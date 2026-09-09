"""Budget/evidence -> readiness; pure files and fixed clocks, never device work."""

import copy
import hashlib
import json

import pytest

from shadow_lab.mage_w4_search.pilot import closeout_report


def artifact(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) if not isinstance(value, str) else value)
    return reference(path)


def reference(path):
    raw = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw)}


def case(tmp_path):
    source_files = {
        name: artifact(tmp_path / f"source-{i}.txt", f"fixture source {i}")
        for i, name in enumerate(
            (
                "shadow_lab/__init__.py",
                "shadow_lab/mage_w4_search/__init__.py",
                "shadow_lab/mage_w4_search/budget.py",
                "worker.py",
            )
        )
    }
    sources = {name: ref["sha256"] for name, ref in source_files.items()}
    harness = artifact(tmp_path / "harness.py", "# corrected fixture harness")
    old_harness = artifact(tmp_path / "old-harness.py", "# original fixture harness")
    leases, runs = [], {}
    for name, host, start, released, remote_stop in (
        ("original", "gooo", 100.0, 184.8, 126.2),
        ("gooo", "gooo", 200.0, 225.8, 225.2),
        ("goooinext", "goooinext", 240.0, 265.9, 265.2),
    ):
        lease = {
            "lease_id": name,
            "request_id": name + "-request",
            "run_id": name + "-run",
            "ticket_id": "06",
            "device_id": host,
            "started_at": start,
            "hard_deadline": start + 30,
            "reserved_seconds": 30.0,
        }
        leases.append({**lease, "stopped_at": released, "stop_reason": "verified group absence"})
        stop = {
            **{
                k: lease[k]
                for k in (
                    "lease_id",
                    "request_id",
                    "run_id",
                    "ticket_id",
                    "device_id",
                    "hard_deadline",
                )
            },
            "started_at": start + 0.3,
            "stopped_at": remote_stop,
            "observed_at": remote_stop,
            "process_group": 123,
            "exit_code": -9,
            "termination": "killed",
            "stop_confirmed": True,
        }
        clock = {
            "offset_seconds": 0.2,
            "uncertainty_seconds": 0.01,
            "at": start - 0.1,
            "bound_seconds": 2,
            "cleanup_seconds": 8,
        }
        group = {"process_group_absent": True, "checked_at": released, "receipt": stop}
        worker = {
            "event": "worker_started",
            "pid": 123,
            "process_group": 123,
            "at": start + 0.31,
            "sigterm": "ignored",
            "cuda_visible_devices": "",
        }
        audit = {
            "lease": lease,
            "stop": stop,
            "clock": clock,
            "source_sha256": sources,
            "independent_group_check": group,
            "transport_events": [],
            "harness_sha256": harness["sha256"],
        }
        prepared = {"source_sha256": sources, "harness_sha256": harness["sha256"]}
        runs[name] = {
            kind: artifact(tmp_path / name / (kind + ".json"), value)
            for kind, value in {
                "lease": lease,
                "stop": stop,
                "clock": clock,
                "audit": audit,
                "worker_log": worker,
                "prepared": prepared,
            }.items()
        }
    snapshot = {
        "campaign_id": "pilot",
        "at": 280.0,
        "leases": leases,
        "used_seconds": 136.5,
        "devices": {"gooo": {"used_seconds": 110.6}, "goooinext": {"used_seconds": 25.9}},
        "timing_events": [],
    }
    disposition = {
        "schema": "mage-late-settlement-disposition/v1",
        "kind": "late_settlement_after_verified_stop",
        "campaign_id": "pilot",
        "lease_id": "original",
        "original": runs["original"],
        "remediation": {
            "runs": {h: runs[h] for h in ("gooo", "goooinext")},
            "harness": harness,
            "original_harness": old_harness,
        },
        "source_files": source_files,
    }
    return snapshot, artifact(tmp_path / "disposition.json", disposition), disposition


def report(snapshot, ref):
    return closeout_report(snapshot, components=[], evidence={}, runtime_dispositions=[ref])


def test_verified_late_settlement_lifts_only_runtime_block_preserving_every_charge(tmp_path):
    snapshot, ref, _ = case(tmp_path)
    before = copy.deepcopy(snapshot)
    blocked = closeout_report(snapshot, components=[], evidence={})
    assert blocked["readiness"]["status"] == "runtime-blocked"
    result = report(snapshot, ref)
    assert result["readiness"]["status"] == "insufficient-evidence"
    assert result["accounting"]["deadline_overruns"] == blocked["accounting"]["deadline_overruns"]
    assert result["accounting"]["used_seconds"] == pytest.approx(136.5)
    assert result["runtime_incidents"]["resolved"][0]["lease_id"] == "original"
    assert result["runtime_incidents"]["unresolved"] == []
    assert "ocr_lock" in result["readiness"]["missing"]
    assert "measured_group_calibrations" in result["readiness"]["missing"]
    assert result["formal_execution_authorized"] is False
    assert snapshot == before


def rewrite_run(disposition, name, **updates):
    from pathlib import Path

    refs = (
        disposition["original"] if name == "original" else disposition["remediation"]["runs"][name]
    )
    values = {key: json.loads(Path(ref["path"]).read_text()) for key, ref in refs.items()}
    for key, change in updates.items():
        values[key].update(change)
    values["audit"].update(lease=values["lease"], stop=values["stop"], clock=values["clock"])
    if "receipt" in values["audit"]["independent_group_check"]:
        values["audit"]["independent_group_check"]["receipt"] = values["stop"]
    for key, value in values.items():
        refs[key] = artifact(Path(refs[key]["path"]), value)


def test_impossible_observation_chronology_cannot_resolve_a_charge_overrun(tmp_path):
    snapshot, _, disposition = case(tmp_path)
    rewrite_run(disposition, "original", stop={"observed_at": 185.0})
    ref = artifact(tmp_path / "disposition.json", disposition)
    result = report(snapshot, ref)
    assert result["readiness"]["status"] == "runtime-blocked"
    assert result["runtime_incidents"]["resolved"] == []


@pytest.mark.parametrize(
    "problem",
    [
        "old_late",
        "new_late",
        "wrong_lease",
        "missing_group",
        "bad_source",
        "bare_resolved",
        "bad_hash",
    ],
)
def test_invalid_or_incomplete_disposition_never_unblocks_runtime(tmp_path, problem):
    from pathlib import Path

    snapshot, _, disposition = case(tmp_path)
    if problem in {"old_late", "new_late"}:
        name, stopped = ("original", 131.2) if problem == "old_late" else ("gooo", 231.2)
        rewrite_run(disposition, name, stop={"stopped_at": stopped, "observed_at": stopped})
    elif problem == "wrong_lease":
        rewrite_run(disposition, "original", stop={"lease_id": "different-lease"})
    elif problem == "missing_group":
        rewrite_run(
            disposition,
            "original",
            audit={
                "independent_group_check": {
                    "process_group_absent": False,
                    "receipt": {},
                    "checked_at": 184.8,
                }
            },
        )
    elif problem == "bad_source":
        rewrite_run(disposition, "gooo", prepared={"source_sha256": {}})
    elif problem == "bare_resolved":
        disposition = {"lease_id": "original", "resolved": True}
    elif problem == "bad_hash":
        Path(disposition["original"]["stop"]["path"]).write_text("{}")
    ref = artifact(tmp_path / "disposition.json", disposition)
    result = report(snapshot, ref)
    assert result["readiness"]["status"] == "runtime-blocked"
    assert result["runtime_incidents"]["resolved"] == []
    assert result["accounting"]["used_seconds"] == pytest.approx(136.5)
    assert result["formal_execution_authorized"] is False


@pytest.mark.parametrize("problem", ["active", "another_overrun"])
def test_resolving_old_incident_cannot_clear_other_runtime_blocks(tmp_path, problem):
    snapshot, ref, _ = case(tmp_path)
    new = {
        "lease_id": "other",
        "request_id": "other",
        "run_id": "other",
        "ticket_id": "06",
        "device_id": "gooo",
        "started_at": 270.0,
        "hard_deadline": 300.0,
        "reserved_seconds": 30.0,
        "stopped_at": None,
        "stop_reason": None,
    }
    added = 10.0
    if problem == "another_overrun":
        added = 40.0
        new["stopped_at"] = 310.0
        snapshot["at"] = 320.0
    snapshot["leases"].append(new)
    snapshot["used_seconds"] += added
    snapshot["devices"]["gooo"]["used_seconds"] += added
    result = report(snapshot, ref)
    assert result["readiness"]["status"] == "runtime-blocked"
    assert result["formal_execution_authorized"] is False
    assert result["accounting"]["used_seconds"] == pytest.approx(136.5 + added)


def test_budget_exhaustion_is_a_normal_stop_without_runtime_failure_or_formal_allowance(tmp_path):
    snapshot, ref, _ = case(tmp_path)
    snapshot["leases"].append(
        {
            "lease_id": "completed",
            "request_id": "completed",
            "run_id": "completed",
            "ticket_id": "08",
            "device_id": "gooo",
            "started_at": 300.0,
            "stopped_at": 28989.4,
            "hard_deadline": 29000.0,
            "reserved_seconds": 28700.0,
        }
    )
    snapshot.update(at=30000.0, used_seconds=28825.9)
    snapshot["devices"]["gooo"]["used_seconds"] = 28800.0
    before = copy.deepcopy(snapshot)
    result = report(snapshot, ref)
    assert result["stop_reason"] == "budget-exhausted"
    assert result["runtime_incidents"]["budget_exhausted"] is True
    assert result["runtime_incidents"]["runtime_blocked"] is False
    assert result["readiness"]["status"] == "insufficient-evidence"
    assert "formal_total_device_seconds" in result["readiness"]["missing"]
    assert result["formal_budget_seconds"] is None
    assert result["formal_execution_authorized"] is False
    assert result["accounting"]["used_seconds"] == pytest.approx(28825.9)
    assert result["accounting"]["remaining_seconds"] == pytest.approx(28774.1)
    assert result["accounting"]["remaining_per_device_seconds"]["gooo"] == 0
    assert result["runtime_incidents"]["charge_overruns"][0]["seconds"] == pytest.approx(54.8)
    assert snapshot == before


def test_html_explains_resolved_incident_without_hiding_historical_charge(tmp_path):
    from shadow_lab.mage_w4_search.pilot import render_closeout_html

    snapshot, ref, _ = case(tmp_path)
    rendered = render_closeout_html(report(snapshot, ref))
    assert "已核验的迟到结算" in rendered
    assert "54.800000 秒" in rendered
    assert "历史预留超时仍计费" in rendered


def test_report_cli_consumes_append_only_disposition_from_explicit_manifest(tmp_path):
    import subprocess
    import sys

    snapshot, ref, _ = case(tmp_path)
    artifact(tmp_path / "snapshot.json", snapshot)
    artifact(
        tmp_path / "input.json", {"components": [], "evidence": {}, "runtime_dispositions": [ref]}
    )
    output = tmp_path / "new-report.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "shadow_lab.mage_w4_search.pilot",
            "--budget-snapshot",
            str(tmp_path / "snapshot.json"),
            "--evidence-manifest",
            str(tmp_path / "input.json"),
            "--output-json",
            str(output),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    document = json.loads(output.read_text())
    assert document["readiness"]["status"] == "insufficient-evidence"
    assert len(document["runtime_incidents"]["resolved"]) == 1
    assert document["accounting"]["used_seconds"] == pytest.approx(136.5)
    assert document["formal_execution_authorized"] is False
