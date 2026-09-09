"""A frozen engineering replay cannot become optimizer work; models remain fixtures."""

import json
from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone

import pytest

from shadow_lab.mage_w4_search.contracts import CandidateRequest, digest_json
from shadow_lab.mage_w4_search.development_replay import prepare_development_replay


def source_fixture(tmp_path):
    from test_mage_w4_coordinator import center
    from test_mage_w4_evaluation_worker import request_inputs

    from shadow_lab.mage_w4_search.center_execution import prepare_center_job
    from shadow_lab.mage_w4_search.ocr import EngineIdentity, OCRConfig
    from shadow_lab.mage_w4_search.pilot import artifact_reference

    plan, manifest, shared = request_inputs()
    config = OCRConfig(EngineIdentity("fixture", "1", (("weights", "e" * 64),)))
    shared["inputs"] = {"ocr_config": asdict(config)}
    shared["run_root"] = str(tmp_path)
    next(row for row in shared["dependencies"] if row["role"] == "ocr_scorer")["sha256"] = (
        config.sha256
    )
    q0 = center(tmp_path, "q0", "a" * 64)
    candidate = center(tmp_path, "candidate", "b" * 64, fixed=q0)
    plan.update(schema="mage-w4-center-plan/v1", ticket_id="06", center_version=0)
    plan["candidate"] = {
        "kind": "adjacent",
        "candidate_id": "candidate",
        "group_id": 0,
        "mutation_count": 1,
        "seed": 12,
        "candidate_index": 0,
        "max_attempts": 2,
    }
    receipt = {
        "status": "candidate",
        "base_sha256": q0.digest,
        "candidate_sha256": candidate.digest,
        "fixed_sha256": shared["fixed_state_sha256"],
        "group_id": "block_00",
        "actual_changes": 1,
        "requested_changes": 1,
        "seed": 12,
        "candidate_index": 0,
        "max_attempts": 2,
    }
    source = prepare_center_job(
        "source",
        plan,
        manifest,
        shared=shared,
        q0=q0,
        current=q0,
        evaluated=candidate,
        template={"path": "fixture", "sha256": "e" * 64},
        candidate_receipt=receipt,
    )
    directory = tmp_path / "jobs/source"
    directory.mkdir(parents=True)
    path = directory / "prepared.json"
    path.write_text(json.dumps(source))
    return source, manifest, artifact_reference(path)


def replay_fixture(tmp_path):
    source, manifest, artifact = source_fixture(tmp_path)
    replay = prepare_development_replay(
        source,
        manifest,
        source_artifact=artifact,
        run_id="replay",
        device={"id": "goooinext"},
        registered_at_utc=(datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat(),
        rule={"max_abs_pixel_error": 0, "require_identical_png": True, "max_abs_score_error": 0},
    )
    return source, replay, manifest


def test_replay_freezes_the_same_payload_and_data_but_distinct_execution_identity(tmp_path):
    from shadow_lab.mage_w4_search.evaluation_worker import reconstruct_evaluation_job

    source, replay, manifest = replay_fixture(tmp_path)
    before, after = (CandidateRequest.from_dict(value["request"]) for value in (source, replay))
    assert after.request_id.startswith("evaluation-replay-")
    assert after.ticket_id == "06" and after.device_id == "goooinext"
    assert (
        replace(
            after,
            request_id=before.request_id,
            run_id=before.run_id,
            device_id=before.device_id,
            runtime_identity=before.runtime_identity,
        )
        == before
    )
    assert replay["evaluation_job"]["eligible_for_center_update"] is False
    assert digest_json(
        reconstruct_evaluation_job(replay["evaluation_job"], manifest)
    ) == digest_json(replay)


@pytest.mark.parametrize(
    "field",
    [
        "candidate_digest",
        "center_digest",
        "q0_digest",
        "mutation_seed",
        "samples",
        "runtime_identity",
    ],
)
def test_replay_rejects_any_nonexecution_identity_change(tmp_path, field):
    from shadow_lab.mage_w4_search.development_replay import verify_replay_prepared

    _, replay, manifest = replay_fixture(tmp_path)
    replay["request"][field] = "changed"
    with pytest.raises(ValueError, match="identity"):
        verify_replay_prepared(replay, manifest)


def test_replay_cannot_enter_optimizer_or_ordinary_launcher(tmp_path):
    from types import SimpleNamespace

    from test_mage_w4_coordinator import setup
    from test_mage_w4_launch import launcher

    from shadow_lab.mage_w4_search.coordinator import CandidateAssignment

    fixture_root = tmp_path / "replay"
    fixture_root.mkdir()
    _, replay, manifest = replay_fixture(fixture_root)
    request = CandidateRequest.from_dict(replay["request"])
    state_root = tmp_path / "state"
    state_root.mkdir()
    coordinator, q0, *_ = setup(state_root)
    with pytest.raises(ValueError, match="replay"):
        coordinator.begin_round(
            "bad", (CandidateAssignment(request=request, manifest=manifest, center=q0),)
        )
    with pytest.raises(ValueError, match="replay"):
        coordinator.submit("bad", SimpleNamespace(request=request), None, None)
    module = launcher()
    with pytest.raises(ValueError, match="replay"):
        module.worker_command("evaluation", request)
    assert "run" in module.worker_command("development-replay", request)


def execute_fixture(prepared, manifest, directory, *, extras=None, clip=0.3):
    import time

    from test_mage_w4_evaluation_worker import FixtureScorer, fixture_generator

    from shadow_lab.mage_qad_checkpoint import content_hash
    from shadow_lab.mage_w4_search.budget import DeviceLease
    from shadow_lab.mage_w4_search.evaluation_worker import run_evaluation_job
    from shadow_lab.mage_w4_search.ocr import RawSegment

    directory.mkdir(parents=True, exist_ok=True)
    now = time.time()
    request = CandidateRequest.from_dict(prepared["request"])
    lease = DeviceLease(
        "lease-" + request.run_id,
        request.run_id + "-occupancy",
        "06",
        request.device_id,
        now - 0.001,
        now + 300,
        300.001,
        request.run_id,
    )
    leased = replace(request, lease_id=lease.lease_id)
    grant = {
        **prepared["preflight"],
        "status": "authorized",
        "request": asdict(leased),
        "request_digest": leased.request_digest,
        "lease": asdict(lease),
        **(extras or {}),
    }
    for name, value in (
        ("prepared.json", prepared),
        ("lease.json", asdict(lease)),
        ("grant.json", grant),
    ):
        (directory / name).write_text(json.dumps(value))
    shared = prepared["evaluation_job"]["shared"]
    external = fixture_generator(shared, [])

    def generation(request, manifest, output):
        report = external(request, manifest, output)
        prompts = {row.prompt_id: row.prompt for row in manifest.records}
        report["request_sha256"] = content_hash(
            [
                {"sample_id": row.sample_id, "prompt": prompts[row.prompt_id], "seed": row.seed}
                for row in request.samples
            ]
        )
        (output / "report.json").write_text(json.dumps(report))
        return report

    result = run_evaluation_job(
        directory,
        manifest,
        device_id=request.device_id,
        fixture_generation=generation,
        fixture_scorers=(
            FixtureScorer(clip, []),
            FixtureScorer(0.2, []),
            lambda *_: (RawSegment("OPEN", (0, 0, 2, 1)),),
        ),
    )
    stopped = result["finished_at"] + 0.001
    for name, value in {
        "clock.json": {"at": lease.started_at - 0.1, "offset_seconds": 0, "uncertainty_seconds": 0},
        "stopped.json": {
            **asdict(lease),
            "started_at": result["started_at"],
            "stopped_at": stopped,
            "observed_at": stopped,
            "exit_code": 0,
            "stop_confirmed": True,
            "process_group": 999,
        },
        "independent-group-check.json": {
            "absent": True,
            "process_group": 999,
            "checked_at": stopped + 0.001,
            "device_id": request.device_id,
            "lease_id": lease.lease_id,
        },
    }.items():
        (directory / name).write_text(json.dumps(value))
    return {**asdict(lease), "stopped_at": stopped + 0.002}


def snapshot_for(*rows):
    def seconds(row):
        return row["stopped_at"] - row["started_at"]

    return {
        "campaign_id": "fixture",
        "at": max(row["stopped_at"] for row in rows) + 0.001,
        "leases": list(rows),
        "used_seconds": sum(map(seconds, rows)),
        "devices": {
            device: {
                "used_seconds": sum(seconds(row) for row in rows if row["device_id"] == device)
            }
            for device in ("gooo", "goooinext")
        },
        "timing_events": [],
    }


def source_verified_fixture(tmp_path):
    from shadow_lab.mage_w4_search.development_replay import (
        replay_registration_binding,
        verify_replay_source,
    )

    source, replay, manifest = replay_fixture(tmp_path)
    binding = replay_registration_binding(
        replay, manifest, CandidateRequest.from_dict(source["request"])
    )
    row = execute_fixture(
        source,
        manifest,
        tmp_path / "jobs/source",
        extras={"development_replay_registration": binding},
    )
    snapshot = snapshot_for(row)
    receipt = verify_replay_source(replay, manifest, snapshot, execution_kind="fixture")
    return source, replay, manifest, row, snapshot, receipt


def test_source_is_verified_on_compute_side_before_control_authorizes_one_replay(tmp_path):
    from shadow_lab.mage_w4_search.development_replay import authorize_replay_control

    source, replay, manifest, row, snapshot, receipt = source_verified_fixture(tmp_path)
    binding = authorize_replay_control(
        replay, manifest, receipt, snapshot, execution_kind="fixture"
    )
    assert receipt["execution_kind"] == "fixture"
    assert receipt["source_evidence"]["batch_sha256"]
    assert binding["source_lease_id"] == row["lease_id"]
    assert binding["destination_job_sha256"] == replay["evaluation_job_sha256"]
    assert snapshot["leases"] == [row]  # control has no authority to reserve/release
    snapshot["leases"][0] = {**row, "stopped_at": None}
    with pytest.raises(ValueError, match="closed"):
        authorize_replay_control(replay, manifest, receipt, snapshot, execution_kind="fixture")


def test_source_without_prior_pair_binding_cannot_be_retroactively_registered(tmp_path):
    from shadow_lab.mage_w4_search.development_replay import verify_replay_source

    source, replay, manifest = replay_fixture(tmp_path)
    row = execute_fixture(source, manifest, tmp_path / "jobs/source")
    with pytest.raises(ValueError, match="before source"):
        verify_replay_source(replay, manifest, snapshot_for(row), execution_kind="fixture")


@pytest.mark.parametrize("clip,expected", [(0.3, "matched"), (0.31, "failed")])
def test_complete_replay_verifies_both_batches_and_keeps_score_difference_visible(
    tmp_path, clip, expected
):
    from shadow_lab.mage_w4_search.development_replay import (
        authorize_replay_control,
        close_development_replay,
    )
    from shadow_lab.mage_w4_search.pilot import artifact_reference

    _, replay, manifest, source_row, snapshot, receipt = source_verified_fixture(tmp_path)
    directory = tmp_path / "jobs/replay"
    directory.mkdir()
    path = directory / "replay-source-verification.json"
    path.write_text(json.dumps(receipt))
    binding = authorize_replay_control(
        replay, manifest, receipt, snapshot, execution_kind="fixture"
    )
    binding["source_verification_file_sha256"] = artifact_reference(path)["sha256"]
    target_row = execute_fixture(
        replay, manifest, directory, extras={"development_replay": binding}, clip=clip
    )
    summary = close_development_replay(
        replay, manifest, snapshot_for(source_row, target_row), directory, execution_kind="fixture"
    )
    assert summary["status"] == expected
    assert summary["execution_kind"] == "fixture"
    assert summary["eligible_for_center_update"] is False
    assert summary["optimizer_candidate_count"] == 0
    assert summary["rows"][0]["identical_png"] is True
    assert summary["rows"][0]["max_abs_pixel_error"] == 0
    assert summary["rows"][0]["absolute_score_differences"]["clip"] == abs(clip - 0.3)
    assert len(summary["accounted_lease_ids"]) == 2


def test_target_without_verified_source_grant_fails_before_fixture_generation(tmp_path):
    source, replay, manifest = replay_fixture(tmp_path)
    with pytest.raises(ValueError, match="source verification"):
        execute_fixture(replay, manifest, tmp_path / "jobs/replay")
    assert not (tmp_path / "jobs/replay/generation").exists()
    result = json.loads((tmp_path / "jobs/replay/result.json").read_text())
    assert result["status"] == "failed"


def test_launcher_metadata_checks_closed_authority_without_calling_batch_verifier(
    tmp_path, monkeypatch
):
    from test_mage_w4_launch import launcher

    from shadow_lab.mage_w4_search import development

    source, replay, manifest, row, snapshot, receipt = source_verified_fixture(tmp_path)
    module = launcher()

    def forbidden(*args, **kwargs):
        raise AssertionError("Mac control must not perform batch/OCR mathematics")

    monkeypatch.setattr(development, "verify_evaluation_job", forbidden)
    raw = json.dumps(receipt)
    binding = module.replay_launch_metadata(
        replay, manifest, receipt_text=raw, snapshot=snapshot, execution_kind="fixture"
    )
    assert binding["development_replay"]["source_lease_id"] == row["lease_id"]
    assert (
        module.replay_launch_metadata(source, manifest, registration=replay)[
            "development_replay_registration"
        ]["destination_job_sha256"]
        == replay["evaluation_job_sha256"]
    )
    with pytest.raises(ValueError, match="identity/status"):
        module.replay_launch_metadata(replay, manifest, receipt_text=raw, snapshot=snapshot)
    assert snapshot["leases"] == [row]


def test_native_replay_uses_actual_full_center_loader_without_q0_fallback(tmp_path, monkeypatch):
    from shadow_lab.mage_w4_search import center_execution, evaluation_worker, native

    _, replay, manifest = replay_fixture(tmp_path)
    job = replay["evaluation_job"]
    job["shared"]["inventory"] = {}
    observed = []

    class BoundaryReached(Exception):
        pass

    def center_loader(job, request, stage):
        observed.append(request.candidate_digest)
        raise BoundaryReached

    def forbidden(*args, **kwargs):
        raise AssertionError("replay cannot use the ordinary q0 loader")

    monkeypatch.setattr(center_execution, "load_execution_center", center_loader)
    monkeypatch.setattr(native, "load_quantized_assets", forbidden)
    with pytest.raises(BoundaryReached):
        evaluation_worker._generate_real(
            job,
            CandidateRequest.from_dict(replay["request"]),
            manifest,
            tmp_path / "output",
            lambda _: None,
        )
    assert observed == [replay["request"]["candidate_digest"]]


def test_replay_cli_failure_retains_cpu_timing_without_lease_and_refuses_overwrite(
    tmp_path, monkeypatch
):
    import importlib.util
    import sys
    from pathlib import Path

    path = Path(__file__).parents[2] / "scripts/mage_w4_replay.py"
    spec = importlib.util.spec_from_file_location("replay_cli_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(path),
            "prepare",
            "--run-root",
            str(tmp_path),
            "--job-id",
            "replay",
            "--device-id",
            "goooinext",
            "--source-prepared",
            str(tmp_path / "source"),
            "--rule",
            str(tmp_path / "rule"),
            "--registered-at-utc",
            "2026-09-09T00:00:00Z",
        ],
    )
    with pytest.raises(ValueError, match="GB10"):
        module.main()
    timing = json.loads((tmp_path / "jobs/replay/host-timing.json").read_text())
    assert timing["status"] == "failed" and timing["host_cpu_seconds"] >= 0
    assert timing["wall_seconds"] >= 0 and timing["finished_at"] >= timing["started_at"]
    assert list((tmp_path / "jobs/replay").iterdir()) == [tmp_path / "jobs/replay/host-timing.json"]
    with pytest.raises(FileExistsError):
        module.main()


def test_replay_cli_prepare_retains_success_timing_outside_frozen_identity(tmp_path, monkeypatch):
    import importlib.util
    import sys
    from pathlib import Path

    source, manifest, artifact = source_fixture(tmp_path)
    (tmp_path / "data-v1").mkdir()
    (tmp_path / "data-v1/manifest.json").write_text(json.dumps(asdict(manifest)))
    rule = {"max_abs_pixel_error": 0, "require_identical_png": True, "max_abs_score_error": 0}
    (tmp_path / "rule.json").write_text(json.dumps(rule))
    path = Path(__file__).parents[2] / "scripts/mage_w4_replay.py"
    spec = importlib.util.spec_from_file_location("replay_prepare_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Only the host/hardware discovery and external installed inputs are fixtures;
    # request construction and all full-center byte checks remain production code.
    monkeypatch.setattr(module.platform, "system", lambda: "Linux")
    monkeypatch.setattr(module.platform, "machine", lambda: "aarch64")
    monkeypatch.setattr(module.subprocess, "check_output", lambda *a, **kw: "fixture NVIDIA GB10")
    monkeypatch.setattr(module, "verify_evaluation_inputs", lambda _: None)
    registered = "2026-09-09T00:00:00Z"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(path),
            "prepare",
            "--run-root",
            str(tmp_path),
            "--job-id",
            "replay",
            "--device-id",
            "goooinext",
            "--source-prepared",
            artifact["path"],
            "--rule",
            str(tmp_path / "rule.json"),
            "--registered-at-utc",
            registered,
        ],
    )
    module.main()
    expected = prepare_development_replay(
        source,
        manifest,
        source_artifact=artifact,
        run_id="replay",
        device={"id": "goooinext", "nvidia_identity": "fixture NVIDIA GB10"},
        registered_at_utc=registered,
        rule=rule,
    )
    prepared = json.loads((tmp_path / "jobs/replay/prepared.json").read_text())
    assert digest_json(prepared) == digest_json(expected)
    timing = json.loads((tmp_path / "jobs/replay/host-timing.json").read_text())
    assert timing["status"] == "prepared" and timing["host_cpu_seconds"] >= 0
    assert not (tmp_path / "jobs/replay/lease.json").exists()
    assert "host_cpu_seconds" not in prepared
