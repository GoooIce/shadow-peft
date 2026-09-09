"""Changing the evidence output must never create another pilot allowance."""

import importlib.util
import json
import shlex
import sys
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from shadow_lab.mage_w4_search.budget import BudgetLedger


def launcher():
    path = Path(__file__).parents[2] / "scripts/mage_w4_launch.py"
    spec = importlib.util.spec_from_file_location("mage_w4_launch_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_missing_canonical_budget_is_not_silently_initialized(tmp_path, monkeypatch):
    module = launcher()
    path = tmp_path / "pilot-budget.sqlite3"
    monkeypatch.setattr(module, "PILOT_LEDGER", path)
    with pytest.raises(ValueError, match="existing ticket 01"):
        module.open_pilot_ledger()
    assert not path.exists()


def test_reopened_canonical_budget_retains_previous_device_occupancy(tmp_path, monkeypatch):
    module = launcher()
    path = tmp_path / "pilot-budget.sqlite3"
    original = BudgetLedger(path, clock=lambda: 1000.0)
    lease = original.reserve("old-run", "02", "gooo", 10)
    monkeypatch.setattr(module, "PILOT_LEDGER", path)
    recovered = module.open_pilot_ledger().snapshot()
    assert recovered["devices"]["gooo"]["active_lease_id"] == lease.lease_id
    assert recovered["devices"]["gooo"]["used_seconds"] > 0


def test_launcher_authorizes_the_coordinator_claim_without_another_reservation(tmp_path):
    from test_mage_w4_coordinator import assignment, setup

    module = launcher()
    coordinator, q0, _, budget, _ = setup(tmp_path)
    planned, _ = assignment(coordinator, tmp_path, q0)
    coordinator.begin_round("round-launch", (planned,))
    request = coordinator.claim("round-launch", "candidate", "gooo", reserve_seconds=10)
    before = budget.snapshot()
    lease, grant = module.authorize_existing_claim(budget, request, planned.manifest)
    assert lease.lease_id == request.lease_id == grant["request"]["lease_id"]
    assert grant["request_digest"] == planned.request.request_digest
    assert budget.snapshot()["leases"] == before["leases"]
    assert budget.snapshot()["committed_seconds"] == 10
    with pytest.raises(ValueError, match="existing coordinator claim"):
        module.authorize_existing_claim(budget, planned.request, planned.manifest)


def test_missing_local_lease_receipt_recovers_original_record_without_reserving(tmp_path):
    from test_mage_w4_coordinator import assignment, setup

    module = launcher()
    coordinator, q0, _, budget, now = setup(tmp_path)
    planned, _ = assignment(coordinator, tmp_path, q0)
    coordinator.begin_round("round-recover", (planned,))
    claimed = coordinator.claim("round-recover", "candidate", "gooo", reserve_seconds=10)
    now[0] += 12
    recovered = module.find_recorded_lease(budget, planned.request)
    assert recovered.lease_id == claimed.lease_id
    assert len(budget.snapshot()["leases"]) == 1
    assert budget.snapshot()["leases"][0]["stopped_at"] is None
    assert budget.snapshot()["overrun_seconds"] == 2


@pytest.mark.parametrize("explicit_root", [False, True])
def test_launch_cli_uses_one_ledger_and_the_frozen_remote_root(
    tmp_path, monkeypatch, explicit_root
):
    from test_mage_w4_evaluation_worker import request_inputs

    from shadow_lab.mage_w4_search.evaluation_worker import prepare_evaluation_job

    module = launcher()
    remote_root = "/home/goooice/runs/calibration-v2" if explicit_root else module.REMOTE_ROOT
    evidence = tmp_path / "evidence-v2"
    (evidence / "data-v1").mkdir(parents=True)
    plan, manifest, shared = request_inputs()
    shared["run_root"] = remote_root
    prepared = prepare_evaluation_job("root-fixture", plan, manifest, shared=shared)
    (evidence / "data-v1/manifest.json").write_text(json.dumps(asdict(manifest)))
    ledger_path = tmp_path / "canonical/pilot-budget.sqlite3"
    ledger_path.parent.mkdir()
    budget = BudgetLedger(ledger_path)
    monkeypatch.setattr(module, "PILOT_LEDGER", ledger_path)
    local = evidence / "gooo/jobs/root-fixture"
    commands = []

    def transport(host, command, **kwargs):
        assert host == "gooo"
        commands.append(command)
        words = shlex.split(command)
        if words == ["cat", remote_root + "/jobs/root-fixture/prepared.json"]:
            return SimpleNamespace(stdout=json.dumps(prepared), returncode=0)
        if "import time; print(time.time())" in words:
            return SimpleNamespace(stdout=str(time.time()), returncode=0)
        if command.startswith("nohup setsid "):
            assert words[words.index("--run-root") + 1] == remote_root
            return SimpleNamespace(stdout="", returncode=0)
        if words == ["cat", remote_root + "/jobs/root-fixture/stopped.json"]:
            lease = json.loads((local / "lease.json").read_text())
            return SimpleNamespace(
                stdout=json.dumps({"lease_id": lease["lease_id"], "stop_confirmed": True}),
                returncode=0,
            )
        raise AssertionError(command)

    monkeypatch.setattr(module, "remote", transport)
    transfers = []
    monkeypatch.setattr(module.subprocess, "run", lambda command, **kw: transfers.append(command))
    args = [
        "mage_w4_launch.py",
        "--host",
        "gooo",
        "--job-id",
        "root-fixture",
        "--worker-kind",
        "evaluation",
        "--seconds",
        "120",
        "--evidence-root",
        str(evidence),
    ]
    if explicit_root:
        args += ["--remote-root", remote_root]
    monkeypatch.setattr(sys, "argv", args)
    module.main()
    assert len(budget.snapshot()["leases"]) == 1
    assert budget.snapshot()["leases"][0]["stopped_at"] is not None
    assert not (evidence / "pilot-budget.sqlite3").exists()
    assert json.loads((local / "launch-target.json").read_text())["remote_root"] == remote_root
    assert any("gooo:" + remote_root + "/jobs/root-fixture/" in row for row in transfers)
    before = budget.snapshot()["leases"]
    # Old v1 evidence has no target sidecar; complete newer jobs can recover from their frozen root.
    (local / "launch-target.json").unlink()
    commands.clear()
    monkeypatch.setattr(sys, "argv", [*args, "--recover"])
    module.main()
    assert budget.snapshot()["leases"] == before
    assert not any(command.startswith("nohup ") for command in commands)
    commands.clear()
    monkeypatch.setattr(
        sys, "argv", [*args, "--recover", "--remote-root", "/home/goooice/runs/wrong-root"]
    )
    with pytest.raises(ValueError, match="launch target"):
        module.main()
    assert commands == []
    assert budget.snapshot()["leases"] == before


@pytest.mark.parametrize("remote_root", ["relative/run", "/tmp/../wrong"])
def test_launch_cli_rejects_nonabsolute_or_ambiguous_roots_before_io(
    tmp_path, monkeypatch, capsys, remote_root
):
    module = launcher()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mage_w4_launch.py",
            "--host",
            "gooo",
            "--job-id",
            "invalid-root",
            "--remote-root",
            remote_root,
            "--evidence-root",
            str(tmp_path / "evidence"),
        ],
    )
    with pytest.raises(SystemExit):
        module.main()
    assert "absolute canonical POSIX path" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []
