"""Actual local pipes exercise the SSH-stdio protocol without device work."""

import json
import sys
from dataclasses import asdict

import pytest
from test_mage_w4_coordinator import assignment, setup

from shadow_lab.mage_w4_search.budget_rpc import RemoteCoordinator


def connection(tmp_path, coordinator, q0, scales, budget):
    q0_path, scales_path = tmp_path / "q0.json", tmp_path / "scales.json"
    q0_path.write_text(json.dumps(asdict(q0)))
    scales_path.write_text(json.dumps(asdict(scales)))
    config = {
        "schema": "mage-w4-coordinator-config/v1",
        "state_path": str(coordinator.path),
        "q0_path": str(q0_path),
        "scale_receipt_path": str(scales_path),
        "rng_seed": 41,
    }
    command = [sys.executable, "-m", "shadow_lab.mage_w4_search.budget_rpc"]
    return RemoteCoordinator(command, config, budget, execution_kind="fixture")


def test_subprocess_claim_and_duplicate_rpc_share_one_authority_and_lease(tmp_path):
    coordinator, q0, scales, budget, _ = setup(tmp_path)
    planned, _ = assignment(coordinator, tmp_path, q0)
    coordinator.begin_round("round-rpc", (planned,))
    with connection(tmp_path, coordinator, q0, scales, budget) as remote:
        assert remote.initialization_timing["host_cpu_seconds"] >= 0
        args = {
            "round_id": "round-rpc",
            "candidate_id": "candidate",
            "device_id": "gooo",
            "reserve_seconds": 10,
            "expected_request_digest": planned.request.request_digest,
        }
        first = remote.call("claim", args)
        assert remote.last_operation_timing["host_cpu_seconds"] >= 0
        assert remote.last_operation_timing["wall_seconds"] >= 0
        second = remote.call("claim", args)
        assert first["lease_id"] == second["lease_id"]
        assert (
            remote.call("work_request", {"round_id": "round-rpc", "candidate_id": "candidate"})
            == first
        )
        with pytest.raises(ValueError, match="unsupported"):
            remote.call("release", {"lease_id": first["lease_id"]})
    assert len(budget.snapshot()["leases"]) == 1
    assert budget.snapshot()["committed_seconds"] == 10


def test_pipe_loss_after_claim_keeps_occupancy_and_original_claim_is_recoverable(tmp_path):
    coordinator, q0, scales, budget, _ = setup(tmp_path)
    planned, _ = assignment(coordinator, tmp_path, q0)
    coordinator.begin_round("round-disconnect", (planned,))
    remote = connection(tmp_path, coordinator, q0, scales, budget)
    with remote:
        claimed = remote.call(
            "claim",
            {
                "round_id": "round-disconnect",
                "candidate_id": "candidate",
                "device_id": "gooo",
                "reserve_seconds": 10,
                "expected_request_digest": planned.request.request_digest,
            },
        )
        remote.process.kill()
        remote.process.wait(timeout=5)
        with pytest.raises((EOFError, BrokenPipeError, ConnectionError)):
            remote.call("snapshot", {})
    assert budget.snapshot()["devices"]["gooo"]["active_lease_id"] == claimed["lease_id"]
    with connection(tmp_path, coordinator, q0, scales, budget) as recovered:
        assert (
            recovered.call(
                "work_request", {"round_id": "round-disconnect", "candidate_id": "candidate"}
            )["lease_id"]
            == claimed["lease_id"]
        )
    assert budget.snapshot()["leases"][0]["stopped_at"] is None


def test_lost_reply_before_work_commit_can_reconcile_original_lease_without_reserve(tmp_path):
    coordinator, q0, scales, budget, _ = setup(tmp_path)
    planned, _ = assignment(coordinator, tmp_path, q0)
    coordinator.begin_round("round-uncommitted", (planned,))
    # The authoritative reserve has committed, but the GB10 work-row transaction has not.
    lease = budget.reserve(
        planned.request.request_id, "06", "gooo", 10, run_id=planned.request.run_id
    )
    assert coordinator.work_request("round-uncommitted", "candidate").lease_id is None
    with connection(tmp_path, coordinator, q0, scales, budget) as remote:
        args = {
            "round_id": "round-uncommitted",
            "candidate_id": "candidate",
            "lease_id": lease.lease_id,
            "expected_request_digest": planned.request.request_digest,
        }
        first = remote.call("reconcile_claim", args)
        assert remote.call("reconcile_claim", args) == first
        assert first["lease_id"] == lease.lease_id
        state = remote.call("snapshot", {})
        assert state["center_version"] == 0
        assert state["events"][-1]["kind"] == "claim_reconciled_without_launch"
    assert len(budget.snapshot()["leases"]) == 1
    assert budget.snapshot()["leases"][0]["stopped_at"] is None


def test_lost_reserve_reply_is_not_rolled_back_and_remote_cannot_release(tmp_path):
    coordinator, q0, scales, budget, _ = setup(tmp_path)
    source = """
import json,sys
hello=json.loads(sys.stdin.readline()); authority=hello['authority']['sha256']
def send(value): print(json.dumps(value),flush=True)
send({'type':'ready','authority_sha256':authority})
operation=json.loads(sys.stdin.readline())
send({'type':'budget-call','sequence':1,'authority_sha256':authority,'method':'reserve',
      'args':{'request_id':'lost-reply','ticket_id':'06','device_id':'gooo','seconds':10,'run_id':'lost-reply'}})
"""
    remote = connection(tmp_path, coordinator, q0, scales, budget)
    remote.command = [sys.executable, "-c", source]
    with remote, pytest.raises((EOFError, BrokenPipeError, ConnectionError)):
        remote.call("fixture", {})
    leases = budget.snapshot()["leases"]
    assert len(leases) == 1 and leases[0]["stopped_at"] is None
    source = """
import json,sys
hello=json.loads(sys.stdin.readline()); authority=hello['authority']['sha256']
def send(value): print(json.dumps(value),flush=True)
send({'type':'ready','authority_sha256':authority})
operation=json.loads(sys.stdin.readline())
send({'type':'budget-call','sequence':1,'authority_sha256':authority,'method':'release',
      'args':{'lease_id':sys.argv[1],'reason':'not authorized'}})
reply=json.loads(sys.stdin.readline())
send({'type':'result','sequence':1,'ok':True,'result':reply})
sys.stdin.readline()
"""
    remote = connection(tmp_path, coordinator, q0, scales, budget)
    remote.command = [sys.executable, "-c", source, leases[0]["lease_id"]]
    with remote:
        denied = remote.call("fixture", {})
        assert denied["ok"] is False and "unsupported" in denied["error"]
    assert budget.snapshot()["leases"][0]["stopped_at"] is None


@pytest.mark.parametrize("fault", ["authority", "sequence"])
def test_protocol_refuses_a_forged_authority_or_duplicate_sequence_before_reserve(tmp_path, fault):
    coordinator, q0, scales, budget, _ = setup(tmp_path)
    source = """
import json,sys
hello=json.loads(sys.stdin.readline()); authority=hello['authority']['sha256']
def send(value): print(json.dumps(value),flush=True)
send({'type':'ready','authority_sha256':authority})
operation=json.loads(sys.stdin.readline())
send({'type':'budget-call','sequence':0 if sys.argv[1]=='sequence' else 1,
      'authority_sha256':'0'*64 if sys.argv[1]=='authority' else authority,'method':'reserve',
      'args':{'request_id':'wrong','ticket_id':'06','device_id':'gooo','seconds':10,'run_id':'wrong'}})
sys.stdin.readline()
"""
    remote = connection(tmp_path, coordinator, q0, scales, budget)
    remote.command = [sys.executable, "-c", source, fault]
    with remote, pytest.raises(ValueError, match="authority/sequence"):
        remote.call("fixture", {})
    assert not budget.snapshot()["leases"]
