"""Development round orchestration without a model or formal confirmation."""

import hashlib
import sqlite3
from dataclasses import replace

import pytest
from test_mage_w4_rewards import batch, scale_plan, screening_batch

from shadow_lab.mage_w4_search.budget import BudgetLedger
from shadow_lab.mage_w4_search.contracts import ArtifactIdentity
from shadow_lab.mage_w4_search.coordinator import (
    CandidateAssignment,
    DevelopmentConfirmation,
    DevelopmentCoordinator,
    FullCenter,
)
from shadow_lab.mage_w4_search.rewards import calibrate_scales


def center(tmp_path, name, digest, fixed=None):
    artifacts = []
    for role in ("quantized_packed", "weight_scales", "quantized_remainder", "layer_manifest"):
        if fixed is not None and role != "quantized_packed":
            artifacts.append(next(item for item in fixed.artifacts if item.role == role))
            continue
        path = tmp_path / f"{name}-{role}.bin"
        data = f"{name}-{role}".encode()
        path.write_bytes(data)
        artifacts.append(
            ArtifactIdentity(role, hashlib.sha256(data).hexdigest(), len(data), str(path))
        )
    return FullCenter(digest, tuple(artifacts))


def setup(tmp_path):
    q0 = center(tmp_path, "q0", "a" * 64)
    baseline = batch((("p1", 1, 0.0, 0.0, 0.0), ("p2", 1, 1.0, 2.0, 0.5), ("p3", 1, 2.0, 4.0, 1.0)))
    actual_artifacts = {item.role: item for item in q0.artifacts}
    baseline = replace(
        baseline,
        request=replace(
            baseline.request,
            dependencies=tuple(
                actual_artifacts.get(item.role, item) for item in baseline.request.dependencies
            ),
        ),
    )
    scales = calibrate_scales(scale_plan(baseline), baseline)
    now = [1_000.0]
    budget = BudgetLedger(tmp_path / "budget.sqlite", clock=lambda: now[0])
    coordinator = DevelopmentCoordinator(
        tmp_path / "search.sqlite", q0=q0, scales=scales, budget=budget, rng_seed=41
    )
    return coordinator, q0, scales, budget, now


def test_reopen_preserves_q0_full_center_rng_and_the_same_consumed_budget(tmp_path):
    coordinator, q0, scales, budget, now = setup(tmp_path)
    initial = coordinator.snapshot()
    seeds = coordinator.next_mutation_seeds(2)
    assert len(set(seeds)) == 2
    lease = budget.reserve("one-workload", "06", "gooo", 20, run_id="fixture-run")
    now[0] += 10
    budget.release(lease.lease_id, reason="fixture worker stopped")
    recovered = DevelopmentCoordinator(
        tmp_path / "search.sqlite", q0=q0, scales=scales, budget=budget, rng_seed=41
    )
    restored = recovered.snapshot()
    assert restored["center"] == initial["center"]
    assert restored["q0"]["native_payload_sha256"] == "a" * 64
    assert restored["budget"]["used_seconds"] == 10
    assert recovered.next_mutation_seeds(2) == seeds
    assert restored["center_identity"] == "development_q0"
    assert restored["independently_confirmed"] is False


def assignment(
    coordinator,
    tmp_path,
    q0,
    candidate_id="candidate",
    *,
    device="gooo",
    seed_index=0,
    candidate_digest="b" * 64,
    values=(0.1, 1.0, 0.5),
    current=None,
):
    state = coordinator.snapshot()
    candidate = center(tmp_path, candidate_id, candidate_digest, fixed=q0)
    scored = screening_batch(
        candidate_id,
        values,
        candidate_digest=candidate_digest,
        center_digest=current.digest if current else q0.digest,
        device=device,
    )
    candidate_artifacts = {row.role: row for row in candidate.artifacts}
    request = replace(
        scored.request,
        ticket_id="06",
        center_version=state["center_version"],
        group_id=state["group_index"],
        mutation_seed=coordinator.next_mutation_seeds(seed_index + 1)[seed_index],
        dependencies=tuple(
            candidate_artifacts.get(row.role, row) for row in scored.request.dependencies
        ),
    )
    scored = replace(scored, request=request)
    return CandidateAssignment(request, scored.manifest, candidate), scored


def test_frozen_same_center_work_is_assigned_once_to_its_own_gb10(tmp_path):
    coordinator, q0, _, budget, _ = setup(tmp_path)
    first, _ = assignment(coordinator, tmp_path, q0, "first")
    second, _ = assignment(
        coordinator,
        tmp_path,
        q0,
        "second",
        device="goooinext",
        seed_index=1,
        candidate_digest="c" * 64,
    )
    coordinator.begin_round("round-1", (first, second))
    with pytest.raises(ValueError, match="assigned device"):
        coordinator.claim("round-1", "first", "goooinext", reserve_seconds=10)
    claimed = coordinator.claim("round-1", "first", "gooo", reserve_seconds=10)
    assert claimed.lease_id
    duplicate = coordinator.claim("round-1", "first", "gooo", reserve_seconds=10)
    assert duplicate.lease_id == claimed.lease_id
    second_claim = coordinator.claim("round-1", "second", "goooinext", reserve_seconds=10)
    assert second_claim.lease_id != claimed.lease_id
    assert budget.snapshot()["committed_seconds"] == 20
    assert coordinator.snapshot()["proposal_cursor"] == 2
    with pytest.raises(ValueError, match="incomplete"):
        coordinator.screen("round-1")


def reference(q0, planned, name="q0", values=(0.0, 0.0, 0.5), current=None):
    full = current or q0
    scored = screening_batch(
        name,
        values,
        candidate_digest=full.digest,
        center_digest=planned.request.center_digest,
        device=planned.request.device_id,
    )
    artifacts = {item.role: item for item in full.artifacts}
    return replace(
        scored,
        request=replace(
            scored.request,
            center_version=planned.request.center_version,
            group_id=planned.request.group_id,
            dependencies=tuple(
                artifacts.get(item.role, item) for item in scored.request.dependencies
            ),
        ),
    )


def test_complete_best_candidate_is_atomically_inherited_only_as_a_development_center(tmp_path):
    coordinator, q0, scales, budget, _ = setup(tmp_path)
    planned, scored = assignment(coordinator, tmp_path, q0)
    coordinator.begin_round("round-1", (planned,))
    claimed = coordinator.claim("round-1", "candidate", "gooo", reserve_seconds=10)
    scored = replace(scored, request=claimed)
    q0_scores = reference(q0, planned)
    coordinator.submit("round-1", scored, q0_scores, q0_scores)
    screened = coordinator.screen("round-1")
    assert screened["selected_candidate_id"] == "candidate"
    decision = DevelopmentConfirmation(
        "round-1", "candidate", planned.center.digest, screened["sha256"], True, "f" * 64
    )
    completed = coordinator.finish_round("round-1", decision)
    assert completed["decision"] == "development_accept"
    state = coordinator.snapshot()
    assert state["center_version"] == 1
    assert state["center"] == {
        "native_payload_sha256": planned.center.digest,
        "artifacts": [vars(item) for item in planned.center.artifacts],
    }
    assert state["q0"]["native_payload_sha256"] == q0.digest
    assert state["group_index"] == 1
    assert state["consecutive_no_update_groups"] == 0
    assert state["center_identity"] == "development_mock_accepted"
    assert state["independently_confirmed"] is False
    recovered = DevelopmentCoordinator(
        tmp_path / "search.sqlite", q0=q0, scales=scales, budget=budget, rng_seed=41
    )
    assert recovered.finish_round("round-1", decision) == completed
    assert recovered.snapshot()["center_version"] == 1
    assert recovered.submit("round-1", scored, q0_scores, q0_scores)["status"] == "duplicate"


def test_failed_winner_keeps_center_and_cannot_send_runner_up_to_confirmation(tmp_path):
    coordinator, q0, _, _, _ = setup(tmp_path)
    first, scored_first = assignment(coordinator, tmp_path, q0, "first", values=(0.2, 1.0, 0.5))
    second, scored_second = assignment(
        coordinator,
        tmp_path,
        q0,
        "second",
        device="goooinext",
        seed_index=1,
        candidate_digest="c" * 64,
        values=(0.1, 1.0, 0.5),
    )
    coordinator.begin_round("round-1", (first, second))
    for planned, scored in ((second, scored_second), (first, scored_first)):
        claimed = coordinator.claim(
            "round-1", planned.request.candidate_id, planned.request.device_id, reserve_seconds=10
        )
        q0_scores = reference(q0, planned)
        coordinator.submit("round-1", replace(scored, request=claimed), q0_scores, q0_scores)
    screened = coordinator.screen("round-1")
    assert screened["selected_candidate_id"] == "first"
    failed = DevelopmentConfirmation(
        "round-1", "first", first.center.digest, screened["sha256"], False, "f" * 64
    )
    assert coordinator.finish_round("round-1", failed)["decision"] == "confirmation_failed"
    state = coordinator.snapshot()
    assert state["center_version"] == 0
    assert state["center"]["native_payload_sha256"] == q0.digest
    assert state["group_index"] == 1
    assert state["consecutive_no_update_groups"] == 1
    runner_up = DevelopmentConfirmation(
        "round-1", "second", second.center.digest, screened["sha256"], True, "f" * 64
    )
    with pytest.raises(ValueError, match="already closed"):
        coordinator.finish_round("round-1", runner_up)


def test_technical_abort_preserves_progress_and_rejects_late_results(tmp_path):
    coordinator, q0, scales, budget, now = setup(tmp_path)
    planned, scored = assignment(coordinator, tmp_path, q0)
    coordinator.begin_round("round-1", (planned,))
    claimed = coordinator.claim("round-1", "candidate", "gooo", reserve_seconds=10)
    now[0] += 2
    budget.release(claimed.lease_id, reason="fixture worker stopped after failure")
    coordinator.abort_round("round-1", reason="worker crashed")
    recovered = DevelopmentCoordinator(
        tmp_path / "search.sqlite", q0=q0, scales=scales, budget=budget, rng_seed=41
    )
    state = recovered.snapshot()
    assert state["group_index"] == state["completed_groups"] == 0
    assert state["proposal_cursor"] == 1
    assert state["budget"]["used_seconds"] == 2
    q0_scores = reference(q0, planned)
    with pytest.raises(ValueError, match="late"):
        recovered.submit("round-1", replace(scored, request=claimed), q0_scores, q0_scores)


def test_two_complete_sweeps_without_update_stop_and_do_not_count_aborted_groups(tmp_path):
    coordinator, q0, _, budget, now = setup(tmp_path)
    for index in range(26):
        planned, scored = assignment(
            coordinator,
            tmp_path,
            q0,
            f"candidate-{index}",
            values=(-0.1, 1.0, 0.5),
            candidate_digest=hashlib.sha256(f"candidate-{index}".encode()).hexdigest(),
        )
        round_id = f"round-{index}"
        coordinator.begin_round(round_id, (planned,))
        claimed = coordinator.claim(
            round_id, planned.request.candidate_id, "gooo", reserve_seconds=10
        )
        q0_scores = reference(q0, planned)
        coordinator.submit(round_id, replace(scored, request=claimed), q0_scores, q0_scores)
        assert coordinator.finish_round(round_id)["decision"] == "no_feasible_candidate"
        now[0] += 0.1
        budget.release(claimed.lease_id, reason="fixture worker stopped")
    state = coordinator.snapshot()
    assert state["completed_groups"] == state["consecutive_no_update_groups"] == 26
    assert state["group_index"] == 0
    assert state["status"] == "stopped"
    assert state["stop_reason"] == "two_complete_sweeps_without_update"
    assert state["center_version"] == 0
    extra, _ = assignment(coordinator, tmp_path, q0, "extra")
    with pytest.raises(ValueError, match="stopped"):
        coordinator.begin_round("one-too-many", (extra,))


def test_failed_storage_commit_rolls_back_entire_accept_and_retry_applies_it_once(
    tmp_path, monkeypatch
):
    coordinator, q0, scales, budget, _ = setup(tmp_path)
    planned, scored = assignment(coordinator, tmp_path, q0)
    coordinator.begin_round("round-1", (planned,))
    claimed = coordinator.claim("round-1", "candidate", "gooo", reserve_seconds=10)
    q0_scores = reference(q0, planned)
    coordinator.submit("round-1", replace(scored, request=claimed), q0_scores, q0_scores)
    screened = coordinator.screen("round-1")
    decision = DevelopmentConfirmation(
        "round-1", "candidate", planned.center.digest, screened["sha256"], True, "f" * 64
    )
    connect = sqlite3.connect

    def storage_without_commit(*args, **kwargs):
        # Real SQLite executes the update transaction; its external storage
        # authorization boundary then refuses COMMIT, like a storage failure.
        database = connect(*args, **kwargs)
        database.set_authorizer(
            lambda operation, name, *rest: (
                sqlite3.SQLITE_DENY
                if operation == sqlite3.SQLITE_TRANSACTION and name == "COMMIT"
                else sqlite3.SQLITE_OK
            )
        )
        return database

    with monkeypatch.context() as patch:
        patch.setattr(sqlite3, "connect", storage_without_commit)
        with pytest.raises(sqlite3.DatabaseError):
            coordinator.finish_round("round-1", decision)
    recovered = DevelopmentCoordinator(
        tmp_path / "search.sqlite", q0=q0, scales=scales, budget=budget, rng_seed=41
    )
    state = recovered.snapshot()
    assert state["center_version"] == state["completed_groups"] == state["group_index"] == 0
    assert state["active_round"] == "round-1"
    assert state["center"]["native_payload_sha256"] == q0.digest
    assert not [row for row in state["events"] if row["kind"] == "round_finished"]
    recovered.finish_round("round-1", decision)
    recovered.finish_round("round-1", decision)
    assert recovered.snapshot()["center_version"] == 1


def test_hard_deadline_rejects_result_and_abort_does_not_release_device(tmp_path):
    coordinator, q0, _, budget, now = setup(tmp_path)
    planned, scored = assignment(coordinator, tmp_path, q0)
    coordinator.begin_round("round-1", (planned,))
    claimed = coordinator.claim("round-1", "candidate", "gooo", reserve_seconds=10)
    now[0] += 11
    q0_scores = reference(q0, planned)
    with pytest.raises(ValueError, match="hard deadline"):
        coordinator.submit("round-1", replace(scored, request=claimed), q0_scores, q0_scores)
    coordinator.abort_round("round-1", reason="watchdog reported hard deadline")
    state = coordinator.snapshot()
    assert state["budget"]["devices"]["gooo"]["active_lease_id"] == claimed.lease_id
    assert state["budget"]["used_seconds"] == 11
    assert state["completed_groups"] == 0
    budget.release(claimed.lease_id, reason="fixture supervisor acknowledged stop")


def test_changed_checkpoint_file_cannot_be_accepted(tmp_path):
    coordinator, q0, _, _, _ = setup(tmp_path)
    planned, scored = assignment(coordinator, tmp_path, q0)
    coordinator.begin_round("round-1", (planned,))
    claimed = coordinator.claim("round-1", "candidate", "gooo", reserve_seconds=10)
    q0_scores = reference(q0, planned)
    coordinator.submit("round-1", replace(scored, request=claimed), q0_scores, q0_scores)
    screened = coordinator.screen("round-1")
    decision = DevelopmentConfirmation(
        "round-1", "candidate", planned.center.digest, screened["sha256"], True, "f" * 64
    )
    from pathlib import Path

    Path(planned.center.artifacts[0].path).write_bytes(b"different model bytes")
    with pytest.raises(ValueError, match="content changed"):
        coordinator.finish_round("round-1", decision)
    assert coordinator.snapshot()["center_version"] == 0


def test_rejected_payload_cannot_resit_under_a_new_candidate_name(tmp_path):
    coordinator, q0, _, budget, _ = setup(tmp_path)
    planned, scored = assignment(coordinator, tmp_path, q0)
    coordinator.begin_round("round-1", (planned,))
    claimed = coordinator.claim("round-1", "candidate", "gooo", reserve_seconds=10)
    q0_scores = reference(q0, planned)
    coordinator.submit("round-1", replace(scored, request=claimed), q0_scores, q0_scores)
    screened = coordinator.screen("round-1")
    failed = DevelopmentConfirmation(
        "round-1", "candidate", planned.center.digest, screened["sha256"], False, "f" * 64
    )
    coordinator.finish_round("round-1", failed)
    budget.release(claimed.lease_id, reason="fixture stopped")
    retry, _ = assignment(
        coordinator, tmp_path, q0, "renamed-candidate", candidate_digest=planned.center.digest
    )
    with pytest.raises(ValueError, match="failed payload"):
        coordinator.begin_round("round-2", (retry,))


def test_consumed_hard_limit_stops_before_registering_another_round(tmp_path):
    coordinator, q0, _, budget, now = setup(tmp_path)
    planned, _ = assignment(coordinator, tmp_path, q0)
    lease = budget.reserve("use-device-cap", "06", "gooo", 28_800, run_id="fixture-run")
    now[0] += 28_800
    budget.release(lease.lease_id, reason="fixture stopped at cap")
    with pytest.raises(ValueError, match="budget"):
        coordinator.begin_round("round-1", (planned,))
    state = coordinator.snapshot()
    assert state["status"] == "stopped"
    assert state["stop_reason"] == "device_budget_exhausted"
    assert state["proposal_cursor"] == 0


def test_recovered_assignment_accepts_absent_optional_exposure_log(tmp_path):
    import json

    coordinator, q0, _, _, _ = setup(tmp_path)
    planned, _ = assignment(coordinator, tmp_path, q0)
    coordinator.begin_round("legacy-json", (planned,))
    # Simulate an earlier JSON archive whose optional exposure log was omitted.
    with sqlite3.connect(coordinator.path) as db:
        row = db.execute("SELECT planned FROM work WHERE round_id=?", ("legacy-json",)).fetchone()
        payload = json.loads(row[0])
        payload["manifest"].pop("exposure_log")
        db.execute(
            "UPDATE work SET planned=? WHERE round_id=?", (json.dumps(payload), "legacy-json")
        )
    claimed = coordinator.claim("legacy-json", "candidate", "gooo", reserve_seconds=10)
    assert claimed.request_digest == planned.request.request_digest
    assert claimed.lease_id
