"""Full quantized centers bridge byte archives and authorized development work."""

import json
from dataclasses import asdict

import pytest

from shadow_lab.mage_w4_search.center_execution import (
    load_execution_center,
    load_full_center,
    map_center_archive,
    materialize_candidate,
    prepare_center_job,
    prepare_center_payload,
    register_q0_center,
)


def test_saved_candidate_can_become_the_next_actual_packed_center(tmp_path):
    pytest.importorskip("torch")
    from test_mage_w4_mutation import _assets

    from shadow_lab.mage_w4_search.mutation import build_layer_groups, propose_candidate
    from shadow_lab.mage_w4_search.native import save_quantized_assets

    original = _assets()
    source = save_quantized_assets(original, tmp_path / "input")
    q0 = register_q0_center(
        source["packed_path"],
        source["remainder_path"],
        tmp_path / "q0",
        allowed_assets=source["file_sha256"],
        expected_native_sha256=original.sha256,
    )
    first = propose_candidate(
        original,
        build_layer_groups(original),
        group_id="block_00",
        mutation_count=1,
        seed=10,
        candidate_index=0,
    )
    current = materialize_candidate(first.assets, tmp_path / "current", fixed_center=q0)
    restored = load_full_center(current)
    assert restored.sha256 == first.assets.sha256 != original.sha256
    second = propose_candidate(
        restored,
        build_layer_groups(restored),
        group_id="block_01",
        mutation_count=1,
        seed=11,
        candidate_index=0,
    )
    candidate = materialize_candidate(second.assets, tmp_path / "candidate", fixed_center=q0)
    assert load_full_center(candidate).sha256 == second.assets.sha256
    assert second.receipt["base_sha256"] == current.digest
    assert load_full_center(q0).sha256 == original.sha256

    def fixed(center):
        return [asdict(row) for row in center.artifacts if row.role != "quantized_packed"]

    assert fixed(q0) == fixed(current) == fixed(candidate)
    assert not (tmp_path / "candidate/remainder.pt").exists()
    with pytest.raises(FileExistsError):
        materialize_candidate(second.assets, tmp_path / "candidate", fixed_center=q0)


def test_current_candidate_and_q0_reference_bind_real_files_and_keep_one_runtime(tmp_path):
    from test_mage_w4_coordinator import center
    from test_mage_w4_evaluation_worker import request_inputs

    from shadow_lab.mage_w4_search.contracts import CandidateRequest
    from shadow_lab.mage_w4_search.rewards import fixed_generation_identity

    plan, manifest, shared = request_inputs()
    q0 = center(tmp_path, "q0", "a" * 64)
    current = center(tmp_path, "current", "b" * 64, fixed=q0)
    proposal = center(tmp_path, "candidate", "c" * 64, fixed=q0)
    plan.update(schema="mage-w4-center-plan/v1", ticket_id="06", center_version=2)
    common = dict(
        shared=shared, q0=q0, current=current, template={"path": "fixture", "sha256": "e" * 64}
    )
    references = []
    for kind, selected in (("q0", q0), ("current", current)):
        plan["candidate"] = {"kind": kind, "candidate_id": kind, "group_id": 1}
        prepared = prepare_center_job(kind, plan, manifest, evaluated=selected, **common)
        request = CandidateRequest.from_dict(prepared["request"])
        assert (request.center_version, request.center_digest, request.q0_digest) == (
            2,
            "b" * 64,
            "a" * 64,
        )
        assert request.candidate_digest == selected.digest and request.ticket_id == "06"
        assert (
            next(row.sha256 for row in request.dependencies if row.role == "quantized_packed")
            == selected.artifacts[0].sha256
        )
        assert all(row.path is None for row in request.dependencies)
        references.append(request)
    plan["candidate"] = {
        "kind": "adjacent",
        "candidate_id": "next",
        "group_id": 1,
        "mutation_count": 1,
        "seed": 14,
        "candidate_index": 0,
        "max_attempts": 2,
    }
    receipt = {
        "status": "candidate",
        "base_sha256": current.digest,
        "candidate_sha256": proposal.digest,
        "fixed_sha256": shared["fixed_state_sha256"],
        "group_id": "block_01",
        "actual_changes": 1,
        "requested_changes": 1,
        "seed": 14,
        "candidate_index": 0,
        "max_attempts": 2,
    }
    prepared = prepare_center_job(
        "next", plan, manifest, evaluated=proposal, candidate_receipt=receipt, **common
    )
    request = CandidateRequest.from_dict(prepared["request"])
    assert request.candidate_digest == proposal.digest and request.center_digest == current.digest
    assert request.runtime_identity == shared["runtime_identity"]
    assert fixed_generation_identity(references[0]) == fixed_generation_identity(request)
    with pytest.raises(ValueError, match="receipt|payload"):
        prepare_center_job(
            "wrong", plan, manifest, evaluated=q0, candidate_receipt=receipt, **common
        )


def test_versioned_complete_worker_uses_frozen_center_job_and_original_grant(tmp_path):
    from test_mage_w4_coordinator import center
    from test_mage_w4_evaluation_worker import FixtureScorer, fixture_generator, fixture_job

    from shadow_lab.mage_w4_search.contracts import CandidateRequest
    from shadow_lab.mage_w4_search.evaluation_worker import run_evaluation_job
    from shadow_lab.mage_w4_search.ocr import RawSegment

    manifest, shared = fixture_job(tmp_path)
    old = json.loads((tmp_path / "prepared.json").read_text())
    q0 = center(tmp_path, "q0", "a" * 64)
    current = center(tmp_path, "current", "b" * 64, fixed=q0)
    plan = {
        **old["evaluation_job"]["plan"],
        "schema": "mage-w4-center-plan/v1",
        "ticket_id": "08",
        "center_version": 3,
        "candidate": {"kind": "current", "candidate_id": "current", "group_id": 2},
    }
    prepared = prepare_center_job(
        old["request"]["run_id"],
        plan,
        manifest,
        shared=shared,
        q0=q0,
        current=current,
        evaluated=current,
        template={"path": "fixture", "sha256": "e" * 64},
    )
    lease = json.loads((tmp_path / "lease.json").read_text())
    lease["ticket_id"] = "08"
    granted = {**prepared["request"], "lease_id": lease["lease_id"]}
    grant = {**prepared["preflight"], "status": "authorized", "request": granted, "lease": lease}
    for name, value in (("prepared.json", prepared), ("lease.json", lease), ("grant.json", grant)):
        (tmp_path / name).write_text(json.dumps(value))
    seen = []
    result = run_evaluation_job(
        tmp_path,
        manifest,
        device_id="gooo",
        fixture_generation=fixture_generator(shared, seen),
        fixture_scorers=(
            FixtureScorer(0.3, seen),
            FixtureScorer(0.2, seen),
            lambda image, config: (RawSegment("OPEN", (0.0, 0.0, 2.0, 1.0)),),
        ),
    )
    assert result["status"] == "ok" and result["execution_kind"] == "fixture"
    scored = json.loads((tmp_path / "scores/batch.json").read_text())
    request = CandidateRequest.from_dict(scored["request"])
    assert (request.ticket_id, request.center_version, request.candidate_digest) == (
        "08",
        3,
        "b" * 64,
    )
    assert request.lease_id == lease["lease_id"]


def test_worker_loads_materialized_current_candidate_and_checks_receipt_replay(tmp_path):
    pytest.importorskip("torch")
    import hashlib

    from test_mage_w4_evaluation_worker import request_inputs
    from test_mage_w4_mutation import _assets

    from shadow_lab.mage_w4_search.center_execution import decode_full_center
    from shadow_lab.mage_w4_search.contracts import CandidateRequest
    from shadow_lab.mage_w4_search.native import save_quantized_assets

    original = _assets()
    saved = save_quantized_assets(original, tmp_path / "input")
    q0 = register_q0_center(
        saved["packed_path"],
        saved["remainder_path"],
        tmp_path / "q0",
        allowed_assets=saved["file_sha256"],
        expected_native_sha256=original.sha256,
    )
    plan, manifest, shared = request_inputs()
    shared.update(q0_digest=original.sha256, fixed_state_sha256=original.fixed_sha256)
    plan.update(schema="mage-w4-center-plan/v1", ticket_id="06", center_version=0)
    plan["candidate"] = {
        "kind": "adjacent",
        "candidate_id": "first",
        "group_id": 0,
        "mutation_count": 1,
        "seed": 11,
        "candidate_index": 0,
        "max_attempts": 2,
    }
    first, _ = prepare_center_payload(plan, q0=q0, current=q0, directory=tmp_path / "first")
    plan.update(center_version=1)
    plan["candidate"].update(candidate_id="second", group_id=1, seed=12)
    second, receipt = prepare_center_payload(
        plan, q0=q0, current=first, directory=tmp_path / "second"
    )
    origin = tmp_path / "template.json"
    origin.write_text("{}")
    prepared = prepare_center_job(
        "second",
        plan,
        manifest,
        shared=shared,
        q0=q0,
        current=first,
        evaluated=second,
        candidate_receipt=receipt,
        template={"path": str(origin), "sha256": hashlib.sha256(origin.read_bytes()).hexdigest()},
    )
    request = CandidateRequest.from_dict(prepared["request"])
    seen = []
    loaded = load_execution_center(prepared["evaluation_job"], request, seen.append)
    assert loaded.sha256 == second.digest != first.digest != q0.digest
    assert receipt["base_sha256"] == first.digest
    assert "candidate_replay" in seen
    assert (
        decode_full_center(prepared["evaluation_job"]["centers"]["evaluated"]).digest
        == loaded.sha256
    )
    prepared["evaluation_job"]["candidate_receipt"]["base_sha256"] = q0.digest
    with pytest.raises(ValueError):
        load_execution_center(prepared["evaluation_job"], request, seen.append)


def test_coordinator_archive_maps_paths_without_changing_bytes_or_native_identity(tmp_path):
    import shutil

    from test_mage_w4_coordinator import center

    remote = center(tmp_path, "remote", "a" * 64)
    local = tmp_path / "archive"
    local.mkdir()
    paths = {}
    for row in remote.artifacts:
        path = local / row.role
        shutil.copyfile(row.path, path)
        paths[row.role] = str(path)
    mapped = map_center_archive(remote, paths)
    assert mapped.digest == remote.digest
    assert mapped.artifact_manifest_digest == remote.artifact_manifest_digest
    assert {row.path for row in mapped.artifacts}.isdisjoint(row.path for row in remote.artifacts)
    (local / "quantized_packed").write_bytes(b"changed archive")
    with pytest.raises(ValueError, match="content changed"):
        map_center_archive(remote, paths)


@pytest.mark.parametrize("fails", [False, True])
def test_center_prepare_cli_retains_failed_jobs_and_cpu_timing(tmp_path, monkeypatch, fails):
    import importlib.util
    import sys
    from pathlib import Path

    script = Path(__file__).parents[2] / "scripts/mage_w4_center.py"
    spec = importlib.util.spec_from_file_location("center_cli_fixture", script)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    monkeypatch.setattr(cli.platform, "system", lambda: "Linux")
    monkeypatch.setattr(cli.platform, "machine", lambda: "aarch64")

    def boundary(*args, **kwargs):
        if fails:
            raise ValueError("fixture missing current payload")
        return {
            "evaluation_job": {
                "candidate_receipt": None,
                "centers": {"evaluated": {"fixture": True}},
            }
        }

    monkeypatch.setattr(cli, "prepare_center_on_gb10", boundary)
    plan = tmp_path / "plan.json"
    plan.write_text("{}")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(script),
            "prepare",
            "--run-root",
            str(tmp_path),
            "--job-id",
            "center-fixture",
            "--device-id",
            "gooo",
            "--template",
            str(plan),
            "--plan",
            str(plan),
            "--q0",
            str(plan),
            "--current",
            str(plan),
        ],
    )
    if fails:
        with pytest.raises(ValueError, match="missing current payload"):
            cli.main()
    else:
        cli.main()
    job = tmp_path / "jobs/center-fixture"
    timing = json.loads((job / "preparation-timing.json").read_text())
    assert timing["status"] == ("failed" if fails else "prepared")
    assert timing["host_cpu_seconds"] >= 0 and timing["wall_seconds"] >= 0
    assert timing["finished_at"] >= timing["started_at"]
    assert not (job / "lease.json").exists()
    assert not tuple(tmp_path.rglob("*.sqlite3"))
    with pytest.raises(FileExistsError):
        cli.main()


@pytest.mark.parametrize("ticket", ["06", "08"])
def test_center_complete_worker_batch_passes_full_closed_job_verifier(tmp_path, ticket):
    from test_mage_w4_coordinator import center
    from test_mage_w4_development import complete_fixture_job

    from shadow_lab.mage_w4_search.development import verify_evaluation_job

    def configure(manifest, shared):
        old = json.loads((tmp_path / "prepared.json").read_text())
        q0 = center(tmp_path, "q0", "a" * 64)
        current = center(tmp_path, "current", "b" * 64, fixed=q0)
        plan = {
            **old["evaluation_job"]["plan"],
            "schema": "mage-w4-center-plan/v1",
            "ticket_id": ticket,
            "center_version": 3,
            "candidate": {"kind": "current", "candidate_id": "current", "group_id": 2},
        }
        prepared = prepare_center_job(
            old["request"]["run_id"],
            plan,
            manifest,
            shared=shared,
            q0=q0,
            current=current,
            evaluated=current,
            template={"path": "fixture", "sha256": "e" * 64},
        )
        lease = json.loads((tmp_path / "lease.json").read_text())
        lease["ticket_id"] = ticket
        grant = {
            **prepared["preflight"],
            "status": "authorized",
            "lease": lease,
            "request": {**prepared["request"], "lease_id": lease["lease_id"]},
        }
        for name, value in (
            ("prepared.json", prepared),
            ("lease.json", lease),
            ("grant.json", grant),
        ):
            (tmp_path / name).write_text(json.dumps(value))

    request, manifest, snapshot = complete_fixture_job(tmp_path, configure=configure)
    batch, proof = verify_evaluation_job(
        tmp_path, request, manifest, snapshot, execution_kind="fixture"
    )
    assert batch.request.ticket_id == ticket and batch.request.center_version == 3
    assert proof["request_digest"] == request.request_digest
    assert proof["formal_execution_authorized"] is False
