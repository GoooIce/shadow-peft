"""File adapter behavior with explicitly fabricated observations; no GB10 work."""

import json
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

import pytest

from shadow_lab.mage_w4_search import group_execution as groups


def setup_grid():
    from test_mage_w4_rewards import frozen_scales, screening_batch

    reference = screening_batch("q0", (1.0, 2.0, 0.5))
    config = {
        "selection": {"text_prompts": 3, "nontext_prompts": 0, "seeds": [1]},
        "edit_counts": [1, 16],
        "proposals_per_count": 1,
        "max_attempts": 8,
        "random_seed": 20260909,
        "group_devices": {str(i): "gooo" if i % 2 == 0 else "goooinext" for i in range(13)},
        "execution_kind": "fixture",
    }
    scales = frozen_scales()
    registration = groups.register_execution(
        reference.manifest, scales, config=config, registered_at_utc="2026-01-15T00:00:00Z"
    )
    return reference, scales, registration


def test_grid_routes_original_group_zero_jobs_without_relabeling_after_scores():
    reference, scales, registration = setup_grid()
    assert len(registration["jobs"]) == 26
    assert [row["ticket_id"] for row in registration["jobs"]] == ["05"] * 2 + ["08"] * 24
    assert [row["device_id"] for row in registration["jobs"][:4]] == ["gooo"] * 2 + [
        "goooinext"
    ] * 2
    assert [row["candidate_id"] for row in registration["jobs"][:2]] == ["g00-m1-r0", "g00-m16-r0"]
    roundtrip = json.loads(json.dumps(registration))
    replay = groups.replay_registration(roundtrip, reference.manifest, scales)
    assert replay["sha256"] == registration["sha256"]
    roundtrip["jobs"][0]["ticket_id"] = "08"
    with pytest.raises(ValueError, match="changed"):
        groups.replay_registration(roundtrip, reference.manifest, scales)


def test_grid_does_not_accept_unreliable_or_fixture_scales_as_real():
    reference, scales, registration = setup_grid()
    args = dict(config=registration["config"], registered_at_utc="2026-01-15T00:00:00Z")
    with pytest.raises(ValueError, match="ready"):
        groups.register_execution(reference.manifest, replace(scales, status="unreliable"), **args)
    args["config"] = {**args["config"], "execution_kind": "gb10_measured"}
    with pytest.raises(ValueError, match="measured"):
        groups.register_execution(reference.manifest, scales, **args)


def test_retired_candidate_rows_cannot_be_selected_into_new_grid():
    reference, scales, registration = setup_grid()
    manifest = replace(
        reference.manifest,
        records=(
            replace(reference.manifest.records[0], retired=True),
            *reference.manifest.records[1:],
        ),
    )
    with pytest.raises(ValueError, match="text count"):
        groups.register_execution(
            manifest,
            scales,
            config=registration["config"],
            registered_at_utc="2026-01-15T00:00:00Z",
        )


def test_complete_group_uses_fixed_paired_math_and_counts_attempt_receipt_entries(
    tmp_path, monkeypatch
):
    from test_mage_w4_rewards import screening_batch

    reference, scales, registration = setup_grid()
    batches, prepared, proofs = {}, {}, {}
    reference_dir = tmp_path / "reference"
    reference_dir.mkdir()
    batches[str(reference_dir)] = reference
    prepared[str(reference_dir)] = {"request": asdict(reference.request)}
    proofs[str(reference_dir)] = {"lease_id": "reference", "occupied_seconds": 100, "artifacts": []}
    candidates = {}
    for i, job in enumerate(registration["jobs"][:2]):
        ident, plan = job["candidate_id"], job["plan"]
        directory = tmp_path / ident
        directory.mkdir()
        (directory / "lease.json").write_text(
            json.dumps(
                {"started_at": datetime.fromisoformat("2026-02-01T00:00:00+00:00").timestamp()}
            )
        )
        (directory / "clock.json").write_text('{"offset_seconds":0,"uncertainty_seconds":0}')
        at = datetime.fromisoformat("2026-02-01T00:00:00+00:00").timestamp()
        (directory / "preparation-timing.json").write_text(
            json.dumps(
                {
                    "schema": "mage-w4-evaluation-preparation-timing/v1",
                    "status": "prepared",
                    "job_id": ident,
                    "device_id": "gooo",
                    "started_at": at - 2,
                    "finished_at": at - 1,
                }
            )
        )
        candidate = screening_batch(
            ident, (1.1 + i * 0.1, 2.1, 0.6), candidate_digest=("b" if i == 0 else "d") * 64
        )
        request = replace(
            candidate.request,
            run_id=ident,
            mutation_seed=plan["candidate"]["seed"],
            effective_edits=plan["candidate"]["mutation_count"],
        )
        candidate = replace(candidate, request=request)
        batches[str(directory)] = candidate
        prepared[str(directory)] = {
            "request": asdict(request),
            "evaluation_job": {
                "plan": plan,
                "candidate_receipt": {"attempts": [{"attempt": 0, "status": "candidate"}]},
            },
        }
        proofs[str(directory)] = {"lease_id": ident, "occupied_seconds": 120 + i, "artifacts": []}
        candidates[ident] = str(directory)
    for directory, value in prepared.items():
        (Path(directory) / "prepared.json").write_text(json.dumps(value))
    # Verification itself is exercised by the complete worker integration tests.
    # This boundary fixture isolates wiring into the REAL unchanged paired math.
    monkeypatch.setattr(
        groups,
        "verify_evaluation_job",
        lambda directory, *a, **kw: (batches[str(directory)], proofs[str(directory)]),
    )
    kwargs = dict(
        group_id=0, candidate_jobs=candidates, reference_jobs={"gooo": str(reference_dir)}
    )
    receipt, proof = groups.close_group(registration, reference.manifest, scales, {}, **kwargs)
    assert receipt["status"] == "fixture-selection-only"
    assert receipt["selected_candidate_id"] == "g00-m16-r0"
    assert receipt["counts"]["scored"] == 2
    assert [row["attempts"] for row in receipt["outcomes"]] == [1, 1]
    assert proof["sum_of_referenced_occupancies_seconds"] == 341
    assert proof["reference_occupancy_is_shared_across_groups"] is True
    assert proof["these_are_not_new_charges"] is True
    assert receipt["formal_density_frozen"] is False
    with pytest.raises(ValueError, match="every registered"):
        groups.close_group(
            registration,
            reference.manifest,
            scales,
            {},
            **{
                **kwargs,
                "candidate_jobs": {next(iter(candidates)): next(iter(candidates.values()))},
            },
        )
    first = next(iter(candidates.values()))
    timing_path = Path(first) / "preparation-timing.json"
    original_timing = json.loads(timing_path.read_text())
    timing_path.write_text(json.dumps({**original_timing, "started_at": 0}))
    with pytest.raises(ValueError, match="preparation"):
        groups.close_group(registration, reference.manifest, scales, {}, **kwargs)
    timing_path.write_text(json.dumps(original_timing))
    (Path(first) / "lease.json").write_text('{"started_at":0}')
    with pytest.raises(ValueError, match="predates"):
        groups.close_group(registration, reference.manifest, scales, {}, **kwargs)


def test_cli_failure_keeps_timing_and_does_not_create_a_group_receipt(tmp_path, monkeypatch):
    import importlib.util
    import sys

    path = Path(__file__).resolve().parents[2] / "scripts/mage_w4_group_pilot.py"
    spec = importlib.util.spec_from_file_location("group_cli_fixture", path)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    reference, _, _ = setup_grid()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(asdict(reference.manifest)))
    snapshot = tmp_path / "snapshot.json"
    snapshot.write_text("{}")
    inputs = tmp_path / "inputs.json"
    inputs.write_text(
        json.dumps(
            {
                "execution_kind": "fixture",
                "manifest_path": str(manifest),
                "budget_snapshot_path": str(snapshot),
            }
        )
    )
    output = tmp_path / "closeout"
    monkeypatch.setattr(
        sys, "argv", [str(path), "close-group", "--inputs", str(inputs), "--output", str(output)]
    )

    def failed_scale(*args):
        raise ValueError("fixture missing original scale evidence")

    monkeypatch.setattr(cli, "load_fixed_scale", failed_scale)
    with pytest.raises(ValueError, match="missing original"):
        cli.main()
    summary = json.loads((output / "summary.json").read_text())
    assert summary["status"] == "failed" and summary["creates_lease"] is False
    assert summary["finished_at"] >= summary["started_at"]
    assert not (output / "group-receipt.json").exists()
