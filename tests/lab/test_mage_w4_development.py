"""Development preregistration and closeout; all observations here are fixtures."""

from dataclasses import asdict, replace

import pytest

from shadow_lab.mage_w4_search.contracts import (
    ArtifactIdentity,
    CandidateRequest,
    DataManifest,
    PromptRecord,
    digest_json,
)
from shadow_lab.mage_w4_search.development import (
    close_scale_registration,
    close_screen_registration,
    decode_calibration_report,
    register_scale_plan,
    register_screen_plan,
    select_development_samples,
    verify_evaluation_job,
)
from shadow_lab.mage_w4_search.ocr import (
    CalibrationPlan,
    EngineIdentity,
    HumanTranscription,
    OCRConfig,
    RawSegment,
    compare_ocr,
    recognize,
)
from shadow_lab.mage_w4_search.rewards import SampleScores, ScoredBatch


def fixture_manifest():
    return DataManifest(
        tuple(
            PromptRecord(
                f"{purpose}-{i:02}",
                f"fixture {purpose} {i}",
                f"{purpose}-source-{i}",
                "development",
                (purpose,),
                ("OPEN",) if i < (5 if purpose == "scale_calibration" else 4) else (),
                i < (5 if purpose == "scale_calibration" else 4),
            )
            for purpose in ("scale_calibration", "candidate_development")
            for i in range(35)
        )
    )


SELECTION = {"text_prompts": 5, "nontext_prompts": 27, "seeds": [20260909, 20260910]}
RELIABILITY = {
    "min_prompts": 32,
    "min_ocr_prompts": 5,
    "min_source_groups": 5,
    "maximum_leave_group_out_relative_change": 0.5,
}


def q0_request(manifest, samples):
    return CandidateRequest(
        request_id="fixture-request",
        run_id="fixture-scale",
        ticket_id="05",
        center_version=0,
        center_digest="a" * 64,
        q0_digest="a" * 64,
        candidate_id="q0",
        candidate_digest="a" * 64,
        group_id=0,
        mutation_seed=0,
        effective_edits=0,
        samples=samples,
        purpose="scale_calibration",
        device_id="gooo",
        dependencies=tuple(
            ArtifactIdentity(role, "b" * 64, 1)
            for role in (
                "quantized_packed",
                "quantized_remainder",
                "weight_scales",
                "layer_manifest",
                "text_encoder",
                "vae",
                "clip_scorer",
                "hps_scorer",
                "ocr_scorer",
            )
        ),
        runtime_identity={
            "software": "fixture",
            "kernel": "fixture",
            "sampler": "fixture",
            "generation": {"steps": 4},
            "device": "gooo",
        },
        manifest_digest=manifest.digest,
    )


def test_selection_is_deterministic_and_one_source_per_prompt():
    manifest = fixture_manifest()
    samples = select_development_samples(manifest, "scale_calibration", **SELECTION)
    assert len(samples) == 64
    assert len({sample.source_group for sample in samples}) == 32
    assert samples[:2][0].seed == 20260909
    assert samples[:2][1].seed == 20260910
    assert samples == select_development_samples(
        replace(manifest, records=tuple(reversed(manifest.records))),
        "scale_calibration",
        **SELECTION,
    )


@pytest.mark.parametrize("complete_text", [False, True])
def test_retired_calibration_records_do_not_change_active_selection(complete_text):
    manifest = fixture_manifest()
    expected = select_development_samples(manifest, "scale_calibration", **SELECTION)
    retired = PromptRecord(
        "000-retired",
        "Earlier retired calibration",
        "retired-source",
        "development",
        ("scale_calibration",),
        ("OLD",) if complete_text else (),
        complete_text,
        retired=True,
    )
    successor = replace(manifest, records=(retired, *manifest.records))
    samples = select_development_samples(successor, "scale_calibration", **SELECTION)
    assert len(samples) == 64
    assert all(sample.prompt_id != retired.prompt_id for sample in samples)
    assert samples == expected


def test_registration_freezes_complete_q0_and_axis_source_counts_before_scores():
    manifest = fixture_manifest()
    samples = select_development_samples(manifest, "scale_calibration", **SELECTION)
    request = q0_request(manifest, samples)
    registration = register_scale_plan(
        request,
        manifest,
        selection=SELECTION,
        reliability=RELIABILITY,
        registered_at_utc="2026-09-09T00:00:00Z",
    )
    assert registration["request_digest"] == request.request_digest
    assert registration["source_group_counts"] == {"clip": 32, "hps": 32, "ocr": 5}
    assert registration["plan"]["min_source_groups"] == 5
    assert registration["formal_execution_authorized"] is False
    with pytest.raises(ValueError, match="OCR source"):
        register_scale_plan(
            request,
            manifest,
            selection=SELECTION,
            reliability={**RELIABILITY, "min_source_groups": 32},
            registered_at_utc="2026-09-09T00:00:00Z",
        )


def test_overlapping_text_sources_are_rejected_before_selection():
    manifest = fixture_manifest()
    rows = list(manifest.records)
    rows[1] = replace(rows[1], source_group=rows[0].source_group)
    with pytest.raises(ValueError, match="independent"):
        select_development_samples(
            replace(manifest, records=tuple(rows)), "scale_calibration", **SELECTION
        )


def test_public_ocr_report_decoder_preserves_its_typed_sha():
    configs = tuple(
        OCRConfig(EngineIdentity(name, "fixture", (("model", "b" * 64),)))
        for name in ("PaddleOCR", "EasyOCR")
    )
    image = b"fixture image"
    import hashlib

    image_sha = hashlib.sha256(image).hexdigest()
    plan = CalibrationPlan((image_sha,), configs, ("c" * 64,), "2026-01-01T00:00:00Z", 0.25)
    human = HumanTranscription(
        image_sha, ("OPEN",), "human_image_transcription", "fixture-human", "d" * 64
    )
    observations = [
        recognize(image, config, lambda *_: [RawSegment("OPEN", (0, 0, 20, 10))])
        for config in configs
    ]
    report = compare_ocr(plan, [human], observations, execution_kind="fixture")
    decoded = decode_calibration_report(asdict(report))
    assert decoded == report
    assert decoded.sha256 == report.sha256
    assert decoded.sha256 != digest_json(asdict(report))


def scores_for(request, manifest, *, clip_change=0, ocr_constant=None):
    prompts = sorted({row.prompt_id for row in request.samples})
    text_ids = {row.prompt_id for row in manifest.records if row.complete_target_inventory}
    rows = []
    for sample in request.samples:
        index = prompts.index(sample.prompt_id)
        ocr = (
            None
            if sample.prompt_id not in text_ids
            else (index / 5 if ocr_constant is None else ocr_constant)
        )
        rows.append(
            SampleScores(
                sample.sample_id,
                digest_json((request.candidate_id, sample.sample_id)),
                index + clip_change,
                index * 2,
                ocr,
                "not_applicable" if ocr is None else "ok",
                "d" * 64,
            )
        )
    return ScoredBatch(
        request,
        manifest,
        tuple(rows),
        tuple(prompt for prompt in prompts if prompt in text_ids),
        (("clip", "b" * 64), ("hps", "b" * 64), ("ocr", "b" * 64)),
        "e" * 64,
        "2026-09-09T01:00:00Z",
        "fixture",
    )


def scale_case():
    manifest = fixture_manifest()
    request = q0_request(
        manifest, select_development_samples(manifest, "scale_calibration", **SELECTION)
    )
    registration = register_scale_plan(
        request,
        manifest,
        selection=SELECTION,
        reliability=RELIABILITY,
        registered_at_utc="2026-09-09T00:00:00Z",
    )
    return manifest, request, registration


def test_scale_closeout_is_unreliable_without_any_epsilon_or_axis_deletion():
    manifest, request, registration = scale_case()
    receipt = close_scale_registration(registration, scores_for(request, manifest, ocr_constant=0))
    assert receipt.status == "unreliable"
    assert receipt.scales is None
    assert "ocr:nonpositive-scale" in receipt.reasons
    assert receipt.prompt_counts == {"clip": 32, "hps": 32, "ocr": 5}


def test_registration_request_identity_cannot_change_when_scores_arrive():
    manifest, request, registration = scale_case()
    altered = replace(request, run_id="unregistered-run")
    with pytest.raises(ValueError, match="registered request"):
        close_scale_registration(registration, scores_for(altered, manifest))


def screen_case():
    manifest, request, registration = scale_case()
    scales = close_scale_registration(registration, scores_for(request, manifest))
    selection = {"text_prompts": 4, "nontext_prompts": 4, "seeds": [20260909, 20260910]}
    q0 = replace(
        request,
        request_id="reference",
        run_id="reference",
        purpose="candidate_development",
        samples=select_development_samples(manifest, "candidate_development", **selection),
    )
    candidates = [
        replace(
            q0,
            request_id=f"candidate-{index}",
            run_id=f"candidate-{index}",
            candidate_id=f"candidate-{index}",
            candidate_digest=str(index + 1) * 64,
            mutation_seed=index,
            effective_edits=count,
        )
        for index, count in enumerate((1024, 65536))
    ]
    plan = register_screen_plan(
        scales,
        candidates,
        {"gooo": q0},
        manifest,
        selection=selection,
        candidate_order=("candidate-1", "candidate-0"),
        registered_at_utc="2026-09-09T00:30:00Z",
    )
    return scales, q0, candidates, manifest, plan


def test_screen_keeps_both_axes_references_and_preregistered_tie_order():
    scales, q0, candidates, manifest, plan = screen_case()
    batches = [scores_for(row, manifest, clip_change=0.5) for row in candidates]
    report = close_screen_registration(plan, scales, batches, {"gooo": scores_for(q0, manifest)})
    assert report.selected_candidate_id == "candidate-1"
    assert report.independently_confirmed is False
    assert report.stage == "development_screening"
    assert all(
        row.current.reference_batch_sha256 == row.q0.reference_batch_sha256
        for row in report.candidates
    )


def test_screen_retains_bad_candidate_and_refuses_missing_planned_result():
    scales, q0, candidates, manifest, plan = screen_case()
    batches = [
        scores_for(candidates[0], manifest, clip_change=-100),
        scores_for(candidates[1], manifest, clip_change=0.5),
    ]
    refs = {"gooo": scores_for(q0, manifest)}
    report = close_screen_registration(plan, scales, batches, refs)
    assert len(report.candidates) == 2
    bad = next(row for row in report.candidates if row.candidate_id == "candidate-0")
    assert bad.feasible is False and bad.current.failed_axes == ("clip",)
    with pytest.raises(ValueError, match="exactly cover"):
        close_screen_registration(plan, scales, batches[:1], refs)


def test_scale_registration_json_roundtrip_preserves_replay_and_rejects_tampering(tmp_path):
    import json

    manifest, request, registration = scale_case()
    batch = scores_for(request, manifest)
    path = tmp_path / "registration.json"
    path.write_text(json.dumps(registration))
    stored = json.loads(path.read_text())
    assert close_scale_registration(stored, batch) == close_scale_registration(registration, batch)
    stored["selection"]["nontext_prompts"] = 26
    with pytest.raises(ValueError, match="identity mismatch"):
        close_scale_registration(stored, batch)


def test_screen_registration_json_roundtrip_preserves_selection(tmp_path):
    import json

    scales, q0, candidates, manifest, registration = screen_case()
    batches = [scores_for(row, manifest, clip_change=0.5) for row in candidates]
    references = {"gooo": scores_for(q0, manifest)}
    path = tmp_path / "screen-registration.json"
    path.write_text(json.dumps(registration))
    stored = json.loads(path.read_text())
    actual = close_screen_registration(stored, scales, batches, references)
    assert actual == close_screen_registration(registration, scales, batches, references)
    assert actual.selected_candidate_id == "candidate-1"


def complete_fixture_job(tmp_path, configure=None):
    """Exercise the actual complete worker with only its external models replaced."""
    import json

    from test_mage_w4_evaluation_worker import FixtureScorer, fixture_generator, fixture_job

    from shadow_lab.mage_qad_checkpoint import content_hash
    from shadow_lab.mage_w4_search.evaluation_worker import run_evaluation_job

    manifest, shared = fixture_job(tmp_path)
    if configure is not None:
        configure(manifest, shared)
    observed = []
    external = fixture_generator(shared, observed)

    def generation(request, manifest, directory):
        report = external(request, manifest, directory)
        prompts = {row.prompt_id: row.prompt for row in manifest.records}
        report["request_sha256"] = content_hash(
            [
                {"sample_id": row.sample_id, "prompt": prompts[row.prompt_id], "seed": row.seed}
                for row in request.samples
            ]
        )
        (directory / "report.json").write_text(json.dumps(report))
        return report

    result = run_evaluation_job(
        tmp_path,
        manifest,
        device_id="gooo",
        fixture_generation=generation,
        fixture_scorers=(
            FixtureScorer(0.3, observed),
            FixtureScorer(0.2, observed),
            lambda *_: (RawSegment("OPEN", (0, 0, 2, 1)),),
        ),
    )
    lease = json.loads((tmp_path / "lease.json").read_text())
    stopped = result["finished_at"] + 0.01
    stop = {
        **lease,
        "started_at": result["started_at"],
        "stopped_at": stopped,
        "observed_at": stopped,
        "exit_code": 0,
        "stop_confirmed": True,
        "process_group": 999,
    }
    files = {
        "clock.json": {
            "at": lease["started_at"] - 0.1,
            "offset_seconds": 0,
            "uncertainty_seconds": 0,
        },
        "stopped.json": stop,
        "independent-group-check.json": {
            "process_group": 999,
            "absent": True,
            "checked_at": stopped + 0.01,
            "device_id": "gooo",
            "lease_id": lease["lease_id"],
        },
    }
    for name, value in files.items():
        (tmp_path / name).write_text(json.dumps(value))
    seconds = stopped + 0.02 - lease["started_at"]
    snapshot = {
        "campaign_id": "fixture",
        "at": stopped + 0.03,
        "leases": [{**lease, "stopped_at": stopped + 0.02}],
        "used_seconds": seconds,
        "devices": {"gooo": {"used_seconds": seconds}, "goooinext": {"used_seconds": 0}},
        "timing_events": [],
    }
    prepared = json.loads((tmp_path / "prepared.json").read_text())
    request = CandidateRequest.from_dict(prepared["request"])
    return request, manifest, snapshot


def test_verified_job_replays_real_module_formats_without_claiming_device_evidence(tmp_path):
    request, manifest, snapshot = complete_fixture_job(tmp_path)
    batch, evidence = verify_evaluation_job(
        tmp_path, request, manifest, snapshot, execution_kind="fixture"
    )
    assert batch.execution_kind == "fixture"
    assert batch.samples[0].ocr == 1
    assert evidence["occupied_seconds"] == snapshot["used_seconds"]
    assert evidence["batch_sha256"] == batch.sha256
    assert evidence["formal_execution_authorized"] is False


def test_closed_evaluation_rejects_changed_score_receipt_and_active_lease(tmp_path):
    import json

    request, manifest, snapshot = complete_fixture_job(tmp_path)
    snapshot["leases"][0]["stopped_at"] = None
    snapshot["at"] = snapshot["leases"][0]["started_at"] + snapshot["used_seconds"]
    with pytest.raises(ValueError, match="closed"):
        verify_evaluation_job(tmp_path, request, manifest, snapshot, execution_kind="fixture")
    snapshot["leases"][0]["stopped_at"] = snapshot["at"]
    path = tmp_path / "scores" / (request.samples[0].sample_id + ".json")
    score = json.loads(path.read_text())
    score["clip"] = 100
    path.write_text(json.dumps(score))
    with pytest.raises(ValueError, match="score receipt"):
        verify_evaluation_job(tmp_path, request, manifest, snapshot, execution_kind="fixture")


def test_cli_selection_and_registration_preserve_frozen_inputs(tmp_path):
    import json
    import os
    import subprocess
    import sys
    from pathlib import Path

    manifest, request, _ = scale_case()
    root = Path(__file__).resolve().parents[2]
    manifest_path, prepared_path = tmp_path / "manifest.json", tmp_path / "prepared.json"
    manifest_path.write_text(json.dumps(asdict(manifest)))
    prepared_path.write_text(json.dumps({"request": asdict(request)}))
    input_path = tmp_path / "inputs.json"
    shared = {
        "manifest_path": str(manifest_path),
        "execution_kind": "fixture",
        "selection": SELECTION,
    }
    input_path.write_text(json.dumps({**shared, "purpose": "scale_calibration"}))
    command = [sys.executable, str(root / "scripts/mage_w4_development.py")]
    environment = {**os.environ, "PYTHONPATH": str(root / "src")}
    subprocess.run(
        [*command, "select", "--inputs", str(input_path), "--output", str(tmp_path / "selection")],
        env=environment,
        cwd=root,
        text=True,
        capture_output=True,
        check=True,
    )
    selected = json.loads((tmp_path / "selection/selection.json").read_text())
    assert selected["samples"] == json.loads(json.dumps([asdict(row) for row in request.samples]))
    input_path.write_text(
        json.dumps(
            {
                **shared,
                "prepared_path": str(prepared_path),
                "reliability": RELIABILITY,
                "registered_at_utc": "2026-09-09T00:00:00Z",
            }
        )
    )
    output = tmp_path / "registration"
    arguments = [*command, "register-scale", "--inputs", str(input_path), "--output", str(output)]
    subprocess.run(arguments, env=environment, cwd=root, text=True, capture_output=True, check=True)
    registration = json.loads((output / "registration.json").read_text())
    assert registration["source_group_counts"] == {"clip": 32, "hps": 32, "ocr": 5}
    assert registration["request_digest"] == request.request_digest
    assert json.loads((output / "summary.json").read_text())["formal_execution_authorized"] is False
    retry = subprocess.run(arguments, env=environment, cwd=root, text=True, capture_output=True)
    assert retry.returncode != 0 and "output directory exists" in retry.stderr


def test_screen_registration_rejects_sampler_drift_before_generating_candidates():
    scales, q0, candidates, manifest, _ = screen_case()
    changed = {**q0.runtime_identity, "generation": {"steps": 8}}
    q0 = replace(q0, runtime_identity=changed)
    candidates = [replace(row, runtime_identity=changed) for row in candidates]
    with pytest.raises(ValueError, match="calibrated generation"):
        register_screen_plan(
            scales,
            candidates,
            {"gooo": q0},
            manifest,
            selection={"text_prompts": 4, "nontext_prompts": 4, "seeds": [20260909, 20260910]},
            candidate_order=("candidate-0", "candidate-1"),
            registered_at_utc="2026-09-09T00:30:00Z",
        )
