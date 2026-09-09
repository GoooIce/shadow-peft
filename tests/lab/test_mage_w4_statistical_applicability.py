"""Actual applicability inputs remain on GB10; these are fixed score fixtures."""

from dataclasses import asdict, replace

import pytest
from test_mage_w4_development import (
    RELIABILITY,
    SELECTION,
    fixture_manifest,
    q0_request,
    scale_case,
    scores_for,
    screen_case,
)

from shadow_lab.mage_w4_search.development import register_scale_plan, select_development_samples
from shadow_lab.mage_w4_search.rewards import SampleScores, ScoredBatch
from shadow_lab.mage_w4_search.statistical_applicability import (
    applicability_report,
    conditional_ocr_gates,
    load_applicability_report,
    read_coverage_evidence,
    support_audit,
)


def scale_inputs():
    manifest = fixture_manifest()
    request = q0_request(
        manifest, select_development_samples(manifest, "scale_calibration", **SELECTION)
    )
    registration = register_scale_plan(
        request,
        manifest,
        selection=SELECTION,
        reliability=RELIABILITY,
        registered_at_utc="2026-09-09T00:00:00+00:00",
    )
    text_ids = tuple(sorted({s.prompt_id for s in request.samples})[:5])
    batch = ScoredBatch(
        replace(request, lease_id="fixture-lease"),
        manifest,
        tuple(
            SampleScores(
                s.sample_id,
                "c" * 64,
                i / 100,
                i / 200,
                0.0 if s.prompt_id in text_ids else None,
                "ok" if s.prompt_id in text_ids else "not_applicable",
                "d" * 64,
            )
            for i, s in enumerate(request.samples)
        ),
        text_ids,
        tuple((axis, "b" * 64) for axis in ("clip", "hps", "ocr")),
        "e" * 64,
        "2026-09-09T00:00:01+00:00",
        "fixture",
    )
    return registration, batch


def test_unreliable_scale_keeps_missing_effects_and_costs_without_inventing_support():
    registration, baseline = scale_inputs()
    report = applicability_report(registration, baseline)
    assert report["status"] == "insufficient-evidence"
    assert report["scale"]["status"] == "unreliable"
    assert report["screen"] is None
    assert report["paired_effects"] is None
    assert report["cost"] is None
    assert report["support"]["axes"]["ocr"]["bound"] == [0, 1]
    assert report["support"]["axes"]["clip"]["bound"] is None
    assert report["support"]["axes"]["hps"]["bound"] is None
    assert report["independence"]["independence_established"] is False
    assert report["independence"]["source_clusters"] == 32
    assert report["independence"]["generation_seeds"] == SELECTION["seeds"]
    assert report["formal_execution_authorized"] is False
    assert report["independently_confirmed"] is False
    assert asdict(baseline)["execution_kind"] == "fixture"


def job_proof(batch):
    return {
        "schema": "mage-w4-development-job-evidence/v1",
        "batch_sha256": batch.sha256,
        "request_digest": batch.request.request_digest,
        "execution_kind": "fixture",
        "lease_id": batch.request.lease_id,
        "occupied_seconds": 640.0,
        "device_id": batch.request.device_id,
        "artifacts": [{"path": "fixture-evidence.json", "sha256": "f" * 64, "size_bytes": 1}],
        "formal_execution_authorized": False,
    }


def test_ocr_gate_is_conditional_not_power_and_shared_cost_is_counted_once():
    registration, baseline = scale_inputs()
    proof = job_proof(baseline)
    report = applicability_report(
        registration,
        baseline,
        job_evidence=[proof, proof],
        risk_scenarios=[{"stage": "confirmation", "attempt_alpha": 0.025}],
    )
    assert report["cost"]["unique_occupied_seconds"] == 640
    gate = report["conditional_gate_diagnostics"][0]
    assert gate["ocr_source_clusters"] == 5
    assert gate["required_empirical_ocr_improvement"] > 1
    assert gate["can_pass_with_any_supported_ocr_difference"] is False
    assert gate["zero_effect_single_prompt_source_requirement"] == 112696
    assert gate["projected_zero_effect_ocr_cost"]["two_models_seconds"] == 4507840
    assert gate["joint_composite_radius"] is None
    assert gate["power"] is None
    assert gate["formal_applicability"] is False
    with pytest.raises(ValueError, match="job evidence"):
        applicability_report(
            registration, baseline, job_evidence=[{**proof, "occupied_seconds": -1}]
        )


def test_complete_screen_keeps_rejected_candidate_and_descriptive_paired_variance():
    manifest, request, registration = scale_case()
    baseline = scores_for(request, manifest)
    _, q0, candidates, _, plan = screen_case()
    batches = [
        scores_for(candidates[0], manifest, clip_change=-100),
        scores_for(candidates[1], manifest, clip_change=0.5),
    ]
    screen = {
        "registration": plan,
        "candidates": batches,
        "references": {"gooo": scores_for(q0, manifest)},
    }
    report = applicability_report(registration, baseline, screen_inputs=screen)
    assert report["screen"]["selected_candidate_id"] == "candidate-1"
    effects = {row["candidate_id"]: row for row in report["paired_effects"]}
    assert effects["candidate-0"]["feasible"] is False
    assert effects["candidate-1"]["current"]["differences"]["clip"] == 0.5
    assert effects["candidate-1"]["current"]["paired_variances"]["clip"] == 0
    assert report["status"] == "insufficient-evidence"
    with pytest.raises(ValueError, match="exactly cover"):
        applicability_report(
            registration, baseline, screen_inputs={**screen, "candidates": batches[:1]}
        )


def test_conditional_radius_matches_public_joint_method_for_unequal_source_clusters():
    from test_mage_w4_statistical import fixture

    from shadow_lab.mage_w4_search.statistical import joint_decision

    protocol, evidence = fixture(count=4, seeds=(1, 2))
    part = protocol.slices[0]
    samples = tuple(
        replace(
            sample,
            source_group="shared" if sample.prompt_id in {"p-0", "p-1"} else sample.source_group,
        )
        for sample in part.samples
    )
    protocol = replace(protocol, slices=(replace(part, samples=samples),))
    evidence = replace(
        evidence,
        rows=tuple(
            replace(row, sample_id=sample.sample_id)
            for row, sample in zip(evidence.rows, samples, strict=True)
        ),
    )
    actual = joint_decision(protocol, evidence, alpha=0.025, stage="confirmation")
    gate = conditional_ocr_gates(
        samples, part.ocr_prompt_ids, [{"stage": "confirmation", "attempt_alpha": 0.025}]
    )[0]
    assert gate["ocr_radius"] == pytest.approx(actual["checks"]["current.ocr"]["radius"])
    assert gate["weight_concentration"] == 0.375
    assert gate["ocr_source_clusters"] == 3


def test_matching_external_source_hash_does_not_prove_floating_support(tmp_path):
    import hashlib

    _, baseline = scale_inputs()
    path = tmp_path / "model.py"
    path.write_text("# fixture normalized model implementation\n")
    identity = hashlib.sha256(path.read_bytes()).hexdigest()
    request = replace(
        baseline.request,
        runtime_identity={
            **baseline.request.runtime_identity,
            "external_source_sha256": {"hpsv2": {"src/open_clip/model.py": identity}},
        },
    )
    audit = support_audit(request, external_model_paths={"hps": path})
    assert audit["sources"]["hps"]["file_verified"] is True
    assert audit["axes"]["hps"]["bound"] is None
    path.write_text("# different model implementation\n")
    with pytest.raises(ValueError, match="external model source"):
        support_audit(request, external_model_paths={"hps": path})


def test_real_applicability_cannot_be_requested_from_mac_before_reading_data():
    import platform

    if platform.system() != "Darwin":
        pytest.skip("host boundary specifically exercises the Mac fixture environment")
    with pytest.raises(ValueError, match="GB10 Linux ARM"):
        load_applicability_report(
            {
                "schema": "mage-w4-statistical-applicability-inputs/v1",
                "execution_kind": "gb10_measured",
                "manifest_path": "must-not-be-read.json",
            }
        )


def test_prior_synthetic_coverage_is_linked_by_file_sha_not_promoted_to_real_applicability(
    tmp_path,
):
    import hashlib
    import json
    from pathlib import Path

    from shadow_lab.mage_w4_search.statistical import METHOD

    source = Path(__file__).parents[2] / "src/shadow_lab/mage_w4_search/statistical.py"
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    coverage = tmp_path / "coverage.json"
    coverage.write_text(
        json.dumps(
            {
                "method": METHOD,
                "execution_kind": "synthetic-cpu-validation",
                "status": "synthetic-coverage-inconclusive-or-failed",
                "results": [],
            }
        )
    )
    manifest = tmp_path / "source.json"
    manifest.write_text(
        json.dumps({"source_sha256": {"shadow_lab/mage_w4_search/statistical.py": source_hash}})
    )
    resources = tmp_path / "resources.json"
    resources.write_text(
        json.dumps(
            {
                "exit_status": 0,
                "timed_out": False,
                "execution_kind": "synthetic-cpu-validation",
                "output_sha256": hashlib.sha256(coverage.read_bytes()).hexdigest(),
                "source_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                "source_sha256": json.loads(manifest.read_text())["source_sha256"],
            }
        )
    )
    paths = {"coverage": coverage, "resources": resources, "source_manifest": manifest}
    linked = read_coverage_evidence(paths)
    assert linked["recorded_diagnostic"]["status"] == "synthetic-coverage-inconclusive-or-failed"
    assert linked["establishes_real_applicability"] is False
    coverage.write_text(coverage.read_text() + " ")
    with pytest.raises(ValueError, match="coverage.*identity"):
        read_coverage_evidence(paths)
