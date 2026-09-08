"""Fixed q0 scales and same-device paired candidate screening."""

from dataclasses import replace

import pytest

from shadow_lab.mage_w4_search.contracts import (
    ArtifactIdentity,
    CandidateRequest,
    DataManifest,
    PromptRecord,
    RequestedSample,
    digest_json,
)
from shadow_lab.mage_w4_search.rewards import (
    SampleScores,
    ScalePlan,
    ScoredBatch,
    calibrate_scales,
    screen_candidates,
)


def batch(
    values,
    *,
    candidate_id="q0",
    candidate_digest="a" * 64,
    center_digest="a" * 64,
    purpose="scale_calibration",
    ocr_prompts=None,
    device="gooo",
):
    # Each tuple is (prompt_id, seed, CLIP, HPS, OCR). Fixtures are fabricated;
    # image/generation hashes provide identities, never native-run evidence.
    prompt_ids = tuple(dict.fromkeys(row[0] for row in values))
    text_ids = prompt_ids if ocr_prompts is None else ocr_prompts
    records = tuple(
        PromptRecord(
            prompt,
            f"Only text {prompt}" if prompt in text_ids else f"A picture of {prompt}",
            f"source-{prompt}",
            "development",
            (purpose,),
            (prompt,) if prompt in text_ids else (),
            prompt in text_ids,
        )
        for prompt in prompt_ids
    )
    manifest = DataManifest(records)
    record_map = {record.prompt_id: record for record in records}
    requested = tuple(
        RequestedSample(
            prompt,
            digest_json(record_map[prompt].prompt),
            f"source-{prompt}",
            seed,
        )
        for prompt, seed, *_ in values
    )
    dependencies = tuple(
        ArtifactIdentity(role, "c" * 64, 1)
        for role in (
            "quantized_packed",
            "weight_scales",
            "quantized_remainder",
            "layer_manifest",
            "text_encoder",
            "vae",
            "clip_scorer",
            "hps_scorer",
            "ocr_scorer",
        )
    )
    request = CandidateRequest(
        request_id=candidate_id,
        run_id="fixture-run",
        ticket_id="05",
        center_version=0,
        center_digest=center_digest,
        q0_digest="a" * 64,
        candidate_id=candidate_id,
        candidate_digest=candidate_digest,
        group_id=0,
        mutation_seed=1,
        effective_edits=0 if candidate_id == "q0" else 1,
        samples=requested,
        purpose=purpose,
        device_id=device,
        dependencies=dependencies,
        runtime_identity={
            "software": "fixture",
            "kernel": "native-fixture",
            "sampler": "fixture",
            "generation": {"steps": 4},
            "device": device,
        },
        manifest_digest=manifest.digest,
    )
    samples = tuple(
        SampleScores(
            sample.sample_id,
            digest_json((candidate_id, sample.sample_id)),
            clip,
            hps,
            ocr,
            "ok" if sample.prompt_id in text_ids else "not_applicable",
            "d" * 64,
        )
        for sample, (_, _, clip, hps, ocr) in zip(requested, values, strict=True)
    )
    return ScoredBatch(
        request=request,
        manifest=manifest,
        samples=samples,
        ocr_prompt_ids=text_ids,
        scorer_sha256=(("clip", "c" * 64), ("hps", "c" * 64), ("ocr", "c" * 64)),
        native_receipt_sha256="e" * 64,
        started_at_utc="2026-02-01T00:00:00+00:00",
        execution_kind="fixture",
    )


def scale_plan(baseline, **changes):
    return ScalePlan(
        q0_digest="a" * 64,
        samples=baseline.request.samples,
        ocr_prompt_ids=baseline.ocr_prompt_ids,
        scorer_sha256=baseline.scorer_sha256,
        manifest_digest=baseline.manifest.digest,
        registered_at_utc="2026-01-01T00:00:00+00:00",
        min_prompts=3,
        min_ocr_prompts=3,
        min_source_groups=3,
        maximum_leave_group_out_relative_change=0.5,
        **changes,
    )


def test_q0_sample_standard_deviation_averages_seeds_before_equal_prompt_weighting():
    baseline = batch(
        (
            ("p1", 1, 0.0, 0.0, 0.0),
            ("p1", 2, 0.0, 0.0, 0.0),
            ("p2", 1, 1.0, 2.0, 0.5),
            ("p3", 1, 2.0, 4.0, 1.0),
        )
    )
    calibrated = calibrate_scales(scale_plan(baseline), baseline)
    assert calibrated.status == "ready"
    assert calibrated.scales.clip == 1.0
    assert calibrated.scales.hps == 2.0
    assert calibrated.scales.ocr == 0.5
    assert calibrated.margins.clip == 0.05
    assert calibrated.margins.hps == 0.1
    assert calibrated.margins.ocr == 0.01
    assert calibrated.prompt_counts == {"clip": 3, "hps": 3, "ocr": 3}
    assert calibrated.method == "sample-standard-deviation-ddof-1"


def test_scale_failure_cannot_be_repaired_by_epsilon_or_candidate_distribution():
    constant = batch((("p1", 1, 1.0, 1.0, 0.5), ("p2", 1, 1.0, 2.0, 0.5), ("p3", 1, 1.0, 3.0, 0.5)))
    unreliable = calibrate_scales(scale_plan(constant), constant)
    assert unreliable.status == "unreliable"
    assert unreliable.scales is unreliable.margins is None
    assert "clip:nonpositive-scale" in unreliable.reasons
    assert "ocr:nonpositive-scale" in unreliable.reasons
    candidate = replace(constant, request=replace(constant.request, candidate_digest="b" * 64))
    with pytest.raises(ValueError, match="q0"):
        calibrate_scales(scale_plan(constant), candidate)
    with pytest.raises(ValueError, match="preregistration"):
        calibrate_scales(
            replace(scale_plan(constant), registered_at_utc="2100-01-01T00:00:00+00:00"), constant
        )


def test_reliability_counts_prompt_and_source_units_and_rejects_unstable_scale():
    baseline = batch((("p1", 1, 0.0, 0.0, 0.0), ("p2", 1, 1.0, 2.0, 0.5), ("p3", 1, 2.0, 4.0, 1.0)))
    plan = replace(scale_plan(baseline), maximum_leave_group_out_relative_change=0.1)
    unstable = calibrate_scales(plan, baseline)
    assert unstable.status == "unreliable"
    assert "clip:unstable-under-source-group-removal" in unstable.reasons
    too_few = replace(scale_plan(baseline), min_prompts=4)
    assert "clip:insufficient-prompts" in calibrate_scales(too_few, baseline).reasons
    with pytest.raises(ValueError, match="samples"):
        calibrate_scales(
            replace(scale_plan(baseline), samples=baseline.request.samples[:-1]), baseline
        )


def test_ocr_scale_uses_only_eligible_prompts_and_failure_invalidates_whole_batch():
    baseline = batch(
        (
            ("p1", 1, 0.0, 0.0, 0.0),
            ("p2", 1, 1.0, 2.0, 0.5),
            ("p3", 1, 2.0, 4.0, 1.0),
            ("p4", 1, 3.0, 6.0, None),
        ),
        ocr_prompts=("p1", "p2", "p3"),
    )
    calibrated = calibrate_scales(scale_plan(baseline), baseline)
    assert calibrated.status == "ready"
    assert calibrated.prompt_counts == {"clip": 4, "hps": 4, "ocr": 3}
    assert calibrated.scales.ocr == 0.5
    broken = replace(baseline.samples[0], ocr_status="engine_failure", ocr=None)
    with pytest.raises(ValueError, match="OCR"):
        calibrate_scales(
            scale_plan(baseline), replace(baseline, samples=(broken,) + baseline.samples[1:])
        )


def frozen_scales():
    baseline = batch((("p1", 1, 0.0, 0.0, 0.0), ("p2", 1, 1.0, 2.0, 0.5), ("p3", 1, 2.0, 4.0, 1.0)))
    return calibrate_scales(scale_plan(baseline), baseline)


def screening_batch(candidate_id, values, **changes):
    return batch(
        tuple((prompt, 1, *values) for prompt in ("s1", "s2", "s3")),
        candidate_id=candidate_id,
        purpose="candidate_development",
        **changes,
    )


def test_positive_composite_cannot_compensate_for_failed_clip_gate_and_ties_are_fixed():
    scales = frozen_scales()
    baseline = screening_batch("q0", (0.0, 0.0, 0.5))
    bad = screening_batch("bad", (-0.051, 2.0, 0.5), candidate_digest="b" * 64)
    first = screening_batch("first", (-0.05, 2.0, 0.5), candidate_digest="c" * 64)
    second = screening_batch("second", (-0.05, 2.0, 0.5), candidate_digest="d" * 64)
    references = {row.request.candidate_id: baseline for row in (bad, first, second)}
    screened = screen_candidates(
        scales,
        (second, bad, first),
        references,
        references,
        candidate_order=("bad", "first", "second"),
    )
    assert screened.selected_candidate_id == "first"
    assert screened.status == "best_feasible_candidate"
    results = {row.candidate_id: row for row in screened.candidates}
    assert results["bad"].current.composite_improvement > 0
    assert results["bad"].feasible is False
    assert results["first"].feasible is True
    assert results["first"].current.differences.clip == -0.05
    assert results["first"].current.composite_improvement == pytest.approx(0.225)
    assert results["first"].current.paired_variances == {"clip": 0.0, "hps": 0.0, "ocr": 0.0}
    assert screened.stage == "development_screening"
    assert screened.independently_confirmed is False


def test_current_only_pass_does_not_bypass_fixed_q0():
    center = "f" * 64
    current = screening_batch(
        "current", (-0.04, 0.0, 0.5), candidate_digest=center, center_digest=center
    )
    q0 = screening_batch("q0", (0.0, 0.0, 0.5), center_digest=center)
    candidate = screening_batch(
        "candidate", (-0.06, 2.0, 0.5), candidate_digest="b" * 64, center_digest=center
    )
    screened = screen_candidates(
        frozen_scales(),
        (candidate,),
        {"candidate": current},
        {"candidate": q0},
        candidate_order=("candidate",),
    )
    assert screened.status == "no_feasible_candidate"
    assert screened.selected_candidate_id is None
    assert screened.candidates[0].current.passed is True
    assert screened.candidates[0].q0.passed is False


def test_exact_decimal_margin_is_inclusive_without_relaxing_it():
    baseline = screening_batch("q0", (1.0, 0.0, 0.5))
    candidate = screening_batch("candidate", (0.95, 2.0, 0.49), candidate_digest="b" * 64)
    result = screen_candidates(
        frozen_scales(),
        (candidate,),
        {"candidate": baseline},
        {"candidate": baseline},
        candidate_order=("candidate",),
    )
    assert result.candidates[0].feasible is True
    bad = replace(
        candidate, samples=tuple(replace(row, clip=0.949999999) for row in candidate.samples)
    )
    assert (
        screen_candidates(
            frozen_scales(),
            (bad,),
            {"candidate": baseline},
            {"candidate": baseline},
            candidate_order=("candidate",),
        ).selected_candidate_id
        is None
    )


@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "duplicate",
        "different_device",
        "stale_center",
        "scorer",
        "runtime",
        "ocr_failure",
    ],
)
def test_incomplete_or_mismatched_paired_results_are_rejected(change):
    baseline = screening_batch("q0", (0.0, 0.0, 0.5))
    candidate = screening_batch("candidate", (0.1, 1.0, 0.5), candidate_digest="b" * 64)
    if change == "missing":
        baseline = replace(baseline, samples=baseline.samples[:-1])
    elif change == "duplicate":
        baseline = replace(baseline, samples=baseline.samples + baseline.samples[:1])
    elif change == "different_device":
        baseline = screening_batch("q0", (0.0, 0.0, 0.5), device="goooinext")
    elif change == "stale_center":
        baseline = replace(baseline, request=replace(baseline.request, center_version=1))
    elif change == "scorer":
        baseline = replace(
            baseline, scorer_sha256=(("clip", "f" * 64), ("hps", "c" * 64), ("ocr", "c" * 64))
        )
    elif change == "runtime":
        baseline = replace(
            baseline,
            request=replace(
                baseline.request,
                runtime_identity={**baseline.request.runtime_identity, "sampler": "different"},
            ),
        )
    else:
        baseline = replace(
            baseline,
            samples=(replace(baseline.samples[0], ocr=None, ocr_status="engine_failure"),)
            + baseline.samples[1:],
        )
    with pytest.raises(ValueError):
        screen_candidates(
            frozen_scales(),
            (candidate,),
            {"candidate": baseline},
            {"candidate": baseline},
            candidate_order=("candidate",),
        )


def test_candidates_share_frozen_samples_even_when_each_has_same_device_references():
    first = screening_batch("first", (0.1, 1.0, 0.5), candidate_digest="b" * 64)
    q0_first = screening_batch("q0", (0.0, 0.0, 0.5))
    second = batch(
        (("other1", 1, 0.1, 1.0, 0.5), ("other2", 1, 0.1, 1.0, 0.5), ("other3", 1, 0.1, 1.0, 0.5)),
        candidate_id="second",
        candidate_digest="d" * 64,
        purpose="candidate_development",
        device="goooinext",
    )
    q0_second = batch(
        (("other1", 1, 0.0, 0.0, 0.5), ("other2", 1, 0.0, 0.0, 0.5), ("other3", 1, 0.0, 0.0, 0.5)),
        purpose="candidate_development",
        device="goooinext",
    )
    refs = {"first": q0_first, "second": q0_second}
    with pytest.raises(ValueError, match="batch sample"):
        screen_candidates(
            frozen_scales(), (first, second), refs, refs, candidate_order=("first", "second")
        )


def test_new_common_generation_configuration_cannot_reuse_old_fixed_scales():
    baseline = screening_batch("q0", (0.0, 0.0, 0.5))
    candidate = screening_batch("candidate", (0.1, 1.0, 0.5), candidate_digest="b" * 64)

    def changed(item):
        return replace(
            item,
            request=replace(
                item.request,
                runtime_identity={
                    **item.request.runtime_identity,
                    "generation": {"steps": 5},
                },
            ),
        )

    candidate, baseline = changed(candidate), changed(baseline)
    with pytest.raises(ValueError, match="fixed generation"):
        screen_candidates(
            frozen_scales(),
            (candidate,),
            {"candidate": baseline},
            {"candidate": baseline},
            candidate_order=("candidate",),
        )
