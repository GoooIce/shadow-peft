"""Joint decisions preserve prompt weights and independent source units."""

import math
from dataclasses import replace

import pytest

from shadow_lab.mage_w4_search.contracts import RequestedSample, digest_json
from shadow_lab.mage_w4_search.statistical import (
    EvidenceError,
    EvidenceLedger,
    JointProtocol,
    PairedEvidence,
    PairedRow,
    SliceSpec,
    coverage_validation,
    joint_decision,
)


def fixture(count=100, seeds=(1,)):
    samples = tuple(
        RequestedSample(f"p-{i}", f"{i:064x}", f"g-{i}", seed)
        for i in range(count)
        for seed in seeds
    )
    part = SliceSpec(
        "slice-1", "confirmation", samples, tuple(f"p-{i}" for i in range(count)), "f" * 64
    )
    protocol = JointProtocol(
        "protocol-fixture",
        "a" * 64,
        {"clip": 1.0, "hps": 1.0, "ocr": 1.0},
        {axis: (0.0, 1.0) for axis in ("clip", "hps", "ocr")},
        (0.025, 0.025),
        2,
        2,
        "b" * 64,
        "c" * 64,
        (part,),
        scorer_digest="e" * 64,
    )
    rows = tuple(
        PairedRow(
            sample.sample_id,
            {"clip": 0.8, "hps": 0.8, "ocr": 0.8},
            {"clip": 0.2, "hps": 0.2, "ocr": 0.2},
            {"clip": 0.2, "hps": 0.2, "ocr": 0.2},
        )
        for sample in samples
    )
    evidence = PairedEvidence(
        "candidate-1", "d" * 64, "a" * 64, "a" * 64, 0, "slice-1", "e" * 64, rows
    )
    return protocol, evidence


def test_strong_paired_gain_passes_seven_joint_checks_without_turning_seeds_into_sample_size():
    protocol, evidence = fixture()
    result = joint_decision(protocol, evidence, alpha=0.025, stage="confirmation")
    assert result["status"] == "accepted"
    assert result["formal"] is False
    assert len(result["checks"]) == 7
    assert result["checks"]["current.clip"]["source_clusters"] == 100

    repeated_protocol, repeated = fixture(seeds=(1, 2, 3))
    repeated_result = joint_decision(repeated_protocol, repeated, alpha=0.025, stage="confirmation")
    assert repeated_result["checks"] == result["checks"]


def test_missing_ocr_small_source_population_and_out_of_support_scores_cannot_pass():
    protocol, evidence = fixture()
    missing = replace(evidence.rows[0], candidate={"clip": 0.8, "hps": 0.8, "ocr": None})
    result = joint_decision(
        protocol,
        replace(evidence, rows=(missing,) + evidence.rows[1:]),
        alpha=0.025,
        stage="confirmation",
    )
    assert result["status"] == "invalid-request"
    assert "missing-or-invalid-score" in result["reasons"]
    small, few = fixture(count=1, seeds=(1, 2, 3, 4))
    result = joint_decision(small, few, alpha=0.025, stage="confirmation")
    assert result["status"] == "insufficient-evidence"
    assert "too-few-source-clusters" in result["reasons"]
    outside = replace(evidence.rows[0], candidate={"clip": 1.001, "hps": 0.8, "ocr": 0.8})
    result = joint_decision(
        protocol,
        replace(evidence, rows=(outside,) + evidence.rows[1:]),
        alpha=0.025,
        stage="confirmation",
    )
    assert "score-outside-frozen-support" in result["reasons"]


def test_visible_failed_candidate_cannot_resit_and_reopen_preserves_risk(tmp_path):
    protocol, evidence = fixture()
    second_samples = tuple(
        replace(
            sample,
            prompt_id="new-" + sample.prompt_id,
            source_group="new-" + sample.source_group,
            prompt_digest=digest_json("new-" + sample.prompt_digest),
        )
        for sample in protocol.slices[0].samples
    )
    second = SliceSpec(
        "slice-2",
        "confirmation",
        second_samples,
        tuple("new-" + name for name in protocol.slices[0].ocr_prompt_ids),
        "1" * 64,
    )
    protocol = replace(protocol, slices=protocol.slices + (second,))
    path = tmp_path / "evidence.sqlite"
    ledger = EvidenceLedger(path, protocol, clock=lambda: 1000)
    claim = ledger.claim_confirmation(
        "attempt-1",
        "slice-1",
        "candidate-1",
        "d" * 64,
        current_digest="a" * 64,
        center_version=0,
        selection_digest="2" * 64,
    )
    assert claim["alpha"] == 0.025
    ledger.mark_visible("attempt-1", "3" * 64)
    recovered = EvidenceLedger(path, protocol, clock=lambda: 1001)
    recovered.abort("attempt-1", reason="worker-failed-after-score-visible")
    assert recovered.snapshot()["confirmation_alpha_committed"] == 0.025
    with pytest.raises(EvidenceError, match="retired-attempt"):
        recovered.claim_confirmation(
            "attempt-1",
            "slice-1",
            "candidate-1",
            "d" * 64,
            current_digest="a" * 64,
            center_version=0,
            selection_digest="2" * 64,
        )
    with pytest.raises(EvidenceError, match="candidate-already-attempted"):
        recovered.claim_confirmation(
            "attempt-2",
            "slice-2",
            "candidate-renamed",
            "d" * 64,
            current_digest="a" * 64,
            center_version=0,
            selection_digest="4" * 64,
        )


def test_confirmed_center_is_replayable_and_final_opens_only_once(tmp_path):
    protocol, evidence = fixture()
    final_samples = tuple(
        replace(
            sample,
            prompt_id="final-" + sample.prompt_id,
            source_group="final-" + sample.source_group,
            prompt_digest=digest_json("final-" + sample.prompt_digest),
        )
        for sample in protocol.slices[0].samples
    )
    final = SliceSpec(
        "final-1",
        "final",
        final_samples,
        tuple("final-" + name for name in protocol.slices[0].ocr_prompt_ids),
        "4" * 64,
    )
    protocol = replace(protocol, slices=protocol.slices + (final,))
    ledger = EvidenceLedger(tmp_path / "evidence.sqlite", protocol, clock=lambda: 1000)
    with pytest.raises(EvidenceError, match="no-confirmed-center"):
        ledger.claim_final("final-attempt", "final-1", "candidate-1", "d" * 64)
    ledger.claim_confirmation(
        "attempt-1",
        "slice-1",
        "candidate-1",
        "d" * 64,
        current_digest="a" * 64,
        center_version=0,
        selection_digest="2" * 64,
    )
    result = ledger.submit("attempt-1", evidence)
    assert result["status"] == "accepted"
    assert (
        ledger.submit("attempt-1", replace(evidence, rows=tuple(reversed(evidence.rows)))) == result
    )
    assert ledger.snapshot()["latest_center"]["version"] == 1
    ledger.claim_final("final-attempt", "final-1", "candidate-1", "d" * 64)
    with pytest.raises(EvidenceError, match="final-already-opened"):
        ledger.claim_final("another-final", "final-1", "candidate-1", "d" * 64)
    assert ledger.snapshot()["confirmation_alpha_committed"] == 0.025
    assert ledger.snapshot()["final_alpha_committed"] == 0.05


def test_prompt_weights_and_ocr_denominator_survive_source_grouping_and_input_order():
    protocol, evidence = fixture(count=3)
    original = protocol.slices[0]
    samples = tuple(
        replace(sample, source_group="shared" if i < 2 else "other")
        for i, sample in enumerate(original.samples)
    )
    part = replace(original, samples=samples, ocr_prompt_ids=("p-1", "p-2"))
    protocol = replace(protocol, slices=(part,))
    rows = []
    for index, sample in enumerate(samples):
        gain = 0.8 if index < 2 else 0.2
        value = {"clip": gain, "hps": gain, "ocr": gain if index else None}
        reference = {"clip": 0.0, "hps": 0.0, "ocr": 0.0 if index else None}
        rows.append(PairedRow(sample.sample_id, value, reference, reference))
    evidence = replace(evidence, rows=tuple(rows))
    result = joint_decision(protocol, evidence, alpha=0.025, stage="confirmation")
    assert result["checks"]["current.clip"]["estimate"] == pytest.approx(0.6)
    assert result["checks"]["current.ocr"]["estimate"] == pytest.approx(0.5)
    assert result["checks"]["composite"]["estimate"] == pytest.approx(0.575)
    assert (
        joint_decision(
            protocol,
            replace(evidence, rows=tuple(reversed(rows))),
            alpha=0.025,
            stage="confirmation",
        )
        == result
    )


def test_zero_composite_lower_bound_is_insufficient_even_when_all_noninferiority_gates_pass():
    protocol, evidence = fixture()
    alpha = 7 * math.exp(-12.5)
    protocol = replace(protocol, confirmation_alphas=(alpha,))
    rows = tuple(
        replace(
            row,
            candidate={axis: 0.5 for axis in ("clip", "hps", "ocr")},
            current={axis: 0.0 for axis in ("clip", "hps", "ocr")},
            q0={axis: 0.0 for axis in ("clip", "hps", "ocr")},
        )
        for row in evidence.rows
    )
    result = joint_decision(
        protocol, replace(evidence, rows=rows), alpha=alpha, stage="confirmation"
    )
    assert result["checks"]["composite"]["lower"] == pytest.approx(0.0, abs=1e-14)
    assert result["status"] == "insufficient-evidence"


def test_small_coverage_smoke_reports_its_simulation_error_without_authorizing_formal_data():
    report = coverage_validation(trials=2, source_clusters=8, seed=7)
    assert len(report["results"]) == 4
    assert report["status"] == "synthetic-coverage-inconclusive-or-failed"
    assert (
        report["formal_applicability"] == "blocked-until-real-support-independence-and-cost-audits"
    )
    assert all(item["noncoverage_mc_upper"] > 0.025 for item in report["results"])


def test_noninferiority_equality_passes_but_composite_gain_cannot_compensate_a_breach():
    protocol, evidence = fixture()
    alpha = 7 * math.exp(-12.5)
    protocol = replace(protocol, confirmation_alphas=(alpha,))
    reference = dict.fromkeys(("clip", "hps", "ocr"), 0.0)
    rows = tuple(
        replace(
            row, candidate={"clip": 0.45, "hps": 1.0, "ocr": 1.0}, current=reference, q0=reference
        )
        for row in evidence.rows
    )
    accepted = joint_decision(
        protocol, replace(evidence, rows=rows), alpha=alpha, stage="confirmation"
    )
    assert accepted["checks"]["current.clip"]["lower"] == pytest.approx(-0.05)
    assert accepted["status"] == "accepted"
    worse = tuple(replace(row, candidate={"clip": 0.449, "hps": 1.0, "ocr": 1.0}) for row in rows)
    rejected = joint_decision(
        protocol, replace(evidence, rows=worse), alpha=alpha, stage="confirmation"
    )
    assert rejected["checks"]["composite"]["passed"] is True
    assert rejected["status"] == "noninferiority-failed"


def test_same_prompt_cannot_be_relabelled_into_a_fresh_slice():
    protocol, evidence = fixture(count=2)
    old = protocol.slices[0]
    renamed = tuple(
        replace(
            sample,
            prompt_id="renamed-" + sample.prompt_id,
            source_group="renamed-" + sample.source_group,
        )
        for sample in old.samples
    )
    other = replace(
        old,
        slice_id="fresh-label",
        samples=renamed,
        ocr_prompt_ids=tuple("renamed-" + name for name in old.ocr_prompt_ids),
    )
    rejected = joint_decision(
        replace(protocol, slices=(old, other)), evidence, alpha=0.025, stage="confirmation"
    )
    assert "prompt-content-overlap-between-slices" in rejected["reasons"]
