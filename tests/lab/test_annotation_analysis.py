from __future__ import annotations

import pytest

from shadow_lab.annotation.analysis import summarize_annotations


def _answer(
    event_id: str,
    comparison_id: str,
    annotator: str,
    decision: str,
    *,
    qc_kind: str = "none",
    qc_group_id: str | None = None,
) -> dict:
    canonical = {"left": "candidate_a", "right": "candidate_b"}.get(
        decision, decision
    )
    return {
        "event_id": event_id,
        "comparison_id": comparison_id,
        "annotator_id": annotator,
        "event_type": "answer",
        "is_superseded": False,
        "presentation_decision": decision,
        "canonical_decision": canonical,
        "qc_kind": qc_kind,
        "qc_group_id": qc_group_id,
        "reveal": {"left_arm_id": "control", "right_arm_id": "candidate"},
    }


def test_summary_reports_bias_agreement_repeat_consistency_and_arm_ci() -> None:
    answers = [
        _answer("1", "c1", "u1", "left"),
        _answer("2", "c1", "u2", "left"),
        _answer("3", "c2", "u1", "right", qc_kind="repeat", qc_group_id="r1"),
        _answer("4", "c3", "u1", "right", qc_kind="repeat", qc_group_id="r1"),
        _answer("5", "c4", "u2", "tie"),
        _answer("6", "c5", "u2", "skip"),
    ]

    report = summarize_annotations(
        {"export_mode": "revealed", "answers": answers},
        bootstrap_samples=100,
        bootstrap_seed=7,
    )

    assert report["decision_counts"] == {"left": 2, "tie": 1, "right": 2, "skip": 1}
    assert report["left_choice_rate"] == pytest.approx(0.5)
    assert report["agreement"]["exact_agreement"] == 1.0
    assert report["repeat_consistency"]["consistency"] == 1.0
    assert set(report["bradley_terry_log_strength"]) == {"control", "candidate"}
    assert len(report["arm_scores"]["candidate"]["paired_bootstrap_ci95"]) == 2


def test_blind_summary_never_requires_reveal_mapping() -> None:
    answer = _answer("1", "c1", "u1", "skip")
    answer.pop("reveal")

    report = summarize_annotations({"export_mode": "blind", "answers": [answer]})

    assert report["skip_rate"] == 1.0
    assert "arm_scores" not in report
