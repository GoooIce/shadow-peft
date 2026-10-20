"""RESTRAIN-lite core (shadow_lab.restrain) — external-behavior tests (issue #31).

Anchors from the spec (``docs/specs/restrain-lite-structured-flywheel.md``
§Implementation Decisions 2/3, Testing Decisions「聚类缝」「权重缝」):

- verdict synonyms collapse to ``pass`` / ``fail`` / ``rework``; ``不合格``
  must never be read as ``合格``; 判定/结论 pick the label nearest the marker.
- number answers normalize number+unit (``245℃`` == ``245 摄氏度``) and cluster
  under a relative tolerance.
- unclusterable prompts carry a flag and never raise.
- vote weights sum to 1, tilt toward the majority, never hit 1/0; all-zero
  counts yield empty weights instead of dividing by zero.
- low-consistency prompts get ``prompt_weight == 0`` while the raw
  ``consistency`` stays available for reports.
- rows carry the exact field set from the spec and survive a JSON round-trip.

Everything here is a pure function: no torch, no numpy, deterministic.
"""

from __future__ import annotations

import json
import math

import pytest

from shadow_lab.restrain import (
    build_self_train_row,
    cluster_answers,
    consistency,
    extract_answer,
    prompt_weight,
    vote_weights,
)

# ---------------------------------------------------------------------------
# extract_answer: quality_verdict
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("放行", "pass"),
        ("合格", "pass"),
        ("结论：放行", "pass"),
        ("判定：合格", "pass"),
        ("拦截", "fail"),
        ("不合格", "fail"),
        ("结论：不合格", "fail"),
        ("判定：拦截", "fail"),
        ("返修", "rework"),
        ("结论：返修", "rework"),
        ("**判定**：**放行**", "pass"),
        ("```text\n结论：拦截\n```", "fail"),
    ],
)
def test_verdict_synonyms_normalize_to_one_label(text, expected):
    assert extract_answer(text, "quality_verdict") == expected


def test_verdict_buhege_is_not_matched_as_hege():
    # ``不合格`` contains ``合格``; alternation must prefer the longer fail label.
    assert extract_answer("判定：不合格", "quality_verdict") == "fail"
    assert extract_answer("不 合格", "quality_verdict") == "fail"


def test_verdict_takes_the_label_nearest_the_judgment_marker():
    assert extract_answer("判定：合格。结论：返修。", "quality_verdict") == "rework"
    # The verdict right after the last marker wins, not a later stray mention.
    assert extract_answer("结论：不合格，不能放行。", "quality_verdict") == "fail"


@pytest.mark.parametrize(
    "text",
    ["", "   ", "**", "```", "判定：", "结论：", "不知道", "245℃", "N/A"],
)
def test_verdict_without_a_label_returns_none(text):
    assert extract_answer(text, "quality_verdict") is None


# ---------------------------------------------------------------------------
# extract_answer: process_calc
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    ["245℃", "245 摄氏度", "245°C", "245 °C"],
)
def test_temperature_units_normalize_to_celsius(text):
    assert extract_answer(text, "process_calc") == (245.0, "℃")


@pytest.mark.parametrize(
    "text",
    ["长度 12.5 mm", "12.5毫米", "12.5 毫米", "12.5MM"],
)
def test_length_units_normalize_to_mm(text):
    assert extract_answer(text, "process_calc") == (12.5, "mm")


def test_number_extraction_prefers_the_first_value_with_a_recognized_unit():
    assert extract_answer("先做3次，结果 245℃", "process_calc") == (245.0, "℃")


def test_bare_and_negative_numbers_are_answers():
    assert extract_answer("结果是 245", "process_calc") == (245.0, "")
    assert extract_answer("偏移 -3.5 ℃", "process_calc") == (-3.5, "℃")


@pytest.mark.parametrize("text", ["", "   ", "```", "温度很高", "暂无数据", "N/A"])
def test_number_answers_without_a_number_return_none(text):
    assert extract_answer(text, "process_calc") is None


def test_extract_answer_returns_none_for_other_tasks():
    assert extract_answer("放行 245℃", "open_qa") is None
    assert extract_answer("放行", None) is None


# ---------------------------------------------------------------------------
# cluster_answers: 聚类缝
# ---------------------------------------------------------------------------


def test_verdict_synonyms_land_in_the_same_cluster():
    result = cluster_answers(
        ["结论：放行", "合格", "判定：合格", "返修"], "quality_verdict"
    )

    assert result.unclusterable is False
    assert result.cluster_method == "verdict_label"
    assert result.counts == {"pass": 3, "rework": 1}
    assert result.clusters["pass"] == ["结论：放行", "合格", "判定：合格"]


def test_number_units_land_in_the_same_cluster():
    result = cluster_answers(["245℃", "245 摄氏度", "245°C"], "process_calc")

    assert result.unclusterable is False
    assert result.cluster_method == "number_tolerance"
    assert list(result.counts.values()) == [3]
    assert sum(result.counts.values()) == 3


def test_number_tolerance_merges_close_values_and_splits_far_values():
    close = cluster_answers(["245", "245.0000001"], "process_calc")
    assert close.unclusterable is False
    assert sum(close.counts.values()) == 2 and len(close.counts) == 1

    far = cluster_answers(["245", "245.1"], "process_calc")
    assert far.unclusterable is True
    assert len(far.counts) == 2


def test_number_tolerance_is_configurable():
    result = cluster_answers(["245", "245.1"], "process_calc", number_rtol=1e-3)

    assert result.unclusterable is False
    assert len(result.counts) == 1


def test_number_clusters_need_a_dominant_plurality():
    result = cluster_answers(["245℃", "245℃", "245.1℃"], "process_calc")

    assert result.unclusterable is False
    assert sorted(result.counts.values(), reverse=True) == [2, 1]


def test_unit_mismatch_is_unclusterable_when_no_cluster_dominates():
    result = cluster_answers(["245℃", "245mm"], "process_calc")

    assert result.unclusterable is True
    assert set(result.counts.values()) == {1}


def test_empty_or_unparseable_answers_are_skipped_without_raising():
    result = cluster_answers(["", "   ", "不知道", "```"], "process_calc")

    assert result.unclusterable is True
    assert result.clusters == {}
    assert result.counts == {}

    verdict = cluster_answers(["", "放行", "**"], "quality_verdict")
    assert verdict.unclusterable is False
    assert verdict.counts == {"pass": 1}


def test_unsupported_task_is_unclusterable_without_raising():
    result = cluster_answers(["放行", "245℃"], "open_qa")

    assert result.unclusterable is True
    assert result.counts == {}
    assert result.cluster_method == "none"


def test_clustering_is_deterministic():
    answers = ["245℃", "246℃", "245 摄氏度", "245.1℃"]

    assert cluster_answers(answers, "process_calc") == cluster_answers(
        answers, "process_calc"
    )


# ---------------------------------------------------------------------------
# weights: 权重缝
# ---------------------------------------------------------------------------


def test_uniform_votes_give_uniform_weights_that_sum_to_one():
    weights = vote_weights([4, 4])

    assert weights == pytest.approx([0.5, 0.5])
    assert sum(weights) == pytest.approx(1.0)


def test_majority_votes_tilt_the_weight_but_never_reach_one_or_zero():
    weights = vote_weights([3, 1])

    # g(f) = exp(-(1-f)^2 / (2*0.5^2)); [3,1] => [0.7310586, 0.2689414].
    assert weights == pytest.approx([0.7310586, 0.2689414], abs=1e-7)
    assert sum(weights) == pytest.approx(1.0)
    assert 0.0 < weights[1] < 0.5 < weights[0] < 1.0


def test_vote_weights_keep_mapping_keys():
    weights = vote_weights({"pass": 3, "fail": 1})

    assert set(weights) == {"pass", "fail"}
    assert weights["pass"] > weights["fail"]
    assert sum(weights.values()) == pytest.approx(1.0)


@pytest.mark.parametrize("counts", [[], [0, 0, 0], {}])
def test_all_zero_counts_return_empty_weights_without_dividing_by_zero(counts):
    assert vote_weights(counts) in ([], {})


def test_consistency_is_high_for_unanimous_votes_and_low_for_a_flat_spread():
    assert consistency([8]) == pytest.approx(1.0)
    assert consistency([4, 4]) == pytest.approx(math.exp(-0.5), abs=1e-9)
    assert consistency([1] * 8) == pytest.approx(math.exp(-1.53125), abs=1e-9)
    assert consistency([1] * 8) < consistency([4, 4])


def test_consistency_of_empty_counts_is_zero():
    assert consistency([]) == 0.0
    assert consistency({}) == 0.0


@pytest.mark.parametrize("counts", [[2, 2, 2, 2], [1, 1, 1], [0, 0]])
def test_prompt_weight_is_zero_below_kappa(counts):
    assert prompt_weight(counts, kappa=3) == 0


@pytest.mark.parametrize("counts", [[3, 1], [8], [5, 2, 1]])
def test_prompt_weight_is_one_at_or_above_kappa(counts):
    assert prompt_weight(counts, kappa=3) == 1


def test_prompt_weight_keeps_raw_consistency_in_the_report():
    counts = [2, 1]
    assert prompt_weight(counts, kappa=3) == 0
    assert consistency(counts) > 0.0


# ---------------------------------------------------------------------------
# build_self_train_row: 数据缝
# ---------------------------------------------------------------------------

_ROW_FIELDS = {
    "prompt",
    "task",
    "answers",
    "counts",
    "reference_counts",
    "weights",
    "prompt_weight",
    "consistency",
    "cluster_method",
    "adapter_version",
}


def test_self_train_row_has_the_spec_fields_and_survives_json_round_trip():
    row = build_self_train_row(
        "阀门状态如何？",
        "quality_verdict",
        ["放行", "合格", "合格", "返修"],
        reference_counts=[3, 1],
        adapter_version="edge-v3",
    )

    assert row is not None
    assert set(row) == _ROW_FIELDS
    assert row["answers"] == ["pass", "rework"]
    assert row["counts"] == [3, 1]
    assert row["reference_counts"] == [3, 1]
    assert row["weights"] == pytest.approx([0.7310586, 0.2689414], abs=1e-7)
    assert row["prompt_weight"] == 1
    assert row["consistency"] == pytest.approx(math.exp(-0.125), abs=1e-9)
    assert row["cluster_method"] == "verdict_label"
    assert row["adapter_version"] == "edge-v3"

    assert json.loads(json.dumps(row)) == row


def test_number_row_keeps_low_prompt_weight_and_raw_consistency():
    row = build_self_train_row(
        "出口温度？",
        "process_calc",
        ["245℃", "245 摄氏度", "246℃"],
    )

    assert row is not None
    assert row["answers"] == ["245℃", "246℃"]
    assert row["counts"] == [2, 1]
    assert row["reference_counts"] == [0, 0]
    assert row["prompt_weight"] == 0
    assert row["consistency"] == pytest.approx(math.exp(-2 / 9), abs=1e-9)
    assert sum(row["weights"]) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("task", "answers"),
    [
        ("process_calc", ["245℃", "245mm"]),
        ("process_calc", ["", "不知道"]),
        ("quality_verdict", ["", "***"]),
        ("open_qa", ["放行", "245℃"]),
    ],
)
def test_unclusterable_rows_are_skipped(task, answers):
    assert build_self_train_row("Q?", task, answers) is None


def test_self_train_row_is_deterministic():
    kwargs = dict(
        prompt="P",
        task="quality_verdict",
        answers=["放行", "合格", "返修"],
        adapter_version="edge-v3",
    )
    assert build_self_train_row(**kwargs) == build_self_train_row(**kwargs)
