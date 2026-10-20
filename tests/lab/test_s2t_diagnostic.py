"""S2T-Local diagnostic core (``shadow_lab.s2t``) — external-behavior tests.

Everything is synthetic logits: the module is torch-free and accepts plain
lists/sequences, so the diagnostic stays import-light and deterministic.

Anchors:
- Hit@K = base top-1 is in the shadow top-K; Agree@1 = the two top-1s match;
  rank is the 1-based position in the deterministic shadow order, also when
  the base token is outside top-K.
- Empty candidates and all ``-inf`` logits are clean misses (rank ``None``),
  never NaN.
- ``K < 1`` raises; ``K > vocab`` truncates to the vocabulary size.
- Stratified counts reconcile with the overall token count.
- go/no-go boundaries: 0.85 -> go, 0.70 -> pilot, below -> no_go.
"""

from __future__ import annotations

import ast
import csv
import io
import json
import math
from pathlib import Path

import pytest

from shadow_lab.s2t import (
    AGREE1_METRIC,
    CSV_FIELDS,
    DEFAULT_KS,
    GO_THRESHOLD,
    HIT8_METRIC,
    PILOT_THRESHOLD,
    POSITION_BUCKETS,
    REPORT_FIELDS,
    SCHEMA_VERSION,
    BootstrapResult,
    GoNoGoDecision,
    bootstrap_ci,
    build_report,
    go_no_go,
    group_metrics,
    position_bucket,
    summarize_hits,
    to_csv,
    to_json,
    token_hit,
)


def _record(
    *,
    base_top1: int | None = None,
    shadow_logits: list[float] | None = None,
    shadow_topk_ids: list[int] | None = None,
    base_logits: list[float] | None = None,
    task: str | None = "fault",
    position: int | None = 0,
    prompt_id: str | None = "p0",
) -> dict:
    record: dict = {}
    if base_top1 is not None:
        record["base_top1"] = base_top1
    if shadow_logits is not None:
        record["shadow_logits"] = shadow_logits
    if shadow_topk_ids is not None:
        record["shadow_topk_ids"] = shadow_topk_ids
    if base_logits is not None:
        record["base_logits"] = base_logits
    if task is not None:
        record["task"] = task
    if position is not None:
        record["position"] = position
    if prompt_id is not None:
        record["prompt_id"] = prompt_id
    return record


# ---------------------------------------------------------------------------
# token_hit: one token, full shadow logits
# ---------------------------------------------------------------------------


def test_token_hit_inside_topk_reports_hit_agree_and_true_rank():
    # shadow order: 1 > 3 > 2 > 0 > 4
    hit = token_hit([0.0, 5.0, 3.0, 4.0, 1.0], base_top1=3, k=2)

    assert hit.hit is True
    assert hit.agree is False
    assert hit.rank == 2
    assert hit.shadow_top1 == 1
    assert hit.effective_k == 2
    assert bool(hit) is True


def test_token_hit_outside_topk_still_reports_true_rank():
    hit = token_hit([9.0, 8.0, 7.0, 6.0, 5.0], base_top1=4, k=2)

    assert hit.hit is False
    assert hit.agree is False
    assert hit.rank == 5
    assert hit.shadow_top1 == 0


@pytest.mark.parametrize("base_top1", [0, 1, 2, 3])
def test_hit_at_1_equals_agree_at_1(base_top1):
    logits = [0.0, 9.0, 4.0, 1.0]
    at_one = token_hit(logits, base_top1, 1)

    assert at_one.hit == at_one.agree


def test_ties_break_toward_lower_token_id_and_rank_is_ordinal():
    logits = [5.0, 5.0, 1.0]

    at_one = token_hit(logits, base_top1=1, k=1)
    assert at_one.shadow_top1 == 0
    assert at_one.rank == 2
    assert at_one.hit is False
    assert at_one.agree is False

    at_two = token_hit(logits, base_top1=1, k=2)
    assert at_two.hit is True
    assert at_two.agree is False
    assert at_two.rank == 2


def test_empty_candidates_are_a_clean_miss():
    hit = token_hit([], base_top1=0, k=8)

    assert hit.hit is False
    assert hit.agree is False
    assert hit.rank is None
    assert hit.shadow_top1 is None
    assert hit.effective_k == 0


def test_all_neg_inf_is_a_clean_miss():
    hit = token_hit([-math.inf, -math.inf, -math.inf], base_top1=0, k=2)

    assert hit.hit is False
    assert hit.agree is False
    assert hit.rank is None
    assert hit.shadow_top1 is None


def test_k_above_vocab_truncates_to_vocab():
    hit = token_hit([0.0, 2.0, 1.0], base_top1=0, k=99)

    assert hit.hit is True
    assert hit.effective_k == 3


@pytest.mark.parametrize("k", [0, -1])
def test_k_below_one_raises(k):
    with pytest.raises(ValueError, match="k"):
        token_hit([0.0, 1.0], base_top1=0, k=k)


def test_base_top1_outside_vocab_raises():
    with pytest.raises(ValueError, match="base_top1"):
        token_hit([0.0, 1.0], base_top1=2, k=1)


def test_nan_logits_are_rejected():
    with pytest.raises(ValueError, match="NaN"):
        token_hit([0.0, math.nan, 1.0], base_top1=0, k=1)


def test_module_does_not_import_torch():
    source = Path(__file__).parents[2] / "src" / "shadow_lab" / "s2t.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert "torch" not in imported


# ---------------------------------------------------------------------------
# summarize_hits: token rates, rank quantiles, partial top-K records
# ---------------------------------------------------------------------------


def test_summarize_hits_rates_and_rank_quantiles():
    records = [
        # shadow order 1 > 0 > 2 > 3 -> base 1 is rank 1 (agreement)
        _record(base_top1=1, shadow_logits=[0.0, 5.0, 2.0, 1.0], prompt_id="p0"),
        # shadow order 2 > 1 > 3 > 0 -> base 0 is rank 4
        _record(base_top1=0, shadow_logits=[0.0, 4.0, 5.0, 3.0], prompt_id="p0", task="calc"),
        # shadow order 3 > 2 > 1 > 0 -> base 2 is rank 2
        _record(base_top1=2, shadow_logits=[0.0, 1.0, 2.0, 3.0], prompt_id="p1", task="calc"),
    ]

    summary = summarize_hits(records)

    assert summary["n"] == 3
    assert summary["n_prompts"] == 2
    assert summary["hit_at_1"] == pytest.approx(1 / 3)
    assert summary["hit_at_4"] == 1.0
    assert summary["hit_at_8"] == 1.0
    assert summary["hit_at_16"] == 1.0
    assert summary["agree_at_1"] == pytest.approx(1 / 3)
    assert summary["n_rank_known"] == 3
    # ranks [1, 2, 4] -> linear interpolation: p90 = 2 + 0.8 * (4 - 2)
    assert summary["rank_p50"] == 2.0
    assert summary["rank_p90"] == pytest.approx(3.6)
    assert summary["rank_p95"] == pytest.approx(3.8)
    assert summary["rank_p99"] == pytest.approx(3.96)
    assert set(DEFAULT_KS) <= {
        int(key.rsplit("_", 1)[1]) for key in summary if key.startswith("hit_at_")
    }


def test_summarize_hits_accepts_partial_topk_records():
    records = [
        _record(base_top1=3, shadow_topk_ids=[1, 3, 0]),
        _record(base_top1=7, shadow_topk_ids=[1, 3, 0]),
    ]

    summary = summarize_hits(records, ks=(1, 4))

    assert summary["n"] == 2
    assert summary["hit_at_1"] == 0.0
    assert summary["hit_at_4"] == 0.5
    assert summary["agree_at_1"] == 0.0
    assert summary["n_rank_known"] == 1
    assert summary["rank_p50"] == 2.0


def test_summarize_hits_derives_base_top1_from_optional_base_logits():
    records = [_record(base_logits=[0.0, 7.0, 1.0], shadow_logits=[0.0, 1.0, 2.0])]

    summary = summarize_hits(records)

    # base top-1 is token 1; shadow order is 2 > 1 > 0, so rank 2.
    assert summary["hit_at_1"] == 0.0
    assert summary["hit_at_4"] == 1.0
    assert summary["rank_p50"] == 2.0


def test_summarize_hits_counts_degenerate_candidates_without_nan():
    records = [
        _record(base_top1=0, shadow_logits=[]),
        _record(base_top1=0, shadow_logits=[-math.inf, -math.inf]),
        _record(base_top1=0, shadow_logits=[0.0, -math.inf]),
    ]

    summary = summarize_hits(records)

    assert summary["n"] == 3
    assert summary["hit_at_1"] == pytest.approx(1 / 3)
    assert summary["n_empty_candidates"] == 1
    assert summary["n_all_neg_inf"] == 1
    assert summary["n_rank_known"] == 1
    assert summary["rank_p50"] == 1.0
    for key, value in summary.items():
        if isinstance(value, float):
            assert math.isfinite(value), f"{key} is not finite: {value}"


def test_summarize_hits_is_deterministic_and_supports_custom_ks():
    records = [_record(base_top1=0, shadow_logits=[0.0, 1.0])]

    first = summarize_hits(records, ks=(2, 8))
    second = summarize_hits(records, ks=(2, 8))

    assert first == second
    assert first["hit_at_2"] == 1.0
    assert first["hit_at_8"] == 1.0


def test_summarize_hits_empty_records_raise():
    with pytest.raises(ValueError, match="at least one record"):
        summarize_hits([])


def test_summarize_hits_invalid_ks_raise():
    records = [_record(base_top1=0, shadow_logits=[0.0, 1.0])]
    with pytest.raises(ValueError, match="ks"):
        summarize_hits(records, ks=(0,))
    with pytest.raises(ValueError, match="ks"):
        summarize_hits(records, ks=())


# ---------------------------------------------------------------------------
# Stratification
# ---------------------------------------------------------------------------


def test_group_metrics_by_task_counts_reconcile_with_total():
    records = [
        _record(base_top1=0, shadow_logits=[1.0, 0.0], task="fault", prompt_id="p0"),
        _record(base_top1=0, shadow_logits=[0.0, 1.0], task="fault", prompt_id="p0"),
        _record(base_top1=0, shadow_logits=[1.0, 0.0], task="fault", prompt_id="p1"),
        _record(base_top1=0, shadow_logits=[0.0, 1.0], task="calc", prompt_id="p1"),
        _record(base_top1=0, shadow_logits=[1.0, 0.0], task="calc", prompt_id="p2"),
        _record(base_top1=0, shadow_logits=[1.0, 0.0], task=None, prompt_id="p2"),
    ]

    overall = summarize_hits(records)
    groups = group_metrics(records, by="task")

    assert sum(metric["n"] for metric in groups.values()) == overall["n"] == 6
    assert set(groups) == {"calc", "fault", "unspecified"}
    assert groups["fault"]["n"] == 3
    assert groups["calc"]["n"] == 2
    assert groups["unspecified"]["n"] == 1
    assert groups["fault"]["hit_at_1"] == pytest.approx(2 / 3)


def test_group_metrics_by_position_bucket_counts_reconcile_with_total():
    positions = [0, 1, 3, 4, 7, 8, 15, 16, 99]
    records = [
        _record(base_top1=0, shadow_logits=[1.0, 0.0], position=position)
        for position in positions
    ]

    overall = summarize_hits(records)
    groups = group_metrics(records, by="position_bucket")

    assert sum(metric["n"] for metric in groups.values()) == overall["n"] == 9
    assert list(groups) == list(POSITION_BUCKETS)
    assert [groups[bucket]["n"] for bucket in POSITION_BUCKETS] == [1, 2, 2, 2, 2]


@pytest.mark.parametrize(
    ("position", "bucket"),
    [
        (0, "0"),
        (1, "1-3"),
        (3, "1-3"),
        (4, "4-7"),
        (7, "4-7"),
        (8, "8-15"),
        (15, "8-15"),
        (16, "16+"),
        (10_000, "16+"),
    ],
)
def test_position_bucket_boundaries(position, bucket):
    assert position_bucket(position) == bucket


def test_position_bucket_rejects_negative_positions():
    with pytest.raises(ValueError, match="position"):
        position_bucket(-1)


def test_group_metrics_rejects_unknown_grouping():
    records = [_record(base_top1=0, shadow_logits=[1.0, 0.0])]
    with pytest.raises(ValueError, match="by"):
        group_metrics(records, by="prompt")


def test_group_metrics_position_bucket_requires_position():
    records = [_record(base_top1=0, shadow_logits=[1.0, 0.0], position=None)]
    with pytest.raises(ValueError, match="position"):
        group_metrics(records, by="position_bucket")


# ---------------------------------------------------------------------------
# bootstrap_ci: deterministic percentile interval for Hit@8 / Agree@1
# ---------------------------------------------------------------------------


def test_bootstrap_ci_raw_values_contains_estimate_and_is_reproducible():
    values = [0, 1, 1, 0, 1, 1, 1, 0, 1, 1]

    first = bootstrap_ci(values, n_boot=200, seed=7)
    second = bootstrap_ci(values, n_boot=200, seed=7)

    assert isinstance(first, BootstrapResult)
    assert first == second
    assert first.metric == "mean"
    assert first.n == 10
    assert first.estimate == pytest.approx(0.7)
    assert first.low <= first.estimate <= first.high
    assert 0.0 <= first.low <= first.high <= 1.0
    assert tuple(first) == (first.low, first.high)
    assert first.to_dict()["n"] == 10


def test_bootstrap_ci_all_identical_values_is_degenerate_interval():
    result = bootstrap_ci([1, 1, 1, 1], n_boot=50, seed=0)

    assert result.estimate == 1.0
    assert result.low == 1.0
    assert result.high == 1.0
    assert result.n == 4


def test_bootstrap_ci_extracts_hit8_and_agree1_from_records():
    records = [
        _record(base_top1=0, shadow_logits=[1.0, 0.0]),
        _record(base_top1=0, shadow_logits=[0.0, 1.0]),
        _record(base_top1=0, shadow_topk_ids=[0, 1]),
        _record(base_top1=9, shadow_topk_ids=[0, 1]),
    ]
    summary = summarize_hits(records)

    hit8 = bootstrap_ci(records, metric="hit@8", n_boot=100, seed=11)
    agree = bootstrap_ci(records, metric="agree@1", n_boot=100, seed=11)

    assert hit8.metric == HIT8_METRIC
    assert hit8.n == 4
    assert hit8.estimate == pytest.approx(summary["hit_at_8"])
    assert agree.metric == AGREE1_METRIC
    assert agree.n == 4
    assert agree.estimate == pytest.approx(summary["agree_at_1"])


def test_bootstrap_ci_accepts_metric_alias_and_confidence():
    records = [_record(base_top1=0, shadow_logits=[1.0, 0.0])]

    result = bootstrap_ci(records, metric="hit_at_8", n_boot=20, seed=2, confidence=0.9)

    assert result.metric == HIT8_METRIC
    assert result.confidence == 0.9


def test_bootstrap_ci_empty_values_raise():
    with pytest.raises(ValueError, match="at least one value"):
        bootstrap_ci([], n_boot=10)


def test_bootstrap_ci_invalid_n_boot_raises():
    with pytest.raises(ValueError, match="n_boot"):
        bootstrap_ci([1, 0], n_boot=0)


def test_bootstrap_ci_unknown_metric_raises():
    with pytest.raises(ValueError, match="metric"):
        bootstrap_ci([1, 0], metric="rank")


# ---------------------------------------------------------------------------
# go_no_go: engineering thresholds from the spec
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("hit8", "decision"),
    [
        (1.0, "go"),
        (0.85, "go"),
        (0.849999, "pilot"),
        (0.70, "pilot"),
        (0.699999, "no_go"),
        (0.0, "no_go"),
    ],
)
def test_go_no_go_threshold_boundaries(hit8, decision):
    result = go_no_go(hit8, n=100, ci=(0.60, 0.95))

    assert isinstance(result, GoNoGoDecision)
    assert result.decision == decision
    assert result.hit8 == pytest.approx(hit8)
    assert result.n == 100
    assert (result.ci_low, result.ci_high) == (0.60, 0.95)
    assert result.go_threshold == GO_THRESHOLD
    assert result.pilot_threshold == PILOT_THRESHOLD
    assert result["decision"] == decision
    assert result["n"] == 100
    assert result.get("ci") == (0.60, 0.95)
    assert result.to_dict()["decision"] == decision


def test_go_no_go_accepts_bootstrap_result():
    records = [
        _record(base_top1=0, shadow_logits=[1.0, 0.0]),
        _record(base_top1=0, shadow_logits=[0.0, 1.0]),
        _record(base_top1=9, shadow_topk_ids=[0, 1]),
        _record(base_top1=9, shadow_topk_ids=[0, 1]),
    ]
    estimate = bootstrap_ci(records, metric="hit@8", n_boot=100, seed=5)

    result = go_no_go(estimate)

    assert estimate.estimate == 0.5
    assert result.decision == "no_go"
    assert result.n == 4
    assert (result.ci_low, result.ci_high) == (estimate.low, estimate.high)


@pytest.mark.parametrize("hit8", [-0.01, 1.01, math.nan])
def test_go_no_go_rejects_invalid_hit8(hit8):
    with pytest.raises(ValueError, match="hit8"):
        go_no_go(hit8)


# ---------------------------------------------------------------------------
# Report assembly and stable JSON/CSV exports
# ---------------------------------------------------------------------------


def _report_records() -> list[dict]:
    records = [
        _record(
            base_top1=0,
            shadow_logits=[1.0, 0.0],
            task="fault" if index < 5 else "calc",
            position=index,
            prompt_id=f"p{index // 2}",
        )
        for index in range(9)
    ]
    records.append(
        _record(
            base_top1=9,
            shadow_topk_ids=[0, 1],
            task="calc",
            position=20,
            prompt_id="p9",
        )
    )
    return records


def test_build_report_assembles_metrics_cis_and_decision():
    records = _report_records()

    report = build_report(records, n_boot=100, seed=3)

    assert set(REPORT_FIELDS) <= set(report)
    assert report["schema_version"] == SCHEMA_VERSION
    assert report["shadow_source"] == "initial_shadow_hidden"
    assert "initial shadow" in report["shadow_source_note"]
    assert report["n_tokens"] == 10
    assert report["n_prompts"] == 6
    assert list(report["ks"]) == list(DEFAULT_KS)
    assert report["hit_at_8"] == pytest.approx(0.9)
    assert report["agree_at_1"] == pytest.approx(0.9)
    assert report["hit_at_1"] == pytest.approx(0.9)
    assert report["go_no_go"] == "go"
    assert report["ci_n_boot"] == 100
    assert report["ci_seed"] == 3
    assert len(report["hit_at_8_ci"]) == 2
    assert len(report["agree_at_1_ci"]) == 2
    assert report["hit_at_8_ci"][0] <= report["hit_at_8"] <= report["hit_at_8_ci"][1]
    assert sum(metric["n"] for metric in report["by_task"].values()) == 10
    assert sum(metric["n"] for metric in report["by_position_bucket"].values()) == 10


def test_build_report_keeps_standard_fields_for_custom_ks_and_derives_hit8():
    records = _report_records()

    report = build_report(records, ks=(2,), n_boot=20, seed=0)

    assert report["hit_at_2"] == pytest.approx(0.9)
    assert report["hit_at_8"] == pytest.approx(0.9)
    assert report["go_no_go"] == "go"
    assert list(report["ks"]) == [2]


def test_to_json_is_stable_round_trips_and_has_no_nan():
    report = build_report(_report_records(), n_boot=50, seed=4)

    first = to_json(report)
    second = to_json(report)

    assert first == second
    assert "NaN" not in first
    payload = json.loads(first)
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["hit_at_8"] == pytest.approx(0.9)
    assert payload["go_no_go"] == "go"
    assert set(REPORT_FIELDS) <= set(payload)


def test_to_json_rejects_nan_instead_of_emitting_invalid_payload():
    with pytest.raises(ValueError):
        to_json({"hit_at_8": math.nan})


def test_to_csv_has_stable_header_and_reconciled_group_rows():
    report = build_report(_report_records(), n_boot=50, seed=4)

    first = to_csv(report)
    second = to_csv(report)

    assert first == second
    reader = csv.DictReader(io.StringIO(first))
    assert reader.fieldnames == list(CSV_FIELDS)
    rows = list(reader)
    assert len(rows) == 1 + len(report["by_task"]) + len(report["by_position_bucket"])

    overall = rows[0]
    assert overall["scope"] == "overall"
    assert overall["group"] == "all"
    assert float(overall["hit_at_8"]) == pytest.approx(0.9)
    assert overall["go_no_go"] == "go"
    assert all(row["group"] for row in rows)
    assert sum(float(row["n_tokens"]) for row in rows if row["scope"] == "task") == 10
    assert sum(
        float(row["n_tokens"]) for row in rows if row["scope"] == "position_bucket"
    ) == 10


def test_to_csv_accepts_a_plain_summary():
    summary = summarize_hits(_report_records())

    rows = list(csv.DictReader(io.StringIO(to_csv(summary))))

    assert len(rows) == 1
    assert rows[0]["scope"] == "overall"
    assert float(rows[0]["hit_at_8"]) == pytest.approx(summary["hit_at_8"])


def test_bootstrap_ci_defaults_records_to_hit8():
    records = [
        _record(base_top1=0, shadow_logits=[1.0, 0.0]),
        _record(base_top1=9, shadow_topk_ids=[0, 1]),
    ]
    summary = summarize_hits(records)

    result = bootstrap_ci(records, n_boot=50, seed=3)

    assert result.metric == HIT8_METRIC
    assert result.n == 2
    assert result.estimate == pytest.approx(summary["hit_at_8"])


def test_build_report_tolerates_records_without_positions():
    records = [_record(base_top1=0, shadow_logits=[1.0, 0.0], position=None)]

    report = build_report(records, n_boot=20, seed=0)

    assert report["n_tokens"] == 1
    assert report["by_position_bucket"] == {}
    assert report["by_task"]
