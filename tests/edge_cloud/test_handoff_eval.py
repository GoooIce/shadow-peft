from __future__ import annotations

import csv
import json
from dataclasses import fields
from pathlib import Path

import pytest

from edge_cloud import knowledge
from edge_cloud.backends import MockStreamingEdgeBackend
from edge_cloud.boundaries import detect_boundary
from edge_cloud.handoff import HandoffDecision, HandoffFeatures, HandoffStep
from edge_cloud.handoff_eval import (
    NUMERIC_HANDOFF_FEATURES,
    STRATEGIES,
    EvalSample,
    apply_cloud_answer,
    bootstrap_ci,
    count_tokens,
    export_candidates,
    fit_candidates,
    load_candidates,
    load_dataset,
    load_results,
    run_evaluation,
    synthetic_dataset,
    write_candidates,
    write_outputs,
)
from edge_cloud.probe import PROBE_VERSION, ProbeModel, write_probe
from edge_cloud.protocol import InferRequest


class _NeverPolicy:
    def reset(self, context: dict | None = None) -> None:
        del context

    def observe(self, step: HandoffStep) -> HandoffDecision:
        del step
        return HandoffDecision(action="continue")


def test_token_counter_matches_the_mock_streaming_backend() -> None:
    for entry in knowledge.ENTRIES:
        request = InferRequest(prompt=entry.keywords[0])
        generation = MockStreamingEdgeBackend().generate_until(request, _NeverPolicy())

        assert count_tokens(generation.text) == generation.output_tokens


def test_synthetic_dataset_is_deterministic_and_uses_prefix_semantics() -> None:
    first = synthetic_dataset(12, seed=7)
    second = synthetic_dataset(12, seed=7)

    assert first == second
    assert [sample.sample_id for sample in first] == [f"s{index:04d}" for index in range(12)]

    hard = [sample for sample in first if sample.cloud_better_from is not None]
    easy = [sample for sample in first if sample.cloud_better_from is None]
    assert hard and easy

    for sample in first:
        entry = knowledge.lookup(sample.prompt)
        assert entry.key != "unknown", sample.prompt
        assert sample.cloud_answer == entry.cloud_answer
        assert sample.expected == sample.cloud_answer
        assert sample.shadow_gap is not None

    for sample in hard:
        # The edge answer is a proper boundary-truncated prefix of the cloud answer.
        assert sample.edge_answer != sample.cloud_answer
        assert sample.cloud_answer.startswith(sample.edge_answer)
        assert detect_boundary(sample.edge_answer) in {"sentence", "newline"}
        assert sample.cloud_better_from == sample.risk_at
        assert sample.risk_at is not None
        # The oracle can only hand off when the label points inside the stream.
        assert 0 <= sample.risk_at < count_tokens(sample.edge_answer)

    for sample in easy:
        assert sample.edge_answer == sample.cloud_answer
        assert sample.risk_at is None
        assert sample.cloud_better_from is None


def test_streaming_backend_can_override_the_knowledge_base_answer() -> None:
    answer = "自定义端侧答案，截断在边界。"

    generation = MockStreamingEdgeBackend(answer=answer).generate_until(
        InferRequest(prompt="回流焊峰值温度"), _NeverPolicy()
    )

    assert generation.text == answer
    assert generation.output_tokens == count_tokens(answer)
    with pytest.raises(ValueError, match="answer"):
        MockStreamingEdgeBackend(answer="   ")


def test_synthetic_dataset_uses_a_different_seed_for_different_perturbations() -> None:
    first = synthetic_dataset(8, seed=1)
    second = synthetic_dataset(8, seed=2)

    assert [sample.prompt for sample in first] != [sample.prompt for sample in second]


def test_load_dataset_defaults_optional_fields_and_preserves_order(tmp_path: Path) -> None:
    path = tmp_path / "dataset.jsonl"
    rows = [
        {
            "prompt": "回流焊峰值温度是多少？",
            "task": "process_qa",
            "expected": "235–245℃",
            "edge_answer": "235–245℃",
            "cloud_answer": "235–245℃ 并配合厂商曲线",
        },
        {
            "prompt": "E-104 怎么处理？",
            "task": "fault_diagnosis",
            "expected": "检查吸嘴",
            "edge_answer": "检查吸嘴",
            "cloud_answer": "检查吸嘴与真空管路",
            "cloud_better_from": 2,
            "risk_at": 2,
            "edge_quality": 0.8,
            "shadow_gap": 0.2,
            "max_tokens": 16,
        },
    ]
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )

    samples = load_dataset(path)

    assert [sample.sample_id for sample in samples] == ["s0000", "s0001"]
    assert samples[0].cloud_better_from is None
    assert samples[0].risk_at is None
    assert samples[0].edge_quality is None
    assert samples[0].shadow_gap is None
    assert samples[0].max_tokens == 256
    assert samples[1].cloud_better_from == 2
    assert samples[1].edge_quality == 0.8
    assert samples[1].shadow_gap == 0.2
    assert samples[1].max_tokens == 16


def test_load_dataset_rejects_an_invalid_max_tokens(tmp_path: Path) -> None:
    path = tmp_path / "dataset.jsonl"
    path.write_text(
        json.dumps(
            {
                "prompt": "p",
                "task": "general",
                "expected": "e",
                "edge_answer": "a",
                "cloud_answer": "c",
                "max_tokens": 0,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_dataset(path)


def test_dataset_dataclass_defaults_match_the_jsonl_contract() -> None:
    sample = EvalSample(
        sample_id="s0000",
        prompt="p",
        task="general",
        expected="e",
        edge_answer="a",
        cloud_answer="c",
    )

    assert sample.cloud_better_from is None
    assert sample.risk_at is None
    assert sample.edge_quality is None
    assert sample.shadow_gap is None


def test_numeric_feature_names_are_handoff_feature_fields() -> None:
    feature_fields = {field.name for field in fields(HandoffFeatures)}

    assert set(NUMERIC_HANDOFF_FEATURES) | {"boundary_kind", "task"} == feature_fields
    assert "boundary_kind" not in NUMERIC_HANDOFF_FEATURES
    assert "task" not in NUMERIC_HANDOFF_FEATURES


def _probe_artifact(tmp_path: Path, samples: list[EvalSample]) -> Path:
    rows = export_candidates(samples)
    write_candidates(rows, tmp_path / "candidates.jsonl")
    loaded = load_candidates(tmp_path / "candidates.jsonl")
    payload = fit_candidates(loaded, threshold=0.5, trained_at="2026-01-01T00:00:00+00:00")
    path = tmp_path / "probe.json"
    write_probe(payload, path)
    return path


def test_all_strategies_share_the_result_schema_and_artifacts(tmp_path: Path) -> None:
    samples = synthetic_dataset(8, seed=5)
    run = run_evaluation(samples, strategies=STRATEGIES, seed=5, n_boot=30)

    results = run.results
    assert results["schema_version"] == "exp9-handoff-eval/v1"
    assert results["seed"] == 5
    assert results["n"] == 8
    assert results["backend"]["kind"] == "synthetic"
    assert results["backend"]["streaming"] is True
    assert set(results["strategies"]) == set(STRATEGIES)

    metric_keys = {name: set(metrics) for name, metrics in results["strategies"].items()}
    assert len(set(map(frozenset, metric_keys.values()))) == 1
    for strategy in STRATEGIES:
        metrics = results["strategies"][strategy]
        assert metrics["n"] == 8
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert metrics["accuracy"] == metrics["exact_match_rate"]
        assert 0.0 <= metrics["contains_expected_rate"] <= 1.0
        assert results["curves"][strategy]["risk_coverage"]
        assert results["curves"][strategy]["cost_accuracy"]
        assert set(results["ci"][strategy]) == {"accuracy", "cost"}

    paths = write_outputs(run, tmp_path / "run")
    assert paths["results"].name == "results.json"
    assert paths["summary"].name == "summary.csv"
    assert paths["curves"].name == "curves.csv"
    assert paths["per_prompt"].name == "per_prompt.jsonl"
    assert load_results(paths["results"]) == results

    rows = paths["per_prompt"].read_text(encoding="utf-8").splitlines()
    assert len(rows) == len(STRATEGIES) * len(samples)
    first = json.loads(rows[0])
    assert first["strategy"] == "edge_only"
    assert first["sample_id"] == "s0000"
    assert "exact_match" in first
    assert "correct" not in first

    with paths["summary"].open("r", encoding="utf-8", newline="") as handle:
        summary = list(csv.DictReader(handle))
    assert [row["strategy"] for row in summary] == list(STRATEGIES)

    with paths["curves"].open("r", encoding="utf-8", newline="") as handle:
        curve_rows = list(csv.DictReader(handle))
    expected_points = sum(
        len(curves["risk_coverage"]) + len(curves["cost_accuracy"])
        for curves in results["curves"].values()
    )
    assert len(curve_rows) == expected_points
    assert {row["curve"] for row in curve_rows} == {"risk_coverage", "cost_accuracy"}
    oracle_cuts = [
        row for row in curve_rows if row["strategy"] == "oracle" and row["curve"] == "cost_accuracy"
    ]
    assert all(row["x_name"] == "cut" for row in oracle_cuts)


def test_repeated_runs_are_byte_identical(tmp_path: Path) -> None:
    samples = synthetic_dataset(10, seed=13)

    first = run_evaluation(samples, seed=13, n_boot=25)
    second = run_evaluation(samples, seed=13, n_boot=25)
    first_paths = write_outputs(first, tmp_path / "first")
    second_paths = write_outputs(second, tmp_path / "second")

    for name in ("results", "summary", "curves", "per_prompt"):
        assert first_paths[name].read_bytes() == second_paths[name].read_bytes(), name


def test_oracle_accuracy_is_at_least_probe_and_legacy(tmp_path: Path) -> None:
    samples = synthetic_dataset(24, seed=11)
    probe_path = _probe_artifact(tmp_path, samples)

    run = run_evaluation(
        samples,
        strategies=STRATEGIES,
        seed=11,
        n_boot=20,
        probe_path=probe_path,
    )
    metrics = run.results["strategies"]

    assert metrics["probe"]["probe_fallback_rate"] == 0.0
    assert metrics["oracle"]["accuracy"] >= metrics["probe"]["accuracy"]
    assert metrics["oracle"]["accuracy"] >= metrics["legacy"]["accuracy"]


def test_hard_samples_need_a_handoff_and_easy_samples_are_correct_on_edge() -> None:
    samples = synthetic_dataset(30, seed=9)
    run = run_evaluation(samples, strategies=STRATEGIES, seed=9, n_boot=0)

    hard_ids = {sample.sample_id for sample in samples if sample.cloud_better_from is not None}
    easy_ids = {sample.sample_id for sample in samples if sample.cloud_better_from is None}
    by_strategy = {
        name: [record for record in run.per_prompt if record["strategy"] == name]
        for name in STRATEGIES
    }

    oracle_hard = [record for record in by_strategy["oracle"] if record["sample_id"] in hard_ids]
    assert len(oracle_hard) == len(hard_ids)
    assert sum(record["exact_match"] for record in oracle_hard) / len(oracle_hard) >= 0.9

    edge_hard = [record for record in by_strategy["edge_only"] if record["sample_id"] in hard_ids]
    assert edge_hard and not any(record["exact_match"] for record in edge_hard)
    edge_easy = [record for record in by_strategy["edge_only"] if record["sample_id"] in easy_ids]
    assert edge_easy and all(record["exact_match"] for record in edge_easy)
    assert 0.0 < run.results["strategies"]["edge_only"]["accuracy"] < 0.6

    for strategy in STRATEGIES:
        metrics = run.results["strategies"][strategy]
        assert metrics["accuracy"] == metrics["exact_match_rate"]


def test_probe_without_an_artifact_falls_back_to_legacy() -> None:
    samples = synthetic_dataset(8, seed=2)

    run = run_evaluation(samples, strategies=("legacy", "probe"), seed=2, n_boot=10)
    metrics = run.results["strategies"]

    assert metrics["probe"]["probe_fallback_rate"] == 1.0
    assert metrics["probe"]["accuracy"] == metrics["legacy"]["accuracy"]
    assert metrics["probe"]["escalation_rate"] == metrics["legacy"]["escalation_rate"]


def test_bootstrap_small_sample_interval_is_wider_than_large_sample() -> None:
    small = [1.0, 0.0] * 4
    large = [1.0, 0.0] * 32

    small_ci = bootstrap_ci(small, n_boot=500, seed=3)
    large_ci = bootstrap_ci(large, n_boot=500, seed=3)

    assert small_ci["low"] is not None and small_ci["high"] is not None
    assert large_ci["low"] is not None and large_ci["high"] is not None
    assert small_ci["high"] - small_ci["low"] > large_ci["high"] - large_ci["low"]
    assert small_ci["point"] == pytest.approx(0.5)
    assert large_ci["point"] == pytest.approx(0.5)


def test_exported_candidates_cover_both_labels_and_fit_a_loadable_probe(
    tmp_path: Path,
) -> None:
    samples = synthetic_dataset(12, seed=3)
    rows = export_candidates(samples)

    assert {row["label"] for row in rows} == {0, 1}
    assert all(set(row["features"]) == set(NUMERIC_HANDOFF_FEATURES) for row in rows)
    assert all(isinstance(row["prompt_id"], str) and row["prompt_id"] for row in rows)

    path = write_candidates(rows, tmp_path / "candidates.jsonl")
    loaded = load_candidates(path)
    assert loaded == rows

    payload = fit_candidates(loaded, trained_at="2026-01-01T00:00:00+00:00")
    assert payload["version"] == PROBE_VERSION
    probe_path = tmp_path / "probe.json"
    write_probe(payload, probe_path)
    model = ProbeModel.load(probe_path)

    assert set(model.feature_names) == set(NUMERIC_HANDOFF_FEATURES)
    score = model.score(loaded[0]["features"])
    assert 0.0 < score < 1.0


def test_continuation_strips_a_true_prefix_and_joins_a_full_answer() -> None:
    prefix = "判定：返修。"
    cloud = "判定：返修。依据 IPC-A-610 Class 2，空洞不超过 25%。"
    final, continuation, mismatch = apply_cloud_answer(prefix, cloud)
    assert final == cloud
    assert continuation == "依据 IPC-A-610 Class 2，空洞不超过 25%。"
    assert mismatch is False

    other = "E-104 表示贴片头真空压力不足：先检查真空管路。"
    full = "E-104 表示贴片头真空压力不足。处理顺序：1）检查吸嘴；2）检查过滤棉。"
    final, continuation, mismatch = apply_cloud_answer(other, full)
    assert final == other + full
    assert continuation == full
    assert mismatch is True

    final, continuation, mismatch = apply_cloud_answer("", full)
    assert final == full
    assert continuation == full
    assert mismatch is False


def test_curves_report_coverage_and_cost_operating_points() -> None:
    samples = synthetic_dataset(8, seed=6)
    run = run_evaluation(samples, seed=6, n_boot=0)

    for strategy in STRATEGIES:
        risk_curve = run.results["curves"][strategy]["risk_coverage"]
        cost_curve = run.results["curves"][strategy]["cost_accuracy"]
        assert len(risk_curve) == len(samples)
        assert [point["coverage"] for point in risk_curve] == [
            pytest.approx((index + 1) / len(samples)) for index in range(len(samples))
        ]
        assert risk_curve[-1]["coverage"] == 1.0
        assert cost_curve
        for point in cost_curve:
            assert 0.0 <= point["accuracy"] <= 1.0
            assert point["cost"] >= 0.0
        if strategy == "oracle":
            assert all("cut" in point for point in cost_curve)
        elif strategy != "edge_only":
            assert all("threshold" in point for point in cost_curve)

    assert all(ci["accuracy"]["low"] is None for ci in run.results["ci"].values())


def test_missing_cloud_answer_degrades_instead_of_crashing() -> None:
    sample = EvalSample(
        sample_id="s0000",
        prompt="回流焊峰值温度是多少？",
        task="process_qa",
        expected="回流焊峰值温度 235–245℃。",
        edge_answer="回流焊峰值温度通常控制在 235–245℃。",
        cloud_answer="",
        cloud_better_from=0,
        risk_at=0,
        edge_quality=0.9,
        shadow_gap=0.2,
    )

    run = run_evaluation([sample], strategies=("legacy", "probe", "oracle"), seed=1, n_boot=0)

    for strategy in ("legacy", "probe", "oracle"):
        metrics = run.results["strategies"][strategy]
        assert metrics["degraded_rate"] == 1.0
        assert metrics["cloud_calls"] == 0
    for record in run.per_prompt:
        assert record["final_text"] == record["edge_text"]


def test_reader_accepts_a_real_backend_shape_and_rejects_bad_schemas(
    tmp_path: Path,
) -> None:
    samples = synthetic_dataset(4, seed=4)
    run = run_evaluation(samples, seed=4, n_boot=0, backend_kind="transformers")
    paths = write_outputs(run, tmp_path / "run")
    loaded = load_results(paths["results"])

    assert loaded["backend"]["kind"] == "transformers"
    assert loaded["schema_version"] == "exp9-handoff-eval/v1"

    payload = json.loads(paths["results"].read_text(encoding="utf-8"))
    payload["schema_version"] = "exp9-handoff-eval/v99"
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        load_results(bad)


def test_unknown_strategy_is_rejected() -> None:
    with pytest.raises(ValueError, match="未知策略"):
        run_evaluation(synthetic_dataset(2), strategies=("edge_only", "magic"))
