"""exp15 cost-policy experiment — external-behavior tests (issue #52).

Spec: issue #48 §6–§7. The tests drive the experiment script's public
functions only: the synthetic builder, the replay JSONL loader, the evidence
gate, the go/no-go evaluator, the experiment report and the CLI artifacts.
No private helper and no router/backend is touched.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from edge_cloud.config import EdgeConfig
from edge_cloud.cost import CostModel
from edge_cloud.cost_policy import (
    DEFAULT_THRESHOLD_GRID,
    CostSample,
    fit_cost_policy,
)
from edge_cloud.handoff_eval import count_tokens

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "exp15_cost_policy.py"


@pytest.fixture(scope="module")
def exp15():
    spec = importlib.util.spec_from_file_location("exp15_cost_policy", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # ``from __future__ import annotations`` makes dataclasses resolve their
    # field types through ``sys.modules``, so register before executing.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Synthetic dataset
# ---------------------------------------------------------------------------


def test_synthetic_samples_cover_every_knowledge_entry_and_both_splits(exp15):
    samples = exp15.build_synthetic_samples(variants_per_entry=4)
    entries = exp15.knowledge.ENTRIES

    assert len(samples) == len(entries) * 4
    assert {sample.task for sample in samples} == {entry.task for entry in entries}
    assert {sample.split for sample in samples} == {"train", "val"}
    for task in {entry.task for entry in entries}:
        assert {sample.split for sample in samples if sample.task == task} == {
            "train",
            "val",
        }
    assert len({sample.sample_id for sample in samples}) == len(samples)

    by_task_and_score = {(entry.task, entry.edge_quality): entry for entry in entries}
    for sample in samples:
        entry = by_task_and_score[(sample.task, sample.score)]
        assert sample.edge_correct is (entry.edge_quality >= 0.5)
        assert sample.cloud_correct is (entry.cloud_quality >= 0.5)
        assert sample.edge_cost > 0.0
        assert sample.cloud_only_cost > 0.0
        assert sample.escalated_cost > sample.cloud_only_cost
        assert sample.edge_latency_ms is not None
        assert sample.escalated_latency_ms is not None


def test_synthetic_samples_are_deterministic_for_a_fixed_seed(exp15):
    first = exp15.build_synthetic_samples(variants_per_entry=4, seed=15)
    second = exp15.build_synthetic_samples(variants_per_entry=4, seed=15)

    assert [sample.to_dict() for sample in first] == [sample.to_dict() for sample in second]
    other_seed = exp15.build_synthetic_samples(variants_per_entry=4, seed=16)
    assert [sample.to_dict() for sample in first] != [sample.to_dict() for sample in other_seed]


def test_synthetic_costs_follow_count_tokens_and_the_cost_model(exp15):
    model = CostModel(
        cost_per_edge_token=1.0,
        cost_per_cloud_input_token=10.0,
        cost_per_cloud_output_token=100.0,
        cost_per_cloud_call=1000.0,
        cloud_rtt_ms=0.0,
    )
    samples = exp15.build_synthetic_samples(variants_per_entry=2, seed=15, cost_model=model)
    entries = list(exp15.knowledge.ENTRIES)

    for sample in samples:
        entry = next(entry for entry in entries if sample.sample_id.startswith(f"syn-{entry.key}-"))
        edge_tokens = count_tokens(entry.edge_answer)
        cloud_tokens = count_tokens(entry.cloud_answer)
        assert sample.edge_cost == pytest.approx(edge_tokens)
        # cloud_only = prompt*10 + cloud*100 + 1000; the prompt prefill is
        # strictly positive, so the cloud answer term cannot account for it all.
        assert sample.cloud_only_cost - cloud_tokens * 100.0 - 1000.0 > 0.0
        # escalated = edge + (prompt + edge)*10 + cloud*100 + 1000, so the
        # difference against cloud_only is exactly edge decode + edge prefill.
        assert sample.escalated_cost - sample.cloud_only_cost == pytest.approx(11.0 * edge_tokens)


# ---------------------------------------------------------------------------
# Replay JSONL loader
# ---------------------------------------------------------------------------


def _row(**overrides):
    row = {
        "sample_id": "r0001",
        "task": "process_qa",
        "score": 0.7,
        "edge_correct": True,
        "cloud_correct": True,
        "edge_cost": 0.01,
        "escalated_cost": 0.05,
        "cloud_only_cost": 0.08,
        "split": "train",
        "edge_latency_ms": 12.0,
        "escalated_latency_ms": 140.0,
    }
    row.update(overrides)
    return row


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    return path


def test_load_replay_round_trips_cost_samples(tmp_path, exp15):
    path = _write_jsonl(
        tmp_path / "replay.jsonl",
        [_row(), _row(sample_id="r0002", split="val", score=0.4)],
    )

    samples = exp15.load_cost_samples(path)

    assert len(samples) == 2
    assert all(isinstance(sample, CostSample) for sample in samples)
    assert samples[0] == CostSample(
        sample_id="r0001",
        task="process_qa",
        score=0.7,
        edge_correct=True,
        cloud_correct=True,
        edge_cost=0.01,
        escalated_cost=0.05,
        cloud_only_cost=0.08,
        split="train",
        edge_latency_ms=12.0,
        escalated_latency_ms=140.0,
    )
    assert samples[1].split == "val"
    assert samples[1].score == pytest.approx(0.4)


def test_load_replay_requires_an_explicit_split(tmp_path, exp15):
    row = _row()
    row.pop("split")
    path = _write_jsonl(tmp_path / "replay.jsonl", [row])

    with pytest.raises(ValueError, match="split"):
        exp15.load_cost_samples(path)


def test_load_replay_rejects_unknown_splits_and_missing_counterfactuals(tmp_path, exp15):
    with pytest.raises(ValueError, match="split"):
        exp15.load_cost_samples(_write_jsonl(tmp_path / "bad_split.jsonl", [_row(split="test")]))

    row = _row()
    row.pop("cloud_correct")
    with pytest.raises(ValueError, match="cloud_correct"):
        exp15.load_cost_samples(_write_jsonl(tmp_path / "missing_label.jsonl", [row]))


def test_load_replay_rejects_empty_files(tmp_path, exp15):
    path = tmp_path / "empty.jsonl"
    path.write_text("\n", encoding="utf-8")

    with pytest.raises(ValueError, match="空"):
        exp15.load_cost_samples(path)


# ---------------------------------------------------------------------------
# Replay samples, evidence gate and go/no-go
# ---------------------------------------------------------------------------


def _replay_sample(
    task: str,
    score: float,
    *,
    correct_from: float,
    split: str,
    index: int,
) -> CostSample:
    return CostSample(
        sample_id=f"{task}-{index:04d}",
        task=task,
        score=score,
        edge_correct=score >= correct_from,
        cloud_correct=True,
        edge_cost=0.1,
        escalated_cost=0.5,
        cloud_only_cost=1.0,
        split=split,
    )


def _replay_samples(
    *, per_task: int = 250, correct_from: dict[str, float] | None = None
) -> list[CostSample]:
    cuts = correct_from or {"task_a": 0.3, "task_b": 0.7}
    samples: list[CostSample] = []
    for task, cut in cuts.items():
        for index in range(per_task):
            score = round(0.1 + (index % 9) * 0.1, 1)
            split = "train" if index % 2 == 0 else "val"
            samples.append(_replay_sample(task, score, correct_from=cut, split=split, index=index))
    return samples


def _eval_report(
    accuracy: float,
    cost: float,
    *,
    objective: float | None = None,
    escalation: float = 0.4,
    tasks: dict | None = None,
) -> dict:
    if objective is None:
        objective = accuracy - 0.1 * cost
    if tasks is None:
        tasks = {
            "task_a": {
                "n": 50,
                "accuracy": accuracy,
                "cost": cost,
                "cost_norm": cost,
                "objective": objective,
                "escalation_rate": escalation,
                "cloud_calls": int(50 * escalation),
                "threshold": 0.3,
                "always_cloud": False,
                "latency_p95_ms": None,
            }
        }
    return {
        "accuracy": accuracy,
        "cost": cost,
        "cost_norm": cost,
        "objective": objective,
        "escalation_rate": escalation,
        "cloud_calls": 20,
        "per_task": tasks,
        "global_threshold": 0.5,
        "thresholds_by_task": {"task_a": 0.3},
    }


def test_go_no_go_passes_when_cost_drops_twenty_percent_at_same_accuracy(exp15):
    per_task = _eval_report(0.9, 0.8)
    baseline = _eval_report(0.9, 1.0)

    gate = exp15.evaluate_go_no_go(
        per_task, baseline, per_task, baseline, fitted_tasks=("task_a",), min_samples=1
    )

    assert gate["status"] == "go"
    assert gate["failed_checks"] == []
    assert gate["checks"]["primary_condition"] is True


def test_go_no_go_passes_on_the_second_condition(exp15):
    per_task = _eval_report(0.91, 1.0)
    baseline = _eval_report(0.9, 1.0)

    gate = exp15.evaluate_go_no_go(
        per_task, baseline, per_task, baseline, fitted_tasks=("task_a",), min_samples=1
    )

    assert gate["status"] == "go"
    assert gate["checks"]["accuracy_plus_1pp"] is True
    assert gate["checks"]["cost_at_most_80pct"] is False


def test_go_no_go_fails_when_neither_condition_holds(exp15):
    per_task = _eval_report(0.9, 0.9)
    baseline = _eval_report(0.9, 1.0)

    gate = exp15.evaluate_go_no_go(
        per_task, baseline, per_task, baseline, fitted_tasks=("task_a",), min_samples=1
    )

    assert gate["status"] == "no_go"
    assert gate["failed_checks"] == ["primary_condition"]


def test_go_no_go_fails_on_escalation_cap_and_direction_inversion(exp15):
    per_task_tasks = {
        "task_a": {
            "n": 50,
            "accuracy": 0.9,
            "cost": 0.8,
            "cost_norm": 0.8,
            "objective": 0.82,
            "escalation_rate": 0.7,
            "cloud_calls": 35,
            "threshold": 0.3,
            "always_cloud": False,
            "latency_p95_ms": None,
        }
    }
    per_task_val = _eval_report(0.9, 0.8, objective=0.82, tasks=per_task_tasks)
    baseline_val = _eval_report(0.9, 1.0, objective=0.9)
    per_task_train = _eval_report(0.9, 0.8, objective=0.82, tasks=per_task_tasks)
    baseline_train = _eval_report(0.9, 1.0, objective=0.8)

    gate = exp15.evaluate_go_no_go(
        per_task_val,
        baseline_val,
        per_task_train,
        baseline_train,
        fitted_tasks=("task_a",),
        min_samples=1,
    )

    assert gate["status"] == "no_go"
    assert gate["failed_checks"] == [
        "no_task_over_escalation_cap",
        "no_direction_inversion",
    ]
    assert gate["escalation_violations"] == [{"task": "task_a", "rate": 0.7}]
    assert gate["inverted_tasks"] == ["task_a"]


def test_go_no_go_exempts_always_cloud_tasks_from_the_escalation_cap(exp15):
    tasks = {
        "task_a": {
            "n": 50,
            "accuracy": 0.9,
            "cost": 0.8,
            "cost_norm": 0.8,
            "objective": 0.82,
            "escalation_rate": 0.4,
            "cloud_calls": 20,
            "threshold": 0.3,
            "always_cloud": False,
            "latency_p95_ms": None,
        },
        "quality_verdict": {
            "n": 50,
            "accuracy": 0.97,
            "cost": 0.5,
            "cost_norm": 0.5,
            "objective": 0.92,
            "escalation_rate": 1.0,
            "cloud_calls": 50,
            "threshold": None,
            "always_cloud": True,
            "latency_p95_ms": None,
        },
    }
    per_task = _eval_report(0.9, 0.8, tasks=tasks)
    baseline = _eval_report(0.9, 1.0)
    baseline["per_task"] = tasks

    gate = exp15.evaluate_go_no_go(
        per_task,
        baseline,
        per_task,
        baseline,
        fitted_tasks=("task_a",),
        min_samples=1,
        always_cloud_tasks=("quality_verdict",),
    )

    assert gate["status"] == "go"
    assert gate["escalation_violations"] == []
    assert gate["always_cloud_exempt_tasks"] == ["quality_verdict"]


def test_evidence_gate_requires_replay_and_enough_samples(exp15):
    samples = _replay_samples(per_task=20)
    fit = fit_cost_policy(
        samples, cost_weight=0.1, min_samples=5, threshold_grid=DEFAULT_THRESHOLD_GRID
    )

    evidence = exp15.evaluate_evidence(samples, dataset_source="replay", fit=fit, min_samples=5)

    assert evidence["claims_gain"] is False
    assert evidence["evidence"] == "insufficient"
    assert any("500" in reason for reason in evidence["reasons"])
    assert evidence["requirements"]["total_samples"]["value"] == 40


def test_evidence_gate_rejects_synthetic_data(exp15):
    samples = exp15.build_synthetic_samples(variants_per_entry=2, seed=15)
    fit = fit_cost_policy(samples, cost_weight=0.1, min_samples=1)

    evidence = exp15.evaluate_evidence(samples, dataset_source="synthetic", fit=fit, min_samples=1)

    assert evidence["claims_gain"] is False
    assert evidence["evidence"] == "insufficient"
    assert any("synthetic" in reason for reason in evidence["reasons"])


def test_evidence_gate_passes_for_large_replay(exp15):
    samples = _replay_samples(per_task=250)
    fit = fit_cost_policy(samples, cost_weight=0.1, min_samples=30)

    evidence = exp15.evaluate_evidence(samples, dataset_source="replay", fit=fit, min_samples=30)

    assert evidence["claims_gain"] is True
    assert evidence["evidence"] == "replay"
    assert evidence["reasons"] == []
    assert evidence["requirements"]["per_task_min_samples"]["tasks"] == {
        "task_a": {"train": 125, "val": 125, "passed": True},
        "task_b": {"train": 125, "val": 125, "passed": True},
    }


def test_experiment_synthetic_never_claims_gain(exp15):
    samples = exp15.build_synthetic_samples(variants_per_entry=2, seed=15)

    result = exp15.run_experiment(
        samples,
        dataset_source="synthetic",
        seed=15,
        min_samples=1,
        bootstrap=0,
        require_cloud_tasks=("quality_verdict",),
    )
    report = result.report

    assert report["dataset_source"] == "synthetic"
    assert report["evidence"]["claims_gain"] is False
    assert report["evidence"]["evidence"] == "insufficient"
    assert report["selected"]["go_no_go"]["status"] == "pipeline_only"
    assert [entry["lambda"] for entry in report["lambda_sweep"]] == list(exp15.DEFAULT_LAMBDA_GRID)
    assert report["selected"]["lambda"] == pytest.approx(exp15.DEFAULT_SELECT_LAMBDA)
    assert result.policy["escalation_threshold_by_task"]
    assert report["selected"]["references"]["edge_only"]["escalation_rate"] == 0.0
    assert report["selected"]["references"]["cloud_only"]["escalation_rate"] == 1.0


def test_experiment_small_replay_is_evidence_insufficient(exp15):
    samples = _replay_samples(per_task=20)

    result = exp15.run_experiment(
        samples,
        dataset_source="replay",
        seed=15,
        min_samples=5,
        bootstrap=0,
        require_cloud_tasks=(),
    )
    report = result.report

    assert report["evidence"]["claims_gain"] is False
    assert report["selected"]["go_no_go"]["status"] == "evidence_insufficient"
    assert any("500" in reason for reason in report["evidence"]["reasons"])


def test_experiment_full_replay_can_claim_gain(exp15):
    samples = _replay_samples(per_task=250)

    result = exp15.run_experiment(
        samples,
        dataset_source="replay",
        seed=15,
        min_samples=30,
        bootstrap=0,
        require_cloud_tasks=(),
    )
    report = result.report

    assert report["evidence"]["claims_gain"] is True
    assert report["evidence"]["evidence"] == "replay"
    assert report["selected"]["go_no_go"]["status"] in {"go", "no_go"}
    assert report["counts"] == {
        "total": 500,
        "train": 250,
        "val": 250,
        "tasks": {
            "task_a": {"train": 125, "val": 125},
            "task_b": {"train": 125, "val": 125},
        },
    }


# ---------------------------------------------------------------------------
# Summary/curve tables and artifacts
# ---------------------------------------------------------------------------

SUMMARY_COLUMNS = (
    "lambda",
    "arm",
    "task",
    "n",
    "accuracy",
    "cost_per_request",
    "cost_norm",
    "objective",
    "escalation_rate",
    "cloud_calls",
    "cloud_calls_per_example",
    "latency_p95_ms",
    "threshold",
    "always_cloud",
    "accuracy_ci_low",
    "accuracy_ci_high",
    "cost_ci_low",
    "cost_ci_high",
    "status",
    "failed_checks",
)

CURVE_COLUMNS = (
    "lambda",
    "scope",
    "task",
    "threshold",
    "n",
    "accuracy",
    "cost_per_request",
    "cost_norm",
    "objective",
    "escalation_rate",
    "is_selected_threshold",
    "is_baseline_threshold",
)


def _synthetic_result(exp15, **overrides):
    samples = exp15.build_synthetic_samples(variants_per_entry=2, seed=15)
    kwargs = {
        "dataset_source": "synthetic",
        "seed": 15,
        "min_samples": 1,
        "bootstrap": 0,
        "selected_lambda": 0.1,
        "require_cloud_tasks": ("quality_verdict",),
        "smoke": True,
    }
    kwargs.update(overrides)
    return exp15.run_experiment(samples, **kwargs)


def test_summary_rows_cover_every_lambda_arm_and_per_task(exp15):
    result = _synthetic_result(exp15)
    rows = list(result.summary_rows)
    report = result.report

    assert {row["arm"] for row in rows} == {
        "per_task",
        "global_baseline",
        "edge_only",
        "cloud_only",
        "current_0.62",
    }
    per_task_lambdas = {
        row["lambda"] for row in rows if row["arm"] == "per_task" and row["task"] == ""
    }
    assert per_task_lambdas == set(report["config"]["lambda_grid"])

    selected = report["selected"]
    for task, entry in selected["val"]["per_task"].items():
        matching = [
            row
            for row in rows
            if row["arm"] == "per_task"
            and row["lambda"] == selected["lambda"]
            and row["task"] == task
        ]
        assert len(matching) == 1
        assert matching[0]["n"] == entry["n"]
        assert matching[0]["escalation_rate"] == pytest.approx(entry["escalation_rate"])
    assert len([row for row in rows if row["arm"] == "per_task"]) == (
        report["config"]["lambda_grid"].__len__() * (len(selected["val"]["per_task"]) + 1)
    )


def test_curve_rows_cover_global_and_every_fitted_task(exp15):
    result = _synthetic_result(exp15)
    rows = list(result.curve_rows)
    report = result.report
    grid = report["config"]["threshold_grid"]

    assert {row["scope"] for row in rows} == {"global", "task"}
    global_rows = [row for row in rows if row["scope"] == "global"]
    assert [row["threshold"] for row in global_rows] == grid
    assert any(row["is_selected_threshold"] for row in global_rows)
    assert any(row["is_baseline_threshold"] for row in global_rows)
    assert all(row["n"] == report["counts"]["val"] for row in global_rows)

    task_rows = [row for row in rows if row["scope"] == "task"]
    tasks = {row["task"] for row in task_rows}
    assert tasks == {
        task
        for task in report["selected"]["val"]["per_task"]
        if task not in report["config"]["require_cloud_tasks"]
    }
    for task in tasks:
        subset = [row for row in task_rows if row["task"] == task]
        assert [row["threshold"] for row in subset] == grid
        assert sum(row["is_selected_threshold"] for row in subset) == 1


def test_write_artifacts_writes_every_required_file(tmp_path, exp15):
    result = _synthetic_result(exp15)
    out = tmp_path / "exp15"

    paths = exp15.write_artifacts(result, out)

    assert set(paths) == {"policy", "report", "summary", "curves", "dataset"}
    for name in ("policy.json", "report.json", "summary.csv", "curves.csv", "dataset.jsonl"):
        assert (out / name).exists(), name

    policy = json.loads((out / "policy.json").read_text(encoding="utf-8"))
    config = EdgeConfig.from_dict(policy)
    assert config.escalation_threshold == pytest.approx(
        result.report["selected"]["fit"]["global_threshold"]
    )
    assert config.escalation_threshold_by_task == result.policy["escalation_threshold_by_task"]

    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert report == result.report
    assert report["schema_version"] == exp15.SCHEMA_VERSION
    assert report["dataset_source"] == "synthetic"
    assert report["evidence"]["claims_gain"] is False

    dataset_rows = [
        json.loads(line)
        for line in (out / "dataset.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(dataset_rows) == report["counts"]["total"]
    assert "split" in dataset_rows[0]
    assert "cloud_correct" in dataset_rows[0]

    with (out / "summary.csv").open(encoding="utf-8", newline="") as handle:
        summary = list(csv.DictReader(handle))
    assert tuple(summary[0]) == SUMMARY_COLUMNS
    assert {row["arm"] for row in summary} == {
        "per_task",
        "global_baseline",
        "edge_only",
        "cloud_only",
        "current_0.62",
    }

    with (out / "curves.csv").open(encoding="utf-8", newline="") as handle:
        curves = list(csv.DictReader(handle))
    assert tuple(curves[0]) == CURVE_COLUMNS
    assert {row["scope"] for row in curves} == {"global", "task"}


def test_cli_synthetic_smoke_is_deterministic(tmp_path):
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    outputs = []
    for name in ("first", "second"):
        out = tmp_path / name
        completed = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--synthetic",
                "--smoke",
                "--out",
                str(out),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=120,
            env=env,
        )
        assert completed.returncode == 0, completed.stderr
        assert "artifacts:" in completed.stdout
        outputs.append(out)

    for name in ("policy.json", "report.json", "summary.csv", "curves.csv", "dataset.jsonl"):
        first = (outputs[0] / name).read_bytes()
        second = (outputs[1] / name).read_bytes()
        assert first == second, name

    report = json.loads((outputs[0] / "report.json").read_text(encoding="utf-8"))
    assert report["smoke"] is True
    assert report["config"]["min_samples"] == 1
    assert report["config"]["bootstrap"] == 0
    assert report["counts"]["total"] == len(exp15_knowledge_entries()) * 2
    assert report["evidence"]["claims_gain"] is False
    assert report["selected"]["go_no_go"]["status"] == "pipeline_only"


def exp15_knowledge_entries():
    from edge_cloud import knowledge

    return knowledge.ENTRIES


def test_cli_replay_fixture_reports_insufficient_evidence(tmp_path):
    samples = _replay_samples(per_task=10)
    dataset = tmp_path / "replay.jsonl"
    dataset.write_text(
        "\n".join(json.dumps(sample.to_dict(), ensure_ascii=False) for sample in samples) + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "exp15"
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--dataset",
            str(dataset),
            "--min-samples",
            "2",
            "--bootstrap",
            "0",
            "--out",
            str(out),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )

    assert completed.returncode == 0, completed.stderr
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert report["dataset_source"] == "replay"
    assert report["evidence"]["claims_gain"] is False
    assert report["evidence"]["evidence"] == "insufficient"
    assert report["selected"]["go_no_go"]["status"] == "evidence_insufficient"
    policy = json.loads((out / "policy.json").read_text(encoding="utf-8"))
    assert EdgeConfig.from_dict(policy).escalation_threshold >= 0.0


def test_cli_requires_a_source_and_help_lists_flags():
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    )
    assert completed.returncode == 0, completed.stderr
    for flag in (
        "--synthetic",
        "--dataset",
        "--lambdas",
        "--select-lambda",
        "--min-samples",
        "--smoke",
        "--out",
    ):
        assert flag in completed.stdout
