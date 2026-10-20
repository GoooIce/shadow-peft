"""exp13 memory-reranking evaluation — external-behavior tests (issue #40).

Spec: ``docs/specs/jitrl-memory-hotfix.md`` §Implementation Decisions 6 and
Further Notes「建议的首轮验收口径」.

The tests drive the evaluation script's public functions only: the synthetic
dataset, scoring, the memory/no-memory arms through ``Router.infer``, the
report and the CLI.  No private helper is called and no real model is loaded.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "exp13_jitrl_memory.py"


@pytest.fixture(scope="module")
def exp13():
    spec = importlib.util.spec_from_file_location("exp13_jitrl_memory", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # ``from __future__ import annotations`` makes dataclasses resolve their
    # field types through ``sys.modules``, so the module must be registered
    # before executing it.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def test_dataset_is_deterministic_for_a_fixed_seed(exp13):
    first = exp13.build_cases(seed=7)
    second = exp13.build_cases(seed=7)

    assert first == second
    assert exp13.cases_fingerprint(first) == exp13.cases_fingerprint(second)


def test_dataset_changes_with_the_seed(exp13):
    assert exp13.cases_fingerprint(exp13.build_cases(seed=7)) != exp13.cases_fingerprint(
        exp13.build_cases(seed=8)
    )


def test_each_knowledge_entry_has_six_paraphrases_and_half_have_memory(exp13):
    cases = exp13.build_cases(seed=13)
    per_entry = Counter(case.entry_key for case in cases)

    assert set(per_entry) == {entry.key for entry in exp13.knowledge.ENTRIES}
    assert set(per_entry.values()) == {6}
    assert len(cases) % 2 == 0
    memory_cases = [case for case in cases if case.has_memory]
    assert len(memory_cases) * 2 == len(cases)
    assert {case.tenant for case in memory_cases} == {exp13.MEMORY_TENANT}
    assert {case.tenant for case in cases if not case.has_memory} == {exp13.NO_MEMORY_TENANT}
    assert len({case.prompt for case in cases}) == len(cases)


def test_paraphrase_count_is_configurable_within_the_spec_range(exp13):
    cases = exp13.build_cases(seed=13, paraphrases_per_entry=4)

    assert set(Counter(case.entry_key for case in cases).values()) == {4}
    with pytest.raises(ValueError):
        exp13.build_cases(seed=13, paraphrases_per_entry=9)


def test_dataset_gold_scoring_accepts_cloud_and_rejects_wrong_answers(exp13):
    cases = exp13.build_cases(seed=13)

    for case in cases:
        assert exp13.score_answer(case, case.cloud_answer) is True
        assert exp13.score_answer(case, case.wrong_answer) is False
        assert exp13.score_answer(case, case.edge_answer) is (
            case.entry_key in exp13.EASY_ENTRY_KEYS
        )


# ---------------------------------------------------------------------------
# Retrieval and the two arms
# ---------------------------------------------------------------------------


def test_memory_retrieval_hits_only_the_memory_tenant(exp13):
    cases = exp13.build_cases(seed=13)
    memory = exp13.build_memory(cases)

    for case in cases:
        hits = memory.retrieve(
            case.prompt,
            task=case.task,
            tenant=case.tenant,
            adapter_version=exp13.ADAPTER_VERSION,
            k=exp13.DEFAULT_TOP_K,
            threshold=exp13.DEFAULT_THRESHOLD,
        )
        if case.has_memory:
            assert hits, f"memory case missed: {case.case_id}"
            assert sum(
                1 for hit in hits if case.cloud_answer in hit.entry.candidates
            ) >= 2
        else:
            assert hits == [], f"line-b unexpectedly hit: {case.case_id}"


def test_rerank_arm_improves_accuracy_without_more_cloud_calls(exp13):
    report = exp13.run_evaluation(seed=13)
    no_memory = report["arms"]["no_memory"]
    rerank = report["arms"]["rerank"]
    comparison = report["comparison"]

    assert no_memory["accuracy"] == pytest.approx(0.5)
    assert rerank["accuracy"] > no_memory["accuracy"]
    assert comparison["accuracy_gain_pp"] >= 3.0
    assert no_memory["cloud_calls"] > 0
    assert rerank["cloud_calls"] <= no_memory["cloud_calls"]
    assert comparison["memory_hit_rate"] == pytest.approx(0.5)
    assert rerank["hit_correct"] + rerank["hit_wrong"] + rerank["missed"] == rerank["total"]
    assert rerank["hit_wrong"] == 0
    assert rerank["missed"] == rerank["total"] // 2
    assert comparison["false_bias_count"] == 0
    assert comparison["false_bias_rate"] == 0.0


def test_no_memory_arm_keeps_the_single_answer_baseline(exp13):
    report = exp13.run_evaluation(seed=13)
    no_memory = report["arms"]["no_memory"]

    assert no_memory["memory_hit_rate"] == 0.0
    assert no_memory["hit_correct"] == 0
    assert no_memory["hit_wrong"] == 0
    assert no_memory["missed"] == no_memory["total"]
    assert no_memory["latency_ms"] is None
    assert all(case["memory_hit"] is False for case in no_memory["cases"])


def test_rerank_arm_reports_the_three_hit_categories_and_latency(exp13):
    report = exp13.run_evaluation(seed=13)
    rerank = report["arms"]["rerank"]
    latency = rerank["latency_ms"]

    assert latency is not None
    assert latency["count"] > 0
    assert 0.0 <= latency["p50"] <= latency["p95"] <= latency["max"]
    # Every memory case is a retrieval hit; every line-b case is a miss.
    assert rerank["hit_correct"] + rerank["hit_wrong"] == rerank["total"] // 2
    assert rerank["cases"]
    assert {case["memory_hit"] for case in rerank["cases"]} == {True, False}
    assert all(
        case["memory_hit"] is case["has_memory"] for case in rerank["cases"]
    )


# ---------------------------------------------------------------------------
# False-bias guard, latency statistics, report and acceptance
# ---------------------------------------------------------------------------


def test_guard_probes_block_insufficient_or_negative_memory(exp13):
    probes = exp13.run_guard_probes()

    assert probes["total"] == 4
    assert probes["blocked"] == probes["total"]
    assert probes["misleading_total"] == 3
    assert probes["misleading_blocked"] == probes["misleading_total"]
    reasons = {detail["name"]: detail["reason"] for detail in probes["details"]}
    assert reasons == {
        "insufficient_support": "insufficient_support",
        "negative_returns": "low_return",
        "top1_supported": "top1_unchanged",
        "unsupported_answer": "no_support",
    }
    assert all(detail["blocked"] for detail in probes["details"])


def test_latency_summary_reads_the_router_metric_snapshot(exp13):
    snapshot = {
        "latency_ms": {
            "memory_latency_ms": {
                "count": 4.0,
                "avg": 2.25,
                "p50": 1.5,
                "p95": 4.0,
                "max": 5.0,
            }
        }
    }

    assert exp13.latency_summary(snapshot) == {
        "count": 4.0,
        "avg": 2.25,
        "p50": 1.5,
        "p95": 4.0,
        "max": 5.0,
    }
    assert exp13.latency_summary({"latency_ms": {}}) is None


def test_report_exposes_the_required_fields_and_guard_probes(exp13):
    report = exp13.run_evaluation(seed=13)

    for key in (
        "experiment",
        "seed",
        "dataset",
        "config",
        "arms",
        "comparison",
        "acceptance",
        "guard_probes",
        "environment",
    ):
        assert key in report
    assert report["dataset"]["case_count"] == 48
    assert report["dataset"]["fingerprint"].startswith("sha256:")
    assert report["dataset"]["memory_fingerprint"].startswith("sha256:")
    assert report["dataset"]["memory_entries"] == 48 * 3 // 2
    assert report["guard_probes"] == exp13.run_guard_probes()
    for arm in ("no_memory", "rerank"):
        for key in (
            "total",
            "correct",
            "accuracy",
            "cloud_calls",
            "hit_correct",
            "hit_wrong",
            "missed",
            "memory_hit_rate",
            "latency_ms",
            "cases",
        ):
            assert key in report["arms"][arm]
    assert report["acceptance"]["verdict"] in {"go", "no-go"}


def test_report_metrics_are_deterministic(exp13):
    first = exp13.run_evaluation(seed=13)
    second = exp13.run_evaluation(seed=13)

    assert first["dataset"] == second["dataset"]
    for arm in ("no_memory", "rerank"):
        for key in (
            "total",
            "correct",
            "accuracy",
            "cloud_calls",
            "hit_correct",
            "hit_wrong",
            "missed",
            "reranked_cases",
            "cases",
        ):
            assert first["arms"][arm][key] == second["arms"][arm][key]
    for key in (
        "accuracy_gain_pp",
        "false_bias_count",
        "false_bias_rate",
        "memory_hit_rate",
        "cloud_calls_no_memory",
        "cloud_calls_rerank",
    ):
        assert first["comparison"][key] == second["comparison"][key]
    assert first["acceptance"] == second["acceptance"]


def _passing_comparison() -> dict[str, object]:
    return {
        "accuracy_gain_pp": 3.0,
        "false_bias_rate": 0.0,
        "latency_p95_ms": 10.0,
        "cloud_calls_delta": 0,
    }


def test_acceptance_passes_at_the_thresholds(exp13):
    acceptance = exp13.evaluate_acceptance(_passing_comparison())

    assert acceptance["checks"] == {
        "accuracy_gain_pp": True,
        "false_bias_rate": True,
        "latency_p95_ms": True,
        "cloud_calls_not_up": True,
    }
    assert acceptance["failed_checks"] == []
    assert acceptance["verdict"] == "go"


@pytest.mark.parametrize(
    ("overrides", "failed"),
    [
        ({"accuracy_gain_pp": 2.99}, "accuracy_gain_pp"),
        ({"false_bias_rate": 0.02}, "false_bias_rate"),
        ({"latency_p95_ms": 50.0}, "latency_p95_ms"),
        ({"latency_p95_ms": None}, "latency_p95_ms"),
        ({"cloud_calls_delta": 1}, "cloud_calls_not_up"),
    ],
)
def test_acceptance_marks_no_go_when_a_gate_fails(exp13, overrides, failed):
    comparison = {**_passing_comparison(), **overrides}

    acceptance = exp13.evaluate_acceptance(comparison)

    assert acceptance["failed_checks"] == [failed]
    assert acceptance["verdict"] == "no-go"


# ---------------------------------------------------------------------------
# CLI and artifacts
# ---------------------------------------------------------------------------


def test_cli_help_is_available(exp13):
    with pytest.raises(SystemExit) as excinfo:
        exp13.build_parser().parse_args(["--help"])
    assert excinfo.value.code == 0
    with pytest.raises(SystemExit) as excinfo:
        exp13.build_parser().parse_args(["run", "--help"])
    assert excinfo.value.code == 0


def test_cli_run_parses_its_arguments(exp13):
    args = exp13.build_parser().parse_args(
        [
            "run",
            "--out",
            "runs-local/exp13",
            "--seed",
            "7",
            "--paraphrases",
            "4",
            "--alpha",
            "0.25",
            "--threshold",
            "0.5",
            "--top-k",
            "8",
            "--min-support",
            "3",
            "--latency-budget-ms",
            "25",
        ]
    )

    assert args.command == "run"
    assert args.out == "runs-local/exp13"
    assert args.seed == 7
    assert args.paraphrases == 4
    assert args.alpha == pytest.approx(0.25)
    assert args.threshold == pytest.approx(0.5)
    assert args.top_k == 8
    assert args.min_support == 3
    assert args.latency_budget_ms == pytest.approx(25.0)
    assert args.no_write is False


def test_cli_help_runs_as_a_standalone_script():
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    for argv in (["--help"], ["run", "--help"]):
        completed = subprocess.run(
            [sys.executable, str(SCRIPT), *argv],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=120,
            env=env,
        )
        assert completed.returncode == 0, completed.stderr
    for flag in (
        "--out",
        "--seed",
        "--paraphrases",
        "--alpha",
        "--threshold",
        "--top-k",
        "--min-support",
        "--latency-budget-ms",
        "--no-write",
    ):
        assert flag in completed.stdout


def test_cli_run_writes_report_dataset_memory_and_outcomes(tmp_path, exp13):
    out = tmp_path / "exp13"

    assert exp13.main(["run", "--out", str(out), "--seed", "13"]) == 0

    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    for name in (
        "report.json",
        "dataset.jsonl",
        "memory.jsonl",
        "outcomes_no_memory.jsonl",
        "outcomes_rerank.jsonl",
        "summary.md",
    ):
        assert (out / name).exists(), name

    dataset = [
        json.loads(line)
        for line in (out / "dataset.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    memory = [
        json.loads(line)
        for line in (out / "memory.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    outcomes = [
        json.loads(line)
        for line in (out / "outcomes_rerank.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(dataset) == report["dataset"]["case_count"]
    assert len(memory) == report["dataset"]["memory_entries"]
    assert outcomes == report["arms"]["rerank"]["cases"]
    summary = (out / "summary.md").read_text(encoding="utf-8")
    assert report["dataset"]["fingerprint"] in summary
    assert report["acceptance"]["verdict"] in summary
    for arm in ("no_memory", "rerank"):
        assert str(report["arms"][arm]["accuracy_pct"]) in summary


