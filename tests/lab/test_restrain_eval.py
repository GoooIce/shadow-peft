"""RESTRAIN-lite evaluation and report seams (issue #34).

Spec: ``docs/specs/restrain-lite-structured-flywheel.md`` §Implementation
Decisions 5 and Testing Decisions「评估缝」:

- verdict answers match after ``extract_answer`` normalization; numeric
  answers match within a relative tolerance;
- entropy is 0 for a single answer and maximal for a uniform distribution;
  top-1 share is 1 for a single answer; empty counts never produce NaN;
- bootstrap intervals are wider on small samples instead of pretending
  significance;
- the report carries the spec fields, ``claims_gain`` requires +1pp accuracy
  and entropy >= 80% of the best-answer baseline, and no baseline means no
  claim;
- the training helper runs 1–2 steps on a tiny random Llama with a fixed seed;
- the CLI parser and the offline prediction path work without loading a model.

Synthetic records and a tiny random Llama, CPU, fixed seeds.
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import pytest

from shadow_lab.restrain_eval import (
    REPORT_FIELDS,
    accuracy,
    answer_counts,
    answer_entropy,
    bootstrap_ci,
    build_report,
    grade_structured,
    low_consistency_rate,
    skip_rates,
    top1_share,
)

ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# grade_structured / accuracy
# ---------------------------------------------------------------------------


def test_verdict_grading_normalizes_synonyms_before_comparing():
    assert grade_structured("结论：合格", "放行", "quality_verdict") is True
    assert grade_structured("判断为不合格", "拦截", "quality_verdict") is True
    assert grade_structured("判定：返修", "放行", "quality_verdict") is False


def test_verdict_grading_accepts_the_canonical_export_labels():
    # Self-train targets are "pass"/"fail"/"rework", so generated answers and
    # held-out expected values may use the canonical spelling.
    assert grade_structured("pass", "放行", "quality_verdict") is True
    assert grade_structured("结论：PASS", "pass", "quality_verdict") is True
    assert grade_structured("fail", "pass", "quality_verdict") is False
    assert grade_structured("maybe", "pass", "quality_verdict") is False


def test_answer_counts_accepts_chinese_and_canonical_verdicts():
    assert answer_counts(["放行", "pass", "合格", "拦截"], "quality_verdict") == {
        "pass": 3,
        "fail": 1,
    }
    assert answer_counts(["245℃", "245 摄氏度", "250℃"], "process_calc") == {
        "245℃": 2,
        "250℃": 1,
    }


def test_number_grading_uses_a_relative_tolerance():
    assert grade_structured("最终温度 245℃。", "245 摄氏度", "process_calc") is True
    assert grade_structured("最终温度 245℃。", "245.5℃", "process_calc") is True
    assert grade_structured("最终温度 245℃。", "250℃", "process_calc") is False
    # The unit is part of the structured answer.
    assert grade_structured("长度 245mm", "245℃", "process_calc") is False


def test_unparseable_answers_fail_and_ungradable_tasks_return_none():
    assert grade_structured("我不知道", "放行", "quality_verdict") is False
    assert grade_structured("", "放行", "quality_verdict") is False
    assert grade_structured("放行", None, "quality_verdict") is None
    assert grade_structured("放行", "放行", "general") is None
    assert grade_structured(None, "放行", "quality_verdict") is False


def test_accuracy_averages_only_gradable_records():
    records = [
        {"task": "quality_verdict", "answer": "合格", "expected": "放行"},
        {"task": "process_calc", "answer": "245℃", "expected": "245℃"},
        {"task": "process_calc", "answer": "250℃", "expected": "245℃"},
        {"task": "general", "answer": "随便", "expected": "随便"},
    ]

    assert accuracy(records) == pytest.approx(2 / 3)
    assert accuracy([]) is None
    assert accuracy([{"task": "general", "answer": "x", "expected": "y"}]) is None


def test_accuracy_reads_the_documented_prediction_keys():
    records = [
        {"task": "quality_verdict", "answer": "拦截", "expected": "放行"},
        {"task": "quality_verdict", "response": "放行", "expected": "放行"},
    ]
    assert accuracy(records) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# entropy / top-1 share
# ---------------------------------------------------------------------------


def test_single_answer_has_zero_entropy_and_full_top1_share():
    counts = {"pass": 8}
    assert answer_entropy(counts) == 0.0
    assert top1_share(counts) == 1.0


def test_uniform_distribution_reaches_the_maximum_entropy():
    counts = {"a": 2, "b": 2, "c": 2, "d": 2}
    assert answer_entropy(counts) == pytest.approx(2.0)  # log2(4)
    assert top1_share(counts) == 0.25
    assert answer_entropy([1, 1, 1]) == pytest.approx(math.log2(3))


def test_empty_counts_never_produce_nan():
    assert answer_entropy({}) is None
    assert answer_entropy([]) is None
    assert answer_entropy({"a": 0}) is None
    assert top1_share({}) == 0.0
    assert top1_share([]) == 0.0
    assert not math.isnan(top1_share({}))


# ---------------------------------------------------------------------------
# low consistency / skip rates / bootstrap
# ---------------------------------------------------------------------------


def test_low_consistency_rate_uses_the_kappa_gate():
    rows = [
        {"counts": [5, 3]},  # max 5 >= 3
        {"counts": [2, 1, 1]},  # max 2 < 3
        {"counts": [3, 3]},  # max 3 >= 3
        {"counts": []},  # no votes: skipped by the exporter
    ]
    assert low_consistency_rate(rows) == pytest.approx(2 / 4)
    assert low_consistency_rate(rows, kappa=5) == pytest.approx(3 / 4)
    assert low_consistency_rate([]) == 0.0


def test_skip_rates_use_the_export_report_counts():
    report = {
        "exported": 6,
        "skipped_empty": 1,
        "skipped_unsupported": 2,
        "skipped_unclusterable": 1,
        "rollout_failures": 4,
        "n": 4,
    }

    rates = skip_rates(report)

    assert rates["skip_rate"] == pytest.approx(4 / 10)
    assert rates["unclusterable_rate"] == pytest.approx(1 / 10)
    assert rates["unsupported_rate"] == pytest.approx(2 / 10)
    assert rates["empty_rate"] == pytest.approx(1 / 10)
    # n rollouts for every prompt that was actually rolled out.
    assert rates["rollout_failure_rate"] == pytest.approx(4 / (4 * 7))
    assert skip_rates({})["skip_rate"] == 0.0


def test_bootstrap_interval_is_wider_on_small_samples():
    small = [0.0, 1.0]
    large = [0.0, 1.0] * 50

    small_lo, small_hi = bootstrap_ci(small, n_boot=500, seed=0)
    large_lo, large_hi = bootstrap_ci(large, n_boot=500, seed=0)

    assert small_hi - small_lo > large_hi - large_lo
    assert large_lo <= 0.5 <= large_hi


def test_bootstrap_is_deterministic_and_validates_inputs():
    assert bootstrap_ci([0.0, 1.0, 1.0], n_boot=100, seed=7) == bootstrap_ci(
        [0.0, 1.0, 1.0], n_boot=100, seed=7
    )
    with pytest.raises(ValueError, match="at least one"):
        bootstrap_ci([])
    with pytest.raises(ValueError, match="n_boot"):
        bootstrap_ci([0.5], n_boot=0)


def test_bootstrap_accepts_an_entropy_statistic_on_counts():
    def mean_entropy(sample):
        return sum(answer_entropy(counts) for counts in sample) / len(sample)

    lo, hi = bootstrap_ci(
        [{"pass": 8}, {"pass": 4, "fail": 4}],
        n_boot=100,
        seed=2,
        statistic=mean_entropy,
    )
    assert lo == pytest.approx(0.0)
    assert hi == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# build_report
# ---------------------------------------------------------------------------


def test_build_report_has_every_spec_field_and_the_config_block():
    export_report = {
        "exported": 6,
        "skipped_empty": 1,
        "skipped_unsupported": 2,
        "skipped_unclusterable": 1,
        "rollout_failures": 4,
        "n": 4,
    }
    report = build_report(
        accuracy=2 / 3,
        outcomes=[True, True, False],
        answer_entropy=answer_entropy({"pass": 3, "fail": 1}) / 2,
        top1_share=(0.75 + 1.0) / 2,
        low_consistency_rate=0.5,
        export_report=export_report,
        baseline={"accuracy": 0.5, "answer_entropy": 0.4},
        escalation_rate=0.25,
        cloud_calls=3,
        sigma=0.4,
        kappa=2,
        n=8,
        delta=0.01,
        n_boot=100,
        ci_seed=1,
    )

    assert set(REPORT_FIELDS) <= set(report)
    assert report["accuracy"] == pytest.approx(2 / 3)
    assert report["accuracy_ci"] is not None
    assert report["low_consistency_rate"] == pytest.approx(0.5)
    assert report["skip_rate"] == pytest.approx(0.4)
    assert report["rollout_failure_rate"] == pytest.approx(4 / (4 * 7))
    assert report["escalation_rate"] == 0.25
    assert report["cloud_calls"] == 3
    assert report["baseline_best_answer"] == 0.5
    assert report["config"] == {"sigma": 0.4, "kappa": 2, "n": 8, "delta": 0.01}
    assert report["claims_gain"] is True


def test_claims_gain_requires_one_point_and_eighty_percent_entropy():
    baseline = {"accuracy": 0.60, "answer_entropy": 1.0}

    def claim(accuracy_value: float, entropy_value: float) -> bool:
        return build_report(
            accuracy=accuracy_value,
            answer_entropy=entropy_value,
            baseline=baseline,
        )["claims_gain"]

    assert claim(0.61, 0.80) is True  # exactly +1pp and exactly 80%
    assert claim(0.609, 0.90) is False  # less than +1pp
    assert claim(0.62, 0.79) is False  # entropy below 80%


def test_no_baseline_means_no_gain_claim():
    report = build_report(accuracy=1.0, answer_entropy=0.0, outcomes=[True])
    assert report["baseline_best_answer"] is None
    assert report["claims_gain"] is False
    assert report["accuracy_ci"] is None  # one graded record, no invented interval


def test_build_report_rejects_out_of_range_rates():
    with pytest.raises(ValueError, match="skip_rate"):
        build_report(skip_rate=1.5)
    with pytest.raises(ValueError, match="cloud_calls"):
        build_report(cloud_calls=-1)


# ---------------------------------------------------------------------------
# Training helper (tiny model)
# ---------------------------------------------------------------------------

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("trl")  # train_weighted_sft reuses shadow_lab.trainers' loss

from transformers import LlamaConfig, LlamaForCausalLM  # noqa: E402

from shadow_lab.restrain_eval import train_weighted_sft  # noqa: E402

VOCAB = 64


class _StubTokenizer:
    """One id per character; deterministic and pad-free."""

    pad_token_id = 0
    eos_token_id = 1
    chat_template = None

    def __call__(self, text: str, add_special_tokens: bool = True):
        del add_special_tokens
        return {"input_ids": [3 + (ord(char) % (VOCAB - 4)) for char in text]}


def _tiny_llama(seed: int = 0) -> LlamaForCausalLM:
    torch.manual_seed(seed)
    return LlamaForCausalLM(
        LlamaConfig(
            vocab_size=VOCAB,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=64,
        )
    )


def _train_rows() -> list[dict[str, Any]]:
    return [
        {"prompt": "这批零件能放行吗", "answer": "pass", "task": "quality_verdict", "weight": 1.0},
        {"prompt": "最终温度是多少", "answer": "245℃", "task": "process_calc", "weight": 0.5},
        {"prompt": "长度是多少", "answer": "12mm", "task": "process_calc", "weight": 0.0},
    ]


def test_train_weighted_sft_runs_two_steps_and_filters_zero_weights():
    model = _tiny_llama(seed=5)
    result = train_weighted_sft(
        model,
        _StubTokenizer(),
        _train_rows(),
        max_steps=2,
        batch_size=2,
        seed=3,
        learning_rate=1e-3,
    )

    assert result["steps"] == 2
    assert result["rows"] == 2  # the zero-weight row never enters the loss
    assert result["tokens_seen"] > 0
    assert math.isfinite(result["loss"])
    assert all(torch.isfinite(p).all() for p in model.parameters())


def test_train_weighted_sft_is_deterministic_for_a_fixed_seed():
    first = train_weighted_sft(
        _tiny_llama(seed=9),
        _StubTokenizer(),
        _train_rows(),
        max_steps=2,
        batch_size=1,
        seed=4,
        learning_rate=1e-3,
    )
    second = train_weighted_sft(
        _tiny_llama(seed=9),
        _StubTokenizer(),
        _train_rows(),
        max_steps=2,
        batch_size=1,
        seed=4,
        learning_rate=1e-3,
    )

    assert first["loss"] == pytest.approx(second["loss"], abs=1e-6)
    assert first["tokens_seen"] == second["tokens_seen"]


def test_train_weighted_sft_respects_the_token_budget():
    result = train_weighted_sft(
        _tiny_llama(seed=2),
        _StubTokenizer(),
        _train_rows(),
        max_steps=1,
        token_budget=12,
        seed=0,
    )
    assert result["tokens_seen"] <= 12 or result["token_budget_overflow"] is True


# ---------------------------------------------------------------------------
# CLI parser and offline path (no model loading)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cli_module():
    path = ROOT / "scripts" / "exp12_restrain.py"
    spec = importlib.util.spec_from_file_location("exp12_restrain", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_help_is_available_for_every_subcommand(cli_module):
    for argv in ([], ["export"], ["train"], ["eval"], ["run"]):
        with pytest.raises(SystemExit) as excinfo:
            cli_module.build_parser().parse_args([*argv, "--help"])
        assert excinfo.value.code == 0


def test_cli_parses_export_and_train_arguments(cli_module):
    export_args = cli_module.build_parser().parse_args(
        [
            "export",
            "--feedback-store",
            "feedback.jsonl",
            "--model",
            "base-model",
            "--n",
            "4",
            "--tasks",
            "quality_verdict,process_calc",
            "--max-new-tokens",
            "32",
            "--out",
            "self_train.jsonl",
        ]
    )
    assert export_args.command == "export"
    assert export_args.n == 4
    assert export_args.tasks == ["quality_verdict", "process_calc"]
    assert export_args.max_new_tokens == 32

    train_args = cli_module.build_parser().parse_args(
        [
            "train",
            "--self-train",
            "self_train.jsonl",
            "--sft",
            "best_answer.jsonl",
            "--model",
            "base-model",
            "--weights",
            "--max-steps",
            "2",
            "--out",
            "runs/exp12/train",
        ]
    )
    assert train_args.weights is True
    assert train_args.max_steps == 2


def test_cli_eval_scores_predictions_offline(tmp_path, cli_module):
    predictions = tmp_path / "predictions.jsonl"
    rows = [
        {"prompt": "p1", "task": "quality_verdict", "answer": "放行", "expected": "放行"},
        {"prompt": "p1", "task": "quality_verdict", "answer": "合格", "expected": "放行"},
        {"prompt": "p2", "task": "process_calc", "answer": "245℃", "expected": "245℃"},
        {"prompt": "p3", "task": "process_calc", "answer": "250℃", "expected": "245℃"},
    ]
    predictions.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    report_path = tmp_path / "report.json"

    code = cli_module.main(
        ["eval", "--predictions", str(predictions), "--out", str(report_path)]
    )

    assert code == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert set(REPORT_FIELDS) <= set(report)
    assert report["accuracy"] == pytest.approx(3 / 4)
    assert report["answer_entropy"] == pytest.approx(
        (answer_entropy({"pass": 2}) + 0.0 + 0.0) / 3
    )
    assert report["top1_share"] == 1.0
    assert report["baseline_best_answer"] is None


def test_cli_export_from_jsonl_passes_rows_through(tmp_path, cli_module):
    source = tmp_path / "self_train.jsonl"
    rows = [
        {
            "prompt": "p1",
            "task": "quality_verdict",
            "answers": ["pass"],
            "counts": [4],
            "reference_counts": [],
            "weights": [1.0],
            "prompt_weight": 1,
            "consistency": 0.75,
            "cluster_method": "verdict_label",
            "adapter_version": "v1",
        }
    ]
    source.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    out = tmp_path / "exported.jsonl"

    code = cli_module.main(["export", "--from-jsonl", str(source), "--out", str(out)])

    assert code == 0
    assert [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()] == rows
