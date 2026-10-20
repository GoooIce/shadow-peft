"""Evaluation/report seams for the on-policy GKD flywheel (issue #21).

Only public behavior is asserted: paired joint/shadow perplexity under a tiny
random Llama, structured-task grading, bootstrap intervals, the batch contract
handed to ``OnPolicyGKDTrainer`` (``source`` branch labels), report fields and
the CLI parser. CPU, fixed seeds, no real checkpoints.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from transformers import LlamaConfig, LlamaForCausalLM  # noqa: E402

from shadow_lab.onpolicy import bucket_rows_by_length  # noqa: E402
from shadow_lab.onpolicy_eval import (  # noqa: E402
    REPORT_FIELDS,
    OnPolicyTrainDataset,
    bootstrap_ci,
    build_report,
    collate_training_rows,
    count_training_tokens,
    data_fingerprint,
    evaluate_ppl,
    grade_task,
    task_accuracy,
    task_outcomes,
    truncate_training_rows_by_token_budget,
)
from shadow_peft import ShadowConfig, ShadowForCausalLM, get_shadow_model  # noqa: E402

VOCAB = 64
ROOT = Path(__file__).resolve().parents[2]


class _StubTokenizer:
    """Minimal tokenizer: one id per character, deterministic and pad-free."""

    pad_token_id = 0
    eos_token_id = None
    chat_template = None

    def __call__(self, text, add_special_tokens=True):  # noqa: ARG002
        return {"input_ids": [3 + (ord(char) % (VOCAB - 4)) for char in text]}


def _tiny_shadow(seed: int = 0) -> ShadowForCausalLM:
    torch.manual_seed(seed)
    base = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=VOCAB,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=256,
        )
    )
    peft = get_shadow_model(
        base,
        ShadowConfig(
            num_shadow_layers=1,
            injection_hidden_size=8,
            gate_hidden_size=10,
            alpha=0.1,
            dropout=0.0,
        ),
    )
    return ShadowForCausalLM(peft, shadow_loss_weight=0.05, inference_mode="base_shadow")


# ---------------------------------------------------------------------------
# evaluate_ppl
# ---------------------------------------------------------------------------


def test_evaluate_ppl_reports_joint_and_shadow_and_restores_mode():
    model = _tiny_shadow()
    tokenizer = _StubTokenizer()
    model.train()
    model.set_inference_mode("shadow_only")

    report = evaluate_ppl(
        model,
        tokenizer,
        ["the quick brown fox jumps over the lazy dog " * 2],
        seq_len=16,
        max_windows=2,
    )

    assert set(report) == {
        "joint_ppl",
        "shadow_ppl",
        "tokens",
        "joint_losses",
        "shadow_losses",
        "window_tokens",
    }
    for key in ("joint_ppl", "shadow_ppl"):
        assert math.isfinite(report[key])
        assert report[key] > 0.0
    assert report["tokens"] > 0
    # The caller's inference mode and train/eval flag survive the evaluation.
    assert model.inference_mode == "shadow_only"
    assert model.training is True


def test_evaluate_ppl_rejects_streams_too_short_to_score():
    model = _tiny_shadow()
    with pytest.raises(ValueError, match="too short"):
        evaluate_ppl(model, _StubTokenizer(), ["x"], seq_len=16, max_windows=2)


def test_evaluate_ppl_returns_per_window_losses_for_bootstrap():
    model = _tiny_shadow()
    tokenizer = _StubTokenizer()

    report = evaluate_ppl(
        model,
        tokenizer,
        ["the quick brown fox jumps over the lazy dog " * 2],
        seq_len=8,
        max_windows=3,
    )

    losses = report["joint_losses"]
    tokens = report["window_tokens"]
    assert len(losses) == len(report["shadow_losses"]) == len(tokens) == 3
    assert all(math.isfinite(loss) and loss > 0.0 for loss in losses)
    assert all(isinstance(count, int) and count > 0 for count in tokens)
    point = math.exp(
        sum(loss * count for loss, count in zip(losses, tokens, strict=True))
        / sum(tokens)
    )
    assert point == pytest.approx(report["joint_ppl"], rel=1e-12)


# ---------------------------------------------------------------------------
# Structured grading
# ---------------------------------------------------------------------------


def test_grade_quality_verdict_uses_the_leading_verdict_term():
    assert (
        grade_task(
            "判定：返修。空洞面积超过 25%，不满足放行条件。", "返修", "quality_verdict"
        )
        is True
    )
    assert grade_task("判定：返修。", "放行", "quality_verdict") is False
    assert grade_task("结论：不合格，建议拦截。", "reject", "quality_verdict") is True
    assert grade_task("判定：放行。", "pass", "quality_verdict") is True
    assert grade_task("Verdict: REWORK due to voids.", "返修", "quality_verdict") is True
    assert grade_task("无法给出结论。", "返修", "quality_verdict") is False


def test_grade_process_calc_uses_numeric_tolerance():
    assert (
        grade_task(
            "30000 DPPM。计算：6 ÷ 200 × 1,000,000 = 30,000 DPPM。",
            30000,
            "process_calc",
        )
        is True
    )
    # Wrong order of magnitude is not within tolerance.
    assert grade_task("约为 3000 DPPM。", 30000, "process_calc") is False
    # Rounding within 1% still counts.
    assert grade_task("结果 29,998 DPPM", "30000", "process_calc") is True
    # Percent answers grade against both 25 and 0.25.
    assert grade_task("空洞面积约 25%", 0.25, "process_calc") is True
    assert grade_task("没有数字", 30000, "process_calc") is False


def test_open_ended_tasks_are_not_fake_graded():
    assert grade_task("any answer", "any expected", "standard_qa") is None
    assert grade_task(None, "返修", "quality_verdict") is None
    assert grade_task("判定：返修。", None, "quality_verdict") is None


def test_task_accuracy_skips_open_ended_records():
    records = [
        {"task": "quality_verdict", "response": "判定：返修。", "expected": "返修"},
        {"task": "process_calc", "response": "约为 3000 DPPM", "expected": "30000"},
        {"task": "general", "response": "anything", "expected": "x"},
        {"task": "quality_verdict", "expected": "返修"},
    ]

    assert task_accuracy(records) == 0.5
    assert task_accuracy([{"task": "general", "response": "x"}]) is None
    assert task_accuracy([]) is None


# ---------------------------------------------------------------------------
# Bootstrap interval
# ---------------------------------------------------------------------------


def test_bootstrap_interval_is_wider_for_small_samples():
    small = bootstrap_ci([0.0, 1.0, 0.0, 1.0], n_boot=1000, seed=3)
    large = bootstrap_ci([0.0, 1.0] * 500, n_boot=1000, seed=3)

    assert small[0] < small[1]
    assert large[0] < large[1]
    assert (small[1] - small[0]) > (large[1] - large[0])
    assert bootstrap_ci([0.0, 1.0, 0.0, 1.0], n_boot=1000, seed=3) == small


def test_bootstrap_interval_rejects_empty_values_and_zero_bootstraps():
    with pytest.raises(ValueError, match="at least one value"):
        bootstrap_ci([])
    with pytest.raises(ValueError, match="n_boot"):
        bootstrap_ci([1.0], n_boot=0)


# ---------------------------------------------------------------------------
# Training batch contract
# ---------------------------------------------------------------------------


class _ChatTokenizer(_StubTokenizer):
    """Stub with a chat template: each message renders as role id + content."""

    chat_template = "stub"

    def apply_chat_template(
        self, messages, tokenize=True, add_generation_prompt=False
    ):  # noqa: ARG002
        role_ids = {"system": 1, "user": 2, "assistant": 3}
        ids: list[int] = []
        for message in messages:
            ids.append(role_ids[message["role"]])
            ids.extend(self(message["content"])["input_ids"])
        if add_generation_prompt:
            ids.append(role_ids["assistant"])
        return ids


def test_onpolicy_train_dataset_maps_prompt_metadata_to_branch_label():
    tokenizer = _ChatTokenizer()
    rollout = {
        "prompt": "q",
        "source": "cloud",
        "prompt_ids": [5, 6, 7],
        "rollout_ids": [8, 9, 10],
    }
    supervised = {
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "ans"},
        ],
        "meta": {"source": "feedback"},
    }

    dataset = OnPolicyTrainDataset(
        tokenizer,
        rollout_rows=[rollout],
        supervised_rows=[supervised],
        max_seq_len=64,
    )

    assert len(dataset) == 2
    onpolicy, supervised_item = dataset[0], dataset[1]
    assert onpolicy["source"] == "onpolicy"  # not the prompt's "cloud" provenance
    assert onpolicy["input_ids"] == [5, 6, 7, 8, 9, 10]
    assert onpolicy["labels"] == [-100, -100, -100, 8, 9, 10]
    assert supervised_item["source"] == "supervised"
    answer_ids = tokenizer("ans", add_special_tokens=True)["input_ids"]
    assert supervised_item["input_ids"][-len(answer_ids) :] == answer_ids
    assert supervised_item["labels"][:-len(answer_ids)] == [
        -100
    ] * (len(supervised_item["input_ids"]) - len(answer_ids))
    assert supervised_item["labels"][-len(answer_ids) :] == answer_ids
    # Original provenance is preserved for audit, outside the batch contract.
    assert dataset.prompt_sources == ["cloud", "feedback"]
    assert set(dataset[0]) == {"input_ids", "attention_mask", "labels", "source"}


def test_onpolicy_train_dataset_truncates_prompt_before_rollout_tokens():
    dataset = OnPolicyTrainDataset(
        _StubTokenizer(),
        rollout_rows=[{"prompt_ids": [1, 2, 3, 4], "rollout_ids": [5, 6, 7]}],
        max_seq_len=5,
    )

    item = dataset[0]

    assert item["input_ids"] == [3, 4, 5, 6, 7]
    assert item["labels"] == [-100, -100, 5, 6, 7]


def test_collate_training_rows_pads_labels_and_keeps_branch_sources():
    rows = [
        {
            "input_ids": [5, 6, 7],
            "attention_mask": [1, 1, 1],
            "labels": [-100, 6, 7],
            "source": "onpolicy",
            "prompt_source": "cloud",
        },
        {
            "input_ids": [8, 9],
            "attention_mask": [1, 1],
            "labels": [8, 9],
            "source": "supervised",
        },
    ]

    batch = collate_training_rows(rows, pad_id=0)

    assert batch["input_ids"].shape == (2, 3)
    assert batch["input_ids"][0].tolist() == [5, 6, 7]
    assert batch["input_ids"][1].tolist() == [8, 9, 0]
    assert batch["attention_mask"][1].tolist() == [1, 1, 0]
    assert batch["labels"][0].tolist() == [-100, 6, 7]
    assert batch["labels"][1].tolist() == [8, 9, -100]
    assert batch["source"] == ["onpolicy", "supervised"]
    assert "prompt_source" not in batch


def test_collate_training_rows_rejects_prompt_metadata_as_source():
    row = {"input_ids": [1], "labels": [1], "source": "cloud"}

    with pytest.raises(ValueError, match="prompt_source"):
        collate_training_rows([row], pad_id=0)
    with pytest.raises(ValueError, match="at least one row"):
        collate_training_rows([], pad_id=0)


# ---------------------------------------------------------------------------
# Token accounting and same-budget row selection
# ---------------------------------------------------------------------------


def _loss_rows():
    return [
        {"source": "supervised", "labels": [-100, 1, 2]},  # 2 loss tokens
        {"source": "onpolicy", "labels": [-100, -100, 3]},  # 1 loss token
        {"source": "supervised", "labels": [-100, 4, 5, 6]},  # 3 loss tokens
    ]


def test_count_training_tokens_only_counts_branches_in_the_loss():
    rows = _loss_rows()

    assert count_training_tokens(rows, 0.0) == {
        "tokens_supervised": 5,
        "tokens_onpolicy": 0,
        "tokens_seen": 5,
    }
    assert count_training_tokens(rows, 1.0) == {
        "tokens_supervised": 0,
        "tokens_onpolicy": 1,
        "tokens_seen": 1,
    }
    assert count_training_tokens(rows, 0.5) == {
        "tokens_supervised": 5,
        "tokens_onpolicy": 1,
        "tokens_seen": 6,
    }


def test_truncate_training_rows_by_token_budget_keeps_whole_rows():
    rows = _loss_rows()

    selected = truncate_training_rows_by_token_budget(rows, 0.5, token_budget=4)

    assert selected == rows[:2]
    assert count_training_tokens(selected, 0.5)["tokens_seen"] == 3


def test_truncate_training_rows_by_token_budget_ignores_inactive_branches():
    rows = _loss_rows()

    supervised = truncate_training_rows_by_token_budget(rows, 0.0, token_budget=4)
    onpolicy = truncate_training_rows_by_token_budget(rows, 1.0, token_budget=4)

    # λ=0 only counts supervised rows, so the on-policy row is dropped even
    # though it is next in file order; λ=1 is the mirror image.
    assert supervised == [rows[0]]
    assert onpolicy == [rows[1]]


def test_truncate_training_rows_by_token_budget_keeps_at_least_one_row():
    rows = _loss_rows()

    selected = truncate_training_rows_by_token_budget(rows, 0.5, token_budget=1)

    # Mixed training keeps one row per branch even at a tiny budget; the
    # overshoot is explicit instead of silently dropping the second branch.
    assert selected == rows[:2]
    assert selected.tokens_used == 3
    assert selected.overflow is True


def test_truncate_training_rows_by_token_budget_none_keeps_all_active_rows():
    rows = _loss_rows()

    assert truncate_training_rows_by_token_budget(rows, 0.5, token_budget=None) == rows
    assert truncate_training_rows_by_token_budget(rows, 1.0, token_budget=None) == [rows[1]]


def _mixed_budget_rows():
    rows = []
    for index in range(4):
        rows.append(
            {
                "id": f"sup{index}",
                "source": "supervised",
                "labels": [-100] + [1] * 100,
            }
        )
    for index in range(4):
        rows.append(
            {
                "id": f"onp{index}",
                "source": "onpolicy",
                "labels": [-100] + [2] * 100,
            }
        )
    return rows


def test_mixed_budget_keeps_both_branches_round_robin():
    rows = _mixed_budget_rows()

    selection = truncate_training_rows_by_token_budget(rows, 0.5, token_budget=300)

    assert [row["id"] for row in selection] == ["sup0", "onp0", "sup1"]
    assert count_training_tokens(selection, 0.5) == {
        "tokens_supervised": 200,
        "tokens_onpolicy": 100,
        "tokens_seen": 300,
    }
    assert selection.tokens_used == 300
    assert selection.token_budget == 300
    assert selection.overflow is False
    assert (
        truncate_training_rows_by_token_budget(rows, 0.5, token_budget=300)
        == selection
    )


def test_mixed_budget_keeps_one_row_per_branch_when_budget_is_tight():
    rows = _mixed_budget_rows()

    for budget in (50, 150):
        selection = truncate_training_rows_by_token_budget(
            rows, 0.5, token_budget=budget
        )
        assert [row["id"] for row in selection] == ["sup0", "onp0"]
        assert count_training_tokens(selection, 0.5)["tokens_seen"] == 200
        assert selection.tokens_used == 200
        assert selection.overflow is True


def test_single_branch_budget_keeps_the_source_order_prefix():
    rows = _mixed_budget_rows()

    supervised = truncate_training_rows_by_token_budget(rows, 0.0, token_budget=250)
    onpolicy = truncate_training_rows_by_token_budget(rows, 1.0, token_budget=250)

    assert [row["id"] for row in supervised] == ["sup0", "sup1"]
    assert [row["id"] for row in onpolicy] == ["onp0", "onp1"]
    assert supervised.overflow is False
    assert onpolicy.overflow is False


def test_task_outcomes_returns_only_gradable_verdicts():
    records = [
        {"task": "quality_verdict", "response": "判定：返修。", "expected": "返修"},
        {"task": "process_calc", "response": "30000 DPPM", "expected": "30000"},
        {"task": "general", "response": "anything", "expected": "x"},
    ]

    assert task_outcomes(records) == [True, True]


# ---------------------------------------------------------------------------
# Length bucketing
# ---------------------------------------------------------------------------


def test_bucket_rows_by_length_is_deterministic_and_lossless():
    rows = [
        {"id": "long", "labels": [1, 2, 3, 4, 5]},
        {"id": "short", "labels": [9]},
        {"id": "mid-a", "labels": [7, 8, 9]},
        {"id": "mid-b", "labels": [4, 5, 6]},
    ]

    buckets = bucket_rows_by_length(rows, batch_size=2)

    assert [[row["id"] for row in bucket] for bucket in buckets] == [
        ["short", "mid-a"],
        ["mid-b", "long"],
    ]
    assert [row for bucket in buckets for row in bucket] == [
        rows[1],
        rows[2],
        rows[3],
        rows[0],
    ]
    assert bucket_rows_by_length(rows, batch_size=2) == buckets
    with pytest.raises(ValueError, match="batch_size"):
        bucket_rows_by_length(rows, batch_size=0)
    assert bucket_rows_by_length([], batch_size=2) == []


def test_onpolicy_train_dataset_sorts_by_loss_length_by_default():
    rows = [
        {"prompt_ids": [1, 2, 3, 4], "rollout_ids": [5]},
        {"prompt_ids": [6], "rollout_ids": [7, 8, 9]},
        {"prompt_ids": [1], "rollout_ids": [2, 3]},
    ]

    dataset = OnPolicyTrainDataset(_StubTokenizer(), rollout_rows=rows, max_seq_len=32)

    lengths = [sum(label != -100 for label in item["labels"]) for item in dataset]
    assert lengths == [1, 2, 3]
    assert dataset.prompt_sources == ["", "", ""]

    unsorted = OnPolicyTrainDataset(
        _StubTokenizer(), rollout_rows=rows, max_seq_len=32, sort_by_length=False
    )
    assert [item["labels"] for item in unsorted] == [
        [-100, -100, -100, -100, 5],
        [-100, 7, 8, 9],
        [-100, 2, 3],
    ]


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _report_kwargs(**overrides):
    kwargs = {
        "joint_ppl": 10.0,
        "shadow_ppl": 9.0,
        "task_accuracy": 0.75,
        "rollout_mode": "base_shadow",
        "lambda_onpolicy": 0.5,
        "divergence": "jsd",
        "teacher_support": "topk",
        "tokens_seen": 1234,
        "wall_clock": 1.5,
        "discard_rate": 0.25,
        "prompts": ["b", "a"],
        "teacher_version": "qwen3-4b",
    }
    kwargs.update(overrides)
    return kwargs


def test_build_report_contains_every_core_field():
    report = build_report(**_report_kwargs())

    assert set(report) == set(REPORT_FIELDS)
    assert report["joint_ppl"] == 10.0
    assert report["shadow_ppl"] == 9.0
    assert report["task_accuracy"] == 0.75
    assert report["lambda"] == 0.5
    assert report["no_go"] is False
    assert report["claims_gain"] is False
    assert report["token_budget_overflow"] is False
    assert len(report["data_fingerprint"]) == 64


def test_build_report_fails_when_a_core_field_is_missing():
    kwargs = _report_kwargs()
    kwargs.pop("joint_ppl")

    with pytest.raises(TypeError):
        build_report(**kwargs)


def test_data_fingerprint_tracks_data_identity_only():
    base = data_fingerprint(
        ["b", "a"], teacher_version="qwen3-4b", lambda_onpolicy=0.5, divergence="jsd"
    )

    assert (
        data_fingerprint(
            ["a", "b", "a"],
            teacher_version="qwen3-4b",
            lambda_onpolicy=0.5,
            divergence="jsd",
        )
        == base
    )
    for changed in (
        {"prompts": ["a", "c"]},
        {"teacher_version": "qwen3-8b"},
        {"lambda_onpolicy": 1.0},
        {"divergence": "forward_kl"},
    ):
        kwargs = {
            "prompts": ["b", "a"],
            "teacher_version": "qwen3-4b",
            "lambda_onpolicy": 0.5,
            "divergence": "jsd",
        }
        kwargs.update(changed)
        assert data_fingerprint(**kwargs) != base


def test_no_go_flags_shadow_gain_without_joint_gain():
    baseline = {"joint_ppl": 10.0, "shadow_ppl": 10.0}

    worse_joint = build_report(
        **_report_kwargs(joint_ppl=10.5, shadow_ppl=9.0, baseline_sft=baseline)
    )
    both_better = build_report(
        **_report_kwargs(joint_ppl=9.0, shadow_ppl=9.0, baseline_sft=baseline)
    )
    shadow_worse = build_report(
        **_report_kwargs(joint_ppl=9.0, shadow_ppl=11.0, baseline_sft=baseline)
    )

    assert worse_joint["no_go"] is True
    assert both_better["no_go"] is False
    assert shadow_worse["no_go"] is False
    assert worse_joint["baseline_sft"] == baseline


def test_no_go_covers_joint_and_task_accuracy_regressions():
    baseline = {"joint_ppl": 10.0, "shadow_ppl": 10.0, "task_accuracy": 0.8}

    joint_regression = build_report(
        **_report_kwargs(joint_ppl=10.5, shadow_ppl=10.5, task_accuracy=0.9, baseline_sft=baseline)
    )
    task_regression = build_report(
        **_report_kwargs(joint_ppl=9.0, shadow_ppl=9.0, task_accuracy=0.8, baseline_sft=baseline)
    )
    task_improved = build_report(
        **_report_kwargs(joint_ppl=9.0, shadow_ppl=9.0, task_accuracy=0.83, baseline_sft=baseline)
    )

    assert joint_regression["no_go"] is True
    assert task_regression["no_go"] is True
    assert task_improved["no_go"] is False


def test_claims_gain_requires_a_joint_guard_and_measurable_win():
    baseline = {"joint_ppl": 10.0, "shadow_ppl": 10.0}

    task_win = build_report(
        **_report_kwargs(
            joint_ppl=9.9,
            shadow_ppl=10.0,
            task_accuracy=0.54,
            baseline_sft={**baseline, "task_accuracy": 0.5},
        )
    )
    shadow_win = build_report(
        **_report_kwargs(
            joint_ppl=10.0,
            shadow_ppl=9.7,
            task_accuracy=None,
            baseline_sft=baseline,
        )
    )
    joint_regression = build_report(
        **_report_kwargs(
            joint_ppl=10.1,
            shadow_ppl=9.0,
            task_accuracy=0.9,
            baseline_sft={**baseline, "task_accuracy": 0.5},
        )
    )
    weak_task_win = build_report(
        **_report_kwargs(
            joint_ppl=9.9,
            shadow_ppl=10.0,
            task_accuracy=0.51,
            baseline_sft={**baseline, "task_accuracy": 0.5},
        )
    )

    assert task_win["claims_gain"] is True
    assert shadow_win["claims_gain"] is True
    assert joint_regression["claims_gain"] is False
    assert weak_task_win["claims_gain"] is False


def test_no_baseline_never_claims_a_gain_or_a_no_go():
    report = build_report(**_report_kwargs(baseline_sft=None))

    assert report["no_go"] is False
    assert report["claims_gain"] is False


def test_build_report_adds_bootstrap_intervals_and_token_split():
    kwargs = _report_kwargs(
        tokens_seen=150,
        tokens_supervised=100,
        tokens_onpolicy=50,
        token_budget=148,
        token_budget_overflow=True,
        joint_losses=[2.0, 2.1, 2.2, 2.3],
        shadow_losses=[1.9, 2.0, 2.1, 2.2],
        window_tokens=[8, 8, 8, 8],
        task_outcomes=[True, False, True, True],
        n_boot=300,
        ci_seed=7,
    )

    report = build_report(**kwargs)
    again = build_report(**kwargs)

    assert report["tokens_supervised"] == 100
    assert report["tokens_onpolicy"] == 50
    assert report["token_budget"] == 148
    assert report["token_budget_overflow"] is True
    assert report["ci_n_boot"] == 300
    assert report["ci_seed"] == 7
    for key in ("joint_ppl_ci", "shadow_ppl_ci", "task_accuracy_ci"):
        assert len(report[key]) == 2
        assert report[key][0] <= report[key][1]
        assert report[key] == again[key]
    assert report["task_accuracy_ci"][0] <= 0.75 <= report["task_accuracy_ci"][1]


def test_single_task_record_gets_no_accuracy_interval():
    report = build_report(**_report_kwargs(task_outcomes=[True]))

    assert report["task_accuracy_ci"] is None
    assert report["joint_ppl_ci"] is None
    assert report["shadow_ppl_ci"] is None


def test_tokens_seen_must_match_the_branch_split_when_both_are_given():
    with pytest.raises(ValueError, match="tokens_seen"):
        build_report(
            **_report_kwargs(
                tokens_seen=999,
                tokens_supervised=100,
                tokens_onpolicy=50,
            )
        )


def test_build_report_rejects_invalid_metrics():
    with pytest.raises(ValueError, match="joint_ppl"):
        build_report(**_report_kwargs(joint_ppl=float("inf")))
    with pytest.raises(ValueError, match="discard_rate"):
        build_report(**_report_kwargs(discard_rate=1.5))
    with pytest.raises(ValueError, match="teacher_support"):
        build_report(**_report_kwargs(teacher_support="approx"))


# ---------------------------------------------------------------------------
# CLI parser (no model loading)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cli_module():
    path = ROOT / "scripts" / "exp10_onpolicy_gkd.py"
    spec = importlib.util.spec_from_file_location("exp10_onpolicy_gkd", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_help_is_available_for_every_subcommand(cli_module):
    for argv in ([], ["rollouts"], ["train"], ["eval"], ["run"]):
        with pytest.raises(SystemExit) as excinfo:
            cli_module.build_parser().parse_args([*argv, "--help"])
        assert excinfo.value.code == 0


def test_cli_parses_rollouts_arguments(cli_module):
    args = cli_module.build_parser().parse_args(
        [
            "rollouts",
            "--prompts",
            "prompts.jsonl",
            "--out",
            "rollouts.jsonl",
            "--model",
            "base-model",
            "--shadow-ckpt",
            "adapter",
            "--shadow-model-path",
            "shadow-model",
            "--rollout-mode",
            "shadow_only",
            "--g",
            "3",
            "--max-new-tokens",
            "32",
            "--seed",
            "7",
        ]
    )

    assert args.command == "rollouts"
    assert args.prompts == "prompts.jsonl"
    assert args.out == "rollouts.jsonl"
    assert args.shadow_ckpt == "adapter"
    assert args.shadow_model_path == "shadow-model"
    assert args.rollout_mode == "shadow_only"
    assert args.num_rollouts == 3
    assert args.max_new_tokens == 32
    assert args.seed == 7


def test_cli_parses_train_eval_and_run_arguments(cli_module):
    parser = cli_module.build_parser()
    train = parser.parse_args(
        [
            "train",
            "--model",
            "base-model",
            "--teacher",
            "teacher-model",
            "--rollouts",
            "rollouts.jsonl",
            "--supervised",
            "sft.jsonl",
            "--out",
            "adapter",
            "--lambda",
            "0.25",
            "--divergence",
            "jsd",
            "--topk",
            "16",
            "--shadow-aux-weight",
            "0.1",
            "--max-steps",
            "3",
        ]
    )
    assert train.command == "train"
    assert train.lambda_onpolicy == 0.25
    assert train.divergence == "jsd"
    assert train.topk == 16
    assert train.shadow_aux_weight == 0.1
    assert train.max_steps == 3
    assert train.supervised == "sft.jsonl"

    evaluation = parser.parse_args(
        [
            "eval",
            "--model",
            "base-model",
            "--shadow-ckpt",
            "adapter",
            "--heldout",
            "heldout.jsonl",
            "--tasks",
            "tasks.jsonl",
            "--out",
            "report.json",
        ]
    )
    assert evaluation.command == "eval"
    assert evaluation.heldout == "heldout.jsonl"
    assert evaluation.tasks == "tasks.jsonl"

    run = parser.parse_args(
        [
            "run",
            "--prompts",
            "prompts.jsonl",
            "--model",
            "base-model",
            "--shadow-ckpt",
            "adapter",
            "--teacher",
            "teacher-model",
            "--heldout",
            "heldout.jsonl",
            "--out",
            "runs/exp10/smoke",
            "--smoke",
        ]
    )
    assert run.command == "run"
    assert run.smoke is True


def test_cli_rejects_missing_required_arguments(cli_module):
    with pytest.raises(SystemExit) as excinfo:
        cli_module.build_parser().parse_args(
            ["rollouts", "--out", "rollouts.jsonl", "--model", "m"]
        )
    assert excinfo.value.code == 2


def test_cli_token_ids_accepts_batch_encoding_like_mappings(cli_module):
    from collections import UserDict

    assert cli_module._token_ids(None, UserDict({"input_ids": [1, 2, 3]})) == [1, 2, 3]
    assert cli_module._token_ids(None, {"input_ids": [[4, 5, 6]]}) == [4, 5, 6]
    assert cli_module._token_ids(None, [7, 8]) == [7, 8]


def test_cli_parses_token_budget_length_sorting_and_ci_knobs(cli_module):
    parser = cli_module.build_parser()

    train = parser.parse_args(
        [
            "train",
            "--model",
            "base-model",
            "--teacher",
            "teacher-model",
            "--rollouts",
            "rollouts.jsonl",
            "--out",
            "adapter",
            "--token-budget",
            "448",
        ]
    )
    no_sort = parser.parse_args(
        [
            "train",
            "--model",
            "base-model",
            "--teacher",
            "teacher-model",
            "--rollouts",
            "rollouts.jsonl",
            "--out",
            "adapter",
            "--no-sort-by-length",
        ]
    )
    evaluation = parser.parse_args(
        [
            "eval",
            "--model",
            "base-model",
            "--shadow-ckpt",
            "adapter",
            "--heldout",
            "heldout.jsonl",
            "--out",
            "report.json",
            "--ci-n-boot",
            "50",
            "--ci-seed",
            "3",
            "--tokens-supervised",
            "100",
            "--tokens-onpolicy",
            "20",
            "--token-budget",
            "448",
            "--token-budget-overflow",
        ]
    )

    assert train.token_budget == 448
    assert train.sort_by_length is True
    assert no_sort.sort_by_length is False
    assert evaluation.ci_n_boot == 50
    assert evaluation.ci_seed == 3
    assert evaluation.tokens_supervised == 100
    assert evaluation.tokens_onpolicy == 20
    assert evaluation.token_budget == 448
    assert evaluation.token_budget_overflow is True


def test_cli_fill_task_responses_mutates_task_records(cli_module):
    records = [{"prompt": "a"}, {"prompt": "b"}]

    out = cli_module._fill_task_responses(records, ["answer-a", "answer-b"])

    assert out is records
    assert [record["response"] for record in records] == ["answer-a", "answer-b"]
