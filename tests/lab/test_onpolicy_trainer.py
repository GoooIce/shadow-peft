"""Trainer-seam tests for the on-policy GKD trainer (issue #20).

``onpolicy_gkd_loss`` is the pure loss composition and is tested directly so
the arithmetic needs no TRL trainer. A few trainer-level tests drive
``OnPolicyGKDTrainer.compute_loss`` on a bare instance (``object.__new__``):
they cover the ``source`` split, the teacher forward, a real backward pass and
two optimizer steps without paying for ``SFTTrainer.__init__``.

Tiny random Llama student/teacher, CPU, fixed seeds.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("trl")  # shadow_lab.trainers imports TRL at module load.

from transformers import LlamaConfig, LlamaForCausalLM  # noqa: E402

from shadow_lab.onpolicy_eval import evaluate_ppl  # noqa: E402
from shadow_lab.trainers import (  # noqa: E402
    OnPolicyGKDTrainer,
    group_parameters_by_shadow,
    onpolicy_gkd_loss,
    validate_teacher_vocab,
)
from shadow_peft import ShadowConfig, ShadowForCausalLM, get_shadow_model  # noqa: E402
from shadow_peft.task_models import _shifted_ce_loss, _shifted_kd_loss  # noqa: E402

SHADOW_PARAM_PREFIXES = (
    "peft_model.shadow_model",
    "peft_model.shadow_injection_model",
    "peft_model.shadow_update_model",
)


def _tiny_llama(vocab_size: int = 32, seed: int = 0) -> LlamaForCausalLM:
    torch.manual_seed(seed)
    cfg = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=32,
    )
    return LlamaForCausalLM(cfg)


def _tiny_shadow_student(vocab_size: int = 32, seed: int = 0) -> ShadowForCausalLM:
    base = _tiny_llama(vocab_size, seed)
    cfg = ShadowConfig(
        num_shadow_layers=1,
        injection_hidden_size=8,
        gate_hidden_size=10,
        alpha=0.1,
        dropout=0.0,
    )
    peft = get_shadow_model(base, cfg)
    return ShadowForCausalLM(
        peft,
        shadow_loss_weight=0.1,
        inference_mode="base_shadow",
        distill_weight=0.5,
        distill_temperature=1.0,
    )


def _labels(batch: int = 4, seq: int = 8, vocab_size: int = 32, seed: int = 11) -> torch.Tensor:
    torch.manual_seed(seed)
    labels = torch.randint(0, vocab_size, (batch, seq))
    labels[:, :2] = -100  # prompt positions, ignored by both shifted losses
    return labels


def _trainer_batch(vocab_size: int = 32, batch: int = 4, seq: int = 8):
    torch.manual_seed(21)
    input_ids = torch.randint(2, vocab_size, (batch, seq))
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    labels[:, :2] = -100
    return input_ids, attention_mask, labels


def _bare_trainer(
    teacher_model=None,
    *,
    lambda_onpolicy: float = 0.5,
    divergence: str = "forward_kl",
    topk: int | None = None,
    shadow_aux_weight: float = 0.0,
    teacher_temperature: float = 1.0,
) -> OnPolicyGKDTrainer:
    """OnPolicyGKDTrainer without SFTTrainer.__init__.

    ``compute_loss`` is the seam under test and only reads these knobs.
    """
    trainer = object.__new__(OnPolicyGKDTrainer)
    trainer.teacher_model = teacher_model
    trainer.lambda_onpolicy = lambda_onpolicy
    trainer.divergence = divergence
    trainer.topk = topk
    trainer.shadow_aux_weight = shadow_aux_weight
    trainer.teacher_temperature = teacher_temperature
    return trainer


# ---------------------------------------------------------------------------
# Pure loss composition
# ---------------------------------------------------------------------------


def test_kd_is_zero_when_student_matches_teacher():
    torch.manual_seed(1)
    logits = torch.randn(2, 6, 32)
    labels = _labels(batch=2, seq=6)

    out = onpolicy_gkd_loss(
        logits, logits.clone(), None, labels, source="onpolicy", lambda_onpolicy=0.3
    )

    assert out["ce"].item() == 0.0
    assert abs(out["kd"].item()) < 1e-6
    assert abs(out["total"].item()) < 1e-6


def test_lambda_zero_returns_supervised_ce_only():
    torch.manual_seed(2)
    student = torch.randn(2, 6, 32)
    teacher = student + 0.5 * torch.randn_like(student)
    labels = _labels(batch=2, seq=6)

    out = onpolicy_gkd_loss(
        student,
        teacher,
        None,
        labels,
        source="supervised",
        lambda_onpolicy=0.0,
        topk=4,
    )

    expected_ce = _shifted_ce_loss(student, labels)
    assert torch.allclose(out["ce"], expected_ce)
    assert out["kd"].item() == 0.0
    assert torch.allclose(out["total"], expected_ce)


def test_lambda_one_returns_onpolicy_kd_only():
    torch.manual_seed(3)
    student = torch.randn(2, 6, 32)
    teacher = student + 0.5 * torch.randn_like(student)
    labels = _labels(batch=2, seq=6)

    out = onpolicy_gkd_loss(
        student,
        teacher,
        None,
        labels,
        source="onpolicy",
        lambda_onpolicy=1.0,
        divergence="jsd",
        topk=8,
        temperature=2.0,
    )

    expected_kd = _shifted_kd_loss(
        student, teacher, labels, 2.0, divergence="jsd", topk=8
    )
    assert out["ce"].item() == 0.0
    assert torch.allclose(out["kd"], expected_kd)
    assert torch.allclose(out["total"], expected_kd)


def test_mixed_source_splits_supervised_and_onpolicy_rows():
    torch.manual_seed(4)
    student = torch.randn(4, 8, 32)
    teacher = student + 0.2 * torch.randn_like(student)
    labels = _labels()
    source = ["supervised", "onpolicy", "supervised", "onpolicy"]
    supervised = torch.tensor([True, False, True, False])
    onpolicy = ~supervised

    out = onpolicy_gkd_loss(
        student,
        teacher,
        None,
        labels,
        source=source,
        lambda_onpolicy=0.25,
        divergence="reverse_kl",
        topk=8,
    )

    expected_ce = _shifted_ce_loss(student[supervised], labels[supervised])
    expected_kd = _shifted_kd_loss(
        student[onpolicy],
        teacher[onpolicy],
        labels[onpolicy],
        1.0,
        divergence="reverse_kl",
        topk=8,
    )
    assert torch.allclose(out["ce"], expected_ce)
    assert torch.allclose(out["kd"], expected_kd)
    assert torch.allclose(out["total"], 0.75 * expected_ce + 0.25 * expected_kd)

    # The teacher may carry only the on-policy rows, which is what
    # compute_loss forwards for large batches.
    out_subset = onpolicy_gkd_loss(
        student,
        teacher[onpolicy],
        None,
        labels,
        source=source,
        lambda_onpolicy=0.25,
        divergence="reverse_kl",
        topk=8,
    )
    assert torch.allclose(out_subset["total"], out["total"])

    # A 0/1 tensor source selects the same rows as the string list.
    out_tensor = onpolicy_gkd_loss(
        student,
        teacher,
        None,
        labels,
        source=torch.tensor([0, 1, 0, 1]),
        lambda_onpolicy=0.25,
        divergence="reverse_kl",
        topk=8,
    )
    assert torch.allclose(out_tensor["total"], out["total"])


def test_shadow_aux_weight_adds_shadow_ce():
    torch.manual_seed(5)
    student = torch.randn(2, 6, 32)
    teacher = student + 0.5 * torch.randn_like(student)
    shadow = student + 0.5 * torch.randn_like(student)
    labels = _labels(batch=2, seq=6)

    out = onpolicy_gkd_loss(
        student,
        teacher,
        shadow,
        labels,
        source="supervised",
        lambda_onpolicy=0.5,
        shadow_aux_weight=0.5,
    )

    expected_ce = _shifted_ce_loss(student, labels)
    expected_aux = _shifted_ce_loss(shadow, labels)
    assert torch.allclose(out["aux"], expected_aux)
    assert torch.allclose(out["total"], 0.5 * expected_ce + 0.5 * expected_aux)


def test_shadow_aux_is_off_by_default():
    torch.manual_seed(6)
    student = torch.randn(2, 6, 32)
    shadow = student + 0.5 * torch.randn_like(student)
    labels = _labels(batch=2, seq=6)

    out = onpolicy_gkd_loss(student, None, shadow, labels, source="supervised")

    assert out["aux"].item() == 0.0
    assert torch.allclose(out["total"], 0.5 * _shifted_ce_loss(student, labels))


def test_missing_source_defaults_to_supervised():
    torch.manual_seed(7)
    student = torch.randn(2, 6, 32)
    labels = _labels(batch=2, seq=6)

    out = onpolicy_gkd_loss(student, None, None, labels)

    assert out["kd"].item() == 0.0
    assert torch.allclose(out["total"], 0.5 * _shifted_ce_loss(student, labels))


def test_lambda_zero_does_not_require_teacher():
    torch.manual_seed(8)
    student = torch.randn(2, 6, 32)
    labels = _labels(batch=2, seq=6)

    out = onpolicy_gkd_loss(
        student,
        None,
        None,
        labels,
        source=["supervised", "onpolicy"],
        lambda_onpolicy=0.0,
    )

    expected = _shifted_ce_loss(student[:1], labels[:1])
    assert torch.allclose(out["total"], expected)


def test_vocab_mismatch_raises_with_both_sizes():
    torch.manual_seed(9)
    student = torch.randn(1, 5, 32)
    teacher = torch.randn(1, 5, 40)
    labels = _labels(batch=1, seq=5)

    with pytest.raises(ValueError, match=r"teacher has vocab 40.*student has vocab 32"):
        onpolicy_gkd_loss(student, teacher, None, labels, source="onpolicy")


# ---------------------------------------------------------------------------
# Trainer seam
# ---------------------------------------------------------------------------


def test_compute_loss_mixed_batch_backward_and_optimizer_steps():
    student = _tiny_shadow_student()
    teacher = _tiny_llama(seed=1)
    trainer = _bare_trainer(
        teacher, lambda_onpolicy=0.5, topk=8, shadow_aux_weight=0.1
    )
    input_ids, attention_mask, labels = _trainer_batch()
    source = ["supervised", "onpolicy", "supervised", "onpolicy"]

    loss, outputs = trainer.compute_loss(
        student,
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "source": source,
        },
        return_outputs=True,
    )

    assert torch.isfinite(loss)
    assert outputs.logits.shape == (4, 8, 32)
    assert outputs.shadow_logits is not None

    loss.backward()
    assert all(p.grad is None for p in teacher.parameters())
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for name, p in student.named_parameters()
        if name.startswith(SHADOW_PARAM_PREFIXES)
    )

    optimizer = torch.optim.SGD(
        [p for p in student.parameters() if p.requires_grad], lr=1e-2
    )
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    second_loss = trainer.compute_loss(
        student,
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "source": source,
        },
    )
    assert torch.isfinite(second_loss)
    second_loss.backward()
    optimizer.step()


def test_compute_loss_lambda_zero_skips_teacher():
    student = _tiny_shadow_student()
    trainer = _bare_trainer(None, lambda_onpolicy=0.0, topk=8)
    input_ids, attention_mask, labels = _trainer_batch()

    loss = trainer.compute_loss(
        student,
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "source": ["supervised", "onpolicy", "supervised", "onpolicy"],
        },
    )

    assert torch.isfinite(loss)


def test_compute_loss_vocab_mismatch_raises():
    student = _tiny_shadow_student()
    teacher = _tiny_llama(vocab_size=40, seed=2)
    trainer = _bare_trainer(teacher)
    input_ids, attention_mask, labels = _trainer_batch()

    with pytest.raises(ValueError, match="vocab"):
        trainer.compute_loss(
            student,
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "source": "onpolicy",
            },
        )


# ---------------------------------------------------------------------------
# Teacher vocabulary, evaluation loop and LR grouping
# ---------------------------------------------------------------------------


def test_validate_teacher_vocab_reports_both_sizes():
    student = _tiny_shadow_student(vocab_size=32)
    teacher = _tiny_llama(vocab_size=40, seed=2)

    with pytest.raises(
        ValueError, match=r"teacher vocabulary 40.*student vocabulary 32"
    ):
        validate_teacher_vocab(student, teacher)


def test_trainer_init_rejects_teacher_vocab_mismatch_even_at_lambda_zero():
    student = _tiny_shadow_student(vocab_size=32)
    teacher = _tiny_llama(vocab_size=40, seed=2)

    with pytest.raises(
        ValueError, match=r"teacher vocabulary 40.*student vocabulary 32"
    ):
        OnPolicyGKDTrainer(
            model=student,
            args=None,
            teacher_model=teacher,
            lambda_onpolicy=0.0,
        )


def test_evaluation_loop_strips_source_before_the_model_forward():
    student = _tiny_shadow_student()
    seen: list[dict] = []
    original_forward = student.forward

    def recording_forward(*args, **kwargs):
        seen.append(dict(kwargs))
        return original_forward(*args, **kwargs)

    student.forward = recording_forward
    trainer = object.__new__(OnPolicyGKDTrainer)
    trainer.model = student
    trainer.args = SimpleNamespace()
    trainer._wrap_model = lambda model, training=False, dataloader=None: model
    trainer._prepare_inputs = lambda inputs: inputs
    input_ids, attention_mask, labels = _trainer_batch(batch=2, seq=8)

    output = trainer.evaluation_loop(
        [
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "source": ["supervised", "onpolicy"],
            }
        ],
        "eval",
    )

    assert seen
    assert all("source" not in kwargs for kwargs in seen)
    assert output.metrics["eval_samples"] == 2
    assert torch.isfinite(torch.tensor(output.metrics["eval_loss"]))


def test_shadow_lr_grouping_includes_injection_and_update_modules():
    student = _tiny_shadow_student()

    groups = group_parameters_by_shadow(student)
    names_by_id = {id(param): name for name, param in student.named_parameters()}
    shadow_names = {
        names_by_id[id(param)]
        for key in ("shadow_decay", "shadow_nodecay")
        for param in groups[key]
    }
    base_names = {
        names_by_id[id(param)]
        for key in ("base_decay", "base_nodecay")
        for param in groups[key]
    }

    assert any(name.startswith("peft_model.shadow_injection_model.") for name in shadow_names)
    assert any(name.startswith("peft_model.shadow_update_model.") for name in shadow_names)
    assert not any(name.startswith("peft_model.base_model.") for name in shadow_names)
    assert not any("shadow_" in name for name in base_names)


# ---------------------------------------------------------------------------
# Checkpoint round-trip
# ---------------------------------------------------------------------------


class _StubTokenizer:
    pad_token_id = 0
    eos_token_id = None
    chat_template = None

    def __call__(self, text, add_special_tokens=True):  # noqa: ARG002
        return {"input_ids": [3 + (ord(char) % 28) for char in text]}


def test_checkpoint_roundtrip_preserves_joint_and_shadow_ppl(tmp_path):
    student = _tiny_shadow_student()
    input_ids, attention_mask, labels = _trainer_batch(batch=2, seq=8)
    optimizer = torch.optim.SGD(
        [param for param in student.parameters() if param.requires_grad], lr=1e-2
    )
    loss = student(
        input_ids=input_ids, attention_mask=attention_mask, labels=labels
    ).loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    texts = ["the quick brown fox jumps over the lazy dog " * 2]
    before = evaluate_ppl(
        student, _StubTokenizer(), texts, seq_len=8, max_windows=2
    )
    adapter_dir = tmp_path / "adapter"
    student.save_pretrained(adapter_dir)

    torch.manual_seed(0)
    reloaded = ShadowForCausalLM.from_pretrained(
        _tiny_llama(vocab_size=32, seed=0), adapter_dir
    )
    after = evaluate_ppl(
        reloaded, _StubTokenizer(), texts, seq_len=8, max_windows=2
    )

    assert after["joint_ppl"] == pytest.approx(before["joint_ppl"], rel=1e-10)
    assert after["shadow_ppl"] == pytest.approx(before["shadow_ppl"], rel=1e-10)
    assert after["tokens"] == before["tokens"]


# ---------------------------------------------------------------------------
# Dynamic mass-adaptive top-k support
# ---------------------------------------------------------------------------


def test_dynamic_topk_mass_one_matches_fixed_topk():
    torch.manual_seed(31)
    student = torch.randn(2, 6, 32)
    teacher = torch.randn(2, 6, 32)
    labels = _labels(batch=2, seq=6)

    fixed = _shifted_kd_loss(
        student, teacher, labels, 1.0, divergence="jsd", topk=8
    )
    dynamic, stats = _shifted_kd_loss(
        student,
        teacher,
        labels,
        1.0,
        divergence="jsd",
        topk=8,
        topk_mass=1.0,
        topk_min=1,
        return_stats=True,
    )

    assert torch.allclose(fixed, dynamic, atol=1e-6)
    assert stats["support_k_mean"] == 8.0
    assert stats["support_k_min"] == 8
    assert stats["support_k_max"] == 8
    assert stats["support_k_max_fraction"] == 1.0


def test_dynamic_topk_mass_adapts_width_and_reports_valid_tokens():
    # Row 0 is peaked at token 0 for every position; row 1 is flat.
    teacher = torch.zeros(2, 6, 32)
    teacher[0, :, 0] = 20.0
    labels = _labels(batch=2, seq=6)

    _, stats = _shifted_kd_loss(
        torch.zeros_like(teacher),
        teacher,
        labels,
        1.0,
        divergence="jsd",
        topk=8,
        topk_mass=0.9,
        topk_min=2,
        return_stats=True,
    )

    # Peak rows stop at topk_min; flat rows need the full max k.
    assert stats["support_tokens"] == 8  # 2 rows x 4 loss-participating positions
    assert stats["support_k_min"] == 2
    assert stats["support_k_max"] == 8
    assert 2.0 < stats["support_k_mean"] < 8.0


def test_dynamic_topk_rejects_missing_max_k_and_bad_bounds():
    student = torch.randn(1, 4, 32)
    teacher = torch.randn(1, 4, 32)
    labels = _labels(batch=1, seq=4)

    with pytest.raises(ValueError, match="topk_mass requires topk"):
        _shifted_kd_loss(
            student, teacher, labels, 1.0, topk=None, topk_mass=0.9
        )
    with pytest.raises(ValueError, match="topk_mass must be in"):
        _shifted_kd_loss(
            student, teacher, labels, 1.0, topk=8, topk_mass=0.0
        )
    with pytest.raises(ValueError, match="topk_min"):
        _shifted_kd_loss(
            student,
            teacher,
            labels,
            1.0,
            topk=8,
            topk_mass=0.9,
            topk_min=9,
        )


def test_onpolicy_gkd_loss_exposes_dynamic_support_stats():
    torch.manual_seed(32)
    student = torch.randn(2, 6, 32)
    teacher = torch.randn(2, 6, 32)
    labels = _labels(batch=2, seq=6)

    out = onpolicy_gkd_loss(
        student,
        teacher,
        None,
        labels,
        source="onpolicy",
        lambda_onpolicy=1.0,
        divergence="jsd",
        topk=8,
        topk_mass=0.9,
        topk_min=1,
    )
    expected = _shifted_kd_loss(
        student,
        teacher,
        labels,
        1.0,
        divergence="jsd",
        topk=8,
        topk_mass=0.9,
        topk_min=1,
    )

    assert out["kd_stats"]["support_tokens"] > 0
    assert torch.allclose(out["kd"], expected)


def test_trainer_accumulates_dynamic_support_stats():
    trainer = object.__new__(OnPolicyGKDTrainer)
    trainer._support_accum = None

    trainer._accumulate_support_stats(
        {
            "support_tokens": 2,
            "support_k_mean": 3.0,
            "support_k_min": 2,
            "support_k_max": 5,
            "support_k_max_fraction": 0.5,
        }
    )
    trainer._accumulate_support_stats(
        {
            "support_tokens": 2,
            "support_k_mean": 5.0,
            "support_k_min": 4,
            "support_k_max": 6,
            "support_k_max_fraction": 0.0,
        }
    )

    summary = trainer.support_summary()
    assert summary["tokens"] == 4
    assert summary["k_mean"] == pytest.approx(4.0)
    assert summary["k_min"] == 2
    assert summary["k_max"] == 6
    assert summary["k_max_fraction"] == pytest.approx(0.25)
