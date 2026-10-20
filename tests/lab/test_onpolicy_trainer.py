"""Trainer-seam tests for the on-policy GKD trainer (issue #20).

``onpolicy_gkd_loss`` is the pure loss composition and is tested directly so
the arithmetic needs no TRL trainer. A few trainer-level tests drive
``OnPolicyGKDTrainer.compute_loss`` on a bare instance (``object.__new__``):
they cover the ``source`` split, the teacher forward, a real backward pass and
two optimizer steps without paying for ``SFTTrainer.__init__``.

Tiny random Llama student/teacher, CPU, fixed seeds.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("trl")  # shadow_lab.trainers imports TRL at module load.

from transformers import LlamaConfig, LlamaForCausalLM  # noqa: E402

from shadow_lab.trainers import OnPolicyGKDTrainer, onpolicy_gkd_loss  # noqa: E402
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
