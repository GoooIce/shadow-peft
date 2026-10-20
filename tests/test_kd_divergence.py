from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

import torch.nn.functional as F  # noqa: E402

from shadow_peft.task_models import _shifted_kd_loss  # noqa: E402

DIVERGENCES = ("forward_kl", "reverse_kl", "jsd")


def _tensors(batch: int = 2, seq: int = 6, vocab: int = 16, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    student = torch.randn(batch, seq, vocab, generator=generator)
    teacher = torch.randn(batch, seq, vocab, generator=generator)
    labels = torch.randint(0, vocab, (batch, seq), generator=generator)
    labels[:, :2] = -100  # exercise shift/ignore_index alignment
    return student, teacher, labels


def _legacy_forward_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Frozen copy of `_shifted_kd_loss` before divergence/top-k were added.

    The new default must reproduce this bit for bit, not merely within a
    tolerance: existing training runs depend on the exact loss values.
    """
    shift_student = student_logits[..., :-1, :].contiguous()
    shift_teacher = teacher_logits[..., :-1, :].contiguous().to(dtype=shift_student.dtype)
    shift_labels = labels[..., 1:].contiguous()
    student_log_probs = F.log_softmax(
        shift_student.view(-1, shift_student.size(-1)) / temperature, dim=-1
    )
    teacher_probs = F.softmax(
        shift_teacher.view(-1, shift_teacher.size(-1)) / temperature, dim=-1
    )
    valid = shift_labels.view(-1) != -100
    kd = F.kl_div(student_log_probs[valid], teacher_probs[valid], reduction="batchmean")
    return kd * temperature * temperature


# ---------------------------------------------------------------------------
# Default forward KL: exact backward compatibility
# ---------------------------------------------------------------------------


def test_default_forward_kl_is_bitwise_identical_to_legacy_implementation():
    student, teacher, labels = _tensors()
    for temperature in (0.5, 1.0, 1.7, 2.0):
        got = _shifted_kd_loss(student, teacher, labels, temperature)
        expected = _legacy_forward_kl(student, teacher, labels, temperature)
        assert torch.equal(got, expected)


def test_explicit_forward_kl_matches_implicit_default_bitwise():
    student, teacher, labels = _tensors(vocab=12, seed=1)
    implicit = _shifted_kd_loss(student, teacher, labels, 1.3)
    explicit = _shifted_kd_loss(student, teacher, labels, 1.3, divergence="forward_kl", topk=None)
    assert torch.equal(implicit, explicit)


# ---------------------------------------------------------------------------
# reverse KL and JSD: mathematical properties
# ---------------------------------------------------------------------------


def test_reverse_kl_zero_when_student_equals_teacher():
    student, _, labels = _tensors(seed=2)
    loss = _shifted_kd_loss(
        student, student.clone(), labels, 1.3, divergence="reverse_kl"
    )
    assert abs(loss.item()) < 1e-6


def test_reverse_kl_matches_explicit_student_to_teacher_kl():
    student, teacher, labels = _tensors(vocab=8, seed=3)
    temperature = 1.5
    shift_student = student[..., :-1, :].reshape(-1, 8) / temperature
    shift_teacher = teacher[..., :-1, :].reshape(-1, 8) / temperature
    valid = labels[..., 1:].reshape(-1) != -100
    student_log_probs = F.log_softmax(shift_student, dim=-1)
    teacher_log_probs = F.log_softmax(shift_teacher, dim=-1)
    per_token = (student_log_probs.exp() * (student_log_probs - teacher_log_probs)).sum(-1)
    expected = per_token[valid].mean() * temperature * temperature

    got = _shifted_kd_loss(student, teacher, labels, temperature, divergence="reverse_kl")
    assert torch.allclose(got, expected, atol=1e-6, rtol=1e-5)


def test_jsd_zero_for_identical_symmetric_and_bounded():
    student, teacher, labels = _tensors(vocab=8, seed=4)

    identical = _shifted_kd_loss(student, student.clone(), labels, 1.0, divergence="jsd")
    assert abs(identical.item()) < 1e-6

    forward = _shifted_kd_loss(student, teacher, labels, 1.0, divergence="jsd")
    backward = _shifted_kd_loss(teacher, student, labels, 1.0, divergence="jsd")
    assert torch.allclose(forward, backward, atol=1e-7)
    assert 0.0 <= forward.item() <= math.log(2.0) + 1e-6


def test_jsd_matches_explicit_two_distribution_formula():
    student, teacher, labels = _tensors(vocab=8, seed=5)
    temperature = 2.0
    valid = labels[..., 1:].reshape(-1) != -100
    student_probs = F.softmax(student[..., :-1, :].reshape(-1, 8) / temperature, dim=-1)
    teacher_probs = F.softmax(teacher[..., :-1, :].reshape(-1, 8) / temperature, dim=-1)
    mixture = 0.5 * (student_probs + teacher_probs)
    kl_student = (student_probs * (student_probs.log() - mixture.log())).sum(-1)
    kl_teacher = (teacher_probs * (teacher_probs.log() - mixture.log())).sum(-1)
    expected = (0.5 * kl_student + 0.5 * kl_teacher)[valid].mean()

    got = _shifted_kd_loss(student, teacher, labels, temperature, divergence="jsd")
    assert torch.allclose(got, expected, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("divergence", DIVERGENCES)
def test_only_non_ignored_shifted_positions_contribute(divergence):
    # labels[0] = -100 and labels[2] = -100 mean shifted positions 1 is
    # ignored; positions 0 and 2 keep identical student/teacher logits.
    student = torch.randn(1, 4, 8, generator=torch.Generator().manual_seed(6))
    teacher = student.clone()
    labels = torch.tensor([[-100, 3, -100, 5]])
    student[0, 1] = torch.tensor([10.0] + [0.0] * 7)
    teacher[0, 1] = torch.tensor([0.0, 10.0] + [0.0] * 6)

    loss = _shifted_kd_loss(student, teacher, labels, 1.0, divergence=divergence)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# top-k union: renormalization and NaN safety
# ---------------------------------------------------------------------------


def test_topk_uses_student_teacher_union_and_renormalizes():
    # One valid shifted position, vocab 4, topk=1: the student's top-1 is 0,
    # the teacher's top-1 is 2. Both distributions must be renormalized over
    # the union {0, 2}, not over the full vocabulary and not over an
    # intersection (which would drop one side entirely).
    student = torch.zeros(1, 2, 4)
    teacher = torch.zeros(1, 2, 4)
    student[0, 0] = torch.tensor([2.0, 0.0, 0.0, 0.0])
    teacher[0, 0] = torch.tensor([0.0, 0.0, 3.0, 0.0])
    labels = torch.tensor([[1, 0]])

    p0 = math.exp(2.0) / (math.exp(2.0) + 1.0)
    p2 = 1.0 / (math.exp(2.0) + 1.0)
    q0 = 1.0 / (1.0 + math.exp(3.0))
    q2 = math.exp(3.0) / (1.0 + math.exp(3.0))
    expected = q0 * math.log(q0 / p0) + q2 * math.log(q2 / p2)

    got = _shifted_kd_loss(student, teacher, labels, 1.0, topk=1)
    assert got.item() == pytest.approx(expected, rel=1e-5, abs=1e-6)

    # topk=None must stay on the full-vocabulary path (tail mass matters here).
    full = _shifted_kd_loss(student, teacher, labels, 1.0)
    assert not torch.allclose(got, full, atol=1e-4)


@pytest.mark.parametrize("divergence", DIVERGENCES)
def test_topk_zero_when_student_equals_teacher(divergence):
    student, _, labels = _tensors(vocab=10, seed=7)
    loss = _shifted_kd_loss(
        student, student.clone(), labels, 1.0, divergence=divergence, topk=3
    )
    assert abs(loss.item()) < 1e-6


def test_topk_jsd_remains_symmetric_and_bounded():
    student, teacher, labels = _tensors(vocab=10, seed=8)
    forward = _shifted_kd_loss(student, teacher, labels, 1.0, divergence="jsd", topk=3)
    backward = _shifted_kd_loss(teacher, student, labels, 1.0, divergence="jsd", topk=3)
    assert torch.allclose(forward, backward, atol=1e-7)
    assert 0.0 <= forward.item() <= math.log(2.0) + 1e-6


@pytest.mark.parametrize("divergence", DIVERGENCES)
def test_topk_all_negative_infinity_candidates_stay_finite(divergence):
    # An API teacher that only exposes its top-k often leaves the rest at
    # -inf; a candidate union can then contain no finite teacher logit.
    student = torch.randn(1, 3, 8, generator=torch.Generator().manual_seed(9))
    teacher = torch.full((1, 3, 8), float("-inf"))
    labels = torch.tensor([[1, 2, 3]])
    loss = _shifted_kd_loss(student, teacher, labels, 1.0, divergence=divergence, topk=2)
    assert torch.isfinite(loss)


@pytest.mark.parametrize("divergence", DIVERGENCES)
def test_topk_all_zero_logits_stay_finite(divergence):
    student = torch.zeros(1, 3, 8)
    teacher = torch.zeros(1, 3, 8)
    labels = torch.tensor([[1, 2, 3]])
    loss = _shifted_kd_loss(student, teacher, labels, 1.0, divergence=divergence, topk=2)
    assert torch.isfinite(loss)


def test_topk_equal_to_vocab_matches_full_vocabulary_path():
    student, teacher, labels = _tensors(vocab=8, seed=10)
    full = _shifted_kd_loss(student, teacher, labels, 1.0, divergence="reverse_kl")
    topk_all = _shifted_kd_loss(
        student, teacher, labels, 1.0, divergence="reverse_kl", topk=8
    )
    assert torch.allclose(topk_all, full, atol=1e-6)


@pytest.mark.parametrize("topk", [0, -1, 9, 3.5])
def test_invalid_topk_raises_value_error(topk):
    student, teacher, labels = _tensors(vocab=8, seed=11)
    with pytest.raises(ValueError, match="topk"):
        _shifted_kd_loss(student, teacher, labels, 1.0, topk=topk)


def test_invalid_divergence_raises_value_error():
    student, teacher, labels = _tensors(vocab=8, seed=12)
    with pytest.raises(ValueError, match="divergence"):
        _shifted_kd_loss(student, teacher, labels, 1.0, divergence="kl")


# ---------------------------------------------------------------------------
# Interface errors
# ---------------------------------------------------------------------------


def test_vocab_mismatch_reports_both_vocab_sizes():
    student = torch.randn(1, 3, 10)
    teacher = torch.randn(1, 3, 12)
    labels = torch.tensor([[1, 2, 3]])
    with pytest.raises(ValueError) as excinfo:
        _shifted_kd_loss(student, teacher, labels, 1.0)
    message = str(excinfo.value)
    assert "vocab" in message
    assert "10" in message
    assert "12" in message


@pytest.mark.parametrize("divergence", DIVERGENCES)
def test_topk_loss_backpropagates_finite_gradients(divergence):
    # Training uses this path; masked -inf candidates must not leak NaN into
    # the gradient through 0 * (-inf) arithmetic.
    student = torch.randn(1, 3, 8, generator=torch.Generator().manual_seed(13)).requires_grad_()
    teacher = torch.full((1, 3, 8), float("-inf"))
    teacher[0, 0, 4] = 5.0
    labels = torch.tensor([[1, 2, 3]])
    loss = _shifted_kd_loss(student, teacher, labels, 1.0, divergence=divergence, topk=2)
    assert torch.isfinite(loss)
    loss.backward()
    assert student.grad is not None
    assert torch.isfinite(student.grad).all()


# ---------------------------------------------------------------------------
# ShadowForCausalLM.forward pass-through
# ---------------------------------------------------------------------------


def _make_shadow_for_causal_lm(
    *, distill_weight: float, distill_temperature: float = 1.5, seed: int = 0
):
    from transformers import LlamaConfig, LlamaForCausalLM

    from shadow_peft import ShadowConfig, ShadowForCausalLM, get_shadow_model

    torch.manual_seed(seed)
    config = LlamaConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
    )
    base = LlamaForCausalLM(config)
    shadow_config = ShadowConfig(
        num_shadow_layers=1,
        injection_hidden_size=8,
        gate_hidden_size=10,
        alpha=0.1,
        dropout=0.0,
    )
    peft = get_shadow_model(base, shadow_config)
    return ShadowForCausalLM(
        peft,
        shadow_loss_weight=0.05,
        inference_mode="base_shadow",
        distill_weight=distill_weight,
        distill_temperature=distill_temperature,
    )


def _forward_batch(vocab: int = 128, batch: int = 2, seq: int = 8):
    torch.manual_seed(1234)
    input_ids = torch.randint(0, vocab, (batch, seq))
    labels = input_ids.clone()
    labels[:, :2] = -100
    return input_ids, labels


def test_forward_kd_defaults_are_transparent_and_bitwise_unchanged():
    model = _make_shadow_for_causal_lm(distill_weight=1.0)
    model.eval()
    input_ids, labels = _forward_batch()
    with torch.no_grad():
        teacher_logits = model(input_ids=input_ids).logits
        teacher_logits = teacher_logits + 0.25 * torch.randn_like(teacher_logits)
        implicit = model(input_ids=input_ids, labels=labels, teacher_logits=teacher_logits)
        explicit = model(
            input_ids=input_ids,
            labels=labels,
            teacher_logits=teacher_logits,
            kd_divergence="forward_kl",
            kd_topk=None,
        )
    expected = _shifted_kd_loss(
        implicit.logits, teacher_logits, labels, model.distill_temperature
    )
    assert implicit.kd_loss is not None
    assert torch.equal(implicit.kd_loss, expected)
    assert torch.equal(implicit.kd_loss, explicit.kd_loss)
    assert torch.equal(implicit.loss, explicit.loss)


def test_forward_passes_kd_divergence_and_topk_to_the_loss():
    model = _make_shadow_for_causal_lm(distill_weight=1.0)
    model.eval()
    input_ids, labels = _forward_batch()
    with torch.no_grad():
        teacher_logits = model(input_ids=input_ids).logits
        teacher_logits = teacher_logits + 0.25 * torch.randn_like(teacher_logits)
        out = model(
            input_ids=input_ids,
            labels=labels,
            teacher_logits=teacher_logits,
            kd_divergence="jsd",
            kd_topk=4,
        )
    expected = _shifted_kd_loss(
        out.logits,
        teacher_logits,
        labels,
        model.distill_temperature,
        divergence="jsd",
        topk=4,
    )
    assert out.kd_loss is not None
    assert torch.equal(out.kd_loss, expected)
    assert torch.isfinite(out.loss)


def test_forward_kd_args_are_noop_when_distill_weight_is_zero():
    model = _make_shadow_for_causal_lm(distill_weight=0.0)
    model.eval()
    input_ids, labels = _forward_batch()
    with torch.no_grad():
        plain = model(input_ids=input_ids, labels=labels)
        with_kd_args = model(
            input_ids=input_ids,
            labels=labels,
            teacher_logits=torch.randn_like(plain.logits),
            kd_divergence="reverse_kl",
            kd_topk=2,
        )
    assert with_kd_args.kd_loss is None
    assert torch.equal(plain.loss, with_kd_args.loss)
