from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from transformers import LlamaConfig, LlamaForCausalLM  # noqa: E402

from shadow_peft import (  # noqa: E402
    ShadowConfig,
    ShadowForCausalLM,
    get_shadow_model,
)


def _tiny_llama(vocab_size: int = 128, num_layers: int = 4) -> LlamaForCausalLM:
    cfg = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=num_layers,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
    )
    return LlamaForCausalLM(cfg)


def _make_shadow_for_causal_lm(
    *,
    distill_weight: float = 0.0,
    distill_temperature: float = 1.0,
    distill_hidden_weight: float = 0.0,
    seed: int = 0,
) -> ShadowForCausalLM:
    torch.manual_seed(seed)
    base = _tiny_llama(num_layers=4)
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
        shadow_loss_weight=0.05,
        inference_mode="base_shadow",
        distill_weight=distill_weight,
        distill_temperature=distill_temperature,
        distill_hidden_weight=distill_hidden_weight,
    )


def _batch(vocab_size: int = 128, batch: int = 2, seq: int = 8):
    torch.manual_seed(1234)
    input_ids = torch.randint(0, vocab_size, (batch, seq))
    labels = input_ids.clone()
    labels[:, :2] = -100  # exercise ignore_index alignment
    return input_ids, labels


def test_default_distill_weight_ignores_teacher_logits():
    m = _make_shadow_for_causal_lm()
    assert m.distill_weight == 0.0
    m.eval()
    input_ids, labels = _batch()
    with torch.no_grad():
        out_no_teacher = m(input_ids=input_ids, labels=labels)
        out_with_teacher = m(
            input_ids=input_ids,
            labels=labels,
            teacher_logits=torch.randn_like(out_no_teacher.logits),
        )
    assert out_no_teacher.kd_loss is None
    assert out_with_teacher.kd_loss is None
    assert torch.allclose(out_no_teacher.loss, out_with_teacher.loss, atol=0.0, rtol=0.0)


def test_distill_loss_finite_and_grads_only_in_shadow():
    m = _make_shadow_for_causal_lm(distill_weight=1.0, distill_temperature=2.0)
    m.train()
    input_ids, labels = _batch()
    with torch.no_grad():
        student_logits = m(input_ids=input_ids).logits
    teacher_logits = student_logits + torch.randn_like(student_logits) * 0.5

    out = m(input_ids=input_ids, labels=labels, teacher_logits=teacher_logits)
    assert out.kd_loss is not None
    assert torch.isfinite(out.kd_loss)
    assert torch.isfinite(out.loss)
    # loss = (ce + shadow_loss_weight * shadow_ce) + distill_weight * kd.
    with torch.no_grad():
        out_no_kd = m(input_ids=input_ids, labels=labels)
    expected = out_no_kd.loss.detach() + m.distill_weight * out.kd_loss.detach()
    assert torch.allclose(out.loss.detach(), expected, atol=1e-5, rtol=1e-5)

    out.loss.backward()
    for name, p in m.named_parameters():
        if not p.requires_grad:
            assert p.grad is None, f"frozen param got grad: {name}"
    # Sanity: the grads actually live in the shadow adapter modules.
    shadow_prefixes = ("peft_model.shadow_model", "peft_model.shadow_injection_model", "peft_model.shadow_update_model")
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for name, p in m.named_parameters()
        if name.startswith(shadow_prefixes)
    )


def test_kd_loss_zero_when_teacher_equals_student():
    m = _make_shadow_for_causal_lm(distill_weight=1.0)
    m.eval()
    input_ids, labels = _batch()
    with torch.no_grad():
        student_logits = m(input_ids=input_ids).logits
        out = m(input_ids=input_ids, labels=labels, teacher_logits=student_logits.clone())
    assert out.kd_loss is not None
    assert abs(out.kd_loss.item()) < 1e-6


def test_vocab_mismatch_raises():
    m = _make_shadow_for_causal_lm(distill_weight=1.0)
    input_ids, labels = _batch()
    batch, seq = input_ids.shape
    bad_teacher = torch.randn(batch, seq, 128 + 7)
    with pytest.raises(ValueError, match="vocab"):
        m(input_ids=input_ids, labels=labels, teacher_logits=bad_teacher)


def test_run_shadow_peft_args_defaults(monkeypatch):
    """The new CLI flags must not break parse_args() defaults (skipped if script deps missing)."""
    experiment_dir = Path(__file__).resolve().parents[1] / "experiment"
    sys.path.insert(0, str(experiment_dir))
    try:
        try:
            import run_shadow_peft
        except ImportError as exc:
            pytest.skip(f"run_shadow_peft.py dependencies not available: {exc}")
        monkeypatch.setattr(sys, "argv", ["run_shadow_peft.py"])
        args = run_shadow_peft.parse_args()
        assert args.quantize_base_group_size is None
        assert args.distill_from is None
        assert args.distill_weight == 1.0
        assert args.distill_temperature == 1.0

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "run_shadow_peft.py",
                "--quantize_base_group_size",
                "128",
                "--distill_from",
                "Qwen/Qwen3-0.6B",
                "--distill_weight",
                "0.5",
                "--distill_temperature",
                "2.0",
            ],
        )
        args = run_shadow_peft.parse_args()
        assert args.quantize_base_group_size == 128
        assert args.distill_from == "Qwen/Qwen3-0.6B"
        assert args.distill_weight == 0.5
        assert args.distill_temperature == 2.0
    finally:
        sys.path.remove(str(experiment_dir))


def test_hidden_distill_off_by_default():
    m = _make_shadow_for_causal_lm()
    assert m.distill_hidden_weight == 0.0
    m.eval()
    input_ids, labels = _batch()
    with torch.no_grad():
        out_ref = m(input_ids=input_ids, labels=labels)
        # Even with teacher hidden states passed, weight=0 means no-op.
        out = m(
            input_ids=input_ids,
            labels=labels,
            teacher_hidden_states=tuple(
                torch.randn(2, 8, 32) for _ in range(5)
            ),
        )
    assert out.hidden_loss is None
    assert torch.allclose(out_ref.loss, out.loss, atol=0.0, rtol=0.0)


def test_hidden_distill_zero_when_identical():
    m = _make_shadow_for_causal_lm(distill_hidden_weight=1.0)
    m.eval()
    input_ids, labels = _batch()
    with torch.no_grad():
        ref = m(input_ids=input_ids, output_hidden_states=True)
        assert ref.hidden_states is not None
        out = m(
            input_ids=input_ids,
            labels=labels,
            teacher_hidden_states=tuple(h.clone() for h in ref.hidden_states),
        )
    assert out.hidden_loss is not None
    assert abs(out.hidden_loss.item()) < 1e-5


def test_hidden_distill_finite_and_loss_composition():
    m = _make_shadow_for_causal_lm(distill_hidden_weight=2.0)
    m.train()
    input_ids, labels = _batch()
    with torch.no_grad():
        ref = m(input_ids=input_ids, output_hidden_states=True)
    teacher_hs = tuple(h + 0.1 * torch.randn_like(h) for h in ref.hidden_states)

    out = m(input_ids=input_ids, labels=labels, teacher_hidden_states=teacher_hs)
    assert out.hidden_loss is not None
    assert torch.isfinite(out.hidden_loss)
    with torch.no_grad():
        out_no_hidden = m(input_ids=input_ids, labels=labels)
    expected = out_no_hidden.loss.detach() + m.distill_hidden_weight * out.hidden_loss.detach()
    assert torch.allclose(out.loss.detach(), expected, atol=1e-5, rtol=1e-5)

    out.loss.backward()
    for name, p in m.named_parameters():
        if not p.requires_grad:
            assert p.grad is None, f"frozen param got grad: {name}"
    shadow_prefixes = (
        "peft_model.shadow_model",
        "peft_model.shadow_injection_model",
        "peft_model.shadow_update_model",
    )
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for name, p in m.named_parameters()
        if name.startswith(shadow_prefixes)
    )


def test_hidden_layer_count_mismatch_raises():
    m = _make_shadow_for_causal_lm(distill_hidden_weight=1.0)
    input_ids, labels = _batch()
    bad_teacher = (torch.randn(2, 8, 32), torch.randn(2, 8, 32))  # 2 != 5 layers
    with pytest.raises(ValueError, match="layer count"):
        m(input_ids=input_ids, labels=labels, teacher_hidden_states=bad_teacher)
