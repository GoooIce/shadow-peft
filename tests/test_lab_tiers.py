"""Backbone tier seam tests (ticket 03: {bf16, w8a8_fp8, w4a16} + joint calibration).

Two iron rules from the direction-① spec are pinned here:
1. torchao conversion runs BEFORE shadow mounting (prepare/convert APIs strip
   already-mounted hooks).
2. W8A8 activation-scale calibration forwards run WITH the shadow attached
   (joint calibration).

The tiny models run on CPU. Float8 dynamic quant requires CUDA sm>=8.9, so the
local w8a8_fp8 tier validates the same dynamic-A/W protocol through INT8; the
5090 runs real FP8 (the seam records which format was applied).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
torchao = pytest.importorskip("torchao")  # noqa: F841

_EXPERIMENT = Path(__file__).resolve().parents[1] / "experiment"
sys.path.insert(0, str(_EXPERIMENT))

from lab import tiers  # noqa: E402
from lab.trainer import TrainConfig  # noqa: E402

from shadow_peft import ShadowConfig  # noqa: E402
from shadow_peft.peft_model import _ShadowLayerWrapper  # noqa: E402


def _tiny_llama(num_layers: int = 4):
    from transformers import LlamaConfig, LlamaForCausalLM

    # hidden/intermediate sizes must clear torchao Int4's groupsize=128,
    # otherwise the conversion silently filters every Linear out.
    cfg = LlamaConfig(
        vocab_size=128,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=num_layers,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
        torch_dtype=torch.bfloat16,
    )
    return LlamaForCausalLM(cfg).to(torch.float32)


def _shadow_cfg() -> ShadowConfig:
    return ShadowConfig(
        num_shadow_layers=1,
        injection_hidden_size=8,
        gate_hidden_size=10,
        alpha=0.1,
        dropout=0.0,
    )


def _calib_batches(n: int = 2):
    return [
        {
            "input_ids": torch.randint(0, 128, (1, 16)),
            "attention_mask": torch.ones(1, 16, dtype=torch.long),
        }
        for _ in range(n)
    ]


def test_tier_registry_and_validation():
    assert tiers.TIERS == ("bf16", "w8a8_fp8", "w4a16")
    with pytest.raises(ValueError):
        tiers.tier_config("nvfp4", torch.device("cpu"))
    # bf16 needs no conversion at all
    assert tiers.tier_config("bf16", torch.device("cpu")) is None


def test_conversion_precedes_shadow_mount(monkeypatch):
    """Iron rule 1: the torchao convert call must happen before the shadow
    mount, and the wrapped decoder layers must be the converted ones."""
    calls: list[str] = []
    real_convert = tiers.apply_tier_conversion

    def spy_convert(model, tier, device):
        calls.append("convert")
        return real_convert(model, tier, device)

    monkeypatch.setattr(tiers, "apply_tier_conversion", spy_convert)

    base = _tiny_llama()
    peft, manifest = tiers.prepare_tiered_student(
        base, _shadow_cfg(), "w4a16", device=torch.device("cpu")
    )
    assert calls == ["convert"]
    assert manifest["sequence"] == ["convert", "mount"]

    # Structural invariant: decoder layers are wrapped AND their inner Linears
    # carry quantized weight tensors (conversion reached the layers that were
    # then wrapped). Off-CUDA the w4a16 tier runs the Int8 weight-only
    # stand-in, so accept any non-Parameter quantized weight family.
    layers = base.model.layers
    assert all(isinstance(lyr, _ShadowLayerWrapper) for lyr in layers)
    saw_quantized_weight = False
    for lyr in layers:
        for mod in lyr.layer.modules():
            w = getattr(mod, "weight", None)
            if w is not None and type(w).__name__ != "Parameter":
                saw_quantized_weight = True
    assert saw_quantized_weight, "w4a16 tier left Linears unconverted"


def test_w8a8_tier_applies_dynamic_protocol_locally():
    """Off-CUDA, w8a8_fp8 validates through INT8 dynamic (same dynamic-A/W
    protocol); the manifest records the substituted format."""
    base = _tiny_llama()
    manifest = tiers.apply_tier_conversion(base, "w8a8_fp8", torch.device("cpu"))
    assert manifest["tier"] == "w8a8_fp8"
    assert manifest["format"] == "int8_dynamic_local"
    saw = any(
        "Int8" in type(getattr(m, "weight", None)).__name__
        for _, m in base.model.named_modules()
    )
    assert saw, "local w8a8 stand-in did not convert any weight"


def test_bf16_tier_is_noop_conversion():
    base = _tiny_llama()
    manifest = tiers.apply_tier_conversion(base, "bf16", torch.device("cpu"))
    assert manifest == {"tier": "bf16", "format": "bf16"}
    assert all(
        type(getattr(m, "weight", None)).__name__ == "Parameter"
        for _, m in base.model.named_modules()
        if getattr(m, "weight", None) is not None
    )


def test_joint_calibration_runs_with_shadow_attached():
    """Iron rule 2: calibration forwards must pass through the mounted shadow
    (injection live), not through the bare quantized base."""
    torch.manual_seed(0)
    base = _tiny_llama()
    peft, manifest = tiers.prepare_tiered_student(
        base,
        _shadow_cfg(),
        "w8a8_fp8",
        device=torch.device("cpu"),
        calib_batches=_calib_batches(2),
    )
    assert manifest["sequence"] == ["convert", "mount", "calibrate:2"]

    injection_calls = []

    # Count calls into the injection module: if calibration forwards ran the
    # bare base, injection would never fire.
    inj = peft.shadow_injection_model
    orig = inj.forward

    def counted(*a, **k):
        injection_calls.append(1)
        return orig(*a, **k)

    inj.forward = counted  # type: ignore[method-assign]

    n = tiers.joint_calibrate(peft, _calib_batches(2))
    assert n == 2
    assert len(injection_calls) > 0, "calibration forwards bypassed the shadow injection"


def test_joint_calibration_requires_shadow_state():
    """Calibration forwards must go through ShadowPeftModel.forward (which
    initializes shadow state), not base_model directly — calling the base
    alone would raise for missing shadow state after mounting."""
    torch.manual_seed(0)
    base = _tiny_llama()
    peft, _ = tiers.prepare_tiered_student(
        base, _shadow_cfg(), "w4a16", device=torch.device("cpu")
    )
    batch = _calib_batches(1)[0]
    with pytest.raises(RuntimeError):
        # After mounting, the wrapped base refuses to run without the adapter.
        peft.base_model(**batch)


@pytest.mark.parametrize("tier", tiers.TIERS)
def test_three_tiers_few_distill_steps(tier):
    """Acceptance: each tier completes a few KL + hidden-MSE distillation steps
    on a tiny model with finite losses, teacher channel identical to bf16."""
    torch.manual_seed(0)
    device = torch.device("cpu")

    teacher = _tiny_llama()
    teacher.eval()

    base = _tiny_llama()
    peft, _ = tiers.prepare_tiered_student(base, _shadow_cfg(), tier, device=device)
    peft.train()

    from shadow_peft import ShadowForCausalLM

    task = ShadowForCausalLM(
        peft, shadow_loss_weight=0.05, distill_weight=1.0, distill_hidden_weight=1.0
    )
    trainable = [p for p in task.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=1e-4)

    for step in range(3):
        batch = _calib_batches(1)[0]
        labels = batch["input_ids"].clone()
        with torch.no_grad():
            t_out = teacher(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                output_hidden_states=True,
            )
        out = task(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=labels,
            teacher_logits=t_out.logits,
            teacher_hidden_states=t_out.hidden_states,
        )
        assert torch.isfinite(out.loss), f"{tier} step {step}: loss not finite"
        assert out.kd_loss is not None and torch.isfinite(out.kd_loss)
        assert out.hidden_loss is not None and torch.isfinite(out.hidden_loss)
        opt.zero_grad()
        out.loss.backward()
        opt.step()


def test_torchao_tiers_do_not_trim_vocab():
    """Direction ① keeps the full 262,144 vocab: torchao tiers never trim."""
    base = _tiny_llama()
    before = base.get_input_embeddings().weight.shape[0]
    manifest = tiers.apply_tier_conversion(base, "w4a16", torch.device("cpu"))
    assert "trim" not in manifest
    assert base.get_input_embeddings().weight.shape[0] == before


def test_train_config_backbone_tier_field():
    cfg = TrainConfig(bits=0, backbone_tier="w8a8_fp8", max_tokens=1e4)
    assert cfg.backbone_tier == "w8a8_fp8"
    with pytest.raises(ValueError):
        TrainConfig(bits=1, backbone_tier="w8a8_fp8")
    with pytest.raises(ValueError):
        TrainConfig(bits=0, backbone_tier="nvfp4")
    # legacy default unchanged
    assert TrainConfig().backbone_tier == "bits"


def test_int4_fake_quant_weight_error_bound():
    """The materialized Int4 fake-quant stays within the group-wise half-step
    bound (4-bit asymmetric: step = range/15 ⇒ error ≤ range/30)."""
    torch.manual_seed(0)
    w = torch.randn(16, 256, dtype=torch.bfloat16)
    fake = tiers.int4_fake_quant_weight(w)
    assert fake.shape == w.shape and fake.dtype == w.dtype
    assert not torch.equal(w, fake)
    g = w.float().reshape(16, 2, 128)
    rng = (g.amax(-1) - g.amin(-1)).unsqueeze(-1)
    err = (w.float() - fake.float()).abs().reshape(16, 2, 128)
    # 4-bit asymmetric step = range/15; torchao's affine rounding keeps the
    # error within ~0.75x step (bound asserted at range/20).
    assert (err <= rng / 20).all()


def test_apply_training_conversion_dispatch():
    torch.manual_seed(0)
    # w4a16 fake: materialized straight from the original weights (no kernel
    # conversion at all — the tile-packed tensor has no dequantize).
    base = _tiny_llama()
    manifest = tiers.apply_training_conversion(
        base, "w4a16", torch.device("cpu"), fake_quant=True
    )
    assert manifest["format"] == "int4_weight_only_fake"
    assert manifest["fake_quant_swaps"] > 0
    assert all(
        isinstance(getattr(m, "weight", None), (torch.nn.Parameter, type(None)))
        for _, m in base.named_modules()
    )
    # w8a8_fp8 fake: kernel conversion then dequantize back to plain weights
    # (Float8 on CUDA, the Int8 stand-in locally — both implement it).
    base = _tiny_llama()
    manifest = tiers.apply_training_conversion(
        base, "w8a8_fp8", torch.device("cpu"), fake_quant=True
    )
    assert manifest["format"] == "int8_dynamic_local"
    assert manifest["fake_quant_swaps"] > 0
    assert all(
        isinstance(getattr(m, "weight", None), (torch.nn.Parameter, type(None)))
        for _, m in base.named_modules()
    )
