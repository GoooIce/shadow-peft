"""Tests for the Bonsai-style ternary g128 surrogate and MLX affine layout."""

from __future__ import annotations

import weakref

import pytest

torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402

import shadow_peft.quantization.affine_2bit as affine_2bit  # noqa: E402
from shadow_peft import (  # noqa: E402
    QuantizedEmbedding2Bit,
    QuantizedLinear2Bit,
    dequantize_2bit_ternary,
    quantize_2bit_bonsai,
    quantize_model_2bit,
)


def _unpack_codes(packed: torch.Tensor) -> torch.Tensor:
    shifts = 2 * torch.arange(16, dtype=torch.int32, device=packed.device)
    return ((packed.unsqueeze(-1) >> shifts) & 0b11).reshape(packed.shape[0], -1)


def _report_reference(
    weight: torch.Tensor,
    group_size: int = 128,
    refine_iters: int = 8,
    thr_ratio: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Small, deliberately scalar packer following deep-research-report.md."""
    if weight.dtype not in (torch.float16, torch.bfloat16):
        weight = weight.to(torch.float16)
    dtype = weight.dtype
    rows, cols = weight.shape
    groups = weight.float().reshape(rows, cols // group_size, group_size)
    scales = groups.abs().mean(dim=-1).clamp_min(1e-8)

    def assign() -> torch.Tensor:
        threshold = thr_ratio * scales.unsqueeze(-1)
        return torch.where(
            groups > threshold,
            torch.ones_like(groups),
            torch.where(groups < -threshold, -torch.ones_like(groups), 0.0),
        )

    ternary = assign()
    for iteration in range(refine_iters):
        denom = (ternary * ternary).sum(dim=-1).clamp_min(1.0)
        scales = ((groups * ternary).sum(dim=-1) / denom).abs().clamp_min(1e-8)
        if iteration + 1 < refine_iters:
            ternary = assign()

    codes = (ternary + 1).to(torch.int64).reshape(rows, cols)
    words: list[list[int]] = []
    for row in codes.tolist():
        packed_row = []
        for start in range(0, cols, 16):
            word = 0
            for offset, code in enumerate(row[start : start + 16]):
                word |= code << (2 * offset)
            # Preserve the uint32 bit pattern in torch's signed int32 storage.
            packed_row.append(word if word < 2**31 else word - 2**32)
        words.append(packed_row)

    stored_scales = scales.to(dtype)
    stored_scales = torch.where(
        stored_scales == 0,
        torch.full_like(stored_scales, torch.finfo(dtype).tiny),
        stored_scales,
    )
    return torch.tensor(words, dtype=torch.int32), stored_scales, -stored_scales


def test_bonsai_defaults_match_report_reference_and_lsb_packing():
    torch.manual_seed(7)
    weight = torch.randn(3, 256, dtype=torch.float16)

    actual = quantize_2bit_bonsai(weight)
    expected = _report_reference(weight)

    for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
        assert torch.equal(actual_tensor, expected_tensor)

    codes = _unpack_codes(actual[0])
    assert codes.min().item() >= 0
    assert codes.max().item() <= 2


def test_bonsai_row_chunking_is_bit_exact(monkeypatch):
    torch.manual_seed(17)
    weight = torch.randn(7, 64, dtype=torch.float16)
    expected = _report_reference(weight, group_size=32)

    # Force four row chunks (2 + 2 + 2 + 1) in a tiny fixture.
    monkeypatch.setattr(affine_2bit, "_BONSAI_MAX_CHUNK_ELEMENTS", 128)
    actual = quantize_2bit_bonsai(weight, group_size=32)

    for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
        assert torch.equal(actual_tensor, expected_tensor)


def test_bonsai_strict_threshold_keeps_equal_boundaries_zero():
    # absmean is exactly 1, hence the +/-0.5 entries sit exactly on the
    # default threshold.  refine_iters=0 isolates the assignment boundary.
    positive = torch.tensor([0.5, 1.5, *([1.0] * 14)], dtype=torch.float16)
    negative = -positive
    packed, _, _ = quantize_2bit_bonsai(
        torch.stack((positive, negative)), group_size=16, refine_iters=0
    )
    codes = _unpack_codes(packed)

    assert codes[0, 0].item() == 1  # +0.5 -> ternary 0
    assert codes[1, 0].item() == 1  # -0.5 -> ternary 0
    assert codes[0, 1].item() == 2  # above +threshold
    assert codes[1, 1].item() == 0  # below -threshold


def test_bonsai_zero_group_is_finite_positive_and_reconstructs_zero():
    weight = torch.zeros(2, 128, dtype=torch.float16)
    packed, scales, biases = quantize_2bit_bonsai(weight)
    restored = dequantize_2bit_ternary(packed, scales, biases, group_size=128)

    assert torch.isfinite(scales).all()
    assert (scales > 0).all()
    assert torch.equal(biases, -scales)
    assert torch.equal(restored, weight)
    assert (_unpack_codes(packed) == 1).all()


def test_bonsai_preserves_representable_fp16_subnormal_scales():
    value = torch.tensor(3e-5, dtype=torch.float16).item()
    weight = torch.full((1, 128), value, dtype=torch.float16)
    packed, scales, biases = quantize_2bit_bonsai(weight)
    restored = dequantize_2bit_ternary(packed, scales, biases)

    assert 0 < scales.item() < torch.finfo(torch.float16).tiny
    assert torch.equal(restored, weight)


def test_bonsai_least_squares_refinement_does_not_increase_mse():
    torch.manual_seed(11)
    weight = torch.randn(4, 256, dtype=torch.float16)
    errors = []
    for refine_iters in range(9):
        packed, scales, biases = quantize_2bit_bonsai(
            weight, refine_iters=refine_iters
        )
        restored = dequantize_2bit_ternary(packed, scales, biases)
        errors.append((restored.float() - weight.float()).square().mean().item())

    assert all(
        after <= before + 1e-7
        for before, after in zip(errors, errors[1:], strict=False)
    )


def test_quantize_model_bonsai_uses_method_specific_defaults():
    torch.manual_seed(13)
    model = nn.Sequential(nn.Linear(32, 16, bias=False)).half()
    original = model[0].weight.detach().clone()

    manifest = quantize_model_2bit(model, group_size=32, method="bonsai")

    assert isinstance(model[0], QuantizedLinear2Bit)
    assert manifest["quantization"]["method"] == "bonsai"
    assert manifest["quantization"]["refine_iters"] == 8
    expected = quantize_2bit_bonsai(original, group_size=32)
    assert torch.equal(model[0].weight, expected[0])
    assert torch.equal(model[0].scales, expected[1])
    assert torch.equal(model[0].biases, expected[2])


def test_quantize_model_default_remains_legacy_affine():
    model = nn.Sequential(nn.Linear(32, 16, bias=False)).half()
    manifest = quantize_model_2bit(model, group_size=32)

    assert manifest["quantization"]["method"] == "affine"
    assert manifest["quantization"]["refine_iters"] == 0
    assert manifest["quantization"]["quantize_tied_weights"] is False


def test_quantized_bonsai_float32_model_accepts_float32_activations():
    torch.manual_seed(17)
    model = nn.Sequential(nn.Linear(128, 32, bias=True)).eval()
    quantize_model_2bit(model, method="bonsai")
    inputs = torch.randn(2, 128)

    output = model(inputs)

    assert output.dtype == torch.float32
    assert torch.isfinite(output).all()


def test_quantized_linear_load_state_dict_invalidates_eval_cache():
    torch.manual_seed(19)
    module = QuantizedLinear2Bit.from_linear(nn.Linear(128, 32).half()).eval()
    inputs = torch.randn(2, 128, dtype=torch.float16)
    before = module(inputs)
    assert module._cached_weight is not None

    state = {name: tensor.clone() for name, tensor in module.state_dict().items()}
    state["scales"].mul_(2)
    state["biases"].mul_(2)
    module.load_state_dict(state)

    assert module._cached_weight is None
    after = module(inputs)
    assert not torch.equal(before, after)


def test_quantize_model_bonsai_quantizes_tied_head_and_preserves_eval_mode():
    class TiedModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.Embedding(64, 32)
            self.lm_head = nn.Linear(32, 64, bias=False)
            self.lm_head.weight = self.embed_tokens.weight

    model = TiedModel().half().eval()
    manifest = quantize_model_2bit(model, group_size=32, method="bonsai")

    assert isinstance(model.embed_tokens, QuantizedEmbedding2Bit)
    assert isinstance(model.lm_head, QuantizedLinear2Bit)
    assert not model.embed_tokens.training
    assert not model.lm_head.training
    assert manifest["tied"] == []
    assert manifest["quantized_tied"] == ["lm_head"]
    assert manifest["quantization"]["quantize_tied_weights"] is True
    assert torch.equal(model.embed_tokens.weight, model.lm_head.weight)
    assert torch.equal(model.embed_tokens.scales, model.lm_head.scales)


def test_quantize_model_releases_replaced_untied_weights(monkeypatch):
    class TwoLinears(nn.Module):
        def __init__(self):
            super().__init__()
            self.first = nn.Linear(32, 32, bias=False)
            self.second = nn.Linear(32, 32, bias=False)

    original = affine_2bit.quantize_2bit_bonsai
    previous_weight: weakref.ReferenceType[torch.Tensor] | None = None

    def tracking_quantizer(weight, *args, **kwargs):
        nonlocal previous_weight
        if previous_weight is not None:
            assert previous_weight() is None
        previous_weight = weakref.ref(weight)
        return original(weight, *args, **kwargs)

    monkeypatch.setattr(affine_2bit, "quantize_2bit_bonsai", tracking_quantizer)
    model = TwoLinears().half()
    quantize_model_2bit(model, group_size=32, method="bonsai")

    assert isinstance(model.first, QuantizedLinear2Bit)
    assert isinstance(model.second, QuantizedLinear2Bit)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"group_size": 0}, "group_size"),
        ({"group_size": 24}, "group_size"),
        ({"thr_ratio": 0.0}, "thr_ratio"),
        ({"method": "official"}, "method"),
        ({"refine_iters": -1}, "refine_iters"),
    ],
)
def test_quantize_model_validates_options_without_quantizable_layers(kwargs, message):
    with pytest.raises(ValueError, match=message):
        quantize_model_2bit(nn.Identity(), **kwargs)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"refine_iters": -1}, "refine_iters"),
        ({"thr_ratio": 0.0}, "thr_ratio"),
        ({"thr_ratio": 1.1}, "thr_ratio"),
    ],
)
def test_bonsai_rejects_invalid_solver_parameters(kwargs, message):
    with pytest.raises(ValueError, match=message):
        quantize_2bit_bonsai(torch.randn(1, 128), **kwargs)
