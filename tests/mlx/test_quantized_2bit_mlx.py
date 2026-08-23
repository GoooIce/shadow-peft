"""
Tests for ternary 2-bit affine quantized base models (MLX).

``bits=2`` is natively supported by MLX, so unlike the 1-bit module this suite
always runs. It mirrors the torch 2-bit pins through the MLX-format codec
(``pack_2bit_ternary`` + ``mx.dequantize``).
"""

from __future__ import annotations

import pytest

pytest.importorskip("mlx")
pytest.importorskip("mlx_lm")

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402

from shadow_peft_mlx import pack_2bit_ternary, quantize_model_2bit  # noqa: E402

GROUP_SIZE = 128


@pytest.fixture
def llama_128(llama_factory):
    # hidden/intermediate dims divisible by GROUP_SIZE.
    return llama_factory(
        seed=41, hidden_size=128, num_attention_heads=4, num_key_value_heads=2
    )


def test_pack_2bit_shapes_dtypes_and_bias():
    mx.random.seed(0)
    w = mx.random.normal((8, 256))
    packed, scales, biases = pack_2bit_ternary(w, group_size=GROUP_SIZE)
    assert packed.shape == (8, 16)  # 256 / 16 codes per word
    assert packed.dtype == mx.uint32
    assert scales.shape == (8, 2)
    assert biases.shape == (8, 2)
    assert scales.dtype == mx.float16
    # Ternary affine grid: bias must be -scale (dequant = scale * (code - 1)).
    assert mx.allclose(biases.astype(mx.float32), -scales.astype(mx.float32), atol=1e-3)


def test_dequantized_values_are_ternary():
    mx.random.seed(0)
    w = mx.random.normal((4, 128))
    packed, scales, biases = pack_2bit_ternary(w, group_size=GROUP_SIZE)
    w_hat = mx.dequantize(packed, scales, biases, group_size=GROUP_SIZE, bits=2)
    for g in range(1):
        sg = scales[0, g].astype(mx.float32).item()
        vals = w_hat[0, g * GROUP_SIZE : (g + 1) * GROUP_SIZE]
        uniq = set(vals.astype(mx.float32).flatten().tolist())
        assert len(uniq) <= 3
        for v in uniq:
            assert any(abs(v - t) < 1e-2 for t in (-sg, 0.0, sg))


def test_pack_2bit_rejects_bad_inputs():
    w = mx.random.normal((4, 128))
    with pytest.raises(ValueError):
        pack_2bit_ternary(w, group_size=GROUP_SIZE, iters=-1)
    with pytest.raises(ValueError):
        pack_2bit_ternary(w, group_size=GROUP_SIZE, thr_ratio=0.0)
    with pytest.raises(ValueError):
        pack_2bit_ternary(mx.random.normal((4, 100)), group_size=GROUP_SIZE)


def test_refine_reduces_mse():
    mx.random.seed(0)
    w = mx.random.normal((4, 256))
    mses = []
    for iters in (0, 3, 8):
        packed, scales, biases = pack_2bit_ternary(w, group_size=GROUP_SIZE, iters=iters)
        w_hat = mx.dequantize(packed, scales, biases, group_size=GROUP_SIZE, bits=2)
        mses.append(((w - w_hat.astype(mx.float32)) ** 2).mean().item())
    assert mses[1] < mses[0] * 0.95
    assert mses[2] <= mses[1] + 1e-7


def test_quantize_model_2bit_manifest_and_forward(llama_128):
    manifest = quantize_model_2bit(llama_128, group_size=GROUP_SIZE, refine_iters=2)
    q = manifest["quantization"]
    assert q["bits"] == 2
    assert q["ternary"] is True
    assert q["thr_ratio"] == 0.5
    assert q["refine_iters"] == 2
    assert len(manifest["quantized"]) > 0

    llama_128.eval()
    ids = mx.array([[1, 5, 7, 9]])
    out = llama_128(ids)
    assert out.shape == (1, 4, llama_128.args.vocab_size)
    assert bool(mx.isfinite(out).all().item())


def test_quantize_model_2bit_skip_patterns(llama_128):
    manifest = quantize_model_2bit(
        llama_128, group_size=GROUP_SIZE, skip_patterns=("embed",)
    )
    assert any("embed" in p for p in manifest["skipped"])
    # Embedding untouched: still an nn.Embedding, not quantized.
    assert isinstance(llama_128.model.embed_tokens, nn.Embedding)


def test_quantize_model_2bit_trims_vocab(llama_factory):
    model = llama_factory(seed=7, hidden_size=128)
    full_vocab = model.args.vocab_size
    trim_to = full_vocab - 8
    manifest = quantize_model_2bit(model, group_size=GROUP_SIZE, trim_vocab_to=trim_to)
    assert manifest["trimmed"]["model.embed_tokens"] == [full_vocab, trim_to]
    assert model.args.vocab_size == trim_to
