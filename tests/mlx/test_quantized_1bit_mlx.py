"""
Tests for 1-bit affine quantized base models (MLX).

Requires an MLX build containing bits=1 affine quantization support
(ml-explore/mlx PR #3161); the whole module skips on released versions.
"""

from __future__ import annotations

import pytest

pytest.importorskip("mlx")
pytest.importorskip("mlx_lm")

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
from mlx.utils import tree_flatten  # noqa: E402

from shadow_peft_mlx import (  # noqa: E402
    ShadowForCausalLM,
    get_shadow_model,
    pack_1bit_refined,
    quantize_model_1bit,
)

GROUP_SIZE = 128


def _has_1bit_support() -> bool:
    try:
        mx.quantize(
            mx.zeros((1, GROUP_SIZE), dtype=mx.float16), group_size=GROUP_SIZE, bits=1
        )
    except ValueError:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _has_1bit_support(),
    reason="installed MLX lacks bits=1 support (see ml-explore/mlx PR #3161)",
)


@pytest.fixture
def llama_128(llama_factory):
    # hidden/intermediate dims divisible by GROUP_SIZE.
    return llama_factory(
        seed=31, hidden_size=128, num_attention_heads=4, num_key_value_heads=2
    )


@pytest.fixture
def quantized_1bit_llama(llama_128):
    manifest = quantize_model_1bit(llama_128, group_size=GROUP_SIZE)
    llama_128.eval()
    return llama_128, manifest


def test_all_layers_quantized_1bit(quantized_1bit_llama):
    base, manifest = quantized_1bit_llama
    layer = base.model.layers[0]
    for proj in (
        layer.self_attn.q_proj,
        layer.self_attn.k_proj,
        layer.self_attn.v_proj,
        layer.self_attn.o_proj,
        layer.mlp.gate_proj,
        layer.mlp.up_proj,
        layer.mlp.down_proj,
    ):
        assert isinstance(proj, nn.QuantizedLinear)
        assert proj.bits == 1
        assert proj.group_size == GROUP_SIZE
    assert isinstance(base.model.embed_tokens, nn.QuantizedEmbedding)
    assert base.model.embed_tokens.bits == 1
    assert manifest["quantized"]
    assert manifest["skipped"] == {}
    assert manifest["quantization"] == {
        "mode": "affine",
        "bits": 1,
        "group_size": GROUP_SIZE,
        "refine_iters": 0,
        "trim_vocab_to": None,
    }


def test_packed_shapes_and_dtypes(quantized_1bit_llama):
    base, _ = quantized_1bit_llama
    q_proj = base.model.layers[0].self_attn.q_proj
    # 128 weights per row -> 4 uint32 words, 1 scale/bias group.
    assert q_proj.weight.shape == (128, 4)
    assert q_proj.weight.dtype == mx.uint32
    assert q_proj.scales.shape == (128, 1)
    assert q_proj.biases.shape == (128, 1)
    embed = base.model.embed_tokens
    assert embed.weight.shape == (64, 4)
    assert embed.scales.shape == (64, 1)


def test_dequantize_hits_group_extremes(llama_128):
    original = llama_128.model.layers[0].self_attn.q_proj.weight * 1
    mx.eval(original)
    quantize_model_1bit(llama_128, group_size=GROUP_SIZE)
    q_proj = llama_128.model.layers[0].self_attn.q_proj
    restored = mx.dequantize(
        q_proj.weight,
        q_proj.scales,
        q_proj.biases,
        group_size=GROUP_SIZE,
        bits=1,
    )
    mx.eval(restored)
    assert restored.shape == original.shape
    lo = mx.min(original, axis=-1, keepdims=True)
    hi = mx.max(original, axis=-1, keepdims=True)
    is_lo = mx.abs(restored - lo) < 1e-3
    is_hi = mx.abs(restored - hi) < 1e-3
    assert mx.all(is_lo | is_hi).item()


def test_skip_patterns(llama_128):
    manifest = quantize_model_1bit(
        llama_128, group_size=GROUP_SIZE, skip_patterns=("q_proj",)
    )
    layer = llama_128.model.layers[0]
    assert type(layer.self_attn.q_proj) is nn.Linear
    assert isinstance(layer.self_attn.k_proj, nn.QuantizedLinear)
    assert manifest["skipped"]
    assert all("skip_patterns" in r for r in manifest["skipped"].values())


def test_shadow_wrap_identity_on_1bit_base(quantized_1bit_llama, shadow_cfg):
    base, _ = quantized_1bit_llama
    ids = mx.array([[1, 5, 7, 9, 3]])
    ref = base(ids)
    peft = get_shadow_model(base, shadow_cfg)
    peft.eval()
    assert mx.abs(peft(ids) - ref).max().item() == 0.0


def test_gradients_only_shadow_params_1bit(quantized_1bit_llama, shadow_cfg):
    base, _ = quantized_1bit_llama
    task = ShadowForCausalLM(get_shadow_model(base, shadow_cfg))
    ids = mx.array([[1, 5, 7, 9, 3]])

    def lf(m, a, b):
        return m(a, labels=b).loss

    _, grads = nn.value_and_grad(task, lf)(task, ids, ids)
    keys = [k for k, _ in tree_flatten(grads)]
    assert keys
    assert all(k.startswith("peft_model.shadow_") for k in keys)


def test_forward_finite_1bit(quantized_1bit_llama):
    base, _ = quantized_1bit_llama
    logits = base(mx.array([[1, 5, 7, 9, 3]]))
    assert mx.isfinite(logits).all().item()


def test_pack_1bit_refined_reduces_mse_monotonically():
    mx.random.seed(7)
    w = mx.random.normal((8, 256)).astype(mx.float16)

    def mse(iters: int) -> float:
        p, s, b = pack_1bit_refined(w, group_size=128, iters=iters)
        deq = mx.dequantize(p, s, b, group_size=128, bits=1)
        return ((deq.astype(mx.float32) - w.astype(mx.float32)) ** 2).mean().item()

    mses = [mse(i) for i in (0, 1, 2, 5)]
    assert mses[1] < mses[0] * 0.5
    assert all(b <= a + 1e-7 for a, b in zip(mses, mses[1:], strict=False))


def test_pack_1bit_refined_format_and_qmm():
    mx.random.seed(8)
    w = mx.random.normal((4, 256)).astype(mx.float16)
    p, s, b = pack_1bit_refined(w, group_size=128, iters=10)
    assert p.dtype == mx.uint32 and p.shape == (4, 8)
    assert s.shape == b.shape == (4, 2) and s.dtype == mx.float16

    x = mx.random.normal((2, 256)).astype(mx.float16)
    y_ref = x @ mx.dequantize(p, s, b, group_size=128, bits=1).T
    y_qmm = mx.quantized_matmul(x, p, s, b, transpose=True, group_size=128, bits=1)
    assert mx.abs(y_ref - y_qmm).max().item() < 1e-1


def test_pack_1bit_refined_constant_group():
    w = mx.full((2, 128), 0.5, dtype=mx.float16)
    p, s, b = pack_1bit_refined(w, group_size=128, iters=10)
    deq = mx.dequantize(p, s, b, group_size=128, bits=1)
    assert mx.isfinite(deq).all().item()
    assert mx.abs(deq - 0.5).max().item() < 1e-3


def test_quantize_model_1bit_refine_integration(llama_128):
    original = llama_128.model.layers[0].self_attn.q_proj.weight * 1
    mx.eval(original)
    manifest = quantize_model_1bit(llama_128, group_size=GROUP_SIZE, refine_iters=5)
    assert manifest["quantization"]["refine_iters"] == 5

    q_proj = llama_128.model.layers[0].self_attn.q_proj
    assert isinstance(q_proj, nn.QuantizedLinear)
    assert q_proj.weight.shape == (128, 4) and q_proj.weight.dtype == mx.uint32

    # Refined packing must beat naive packing in weight space.
    deq = mx.dequantize(
        q_proj.weight, q_proj.scales, q_proj.biases, group_size=GROUP_SIZE, bits=1
    )
    naive_p, naive_s, naive_b = mx.quantize(original, group_size=GROUP_SIZE, bits=1)
    naive_deq = mx.dequantize(naive_p, naive_s, naive_b, group_size=GROUP_SIZE, bits=1)
    ref = original.astype(mx.float32)
    mse_refined = ((deq.astype(mx.float32) - ref) ** 2).mean().item()
    mse_naive = ((naive_deq.astype(mx.float32) - ref) ** 2).mean().item()
    assert mse_refined < mse_naive

    # Forward still works end to end.
    logits = llama_128(mx.array([[1, 5, 7, 9, 3]]))
    assert mx.isfinite(logits).all().item()


def test_trim_vocab_mlx(llama_128):
    # vocab 64 -> trim 48; untied lm_head is trimmed along with the embedding.
    manifest = quantize_model_1bit(llama_128, group_size=GROUP_SIZE, trim_vocab_to=48)

    embed = llama_128.model.embed_tokens
    assert isinstance(embed, nn.QuantizedEmbedding)
    assert embed.weight.shape == (48, 4)  # 48 rows, 128/32 words per row
    assert manifest["trimmed"]["model.embed_tokens"] == [64, 48]
    assert manifest["quantization"]["trim_vocab_to"] == 48

    logits = llama_128(mx.array([[1, 5, 7, 9, 3]]))
    assert logits.shape[-1] == 48
    assert mx.isfinite(logits).all().item()


def test_trim_vocab_validation(llama_128):
    with pytest.raises(ValueError):
        quantize_model_1bit(llama_128, group_size=GROUP_SIZE, trim_vocab_to=0)
