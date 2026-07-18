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
