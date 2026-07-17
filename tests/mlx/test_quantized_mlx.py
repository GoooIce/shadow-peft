"""
Tests for ShadowPEFT (MLX) on a quantized base model (QLoRA-style).

Verifies that the Shadow adapter works end-to-end when the base model's Linear and
Embedding layers are 4-bit quantized: identity wrap, cached decode, gradient
isolation, training, save/load, and export.
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
    ShadowPeftModel,
    get_shadow_model,
    train,
)


@pytest.fixture
def quantized_llama(llama_factory):
    base = llama_factory(seed=30)
    nn.quantize(base, group_size=32, bits=4)
    base.eval()
    return base


def test_quantized_layers_in_place(quantized_llama):
    layer = quantized_llama.model.layers[0]
    assert isinstance(layer.self_attn.q_proj, nn.QuantizedLinear)
    assert isinstance(quantized_llama.model.embed_tokens, nn.QuantizedEmbedding)


def test_identity_after_wrap_quantized(quantized_llama, shadow_cfg):
    ids = mx.array([[1, 5, 7, 9, 3]])
    ref = quantized_llama(ids)
    peft = get_shadow_model(quantized_llama, shadow_cfg)
    peft.eval()
    assert mx.abs(peft(ids) - ref).max().item() == 0.0


def test_cached_decode_quantized(quantized_llama, shadow_cfg):
    peft = get_shadow_model(quantized_llama, shadow_cfg)
    peft.eval()
    prefix = mx.array([[1, 5, 7, 9]])
    nxt = mx.array([[11]])

    ref_full = peft(mx.concatenate([prefix, nxt], axis=1))
    caches = peft.make_cache()
    peft(prefix, cache=caches)
    step = peft(nxt, cache=caches)
    assert mx.abs(step[:, -1] - ref_full[:, -1]).max().item() < 1e-5


def test_gradients_only_shadow_params_quantized(quantized_llama, shadow_cfg):
    task = ShadowForCausalLM(get_shadow_model(quantized_llama, shadow_cfg))
    ids = mx.array([[1, 5, 7, 9, 3]])

    def lf(m, a, b):
        return m(a, labels=b).loss

    _, grads = nn.value_and_grad(task, lf)(task, ids, ids)
    keys = [k for k, _ in tree_flatten(grads)]
    assert keys
    assert all(k.startswith("peft_model.shadow_") for k in keys)


def test_training_leaves_quantized_base_untouched(quantized_llama, shadow_cfg):
    base = quantized_llama
    packed_before = base.model.layers[0].self_attn.q_proj.weight * 1
    task = ShadowForCausalLM(get_shadow_model(base, shadow_cfg))
    ids = mx.array([[1, 5, 7, 9, 3]])

    history = train(task, [(ids, ids)], lr=1e-2, epochs=4, log_every=0)
    assert history[-1][1] < history[0][1]
    packed_after = base.model.layers[0].self_attn.q_proj.weight
    assert mx.abs(packed_after - packed_before).max().item() == 0.0


def test_save_load_roundtrip_quantized(tmp_path, quantized_llama, shadow_cfg):
    base = quantized_llama
    peft = get_shadow_model(base, shadow_cfg)
    task = ShadowForCausalLM(peft)
    ids = mx.array([[1, 5, 7, 9, 3]])
    train(task, [(ids, ids)], lr=1e-2, epochs=2, log_every=0)
    task.eval()

    ref = peft(ids)
    peft.save_pretrained(tmp_path)
    reloaded = ShadowPeftModel.from_pretrained(base, tmp_path)
    reloaded.eval()
    assert mx.abs(reloaded(ids) - ref).max().item() == 0.0


def test_export_shadow_quantized_base(quantized_llama, shadow_cfg):
    peft = get_shadow_model(quantized_llama, shadow_cfg)
    peft.eval()
    exported = peft.export_shadow()
    exported.eval()

    ids = mx.array([[1, 5, 7, 9, 3]])
    logits = exported(ids)
    assert logits.shape == (1, 5, 64)
    assert mx.isfinite(logits).all().item()
    # Export produces plain float modules (dequantized), independent of the base.
    assert isinstance(exported.model.embed_tokens, nn.Embedding)
    assert not isinstance(exported.model.embed_tokens, nn.QuantizedEmbedding)
