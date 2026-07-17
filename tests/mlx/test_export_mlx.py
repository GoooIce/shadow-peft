"""Tests for export_shadow() and the pseudo-inverse projection utilities (MLX)."""

from __future__ import annotations

import pytest

pytest.importorskip("mlx")
pytest.importorskip("mlx_lm")

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
from mlx_lm.models import llama  # noqa: E402

from shadow_peft_mlx import (  # noqa: E402
    ProjectedCausalLM,
    compute_pinv_projection,
    get_shadow_model,
)


def test_export_shadow_forward(llama_factory, shadow_cfg):
    peft = get_shadow_model(llama_factory(seed=20), shadow_cfg)
    peft.eval()
    exported = peft.export_shadow()
    exported.eval()

    ids = mx.array([[1, 5, 7, 9, 3]])
    logits = exported(ids)
    assert logits.shape == (1, 5, 64)
    assert mx.isfinite(logits).all().item()

    # Exported model is fully decoupled: shadow backbone weights, base embeddings.
    assert exported.model.embed_tokens is not peft.base_model.model.embed_tokens
    assert mx.abs(
        exported.model.embed_tokens.weight - peft.base_model.model.embed_tokens.weight
    ).max().item() == 0.0


def test_export_shadow_hidden_mismatch_returns_projected(llama_factory, shadow_cfg):
    # Explicit shadow backbone with a smaller hidden size than the base (16 vs 32).
    small_args = llama.ModelArgs(
        model_type="llama",
        hidden_size=16,
        num_hidden_layers=1,
        intermediate_size=32,
        num_attention_heads=2,
        num_key_value_heads=2,
        rms_norm_eps=1e-5,
        vocab_size=64,
    )
    shadow16 = llama.LlamaModel(small_args)
    peft = get_shadow_model(llama_factory(seed=21), shadow_cfg, shadow_model=shadow16)
    peft.eval()
    assert peft.shadow_hidden_projection is not None

    exported = peft.export_shadow()
    assert isinstance(exported, ProjectedCausalLM)
    exported.eval()

    ids = mx.array([[1, 5, 7, 9, 3]])
    assert exported(ids).shape == (1, 5, 64)

    # Cached prefill + decode step works on the exported projected model.
    caches = exported.make_cache()
    exported(ids, cache=caches)
    step = exported(mx.array([[11]]), cache=caches)
    assert step.shape == (1, 1, 64)


def test_pinv_projection_exact_case():
    # W_ref = W_target @ P_true has an exact solution; pinv must recover it.
    mx.random.seed(0)
    w_target = mx.random.normal((64, 32))  # (vocab, base_hidden)
    p_true = mx.random.normal((32, 8))  # (base_hidden, shadow_hidden)
    w_ref = w_target @ p_true  # (vocab, shadow_hidden)

    w_proj = compute_pinv_projection(w_ref, w_target)
    err = mx.linalg.norm(w_target @ w_proj - w_ref) / mx.linalg.norm(w_ref)
    assert float(err) < 1e-5


def test_projected_wrap_optimal_init(llama_factory):
    base = llama_factory(seed=22)  # hidden_size=32
    base.eval()
    d_base, d_shadow, vocab = 32, 32, 64

    lm_head = nn.Linear(d_base, vocab, bias=False)
    projection = nn.Linear(d_shadow, d_base, bias=False)
    p_true = mx.random.normal((d_base, d_shadow)) * 0.1
    reference = lm_head.weight @ p_true  # (vocab, d_shadow)

    wrapped = ProjectedCausalLM.wrap(
        shadow_model=base,
        shadow_hidden_projection=projection,
        lm_head=lm_head,
        init_optimal_projection=True,
        reference_lm_head_weight=reference,
    )
    err = mx.linalg.norm(wrapped.lm_head.weight @ wrapped.shadow_hidden_projection.weight - reference)
    rel = err / mx.linalg.norm(reference)
    assert float(rel) < 1e-4

    ids = mx.array([[1, 5, 7]])
    assert wrapped(ids).shape == (1, 3, 64)
