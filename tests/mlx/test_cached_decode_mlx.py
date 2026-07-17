"""
Tests for ShadowPEFT (MLX) KV-cache incremental decode.

Mirrors tests/test_cached_decode.py (torch): cached prefill + decode must match
full-sequence recompute, proving the dual KV-cache integration is correct.
"""

from __future__ import annotations

import pytest

pytest.importorskip("mlx")
pytest.importorskip("mlx_lm")

import mlx.core as mx  # noqa: E402
from mlx_lm.generate import generate_step  # noqa: E402

from shadow_peft_mlx import ShadowForCausalLM, get_shadow_model  # noqa: E402


def _greedy_fullseq(model, prefix: mx.array, max_new: int) -> list[int]:
    """Manual greedy decode with full-sequence (uncached) forwards."""
    ids = prefix
    tokens = []
    for _ in range(max_new):
        logits = model(ids)
        nxt = mx.argmax(logits[:, -1, :], axis=-1)
        tokens.append(nxt.item())
        ids = mx.concatenate([ids, nxt[None]], axis=1)
    return tokens


def test_prefill_matches_uncached_llama(llama_factory, shadow_cfg):
    peft = get_shadow_model(llama_factory(seed=5), shadow_cfg)
    peft.eval()
    ids = mx.array([[1, 5, 7, 9, 3]])
    ref = peft(ids)

    caches = peft.make_cache()
    out = peft(ids, cache=caches)
    assert caches[0].offset == ids.shape[1]
    assert peft._runtime.shadow_cache is not None
    assert mx.abs(out - ref).max().item() == 0.0


def test_prefill_matches_uncached_qwen2(qwen2_factory, shadow_cfg):
    peft = get_shadow_model(qwen2_factory(seed=5), shadow_cfg)
    peft.eval()
    ids = mx.array([[1, 5, 7, 9, 3]])
    ref = peft(ids)

    caches = peft.make_cache()
    out = peft(ids, cache=caches)
    assert caches[0].offset == ids.shape[1]
    assert mx.abs(out - ref).max().item() == 0.0


def test_decode_step_matches_fullseq(llama_factory, shadow_cfg):
    peft = get_shadow_model(llama_factory(seed=6), shadow_cfg)
    peft.eval()
    prefix = mx.array([[1, 5, 7, 9]])
    nxt = mx.array([[11]])

    # Uncached reference first: an uncached call resets the shadow cache by design.
    ref_full = peft(mx.concatenate([prefix, nxt], axis=1))

    caches = peft.make_cache()
    peft(prefix, cache=caches)
    step_logits = peft(nxt, cache=caches)

    maxdiff = mx.abs(step_logits[:, -1] - ref_full[:, -1]).max().item()
    assert maxdiff < 1e-5, f"cached decode maxdiff {maxdiff:.2e}"
    assert peft._runtime.shadow_cache[0].offset == prefix.shape[1] + 1


def test_decode_step_matches_fullseq_qwen2(qwen2_factory, shadow_cfg):
    peft = get_shadow_model(qwen2_factory(seed=6), shadow_cfg)
    peft.eval()
    prefix = mx.array([[1, 5, 7, 9]])
    nxt = mx.array([[11]])

    ref_full = peft(mx.concatenate([prefix, nxt], axis=1))

    caches = peft.make_cache()
    peft(prefix, cache=caches)
    step_logits = peft(nxt, cache=caches)

    maxdiff = mx.abs(step_logits[:, -1] - ref_full[:, -1]).max().item()
    assert maxdiff < 1e-5, f"cached decode maxdiff {maxdiff:.2e}"


def test_step_by_step_logits_alignment(llama_factory, shadow_cfg):
    # Two identically seeded instances: the full-seq (uncached) path resets the
    # shadow cache, so it cannot share an instance with the cached path.
    peft_cached = get_shadow_model(llama_factory(seed=42), shadow_cfg)
    peft_cached.eval()
    peft_full = get_shadow_model(llama_factory(seed=42), shadow_cfg)
    peft_full.eval()

    prefix = mx.array([[3, 1, 4]])
    max_new = 4

    full_logits = peft_full(prefix)[:, -1, :]
    caches = peft_cached.make_cache()
    cached_logits = peft_cached(prefix, cache=caches)[:, -1, :]
    maxdiff = mx.abs(cached_logits - full_logits).max().item()
    assert maxdiff < 1e-5, f"prefill maxdiff {maxdiff:.2e}"

    generated: list[int] = []
    for step in range(max_new):
        nxt = mx.argmax(cached_logits, axis=-1)
        generated.append(nxt.item())

        ids_full = mx.concatenate([prefix, mx.array([generated])], axis=1)
        full_logits = peft_full(ids_full)[:, -1, :]

        if step < max_new - 1:
            cached_logits = peft_cached(nxt[None], cache=caches)[:, -1, :]
            maxdiff = mx.abs(cached_logits - full_logits).max().item()
            assert maxdiff < 1e-5, f"step {step} maxdiff {maxdiff:.2e}"


def test_generate_step_matches_manual_greedy(llama_factory, shadow_cfg):
    peft = get_shadow_model(llama_factory(seed=7), shadow_cfg)
    peft.eval()
    prompt = mx.array([3, 1, 4])

    gen_tokens = [t for t, _ in generate_step(prompt, peft, max_tokens=6)]
    manual_tokens = _greedy_fullseq(peft, prompt[None], 6)
    assert gen_tokens == manual_tokens


def test_uncached_call_resets_shadow_cache(llama_factory, shadow_cfg):
    peft = get_shadow_model(llama_factory(seed=8), shadow_cfg)
    peft.eval()
    ids = mx.array([[1, 5, 7, 9, 3]])

    caches = peft.make_cache()
    peft(ids, cache=caches)
    assert peft._runtime.shadow_cache is not None

    out1 = peft(ids)
    assert peft._runtime.shadow_cache is None
    out2 = peft(ids)
    assert mx.abs(out1 - out2).max().item() == 0.0


def test_cached_shadow_logits_shape(llama_factory, shadow_cfg):
    task = ShadowForCausalLM(get_shadow_model(llama_factory(seed=9), shadow_cfg))
    task.eval()
    ids = mx.array([[1, 5, 7, 9, 3]])

    caches = task.peft_model.make_cache()
    out = task(ids, cache=caches)
    assert out.shadow_logits is not None
    assert out.shadow_logits.shape == out.logits.shape
