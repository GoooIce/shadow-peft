"""
Tests for ShadowPEFT KV-cache incremental decode.

Verifies that cached generation (use_cache=True) produces identical results
to full-sequence recompute (use_cache=False), proving the cache integration
is mathematically correct.
"""

from __future__ import annotations

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


def _tiny_qwen2(num_layers: int = 4):
    Qwen2Config = getattr(transformers, "Qwen2Config", None)
    Qwen2ForCausalLM = getattr(transformers, "Qwen2ForCausalLM", None)
    if Qwen2Config is None or Qwen2ForCausalLM is None:
        pytest.skip("Qwen2 is not available in this transformers version")
    cfg = Qwen2Config(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=num_layers,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
    )
    return Qwen2ForCausalLM(cfg)


def _build_causallm(base):
    cfg = ShadowConfig(
        num_shadow_layers=1,
        injection_hidden_size=8,
        gate_hidden_size=10,
        alpha=0.1,
        dropout=0.0,
    )
    peft = get_shadow_model(base, cfg)
    return ShadowForCausalLM(peft, inference_mode="base_shadow")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_generate_cached_matches_nocache():
    """generate(use_cache=True) and generate(use_cache=False) must produce identical tokens."""
    torch.manual_seed(42)
    base = _tiny_llama(num_layers=4)
    m = _build_causallm(base)
    m.eval()

    input_ids = torch.randint(0, base.config.vocab_size, (1, 4))
    with torch.no_grad():
        gen_cached = m.generate(input_ids=input_ids, max_new_tokens=8, use_cache=True, do_sample=False)
        m.peft_model._shadow_cache = None  # reset for the no-cache run
        gen_nocache = m.generate(input_ids=input_ids, max_new_tokens=8, use_cache=False, do_sample=False)

    assert gen_cached.tolist() == gen_nocache.tolist(), (
        f"Cached {gen_cached.tolist()} != nocache {gen_nocache.tolist()}"
    )


def test_generate_cached_matches_nocache_qwen2():
    """Same test for Qwen2 architecture."""
    torch.manual_seed(42)
    base = _tiny_qwen2(num_layers=4)
    m = _build_causallm(base)
    m.eval()

    input_ids = torch.randint(0, base.config.vocab_size, (1, 4))
    with torch.no_grad():
        gen_cached = m.generate(input_ids=input_ids, max_new_tokens=8, use_cache=True, do_sample=False)
        m.peft_model._shadow_cache = None
        gen_nocache = m.generate(input_ids=input_ids, max_new_tokens=8, use_cache=False, do_sample=False)

    assert gen_cached.tolist() == gen_nocache.tolist()


def test_forward_prefill_then_decode():
    """
    Manually prefill with use_cache=True, then do a single decode step.
    Compare with full-sequence forward on the extended input.
    """
    torch.manual_seed(42)
    base = _tiny_llama(num_layers=4)
    m = _build_causallm(base)
    m.eval()

    prefix = torch.randint(0, base.config.vocab_size, (1, 4))
    next_token = torch.tensor([[77]])
    full_ids = torch.cat([prefix, next_token], dim=1)

    with torch.no_grad():
        # Prefill
        out_prefill = m(input_ids=prefix, use_cache=True)
        base_cache = out_prefill.past_key_values
        assert base_cache is not None, "Prefill should return a DynamicCache"
        assert base_cache.get_seq_length() == 4

        # Decode step: pass only the new token + cache
        out_decode = m(input_ids=next_token, past_key_values=base_cache, use_cache=True)
        logits_decode = out_decode.logits  # [1, 1, vocab]

        # Ground truth: full-sequence forward
        m.peft_model._shadow_cache = None  # reset shadow cache
        out_full = m(input_ids=full_ids, use_cache=False)
        logits_full = out_full.logits[:, -1:, :]  # [1, 1, vocab]

    maxdiff = (logits_decode - logits_full).abs().max().item()
    assert maxdiff < 1e-4, f"Cached decode logits maxdiff {maxdiff:.2e} exceeds threshold 1e-4"


def test_step_by_step_logits_alignment():
    """
    Greedy decode token-by-token with cache, comparing logits at each step
    against full-sequence recompute.

    Uses two separate model instances (with identical seeds) so the cached path's
    _shadow_cache is not corrupted by the full-seq ground-truth runs.
    """
    torch.manual_seed(42)
    base = _tiny_llama(num_layers=4)
    m_cached = _build_causallm(base)
    m_cached.eval()

    # Second instance for ground-truth full-seq runs (same seed → same weights).
    torch.manual_seed(42)
    base2 = _tiny_llama(num_layers=4)
    m_full = _build_causallm(base2)
    m_full.eval()

    prefix = torch.randint(0, base.config.vocab_size, (1, 3))
    max_new_tokens = 5

    with torch.no_grad():
        # Ground truth: full-sequence forward on prefix gives the logits for position len-1
        out_full_prefix = m_full(input_ids=prefix, use_cache=False)
        full_logits = out_full_prefix.logits[:, -1, :]

        # Cached prefill
        out = m_cached(input_ids=prefix, use_cache=True)
        cache = out.past_key_values
        cached_logits = out.logits[:, -1, :]

        maxdiff = (cached_logits - full_logits).abs().max().item()
        assert maxdiff < 1e-4, f"Prefill: cached vs full logits maxdiff {maxdiff:.2e}"

        generated = []
        for step in range(max_new_tokens):
            # Pick next token from cached logits
            next_tok = cached_logits.argmax(dim=-1, keepdim=True)
            generated.append(next_tok.item())

            # Full-seq ground truth for the extended sequence
            full_ids = torch.cat([prefix] + [torch.tensor([[t]]) for t in generated], dim=1)
            out_full = m_full(input_ids=full_ids, use_cache=False)
            full_logits = out_full.logits[:, -1, :]

            # Cached decode step
            if step < max_new_tokens - 1:
                out = m_cached(input_ids=next_tok, past_key_values=cache, use_cache=True)
                cached_logits = out.logits[:, -1, :]
                maxdiff = (cached_logits - full_logits).abs().max().item()
                assert maxdiff < 1e-4, f"Step {step}: cached vs full logits maxdiff {maxdiff:.2e}"

    # Verify the cached greedy sequence matches a cached generate() call.
    # generate() may stop early on EOS; compare up to the shorter sequence.
    cached_gen = m_cached.generate(input_ids=prefix, max_new_tokens=max_new_tokens, use_cache=True, do_sample=False)
    gen_tokens = cached_gen[0, prefix.shape[1]:].tolist()
    assert generated[:len(gen_tokens)] == gen_tokens, (
        f"Manual decode {generated[:len(gen_tokens)]} != generate() {gen_tokens}"
    )


def test_backward_compat_nocache_default():
    """
    Default forward (use_cache not specified) must behave exactly as before:
    no cache returned, same logits as use_cache=False.
    """
    torch.manual_seed(42)
    base = _tiny_llama(num_layers=4)
    m = _build_causallm(base)
    m.eval()

    input_ids = torch.randint(0, base.config.vocab_size, (1, 6))

    with torch.no_grad():
        out_default = m(input_ids=input_ids)
        out_explicit = m(input_ids=input_ids, use_cache=False)

    assert out_default.past_key_values is None, "Default forward should not return a cache"
    assert torch.allclose(out_default.logits, out_explicit.logits, atol=0.0, rtol=0.0)


def test_cached_forward_does_not_break_shadow_logits():
    """Shadow logits must still be computed correctly in cached mode."""
    torch.manual_seed(42)
    base = _tiny_llama(num_layers=4)
    m = _build_causallm(base)
    m.eval()

    input_ids = torch.randint(0, base.config.vocab_size, (1, 5))
    with torch.no_grad():
        out = m(input_ids=input_ids, use_cache=True)
    assert out.shadow_logits is not None
    assert out.shadow_logits.shape == out.logits.shape
