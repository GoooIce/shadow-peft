"""CachedBlockDrafter + cached block-engine tests.

The exact tier (attention-only Llama) proves the engine's lossless property
for any drafter; the cache-integrity tests pin the refinement rollback (the
O(1) crop snapshot must leave the trunk exactly where it was).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from transformers import Lfm2Config, Lfm2ForCausalLM, LlamaConfig, LlamaForCausalLM  # noqa: E402

from shadow_lab.block_drafter import CachedBlockDrafter  # noqa: E402
from shadow_lab.specdec import (  # noqa: E402
    _deep_snapshot,
    block_speculative_generate,
    greedy_reference,
)


def _tiny_llama(seed: int) -> LlamaForCausalLM:
    torch.manual_seed(seed)
    cfg = LlamaConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                      num_attention_heads=4, num_key_value_heads=2, vocab_size=256,
                      max_position_embeddings=512)
    return LlamaForCausalLM(cfg).eval()


def _tiny_lfm2(seed: int) -> Lfm2ForCausalLM:
    torch.manual_seed(seed)
    cfg = Lfm2Config(
        hidden_size=32, intermediate_size=64, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, vocab_size=256,
        layer_types=["conv", "conv", "full_attention", "full_attention"],
        conv_L_cache=3, max_position_embeddings=512,
    )
    return Lfm2ForCausalLM(cfg).eval()


def _ids(n: int = 12, seed: int = 7):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (1, n), generator=g)


def _drafter(lm, k=4, **kw) -> CachedBlockDrafter:
    d = CachedBlockDrafter(lm, k=k, **kw)
    with torch.no_grad():  # random low-rank heads start near-identity; scale for divergence
        for tf in d.heads.tf:
            tf.weight.mul_(3.0)
    return d.eval()


# ---------- lossless proof (exact tier) ----------

def test_block_engine_lossless_exact():
    target = _tiny_llama(0)
    drafter = _drafter(_tiny_llama(1), k=4)
    ids = _ids()
    ref = greedy_reference(target, ids, 40)
    for k, r in ((2, 0), (4, 0), (4, 2)):
        out, stats = block_speculative_generate(target, drafter, ids, k=k, r=r,
                                                max_new_tokens=40)
        assert torch.equal(out, ref), f"lossless broken at k={k}, r={r}"
        assert stats["k_eff_mean"] == k
        assert stats["tau"] <= k + 1


def test_block_engine_partial_and_defer_paths():
    """Force full-accept then mismatch rounds (deferred pending + rollback)."""
    import copy
    target = _tiny_llama(0)
    drafter = _drafter(copy.deepcopy(target), k=4)  # oracle-ish trunk, wild heads
    ids = _ids(16)
    ref = greedy_reference(target, ids, 48)
    for defer in (True, False):
        out, stats = block_speculative_generate(target, drafter, ids, k=4, r=1,
                                                max_new_tokens=48, defer_bonus=defer)
        assert torch.equal(out, ref), f"defer={defer}"
        assert sum(stats["accept_hist"][:4]) > 0  # partial rounds exercised


# ---------- drafter mechanics ----------

def test_refinement_leaves_cache_bit_identical():
    lm = _tiny_lfm2(0)
    drafter = _drafter(lm, k=4)
    ids = _ids(9)
    drafter.prefill(ids)
    truth = _deep_snapshot(drafter._cache)
    len_before = drafter._cache.get_seq_length()
    h_before = drafter._h_last.clone()
    b1 = drafter.draft(4, r=2)
    assert drafter._cache.get_seq_length() == len_before, "refinement leaked cache growth"
    for layer, t in zip(drafter._cache.layers, truth, strict=True):
        for n, v in vars(layer).items():
            if isinstance(v, torch.Tensor):
                assert torch.equal(v, t[n]), f"{n} mutated by refinement"
    assert torch.equal(drafter._h_last, h_before)
    assert b1.shape == (1, 4)
    assert torch.equal(drafter.draft(4, r=2), b1)  # deterministic re-draft


def test_draft_k_use_and_commit_extends():
    lm = _tiny_lfm2(0)
    drafter = _drafter(lm, k=8)
    ids = _ids(9)
    drafter.prefill(ids)
    assert drafter.draft(k_use=3).shape == (1, 3)
    assert drafter.draft().shape == (1, 8)
    len_before = drafter._cache.get_seq_length()
    h = drafter.commit(torch.tensor([5, 9, 2]))
    assert drafter._cache.get_seq_length() == len_before + 3
    assert h.shape == (1, 1, lm.config.hidden_size)


def test_confidence_threshold_truncates_block():
    lm = _tiny_llama(0)
    drafter = CachedBlockDrafter(lm, k=8, with_confidence=True).eval()
    drafter.prefill(_ids(9))
    with torch.no_grad():  # force every position below threshold (sigmoid ~ 0)
        for c in drafter.heads.conf:
            c.bias.fill_(-10.0)
    short = drafter.draft(8, r=0, conf_threshold=0.9)
    assert short.shape[1] == 1
    with torch.no_grad():  # force every position above threshold (sigmoid ~ 1)
        for c in drafter.heads.conf:
            c.bias.fill_(10.0)
    full = drafter.draft(8, r=0, conf_threshold=0.9)
    assert full.shape[1] == 8
