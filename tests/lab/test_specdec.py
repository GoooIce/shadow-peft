"""Specdec engine tests.

Two tiers, because LFM2's conv layer uses *different algorithms* for cached
vs full-sequence forward (summation order differs -> rare near-tie argmax
flips even in fp32):

- exact tier (attention-only Llama): random init gives wide argmax margins,
  so token-identity holds exactly for any draft — the strict lossless proof.
- agreement tier (LFM2 conv hybrid): a rollback bug corrupts tokens
  wholesale, a near-tie flip changes isolated positions — assert high
  agreement + sane accept stats instead of strict equality.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from transformers import Lfm2Config, Lfm2ForCausalLM, LlamaConfig, LlamaForCausalLM  # noqa: E402

from shadow_lab.specdec import greedy_reference, greedy_speculative_generate  # noqa: E402


def _tiny_lfm2(seed: int) -> Lfm2ForCausalLM:
    torch.manual_seed(seed)
    cfg = Lfm2Config(
        hidden_size=32, intermediate_size=64, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, vocab_size=256,
        layer_types=["conv", "conv", "full_attention", "full_attention"],
        conv_L_cache=3, max_position_embeddings=512,
    )
    return Lfm2ForCausalLM(cfg).eval()


def _tiny_llama(seed: int) -> LlamaForCausalLM:
    torch.manual_seed(seed)
    cfg = LlamaConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                      num_attention_heads=4, num_key_value_heads=2, vocab_size=256,
                      max_position_embeddings=512)
    return LlamaForCausalLM(cfg).eval()


def _ids(n: int = 12, seed: int = 7):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (1, n), generator=g)


# ---------- exact tier (attention-only) ----------

def test_lossless_exact_same_model():
    target = _tiny_llama(0)
    ids = _ids()
    ref = greedy_reference(target, ids, 24)
    out, stats = greedy_speculative_generate(target, target, ids, k=3, max_new_tokens=24)
    assert torch.equal(out, ref)
    assert stats["tau"] == pytest.approx(4.0)  # k accepted + bonus every round


def test_lossless_exact_independent_draft():
    target, draft = _tiny_llama(0), _tiny_llama(1)
    ids = _ids()
    ref = greedy_reference(target, ids, 24)
    for k in (1, 2, 4):
        out, stats = greedy_speculative_generate(target, draft, ids, k=k, max_new_tokens=24)
        assert torch.equal(out, ref), f"lossless property broken at k={k}"
        assert stats["tau"] <= k + 1


# ---------- agreement tier (conv hybrid) ----------

def _agreement(a: torch.Tensor, b: torch.Tensor) -> float:
    gen_a, gen_b = a[0, -b.shape[1]:], b[0, -b.shape[1]:]
    return float((gen_a == gen_b).float().mean())


def test_lfm2_oracle_high_agreement():
    target = _tiny_lfm2(0)
    ids = _ids()
    ref = greedy_reference(target, ids, 32)
    out, stats = greedy_speculative_generate(target, target, ids, k=4, max_new_tokens=32)
    # near-tie flips aside, oracle must follow greedy almost everywhere
    assert _agreement(out, ref) >= 0.9
    assert stats["tau"] >= 3.0  # k+1 = 5, tolerate near-tie rejects


def test_lfm2_mismatch_path_and_agreement():
    target = _tiny_lfm2(0)
    draft = _tiny_lfm2(2)
    ids = _ids(16)
    # force divergence: bias one wrong logit on the draft head (untrained tiny
    # models collapse to the same constant token otherwise)
    with torch.no_grad():
        target_next = int(target(input_ids=ids).logits[:, -1].argmax(-1))
    wrong = (target_next + 1) % 256

    def _bias(_module, _args, output):
        output[..., wrong] = output[..., wrong] + 1000.0
        return output

    draft.lm_head.register_forward_hook(_bias)
    out, stats = greedy_speculative_generate(target, draft, ids, k=4, max_new_tokens=32)
    ref = greedy_reference(target, ids, 32)
    assert _agreement(out, ref) >= 0.9
    assert sum(stats["accept_hist"][:4]) > 0  # mismatch/rollback path exercised


# ---------- v1.1 wall-clock path: crop snapshot + deferred bonus ----------

def test_snapshot_crop_matches_deep():
    target, draft = _tiny_llama(0), _tiny_llama(1)
    ids = _ids()
    out_crop, st_crop = greedy_speculative_generate(
        target, draft, ids, k=4, max_new_tokens=40, snapshot="crop")
    out_deep, st_deep = greedy_speculative_generate(
        target, draft, ids, k=4, max_new_tokens=40, snapshot="deep")
    assert torch.equal(out_crop, out_deep)
    assert st_crop["tau"] == st_deep["tau"]
    assert st_crop["accept_hist"] == st_deep["accept_hist"]


def test_crop_restore_bitwise():
    from shadow_lab.specdec import _deep_snapshot, _restore, _snapshot
    m = _tiny_lfm2(0)
    ids = torch.randint(0, 256, (1, 9))
    out = m(input_ids=ids, use_cache=True)
    cache = out.past_key_values
    truth = _deep_snapshot(cache)  # full-clone ground truth of the prefix
    snap = _snapshot(cache)
    nxt = out.logits[:, -1].argmax(-1, keepdim=True)
    o = m(input_ids=nxt, past_key_values=cache)
    cache = o.past_key_values
    assert cache.get_seq_length() == 10  # something was appended
    _restore(cache, snap)
    assert cache.get_seq_length() == 9
    for layer, t in zip(cache.layers, truth, strict=True):
        for n, v in vars(layer).items():
            if isinstance(v, torch.Tensor):
                assert torch.equal(v, t[n]), f"{n} not bit-restored"
            elif isinstance(v, dict):
                for si, st_ in v.items():
                    if isinstance(st_, torch.Tensor):
                        assert torch.equal(st_, t[n][si]), f"{n}[{si}] not bit-restored"


def test_defer_bonus_equivalence_and_partial_rollback():
    target, draft = _tiny_llama(0), _tiny_llama(1)
    ids = _ids()
    # force frequent mismatches so partial rounds (incl. rollback with a
    # deferred pending token) are exercised on the exact tier
    with torch.no_grad():
        t_head = target.lm_head.weight
    orig = draft.lm_head.weight
    with torch.no_grad():
        draft.lm_head.weight.copy_(t_head * 0.6 + orig * 0.4)
    ref = greedy_reference(target, ids, 48)
    for defer in (True, False):
        out, stats = greedy_speculative_generate(
            target, draft, ids, k=4, max_new_tokens=48, defer_bonus=defer)
        assert torch.equal(out, ref), f"lossless broken with defer_bonus={defer}"
        assert sum(stats["accept_hist"][:4]) > 0  # partial rounds really occurred


def test_defer_full_accept_then_mismatch_regression():
    """The GB10 crash path: a full-accept round sets a pending token, the next
    round mismatches and must replay [pending + committed] through the target.

    Built deterministically: draft == target for round 1 (oracle full-accept),
    a counter-armed bias hook corrupts the draft from round 2 on.
    """
    import copy
    target = _tiny_llama(0)
    draft = copy.deepcopy(target)
    k = 4
    calls = {"n": 0}

    def _arm(_module, _args, output):
        calls["n"] += 1
        if calls["n"] > k + 3:  # prefill + chain + bonus, then round-2 chain
            output[..., 0] = output[..., 0] + 1000.0
        return output

    draft.lm_head.register_forward_hook(_arm)
    ids = _ids()
    ref = greedy_reference(target, ids, 40)
    out, stats = greedy_speculative_generate(target, draft, ids, k=k, max_new_tokens=40)
    assert torch.equal(out, ref)
    assert stats["accept_hist"][k] > 0  # round 1 was a full accept
    assert sum(stats["accept_hist"][:k]) > 0  # later rounds mismatched (pending replay)
