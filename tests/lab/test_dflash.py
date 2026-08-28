"""DFlash2-structured block drafter tests (spec exp5).

Lossless guarantee for block drafting: the block engine's output must be
token-identical to the plain cached greedy reference of the target for an
UNTRAINED random drafter (acceptance is target-side; the drafter only
proposes). Also pins block-head shapes and refinement determinism.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from transformers import Lfm2Config, Lfm2ForCausalLM  # noqa: E402

from shadow_lab.dflash import KVInjectedDrafter  # noqa: E402
from shadow_lab.specdec import block_speculative_generate_fullctx, greedy_reference  # noqa: E402


def _tiny_target(seed: int = 0) -> Lfm2ForCausalLM:
    torch.manual_seed(seed)
    cfg = Lfm2Config(
        hidden_size=48, intermediate_size=96, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, vocab_size=256,
        layer_types=["conv", "conv", "full_attention", "full_attention"],
        conv_L_cache=3, max_position_embeddings=512,
    )
    return Lfm2ForCausalLM(cfg).eval()


def _drafter(target, k: int = 4, seed: int = 3) -> KVInjectedDrafter:
    torch.manual_seed(seed)
    n_layers = target.config.num_hidden_layers
    hidden_size = target.config.hidden_size
    return KVInjectedDrafter(
        vocab=target.config.vocab_size, hidden=64, n_layers=2, n_heads=4, n_kv_heads=2,
        target_hidden=hidden_size, target_n_layers=n_layers, k=k,
    ).eval()


def _ids(n: int = 12, seed: int = 7):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (1, n), generator=g)


def test_block_engine_lossless_random_drafter():
    target = _tiny_target()
    drafter = _drafter(target)
    ids = _ids()
    ref = greedy_reference(target, ids, 32)
    for r in (0, 2):
        out, stats = block_speculative_generate_fullctx(target, drafter, ids, k=4, r=r, max_new_tokens=32)
        assert torch.equal(out, ref), f"block lossless broken at r={r}"
        assert stats["tau"] <= 5


def test_block_head_shapes_and_refinement_stability():
    target = _tiny_target()
    drafter = _drafter(target, k=4)
    ids = _ids()
    with torch.no_grad():
        out = target(input_ids=ids, output_hidden_states=True)
        feats = drafter.inject_features(out.hidden_states)
        logits = drafter.forward_block(ids, feats)
        assert logits.shape == (1, 4, 256)
        b0 = drafter.draft_block(ids, feats, r=0)
        b2 = drafter.draft_block(ids, feats, r=2)
        assert b0.shape == (1, 4)
        # deterministic refinement: same input -> same block
        assert torch.equal(b2, drafter.draft_block(ids, feats, r=2))
    # causal forward trains (grad reaches injection projections)
    drafter.train()
    logits = drafter.forward_causal(ids, feats)
    logits.sum().backward()
    inj = drafter.layers[0].k_inj.weight.grad
    assert inj is not None and torch.isfinite(inj).all()


def test_injected_lfm_drafter_engine_path():
    """InjectedLFMDrafter through the chain engine: lossless + seams plumbed."""
    from transformers import LlamaConfig, LlamaForCausalLM

    from shadow_lab.dflash import InjectedLFMDrafter
    from shadow_lab.specdec import greedy_speculative_generate

    torch.manual_seed(0)
    cfg = LlamaConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                      num_attention_heads=4, num_key_value_heads=2, vocab_size=256,
                      max_position_embeddings=512)
    target = LlamaForCausalLM(cfg).eval()
    lm = LlamaForCausalLM(LlamaConfig(**cfg.to_dict())).eval()
    drafter = InjectedLFMDrafter(lm, target_hidden=32)
    assert len(drafter.seams) == 4 and drafter.sources == [1, 2, 3, 4]

    ids = _ids()
    ref = greedy_reference(target, ids, 24)
    out, stats = greedy_speculative_generate(target, drafter, ids, k=3, max_new_tokens=24)
    assert torch.equal(out, ref)

    # injection projections train through the causal path
    lm.train()
    with torch.no_grad():
        tout = target(input_ids=ids, output_hidden_states=True)
    drafter.set_feats(tout.hidden_states)
    lm(input_ids=ids).logits.sum().backward()
    g = drafter.seams[0].proj_k.weight.grad
    assert g is not None and torch.isfinite(g).all()
