"""LFM2 hybrid-backbone shadow tests (conv+attention mix, spec p0-common).

LFM2 slices to an all-conv prefix when building the implicit shadow model —
that must be remapped to attention layers: an all-conv shadow cannot use HF
caches (``get_seq_length`` needs an attention entry) and a 3-token conv memory
cannot track sequence state. Pins the remap and the cached decode path.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("transformers.models.lfm2")

from transformers import Lfm2Config, Lfm2ForCausalLM  # noqa: E402

from shadow_peft import ShadowConfig, get_shadow_model  # noqa: E402
from shadow_peft.task_models import ShadowForCausalLM  # noqa: E402


def _tiny_lfm2(num_layers: int = 6) -> Lfm2ForCausalLM:
    types = ["conv", "conv", "full_attention", "conv", "conv", "full_attention"][:num_layers]
    while len(types) < num_layers:
        types.append("conv")
    cfg = Lfm2Config(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=num_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=256,
        layer_types=types,
        conv_L_cache=3,
        max_position_embeddings=512,
    )
    return Lfm2ForCausalLM(cfg)


def _ids(seq: int = 12):
    return torch.randint(0, 256, (1, seq))


def test_implicit_shadow_remaps_all_conv_prefix():
    peft = get_shadow_model(_tiny_lfm2(), ShadowConfig(num_shadow_layers=2))
    shadow_cfg = peft.shadow_model.config
    # First two base layers are conv; the shadow must not inherit that prefix.
    assert list(shadow_cfg.layer_types) == ["full_attention", "full_attention"]
    fa = shadow_cfg.full_attn_idxs
    assert fa is None or list(fa) == [0, 1]


def test_explicit_shadow_layer_types_override():
    conv = get_shadow_model(
        _tiny_lfm2(), ShadowConfig(num_shadow_layers=2, shadow_layer_types=["conv", "conv"])
    )
    assert list(conv.shadow_model.config.layer_types) == ["conv", "conv"]
    mixed = get_shadow_model(
        _tiny_lfm2(), ShadowConfig(num_shadow_layers=2, shadow_layer_types=["conv", "full_attention"])
    )
    assert list(mixed.shadow_model.config.layer_types) == ["conv", "full_attention"]


def test_cached_prefill_step_and_generate():
    peft = get_shadow_model(_tiny_lfm2(), ShadowConfig(num_shadow_layers=2)).eval()
    ids = _ids()
    with torch.no_grad():
        o1 = peft(input_ids=ids[:, :6], use_cache=True)
        assert o1.past_key_values is not None
        o2 = peft(input_ids=ids[:, 6:], past_key_values=o1.past_key_values, use_cache=True)
        assert o2.logits.shape == (1, 6, 256)
    # Generation goes through the task wrapper (its forward initializes shadow state).
    lm = ShadowForCausalLM(peft).eval()
    with torch.no_grad():
        gen = lm.generate(ids[:, :4], max_new_tokens=4, do_sample=False)
    assert gen.shape[1] == 8


def test_swap_adapter_roundtrip_bit_identical():
    peft = get_shadow_model(_tiny_lfm2(), ShadowConfig(num_shadow_layers=2)).eval()
    ids = _ids()
    with torch.no_grad():
        ref = peft(input_ids=ids).logits
    peft.swap_adapter(peft.adapter_state_dict())
    with torch.no_grad():
        after = peft(input_ids=ids).logits
    assert torch.equal(ref, after)
