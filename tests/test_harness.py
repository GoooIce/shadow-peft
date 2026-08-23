"""Tests for the Mixture-of-Shadow harness (request-level routing + adapter swap)."""

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
from shadow_peft.harness import (  # noqa: E402
    Router,
    RoutingRule,
    Session,
    ShadowEngine,
    ShadowHarnessConfig,
    SpecialistConfig,
)


# --------------------------------------------------------------------------- #
# Tiny-model + tokenizer helpers (mirror tests/test_shadow_peft.py patterns)
# --------------------------------------------------------------------------- #
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


class _TokenizerStub:
    """Minimal tokenizer for tests: encodes to ids, decodes back, no chat template.

    Produces deterministic ids so generation is reproducible across runs.
    """

    def __call__(self, text: str, return_tensors: str = "pt"):
        ids = [(ord(c) % 120 + 4) for c in text[:48]]
        if not ids:
            ids = [1]
        t = torch.tensor([ids], dtype=torch.long)
        return {"input_ids": t, "attention_mask": torch.ones_like(t)}

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        return "".join(chr(int(i) % 120 + 32) for i in ids.tolist())

    def to(self, device):
        return self


def _shadow_cfg() -> ShadowConfig:
    return ShadowConfig(
        num_shadow_layers=1,
        injection_hidden_size=8,
        gate_hidden_size=10,
        alpha=0.1,
        dropout=0.0,
    )


def _build_model():
    torch.manual_seed(0)
    base = _tiny_llama()
    base.eval()
    peft = get_shadow_model(base, _shadow_cfg())
    model = ShadowForCausalLM(peft, shadow_loss_weight=0.0)
    model.eval()
    return model


def _save_random_adapter(base_factory, tmp_path, name: str, seed: int) -> str:
    """Build a fresh shadow adapter with a distinct RNG seed and save it."""
    torch.manual_seed(seed)
    base = base_factory()
    base.eval()
    peft = get_shadow_model(base, _shadow_cfg())
    peft.eval()
    save_dir = tmp_path / f"adapter_{name}"
    peft.save_pretrained(save_dir)
    return str(save_dir)


# --------------------------------------------------------------------------- #
# Router tests (pure Python, no torch model needed)
# --------------------------------------------------------------------------- #
def _router_cfg() -> ShadowHarnessConfig:
    return ShadowHarnessConfig(
        specialists=[
            SpecialistConfig(name="medical", keywords=["diagnose", "symptom"], priority=1),
            SpecialistConfig(name="code", keywords=["python", "function"], priority=2),
            SpecialistConfig(name="general"),
        ],
        default_specialist="general",
        rules=[RoutingRule(field="domain", value="health", specialist="medical")],
    )


def test_router_keyword_match():
    r = Router(_router_cfg())
    assert r.route("how to write a python function").specialist == "code"
    assert r.route("please diagnose my symptom").specialist == "medical"


def test_router_metadata_guardrail_overrides_keyword():
    r = Router(_router_cfg())
    d = r.route("write a python function", metadata={"domain": "health"})
    assert d.specialist == "medical"
    assert d.reason == "metadata_rule"


def test_router_default_fallback():
    r = Router(_router_cfg())
    d = r.route("tell me a joke")
    assert d.specialist == "general"
    assert d.reason == "default_fallback"


def test_router_priority_breaks_ties():
    r = Router(_router_cfg())
    # Both 'medical' (priority 1) and 'code' (priority 2) keywords present -> code wins.
    assert r.route("diagnose this python function").specialist == "code"


def test_router_session_sticky():
    r = Router(_router_cfg())
    sess = Session()
    sess.bind("code")
    d = r.route("now do something else", session=sess)
    assert d.specialist == "code"
    assert d.reason == "session_sticky"


def test_router_metadata_overrides_stickiness():
    cfg = _router_cfg()
    r = Router(cfg)
    sess = Session()
    sess.bind("code")
    d = r.route("anything", metadata={"domain": "health"}, session=sess)
    assert d.specialist == "medical"
    assert d.reason == "metadata_rule"


def test_router_sticky_disabled_when_config_says_so():
    cfg = ShadowHarnessConfig(
        specialists=[
            SpecialistConfig(name="code", keywords=["python"]),
            SpecialistConfig(name="general"),
        ],
        default_specialist="general",
        sticky_sessions=False,
    )
    r = Router(cfg)
    sess = Session()
    sess.bind("code")
    # sticky off -> no keyword hit -> fallback
    assert r.route("hello world", session=sess).specialist == "general"


# --------------------------------------------------------------------------- #
# Session tests
# --------------------------------------------------------------------------- #
def test_session_add_turn_validates_role():
    s = Session()
    s.add_turn("user", "hi")
    s.add_turn("assistant", "hello")
    assert len(s.history) == 2
    with pytest.raises(ValueError):
        s.add_turn("system", "x")


def test_session_reset_clears_state():
    s = Session()
    s.bind("code")
    s.add_turn("user", "hi")
    s.reset()
    assert s.specialist is None
    assert s.history == []


def test_session_auto_id():
    s = Session()
    assert s.session_id != ""
    assert len(s.session_id) == 12


# --------------------------------------------------------------------------- #
# Config validation tests
# --------------------------------------------------------------------------- #
def test_config_rejects_missing_default():
    with pytest.raises(ValueError, match="default_specialist"):
        ShadowHarnessConfig(
            specialists=[SpecialistConfig(name="a")],
            default_specialist="b",
        )


def test_config_rejects_duplicate_names():
    with pytest.raises(ValueError, match="Duplicate"):
        ShadowHarnessConfig(
            specialists=[SpecialistConfig(name="a"), SpecialistConfig(name="a")],
            default_specialist="a",
        )


def test_config_specialist_lookup():
    cfg = _router_cfg()
    assert cfg.specialist("code").priority == 2
    with pytest.raises(KeyError):
        cfg.specialist("nope")


# --------------------------------------------------------------------------- #
# Registry / engine integration tests (need torch model)
# --------------------------------------------------------------------------- #
def _engine_with_two_specialists(tmp_path):
    """Build an engine with two genuinely different saved adapters."""
    model = _build_model()
    tokenizer = _TokenizerStub()

    path_a = _save_random_adapter(_tiny_llama, tmp_path, "alpha", seed=101)
    path_b = _save_random_adapter(_tiny_llama, tmp_path, "beta", seed=202)

    cfg = ShadowHarnessConfig(
        specialists=[
            SpecialistConfig(name="alpha", adapter_path=path_a, keywords=["alpha"]),
            SpecialistConfig(name="beta", adapter_path=path_b, keywords=["beta"]),
            SpecialistConfig(name="general"),
        ],
        default_specialist="general",
    )
    return ShadowEngine(model, tokenizer, cfg), model


def test_ensure_swaps_adapter_and_resets_shadow_cache(tmp_path):
    engine, model = _engine_with_two_specialists(tmp_path)
    peft = model.peft_model

    engine.registry.ensure("alpha")
    assert peft._shadow_cache is None
    alpha_sd = {k: v.detach().clone() for k, v in peft.adapter_state_dict().items()}

    engine.registry.ensure("beta")
    assert peft._shadow_cache is None
    beta_sd = {k: v.detach().clone() for k, v in peft.adapter_state_dict().items()}
    # Implicit shadow model has no embed_tokens (driven by base); compare an attn weight.
    key = "shadow_model.layers.0.self_attn.q_proj.weight"
    assert not torch.equal(alpha_sd[key], beta_sd[key])


def test_swap_is_idempotent_when_already_active(tmp_path):
    engine, model = _engine_with_two_specialists(tmp_path)
    engine.registry.ensure("alpha")
    model.peft_model._shadow_cache = "sentinel"
    engine.registry.ensure("alpha")
    assert model.peft_model._shadow_cache == "sentinel"


def test_swap_output_matches_direct_load_baseline(tmp_path):
    """After swapping to a specialist, output == loading that specialist into a fresh model."""
    engine, model = _engine_with_two_specialists(tmp_path)
    tokenizer = engine.tokenizer

    # Baseline: fresh model, load alpha directly.
    torch.manual_seed(0)
    base_ref = _tiny_llama()
    base_ref.eval()
    peft_ref = get_shadow_model(base_ref, _shadow_cfg())
    from safetensors.torch import load_file

    alpha_path = engine.config.specialist("alpha").adapter_path
    peft_ref.load_adapter_state_dict(load_file(f"{alpha_path}/shadow_adapter.safetensors"))
    peft_ref.eval()
    ref_model = ShadowForCausalLM(peft_ref, shadow_loss_weight=0.0)
    ref_model.eval()

    inputs = tokenizer("alpha query", return_tensors="pt")

    # Baseline forward.
    inputs_ref = {k: v.to(ref_model.device) for k, v in inputs.items()}
    with torch.no_grad():
        ref_out = ref_model(**inputs_ref).logits

    # Via engine: swap to alpha, then a raw forward.
    engine.registry.ensure("alpha")
    model.set_inference_mode("base_shadow")
    inputs_live = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        live_out = model(**inputs_live).logits

    assert torch.allclose(ref_out, live_out, atol=1e-5, rtol=1e-4), (
        "Swapped-adapter output should match a fresh model loaded with the same adapter."
    )


def test_swap_ab_a_is_lossless(tmp_path):
    """alpha -> beta -> alpha: the second alpha output matches the first."""
    engine, model = _engine_with_two_specialists(tmp_path)
    tokenizer = engine.tokenizer

    def _logits():
        inputs = tokenizer("alpha query", return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            return model(**inputs).logits.clone()

    engine.registry.ensure("alpha")
    first = _logits()
    engine.registry.ensure("beta")
    engine.registry.ensure("alpha")
    second = _logits()
    assert torch.allclose(first, second, atol=0.0, rtol=0.0)


def test_engine_generate_routes_and_returns_info(tmp_path):
    engine, _model = _engine_with_two_specialists(tmp_path)
    text, info = engine.generate("alpha please", max_new_tokens=3)
    assert info["specialist"] == "alpha"
    assert info["reason"] == "keyword"
    assert isinstance(text, str)


def test_engine_session_sticky_and_history(tmp_path):
    engine, _model = _engine_with_two_specialists(tmp_path)
    sess = engine.session()

    engine.generate("beta query", session=sess, max_new_tokens=2)
    assert sess.specialist == "beta"
    assert len(sess.history) == 2  # user + assistant

    _, info = engine.generate("unrelated text", session=sess, max_new_tokens=2)
    assert info["specialist"] == "beta"
    assert info["reason"] == "session_sticky"
    assert len(sess.history) == 4


def test_engine_history_isolation_across_sessions(tmp_path):
    engine, _model = _engine_with_two_specialists(tmp_path)
    s1 = engine.session()
    s2 = engine.session()

    engine.generate("alpha query", session=s1, max_new_tokens=1)
    engine.generate("beta query", session=s2, max_new_tokens=1)

    # Each session only holds its own turns.
    assert s1.history[0]["content"] == "alpha query"
    assert s2.history[0]["content"] == "beta query"
    assert len(s1.history) == 2
    assert len(s2.history) == 2
    assert s1.session_id != s2.session_id


def test_engine_metadata_overrides_stickiness(tmp_path):
    engine, _model = _engine_with_two_specialists(tmp_path)
    sess = engine.session()
    engine.generate("alpha query", session=sess, max_new_tokens=1)
    assert sess.specialist == "alpha"

    # Swap in a config with a metadata rule that forces beta.
    cfg = ShadowHarnessConfig(
        specialists=[
            SpecialistConfig(name="alpha", keywords=["alpha"]),
            SpecialistConfig(name="beta"),
            SpecialistConfig(name="general"),
        ],
        default_specialist="general",
        rules=[RoutingRule(field="force", value="beta", specialist="beta")],
    )
    engine.config = cfg
    engine.router = Router(cfg)
    _, info = engine.generate("alpha query", metadata={"force": "beta"}, session=sess, max_new_tokens=1)
    assert info["specialist"] == "beta"
    assert info["reason"] == "metadata_rule"


def test_swap_adapter_clears_shadow_runtime_state(tmp_path):
    """The swap seam: shadow runtime state resets BEFORE the weights load."""
    engine, model = _engine_with_two_specialists(tmp_path)
    peft = model.peft_model
    engine.registry.ensure("alpha")
    # Plant stale runtime state as an interrupted generation would leave behind.
    peft._shadow_cache = "stale"
    peft._shadow_hidden_states = "stale"
    beta_sd = engine.registry._adapter_states["beta"]
    peft.swap_adapter(beta_sd)
    assert peft._shadow_cache is None
    assert peft._shadow_hidden_states is None
    key = "shadow_model.layers.0.self_attn.q_proj.weight"
    assert torch.equal(peft.adapter_state_dict()[key], beta_sd[key])


def test_registry_restores_previous_adapter_on_failed_swap(tmp_path):
    """A swap failing mid-load must roll back to the previous specialist.

    Dropping a required ``shadow_update_model`` key makes the strict load fail
    on the LAST of the four modules, i.e. after ``shadow_model`` /
    injection already hold the new specialist's weights -- the mixed-state
    hazard. The registry must restore alpha and re-raise the original error.
    """
    engine, model = _engine_with_two_specialists(tmp_path)
    peft = model.peft_model
    engine.registry.ensure("alpha")

    key = "shadow_model.layers.0.self_attn.q_proj.weight"
    alpha_weight = engine.registry._adapter_states["alpha"][key].detach().clone()

    beta_state = dict(engine.registry._adapter_states["beta"])
    dropped = sorted(k for k in beta_state if k.startswith("shadow_update_model."))[0]
    del beta_state[dropped]
    engine.registry._adapter_states["beta"] = beta_state

    with pytest.raises(RuntimeError):
        engine.registry.ensure("beta")

    # Rolled back: still serving alpha, with alpha's actual weights intact.
    assert engine.registry.current == "alpha"
    assert torch.equal(peft.adapter_state_dict()[key], alpha_weight)
    # Shadow state was cleared before the (failed) load and stays cleared.
    assert peft._shadow_cache is None
    assert peft._shadow_hidden_states is None


def test_harness_core_is_single_source():
    """torch and MLX harness types are the same backend-free classes from shadow_core."""
    import shadow_core
    import shadow_peft
    import shadow_peft_mlx

    assert shadow_peft.ShadowConfig is shadow_core.ShadowConfig
    assert shadow_peft_mlx.ShadowConfig is shadow_core.ShadowConfig
    from shadow_peft.harness import Router as RouterTorch
    from shadow_peft_mlx.harness import Router as RouterMlx
    from shadow_core.harness import Router as RouterCore

    assert RouterTorch is RouterCore
    assert RouterMlx is RouterCore
