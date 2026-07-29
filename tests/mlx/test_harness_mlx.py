"""Tests for the Mixture-of-Shadow harness on the MLX backend."""

from __future__ import annotations

import pytest

pytest.importorskip("mlx")
pytest.importorskip("mlx_lm")

import mlx.core as mx  # noqa: E402

# Reuse the shared tiny-model factories.
from conftest import make_config, make_llama  # type: ignore[import-not-found]  # noqa: E402

from shadow_peft_mlx import (  # noqa: E402
    ShadowForCausalLM,
    get_shadow_model,
)
from shadow_peft_mlx.harness import (  # noqa: E402
    Router,
    RoutingRule,
    Session,
    ShadowEngineMlx,
    ShadowHarnessConfig,
    SpecialistConfig,
)


# --------------------------------------------------------------------------- #
# Tokenizer stub (mlx_lm tokenizers have encode/decode/eos_token_ids)
# --------------------------------------------------------------------------- #
class _TokenizerStub:
    def encode(self, text, add_special_tokens: bool = False):
        ids = [(ord(c) % 60 + 4) for c in text[:32]]
        return ids or [1]

    def decode(self, tokens):
        return "".join(chr(int(t) % 90 + 33) for t in tokens)

    @property
    def eos_token_ids(self):
        # A token id the stub encoder never emits, so generation runs to max_tokens.
        return {0}


# --------------------------------------------------------------------------- #
# Fixtures / builders
# --------------------------------------------------------------------------- #
def _build_task():
    mx.random.seed(0)
    base = make_llama()
    base.eval()
    peft = get_shadow_model(base, make_config())
    task = ShadowForCausalLM(peft, shadow_loss_weight=0.0)
    task.eval()
    return task


def _save_random_adapter(tmp_path, name: str, seed: int) -> str:
    """Build a fresh shadow adapter with a distinct RNG seed and save it."""
    mx.random.seed(seed)
    base = make_llama()
    base.eval()
    peft = get_shadow_model(base, make_config())
    peft.eval()
    save_dir = tmp_path / f"adapter_{name}"
    peft.save_pretrained(save_dir)
    return str(save_dir)


def _engine_with_two_specialists(tmp_path):
    task = _build_task()
    tokenizer = _TokenizerStub()
    path_a = _save_random_adapter(tmp_path, "alpha", seed=101)
    path_b = _save_random_adapter(tmp_path, "beta", seed=202)
    cfg = ShadowHarnessConfig(
        specialists=[
            SpecialistConfig(name="alpha", adapter_path=path_a, keywords=["alpha"]),
            SpecialistConfig(name="beta", adapter_path=path_b, keywords=["beta"]),
            SpecialistConfig(name="general"),
        ],
        default_specialist="general",
    )
    return ShadowEngineMlx(task, tokenizer, cfg), task


# --------------------------------------------------------------------------- #
# Router / session / config tests (pure Python, mirror of torch tests)
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


def test_router_session_sticky():
    r = Router(_router_cfg())
    sess = Session()
    sess.bind("code")
    d = r.route("now do something else", session=sess)
    assert d.specialist == "code"
    assert d.reason == "session_sticky"


def test_router_metadata_overrides_stickiness():
    r = Router(_router_cfg())
    sess = Session()
    sess.bind("code")
    d = r.route("anything", metadata={"domain": "health"}, session=sess)
    assert d.specialist == "medical"
    assert d.reason == "metadata_rule"


def test_session_add_turn_validates_role():
    s = Session()
    s.add_turn("user", "hi")
    s.add_turn("assistant", "hello")
    assert len(s.history) == 2
    with pytest.raises(ValueError):
        s.add_turn("system", "x")


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


# --------------------------------------------------------------------------- #
# Registry / engine integration tests (MLX model)
# --------------------------------------------------------------------------- #
def test_ensure_swaps_adapter_and_resets_shadow_cache(tmp_path):
    engine, task = _engine_with_two_specialists(tmp_path)
    peft = task.peft_model

    engine.registry.ensure("alpha")
    assert peft._runtime.shadow_cache is None
    alpha_params = {k: mx.array(v) for k, v in peft.adapter_parameters().items()}

    engine.registry.ensure("beta")
    assert peft._runtime.shadow_cache is None
    beta_params = {k: mx.array(v) for k, v in peft.adapter_parameters().items()}
    key = "shadow_model.layers.0.self_attn.q_proj.weight"
    assert not mx.array_equal(alpha_params[key], beta_params[key])


def test_swap_is_idempotent_when_already_active(tmp_path):
    engine, task = _engine_with_two_specialists(tmp_path)
    engine.registry.ensure("alpha")
    task.peft_model._runtime.shadow_cache = "sentinel"
    engine.registry.ensure("alpha")
    assert task.peft_model._runtime.shadow_cache == "sentinel"


def test_swap_output_matches_direct_load_baseline(tmp_path):
    """After swapping to a specialist, forward logits == a fresh model loaded with it."""
    engine, task = _engine_with_two_specialists(tmp_path)

    # Baseline: fresh model, load alpha directly.
    mx.random.seed(0)
    base_ref = make_llama()
    base_ref.eval()
    peft_ref = get_shadow_model(base_ref, make_config())
    alpha_path = engine.config.specialist("alpha").adapter_path
    peft_ref.load_weights(mx.load(f"{alpha_path}/shadow_adapter.safetensors"), strict=False)
    peft_ref.eval()

    inputs = mx.array([[3, 1, 4, 1, 5, 9]])
    ref_logits = peft_ref(inputs)
    mx.eval(ref_logits)

    engine.registry.ensure("alpha")
    task.set_inference_mode("base_shadow")
    live_logits = task.peft_model(inputs)
    mx.eval(live_logits)

    assert mx.allclose(ref_logits, live_logits, atol=1e-4, rtol=1e-3), (
        "Swapped-adapter output should match a fresh model loaded with the same adapter."
    )


def test_swap_ab_a_is_lossless(tmp_path):
    """alpha -> beta -> alpha: the second alpha logits match the first."""
    engine, task = _engine_with_two_specialists(tmp_path)
    inputs = mx.array([[3, 1, 4, 1, 5, 9]])

    def _logits():
        out = task.peft_model(inputs)
        mx.eval(out)
        return out

    engine.registry.ensure("alpha")
    first = _logits()
    engine.registry.ensure("beta")
    engine.registry.ensure("alpha")
    second = _logits()
    assert mx.allclose(first, second, atol=0.0, rtol=0.0)


def test_engine_generate_routes_and_returns_info(tmp_path):
    engine, _task = _engine_with_two_specialists(tmp_path)
    text, info = engine.generate("alpha please", max_tokens=3)
    assert info["specialist"] == "alpha"
    assert info["reason"] == "keyword"
    assert isinstance(text, str)


def test_engine_session_sticky_and_history(tmp_path):
    engine, _task = _engine_with_two_specialists(tmp_path)
    sess = engine.session()

    engine.generate("beta query", session=sess, max_tokens=2)
    assert sess.specialist == "beta"
    assert len(sess.history) == 2  # user + assistant

    _, info = engine.generate("unrelated text", session=sess, max_tokens=2)
    assert info["specialist"] == "beta"
    assert info["reason"] == "session_sticky"
    assert len(sess.history) == 4


def test_engine_history_isolation_across_sessions(tmp_path):
    engine, _task = _engine_with_two_specialists(tmp_path)
    s1 = engine.session()
    s2 = engine.session()

    engine.generate("alpha query", session=s1, max_tokens=1)
    engine.generate("beta query", session=s2, max_tokens=1)

    assert s1.history[0]["content"] == "alpha query"
    assert s2.history[0]["content"] == "beta query"
    assert len(s1.history) == 2
    assert len(s2.history) == 2
    assert s1.session_id != s2.session_id


def test_registry_unknown_specialist_raises(tmp_path):
    engine, _task = _engine_with_two_specialists(tmp_path)
    with pytest.raises(KeyError):
        engine.registry.ensure("nope")
