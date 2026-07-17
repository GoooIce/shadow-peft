from __future__ import annotations

import pytest

pytest.importorskip("mlx")
pytest.importorskip("mlx_lm")

import mlx.core as mx  # noqa: E402
from mlx_lm.models import llama, qwen2  # noqa: E402

from shadow_peft_mlx import ShadowConfig  # noqa: E402

VOCAB_SIZE = 64


def make_llama(
    *,
    num_layers: int = 3,
    hidden_size: int = 32,
    num_attention_heads: int = 4,
    num_key_value_heads: int = 2,
    vocab_size: int = VOCAB_SIZE,
):
    args = llama.ModelArgs(
        model_type="llama",
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        intermediate_size=hidden_size * 2,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        rms_norm_eps=1e-5,
        vocab_size=vocab_size,
    )
    return llama.Model(args)


def make_qwen2(
    *,
    num_layers: int = 3,
    hidden_size: int = 32,
    num_attention_heads: int = 4,
    num_key_value_heads: int = 2,
    vocab_size: int = VOCAB_SIZE,
):
    args = qwen2.ModelArgs(
        model_type="qwen2",
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        intermediate_size=hidden_size * 2,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        rms_norm_eps=1e-6,
        vocab_size=vocab_size,
    )
    return qwen2.Model(args)


def make_config(**overrides) -> ShadowConfig:
    base = {
        "num_shadow_layers": 1,
        "injection_hidden_size": 8,
        "gate_hidden_size": 10,
        "alpha": 0.1,
        "dropout": 0.0,
    }
    base.update(overrides)
    return ShadowConfig(**base)


@pytest.fixture
def shadow_cfg() -> ShadowConfig:
    return make_config()


@pytest.fixture
def llama_factory():
    def _make(seed: int | None = None, **kwargs):
        if seed is not None:
            mx.random.seed(seed)
        model = make_llama(**kwargs)
        model.eval()
        return model

    return _make


@pytest.fixture
def qwen2_factory():
    def _make(seed: int | None = None, **kwargs):
        if seed is not None:
            mx.random.seed(seed)
        model = make_qwen2(**kwargs)
        model.eval()
        return model

    return _make
