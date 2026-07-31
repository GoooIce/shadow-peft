from __future__ import annotations

from dataclasses import asdict

import pytest

pytest.importorskip("mlx")
pytest.importorskip("mlx_lm")

import mlx.core as mx  # noqa: E402
from mlx_lm.models import qwen3_5  # noqa: E402
from mlx_lm.models.cache import ArraysCache  # noqa: E402

from shadow_peft_mlx import (  # noqa: E402
    ShadowConfig,
    ShadowForCausalLM,
    get_shadow_model,
)


def test_qwen35_unified_wrapper_builds_reduced_shadow_and_resolves_head():
    text_args = qwen3_5.TextModelArgs(
        model_type="qwen3_5_text",
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        vocab_size=64,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        full_attention_interval=4,
    )
    model = qwen3_5.Model(qwen3_5.ModelArgs(model_type="qwen3_5", text_config=asdict(text_args)))
    config = ShadowConfig(
        num_shadow_layers=1,
        shadow_intermediate_size=32,
        injection_hidden_size=4,
        gate_hidden_size=4,
        dropout=0.0,
    )

    peft = get_shadow_model(model, config)
    task = ShadowForCausalLM(peft, shadow_loss_weight=0.0)

    assert len(peft.layers) == 4
    assert len(peft.shadow_model.layers) == 1
    assert peft.shadow_model.ssm_idx == 0
    assert peft.shadow_model.fa_idx == 1
    assert peft.base_hidden_size == 32
    assert task.lm_head is model.language_model.lm_head


def test_qwen35_shadow_uses_hybrid_cache_during_chunked_prefill():
    text_args = qwen3_5.TextModelArgs(
        model_type="qwen3_5_text",
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        vocab_size=64,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        full_attention_interval=4,
    )
    model = qwen3_5.Model(
        qwen3_5.ModelArgs(model_type="qwen3_5", text_config=asdict(text_args))
    )
    peft = get_shadow_model(
        model,
        ShadowConfig(
            num_shadow_layers=1,
            shadow_intermediate_size=32,
            injection_hidden_size=4,
            gate_hidden_size=4,
            dropout=0.0,
        ),
    )

    cache = peft.make_cache()
    first, _ = peft.forward_hidden(mx.array([[1, 2, 3]]), cache=cache)
    second, _ = peft.forward_hidden(mx.array([[4, 5]]), cache=cache)
    mx.eval(first, second)

    assert first.shape == (1, 3, 32)
    assert second.shape == (1, 2, 32)
    assert isinstance(peft._runtime.shadow_cache[0], ArraysCache)
    assert all(value is not None for value in peft._runtime.shadow_cache[0].cache)
