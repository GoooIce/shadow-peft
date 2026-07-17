"""Tests for the MLX ShadowPeftModel: wrapping, freezing, identity, save/load."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("mlx")
pytest.importorskip("mlx_lm")

import mlx.core as mx  # noqa: E402
from mlx.utils import tree_flatten  # noqa: E402

from shadow_peft_mlx import ShadowConfig, ShadowPeftModel, get_shadow_model  # noqa: E402
from shadow_peft_mlx.peft_model import _ShadowLayerWrapper  # noqa: E402


def test_wraps_and_freezes_base(llama_factory, shadow_cfg):
    base = llama_factory()
    peft = get_shadow_model(base, shadow_cfg)

    # Every decoder layer is wrapped (layer 0 as a no-op passthrough).
    assert all(isinstance(layer, _ShadowLayerWrapper) for layer in peft.layers)

    # Base contributes no trainable parameters; adapters do.
    assert not tree_flatten(peft.base_model.trainable_parameters())
    assert tree_flatten(peft.shadow_model.trainable_parameters())
    assert tree_flatten(peft.shadow_injection_model.trainable_parameters())
    assert tree_flatten(peft.shadow_update_model.trainable_parameters())


def test_identity_after_wrap_llama(llama_factory, shadow_cfg):
    base = llama_factory(seed=1)
    ids = mx.array([[1, 5, 7, 9, 3]])
    ref = base(ids)
    peft = get_shadow_model(base, shadow_cfg)
    peft.eval()
    out = peft(ids)
    assert mx.abs(out - ref).max().item() == 0.0


def test_identity_after_wrap_qwen2(qwen2_factory, shadow_cfg):
    base = qwen2_factory(seed=1)
    ids = mx.array([[1, 5, 7, 9, 3]])
    ref = base(ids)
    peft = get_shadow_model(base, shadow_cfg)
    peft.eval()
    out = peft(ids)
    assert mx.abs(out - ref).max().item() == 0.0


def test_save_load_roundtrip_matches_outputs(tmp_path: Path, llama_factory, shadow_cfg):
    base = llama_factory(seed=3)
    peft = get_shadow_model(base, shadow_cfg)
    peft.eval()

    ids = mx.array([[1, 5, 7, 9, 3], [2, 4, 6, 8, 10]])
    out1 = peft(ids)

    save_dir = tmp_path / "shadow_adapter"
    peft.save_pretrained(save_dir)
    assert (save_dir / "shadow_config.json").exists()
    assert (save_dir / "shadow_adapter.safetensors").exists()

    # Reload onto the same base instance (re-wrap replaces stale wrappers in-place).
    peft2 = ShadowPeftModel.from_pretrained(base, save_dir, is_trainable=False)
    peft2.eval()
    out2 = peft2(ids)
    assert mx.abs(out1 - out2).max().item() == 0.0


def test_from_pretrained_trainability(tmp_path: Path, llama_factory, shadow_cfg):
    base = llama_factory(seed=4)
    peft = get_shadow_model(base, shadow_cfg)
    save_dir = tmp_path / "ckpt"
    peft.save_pretrained(save_dir)

    frozen = ShadowPeftModel.from_pretrained(base, save_dir, is_trainable=False)
    assert not tree_flatten(frozen.trainable_parameters())

    trainable = ShadowPeftModel.from_pretrained(base, save_dir, is_trainable=True)
    assert tree_flatten(trainable.trainable_parameters())
    # Base stays frozen either way.
    assert not tree_flatten(trainable.base_model.trainable_parameters())


def test_adapter_parameters_excludes_base(llama_factory, shadow_cfg):
    peft = get_shadow_model(llama_factory(), shadow_cfg)
    keys = set(peft.adapter_parameters().keys())
    assert keys
    assert all(
        k.startswith(
            (
                "shadow_model.",
                "shadow_hidden_projection.",
                "shadow_injection_model.",
                "shadow_update_model.",
            )
        )
        for k in keys
    )


def test_config_json_roundtrip(tmp_path: Path):
    cfg = ShadowConfig(
        num_shadow_layers=2,
        injection_hidden_size=32,
        gate_hidden_size=12,
        alpha=0.2,
        dropout=0.1,
        shadow_intermediate_size=48,
        modules_to_save=["shadow_lm_head"],
    )
    cfg.save_pretrained(tmp_path)
    loaded = ShadowConfig.from_pretrained(tmp_path)
    assert loaded == cfg


def test_requires_at_least_two_layers(llama_factory, shadow_cfg):
    base = llama_factory(num_layers=1)
    with pytest.raises(ValueError, match="at least 2 decoder layers"):
        get_shadow_model(base, shadow_cfg)


def test_print_trainable_parameters(llama_factory, shadow_cfg, capsys):
    peft = get_shadow_model(llama_factory(), shadow_cfg)
    peft.print_trainable_parameters()
    out = capsys.readouterr().out
    assert "Trainable params:" in out
