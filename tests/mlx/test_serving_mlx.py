"""
Serving round-trip test: an exported shadow model saved via `save_servable_model`
must load back through the standard `mlx_lm.utils.load_model` path (the same one
`mlx_lm.server` uses) and reproduce outputs bit-exactly.
"""

from __future__ import annotations

import pytest

pytest.importorskip("mlx")
pytest.importorskip("mlx_lm")

import mlx.core as mx  # noqa: E402
from mlx_lm.generate import generate_step  # noqa: E402
from mlx_lm.utils import load_model  # noqa: E402

from shadow_peft_mlx import get_shadow_model, save_servable_model  # noqa: E402


def test_exported_model_loadable_via_mlx_lm(tmp_path, llama_factory, shadow_cfg):
    peft = get_shadow_model(llama_factory(seed=40), shadow_cfg)
    peft.eval()
    exported = peft.export_shadow()
    exported.eval()

    save_servable_model(exported, tmp_path)
    assert (tmp_path / "config.json").exists()
    assert list(tmp_path.glob("model*.safetensors"))

    reloaded, config = load_model(tmp_path)
    reloaded.eval()
    assert config["model_type"] == "llama"
    assert config["num_hidden_layers"] == shadow_cfg.num_shadow_layers

    ids = mx.array([[1, 5, 7, 9, 3]])
    out_exported = exported(ids)
    out_reloaded = reloaded(ids)
    assert mx.abs(out_exported - out_reloaded).max().item() == 0.0

    # End-to-end generation works on the reloaded model (KV cache included).
    tokens = [t for t, _ in generate_step(mx.array([1, 5, 7]), reloaded, max_tokens=4)]
    assert len(tokens) == 4
