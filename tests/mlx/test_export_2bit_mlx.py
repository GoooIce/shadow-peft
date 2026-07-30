"""Tests for shadow_peft_mlx.export_2bit.save_2bit_mlx_checkpoint.

Uses a synthetic Gemma-like torch model (small, in-memory) to verify the
MLX container format written by the exporter: packed weight dtype = U32,
scales/biases = F16, config.json carries the quantization block, and the
safetensors keys follow the HF naming that mlx_vlm sanitize handles.
"""

from __future__ import annotations

import json
import struct

import pytest

torch = pytest.importorskip("torch")

import torch.nn as nn  # noqa: E402
from safetensors.torch import load_file as st_load  # noqa: E402

from shadow_peft.quantization import (  # noqa: E402
    quantize_model_2bit,
    save_quantized_checkpoint,
)
from shadow_peft_mlx import save_2bit_mlx_checkpoint  # noqa: E402


# --- synthetic Gemma-like model -------------------------------------------- #
class _GemmaLikeLayer(nn.Module):
    def __init__(self, hidden=256, inter=512, group_size=128):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(hidden)
        self.self_attn = nn.ModuleDict(
            {
                "q_proj": nn.Linear(hidden, hidden, bias=False),
                "o_proj": nn.Linear(hidden, hidden, bias=False),
            }
        )
        self.mlp = nn.ModuleDict(
            {
                "gate_proj": nn.Linear(hidden, inter, bias=False),
                "down_proj": nn.Linear(inter, hidden, bias=False),
            }
        )

    def forward(self, x):
        return x


class _GemmaLikeModel(nn.Module):
    """Minimal module tree with HF/Gemma naming: model.language_model.layers.*."""

    def __init__(self, hidden=256, inter=512):
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = nn.Module()
        self.model.language_model.embed_tokens = nn.Embedding(64, hidden)
        self.model.language_model.norm = nn.LayerNorm(hidden)
        self.model.language_model.layers = nn.ModuleList(
            [_GemmaLikeLayer(hidden, inter) for _ in range(2)]
        )
        self.lm_head = nn.Linear(hidden, 64, bias=False)
        self.lm_head.weight = self.model.language_model.embed_tokens.weight
        # config stand-in
        self.config = {
            "model_type": "gemma4_unified",
            "architectures": ["Gemma4UnifiedForConditionalGeneration"],
            "text_config": {"hidden_size": hidden, "num_hidden_layers": 2},
        }


def _quantize_and_export(tmp_path):
    # The production converter loads BF16.  The MLX exporter must still emit
    # FP16 scales/biases, as required by the Bonsai-style container.
    model = _GemmaLikeModel().bfloat16()
    manifest = quantize_model_2bit(model, group_size=128, refine_iters=0)
    mlx_dir = tmp_path / "mlx"
    torch_dir = tmp_path / "torch"
    info = save_2bit_mlx_checkpoint(model, model.config, manifest, mlx_dir, group_size=128)
    save_quantized_checkpoint(model, manifest, torch_dir)
    return mlx_dir, torch_dir, info, manifest


def _safetensors_header_dtype(path, key):
    """Read the safetensors header and return the dtype string for ``key``."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    return header[key]["dtype"]


# --- tests ----------------------------------------------------------------- #
class TestSave2BitMlxCheckpoint:
    def test_config_json_has_quantization_block(self, tmp_path):
        mlx_dir, _, info, _ = _quantize_and_export(tmp_path)
        cfg = json.loads((mlx_dir / "config.json").read_text())
        assert cfg["quantization"] == {"bits": 2, "group_size": 128, "mode": "affine"}
        assert cfg["text_config"]["quantization"] == cfg["quantization"]
        # original fields preserved
        assert cfg["model_type"] == "gemma4_unified"

    def test_packed_weight_dtype_is_u32(self, tmp_path):
        mlx_dir, _, info, _ = _quantize_and_export(tmp_path)
        st_path = mlx_dir / info["shards"][0]
        dtype = _safetensors_header_dtype(
            str(st_path), "model.language_model.layers.0.self_attn.q_proj.weight"
        )
        assert dtype == "U32"

    def test_scales_biases_dtype_is_f16(self, tmp_path):
        mlx_dir, _, info, _ = _quantize_and_export(tmp_path)
        st_path = mlx_dir / info["shards"][0]
        assert _safetensors_header_dtype(
            str(st_path), "model.language_model.layers.0.self_attn.q_proj.scales"
        ) == "F16"
        assert _safetensors_header_dtype(
            str(st_path), "model.language_model.layers.0.self_attn.q_proj.biases"
        ) == "F16"

    def test_torch_container_is_i32(self, tmp_path):
        _, torch_dir, _, _ = _quantize_and_export(tmp_path)
        dtype = _safetensors_header_dtype(
            str(torch_dir / "model.safetensors"),
            "model.language_model.layers.0.self_attn.q_proj.weight",
        )
        assert dtype == "I32"

    def test_quantized_count_matches_manifest(self, tmp_path):
        _, _, info, manifest = _quantize_and_export(tmp_path)
        assert info["quantized_count"] == len(manifest["quantized"])

    def test_unquantized_layers_kept(self, tmp_path):
        """Norm/embed scalars are NOT quantized, so must be present unmodified."""
        mlx_dir, _, info, _ = _quantize_and_export(tmp_path)
        tensors = st_load(str(mlx_dir / info["shards"][0]))
        # norm weight retained (not a quantized triplet — no .scales)
        assert "model.language_model.norm.weight" in tensors
        # embed_tokens IS quantized (it's nn.Embedding) -> has triplet
        assert "model.language_model.embed_tokens.weight" in tensors
        assert "model.language_model.embed_tokens.scales" in tensors

    def test_bit_pattern_identical_mlx_vs_torch(self, tmp_path):
        """MLX uint32 and torch int32 are the same 32 bits (zero-copy view)."""
        mlx_dir, torch_dir, info, _ = _quantize_and_export(tmp_path)
        key = "model.language_model.layers.0.self_attn.q_proj.weight"
        mlx_w = st_load(str(mlx_dir / info["shards"][0]))[key]
        torch_w = st_load(str(torch_dir / "model.safetensors"))[key]
        # view both as int32 to compare bit patterns
        assert torch.equal(mlx_w.view(torch.int32), torch_w)

    def test_can_omit_tied_head_for_embedding_scored_runtime(self, tmp_path):
        model = _GemmaLikeModel().bfloat16()
        manifest = quantize_model_2bit(model, group_size=128, method="bonsai")
        info = save_2bit_mlx_checkpoint(
            model,
            model.config,
            manifest,
            tmp_path,
            exclude_patterns=("lm_head.",),
        )

        assert sorted(info["excluded"]) == [
            "lm_head.biases",
            "lm_head.scales",
            "lm_head.weight",
        ]
        for shard in info["shards"]:
            assert all(
                not key.startswith("lm_head.") for key in st_load(str(tmp_path / shard))
            )
