"""Round-trip and validation tests for Torch 1-bit checkpoint loading."""

from __future__ import annotations

import json
import math

import pytest

torch = pytest.importorskip("torch")

from safetensors.torch import load_file as safetensors_load_file  # noqa: E402
from torch import nn  # noqa: E402

import shadow_peft.quantization.affine_1bit as affine_1bit  # noqa: E402
from shadow_peft import (  # noqa: E402
    QuantizedEmbedding1BitSign,
    QuantizedLinear1BitSign,
    load_quantized_checkpoint_1bit,
    quantize_model_1bit,
    rebuild_quantized_model_1bit,
    save_quantized_checkpoint,
)


class _ScaledEmbedding(nn.Embedding):
    def __init__(self, num_embeddings: int, embedding_dim: int, scale: float):
        super().__init__(num_embeddings, embedding_dim)
        self.scalar_embed_scale = scale
        self.register_buffer("embed_scale", torch.tensor(scale), persistent=False)

    def forward(self, input_ids):
        return super().forward(input_ids) * self.embed_scale.to(self.weight.dtype)


class _TinyTiedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = _ScaledEmbedding(16, 32, math.sqrt(32))
        self.proj = nn.Linear(32, 32, bias=True)
        self.norm = nn.LayerNorm(32)
        self.lm_head = nn.Linear(32, 16, bias=False)
        self.lm_head.weight = self.embed_tokens.weight

    def forward(self, input_ids):
        hidden = self.norm(self.proj(self.embed_tokens(input_ids)))
        return self.lm_head(hidden)


def _meta_tiny() -> _TinyTiedModel:
    with torch.device("meta"):
        return _TinyTiedModel().bfloat16().eval()


def test_torch_1bit_checkpoint_roundtrip_from_meta_skeleton(tmp_path):
    torch.manual_seed(101)
    model = _TinyTiedModel().bfloat16().eval()
    manifest = quantize_model_1bit(model, group_size=32, method="bonsai")
    input_ids = torch.tensor([[1, 1, 3], [2, 3, 2]])
    expected = model(input_ids)
    save_quantized_checkpoint(model, manifest, tmp_path)

    loaded = load_quantized_checkpoint_1bit(tmp_path, model=_meta_tiny())

    assert isinstance(loaded.embed_tokens, QuantizedEmbedding1BitSign)
    assert isinstance(loaded.proj, QuantizedLinear1BitSign)
    assert isinstance(loaded.lm_head, QuantizedLinear1BitSign)
    assert loaded.embed_tokens.scalar_embed_scale == pytest.approx(math.sqrt(32))
    assert loaded.embed_tokens.scales.dtype == torch.bfloat16
    assert loaded.norm.weight.dtype == torch.bfloat16
    assert loaded.embed_tokens.weight.device.type == "cpu"
    assert loaded.embed_tokens.weight.data_ptr() != loaded.lm_head.weight.data_ptr()
    torch.testing.assert_close(loaded(input_ids), expected, rtol=0, atol=0)


def test_torch_1bit_loader_accepts_in_memory_state_and_validates_before_mutation(tmp_path):
    model = _TinyTiedModel().bfloat16().eval()
    manifest = quantize_model_1bit(model, group_size=32, method="bonsai")
    save_quantized_checkpoint(model, manifest, tmp_path)
    state = safetensors_load_file(str(tmp_path / "model.safetensors"))

    loaded = load_quantized_checkpoint_1bit(
        state,
        model=_meta_tiny(),
        manifest=manifest,
    )
    assert isinstance(loaded.proj, QuantizedLinear1BitSign)

    invalid = dict(state)
    invalid["proj.weight"] = invalid["proj.weight"][:, :-1]
    untouched = _meta_tiny()
    with pytest.raises(ValueError, match="proj.weight.*shape"):
        rebuild_quantized_model_1bit(untouched, manifest, invalid)
    assert isinstance(untouched.proj, nn.Linear)

    wrong_module = json.loads(json.dumps(manifest))
    wrong_module["quantized"] = ["norm"]
    wrong_module["quantized_tied"] = []
    wrong_module["tied_to"] = {}
    with pytest.raises(TypeError, match="nn.Linear or nn.Embedding"):
        rebuild_quantized_model_1bit(_meta_tiny(), wrong_module)


def test_quantized_embedding_decodes_only_unique_rows(monkeypatch):
    torch.manual_seed(103)
    source = _ScaledEmbedding(128, 32, 3.0).bfloat16()
    module = QuantizedEmbedding1BitSign.from_embedding(source, group_size=32)
    input_ids = torch.tensor([[7, 7, 9], [11, 9, 7]])
    dense = affine_1bit.dequantize_1bit_affine(
        module.weight,
        module.scales,
        module.biases,
        module.group_size,
    )
    expected = nn.functional.embedding(input_ids, dense) * 3.0

    decoded_rows: list[int] = []
    original = affine_1bit.dequantize_1bit_affine

    def tracked(packed, scales, biases, group_size=128):
        decoded_rows.append(packed.shape[0])
        return original(packed, scales, biases, group_size)

    monkeypatch.setattr(affine_1bit, "dequantize_1bit_affine", tracked)
    actual = module(input_ids)

    assert decoded_rows == [3]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_gemma4_meta_rotary_buffers_are_recomputed():
    from transformers.models.gemma4_unified.configuration_gemma4_unified import (
        Gemma4UnifiedTextConfig,
    )
    from transformers.models.gemma4_unified.modeling_gemma4_unified import (
        Gemma4UnifiedTextRotaryEmbedding,
    )

    config = Gemma4UnifiedTextConfig(
        num_hidden_layers=2,
        layer_types=["sliding_attention", "full_attention"],
    )
    expected = Gemma4UnifiedTextRotaryEmbedding(config)
    with torch.device("meta"):
        actual = Gemma4UnifiedTextRotaryEmbedding(config)
    wrapper = nn.Module()
    wrapper.rotary = actual

    affine_1bit._materialize_gemma4_rotary_buffers(wrapper, torch.device("cpu"))

    for layer_type in config.layer_types:
        torch.testing.assert_close(
            getattr(actual, f"{layer_type}_inv_freq"),
            getattr(expected, f"{layer_type}_inv_freq"),
        )
        torch.testing.assert_close(
            getattr(actual, f"{layer_type}_original_inv_freq"),
            getattr(expected, f"{layer_type}_original_inv_freq"),
        )
