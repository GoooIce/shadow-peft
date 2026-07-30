"""Tests for the Bonsai-style binary g128 surrogate and MLX container."""

from __future__ import annotations

import json
import runpy
import struct
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from safetensors.torch import load_file as safetensors_load  # noqa: E402
from torch import nn  # noqa: E402

from shadow_peft import (  # noqa: E402
    QuantizedEmbedding1BitSign,
    QuantizedLinear1BitSign,
    dequantize_1bit_affine,
    quantize_1bit_bonsai,
    quantize_1bit_sign,
    quantize_model_1bit,
)


def _unpack_bits(packed: torch.Tensor) -> torch.Tensor:
    shifts = torch.arange(32, dtype=torch.int32, device=packed.device)
    return ((packed.unsqueeze(-1) >> shifts) & 1).reshape(packed.shape[0], -1)


def _report_reference(
    weight: torch.Tensor,
    group_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Small scalar packer following the public Bonsai/Q1_0 representation."""
    dtype = (
        weight.dtype
        if weight.dtype in (torch.float16, torch.bfloat16)
        else torch.float16
    )
    rows, cols = weight.shape
    groups = weight.float().reshape(rows, cols // group_size, group_size)
    d = groups.abs().mean(dim=-1)
    bits = (groups >= 0).to(torch.int64).reshape(rows, cols)

    words: list[list[int]] = []
    for row in bits.tolist():
        packed_row = []
        for start in range(0, cols, 32):
            word = 0
            for offset, bit in enumerate(row[start : start + 32]):
                word |= bit << offset
            packed_row.append(word if word < 2**31 else word - 2**32)
        words.append(packed_row)

    scales = (2 * d).to(dtype)
    return torch.tensor(words, dtype=torch.int32), scales, -scales / 2


def test_bonsai_1bit_defaults_match_independent_reference():
    torch.manual_seed(23)
    weight = torch.randn(3, 256, dtype=torch.float16)

    actual = quantize_1bit_bonsai(weight)
    expected = _report_reference(weight)

    for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
        assert torch.equal(actual_tensor, expected_tensor)
    assert torch.equal(_unpack_bits(actual[0]), (weight >= 0).to(torch.int32))


def test_bonsai_1bit_lsb_first_word():
    values = torch.tensor(
        [-1.0 if index % 3 == 0 else 1.0 for index in range(32)],
        dtype=torch.float16,
    ).unsqueeze(0)

    packed, _, _ = quantize_1bit_bonsai(values, group_size=32)

    expected = sum((int(value >= 0) << index) for index, value in enumerate(values[0]))
    assert (packed.item() & 0xFFFFFFFF) == expected


def test_bonsai_1bit_uses_fp32_signs_before_fp16_storage_cast():
    weight = torch.ones(1, 32, dtype=torch.float32)
    weight[0, 0] = -1e-8

    packed, scales, _ = quantize_1bit_bonsai(weight, group_size=32)

    assert (packed.item() & 1) == 0
    assert scales.dtype == torch.float16


def test_bonsai_1bit_zero_group_reconstructs_zero_exactly():
    weight = torch.zeros(2, 128, dtype=torch.bfloat16)
    packed, scales, biases = quantize_1bit_bonsai(weight)
    restored = dequantize_1bit_affine(packed, scales, biases)

    assert (_unpack_bits(packed) == 1).all()
    assert torch.equal(scales, torch.zeros_like(scales))
    assert torch.equal(biases, torch.zeros_like(biases))
    assert torch.equal(restored, weight)


def test_bonsai_1bit_refinement_is_a_fixed_point_and_sign_alias_matches():
    torch.manual_seed(29)
    weight = torch.randn(4, 256, dtype=torch.float16)

    baseline = quantize_1bit_bonsai(weight, refine_iters=0)
    refined = quantize_1bit_bonsai(weight, refine_iters=8)
    legacy_sign = quantize_1bit_sign(weight, refine_iters=8)

    for expected, actual, alias in zip(baseline, refined, legacy_sign, strict=True):
        assert torch.equal(expected, actual)
        assert torch.equal(expected, alias)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"group_size": 0}, "group_size"),
        ({"group_size": 48}, "group_size"),
        ({"refine_iters": -1}, "refine_iters"),
    ],
)
def test_bonsai_1bit_rejects_invalid_options(kwargs, message):
    with pytest.raises(ValueError, match=message):
        quantize_1bit_bonsai(torch.randn(1, 128), **kwargs)


def test_bonsai_1bit_rejects_scale_overflow():
    weight = torch.full((1, 32), 40000.0, dtype=torch.float16)
    with pytest.raises(ValueError, match="scale is not finite"):
        quantize_1bit_bonsai(weight, group_size=32)


def test_quantize_model_1bit_bonsai_accepts_float32_activations_and_preserves_eval():
    torch.manual_seed(31)
    model = nn.Sequential(nn.Linear(128, 32, bias=True)).eval()

    manifest = quantize_model_1bit(model, method="bonsai")
    output = model(torch.randn(2, 128))

    assert isinstance(model[0], QuantizedLinear1BitSign)
    assert not model[0].training
    assert output.dtype == torch.float32
    assert torch.isfinite(output).all()
    assert manifest["quantization"]["variant"] == "sign"
    assert manifest["quantization"]["method"] == "bonsai"
    assert manifest["quantization"]["refine_iters"] == 0
    assert manifest["quantization"]["quantize_tied_weights"] is True


def test_quantize_model_1bit_bonsai_quantizes_tied_head_separately():
    class TiedModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.Embedding(64, 32)
            self.lm_head = nn.Linear(32, 64, bias=False)
            self.lm_head.weight = self.embed_tokens.weight

    model = TiedModel().half().eval()
    manifest = quantize_model_1bit(model, group_size=32, method="bonsai")

    assert isinstance(model.embed_tokens, QuantizedEmbedding1BitSign)
    assert isinstance(model.lm_head, QuantizedLinear1BitSign)
    assert not model.embed_tokens.training
    assert not model.lm_head.training
    assert manifest["tied"] == []
    assert manifest["quantized_tied"] == ["lm_head"]
    assert manifest["tied_to"] == {"lm_head": "embed_tokens"}
    assert torch.equal(model.embed_tokens.weight, model.lm_head.weight)
    assert torch.equal(model.embed_tokens.scales, model.lm_head.scales)
    assert model.embed_tokens.weight.data_ptr() != model.lm_head.weight.data_ptr()


def test_quantize_model_1bit_legacy_defaults_remain_compatible():
    class TiedModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.Embedding(64, 32)
            self.lm_head = nn.Linear(32, 64, bias=False)
            self.lm_head.weight = self.embed_tokens.weight

    default_model = TiedModel().half()
    default_manifest = quantize_model_1bit(default_model, group_size=32)
    assert default_manifest["quantization"] == {
        "mode": "affine",
        "bits": 1,
        "variant": "affine",
        "group_size": 32,
        "refine_iters": 0,
        "trim_vocab_to": None,
    }
    assert default_manifest["tied"] == ["lm_head"]

    sign_model = TiedModel().half()
    sign_manifest = quantize_model_1bit(sign_model, group_size=32, variant="sign")
    assert sign_manifest["quantization"]["variant"] == "sign"
    assert "method" not in sign_manifest["quantization"]
    assert sign_manifest["tied"] == ["lm_head"]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"group_size": 0}, "group_size"),
        ({"group_size": 48}, "group_size"),
        ({"refine_iters": -1}, "refine_iters"),
        ({"method": "official"}, "method"),
        ({"variant": "ternary"}, "variant"),
        ({"variant": "sign", "method": "affine"}, "conflicts"),
    ],
)
def test_quantize_model_1bit_validates_before_model_mutation(kwargs, message):
    with pytest.raises(ValueError, match=message):
        quantize_model_1bit(nn.Identity(), **kwargs)


def test_quantized_1bit_load_state_dict_invalidates_eval_cache():
    torch.manual_seed(37)
    module = QuantizedLinear1BitSign.from_linear(nn.Linear(128, 32).half()).eval()
    inputs = torch.randn(2, 128, dtype=torch.float16)
    before = module(inputs)
    assert module._cached_weight is not None

    state = {name: tensor.clone() for name, tensor in module.state_dict().items()}
    state["scales"].mul_(2)
    state["biases"].mul_(2)
    module.load_state_dict(state)

    assert module._cached_weight is None
    after = module(inputs)
    assert not torch.equal(before, after)


def test_bonsai_1bit_trim_preserves_padding_idx_and_rejects_removed_index():
    class EmbeddingModel(nn.Module):
        def __init__(self, padding_idx: int):
            super().__init__()
            self.embed_tokens = nn.Embedding(10, 32, padding_idx=padding_idx)

    model = EmbeddingModel(padding_idx=3).half()
    quantize_model_1bit(
        model,
        group_size=32,
        trim_vocab_to=8,
        method="bonsai",
    )
    assert isinstance(model.embed_tokens, QuantizedEmbedding1BitSign)
    assert model.embed_tokens.padding_idx == 3

    with pytest.raises(ValueError, match="padding_idx"):
        quantize_model_1bit(
            EmbeddingModel(padding_idx=9).half(),
            group_size=32,
            trim_vocab_to=8,
            method="bonsai",
        )


def _load_exporter():
    root = Path(__file__).resolve().parents[1]
    exports = runpy.run_path(str(root / "src/shadow_peft_mlx/export_2bit.py"))
    return exports["save_1bit_mlx_checkpoint"]


def _header_dtype(path: Path, key: str) -> str:
    with path.open("rb") as handle:
        header_size = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_size))
    return header[key]["dtype"]


def test_bonsai_exporter_public_import_does_not_require_mlx():
    from shadow_peft_mlx import save_1bit_mlx_checkpoint

    assert callable(save_1bit_mlx_checkpoint)


def test_bonsai_1bit_mlx_export_contract(tmp_path):
    class TiedModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.Embedding(64, 32)
            self.proj = nn.Linear(32, 32, bias=False)
            self.norm = nn.LayerNorm(32)
            self.lm_head = nn.Linear(32, 64, bias=False)
            self.lm_head.weight = self.embed_tokens.weight

    model = TiedModel().bfloat16().eval()
    manifest = quantize_model_1bit(model, group_size=32, method="bonsai")
    torch_state = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    config = {"model_type": "synthetic", "text_config": {"hidden_size": 32}}
    original_config = json.loads(json.dumps(config))

    save_1bit = _load_exporter()
    info = save_1bit(
        model,
        config,
        manifest,
        tmp_path,
        group_size=32,
    )

    shard = tmp_path / info["shards"][0]
    exported = safetensors_load(str(shard))
    assert info["bits"] == 1
    assert _header_dtype(shard, "embed_tokens.weight") == "U32"
    assert _header_dtype(shard, "embed_tokens.scales") == "F16"
    assert _header_dtype(shard, "embed_tokens.biases") == "F16"
    assert _header_dtype(shard, "norm.weight") == "F16"
    assert torch.equal(
        exported["embed_tokens.weight"].view(torch.int32),
        torch_state["embed_tokens.weight"],
    )
    torch.testing.assert_close(
        exported["embed_tokens.biases"],
        -exported["embed_tokens.scales"] / 2,
        rtol=0,
        atol=0,
    )
    assert all(not key.startswith("lm_head.") for key in exported)
    assert any(key.startswith("lm_head.") for key in info["excluded"])

    saved_config = json.loads((tmp_path / "config.json").read_text())
    expected_quantization = {"bits": 1, "group_size": 32}
    assert saved_config["quantization"] == expected_quantization
    assert saved_config["text_config"]["quantization"] == expected_quantization
    assert "mode" not in saved_config["quantization"]
    assert config == original_config


def test_bonsai_1bit_export_keeps_embedding_when_registered_after_head(tmp_path):
    class ReverseTiedModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.lm_head = nn.Linear(32, 64, bias=False)
            self.embed_tokens = nn.Embedding(64, 32)
            self.embed_tokens.weight = self.lm_head.weight

    model = ReverseTiedModel().half()
    manifest = quantize_model_1bit(model, group_size=32, method="bonsai")
    assert manifest["tied_to"] == {"embed_tokens": "lm_head"}

    info = _load_exporter()(
        model,
        {"tie_word_embeddings": True},
        manifest,
        tmp_path,
        group_size=32,
    )
    exported = safetensors_load(str(tmp_path / info["shards"][0]))

    assert "embed_tokens.weight" in exported
    assert "embed_tokens.scales" in exported
    assert all(not key.startswith("lm_head.") for key in exported)


def test_bonsai_1bit_export_rejects_manifest_mismatch(tmp_path):
    model = nn.Sequential(nn.Linear(32, 32, bias=False)).half()
    manifest = quantize_model_1bit(model, group_size=32, method="bonsai")
    save_1bit = _load_exporter()

    with pytest.raises(ValueError, match="group_size"):
        save_1bit(model, {}, manifest, tmp_path, group_size=64)

    missing_bits = json.loads(json.dumps(manifest))
    del missing_bits["quantization"]["bits"]
    with pytest.raises(ValueError, match="bits"):
        save_1bit(model, {}, missing_bits, tmp_path / "missing-bits", group_size=32)

    model_96 = nn.Sequential(nn.Linear(96, 32, bias=False)).half()
    manifest_96 = quantize_model_1bit(
        model_96, group_size=96, method="bonsai"
    )
    with pytest.raises(ValueError, match="32, 64, or 128"):
        save_1bit(model_96, {}, manifest_96, tmp_path / "g96", group_size=96)


def test_bonsai_1bit_export_rejects_affine_or_asymmetric_state(tmp_path):
    save_1bit = _load_exporter()

    affine_model = nn.Sequential(nn.Linear(32, 32, bias=False)).half()
    affine_manifest = quantize_model_1bit(affine_model, group_size=32)
    with pytest.raises(ValueError, match="Bonsai/sign manifest"):
        save_1bit(affine_model, {}, affine_manifest, tmp_path / "affine", group_size=32)

    bonsai_model = nn.Sequential(nn.Linear(32, 32, bias=False)).half()
    bonsai_manifest = quantize_model_1bit(
        bonsai_model, group_size=32, method="bonsai"
    )
    with torch.no_grad():
        bonsai_model[0].biases[0, 0] += torch.tensor(0.125, dtype=torch.float16)
    with pytest.raises(ValueError, match="bias == -scale / 2"):
        save_1bit(
            bonsai_model,
            {},
            bonsai_manifest,
            tmp_path / "asymmetric",
            group_size=32,
        )

    clean_model = nn.Sequential(nn.Linear(32, 32, bias=False)).half()
    clean_manifest = quantize_model_1bit(
        clean_model, group_size=32, method="bonsai"
    )
    with pytest.raises(ValueError, match="partially remove"):
        save_1bit(
            clean_model,
            {},
            clean_manifest,
            tmp_path / "partial-triplet",
            group_size=32,
            exclude_patterns=("biases",),
        )

    two_layer_model = nn.Sequential(
        nn.Linear(32, 32, bias=False),
        nn.Linear(32, 32, bias=False),
    ).half()
    incomplete_manifest = quantize_model_1bit(
        two_layer_model, group_size=32, method="bonsai"
    )
    incomplete_manifest["quantized"].pop()
    with pytest.raises(ValueError, match="missing from manifest"):
        save_1bit(
            two_layer_model,
            {},
            incomplete_manifest,
            tmp_path / "unlisted",
            group_size=32,
        )


def test_bonsai_1bit_export_syncs_trimmed_vocab_config(tmp_path):
    model = nn.Sequential(nn.Embedding(64, 32)).half()
    manifest = quantize_model_1bit(
        model,
        group_size=32,
        method="bonsai",
        trim_vocab_to=48,
    )
    config = {"vocab_size": 64, "text_config": {"vocab_size": 64}}
    info = _load_exporter()(
        model,
        config,
        manifest,
        tmp_path,
        group_size=32,
    )

    assert info["config"]["vocab_size"] == 48
    assert info["config"]["text_config"]["vocab_size"] == 48
    assert config == {"vocab_size": 64, "text_config": {"vocab_size": 64}}


@pytest.mark.parametrize(
    ("trim_vocab_to", "skip_patterns"),
    [(80, ()), (48, ("embed_tokens",))],
)
def test_bonsai_1bit_export_does_not_sync_unapplied_trim(
    tmp_path, trim_vocab_to, skip_patterns
):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.Embedding(64, 32)
            self.proj = nn.Linear(32, 32, bias=False)

    model = Model().half()
    manifest = quantize_model_1bit(
        model,
        group_size=32,
        method="bonsai",
        trim_vocab_to=trim_vocab_to,
        skip_patterns=skip_patterns,
    )
    assert manifest["trimmed"] == {}

    config = {"vocab_size": 64, "text_config": {"vocab_size": 64}}
    info = _load_exporter()(
        model,
        config,
        manifest,
        tmp_path,
        group_size=32,
    )

    assert info["config"]["vocab_size"] == 64
    assert info["config"]["text_config"]["vocab_size"] == 64
