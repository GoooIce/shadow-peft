from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

import torch.nn.functional as F  # noqa: E402
from safetensors.torch import load_file as safetensors_load_file  # noqa: E402
from torch import nn  # noqa: E402
from transformers import LlamaConfig, LlamaForCausalLM  # noqa: E402

from shadow_peft import (  # noqa: E402
    QuantizedEmbedding1Bit,
    QuantizedLinear1Bit,
    dequantize_1bit_affine,
    quantize_1bit_affine,
    quantize_model_1bit,
    save_quantized_checkpoint,
)


def _tiny_llama(tie_word_embeddings: bool = True) -> LlamaForCausalLM:
    cfg = LlamaConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
        tie_word_embeddings=tie_word_embeddings,
    )
    return LlamaForCausalLM(cfg)


def test_roundtrip_values_hit_group_extremes():
    torch.manual_seed(0)
    weight = torch.randn(4, 256, dtype=torch.float16)
    packed, scales, biases = quantize_1bit_affine(weight, group_size=128)
    w_hat = dequantize_1bit_affine(packed, scales, biases, group_size=128)

    groups = weight.reshape(4, 2, 128).float()
    w_min = groups.amin(dim=-1, keepdim=True)
    w_max = groups.amax(dim=-1, keepdim=True)
    flat = w_hat.reshape(4, 2, 128).float()
    dist = torch.minimum((flat - w_min).abs(), (flat - w_max).abs())
    # fp16 ulp near |w| <= 4 is ~0.004; allow a couple of ulps.
    assert dist.max().item() <= 1e-2


def test_packed_shapes_and_dtypes():
    torch.manual_seed(0)
    weight = torch.randn(8, 256, dtype=torch.float16)
    packed, scales, biases = quantize_1bit_affine(weight, group_size=128)
    assert packed.shape == (8, 8)  # 256 / 32
    assert packed.dtype == torch.int32
    assert scales.shape == (8, 2)  # 256 / 128
    assert biases.shape == (8, 2)
    assert scales.dtype == torch.float16
    assert biases.dtype == torch.float16

    # Non-fp16/bf16 inputs are cast to fp16.
    packed32, scales32, biases32 = quantize_1bit_affine(weight.float(), group_size=128)
    assert scales32.dtype == torch.float16
    assert torch.equal(packed, packed32)

    # bf16 input keeps bf16 scales.
    _, scales_bf16, _ = quantize_1bit_affine(weight.bfloat16(), group_size=128)
    assert scales_bf16.dtype == torch.bfloat16


def test_bit_order_lsb_first():
    # q[j] = 1 for odd j, 0 for even j (w_min=0, w_max=1, scale=1).
    weight = torch.arange(32, dtype=torch.float16).remainder(2).reshape(1, 32)
    packed, scales, biases = quantize_1bit_affine(weight, group_size=32)
    assert packed.shape == (1, 1)
    expected = torch.tensor(0xAAAAAAAA, dtype=torch.int64).to(torch.int32).item()
    assert packed[0, 0].item() == expected
    # Weight 0 occupies bit 0 (LSB-first).
    assert (packed[0, 0].item() >> 0) & 1 == 0
    assert (packed[0, 0].item() >> 1) & 1 == 1

    w_hat = dequantize_1bit_affine(packed, scales, biases, group_size=32)
    assert torch.equal(w_hat, weight)


def test_degenerate_group_constant_weights():
    weight = torch.full((2, 64), 3.25, dtype=torch.float16)
    packed, scales, biases = quantize_1bit_affine(weight, group_size=32)
    w_hat = dequantize_1bit_affine(packed, scales, biases, group_size=32)
    assert not torch.isnan(w_hat).any()
    assert torch.equal(w_hat, weight)


def test_quantize_rejects_bad_dims():
    with pytest.raises(ValueError):
        quantize_1bit_affine(torch.randn(4, 96, dtype=torch.float16), group_size=128)
    with pytest.raises(ValueError):
        quantize_1bit_affine(torch.randn(4, 96, dtype=torch.float16), group_size=48)


def test_quantized_linear_matches_dequantized_reference():
    torch.manual_seed(0)
    linear = nn.Linear(64, 16, bias=True)
    qlin = QuantizedLinear1Bit.from_linear(linear, group_size=32)
    x = torch.randn(3, 64, dtype=torch.float16)

    weight = dequantize_1bit_affine(qlin.weight, qlin.scales, qlin.biases, group_size=32)
    expected = F.linear(x, weight, qlin.bias)
    qlin.eval()
    with torch.no_grad():
        out = qlin(x)
    assert torch.equal(out, expected)


def test_quantized_linear_eval_cache_and_train_invalidation():
    torch.manual_seed(0)
    qlin = QuantizedLinear1Bit.from_linear(nn.Linear(64, 16), group_size=32)
    x = torch.randn(2, 64, dtype=torch.float16)

    qlin.eval()
    assert qlin._cached_weight is None
    with torch.no_grad():
        out1 = qlin(x)
        out2 = qlin(x)
    assert qlin._cached_weight is not None
    assert torch.equal(out1, out2)

    qlin.train()
    assert qlin._cached_weight is None
    out3 = qlin(x)
    assert qlin._cached_weight is None
    assert torch.equal(out1, out3)


def test_quantized_embedding_matches_dequantized_reference():
    torch.manual_seed(0)
    emb = nn.Embedding(64, 32, padding_idx=0)
    qemb = QuantizedEmbedding1Bit.from_embedding(emb, group_size=32)
    ids = torch.randint(0, 64, (2, 8))

    weight = dequantize_1bit_affine(qemb.weight, qemb.scales, qemb.biases, group_size=32)
    expected = F.embedding(ids, weight, padding_idx=0)
    assert torch.equal(qemb(ids), expected)
    assert qemb.padding_idx == 0


def test_quantize_model_replaces_layers_and_skips_tied_lm_head():
    torch.manual_seed(0)
    model = _tiny_llama(tie_word_embeddings=True).half()
    assert model.lm_head.weight is model.model.embed_tokens.weight

    manifest = quantize_model_1bit(model, group_size=32)

    assert isinstance(model.model.embed_tokens, QuantizedEmbedding1Bit)
    for layer in model.model.layers:
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            assert isinstance(getattr(layer.self_attn, name), QuantizedLinear1Bit)
        for name in ("gate_proj", "up_proj", "down_proj"):
            assert isinstance(getattr(layer.mlp, name), QuantizedLinear1Bit)

    # Tied lm_head is left as a plain Linear using the original fp weight.
    assert isinstance(model.lm_head, nn.Linear)
    assert manifest["tied"] == ["lm_head"]
    assert manifest["skipped"] == {}
    assert manifest["quantization"] == {
        "mode": "affine",
        "bits": 1,
        "group_size": 32,
        "refine_iters": 0,
        "trim_vocab_to": None,
    }
    assert "model.embed_tokens" in manifest["quantized"]

    model.eval()
    input_ids = torch.randint(0, model.config.vocab_size, (2, 8))
    with torch.no_grad():
        logits = model(input_ids=input_ids).logits
    assert logits.shape == (2, 8, model.config.vocab_size)


def test_quantize_model_skip_patterns_and_bad_dims():
    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Linear(64, 32),
        nn.Linear(48, 32),  # 48 % 32 != 0 -> skipped
        nn.Linear(64, 64),
    )
    manifest = quantize_model_1bit(model, group_size=32, skip_patterns=("2",))
    assert isinstance(model[0], QuantizedLinear1Bit)
    assert isinstance(model[1], nn.Linear)
    assert isinstance(model[2], nn.Linear)
    assert "not divisible" in manifest["skipped"]["1"]
    assert "skip pattern" in manifest["skipped"]["2"]
    assert manifest["quantized"] == ["0"]


def test_save_load_roundtrip(tmp_path: Path):
    torch.manual_seed(0)
    model1 = _tiny_llama().half()
    model1.eval()
    manifest = quantize_model_1bit(model1, group_size=32)
    input_ids = torch.randint(0, model1.config.vocab_size, (2, 8))
    with torch.no_grad():
        out1 = model1(input_ids=input_ids).logits

    save_dir = tmp_path / "quantized"
    save_quantized_checkpoint(model1, manifest, save_dir)
    assert (save_dir / "model.safetensors").exists()
    assert (save_dir / "quantization.json").exists()

    torch.manual_seed(0)
    model2 = _tiny_llama().half()
    quantize_model_1bit(model2, group_size=32)
    state = safetensors_load_file(str(save_dir / "model.safetensors"))
    model2.load_state_dict(state, strict=True)
    model2.eval()
    with torch.no_grad():
        out2 = model2(input_ids=input_ids).logits
    assert torch.equal(out1, out2)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_mps_matches_cpu():
    torch.manual_seed(0)
    weight = torch.randn(4, 256, dtype=torch.float16)
    packed_cpu, scales_cpu, biases_cpu = quantize_1bit_affine(weight, group_size=128)
    packed_mps, scales_mps, biases_mps = quantize_1bit_affine(
        weight.to("mps"), group_size=128
    )
    assert torch.equal(packed_cpu, packed_mps.cpu())
    assert torch.equal(scales_cpu, scales_mps.cpu())
    assert torch.equal(biases_cpu, biases_mps.cpu())

    w_cpu = dequantize_1bit_affine(packed_cpu, scales_cpu, biases_cpu, group_size=128)
    w_mps = dequantize_1bit_affine(
        packed_mps, scales_mps, biases_mps, group_size=128
    ).cpu()
    assert torch.equal(w_cpu, w_mps)


def _group_mse(weight: torch.Tensor, packed, scales, biases, group_size: int) -> float:
    w_hat = dequantize_1bit_affine(packed, scales, biases, group_size=group_size)
    return (w_hat.float() - weight.float()).pow(2).mean().item()


def test_refine_reduces_mse_and_is_monotone():
    torch.manual_seed(0)
    weight = torch.randn(8, 256, dtype=torch.float16)
    mses = []
    for iters in (0, 1, 2, 5):
        packed, scales, biases = quantize_1bit_affine(
            weight, group_size=128, refine_iters=iters
        )
        mses.append(_group_mse(weight, packed, scales, biases, 128))
    assert mses[1] < mses[0] * 0.5  # refinement removes most of the error
    assert all(b <= a + 1e-9 for a, b in zip(mses, mses[1:], strict=False))


def test_refine_format_and_determinism():
    torch.manual_seed(0)
    weight = torch.randn(8, 256, dtype=torch.float16)
    p1, s1, b1 = quantize_1bit_affine(weight, group_size=128, refine_iters=10)
    p2, s2, b2 = quantize_1bit_affine(weight, group_size=128, refine_iters=10)
    assert p1.shape == (8, 8) and p1.dtype == torch.int32
    assert s1.shape == b1.shape == (8, 2) and s1.dtype == torch.float16
    assert torch.equal(p1, p2) and torch.equal(s1, s2) and torch.equal(b1, b2)


def test_refine_degenerate_group_constant_weights():
    weight = torch.full((2, 64), 3.25, dtype=torch.float16)
    packed, scales, biases = quantize_1bit_affine(
        weight, group_size=32, refine_iters=10
    )
    w_hat = dequantize_1bit_affine(packed, scales, biases, group_size=32)
    assert not torch.isnan(w_hat).any()
    assert torch.allclose(w_hat, weight, atol=1e-3)


def test_refine_rejects_negative_iters():
    with pytest.raises(ValueError):
        quantize_1bit_affine(
            torch.randn(4, 128, dtype=torch.float16), group_size=128, refine_iters=-1
        )


def test_refine_manifest_records_iters():
    model = _tiny_llama().half()
    manifest = quantize_model_1bit(model, group_size=32, refine_iters=3)
    assert manifest["quantization"]["refine_iters"] == 3


def test_trim_vocab_embedding_and_tied_lm_head():
    torch.manual_seed(0)
    model = _tiny_llama(tie_word_embeddings=True).half()  # vocab 128, hidden 32
    manifest = quantize_model_1bit(model, group_size=32, trim_vocab_to=96)

    emb = model.model.embed_tokens
    assert isinstance(emb, QuantizedEmbedding1Bit)
    assert emb.num_embeddings == 96
    assert emb.weight.shape == (96, 1)  # hidden 32 -> 1 word per row

    # Tied lm_head is re-pointed at the trimmed fp weight (not quantized).
    head = model.lm_head
    assert isinstance(head, nn.Linear)
    assert not isinstance(head, QuantizedLinear1Bit)
    assert head.weight.shape == (96, 32)

    assert manifest["trimmed"]["model.embed_tokens"] == [128, 96]
    assert manifest["trimmed"]["lm_head"] == [128, 96]
    assert manifest["quantization"]["trim_vocab_to"] == 96
    assert manifest["tied"] == ["lm_head"]
    assert model.config.vocab_size == 96  # config kept consistent with the trim

    ids = torch.randint(0, 96, (2, 8))
    out = model(input_ids=ids)
    assert out.logits.shape == (2, 8, 96)
    assert torch.isfinite(out.logits).all()


def test_trim_vocab_to_none_by_default():
    torch.manual_seed(0)
    model = _tiny_llama().half()
    manifest = quantize_model_1bit(model, group_size=32)
    assert manifest["trimmed"] == {}
    assert model.model.embed_tokens.weight.shape == (128, 1)
