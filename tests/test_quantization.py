from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

import torch.nn.functional as F  # noqa: E402
from torch import nn  # noqa: E402
from transformers import LlamaConfig, LlamaForCausalLM  # noqa: E402

from shadow_peft import (  # noqa: E402
    QuantizedEmbedding1Bit,
    QuantizedEmbedding2Bit,
    QuantizedLinear1Bit,
    QuantizedLinear2Bit,
    dequantize_1bit_affine,
    dequantize_2bit_ternary,
    quantize_1bit_affine,
    quantize_2bit_ternary,
    quantize_model_1bit,
    quantize_model_2bit,
)
from shadow_peft.quantization import (  # noqa: E402
    AFFINE_1BIT,
    TERNARY_2BIT,
    load_quantized_checkpoint,
    quantize_model,
    save_quantized_checkpoint,
    unpack_codes,
)
from shadow_peft.quantization.modules import classes_for_scheme  # noqa: E402

SCHEMES = [AFFINE_1BIT, TERNARY_2BIT]
scheme_ids = [s.label for s in SCHEMES]


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


def _expected_quantization_meta(scheme, **overrides) -> dict:
    meta = {
        "mode": "affine",
        "bits": scheme.bits,
        "group_size": 32,
        "refine_iters": 0,
        "trim_vocab_to": None,
    }
    if scheme.ternary:
        meta = {
            "mode": "affine",
            "bits": scheme.bits,
            "ternary": True,
            "group_size": 32,
            "refine_iters": 0,
            "thr_ratio": 0.5,
            "trim_vocab_to": None,
        }
    meta.update(overrides)
    return meta


# ---------------------------------------------------------------------------
# The codec: unpack_codes is the single home of the shift-and-mask idiom
# ---------------------------------------------------------------------------


def test_unpack_codes_lsb_first_bits1():
    # codes 1,0,1,0,... pack to 0xAAAAAAAA (weight j occupies bit j).
    codes = torch.tensor([[1, 0, 1, 0]], dtype=torch.int32)
    packed = (codes * 2 ** torch.arange(4, dtype=torch.int64)).sum(-1).to(torch.int32)
    unpacked = unpack_codes(packed, 1)
    assert unpacked.shape == (1, 32)  # 32 one-bit codes per word
    assert unpacked[0, :4].tolist() == [1, 0, 1, 0]
    assert unpacked[0, 4:].sum() == 0  # padding slots are zero


def test_unpack_codes_lsb_first_bits2():
    # 2-bit codes c_j occupy bits 2j and 2j+1.
    codes = torch.tensor([[0, 1, 2, 1]], dtype=torch.int32)
    packed = (codes * 4 ** torch.arange(4, dtype=torch.int64)).sum(-1).to(torch.int32)
    unpacked = unpack_codes(packed, 2)
    assert unpacked.shape == (1, 16)  # 16 two-bit codes per word
    assert unpacked[0, :4].tolist() == [0, 1, 2, 1]
    assert unpacked[0, 4:].sum() == 0  # padding slots are zero


def test_unpack_codes_rejects_bad_bits():
    with pytest.raises(ValueError):
        unpack_codes(torch.zeros(1, 1, dtype=torch.int32), 5)


# ---------------------------------------------------------------------------
# Parametrized over the scheme table: one test surface for every scheme
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scheme", SCHEMES, ids=scheme_ids)
def test_packed_shapes_and_dtypes(scheme):
    torch.manual_seed(0)
    weight = torch.randn(8, 256, dtype=torch.float16)
    packed, scales, biases = scheme.quantize(weight, group_size=128, refine_iters=0, thr_ratio=0.5)
    assert packed.shape == (8, 256 // scheme.words_per_int)
    assert packed.dtype == torch.int32
    assert scales.shape == (8, 2)
    assert biases.shape == (8, 2)
    assert scales.dtype == torch.float16
    assert biases.dtype == torch.float16

    # Non-fp16/bf16 inputs are cast to fp16.
    _, scales32, _ = scheme.quantize(weight.float(), group_size=128, refine_iters=0, thr_ratio=0.5)
    assert scales32.dtype == torch.float16

    # bf16 input keeps bf16 scales.
    _, scales_bf16, _ = scheme.quantize(weight.bfloat16(), group_size=128, refine_iters=0, thr_ratio=0.5)
    assert scales_bf16.dtype == torch.bfloat16


@pytest.mark.parametrize("scheme", SCHEMES, ids=scheme_ids)
def test_quantize_rejects_bad_dims(scheme):
    with pytest.raises(ValueError):
        scheme.quantize(torch.randn(4, 96, dtype=torch.float16), group_size=128, refine_iters=0, thr_ratio=0.5)
    with pytest.raises(ValueError):
        scheme.quantize(
            torch.randn(4, 96, dtype=torch.float16),
            group_size=scheme.group_multiple + scheme.group_multiple // 2,  # not a multiple
            refine_iters=0,
            thr_ratio=0.5,
        )
    with pytest.raises(ValueError):
        scheme.quantize(torch.randn(4, 128, dtype=torch.float16), group_size=128, refine_iters=-1, thr_ratio=0.5)


@pytest.mark.parametrize("scheme", SCHEMES, ids=scheme_ids)
def test_dequantized_weight_matches_codes_through_codec(scheme):
    """dequant must agree with unpack_codes + the affine formula."""
    torch.manual_seed(0)
    weight = torch.randn(4, 128, dtype=torch.float16)
    packed, scales, biases = scheme.quantize(weight, group_size=128, refine_iters=0, thr_ratio=0.5)
    w_hat = scheme.dequantize(packed, scales, biases, group_size=128)

    codes = unpack_codes(packed, scheme.bits).reshape(4, 1, 128).to(scales.dtype)
    w_ref = (scales.unsqueeze(-1) * codes + biases.unsqueeze(-1)).reshape(4, 128)
    assert torch.equal(w_hat, w_ref)


@pytest.mark.parametrize("scheme", SCHEMES, ids=scheme_ids)
def test_refine_reduces_mse_and_is_monotone(scheme):
    torch.manual_seed(0)
    weight = torch.randn(8, 256, dtype=torch.float16)
    mses = []
    for iters in (0, 1, 2, 5):
        packed, scales, biases = scheme.quantize(weight, group_size=128, refine_iters=iters, thr_ratio=0.5)
        w_hat = scheme.dequantize(packed, scales, biases, group_size=128)
        mses.append((w_hat.float() - weight.float()).pow(2).mean().item())
    assert mses[1] < mses[0] * 0.95  # refinement removes most of the error
    assert all(b <= a + 1e-7 for a, b in zip(mses, mses[1:], strict=False))


@pytest.mark.parametrize("scheme", SCHEMES, ids=scheme_ids)
def test_refine_format_and_determinism(scheme):
    torch.manual_seed(0)
    weight = torch.randn(8, 256, dtype=torch.float16)
    p1, s1, b1 = scheme.quantize(weight, group_size=128, refine_iters=10, thr_ratio=0.5)
    p2, s2, b2 = scheme.quantize(weight, group_size=128, refine_iters=10, thr_ratio=0.5)
    assert p1.shape == (8, 256 // scheme.words_per_int) and p1.dtype == torch.int32
    assert s1.shape == b1.shape == (8, 2) and s1.dtype == torch.float16
    assert torch.equal(p1, p2) and torch.equal(s1, s2) and torch.equal(b1, b2)


@pytest.mark.parametrize("scheme", SCHEMES, ids=scheme_ids)
def test_quantized_linear_matches_dequantized_reference(scheme):
    torch.manual_seed(0)
    linear_cls, _ = classes_for_scheme(scheme)
    linear = nn.Linear(64, 16, bias=True)
    qlin = linear_cls.from_linear(linear, group_size=32)
    x = torch.randn(3, 64, dtype=torch.float16)

    weight = qlin.dequantize_weight()
    expected = F.linear(x, weight, qlin.bias)
    qlin.eval()
    with torch.no_grad():
        out = qlin(x)
    assert torch.equal(out, expected)


@pytest.mark.parametrize("scheme", SCHEMES, ids=scheme_ids)
def test_quantized_linear_eval_cache_and_train_invalidation(scheme):
    torch.manual_seed(0)
    linear_cls, _ = classes_for_scheme(scheme)
    qlin = linear_cls.from_linear(nn.Linear(64, 16), group_size=32)
    x = torch.randn(2, 64, dtype=torch.float16)

    qlin.eval()
    assert qlin._cached_weight is None
    with torch.no_grad():
        out1 = qlin(x)
        out2 = qlin(x)
    assert qlin._cached_weight is not None
    assert torch.equal(out1, out2)

    qlin.train()
    assert qlin._cached_weight is None  # train mode invalidates the cache
    out3 = qlin(x)
    assert qlin._cached_weight is None
    assert torch.equal(out1, out3)


@pytest.mark.parametrize("scheme", SCHEMES, ids=scheme_ids)
def test_quantized_embedding_matches_dequantized_reference(scheme):
    torch.manual_seed(0)
    _, embedding_cls = classes_for_scheme(scheme)
    emb = nn.Embedding(64, 32, padding_idx=0)
    qemb = embedding_cls.from_embedding(emb, group_size=32)
    ids = torch.randint(0, 64, (2, 8))

    expected = F.embedding(ids, qemb.dequantize_weight(), padding_idx=0)
    assert torch.equal(qemb(ids), expected)
    assert qemb.padding_idx == 0


@pytest.mark.parametrize("scheme", SCHEMES, ids=scheme_ids)
def test_from_linear_thr_ratio_default_is_unified(scheme):
    """thr_ratio defaults to 0.5 everywhere (the 0.7 legacy default is gone)."""
    import inspect

    linear_cls, embedding_cls = classes_for_scheme(scheme)
    assert inspect.signature(linear_cls.from_linear).parameters["thr_ratio"].default == 0.5
    assert inspect.signature(embedding_cls.from_embedding).parameters["thr_ratio"].default == 0.5


@pytest.mark.parametrize("scheme", SCHEMES, ids=scheme_ids)
def test_quantize_model_replaces_layers_and_skips_tied_lm_head(scheme):
    torch.manual_seed(0)
    model = _tiny_llama(tie_word_embeddings=True).half()
    assert model.lm_head.weight is model.model.embed_tokens.weight

    manifest = quantize_model(model, scheme, group_size=32)

    linear_cls, embedding_cls = classes_for_scheme(scheme)
    assert isinstance(model.model.embed_tokens, embedding_cls)
    for layer in model.model.layers:
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            assert isinstance(getattr(layer.self_attn, name), linear_cls)
        for name in ("gate_proj", "up_proj", "down_proj"):
            assert isinstance(getattr(layer.mlp, name), linear_cls)

    # Tied lm_head is left as a plain Linear using the original fp weight.
    assert isinstance(model.lm_head, nn.Linear)
    assert manifest["tied"] == ["lm_head"]
    assert manifest["skipped"] == {}
    assert manifest["quantization"] == _expected_quantization_meta(scheme)
    assert "model.embed_tokens" in manifest["quantized"]

    model.eval()
    input_ids = torch.randint(0, model.config.vocab_size, (2, 8))
    with torch.no_grad():
        logits = model(input_ids=input_ids).logits
    assert logits.shape == (2, 8, model.config.vocab_size)


@pytest.mark.parametrize("scheme", SCHEMES, ids=scheme_ids)
def test_quantize_model_skip_patterns_and_bad_dims(scheme):
    torch.manual_seed(0)
    linear_cls, _ = classes_for_scheme(scheme)
    model = nn.Sequential(
        nn.Linear(64, 32),
        nn.Linear(48, 32),  # 48 % 32 != 0 -> skipped
        nn.Linear(64, 64),
    )
    manifest = quantize_model(model, scheme, group_size=32, skip_patterns=("2",))
    assert isinstance(model[0], linear_cls)
    assert isinstance(model[1], nn.Linear)
    assert isinstance(model[2], nn.Linear)
    assert "not divisible" in manifest["skipped"]["1"]
    assert "skip pattern" in manifest["skipped"]["2"]
    assert manifest["quantized"] == ["0"]


@pytest.mark.parametrize("scheme", SCHEMES, ids=scheme_ids)
def test_trim_vocab_embedding_and_tied_lm_head(scheme):
    torch.manual_seed(0)
    model = _tiny_llama(tie_word_embeddings=True).half()  # vocab 128, hidden 32
    manifest = quantize_model(model, scheme, group_size=32, trim_vocab_to=96)

    linear_cls, embedding_cls = classes_for_scheme(scheme)
    emb = model.model.embed_tokens
    assert isinstance(emb, embedding_cls)
    assert emb.num_embeddings == 96
    assert emb.weight.shape == (96, 32 // scheme.words_per_int)

    # Tied lm_head is re-pointed at the trimmed fp weight (not quantized).
    head = model.lm_head
    assert isinstance(head, nn.Linear)
    assert not isinstance(head, linear_cls)
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


@pytest.mark.parametrize("scheme", SCHEMES, ids=scheme_ids)
def test_trim_vocab_to_none_by_default(scheme):
    torch.manual_seed(0)
    model = _tiny_llama().half()
    manifest = quantize_model(model, scheme, group_size=32)
    assert manifest["trimmed"] == {}
    assert model.model.embed_tokens.weight.shape == (128, 32 // scheme.words_per_int)


@pytest.mark.parametrize("scheme", SCHEMES, ids=scheme_ids)
def test_refine_manifest_records_iters(scheme):
    model = _tiny_llama().half()
    manifest = quantize_model(model, scheme, group_size=32, refine_iters=3)
    assert manifest["quantization"]["refine_iters"] == 3


@pytest.mark.parametrize("scheme", SCHEMES, ids=scheme_ids)
def test_save_load_roundtrip_through_loader(scheme, tmp_path: Path):
    """The real save -> load pair: a fresh model rebuilt from the checkpoint
    reproduces the packed model bit-for-bit (logits identical)."""
    torch.manual_seed(0)
    model1 = _tiny_llama().half()
    model1.eval()
    manifest = quantize_model(model1, scheme, group_size=32, refine_iters=2)
    input_ids = torch.randint(0, model1.config.vocab_size, (2, 8))
    with torch.no_grad():
        out1 = model1(input_ids=input_ids).logits

    save_dir = tmp_path / "quantized"
    save_quantized_checkpoint(model1, manifest, save_dir)
    assert (save_dir / "model.safetensors").exists()
    assert (save_dir / "quantization.json").exists()

    torch.manual_seed(1234)  # different init weights: they must be overwritten
    model2 = _tiny_llama().half()
    loaded_manifest = load_quantized_checkpoint(model2, save_dir)
    assert loaded_manifest["quantization"]["bits"] == scheme.bits
    model2.eval()
    with torch.no_grad():
        out2 = model2(input_ids=input_ids).logits
    assert torch.equal(out1, out2)


@pytest.mark.parametrize("scheme", SCHEMES, ids=scheme_ids)
def test_save_load_roundtrip_with_skip_patterns_derived_from_manifest(scheme, tmp_path: Path):
    """A checkpoint saved with custom skip patterns loads with DEFAULT loader
    arguments: the loader derives the skips from the manifest itself."""
    torch.manual_seed(0)
    model1 = _tiny_llama().half()
    model1.eval()
    skip = "model.layers.0"
    manifest = quantize_model(model1, scheme, group_size=32, skip_patterns=(skip,))
    # The pattern skips modules that would otherwise be quantized.
    assert isinstance(model1.model.layers[0].self_attn.q_proj, nn.Linear)
    linear_cls, _ = classes_for_scheme(scheme)
    assert isinstance(model1.model.layers[1].self_attn.q_proj, linear_cls)

    input_ids = torch.randint(0, model1.config.vocab_size, (2, 8))
    with torch.no_grad():
        out1 = model1(input_ids=input_ids).logits

    save_dir = tmp_path / "quantized"
    save_quantized_checkpoint(model1, manifest, save_dir)

    torch.manual_seed(1234)  # different init weights: they must be overwritten
    model2 = _tiny_llama().half()
    loaded_manifest = load_quantized_checkpoint(model2, save_dir)  # default args
    # (a) the skipped module stays a plain Linear, (b) others are quantized.
    assert isinstance(model2.model.layers[0].self_attn.q_proj, nn.Linear)
    assert not isinstance(model2.model.layers[0].self_attn.q_proj, linear_cls)
    assert isinstance(model2.model.layers[1].self_attn.q_proj, linear_cls)
    assert loaded_manifest["quantization"]["bits"] == scheme.bits
    model2.eval()
    with torch.no_grad():
        out2 = model2(input_ids=input_ids).logits
    assert torch.equal(out1, out2)


@pytest.mark.parametrize("scheme", SCHEMES, ids=scheme_ids)
def test_legacy_named_entrypoints_match_scheme_table(scheme):
    """quantize_model_1bit / quantize_model_2bit are the scheme table."""
    if scheme.bits == 1:
        fn = quantize_model_1bit
    else:
        fn = quantize_model_2bit
    torch.manual_seed(0)
    model_a = _tiny_llama().half()
    model_b = _tiny_llama().half()
    manifest_a = fn(model_a, group_size=32)
    manifest_b = quantize_model(model_b, scheme, group_size=32)
    assert manifest_a == manifest_b


# ---------------------------------------------------------------------------
# Scheme-specific format pins
# ---------------------------------------------------------------------------


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


def test_refine_degenerate_group_constant_weights():
    weight = torch.full((2, 64), 3.25, dtype=torch.float16)
    packed, scales, biases = quantize_1bit_affine(weight, group_size=32, refine_iters=10)
    w_hat = dequantize_1bit_affine(packed, scales, biases, group_size=32)
    assert not torch.isnan(w_hat).any()
    assert torch.allclose(w_hat, weight, atol=1e-3)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
@pytest.mark.parametrize("scheme", SCHEMES, ids=scheme_ids)
def test_mps_matches_cpu(scheme):
    torch.manual_seed(0)
    weight = torch.randn(4, 256, dtype=torch.float16)
    packed_cpu, scales_cpu, biases_cpu = scheme.quantize(weight, group_size=128, refine_iters=0, thr_ratio=0.5)
    packed_mps, scales_mps, biases_mps = scheme.quantize(
        weight.to("mps"), group_size=128, refine_iters=0, thr_ratio=0.5
    )
    assert torch.equal(packed_cpu, packed_mps.cpu())
    assert torch.equal(scales_cpu, scales_mps.cpu())
    assert torch.equal(biases_cpu, biases_mps.cpu())

    w_cpu = scheme.dequantize(packed_cpu, scales_cpu, biases_cpu, group_size=128)
    w_mps = scheme.dequantize(packed_mps, scales_mps, biases_mps, group_size=128).cpu()
    assert torch.equal(w_cpu, w_mps)


# --- ternary 2-bit pins -----------------------------------------------------


def test_code_3_never_produced():
    """Safety: unpacked codes must only be in {0, 1, 2}, never 3."""
    torch.manual_seed(0)
    weight = torch.randn(8, 256, dtype=torch.float16)
    packed, _, _ = quantize_2bit_ternary(weight, group_size=128)
    codes = unpack_codes(packed, 2)
    assert codes.max().item() <= 2


def test_dequant_values_are_ternary():
    """Dequantized weights must be exactly {-s_g, 0, +s_g} per group."""
    torch.manual_seed(0)
    weight = torch.randn(8, 256, dtype=torch.float16)
    packed, scales, biases = quantize_2bit_ternary(weight, group_size=128)
    w_hat = dequantize_2bit_ternary(packed, scales, biases, group_size=128)

    for g in range(2):
        sg = scales[0, g].item()
        group_vals = w_hat[0, g * 128 : (g + 1) * 128].float()
        unique_vals = group_vals.unique()
        assert len(unique_vals) <= 3
        for v in unique_vals:
            assert any(abs(v.item() - t) < 1e-3 for t in (-sg, 0.0, sg))


def test_bias_is_negative_scale():
    """bias must equal -scale (so dequant = scale*(code-1))."""
    torch.manual_seed(0)
    weight = torch.randn(4, 128, dtype=torch.float16)
    _, scales, biases = quantize_2bit_ternary(weight, group_size=128)
    torch.testing.assert_close(biases.float(), -scales.float(), rtol=0, atol=1e-3)


def test_roundtrip_mse_better_than_max():
    """Round-trip error should be much less than the weight magnitude."""
    torch.manual_seed(0)
    weight = torch.randn(8, 256, dtype=torch.float16)
    packed, scales, biases = quantize_2bit_ternary(weight, group_size=128, refine_iters=8)
    w_hat = dequantize_2bit_ternary(packed, scales, biases, group_size=128)
    mse = ((weight.float() - w_hat.float()) ** 2).mean().item()
    signal = weight.float().var().item()
    assert mse < signal  # error smaller than signal variance


def test_quantize_rejects_bad_thr_ratio():
    with pytest.raises(ValueError):
        quantize_2bit_ternary(torch.randn(4, 128, dtype=torch.float16), group_size=128, thr_ratio=0.0)
    with pytest.raises(ValueError):
        quantize_2bit_ternary(torch.randn(4, 128, dtype=torch.float16), group_size=128, thr_ratio=1.5)


def test_manifest_records_thr_ratio():
    model = _tiny_llama().half()
    manifest = quantize_model_2bit(model, group_size=32, thr_ratio=0.6)
    assert manifest["quantization"]["thr_ratio"] == 0.6


# --- compatibility surface ---------------------------------------------------


def test_scheme_pinned_class_names_importable():
    """The per-bit class names remain real classes (isinstance works)."""
    assert issubclass(QuantizedLinear1Bit, __import__(
        "shadow_peft.quantization", fromlist=["QuantizedLinear"]
    ).QuantizedLinear)
    assert issubclass(QuantizedLinear2Bit, __import__(
        "shadow_peft.quantization", fromlist=["QuantizedLinear"]
    ).QuantizedLinear)
    assert issubclass(QuantizedEmbedding1Bit, __import__(
        "shadow_peft.quantization", fromlist=["QuantizedEmbedding"]
    ).QuantizedEmbedding)
    assert issubclass(QuantizedEmbedding2Bit, __import__(
        "shadow_peft.quantization", fromlist=["QuantizedEmbedding"]
    ).QuantizedEmbedding)


def test_quantized_embedding_1bit_has_dequantize_weight_parity():
    """The 1-bit Embedding exposes dequantize_weight() like its 2-bit twin."""
    torch.manual_seed(0)
    qemb = QuantizedEmbedding1Bit.from_embedding(nn.Embedding(64, 32), group_size=32)
    w = qemb.dequantize_weight()
    assert w.shape == (64, 32)
    expected = dequantize_1bit_affine(qemb.weight, qemb.scales, qemb.biases, group_size=32)
    assert torch.equal(w, expected)


# --- 8-bit affine (shadow-side PTQ) -------------------------------------------


def test_affine_8bit_scheme_row():
    """get_scheme(8) returns the INT8 row with correct table fields and dim checks."""
    from shadow_peft.quantization import AFFINE_8BIT, get_scheme

    s = get_scheme(8)
    assert s is AFFINE_8BIT
    assert (s.bits, s.label, s.levels, s.words_per_int, s.ternary) == (
        8, "affine_8bit", 256, 4, False,
    )
    s.check_dims(128, 32)
    s.check_dims(128, 128)
    with pytest.raises(ValueError):
        s.check_dims(128, 33)
    with pytest.raises(ValueError):
        s.check_dims(130, 128)


def test_affine_8bit_roundtrip_within_half_step():
    """Packing (4 codes per int32 word) and roundtrip error stay within the
    8-bit affine half-step (plus fp16 scale/bias storage slack)."""
    from shadow_peft.quantization import dequantize_8bit_affine, quantize_8bit_affine

    torch.manual_seed(0)
    w = torch.randn(16, 256) * 0.01
    packed, scales, biases = quantize_8bit_affine(w, group_size=128)
    assert packed.dtype == torch.int32
    assert packed.shape == (16, 64)  # 256 values / 4 per int32 word
    assert scales.shape == biases.shape == (16, 2)

    w_hat = dequantize_8bit_affine(packed, scales, biases, group_size=128)
    half_step = scales.float() / 2.0  # scale = (max-min)/255 per group
    err = (w - w_hat).abs().reshape(16, 2, 128)
    assert (err <= half_step.unsqueeze(-1) + 5e-4).all()


def test_quantize_model_8bit_walker():
    """The walker drives the 8-bit row like the 1/2-bit ones: in-place module
    replacement, packing density, forward stays close to fp."""
    from shadow_peft.quantization import QuantizedLinear8Bit, quantize_model_8bit

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(256, 128), nn.Linear(128, 64)).half()
    x = torch.randn(4, 256).half()
    ref = model(x)

    manifest = quantize_model_8bit(model, group_size=128)
    assert manifest["quantization"]["bits"] == 8
    assert manifest["quantization"]["group_size"] == 128
    assert len(manifest["quantized"]) == 2

    q0 = model[0]
    assert isinstance(q0, QuantizedLinear8Bit)
    assert q0.weight.shape == (128, 64)  # 4 codes per int32 word

    out = model(x)
    assert torch.allclose(out, ref, atol=0.05)


def test_unpack_codes_bits8_registry_path():
    """bits=8 resolves through the registered scheme table (not the raw
    whitelist) and keeps the LSB-first 4-codes-per-word layout."""
    from shadow_peft.quantization import unpack_codes

    word = 1 | (2 << 8) | (3 << 16) | (250 << 24)
    word_signed = word - (1 << 32) if word >= (1 << 31) else word  # two's complement
    packed = torch.tensor([[word_signed]], dtype=torch.int32)
    codes = unpack_codes(packed, 8)
    assert codes.shape == (1, 1, 4)
    assert codes.flatten().tolist() == [1, 2, 3, 250]
