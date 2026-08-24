"""Gemma3/Gemma4 multimodal mount tests (ticket 01: `language_model` + `text_config`).

Gemma3 4B+ and all Gemma4 sizes load as multimodal wrappers whose decoder stack
lives at ``model.model.language_model.layers`` and whose hidden size lives under
``config.text_config``. These tests pin the resolved paths without downloading
any real checkpoint (12B weights are never touched).
"""

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from shadow_peft import ShadowConfig, ShadowPeftModel, get_shadow_model  # noqa: E402
from shadow_peft.model_utils import (  # noqa: E402
    _get_backbone,
    _get_decoder_layers,
    _get_hidden_size,
)

ShadowCfg = ShadowConfig

VOCAB = 128


def _shadow_cfg() -> ShadowConfig:
    return ShadowConfig(
        num_shadow_layers=1,
        injection_hidden_size=8,
        gate_hidden_size=10,
        alpha=0.1,
        dropout=0.0,
    )


def _tiny_gemma3_mm(num_layers: int = 4):
    Gemma3Config = getattr(transformers, "Gemma3Config", None)
    Gemma3ForConditionalGeneration = getattr(transformers, "Gemma3ForConditionalGeneration", None)
    if Gemma3Config is None or Gemma3ForConditionalGeneration is None:
        pytest.skip("Gemma3 multimodal classes are not available in this transformers version")
    from transformers.models.gemma3.configuration_gemma3 import Gemma3TextConfig

    text = Gemma3TextConfig(
        vocab_size=VOCAB,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=num_layers,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=8,
        max_position_embeddings=64,
        sliding_window=16,
    )
    return Gemma3ForConditionalGeneration(Gemma3Config(text_config=text))


def _tiny_gemma4_mm(num_layers: int = 4):
    Gemma4Config = getattr(transformers, "Gemma4Config", None)
    Gemma4ForConditionalGeneration = getattr(transformers, "Gemma4ForConditionalGeneration", None)
    if Gemma4Config is None or Gemma4ForConditionalGeneration is None:
        pytest.skip("Gemma4 classes are not available in this transformers version")
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig

    text = Gemma4TextConfig(
        vocab_size=VOCAB,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=num_layers,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=8,
        max_position_embeddings=64,
        sliding_window=16,
    )
    return Gemma4ForConditionalGeneration(Gemma4Config(text_config=text))


def _tiny_gemma4_unified(num_layers: int = 4):
    Gemma4UnifiedConfig = getattr(transformers, "Gemma4UnifiedConfig", None)
    Gemma4UnifiedForConditionalGeneration = getattr(
        transformers, "Gemma4UnifiedForConditionalGeneration", None
    )
    Gemma4UnifiedTextConfig = getattr(transformers, "Gemma4UnifiedTextConfig", None)
    if (
        Gemma4UnifiedConfig is None
        or Gemma4UnifiedForConditionalGeneration is None
        or Gemma4UnifiedTextConfig is None
    ):
        pytest.skip("Gemma4Unified classes are not available in this transformers version")
    text = Gemma4UnifiedTextConfig(
        vocab_size=VOCAB,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=num_layers,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=8,
        max_position_embeddings=64,
        sliding_window=16,
    )
    return Gemma4UnifiedForConditionalGeneration(Gemma4UnifiedConfig(text_config=text))


@pytest.mark.parametrize(
    "make_base",
    [_tiny_gemma3_mm, _tiny_gemma4_mm, _tiny_gemma4_unified],
    ids=["gemma3-mm", "gemma4-mm", "gemma4-unified"],
)
def test_backbone_resolution_pierces_multimodal_wrapper(make_base):
    torch.manual_seed(0)
    base = make_base(num_layers=4)
    backbone = _get_backbone(base)
    layers = getattr(backbone, "layers", None)
    assert isinstance(layers, torch.nn.ModuleList), (
        f"backbone resolved to {type(backbone).__name__} without a decoder-layer ModuleList"
    )
    assert len(layers) == 4

    resolved_backbone, resolved_layers, attr_name = _get_decoder_layers(base)
    assert resolved_layers is layers
    assert attr_name == "layers"
    # The resolved backbone must be the text model, not the multimodal container.
    assert "language_model" not in (
        resolved_backbone._modules.keys() if hasattr(resolved_backbone, "_modules") else []
    )


@pytest.mark.parametrize(
    "make_base",
    [_tiny_gemma3_mm, _tiny_gemma4_mm, _tiny_gemma4_unified],
    ids=["gemma3-mm", "gemma4-mm", "gemma4-unified"],
)
def test_hidden_size_reads_text_config(make_base):
    base = make_base()
    assert _get_hidden_size(base) == 32


@pytest.mark.parametrize(
    "make_base",
    [_tiny_gemma3_mm, _tiny_gemma4_mm, _tiny_gemma4_unified],
    ids=["gemma3-mm", "gemma4-mm", "gemma4-unified"],
)
def test_mount_forward_and_swap_roundtrip(make_base, tmp_path: Path):
    torch.manual_seed(0)
    base1 = make_base(num_layers=4)
    base1.eval()
    base_state = {k: v.detach().clone() for k, v in base1.state_dict().items()}

    m1 = get_shadow_model(base1, _shadow_cfg())
    m1.eval()
    input_ids = torch.randint(0, VOCAB, (2, 8))
    with torch.no_grad():
        out1 = m1(input_ids=input_ids).logits
    assert out1.shape == (2, 8, VOCAB)
    assert torch.isfinite(out1).all()

    save_dir = tmp_path / "shadow_adapter"
    m1.save_pretrained(save_dir)

    base2 = make_base(num_layers=4)
    base2.load_state_dict(base_state)
    base2.eval()
    m2 = ShadowPeftModel.from_pretrained(base2, save_dir, is_trainable=False)
    m2.eval()
    with torch.no_grad():
        out2 = m2(input_ids=input_ids).logits
    assert torch.allclose(out1, out2, atol=0.0, rtol=0.0)

    # swap_adapter: swapping the same adapter back in must reset shadow runtime
    # state and reproduce identical outputs.
    m2.swap_adapter(m2.adapter_state_dict())
    with torch.no_grad():
        out3 = m2(input_ids=input_ids).logits
    assert torch.allclose(out1, out3, atol=0.0, rtol=0.0)


def test_qwen3_zero_regression():
    torch.manual_seed(0)
    Qwen3Config = getattr(transformers, "Qwen3Config", None)
    Qwen3ForCausalLM = getattr(transformers, "Qwen3ForCausalLM", None)
    if Qwen3Config is None or Qwen3ForCausalLM is None:
        pytest.skip("Qwen3 is not available in this transformers version")
    cfg = Qwen3Config(
        vocab_size=VOCAB,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
    )
    base = Qwen3ForCausalLM(cfg)
    backbone, layers, _ = _get_decoder_layers(base)
    assert backbone is base.model
    assert len(layers) == 4
    assert _get_hidden_size(base) == 32

    m = get_shadow_model(base, _shadow_cfg())
    m.eval()
    input_ids = torch.randint(0, VOCAB, (1, 6))
    with torch.no_grad():
        out = m(input_ids=input_ids).logits
    assert out.shape == (1, 6, VOCAB)
