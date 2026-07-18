from .config import ShadowConfig
from .peft_model import ShadowPeftModel, get_shadow_model, prepare_shadow_model
from .projected_causal_lm import (
    AutoModelForCausalLMWithHiddenProjection,
    AutoModelForCausalLMWithHiddenProjectionConfig,
)
from .quantization import (
    QuantizedEmbedding1Bit,
    QuantizedLinear1Bit,
    dequantize_1bit_affine,
    quantize_1bit_affine,
    quantize_model_1bit,
    save_quantized_checkpoint,
)
from .task_models import ShadowForCausalLM, ShadowForSequenceClassification
from .version import __version__  # noqa: F401

__all__ = [
    "ShadowConfig",
    "AutoModelForCausalLMWithHiddenProjection",
    "AutoModelForCausalLMWithHiddenProjectionConfig",
    "QuantizedEmbedding1Bit",
    "QuantizedLinear1Bit",
    "ShadowForCausalLM",
    "ShadowForSequenceClassification",
    "ShadowPeftModel",
    "dequantize_1bit_affine",
    "get_shadow_model",
    "prepare_shadow_model",
    "quantize_1bit_affine",
    "quantize_model_1bit",
    "save_quantized_checkpoint",
]

# Register with transformers' Auto classes so that checkpoints with
# model_type="causal_lm_with_hidden_projection" are loaded automatically
# via AutoConfig.from_pretrained / AutoModelForCausalLM.from_pretrained.
from transformers import AutoConfig, AutoModelForCausalLM

AutoConfig.register(
    AutoModelForCausalLMWithHiddenProjectionConfig.model_type,
    AutoModelForCausalLMWithHiddenProjectionConfig,
)
AutoModelForCausalLM.register(
    AutoModelForCausalLMWithHiddenProjectionConfig,
    AutoModelForCausalLMWithHiddenProjection,
)
