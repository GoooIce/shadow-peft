"""MLX backend public API with dependency-safe lazy imports."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from .export_2bit import save_1bit_mlx_checkpoint, save_2bit_mlx_checkpoint
from .version import __version__

_LAZY_IMPORTS = {
    "ProjectedCausalLM": (".projected_causal_lm", "ProjectedCausalLM"),
    "ShadowCausalLMOutput": (".task_models", "ShadowCausalLMOutput"),
    "ShadowConfig": (".config", "ShadowConfig"),
    "ShadowForCausalLM": (".task_models", "ShadowForCausalLM"),
    "ShadowForSequenceClassification": (
        ".task_models",
        "ShadowForSequenceClassification",
    ),
    "ShadowPeftModel": (".peft_model", "ShadowPeftModel"),
    "ShadowSequenceClassifierOutput": (
        ".task_models",
        "ShadowSequenceClassifierOutput",
    ),
    "compute_pinv_projection": (".projected_causal_lm", "compute_pinv_projection"),
    "convert_checkpoint": (".convert", "convert_checkpoint"),
    "ensure_1bit_support": (".quantization", "ensure_1bit_support"),
    "get_shadow_model": (".peft_model", "get_shadow_model"),
    "loss_fn": (".trainer", "loss_fn"),
    "pack_1bit_bonsai": (".quantization", "pack_1bit_bonsai"),
    "pack_1bit_refined": (".quantization", "pack_1bit_refined"),
    "prepare_shadow_model": (".peft_model", "prepare_shadow_model"),
    "quantize_model_1bit": (".quantization", "quantize_model_1bit"),
    "save_servable_model": (".model_utils", "save_servable_model"),
    "train": (".trainer", "train"),
}


def __getattr__(name: str) -> Any:
    """Load MLX-dependent symbols only when they are actually requested."""
    target = _LAZY_IMPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = target
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


__all__ = [
    "ProjectedCausalLM",
    "ShadowCausalLMOutput",
    "ShadowConfig",
    "ShadowForCausalLM",
    "ShadowForSequenceClassification",
    "ShadowPeftModel",
    "ShadowSequenceClassifierOutput",
    "__version__",
    "compute_pinv_projection",
    "convert_checkpoint",
    "ensure_1bit_support",
    "get_shadow_model",
    "loss_fn",
    "pack_1bit_bonsai",
    "pack_1bit_refined",
    "prepare_shadow_model",
    "quantize_model_1bit",
    "save_1bit_mlx_checkpoint",
    "save_2bit_mlx_checkpoint",
    "save_servable_model",
    "train",
]
