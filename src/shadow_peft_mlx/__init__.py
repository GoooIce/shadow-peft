from .config import ShadowConfig
from .peft_model import ShadowPeftModel, get_shadow_model, prepare_shadow_model
from .projected_causal_lm import ProjectedCausalLM, compute_pinv_projection
from .task_models import (
    ShadowCausalLMOutput,
    ShadowForCausalLM,
    ShadowForSequenceClassification,
    ShadowSequenceClassifierOutput,
)
from .trainer import loss_fn, train
from .version import __version__

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
    "get_shadow_model",
    "loss_fn",
    "prepare_shadow_model",
    "train",
]
