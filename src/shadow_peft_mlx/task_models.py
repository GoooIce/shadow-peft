from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from .model_utils import _get_inner_model
from .peft_model import ShadowPeftModel

InferenceMode = Literal["base_shadow", "shadow_only"]

_SHADOW_MODULES_FILE = "shadow_modules.safetensors"


@dataclass
class ShadowCausalLMOutput:
    loss: mx.array | None
    logits: mx.array
    shadow_logits: mx.array | None = None


@dataclass
class ShadowSequenceClassifierOutput:
    loss: mx.array | None
    logits: mx.array
    shadow_logits: mx.array | None = None


def _shifted_ce_loss(logits: mx.array, labels: mx.array) -> mx.array:
    """Standard causal LM loss: predict token t+1 from token t; -100 positions ignored."""
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    mask = shift_labels != -100
    safe_labels = mx.where(mask, shift_labels, 0)
    ce = nn.losses.cross_entropy(
        shift_logits.reshape(-1, shift_logits.shape[-1]),
        safe_labels.reshape(-1),
        reduction="none",
    )
    ce = ce * mask.reshape(-1).astype(ce.dtype)
    denom = mx.maximum(mask.sum(), 1)
    return ce.astype(mx.float32).sum() / denom


def _save_trainable_heads(
    save_directory: str | Path,
    heads: dict[str, nn.Module | None],
    requested: set[str],
) -> None:
    """
    Save trainable task heads (optionally restricted to `requested`), mirroring the
    torch `_save_modules_to_save` behavior.
    """
    state: dict[str, mx.array] = {}
    for name, module in heads.items():
        if module is None:
            continue
        if requested and name not in requested:
            continue
        if not tree_flatten(module.trainable_parameters()):
            continue
        for k, v in tree_flatten(module.parameters()):
            state[f"{name}.{k}"] = v
    if state:
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)
        mx.save_safetensors(str(save_dir / _SHADOW_MODULES_FILE), state)


def _load_heads_if_present(wrapper: nn.Module, pretrained_shadow_path: str | Path) -> None:
    st_path = Path(pretrained_shadow_path) / _SHADOW_MODULES_FILE
    if st_path.exists():
        wrapper.load_weights(str(st_path), strict=False)


class ShadowForCausalLM(nn.Module):
    """
    Task wrapper for causal LM (MLX version).

    - **base_shadow**: returns both `logits` (base path) and `shadow_logits` (shadow path).
    - **shadow_only**: returns `logits` equal to shadow logits (and also in `shadow_logits`).

    Loss (when `labels` given): `base_CE + shadow_loss_weight * shadow_CE`.
    """

    def __init__(
        self,
        peft_model: ShadowPeftModel,
        *,
        shadow_loss_weight: float = 0.05,
        inference_mode: InferenceMode = "base_shadow",
    ) -> None:
        super().__init__()
        self.peft_model = peft_model
        self.shadow_loss_weight = float(shadow_loss_weight)
        self.inference_mode: InferenceMode = inference_mode

        base_inner = _get_inner_model(peft_model.base_model)
        vocab_size = int(base_inner.args.vocab_size)
        hidden = peft_model.base_hidden_size

        # `lm_head` is a direct reference to the base model's head (None when the
        # architecture ties embeddings and scores via `embed_tokens.as_linear`).
        self.lm_head = getattr(peft_model.base_model, "lm_head", None)
        self.shadow_lm_head = nn.Linear(hidden, vocab_size, bias=False)
        if self.lm_head is not None:
            self.shadow_lm_head.weight = self.lm_head.weight * 1
        else:
            self.shadow_lm_head.weight = base_inner.embed_tokens.weight * 1

        # Heads are frozen by default (mirrors the torch implementation); unfreeze via
        # ShadowConfig.modules_to_save.
        self.shadow_lm_head.freeze()
        requested = set(peft_model.shadow_config.modules_to_save or [])
        if "shadow_lm_head" in requested:
            self.shadow_lm_head.unfreeze()
        if "lm_head" in requested:
            if self.lm_head is None:
                raise ValueError(
                    "modules_to_save requested 'lm_head', but the base model ties "
                    "embeddings and has no separate lm_head module."
                )
            self.lm_head.unfreeze()

    def set_inference_mode(self, mode: InferenceMode) -> None:
        self.inference_mode = mode

    def print_trainable_parameters(self) -> None:
        self.peft_model.print_trainable_parameters()

    def save_pretrained(self, save_directory: str | Path) -> None:
        self.peft_model.save_pretrained(save_directory)
        requested = set(self.peft_model.shadow_config.modules_to_save or [])
        _save_trainable_heads(
            save_directory,
            {"lm_head": self.lm_head, "shadow_lm_head": self.shadow_lm_head},
            requested,
        )

    @classmethod
    def from_pretrained(
        cls,
        model: nn.Module,
        pretrained_shadow_path: str | Path,
        *,
        is_trainable: bool = False,
        shadow_model: nn.Module | None = None,
        shadow_loss_weight: float = 0.05,
        inference_mode: InferenceMode = "base_shadow",
    ) -> ShadowForCausalLM:
        peft = ShadowPeftModel.from_pretrained(
            model,
            pretrained_shadow_path,
            is_trainable=is_trainable,
            shadow_model=shadow_model,
        )
        wrapper = cls(peft, shadow_loss_weight=shadow_loss_weight, inference_mode=inference_mode)
        _load_heads_if_present(wrapper, pretrained_shadow_path)
        return wrapper

    def __call__(
        self,
        input_ids: mx.array,
        labels: mx.array | None = None,
        cache: list | None = None,
    ) -> ShadowCausalLMOutput:
        if self.inference_mode == "shadow_only":
            shadow_hidden = self.peft_model._compute_shadow_hidden(input_ids, shadow_cache=None)
            shadow_logits = self.shadow_lm_head(shadow_hidden)
            loss = _shifted_ce_loss(shadow_logits, labels) if labels is not None else None
            return ShadowCausalLMOutput(loss=loss, logits=shadow_logits, shadow_logits=shadow_logits)

        logits, shadow_hidden = self.peft_model.forward_with_shadow(input_ids, cache=cache)
        shadow_logits = self.shadow_lm_head(shadow_hidden)

        loss = None
        if labels is not None:
            loss = _shifted_ce_loss(logits, labels)
            if self.shadow_loss_weight > 0:
                loss = loss + self.shadow_loss_weight * _shifted_ce_loss(shadow_logits, labels)

        return ShadowCausalLMOutput(loss=loss, logits=logits, shadow_logits=shadow_logits)

    def generate(self, prompt: mx.array, *, max_tokens: int = 256, sampler=None, **kwargs):
        """Convenience wrapper around `mlx_lm.generate.generate_step`."""
        from mlx_lm.generate import generate_step

        yield from generate_step(
            prompt, self.peft_model, max_tokens=max_tokens, sampler=sampler, **kwargs
        )


class ShadowForSequenceClassification(nn.Module):
    """
    Task wrapper for sequence classification (MLX version).

    mlx-lm ships no classification models, so the base path uses a freshly initialized
    `classifier_head` on top of the (wrapped) base backbone's last-token hidden state.
    Both heads are trainable by default (mirrors the torch implementation); restrict via
    `ShadowConfig.modules_to_save`.
    """

    def __init__(
        self,
        peft_model: ShadowPeftModel,
        num_labels: int,
        *,
        shadow_loss_weight: float = 0.05,
        inference_mode: InferenceMode = "base_shadow",
    ) -> None:
        super().__init__()
        self.peft_model = peft_model
        self.shadow_loss_weight = float(shadow_loss_weight)
        self.inference_mode: InferenceMode = inference_mode

        hidden = peft_model.base_hidden_size
        self.classifier_head = nn.Linear(hidden, num_labels)
        self.shadow_classifier_head = nn.Linear(hidden, num_labels)
        self.shadow_classifier_head.weight = self.classifier_head.weight * 1
        self.shadow_classifier_head.bias = self.classifier_head.bias * 1

        # Default: both heads trainable. An explicit non-empty modules_to_save overrides.
        requested = set(peft_model.shadow_config.modules_to_save or [])
        if requested:
            self.classifier_head.freeze()
            self.shadow_classifier_head.freeze()
            if "classifier_head" in requested:
                self.classifier_head.unfreeze()
            if "shadow_classifier_head" in requested:
                self.shadow_classifier_head.unfreeze()

    def set_inference_mode(self, mode: InferenceMode) -> None:
        self.inference_mode = mode

    def save_pretrained(self, save_directory: str | Path) -> None:
        self.peft_model.save_pretrained(save_directory)
        requested = set(self.peft_model.shadow_config.modules_to_save or [])
        _save_trainable_heads(
            save_directory,
            {
                "classifier_head": self.classifier_head,
                "shadow_classifier_head": self.shadow_classifier_head,
            },
            requested,
        )

    @classmethod
    def from_pretrained(
        cls,
        model: nn.Module,
        pretrained_shadow_path: str | Path,
        num_labels: int,
        *,
        is_trainable: bool = False,
        shadow_model: nn.Module | None = None,
        shadow_loss_weight: float = 0.05,
        inference_mode: InferenceMode = "base_shadow",
    ) -> ShadowForSequenceClassification:
        peft = ShadowPeftModel.from_pretrained(
            model,
            pretrained_shadow_path,
            is_trainable=is_trainable,
            shadow_model=shadow_model,
        )
        wrapper = cls(
            peft,
            num_labels,
            shadow_loss_weight=shadow_loss_weight,
            inference_mode=inference_mode,
        )
        _load_heads_if_present(wrapper, pretrained_shadow_path)
        return wrapper

    def __call__(
        self,
        input_ids: mx.array,
        labels: mx.array | None = None,
    ) -> ShadowSequenceClassifierOutput:
        if self.inference_mode == "shadow_only":
            shadow_hidden = self.peft_model._compute_shadow_hidden(input_ids, shadow_cache=None)
            shadow_logits = self.shadow_classifier_head(shadow_hidden[:, -1, :])
            loss = None
            if labels is not None:
                loss = nn.losses.cross_entropy(shadow_logits, labels, reduction="mean")
            return ShadowSequenceClassifierOutput(
                loss=loss, logits=shadow_logits, shadow_logits=shadow_logits
            )

        hidden, shadow_hidden = self.peft_model.forward_hidden(input_ids)
        base_logits = self.classifier_head(hidden[:, -1, :])
        shadow_logits = self.shadow_classifier_head(shadow_hidden[:, -1, :])

        loss = None
        if labels is not None:
            loss = nn.losses.cross_entropy(base_logits, labels, reduction="mean")
            if self.shadow_loss_weight > 0:
                loss = loss + self.shadow_loss_weight * nn.losses.cross_entropy(
                    shadow_logits, labels, reduction="mean"
                )

        return ShadowSequenceClassifierOutput(
            loss=loss, logits=base_logits, shadow_logits=shadow_logits
        )
