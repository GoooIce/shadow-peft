"""Shadow-aware SFTTrainer subclasses for benchmark training and evaluation.

Requires ``trl``. The MMLU evaluation_loop lives here exactly once (the
canonical generation-based loop); the GSM8K trainer takes an
``answer_extractor`` seam for boxed/standard extraction strategies.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import torch
import torch.nn.functional as F
from transformers import Trainer
from transformers.trainer_utils import EvalLoopOutput

from shadow_peft.task_models import _shifted_kd_loss

try:
    from trl import SFTTrainer
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The `trl` package is required for the eval trainers. Install it with `pip install trl`."
    ) from exc

from .eval import (
    DEFAULT_GSM8K_GENERATION_TOKENS,
    DEFAULT_MMLU_GENERATION_TOKENS,
    DEFAULT_SQUAD_V2_GENERATION_TOKENS,
    _clean_generated_text,
    _shifted_ce_loss,
    _unwrap_model,
    extract_answer_from_text,
    extract_gsm8k_answer_from_text,
)

# ---------------------------------------------------------------------------
# Trainer mixins
# ---------------------------------------------------------------------------

SHADOW_PARAM_PATTERNS = (
    "shadow_model.",
    "shadow_lm_head",
    "shadow_adapter",
    "shadow_proj",
    "shadow_hidden_projection",
    "shadow_classifier",
    # The joint GKD loss mainly trains these two adapter groups.
    "shadow_injection_model.",
    "shadow_update_model.",
)


def _is_shadow_parameter(name: str) -> bool:
    return any(pattern in name for pattern in SHADOW_PARAM_PATTERNS)


def group_parameters_by_shadow(model) -> dict[str, list[torch.nn.Parameter]]:
    """Split trainable parameters into base/shadow x decay/no-decay groups.

    Shadow identification uses :data:`SHADOW_PARAM_PATTERNS`. Parameters that
    do not require grad are skipped, mirroring optimizer construction. The
    four lists feed the differential-LR optimizer groups (shadow parameters
    get ``learning_rate * shadow_lr_multiplier``).
    """
    groups: dict[str, list[torch.nn.Parameter]] = {
        "base_decay": [],
        "base_nodecay": [],
        "shadow_decay": [],
        "shadow_nodecay": [],
    }
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        prefix = "shadow" if _is_shadow_parameter(name) else "base"
        suffix = "nodecay" if len(param.shape) == 1 or name.endswith(".bias") else "decay"
        groups[f"{prefix}_{suffix}"].append(param)
    return groups


class DifferentialLRMixin:
    """
    Mixin that adds differential learning rate support for shadow parameters.

    Usage: Mix this into any Trainer class and set self.shadow_lr_multiplier before calling super().__init__().
    """

    def _create_optimizer_with_differential_lr(self, shadow_lr_multiplier: float):
        """Create optimizer with different LRs for shadow vs base parameters."""
        if self.optimizer is not None:
            return self.optimizer

        groups = group_parameters_by_shadow(self.model)
        base_decay = groups["base_decay"]
        base_nodecay = groups["base_nodecay"]
        shadow_decay = groups["shadow_decay"]
        shadow_nodecay = groups["shadow_nodecay"]

        print(">>> model architecture:")
        print(self.model)
        for name, param in self.model.named_parameters():
            if param.requires_grad and _is_shadow_parameter(name):
                print(f">>> Shadow parameter: {name}")

        base_lr = self.args.learning_rate
        shadow_lr = base_lr * shadow_lr_multiplier

        # Build parameter groups
        optimizer_grouped_parameters = []

        if base_decay:
            optimizer_grouped_parameters.append({
                "params": base_decay,
                "weight_decay": self.args.weight_decay,
                "lr": base_lr,
            })

        if base_nodecay:
            optimizer_grouped_parameters.append({
                "params": base_nodecay,
                "weight_decay": 0.0,
                "lr": base_lr,
            })

        if shadow_decay:
            optimizer_grouped_parameters.append({
                "params": shadow_decay,
                "weight_decay": self.args.weight_decay,
                "lr": shadow_lr,
            })

        if shadow_nodecay:
            optimizer_grouped_parameters.append({
                "params": shadow_nodecay,
                "weight_decay": 0.0,
                "lr": shadow_lr,
            })

        # Get optimizer class and kwargs from training args
        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

        # Remove 'lr' from optimizer_kwargs if present (we set it per group)
        optimizer_kwargs.pop('lr', None)

        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        # Log the setup
        if shadow_lr_multiplier != 1.0:
            print("\n=== Differential Learning Rate Setup ===")
            print(f"Base LR: {base_lr}")
            print(f"Shadow LR: {shadow_lr} (multiplier: {shadow_lr_multiplier})")
            print(f"Base params (decay): {len(base_decay)}")
            print(f"Base params (no decay): {len(base_nodecay)}")
            print(f"Shadow params (decay): {len(shadow_decay)}")
            print(f"Shadow params (no decay): {len(shadow_nodecay)}")
            print("=" * 40 + "\n")

        return self.optimizer

    def create_optimizer(self):
        """Override to use differential LR if shadow_lr_multiplier is set."""
        shadow_lr_multiplier = getattr(self, 'shadow_lr_multiplier', 1.0)
        if shadow_lr_multiplier != 1.0:
            return self._create_optimizer_with_differential_lr(shadow_lr_multiplier)
        return super().create_optimizer()


class DistillTeacherMixin:
    """
    Mixin that runs a frozen teacher in `compute_loss` and passes its logits to the student.

    Usage: Mix this into any Trainer class whose student is a ShadowForCausalLM and set
    self.teacher_model before calling super().__init__(). When teacher_model is None (or the
    student has distill_weight == 0), compute_loss behaves exactly as before.
    """

    def _inject_teacher_logits(self, model, inputs):
        teacher = getattr(self, "teacher_model", None)
        if teacher is None or inputs.get("labels") is None:
            return inputs
        actual = _unwrap_model(model)
        if getattr(actual, "distill_weight", 0.0) <= 0:
            return inputs
        device = inputs["input_ids"].device
        if next(teacher.parameters()).device != device:
            teacher.to(device)
        teacher_inputs = {
            k: inputs[k] for k in ("input_ids", "attention_mask", "position_ids") if k in inputs
        }
        with torch.no_grad():
            outputs = teacher(**teacher_inputs)
        inputs["teacher_logits"] = outputs.logits
        return inputs


def generate_from_shadow(model, **gen_kwargs) -> torch.Tensor:
    prev = getattr(model, "inference_mode", "base_shadow")
    model.set_inference_mode("shadow_only")
    try:
        return model.generate(**gen_kwargs)
    finally:
        model.set_inference_mode(prev)


def weighted_shifted_ce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    sample_weights,
) -> torch.Tensor:
    """Next-token CE averaged per sample, then weighted across the batch.

    ``sample_weights`` is one non-negative weight per sample (``(batch,)``).
    Each sample's CE is the mean over its non-``-100`` target tokens and the
    batch loss is ``sum(w_i * CE_i) / sum(w_i)``. Uniform positive weights
    reduce to :func:`shadow_lab.eval._shifted_ce_loss` bit for bit, so
    all-ones weights reproduce the unweighted SFT baseline exactly.
    """
    weights = torch.as_tensor(sample_weights, dtype=logits.dtype, device=logits.device)
    if weights.dim() == 0:
        weights = weights.expand(labels.size(0))
    if weights.dim() != 1 or weights.numel() != labels.size(0):
        raise ValueError(
            "sample_weights must be 1-D with one weight per sample "
            f"(got shape {tuple(weights.shape)} for a batch of {labels.size(0)})."
        )
    if bool(torch.all(weights == 0)):
        # Nothing contributes; keep the graph so backward yields zero grads.
        return logits.sum() * 0.0
    if bool(torch.all(weights == weights[0])):
        return _shifted_ce_loss(logits, labels)
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    per_token = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    ).view(shift_labels.shape)
    supervised = (shift_labels != -100).to(per_token.dtype)
    token_counts = supervised.sum(dim=-1).clamp(min=1.0)
    per_sample = (per_token * supervised).sum(dim=-1) / token_counts
    return (weights * per_sample).sum() / weights.sum()


# ---------------------------------------------------------------------------
# Shadow-aware trainers
# ---------------------------------------------------------------------------

class ShadowSFTTrainer(DifferentialLRMixin, DistillTeacherMixin, SFTTrainer):
    """SFTTrainer variant that evaluates shadow_logits during evaluation."""

    def __init__(self, *args, shadow_lr_multiplier=1.0, teacher_model=None, **kwargs):
        self.shadow_lr_multiplier = shadow_lr_multiplier
        self.teacher_model = teacher_model
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        inputs = self._inject_teacher_logits(model, inputs)
        return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)

    def evaluation_loop(self, dataloader, description, prediction_loss_only=None, ignore_keys=None, metric_key_prefix="eval"):
        model = self._wrap_model(self.model, training=False, dataloader=dataloader)
        model.eval()

        total_loss = 0.0
        total_shadow_loss = 0.0
        num_samples = 0

        for step, inputs in enumerate(dataloader):  # noqa: B007
            inputs = self._prepare_inputs(inputs)
            with torch.no_grad():
                outputs = model(**inputs)
                if outputs.loss is not None:
                    total_loss += outputs.loss.item()

                if getattr(outputs, "shadow_logits", None) is not None:
                    labels = inputs.get("labels")
                    if labels is not None:
                        total_shadow_loss += float(_shifted_ce_loss(outputs.shadow_logits, labels).item())

                batch_size = inputs["input_ids"].shape[0]
                num_samples += batch_size

        avg_loss = total_loss / (step + 1) if step >= 0 else 0.0
        metrics = {
            f"{metric_key_prefix}_loss": avg_loss,
            f"{metric_key_prefix}_samples": num_samples,
        }
        if total_shadow_loss > 0:
            metrics[f"{metric_key_prefix}_shadow_loss"] = total_shadow_loss / (step + 1)

        return EvalLoopOutput(predictions=None, label_ids=None, metrics=metrics, num_samples=num_samples)


class WeightedSFTTrainer(ShadowSFTTrainer):
    """SFTTrainer that applies optional per-sample ``sample_weights``.

    A batch may carry a ``sample_weights`` entry with one non-negative weight
    per sample. When it does, ``compute_loss`` replaces the next-token CE with
    :func:`weighted_shifted_ce_loss` (the weighted path trains the CE only; no
    teacher-logit distillation is added); when it does not, the trainer behaves
    exactly like :class:`ShadowSFTTrainer` (teacher injection included).

    ``sample_weights`` are training-only: keep them out of evaluation batches,
    which the inherited ``evaluation_loop`` forwards straight to the model.
    """

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        sample_weights = inputs.pop("sample_weights", None)
        if sample_weights is None:
            return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)
        labels = inputs.pop("labels", None)
        if labels is None:
            raise ValueError("WeightedSFTTrainer requires `labels` when `sample_weights` is set.")
        outputs = model(**inputs)
        loss = weighted_shifted_ce_loss(outputs.logits, labels, sample_weights)
        return (loss, outputs) if return_outputs else loss


class MMLUTrainer(DifferentialLRMixin, DistillTeacherMixin, SFTTrainer):
    """SFTTrainer variant for MMLU with answer extraction and accuracy computation.

    The canonical MMLU evaluation loop — generation-based answer extraction
    with optional shadow-path generation, plus thinking suppression via
    ``stop_strings`` when the tokenizer advertises ``thinking_tokens``.
    """

    def __init__(self, *args, shadow_lr_multiplier=1.0, teacher_model=None, **kwargs):
        self.shadow_lr_multiplier = shadow_lr_multiplier
        self.teacher_model = teacher_model
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        inputs.pop("prompt_input_ids", None)
        inputs.pop("prompt_attention_mask", None)
        inputs.pop("answer_letter", None)
        inputs.pop("answer_idx", None)
        inputs = self._inject_teacher_logits(model, inputs)
        return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)

    def evaluation_loop(self, dataloader, description, prediction_loss_only=None, ignore_keys=None, metric_key_prefix="eval"):
        model = self._wrap_model(self.model, training=False, dataloader=dataloader)
        model.eval()

        all_predictions = []
        all_shadow_predictions = []
        all_labels = []
        total_loss = 0.0
        total_shadow_loss = 0.0
        num_samples = 0

        actual_model = model.module if hasattr(model, "module") else model
        do_shadow_gen = bool(getattr(self.args, "print_shadow_output", False))

        compute_dtype = torch.bfloat16 if self.args.bf16 else (torch.float16 if self.args.fp16 else torch.float32)
        max_new_tokens = getattr(self.args, "generation_max_length", None)
        if max_new_tokens is None:
            max_new_tokens = DEFAULT_MMLU_GENERATION_TOKENS
        pad_token_id = getattr(
            self.processing_class, "pad_token_id", getattr(self.model.config, "pad_token_id", None)
        )
        eos_token_id = getattr(
            self.processing_class, "eos_token_id", getattr(self.model.config, "eos_token_id", None)
        )
        if pad_token_id is None and eos_token_id is not None:
            pad_token_id = eos_token_id

        for step, inputs in enumerate(dataloader):  # noqa: B007
            inputs = self._prepare_inputs(inputs)
            prompt_input_ids = inputs.pop("prompt_input_ids", None)
            prompt_attention_mask = inputs.pop("prompt_attention_mask", None)
            answer_letters = inputs.pop("answer_letters", None)
            inputs.pop("answer_indices", None)

            with torch.no_grad():
                outputs = model(**inputs)
                if outputs.loss is not None:
                    total_loss += outputs.loss.item()

                if getattr(outputs, "shadow_logits", None) is not None:
                    labels = inputs.get("labels")
                    if labels is not None:
                        total_shadow_loss += float(_shifted_ce_loss(outputs.shadow_logits, labels).item())

                input_ids = inputs["input_ids"]
                attention_mask = inputs["attention_mask"]
                batch_size = input_ids.shape[0]

                for i in range(batch_size):
                    if prompt_input_ids is not None and prompt_attention_mask is not None:
                        gen_input_ids = prompt_input_ids[i : i + 1, :]
                        gen_attention_mask = prompt_attention_mask[i : i + 1, :]
                    else:
                        gen_input_ids = input_ids[i : i + 1, :]
                        gen_attention_mask = attention_mask[i : i + 1, :]

                    if gen_attention_mask is not None:
                        true_len = int(gen_attention_mask.long().sum(dim=-1).item())
                        true_len = max(true_len, 1)
                        padding_side = getattr(self.processing_class, "padding_side", "right")
                        if padding_side == "left":
                            gen_input_ids = gen_input_ids[:, -true_len:]
                            gen_attention_mask = gen_attention_mask[:, -true_len:]
                        else:
                            gen_input_ids = gen_input_ids[:, :true_len]
                            gen_attention_mask = gen_attention_mask[:, :true_len]

                    with torch.autocast(
                        device_type="cuda",
                        dtype=compute_dtype,
                        enabled=bool(self.args.bf16 or self.args.fp16),
                    ):
                        gen_kwargs = {
                            "input_ids": gen_input_ids,
                            "attention_mask": gen_attention_mask,
                            "max_new_tokens": max_new_tokens,
                            "do_sample": False,
                            "pad_token_id": pad_token_id,
                            "eos_token_id": eos_token_id,
                            "use_cache": False,
                        }
                        if hasattr(self.processing_class, "thinking_tokens"):
                            gen_kwargs["stop_strings"] = ["<think>", "<|im_start|>think"]
                        generated = actual_model.generate(**gen_kwargs)

                    prompt_length = gen_input_ids.shape[-1]
                    new_tokens = generated[0, prompt_length:]
                    generated_text_raw = self.processing_class.decode(
                        new_tokens,
                        skip_special_tokens=True,
                    )
                    generated_text = _clean_generated_text(generated_text_raw)
                    predicted_answer = extract_answer_from_text(generated_text)
                    all_predictions.append(predicted_answer.upper() if predicted_answer else "INVALID")

                    shadow_generated_text = None
                    shadow_predicted = None
                    shadow_info = ""

                    if do_shadow_gen and hasattr(actual_model, "shadow_lm_head"):
                        with torch.autocast(
                            device_type="cuda",
                            dtype=compute_dtype,
                            enabled=bool(self.args.bf16 or self.args.fp16),
                        ):
                            shadow_generated_ids = generate_from_shadow(
                                model=actual_model,
                                input_ids=gen_input_ids,
                                attention_mask=gen_attention_mask,
                                max_new_tokens=max_new_tokens,
                                pad_token_id=pad_token_id,
                                eos_token_id=eos_token_id,
                                use_cache=False,
                            )

                        shadow_generated_text_raw = self.processing_class.decode(
                            shadow_generated_ids[0, prompt_length:],
                            skip_special_tokens=True,
                        )
                        shadow_generated_text = _clean_generated_text(shadow_generated_text_raw)
                        shadow_predicted = extract_answer_from_text(shadow_generated_text)
                        all_shadow_predictions.append(
                            shadow_predicted.upper() if shadow_predicted else "INVALID"
                        )

                    if answer_letters and i < len(answer_letters) and answer_letters[i] is not None:
                        true_answer = str(answer_letters[i]).upper()
                        all_labels.append(true_answer)
                        if do_shadow_gen and shadow_generated_text is not None:
                            shadow_info = (
                                f" | shadow_gen: '{shadow_generated_text}' | shadow_pred: {shadow_predicted}"
                            )
                        print(
                            f"[Sample {num_samples}] generated: '{generated_text}' | predicted: {predicted_answer} | true: {true_answer}{shadow_info}",
                            flush=True,
                        )
                    else:
                        print(
                            f"[Sample {num_samples}] Warning: No answer_letter found for sample {i}",
                            flush=True,
                        )
                    num_samples += 1

        correct = sum(1 for p, label in zip(all_predictions, all_labels, strict=False) if p == label)
        accuracy = correct / len(all_labels) if all_labels else 0.0
        avg_loss = total_loss / (step + 1) if step >= 0 else 0.0

        metrics: dict[str, float] = {
            f"{metric_key_prefix}_loss": avg_loss,
            f"{metric_key_prefix}_accuracy": accuracy,
            f"{metric_key_prefix}_samples": num_samples,
        }
        if total_shadow_loss > 0:
            metrics[f"{metric_key_prefix}_shadow_loss"] = total_shadow_loss / (step + 1)
        if do_shadow_gen and all_shadow_predictions and all_labels:
            shadow_correct = sum(1 for p, label in zip(all_shadow_predictions, all_labels, strict=False) if p == label)
            metrics[f"{metric_key_prefix}_shadow_accuracy"] = shadow_correct / len(all_labels)

        return EvalLoopOutput(
            predictions=all_predictions,
            label_ids=all_labels,
            metrics=metrics,
            num_samples=num_samples,
        )


class GSM8KTrainer(DifferentialLRMixin, DistillTeacherMixin, SFTTrainer):
    """SFTTrainer variant for GSM8K with final-answer extraction accuracy.

    ``answer_extractor`` is the seam for extraction strategies: the standard
    gold-format parser by default, the boxed parser for OOD/base-model evals.
    """

    def __init__(
        self,
        *args,
        shadow_lr_multiplier=1.0,
        teacher_model=None,
        answer_extractor: Callable[[str], str] = extract_gsm8k_answer_from_text,
        **kwargs,
    ):
        self.shadow_lr_multiplier = shadow_lr_multiplier
        self.teacher_model = teacher_model
        self.answer_extractor = answer_extractor
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        inputs.pop("prompt_input_ids", None)
        inputs.pop("prompt_attention_mask", None)
        inputs.pop("gold_answers", None)
        inputs = self._inject_teacher_logits(model, inputs)
        return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)

    def evaluation_loop(
        self,
        dataloader,
        description,
        prediction_loss_only=None,
        ignore_keys=None,
        metric_key_prefix="eval",
    ):
        model = self._wrap_model(self.model, training=False, dataloader=dataloader)
        model.eval()

        total_loss = 0.0
        total_shadow_loss = 0.0
        num_batches = 0
        num_samples = 0

        correct = 0
        shadow_correct = 0
        total_answered = 0

        actual_model = model.module if hasattr(model, "module") else model
        has_shadow = hasattr(actual_model, "shadow_lm_head")
        do_shadow_gen = bool(getattr(self.args, "print_shadow_output", False))

        compute_dtype = (
            torch.bfloat16
            if self.args.bf16
            else (torch.float16 if self.args.fp16 else torch.float32)
        )
        max_new_tokens = getattr(self.args, "generation_max_length", None)
        if max_new_tokens is None:
            max_new_tokens = DEFAULT_GSM8K_GENERATION_TOKENS

        pad_token_id = getattr(
            self.processing_class,
            "pad_token_id",
            getattr(self.model.config, "pad_token_id", None),
        )
        eos_token_id = getattr(
            self.processing_class,
            "eos_token_id",
            getattr(self.model.config, "eos_token_id", None),
        )
        if pad_token_id is None and eos_token_id is not None:
            pad_token_id = eos_token_id

        use_autocast = bool(torch.cuda.is_available() and (self.args.bf16 or self.args.fp16))

        for _, inputs in enumerate(dataloader):
            inputs = self._prepare_inputs(inputs)
            prompt_input_ids = inputs.pop("prompt_input_ids", None)
            prompt_attention_mask = inputs.pop("prompt_attention_mask", None)
            gold_answers = inputs.pop("gold_answers", None)

            with torch.inference_mode():
                outputs = model(**inputs)
                if outputs.loss is not None:
                    total_loss += float(outputs.loss.item())
                num_batches += 1

                labels = inputs.get("labels")
                if labels is not None and getattr(outputs, "shadow_logits", None) is not None:
                    total_shadow_loss += float(_shifted_ce_loss(outputs.shadow_logits, labels).item())

            if prompt_input_ids is None or prompt_attention_mask is None or gold_answers is None:
                continue

            batch_size = prompt_input_ids.shape[0]
            for i in range(batch_size):
                gen_input_ids = prompt_input_ids[i : i + 1, :]
                gen_attention_mask = prompt_attention_mask[i : i + 1, :]

                # Trim per-sample padding before generation.
                if gen_attention_mask is not None:
                    true_len = int(gen_attention_mask.long().sum(dim=-1).item())
                    true_len = max(true_len, 1)
                    padding_side = getattr(self.processing_class, "padding_side", "right")
                    if padding_side == "left":
                        gen_input_ids = gen_input_ids[:, -true_len:]
                        gen_attention_mask = gen_attention_mask[:, -true_len:]
                    else:
                        gen_input_ids = gen_input_ids[:, :true_len]
                        gen_attention_mask = gen_attention_mask[:, :true_len]

                if use_autocast:
                    ctx = torch.autocast(device_type="cuda", dtype=compute_dtype, enabled=True)
                else:
                    ctx = torch.no_grad()

                with ctx:
                    generated = model.generate(
                        input_ids=gen_input_ids,
                        attention_mask=gen_attention_mask,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=pad_token_id,
                        eos_token_id=eos_token_id,
                        use_cache=False,
                    )

                prompt_length = gen_input_ids.shape[-1]
                new_tokens = generated[0, prompt_length:]
                generated_text_raw = self.processing_class.decode(new_tokens, skip_special_tokens=True)
                pred = self.answer_extractor(generated_text_raw)
                gold = str(gold_answers[i]).strip()

                if pred and gold and pred == gold:
                    correct += 1
                total_answered += 1

                shadow_text_raw = None
                shadow_pred = None
                if has_shadow and do_shadow_gen:
                    with ctx:
                        shadow_generated_ids = generate_from_shadow(
                            model=actual_model,
                            input_ids=gen_input_ids,
                            attention_mask=gen_attention_mask,
                            max_new_tokens=max_new_tokens,
                            pad_token_id=pad_token_id,
                            eos_token_id=eos_token_id,
                            use_cache=False,
                        )
                    shadow_new_tokens = shadow_generated_ids[0, prompt_length:]
                    shadow_text_raw = self.processing_class.decode(
                        shadow_new_tokens, skip_special_tokens=True
                    )
                    shadow_pred = self.answer_extractor(shadow_text_raw)
                    if shadow_pred and gold and shadow_pred == gold:
                        shadow_correct += 1

                running_acc = correct / total_answered if total_answered > 0 else 0.0
                running_loss = total_loss / max(num_batches, 1)
                msg = (
                    f"[Sample {num_samples}] "
                    f"acc: {running_acc:.4f} ({correct}/{total_answered}) | "
                    f"loss: {running_loss:.4f} | "
                    f"generated: '{generated_text_raw}' | predicted: {pred} | true: {gold}"
                )
                if has_shadow and do_shadow_gen and shadow_text_raw is not None:
                    running_shadow_acc = shadow_correct / total_answered if total_answered > 0 else 0.0
                    msg += (
                        f" | shadow_acc: {running_shadow_acc:.4f} ({shadow_correct}/{total_answered})"
                        f" | shadow_gen: '{shadow_text_raw}' | shadow_pred: {shadow_pred}"
                    )
                print(msg, flush=True)
                num_samples += 1

        avg_loss = total_loss / max(num_batches, 1)
        metrics = {
            f"{metric_key_prefix}_loss": avg_loss,
            f"{metric_key_prefix}_samples": num_samples,
        }
        if total_answered > 0:
            metrics[f"{metric_key_prefix}_answer_accuracy"] = correct / total_answered
            if has_shadow and do_shadow_gen:
                metrics[f"{metric_key_prefix}_shadow_answer_accuracy"] = shadow_correct / total_answered
        if total_shadow_loss > 0 and num_batches > 0:
            metrics[f"{metric_key_prefix}_shadow_loss"] = total_shadow_loss / num_batches

        return EvalLoopOutput(predictions=None, label_ids=None, metrics=metrics, num_samples=num_samples)


class SquadV2Trainer(DifferentialLRMixin, DistillTeacherMixin, SFTTrainer):
    """SFTTrainer variant for SQuAD v2 using the official EM/F1 metric."""

    def __init__(self, *args, shadow_lr_multiplier=1.0, teacher_model=None, **kwargs):
        self.shadow_lr_multiplier = shadow_lr_multiplier
        self.teacher_model = teacher_model
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        inputs.pop("prompt_input_ids", None)
        inputs.pop("prompt_attention_mask", None)
        inputs.pop("squad_ids", None)
        inputs.pop("squad_answers", None)
        inputs = self._inject_teacher_logits(model, inputs)
        return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)

    def evaluation_loop(
        self,
        dataloader,
        description,
        prediction_loss_only=None,
        ignore_keys=None,
        metric_key_prefix="eval",
    ):
        model = self._wrap_model(self.model, training=False, dataloader=dataloader)
        model.eval()

        import evaluate

        metric = evaluate.load("squad_v2")
        shadow_metric = None

        total_loss = 0.0
        total_shadow_loss = 0.0
        num_batches = 0
        num_samples = 0

        actual_model = model.module if hasattr(model, "module") else model
        do_shadow_gen = bool(getattr(self.args, "print_shadow_output", False))
        has_shadow = hasattr(actual_model, "shadow_lm_head")
        if has_shadow and do_shadow_gen:
            shadow_metric = evaluate.load("squad_v2")

        compute_dtype = (
            torch.bfloat16
            if self.args.bf16
            else (torch.float16 if self.args.fp16 else torch.float32)
        )
        max_new_tokens = getattr(self.args, "generation_max_length", None)
        if max_new_tokens is None:
            max_new_tokens = DEFAULT_SQUAD_V2_GENERATION_TOKENS

        pad_token_id = getattr(
            self.processing_class,
            "pad_token_id",
            getattr(self.model.config, "pad_token_id", None),
        )
        eos_token_id = getattr(
            self.processing_class,
            "eos_token_id",
            getattr(self.model.config, "eos_token_id", None),
        )
        if pad_token_id is None and eos_token_id is not None:
            pad_token_id = eos_token_id

        use_autocast = bool(torch.cuda.is_available() and (self.args.bf16 or self.args.fp16))

        for _, inputs in enumerate(dataloader):
            inputs = self._prepare_inputs(inputs)
            prompt_input_ids = inputs.pop("prompt_input_ids", None)
            prompt_attention_mask = inputs.pop("prompt_attention_mask", None)
            squad_ids = inputs.pop("squad_ids", None)
            squad_answers = inputs.pop("squad_answers", None)

            with torch.inference_mode():
                outputs = model(**inputs)
                if outputs.loss is not None:
                    total_loss += float(outputs.loss.item())
                num_batches += 1

                labels = inputs.get("labels")
                if labels is not None and getattr(outputs, "shadow_logits", None) is not None:
                    total_shadow_loss += float(_shifted_ce_loss(outputs.shadow_logits, labels).item())

            if (
                prompt_input_ids is None
                or prompt_attention_mask is None
                or squad_ids is None
                or squad_answers is None
            ):
                continue

            batch_size = prompt_input_ids.shape[0]
            for i in range(batch_size):
                gen_input_ids = prompt_input_ids[i : i + 1, :]
                gen_attention_mask = prompt_attention_mask[i : i + 1, :]

                # Trim per-sample padding before generation.
                if gen_attention_mask is not None:
                    true_len = int(gen_attention_mask.long().sum(dim=-1).item())
                    true_len = max(true_len, 1)
                    padding_side = getattr(self.processing_class, "padding_side", "right")
                    if padding_side == "left":
                        gen_input_ids = gen_input_ids[:, -true_len:]
                        gen_attention_mask = gen_attention_mask[:, -true_len:]
                    else:
                        gen_input_ids = gen_input_ids[:, :true_len]
                        gen_attention_mask = gen_attention_mask[:, :true_len]

                if use_autocast:
                    ctx = torch.autocast(device_type="cuda", dtype=compute_dtype, enabled=True)
                else:
                    ctx = torch.no_grad()

                with ctx:
                    generated = actual_model.generate(
                        input_ids=gen_input_ids,
                        attention_mask=gen_attention_mask,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=pad_token_id,
                        eos_token_id=eos_token_id,
                        use_cache=False,
                    )

                prompt_length = gen_input_ids.shape[-1]
                new_tokens = generated[0, prompt_length:]
                pred_raw = self.processing_class.decode(new_tokens, skip_special_tokens=True).strip()
                pred_clean = _clean_generated_text(pred_raw)

                # Map "unanswerable" => no-answer.
                if pred_clean.lower() in {"unanswerable", "no answer", "n/a", "none", ""}:
                    prediction_text = ""
                    no_answer_probability = 1.0
                else:
                    prediction_text = pred_clean
                    no_answer_probability = 0.0

                qid = str(squad_ids[i])
                shadow_info = ""
                if has_shadow and do_shadow_gen:
                    with ctx:
                        shadow_generated = generate_from_shadow(
                            model=actual_model,
                            input_ids=gen_input_ids,
                            attention_mask=gen_attention_mask,
                            max_new_tokens=max_new_tokens,
                            pad_token_id=pad_token_id,
                            eos_token_id=eos_token_id,
                            use_cache=False,
                        )
                    shadow_new_tokens = shadow_generated[0, prompt_length:]
                    shadow_raw = self.processing_class.decode(
                        shadow_new_tokens, skip_special_tokens=True
                    ).strip()
                    shadow_text = _clean_generated_text(shadow_raw)
                    shadow_info = f" | shadow_gen: '{shadow_text}'"
                    if shadow_metric is not None:
                        if shadow_text.lower() in {"unanswerable", "no answer", "n/a", "none", ""}:
                            shadow_prediction_text = ""
                            shadow_no_answer_probability = 1.0
                        else:
                            shadow_prediction_text = shadow_text
                            shadow_no_answer_probability = 0.0
                        shadow_info = f" | shadow_gen: '{shadow_prediction_text}'"
                        shadow_metric.add(
                            prediction={
                                "id": qid,
                                "prediction_text": shadow_prediction_text,
                                "no_answer_probability": float(shadow_no_answer_probability),
                            },
                            reference={
                                "id": qid,
                                "answers": squad_answers[i],
                            },
                        )

                print(
                    f"[Sample {num_samples}] generated: '{prediction_text}' | true: {squad_answers[i]}{shadow_info}",
                    flush=True,
                )
                metric.add(
                    prediction={
                        "id": qid,
                        "prediction_text": prediction_text,
                        "no_answer_probability": float(no_answer_probability),
                    },
                    reference={
                        "id": qid,
                        "answers": squad_answers[i],
                    },
                )
                num_samples += 1

        avg_loss = total_loss / max(num_batches, 1)
        squad_metrics = metric.compute()
        metrics = {
            f"{metric_key_prefix}_loss": avg_loss,
            f"{metric_key_prefix}_samples": num_samples,
        }
        for k, v in squad_metrics.items():
            metrics[f"{metric_key_prefix}_{k}"] = v
        if shadow_metric is not None:
            shadow_squad_metrics = shadow_metric.compute()
            for k, v in shadow_squad_metrics.items():
                metrics[f"{metric_key_prefix}_shadow_{k}"] = v
        if total_shadow_loss > 0 and num_batches > 0:
            metrics[f"{metric_key_prefix}_shadow_loss"] = total_shadow_loss / num_batches

        return EvalLoopOutput(
            predictions=None,
            label_ids=None,
            metrics=metrics,
            num_samples=num_samples,
        )


# ---------------------------------------------------------------------------
# On-policy GKD flywheel
# ---------------------------------------------------------------------------

def _onpolicy_mask(
    source: str | Sequence[str] | torch.Tensor | None,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Normalize a batch's ``source`` column to a boolean on-policy mask.

    ``source`` is normally one string per row (``"supervised"`` or
    ``"onpolicy"``); a single string broadcasts, a tensor may be a boolean
    mask or 0/1 integers, and ``None`` treats every row as supervised.
    """
    if source is None:
        return torch.zeros(batch_size, dtype=torch.bool, device=device)

    if isinstance(source, str):
        values = [source] * batch_size
    elif torch.is_tensor(source):
        if source.dim() != 1 or source.numel() != batch_size:
            raise ValueError(
                f"source tensor must be 1-D with {batch_size} entries, "
                f"got shape {tuple(source.shape)}."
            )
        if source.dtype == torch.bool:
            return source.to(device)
        ints = source.to(device="cpu", dtype=torch.long).tolist()
        if any(value not in (0, 1) for value in ints):
            raise ValueError(
                "source tensor must be boolean or 0/1 integers "
                "(0=supervised, 1=onpolicy)."
            )
        return torch.tensor([bool(value) for value in ints], dtype=torch.bool, device=device)
    else:
        values = list(source)

    if len(values) != batch_size:
        raise ValueError(f"source must have {batch_size} entries, got {len(values)}.")
    mask = []
    for value in values:
        if value == "supervised":
            mask.append(False)
        elif value == "onpolicy":
            mask.append(True)
        else:
            raise ValueError(
                f"Unknown source {value!r}; expected 'supervised' or 'onpolicy'."
            )
    return torch.tensor(mask, dtype=torch.bool, device=device)


def onpolicy_gkd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor | None,
    shadow_logits: torch.Tensor | None,
    labels: torch.Tensor,
    *,
    source: str | Sequence[str] | torch.Tensor | None = None,
    lambda_onpolicy: float = 0.5,
    divergence: str = "forward_kl",
    topk: int | None = None,
    shadow_aux_weight: float = 0.0,
    temperature: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Compose one on-policy GKD batch into its loss terms.

    ``total = (1 - lambda) * CE_supervised + lambda * KD_onpolicy
    + shadow_aux_weight * CE_shadow``

    - ``source`` splits the rows into ``"supervised"`` and ``"onpolicy"``.
      Supervised rows contribute a shifted CE on the joint ``base_shadow``
      logits; on-policy rows contribute ``_shifted_kd_loss`` against the
      teacher's logits. The two row sets never share a term.
    - The auxiliary term is the shadow-only shifted CE (chosen over a KL so
      it keeps the same target semantics as the supervised CE). It is off by
      default (weight 0) and skipped when ``shadow_logits`` is missing.
    - The teacher is only required when ``lambda_onpolicy > 0`` and the batch
      contains on-policy rows. ``teacher_logits`` may cover the whole batch
      or only the on-policy rows.
    - Vocabulary mismatches surface from ``_shifted_kd_loss`` with both
      vocabulary sizes.

    The model's own ``outputs.loss`` is deliberately not used here: it
    stacks ``shadow_loss_weight`` and ``distill_weight``, which would
    double-count the shadow and teacher terms composed explicitly below.

    Returns ``{"ce", "kd", "aux", "total"}``; terms that do not participate
    are zero scalars.
    """
    if student_logits.dim() != 3:
        raise ValueError(
            f"student_logits must be [batch, seq, vocab], got {tuple(student_logits.shape)}."
        )
    if labels.shape != student_logits.shape[:2]:
        raise ValueError(
            f"labels shape {tuple(labels.shape)} must match student_logits "
            f"batch/seq {tuple(student_logits.shape[:2])}."
        )
    if not 0.0 <= lambda_onpolicy <= 1.0:
        raise ValueError(f"lambda_onpolicy must be in [0, 1], got {lambda_onpolicy}.")
    if shadow_aux_weight < 0:
        raise ValueError(f"shadow_aux_weight must be >= 0, got {shadow_aux_weight}.")
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}.")
    if teacher_logits is not None and (
        teacher_logits.dim() != 3 or teacher_logits.size(1) != student_logits.size(1)
    ):
        raise ValueError(
            "teacher_logits must be [rows, seq, vocab] with the student's "
            f"sequence length, got {tuple(teacher_logits.shape)}."
        )
    if shadow_logits is not None and shadow_logits.shape != student_logits.shape:
        raise ValueError(
            f"shadow_logits shape {tuple(shadow_logits.shape)} must match "
            f"student_logits {tuple(student_logits.shape)}."
        )

    batch_size = labels.size(0)
    onpolicy = _onpolicy_mask(source, batch_size, labels.device)
    supervised = ~onpolicy

    zero = student_logits.new_zeros(())
    ce = zero
    kd = zero
    aux = zero
    total = zero
    has_term = False

    if lambda_onpolicy < 1.0 and bool(supervised.any()):
        ce = _shifted_ce_loss(student_logits[supervised], labels[supervised])
        total = total + (1.0 - lambda_onpolicy) * ce
        has_term = True

    if lambda_onpolicy > 0.0 and bool(onpolicy.any()):
        if teacher_logits is None:
            raise ValueError(
                "Batch has on-policy rows but teacher_logits is None while "
                "lambda_onpolicy > 0."
            )
        if teacher_logits.size(0) == batch_size:
            teacher_for_kd = teacher_logits[onpolicy]
        elif teacher_logits.size(0) == int(onpolicy.sum()):
            teacher_for_kd = teacher_logits
        else:
            raise ValueError(
                "teacher_logits must have one row per batch element or per "
                f"on-policy element; got {teacher_logits.size(0)} rows for "
                f"{batch_size} rows ({int(onpolicy.sum())} on-policy)."
            )
        kd = _shifted_kd_loss(
            student_logits[onpolicy],
            teacher_for_kd,
            labels[onpolicy],
            temperature,
            divergence=divergence,
            topk=topk,
        )
        total = total + lambda_onpolicy * kd
        has_term = True

    if shadow_aux_weight > 0.0 and shadow_logits is not None:
        aux = _shifted_ce_loss(shadow_logits, labels)
        total = total + shadow_aux_weight * aux
        has_term = True

    if not has_term:
        # Keep the zero loss connected to the student so an all-on-policy
        # batch with lambda=0 (or an empty branch) stays backprop-safe.
        total = student_logits.sum() * 0.0

    return {"ce": ce, "kd": kd, "aux": aux, "total": total}


def _model_vocab_size(model, role: str) -> int:
    """Best-effort vocabulary size for a student/teacher model."""
    config = getattr(model, "config", None)
    vocab_size = getattr(config, "vocab_size", None)
    if vocab_size is not None and int(vocab_size) > 0:
        return int(vocab_size)
    peft_model = getattr(model, "peft_model", None)
    base_model = getattr(peft_model, "base_model", None)
    for candidate in (model, peft_model, base_model):
        if candidate is None:
            continue
        get_embeddings = getattr(candidate, "get_input_embeddings", None)
        if callable(get_embeddings):
            embeddings = get_embeddings()
            weight = getattr(embeddings, "weight", None)
            if weight is not None:
                return int(weight.shape[0])
        lm_head = getattr(candidate, "lm_head", None)
        weight = getattr(lm_head, "weight", None)
        if weight is not None:
            return int(weight.shape[0])
    raise ValueError(f"cannot determine the {role} vocabulary size")


def validate_teacher_vocab(student, teacher) -> int:
    """Require a shared tokenizer vocabulary before any distillation run.

    This runs even when ``lambda_onpolicy=0`` (all-supervised batches): a
    mismatched teacher is a configuration error, not something to discover
    halfway through a run. Returns the shared size; raises ``ValueError``
    naming both sizes otherwise.
    """
    student_vocab = _model_vocab_size(student, "student")
    teacher_vocab = _model_vocab_size(teacher, "teacher")
    if student_vocab != teacher_vocab:
        raise ValueError(
            f"teacher vocabulary {teacher_vocab} does not match student "
            f"vocabulary {student_vocab}; on-policy KD needs a shared "
            "tokenizer."
        )
    return student_vocab


class OnPolicyGKDTrainer(ShadowSFTTrainer):
    """GKD-style flywheel trainer: supervised SFT plus on-policy distillation.

    Each batch carries a ``source`` column (``"supervised"`` / ``"onpolicy"``);
    the trainer splits the rows and composes::

        loss = (1 - lambda_onpolicy) * CE_supervised
             + lambda_onpolicy * KD_onpolicy
             + shadow_aux_weight * CE_shadow

    Supervised rows use a shifted CE on the joint ``base_shadow`` logits.
    On-policy rows run the frozen teacher over the same tokens under
    ``torch.no_grad()`` and distill with ``_shifted_kd_loss`` (temperature,
    divergence and top-k configurable). ``shadow_logits`` gets an optional
    shadow-only shifted CE; the default weight 0 leaves the auxiliary path
    out of the objective. Only positions with ``labels != -100`` contribute.
    Teacher vocabulary mismatches raise from ``_shifted_kd_loss``.

    The model's own ``outputs.loss`` is not used: it already stacks the
    model-level ``shadow_loss_weight`` / ``distill_weight``, which would
    double-count terms this trainer composes explicitly.

    Constructor knobs mirror the spec: ``lambda_onpolicy=0.5``,
    ``divergence="forward_kl"``, ``topk=None``, ``shadow_aux_weight=0.0``,
    ``teacher_temperature=1.0``; ``shadow_lr_multiplier`` and
    ``teacher_model`` pass through to ``ShadowSFTTrainer``. The teacher is
    frozen and put in eval mode.
    """

    def __init__(
        self,
        *args,
        lambda_onpolicy: float = 0.5,
        divergence: str = "forward_kl",
        topk: int | None = None,
        shadow_aux_weight: float = 0.0,
        teacher_temperature: float = 1.0,
        shadow_lr_multiplier: float = 1.0,
        teacher_model=None,
        **kwargs,
    ):
        if not 0.0 <= lambda_onpolicy <= 1.0:
            raise ValueError(f"lambda_onpolicy must be in [0, 1], got {lambda_onpolicy}.")
        if shadow_aux_weight < 0:
            raise ValueError(f"shadow_aux_weight must be >= 0, got {shadow_aux_weight}.")
        if teacher_temperature <= 0:
            raise ValueError(
                f"teacher_temperature must be > 0, got {teacher_temperature}."
            )
        model = kwargs.get("model")
        if model is None and args:
            model = args[0]
        if model is not None and teacher_model is not None:
            validate_teacher_vocab(model, teacher_model)
        self.lambda_onpolicy = float(lambda_onpolicy)
        self.divergence = divergence
        self.topk = topk
        self.shadow_aux_weight = float(shadow_aux_weight)
        self.teacher_temperature = float(teacher_temperature)
        if teacher_model is not None:
            teacher_model.requires_grad_(False)
            teacher_model.eval()
        super().__init__(
            *args,
            shadow_lr_multiplier=shadow_lr_multiplier,
            teacher_model=teacher_model,
            **kwargs,
        )

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        source = inputs.pop("source", None)
        labels = inputs.get("labels")
        if labels is None:
            raise ValueError(
                "OnPolicyGKDTrainer.compute_loss requires `labels` in the batch."
            )
        model_inputs = {key: value for key, value in inputs.items() if key != "labels"}
        outputs = model(**model_inputs)

        teacher_logits = None
        onpolicy = _onpolicy_mask(source, labels.size(0), labels.device)
        if self.lambda_onpolicy > 0.0 and bool(onpolicy.any()):
            teacher = getattr(self, "teacher_model", None)
            if teacher is None:
                raise ValueError(
                    "OnPolicyGKDTrainer got on-policy rows but teacher_model is "
                    "None while lambda_onpolicy > 0."
                )
            device = labels.device
            if next(teacher.parameters()).device != device:
                teacher.to(device)
            teacher_inputs = {}
            for key in ("input_ids", "attention_mask", "position_ids"):
                value = model_inputs.get(key)
                if value is None:
                    continue
                if (
                    torch.is_tensor(value)
                    and value.dim() > 0
                    and value.size(0) == labels.size(0)
                ):
                    value = value[onpolicy]
                teacher_inputs[key] = value
            with torch.no_grad():
                teacher_outputs = teacher(**teacher_inputs)
            teacher_logits = getattr(teacher_outputs, "logits", None)
            if teacher_logits is None:
                raise TypeError("Teacher forward returned no `logits`.")

        losses = onpolicy_gkd_loss(
            outputs.logits,
            teacher_logits,
            getattr(outputs, "shadow_logits", None),
            labels,
            source=source,
            lambda_onpolicy=self.lambda_onpolicy,
            divergence=self.divergence,
            topk=self.topk,
            shadow_aux_weight=self.shadow_aux_weight,
            temperature=self.teacher_temperature,
        )
        loss = losses["total"]
        if return_outputs:
            return loss, outputs
        return loss

    @staticmethod
    def _strip_source(loader):
        """Yield batches without the training-only ``source`` column."""
        for inputs in loader:
            if isinstance(inputs, Mapping):
                inputs = dict(inputs)
                inputs.pop("source", None)
            yield inputs

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Keep ``source`` away from the model during the HF eval path."""
        inputs = dict(inputs)
        inputs.pop("source", None)
        return super().prediction_step(
            model, inputs, prediction_loss_only, ignore_keys=ignore_keys
        )

    def evaluation_loop(
        self,
        dataloader,
        description,
        prediction_loss_only=None,
        ignore_keys=None,
        metric_key_prefix="eval",
    ):
        """Evaluate with the standard CE loss, never passing ``source`` down.

        ``source`` selects the training loss branch; it is not a model input.
        The inherited loop forwards every batch straight to the model, so the
        loader is filtered here (and in :meth:`prediction_step`).
        """
        return super().evaluation_loop(
            self._strip_source(dataloader),
            description,
            prediction_loss_only=prediction_loss_only,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )
