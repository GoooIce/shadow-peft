"""Shadow-aware SFTTrainer subclasses for benchmark training and evaluation.

Requires ``trl``. The MMLU evaluation_loop lives here exactly once (the
canonical generation-based loop); the GSM8K trainer takes an
``answer_extractor`` seam for boxed/standard extraction strategies.
"""

from __future__ import annotations

from typing import Callable, Dict

import torch
from transformers import Trainer
from transformers.trainer_utils import EvalLoopOutput

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

class DifferentialLRMixin:
    """
    Mixin that adds differential learning rate support for shadow parameters.

    Usage: Mix this into any Trainer class and set self.shadow_lr_multiplier before calling super().__init__().
    """

    def _create_optimizer_with_differential_lr(self, shadow_lr_multiplier: float):
        """Create optimizer with different LRs for shadow vs base parameters."""
        if self.optimizer is not None:
            return self.optimizer

        # Identify shadow parameter patterns
        shadow_param_patterns = [
            'shadow_model.',
            'shadow_lm_head',
            'shadow_adapter',
            'shadow_proj',
            'shadow_hidden_projection',
            'shadow_classifier',
        ]

        # Categorize parameters into 4 groups
        base_decay = []
        base_nodecay = []
        shadow_decay = []
        shadow_nodecay = []

        print(">>> model architecture:")
        print(self.model)
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            # Check if this is a shadow parameter
            is_shadow = any(pattern in name for pattern in shadow_param_patterns)

            # Check if this parameter should have weight decay
            no_decay = len(param.shape) == 1 or name.endswith(".bias")

            if is_shadow:
                print(f">>> Shadow parameter: {name}")
                if no_decay:
                    shadow_nodecay.append(param)
                else:
                    shadow_decay.append(param)
            else:
                if no_decay:
                    base_nodecay.append(param)
                else:
                    base_decay.append(param)

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
            print(f"\n=== Differential Learning Rate Setup ===")
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

        for step, inputs in enumerate(dataloader):
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

        for step, inputs in enumerate(dataloader):
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

        correct = sum(1 for p, l in zip(all_predictions, all_labels) if p == l)
        accuracy = correct / len(all_labels) if all_labels else 0.0
        avg_loss = total_loss / (step + 1) if step >= 0 else 0.0

        metrics: Dict[str, float] = {
            f"{metric_key_prefix}_loss": avg_loss,
            f"{metric_key_prefix}_accuracy": accuracy,
            f"{metric_key_prefix}_samples": num_samples,
        }
        if total_shadow_loss > 0:
            metrics[f"{metric_key_prefix}_shadow_loss"] = total_shadow_loss / (step + 1)
        if do_shadow_gen and all_shadow_predictions and all_labels:
            shadow_correct = sum(1 for p, l in zip(all_shadow_predictions, all_labels) if p == l)
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
