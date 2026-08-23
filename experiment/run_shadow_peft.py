from __future__ import annotations

"""
Training CLI for Shadow-PEFT / LoRA experiments (MMLU, GSM8K, SQuAD v2, generation, classification).

Benchmark evaluation machinery (trainers, collators, answer extraction) lives in
the experiment lab: experiment/lab/eval.py. The eval CLIs are run_eval.py,
run_shadow_peft_eval.py and run_shadow_peft_ood_eval.py.
"""

import argparse
import copy
import json
import os
import sys
from typing import Dict, List, Optional

sys.dont_write_bytecode = True  # avoid __pycache__ permission issues in some environments

import evaluate
import numpy as np
import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)
from trl import SFTConfig, SFTTrainer

from lab.data import (
    ClassificationDatasetBundle,
    ExperimentDatasetBundle,
    GSM8KDatasetBundle,
    MMLUDatasetBundle,
    SquadV2DatasetBundle,
    build_classification_datasets,
    build_datasets,
    build_gsm8k_datasets,
    build_mmlu_datasets,
    build_squad_v2_datasets,
)
from lab.eval import (
    GSM8K_SUITE,
    MMLU_SUITE,
    ClassificationCollatorWithPositions,
    GSM8KDataCollator,
    LMDataCollatorWithPositions,
    MMLUDataCollator,
    SquadV2DataCollator,
    _evaluate_mmlu_subsets,
    _pick_reference_eval_dataset,
    _resolve_mmlu_eval_subsets,
    _set_attn_impl,
    _sync_pad_token,
    prepare_causal_model,
    prepare_tokenizer,
)
from lab.trainers import (
    GSM8KTrainer,
    MMLUTrainer,
    ShadowSFTTrainer,
    SquadV2Trainer,
)
from shadow_peft import (
    ShadowConfig,
    ShadowForCausalLM,
    ShadowForSequenceClassification,
    get_shadow_model,
    prepare_shadow_model,
)

CLASSIFICATION_SUITE = [
    {
        "id": "amazon_reviews_multi_en",
        "dataset_name": "SetFit/amazon_reviews_multi_en",
        "dataset_config": None,
        "train_split": "train",
        "eval_split": "test",
        "text_column": "text",
        "label_column": "label",
        "max_seq_length": 256,
        "shadow_alpha": 4.0,
        "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    },
    {
        "id": "20_newsgroups",
        "dataset_name": "SetFit/20_newsgroups",
        "dataset_config": None,
        "train_split": "train",
        "eval_split": "test",
        "text_column": "text",
        "label_column": "label",
        "max_seq_length": 256,
        "shadow_alpha": 2.0,
        "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    },
    {
        "id": "ag_news",
        "dataset_name": "SetFit/ag_news",
        "dataset_config": None,
        "train_split": "train",
        "eval_split": "test",
        "text_column": "text",
        "label_column": "label",
        "max_seq_length": 256,
        "shadow_alpha": 2.0,
        "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LoRA vs Shadow-PEFT experiments.")
    parser.add_argument(
        "--task",
        type=str,
        choices=("generation", "classification", "mmlu", "gsm8k", "squad_v2"),
        default="mmlu",
    )
    parser.add_argument(
        "--suite",
        type=str,
        choices=("generation", "classification", "mmlu", "gsm8k", "squad_v2", "all"),
        default=None,
    )
    parser.add_argument(
        "--classification_id",
        type=str,
        default=None,
        choices=[spec["id"] for spec in CLASSIFICATION_SUITE],
        help="Run only one classification suite entry when using --suite classification/all.",
    )
    parser.add_argument("--mmlu_subset", type=str, default="all")
    parser.add_argument("--gsm8k_subset", type=str, choices=("main", "socratic"), default="main")
    parser.add_argument("--gsm8k_answer_mode", type=str, choices=("thinking", "final"), default="thinking")
    parser.add_argument("--squad_answer_mode", type=str, choices=("thinking", "final"), default="final")

    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument(
        "--pretrained_shadow_model_name",
        type=str,
        default=None,
        help=(
            "Optional Hugging Face model name/path to use as the explicit Shadow backbone. "
            "If provided, Shadow-PEFT will be constructed with "
            "`get_shadow_model(base_model, shadow_config, shadow_model=shadow_model)`."
        ),
    )
    parser.add_argument(
        "--remove_shadow_embed_tokens",
        type=int,
        default=None,
        choices=[0, 1],
        help=(
            "Only relevant when --pretrained_shadow_model_name is set. If 1, call "
            "`prepare_shadow_model(shadow_model, remove_embed_tokens=True)` before constructing "
            "ShadowPEFT, so the explicit shadow backbone is driven by base `inputs_embeds`."
        ),
    )
    parser.add_argument("--dataset_name", type=str, default="tiny_shakespeare")
    parser.add_argument("--dataset_config", type=str, default=None)
    parser.add_argument("--text_column", type=str, default="text")
    parser.add_argument("--text_pair_column", type=str, default=None)
    parser.add_argument("--text_template", type=str, default=None)
    parser.add_argument("--label_column", type=str, default="label")
    parser.add_argument("--block_size", type=int, default=512)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--generation_max_length", type=int, default=None)
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="flash_attention_2",
        choices=("sdpa", "eager", "flash_attention_2"),
    )

    parser.add_argument("--per_device_train_batch_size", type=int, default=128)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=128)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--eval_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=100000)
    parser.add_argument("--save_steps", type=int, default=0)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--report_to", type=str, default="wandb")
    parser.add_argument("--run_name", type=str, default="shadow_peft_experiment")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--print_shadow_output", type=int, default=1, choices=[0, 1])
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--eval_split", type=str, default="validation")
    parser.add_argument(
        "--mode",
        type=str,
        default="both",
        choices=("lora", "dora", "shadow", "both"),
        help="Which method(s) to run. 'dora' is an alias for running LoRA with --peft_method dora.",
    )
    parser.add_argument("--trainer", type=str, default="sft", choices=("sft", "grpo", "both"))
    parser.add_argument("--fp16", type=int, default=0, choices=[0, 1])
    parser.add_argument("--bf16", type=int, default=1, choices=[0, 1])
    parser.add_argument("--classification_metric", type=str, default="accuracy")

    # LoRA
    parser.add_argument("--peft_method", type=str, default="lora", choices=("lora", "dora"))
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", type=str, nargs="+", default=["q_proj", "k_proj", "v_proj", "o_proj"])

    # Shadow-PEFT
    parser.add_argument("--shadow_layers", type=int, default=1)
    parser.add_argument("--shadow_intermediate_size", type=int, default=128)
    parser.add_argument("--shadow_num_attention_heads", type=int, default=None)
    parser.add_argument("--shadow_num_key_value_heads", type=int, default=None)
    parser.add_argument("--shadow_head_dim", type=int, default=None)
    parser.add_argument("--shadow_loss_weight", type=float, default=0.05)
    parser.add_argument("--shadow_alpha", type=float, default=1.0)
    parser.add_argument("--shadow_dropout", type=float, default=0.2)
    parser.add_argument("--injection_hidden_size", type=int, default=16)
    parser.add_argument("--gate_hidden_size", type=int, default=10)
    parser.add_argument(
        "--modules_to_save",
        type=str,
        nargs="+",
        default=None,
        help="Optional extra trainable modules to save in Shadow checkpoints (PEFT-style). "
        "Examples: seqcls: classifier_head shadow_classifier_head; causal LM: lm_head shadow_lm_head.",
    )
    parser.add_argument(
        "--shadow_lr_multiplier",
        type=float,
        default=1.0,
        help="Learning rate multiplier for shadow-only parameters (shadow_model, shadow_lm_head, shadow_adapter). "
        "E.g., 0.1 means shadow params get 10%% of base LR. Default: 1.0 (same as base).",
    )

    # 1-bit quantized base + KL distillation (P2)
    parser.add_argument(
        "--quantize_base_group_size",
        type=int,
        default=None,
        help="If set (e.g. 128), quantize the frozen base model to 1-bit weights with this group size "
        "before building Shadow-PEFT. Default: None (no quantization).",
    )
    parser.add_argument(
        "--distill_from",
        type=str,
        default=None,
        help="Optional teacher model name/path (same architecture as the base, e.g. the fp16 original). "
        "When set, a frozen teacher is kept alongside and its logits are distilled into the student. "
        "Default: None (no distillation).",
    )
    parser.add_argument("--distill_weight", type=float, default=1.0)
    parser.add_argument("--distill_temperature", type=float, default=1.0)

    # MMLU fewshot
    parser.add_argument("--use_few_shot", type=int, default=0, choices=[0, 1])
    return parser.parse_args()


def _normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    """Backwards/UX compatibility normalizations."""
    # Support `--mode dora` as a shorthand for running the LoRA path with DoRA enabled.
    if getattr(args, "mode", None) == "dora":
        args.mode = "lora"
        args.peft_method = "dora"
    return args


def _make_run_name(args: argparse.Namespace, suffix: str) -> Optional[str]:
    if args.run_name is None:
        return None
    return f"{args.run_name}-{args.dataset_name}-{suffix}"


def _resolve_save_strategy(args: argparse.Namespace) -> str:
    # Avoid checkpoint saving by default (it can fail with tied/shared tensors under safetensors).
    return "no" if int(getattr(args, "save_steps", 0)) <= 0 else "steps"


def _configure_model_no_thinking(model, tokenizer):
    """Best-effort: avoid generating thinking blocks for chatty models (e.g. Qwen)."""
    gen_cfg = getattr(model, "generation_config", None)
    if gen_cfg is not None and hasattr(gen_cfg, "enable_thinking"):
        gen_cfg.enable_thinking = False

    if hasattr(tokenizer, "convert_tokens_to_ids") and gen_cfg is not None:
        stop_tokens = []
        for token in ["<think>", "<|im_start|>think"]:
            token_id = tokenizer.convert_tokens_to_ids(token)
            unk_id = getattr(tokenizer, "unk_token_id", None)
            if isinstance(token_id, int) and unk_id is not None and token_id != unk_id:
                stop_tokens.append(token_id)
        if stop_tokens and hasattr(gen_cfg, "eos_token_id"):
            if isinstance(gen_cfg.eos_token_id, list):
                gen_cfg.eos_token_id.extend(stop_tokens)
            else:
                gen_cfg.eos_token_id = [gen_cfg.eos_token_id] + stop_tokens


def _save_shadow_peft_adapter(model, tokenizer, output_dir: str) -> None:
    """
    Save ONLY Shadow-PEFT adapter weights/config via ShadowPEFT's `save_pretrained`.

    This intentionally avoids saving backbone weights (e.g. `pytorch_model.bin`).
    """
    os.makedirs(output_dir, exist_ok=True)
    actual = model.module if hasattr(model, "module") else model
    if not hasattr(actual, "save_pretrained"):
        raise AttributeError("Shadow model does not implement save_pretrained().")
    actual.save_pretrained(output_dir)
    # Tokenizer is safe to save (no backbone weights), and helps with reload/inference.
    if tokenizer is not None:
        tokenizer.save_pretrained(output_dir)


def _maybe_quantize_base_model(args: argparse.Namespace, base_model) -> None:
    """In-place 1-bit quantization of the frozen base model (before building Shadow-PEFT)."""
    group_size = getattr(args, "quantize_base_group_size", None)
    if group_size is None:
        return
    # Lazy import: the quantization module is only required when this flag is set.
    from shadow_peft.quantization import quantize_model_1bit

    manifest = quantize_model_1bit(base_model, group_size=int(group_size))
    manifest = manifest or {}
    quantized = manifest.get("quantized_layers") or manifest.get("quantized") or []
    skipped = manifest.get("skipped_layers") or manifest.get("skipped") or []
    print(
        f">>> Quantized base model to 1-bit (group_size={int(group_size)}): "
        f"{len(quantized)} quantized layers, {len(skipped)} skipped layers."
    )


def _prepare_distill_teacher(args: argparse.Namespace, tokenizer):
    """Load a frozen fp16/bf16 teacher on the same device flow as the base model (None if disabled)."""
    teacher_name = getattr(args, "distill_from", None)
    if not teacher_name:
        return None
    teacher = prepare_causal_model(teacher_name, args.attn_implementation)
    _sync_pad_token(teacher, tokenizer)
    if args.bf16:
        teacher = teacher.to(torch.bfloat16)
    elif args.fp16:
        teacher = teacher.to(torch.float16)
    teacher.eval()
    teacher.requires_grad_(False)
    print(f">>> Distillation teacher loaded: {teacher_name} (dtype={next(teacher.parameters()).dtype}).")
    return teacher


def _shadow_config_from_args(args: argparse.Namespace) -> ShadowConfig:
    return ShadowConfig(
        num_shadow_layers=int(args.shadow_layers),
        shadow_intermediate_size=int(args.shadow_intermediate_size),
        shadow_num_attention_heads=(
            None if getattr(args, "shadow_num_attention_heads", None) is None else int(args.shadow_num_attention_heads)
        ),
        shadow_num_key_value_heads=(
            None if getattr(args, "shadow_num_key_value_heads", None) is None else int(args.shadow_num_key_value_heads)
        ),
        shadow_head_dim=None if getattr(args, "shadow_head_dim", None) is None else int(args.shadow_head_dim),
        injection_hidden_size=int(args.injection_hidden_size),
        gate_hidden_size=int(args.gate_hidden_size),
        alpha=float(args.shadow_alpha),
        dropout=float(args.shadow_dropout),
        modules_to_save=list(getattr(args, "modules_to_save", None) or []),
    )


def _build_shadow_peft_causal_lm(
    args: argparse.Namespace, base_model, *, shadow_model=None
) -> ShadowForCausalLM:
    peft = get_shadow_model(base_model, _shadow_config_from_args(args), shadow_model=shadow_model)
    # Distillation only activates when a teacher is provided; otherwise keep the exact previous behavior.
    distill_enabled = getattr(args, "distill_from", None) is not None
    return ShadowForCausalLM(
        peft,
        shadow_loss_weight=float(args.shadow_loss_weight),
        inference_mode="base_shadow",
        distill_weight=float(getattr(args, "distill_weight", 1.0)) if distill_enabled else 0.0,
        distill_temperature=float(getattr(args, "distill_temperature", 1.0)),
    )


def _build_shadow_peft_seqcls(
    args: argparse.Namespace, base_model, *, shadow_model=None
) -> ShadowForSequenceClassification:
    peft = get_shadow_model(base_model, _shadow_config_from_args(args), shadow_model=shadow_model)
    return ShadowForSequenceClassification(peft, shadow_loss_weight=float(args.shadow_loss_weight), inference_mode="base_shadow")


def _build_peft_config(args: argparse.Namespace, task_type):
    base_kwargs = dict(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.lora_target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=task_type,
    )
    if getattr(args, "peft_method", "lora") != "dora":
        return LoraConfig(**base_kwargs)
    for flag in ("use_dora", "enable_dora", "dora"):
        try:
            return LoraConfig(**base_kwargs, **{flag: True})
        except TypeError:
            continue
    cfg = LoraConfig(**base_kwargs)
    for attr in ("use_dora", "enable_dora", "dora"):
        if hasattr(cfg, attr):
            setattr(cfg, attr, True)
            return cfg
    raise RuntimeError("Your installed `peft` does not appear to support DoRA.")


def run_generation_trainer(
    model,
    tokenizer,
    training_args: SFTConfig,
    datasets: ExperimentDatasetBundle,
    trainer_cls=SFTTrainer,
    trainer_kwargs: Optional[Dict] = None,
    *,
    save_model: bool = True,
):
    data_collator = LMDataCollatorWithPositions(tokenizer)
    dataset_text_field = "text" if "text" in datasets.train_dataset.column_names else None
    init_kwargs = dict(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=datasets.train_dataset,
        eval_dataset=datasets.eval_dataset,
        data_collator=data_collator,
    )
    if dataset_text_field is not None:
        init_kwargs["dataset_text_field"] = dataset_text_field
    if trainer_kwargs:
        init_kwargs.update(trainer_kwargs)
    trainer = trainer_cls(**init_kwargs)
    trainer.train()
    metrics = trainer.evaluate()
    if save_model:
        trainer.save_model(training_args.output_dir)
    return metrics


def generation_lora(args, tokenizer, datasets):
    model = prepare_causal_model(args.model_name, args.attn_implementation)
    _sync_pad_token(model, tokenizer)
    if args.bf16:
        model = model.to(torch.bfloat16)
    elif args.fp16:
        model = model.to(torch.float16)
    lora_config = _build_peft_config(args, task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    training_args = SFTConfig(
        output_dir=os.path.join(args.output_dir, getattr(args, "peft_method", "lora")),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_accumulation_steps=args.eval_accumulation_steps,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        eval_strategy="steps",
        save_strategy=_resolve_save_strategy(args),
        save_steps=args.save_steps,
        save_total_limit=1,
        warmup_ratio=args.warmup_ratio,
        report_to=args.report_to,
        run_name=_make_run_name(args, f"{getattr(args, 'peft_method', 'lora')}-gen"),
        # save_safetensors=False,
        fp16=args.fp16,
        bf16=args.bf16,
    )
    setattr(training_args, "generation_max_length", getattr(args, "generation_max_length", None))
    return run_generation_trainer(model, tokenizer, training_args, datasets, save_model=True)


def generation_shadow(args, tokenizer, datasets):
    base_model = prepare_causal_model(args.model_name, args.attn_implementation)
    _sync_pad_token(base_model, tokenizer)
    _maybe_quantize_base_model(args, base_model)
    shadow_model = None
    if getattr(args, "pretrained_shadow_model_name", None):
        shadow_model = prepare_causal_model(args.pretrained_shadow_model_name, args.attn_implementation)
        _sync_pad_token(shadow_model, tokenizer)
        if int(getattr(args, "remove_shadow_embed_tokens", 0) or 0) == 1:
            shadow_model = prepare_shadow_model(shadow_model, remove_embed_tokens=True)
    model = _build_shadow_peft_causal_lm(args, base_model, shadow_model=shadow_model)
    if args.bf16:
        model = model.to(torch.bfloat16)
    elif args.fp16:
        model = model.to(torch.float16)
    model.print_trainable_parameters()
    training_args = SFTConfig(
        output_dir=os.path.join(args.output_dir, "shadow_peft"),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_accumulation_steps=args.eval_accumulation_steps,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        eval_strategy="steps",
        save_strategy=_resolve_save_strategy(args),
        save_steps=args.save_steps,
        save_total_limit=1,
        warmup_ratio=args.warmup_ratio,
        report_to=args.report_to,
        run_name=_make_run_name(args, "shadow_peft-gen"),
        # save_safetensors=False,
        fp16=args.fp16,
        bf16=args.bf16,
    )
    setattr(training_args, "generation_max_length", getattr(args, "generation_max_length", None))
    metrics = run_generation_trainer(
        model,
        tokenizer,
        training_args,
        datasets,
        trainer_cls=ShadowSFTTrainer,
        trainer_kwargs={
            "shadow_lr_multiplier": float(getattr(args, "shadow_lr_multiplier", 1.0)),
            "teacher_model": _prepare_distill_teacher(args, tokenizer),
        },
        save_model=False,  # avoid saving backbone weights via Trainer
    )
    _save_shadow_peft_adapter(model, tokenizer, training_args.output_dir)
    return metrics


def run_generation_task(args: argparse.Namespace) -> Dict[str, Dict[str, float]]:
    tokenizer = prepare_tokenizer(args.model_name)
    datasets = build_datasets(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        tokenizer=tokenizer,
        block_size=args.block_size,
        train_split=args.train_split,
        eval_split=args.eval_split,
        text_column=args.text_column,
        text_template=args.text_template,
    )
    os.makedirs(args.output_dir, exist_ok=True)
    results: Dict[str, Dict[str, float]] = {}
    if args.mode in ("both", "lora"):
        results["lora"] = generation_lora(args, tokenizer, datasets)
    if args.mode in ("both", "shadow"):
        results["shadow_peft"] = generation_shadow(args, tokenizer, datasets)
    summary_path = os.path.join(args.output_dir, "metrics.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(f"Wrote aggregated metrics to {summary_path}")
    return results


def gsm8k_lora(args, tokenizer, datasets: GSM8KDatasetBundle):
    model = prepare_causal_model(args.model_name, args.attn_implementation)
    _sync_pad_token(model, tokenizer)
    if args.bf16:
        model = model.to(torch.bfloat16)
    elif args.fp16:
        model = model.to(torch.float16)

    lora_config = _build_peft_config(args, task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    _configure_model_no_thinking(model, tokenizer)

    training_args = SFTConfig(
        output_dir=os.path.join(args.output_dir, getattr(args, "peft_method", "lora")),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_accumulation_steps=args.eval_accumulation_steps,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        eval_strategy="steps",
        save_strategy=_resolve_save_strategy(args),
        save_steps=args.save_steps,
        save_total_limit=1,
        warmup_ratio=args.warmup_ratio,
        report_to=args.report_to,
        run_name=_make_run_name(args, f"{getattr(args, 'peft_method', 'lora')}-gsm8k"),
        # save_safetensors=False,
        fp16=args.fp16,
        bf16=args.bf16,
        max_length=args.max_seq_length,
        remove_unused_columns=False,
    )
    setattr(training_args, "print_shadow_output", bool(getattr(args, "print_shadow_output", False)))
    setattr(training_args, "generation_max_length", getattr(args, "generation_max_length", None))

    collator = GSM8KDataCollator(tokenizer)
    trainer = GSM8KTrainer(
        model=model,
        args=training_args,
        train_dataset=datasets.train_dataset,
        eval_dataset=datasets.eval_dataset,
        data_collator=collator,
        processing_class=tokenizer,
    )
    trainer.train()
    metrics = trainer.evaluate()
    trainer.save_model(training_args.output_dir)
    return metrics


def gsm8k_shadow(args, tokenizer, datasets: GSM8KDatasetBundle):
    base_model = prepare_causal_model(args.model_name, args.attn_implementation)
    _sync_pad_token(base_model, tokenizer)
    _maybe_quantize_base_model(args, base_model)
    shadow_model = None
    if getattr(args, "pretrained_shadow_model_name", None):
        shadow_model = prepare_causal_model(args.pretrained_shadow_model_name, args.attn_implementation)
        _sync_pad_token(shadow_model, tokenizer)
        if int(getattr(args, "remove_shadow_embed_tokens", 0) or 0) == 1:
            shadow_model = prepare_shadow_model(shadow_model, remove_embed_tokens=True)
    model = _build_shadow_peft_causal_lm(args, base_model, shadow_model=shadow_model)
    if args.bf16:
        model = model.to(torch.bfloat16)
    elif args.fp16:
        model = model.to(torch.float16)
    model.print_trainable_parameters()

    _configure_model_no_thinking(model, tokenizer)

    training_args = SFTConfig(
        output_dir=os.path.join(args.output_dir, "shadow_peft"),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_accumulation_steps=args.eval_accumulation_steps,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        eval_strategy="steps",
        save_strategy=_resolve_save_strategy(args),
        save_steps=args.save_steps,
        save_total_limit=1,
        warmup_ratio=args.warmup_ratio,
        report_to=args.report_to,
        run_name=_make_run_name(args, "shadow_peft-gsm8k"),
        # save_safetensors=False,
        fp16=args.fp16,
        bf16=args.bf16,
        max_length=args.max_seq_length,
        remove_unused_columns=False,
    )
    setattr(training_args, "print_shadow_output", bool(getattr(args, "print_shadow_output", False)))
    setattr(training_args, "generation_max_length", getattr(args, "generation_max_length", None))

    collator = GSM8KDataCollator(tokenizer)
    trainer = GSM8KTrainer(
        model=model,
        args=training_args,
        train_dataset=datasets.train_dataset,
        eval_dataset=datasets.eval_dataset,
        data_collator=collator,
        processing_class=tokenizer,
        shadow_lr_multiplier=float(getattr(args, "shadow_lr_multiplier", 1.0)),
        teacher_model=_prepare_distill_teacher(args, tokenizer),
    )
    trainer.train()
    metrics = trainer.evaluate()
    _save_shadow_peft_adapter(model, tokenizer, training_args.output_dir)
    return metrics


def run_gsm8k_task(args: argparse.Namespace) -> Dict[str, Dict[str, float]]:
    if args.trainer not in ("sft", "both"):
        raise NotImplementedError("GSM8K currently supports only --trainer sft in run_shadow_peft.py.")

    tokenizer = prepare_tokenizer(args.model_name)
    datasets: GSM8KDatasetBundle = build_gsm8k_datasets(
        tokenizer=tokenizer,
        subset=args.gsm8k_subset,
        max_length=args.max_seq_length,
        answer_mode=("final" if getattr(args, "gsm8k_answer_mode", "thinking") == "final" else "thinking"),
    )

    os.makedirs(args.output_dir, exist_ok=True)
    results: Dict[str, Dict[str, float]] = {}
    if args.mode in ("both", "lora"):
        results["lora_sft"] = gsm8k_lora(args, tokenizer, datasets)
    if args.mode in ("both", "shadow"):
        results["shadow_peft_sft"] = gsm8k_shadow(args, tokenizer, datasets)

    summary_path = os.path.join(args.output_dir, "metrics.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(f"Wrote aggregated metrics to {summary_path}")
    return results


def squad_v2_lora(args, tokenizer, datasets: SquadV2DatasetBundle):
    model = prepare_causal_model(args.model_name, args.attn_implementation)
    _sync_pad_token(model, tokenizer)
    if args.bf16:
        model = model.to(torch.bfloat16)
    elif args.fp16:
        model = model.to(torch.float16)

    lora_config = _build_peft_config(args, task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    _configure_model_no_thinking(model, tokenizer)

    training_args = SFTConfig(
        output_dir=os.path.join(args.output_dir, getattr(args, "peft_method", "lora")),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_accumulation_steps=args.eval_accumulation_steps,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        eval_strategy="steps",
        save_strategy=_resolve_save_strategy(args),
        save_steps=args.save_steps,
        save_total_limit=1,
        warmup_ratio=args.warmup_ratio,
        report_to=args.report_to,
        run_name=_make_run_name(args, f"{getattr(args, 'peft_method', 'lora')}-squad_v2"),
        # save_safetensors=False,
        fp16=args.fp16,
        bf16=args.bf16,
        max_length=args.max_seq_length,
        remove_unused_columns=False,
    )
    setattr(training_args, "print_shadow_output", bool(getattr(args, "print_shadow_output", False)))
    setattr(training_args, "generation_max_length", getattr(args, "generation_max_length", None))

    collator = SquadV2DataCollator(tokenizer)
    trainer = SquadV2Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets.train_dataset,
        eval_dataset=datasets.eval_dataset,
        data_collator=collator,
        processing_class=tokenizer,
    )
    trainer.train()
    metrics = trainer.evaluate()
    trainer.save_model(training_args.output_dir)
    return metrics


def squad_v2_shadow(args, tokenizer, datasets: SquadV2DatasetBundle):
    base_model = prepare_causal_model(args.model_name, args.attn_implementation)
    _sync_pad_token(base_model, tokenizer)
    _maybe_quantize_base_model(args, base_model)
    shadow_model = None
    if getattr(args, "pretrained_shadow_model_name", None):
        shadow_model = prepare_causal_model(args.pretrained_shadow_model_name, args.attn_implementation)
        _sync_pad_token(shadow_model, tokenizer)
        if int(getattr(args, "remove_shadow_embed_tokens", 0) or 0) == 1:
            shadow_model = prepare_shadow_model(shadow_model, remove_embed_tokens=True)
    model = _build_shadow_peft_causal_lm(args, base_model, shadow_model=shadow_model)
    if args.bf16:
        model = model.to(torch.bfloat16)
    elif args.fp16:
        model = model.to(torch.float16)
    model.print_trainable_parameters()

    _configure_model_no_thinking(model, tokenizer)

    training_args = SFTConfig(
        output_dir=os.path.join(args.output_dir, "shadow_peft"),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_accumulation_steps=args.eval_accumulation_steps,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        eval_strategy="steps",
        save_strategy=_resolve_save_strategy(args),
        save_steps=args.save_steps,
        save_total_limit=1,
        warmup_ratio=args.warmup_ratio,
        report_to=args.report_to,
        run_name=_make_run_name(args, "shadow_peft-squad_v2"),
        # save_safetensors=False,
        fp16=args.fp16,
        bf16=args.bf16,
        max_length=args.max_seq_length,
        remove_unused_columns=False,
    )
    setattr(training_args, "print_shadow_output", bool(getattr(args, "print_shadow_output", False)))
    setattr(training_args, "generation_max_length", getattr(args, "generation_max_length", None))

    collator = SquadV2DataCollator(tokenizer)
    trainer = SquadV2Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets.train_dataset,
        eval_dataset=datasets.eval_dataset,
        data_collator=collator,
        processing_class=tokenizer,
        shadow_lr_multiplier=float(getattr(args, "shadow_lr_multiplier", 1.0)),
        teacher_model=_prepare_distill_teacher(args, tokenizer),
    )
    trainer.train()
    metrics = trainer.evaluate()
    _save_shadow_peft_adapter(model, tokenizer, training_args.output_dir)
    return metrics


def run_squad_v2_task(args: argparse.Namespace) -> Dict[str, Dict[str, float]]:
    if args.trainer not in ("sft", "both"):
        raise NotImplementedError("SQuAD v2 currently supports only --trainer sft in run_shadow_peft.py.")

    tokenizer = prepare_tokenizer(args.model_name)
    datasets: SquadV2DatasetBundle = build_squad_v2_datasets(
        tokenizer=tokenizer,
        max_length=args.max_seq_length,
        answer_mode=("final" if getattr(args, "squad_answer_mode", "final") == "final" else "thinking"),
    )

    os.makedirs(args.output_dir, exist_ok=True)
    results: Dict[str, Dict[str, float]] = {}
    if args.mode in ("both", "lora"):
        results["lora_sft"] = squad_v2_lora(args, tokenizer, datasets)
    if args.mode in ("both", "shadow"):
        results["shadow_peft_sft"] = squad_v2_shadow(args, tokenizer, datasets)

    summary_path = os.path.join(args.output_dir, "metrics.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(f"Wrote aggregated metrics to {summary_path}")
    return results


def mmlu_lora(args, tokenizer, datasets: MMLUDatasetBundle):
    model = prepare_causal_model(args.model_name, args.attn_implementation)
    _sync_pad_token(model, tokenizer)
    if args.bf16:
        model = model.to(torch.bfloat16)
    elif args.fp16:
        model = model.to(torch.float16)
    lora_config = _build_peft_config(args, task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    training_args = SFTConfig(
        output_dir=os.path.join(args.output_dir, getattr(args, "peft_method", "lora")),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_accumulation_steps=args.eval_accumulation_steps,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        eval_strategy="steps",
        save_strategy=_resolve_save_strategy(args),
        save_steps=args.save_steps,
        save_total_limit=1,
        warmup_ratio=args.warmup_ratio,
        report_to=args.report_to,
        run_name=_make_run_name(args, f"{getattr(args, 'peft_method', 'lora')}-mmlu"),
        # save_safetensors=False,
        fp16=args.fp16,
        bf16=args.bf16,
        max_length=args.max_seq_length,
        remove_unused_columns=False,
    )
    setattr(training_args, "print_shadow_output", bool(getattr(args, "print_shadow_output", False)))
    setattr(training_args, "shadow_gen_max_samples", int(getattr(args, "shadow_gen_max_samples", 0)))
    setattr(training_args, "shadow_gen_sample_rate", float(getattr(args, "shadow_gen_sample_rate", 1.0)))
    setattr(training_args, "generation_max_length", getattr(args, "generation_max_length", None))
    data_collator = MMLUDataCollator(tokenizer)
    eval_reference_dataset = _pick_reference_eval_dataset(datasets.eval_datasets)
    trainer = MMLUTrainer(
        model=model,
        args=training_args,
        train_dataset=datasets.train_dataset,
        eval_dataset=eval_reference_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
    )
    trainer.train()
    metrics = _evaluate_mmlu_subsets(trainer, datasets.eval_datasets)
    trainer.save_model(training_args.output_dir)
    return metrics


def mmlu_shadow(args, tokenizer, datasets: MMLUDatasetBundle):
    base_model = prepare_causal_model(args.model_name, args.attn_implementation)
    _sync_pad_token(base_model, tokenizer)
    _maybe_quantize_base_model(args, base_model)
    shadow_model = None
    if getattr(args, "pretrained_shadow_model_name", None):
        shadow_model = prepare_causal_model(args.pretrained_shadow_model_name, args.attn_implementation)
        _sync_pad_token(shadow_model, tokenizer)
        if int(getattr(args, "remove_shadow_embed_tokens", 0) or 0) == 1:
            shadow_model = prepare_shadow_model(shadow_model, remove_embed_tokens=True)
    model = _build_shadow_peft_causal_lm(args, base_model, shadow_model=shadow_model)
    if args.bf16:
        model = model.to(torch.bfloat16)
    elif args.fp16:
        model = model.to(torch.float16)
    model.print_trainable_parameters()
    training_args = SFTConfig(
        output_dir=os.path.join(args.output_dir, "shadow_peft"),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_accumulation_steps=args.eval_accumulation_steps,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        eval_strategy="steps",
        save_strategy=_resolve_save_strategy(args),
        save_steps=args.save_steps,
        save_total_limit=1,
        warmup_ratio=args.warmup_ratio,
        report_to=args.report_to,
        run_name=_make_run_name(args, "shadow_peft-mmlu"),
        # save_safetensors=False,
        fp16=args.fp16,
        bf16=args.bf16,
        max_length=args.max_seq_length,
        remove_unused_columns=False,
    )
    setattr(training_args, "print_shadow_output", bool(getattr(args, "print_shadow_output", False)))
    setattr(training_args, "shadow_gen_max_samples", int(getattr(args, "shadow_gen_max_samples", 0)))
    setattr(training_args, "shadow_gen_sample_rate", float(getattr(args, "shadow_gen_sample_rate", 1.0)))
    setattr(training_args, "generation_max_length", getattr(args, "generation_max_length", None))
    data_collator = MMLUDataCollator(tokenizer)
    eval_reference_dataset = _pick_reference_eval_dataset(datasets.eval_datasets)
    trainer = MMLUTrainer(
        model=model,
        args=training_args,
        train_dataset=datasets.train_dataset,
        eval_dataset=eval_reference_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
        shadow_lr_multiplier=float(getattr(args, "shadow_lr_multiplier", 1.0)),
        teacher_model=_prepare_distill_teacher(args, tokenizer),
    )
    trainer.train()
    metrics = _evaluate_mmlu_subsets(trainer, datasets.eval_datasets)
    _save_shadow_peft_adapter(model, tokenizer, training_args.output_dir)
    return metrics


def run_mmlu_task(args: argparse.Namespace) -> Dict[str, Dict[str, Dict[str, float]]]:
    tokenizer = prepare_tokenizer(args.model_name)
    eval_subsets = _resolve_mmlu_eval_subsets(args.mmlu_subset)
    use_few_shot = getattr(args, "use_few_shot", False)
    datasets = build_mmlu_datasets(
        tokenizer=tokenizer,
        max_length=args.max_seq_length,
        train_subset="auxiliary_train",
        train_split="train",
        eval_subsets=eval_subsets,
        eval_split="test",
        use_few_shot=use_few_shot,
    )
    os.makedirs(args.output_dir, exist_ok=True)
    results: Dict[str, Dict[str, Dict[str, float]]] = {}
    if args.trainer in ("sft", "both"):
        if args.mode in ("both", "lora"):
            results["lora_sft"] = mmlu_lora(args, tokenizer, datasets)
        if args.mode in ("both", "shadow"):
            results["shadow_peft_sft"] = mmlu_shadow(args, tokenizer, datasets)
    summary_path = os.path.join(args.output_dir, "metrics.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(f"Wrote aggregated metrics to {summary_path}")
    return results


def classification_lora(args, tokenizer, datasets: ClassificationDatasetBundle):
    base_model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(datasets.label2id),
        id2label=datasets.id2label,
        label2id=datasets.label2id,
    )
    _set_attn_impl(base_model, args.attn_implementation)
    _sync_pad_token(base_model, tokenizer)
    if args.bf16:
        base_model = base_model.to(torch.bfloat16)
    elif args.fp16:
        base_model = base_model.to(torch.float16)
    lora_config = _build_peft_config(args, task_type=TaskType.SEQ_CLS)
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()
    collator = ClassificationCollatorWithPositions(tokenizer)
    metric = evaluate.load(args.classification_metric)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        preds = np.argmax(logits, axis=-1)
        return metric.compute(predictions=preds, references=labels)

    training_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, getattr(args, "peft_method", "lora")),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_accumulation_steps=args.eval_accumulation_steps,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        eval_strategy="steps",
        save_strategy=_resolve_save_strategy(args),
        save_steps=args.save_steps,
        save_total_limit=1,
        warmup_ratio=args.warmup_ratio,
        report_to=args.report_to,
        run_name=_make_run_name(args, f"{getattr(args, 'peft_method', 'lora')}-cls"),
        # save_safetensors=False,
        fp16=bool(args.fp16),
        bf16=bool(args.bf16),
        # ShadowPEFT task wrappers use `*args/**kwargs` forwards, so Trainer can't infer
        # which columns are used; keep token columns to avoid collator getting only labels.
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets.train_dataset,
        eval_dataset=datasets.eval_dataset,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    metrics = trainer.evaluate()
    trainer.save_model(training_args.output_dir)
    return metrics


def classification_shadow(args, tokenizer, datasets: ClassificationDatasetBundle):
    base_model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(datasets.label2id),
        id2label=datasets.id2label,
        label2id=datasets.label2id,
    )
    _set_attn_impl(base_model, args.attn_implementation)
    _sync_pad_token(base_model, tokenizer)
    shadow_model = None
    if getattr(args, "pretrained_shadow_model_name", None):
        # Use prepare_causal_model first so that AutoModelForCausalLMWithHiddenProjection
        # (projected shadow models) are detected and loaded correctly without triggering
        # an AutoModelForSequenceClassification load error.
        shadow_model = prepare_causal_model(args.pretrained_shadow_model_name, args.attn_implementation)
        from shadow_peft import AutoModelForCausalLMWithHiddenProjection as _ProjCausalLM
        if not isinstance(shadow_model, _ProjCausalLM):
            # Regular causal model: try to reload with a classification head.
            try:
                shadow_model = AutoModelForSequenceClassification.from_pretrained(
                    args.pretrained_shadow_model_name,
                    num_labels=len(datasets.label2id),
                    id2label=datasets.id2label,
                    label2id=datasets.label2id,
                )
            except Exception as e:
                print(f"Warning: could not reload shadow as SeqCls ({e}), using causal model.")
        _set_attn_impl(shadow_model, args.attn_implementation)
        _sync_pad_token(shadow_model, tokenizer)
        if int(getattr(args, "remove_shadow_embed_tokens", 0) or 0) == 1:
            shadow_model = prepare_shadow_model(shadow_model, remove_embed_tokens=True)
    model = _build_shadow_peft_seqcls(args, base_model, shadow_model=shadow_model)
    if args.bf16:
        model = model.to(torch.bfloat16)
    elif args.fp16:
        model = model.to(torch.float16)
    model.peft_model.print_trainable_parameters()  # type: ignore[attr-defined]
    collator = ClassificationCollatorWithPositions(tokenizer)
    metric = evaluate.load(args.classification_metric)
    shadow_metric = evaluate.load(args.classification_metric)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        shadow_logits = None
        if isinstance(logits, (tuple, list)):
            main_logits = logits[0]
            if len(logits) > 1:
                shadow_logits = logits[1]
        else:
            main_logits = logits
        preds = np.argmax(main_logits, axis=-1)
        results = metric.compute(predictions=preds, references=labels)
        if shadow_logits is not None:
            shadow_preds = np.argmax(shadow_logits, axis=-1)
            shadow_results = shadow_metric.compute(predictions=shadow_preds, references=labels)
            results.update({f"shadow_{k}": v for k, v in shadow_results.items()})
        return results

    training_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, "shadow_peft"),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_accumulation_steps=args.eval_accumulation_steps,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        eval_strategy="steps",
        save_strategy=_resolve_save_strategy(args),
        save_steps=args.save_steps,
        save_total_limit=1,
        warmup_ratio=args.warmup_ratio,
        report_to=args.report_to,
        run_name=_make_run_name(args, "shadow_peft-cls"),
        # save_safetensors=False,
        fp16=bool(args.fp16),
        bf16=bool(args.bf16),
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets.train_dataset,
        eval_dataset=datasets.eval_dataset,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    metrics = trainer.evaluate()
    _save_shadow_peft_adapter(model, tokenizer, training_args.output_dir)
    return metrics


def run_classification_task(args: argparse.Namespace) -> Dict[str, Dict[str, float]]:
    tokenizer = prepare_tokenizer(args.model_name)
    datasets = build_classification_datasets(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        tokenizer=tokenizer,
        text_column=args.text_column,
        text_pair_column=args.text_pair_column,
        label_column=args.label_column,
        max_seq_length=args.max_seq_length,
        train_split=args.train_split,
        eval_split=args.eval_split,
    )
    os.makedirs(args.output_dir, exist_ok=True)
    results: Dict[str, Dict[str, float]] = {}
    if args.mode in ("both", "lora"):
        results["lora"] = classification_lora(args, tokenizer, datasets)
    if args.mode in ("both", "shadow"):
        results["shadow_peft"] = classification_shadow(args, tokenizer, datasets)
    summary_path = os.path.join(args.output_dir, "metrics.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(f"Wrote aggregated metrics to {summary_path}")
    return results


def run_suite(args: argparse.Namespace) -> Dict[str, Dict[str, Dict[str, float]]]:
    suite_results: Dict[str, Dict[str, Dict[str, float]]] = {}
    selections: List[tuple[str, List[Dict]]] = []
    if args.suite in ("mmlu", "generation", "all"):
        selections.append(("mmlu", MMLU_SUITE))
    if args.suite in ("gsm8k", "all"):
        selections.append(("gsm8k", GSM8K_SUITE))
    if args.suite in ("squad_v2", "all"):
        selections.append(("squad_v2", [{"id": "squad_v2", "model_name": args.model_name, "max_seq_length": args.max_seq_length}]))
    if args.suite in ("classification", "all"):
        selections.append(("classification", CLASSIFICATION_SUITE))

    for task, specs in selections:
        if task == "classification" and getattr(args, "classification_id", None):
            specs = [s for s in specs if s.get("id") == args.classification_id]
            if not specs:
                raise ValueError(f"Unknown classification_id={args.classification_id}")
        for spec in specs:
            run_args = copy.deepcopy(args)
            run_args.task = task
            for key, value in spec.items():
                if key == "id":
                    continue
                setattr(run_args, key, value)
            run_args.output_dir = os.path.join(args.output_dir, f"{task}_{spec['id']}")
            if args.run_name is not None:
                run_args.run_name = f"{args.run_name}_{spec['id']}"
            print(f"=== Running {task} dataset '{spec['id']}' ===")
            if task == "mmlu":
                result = run_mmlu_task(run_args)
            elif task == "generation":
                result = run_generation_task(run_args)
            elif task == "gsm8k":
                result = run_gsm8k_task(run_args)
            elif task == "squad_v2":
                result = run_squad_v2_task(run_args)
            elif task == "classification":
                result = run_classification_task(run_args)
            else:
                raise NotImplementedError(f"Task not yet implemented in run_shadow_peft.py: {task}")
            suite_results[f"{task}:{spec['id']}"] = result

    summary_path = os.path.join(args.output_dir, "suite_metrics_shadow_peft.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(suite_results, handle, indent=2)
    print(f"Wrote suite metrics to {summary_path}")
    return suite_results


def main():
    args = _normalize_args(parse_args())
    set_seed(args.seed)
    if args.suite is not None:
        run_suite(args)
        return

    if args.task == "mmlu":
        run_mmlu_task(args)
    elif args.task == "generation":
        run_generation_task(args)
    elif args.task == "gsm8k":
        run_gsm8k_task(args)
    elif args.task == "squad_v2":
        run_squad_v2_task(args)
    elif args.task == "classification":
        run_classification_task(args)
    else:
        raise NotImplementedError(
            f"Task not yet implemented in run_shadow_peft.py: {args.task}."
        )


if __name__ == "__main__":
    main()
