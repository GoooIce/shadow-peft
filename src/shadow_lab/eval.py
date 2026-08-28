"""Shadow-aware benchmark evaluation assets: one home for the shared machinery.

Suites, collators, answer extraction, position-id construction, MMLU subset
plumbing, and model/tokenizer preparation — all importable without the
training stack. The SFTTrainer subclasses live in ``lab.trainers`` (they
require ``trl``); the GSM8K trainer there takes an ``answer_extractor`` so
callers swap extraction strategies through the interface instead of
monkey-patching module globals.
"""

from __future__ import annotations

import contextlib
import json
import re

import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    DataCollatorWithPadding,
    Trainer,
)

from .data import extract_gsm8k_final_answer

# MMLU subtasks from lighteval/mmlu dataset
MMLU_SUBTASKS = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
]

MMLU_SUITE = [
    {
        "id": "mmlu_full",
        "mmlu_subset": "all",
        "max_seq_length": 512,
    }
]

GSM8K_SUITE = [
    {
        "id": "gsm8k_main",
        "gsm8k_subset": "main",
        "max_seq_length": 512,
    }
]

SQUAD_V2_SUITE = [
    {
        "id": "squad_v2",
        "max_seq_length": 512,
    }
]

DEFAULT_MMLU_GENERATION_TOKENS = 10
DEFAULT_GSM8K_GENERATION_TOKENS = 256
DEFAULT_SQUAD_V2_GENERATION_TOKENS = 32

REWARD_CORRECT_ANSWER = 1.0
REWARD_WRONG_ANSWER = -0.5
REWARD_INVALID_FORMAT = -1.0


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _unwrap_model(model):
    """Unwrap DDP/DataParallel wrappers."""
    return model.module if hasattr(model, "module") else model


def _shifted_ce_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )


def _resolve_mmlu_eval_subsets(requested_subset: str | None) -> list[str]:
    if requested_subset in (None, "all"):
        return list(MMLU_SUBTASKS)
    if requested_subset == "auxiliary_train":
        raise ValueError("Evaluation subset cannot be 'auxiliary_train'.")
    return [requested_subset]


def _pick_reference_eval_dataset(eval_datasets: dict[str, object]):
    for dataset in eval_datasets.values():
        return dataset
    raise ValueError("At least one evaluation subset is required for MMLU.")


def _evaluate_mmlu_subsets(trainer: Trainer, eval_datasets: dict[str, object]):
    metrics: dict[str, dict[str, float]] = {}
    for subset_name, eval_dataset in eval_datasets.items():
        subset_metrics = trainer.evaluate(eval_dataset=eval_dataset, metric_key_prefix="eval")
        entry: dict[str, float] = {
            "loss": subset_metrics.get("eval_loss"),
            "accuracy": subset_metrics.get("eval_accuracy"),
            "samples": subset_metrics.get("eval_samples"),
        }
        if subset_metrics.get("eval_shadow_loss") is not None:
            entry["shadow_loss"] = subset_metrics.get("eval_shadow_loss")
        if subset_metrics.get("eval_shadow_accuracy") is not None:
            entry["shadow_accuracy"] = subset_metrics.get("eval_shadow_accuracy")
        metrics[subset_name] = entry
    return metrics


# ---------------------------------------------------------------------------
# Model / tokenizer preparation
# ---------------------------------------------------------------------------

def prepare_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        pad_fallback = tokenizer.eos_token or tokenizer.bos_token
        if pad_fallback is None:
            raise ValueError("Tokenizer is missing pad/eos/bos tokens.")
        tokenizer.pad_token = pad_fallback
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    return tokenizer


def _sync_pad_token(model, tokenizer):
    if hasattr(model, "config") and getattr(model.config, "pad_token_id", None) is None:
        if tokenizer.pad_token_id is None:
            raise ValueError("Tokenizer pad_token_id is undefined.")
        model.config.pad_token_id = tokenizer.pad_token_id


def _set_attn_impl(model, attn_impl: str):
    if hasattr(model, "config"):
        with contextlib.suppress(AttributeError):
            model.config._attn_implementation = attn_impl


def prepare_causal_model(model_name: str, attn_impl: str):
    # Support ShadowPEFT's projected shadow model wrapper (AutoModelForCausalLMWithHiddenProjection).
    # This model uses a custom `model_type` and cannot be loaded via AutoModelForCausalLM.
    try:
        from pathlib import Path

        raw_cfg = None
        revision = None

        # Local folder?
        p = Path(model_name)
        cfg_path = p / "config.json"
        if cfg_path.exists():
            with open(cfg_path, encoding="utf-8") as f:
                raw_cfg = json.load(f)
        else:
            # Hub repo id (optionally `repo@revision`)
            repo_spec = str(model_name)
            if "@" in repo_spec and "/" in repo_spec and not repo_spec.startswith(("/", ".", "~")):
                repo_id, revision = repo_spec.rsplit("@", 1)
            else:
                repo_id = repo_spec

            try:
                from huggingface_hub import hf_hub_download

                cfg_file = hf_hub_download(repo_id=repo_id, filename="config.json", revision=revision)
                with open(cfg_file, encoding="utf-8") as f:
                    raw_cfg = json.load(f)
            except Exception:
                raw_cfg = None

        if isinstance(raw_cfg, dict) and raw_cfg.get("model_type") in (
            "shadow_proj_causal_lm",
            "causal_lm_with_hidden_projection",
        ):
            from shadow_peft import AutoModelForCausalLMWithHiddenProjection

            model = AutoModelForCausalLMWithHiddenProjection.from_pretrained(
                model_name,
                revision=revision,
                freeze_backbone=False,
            )
            if hasattr(model.config, "use_cache"):
                model.config.use_cache = False
            return model
    except Exception as err:
        # Fall back to standard HF loading.
        print(f">>> Error loading shadow model: {err}")

    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation=attn_impl)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_name)
        _set_attn_impl(model, attn_impl)
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    return model


# ---------------------------------------------------------------------------
# Position ids + collators
# ---------------------------------------------------------------------------

def _build_position_ids(attention_mask: torch.Tensor, padding_side: str) -> torch.Tensor:
    device = attention_mask.device
    seq_length = attention_mask.size(1)
    if padding_side == "left":
        position_ids = attention_mask.long().cumsum(dim=1) - 1
        position_ids.clamp_(min=0)
        position_ids = position_ids.masked_fill(attention_mask == 0, 0)
        return position_ids.to(device=device).clone()
    base = torch.arange(seq_length, device=device).unsqueeze(0)
    return base.expand_as(attention_mask).clone()


class LMDataCollatorWithPositions:
    def __init__(self, tokenizer):
        self.base = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        self.tokenizer = tokenizer

    def __call__(self, features):
        batch = self.base(features)
        attention_mask = batch.get("attention_mask")
        if attention_mask is None:
            attention_mask = (batch["input_ids"] != self.tokenizer.pad_token_id).long()
            batch["attention_mask"] = attention_mask
        attention_mask = attention_mask.to(dtype=torch.bool)
        batch["attention_mask"] = attention_mask
        batch["position_ids"] = _build_position_ids(attention_mask.long(), self.tokenizer.padding_side)
        return batch


class ClassificationCollatorWithPositions:
    def __init__(self, tokenizer):
        self.base = DataCollatorWithPadding(tokenizer=tokenizer)
        self.tokenizer = tokenizer

    def __call__(self, features):
        batch = self.base(features)
        attention_mask = batch.get("attention_mask")
        if attention_mask is None:
            attention_mask = (batch["input_ids"] != self.tokenizer.pad_token_id).long()
            batch["attention_mask"] = attention_mask
        attention_mask = attention_mask.to(dtype=torch.bool)
        batch["attention_mask"] = attention_mask
        batch["position_ids"] = _build_position_ids(attention_mask.long(), self.tokenizer.padding_side)
        return batch


class GSM8KDataCollator:
    """Data collator for GSM8K that preserves prompt and gold answer."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, features):
        gold_answers = [f.get("gold_answer", "") for f in features]

        input_ids_list = [f["input_ids"] for f in features]
        attention_list = [f.get("attention_mask", [1] * len(f["input_ids"])) for f in features]
        labels_list = [f.get("labels", f["input_ids"]) for f in features]

        max_length = max(len(ids) for ids in input_ids_list)
        batch_input_ids = []
        batch_labels = []
        batch_attention = []
        for ids, mask, labels in zip(input_ids_list, attention_list, labels_list, strict=False):
            pad_len = max_length - len(ids)
            batch_input_ids.append(ids + [self.pad_token_id] * pad_len)
            batch_attention.append(mask + [0] * pad_len)
            batch_labels.append(labels + [-100] * pad_len)

        batch = {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention, dtype=torch.bool),
        }

        batch["position_ids"] = _build_position_ids(
            batch["attention_mask"].long(), self.tokenizer.padding_side
        )

        prompt_input_ids_list = [f.get("prompt_input_ids", f["input_ids"]) for f in features]
        prompt_attention_list = []
        for f, prompt_ids in zip(features, prompt_input_ids_list, strict=False):
            prompt_mask = f.get("prompt_attention_mask")
            if prompt_mask is None:
                prompt_mask = [1] * len(prompt_ids)
            prompt_attention_list.append(prompt_mask)

        max_prompt_len = max(len(ids) for ids in prompt_input_ids_list)
        padded_prompt_ids = []
        padded_prompt_masks = []
        for ids, mask in zip(prompt_input_ids_list, prompt_attention_list, strict=False):
            pad_len = max_prompt_len - len(ids)
            padded_prompt_ids.append(ids + [self.pad_token_id] * pad_len)
            padded_prompt_masks.append(mask + [0] * pad_len)

        batch["prompt_input_ids"] = torch.tensor(padded_prompt_ids, dtype=torch.long)
        batch["prompt_attention_mask"] = torch.tensor(padded_prompt_masks, dtype=torch.bool)
        batch["gold_answers"] = gold_answers
        return batch


class SquadV2DataCollator:
    """Data collator for SQuAD v2 that preserves prompt + id/answers for metric eval."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, features):
        squad_ids = [f.get("squad_id", "") for f in features]
        squad_answers = [f.get("squad_answers", {"text": [], "answer_start": []}) for f in features]

        input_ids_list = [f["input_ids"] for f in features]
        attention_list = [f.get("attention_mask", [1] * len(f["input_ids"])) for f in features]
        labels_list = [f.get("labels", f["input_ids"]) for f in features]

        max_length = max(len(ids) for ids in input_ids_list)
        batch_input_ids = []
        batch_labels = []
        batch_attention = []
        for ids, mask, labels in zip(input_ids_list, attention_list, labels_list, strict=False):
            pad_len = max_length - len(ids)
            batch_input_ids.append(ids + [self.pad_token_id] * pad_len)
            batch_attention.append(mask + [0] * pad_len)
            batch_labels.append(labels + [-100] * pad_len)

        batch = {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention, dtype=torch.bool),
        }

        batch["position_ids"] = _build_position_ids(
            batch["attention_mask"].long(), self.tokenizer.padding_side
        )

        prompt_input_ids_list = [f.get("prompt_input_ids", f["input_ids"]) for f in features]
        prompt_attention_list = []
        for f, prompt_ids in zip(features, prompt_input_ids_list, strict=False):
            prompt_mask = f.get("prompt_attention_mask")
            if prompt_mask is None:
                prompt_mask = [1] * len(prompt_ids)
            prompt_attention_list.append(prompt_mask)

        max_prompt_len = max(len(ids) for ids in prompt_input_ids_list)
        padded_prompt_ids = []
        padded_prompt_masks = []
        for ids, mask in zip(prompt_input_ids_list, prompt_attention_list, strict=False):
            pad_len = max_prompt_len - len(ids)
            padded_prompt_ids.append(ids + [self.pad_token_id] * pad_len)
            padded_prompt_masks.append(mask + [0] * pad_len)

        batch["prompt_input_ids"] = torch.tensor(padded_prompt_ids, dtype=torch.long)
        batch["prompt_attention_mask"] = torch.tensor(padded_prompt_masks, dtype=torch.bool)
        batch["squad_ids"] = squad_ids
        batch["squad_answers"] = squad_answers
        return batch


class MMLUDataCollator:
    """Data collator for MMLU that preserves answer information + prompt inputs."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, features):
        # Extract answer info FIRST before any processing
        answer_letters = [f.get("answer_letter", None) for f in features]
        answer_indices = [f.get("answer_idx", None) for f in features]

        # Remove answer fields from features for tokenization
        features_for_collation = []
        for f in features:
            f_clean = {k: v for k, v in f.items() if k not in ["answer_letter", "answer_idx"]}
            features_for_collation.append(f_clean)

        # Manual padding and batching
        input_ids_list = [f["input_ids"] for f in features_for_collation]
        labels_list = [f.get("labels", f["input_ids"]) for f in features_for_collation]
        max_length = max(len(ids) for ids in input_ids_list)

        batch_input_ids = []
        batch_labels = []
        for ids, labels in zip(input_ids_list, labels_list, strict=False):
            pad_len = max_length - len(ids)
            batch_input_ids.append(ids + [self.pad_token_id] * pad_len)
            batch_labels.append(labels + [-100] * pad_len)

        batch = {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }

        attention_mask = (batch["input_ids"] != self.pad_token_id).bool()
        batch["attention_mask"] = attention_mask
        batch["position_ids"] = _build_position_ids(attention_mask.long(), self.tokenizer.padding_side)

        # Prepare prompt-only inputs (everything up to "Answer: ")
        prompt_input_ids_list = [f.get("prompt_input_ids", f["input_ids"]) for f in features]
        prompt_attention_list = []
        for feature, prompt_ids in zip(features, prompt_input_ids_list, strict=False):
            prompt_mask = feature.get("prompt_attention_mask")
            if prompt_mask is None:
                prompt_mask = [1] * len(prompt_ids)
            elif len(prompt_mask) != len(prompt_ids):
                if len(prompt_mask) > len(prompt_ids):
                    prompt_mask = prompt_mask[: len(prompt_ids)]
                else:
                    prompt_mask = prompt_mask + [0] * (len(prompt_ids) - len(prompt_mask))
            prompt_attention_list.append(prompt_mask)

        max_prompt_len = max(len(ids) for ids in prompt_input_ids_list)
        padded_prompt_ids = []
        padded_prompt_masks = []
        for ids, mask in zip(prompt_input_ids_list, prompt_attention_list, strict=False):
            pad_len = max_prompt_len - len(ids)
            padded_prompt_ids.append(ids + [self.pad_token_id] * pad_len)
            padded_prompt_masks.append(mask + [0] * pad_len)

        batch["prompt_input_ids"] = torch.tensor(padded_prompt_ids, dtype=torch.long)
        batch["prompt_attention_mask"] = torch.tensor(padded_prompt_masks, dtype=torch.bool)

        batch["answer_letters"] = answer_letters
        batch["answer_indices"] = answer_indices
        return batch


# ---------------------------------------------------------------------------
# Generated-text cleaning + answer extraction (the single home)
# ---------------------------------------------------------------------------

_THINK_PATTERN = re.compile(r"<think>.*?</think>", flags=re.IGNORECASE | re.DOTALL)
_IM_TAG_PATTERN = re.compile(r"<\|im_(?:start|end)\|>")
_HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
_ANSWER_PATTERN = re.compile(r"(?:answer|option)\s*[:=-]?\s*([ABCD])", flags=re.IGNORECASE)
_LETTER_PATTERN = re.compile(r"\b([ABCD])\b", flags=re.IGNORECASE)
_BOXED_ANSWER_RE = re.compile(r"\\boxed\s*\{\s*([^}]*)\s*\}")
_NUMBER_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")


def _clean_generated_text(text: str) -> str:
    no_think = _THINK_PATTERN.sub(" ", text)
    no_im = _IM_TAG_PATTERN.sub(" ", no_think)
    no_tags = _HTML_TAG_PATTERN.sub(" ", no_im)
    return re.sub(r"\s+", " ", no_tags).strip()


def extract_answer_from_text(text: str) -> str | None:
    if not text:
        return None
    cleaned = _clean_generated_text(text)
    if not cleaned:
        return None
    match = _LETTER_PATTERN.search(cleaned)
    if match:
        return match.group(1).upper()
    match = _ANSWER_PATTERN.search(cleaned)
    if match:
        return match.group(1).upper()
    return None


def extract_gsm8k_answer_from_text(text: str) -> str:
    """Standard GSM8K extractor: clean the generation, parse the gold format."""
    if not text:
        return ""
    cleaned = _clean_generated_text(text)
    return extract_gsm8k_final_answer(cleaned)


def extract_gsm8k_boxed_answer(text: str) -> str:
    """Boxed-answer GSM8K extractor for prompts that demand $\\boxed{answer}$.

    Strategies, in order: \\boxed{...}, "Final Answer:", "#### " (GSM8K
    style), then the last number in the output.
    """
    if not text:
        return ""

    # Strategy 1: Look for \boxed{...} pattern (single or double $)
    match = _BOXED_ANSWER_RE.search(text)
    if match:
        boxed_content = match.group(1).strip()
        num = _NUMBER_RE.search(boxed_content)
        if num:
            return num.group(0).replace(",", "").strip()
        return boxed_content

    # Strategy 2: Look for "Final Answer:" pattern
    final_answer_pattern = re.compile(r"Final Answer:\s*\$?\\boxed\{([^}]+)\}\$?", re.IGNORECASE)
    match = final_answer_pattern.search(text)
    if match:
        answer_content = match.group(1).strip()
        num = _NUMBER_RE.search(answer_content)
        if num:
            return num.group(0).replace(",", "").strip()
        return answer_content

    # Strategy 3: Look for "#### " pattern (standard GSM8K)
    gsm8k_pattern = re.compile(r"####\s*([^\n\r]+)")
    match = gsm8k_pattern.search(text)
    if match:
        answer_content = match.group(1).strip()
        num = _NUMBER_RE.search(answer_content)
        if num:
            return num.group(0).replace(",", "").strip()
        return answer_content

    # Strategy 4: Fallback - take the last number in the output
    nums = _NUMBER_RE.findall(text)
    if nums:
        return nums[-1].replace(",", "").strip()

    return text.strip()


def compute_mmlu_reward(generated_texts: list[str], ground_truth_answers: list[str]) -> list[float]:
    rewards = []
    for gen_text, true_answer in zip(generated_texts, ground_truth_answers, strict=False):
        pred = extract_answer_from_text(gen_text) if gen_text else None
        if pred is None or pred not in ["A", "B", "C", "D"]:
            reward = REWARD_INVALID_FORMAT
        elif pred == true_answer.upper():
            reward = REWARD_CORRECT_ANSWER
        else:
            reward = REWARD_WRONG_ANSWER
        rewards.append(reward)
    return rewards
