"""On-policy rollout data pipeline for GKD-style distillation (issue #19).

Hard-case prompts exported by the feedback flywheel (see
``FeedbackStore.export_prompts``, issue #18) are sampled with the deployed
student policy (``base_shadow`` by default, ``shadow_only`` as an ablation).
Each prompt contributes ``num_rollouts`` trajectories; the resulting rows are
stable JSONL consumable by the on-policy trainer (issue #20).

Row contract::

    {
        "prompt": str, "task": str, "source": str, "split": "train" | "val",
        "prompt_ids": list[int], "rollout_ids": list[int], "labels": list[int],
        "finish_reason": "eos" | "length", "rollout_mode": str,
        "adapter_version": str, "seed": int,
    }

``labels`` masks every prompt position with ``-100`` and keeps the rollout
tokens, so the trainer can rebuild ``input_ids = prompt_ids + rollout_ids``
and a standard causal-LM shift without any extra alignment work.
"""

from __future__ import annotations

import json
import os
import random
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

ROLLOUT_MODES = ("base_shadow", "shadow_only")
FINISH_REASONS = ("eos", "length")
DISCARD_REASONS = ("empty", "immediate_eos", "too_long")


def _prompt_text(prompt: dict[str, Any]) -> str:
    return str(prompt.get("prompt", "") or "").strip()


def _confidence(prompt: dict[str, Any]) -> float:
    value = prompt.get("confidence")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("-inf")


def _dedupe_prompts(prompts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep one row per prompt text, first-seen order, highest confidence wins.

    Ties go to the last occurrence so a stable export always overrides stale
    copies. Rows with an empty/missing ``prompt`` are dropped.
    """
    order: list[str] = []
    best: dict[str, dict[str, Any]] = {}
    for prompt in prompts:
        text = _prompt_text(prompt)
        if not text:
            continue
        if text not in best:
            order.append(text)
            best[text] = prompt
            continue
        if _confidence(prompt) >= _confidence(best[text]):
            best[text] = prompt
    return [best[text] for text in order]


def load_prompts(path: str | Path) -> list[dict[str, Any]]:
    """Read a ``FeedbackStore.export_prompts`` JSONL file.

    Rows without a usable ``prompt`` are skipped, as are malformed/blank
    lines. Duplicate prompts are collapsed to a single row (highest
    ``confidence``; ties keep the last occurrence) so a hard case cannot
    dominate the rollout set. All other metadata is preserved untouched.
    """
    prompts: list[dict[str, Any]] = []
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                prompts.append(row)
    return _dedupe_prompts(prompts)


def split_prompts(
    prompts: list[dict[str, Any]],
    val_ratio: float = 0.1,
    seed: int = 0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split prompts (never individual rollouts) into train/val lists.

    The split unit is the prompt text: all G rollouts of a prompt land on the
    same side, so no rollout of a validation prompt can leak into training.
    Duplicate prompts are collapsed first (highest confidence wins), the
    remaining prompt keys are shuffled with ``random.Random(seed)``, and both
    returned lists keep the input order. At least one prompt stays in train
    whenever ``val_ratio < 1``; a positive ratio on a non-trivial set always
    reserves at least one validation prompt.
    """
    if not 0.0 <= val_ratio <= 1.0:
        raise ValueError(f"val_ratio must be in [0, 1], got {val_ratio}")

    unique = _dedupe_prompts(prompts)
    if not unique:
        return [], []

    n_val = int(round(len(unique) * val_ratio))
    if val_ratio > 0.0 and len(unique) >= 2 and n_val == 0:
        n_val = 1
    n_val = min(n_val, len(unique) - 1)

    shuffled_keys = [_prompt_text(row) for row in unique]
    random.Random(seed).shuffle(shuffled_keys)
    val_keys = set(shuffled_keys[:n_val])

    train = [row for row in unique if _prompt_text(row) not in val_keys]
    val = [row for row in unique if _prompt_text(row) in val_keys]
    return train, val


def _ids_from_encoded(encoded: Any) -> list[int]:
    """Normalize a tokenizer/chat-template return value to a flat id list."""
    if isinstance(encoded, Mapping):
        encoded = encoded["input_ids"]
    if hasattr(encoded, "tolist"):  # torch.Tensor / numpy array
        encoded = encoded.tolist()
    if encoded and isinstance(encoded[0], (list, tuple)):
        encoded = list(encoded[0])
    return [int(token) for token in encoded]


def _encode_prompt(tokenizer: Any, prompt: str) -> list[int]:
    """Tokenize a raw flywheel prompt the way the deployed chat path does.

    Tokenizers that ship a chat template get the user message rendered with
    ``add_generation_prompt`` (and Qwen-style ``enable_thinking=False`` when
    supported); plain tokenizers fall back to a normal encode.
    """
    if getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": prompt}]
        try:
            encoded = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            encoded = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
            )
        return _ids_from_encoded(encoded)
    return _ids_from_encoded(tokenizer(prompt, add_special_tokens=True))


@dataclass
class RolloutConfig:
    """Sampling and split knobs for one rollout-generation pass."""

    rollout_mode: str = "base_shadow"
    num_rollouts: int = 4
    max_new_tokens: int = 64
    temperature: float = 1.0
    top_p: float = 1.0
    seed: int = 0
    max_prompt_tokens: int | None = None
    max_rollout_tokens: int | None = None
    adapter_version: str = ""
    val_ratio: float = 0.1

    def __post_init__(self) -> None:
        if self.rollout_mode not in ROLLOUT_MODES:
            raise ValueError(
                f"rollout_mode must be one of {ROLLOUT_MODES}, got {self.rollout_mode!r}"
            )
        if self.num_rollouts < 1:
            raise ValueError(f"num_rollouts must be >= 1, got {self.num_rollouts}")
        if self.max_new_tokens < 1:
            raise ValueError(f"max_new_tokens must be >= 1, got {self.max_new_tokens}")
        if self.temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {self.temperature}")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}")
        if self.max_prompt_tokens is not None and self.max_prompt_tokens < 1:
            raise ValueError(
                f"max_prompt_tokens must be >= 1 when set, got {self.max_prompt_tokens}"
            )
        if self.max_rollout_tokens is not None and self.max_rollout_tokens < 1:
            raise ValueError(
                f"max_rollout_tokens must be >= 1 when set, got {self.max_rollout_tokens}"
            )
        if not 0.0 <= self.val_ratio < 1.0:
            raise ValueError(f"val_ratio must be in [0, 1), got {self.val_ratio}")


class RolloutRows(list):
    """List of rollout rows carrying aggregate discard statistics.

    ``stats`` keys: ``prompts``, ``rollouts_per_prompt``, ``attempts``,
    ``kept``, ``discarded``, ``discard_rate``, ``by_reason`` (``empty`` /
    ``immediate_eos`` / ``too_long``), ``skipped_prompts``,
    ``duplicate_prompts``, ``truncated_prompts``.
    """

    def __init__(self, rows=(), *, stats: dict[str, Any] | None = None) -> None:
        super().__init__(rows)
        self.stats = stats or {}


def generate_rollouts(
    model: Any,
    tokenizer: Any,
    prompts: list[dict[str, Any]],
    config: RolloutConfig | None = None,
) -> RolloutRows:
    """Sample ``config.num_rollouts`` trajectories per prompt.

    Uses the deployed ``base_shadow`` path by default (``shadow_only`` is the
    ablation), samples with ``do_sample=True`` at temperature 1.0 / top-p
    1.0, and seeds ``torch.manual_seed(config.seed + prompt_index)`` once per
    prompt so the G trajectories differ but the whole run is bit-reproducible.

    Generation runs under ``torch.no_grad()``; the model's prior
    train/eval flag and inference mode are restored afterwards. Prompts
    longer than ``max_prompt_tokens`` keep their most recent tokens (the
    left is cut). Degenerate trajectories (empty, immediate EOS, longer than
    ``max_rollout_tokens``) are dropped and counted in ``rows.stats``. Each
    kept row masks the prompt labels with ``-100`` and keeps the rollout
    tokens (including a terminating EOS) as labels.
    """
    config = config or RolloutConfig()
    if not isinstance(config, RolloutConfig):
        raise TypeError(f"config must be a RolloutConfig, got {type(config).__name__}")

    skipped_prompts = sum(1 for row in prompts if not _prompt_text(row))
    unique = _dedupe_prompts(prompts)
    duplicate_prompts = len(prompts) - skipped_prompts - len(unique)
    train_prompts, val_prompts = split_prompts(unique, config.val_ratio, config.seed)
    split_by_prompt = {_prompt_text(row): "train" for row in train_prompts}
    split_by_prompt.update({_prompt_text(row): "val" for row in val_prompts})

    max_rollout_tokens = (
        config.max_rollout_tokens
        if config.max_rollout_tokens is not None
        else config.max_new_tokens
    )
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)

    rows = RolloutRows()
    by_reason = {reason: 0 for reason in DISCARD_REASONS}
    attempts = 0
    truncated_prompts = 0

    was_training = bool(getattr(model, "training", False))
    previous_mode = getattr(model, "inference_mode", None)
    set_mode = getattr(model, "set_inference_mode", None)
    if hasattr(model, "eval"):
        model.eval()
    if callable(set_mode):
        set_mode(config.rollout_mode)

    try:
        device = getattr(model, "device", None)
        if device is None:
            try:
                device = next(model.parameters()).device
            except (AttributeError, StopIteration):
                device = torch.device("cpu")

        for prompt_index, prompt_row in enumerate(unique):
            prompt_text = _prompt_text(prompt_row)
            prompt_ids = _encode_prompt(tokenizer, prompt_text)
            if not prompt_ids:
                skipped_prompts += 1
                continue
            if config.max_prompt_tokens is not None and len(prompt_ids) > config.max_prompt_tokens:
                prompt_ids = prompt_ids[-config.max_prompt_tokens :]
                truncated_prompts += 1

            adapter_version = config.adapter_version or str(
                prompt_row.get("adapter_version") or ""
            )
            split = split_by_prompt[prompt_text]
            task = str(prompt_row.get("task") or "general")
            source = str(prompt_row.get("source") or "unknown")
            effective_seed = config.seed + prompt_index

            input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
            attention_mask = torch.ones_like(input_ids)
            generate_kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "max_new_tokens": config.max_new_tokens,
                "do_sample": True,
                "temperature": config.temperature,
                "top_p": config.top_p,
                "pad_token_id": pad_token_id,
                # Explicit None overrides an inherited generation_config EOS.
                "eos_token_id": eos_token_id,
                "use_cache": False,
            }

            torch.manual_seed(effective_seed)
            for _ in range(config.num_rollouts):
                attempts += 1
                with torch.no_grad():
                    output = model.generate(**generate_kwargs)
                if hasattr(output, "sequences"):
                    output = output.sequences
                generated = [int(token) for token in output[0, len(prompt_ids) :].tolist()]

                reason = None
                finish_reason = "length"
                if not generated:
                    reason = "empty"
                elif eos_token_id is not None and eos_token_id in generated:
                    eos_index = generated.index(eos_token_id)
                    if eos_index == 0:
                        reason = "immediate_eos"
                    else:
                        generated = generated[: eos_index + 1]
                        finish_reason = "eos"
                if reason is None and len(generated) > max_rollout_tokens:
                    reason = "too_long"

                if reason is not None:
                    by_reason[reason] += 1
                    continue

                rows.append(
                    {
                        "prompt": prompt_text,
                        "task": task,
                        "source": source,
                        "split": split,
                        "prompt_ids": list(prompt_ids),
                        "rollout_ids": list(generated),
                        "labels": [-100] * len(prompt_ids) + list(generated),
                        "finish_reason": finish_reason,
                        "rollout_mode": config.rollout_mode,
                        "adapter_version": adapter_version,
                        "seed": effective_seed,
                    }
                )
    finally:
        if callable(set_mode) and previous_mode is not None:
            set_mode(previous_mode)
        if hasattr(model, "train"):
            model.train(was_training)

    discarded = attempts - len(rows)
    rows.stats = {
        "prompts": len(unique),
        "rollouts_per_prompt": config.num_rollouts,
        "attempts": attempts,
        "kept": len(rows),
        "discarded": discarded,
        "discard_rate": discarded / attempts if attempts else 0.0,
        "by_reason": by_reason,
        "skipped_prompts": skipped_prompts,
        "duplicate_prompts": duplicate_prompts,
        "truncated_prompts": truncated_prompts,
    }
    return rows


def write_rollouts(rows: list[dict[str, Any]], path: str | Path) -> str:
    """Write rollout rows as stable, atomic JSONL (sorted keys, one row/line).

    The file is written to a ``.tmp`` sibling and moved into place with
    ``os.replace`` so a crashed run never leaves a half-written rollout set.
    Returns the output path as a string.
    """
    output = Path(path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(output.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temp, output)
    return str(output)


def load_rollouts(path: str | Path) -> list[dict[str, Any]]:
    """Read a rollout JSONL file written by :func:`write_rollouts`."""
    rows: list[dict[str, Any]] = []
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows
