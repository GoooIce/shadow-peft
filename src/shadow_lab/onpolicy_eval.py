"""Evaluation, reporting and training-batch seams for the on-policy GKD flywheel.

Issue #21 closes the loop started by the rollout pipeline (issue #19) and the
``OnPolicyGKDTrainer`` (issue #20). It provides:

- :func:`evaluate_ppl` — paired joint (``base_shadow``) and shadow-only
  perplexity over the same held-out windows, restoring the caller's inference
  mode and train/eval flag afterwards.
- :func:`grade_task` / :func:`task_accuracy` — structured grading for the two
  production tasks (``quality_verdict`` verdict keywords, ``process_calc``
  numeric tolerance); open-ended tasks return ``None`` instead of a fake score.
- :func:`bootstrap_ci` — 2.5/97.5 percentile bootstrap interval so small
  held-out sets are reported with their uncertainty.
- :func:`build_report` — the auditable report (both perplexities, task
  accuracy, rollout/loss provenance, token budget, wall clock, discard rate,
  data fingerprint, SFT baseline and the ``no_go`` guard).
- :class:`OnPolicyTrainDataset` / :func:`collate_training_rows` — turn rollout
  rows and supervised SFT rows into the ``input_ids``/``attention_mask``/
  ``labels``/``source`` batch contract the trainer consumes. Rollout metadata
  is **not** the branch label: rollout rows are always ``source="onpolicy"``,
  supervised rows ``source="supervised"``; the original prompt source is kept
  aside as ``prompt_source``.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import torch

from shadow_peft.task_models import _shifted_ce_loss

ROLLOUT_MODES = ("base_shadow", "shadow_only")
DIVERGENCES = ("forward_kl", "reverse_kl", "jsd")
TEACHER_SUPPORTS = ("full", "topk")
BRANCH_SOURCES = ("supervised", "onpolicy")
IGNORE_INDEX = -100

REPORT_FIELDS = (
    "joint_ppl",
    "shadow_ppl",
    "task_accuracy",
    "rollout_mode",
    "lambda",
    "divergence",
    "teacher_support",
    "tokens_seen",
    "wall_clock",
    "discard_rate",
    "data_fingerprint",
    "baseline_sft",
    "no_go",
)

_NUMBER_RE = re.compile(
    r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?%?"
)

# Longest terms first so "不合格" wins over "合格" at the same position, and
# the leftmost match is the verdict the answer actually opens with
# ("判定：返修。……不满足放行条件" grades as rework, not pass).
_VERDICT_SYNONYMS: dict[str, tuple[str, ...]] = {
    "pass": ("放行", "合格", "通过", "pass", "release", "accept", "approve"),
    "rework": ("返修", "重工", "rework", "repair"),
    "reject": ("不合格", "拦截", "报废", "reject", "scrap", "fail"),
}
_VERDICT_TERMS = sorted(
    ((term, label) for label, terms in _VERDICT_SYNONYMS.items() for term in terms),
    key=lambda pair: len(pair[0]),
    reverse=True,
)
_VERDICT_LABEL = {term.casefold(): label for term, label in _VERDICT_TERMS}
_VERDICT_RE = re.compile(
    "|".join(re.escape(term) for term, _label in _VERDICT_TERMS), re.IGNORECASE
)

# Relative tolerance for process_calc: 1% covers rounding in a worked result
# while still rejecting a wrong order of magnitude (3000 vs 30000 DPPM).
NUMBER_REL_TOL = 1e-2

# ---------------------------------------------------------------------------
# Paired perplexity
# ---------------------------------------------------------------------------


def _normalize_ids(encoded: Any) -> list[int]:
    """Normalize a tokenizer return value to a flat list of token ids."""
    if isinstance(encoded, Mapping):
        encoded = encoded["input_ids"]
    if hasattr(encoded, "tolist"):  # torch.Tensor / numpy array
        encoded = encoded.tolist()
    if encoded and isinstance(encoded[0], (list, tuple)):
        encoded = list(encoded[0])
    return [int(token) for token in encoded]


def _encode_text(tokenizer: Any, text: str) -> list[int]:
    return _normalize_ids(tokenizer(text, add_special_tokens=True))


def _ppl_windows(
    texts: Sequence[str],
    tokenizer: Any,
    seq_len: int,
    max_windows: int | None,
) -> list[list[int]]:
    """Tokenize held-out texts into ``seq_len + 1`` windows (context + targets).

    Each text is windowed independently so documents never share a scored
    span. Windows stride by ``seq_len`` and carry their last token as context
    for the next window; only targets within the window are scored. The total
    is capped at ``max_windows`` (``None`` scores everything).
    """
    if seq_len < 2:
        raise ValueError(f"seq_len must be >= 2, got {seq_len}")
    if max_windows is not None and max_windows < 1:
        raise ValueError(f"max_windows must be >= 1 or None, got {max_windows}")

    windows: list[list[int]] = []
    for text in texts:
        ids = _encode_text(tokenizer, text)
        if len(ids) < 2:
            continue
        for start in range(0, len(ids) - 1, seq_len):
            chunk = ids[start : start + seq_len + 1]
            if len(chunk) < 2:
                break
            windows.append(chunk)
            if max_windows is not None and len(windows) >= max_windows:
                return windows
    return windows


def evaluate_ppl(
    model: Any,
    tokenizer: Any,
    texts: Sequence[str],
    *,
    seq_len: int = 2048,
    max_windows: int | None = 8,
) -> dict[str, Any]:
    """Report joint and shadow-only perplexity on the same held-out windows.

    The joint number is the deployed ``base_shadow`` path (``logits``); the
    shadow number comes from the injected shadow head alone (``shadow_only``).
    Both run under ``torch.no_grad()``. The model's previous inference mode and
    train/eval flag are restored even if a forward raises.

    Returns ``{"joint_ppl", "shadow_ppl", "tokens"}`` where ``tokens`` is the
    number of scored next-token targets. At least one window of two tokens is
    required; an all-empty/one-token corpus raises ``ValueError``.
    """
    if isinstance(texts, str):
        raise TypeError("texts must be a sequence of strings, not a bare string")
    set_mode = getattr(model, "set_inference_mode", None)
    if not callable(set_mode):
        raise TypeError(
            "evaluate_ppl needs a ShadowForCausalLM-like model exposing "
            "set_inference_mode('base_shadow'|'shadow_only')."
        )

    windows = _ppl_windows([str(text) for text in texts], tokenizer, seq_len, max_windows)
    if not windows:
        raise ValueError(
            "evaluate_ppl: held-out texts too short to score — need at least "
            "one text with 2+ tokens"
        )

    device = getattr(model, "device", None)
    if device is None:
        device = next(model.parameters()).device
    was_training = bool(getattr(model, "training", False))
    previous_mode = getattr(model, "inference_mode", "base_shadow")

    totals = {"joint": [0.0, 0], "shadow": [0.0, 0]}
    if hasattr(model, "eval"):
        model.eval()
    try:
        with torch.no_grad():
            for mode, key in (("base_shadow", "joint"), ("shadow_only", "shadow")):
                set_mode(mode)
                for ids in windows:
                    x = torch.tensor([ids], dtype=torch.long, device=device)
                    logits = model(input_ids=x).logits
                    loss = _shifted_ce_loss(logits, x)
                    targets = x.size(1) - 1
                    totals[key][0] += float(loss) * targets
                    totals[key][1] += targets
    finally:
        if previous_mode is not None:
            set_mode(previous_mode)
        if hasattr(model, "train"):
            model.train(was_training)

    tokens = totals["joint"][1]
    if not tokens or totals["joint"][1] != totals["shadow"][1]:
        raise RuntimeError("evaluate_ppl: paired window bookkeeping diverged")
    return {
        "joint_ppl": math.exp(totals["joint"][0] / tokens),
        "shadow_ppl": math.exp(totals["shadow"][0] / tokens),
        "tokens": tokens,
    }


# ---------------------------------------------------------------------------
# Structured task grading
# ---------------------------------------------------------------------------


def _canonical_verdict(value: Any) -> str | None:
    text = str(value or "").casefold()
    for term, label in _VERDICT_TERMS:
        if term.casefold() in text:
            return label
    return None


def _numbers_in(text: str) -> list[float]:
    """Parse numbers, including thousands separators and percentages.

    ``25%`` contributes both ``25`` and ``0.25`` so an expected ratio (0.25)
    and an expected percentage (25) both grade the same sentence.
    """
    values: list[float] = []
    for match in _NUMBER_RE.finditer(text):
        raw = match.group(0)
        percent = raw.endswith("%")
        raw = raw.rstrip("%").replace(",", "")
        try:
            number = float(raw)
        except ValueError:
            continue
        values.append(number)
        if percent:
            values.append(number / 100.0)
    return values


def _grade_verdict(text: str, expected: Any) -> bool:
    expected_label = _canonical_verdict(expected)
    if expected_label is None:
        needle = str(expected or "").strip().casefold()
        return bool(needle) and needle in text.casefold()
    match = _VERDICT_RE.search(text)
    if match is None:
        return False
    return _VERDICT_LABEL[match.group(0).casefold()] == expected_label


def _grade_number(text: str, expected: Any) -> bool:
    expected_values = _numbers_in(str(expected))
    if not expected_values:
        return False
    target = expected_values[0]
    tolerance = max(abs(target) * NUMBER_REL_TOL, 1e-9)
    return any(
        math.isclose(candidate, target, rel_tol=NUMBER_REL_TOL, abs_tol=tolerance)
        for candidate in _numbers_in(text)
    )


def grade_task(text: str | None, expected: Any, task: str | None) -> bool | None:
    """Grade one structured answer; ``None`` means the task is not gradable.

    - ``quality_verdict``: the first verdict term in ``text`` must match the
      expected verdict (``放行``/``返修``/``不合格`` and English synonyms).
    - ``process_calc``: any number in ``text`` within 1% relative tolerance of
      the first number in ``expected`` (thousands separators and ``%`` are
      normalized).
    - Every other task is open-ended and returns ``None``; no fabricated score.
    """
    if text is None or expected is None:
        return None
    normalized = str(task or "").strip().lower()
    body = str(text)
    if normalized == "quality_verdict":
        return _grade_verdict(body, expected)
    if normalized == "process_calc":
        return _grade_number(body, expected)
    return None


def _record_text(record: Mapping[str, Any]) -> Any:
    for key in ("response", "text", "output", "prediction", "answer"):
        if record.get(key) is not None:
            return record[key]
    return None


def task_accuracy(records: Iterable[Mapping[str, Any]]) -> float | None:
    """Mean accuracy over gradable records; ``None`` when none can be graded.

    Records carry ``task``, ``expected`` and the model output under one of
    ``response``/``text``/``output``/``prediction``/``answer``. Open-ended
    tasks are skipped, never counted as failures.
    """
    graded = []
    for record in records:
        if not isinstance(record, Mapping):
            continue
        verdict = grade_task(
            _record_text(record), record.get("expected"), record.get("task")
        )
        if verdict is not None:
            graded.append(bool(verdict))
    if not graded:
        return None
    return sum(graded) / len(graded)


# ---------------------------------------------------------------------------
# Bootstrap interval
# ---------------------------------------------------------------------------


def _percentile(sorted_values: Sequence[float], q: float) -> float:
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = (len(sorted_values) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(sorted_values[lower]) * (1.0 - weight) + float(
        sorted_values[upper]
    ) * weight


def bootstrap_ci(
    values: Sequence[float], n_boot: int = 1000, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap interval (2.5%, 97.5%) for the mean of ``values``.

    With few observations the resampled means spread out, so the interval is
    wider than for a large sample. Deterministic for a fixed ``seed``.
    """
    data = [float(value) for value in values]
    if not data:
        raise ValueError("bootstrap_ci needs at least one value")
    if n_boot < 1:
        raise ValueError(f"n_boot must be >= 1, got {n_boot}")
    rng = random.Random(seed)
    count = len(data)
    means = sorted(
        sum(rng.choices(data, k=count)) / count for _ in range(n_boot)
    )
    return _percentile(means, 0.025), _percentile(means, 0.975)


# ---------------------------------------------------------------------------
# Training batch contract
# ---------------------------------------------------------------------------


def _fit_sequence(
    prompt_ids: list[int], response_ids: list[int], max_seq_len: int
) -> tuple[list[int], list[int]]:
    """Right-truncate the response and left-truncate the prompt to fit.

    Keeping the tail of the prompt preserves the most recent instructions;
    keeping the head of the response preserves the start of the trajectory.
    """
    response = response_ids[:max_seq_len]
    budget = max_seq_len - len(response)
    prompt = prompt_ids[-budget:] if budget else []
    return prompt, response


def _fit_prebuilt(
    input_ids: list[int], labels: list[int], attention_mask: list[int], max_seq_len: int
) -> tuple[list[int], list[int], list[int]]:
    if len(input_ids) <= max_seq_len:
        return input_ids, labels, attention_mask
    keep = max_seq_len
    return input_ids[-keep:], labels[-keep:], attention_mask[-keep:]


def _common_prefix(left: Sequence[int], right: Sequence[int]) -> int:
    limit = min(len(left), len(right))
    index = 0
    while index < limit and left[index] == right[index]:
        index += 1
    return index


def _chat_template_ids(
    tokenizer: Any, messages: list[dict[str, str]], *, add_generation_prompt: bool
) -> list[int]:
    try:
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
        )
    except TypeError:
        encoded = tokenizer.apply_chat_template(messages, tokenize=True)
    return _normalize_ids(encoded)


def _tokenize_messages(
    tokenizer: Any, messages: list[dict[str, str]]
) -> tuple[list[int], list[int]]:
    """Tokenize a chat conversation and mask every non-assistant token."""
    normalized = [
        {
            "role": str(message.get("role") or "user"),
            "content": str(message.get("content") or ""),
        }
        for message in messages
    ]
    if not normalized:
        raise ValueError("supervised row has an empty `messages` list")

    if getattr(tokenizer, "chat_template", None):
        input_ids = _chat_template_ids(
            tokenizer, normalized, add_generation_prompt=False
        )
        prefix_messages = normalized[:-1]
        prefix_ids = (
            _chat_template_ids(
                tokenizer, prefix_messages, add_generation_prompt=True
            )
            if prefix_messages
            else []
        )
    else:
        full_text = "\n".join(message["content"] for message in normalized)
        input_ids = _encode_text(tokenizer, full_text)
        prefix_text = "\n".join(
            message["content"] for message in normalized[:-1]
        )
        prefix_ids = _encode_text(tokenizer, prefix_text) if prefix_text else []

    masked = _common_prefix(prefix_ids, input_ids)
    if masked == 0 and normalized[-1]["role"] == "assistant":
        # Some templates render the assistant header separately. Fall back to
        # locating the assistant content verbatim so labels still start there.
        answer_ids = _normalize_ids(
            tokenizer(normalized[-1]["content"], add_special_tokens=False)
        )
        if answer_ids and input_ids[-len(answer_ids) :] == answer_ids:
            masked = len(input_ids) - len(answer_ids)
    labels = [IGNORE_INDEX] * masked + input_ids[masked:]
    if all(label == IGNORE_INDEX for label in labels):
        raise ValueError(
            "supervised row has no assistant tokens to train on; check the "
            "chat template or the message roles"
        )
    return input_ids, labels


class OnPolicyTrainDataset(torch.utils.data.Dataset):
    """Rollout rows plus optional supervised SFT rows, trainer-ready.

    Every item is exactly ``{"input_ids", "attention_mask", "labels",
    "source"}``. Rollout rows become ``source="onpolicy"`` with the prompt
    masked out of the labels; rows read from ``FeedbackStore.export_sft``
    become ``source="supervised"`` with everything before the assistant answer
    masked. The prompt's original provenance (``cloud``/``feedback``/...) is
    preserved in ``prompt_sources`` for reporting and never used as the loss
    branch.
    """

    def __init__(
        self,
        tokenizer: Any,
        *,
        rollout_rows: Iterable[Mapping[str, Any]] = (),
        supervised_rows: Iterable[Mapping[str, Any]] = (),
        max_seq_len: int = 512,
    ) -> None:
        if max_seq_len < 2:
            raise ValueError(f"max_seq_len must be >= 2, got {max_seq_len}")
        self.tokenizer = tokenizer
        self.max_seq_len = int(max_seq_len)
        self.records: list[dict[str, Any]] = []
        self.prompt_sources: list[str] = []
        for index, row in enumerate(rollout_rows):
            self.records.append(self._from_rollout(row, index))
        for index, row in enumerate(supervised_rows):
            self.records.append(self._from_supervised(row, index))
        if not self.records:
            raise ValueError(
                "OnPolicyTrainDataset needs at least one rollout or supervised row"
            )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.records[index]

    def to_list(self) -> list[dict[str, Any]]:
        """Plain records for ``datasets.Dataset.from_list`` / the CLI."""
        return [dict(record) for record in self.records]

    def _from_rollout(self, row: Mapping[str, Any], index: int) -> dict[str, Any]:
        prompt_ids = row.get("prompt_ids")
        rollout_ids = row.get("rollout_ids")
        prompt_source = str(row.get("source") or "")
        if prompt_ids is None or rollout_ids is None:
            input_ids = row.get("input_ids")
            if input_ids is None:
                raise ValueError(
                    f"rollout row {index} needs prompt_ids/rollout_ids or input_ids"
                )
            raw_ids = [int(token) for token in input_ids]
            raw_labels = [
                int(token) for token in row.get("labels", raw_ids)
            ]
            raw_mask = [
                int(value) for value in row.get("attention_mask", [1] * len(raw_ids))
            ]
            input_ids, labels, attention_mask = _fit_prebuilt(
                raw_ids, raw_labels, raw_mask, self.max_seq_len
            )
        else:
            prompt = [int(token) for token in prompt_ids]
            response = [int(token) for token in rollout_ids]
            if not response:
                raise ValueError(f"rollout row {index} has no rollout tokens")
            prompt, response = _fit_sequence(prompt, response, self.max_seq_len)
            input_ids = prompt + response
            labels = [IGNORE_INDEX] * len(prompt) + response
            attention_mask = [1] * len(input_ids)
        self.prompt_sources.append(prompt_source)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "source": "onpolicy",
        }

    def _from_supervised(self, row: Mapping[str, Any], index: int) -> dict[str, Any]:
        meta = row.get("meta") if isinstance(row.get("meta"), Mapping) else {}
        prompt_source = str(
            row.get("prompt_source") or row.get("source") or meta.get("source") or ""
        )
        if row.get("input_ids") is not None:
            raw_ids = [int(token) for token in row["input_ids"]]
            raw_labels = [
                int(token) for token in row.get("labels", raw_ids)
            ]
            raw_mask = [
                int(value) for value in row.get("attention_mask", [1] * len(raw_ids))
            ]
            input_ids, labels, attention_mask = _fit_prebuilt(
                raw_ids, raw_labels, raw_mask, self.max_seq_len
            )
        else:
            messages = row.get("messages")
            if not isinstance(messages, list) or not messages:
                raise ValueError(
                    f"supervised row {index} needs `messages` or pre-tokenized input_ids"
                )
            input_ids, labels = _tokenize_messages(self.tokenizer, messages)
            attention_mask = [1] * len(input_ids)
            input_ids, labels, attention_mask = _fit_prebuilt(
                input_ids, labels, attention_mask, self.max_seq_len
            )
        self.prompt_sources.append(prompt_source)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "source": "supervised",
        }


def collate_training_rows(
    rows: Sequence[Mapping[str, Any]], pad_id: int
) -> dict[str, Any]:
    """Right-pad unified rows into the trainer's batch contract.

    ``source`` must already be the loss-branch label (``"supervised"`` or
    ``"onpolicy"``); anything else — most importantly a rollout row's prompt
    provenance (``"cloud"``/``"feedback"``) — raises instead of silently
    selecting the wrong loss branch. Padding positions get ``pad_id``, an
    attention mask of 0 and a label of ``-100``. Unknown keys (e.g.
    ``prompt_source``) are ignored, so audit metadata cannot reach the model.
    """
    if not rows:
        raise ValueError("collate_training_rows needs at least one row")
    if not isinstance(pad_id, int) or pad_id < 0:
        raise ValueError(f"pad_id must be a non-negative int, got {pad_id!r}")

    sources: list[str] = []
    for index, row in enumerate(rows):
        source = row.get("source")
        if source not in BRANCH_SOURCES:
            raise ValueError(
                f"row {index} has source {source!r}; expected one of {BRANCH_SOURCES}. "
                "Rollout prompt provenance belongs in `prompt_source`, not `source`."
            )
        sources.append(str(source))

    max_len = max(len(row["input_ids"]) for row in rows)
    input_ids = torch.full((len(rows), max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(rows), max_len), dtype=torch.long)
    labels = torch.full((len(rows), max_len), IGNORE_INDEX, dtype=torch.long)
    for index, row in enumerate(rows):
        ids = [int(token) for token in row["input_ids"]]
        row_labels = [int(token) for token in row["labels"]]
        mask = row.get("attention_mask")
        row_mask = (
            [int(value) for value in mask] if mask is not None else [1] * len(ids)
        )
        if not (len(ids) == len(row_labels) == len(row_mask)):
            raise ValueError(
                f"row {index} has mismatched input_ids/labels/attention_mask lengths"
            )
        length = len(ids)
        input_ids[index, :length] = torch.tensor(ids, dtype=torch.long)
        attention_mask[index, :length] = torch.tensor(row_mask, dtype=torch.long)
        labels[index, :length] = torch.tensor(row_labels, dtype=torch.long)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "source": sources,
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def data_fingerprint(
    prompts: Iterable[str] | None = None,
    *,
    prompt_hash: str | None = None,
    teacher_version: str,
    lambda_onpolicy: float,
    divergence: str,
) -> str:
    """Stable hash of the run's data identity.

    Covers the prompt **set** (order- and duplicate-insensitive), the teacher
    version, λ and divergence. Changing any of them changes the fingerprint;
    reordering the prompt file does not.
    """
    if prompt_hash is None:
        if prompts is None:
            raise ValueError("data_fingerprint needs prompts or prompt_hash")
        digest = hashlib.sha256()
        for text in sorted({str(prompt) for prompt in prompts}):
            digest.update(text.encode("utf-8"))
            digest.update(b"\x00")
        prompt_hash = digest.hexdigest()
    payload = json.dumps(
        {
            "prompt_hash": str(prompt_hash),
            "teacher_version": str(teacher_version),
            "lambda": round(float(lambda_onpolicy), 6),
            "divergence": str(divergence),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_report(
    *,
    joint_ppl: float,
    shadow_ppl: float,
    task_accuracy: float | None,
    rollout_mode: str,
    lambda_onpolicy: float,
    divergence: str,
    teacher_support: str,
    tokens_seen: int,
    wall_clock: float,
    discard_rate: float,
    prompts: Iterable[str] | None = None,
    prompt_hash: str | None = None,
    teacher_version: str = "",
    baseline_sft: Mapping[str, Any] | None = None,
    no_go: bool | None = None,
) -> dict[str, Any]:
    """Build the auditable evaluation report; every core field is required.

    ``no_go`` is the deployment guard from the spec: when a same-budget
    ``baseline_sft`` (with ``joint_ppl``/``shadow_ppl``) is supplied, it is
    true exactly when shadow-only perplexity improved but joint perplexity did
    not. Without a baseline it defaults to ``False`` (no claim either way) and
    can be overridden explicitly.
    """
    joint = float(joint_ppl)
    shadow = float(shadow_ppl)
    if not (math.isfinite(joint) and joint > 0.0):
        raise ValueError(f"joint_ppl must be finite and positive, got {joint_ppl!r}")
    if not (math.isfinite(shadow) and shadow > 0.0):
        raise ValueError(f"shadow_ppl must be finite and positive, got {shadow_ppl!r}")
    if rollout_mode not in ROLLOUT_MODES:
        raise ValueError(
            f"rollout_mode must be one of {ROLLOUT_MODES}, got {rollout_mode!r}"
        )
    if divergence not in DIVERGENCES:
        raise ValueError(
            f"divergence must be one of {DIVERGENCES}, got {divergence!r}"
        )
    if teacher_support not in TEACHER_SUPPORTS:
        raise ValueError(
            f"teacher_support must be one of {TEACHER_SUPPORTS}, got {teacher_support!r}"
        )
    if not 0.0 <= float(lambda_onpolicy) <= 1.0:
        raise ValueError(
            f"lambda_onpolicy must be in [0, 1], got {lambda_onpolicy!r}"
        )
    if int(tokens_seen) < 0:
        raise ValueError(f"tokens_seen must be >= 0, got {tokens_seen!r}")
    if not math.isfinite(float(wall_clock)) or float(wall_clock) < 0.0:
        raise ValueError(f"wall_clock must be finite and >= 0, got {wall_clock!r}")
    if not 0.0 <= float(discard_rate) <= 1.0:
        raise ValueError(f"discard_rate must be in [0, 1], got {discard_rate!r}")
    if task_accuracy is not None and not 0.0 <= float(task_accuracy) <= 1.0:
        raise ValueError(
            f"task_accuracy must be None or in [0, 1], got {task_accuracy!r}"
        )

    baseline = dict(baseline_sft) if baseline_sft is not None else None
    if no_go is None:
        no_go = False
        if baseline is not None:
            baseline_joint = float(baseline["joint_ppl"])
            baseline_shadow = float(baseline["shadow_ppl"])
            no_go = shadow < baseline_shadow and not (joint < baseline_joint)

    fingerprint = data_fingerprint(
        prompts,
        prompt_hash=prompt_hash,
        teacher_version=teacher_version,
        lambda_onpolicy=lambda_onpolicy,
        divergence=divergence,
    )
    return {
        "joint_ppl": joint,
        "shadow_ppl": shadow,
        "task_accuracy": None if task_accuracy is None else float(task_accuracy),
        "rollout_mode": rollout_mode,
        "lambda": float(lambda_onpolicy),
        "divergence": divergence,
        "teacher_support": teacher_support,
        "tokens_seen": int(tokens_seen),
        "wall_clock": float(wall_clock),
        "discard_rate": float(discard_rate),
        "data_fingerprint": fingerprint,
        "baseline_sft": baseline,
        "no_go": bool(no_go),
    }
