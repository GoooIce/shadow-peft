"""RESTRAIN-lite evaluation, reporting and weighted SFT loop (issue #34).

Spec: ``docs/specs/restrain-lite-structured-flywheel.md`` §Implementation
Decisions 5 and Testing Decisions「评估缝」. The module is split into three
layers so each can be exercised without a model:

- **grading** — :func:`grade_structured` / :func:`accuracy` reuse
  :func:`shadow_lab.restrain.extract_answer` (and accept the canonical
  ``pass``/``fail``/``rework`` labels): verdict answers match after synonym
  normalization, numeric answers match within a relative tolerance (the unit
  must match when both sides carry one);
- **answer statistics** — :func:`answer_entropy` (bits), :func:`top1_share`,
  :func:`low_consistency_rate`, :func:`skip_rates` and :func:`bootstrap_ci`;
- **report and training** — :func:`build_report` emits the auditable report
  (accuracy/entropy/consistency/skip/failure/escalation/cloud/config and the
  ``claims_gain`` acceptance rule), and :func:`train_weighted_sft` runs the
  same next-token CE as :class:`shadow_lab.trainers.WeightedSFTTrainer`
  through a lightweight manual loop that works on tiny random models.

No model or torch import happens at module import time: the training helper
imports torch only when called, and the CLI keeps heavy imports lazy.
"""

from __future__ import annotations

import math
import random
import re
import time
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

from .restrain import cluster_answers, extract_answer

REPORT_FIELDS = (
    "accuracy",
    "accuracy_ci",
    "answer_entropy",
    "top1_share",
    "low_consistency_rate",
    "skip_rate",
    "rollout_failure_rate",
    "escalation_rate",
    "cloud_calls",
    "baseline_best_answer",
    "config",
    "claims_gain",
)

DEFAULT_NUMBER_RTOL = 1e-2
DEFAULT_DELTA = 0.01
DEFAULT_ENTROPY_KEEP = 0.8


# ---------------------------------------------------------------------------
# Structured grading
# ---------------------------------------------------------------------------


_CANONICAL_VERDICTS = frozenset({"pass", "fail", "rework"})
_VERDICT_FRAME_RE = re.compile(r"(?:判定|结论)\s*[:：]?\s*")


def _record_answer(record: Mapping[str, Any]) -> Any:
    for key in ("answer", "response", "text", "output", "prediction"):
        if key in record and record[key] is not None:
            return record[key]
    return None


def _verdict_label(value: Any) -> str | None:
    """Extract a verdict label, also accepting the canonical English labels.

    ``shadow_lab.restrain.extract_answer`` parses the Chinese labels produced
    by the production prompts; exported self-train rows and their training
    targets use the canonical ``pass``/``fail``/``rework`` spellings. Grading
    and evaluation must accept both sides of that round trip.
    """

    if value is None:
        return None
    text = str(value)
    parsed = extract_answer(text, "quality_verdict")
    if parsed is not None:
        return parsed
    candidate = _VERDICT_FRAME_RE.sub("", text)
    candidate = candidate.strip().strip(" .,;:!?。；：！？").lower()
    return candidate if candidate in _CANONICAL_VERDICTS else None


def answer_counts(
    answers: Iterable[Any],
    task: str | None,
    *,
    number_rtol: float = 1e-6,
) -> dict[str, int]:
    """Cluster answers into ``{label: votes}`` for entropy and top-1 share.

    Verdict answers use :func:`_verdict_label` (Chinese synonyms and canonical
    English labels); numeric answers reuse :func:`shadow_lab.restrain.cluster_answers`
    so the unit/tolerance rules stay in one place.
    """

    normalized_task = str(task or "").strip().lower()
    if normalized_task == "quality_verdict":
        labels = [
            label
            for answer in answers
            if (label := _verdict_label(answer)) is not None
        ]
        return dict(Counter(labels))
    parsed = [
        str(answer) for answer in answers if isinstance(answer, str)
    ]
    return dict(cluster_answers(parsed, normalized_task, number_rtol=number_rtol).counts)


def _number_matches(
    answer: tuple[float, str],
    expected: tuple[float, str],
    *,
    number_rtol: float,
) -> bool:
    answer_value, answer_unit = answer
    expected_value, expected_unit = expected
    if answer_unit and expected_unit and answer_unit != expected_unit:
        return False
    return math.isclose(
        answer_value,
        expected_value,
        rel_tol=number_rtol,
        abs_tol=max(abs(expected_value) * number_rtol, 1e-9),
    )


def grade_structured(
    answer: Any,
    expected: Any,
    task: str | None,
    *,
    number_rtol: float = DEFAULT_NUMBER_RTOL,
) -> bool | None:
    """Grade one structured answer; ``None`` means the item is not gradable.

    ``quality_verdict`` normalizes both sides with ``extract_answer`` (and
    accepts the canonical ``pass``/``fail``/``rework`` labels used by the
    self-train rows) and compares the labels. ``process_calc`` compares the
    first number with a relative tolerance (default 1%); when both sides carry
    a unit the units must match. Any other task, a missing ``expected`` or an expected value
    that cannot be parsed returns ``None``. A model answer that cannot be
    parsed is a miss (``False``), not an ungradable item.
    """

    if number_rtol < 0:
        raise ValueError("number_rtol must be non-negative")
    normalized_task = str(task or "").strip().lower()
    if normalized_task not in {"quality_verdict", "process_calc"}:
        return None
    if expected is None:
        return None

    if normalized_task == "quality_verdict":
        expected_label = _verdict_label(expected)
        if expected_label is None:
            return None
        if answer is None:
            return False
        return _verdict_label(answer) == expected_label

    expected_number = extract_answer(str(expected), normalized_task)
    if expected_number is None:
        return None
    if answer is None:
        return False
    try:
        answer_number = extract_answer(str(answer), normalized_task)
    except (TypeError, ValueError):
        return False
    if answer_number is None:
        return False
    return _number_matches(answer_number, expected_number, number_rtol=number_rtol)


def graded_outcomes(
    records: Iterable[Mapping[str, Any]],
    *,
    number_rtol: float = DEFAULT_NUMBER_RTOL,
) -> list[bool]:
    """Graded booleans for the records that can be graded (open-ended skipped)."""

    outcomes: list[bool] = []
    for record in records:
        if not isinstance(record, Mapping):
            continue
        verdict = grade_structured(
            _record_answer(record),
            record.get("expected", record.get("expected_answer")),
            record.get("task"),
            number_rtol=number_rtol,
        )
        if verdict is not None:
            outcomes.append(bool(verdict))
    return outcomes


def accuracy(
    records: Iterable[Mapping[str, Any]],
    *,
    number_rtol: float = DEFAULT_NUMBER_RTOL,
) -> float | None:
    """Mean accuracy over gradable records; ``None`` when none can be graded."""

    outcomes = graded_outcomes(records, number_rtol=number_rtol)
    if not outcomes:
        return None
    return sum(outcomes) / len(outcomes)


# ---------------------------------------------------------------------------
# Answer statistics
# ---------------------------------------------------------------------------


def _count_values(counts: Mapping[str, int] | Sequence[int]) -> list[float]:
    values = (
        [float(value) for value in counts.values()]
        if isinstance(counts, Mapping)
        else [float(value) for value in counts]
    )
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("counts must be finite and non-negative")
    return values


def answer_entropy(counts: Mapping[str, int] | Sequence[int]) -> float | None:
    """Shannon entropy of the answer distribution in bits.

    A single answer has entropy ``0``; a uniform distribution over ``k``
    answers reaches ``log2(k)``. Empty or all-zero counts have no
    distribution and return ``None`` instead of ``NaN``.
    """

    values = _count_values(counts)
    total = sum(values)
    if total <= 0:
        return None
    entropy = 0.0
    for value in values:
        if value <= 0:
            continue
        probability = value / total
        entropy -= probability * math.log2(probability)
    return entropy


def top1_share(counts: Mapping[str, int] | Sequence[int]) -> float:
    """Largest cluster's share of the votes; ``0.0`` when there are none."""

    values = _count_values(counts)
    total = sum(values)
    if total <= 0:
        return 0.0
    return max(values) / total


def low_consistency_rate(
    rows: Iterable[Mapping[str, Any]],
    *,
    kappa: int = 3,
) -> float:
    """Fraction of rows whose largest cluster has fewer than ``kappa`` votes.

    This is the same gate as :func:`shadow_lab.restrain.prompt_weight`; rows
    without counts count as low consistency because no answer dominates.
    """

    if kappa < 1:
        raise ValueError("kappa must be >= 1")
    row_list = list(rows)
    if not row_list:
        return 0.0
    low = 0
    for row in row_list:
        counts = row.get("counts") if isinstance(row, Mapping) else None
        if isinstance(counts, Mapping):
            values = _count_values(counts)
        elif isinstance(counts, Sequence) and not isinstance(counts, (str, bytes)):
            values = _count_values(list(counts))
        else:
            values = []
        if not values or max(values) < kappa:
            low += 1
    return low / len(row_list)


def skip_rates(report: Mapping[str, Any]) -> dict[str, float | None]:
    """Skip and rollout-failure rates from a ``FeedbackStore`` export report.

    The denominator is every unique prompt the exporter considered for
    self-training: exported rows plus the empty/unsupported/unclusterable
    skips. ``rollout_failure_rate`` needs the rollout count ``n`` (or an
    explicit ``rollout_attempts``) in the report; without either it is
    ``None`` rather than a fabricated rate.
    """

    exported = max(0, int(report.get("exported", 0) or 0))
    skipped_empty = max(0, int(report.get("skipped_empty", 0) or 0))
    skipped_unsupported = max(0, int(report.get("skipped_unsupported", 0) or 0))
    skipped_unclusterable = max(0, int(report.get("skipped_unclusterable", 0) or 0))
    considered = exported + skipped_empty + skipped_unsupported + skipped_unclusterable
    if considered:
        skip_rate = (considered - exported) / considered
        empty_rate = skipped_empty / considered
        unsupported_rate = skipped_unsupported / considered
        unclusterable_rate = skipped_unclusterable / considered
    else:
        skip_rate = empty_rate = unsupported_rate = unclusterable_rate = 0.0

    failures = max(0, int(report.get("rollout_failures", 0) or 0))
    attempts = report.get("rollout_attempts")
    if attempts is None:
        n = report.get("n")
        rolled_out = exported + skipped_unclusterable
        attempts = int(n) * rolled_out if n is not None else None
    attempts = int(attempts) if attempts is not None else 0
    failure_rate = failures / attempts if attempts > 0 else None

    return {
        "skip_rate": skip_rate,
        "empty_rate": empty_rate,
        "unsupported_rate": unsupported_rate,
        "unclusterable_rate": unclusterable_rate,
        "rollout_failure_rate": failure_rate,
    }


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
    values: Sequence[Any],
    *,
    n_boot: int = 1000,
    seed: int = 0,
    statistic: Callable[[Sequence[Any]], float] | None = None,
) -> tuple[float, float]:
    """Percentile bootstrap interval (2.5%, 97.5%) of a sample statistic.

    The default statistic is the mean. Small samples resample into visibly
    wider intervals instead of pretending significance; the result is
    deterministic for a fixed ``seed``. Pass per-prompt entropies with the
    default mean to obtain an entropy interval, or pass the counts themselves
    with ``statistic=answer_entropy``.
    """

    data = list(values)
    if not data:
        raise ValueError("bootstrap_ci needs at least one value")
    if statistic is None:
        numeric = [float(value) for value in data]
        if any(not math.isfinite(value) for value in numeric):
            raise ValueError("bootstrap values must be finite")
        data = numeric

        def compute(sample: Sequence[Any]) -> float:
            return sum(float(value) for value in sample) / len(sample)

    else:
        compute = statistic
    if n_boot < 1:
        raise ValueError(f"n_boot must be >= 1, got {n_boot}")

    rng = random.Random(seed)
    samples = sorted(
        float(compute(rng.choices(data, k=len(data)))) for _ in range(n_boot)
    )
    return _percentile(samples, 0.025), _percentile(samples, 0.975)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _unit_rate(value: Any, name: str) -> float | None:
    if value is None:
        return None
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value!r}")
    return parsed


def _baseline_metrics(
    baseline: Mapping[str, Any] | float | None,
) -> tuple[float | None, float | None]:
    """Return ``(best_answer_accuracy, answer_entropy)`` from a baseline."""

    if baseline is None:
        return None, None
    if isinstance(baseline, Mapping):
        best = None
        for key in ("accuracy", "best_answer_accuracy", "baseline_best_answer", "task_accuracy"):
            if baseline.get(key) is not None:
                best = float(baseline[key])
                break
        entropy = None
        for key in ("answer_entropy", "entropy"):
            if baseline.get(key) is not None:
                entropy = float(baseline[key])
                break
        return best, entropy
    return float(baseline), None


def build_report(
    *,
    accuracy: float | None = None,
    outcomes: Sequence[bool] | None = None,
    answer_entropy: float | None = None,
    top1_share: float | None = None,
    low_consistency_rate: float = 0.0,
    export_report: Mapping[str, Any] | None = None,
    skip_rate: float | None = None,
    rollout_failure_rate: float | None = None,
    escalation_rate: float | None = None,
    cloud_calls: int = 0,
    baseline: Mapping[str, Any] | float | None = None,
    config: Mapping[str, Any] | None = None,
    sigma: float = 0.5,
    kappa: int = 3,
    n: int = 8,
    delta: float = DEFAULT_DELTA,
    n_boot: int = 200,
    ci_seed: int = 0,
) -> dict[str, Any]:
    """Build the RESTRAIN-lite evaluation report.

    ``accuracy``/``outcomes`` may be supplied by the caller (``eval`` grades
    records first); when only ``outcomes`` is given the accuracy is their
    mean and a percentile bootstrap interval is added when at least two
    gradable records exist. ``baseline`` is the best-answer SFT report (a
    mapping with ``accuracy``/``answer_entropy``, or just its accuracy).
    ``claims_gain`` follows the first-round acceptance rule: accuracy is at
    least ``delta`` (1pp by default) above the baseline **and** answer
    entropy is at least 80% of the baseline's. Without a baseline the claim
    is always ``False``.
    """

    if accuracy is not None:
        accuracy = _unit_rate(accuracy, "accuracy")
    if outcomes is not None:
        outcome_list = [bool(outcome) for outcome in outcomes]
        if accuracy is None and outcome_list:
            accuracy = sum(outcome_list) / len(outcome_list)
    else:
        outcome_list = []

    answer_entropy = (
        None if answer_entropy is None else float(answer_entropy)
    )
    if answer_entropy is not None and (
        not math.isfinite(answer_entropy) or answer_entropy < 0
    ):
        raise ValueError("answer_entropy must be finite and non-negative")
    top1_share = _unit_rate(top1_share, "top1_share")
    low_consistency_rate = (
        _unit_rate(low_consistency_rate, "low_consistency_rate") or 0.0
    )

    rates = skip_rates(export_report) if export_report is not None else {}
    skip_rate = _unit_rate(
        skip_rate if skip_rate is not None else rates.get("skip_rate", 0.0),
        "skip_rate",
    ) or 0.0
    rollout_failure_rate = _unit_rate(
        rollout_failure_rate
        if rollout_failure_rate is not None
        else rates.get("rollout_failure_rate"),
        "rollout_failure_rate",
    )
    escalation_rate = _unit_rate(escalation_rate, "escalation_rate")
    if int(cloud_calls) < 0:
        raise ValueError(f"cloud_calls must be >= 0, got {cloud_calls!r}")
    if int(n_boot) < 1:
        raise ValueError(f"n_boot must be >= 1, got {n_boot!r}")

    report_config: dict[str, Any] = {
        "sigma": float(sigma),
        "kappa": int(kappa),
        "n": int(n),
        "delta": float(delta),
    }
    if config is not None:
        report_config.update(dict(config))
    threshold = float(report_config["delta"])

    accuracy_ci = None
    if len(outcome_list) >= 2 and accuracy is not None:
        accuracy_ci = list(
            bootstrap_ci(
                [float(outcome) for outcome in outcome_list],
                n_boot=int(n_boot),
                seed=int(ci_seed),
            )
        )

    baseline_accuracy, baseline_entropy = _baseline_metrics(baseline)
    claims_gain = False
    if accuracy is not None and baseline_accuracy is not None:
        accuracy_ok = accuracy >= baseline_accuracy + threshold - 1e-12
        entropy_ok = (
            answer_entropy is not None
            and baseline_entropy is not None
            and answer_entropy >= DEFAULT_ENTROPY_KEEP * baseline_entropy - 1e-12
        )
        claims_gain = bool(accuracy_ok and entropy_ok)

    return {
        "accuracy": accuracy,
        "accuracy_ci": accuracy_ci,
        "answer_entropy": answer_entropy,
        "top1_share": top1_share,
        "low_consistency_rate": low_consistency_rate,
        "skip_rate": skip_rate,
        "rollout_failure_rate": rollout_failure_rate,
        "escalation_rate": escalation_rate,
        "cloud_calls": int(cloud_calls),
        "baseline_best_answer": baseline_accuracy,
        "config": report_config,
        "claims_gain": claims_gain,
    }


# ---------------------------------------------------------------------------
# Training rows and weighted loop
# ---------------------------------------------------------------------------


def self_train_to_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    use_weights: bool = True,
) -> list[dict[str, Any]]:
    """Turn exported self-train rows into ``prompt``/``answer``/``weight`` rows.

    The training target is the largest cluster's canonical label. With
    ``use_weights`` the sample weight is ``prompt_weight * cluster_weight``
    (the RESTRAIN soft vote weight); without it every sample weighs 1. Rows
    gated out by ``prompt_weight == 0`` are dropped because they never enter
    the loss.
    """

    prepared: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        prompt = row.get("prompt")
        answers = row.get("answers")
        if not isinstance(prompt, str) or not prompt.strip():
            continue
        if not isinstance(answers, Sequence) or isinstance(answers, (str, bytes)):
            continue
        if not answers:
            continue
        counts = row.get("counts")
        if isinstance(counts, Sequence) and not isinstance(counts, (str, bytes)):
            count_values = [int(count) for count in counts]
        else:
            count_values = [1] * len(answers)
        if len(count_values) != len(answers):
            raise ValueError("counts must align with answers")
        best_index = max(range(len(count_values)), key=count_values.__getitem__)
        answer = answers[best_index]
        if not isinstance(answer, str) or not answer.strip():
            continue

        weight = 1.0
        if use_weights:
            prompt_weight = float(row.get("prompt_weight", 1))
            weights = row.get("weights")
            if isinstance(weights, Sequence) and not isinstance(weights, (str, bytes)):
                if len(weights) != len(answers):
                    raise ValueError("weights must align with answers")
                weight = prompt_weight * float(weights[best_index])
            else:
                weight = prompt_weight
        if weight <= 0:
            continue
        prepared.append(
            {
                "prompt": prompt.strip(),
                "answer": answer.strip(),
                "task": str(row.get("task") or ""),
                "weight": weight,
            }
        )
    return prepared


def best_answer_sft_to_rows(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Turn ``FeedbackStore.export_sft`` rows into unweighted training rows."""

    prepared: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, Mapping):
            continue
        prompt = record.get("prompt")
        answer = record.get("answer")
        if isinstance(prompt, str) and isinstance(answer, str):
            if prompt.strip() and answer.strip():
                prepared.append(
                    {
                        "prompt": prompt.strip(),
                        "answer": answer.strip(),
                        "task": str(record.get("task") or ""),
                        "weight": 1.0,
                    }
                )
            continue

        messages = record.get("messages")
        if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
            continue
        user_text = ""
        assistant_text = ""
        for message in messages:
            if not isinstance(message, Mapping):
                continue
            content = message.get("content")
            if not isinstance(content, str):
                continue
            if message.get("role") == "user" and content.strip():
                user_text = content.strip()
            elif message.get("role") == "assistant" and content.strip():
                assistant_text = content.strip()
        if user_text and assistant_text:
            prepared.append(
                {
                    "prompt": user_text,
                    "answer": assistant_text,
                    "task": str(record.get("task") or ""),
                    "weight": 1.0,
                }
            )
    return prepared


def _encode_ids(tokenizer: Any, text: str, *, add_special_tokens: bool) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=add_special_tokens)
    if isinstance(encoded, Mapping):
        ids = encoded["input_ids"]
    else:
        ids = getattr(encoded, "input_ids", encoded)
    if ids and isinstance(ids[0], (list, tuple)):
        ids = ids[0]
    return [int(token_id) for token_id in ids]


def _prompt_ids(tokenizer: Any, prompt: str) -> list[int]:
    if getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": prompt}]
        try:
            encoded = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True
            )
        except TypeError:
            encoded = tokenizer.apply_chat_template(messages, tokenize=True)
        if isinstance(encoded, Mapping):
            encoded = encoded["input_ids"]
        if encoded and isinstance(encoded[0], (list, tuple)):
            encoded = encoded[0]
        ids = [int(token_id) for token_id in encoded]
        if ids:
            return ids
    return _encode_ids(tokenizer, prompt, add_special_tokens=True)


def _answer_ids(tokenizer: Any, answer: str) -> list[int]:
    ids = _encode_ids(tokenizer, answer, add_special_tokens=False)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is not None and (not ids or ids[-1] != eos_id):
        ids.append(int(eos_id))
    if not ids:
        raise ValueError("answer tokenizes to no tokens")
    return ids


def _fit_sequence(
    prompt_ids: list[int], answer_ids: list[int], max_seq_len: int
) -> tuple[list[int], list[int]]:
    answer = answer_ids[:max_seq_len]
    budget = max_seq_len - len(answer)
    prompt = prompt_ids[-budget:] if budget else []
    return prompt, answer


def _supervised_example(
    tokenizer: Any, row: Mapping[str, Any], max_seq_len: int
) -> dict[str, Any] | None:
    prompt = row.get("prompt")
    answer = row.get("answer")
    if not isinstance(prompt, str) or not isinstance(answer, str):
        return None
    weight = float(row.get("weight", 1.0))
    if weight <= 0:
        return None
    prompt_ids, answer_ids = _fit_sequence(
        _prompt_ids(tokenizer, prompt),
        _answer_ids(tokenizer, answer),
        max_seq_len,
    )
    if not answer_ids:
        return None
    input_ids = prompt_ids + answer_ids
    labels = [-100] * len(prompt_ids) + answer_ids
    return {
        "input_ids": input_ids,
        "labels": labels,
        "weight": weight,
        "supervised_tokens": len(answer_ids),
    }


def _collate(
    examples: Sequence[Mapping[str, Any]], pad_token_id: int
) -> dict[str, Any]:
    import torch

    width = max(len(example["input_ids"]) for example in examples)
    input_ids = []
    attention_mask = []
    labels = []
    for example in examples:
        padding = width - len(example["input_ids"])
        input_ids.append(list(example["input_ids"]) + [pad_token_id] * padding)
        attention_mask.append([1] * len(example["input_ids"]) + [0] * padding)
        labels.append(list(example["labels"]) + [-100] * padding)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "sample_weights": torch.tensor(
            [float(example["weight"]) for example in examples], dtype=torch.float32
        ),
    }


def train_weighted_sft(
    model: Any,
    tokenizer: Any,
    rows: Iterable[Mapping[str, Any]],
    *,
    max_steps: int = 1,
    batch_size: int = 1,
    learning_rate: float = 1e-4,
    seed: int = 0,
    max_seq_len: int = 512,
    token_budget: int | None = None,
    limit: int | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    """Run weighted SFT on prepared ``prompt``/``answer``/``weight`` rows.

    The loss is exactly :func:`shadow_lab.trainers.weighted_shifted_ce_loss`
    (the loss behind :class:`WeightedSFTTrainer`): per-sample CE over the
    non-``-100`` answer tokens, averaged as ``sum(w_i * CE_i) / sum(w_i)``.
    Weight-0 rows are filtered before batching. A manual loop is used instead
    of TRL so tiny random models can train one or two steps deterministically
    (``torch.manual_seed(seed)`` and a seeded shuffle). ``token_budget`` caps
    loss-participating answer tokens in source order; when the first row
    alone exceeds the budget it is kept and ``token_budget_overflow`` is
    recorded (zero training rows are never produced silently).
    """

    import torch

    from .trainers import weighted_shifted_ce_loss

    if max_steps < 1:
        raise ValueError("max_steps must be >= 1")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if max_seq_len < 2:
        raise ValueError("max_seq_len must be >= 2")
    if token_budget is not None and token_budget < 0:
        raise ValueError("token_budget must be non-negative")
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")

    source_rows = list(rows)
    if limit is not None:
        source_rows = source_rows[:limit]
    examples: list[dict[str, Any]] = []
    tokens_seen = 0
    overflow = False
    for row in source_rows:
        if not isinstance(row, Mapping):
            continue
        example = _supervised_example(tokenizer, row, max_seq_len)
        if example is None:
            continue
        if token_budget is None:
            examples.append(example)
            tokens_seen += int(example["supervised_tokens"])
            continue
        if tokens_seen + example["supervised_tokens"] > token_budget:
            if not examples:
                examples.append(example)
                tokens_seen += int(example["supervised_tokens"])
                overflow = True
            break
        examples.append(example)
        tokens_seen += int(example["supervised_tokens"])
    if not examples:
        raise ValueError("no training rows left after filtering zero weights")

    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not params:
        raise ValueError("model has no trainable parameters")
    optimizer = torch.optim.AdamW(params, lr=learning_rate)

    torch.manual_seed(seed)
    random.Random(seed).shuffle(examples)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = 0
    target_device = torch.device(device) if device else next(model.parameters()).device
    model.train()

    started = time.perf_counter()
    steps = 0
    last_loss = 0.0
    while steps < max_steps:
        for start in range(0, len(examples), batch_size):
            batch = _collate(examples[start : start + batch_size], int(pad_token_id))
            input_ids = batch["input_ids"].to(target_device)
            attention_mask = batch["attention_mask"].to(target_device)
            labels = batch["labels"].to(target_device)
            weights = batch["sample_weights"].to(target_device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            loss = weighted_shifted_ce_loss(logits, labels, weights)
            loss.backward()
            optimizer.step()
            last_loss = float(loss.detach().item())
            steps += 1
            if steps >= max_steps:
                break

    return {
        "steps": steps,
        "rows": len(examples),
        "tokens_seen": tokens_seen,
        "loss": last_loss,
        "wall_clock": time.perf_counter() - started,
        "seed": int(seed),
        "batch_size": int(batch_size),
        "max_steps": int(max_steps),
        "max_seq_len": int(max_seq_len),
        "token_budget": token_budget,
        "token_budget_overflow": overflow,
    }
