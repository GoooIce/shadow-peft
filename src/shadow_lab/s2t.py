"""S2T-Local diagnostic core: Hit@K, Agree@1, rank, and go/no-go.

The diagnostic decides whether a shadow model's top-K candidates cover the
base model's top-1 *before* any selector head is trained (spec issue #3,
``docs/specs/s2t-local-shadow-selector.md``). This module is deliberately
dependency-free — it does **not** import torch. Callers pass plain sequences
(or any duck-typed tensor exposing ``tolist()``), which keeps the diagnostic
usable on CPU-only analysis machines and makes every metric reproducible.

Metric contract
---------------
``Hit@K``
    The base top-1 token id is among the shadow top-K ids.
``Agree@1``
    The shadow top-1 token id equals the base top-1 token id.
``rank``
    The 1-based position of the base top-1 in the deterministic shadow order:
    descending logit, ties broken toward the lower token id. The rank is also
    reported when the token falls outside top-K. Ties therefore have distinct
    ordinal ranks, and ``rank <= K`` is exactly ``Hit@K``.

    Reports expose two rank views. ``rank_all_*`` uses every token's exact
    ``base_top1_rank_in_shadow`` (records written by the diagnostic runner
    carry it), including tokens outside the captured top-K. ``rank_hit_*``
    keeps the older hit-position view: it only knows ranks from the captured
    candidate list, so a miss outside that list is absent. ``rank_p50`` …
    ``rank_p99`` remain as deprecated aliases of ``rank_hit_*``.

Degenerate inputs never produce NaNs:
- empty candidate lists and empty logit vectors are misses with ``rank=None``;
- an all-``-inf`` shadow distribution carries no preference, so it is also a
  miss with ``rank=None`` (the first vocabulary id is *not* treated as top-1);
- ``K < 1`` raises; ``K > vocab`` is truncated to the vocabulary size and
  recorded as ``effective_k``.
NaN logits raise ``ValueError`` because they make the ordering undefined;
``+-inf`` logits are accepted (``-inf`` is used for masked candidates).

The shadow logits consumed here are expected to come from the **initial**
shadow hidden state, not the injected deployment path. Reports built by
:func:`build_report` carry this caveat explicitly.
"""

from __future__ import annotations

import csv
import io
import json
import math
import operator
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

DEFAULT_KS: tuple[int, ...] = (1, 4, 8, 16)
POSITION_BUCKETS: tuple[str, ...] = ("0", "1-3", "4-7", "8-15", "16+")
GO_THRESHOLD = 0.85
PILOT_THRESHOLD = 0.70
HIT8_METRIC = "hit@8"
AGREE1_METRIC = "agree@1"
SCHEMA_VERSION = "s2t-local-diagnostic/v1"
SHADOW_SOURCE = "initial_shadow_hidden"
SHADOW_SOURCE_NOTE = (
    "shadow_logits were captured from the initial shadow hidden state; the "
    "injected base_shadow deployment path is not measured by this diagnostic."
)
REPORT_FIELDS: tuple[str, ...] = (
    "schema_version",
    "shadow_source",
    "shadow_source_note",
    "n_tokens",
    "n_prompts",
    "ks",
    "hit_at_1",
    "hit_at_4",
    "hit_at_8",
    "hit_at_16",
    "agree_at_1",
    "n_rank_known",
    "n_rank_known_all",
    "rank_hit_p50",
    "rank_hit_p90",
    "rank_hit_p95",
    "rank_hit_p99",
    "rank_all_p50",
    "rank_all_p90",
    "rank_all_p95",
    "rank_all_p99",
    "rank_p50",
    "rank_p90",
    "rank_p95",
    "rank_p99",
    "n_empty_candidates",
    "n_all_neg_inf",
    "hit_at_8_ci",
    "agree_at_1_ci",
    "ci_n_boot",
    "ci_seed",
    "go_no_go",
    "go_threshold",
    "pilot_threshold",
    "go_no_go_rationale",
    "by_task",
    "by_position_bucket",
    "notes",
)
CSV_FIELDS: tuple[str, ...] = (
    "schema_version",
    "scope",
    "group",
    "n_tokens",
    "n_prompts",
    "hit_at_1",
    "hit_at_4",
    "hit_at_8",
    "hit_at_16",
    "agree_at_1",
    "rank_hit_p50",
    "rank_hit_p90",
    "rank_hit_p95",
    "rank_hit_p99",
    "rank_all_p50",
    "rank_all_p90",
    "rank_all_p95",
    "rank_all_p99",
    "rank_p50",
    "rank_p90",
    "rank_p95",
    "rank_p99",
    "n_rank_known",
    "n_rank_known_all",
    "n_empty_candidates",
    "n_all_neg_inf",
    "hit_at_8_ci_low",
    "hit_at_8_ci_high",
    "agree_at_1_ci_low",
    "agree_at_1_ci_high",
    "ci_n_boot",
    "ci_seed",
    "go_no_go",
    "go_threshold",
    "pilot_threshold",
)
_MISSING = object()


@dataclass(frozen=True, slots=True)
class TokenHit:
    """Per-token diagnostic result.

    ``effective_k`` is ``min(k, vocab)``; ``0`` means the shadow distribution
    was empty. ``rank``/``shadow_top1`` are ``None`` when the distribution is
    empty or all ``-inf``.
    """

    hit: bool
    agree: bool
    rank: int | None
    shadow_top1: int | None
    k: int
    effective_k: int

    def __bool__(self) -> bool:
        return self.hit

    def to_dict(self) -> dict[str, Any]:
        return {
            "hit": self.hit,
            "agree": self.agree,
            "rank": self.rank,
            "shadow_top1": self.shadow_top1,
            "k": self.k,
            "effective_k": self.effective_k,
        }


def _as_sequence(values: Any, *, name: str) -> list[Any]:
    if values is None:
        return []
    tolist = getattr(values, "tolist", None)
    if callable(tolist):
        values = tolist()
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of numbers, not {type(values).__name__}")
    try:
        return list(values)
    except TypeError as error:
        raise TypeError(
            f"{name} must be a sequence of numbers, got {type(values).__name__}"
        ) from error


def _logits(values: Any, *, name: str = "shadow_logits") -> list[float]:
    result: list[float] = []
    for index, value in enumerate(_as_sequence(values, name=name)):
        try:
            number = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name}[{index}] is not a number: {value!r}") from error
        if math.isnan(number):
            raise ValueError(f"{name}[{index}] is NaN; the shadow ordering is undefined")
        result.append(number)
    return result


def _index(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer token id, got {value!r}")
    try:
        number = operator.index(value)
    except TypeError as error:
        raise ValueError(f"{name} must be an integer token id, got {value!r}") from error
    if number < 0:
        raise ValueError(f"{name} must be >= 0, got {number}")
    return number


def _positive_k(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError(f"k must be a positive integer, got {value!r}")
    try:
        number = operator.index(value)
    except TypeError as error:
        raise ValueError(f"k must be a positive integer, got {value!r}") from error
    if number < 1:
        raise ValueError(f"k must be >= 1, got {number}")
    return number


def _has_preference(logits: Sequence[float]) -> bool:
    """At least one candidate is better than ``-inf`` (no preference otherwise)."""

    return any(value > -math.inf for value in logits)


def _shadow_top1(logits: Sequence[float]) -> int:
    """Argmax with the deterministic lower-token-id tie-break (O(vocab))."""

    best = 0
    for token in range(1, len(logits)):
        if logits[token] > logits[best]:
            best = token
    return best


def _shadow_rank(logits: Sequence[float], base_top1: int) -> int:
    """Ordinal 1-based rank in descending-logit order (O(vocab), no sort)."""

    target = logits[base_top1]
    # Count candidates ordered before the base token: strictly better logits,
    # plus equal logits with a lower id (the deterministic tie-break).
    return 1 + sum(
        1
        for token, value in enumerate(logits)
        if value > target or (value == target and token < base_top1)
    )


def token_hit(
    shadow_logits: Sequence[float] | Any,
    base_top1: int,
    k: int = 1,
) -> TokenHit:
    """Score one token: Hit@K, Agree@1 and the base token's shadow rank.

    ``k`` is positional to match the ticket's ``token_hit(logits, base, k)``
    form. ``k > vocab`` truncates (see ``effective_k``); ``k < 1`` raises.
    """

    requested_k = _positive_k(k)
    base = _index(base_top1, name="base_top1")
    logits = _logits(shadow_logits)

    if not logits:
        return TokenHit(
            hit=False, agree=False, rank=None, shadow_top1=None,
            k=requested_k, effective_k=0,
        )
    if base >= len(logits):
        raise ValueError(
            f"base_top1 {base} is outside the shadow vocabulary of size {len(logits)}"
        )
    effective_k = min(requested_k, len(logits))
    if not _has_preference(logits):
        return TokenHit(
            hit=False, agree=False, rank=None, shadow_top1=None,
            k=requested_k, effective_k=effective_k,
        )

    rank = _shadow_rank(logits, base)
    shadow_top1 = _shadow_top1(logits)
    return TokenHit(
        hit=rank <= effective_k,
        agree=shadow_top1 == base,
        rank=rank,
        shadow_top1=shadow_top1,
        k=requested_k,
        effective_k=effective_k,
    )


# ---------------------------------------------------------------------------
# Record normalization and summaries
# ---------------------------------------------------------------------------


def _field(record: Any, name: str, default: Any = None) -> Any:
    if isinstance(record, Mapping):
        return record.get(name, default)
    return getattr(record, name, default)


_DIAGNOSTIC_FIELDS = ("base_top1", "base_logits", "shadow_logits", "shadow_topk_ids")


def _looks_like_record(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(key in value for key in _DIAGNOSTIC_FIELDS)
    return any(hasattr(value, key) for key in _DIAGNOSTIC_FIELDS)


def _normalize_ks(ks: int | Sequence[int]) -> tuple[int, ...]:
    values: Sequence[Any]
    if isinstance(ks, int) and not isinstance(ks, bool):
        values = (ks,)
    else:
        try:
            values = list(ks)
        except TypeError as error:
            raise ValueError(f"ks must be a positive integer or a sequence, got {ks!r}") from error
    normalized: list[int] = []
    for value in values:
        try:
            k = _positive_k(value)
        except ValueError as error:
            raise ValueError(f"ks must contain positive integers, got {value!r}") from error
        if k not in normalized:
            normalized.append(k)
    if not normalized:
        raise ValueError("ks must contain at least one K")
    return tuple(normalized)


def _record_base_top1(record: Any) -> int:
    base_top1 = _field(record, "base_top1")
    if base_top1 is not None:
        return _index(base_top1, name="base_top1")
    base_logits = _field(record, "base_logits")
    if base_logits is None:
        raise ValueError("record needs base_top1 or base_logits")
    logits = _logits(base_logits, name="base_logits")
    if not logits or not _has_preference(logits):
        raise ValueError("base_logits has no top-1 candidate")
    return _shadow_top1(logits)


def _record_prompt_id(record: Any) -> str | None:
    prompt_id = _field(record, "prompt_id")
    return None if prompt_id is None else str(prompt_id)


def _task_label(record: Any) -> str:
    task = _field(record, "task")
    return "unspecified" if task is None or task == "" else str(task)


def _exact_rank(record: Any) -> int | None:
    """Read the runner's exact ``base_top1_rank_in_shadow`` when present."""

    value = _field(record, "base_top1_rank_in_shadow")
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(
            f"base_top1_rank_in_shadow must be a positive integer, got {value!r}"
        )
    try:
        rank = operator.index(value)
    except TypeError as error:
        raise ValueError(
            f"base_top1_rank_in_shadow must be a positive integer, got {value!r}"
        ) from error
    if rank < 1:
        raise ValueError(f"base_top1_rank_in_shadow must be >= 1, got {rank}")
    return rank


def _score_record(record: Any, ks: Sequence[int]) -> dict[str, Any]:
    """Normalize one record into per-K hits plus the shared token facts."""

    base = _record_base_top1(record)
    raw_logits = _field(record, "shadow_logits")

    if raw_logits is not None:
        logits = _logits(raw_logits)
        empty = not logits
        all_neg_inf = bool(logits) and not _has_preference(logits)
        if empty or all_neg_inf:
            hits = {k: False for k in ks}
            agree = False
            rank = rank_hit = None
        else:
            rank = _shadow_rank(logits, base)
            rank_hit = rank
            agree = _shadow_top1(logits) == base
            hits = {k: rank <= min(k, len(logits)) for k in ks}
    else:
        raw_ids = _field(record, "shadow_topk_ids")
        if raw_ids is None:
            ids: list[int] = []
        else:
            ids = [
                _index(value, name="shadow_topk_ids")
                for value in _as_sequence(raw_ids, name="shadow_topk_ids")
            ]
        empty = not ids
        all_neg_inf = False
        hits = {k: (not empty and base in ids[: min(k, len(ids))]) for k in ks}
        agree = bool(ids) and ids[0] == base
        if base in ids:
            rank = rank_hit = ids.index(base) + 1
        else:
            exact = _exact_rank(record)
            rank = exact
            rank_hit = None

    return {
        "hits": hits,
        "agree": agree,
        "rank": rank,
        "rank_hit": rank_hit,
        "empty": empty,
        "all_neg_inf": all_neg_inf,
        "prompt_id": _record_prompt_id(record),
        "task": _task_label(record),
        "position": _field(record, "position", _MISSING),
    }


def _percentile(sorted_values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile (same convention as onpolicy_eval)."""

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


def summarize_hits(
    records: Iterable[Any],
    ks: int | Sequence[int] = DEFAULT_KS,
) -> dict[str, Any]:
    """Aggregate per-token diagnostics over ``records``.

    Each record is a mapping (or object) with ``base_top1`` or ``base_logits``
    and either ``shadow_logits`` (full distribution; preferred) or
    ``shadow_topk_ids`` (candidate list, already ordered by the shadow model).
    The runner's ``base_top1_rank_in_shadow`` supplies the exact rank for a
    token outside the captured candidate list; without it such a token has
    ``rank=None``.

    Returned flat schema (stable key order):
    ``n``, ``n_prompts``, ``hit_at_<K>``..., ``agree_at_1``,
    ``n_rank_known``, ``n_rank_known_all``, ``rank_hit_p50/p90/p95/p99``,
    ``rank_all_p50/p90/p95/p99``, deprecated ``rank_p50/p90/p95/p99``
    aliases, ``n_empty_candidates``, ``n_all_neg_inf``.
    """

    record_list = list(records)
    if not record_list:
        raise ValueError("summarize_hits needs at least one record")
    ks_tuple = _normalize_ks(ks)
    rows = [_score_record(record, ks_tuple) for record in record_list]
    n = len(rows)

    prompt_ids = {row["prompt_id"] for row in rows if row["prompt_id"] is not None}
    ranks_hit = sorted(row["rank_hit"] for row in rows if row["rank_hit"] is not None)
    ranks_all = sorted(row["rank"] for row in rows if row["rank"] is not None)

    summary: dict[str, Any] = {"n": n, "n_prompts": len(prompt_ids)}
    for k in ks_tuple:
        summary[f"hit_at_{k}"] = sum(1 for row in rows if row["hits"][k]) / n
    summary["agree_at_1"] = sum(1 for row in rows if row["agree"]) / n
    summary["n_rank_known"] = len(ranks_hit)
    summary["n_rank_known_all"] = len(ranks_all)
    for label, quantile in (("p50", 0.50), ("p90", 0.90), ("p95", 0.95), ("p99", 0.99)):
        hit_value = _percentile(ranks_hit, quantile) if ranks_hit else None
        all_value = _percentile(ranks_all, quantile) if ranks_all else None
        summary[f"rank_hit_{label}"] = hit_value
        summary[f"rank_all_{label}"] = all_value
        summary[f"rank_{label}"] = hit_value
    summary["n_empty_candidates"] = sum(1 for row in rows if row["empty"])
    summary["n_all_neg_inf"] = sum(1 for row in rows if row["all_neg_inf"])
    return summary


def position_bucket(position: int) -> str:
    """Fixed token-position buckets: 0, 1-3, 4-7, 8-15, 16+."""

    index = _index(position, name="position")
    if index == 0:
        return "0"
    if index <= 3:
        return "1-3"
    if index <= 7:
        return "4-7"
    if index <= 15:
        return "8-15"
    return "16+"


def group_metrics(
    records: Iterable[Any],
    by: str = "task",
    ks: int | Sequence[int] = DEFAULT_KS,
) -> dict[str, dict[str, Any]]:
    """Stratify records by ``task`` or ``position_bucket``.

    Every returned group carries the full :func:`summarize_hits` schema, so
    ``sum(metric["n"] for metric in groups.values())`` equals the overall
    token count. Task groups are sorted; position-bucket groups follow
    :data:`POSITION_BUCKETS` order.
    """

    record_list = list(records)
    if not record_list:
        raise ValueError("group_metrics needs at least one record")
    if by not in ("task", "position_bucket"):
        raise ValueError(f"by must be 'task' or 'position_bucket', got {by!r}")

    grouped: dict[str, list[Any]] = {}
    if by == "task":
        for record in record_list:
            grouped.setdefault(_task_label(record), []).append(record)
        order = sorted(grouped)
    else:
        for record in record_list:
            position = _field(record, "position", _MISSING)
            if position is _MISSING or position is None:
                raise ValueError("position_bucket grouping needs a position on every record")
            grouped.setdefault(position_bucket(position), []).append(record)
        order = [bucket for bucket in POSITION_BUCKETS if bucket in grouped]

    return {key: summarize_hits(grouped[key], ks=ks) for key in order}


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals and the go/no-go gate
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BootstrapResult:
    """Percentile-bootstrap interval plus the sample size it was built from.

    Iterable as ``(low, high)`` for callers that only unpack the interval;
    the point estimate and provenance stay available as attributes.
    """

    metric: str
    estimate: float
    low: float
    high: float
    n: int
    n_boot: int
    seed: int
    confidence: float

    def __iter__(self):
        yield self.low
        yield self.high

    def __getitem__(self, index: int) -> float:
        return (self.low, self.high)[index]

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "estimate": self.estimate,
            "ci_low": self.low,
            "ci_high": self.high,
            "n": self.n,
            "n_boot": self.n_boot,
            "seed": self.seed,
            "confidence": self.confidence,
        }


@dataclass(frozen=True, slots=True)
class GoNoGoDecision:
    """Engineering decision with its sample size and interval attached."""

    decision: str
    hit8: float
    n: int | None
    ci_low: float | None
    ci_high: float | None
    go_threshold: float
    pilot_threshold: float
    rationale: str

    @property
    def ci(self) -> tuple[float, float] | None:
        if self.ci_low is None or self.ci_high is None:
            return None
        return (self.ci_low, self.ci_high)

    def __getitem__(self, key: str) -> Any:
        if key in ("decision", "go_no_go"):
            return self.decision
        if key == "ci":
            return self.ci
        if key in self.__dataclass_fields__:
            return getattr(self, key)
        raise KeyError(key)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "hit8": self.hit8,
            "n": self.n,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "go_threshold": self.go_threshold,
            "pilot_threshold": self.pilot_threshold,
            "rationale": self.rationale,
        }


def _parse_metric(metric: str) -> tuple[str, int, str]:
    if not isinstance(metric, str):
        raise ValueError(f"metric must be a string, got {metric!r}")
    name = metric.strip().lower().replace(" ", "")
    if name in ("agree", "agree@1", "agree_at_1", "agree1"):
        return "agree", 1, AGREE1_METRIC
    if name == "hit":
        return "hit", 8, HIT8_METRIC
    for prefix in ("hit@", "hit_at_"):
        if name.startswith(prefix):
            suffix = name[len(prefix):]
            try:
                k = _positive_k(int(suffix))
            except (TypeError, ValueError) as error:
                raise ValueError(f"metric has an invalid K: {metric!r}") from error
            return "hit", k, f"hit@{k}"
    raise ValueError(f"metric must be hit@K or agree@1, got {metric!r}")


def _metric_values(data: Iterable[Any], metric: str) -> tuple[list[float], str]:
    kind, k, canonical = _parse_metric(metric)
    record_list = list(data)
    if not record_list:
        raise ValueError("bootstrap_ci needs at least one value")
    if kind == "hit":
        values = [
            1.0 if _score_record(record, (k,))["hits"][k] else 0.0
            for record in record_list
        ]
    else:
        values = [
            1.0 if _score_record(record, (1,))["agree"] else 0.0
            for record in record_list
        ]
    return values, canonical


def _as_rate_values(data: Iterable[Any]) -> list[float]:
    values: list[float] = []
    for index, value in enumerate(data):
        try:
            number = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"bootstrap_ci value[{index}] is not numeric; pass records with "
                "metric='hit@K' or 'agree@1' instead"
            ) from error
        if not math.isfinite(number):
            raise ValueError(f"bootstrap_ci value[{index}] must be finite, got {number!r}")
        values.append(number)
    return values


def bootstrap_ci(
    data: Iterable[Any],
    metric: str | None = None,
    *,
    n_boot: int = 1000,
    seed: int = 0,
    confidence: float = 0.95,
) -> BootstrapResult:
    """Percentile-bootstrap CI for a rate or for Hit@K / Agree@1 records.

    ``data`` is either a sequence of 0/1 (or numeric) values, or a sequence of
    diagnostic records. With records, pass ``metric="hit@K"`` or
    ``"agree@1"``; omitting ``metric`` defaults records to Hit@8 and numeric
    values to a plain mean. Deterministic for a fixed ``seed``.
    """

    if isinstance(n_boot, bool) or not isinstance(n_boot, int) or n_boot < 1:
        raise ValueError(f"n_boot must be a positive integer, got {n_boot!r}")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence!r}")

    items = list(data)
    if metric is None and items and _looks_like_record(items[0]):
        metric = HIT8_METRIC
    if metric is None:
        values = _as_rate_values(items)
        canonical = "mean"
    else:
        values, canonical = _metric_values(items, metric)
    if not values:
        raise ValueError("bootstrap_ci needs at least one value")

    rng = random.Random(seed)
    count = len(values)
    means = sorted(
        sum(values[rng.randrange(count)] for _ in range(count)) / count
        for _ in range(n_boot)
    )
    alpha = (1.0 - confidence) / 2.0
    return BootstrapResult(
        metric=canonical,
        estimate=sum(values) / count,
        low=_percentile(means, alpha),
        high=_percentile(means, 1.0 - alpha),
        n=count,
        n_boot=n_boot,
        seed=seed,
        confidence=confidence,
    )


def _parse_ci(ci: Any) -> tuple[float | None, float | None]:
    if ci is None:
        return None, None
    if isinstance(ci, BootstrapResult):
        return ci.low, ci.high
    try:
        values = [float(value) for value in ci]
    except (TypeError, ValueError) as error:
        raise ValueError(f"ci must be a (low, high) pair, got {ci!r}") from error
    if len(values) != 2:
        raise ValueError(f"ci must be a (low, high) pair, got {ci!r}")
    low, high = values
    if not (math.isfinite(low) and math.isfinite(high)) or low > high:
        raise ValueError(f"ci must be finite with low <= high, got {ci!r}")
    return low, high


def go_no_go(
    hit8: float | BootstrapResult | Mapping[str, Any],
    n: int | None = None,
    ci: Any = None,
    *,
    go_threshold: float = GO_THRESHOLD,
    pilot_threshold: float = PILOT_THRESHOLD,
) -> GoNoGoDecision:
    """Apply the spec's Hit@8 gate.

    ``>= go_threshold`` -> ``go``; ``>= pilot_threshold`` -> ``pilot``;
    otherwise ``no_go``. The decision uses the point estimate; the interval
    and sample size are carried in the result for the report. ``hit8`` may be
    a rate, a :class:`BootstrapResult`, or a report/summary mapping.
    """

    source: Any = hit8
    if isinstance(hit8, BootstrapResult):
        value = hit8.estimate
        if n is None:
            n = hit8.n
        if ci is None:
            ci = (hit8.low, hit8.high)
    elif isinstance(hit8, Mapping):
        raw = hit8.get("hit_at_8", hit8.get("hit8", hit8.get("hit@8")))
        if raw is None:
            raise ValueError("hit8 mapping needs hit_at_8 (or hit8)")
        source = raw
        value = float(raw)
        if n is None:
            candidate = hit8.get("n", hit8.get("n_tokens"))
            n = None if candidate is None else candidate
        if ci is None:
            ci = hit8.get("hit_at_8_ci")
    else:
        try:
            value = float(hit8)
        except (TypeError, ValueError) as error:
            raise ValueError(f"hit8 must be a rate, got {hit8!r}") from error

    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"hit8 must be a finite rate in [0, 1], got {source!r}")
    if n is not None:
        if isinstance(n, bool):
            raise ValueError(f"n must be a non-negative integer, got {n!r}")
        try:
            n = operator.index(n)
        except TypeError as error:
            raise ValueError(f"n must be a non-negative integer, got {n!r}") from error
        if n < 0:
            raise ValueError(f"n must be >= 0, got {n}")
    if not 0.0 <= pilot_threshold <= go_threshold <= 1.0:
        raise ValueError(
            "thresholds must satisfy 0 <= pilot_threshold <= go_threshold <= 1"
        )

    low, high = _parse_ci(ci)
    if value >= go_threshold:
        decision = "go"
        rationale = (
            f"Hit@8={value:.3f} >= go threshold {go_threshold:.2f}: "
            "candidate coverage supports selector training"
        )
    elif value >= pilot_threshold:
        decision = "pilot"
        rationale = (
            f"Hit@8={value:.3f} is in [{pilot_threshold:.2f}, {go_threshold:.2f}): "
            "pilot on one task and check Agree@1 first"
        )
    else:
        decision = "no_go"
        rationale = (
            f"Hit@8={value:.3f} < pilot threshold {pilot_threshold:.2f}: "
            "stop S2T-Local and return to GKD representation work"
        )

    return GoNoGoDecision(
        decision=decision,
        hit8=value,
        n=n,
        ci_low=low,
        ci_high=high,
        go_threshold=go_threshold,
        pilot_threshold=pilot_threshold,
        rationale=rationale,
    )


# ---------------------------------------------------------------------------
# Report assembly and stable exports
# ---------------------------------------------------------------------------


def build_report(
    records: Iterable[Any],
    ks: int | Sequence[int] = DEFAULT_KS,
    *,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    """Assemble the auditable S2T-Local diagnostic report.

    The report always contains the standard K=1/4/8/16 metrics (so consumers
    have a stable schema), plus any extra ``ks`` supplied by the caller. It
    bootstraps Hit@8 and Agree@1 with the same seed and applies
    :func:`go_no_go` to the Hit@8 point estimate.
    """

    record_list = list(records)
    if not record_list:
        raise ValueError("build_report needs at least one record")
    requested_ks = _normalize_ks(ks)
    metric_ks = tuple(dict.fromkeys((*DEFAULT_KS, *requested_ks)))

    summary = summarize_hits(record_list, ks=metric_ks)
    by_task = group_metrics(record_list, by="task", ks=metric_ks)
    has_positions = all(
        _field(record, "position", None) is not None for record in record_list
    )
    by_position = (
        group_metrics(record_list, by="position_bucket", ks=metric_ks)
        if has_positions
        else {}
    )
    hit_ci = bootstrap_ci(record_list, metric=HIT8_METRIC, n_boot=n_boot, seed=seed)
    agree_ci = bootstrap_ci(record_list, metric=AGREE1_METRIC, n_boot=n_boot, seed=seed)
    decision = go_no_go(hit_ci)

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "shadow_source": SHADOW_SOURCE,
        "shadow_source_note": SHADOW_SOURCE_NOTE,
        "n_tokens": summary["n"],
        "n_prompts": summary["n_prompts"],
        "ks": list(requested_ks),
        "hit_at_1": summary["hit_at_1"],
        "hit_at_4": summary["hit_at_4"],
        "hit_at_8": summary["hit_at_8"],
        "hit_at_16": summary["hit_at_16"],
        "agree_at_1": summary["agree_at_1"],
        "n_rank_known": summary["n_rank_known"],
        "n_rank_known_all": summary["n_rank_known_all"],
        "rank_hit_p50": summary["rank_hit_p50"],
        "rank_hit_p90": summary["rank_hit_p90"],
        "rank_hit_p95": summary["rank_hit_p95"],
        "rank_hit_p99": summary["rank_hit_p99"],
        "rank_all_p50": summary["rank_all_p50"],
        "rank_all_p90": summary["rank_all_p90"],
        "rank_all_p95": summary["rank_all_p95"],
        "rank_all_p99": summary["rank_all_p99"],
        "rank_p50": summary["rank_p50"],
        "rank_p90": summary["rank_p90"],
        "rank_p95": summary["rank_p95"],
        "rank_p99": summary["rank_p99"],
        "n_empty_candidates": summary["n_empty_candidates"],
        "n_all_neg_inf": summary["n_all_neg_inf"],
        "hit_at_8_ci": [hit_ci.low, hit_ci.high],
        "agree_at_1_ci": [agree_ci.low, agree_ci.high],
        "ci_n_boot": n_boot,
        "ci_seed": seed,
        "go_no_go": decision.decision,
        "go_threshold": decision.go_threshold,
        "pilot_threshold": decision.pilot_threshold,
        "go_no_go_rationale": decision.rationale,
        "by_task": by_task,
        "by_position_bucket": by_position,
        "notes": [
            "Hit@K is base top-1 membership in the shadow top-K; Agree@1 is exact top-1 agreement.",
            "Rank is 1-based in descending shadow order with ties broken toward the lower token id.",
            "rank_all_* uses each token's exact base_top1_rank_in_shadow, including ranks outside the captured top-K; n_rank_known_all counts them.",
            "rank_hit_* keeps the older hit-position view (ranks known from the captured top-K list) while rank_all_* covers every exact rank; rank_p50..p99 are deprecated aliases of rank_hit_p50..p99.",
            "Empty and all -inf candidate sets are misses with rank=None; K > vocab truncates to vocab.",
        ],
    }
    for k in requested_ks:
        key = f"hit_at_{k}"
        if key not in report:
            report[key] = summary[key]
    return report


def _plain(value: Any) -> Any:
    """Convert report values to JSON-friendly plain objects."""

    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, (int, float, str)):
        return value
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _plain(to_dict())
    return value


def to_json(report: Mapping[str, Any], *, indent: int | None = 2) -> str:
    """Serialize a report (or summary) with stable keys and no NaN tokens.

    ``sort_keys=True`` makes repeated exports byte-identical regardless of how
    the mapping was assembled; ``allow_nan=False`` turns an accidental NaN
    into an error instead of invalid JSON.
    """

    return json.dumps(
        _plain(report),
        ensure_ascii=False,
        allow_nan=False,
        indent=indent,
        sort_keys=True,
    )


def _csv_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"refusing to export non-finite value {value!r} to CSV")
        return str(round(value, 6))
    return str(value)


def _csv_row(
    *,
    scope: str,
    group: str,
    metrics: Mapping[str, Any],
    report: Mapping[str, Any],
    include_ci: bool,
) -> dict[str, Any]:
    row: dict[str, Any] = dict.fromkeys(CSV_FIELDS)
    row["schema_version"] = report.get("schema_version", SCHEMA_VERSION)
    row["scope"] = scope
    row["group"] = group
    row["n_tokens"] = metrics.get("n", metrics.get("n_tokens"))
    row["n_prompts"] = metrics.get("n_prompts")
    for key in (
        "hit_at_1",
        "hit_at_4",
        "hit_at_8",
        "hit_at_16",
        "agree_at_1",
        "rank_hit_p50",
        "rank_hit_p90",
        "rank_hit_p95",
        "rank_hit_p99",
        "rank_all_p50",
        "rank_all_p90",
        "rank_all_p95",
        "rank_all_p99",
        "rank_p50",
        "rank_p90",
        "rank_p95",
        "rank_p99",
        "n_rank_known",
        "n_rank_known_all",
        "n_empty_candidates",
        "n_all_neg_inf",
    ):
        row[key] = metrics.get(key)
    row["ci_n_boot"] = report.get("ci_n_boot")
    row["ci_seed"] = report.get("ci_seed")
    row["go_threshold"] = report.get("go_threshold")
    row["pilot_threshold"] = report.get("pilot_threshold")
    if include_ci:
        hit_ci = report.get("hit_at_8_ci")
        if hit_ci is not None:
            row["hit_at_8_ci_low"], row["hit_at_8_ci_high"] = hit_ci
        agree_ci = report.get("agree_at_1_ci")
        if agree_ci is not None:
            row["agree_at_1_ci_low"], row["agree_at_1_ci_high"] = agree_ci
        decision = report.get("go_no_go")
        if isinstance(decision, Mapping):
            decision = decision.get("decision")
        row["go_no_go"] = decision
    return row


def to_csv(report: Mapping[str, Any]) -> str:
    """Long-format CSV: one overall row plus one row per task / position bucket.

    The header is the fixed :data:`CSV_FIELDS` tuple. Group rows leave the
    confidence-interval and decision columns blank because only the overall
    sample is bootstrapped; ``n_tokens`` per scope reconciles with the overall
    row.
    """

    overall = dict(report)
    if "n_tokens" in overall and "n" not in overall:
        overall["n"] = overall["n_tokens"]
    rows = [
        _csv_row(scope="overall", group="all", metrics=overall, report=report, include_ci=True)
    ]
    for group, metrics in (report.get("by_task") or {}).items():
        rows.append(
            _csv_row(scope="task", group=str(group), metrics=metrics, report=report, include_ci=False)
        )
    for group, metrics in (report.get("by_position_bucket") or {}).items():
        rows.append(
            _csv_row(
                scope="position_bucket",
                group=str(group),
                metrics=metrics,
                report=report,
                include_ci=False,
            )
        )

    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer,
        fieldnames=list(CSV_FIELDS),
        lineterminator="\n",
        extrasaction="ignore",
    )
    writer.writeheader()
    for row in rows:
        writer.writerow({key: _csv_cell(row.get(key)) for key in CSV_FIELDS})
    return buffer.getvalue()
