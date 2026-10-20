"""RESTRAIN-lite core: structured answer extraction, clustering and weights.

Pure functions for the structured-output flywheel (spec:
``docs/specs/restrain-lite-structured-flywheel.md`` §Implementation
Decisions 2/3). No torch/numpy, no I/O, no randomness: the same answers and
counts always produce the same row.

Conventions
-----------
- Supported tasks are ``quality_verdict`` (verdict label extraction) and
  ``process_calc`` (first number + unit extraction). Any other task yields
  ``None`` / an unclusterable result.
- Vote weights follow RESTRAIN Eq. 2-3 with a Gaussian centred at full
  consensus, ``g(f) = exp(-(1 - f)^2 / (2 * sigma^2))``, so the majority
  answer is weighted up while never collapsing to 1/0. ``consistency`` is
  ``g(max_j f_j)`` (Eq. 5) and is kept raw in reports; ``prompt_weight``
  applies the first-version ``max_count < kappa`` gate on top of it.
- A numeric prompt is ``unclusterable`` when there are no parsed answers or
  when the largest numeric cluster does not strictly lead (a tie means no
  dominant answer). Verdict prompts are unclusterable only when no answer
  parses. ``build_self_train_row`` returns ``None`` for unclusterable rows,
  i.e. the exporter skips them instead of emitting a noisy sample.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

_VERDICT_ALIASES = {
    "放行": "pass",
    "合格": "pass",
    "拦截": "fail",
    "不合格": "fail",
    "返修": "rework",
}

# Frames introduce a verdict; the label nearest the last frame wins. The
# fail spelling is listed first so 不合格 can never be read as 合格.
_VERDICT_TOKEN_RE = re.compile(
    r"(?P<frame>判定|结论)|(?P<verdict>不\s*合格|合格|拦截|放行|返修)"
)

# Characters that carry no answer on their own (markdown/JSON punctuation).
_FORMAT_ONLY_RE = re.compile(r"[\s*_`#>|~=+.,;:!?/\\()\[\]{}<>\"'、，。；：！？…·\-–—]+")

# Canonical spellings for number units. Latin keys are lower-case; matching
# is case-insensitive. Longer aliases must win (毫米 before 米, mm before m).
_UNIT_ALIASES = {
    "℃": "℃",
    "°c": "℃",
    "摄氏度": "℃",
    "度c": "℃",
    "mm": "mm",
    "毫米": "mm",
    "cm": "cm",
    "厘米": "cm",
    "km": "km",
    "千米": "km",
    "公里": "km",
    "μm": "um",
    "um": "um",
    "微米": "um",
    "nm": "nm",
    "纳米": "nm",
    "m": "m",
    "米": "m",
    "in": "in",
    "inch": "in",
    "英寸": "in",
    "ft": "ft",
    "英尺": "ft",
    "kg": "kg",
    "千克": "kg",
    "公斤": "kg",
    "mg": "mg",
    "毫克": "mg",
    "g": "g",
    "克": "g",
    "t": "t",
    "吨": "t",
    "ms": "ms",
    "毫秒": "ms",
    "s": "s",
    "秒": "s",
    "min": "min",
    "分钟": "min",
    "h": "h",
    "hr": "h",
    "小时": "h",
    "l": "L",
    "升": "L",
    "ml": "mL",
    "毫升": "mL",
    "m3": "m3",
    "立方米": "m3",
    "pa": "Pa",
    "帕": "Pa",
    "kpa": "kPa",
    "千帕": "kPa",
    "mpa": "MPa",
    "兆帕": "MPa",
    "bar": "bar",
    "巴": "bar",
    "psi": "psi",
    "v": "V",
    "伏": "V",
    "mv": "mV",
    "毫伏": "mV",
    "a": "A",
    "安": "A",
    "ma": "mA",
    "毫安": "mA",
    "%": "%",
    "％": "%",
    "百分比": "%",
}

_UNIT_PATTERN = "|".join(
    re.escape(alias) for alias in sorted(_UNIT_ALIASES, key=len, reverse=True)
)
_NUMBER_RE = re.compile(
    rf"(?<![A-Za-z0-9_.])"
    rf"(?P<number>[-+−]?\d+(?:,\d{{3}})*(?:\.\d+)?(?:[eE][-+]?\d+)?)"
    rf"\s*(?P<unit>{_UNIT_PATTERN})?(?![A-Za-z])",
    re.IGNORECASE,
)


def _is_blank_or_format(text: str) -> bool:
    return not _FORMAT_ONLY_RE.sub("", text)


def _normalize_verdict_token(token: str) -> str | None:
    return _VERDICT_ALIASES.get(re.sub(r"\s+", "", token))


def _extract_verdict(text: str) -> str | None:
    events = [
        (match.lastgroup, match.group())
        for match in _VERDICT_TOKEN_RE.finditer(text)
    ]
    verdicts = [index for index, (kind, _) in enumerate(events) if kind == "verdict"]
    if not verdicts:
        return None

    frames = [index for index, (kind, _) in enumerate(events) if kind == "frame"]
    if not frames:
        # No judgment marker: the most recent label is the conclusion.
        return _normalize_verdict_token(events[verdicts[-1]][1])

    marker = frames[-1]
    for index in verdicts:
        if index > marker:
            return _normalize_verdict_token(events[index][1])
    # A trailing marker with no label after it: use the closest label before.
    for index in reversed(verdicts):
        if index < marker:
            return _normalize_verdict_token(events[index][1])
    return None


def _normalize_unit(raw: str | None) -> str:
    if not raw:
        return ""
    return _UNIT_ALIASES.get(raw.strip().lower(), raw.strip().lower())


def _parse_number(raw: str) -> float:
    cleaned = raw.replace(",", "").replace("−", "-")
    return float(cleaned)


def _extract_number(text: str) -> tuple[float, str] | None:
    first_bare: tuple[float, str] | None = None
    for match in _NUMBER_RE.finditer(text):
        value = _parse_number(match.group("number"))
        unit = _normalize_unit(match.group("unit"))
        if unit:
            # Prefer the first value that carries a recognized unit; step
            # numbers ("做3次，结果245℃") must not shadow the measurement.
            return value, unit
        if first_bare is None:
            first_bare = (value, "")
    return first_bare


def extract_answer(text: str, task: str | None) -> str | tuple[float, str] | None:
    """Extract the normalized answer from one rollout.

    ``quality_verdict`` returns ``"pass"``, ``"fail"`` or ``"rework"``;
    ``process_calc`` returns ``(value, unit)`` with a canonical unit (``℃``,
    ``mm``, ``mm``-family aliases, ...), or ``(value, "")`` when no unit is
    present. Empty answers, pure formatting tokens, answers with no match and
    unsupported tasks return ``None``.
    """

    if not isinstance(text, str) or _is_blank_or_format(text):
        return None
    if task == "quality_verdict":
        return _extract_verdict(text)
    if task == "process_calc":
        return _extract_number(text)
    return None


# ---------------------------------------------------------------------------
# Answer clustering
# ---------------------------------------------------------------------------

_CLUSTER_METHODS = {
    "quality_verdict": "verdict_label",
    "process_calc": "number_tolerance",
}


@dataclass(frozen=True, slots=True)
class ClusterResult:
    """Clustered answers for one prompt.

    ``clusters`` maps a canonical label to the raw answers that produced it;
    ``counts`` mirrors those labels with vote counts. ``cluster_method`` is
    ``"verdict_label"``, ``"number_tolerance"`` or ``"none"``.
    ``unclusterable`` is True when the prompt must be skipped (no parsed
    answers, or no strictly dominant numeric cluster).
    """

    clusters: dict[str, list[str]]
    counts: dict[str, int]
    cluster_method: str
    unclusterable: bool


def _format_number(value: float) -> str:
    if value == 0:
        return "0"
    if value.is_integer():
        return str(int(value))
    return f"{value:.12g}"


def cluster_answers(
    answers: Sequence[str],
    task: str | None,
    *,
    number_rtol: float = 1e-6,
) -> ClusterResult:
    """Group answers into canonical clusters; never raises on bad answers.

    Verdict answers cluster by normalized label. Numeric answers cluster when
    units match exactly and values satisfy ``math.isclose(rel_tol=number_rtol)``
    against the cluster's first value. Numeric ties leave ``unclusterable``
    true because no answer dominates.
    """

    method = _CLUSTER_METHODS.get(task or "")
    if method is None:
        return ClusterResult({}, {}, "none", True)
    if number_rtol < 0:
        raise ValueError("number_rtol must be non-negative")

    clusters: dict[str, list[str]] = {}
    representatives: dict[str, tuple[float, str]] = {}

    for answer in answers:
        if not isinstance(answer, str):
            continue
        parsed = extract_answer(answer, task)
        if parsed is None:
            continue

        if task == "quality_verdict":
            label = parsed
            clusters.setdefault(label, []).append(answer)
            continue

        value, unit = parsed
        label = None
        for existing, (rep_value, rep_unit) in representatives.items():
            if rep_unit == unit and math.isclose(
                rep_value, value, rel_tol=number_rtol, abs_tol=0.0
            ):
                label = existing
                break
        if label is None:
            base = _format_number(value) + unit
            label = base
            suffix = 2
            while label in representatives:
                label = f"{base}#{suffix}"
                suffix += 1
            representatives[label] = (value, unit)
        clusters.setdefault(label, []).append(answer)

    counts = {label: len(items) for label, items in clusters.items()}
    if not counts:
        return ClusterResult(clusters, counts, method, True)

    unclusterable = False
    if task == "process_calc" and len(counts) > 1:
        ordered = sorted(counts.values(), reverse=True)
        unclusterable = ordered[0] == ordered[1]
    return ClusterResult(clusters, counts, method, unclusterable)


# ---------------------------------------------------------------------------
# Vote, consistency and prompt weights
# ---------------------------------------------------------------------------


def _count_values(counts: Mapping[str, int] | Sequence[int]) -> list[float]:
    if isinstance(counts, Mapping):
        return [float(value) for value in counts.values()]
    return [float(value) for value in counts]


def _gaussian(fraction: float, sigma: float) -> float:
    return math.exp(-((1.0 - fraction) ** 2) / (2.0 * sigma * sigma))


def _check_sigma(sigma: float) -> None:
    if sigma <= 0:
        raise ValueError("sigma must be positive")


def vote_weights(
    counts: Mapping[str, int] | Sequence[int],
    *,
    sigma: float = 0.5,
) -> dict[str, float] | list[float]:
    """Soft pseudo-label weights ``g(f_j) / sum_l g(f_l)``.

    Returns a mapping when *counts* is a mapping, otherwise a list aligned
    with the input. All-zero counts return an empty container instead of
    dividing by zero; positive weights always sum to 1.
    """

    _check_sigma(sigma)
    values = _count_values(counts)
    total = sum(values)
    if total <= 0:
        return {} if isinstance(counts, Mapping) else []

    scores = [_gaussian(value / total, sigma) for value in values]
    denominator = sum(scores)
    weights = [score / denominator for score in scores]
    if isinstance(counts, Mapping):
        return dict(zip(counts.keys(), weights, strict=True))
    return weights


def consistency(
    counts: Mapping[str, int] | Sequence[int],
    sigma: float = 0.5,
) -> float:
    """Prompt consistency ``g(max_j f_j)``; 0.0 for empty/all-zero counts."""

    _check_sigma(sigma)
    values = _count_values(counts)
    total = sum(values)
    if total <= 0:
        return 0.0
    return _gaussian(max(values) / total, sigma)


def prompt_weight(
    counts: Mapping[str, int] | Sequence[int],
    *,
    kappa: int = 3,
) -> int:
    """First-version gate: 1 when the largest cluster has >= kappa votes, else 0."""

    values = _count_values(counts)
    return 0 if max(values, default=0.0) < kappa else 1


# ---------------------------------------------------------------------------
# Export row
# ---------------------------------------------------------------------------


def build_self_train_row(
    prompt: str,
    task: str,
    answers: Sequence[str],
    *,
    reference_counts: Mapping[str, int] | Sequence[int] | None = None,
    adapter_version: str = "",
    sigma: float = 0.5,
    kappa: int = 3,
    number_rtol: float = 1e-6,
) -> dict[str, object] | None:
    """Build one JSON-ready self-training row, or ``None`` when unclusterable.

    ``answers`` holds the canonical cluster labels (one per unique answer),
    with ``counts`` and ``weights`` aligned to it. ``reference_counts`` may
    be omitted (zeros), keyed by cluster label, or a sequence aligned with
    ``answers``. Unclusterable prompts are skipped by returning ``None`` so
    they are never written to the training JSONL.
    """

    clustered = cluster_answers(answers, task, number_rtol=number_rtol)
    if clustered.unclusterable:
        return None

    labels = list(clustered.clusters)
    counts = [clustered.counts[label] for label in labels]
    weight_map = vote_weights(clustered.counts, sigma=sigma)
    weights = [weight_map[label] for label in labels]

    if reference_counts is None:
        references = [0] * len(labels)
    elif isinstance(reference_counts, Mapping):
        references = [int(reference_counts.get(label, 0)) for label in labels]
    else:
        references = [int(count) for count in reference_counts]
        if len(references) != len(labels):
            raise ValueError("reference_counts must align with clustered answers")

    return {
        "prompt": prompt,
        "task": task,
        "answers": labels,
        "counts": counts,
        "reference_counts": references,
        "weights": weights,
        "prompt_weight": prompt_weight(counts, kappa=kappa),
        "consistency": consistency(counts, sigma=sigma),
        "cluster_method": clustered.cluster_method,
        "adapter_version": adapter_version,
    }
