"""n-best answer reranking with JitRL-lite experience memory (issue #38).

Spec: ``docs/specs/jitrl-memory-hotfix.md`` §Implementation Decisions 3 and
Testing Decisions「重排缝」.  The reranker is a pure, deterministic function of
the n-best candidates and the retrieved ``MemoryHit`` list: it never calls a
model, never mutates memory and never trains.

Scoring follows the spec::

    score = logprob + alpha * memory_score

``memory_score`` is the return-weighted sum of retrieved entries whose stored
answer matches the candidate after task-specific normalization.  A memory
signal may only replace the model's top-1 when at least ``min_support``
retrieved entries support the same candidate and their average return is above
``min_return``.  With no hits, or when the gate fails, the model's original
top-1 is returned unchanged.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from shadow_lab.restrain import extract_answer

from .memory import MemoryHit

# Relative tolerance for numeric answers, matching RESTRAIN-lite evaluation.
NUMBER_RTOL = 1e-2

_ENGLISH_VERDICT_RE = re.compile(
    r"(?<![A-Za-z])(?P<verdict>pass|fail|rework)(?![A-Za-z])",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class Candidate:
    """One sampled/beam answer with its sequence log probability."""

    text: str
    logprob: float
    finish_reason: str = "stop"


@dataclass(frozen=True, slots=True)
class RerankResult:
    """The selected candidate and the evidence behind the selection.

    ``scores`` is aligned with the input candidates (model logprob plus the
    optional memory term).  ``memory_score`` and ``support`` describe the
    selected candidate; ``support`` counts retrieved entries in the winning
    return direction.
    """

    index: int
    text: str
    scores: list[float]
    memory_score: float
    support: int
    reranked: bool
    reason: str


def _task_kind(task: str | None) -> str:
    tokens = set(re.findall(r"[a-z0-9]+", str(task or "").lower()))
    if "verdict" in tokens:
        return "verdict"
    if tokens & {"number", "numeric", "calc", "calculation", "measurement"}:
        return "number"
    return "text"


def _normalize_text(text: str) -> str:
    return " ".join(text.strip().casefold().split())


def _normalize_verdict(text: str) -> str | None:
    parsed = extract_answer(text, "quality_verdict")
    if isinstance(parsed, str):
        return parsed
    matches = list(_ENGLISH_VERDICT_RE.finditer(text))
    if not matches:
        return None
    return matches[-1].group("verdict").lower()


def _normalize_number(text: str) -> tuple[float, str] | None:
    parsed = extract_answer(text, "process_calc")
    if isinstance(parsed, tuple):
        return parsed
    return None


def normalize_answer(text: str, task: str | None) -> str | tuple[float, str] | None:
    """Return a comparable key for *text* under *task*.

    - verdict tasks: ``pass`` / ``fail`` / ``rework`` (``不合格`` is matched
      before ``合格`` by the RESTRAIN-lite extractor);
    - number tasks: ``(value, canonical_unit)``, compared with a 1% relative
      tolerance when reranking; when both answers carry a unit the units must
      match;
    - every other task: case-folded text with collapsed whitespace.

    ``None`` means the answer carries no comparable form (empty free text, a
    verdict with no label, or a number task with no number).
    """

    if not isinstance(text, str):
        text = str(text)
    kind = _task_kind(task)
    if kind == "verdict":
        return _normalize_verdict(text)
    if kind == "number":
        return _normalize_number(text)
    normalized = _normalize_text(text)
    return normalized or None


def _answers_match(
    left: str | tuple[float, str] | None,
    right: str | tuple[float, str] | None,
) -> bool:
    if isinstance(left, tuple) and isinstance(right, tuple):
        left_value, left_unit = left
        right_value, right_unit = right
        if left_unit and right_unit and left_unit != right_unit:
            return False
        return math.isclose(
            left_value,
            right_value,
            rel_tol=NUMBER_RTOL,
            abs_tol=max(abs(right_value) * NUMBER_RTOL, 1e-9),
        )
    if isinstance(left, str) and isinstance(right, str):
        return left == right
    return False


def _entry_similarity(
    candidate_key: str | tuple[float, str] | None,
    entry_candidates: Sequence[str],
    task: str | None,
) -> float:
    if candidate_key is None:
        return 0.0
    for answer in entry_candidates:
        if _answers_match(candidate_key, normalize_answer(answer, task)):
            return 1.0
    return 0.0


@dataclass(frozen=True, slots=True)
class _Evidence:
    score: float
    support: int
    mean_return: float


_NO_EVIDENCE = _Evidence(score=0.0, support=0, mean_return=0.0)


def _evidence(
    candidate: Candidate | str,
    hits: Iterable[MemoryHit],
    task: str | None,
) -> _Evidence:
    """Return same-direction memory evidence for one candidate.

    Matching hits are split by return sign.  Only the strictly larger
    direction contributes: positive memories support a rewrite, negative
    memories oppose it, and a tie (or no non-zero return) carries no signal.
    """

    text = candidate.text if isinstance(candidate, Candidate) else str(candidate)
    candidate_key = normalize_answer(text, task)
    if candidate_key is None:
        return _NO_EVIDENCE

    positive: list[tuple[float, float]] = []
    negative: list[tuple[float, float]] = []
    for hit in hits:
        similarity = _entry_similarity(candidate_key, hit.entry.candidates, task)
        if similarity <= 0.0:
            continue
        return_value = float(hit.entry.return_value)
        if return_value > 0.0:
            positive.append((return_value, similarity))
        elif return_value < 0.0:
            negative.append((return_value, similarity))

    if len(positive) > len(negative):
        relevant = positive
    elif len(negative) > len(positive):
        relevant = negative
    else:
        return _NO_EVIDENCE

    support = len(relevant)
    return _Evidence(
        score=sum(value * similarity for value, similarity in relevant),
        support=support,
        mean_return=sum(value for value, _ in relevant) / support,
    )


def memory_score(
    candidate: Candidate | str,
    hits: Iterable[MemoryHit],
    task: str | None,
) -> float:
    """Return-weighted support for *candidate* from retrieved *hits*.

    Each hit whose stored answer normalizes to the candidate contributes
    ``return_value * similarity``.  Only returns in the majority direction are
    counted, so positive and negative memories never cancel each other into a
    false signal.
    """

    return _evidence(candidate, hits, task).score


def _baseline_index(candidates: Sequence[Candidate]) -> int:
    """Index of the highest logprob, ties resolved to the lowest index."""

    best = 0
    for index in range(1, len(candidates)):
        if candidates[index].logprob > candidates[best].logprob:
            best = index
    return best


def _result(
    candidates: Sequence[Candidate],
    index: int,
    *,
    scores: list[float],
    memory_score: float,
    support: int,
    reranked: bool,
    reason: str,
) -> RerankResult:
    return RerankResult(
        index=index,
        text=candidates[index].text,
        scores=scores,
        memory_score=memory_score,
        support=support,
        reranked=reranked,
        reason=reason,
    )


def rerank(
    candidates: Sequence[Candidate],
    hits: Iterable[MemoryHit],
    *,
    alpha: float = 0.5,
    min_support: int = 2,
    min_return: float = 0.0,
) -> RerankResult:
    """Select one n-best candidate, optionally rewritten by memory support.

    ``hits`` are the retrieved ``MemoryHit`` objects in relevance order.  With
    no hits the result is the logprob baseline, bit for bit.  A memory rewrite
    is only allowed for a candidate supported by at least ``min_support`` hits
    whose average return exceeds ``min_return``; otherwise the baseline top-1
    is preserved and ``reranked`` stays false.
    """

    candidate_list = list(candidates)
    if not candidate_list:
        raise ValueError("candidates must not be empty")
    hit_list = list(hits)
    baseline = _baseline_index(candidate_list)
    baseline_scores = [float(candidate.logprob) for candidate in candidate_list]

    if not hit_list:
        return _result(
            candidate_list,
            baseline,
            scores=baseline_scores,
            memory_score=0.0,
            support=0,
            reranked=False,
            reason="no_memory",
        )

    task = hit_list[0].entry.task or None
    evidence = [_evidence(candidate, hit_list, task) for candidate in candidate_list]
    scores = [
        float(candidate.logprob) + alpha * item.score
        for candidate, item in zip(candidate_list, evidence, strict=True)
    ]
    eligible = [
        index
        for index, item in enumerate(evidence)
        if (
            item.support > 0
            and item.support >= min_support
            and item.mean_return > min_return
        )
    ]

    if not eligible:
        max_support = max(item.support for item in evidence)
        if max_support == 0:
            reason = "no_support"
        elif max_support < min_support:
            reason = "insufficient_support"
        else:
            reason = "low_return"
        return _result(
            candidate_list,
            baseline,
            scores=scores,
            memory_score=evidence[baseline].score,
            support=evidence[baseline].support,
            reranked=False,
            reason=reason,
        )

    winner = max(eligible, key=lambda index: (scores[index], -index))
    if winner == baseline:
        return _result(
            candidate_list,
            baseline,
            scores=scores,
            memory_score=evidence[baseline].score,
            support=evidence[baseline].support,
            reranked=False,
            reason="top1_unchanged",
        )
    return _result(
        candidate_list,
        winner,
        scores=scores,
        memory_score=evidence[winner].score,
        support=evidence[winner].support,
        reranked=True,
        reason="reranked",
    )


__all__ = [
    "NUMBER_RTOL",
    "Candidate",
    "RerankResult",
    "memory_score",
    "normalize_answer",
    "rerank",
]
