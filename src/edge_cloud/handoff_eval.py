"""Offline evaluation bench for token-level edge-cloud handoff (交接评测台).

This module is deliberately independent from ``Router``: it owns its own
strategy loop so the bench can be developed and run while the router's handoff
control flow evolves. Four strategies share one prompt set, one deterministic
edge backend and one scripted cloud answer:

``edge_only``
    Full edge generation, never escalates.
``legacy``
    Full edge generation, then the existing ``ConfidenceEstimator`` threshold
    decides whether the whole answer is escalated.
``probe``
    Streams tokens through ``ProbePolicy``. A handoff continues the cloud answer
    from the emitted prefix; when the probe is unavailable or does not trigger,
    the legacy threshold decides on the complete answer.
``oracle``
    Same loop as ``probe`` but driven by ``OraclePolicy`` and the per-prompt
    ``cloud_better_from`` label, i.e. the evaluation-only upper bound.

Everything that affects the numbers is deterministic: synthetic data is seeded,
latency is a closed-form function of token counts and configured prices (never
wall-clock), the cloud answer is scripted from the dataset, and bootstrap
resampling uses ``random.Random(seed)``. Running the same command twice writes
byte-identical ``results.json``, ``summary.csv``, ``curves.csv`` and
``per_prompt.jsonl``.

Accuracy semantics: a final answer is correct only when its whitespace-normalized
form equals the expected cloud answer. A truncated edge prefix continued by the
cloud is exact because the synthetic edge answer is itself a prefix of the cloud
answer. ``contains_expected_rate`` is a diagnostic that also counts a non-prefix
draft joined with the full cloud answer; it never drives the main accuracy, the
risk-coverage curve or the cost-accuracy curve.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from . import knowledge
from .backends import (
    Backend,
    Generation,
    MockStreamingEdgeBackend,
    StreamingBackend,
)
from .confidence import ConfidenceEstimator
from .continuation import assemble_continuation, looks_like_duplicate_prefix
from .cost import CostModel
from .handoff import HandoffDecision, HandoffFeatures, HandoffPolicy, HandoffStep
from .policies import OraclePolicy, ProbePolicy
from .probe import PROBE_VERSION, fit_probe, write_probe  # noqa: F401 - re-exported for scripts
from .protocol import InferRequest

SCHEMA_VERSION = "exp9-handoff-eval/v1"
STRATEGIES: tuple[str, ...] = ("edge_only", "legacy", "probe", "oracle")
BOUNDARY_KINDS: tuple[str, ...] = ("token", "sentence", "newline", "step", "json_field")

# HandoffFeatures fields that are numeric (or boolean) and therefore usable as
# probe inputs. ``boundary_kind`` and ``task`` are categorical and excluded.
NUMERIC_HANDOFF_FEATURES: tuple[str, ...] = (
    "kl_base_shadow",
    "top1_agreement",
    "base_entropy",
    "shadow_entropy",
    "base_margin",
    "shadow_margin",
    "base_top1_in_shadow_topk",
    "low_prob_frac",
    "prefix_tokens",
    "format_ok",
)

# Mirror of ``backends._MOCK_TOKEN``: the bench must count tokens the same way
# the streaming mock does, without importing a private symbol from another
# module. ``test_token_counter_matches_the_mock_streaming_backend`` guards this.
_TOKEN = re.compile(r"[A-Za-z]+(?:['’\-][A-Za-z]+)*|\d+(?:[.,]\d+)*|\s+|.", re.DOTALL)

_PROMPT_TEMPLATES: tuple[str, ...] = (
    "现场提问 {tag}：{keyword} 相关异常，请给出处理建议。",
    "产线工单 {tag}：值班工程师询问 {keyword}，请按步骤回答。",
    "异常记录 {tag}：涉及 {keyword}，需要结论与依据。",
)

DEFAULT_COST_MODEL = CostModel(
    cost_per_edge_token=0.000002,
    cost_per_cloud_input_token=0.000004,
    cost_per_cloud_output_token=0.000012,
    cost_per_cloud_call=0.0005,
    cloud_rtt_ms=120.0,
)

_STRATEGY_SET = frozenset(STRATEGIES)


def count_tokens(text: str) -> int:
    """Count text tokens exactly as ``MockStreamingEdgeBackend`` splits them."""

    return len(_TOKEN.findall(text))


def _round(value: float, digits: int = 6) -> float:
    return round(float(value), digits)


def _quantile(ordered: Sequence[float], q: float) -> float:
    if not ordered:
        return 0.0
    index = max(0, min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1))
    return ordered[index]


def _normalize(text: str) -> str:
    return " ".join(str(text).split())


def _matches_expected(final_text: str, expected: str) -> bool:
    """Strict answer match after whitespace normalization."""

    want = _normalize(expected)
    return bool(want) and _normalize(final_text) == want


def _contains_expected(final_text: str, expected: str) -> bool:
    """Diagnostic-only match: the expected answer appears at the end.

    A non-prefix edge draft joined with a full cloud answer satisfies this but
    not ``_matches_expected``; the main accuracy and every curve use the strict
    form, so a bad prefix can never be counted as a correct answer.
    """

    want = _normalize(expected)
    return bool(want) and _normalize(final_text).endswith(want)


@dataclass(slots=True, frozen=True)
class EvalSample:
    """One prompt plus its scripted edge/cloud answers and handoff label.

    ``cloud_better_from`` is the first prefix token index at which handing the
    remaining generation to the cloud beats staying on the edge. The synthetic
    generator sets it to ``risk_at``; datasets without labels may omit it.
    """

    sample_id: str
    prompt: str
    task: str
    expected: str
    edge_answer: str
    cloud_answer: str
    cloud_better_from: int | None = None
    risk_at: int | None = None
    edge_quality: float | None = None
    shadow_gap: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _optional_int(value: Any, *, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} 必须是非负整数或 null")
    return value


def _optional_float(value: Any, *, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} 必须是有限数值或 null")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} 必须是有限数值或 null")
    return number


_TRUNCATION_BOUNDARIES = "。！？\n"


def _boundary_prefix(text: str, target_fraction: float) -> str:
    """Return a proper prefix of *text* cut at a punctuation/newline boundary.

    The cut closest to ``target_fraction`` of the text is chosen. When the
    knowledge-base answer has no interior boundary inside the requested band,
    the nearest interior boundary is used instead, so the result is always a
    non-empty, non-full prefix ending on a semantic edge.
    """

    n = len(text)
    positions = [
        index + 1
        for index, char in enumerate(text)
        if char in _TRUNCATION_BOUNDARIES and 0 < index + 1 < n
    ]
    if not positions:
        raise ValueError("无法为样本找到可截断的边界")
    target = target_fraction * n
    cut = min(positions, key=lambda position: (abs(position - target), position))
    return text[:cut]


def synthetic_dataset(n: int = 32, seed: int = 9) -> list[EvalSample]:
    """Build a deterministic prompt set from the mock knowledge base.

    Prompts keep an original KB keyword so ``MockStreamingEdgeBackend`` matches
    the intended entry. About 30% of the samples are *easy*: the edge answer
    equals the full cloud answer, with no risk point and no handoff label. The
    rest are *hard*: the edge answer is a boundary-truncated prefix of the cloud
    answer, ``risk_at`` sits before the cut, and ``cloud_better_from`` equals
    ``risk_at``. Every sample carries a shadow gap so probe features are always
    observable.
    """

    if n < 0:
        raise ValueError("n 不能为负数")
    rng = random.Random(seed)
    easy_count = round(n * 0.3)
    difficulty_flags = [True] * easy_count + [False] * (n - easy_count)
    rng.shuffle(difficulty_flags)
    samples: list[EvalSample] = []
    for index, easy in enumerate(difficulty_flags):
        entry = knowledge.ENTRIES[index % len(knowledge.ENTRIES)]
        keyword = rng.choice(entry.keywords)
        template = rng.choice(_PROMPT_TEMPLATES)
        prompt = template.format(tag=f"E9-{index:03d}", keyword=keyword)
        if easy:
            edge_answer = entry.cloud_answer
            risk_at = None
            edge_quality = round(
                min(0.98, max(0.92, entry.edge_quality + rng.uniform(0.01, 0.04))), 4
            )
        else:
            edge_answer = _boundary_prefix(entry.cloud_answer, rng.uniform(0.4, 0.7))
            token_count = max(1, count_tokens(edge_answer))
            # Keep the risky stretch long enough that the legacy confidence
            # estimator escalates on the truncated answer.
            risk_at = rng.randint(0, token_count // 2)
            edge_quality = round(
                max(0.05, min(0.98, entry.edge_quality + rng.uniform(-0.04, 0.04))), 4
            )
        shadow_gap = rng.choice((0.1, 0.2, 0.3, 0.4))
        samples.append(
            EvalSample(
                sample_id=f"s{index:04d}",
                prompt=prompt,
                task=entry.task,
                expected=entry.cloud_answer,
                edge_answer=edge_answer,
                cloud_answer=entry.cloud_answer,
                cloud_better_from=risk_at,
                risk_at=risk_at,
                edge_quality=edge_quality,
                shadow_gap=shadow_gap,
            )
        )
    return samples


def load_dataset(path: str | Path) -> list[EvalSample]:
    """Read a JSONL prompt set; order is preserved and optional fields default to null.

    Required keys: ``prompt``, ``task``, ``expected``, ``edge_answer``,
    ``cloud_answer``. Optional keys: ``cloud_better_from``, ``risk_at``,
    ``edge_quality``, ``shadow_gap``. An explicit ``sample_id``/``id`` is
    honoured; otherwise ids are ``s0000``, ``s0001``, …
    """

    samples: list[EvalSample] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"数据集第 {line_number} 行不是合法 JSON: {exc}") from exc
            if not isinstance(row, Mapping):
                raise ValueError(f"数据集第 {line_number} 行必须是 JSON 对象")
            required = ("prompt", "task", "expected", "edge_answer", "cloud_answer")
            missing = [key for key in required if not isinstance(row.get(key), str)]
            if missing:
                raise ValueError(
                    f"数据集第 {line_number} 行缺少字符串字段: {', '.join(missing)}"
                )
            sample_id = row.get("sample_id", row.get("id"))
            if not isinstance(sample_id, str) or not sample_id:
                sample_id = f"s{len(samples):04d}"
            samples.append(
                EvalSample(
                    sample_id=sample_id,
                    prompt=str(row["prompt"]),
                    task=str(row["task"]),
                    expected=str(row["expected"]),
                    edge_answer=str(row["edge_answer"]),
                    cloud_answer=str(row["cloud_answer"]),
                    cloud_better_from=_optional_int(
                        row.get("cloud_better_from"), name="cloud_better_from"
                    ),
                    risk_at=_optional_int(row.get("risk_at"), name="risk_at"),
                    edge_quality=_optional_float(row.get("edge_quality"), name="edge_quality"),
                    shadow_gap=_optional_float(row.get("shadow_gap"), name="shadow_gap"),
                )
            )
    return samples


@dataclass(frozen=True, slots=True)
class LatencyModel:
    """Closed-form latency so repeated runs never depend on wall-clock noise."""

    edge_ms_per_token: float = 12.0
    cloud_prefill_ms_per_token: float = 0.8
    cloud_decode_ms_per_token: float = 18.0
    cloud_rtt_ms: float = 120.0

    def estimate(
        self,
        *,
        edge_tokens: int,
        cloud_prefill_tokens: int,
        cloud_decode_tokens: int,
        cloud_calls: int,
    ) -> float:
        total = edge_tokens * self.edge_ms_per_token
        if cloud_calls:
            total += (
                self.cloud_rtt_ms
                + cloud_prefill_tokens * self.cloud_prefill_ms_per_token
                + cloud_decode_tokens * self.cloud_decode_ms_per_token
            )
        return _round(total)


class _NeverHandoffPolicy:
    def reset(self, context: dict[str, Any] | None = None) -> None:
        del context

    def observe(self, step: HandoffStep) -> HandoffDecision:
        return HandoffDecision(action="continue", boundary=step.features.boundary_kind)


class _RecordingPolicy:
    """Delegate to a policy while remembering every observed step and decision."""

    def __init__(self, inner: HandoffPolicy | None = None) -> None:
        self.inner: HandoffPolicy = inner or _NeverHandoffPolicy()
        self.context: dict[str, Any] | None = None
        self.steps: list[HandoffStep] = []
        self.decisions: list[HandoffDecision] = []

    def reset(self, context: dict[str, Any] | None = None) -> None:
        self.context = context
        self.steps = []
        self.decisions = []
        self.inner.reset(context)

    def observe(self, step: HandoffStep) -> HandoffDecision:
        decision = self.inner.observe(step)
        self.steps.append(step)
        self.decisions.append(decision)
        return decision


def _mock_backend_factory(sample: EvalSample) -> Backend:
    return MockStreamingEdgeBackend(
        risk_at=sample.risk_at,
        edge_quality=sample.edge_quality,
        shadow_gap=sample.shadow_gap,
        answer=sample.edge_answer,
    )


def _generate_full(
    backend: Backend, request: InferRequest, recorder: _RecordingPolicy
) -> Generation:
    if isinstance(backend, StreamingBackend):
        return backend.generate_until(request, recorder, context={"task": request.task})
    return backend.generate(request)


def _last_trust(recorder: _RecordingPolicy) -> float | None:
    for decision in reversed(recorder.decisions):
        if decision.prefix_trust is not None:
            return float(decision.prefix_trust)
    return None


def apply_cloud_answer(
    partial_trace: str, cloud_answer: str
) -> tuple[str, str, bool]:
    """Return ``(final_text, continuation, mismatch)`` for one cloud answer.

    When the cloud answer starts with the edge prefix it is a true continuation
    and the prefix is stripped. Otherwise the cloud returned a full answer; it
    is joined with ``assemble_continuation`` so a restated prefix is never
    duplicated, and the mismatch is recorded for metrics.
    """

    if not partial_trace:
        return cloud_answer, cloud_answer, False
    if cloud_answer.startswith(partial_trace):
        continuation = cloud_answer[len(partial_trace):]
        final = assemble_continuation(partial_trace, continuation)
        return final, continuation, False
    final = assemble_continuation(partial_trace, cloud_answer)
    restated = looks_like_duplicate_prefix(partial_trace, cloud_answer)
    return final, cloud_answer, not restated


def _evaluate_sample(
    strategy: str,
    sample: EvalSample,
    *,
    factory: Callable[[EvalSample], Backend],
    probe_path: str | None,
    escalation_threshold: float,
    cost_model: CostModel,
    latency_model: LatencyModel,
) -> dict[str, Any]:
    backend = factory(sample)
    request = InferRequest(prompt=sample.prompt, task=sample.task)
    streaming = isinstance(backend, StreamingBackend)
    recorder = _RecordingPolicy()
    probe_unavailable = False
    fallback_reason = ""
    gold = sample.cloud_better_from

    if strategy == "probe":
        if streaming and probe_path:
            policy = ProbePolicy(
                probe_path,
                threshold=None,
                feature_names=NUMERIC_HANDOFF_FEATURES,
            )
            probe_unavailable = not policy.available
            if probe_unavailable:
                fallback_reason = policy.unavailable_reason or "probe_unavailable"
            recorder = _RecordingPolicy(policy)
            generation = backend.generate_until(
                request, recorder, context={"task": sample.task}
            )
            probe_unavailable = probe_unavailable or not policy.available
        else:
            probe_unavailable = True
            fallback_reason = "probe_not_configured" if probe_path is None else "probe_unstreamed"
            generation = _generate_full(backend, request, recorder)
    elif strategy == "oracle":
        if streaming and gold is not None:
            policy = OraclePolicy()
            recorder = _RecordingPolicy(policy)
            generation = backend.generate_until(
                request,
                recorder,
                context={"task": sample.task, "gold_handoff_index": gold},
            )
        else:
            fallback_reason = "oracle_label_missing" if gold is None else "oracle_unstreamed"
            generation = _generate_full(backend, request, recorder)
    else:
        generation = _generate_full(backend, request, recorder)

    confidence = ConfidenceEstimator().estimate(
        generation.text,
        generation.token_logprobs,
        finish_reason=generation.finish_reason,
    )
    edge_tokens = generation.output_tokens or count_tokens(generation.text)
    prompt_tokens = count_tokens(sample.prompt)
    expected_tokens = count_tokens(sample.expected)

    handoff = generation.handoff
    handoff_taken = bool(handoff is not None and handoff.handed_off)

    escalate = handoff_taken
    if not escalate and strategy != "edge_only":
        escalate = confidence.score < escalation_threshold

    final_text = generation.text
    continuation = ""
    cloud_calls = 0
    prefill_tokens = 0
    decode_tokens = 0
    degraded = False
    mismatch = False

    if escalate:
        if sample.cloud_answer.strip():
            cloud_calls = 1
            if handoff_taken:
                partial = handoff.partial_trace if handoff is not None else generation.text
                final_text, continuation, mismatch = apply_cloud_answer(partial, sample.cloud_answer)
                prefill_tokens = prompt_tokens + count_tokens(partial)
                decode_tokens = count_tokens(continuation)
            else:
                final_text = sample.cloud_answer
                continuation = sample.cloud_answer
                prefill_tokens = prompt_tokens + edge_tokens
                decode_tokens = count_tokens(sample.cloud_answer)
        else:
            # A strategy wanted the cloud but the dataset has no answer: keep the
            # edge output and count the request as degraded, never crash.
            degraded = True
            fallback_reason = fallback_reason or "cloud_answer_missing"

    cost = cost_model.estimate(
        edge_tokens=edge_tokens,
        cloud_prefill_tokens=prefill_tokens,
        cloud_decode_tokens_expected=decode_tokens,
        cloud_calls=cloud_calls,
    )
    latency_ms = latency_model.estimate(
        edge_tokens=edge_tokens,
        cloud_prefill_tokens=prefill_tokens,
        cloud_decode_tokens=decode_tokens,
        cloud_calls=cloud_calls,
    )

    if strategy == "probe":
        score = _last_trust(recorder)
        if score is None:
            score = confidence.score
    elif strategy == "oracle":
        score = (gold / max(1, edge_tokens)) if gold is not None else 1.0
    else:
        score = confidence.score

    handoff_index = None
    boundary = None
    reason = None
    if handoff_taken and handoff is not None:
        handoff_index = handoff.handoff_index
        boundary = handoff.boundary
        reason = handoff.reason

    return {
        "sample_id": sample.sample_id,
        "strategy": strategy,
        "prompt": sample.prompt,
        "task": sample.task,
        "expected": sample.expected,
        "edge_text": generation.text,
        "final_text": final_text,
        "cloud_answer": sample.cloud_answer,
        "correct": _matches_expected(final_text, sample.expected),
        "exact_match": _matches_expected(final_text, sample.expected),
        "contains_expected": _contains_expected(final_text, sample.expected),
        "edge_correct": _matches_expected(generation.text, sample.expected),
        "cloud_reference_correct": bool(sample.cloud_answer.strip())
        and _matches_expected(sample.cloud_answer, sample.expected),
        "confidence": confidence.score,
        "score": _round(score),
        "escalated": bool(escalate),
        "handoff": handoff_taken,
        "handoff_index": handoff_index,
        "boundary": boundary,
        "handoff_reason": reason,
        "edge_tokens": int(edge_tokens),
        "prompt_tokens": int(prompt_tokens),
        "expected_tokens": int(expected_tokens),
        "cloud_prefill_tokens": int(prefill_tokens),
        "cloud_decode_tokens": int(decode_tokens),
        "cloud_calls": int(cloud_calls),
        "cost": _round(cost.estimated_cost),
        "latency_ms": _round(latency_ms),
        "degraded": bool(degraded),
        "probe_unavailable": bool(probe_unavailable),
        "continuation_mismatch": bool(mismatch),
        "fallback_reason": fallback_reason,
        "gold_handoff_index": gold,
    }


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _strategy_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    n = len(records)
    positions = sorted(
        int(record["handoff_index"])
        for record in records
        if record["handoff"] and record["handoff_index"] is not None
    )
    boundaries = {kind: 0 for kind in BOUNDARY_KINDS}
    for record in records:
        if record["handoff"] and record["boundary"] in boundaries:
            boundaries[str(record["boundary"])] += 1
    latencies = sorted(float(record["latency_ms"]) for record in records)
    return {
        "n": n,
        "accuracy": _round(_mean([1.0 if r["correct"] else 0.0 for r in records])),
        "exact_match_rate": _round(_mean([1.0 if r["exact_match"] else 0.0 for r in records])),
        "contains_expected_rate": _round(
            _mean([1.0 if r["contains_expected"] else 0.0 for r in records])
        ),
        "escalation_rate": _round(
            _mean([1.0 if r["escalated"] else 0.0 for r in records])
        ),
        "handoff_rate": _round(_mean([1.0 if r["handoff"] else 0.0 for r in records])),
        "handoff_position": {
            "count": len(positions),
            "mean": None if not positions else _round(_mean(positions)),
            "median": None if not positions else _round(_quantile(positions, 0.5)),
            "p95": None if not positions else _round(_quantile(positions, 0.95)),
        },
        "boundary_types": boundaries,
        "cloud_prefill_tokens": int(sum(int(r["cloud_prefill_tokens"]) for r in records)),
        "cloud_decode_tokens": int(sum(int(r["cloud_decode_tokens"]) for r in records)),
        "cloud_calls": int(sum(int(r["cloud_calls"]) for r in records)),
        "total_cost": _round(sum(float(r["cost"]) for r in records)),
        "cost_per_request": _round(_mean([float(r["cost"]) for r in records])),
        "latency_ms": {
            "mean": _round(_mean(latencies)),
            "p50": _round(_quantile(latencies, 0.5)),
            "p95": _round(_quantile(latencies, 0.95)),
        },
        "degraded_rate": _round(_mean([1.0 if r["degraded"] else 0.0 for r in records])),
        "probe_fallback_rate": _round(
            _mean([1.0 if r["probe_unavailable"] else 0.0 for r in records])
        ),
        "continuation_mismatch_rate": _round(
            _mean([1.0 if r["continuation_mismatch"] else 0.0 for r in records])
        ),
    }


def _risk_coverage_curve(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(records, key=lambda record: (-float(record["score"]), record["sample_id"]))
    points: list[dict[str, Any]] = []
    errors = 0
    for index, record in enumerate(ordered, start=1):
        errors += 0 if record["correct"] else 1
        points.append(
            {
                "coverage": _round(index / len(ordered)),
                "risk": _round(errors / index),
                "threshold": _round(float(record["score"])),
            }
        )
    return points


def _edge_only_cost(record: Mapping[str, Any], cost_model: CostModel) -> float:
    return cost_model.estimate(edge_tokens=int(record["edge_tokens"])).estimated_cost


def _full_cloud_cost(record: Mapping[str, Any], cost_model: CostModel) -> float:
    return cost_model.estimate(
        edge_tokens=int(record["edge_tokens"]),
        cloud_prefill_tokens=int(record["prompt_tokens"]) + int(record["edge_tokens"]),
        cloud_decode_tokens_expected=int(record["expected_tokens"]),
        cloud_calls=1,
    ).estimated_cost


def _cost_accuracy_curve(
    strategy: str,
    records: Sequence[Mapping[str, Any]],
    cost_model: CostModel,
) -> list[dict[str, Any]]:
    """Sweep the routing operating point and report average cost vs accuracy."""

    if not records:
        return []
    if strategy == "edge_only":
        return [
            {
                "threshold": None,
                "accuracy": _round(_mean([1.0 if r["correct"] else 0.0 for r in records])),
                "cost": _round(_mean([_edge_only_cost(r, cost_model) for r in records])),
            }
        ]

    points: list[dict[str, Any]] = []
    if strategy == "oracle":
        max_cut = max(int(record["edge_tokens"]) for record in records)
        step = max(1, math.ceil(max_cut / 10))
        cuts = sorted({0, *range(0, max_cut + 1, step), max_cut})
        for cut in cuts:
            total_cost = 0.0
            correct = 0
            for record in records:
                label = record["gold_handoff_index"]
                if label is not None and int(label) <= cut:
                    effective = min(cut, int(record["edge_tokens"]))
                    expected_tokens = int(record["expected_tokens"])
                    decode = max(0, expected_tokens - effective)
                    total_cost += cost_model.estimate(
                        edge_tokens=effective,
                        cloud_prefill_tokens=int(record["prompt_tokens"]) + effective,
                        cloud_decode_tokens_expected=decode,
                        cloud_calls=1,
                    ).estimated_cost
                    correct += 1 if record["cloud_reference_correct"] else 0
                else:
                    total_cost += _edge_only_cost(record, cost_model)
                    correct += 1 if record["edge_correct"] else 0
            points.append(
                {
                    "cut": cut,
                    "accuracy": _round(correct / len(records)),
                    "cost": _round(total_cost / len(records)),
                }
            )
        return points

    thresholds = sorted({0.0, *(float(record["score"]) for record in records), 1.0})
    for threshold in thresholds:
        total_cost = 0.0
        correct = 0
        for record in records:
            if float(record["score"]) < threshold:
                if record["cloud_calls"]:
                    # Reuse the measured cloud path (including a mid-stream cut).
                    total_cost += float(record["cost"])
                    correct += 1 if record["correct"] else 0
                else:
                    total_cost += _full_cloud_cost(record, cost_model)
                    correct += 1 if record["cloud_reference_correct"] else 0
            else:
                total_cost += _edge_only_cost(record, cost_model)
                correct += 1 if record["edge_correct"] else 0
        points.append(
            {
                "threshold": _round(threshold),
                "accuracy": _round(correct / len(records)),
                "cost": _round(total_cost / len(records)),
            }
        )
    return points


def _percentile_sorted(ordered: Sequence[float], q: float) -> float:
    if len(ordered) == 1:
        return ordered[0]
    position = q * (len(ordered) - 1)
    lower = max(0, min(len(ordered) - 1, int(math.floor(position))))
    upper = max(0, min(len(ordered) - 1, int(math.ceil(position))))
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def bootstrap_ci(
    values: Sequence[float], *, n_boot: int = 200, seed: int = 9, level: float = 0.95
) -> dict[str, Any]:
    """Percentile bootstrap interval for the mean, resampling with replacement."""

    data = [float(value) for value in values]
    if not data:
        return {"point": None, "low": None, "high": None, "samples": 0}
    point = _mean(data)
    if n_boot < 1:
        return {"point": _round(point), "low": None, "high": None, "samples": 0}
    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(n_boot):
        total = sum(data[rng.randrange(len(data))] for _ in range(len(data)))
        means.append(total / len(data))
    means.sort()
    alpha = (1.0 - level) / 2.0
    return {
        "point": _round(point),
        "low": _round(_percentile_sorted(means, alpha)),
        "high": _round(_percentile_sorted(means, 1.0 - alpha)),
        "samples": n_boot,
        "level": level,
    }


@dataclass(slots=True)
class EvaluationRun:
    """Results plus the per-prompt records that produced them."""

    results: dict[str, Any]
    per_prompt: list[dict[str, Any]]

    def write(self, out_dir: str | Path) -> dict[str, Path]:
        return write_outputs(self, out_dir)


def _dataset_digest(samples: Sequence[EvalSample]) -> str:
    payload = json.dumps(
        [sample.to_dict() for sample in samples],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def run_evaluation(
    samples: Sequence[EvalSample],
    *,
    strategies: Sequence[str] = STRATEGIES,
    seed: int = 9,
    n_boot: int = 200,
    probe_path: str | Path | None = None,
    escalation_threshold: float = 0.62,
    backend_kind: str = "synthetic",
    backend_factory: Callable[[EvalSample], Backend] | None = None,
    cost_model: CostModel | None = None,
    latency_model: LatencyModel | None = None,
    dataset_source: str = "synthetic",
) -> EvaluationRun:
    """Run every strategy over the same prompt set and return reproducible artifacts."""

    strategy_list = list(strategies)
    unknown = sorted(set(strategy_list) - _STRATEGY_SET)
    if unknown:
        raise ValueError(f"未知策略: {', '.join(unknown)}")
    if not 0.0 <= float(escalation_threshold) <= 1.0:
        raise ValueError("escalation_threshold 必须在 [0, 1] 内")

    factory = backend_factory or _mock_backend_factory
    cost = cost_model or DEFAULT_COST_MODEL
    latency = latency_model or LatencyModel()
    probe = None if probe_path is None else str(probe_path)

    all_records: list[dict[str, Any]] = []
    per_strategy: dict[str, list[dict[str, Any]]] = {}
    for strategy in strategy_list:
        records = [
            _evaluate_sample(
                strategy,
                sample,
                factory=factory,
                probe_path=probe,
                escalation_threshold=float(escalation_threshold),
                cost_model=cost,
                latency_model=latency,
            )
            for sample in samples
        ]
        per_strategy[strategy] = records
        all_records.extend(records)

    strategy_metrics = {
        strategy: _strategy_metrics(per_strategy[strategy]) for strategy in strategy_list
    }
    curves = {
        strategy: {
            "risk_coverage": _risk_coverage_curve(per_strategy[strategy]),
            "cost_accuracy": _cost_accuracy_curve(
                strategy, per_strategy[strategy], cost
            ),
        }
        for strategy in strategy_list
    }
    ci = {
        strategy: {
            "accuracy": bootstrap_ci(
                [1.0 if record["correct"] else 0.0 for record in per_strategy[strategy]],
                n_boot=n_boot,
                seed=seed,
            ),
            "cost": bootstrap_ci(
                [float(record["cost"]) for record in per_strategy[strategy]],
                n_boot=n_boot,
                seed=seed,
            ),
        }
        for strategy in strategy_list
    }

    sample_backend = factory(samples[0]) if samples else None
    backend_info = {
        "kind": backend_kind,
        "edge_model": str(getattr(sample_backend, "model_id", "")),
        "cloud_model": "scripted-dataset",
        "streaming": isinstance(sample_backend, StreamingBackend),
    }
    results: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "backend": backend_info,
        "seed": int(seed),
        "n": len(samples),
        "bootstrap_samples": int(n_boot),
        "dataset": {"source": dataset_source, "digest": _dataset_digest(samples)},
        "strategies": strategy_metrics,
        "curves": curves,
        "ci": ci,
        "config": {
            "escalation_threshold": float(escalation_threshold),
            "probe_path": probe,
            "cost_model": asdict(cost),
            "latency_model": asdict(latency),
            "strategies": strategy_list,
        },
    }
    return EvaluationRun(results=results, per_prompt=all_records)


_STRATEGY_COLUMNS: tuple[str, ...] = (
    "strategy",
    "n",
    "accuracy",
    "exact_match_rate",
    "contains_expected_rate",
    "escalation_rate",
    "handoff_rate",
    "handoff_mean_index",
    "handoff_median_index",
    "handoff_p95_index",
    "cloud_prefill_tokens",
    "cloud_decode_tokens",
    "cloud_calls",
    "total_cost",
    "cost_per_request",
    "latency_p50_ms",
    "latency_p95_ms",
    "degraded_rate",
    "probe_fallback_rate",
    "continuation_mismatch_rate",
)

_CURVE_COLUMNS: tuple[str, ...] = (
    "strategy",
    "curve",
    "x_name",
    "x",
    "accuracy",
    "risk",
    "cost",
)


def _format_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _summary_rows(results: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for strategy, metrics in results["strategies"].items():
        position = metrics["handoff_position"]
        rows.append(
            {
                "strategy": strategy,
                "n": metrics["n"],
                "accuracy": metrics["accuracy"],
                "exact_match_rate": metrics["exact_match_rate"],
                "contains_expected_rate": metrics["contains_expected_rate"],
                "escalation_rate": metrics["escalation_rate"],
                "handoff_rate": metrics["handoff_rate"],
                "handoff_mean_index": position["mean"],
                "handoff_median_index": position["median"],
                "handoff_p95_index": position["p95"],
                "cloud_prefill_tokens": metrics["cloud_prefill_tokens"],
                "cloud_decode_tokens": metrics["cloud_decode_tokens"],
                "cloud_calls": metrics["cloud_calls"],
                "total_cost": metrics["total_cost"],
                "cost_per_request": metrics["cost_per_request"],
                "latency_p50_ms": metrics["latency_ms"]["p50"],
                "latency_p95_ms": metrics["latency_ms"]["p95"],
                "degraded_rate": metrics["degraded_rate"],
                "probe_fallback_rate": metrics["probe_fallback_rate"],
                "continuation_mismatch_rate": metrics["continuation_mismatch_rate"],
            }
        )
    return rows


def _curve_rows(results: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Flatten both curve kinds into one long-form CSV table."""

    rows: list[dict[str, Any]] = []
    for strategy, curves in results["curves"].items():
        for point in curves["risk_coverage"]:
            rows.append(
                {
                    "strategy": strategy,
                    "curve": "risk_coverage",
                    "x_name": "threshold",
                    "x": point["threshold"],
                    "accuracy": None,
                    "risk": point["risk"],
                    "cost": None,
                }
            )
        for point in curves["cost_accuracy"]:
            x_name = "cut" if "cut" in point else "threshold"
            rows.append(
                {
                    "strategy": strategy,
                    "curve": "cost_accuracy",
                    "x_name": x_name,
                    "x": point[x_name],
                    "accuracy": point["accuracy"],
                    "risk": None,
                    "cost": point["cost"],
                }
            )
    return rows


def write_outputs(run: EvaluationRun, out_dir: str | Path) -> dict[str, Path]:
    """Write ``results.json``, ``summary.csv``, ``curves.csv`` and ``per_prompt.jsonl``."""

    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    results_path = target / "results.json"
    summary_path = target / "summary.csv"
    curves_path = target / "curves.csv"
    per_prompt_path = target / "per_prompt.jsonl"

    results_path.write_text(
        json.dumps(run.results, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(_STRATEGY_COLUMNS), lineterminator="\n"
        )
        writer.writeheader()
        for row in _summary_rows(run.results):
            writer.writerow({key: _format_cell(row[key]) for key in _STRATEGY_COLUMNS})
    with curves_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(_CURVE_COLUMNS), lineterminator="\n")
        writer.writeheader()
        for row in _curve_rows(run.results):
            writer.writerow({key: _format_cell(row[key]) for key in _CURVE_COLUMNS})
    with per_prompt_path.open("w", encoding="utf-8") as handle:
        for record in run.per_prompt:
            handle.write(
                json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
                + "\n"
            )
    return {
        "results": results_path,
        "summary": summary_path,
        "curves": curves_path,
        "per_prompt": per_prompt_path,
    }


_RESULTS_REQUIRED = frozenset({"schema_version", "backend", "seed", "n", "strategies", "curves", "ci"})
_METRIC_REQUIRED = frozenset(
    {
        "n",
        "accuracy",
        "exact_match_rate",
        "contains_expected_rate",
        "escalation_rate",
        "handoff_rate",
        "handoff_position",
        "boundary_types",
        "cloud_prefill_tokens",
        "cloud_decode_tokens",
        "total_cost",
        "latency_ms",
        "degraded_rate",
        "probe_fallback_rate",
        "continuation_mismatch_rate",
    }
)


def load_results(path: str | Path) -> dict[str, Any]:
    """Read and validate an evaluation artifact from any backend kind.

    Mock and real-backend runs share the same schema; the reader only checks the
    contract, so a real run can be compared with a smoke run without special
    cases.
    """

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError("results.json 顶层必须是对象")
    missing = sorted(_RESULTS_REQUIRED - set(data))
    if missing:
        raise ValueError(f"results.json 缺少字段: {', '.join(missing)}")
    if data["schema_version"] != SCHEMA_VERSION:
        raise ValueError(
            f"results schema {data['schema_version']!r} 不受支持, 期望 {SCHEMA_VERSION!r}"
        )
    strategies = data["strategies"]
    curves = data["curves"]
    ci = data["ci"]
    if not isinstance(strategies, Mapping) or not strategies:
        raise ValueError("results.strategies 必须是非空对象")
    for name in strategies:
        metrics = strategies[name]
        if not isinstance(metrics, Mapping):
            raise ValueError(f"results.strategies.{name} 必须是对象")
        missing_metrics = sorted(_METRIC_REQUIRED - set(metrics))
        if missing_metrics:
            raise ValueError(
                f"results.strategies.{name} 缺少指标: {', '.join(missing_metrics)}"
            )
        if name not in curves or name not in ci:
            raise ValueError(f"results 缺少策略 {name!r} 的 curves/ci")
        if not {"risk_coverage", "cost_accuracy"} <= set(curves[name]):
            raise ValueError(f"results.curves.{name} 缺少曲线")
        if not {"accuracy", "cost"} <= set(ci[name]):
            raise ValueError(f"results.ci.{name} 缺少置信区间")
    return dict(data)


def export_candidates(
    samples: Sequence[EvalSample],
    *,
    backend_factory: Callable[[EvalSample], Backend] | None = None,
) -> list[dict[str, Any]]:
    """Export one probe training row per observed prefix.

    Labels follow the ticket contract: a prefix whose token index is at or past
    ``cloud_better_from`` should hand off (1), earlier prefixes stay on the edge
    (0). Features are the numeric ``HandoffFeatures`` fields, the same names a
    ``ProbeModel`` scores at serving time.
    """

    factory = backend_factory or _mock_backend_factory
    rows: list[dict[str, Any]] = []
    for sample in samples:
        backend = factory(sample)
        if not isinstance(backend, StreamingBackend):
            raise ValueError("导出候选需要支持 generate_until 的流式端侧后端")
        recorder = _RecordingPolicy()
        backend.generate_until(
            InferRequest(prompt=sample.prompt, task=sample.task),
            recorder,
            context={"task": sample.task},
        )
        gold = sample.cloud_better_from
        for step in recorder.steps:
            features = step.features.to_dict()
            label = 1 if gold is not None and int(features["prefix_tokens"]) >= gold else 0
            rows.append(
                {
                    "prompt_id": sample.sample_id,
                    "features": {name: features[name] for name in NUMERIC_HANDOFF_FEATURES},
                    "label": label,
                }
            )
    return rows


def write_candidates(rows: Sequence[Mapping[str, Any]], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
                + "\n"
            )
    return target


def load_candidates(path: str | Path) -> list[dict[str, Any]]:
    """Read probe candidate rows written by ``--export-candidates``."""

    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"候选第 {line_number} 行不是合法 JSON: {exc}") from exc
            if not isinstance(row, Mapping):
                raise ValueError(f"候选第 {line_number} 行必须是 JSON 对象")
            features = row.get("features")
            label = row.get("label")
            if not isinstance(features, Mapping) or not features:
                raise ValueError(f"候选第 {line_number} 行缺少 features 对象")
            if isinstance(label, bool):
                label = int(label)
            if not isinstance(label, int) or label not in (0, 1):
                raise ValueError(f"候选第 {line_number} 行 label 必须是 0 或 1")
            rows.append(
                {
                    "prompt_id": str(row.get("prompt_id", "")),
                    "features": dict(features),
                    "label": label,
                }
            )
    return rows


def feature_matrix(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[int], list[str]]:
    """Split candidate rows into ``fit_probe`` inputs and a stable feature order."""

    features: list[dict[str, Any]] = []
    labels: list[int] = []
    names: set[str] = set()
    for row in rows:
        row_features = dict(row["features"])
        names.update(row_features)
        features.append(row_features)
        labels.append(int(row["label"]))
    return features, labels, sorted(names)


def fit_candidates(
    rows: Sequence[Mapping[str, Any]],
    *,
    threshold: float = 0.5,
    trained_at: str | None = None,
    calibration_bins: int = 10,
) -> dict[str, Any]:
    """Fit a probe artifact from candidate rows (shared by tests and the CLI)."""

    features, labels, names = feature_matrix(rows)
    if not names:
        raise ValueError("候选行没有可用特征")
    return fit_probe(
        features,
        labels,
        version=PROBE_VERSION,
        threshold=threshold,
        feature_names=names,
        calibration_bins=calibration_bins,
        trained_at=trained_at,
    )


def probe_feature_names() -> tuple[str, ...]:
    """The numeric feature names a fitted probe is expected to reference."""

    available = set(HandoffFeatures.__dataclass_fields__)
    return tuple(name for name in NUMERIC_HANDOFF_FEATURES if name in available)
