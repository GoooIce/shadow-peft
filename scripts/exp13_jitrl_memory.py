#!/usr/bin/env python3
"""exp13: JitRL-lite memory reranking evaluation (issue #40).

Spec: ``docs/specs/jitrl-memory-hotfix.md`` §Implementation Decisions 6 and
Further Notes「建议的首轮验收口径」.

The evaluation is fully synthetic and CPU-only: a deterministic paraphrase set
is generated from ``edge_cloud.knowledge.ENTRIES``, corrected feedback memory is
built for half the lines, and a scripted n-best edge backend plus a fake cloud
drive the real ``Router`` twice — once with ``memory_mode=off`` and once with
``memory_mode=on``.  No model is loaded and no network call is made.

Example::

    PYTHONPATH=src python scripts/exp13_jitrl_memory.py run \
        --out runs-local/exp13 --seed 13

Artifacts: ``report.json``, ``dataset.jsonl``, ``memory.jsonl``,
``outcomes_no_memory.jsonl``, ``outcomes_rerank.jsonl`` and ``summary.md``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import sys
import time
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from edge_cloud import knowledge
from edge_cloud.backends import Generation
from edge_cloud.config import EdgeConfig
from edge_cloud.memory import ExperienceMemory, MemoryEntry, MemoryHit
from edge_cloud.metrics import Metrics
from edge_cloud.protocol import CloudRequest, CloudResponse, InferRequest
from edge_cloud.rerank import Candidate, normalize_answer, rerank
from edge_cloud.router import Router

EXPERIMENT = "exp13-jitrl-memory"
SPEC = "docs/specs/jitrl-memory-hotfix.md §Implementation Decisions 6"
DEFAULT_SEED = 13
DEFAULT_PARAPHRASES = 6
MEMORY_TENANT = "line-a"
NO_MEMORY_TENANT = "line-b"
ADAPTER_VERSION = "base"
MEMORY_RECORDS_PER_CASE = 3
DEFAULT_ALPHA = 0.5
DEFAULT_THRESHOLD = 0.45
DEFAULT_TOP_K = 4
DEFAULT_MIN_SUPPORT = 2
DEFAULT_LATENCY_BUDGET_MS = 50.0
DEFAULT_ESCALATION_THRESHOLD = 0.62

# Per-token probabilities converted to logprobs: 0.64 clears the confidence
# threshold, and 0.62 leaves a reranked answer just above it.  The deliberately
# low-confidence cases carry ``finish_reason="length"`` instead of a lower
# logprob, so the n-best top-1 stays the edge answer (matching ``generate``)
# while the confidence estimator still escalates it.
CONFIDENT_LOGPROB = math.log(0.64)
CLOUD_CANDIDATE_LOGPROB = math.log(0.62)
WRONG_CANDIDATE_LOGPROB = math.log(0.40)
GENERIC_CANDIDATE_LOGPROB = math.log(0.35)
TOKENS_PER_ANSWER = 24


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GoldSpec:
    """Task-specific ground truth for one knowledge entry."""

    kind: str  # "verdict" | "number" | "facts"
    value: Any = None
    facts: tuple[str, ...] = ()


GOLD_SPECS: dict[str, GoldSpec] = {
    "reflow-profile": GoldSpec("facts", facts=("235", "245", "60", "90")),
    "fault-e104": GoldSpec("facts", facts=("真空", "过滤棉")),
    "solder-void": GoldSpec("verdict", value="rework"),
    "ipc-class3": GoldSpec("facts", facts=("10%", "x-ray")),
    "dppm-calc": GoldSpec("number", value=(30000.0, "")),
    "fault-e217": GoldSpec("facts", facts=("冷却液", "流量")),
    "oee": GoldSpec("facts", facts=("smed", "可用率")),
    "estop-reset": GoldSpec("facts", facts=("复位", "3 秒")),
}

WRONG_ANSWERS: dict[str, str] = {
    "reflow-profile": "回流焊温度越高越好，直接设定 300℃ 即可，不需要看曲线。",
    "fault-e104": "E-104 表示贴片头真空压力过高，直接更换整个贴片头即可。",
    "solder-void": "判定：合格。空洞不影响焊接质量，可以直接放行。",
    "ipc-class3": "Class 3 与 Class 2 要求一致，虚焊返修后即可放行。",
    "dppm-calc": "约为 3000 DPPM。",
    "fault-e217": "E-217 重启设备即可恢复，无需检查冷却系统。",
    "oee": "提升 OEE 只需增加加班时间，不需要分析损失结构。",
    "estop-reset": "急停按下后可直接重新启动，无需其他确认。",
}

EASY_ENTRY_KEYS = frozenset(
    entry.key for entry in knowledge.ENTRIES if entry.difficulty == "easy"
)
VERDICT_ENTRY_KEYS = frozenset(
    entry.key for entry in knowledge.ENTRIES if entry.task == "quality_verdict"
)

_QUERY_TEMPLATES = (
    "{k0} 报警是什么原因？应该怎么处理？",
    "产线出现 {k0}，{k2} 异常，排查步骤是什么？",
    "设备报 {k0}，和 {k2} 有关吗？",
    "请说明 {k0} 的排查顺序和 {k3} 检查要点。",
    "现场 {k2} 有问题（{k0}），先做什么？",
    "关于 {k0}，{k3} 需要怎么维护？",
    "{k0} 怎么解决？{k2} 要检查哪些项目？",
    "{k0} 故障处理：{k2}、{k3} 的检查步骤。",
)

_SYNONYMS = {
    "处理": "处置",
    "排查": "检查",
    "检查": "确认",
    "原因": "缘由",
    "应该": "需要",
    "怎么": "如何",
    "什么": "哪些",
}


def _fill_template(template: str, entry: knowledge.KBEntry) -> str:
    keys = entry.keywords
    k3 = keys[3] if len(keys) > 3 else keys[-1]
    return template.format(k0=keys[0], k1=keys[1], k2=keys[2], k3=k3)


def _synonym_rewrite(text: str) -> str:
    rewritten = text
    for source, replacement in _SYNONYMS.items():
        rewritten = rewritten.replace(source, replacement)
    return rewritten


@dataclass(frozen=True, slots=True)
class EvalCase:
    case_id: str
    entry_key: str
    task: str
    prompt: str
    tenant: str
    has_memory: bool
    edge_answer: str
    cloud_answer: str
    wrong_answer: str
    gold: GoldSpec
    edge_low_confidence: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "entry_key": self.entry_key,
            "task": self.task,
            "prompt": self.prompt,
            "tenant": self.tenant,
            "has_memory": self.has_memory,
            "edge_answer": self.edge_answer,
            "cloud_answer": self.cloud_answer,
            "wrong_answer": self.wrong_answer,
            "gold": {
                "kind": self.gold.kind,
                "value": self.gold.value,
                "facts": list(self.gold.facts),
            },
            "edge_low_confidence": self.edge_low_confidence,
        }


def build_cases(
    *, seed: int = DEFAULT_SEED, paraphrases_per_entry: int = DEFAULT_PARAPHRASES
) -> list[EvalCase]:
    """Build the deterministic paraphrase set.

    Every knowledge entry contributes ``paraphrases_per_entry`` prompts (4–8).
    The first half of each entry's prompts belong to ``line-a``, which has
    corrected historical memory; the second half belongs to ``line-b``, which
    has none.  Easy entries (except the verdict task, where memory cannot
    change the normalized answer) make one prompt per line deliberately
    low-confidence to exercise cloud escalation.
    """

    if not 4 <= paraphrases_per_entry <= 8:
        raise ValueError("paraphrases_per_entry 必须在 [4, 8] 内")
    if paraphrases_per_entry % 2 != 0:
        raise ValueError("paraphrases_per_entry 必须是偶数，才能按记忆/无记忆对半切分")

    half = paraphrases_per_entry // 2
    cases: list[EvalCase] = []
    for entry in knowledge.ENTRIES:
        rng = random.Random(f"exp13:{seed}:{entry.key}")
        templates = list(_QUERY_TEMPLATES)
        rng.shuffle(templates)
        for local_index, template in enumerate(templates[:paraphrases_per_entry]):
            has_memory = local_index < half
            group_index = local_index if has_memory else local_index - half
            tenant = MEMORY_TENANT if has_memory else NO_MEMORY_TENANT
            low_confidence = (
                entry.key in EASY_ENTRY_KEYS
                and entry.key not in VERDICT_ENTRY_KEYS
                and group_index == 1
            )
            cases.append(
                EvalCase(
                    case_id=f"{entry.key}/{tenant}/{local_index}",
                    entry_key=entry.key,
                    task=entry.task,
                    prompt=_fill_template(template, entry),
                    tenant=tenant,
                    has_memory=has_memory,
                    edge_answer=entry.edge_answer,
                    cloud_answer=entry.cloud_answer,
                    wrong_answer=WRONG_ANSWERS[entry.key],
                    gold=GOLD_SPECS[entry.key],
                    edge_low_confidence=low_confidence,
                )
            )
    return cases


def _sha256(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def cases_fingerprint(cases: Sequence[EvalCase]) -> str:
    """Content hash of the evaluation cases (independent of wall-clock time)."""

    return _sha256([case.to_dict() for case in cases])


def memory_fingerprint(cases: Sequence[EvalCase]) -> str:
    """Content hash of the historical records derived from the memory cases."""

    return _sha256(_memory_records(cases))


def _memory_records(cases: Sequence[EvalCase]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for case in cases:
        if not case.has_memory:
            continue
        prompts = (
            case.prompt,
            f"再次出现：{case.prompt}",
            _synonym_rewrite(case.prompt),
        )
        for index, prompt in enumerate(prompts[:MEMORY_RECORDS_PER_CASE]):
            records.append(
                {
                    "entry_id": f"mem-{case.case_id.replace('/', '-')}-{index}",
                    "prompt": prompt,
                    "task": case.task,
                    "candidates": [case.cloud_answer],
                    "return_value": 1.0,
                    "accepted": True,
                    "corrected": True,
                    "adapter_version": ADAPTER_VERSION,
                    "tenant": case.tenant,
                    "ttl_s": 30 * 24 * 60 * 60,
                }
            )
    return records


def build_memory(
    cases: Sequence[EvalCase], *, now: float | None = None
) -> ExperienceMemory:
    """Build the corrected-experience memory for ``line-a`` cases.

    Each memory case gets three corrected human records (the same question, a
    repeat prefix and a synonym rewrite), all returning +1 for the cloud answer.
    ``line-b`` has no records at all.
    """

    created_at = time.time() if now is None else float(now)
    memory = ExperienceMemory(default_ttl_s=30 * 24 * 60 * 60)
    for record in _memory_records(cases):
        memory.add(
            MemoryEntry(
                entry_id=record["entry_id"],
                prompt=record["prompt"],
                task=record["task"],
                candidates=list(record["candidates"]),
                return_value=float(record["return_value"]),
                accepted=bool(record["accepted"]),
                corrected=bool(record["corrected"]),
                adapter_version=str(record["adapter_version"]),
                tenant=record["tenant"],
                created_at=created_at,
                ttl_s=int(record["ttl_s"]),
            )
        )
    return memory


def score_answer(case: EvalCase, text: str) -> bool:
    """Task-specific structured correctness for one answer."""

    if not isinstance(text, str) or not text.strip():
        return False
    gold = case.gold
    if gold.kind == "verdict":
        return normalize_answer(text, case.task) == gold.value
    if gold.kind == "number":
        parsed = normalize_answer(text, case.task)
        if not isinstance(parsed, tuple) or not isinstance(gold.value, tuple):
            return False
        value, unit = parsed
        gold_value, gold_unit = gold.value
        if unit and gold_unit and unit != gold_unit:
            return False
        return math.isclose(value, gold_value, rel_tol=1e-2, abs_tol=1e-9)
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return all(fact.casefold() in normalized for fact in gold.facts)


# ---------------------------------------------------------------------------
# Synthetic edge backend and cloud
# ---------------------------------------------------------------------------


class SyntheticNBestBackend:
    """Deterministic n-best edge model with one confident wrong answer.

    The first candidate is the knowledge-base edge answer (the baseline
    ``generate`` answer), the second is the cloud answer (the memory target),
    the third is a plausible wrong answer and the fourth is the generic default.
    Logprobs are chosen so the edge answer stays the n-best top-1; cases marked
    low-confidence are truncated (``finish_reason="length"``) so the confidence
    estimator escalates the baseline while a complete reranked answer does not.
    """

    model_id = "synthetic-nbest"

    def __init__(self, cases: Sequence[EvalCase]) -> None:
        self._by_prompt = {case.prompt: case for case in cases}
        if len(self._by_prompt) != len(cases):
            raise ValueError("prompts must be unique")
        self.generate_calls = 0
        self.nbest_calls: list[int] = []

    def _case(self, request: InferRequest) -> EvalCase:
        try:
            return self._by_prompt[request.prompt]
        except KeyError as exc:  # pragma: no cover - the eval only feeds known cases
            raise KeyError(f"unknown evaluation prompt: {request.prompt!r}") from exc

    def _generation(
        self, text: str, logprob: float, *, finish_reason: str = "stop"
    ) -> Generation:
        return Generation(
            text=text,
            token_logprobs=[logprob] * TOKENS_PER_ANSWER,
            model_id=self.model_id,
            finish_reason=finish_reason,
            latency_ms=0.0,
            output_tokens=TOKENS_PER_ANSWER,
        )

    def _candidates(self, case: EvalCase) -> list[tuple[str, float, str]]:
        edge_finish = "length" if case.edge_low_confidence else "stop"
        return [
            (case.edge_answer, CONFIDENT_LOGPROB, edge_finish),
            (case.cloud_answer, CLOUD_CANDIDATE_LOGPROB, "stop"),
            (case.wrong_answer, WRONG_CANDIDATE_LOGPROB, "stop"),
            (knowledge.DEFAULT_ENTRY.cloud_answer, GENERIC_CANDIDATE_LOGPROB, "stop"),
        ]

    def generate(self, request: InferRequest, *, draft: str | None = None) -> Generation:
        del draft
        self.generate_calls += 1
        case = self._case(request)
        return self._generation(
            case.edge_answer,
            CONFIDENT_LOGPROB,
            finish_reason="length" if case.edge_low_confidence else "stop",
        )

    def generate_nbest(self, request: InferRequest, n: int = 4) -> list[Generation]:
        self.nbest_calls.append(n)
        return [
            self._generation(text, logprob, finish_reason=finish_reason)
            for text, logprob, finish_reason in self._candidates(self._case(request))[:n]
        ]


class SyntheticCloud:
    """Fake cloud that answers every escalated prompt with the cloud answer."""

    model_id = "synthetic-cloud"

    def __init__(self, cases: Sequence[EvalCase]) -> None:
        self._answers = {case.prompt: case.cloud_answer for case in cases}
        self.calls: list[CloudRequest] = []

    def enhance(
        self, request: CloudRequest, *, timeout_ms: int | None = None
    ) -> CloudResponse:
        del timeout_ms
        self.calls.append(request)
        return CloudResponse(
            request_id=request.request_id,
            text=self._answers.get(request.prompt, knowledge.DEFAULT_ENTRY.cloud_answer),
            confidence=0.96,
            model_id=self.model_id,
            latency_ms=8.0,
            confidence_signals={"logprob_geomean": 0.96},
            output_tokens=32,
        )


# ---------------------------------------------------------------------------
# Evaluation arms
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CaseOutcome:
    case_id: str
    entry_key: str
    prompt: str
    tenant: str
    has_memory: bool
    source: str
    answer: str
    correct: bool
    memory_hit: bool
    memory_reranked: bool
    cloud_called: bool
    memory_entry_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "entry_key": self.entry_key,
            "prompt": self.prompt,
            "tenant": self.tenant,
            "has_memory": self.has_memory,
            "source": self.source,
            "answer": self.answer,
            "correct": self.correct,
            "memory_hit": self.memory_hit,
            "memory_reranked": self.memory_reranked,
            "cloud_called": self.cloud_called,
            "memory_entry_ids": list(self.memory_entry_ids),
        }


@dataclass(slots=True)
class ArmResult:
    name: str
    memory_mode: str
    outcomes: list[CaseOutcome]
    cloud_calls: int
    latency_ms: dict[str, float] | None
    metrics: dict[str, Any]


def _edge_config(
    *,
    memory_mode: str,
    tenant: str,
    alpha: float,
    threshold: float,
    top_k: int,
    min_support: int,
    latency_budget_ms: float,
) -> EdgeConfig:
    return EdgeConfig(
        backend={"kind": "synthetic_nbest"},
        cloud_url="http://127.0.0.1:1",
        outbox_path="",
        feedback_path="",
        enable_outbox_flusher=False,
        escalation_threshold=DEFAULT_ESCALATION_THRESHOLD,
        # Evaluation override: no task is forced to the cloud so every task can
        # exercise the memory seam.  The shipped default stays
        # ``["quality_verdict"]``.
        require_cloud_tasks=[],
        adapter_version=ADAPTER_VERSION,
        memory_mode=memory_mode,
        memory_tenant=tenant,
        memory_alpha=alpha,
        memory_threshold=threshold,
        memory_top_k=top_k,
        memory_min_support=min_support,
        memory_latency_budget_ms=latency_budget_ms,
    )


def latency_summary(metrics_snapshot: dict[str, Any]) -> dict[str, float] | None:
    """Extract the router's ``memory_latency_ms`` P50/P95 summary."""

    summary = metrics_snapshot.get("latency_ms", {}).get("memory_latency_ms")
    if not summary:
        return None
    return {
        key: round(float(summary[key]), 3)
        for key in ("count", "avg", "p50", "p95", "max")
    }


def run_arm(
    cases: Sequence[EvalCase],
    memory: ExperienceMemory,
    *,
    memory_mode: str,
    alpha: float = DEFAULT_ALPHA,
    threshold: float = DEFAULT_THRESHOLD,
    top_k: int = DEFAULT_TOP_K,
    min_support: int = DEFAULT_MIN_SUPPORT,
    latency_budget_ms: float = DEFAULT_LATENCY_BUDGET_MS,
) -> ArmResult:
    """Run every case through ``Router.infer`` for one memory mode."""

    backend = SyntheticNBestBackend(cases)
    cloud = SyntheticCloud(cases)
    metrics = Metrics()
    routers = {
        tenant: Router(
            backend,
            _edge_config(
                memory_mode=memory_mode,
                tenant=tenant,
                alpha=alpha,
                threshold=threshold,
                top_k=top_k,
                min_support=min_support,
                latency_budget_ms=latency_budget_ms,
            ),
            cloud_client=cloud,
            memory=memory,
            metrics=metrics,
        )
        for tenant in (MEMORY_TENANT, NO_MEMORY_TENANT)
    }

    outcomes: list[CaseOutcome] = []
    for case in cases:
        calls_before = len(cloud.calls)
        response = routers[case.tenant].infer(
            InferRequest(prompt=case.prompt, task=case.task, deadline_ms=5000)
        )
        outcomes.append(
            CaseOutcome(
                case_id=case.case_id,
                entry_key=case.entry_key,
                prompt=case.prompt,
                tenant=case.tenant,
                has_memory=case.has_memory,
                source=response.source,
                answer=response.text,
                correct=score_answer(case, response.text),
                memory_hit=bool(response.memory_hit),
                memory_reranked=bool(response.memory_reranked),
                cloud_called=len(cloud.calls) > calls_before,
                memory_entry_ids=tuple(response.memory_entry_ids),
            )
        )

    snapshot = metrics.snapshot()
    return ArmResult(
        name="rerank" if memory_mode == "on" else "no_memory",
        memory_mode=memory_mode,
        outcomes=outcomes,
        cloud_calls=len(cloud.calls),
        latency_ms=latency_summary(snapshot),
        metrics=snapshot,
    )


def _arm_payload(arm: ArmResult) -> dict[str, Any]:
    total = len(arm.outcomes)
    correct = sum(1 for outcome in arm.outcomes if outcome.correct)
    hit_correct = sum(
        1 for outcome in arm.outcomes if outcome.memory_hit and outcome.correct
    )
    hit_wrong = sum(
        1 for outcome in arm.outcomes if outcome.memory_hit and not outcome.correct
    )
    missed = sum(1 for outcome in arm.outcomes if not outcome.memory_hit)
    return {
        "name": arm.name,
        "memory_mode": arm.memory_mode,
        "total": total,
        "correct": correct,
        "accuracy": round(correct / total, 6) if total else 0.0,
        "accuracy_pct": round(100.0 * correct / total, 2) if total else 0.0,
        "cloud_calls": arm.cloud_calls,
        "memory_hit_cases": hit_correct + hit_wrong,
        "memory_hit_rate": round((hit_correct + hit_wrong) / total, 6) if total else 0.0,
        "hit_correct": hit_correct,
        "hit_wrong": hit_wrong,
        "missed": missed,
        "reranked_cases": sum(1 for outcome in arm.outcomes if outcome.memory_reranked),
        "latency_ms": arm.latency_ms,
        "cases": [outcome.to_dict() for outcome in arm.outcomes],
    }


def _comparison(
    no_memory: ArmResult, rerank: ArmResult
) -> dict[str, Any]:
    baseline = {outcome.case_id: outcome for outcome in no_memory.outcomes}
    false_bias_cases = [
        outcome.case_id
        for outcome in rerank.outcomes
        if baseline[outcome.case_id].correct and not outcome.correct
    ]
    baseline_correct = sum(1 for outcome in no_memory.outcomes if outcome.correct)
    total = len(no_memory.outcomes)
    no_memory_correct = baseline_correct
    rerank_correct = sum(1 for outcome in rerank.outcomes if outcome.correct)
    latency = rerank.latency_ms or {}
    return {
        "accuracy_gain_pp": round(
            100.0 * (rerank_correct - no_memory_correct) / total, 4
        )
        if total
        else 0.0,
        "false_bias_count": len(false_bias_cases),
        "false_bias_rate": round(len(false_bias_cases) / baseline_correct, 6)
        if baseline_correct
        else 0.0,
        "false_bias_denominator": baseline_correct,
        "false_bias_cases": false_bias_cases,
        "memory_hit_rate": round(
            sum(1 for outcome in rerank.outcomes if outcome.memory_hit) / total, 6
        )
        if total
        else 0.0,
        "latency_p50_ms": latency.get("p50"),
        "latency_p95_ms": latency.get("p95"),
        "cloud_calls_no_memory": no_memory.cloud_calls,
        "cloud_calls_rerank": rerank.cloud_calls,
        "cloud_calls_delta": rerank.cloud_calls - no_memory.cloud_calls,
    }


GUARD_BASELINE = "先确认危险源已排除，再按急停复位按钮并保持 3 秒。"
GUARD_BAD = "急停按下后可直接重新启动，无需复位确认。"
GUARD_UNRELATED = "先更换主轴冷却液并检查流量。"


def _probe_hit(entry_id: str, answer: str, return_value: float) -> MemoryHit:
    entry = MemoryEntry(
        entry_id=entry_id,
        prompt="急停如何复位",
        task="safety",
        candidates=[answer],
        return_value=return_value,
        accepted=return_value > 0,
        corrected=return_value >= 1.0,
        adapter_version=ADAPTER_VERSION,
        tenant=NO_MEMORY_TENANT,
        created_at=0.0,
    )
    return MemoryHit(entry=entry, score=0.9)


def run_guard_probes(
    *,
    alpha: float = DEFAULT_ALPHA,
    min_support: int = DEFAULT_MIN_SUPPORT,
) -> dict[str, Any]:
    """Exercise the false-bias guard with deliberately misleading memories.

    Every probe starts from a correct baseline answer.  Probes 1, 2 and 4 feed
    memory that points at a wrong or unsupported answer; the guard must keep
    the baseline, so the measured false-bias stays zero.  Probe 3 is the
    positive control (memory supports the baseline).
    """

    candidates = [
        Candidate(text=GUARD_BASELINE, logprob=-0.10),
        Candidate(text=GUARD_BAD, logprob=-0.30),
    ]
    probes = (
        (
            "insufficient_support",
            [_probe_hit("probe-bad-0", GUARD_BAD, 1.0)],
            True,
        ),
        (
            "negative_returns",
            [
                _probe_hit("probe-bad-1", GUARD_BAD, -1.0),
                _probe_hit("probe-bad-2", GUARD_BAD, -1.0),
            ],
            True,
        ),
        (
            "top1_supported",
            [
                _probe_hit("probe-good-0", GUARD_BASELINE, 1.0),
                _probe_hit("probe-good-1", GUARD_BASELINE, 1.0),
            ],
            False,
        ),
        (
            "unsupported_answer",
            [
                _probe_hit("probe-unrelated-0", GUARD_UNRELATED, 1.0),
                _probe_hit("probe-unrelated-1", GUARD_UNRELATED, 1.0),
            ],
            True,
        ),
    )
    details = []
    for name, hits, misleading in probes:
        result = rerank(candidates, hits, alpha=alpha, min_support=min_support)
        details.append(
            {
                "name": name,
                "reason": result.reason,
                "blocked": not result.reranked and result.text == GUARD_BASELINE,
                "misleading": misleading,
                "selected_text": result.text,
            }
        )
    return {
        "total": len(details),
        "blocked": sum(1 for detail in details if detail["blocked"]),
        "misleading_total": sum(1 for detail in details if detail["misleading"]),
        "misleading_blocked": sum(
            1 for detail in details if detail["misleading"] and detail["blocked"]
        ),
        "details": details,
    }


def evaluate_acceptance(comparison: dict[str, Any]) -> dict[str, Any]:
    """Apply the spec's first-round acceptance gates and return go/no-go."""

    thresholds = {
        "accuracy_gain_pp": 3.0,
        "false_bias_rate": 0.02,
        "latency_p95_ms": 50.0,
        "cloud_calls_not_up": True,
    }
    p95 = comparison.get("latency_p95_ms")
    checks = {
        "accuracy_gain_pp": float(comparison.get("accuracy_gain_pp", 0.0)) >= 3.0,
        "false_bias_rate": float(comparison.get("false_bias_rate", 1.0)) < 0.02,
        "latency_p95_ms": p95 is not None and float(p95) < 50.0,
        "cloud_calls_not_up": int(comparison.get("cloud_calls_delta", 0)) <= 0,
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "thresholds": thresholds,
        "checks": checks,
        "failed_checks": failed,
        "verdict": "no-go" if failed else "go",
    }


def build_report(
    *,
    seed: int,
    paraphrases_per_entry: int,
    cases: Sequence[EvalCase],
    memory: ExperienceMemory,
    no_memory: ArmResult,
    rerank: ArmResult,
    alpha: float,
    threshold: float,
    top_k: int,
    min_support: int,
    latency_budget_ms: float,
    guard_probes: dict[str, Any] | None = None,
) -> dict[str, Any]:
    comparison = _comparison(no_memory, rerank)
    return {
        "experiment": EXPERIMENT,
        "spec": SPEC,
        "generated_at": time.time(),
        "seed": seed,
        "dataset": {
            "case_count": len(cases),
            "memory_case_count": sum(1 for case in cases if case.has_memory),
            "no_memory_case_count": sum(1 for case in cases if not case.has_memory),
            "paraphrases_per_entry": paraphrases_per_entry,
            "entry_count": len({case.entry_key for case in cases}),
            "fingerprint": cases_fingerprint(cases),
            "memory_fingerprint": memory_fingerprint(cases),
            "memory_entries": len(memory),
            "cases": [case.to_dict() for case in cases],
            "memory_records": _memory_records(cases),
        },
        "config": {
            "alpha": alpha,
            "threshold": threshold,
            "top_k": top_k,
            "min_support": min_support,
            "latency_budget_ms": latency_budget_ms,
            "escalation_threshold": DEFAULT_ESCALATION_THRESHOLD,
            "adapter_version": ADAPTER_VERSION,
            "memory_tenant": MEMORY_TENANT,
            "no_memory_tenant": NO_MEMORY_TENANT,
            "nbest_size": 4,
            "require_cloud_tasks": [],
        },
        "arms": {
            "no_memory": _arm_payload(no_memory),
            "rerank": _arm_payload(rerank),
        },
        "comparison": comparison,
        "acceptance": evaluate_acceptance(comparison),
        "guard_probes": guard_probes
        or {
            "total": 0,
            "blocked": 0,
            "misleading_total": 0,
            "misleading_blocked": 0,
            "details": [],
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
    }


def run_evaluation(
    *,
    seed: int = DEFAULT_SEED,
    paraphrases_per_entry: int = DEFAULT_PARAPHRASES,
    alpha: float = DEFAULT_ALPHA,
    threshold: float = DEFAULT_THRESHOLD,
    top_k: int = DEFAULT_TOP_K,
    min_support: int = DEFAULT_MIN_SUPPORT,
    latency_budget_ms: float = DEFAULT_LATENCY_BUDGET_MS,
    now: float | None = None,
) -> dict[str, Any]:
    """Build the dataset/memory and evaluate both arms locally (CPU only)."""

    cases = build_cases(seed=seed, paraphrases_per_entry=paraphrases_per_entry)
    memory = build_memory(cases, now=now)
    no_memory = run_arm(
        cases,
        memory,
        memory_mode="off",
        alpha=alpha,
        threshold=threshold,
        top_k=top_k,
        min_support=min_support,
        latency_budget_ms=latency_budget_ms,
    )
    rerank = run_arm(
        cases,
        memory,
        memory_mode="on",
        alpha=alpha,
        threshold=threshold,
        top_k=top_k,
        min_support=min_support,
        latency_budget_ms=latency_budget_ms,
    )
    return build_report(
        seed=seed,
        paraphrases_per_entry=paraphrases_per_entry,
        cases=cases,
        memory=memory,
        no_memory=no_memory,
        rerank=rerank,
        alpha=alpha,
        threshold=threshold,
        top_k=top_k,
        min_support=min_support,
        latency_budget_ms=latency_budget_ms,
        guard_probes=run_guard_probes(alpha=alpha, min_support=min_support),
    )


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def render_summary(report: dict[str, Any]) -> str:
    """Render the compact result table committed into the experiment archive."""

    dataset = report["dataset"]
    comparison = report["comparison"]
    acceptance = report["acceptance"]
    guard = report["guard_probes"]
    lines = [
        f"# {EXPERIMENT} results",
        "",
        f"- seed: {report['seed']}",
        f"- cases: {dataset['case_count']} "
        f"(memory {dataset['memory_case_count']} / no-memory {dataset['no_memory_case_count']})",
        f"- dataset fingerprint: `{dataset['fingerprint']}`",
        f"- memory fingerprint: `{dataset['memory_fingerprint']}`",
        "",
        "| arm | accuracy | correct/total | cloud calls | hit+correct | hit+wrong | missed | reranked |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name in ("no_memory", "rerank"):
        arm = report["arms"][name]
        lines.append(
            f"| {name} | {arm['accuracy_pct']:.2f}% | {arm['correct']}/{arm['total']} "
            f"| {arm['cloud_calls']} | {arm['hit_correct']} | {arm['hit_wrong']} "
            f"| {arm['missed']} | {arm['reranked_cases']} |"
        )
    lines.extend(
        [
            "",
            f"- accuracy gain: {comparison['accuracy_gain_pp']} pp",
            f"- false bias: {comparison['false_bias_count']}/"
            f"{comparison['false_bias_denominator']} = {comparison['false_bias_rate']:.4f}",
            f"- extra latency P50/P95: {comparison['latency_p50_ms']} / "
            f"{comparison['latency_p95_ms']} ms",
            f"- cloud calls: {comparison['cloud_calls_no_memory']} -> "
            f"{comparison['cloud_calls_rerank']} (delta {comparison['cloud_calls_delta']})",
            f"- guard probes: {guard['blocked']}/{guard['total']} blocked "
            f"(misleading {guard['misleading_blocked']}/{guard['misleading_total']})",
            f"- acceptance: **{acceptance['verdict']}**"
            + (
                f" (failed: {', '.join(acceptance['failed_checks'])})"
                if acceptance["failed_checks"]
                else ""
            ),
            "",
        ]
    )
    return "\n".join(lines)


def write_artifacts(report: dict[str, Any], out_dir: str | Path) -> Path:
    """Write the report, dataset, memory and per-case outcomes under *out_dir*."""

    out = Path(out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    _write_json(out / "report.json", report)
    _write_jsonl(out / "dataset.jsonl", report["dataset"]["cases"])
    _write_jsonl(out / "memory.jsonl", report["dataset"]["memory_records"])
    _write_jsonl(out / "outcomes_no_memory.jsonl", report["arms"]["no_memory"]["cases"])
    _write_jsonl(out / "outcomes_rerank.jsonl", report["arms"]["rerank"]["cases"])
    (out / "summary.md").write_text(render_summary(report), encoding="utf-8")
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="exp13_jitrl_memory.py",
        description=(
            "exp13: 在确定性故障码改写集上对比 no-memory 与 JitRL-lite 记忆重排；"
            "纯 CPU、合成后端，不加载真实模型。"
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="构建数据集、跑两个对照臂并写出报告")
    run.add_argument("--out", default="runs-local/exp13", help="产物目录")
    run.add_argument("--seed", type=int, default=DEFAULT_SEED)
    run.add_argument(
        "--paraphrases",
        type=int,
        choices=(4, 6, 8),
        default=DEFAULT_PARAPHRASES,
        help="每个知识条目的问法数（4/6/8，记忆与无记忆对半）",
    )
    run.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    run.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    run.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    run.add_argument("--min-support", type=int, default=DEFAULT_MIN_SUPPORT)
    run.add_argument(
        "--latency-budget-ms", type=float, default=DEFAULT_LATENCY_BUDGET_MS
    )
    run.add_argument("--no-write", action="store_true", help="只打印结果，不写产物")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_evaluation(
        seed=args.seed,
        paraphrases_per_entry=args.paraphrases,
        alpha=args.alpha,
        threshold=args.threshold,
        top_k=args.top_k,
        min_support=args.min_support,
        latency_budget_ms=args.latency_budget_ms,
    )
    if not args.no_write:
        out = write_artifacts(report, args.out)
        print(f"artifacts: {out}")
    print(
        json.dumps(
            {
                "comparison": report["comparison"],
                "acceptance": report["acceptance"],
                "guard_probes": {
                    "total": report["guard_probes"]["total"],
                    "blocked": report["guard_probes"]["blocked"],
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
