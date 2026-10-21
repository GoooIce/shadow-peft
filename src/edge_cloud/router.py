"""Local-first router with confidence escalation, budget and circuit breaking.

Request path:

    产线请求 → 端侧小模型 → 置信度评估 ─┬─ 达标 → 本地返回
                                      └─ 不达标 → 云端大模型增强 → 返回
                                                   ↘ 云端不可用 → 本地降级 + 离线队列

The cloud is a *fallback and enhancer*, never a hard dependency: a disconnected
production line keeps answering with the edge model and replays hard cases later.
"""

from __future__ import annotations

import json
import math
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from dataclasses import dataclass
from dataclasses import fields as dataclass_fields
from typing import Any

from .backends import Backend, Generation, StreamingBackend
from .confidence import ConfidenceEstimator, format_ok
from .config import EdgeConfig
from .continuation import assemble_continuation, looks_like_duplicate_prefix
from .cost import CostEstimate, CostModel
from .handoff import HandoffFeatures, HandoffOutcome, HandoffPolicy
from .memory import ExperienceMemory, MemoryHit
from .metrics import Metrics
from .outbox import Outbox
from .policies import FormatGuardPolicy, OraclePolicy, ProbePolicy
from .protocol import CloudRequest, CloudResponse, ConfidenceReport, InferRequest, InferResponse
from .rerank import Candidate, rerank

# The probe artifact names its features with ``HandoffFeatures`` field names, so
# the router advertises the full theoretical set and lets the policy reject an
# artifact whose required features this deployment cannot produce.
_HANDOFF_FEATURE_NAMES = tuple(field.name for field in dataclass_fields(HandoffFeatures))

_LEGACY_MODES = frozenset({"off", "legacy"})

# A streaming handoff must leave enough time for the cloud round trip; the
# legacy path keeps its historical 150 ms guard.
_MIN_HANDOFF_REMAINING_MS = 150.0

# Memory reranking is only meaningful with alternatives to choose from; the
# spec fixes the edge n-best size at 4.
_NBEST_SIZE = 4


@dataclass(frozen=True, slots=True)
class _MemoryHitSet:
    """Retrieved hits plus the time the retrieval itself took."""

    hits: list[MemoryHit]
    retrieve_ms: float


@dataclass(frozen=True, slots=True)
class _MemoryOutcome:
    """What the memory contributed to one response, for the wire fields."""

    entry_ids: list[str]
    score: float
    reranked: bool


class CloudError(RuntimeError):
    def __init__(self, message: str, *, retryable: bool = True, status: int | None = None) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.status = status


class InferenceUnavailable(RuntimeError):
    def __init__(self, message: str, *, trace: list[str]) -> None:
        super().__init__(message)
        self.trace = trace


class CloudClient:
    def __init__(self, base_url: str, *, timeout_s: float = 30.0, api_token: str = "") -> None:
        self.base_url = base_url.rstrip("/")
        for suffix in ("/v1/enhance", "/v1"):
            if self.base_url.endswith(suffix):
                self.base_url = self.base_url[: -len(suffix)]
                break
        self.timeout_s = timeout_s
        self.api_token = api_token

    def _endpoint(self) -> str:
        return f"{self.base_url}/v1/enhance"

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None,
        *,
        timeout_ms: int | None = None,
    ) -> Any:
        url = f"{self.base_url}{path}"
        data = None
        if payload is not None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"
        http_request = urllib.request.Request(url, data=data, headers=headers, method=method)
        timeout_s = self.timeout_s
        if timeout_ms is not None:
            timeout_s = min(timeout_s, max(0.05, timeout_ms / 1000))
        try:
            with urllib.request.urlopen(http_request, timeout=timeout_s) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise CloudError(
                f"云端 HTTP {exc.code}: {detail}",
                retryable=exc.code >= 500 or exc.code == 429,
                status=exc.code,
            ) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise CloudError(f"云端连接失败: {exc}", retryable=True) from exc
        except json.JSONDecodeError as exc:
            raise CloudError(f"云端响应不是合法 JSON: {exc}", retryable=True) from exc

    def enhance(self, request: CloudRequest, *, timeout_ms: int | None = None) -> CloudResponse:
        data = self._request("POST", "/v1/enhance", request.to_dict(), timeout_ms=timeout_ms)
        if not isinstance(data, dict):
            raise CloudError(f"云端响应字段不合法: {data}", retryable=False)

        try:
            raw_output_tokens = data.get("output_tokens", 0)
            if raw_output_tokens is None:
                raw_output_tokens = 0
            if isinstance(raw_output_tokens, bool):
                raise ValueError("output_tokens 必须是整数")
            output_tokens = int(raw_output_tokens)
            if output_tokens < 0:
                raise ValueError("output_tokens 必须非负")
            response = CloudResponse(
                request_id=str(data.get("request_id") or request.request_id),
                text=str(data.get("text") or ""),
                confidence=float(data.get("confidence", 0.0)),
                model_id=str(data.get("model_id") or "cloud"),
                latency_ms=float(data.get("latency_ms", 0.0)),
                confidence_signals=dict(data.get("confidence_signals") or {}),
                finish_reason=str(data.get("finish_reason", "stop")),
                trace=list(data.get("trace") or []),
                metadata=dict(data.get("metadata") or {}),
                continued_from_prefix=bool(data.get("continued_from_prefix", False)),
                corrected_prefix=bool(data.get("corrected_prefix", False)),
                output_tokens=output_tokens,
            )
        except (TypeError, ValueError) as exc:
            raise CloudError(f"云端响应字段不合法: {data}", retryable=False) from exc
        if not response.text.strip():
            raise CloudError("云端返回空结果", retryable=True)
        return response

    def feedback(self, payload: dict[str, Any], *, timeout_ms: int = 3000) -> Any:
        return self._request("POST", "/v1/feedback", payload, timeout_ms=timeout_ms)

    def get_json(self, path: str, *, timeout_ms: int = 3000) -> Any:
        return self._request("GET", path, None, timeout_ms=timeout_ms)

    def post_json(
        self, path: str, payload: dict[str, Any], *, timeout_ms: int = 5000
    ) -> Any:
        return self._request("POST", path, payload, timeout_ms=timeout_ms)


class CircuitBreaker:
    def __init__(self, failures: int = 3, cooldown_s: float = 30.0) -> None:
        self.failures = max(1, failures)
        self.cooldown_s = max(0.0, cooldown_s)
        self._failures = 0
        self._opened_at = 0.0
        self._lock = threading.Lock()

    def allow(self, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        with self._lock:
            if self._failures < self.failures:
                return True
            return now - self._opened_at >= self.cooldown_s

    def record_success(self) -> None:
        with self._lock:
            self._failures = 0
            self._opened_at = 0.0

    def record_failure(self, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        with self._lock:
            self._failures += 1
            if self._failures >= self.failures:
                self._opened_at = now

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "failures": self._failures,
                "threshold": self.failures,
                "open": self._failures >= self.failures,
            }


class CloudBudget:
    """Sliding-window cap on cloud calls, escalation ratio and cost.

    ``max_cost_per_min`` is disabled at 0, which keeps the historical
    calls+ratio behaviour for every existing caller. ``record`` accepts the
    cost of a call so the window can price token-level handoffs; the default
    ``cost=0.0`` keeps ``record(False)`` / ``record(True)`` semantics.
    """

    def __init__(
        self,
        max_calls_per_window: int,
        max_ratio: float,
        window_s: float = 60.0,
        *,
        ratio_min_samples: int = 20,
        max_cost_per_min: float = 0.0,
    ) -> None:
        self.max_calls = max(1, max_calls_per_window)
        self.max_ratio = max(0.0, min(1.0, max_ratio))
        self.window_s = max(1.0, window_s)
        self.ratio_min_samples = max(1, ratio_min_samples)
        self.max_cost_per_min = max(0.0, float(max_cost_per_min))
        self._events: deque[tuple[float, bool, float]] = deque()
        self._lock = threading.Lock()

    def _prune(self, now: float) -> None:
        cutoff = now - self.window_s
        while self._events and self._events[0][0] < cutoff:
            self._events.popleft()

    @staticmethod
    def _event_cost(event: tuple[float, bool, float]) -> float:
        return event[2]

    def allow(
        self, now: float | None = None, estimated_cost: float = 0.0
    ) -> tuple[bool, str]:
        now = time.monotonic() if now is None else now
        try:
            estimated_cost = float(estimated_cost)
        except (TypeError, ValueError):
            estimated_cost = 0.0
        estimated_cost = max(0.0, estimated_cost)
        with self._lock:
            self._prune(now)
            calls = sum(1 for _, escalated, _ in self._events if escalated)
            total = len(self._events)
            if calls >= self.max_calls:
                return False, "max_calls"
            if total >= self.ratio_min_samples:
                projected = (calls + 1) / (total + 1)
                if projected > self.max_ratio:
                    return False, "max_ratio"
            if self.max_cost_per_min > 0.0:
                window_cost = sum(self._event_cost(event) for event in self._events)
                if window_cost + estimated_cost > self.max_cost_per_min:
                    return False, "max_cost"
            return True, ""

    def record(
        self, escalated: bool, now: float | None = None, cost: float = 0.0
    ) -> None:
        now = time.monotonic() if now is None else now
        try:
            cost = max(0.0, float(cost))
        except (TypeError, ValueError):
            cost = 0.0
        with self._lock:
            self._prune(now)
            self._events.append((now, bool(escalated), cost))

    def snapshot(self, now: float | None = None) -> dict[str, Any]:
        now = time.monotonic() if now is None else now
        with self._lock:
            self._prune(now)
            calls = sum(1 for _, escalated, _ in self._events if escalated)
            total = len(self._events)
            window_cost = sum(self._event_cost(event) for event in self._events)
        return {
            "cloud_calls": calls,
            "requests": total,
            "max_calls": self.max_calls,
            "max_ratio": self.max_ratio,
            "max_cost_per_min": self.max_cost_per_min,
            "window_cost": round(window_cost, 9),
        }


class Router:
    def __init__(
        self,
        edge_backend: Backend,
        config: EdgeConfig,
        *,
        cloud_client: CloudClient | None = None,
        confidence: ConfidenceEstimator | None = None,
        metrics: Metrics | None = None,
        outbox: Outbox | None = None,
        handoff_policy: HandoffPolicy | None = None,
        memory: ExperienceMemory | None = None,
    ) -> None:
        self.edge_backend = edge_backend
        self.config = config
        self.cloud_client = cloud_client
        self.confidence = confidence or ConfidenceEstimator()
        self.metrics = metrics or Metrics()
        self.outbox = outbox
        # Test/eval seam: a pre-built policy replaces the configured probe or
        # oracle policy while the router keeps ownership of cloud calls.
        self.handoff_policy = handoff_policy
        # Test/eval seam: an injected memory replaces the configured JSONL file.
        # Loading is lazy so a config hot-switched to ``memory_mode=on`` works
        # without rebuilding the router, and memory stays entirely local
        # (available in offline mode).
        self.memory = memory
        self._memory_loaded = memory is not None
        self._memory_error: str | None = None
        self.breaker = CircuitBreaker(config.circuit_failures, config.circuit_cooldown_s)
        self.budget = CloudBudget(
            config.max_cloud_calls_per_min,
            config.max_cloud_ratio,
            config.cloud_window_s,
            max_cost_per_min=config.max_cost_per_min,
        )

    def infer(self, request: InferRequest) -> InferResponse:
        started = time.monotonic()
        trace: list[str] = []
        expected_format = request.metadata.get("expected_format") or (
            self.config.expected_format_by_task.get(request.task)
        )

        # Read the mode on every request: #13 hot-switches it through config.
        mode = self.config.handoff_mode
        if mode in _LEGACY_MODES:
            return self._infer_whole_segment(request, started, trace, expected_format)
        if request.require_cloud or request.task in set(self.config.require_cloud_tasks):
            # Whole-segment cloud semantics; never enter streaming or send a trace.
            trace.append("任务要求整段云端：跳过 token 级交接")
            return self._infer_whole_segment(request, started, trace, expected_format)
        policy = self._select_handoff_policy(trace, mode)
        if policy is None:
            return self._infer_whole_segment(request, started, trace, expected_format)
        return self._infer_streaming(request, started, trace, expected_format, policy)

    def _infer_whole_segment(
        self,
        request: InferRequest,
        started: float,
        trace: list[str],
        expected_format: str | None,
    ) -> InferResponse:
        edge_started = time.monotonic()
        edge_generation: Generation | None = None
        edge_error: str | None = None
        memory_outcome: _MemoryOutcome | None = None
        try:
            if self._memory_eligible(request):
                edge_generation, memory_outcome = self._generate_with_memory(
                    request, trace, expected_format
                )
            else:
                edge_generation = self.edge_backend.generate(request)
        except Exception as exc:  # noqa: BLE001 - any backend failure must degrade, not crash
            edge_error = str(exc)
            self.metrics.incr("edge_failures_total")
            trace.append(f"端侧推理失败：{exc}")
        edge_wall_ms = (time.monotonic() - edge_started) * 1000
        return self._finish_whole(
            request,
            started,
            trace,
            expected_format,
            edge_generation,
            edge_error,
            edge_wall_ms,
            memory=memory_outcome,
        )

    # --------------------------------------------------------------- memory
    def _memory_eligible(self, request: InferRequest) -> bool:
        """Memory reranking only applies to successful, non-forced whole answers.

        The streaming handoff path is deliberately excluded: it already chose a
        token-level handoff after one decode, and reranking would need a second
        n-best decode that changes the single-handoff semantics.
        """

        if str(self.config.memory_mode).strip().lower() != "on":
            return False
        if request.require_cloud or request.task in set(self.config.require_cloud_tasks):
            return False
        return callable(getattr(self.edge_backend, "generate_nbest", None))

    def _get_memory(self) -> ExperienceMemory | None:
        """Load the configured memory once; ``None`` means it is unavailable."""

        if not self._memory_loaded:
            self._memory_loaded = True
            try:
                if self.config.memory_path:
                    self.memory = ExperienceMemory.load(
                        self.config.memory_path,
                        default_ttl_s=self.config.memory_ttl_s,
                    )
                else:
                    self.memory = ExperienceMemory(
                        default_ttl_s=self.config.memory_ttl_s
                    )
            except Exception as exc:  # noqa: BLE001 - memory must never fail inference
                self.memory = None
                self._memory_error = str(exc)
        if self._memory_error is not None:
            return None
        return self.memory

    def _retrieve_memory(
        self, request: InferRequest, trace: list[str]
    ) -> _MemoryHitSet | None:
        """Retrieve top-k hits within the latency budget, or ``None``.

        A miss, an unavailable memory or an over-budget retrieval returns
        ``None`` so the caller keeps the baseline single-answer path.
        """

        memory = self._get_memory()
        if memory is None:
            self.metrics.incr("memory_fallback_total")
            detail = self._memory_error or "未初始化"
            trace.append(f"记忆不可用，回退端侧基线：{detail}")
            return None
        started = time.perf_counter()
        try:
            hits = memory.retrieve(
                request.prompt,
                task=request.task,
                tenant=self.config.memory_tenant,
                adapter_version=self.config.adapter_version,
                k=self.config.memory_top_k,
                threshold=self.config.memory_threshold,
            )
        except Exception as exc:  # noqa: BLE001 - retrieval failures are a silent fallback
            elapsed_ms = (time.perf_counter() - started) * 1000
            self.metrics.observe("memory_latency_ms", elapsed_ms)
            self.metrics.incr("memory_fallback_total")
            trace.append(f"记忆检索失败，回退端侧基线：{exc}")
            return None
        elapsed_ms = (time.perf_counter() - started) * 1000
        if not hits:
            self.metrics.observe("memory_latency_ms", elapsed_ms)
            return None
        if elapsed_ms > self.config.memory_latency_budget_ms:
            self.metrics.observe("memory_latency_ms", elapsed_ms)
            self.metrics.incr("memory_fallback_total")
            trace.append(
                f"记忆检索用时 {elapsed_ms:.1f}ms 超过预算 "
                f"{self.config.memory_latency_budget_ms:.1f}ms，按 no_memory 回退"
            )
            return None
        return _MemoryHitSet(hits=hits, retrieve_ms=elapsed_ms)

    def _generate_with_memory(
        self,
        request: InferRequest,
        trace: list[str],
        expected_format: str | None,
    ) -> tuple[Generation, _MemoryOutcome | None]:
        """Generate n-best and rerank it with memory, falling back to one answer."""

        hit_set = self._retrieve_memory(request, trace)
        if hit_set is None:
            return self.edge_backend.generate(request), None

        try:
            generations = [
                generation
                for generation in self.edge_backend.generate_nbest(
                    request, n=_NBEST_SIZE
                )
                if isinstance(generation, Generation) and generation.text.strip()
            ]
        except Exception as exc:  # noqa: BLE001 - reranking must not break inference
            self.metrics.incr("memory_fallback_total")
            trace.append(f"端侧 n-best 生成失败，回退单答案：{exc}")
            return self.edge_backend.generate(request), None
        if not generations:
            self.metrics.incr("memory_fallback_total")
            trace.append("端侧 n-best 生成结果为空，回退单答案")
            return self.edge_backend.generate(request), None

        candidates = [
            Candidate(
                text=generation.text,
                logprob=self._sequence_logprob(generation),
                finish_reason=generation.finish_reason,
            )
            for generation in generations
        ]
        rerank_started = time.perf_counter()
        try:
            result = rerank(
                candidates,
                hit_set.hits,
                alpha=self.config.memory_alpha,
                min_support=self.config.memory_min_support,
            )
        except Exception as exc:  # noqa: BLE001 - reranking must not break inference
            elapsed_ms = hit_set.retrieve_ms + (time.perf_counter() - rerank_started) * 1000
            self.metrics.observe("memory_latency_ms", elapsed_ms)
            self.metrics.incr("memory_fallback_total")
            trace.append(f"记忆重排失败，回退端侧基线：{exc}")
            return generations[self._best_logprob_index(candidates)], None

        memory_ms = hit_set.retrieve_ms + (time.perf_counter() - rerank_started) * 1000
        self.metrics.observe("memory_latency_ms", memory_ms)
        baseline_index = self._best_logprob_index(candidates)
        baseline_generation = generations[baseline_index]
        if memory_ms > self.config.memory_latency_budget_ms:
            self.metrics.incr("memory_fallback_total")
            trace.append(
                f"记忆检索+重排用时 {memory_ms:.1f}ms 超过预算 "
                f"{self.config.memory_latency_budget_ms:.1f}ms，按 no_memory 回退"
            )
            return baseline_generation, None

        self.metrics.incr("memory_hit_total")
        entry_ids = [hit.entry.entry_id for hit in hit_set.hits]
        trace.append(
            f"记忆命中 {len(hit_set.hits)} 条（最高相似度 "
            f"{hit_set.hits[0].score:.2f}），重排 {len(candidates)} 个候选"
        )
        selected_generation = generations[result.index]
        reranked = result.reranked and selected_generation.text != baseline_generation.text
        if reranked and format_ok(selected_generation.text, expected_format) == 0.0:
            self.metrics.incr("memory_fallback_total")
            trace.append(
                f"记忆重排答案未通过 {expected_format} 格式校验，回退端侧原答案"
            )
            return baseline_generation, _MemoryOutcome(
                entry_ids=entry_ids,
                score=result.memory_score,
                reranked=False,
            )
        if reranked:
            self.metrics.incr("memory_reranked_total")
            trace.append(f"记忆重排改写端侧答案：{result.reason}")
        else:
            reason = "top1_unchanged" if result.reranked else result.reason
            trace.append(f"记忆未改写端侧答案：{reason}")
        return selected_generation, _MemoryOutcome(
            entry_ids=entry_ids,
            score=result.memory_score,
            reranked=reranked,
        )

    @staticmethod
    def _sequence_logprob(generation: Generation) -> float:
        """Length-normalized log probability used as the rerank model term."""

        values = [
            float(value)
            for value in (generation.token_logprobs or [])
            if math.isfinite(value)
        ]
        return sum(values) / len(values) if values else 0.0

    @staticmethod
    def _best_logprob_index(candidates: list[Candidate]) -> int:
        best = 0
        for index in range(1, len(candidates)):
            if candidates[index].logprob > candidates[best].logprob:
                best = index
        return best

    @staticmethod
    def _memory_fields(memory: _MemoryOutcome | None) -> dict[str, Any]:
        if memory is None:
            return {}
        return {
            "memory_hit": True,
            "memory_entry_ids": list(memory.entry_ids),
            "memory_score": memory.score,
            "memory_reranked": memory.reranked,
        }

    def _finish_whole(
        self,
        request: InferRequest,
        started: float,
        trace: list[str],
        expected_format: str | None,
        edge_generation: Generation | None,
        edge_error: str | None,
        edge_wall_ms: float,
        *,
        memory: _MemoryOutcome | None = None,
    ) -> InferResponse:
        """Existing whole-segment escalation path (legacy and fallbacks)."""

        edge_report: ConfidenceReport | None = None
        if edge_generation is not None:
            self.metrics.incr("edge_success_total")
            self.metrics.observe("edge_latency_ms", edge_wall_ms)
            edge_report = self.confidence.estimate(
                edge_generation.text,
                edge_generation.token_logprobs,
                finish_reason=edge_generation.finish_reason,
                expected_format=expected_format,
            )
            trace.append(
                f"端侧 {edge_generation.model_id} 用时 {edge_wall_ms:.1f}ms，"
                f"置信度 {edge_report.score:.2f}（{edge_report.method}）"
            )

        reason = self._escalation_reason(request, edge_generation, edge_report, edge_error)
        if reason is None:
            self.metrics.incr("requests_total")
            self.metrics.incr("edge_only_total")
            self.budget.record(False)
            trace.append("置信度达到阈值，端侧直接返回")
            response = self._edge_response(
                request,
                edge_generation,
                edge_report,
                started,
                trace,
                expected_format,
                memory=memory,
            )
            self._attach_edge_cost(response, edge_generation)
            return response

        self.metrics.incr("escalation_total")
        self.metrics.incr(f"escalation_reason_{reason}")
        threshold = self._escalation_threshold_for(request.task)
        if self.config.escalation_threshold_by_task:
            trace.append(
                f"触发升级：{reason}（task={request.task} 阈值 {threshold:.2f}）"
            )
        else:
            trace.append(f"触发升级：{reason}（阈值 {threshold:.2f}）")

        edge_response = self._edge_response(
            request,
            edge_generation,
            edge_report,
            started,
            trace,
            expected_format,
            memory=memory,
        )
        if self.config.offline:
            self.budget.record(False)
            trace.append("当前为离线模式，保留端侧结果并写入待同步队列")
            return self._degrade(request, edge_response, reason, started, trace, queued=True)

        if self.cloud_client is None:
            self.budget.record(False)
            trace.append("未配置云端客户端，保留端侧结果")
            return self._degrade(request, edge_response, reason, started, trace, queued=True)

        remaining_ms = request.deadline_ms - (time.monotonic() - started) * 1000
        cost_model = self._cost_model()
        edge_tokens = self._edge_token_count(edge_generation)
        prompt_tokens = self._prompt_tokens(edge_generation) if edge_generation else 0
        estimated = cost_model.estimate(
            edge_tokens=edge_tokens,
            cloud_prefill_tokens=prompt_tokens + edge_tokens,
            cloud_decode_tokens_expected=request.max_tokens,
            cloud_calls=1,
        )
        skip_reason = ""
        if remaining_ms < 150:
            skip_reason = "deadline_exceeded"
        elif not self.breaker.allow():
            skip_reason = "circuit_open"
        else:
            allowed, budget_reason = self.budget.allow(
                estimated_cost=estimated.estimated_cost
            )
            if not allowed:
                skip_reason = f"budget_{budget_reason}"
        if skip_reason:
            self.budget.record(False)
            trace.append(f"跳过云端增强：{skip_reason}")
            return self._degrade(request, edge_response, reason, started, trace, queued=True)

        self.metrics.incr("cloud_attempt_total")
        self.budget.record(True, cost=estimated.estimated_cost)
        cloud_request = CloudRequest(
            prompt=request.prompt,
            request_id=request.request_id,
            task=request.task,
            draft=edge_generation.text if edge_generation else "",
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            deadline_ms=int(min(self.config.cloud_timeout_ms, max(150, remaining_ms))),
            metadata={
                **request.metadata,
                "edge_confidence": edge_report.score if edge_report else None,
            },
        )
        try:
            cloud_response = self.cloud_client.enhance(cloud_request, timeout_ms=int(remaining_ms))
        except Exception as exc:  # noqa: BLE001 - fall back to the edge answer
            self.breaker.record_failure()
            self.metrics.incr("cloud_failure_total")
            trace.append(f"云端增强失败：{exc}")
            return self._degrade(request, edge_response, reason, started, trace, queued=True)

        self.breaker.record_success()
        self.metrics.incr("cloud_success_total")
        self.metrics.incr("requests_total")
        self.metrics.observe("cloud_latency_ms", cloud_response.latency_ms)
        self.metrics.observe("request_latency_ms", (time.monotonic() - started) * 1000)
        cloud_report = ConfidenceReport(
            score=cloud_response.confidence,
            method="cloud_reported",
            signals=dict(cloud_response.confidence_signals),
        )
        trace.append(
            f"云端 {cloud_response.model_id} 用时 {cloud_response.latency_ms:.1f}ms，"
            f"置信度 {cloud_report.score:.2f}"
        )
        trace.append("最终采用云端增强结果")
        response = InferResponse(
            request_id=request.request_id,
            text=cloud_response.text,
            source="cloud",
            confidence=cloud_report.score,
            latency_ms=(time.monotonic() - started) * 1000,
            model_id=cloud_response.model_id,
            adapter_version=self.config.adapter_version,
            escalated=True,
            escalation_reason=reason,
            edge_text=edge_generation.text if edge_generation else None,
            edge_confidence=edge_report.score if edge_report else None,
            edge_latency_ms=edge_generation.latency_ms if edge_generation else None,
            cloud_latency_ms=cloud_response.latency_ms,
            degraded=False,
            trace=trace,
            confidence_signals=cloud_report.signals,
            finish_reason=cloud_response.finish_reason,
            cloud_output_tokens=cloud_response.output_tokens,
            metadata={"edge_model": edge_generation.model_id if edge_generation else None},
            memory_hit=memory is not None,
            memory_entry_ids=list(memory.entry_ids) if memory else [],
            memory_score=memory.score if memory else None,
            # The final answer is the cloud one, so it is not a memory rewrite.
            memory_reranked=False,
        )
        response.cost_estimate = cost_model.estimate(
            edge_tokens=edge_tokens,
            cloud_prefill_tokens=prompt_tokens + edge_tokens,
            cloud_decode_tokens_expected=cloud_response.output_tokens,
            cloud_calls=1,
        ).to_dict()
        return response

    # ------------------------------------------------------------------ handoff
    def _supports_streaming(self) -> bool:
        return isinstance(self.edge_backend, StreamingBackend) or callable(
            getattr(self.edge_backend, "generate_until", None)
        )

    def _select_handoff_policy(
        self, trace: list[str], mode: str
    ) -> HandoffPolicy | None:
        """Return the policy for this request, or ``None`` to use the legacy path."""

        if not self._supports_streaming():
            self.metrics.incr("streaming_unavailable_total")
            self.metrics.incr("legacy_fallback_total")
            trace.append(
                f"端侧后端不支持流式生成（{type(self.edge_backend).__name__}），"
                "回退整段升级路径"
            )
            return None
        if self.handoff_policy is not None:
            return self.handoff_policy
        if mode == "oracle":
            return OraclePolicy()
        policy = ProbePolicy(
            self.config.probe_path,
            self.config.handoff_threshold,
            self.config.handoff_threshold_by_task,
            feature_names=_HANDOFF_FEATURE_NAMES,
        )
        if not policy.available:
            self.metrics.incr("probe_unavailable_total")
            self.metrics.incr("legacy_fallback_total")
            trace.append(f"probe 不可用，回退整段升级路径：{policy.unavailable_reason}")
            return None
        return policy

    def _infer_streaming(
        self,
        request: InferRequest,
        started: float,
        trace: list[str],
        expected_format: str | None,
        policy: HandoffPolicy,
    ) -> InferResponse:
        context: dict[str, Any] = {
            "task": request.task,
            "deadline_ms": request.deadline_ms,
            "expected_format": expected_format,
            "deadline_at": started + request.deadline_ms / 1000.0,
            "min_remaining_ms": max(
                _MIN_HANDOFF_REMAINING_MS, float(self.config.cloud_rtt_ms)
            ),
            "clock": time.monotonic,
        }
        gold = request.metadata.get("gold_handoff_index")
        if (
            self.config.handoff_mode == "oracle"
            and isinstance(gold, int)
            and not isinstance(gold, bool)
            and gold >= 0
        ):
            # A missing label simply never triggers; a malformed one is ignored
            # instead of failing the request.
            context["gold_handoff_index"] = gold

        edge_started = time.monotonic()
        edge_generation: Generation | None = None
        edge_error: str | None = None
        try:
            edge_generation = self.edge_backend.generate_until(
                request,
                self._format_guarded(policy),
                max_boundary_wait_tokens=self.config.max_boundary_wait_tokens,
                expected_format=expected_format,
                context=context,
            )
        except Exception as exc:  # noqa: BLE001 - streaming failure must degrade, not crash
            edge_error = str(exc)
            self.metrics.incr("edge_failures_total")
            trace.append(f"端侧流式推理失败：{exc}")
        edge_wall_ms = (time.monotonic() - edge_started) * 1000

        if edge_generation is None or not self._handed_off(edge_generation):
            if isinstance(policy, ProbePolicy) and not policy.available:
                self.metrics.incr("probe_unavailable_total")
                self.metrics.incr("legacy_fallback_total")
                trace.append(
                    "probe 运行期不可用，回退整段升级路径："
                    f"{policy.unavailable_reason}"
                )
            # No handoff: the generation is a complete answer, so reuse the whole
            # escalation path exactly (including ConfidenceEstimator scoring).
            return self._finish_whole(
                request,
                started,
                trace,
                expected_format,
                edge_generation,
                edge_error,
                edge_wall_ms,
            )
        return self._infer_handoff(
            request, started, trace, expected_format, edge_generation, edge_wall_ms
        )

    @staticmethod
    def _format_guarded(policy: HandoffPolicy) -> HandoffPolicy:
        """Wrap the configured probe/oracle policy with the JSON format guard.

        Injected test/eval policies are returned unchanged: the guard is a
        serving-path rule for the two shipped policies, not a policy rewriter.
        """

        if isinstance(policy, (ProbePolicy, OraclePolicy)):
            return FormatGuardPolicy(policy)
        return policy

    @staticmethod
    def _handed_off(generation: Generation) -> bool:
        return generation.handoff is not None and bool(generation.handoff.handed_off)

    def _infer_handoff(
        self,
        request: InferRequest,
        started: float,
        trace: list[str],
        expected_format: str | None,
        generation: Generation,
        edge_wall_ms: float,
    ) -> InferResponse:
        """Send the edge prefix to the cloud exactly once and assemble the answer."""

        outcome = generation.handoff
        if outcome is None:  # pragma: no cover - guarded by _handed_off
            raise AssertionError("handoff outcome 缺失")
        prefix = generation.text or outcome.partial_trace
        handoff_index = (
            outcome.handoff_index
            if outcome.handoff_index is not None
            else generation.output_tokens
        )
        prefix_tokens = generation.output_tokens or handoff_index
        handoff_reason = str(outcome.reason or "low_prefix_trust")
        boundary = str(outcome.boundary or "token")
        waited_tokens = self._generation_int(generation, "handoff_waited_tokens")
        prompt_tokens = self._prompt_tokens(generation)
        edge_report = self.confidence.estimate(
            prefix,
            generation.token_logprobs,
            finish_reason=generation.finish_reason,
            expected_format=expected_format,
        )

        self.metrics.incr("edge_success_total")
        self.metrics.observe("edge_latency_ms", edge_wall_ms)
        self.metrics.incr("handoff_total")
        self.metrics.incr(f"handoff_reason_{handoff_reason}")
        self.metrics.incr(f"handoff_boundary_{boundary}")
        self.metrics.observe("handoff_index", float(handoff_index))
        self.metrics.observe_hist(
            "handoff_index", float(handoff_index), (4, 8, 16, 32, 64, 128)
        )
        self.metrics.observe("boundary_wait_tokens", float(waited_tokens))
        self.metrics.incr("escalation_total")
        self.metrics.incr(f"escalation_reason_{handoff_reason}")
        trace.append(
            f"端侧 {generation.model_id} 在第 {handoff_index} 个 token 处交接"
            f"（{boundary} 边界，原因 {handoff_reason}）"
        )

        cost_model = self._cost_model()
        estimated = cost_model.estimate(
            edge_tokens=prefix_tokens,
            cloud_prefill_tokens=prompt_tokens + prefix_tokens,
            cloud_decode_tokens_expected=max(1, request.max_tokens - prefix_tokens),
            cloud_calls=1,
        )
        edge_response = self._handoff_edge_response(
            request,
            generation,
            edge_report,
            prefix,
            prefix_tokens,
            handoff_index,
            handoff_reason,
            outcome,
            expected_format,
            started,
            trace,
            cost_model.estimate(edge_tokens=prefix_tokens),
        )

        remaining_ms = request.deadline_ms - (time.monotonic() - started) * 1000
        required_ms = max(_MIN_HANDOFF_REMAINING_MS, float(self.config.cloud_rtt_ms))
        skip_reason = ""
        skip_detail = ""
        if self.config.offline:
            skip_reason = "offline"
            skip_detail = "离线模式"
        elif self.cloud_client is None:
            skip_reason = "offline"
            skip_detail = "未配置云端客户端"
        elif not self.breaker.allow():
            skip_reason = "circuit_open"
            skip_detail = "熔断器打开"
        elif remaining_ms < required_ms:
            skip_reason = "deadline"
            skip_detail = f"剩余 {remaining_ms:.0f}ms < 需要 {required_ms:.0f}ms"
        else:
            allowed, budget_reason = self.budget.allow(
                estimated_cost=estimated.estimated_cost
            )
            if not allowed:
                skip_reason = "budget"
                skip_detail = f"预算窗口不足（{budget_reason}）"
        if skip_reason:
            self.budget.record(False)
            trace.append(f"跳过云端续写：{skip_reason}，{skip_detail}")
            return self._degrade(request, edge_response, skip_reason, started, trace, queued=True)

        self.metrics.incr("cloud_attempt_total")
        self.budget.record(True, cost=estimated.estimated_cost)
        cloud_request = CloudRequest(
            prompt=request.prompt,
            request_id=request.request_id,
            task=request.task,
            draft="",
            max_tokens=max(1, request.max_tokens - prefix_tokens),
            temperature=request.temperature,
            deadline_ms=int(min(self.config.cloud_timeout_ms, max(150, remaining_ms))),
            metadata={
                **request.metadata,
                "edge_confidence": edge_report.score,
            },
            partial_trace=prefix,
            handoff_index=handoff_index,
            edge_tokens_before_handoff=prefix_tokens,
            handoff_reason=handoff_reason,
            prefix_trust=outcome.prefix_trust,
            allow_prefix_rewrite=self.config.allow_prefix_rewrite,
        )
        try:
            cloud_response = self.cloud_client.enhance(
                cloud_request, timeout_ms=int(remaining_ms)
            )
        except Exception as exc:  # noqa: BLE001 - keep the edge prefix
            self.breaker.record_failure()
            self.metrics.incr("cloud_failure_total")
            trace.append(f"云端续写失败：{exc}")
            return self._degrade(
                request, edge_response, handoff_reason, started, trace, queued=True
            )

        self.breaker.record_success()
        self.metrics.incr("cloud_success_total")
        self.metrics.incr("requests_total")
        self.metrics.observe("cloud_latency_ms", cloud_response.latency_ms)
        self.metrics.observe("request_latency_ms", (time.monotonic() - started) * 1000)
        self.metrics.incr("cloud_prefill_tokens", float(prompt_tokens + prefix_tokens))
        self.metrics.observe("cloud_decode_tokens", float(cloud_response.output_tokens))
        cloud_report = ConfidenceReport(
            score=cloud_response.confidence,
            method="cloud_reported",
            signals=dict(cloud_response.confidence_signals),
        )
        trace.append(
            f"云端 {cloud_response.model_id} 续写用时 {cloud_response.latency_ms:.1f}ms，"
            f"置信度 {cloud_report.score:.2f}"
        )

        rewrite = bool(cloud_response.corrected_prefix)
        mismatch = False
        if rewrite:
            final_text = cloud_response.text
            trace.append("云端纠正了端侧前缀，采用云端完整答案")
        else:
            if not cloud_response.continued_from_prefix:
                mismatch = True
                trace.append("云端未声明续写（continued_from_prefix=false）")
            if bool(cloud_response.metadata.get("continuation_mismatch")) or (
                looks_like_duplicate_prefix(prefix, cloud_response.text)
            ):
                mismatch = True
                trace.append("云端重述了端侧前缀，按安全拼接处理")
            if mismatch:
                self.metrics.incr("continuation_mismatch_total")
            if self.config.allow_prefix_rewrite:
                trace.append("云端未声明纠正，按续写处理")
            final_text = assemble_continuation(prefix, cloud_response.text)
            trace.append("最终答案由端侧前缀与云端续写拼接而成")

        degraded = False
        if format_ok(final_text, expected_format) == 0.0:
            degraded = True
            self.metrics.incr("final_format_invalid_total")
            trace.append(f"最终答案未通过 {expected_format} 格式校验，仍保留云端结果")

        cost_estimate = cost_model.estimate(
            edge_tokens=prefix_tokens,
            cloud_prefill_tokens=prompt_tokens + prefix_tokens,
            cloud_decode_tokens_expected=cloud_response.output_tokens,
            cloud_calls=1,
        )
        return InferResponse(
            request_id=request.request_id,
            text=final_text,
            source="cloud",
            confidence=cloud_report.score,
            latency_ms=(time.monotonic() - started) * 1000,
            model_id=cloud_response.model_id,
            adapter_version=self.config.adapter_version,
            escalated=True,
            escalation_reason=handoff_reason,
            edge_text=prefix,
            edge_confidence=edge_report.score,
            edge_latency_ms=generation.latency_ms,
            cloud_latency_ms=cloud_response.latency_ms,
            degraded=degraded,
            trace=trace,
            confidence_signals=cloud_report.signals,
            finish_reason=cloud_response.finish_reason,
            metadata={
                "edge_model": generation.model_id,
                "expected_format": expected_format,
                "handoff_policy_version": self.config.handoff_policy_version,
                "continued_from_prefix": bool(cloud_response.continued_from_prefix),
                "corrected_prefix": bool(cloud_response.corrected_prefix),
            },
            handoff_index=handoff_index,
            edge_tokens_before_handoff=prefix_tokens,
            handoff_reason=handoff_reason,
            prefix_trust=outcome.prefix_trust,
            cloud_output_tokens=cloud_response.output_tokens,
            cost_estimate=cost_estimate.to_dict(),
        )

    def _handoff_edge_response(
        self,
        request: InferRequest,
        generation: Generation,
        report: ConfidenceReport,
        prefix: str,
        prefix_tokens: int,
        handoff_index: int,
        handoff_reason: str,
        outcome: HandoffOutcome,
        expected_format: str | None,
        started: float,
        trace: list[str],
        cost_estimate: CostEstimate,
    ) -> InferResponse:
        return InferResponse(
            request_id=request.request_id,
            text=prefix,
            source="edge",
            confidence=report.score,
            latency_ms=(time.monotonic() - started) * 1000,
            model_id=generation.model_id,
            adapter_version=self.config.adapter_version,
            escalated=False,
            edge_text=prefix,
            edge_confidence=report.score,
            edge_latency_ms=generation.latency_ms,
            degraded=False,
            trace=list(trace),
            confidence_signals=report.signals,
            finish_reason=generation.finish_reason,
            metadata={
                "edge_model": generation.model_id,
                "expected_format": expected_format,
            },
            handoff_index=handoff_index,
            edge_tokens_before_handoff=prefix_tokens,
            handoff_reason=handoff_reason,
            prefix_trust=outcome.prefix_trust,
            cloud_output_tokens=0,
            cost_estimate=cost_estimate.to_dict(),
        )

    # ------------------------------------------------------------------ cost
    def _cost_model(self) -> CostModel:
        return CostModel(
            cost_per_edge_token=self.config.cost_per_edge_token,
            cost_per_cloud_input_token=self.config.cost_per_cloud_input_token,
            cost_per_cloud_output_token=self.config.cost_per_cloud_output_token,
            cost_per_cloud_call=self.config.cost_per_cloud_call,
            cloud_rtt_ms=self.config.cloud_rtt_ms,
        )

    @staticmethod
    def _generation_int(generation: Generation, key: str) -> int:
        raw = generation.raw if isinstance(generation.raw, dict) else {}
        value = raw.get(key, 0)
        if isinstance(value, bool):
            return 0
        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            return 0

    def _prompt_tokens(self, generation: Generation) -> int:
        return self._generation_int(generation, "prompt_tokens")

    @staticmethod
    def _edge_token_count(generation: Generation | None) -> int:
        if generation is None:
            return 0
        if generation.output_tokens > 0:
            return generation.output_tokens
        if generation.token_logprobs:
            return len(generation.token_logprobs)
        return 0

    def _attach_edge_cost(
        self, response: InferResponse, generation: Generation | None
    ) -> None:
        edge_tokens = self._edge_token_count(generation)
        response.cost_estimate = self._cost_model().estimate(
            edge_tokens=edge_tokens
        ).to_dict()

    def _escalation_threshold_for(self, task: str) -> float:
        """Whole-segment threshold for *task*, falling back to the global one."""

        return float(
            self.config.escalation_threshold_by_task.get(
                task, self.config.escalation_threshold
            )
        )

    def _escalation_reason(
        self,
        request: InferRequest,
        generation: Generation | None,
        report: ConfidenceReport | None,
        edge_error: str | None,
    ) -> str | None:
        if edge_error is not None:
            return "edge_unavailable"
        if request.require_cloud or request.task in set(self.config.require_cloud_tasks):
            return "task_requires_cloud"
        if report is None:
            return "no_edge_result"
        if report.signals.get("format_ok", 1.0) == 0.0:
            return "format_invalid"
        if report.score < self._escalation_threshold_for(request.task):
            return "low_confidence"
        return None

    def _edge_response(
        self,
        request: InferRequest,
        generation: Generation | None,
        report: ConfidenceReport | None,
        started: float,
        trace: list[str],
        expected_format: str | None,
        *,
        memory: _MemoryOutcome | None = None,
    ) -> InferResponse:
        if generation is None:
            return InferResponse(
                request_id=request.request_id,
                text="",
                source="none",
                confidence=0.0,
                latency_ms=(time.monotonic() - started) * 1000,
                model_id="none",
                adapter_version=self.config.adapter_version,
                degraded=True,
                trace=trace,
                metadata={"expected_format": expected_format},
            )
        return InferResponse(
            request_id=request.request_id,
            text=generation.text,
            source="edge",
            confidence=report.score if report else 0.0,
            latency_ms=(time.monotonic() - started) * 1000,
            model_id=generation.model_id,
            adapter_version=self.config.adapter_version,
            escalated=False,
            edge_text=generation.text,
            edge_confidence=report.score if report else None,
            edge_latency_ms=generation.latency_ms,
            degraded=False,
            trace=trace,
            confidence_signals=report.signals if report else {},
            finish_reason=generation.finish_reason,
            metadata={"expected_format": expected_format},
            **self._memory_fields(memory),
        )

    def _degrade(
        self,
        request: InferRequest,
        edge_response: InferResponse,
        reason: str,
        started: float,
        trace: list[str],
        *,
        queued: bool,
    ) -> InferResponse:
        self.metrics.incr("requests_total")
        self.metrics.incr("degraded_total")
        edge_response.degraded = True
        edge_response.escalation_reason = reason
        edge_response.trace = list(trace)
        edge_response.latency_ms = (time.monotonic() - started) * 1000
        if queued and self.outbox is not None:
            try:
                self.outbox.enqueue(request, edge_response, reason)
                self.metrics.incr("outbox_enqueued_total")
            except Exception as exc:  # noqa: BLE001 - disk issues must not fail inference
                self.metrics.incr("outbox_error_total")
                trace.append(f"离线队列写入失败：{exc}")
        self.metrics.observe("request_latency_ms", edge_response.latency_ms)
        return edge_response


class OutboxFlusher(threading.Thread):
    """Replay pending offline items when the cloud becomes reachable again."""

    def __init__(
        self,
        outbox: Outbox,
        cloud_client: CloudClient,
        *,
        interval_s: float = 10.0,
        max_attempts: int = 10,
        metrics: Metrics | None = None,
        cloud_timeout_ms: int = 15000,
    ) -> None:
        super().__init__(name="edge-cloud-outbox-flusher", daemon=True)
        self.outbox = outbox
        self.cloud_client = cloud_client
        self.interval_s = max(1.0, interval_s)
        self.max_attempts = max(1, max_attempts)
        self.metrics = metrics or Metrics()
        self.cloud_timeout_ms = cloud_timeout_ms
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        while not self._stop.wait(self.interval_s):
            self.flush_once()

    def flush_once(self) -> int:
        flushed = 0
        for item in self.outbox.pending():
            attempts = int(item.get("attempts", 0))
            if attempts >= self.max_attempts:
                self.outbox.update(item["item_id"], status="dead")
                continue
            try:
                request = InferRequest.from_dict(item.get("request") or {})
                cached = item.get("edge_response") or {}
                edge_text = str(cached.get("text", ""))
                handoff_index = cached.get("handoff_index")
                partial_trace = (
                    str(cached.get("edge_text") or edge_text)
                    if handoff_index is not None
                    else ""
                )
                edge_tokens = cached.get("edge_tokens_before_handoff")
                max_tokens = request.max_tokens
                if partial_trace:
                    max_tokens = max(1, max_tokens - int(edge_tokens or handoff_index or 0))
                cloud_request = CloudRequest(
                    prompt=request.prompt,
                    request_id=request.request_id,
                    task=request.task,
                    draft="" if partial_trace else edge_text,
                    max_tokens=max_tokens,
                    temperature=request.temperature,
                    deadline_ms=self.cloud_timeout_ms,
                    metadata=request.metadata,
                    partial_trace=partial_trace,
                    handoff_index=handoff_index,
                    edge_tokens_before_handoff=edge_tokens,
                    handoff_reason=cached.get("handoff_reason"),
                    prefix_trust=cached.get("prefix_trust"),
                )
            except Exception as exc:  # noqa: BLE001 - corrupt item must not kill the flusher
                self.outbox.update(
                    item["item_id"],
                    status="dead",
                    last_error=f"条目损坏: {exc}"[:300],
                )
                self.metrics.incr("outbox_flush_corrupt_total")
                continue
            try:
                response = self.cloud_client.enhance(
                    cloud_request, timeout_ms=self.cloud_timeout_ms
                )
            except Exception as exc:  # noqa: BLE001 - keep the item pending
                self.outbox.update(
                    item["item_id"], attempts=attempts + 1, last_error=str(exc)[:300]
                )
                self.metrics.incr("outbox_flush_failure_total")
                break
            cloud_text = (
                assemble_continuation(
                    partial_trace, response.text, corrected=response.corrected_prefix
                )
                if partial_trace
                else response.text
            )
            self.outbox.update(
                item["item_id"],
                status="done",
                attempts=attempts + 1,
                cloud_text=cloud_text,
                cloud_confidence=response.confidence,
                last_error="",
            )
            self.metrics.incr("outbox_flush_success_total")
            flushed += 1
        return flushed
