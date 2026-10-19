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
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from typing import Any

from .backends import Backend, Generation
from .confidence import ConfidenceEstimator
from .config import EdgeConfig
from .metrics import Metrics
from .outbox import Outbox
from .protocol import CloudRequest, CloudResponse, ConfidenceReport, InferRequest, InferResponse


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
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
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
    """Sliding-window cap on cloud calls (absolute and escalation ratio)."""

    def __init__(
        self,
        max_calls_per_window: int,
        max_ratio: float,
        window_s: float = 60.0,
        *,
        ratio_min_samples: int = 20,
    ) -> None:
        self.max_calls = max(1, max_calls_per_window)
        self.max_ratio = max(0.0, min(1.0, max_ratio))
        self.window_s = max(1.0, window_s)
        self.ratio_min_samples = max(1, ratio_min_samples)
        self._events: deque[tuple[float, bool]] = deque()
        self._lock = threading.Lock()

    def _prune(self, now: float) -> None:
        cutoff = now - self.window_s
        while self._events and self._events[0][0] < cutoff:
            self._events.popleft()

    def allow(self, now: float | None = None) -> tuple[bool, str]:
        now = time.monotonic() if now is None else now
        with self._lock:
            self._prune(now)
            calls = sum(1 for _, escalated in self._events if escalated)
            total = len(self._events)
            if calls >= self.max_calls:
                return False, "max_calls"
            if total >= self.ratio_min_samples:
                projected = (calls + 1) / (total + 1)
                if projected > self.max_ratio:
                    return False, "max_ratio"
            return True, ""

    def record(self, escalated: bool, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        with self._lock:
            self._prune(now)
            self._events.append((now, escalated))

    def snapshot(self) -> dict[str, Any]:
        now = time.monotonic()
        with self._lock:
            self._prune(now)
            calls = sum(1 for _, escalated in self._events if escalated)
            total = len(self._events)
        return {"cloud_calls": calls, "requests": total, "max_calls": self.max_calls, "max_ratio": self.max_ratio}


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
    ) -> None:
        self.edge_backend = edge_backend
        self.config = config
        self.cloud_client = cloud_client
        self.confidence = confidence or ConfidenceEstimator()
        self.metrics = metrics or Metrics()
        self.outbox = outbox
        self.breaker = CircuitBreaker(config.circuit_failures, config.circuit_cooldown_s)
        self.budget = CloudBudget(
            config.max_cloud_calls_per_min, config.max_cloud_ratio, config.cloud_window_s
        )

    def infer(self, request: InferRequest) -> InferResponse:
        started = time.perf_counter()
        trace: list[str] = []
        expected_format = request.metadata.get("expected_format") or self.config.expected_format_by_task.get(
            request.task
        )

        edge_generation: Generation | None = None
        edge_report: ConfidenceReport | None = None
        edge_error: str | None = None
        edge_started = time.perf_counter()
        try:
            edge_generation = self.edge_backend.generate(request)
        except Exception as exc:  # noqa: BLE001 - any backend failure must degrade, not crash
            edge_error = str(exc)
            self.metrics.incr("edge_failures_total")
            trace.append(f"端侧推理失败：{exc}")
        edge_wall_ms = (time.perf_counter() - edge_started) * 1000

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
            return self._edge_response(request, edge_generation, edge_report, started, trace, expected_format)

        self.metrics.incr("escalation_total")
        self.metrics.incr(f"escalation_reason_{reason}")
        trace.append(f"触发升级：{reason}（阈值 {self.config.escalation_threshold:.2f}）")

        edge_response = self._edge_response(
            request, edge_generation, edge_report, started, trace, expected_format
        )
        if self.config.offline:
            self.budget.record(False)
            trace.append("当前为离线模式，保留端侧结果并写入待同步队列")
            return self._degrade(request, edge_response, reason, started, trace, queued=True)

        if self.cloud_client is None:
            self.budget.record(False)
            trace.append("未配置云端客户端，保留端侧结果")
            return self._degrade(request, edge_response, reason, started, trace, queued=True)

        remaining_ms = request.deadline_ms - (time.perf_counter() - started) * 1000
        skip_reason = ""
        if remaining_ms < 150:
            skip_reason = "deadline_exceeded"
        elif not self.breaker.allow():
            skip_reason = "circuit_open"
        else:
            allowed, budget_reason = self.budget.allow()
            if not allowed:
                skip_reason = f"budget_{budget_reason}"
        if skip_reason:
            self.budget.record(False)
            trace.append(f"跳过云端增强：{skip_reason}")
            return self._degrade(request, edge_response, reason, started, trace, queued=True)

        self.metrics.incr("cloud_attempt_total")
        self.budget.record(True)
        cloud_request = CloudRequest(
            prompt=request.prompt,
            request_id=request.request_id,
            task=request.task,
            draft=edge_generation.text if edge_generation else "",
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            deadline_ms=int(min(self.config.cloud_timeout_ms, max(150, remaining_ms))),
            metadata={**request.metadata, "edge_confidence": edge_report.score if edge_report else None},
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
        self.metrics.observe("request_latency_ms", (time.perf_counter() - started) * 1000)
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
        return InferResponse(
            request_id=request.request_id,
            text=cloud_response.text,
            source="cloud",
            confidence=cloud_report.score,
            latency_ms=(time.perf_counter() - started) * 1000,
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
            metadata={"edge_model": edge_generation.model_id if edge_generation else None},
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
        if report.score < self.config.escalation_threshold:
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
    ) -> InferResponse:
        if generation is None:
            return InferResponse(
                request_id=request.request_id,
                text="",
                source="none",
                confidence=0.0,
                latency_ms=(time.perf_counter() - started) * 1000,
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
            latency_ms=(time.perf_counter() - started) * 1000,
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
        edge_response.latency_ms = (time.perf_counter() - started) * 1000
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
                edge_text = str((item.get("edge_response") or {}).get("text", ""))
                cloud_request = CloudRequest(
                    prompt=request.prompt,
                    request_id=request.request_id,
                    task=request.task,
                    draft=edge_text,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    deadline_ms=self.cloud_timeout_ms,
                    metadata=request.metadata,
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
                response = self.cloud_client.enhance(cloud_request, timeout_ms=self.cloud_timeout_ms)
            except Exception as exc:  # noqa: BLE001 - keep the item pending
                self.outbox.update(
                    item["item_id"], attempts=attempts + 1, last_error=str(exc)[:300]
                )
                self.metrics.incr("outbox_flush_failure_total")
                break
            self.outbox.update(
                item["item_id"],
                status="done",
                attempts=attempts + 1,
                cloud_text=response.text,
                cloud_confidence=response.confidence,
                last_error="",
            )
            self.metrics.incr("outbox_flush_success_total")
            flushed += 1
        return flushed
