"""Edge agent HTTP service (产线端)."""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import replace
from pathlib import Path
from typing import Any

from .backends import Backend, build_backend
from .config import EdgeConfig
from .flywheel import FeedbackStore
from .http_utils import BaseJsonHandler, HttpError, RawResponse, ServiceServer
from .metrics import Metrics
from .outbox import Outbox, OutboxItem
from .protocol import (
    CloudRequest,
    CloudResponse,
    FeedbackRequest,
    InferRequest,
    InferResponse,
)
from .router import CloudClient, CloudError, OutboxFlusher, Router

_CONSOLE_PATH = Path(__file__).parent / "web" / "console.html"


class _HandoffPolicyOutbox(Outbox):
    """Outbox that records the handoff policy active when an item is queued.

    ``OutboxFlusher`` (router.py) rebuilds the cloud request from the stored
    item alone, so it cannot know which policy the edge was running. Stamping
    both the request and the cached response keeps replay auditable and lets the
    replay client below restore ``allow_prefix_rewrite`` without touching
    router.py.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        policy_version: str,
        allow_prefix_rewrite: bool,
    ) -> None:
        super().__init__(path)
        self.policy_version = str(policy_version)
        self.allow_prefix_rewrite = bool(allow_prefix_rewrite)

    def enqueue(
        self,
        request: InferRequest,
        edge_response: InferResponse,
        reason: str,
    ) -> OutboxItem:
        request_metadata = dict(request.metadata)
        request_metadata.setdefault("handoff_policy_version", self.policy_version)
        request_metadata.setdefault("allow_prefix_rewrite", self.allow_prefix_rewrite)
        response_metadata = dict(edge_response.metadata)
        response_metadata.setdefault("handoff_policy_version", self.policy_version)
        response_metadata.setdefault("allow_prefix_rewrite", self.allow_prefix_rewrite)
        return super().enqueue(
            replace(request, metadata=request_metadata),
            replace(edge_response, metadata=response_metadata),
            reason,
        )


class _ReplayPolicyCloudClient:
    """Restore the enqueue-time handoff policy on replayed cloud requests.

    ``OutboxFlusher`` copies the stored request metadata into
    ``CloudRequest.metadata`` but always builds the request with the protocol
    default ``allow_prefix_rewrite=False``. This adapter reads the value
    recorded at enqueue time so replay keeps the original continuation
    semantics; entries queued by older versions keep the default.
    """

    def __init__(self, inner: CloudClient) -> None:
        self._inner = inner

    def enhance(
        self, request: CloudRequest, *, timeout_ms: int | None = None
    ) -> CloudResponse:
        rewrite = request.metadata.get("allow_prefix_rewrite")
        if isinstance(rewrite, bool):
            request.allow_prefix_rewrite = rewrite
        return self._inner.enhance(request, timeout_ms=timeout_ms)


class EdgeService:
    def __init__(
        self,
        config: EdgeConfig,
        *,
        backend: Backend | None = None,
        cloud_client: CloudClient | None = None,
        metrics: Metrics | None = None,
    ) -> None:
        self.config = config
        self.backend = backend or build_backend(config.backend)
        self.cloud = cloud_client or CloudClient(config.cloud_url)
        self.metrics = metrics or Metrics()
        self.outbox = _HandoffPolicyOutbox(
            config.outbox_path,
            policy_version=config.handoff_policy_version,
            allow_prefix_rewrite=config.allow_prefix_rewrite,
        )
        self.feedback = FeedbackStore(config.feedback_path)
        self.router = Router(
            self.backend,
            config,
            cloud_client=self.cloud,
            metrics=self.metrics,
            outbox=self.outbox,
        )
        self._responses: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._requests: OrderedDict[str, InferRequest] = OrderedDict()
        self._cache_lock = threading.Lock()
        self.flusher: OutboxFlusher | None = None
        if config.enable_outbox_flusher:
            self.flusher = OutboxFlusher(
                self.outbox,
                _ReplayPolicyCloudClient(self.cloud),
                interval_s=config.outbox_interval_s,
                metrics=self.metrics,
                cloud_timeout_ms=config.cloud_timeout_ms,
            )
        self.server: ServiceServer | None = None
        self.adapter_channel = "local"

    # ------------------------------------------------------------------ lifecycle
    def start(self, host: str = "127.0.0.1", port: int = 0) -> ServiceServer:
        handler = make_edge_handler(self)
        self.server = ServiceServer(handler, host, port).start()
        if self.flusher is not None:
            self.flusher.start()
        self.refresh_adapter()
        self.metrics.incr("edge_started_total")
        return self.server

    def stop(self) -> None:
        if self.flusher is not None:
            self.flusher.stop()
        if self.server is not None:
            self.server.stop()
            self.server = None

    @property
    def url(self) -> str:
        if self.server is None:
            return ""
        return self.server.url

    # ------------------------------------------------------------------ inference
    def infer(self, request: InferRequest) -> dict[str, Any]:
        response = self.router.infer(request)
        with self._cache_lock:
            self._requests[request.request_id] = request
            self._responses[request.request_id] = response.to_dict()
            while len(self._responses) > self.config.request_log_size:
                old_id, _ = self._responses.popitem(last=False)
                self._requests.pop(old_id, None)
        return response.to_dict()

    def record_feedback(self, request: FeedbackRequest) -> dict[str, Any]:
        with self._cache_lock:
            cached_response = dict(self._responses.get(request.request_id, {}))
            cached_request = self._requests.get(request.request_id)
        if cached_request is not None:
            request.prompt = request.prompt or cached_request.prompt
            if request.task == "general":
                request.task = cached_request.task
        if cached_response:
            request.edge_text = request.edge_text or str(cached_response.get("edge_text") or "")
            if not request.cloud_text and cached_response.get("source") == "cloud":
                request.cloud_text = str(cached_response.get("text") or "")
            request.source = request.source or str(cached_response.get("source") or "")
            request.confidence = request.confidence or float(
                cached_response.get("confidence") or 0.0
            )
            request.escalation_reason = request.escalation_reason or cached_response.get(
                "escalation_reason"
            )
            request.adapter_version = request.adapter_version or str(
                cached_response.get("adapter_version") or ""
            )
            request.final_text = request.final_text or str(cached_response.get("text") or "")
            request.handoff_index = (
                request.handoff_index
                if request.handoff_index is not None
                else cached_response.get("handoff_index")
            )
            request.handoff_reason = request.handoff_reason or cached_response.get(
                "handoff_reason"
            )
            request.prefix_trust = (
                request.prefix_trust
                if request.prefix_trust is not None
                else cached_response.get("prefix_trust")
            )
            request.cost_estimate = request.cost_estimate or cached_response.get(
                "cost_estimate"
            )
            cached_metadata = cached_response.get("metadata")
            if isinstance(cached_metadata, dict):
                request.continued_from_prefix = request.continued_from_prefix or bool(
                    cached_metadata.get("continued_from_prefix")
                )
                request.corrected_prefix = request.corrected_prefix or bool(
                    cached_metadata.get("corrected_prefix")
                )
        if request.corrected_text.strip():
            request.final_text = request.corrected_text.strip()
        elif not request.final_text.strip():
            request.final_text = request.cloud_text or request.edge_text

        record = self.feedback.record(request)
        self.metrics.incr("feedback_total")
        if request.accepted:
            self.metrics.incr("feedback_accepted_total")
        else:
            self.metrics.incr("feedback_rejected_total")
        if request.corrected_text.strip():
            self.metrics.incr("feedback_corrected_total")

        # Upload to the cloud flywheel best-effort; local copy is authoritative offline.
        try:
            self.cloud.feedback(request.to_dict(), timeout_ms=3000)
            self.metrics.incr("feedback_forwarded_total")
        except CloudError:
            self.metrics.incr("feedback_forward_failure_total")
        return record.to_dict()

    # ------------------------------------------------------------------ registry
    def refresh_adapter(self) -> dict[str, Any]:
        try:
            data = self.cloud.get_json(
                f"/v1/registry/current/{self.config.edge_id}", timeout_ms=2000
            )
        except CloudError:
            self.adapter_channel = "local"
            return {"version": self.config.adapter_version, "channel": "local"}
        version = data.get("version") if isinstance(data, dict) else None
        if version:
            self.config.adapter_version = str(version)
            self.adapter_channel = str(data.get("channel", "stable"))
        return {"version": self.config.adapter_version, "channel": self.adapter_channel}

    # ------------------------------------------------------------------ routes
    def handle_get(self, path: str) -> Any:
        if path in {"/", "/console"}:
            return self._console()
        if path == "/healthz":
            return {
                "status": "ok",
                "edge_id": self.config.edge_id,
                "adapter_version": self.config.adapter_version,
                "adapter_channel": self.adapter_channel,
                "offline": self.config.offline,
                "cloud_url": self.config.cloud_url,
                "handoff_mode": self.config.handoff_mode,
                "handoff_policy_version": self.config.handoff_policy_version,
                "handoff_threshold": self.config.handoff_threshold,
            }
        if path == "/v1/stats":
            return self.stats()
        if path == "/metrics":
            return RawResponse(
                self.metrics.prometheus(),
                "text/plain; version=0.0.4; charset=utf-8",
            )
        if path == "/v1/outbox":
            items = self.outbox.items()
            return {"items": items[-50:], "stats": self.outbox.stats()}
        if path == "/v1/cloud/registry":
            return self._proxy("GET", "/v1/registry")
        if path.startswith("/v1/cloud/registry/current/"):
            device_id = path.rsplit("/", 1)[-1] or self.config.edge_id
            return self._proxy("GET", f"/v1/registry/current/{device_id}")
        if path == "/v1/cloud/stats":
            return self._proxy("GET", "/v1/stats")
        raise HttpError(404, f"未知路径: {path}")

    def handle_post(self, path: str, payload: dict[str, Any]) -> Any:
        if path == "/v1/infer":
            request = InferRequest.from_dict(payload)
            return self.infer(request)
        if path == "/v1/feedback":
            request = FeedbackRequest.from_dict(payload)
            return {"ok": True, "record": self.record_feedback(request)}
        if path == "/v1/admin/config":
            return self.update_config(payload)
        if path == "/v1/admin/refresh-adapter":
            return {"ok": True, "adapter": self.refresh_adapter()}
        if path == "/v1/outbox/flush":
            flushed = self.flusher.flush_once() if self.flusher is not None else 0
            return {"flushed": flushed, "outbox": self.outbox.stats()}
        if path == "/v1/cloud/dataset/export":
            return self._proxy("POST", "/v1/dataset/export", payload)
        if path == "/v1/cloud/registry/publish":
            return self._proxy(
                "POST", "/v1/registry/publish", self._with_handoff_metrics(payload)
            )
        if path == "/v1/cloud/registry/promote":
            return self._proxy("POST", "/v1/registry/promote", payload)
        if path == "/v1/cloud/registry/rollback":
            return self._proxy("POST", "/v1/registry/rollback", payload)
        raise HttpError(404, f"未知路径: {path}")

    def stats(self) -> dict[str, Any]:
        return {
            "edge_id": self.config.edge_id,
            "adapter_version": self.config.adapter_version,
            "adapter_channel": self.adapter_channel,
            "offline": self.config.offline,
            "threshold": self.config.escalation_threshold,
            "handoff_mode": self.config.handoff_mode,
            "handoff_policy_version": self.config.handoff_policy_version,
            "handoff_threshold": self.config.handoff_threshold,
            "allow_prefix_rewrite": self.config.allow_prefix_rewrite,
            "metrics": self.metrics.snapshot(),
            "outbox": self.outbox.stats(),
            "budget": self.router.budget.snapshot(),
            "breaker": self.router.breaker.snapshot(),
            "feedback": self.feedback.stats(),
        }

    def update_config(self, payload: dict[str, Any]) -> dict[str, Any]:
        if "escalation_threshold" in payload:
            value = float(payload["escalation_threshold"])
            if not 0.0 <= value <= 1.0:
                raise HttpError(400, "escalation_threshold 必须在 [0, 1] 内")
            self.config.escalation_threshold = value
        if "offline" in payload:
            self.config.offline = bool(payload["offline"])
        if "require_cloud_tasks" in payload:
            tasks = payload["require_cloud_tasks"]
            if not isinstance(tasks, list):
                raise HttpError(400, "require_cloud_tasks 必须是数组")
            self.config.require_cloud_tasks = [str(task) for task in tasks]
        if "max_cloud_calls_per_min" in payload:
            value = max(1, int(payload["max_cloud_calls_per_min"]))
            self.config.max_cloud_calls_per_min = value
            self.router.budget.max_calls = value
        if "max_cloud_ratio" in payload:
            value = float(payload["max_cloud_ratio"])
            if not 0.0 <= value <= 1.0:
                raise HttpError(400, "max_cloud_ratio 必须在 [0, 1] 内")
            self.config.max_cloud_ratio = value
            self.router.budget.max_ratio = value
        if "handoff_mode" in payload or "handoff_threshold" in payload:
            # Validate both values together so a rejected payload never leaves
            # the service half-updated.
            try:
                validated = EdgeConfig(
                    handoff_mode=payload.get("handoff_mode", self.config.handoff_mode),
                    handoff_threshold=payload.get(
                        "handoff_threshold", self.config.handoff_threshold
                    ),
                )
            except (TypeError, ValueError) as exc:
                raise HttpError(400, f"交接配置不合法：{exc}") from exc
            self.config.handoff_mode = validated.handoff_mode
            self.config.handoff_threshold = validated.handoff_threshold
        return {"ok": True, "config": self.config.to_dict()}

    def _with_handoff_metrics(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Attach the active handoff policy to registry metrics.

        Adapter metrics then distinguish "the model got better" from "the
        handoff policy changed" without requiring a registry schema change.
        """
        enriched = dict(payload)
        metrics = enriched.get("metrics")
        if metrics is None:
            metrics = {}
        if not isinstance(metrics, dict):
            return enriched
        metrics = dict(metrics)
        metrics.setdefault("handoff_policy_version", self.config.handoff_policy_version)
        metrics.setdefault("handoff_mode", self.config.handoff_mode)
        enriched["metrics"] = metrics
        return enriched

    def _proxy(self, method: str, path: str, payload: dict[str, Any] | None = None) -> Any:
        try:
            if method == "GET":
                return self.cloud.get_json(path, timeout_ms=5000)
            return self.cloud.post_json(path, payload or {}, timeout_ms=10000)
        except CloudError as exc:
            raise HttpError(502, f"云端管理接口不可用：{exc}") from exc

    def _console(self) -> RawResponse:
        if not _CONSOLE_PATH.exists():
            return RawResponse(
                "<h1>EdgeCloud console</h1><p>console.html 未安装。</p>",
                "text/html; charset=utf-8",
            )
        return RawResponse(_CONSOLE_PATH.read_text(encoding="utf-8"), "text/html; charset=utf-8")


def make_edge_handler(service: EdgeService) -> type[BaseJsonHandler]:
    class EdgeHandler(BaseJsonHandler):
        def route_get(self, path: str) -> Any:
            return service.handle_get(path)

        def route_post(self, path: str, payload: dict[str, Any]) -> Any:
            return service.handle_post(path, payload)

    return EdgeHandler
