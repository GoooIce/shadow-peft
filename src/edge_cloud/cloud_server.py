"""Cloud enhancement service (云端大模型)."""

from __future__ import annotations

from typing import Any

from .backends import Backend, BackendError, Generation, build_backend
from .confidence import ConfidenceEstimator
from .config import CloudConfig
from .continuation import assemble_continuation, looks_like_duplicate_prefix
from .flywheel import FeedbackStore
from .http_utils import BaseJsonHandler, HttpError, RawResponse, ServiceServer
from .metrics import Metrics
from .protocol import CloudRequest, CloudResponse, FeedbackRequest, InferRequest
from .registry import AdapterRegistry, RegistryError

_CONTINUATION_INSTRUCTION = (
    "端侧模型已经生成了下面的推理前缀。必须从此前缀继续，不要重复已有步骤；"
    "不要重写前缀，只输出剩余部分。"
)
_REWRITE_INSTRUCTION = (
    "端侧模型在交接点之前生成了下面的推理前缀，它可能不完整或有误。"
    "请先纠正前缀，再给出完整、自包含的最终答案。"
)


def _continuation_prompt(prompt: str, partial_trace: str) -> str:
    return (
        f"{prompt}\n\n{_CONTINUATION_INSTRUCTION}\n\n"
        f"端侧前缀：\n{partial_trace}\n\n续写："
    )


def _rewrite_prompt(prompt: str, partial_trace: str) -> str:
    return (
        f"{prompt}\n\n{_REWRITE_INSTRUCTION}\n\n"
        f"端侧前缀：\n{partial_trace}\n\n完整答案："
    )


def _usage_output_tokens(raw: dict[str, Any]) -> int | None:
    usage = raw.get("usage")
    if not isinstance(usage, dict):
        return None
    for key in ("completion_tokens", "output_tokens"):
        value = usage.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
    return None


def _estimate_output_tokens(text: str) -> int:
    """Deterministic fallback when the backend reports no token usage.

    Uses three characters per token (the same heuristic as the mock backends),
    rounded up, and zero for empty text; the result only depends on the text.
    """

    stripped = text.strip()
    if not stripped:
        return 0
    return (len(stripped) + 2) // 3


def _output_token_count(generation: Generation) -> int:
    """Output token count, preferring backend usage over local estimates.

    Order: backend usage (``completion_tokens``/``output_tokens``), then the
    number of token logprobs the backend generated, then
    ``_estimate_output_tokens``.
    """

    raw = generation.raw if isinstance(generation.raw, dict) else {}
    usage_tokens = _usage_output_tokens(raw)
    if usage_tokens is not None:
        return usage_tokens
    if generation.token_logprobs:
        return len(generation.token_logprobs)
    return _estimate_output_tokens(generation.text)


class CloudService:
    def __init__(
        self,
        config: CloudConfig,
        *,
        backend: Backend | None = None,
        metrics: Metrics | None = None,
    ) -> None:
        self.config = config
        self.backend = backend or build_backend(config.backend)
        self.metrics = metrics or Metrics()
        self.confidence = ConfidenceEstimator()
        self.feedback = FeedbackStore(config.feedback_path)
        self.registry = AdapterRegistry(config.registry_path)
        self.server: ServiceServer | None = None

    def start(self, host: str = "127.0.0.1", port: int = 0) -> ServiceServer:
        self.server = ServiceServer(make_cloud_handler(self), host, port).start()
        self.metrics.incr("cloud_started_total")
        return self.server

    def stop(self) -> None:
        if self.server is not None:
            self.server.stop()
            self.server = None

    @property
    def url(self) -> str:
        if self.server is None:
            return ""
        return self.server.url

    # ------------------------------------------------------------------ inference
    def enhance(self, request: CloudRequest) -> dict[str, Any]:
        partial_trace = request.partial_trace.strip()
        continuing = bool(partial_trace)
        rewriting = continuing and request.allow_prefix_rewrite

        if rewriting:
            prompt = _rewrite_prompt(request.prompt, partial_trace)
        elif continuing:
            prompt = _continuation_prompt(request.prompt, partial_trace)
        else:
            prompt = request.prompt

        infer_request = InferRequest(
            prompt=prompt,
            request_id=request.request_id,
            task=request.task,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            deadline_ms=request.deadline_ms,
            require_cloud=True,
            metadata=request.metadata,
        )
        expected_format = request.metadata.get("expected_format")
        try:
            generation = self.backend.generate(
                infer_request,
                draft=None if continuing else (request.draft or None),
            )
        except BackendError as exc:
            self.metrics.incr("cloud_backend_failure_total")
            raise HttpError(502, f"云端模型调用失败：{exc}") from exc
        report = self.confidence.estimate(
            generation.text,
            generation.token_logprobs,
            finish_reason=generation.finish_reason,
            expected_format=expected_format,
        )

        text = generation.text
        continued_from_prefix = False
        corrected_prefix = False
        metadata: dict[str, Any] = {"flags": generation.raw}
        if rewriting:
            text = assemble_continuation(partial_trace, generation.text, corrected=True)
            corrected_prefix = True
            draft_note = "已按续写契约纠正端侧前缀"
        elif continuing:
            if looks_like_duplicate_prefix(partial_trace, generation.text):
                text = assemble_continuation(partial_trace, generation.text)
                metadata["continuation_mismatch"] = True
                draft_note = "云端重述了端侧前缀，按整段答案返回"
            else:
                continued_from_prefix = True
                draft_note = "已从端侧前缀续写"
        else:
            draft_note = "已复核端侧草稿" if request.draft else "无端侧草稿"

        self.metrics.incr("requests_total")
        self.metrics.observe("cloud_generation_latency_ms", generation.latency_ms)
        response = CloudResponse(
            request_id=request.request_id,
            text=text,
            confidence=report.score,
            model_id=generation.model_id,
            latency_ms=generation.latency_ms,
            confidence_signals=report.signals,
            finish_reason=generation.finish_reason,
            trace=[
                f"云端 {generation.model_id} 用时 {generation.latency_ms:.1f}ms",
                f"置信度 {report.score:.2f}（{report.method}）",
                draft_note,
            ],
            metadata=metadata,
            continued_from_prefix=continued_from_prefix,
            corrected_prefix=corrected_prefix,
            output_tokens=_output_token_count(generation),
        )
        return response.to_dict()

    def record_feedback(self, request: FeedbackRequest) -> dict[str, Any]:
        record = self.feedback.record(request)
        self.metrics.incr("feedback_total")
        if request.accepted:
            self.metrics.incr("feedback_accepted_total")
        if request.corrected_text.strip():
            self.metrics.incr("feedback_corrected_total")
        return record.to_dict()

    # ------------------------------------------------------------------ routes
    def handle_get(self, path: str) -> Any:
        if path == "/":
            return {
                "service": "edge-cloud cloud enhancer",
                "cloud_id": self.config.cloud_id,
                "model_id": self.backend.model_id,
                "endpoints": [
                    "/healthz",
                    "/v1/enhance",
                    "/v1/feedback",
                    "/v1/stats",
                    "/v1/registry",
                    "/v1/registry/current/{device_id}",
                ],
            }
        if path == "/healthz":
            return {
                "status": "ok",
                "cloud_id": self.config.cloud_id,
                "model_id": self.backend.model_id,
            }
        if path == "/v1/stats":
            return self.stats()
        if path == "/metrics":
            return RawResponse(
                self.metrics.prometheus(),
                "text/plain; version=0.0.4; charset=utf-8",
            )
        if path == "/v1/registry":
            return self.registry.state()
        if path.startswith("/v1/registry/current/"):
            device_id = path.rsplit("/", 1)[-1] or "edge-unknown"
            return self.registry.resolve(device_id)
        raise HttpError(404, f"未知路径: {path}")

    def handle_post(self, path: str, payload: dict[str, Any]) -> Any:
        if path == "/v1/enhance":
            return self.enhance(CloudRequest.from_dict(payload))
        if path == "/v1/feedback":
            return {"ok": True, "record": self.record_feedback(FeedbackRequest.from_dict(payload))}
        if path == "/v1/dataset/export":
            hard_only = bool(payload.get("hard_only", True))
            result = self.feedback.export_sft(self.config.dataset_path, hard_only=hard_only)
            self.metrics.incr("dataset_export_total")
            return result
        if path == "/v1/registry/publish":
            try:
                state = self.registry.publish(
                    str(payload.get("version", "")),
                    artifact=str(payload.get("artifact", "")),
                    notes=str(payload.get("notes", "")),
                    metrics=payload.get("metrics") or {},
                    canary_ratio=float(payload.get("canary_ratio", 0.1)),
                )
            except RegistryError as exc:
                raise HttpError(400, str(exc)) from exc
            self.metrics.incr("adapter_publish_total")
            return state
        if path == "/v1/registry/promote":
            try:
                state = self.registry.promote(str(payload.get("version", "")))
            except RegistryError as exc:
                raise HttpError(400, str(exc)) from exc
            self.metrics.incr("adapter_promote_total")
            return state
        if path == "/v1/registry/rollback":
            try:
                state = self.registry.rollback()
            except RegistryError as exc:
                raise HttpError(400, str(exc)) from exc
            self.metrics.incr("adapter_rollback_total")
            return state
        raise HttpError(404, f"未知路径: {path}")

    def stats(self) -> dict[str, Any]:
        registry_state = self.registry.state()
        return {
            "cloud_id": self.config.cloud_id,
            "model_id": self.backend.model_id,
            "metrics": self.metrics.snapshot(),
            "feedback": self.feedback.stats(),
            "registry": {
                "active": registry_state["active"],
                "canary": registry_state["canary"],
                "versions": sorted(registry_state["versions"].keys()),
            },
        }


def make_cloud_handler(service: CloudService) -> type[BaseJsonHandler]:
    class CloudHandler(BaseJsonHandler):
        def route_get(self, path: str) -> Any:
            return service.handle_get(path)

        def route_post(self, path: str, payload: dict[str, Any]) -> Any:
            return service.handle_post(path, payload)

    return CloudHandler
