"""Wire protocol for the edge-cloud collaborative inference system.

All payloads are plain JSON objects so that the edge agent (产线端) and the
cloud service (云端) can be implemented in different languages later without
changing the contract. The dataclasses here are the single source of truth for
field names and validation rules.
"""

from __future__ import annotations

import json
import math
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any


class ProtocolError(ValueError):
    """Raised when a request payload does not satisfy the wire contract."""


def _bounded_int(value: Any, *, default: int, low: int, high: int, name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"{name} 必须是整数") from exc
    if not low <= parsed <= high:
        raise ProtocolError(f"{name} 必须在 [{low}, {high}] 内")
    return parsed


def _bounded_float(value: Any, *, default: float, low: float, high: float, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"{name} 必须是数字") from exc
    if not low <= parsed <= high:
        raise ProtocolError(f"{name} 必须在 [{low}, {high}] 内")
    return parsed


def _optional_non_negative_int(value: Any, *, name: str) -> int | None:
    """Accept only a JSON integer: not a bool, a float or a string.

    ``bool`` is an ``int`` subclass and JSON parsers turn ``1.5`` into a float;
    silently truncating either hides a producer bug, so both are rejected.
    """

    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProtocolError(f"{name} 必须是整数（不接受布尔值、小数或字符串）")
    if value < 0:
        raise ProtocolError(f"{name} 必须非负")
    return value


def _optional_unit_float(value: Any, *, name: str) -> float | None:
    """Accept a finite real number in [0, 1]; bools and strings are rejected."""

    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProtocolError(f"{name} 必须是数字（不接受布尔值或字符串）")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ProtocolError(f"{name} 必须是有限数字")
    if not 0.0 <= parsed <= 1.0:
        raise ProtocolError(f"{name} 必须在 [0, 1] 内")
    return parsed


def _optional_bool(value: Any, *, name: str) -> bool:
    if value is None:
        return False
    if not isinstance(value, bool):
        raise ProtocolError(f"{name} 必须是布尔值")
    return value


def _optional_cost_estimate(value: Any, *, name: str) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ProtocolError(f"{name} 必须是对象")
    return dict(value)


@dataclass(slots=True)
class InferRequest:
    """A production-line inference request handled first by the edge model."""

    prompt: str
    request_id: str = ""
    task: str = "general"
    max_tokens: int = 256
    temperature: float = 0.0
    deadline_ms: int = 8000
    require_cloud: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise ProtocolError("prompt 不能为空")
        if len(self.prompt) > 8000:
            raise ProtocolError("prompt 超过 8000 字符上限")
        if not self.request_id:
            self.request_id = uuid.uuid4().hex
        if not isinstance(self.task, str) or not self.task.strip():
            self.task = "general"
        self.task = self.task.strip()[:64]
        self.max_tokens = _bounded_int(
            self.max_tokens, default=256, low=1, high=4096, name="max_tokens"
        )
        self.temperature = _bounded_float(
            self.temperature, default=0.0, low=0.0, high=2.0, name="temperature"
        )
        self.deadline_ms = _bounded_int(
            self.deadline_ms, default=8000, low=50, high=120000, name="deadline_ms"
        )
        if not isinstance(self.metadata, dict):
            raise ProtocolError("metadata 必须是对象")

    @classmethod
    def from_dict(cls, payload: Any) -> InferRequest:
        if not isinstance(payload, dict):
            raise ProtocolError("请求体必须是 JSON 对象")
        return cls(
            prompt=payload.get("prompt", ""),
            request_id=str(payload.get("request_id", "") or ""),
            task=str(payload.get("task", "general")),
            max_tokens=payload.get("max_tokens", 256),
            temperature=payload.get("temperature", 0.0),
            deadline_ms=payload.get("deadline_ms", 8000),
            require_cloud=bool(payload.get("require_cloud", False)),
            metadata=payload.get("metadata") or {},
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ConfidenceReport:
    """Confidence of one model output, with the signals that produced it."""

    score: float
    method: str
    signals: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class InferResponse:
    """Result returned to the production-line caller."""

    request_id: str
    text: str
    source: str
    confidence: float
    latency_ms: float
    model_id: str
    adapter_version: str = ""
    escalated: bool = False
    escalation_reason: str | None = None
    edge_text: str | None = None
    edge_confidence: float | None = None
    edge_latency_ms: float | None = None
    cloud_latency_ms: float | None = None
    degraded: bool = False
    trace: list[str] = field(default_factory=list)
    confidence_signals: dict[str, float] = field(default_factory=dict)
    finish_reason: str = "stop"
    created_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)
    handoff_index: int | None = None
    edge_tokens_before_handoff: int | None = None
    handoff_reason: str | None = None
    prefix_trust: float | None = None
    cloud_output_tokens: int = 0
    cost_estimate: dict[str, Any] | None = None
    memory_hit: bool = False
    memory_entry_ids: list[str] = field(default_factory=list)
    memory_score: float | None = None
    memory_reranked: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class CloudRequest:
    """Enhancement request sent from the edge agent to the cloud."""

    prompt: str
    request_id: str = ""
    task: str = "general"
    draft: str = ""
    max_tokens: int = 256
    temperature: float = 0.0
    deadline_ms: int = 15000
    metadata: dict[str, Any] = field(default_factory=dict)
    partial_trace: str = ""
    handoff_index: int | None = None
    edge_tokens_before_handoff: int | None = None
    handoff_reason: str | None = None
    prefix_trust: float | None = None
    allow_prefix_rewrite: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise ProtocolError("prompt 不能为空")
        if not self.request_id:
            self.request_id = uuid.uuid4().hex
        self.max_tokens = _bounded_int(
            self.max_tokens, default=256, low=1, high=4096, name="max_tokens"
        )
        self.temperature = _bounded_float(
            self.temperature, default=0.0, low=0.0, high=2.0, name="temperature"
        )
        self.deadline_ms = _bounded_int(
            self.deadline_ms, default=15000, low=50, high=120000, name="deadline_ms"
        )
        if not isinstance(self.metadata, dict):
            raise ProtocolError("metadata 必须是对象")
        if not isinstance(self.partial_trace, str):
            self.partial_trace = str(self.partial_trace)
        self.handoff_index = _optional_non_negative_int(self.handoff_index, name="handoff_index")
        self.edge_tokens_before_handoff = _optional_non_negative_int(
            self.edge_tokens_before_handoff, name="edge_tokens_before_handoff"
        )
        self.prefix_trust = _optional_unit_float(self.prefix_trust, name="prefix_trust")
        if self.handoff_reason is not None and not isinstance(self.handoff_reason, str):
            self.handoff_reason = str(self.handoff_reason)
        self.allow_prefix_rewrite = bool(self.allow_prefix_rewrite)

    @classmethod
    def from_dict(cls, payload: Any) -> CloudRequest:
        if not isinstance(payload, dict):
            raise ProtocolError("请求体必须是 JSON 对象")
        return cls(
            prompt=payload.get("prompt", ""),
            request_id=str(payload.get("request_id", "") or ""),
            task=str(payload.get("task", "general")),
            draft=str(payload.get("draft", "") or ""),
            max_tokens=payload.get("max_tokens", 256),
            temperature=payload.get("temperature", 0.0),
            deadline_ms=payload.get("deadline_ms", 15000),
            metadata=payload.get("metadata") or {},
            partial_trace=str(payload.get("partial_trace", "") or ""),
            handoff_index=payload.get("handoff_index"),
            edge_tokens_before_handoff=payload.get("edge_tokens_before_handoff"),
            handoff_reason=payload.get("handoff_reason"),
            prefix_trust=payload.get("prefix_trust"),
            allow_prefix_rewrite=bool(payload.get("allow_prefix_rewrite", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class CloudResponse:
    """Enhanced result produced by the large cloud model."""

    request_id: str
    text: str
    confidence: float
    model_id: str
    latency_ms: float
    confidence_signals: dict[str, float] = field(default_factory=dict)
    finish_reason: str = "stop"
    trace: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    continued_from_prefix: bool = False
    corrected_prefix: bool = False
    output_tokens: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class FeedbackRequest:
    """Production-line outcome used by the data flywheel (数据飞轮)."""

    request_id: str
    accepted: bool
    prompt: str = ""
    task: str = "general"
    edge_text: str = ""
    cloud_text: str = ""
    final_text: str = ""
    corrected_text: str = ""
    source: str = ""
    confidence: float = 0.0
    escalation_reason: str | None = None
    adapter_version: str = ""
    note: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    handoff_index: int | None = None
    handoff_reason: str | None = None
    prefix_trust: float | None = None
    continued_from_prefix: bool = False
    corrected_prefix: bool = False
    cost_estimate: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, payload: Any) -> FeedbackRequest:
        if not isinstance(payload, dict):
            raise ProtocolError("请求体必须是 JSON 对象")
        request_id = str(payload.get("request_id", "") or "").strip()
        if not request_id:
            raise ProtocolError("request_id 不能为空")
        accepted = payload.get("accepted")
        if not isinstance(accepted, bool):
            raise ProtocolError("accepted 必须是布尔值")
        metadata = payload.get("metadata") or {}
        if not isinstance(metadata, dict):
            raise ProtocolError("metadata 必须是对象")
        handoff_reason = payload.get("handoff_reason")
        if handoff_reason is not None and not isinstance(handoff_reason, str):
            raise ProtocolError("handoff_reason 必须是字符串")
        return cls(
            request_id=request_id,
            accepted=accepted,
            prompt=str(payload.get("prompt", "") or ""),
            task=str(payload.get("task", "general") or "general"),
            edge_text=str(payload.get("edge_text", "") or ""),
            cloud_text=str(payload.get("cloud_text", "") or ""),
            final_text=str(payload.get("final_text", "") or ""),
            corrected_text=str(payload.get("corrected_text", "") or ""),
            source=str(payload.get("source", "") or ""),
            confidence=float(payload.get("confidence", 0.0) or 0.0),
            escalation_reason=payload.get("escalation_reason"),
            adapter_version=str(payload.get("adapter_version", "") or ""),
            note=str(payload.get("note", "") or ""),
            metadata=metadata,
            handoff_index=_optional_non_negative_int(
                payload.get("handoff_index"), name="handoff_index"
            ),
            handoff_reason=handoff_reason,
            prefix_trust=_optional_unit_float(
                payload.get("prefix_trust"), name="prefix_trust"
            ),
            continued_from_prefix=_optional_bool(
                payload.get("continued_from_prefix", False),
                name="continued_from_prefix",
            ),
            corrected_prefix=_optional_bool(
                payload.get("corrected_prefix", False), name="corrected_prefix"
            ),
            cost_estimate=_optional_cost_estimate(
                payload.get("cost_estimate"), name="cost_estimate"
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def loads(body: bytes | str) -> Any:
    if isinstance(body, bytes):
        try:
            body = body.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ProtocolError("请求体必须是 UTF-8 编码") from exc
    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"JSON 解析失败：{exc.msg}") from exc
