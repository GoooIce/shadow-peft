"""Configuration objects for the edge agent and the cloud service."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

HANDOFF_MODES = ("off", "legacy", "probe", "oracle")
MEMORY_MODES = ("off", "on")


def _unit_interval(value: Any, *, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须是数字") from exc
    if not 0.0 <= parsed <= 1.0:
        raise ValueError(f"{name} 必须在 [0, 1] 内")
    return parsed


def _non_negative_float(value: Any, *, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须是数字") from exc
    if not parsed >= 0.0:
        raise ValueError(f"{name} 必须非负")
    return parsed


def _non_negative_int(value: Any, *, name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须是整数") from exc
    if parsed < 0:
        raise ValueError(f"{name} 必须非负")
    return parsed


@dataclass(slots=True)
class EdgeConfig:
    """Edge agent configuration.

    ``memory_threshold`` defaults to 0.45, not the JitRL spec's 0.75: the
    shipped character n-gram TF-IDF scorer rates a typical synonym rewrite
    (e.g. "E104 真空报警" vs "E-104 真空压力不足") at ~0.47 and exact repeats
    at 1.0, so a 0.75 gate would suppress almost every real hit. Deployments
    can still set 0.75 (or any value in [0, 1]).
    """

    edge_id: str = "edge-01"
    backend: dict[str, Any] = field(default_factory=lambda: {"kind": "mock_edge"})
    cloud_url: str = "http://127.0.0.1:9100"
    agent_port: int = 8080
    cloud_timeout_ms: int = 8000
    escalation_threshold: float = 0.62
    escalation_threshold_by_task: dict[str, float] = field(default_factory=dict)
    require_cloud_tasks: list[str] = field(default_factory=lambda: ["quality_verdict"])
    max_cloud_calls_per_min: int = 60
    max_cloud_ratio: float = 0.6
    cloud_window_s: float = 60.0
    offline: bool = False
    circuit_failures: int = 3
    circuit_cooldown_s: float = 30.0
    outbox_path: str = "runs/edge_cloud/outbox.jsonl"
    feedback_path: str = "runs/edge_cloud/feedback.jsonl"
    adapter_version: str = "base"
    enable_outbox_flusher: bool = True
    outbox_interval_s: float = 10.0
    request_log_size: int = 512
    expected_format_by_task: dict[str, str] = field(
        default_factory=lambda: {"quality_verdict": "verdict", "process_calc": "number"}
    )
    handoff_mode: str = "legacy"
    handoff_threshold: float = 0.5
    handoff_threshold_by_task: dict[str, float] = field(default_factory=dict)
    max_boundary_wait_tokens: int = 12
    allow_prefix_rewrite: bool = False
    probe_path: str = ""
    handoff_policy_version: str = "v1"
    cost_per_edge_token: float = 0.0
    cost_per_cloud_input_token: float = 0.0
    cost_per_cloud_output_token: float = 0.0
    cost_per_cloud_call: float = 0.0
    cloud_rtt_ms: float = 0.0
    max_cost_per_min: float = 0.0
    # JitRL-lite experience memory (issue #39); see the class docstring for the
    # calibrated ``memory_threshold`` default.
    memory_mode: str = "off"
    memory_path: str = ""
    memory_top_k: int = 4
    memory_threshold: float = 0.45
    memory_alpha: float = 0.5
    memory_min_support: int = 2
    memory_ttl_s: int = 30 * 24 * 60 * 60
    memory_tenant: str = "default"
    memory_latency_budget_ms: float = 50.0

    def __post_init__(self) -> None:
        self.escalation_threshold = max(0.0, min(1.0, float(self.escalation_threshold)))
        if not isinstance(self.escalation_threshold_by_task, dict):
            raise ValueError("escalation_threshold_by_task 必须是对象")
        self.escalation_threshold_by_task = {
            str(task): _unit_interval(value, name=f"escalation_threshold_by_task.{task}")
            for task, value in self.escalation_threshold_by_task.items()
        }
        self.max_cloud_ratio = max(0.0, min(1.0, float(self.max_cloud_ratio)))
        self.require_cloud_tasks = [str(task) for task in self.require_cloud_tasks]
        mode = str(self.handoff_mode).strip().lower()
        if mode not in HANDOFF_MODES:
            raise ValueError(f"handoff_mode 必须是 {'|'.join(HANDOFF_MODES)} 之一")
        self.handoff_mode = mode
        self.handoff_threshold = _unit_interval(
            self.handoff_threshold, name="handoff_threshold"
        )
        if not isinstance(self.handoff_threshold_by_task, dict):
            raise ValueError("handoff_threshold_by_task 必须是对象")
        self.handoff_threshold_by_task = {
            str(task): _unit_interval(value, name=f"handoff_threshold_by_task.{task}")
            for task, value in self.handoff_threshold_by_task.items()
        }
        self.max_boundary_wait_tokens = _non_negative_int(
            self.max_boundary_wait_tokens, name="max_boundary_wait_tokens"
        )
        self.allow_prefix_rewrite = bool(self.allow_prefix_rewrite)
        self.probe_path = str(self.probe_path or "")
        self.handoff_policy_version = str(self.handoff_policy_version or "v1")
        self.cost_per_edge_token = _non_negative_float(
            self.cost_per_edge_token, name="cost_per_edge_token"
        )
        self.cost_per_cloud_input_token = _non_negative_float(
            self.cost_per_cloud_input_token, name="cost_per_cloud_input_token"
        )
        self.cost_per_cloud_output_token = _non_negative_float(
            self.cost_per_cloud_output_token, name="cost_per_cloud_output_token"
        )
        self.cost_per_cloud_call = _non_negative_float(
            self.cost_per_cloud_call, name="cost_per_cloud_call"
        )
        self.cloud_rtt_ms = _non_negative_float(self.cloud_rtt_ms, name="cloud_rtt_ms")
        self.max_cost_per_min = _non_negative_float(
            self.max_cost_per_min, name="max_cost_per_min"
        )
        memory_mode = str(self.memory_mode).strip().lower()
        if memory_mode not in MEMORY_MODES:
            raise ValueError(f"memory_mode 必须是 {'|'.join(MEMORY_MODES)} 之一")
        self.memory_mode = memory_mode
        self.memory_path = str(self.memory_path or "")
        self.memory_top_k = _non_negative_int(self.memory_top_k, name="memory_top_k")
        self.memory_threshold = _unit_interval(
            self.memory_threshold, name="memory_threshold"
        )
        self.memory_alpha = _non_negative_float(self.memory_alpha, name="memory_alpha")
        self.memory_min_support = _non_negative_int(
            self.memory_min_support, name="memory_min_support"
        )
        self.memory_ttl_s = _non_negative_int(self.memory_ttl_s, name="memory_ttl_s")
        self.memory_tenant = str(self.memory_tenant or "default")
        self.memory_latency_budget_ms = _non_negative_float(
            self.memory_latency_budget_ms, name="memory_latency_budget_ms"
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> EdgeConfig:
        allowed = {field_name for field_name in cls.__dataclass_fields__}
        return cls(**{key: value for key, value in payload.items() if key in allowed})

    def to_dict(self) -> dict[str, Any]:
        return {field_name: getattr(self, field_name) for field_name in self.__dataclass_fields__}


@dataclass(slots=True)
class CloudConfig:
    cloud_id: str = "cloud-01"
    port: int = 9100
    backend: dict[str, Any] = field(default_factory=lambda: {"kind": "mock_cloud"})
    feedback_path: str = "runs/edge_cloud/cloud_feedback.jsonl"
    registry_path: str = "runs/edge_cloud/registry.json"
    dataset_path: str = "runs/edge_cloud/sft_hard_cases.jsonl"
    system_prompt: str = (
        "你是生产线工艺与质量专家。回答要给出结论、依据和可执行步骤；"
        "涉及数值时写明计算口径；不确定时明确要求补充上下文。"
    )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CloudConfig:
        allowed = {field_name for field_name in cls.__dataclass_fields__}
        return cls(**{key: value for key, value in payload.items() if key in allowed})

    def to_dict(self) -> dict[str, Any]:
        return {field_name: getattr(self, field_name) for field_name in self.__dataclass_fields__}


def load_config(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("配置文件顶层必须是 JSON 对象")
    return payload
