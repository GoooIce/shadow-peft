"""Configuration objects for the edge agent and the cloud service."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class EdgeConfig:
    edge_id: str = "edge-01"
    backend: dict[str, Any] = field(default_factory=lambda: {"kind": "mock_edge"})
    cloud_url: str = "http://127.0.0.1:9100"
    agent_port: int = 8080
    cloud_timeout_ms: int = 8000
    escalation_threshold: float = 0.62
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

    def __post_init__(self) -> None:
        self.escalation_threshold = max(0.0, min(1.0, float(self.escalation_threshold)))
        self.max_cloud_ratio = max(0.0, min(1.0, float(self.max_cloud_ratio)))
        self.require_cloud_tasks = [str(task) for task in self.require_cloud_tasks]

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
