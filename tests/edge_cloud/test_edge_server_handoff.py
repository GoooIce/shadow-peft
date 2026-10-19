from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from edge_cloud.backends import Generation, MockEdgeBackend
from edge_cloud.config import EdgeConfig
from edge_cloud.edge_server import EdgeService
from edge_cloud.http_utils import HttpError
from edge_cloud.protocol import CloudRequest, CloudResponse, InferRequest, InferResponse


class FakeCloud:
    """Cloud double that records enhancement requests and proxied payloads."""

    def __init__(self, text: str = "后续。") -> None:
        self.text = text
        self.calls: list[CloudRequest] = []
        self.posted: list[tuple[str, dict[str, Any]]] = []

    def enhance(
        self, request: CloudRequest, *, timeout_ms: int | None = None
    ) -> CloudResponse:
        self.calls.append(request)
        return CloudResponse(
            request_id=request.request_id,
            text=self.text,
            confidence=0.9,
            model_id="fake-cloud",
            latency_ms=8.0,
            continued_from_prefix=True,
            output_tokens=3,
        )

    def get_json(self, path: str, *, timeout_ms: int = 3000) -> Any:
        return {}

    def post_json(
        self, path: str, payload: dict[str, Any], *, timeout_ms: int = 5000
    ) -> Any:
        self.posted.append((path, payload))
        return {"ok": True}

    def feedback(self, payload: dict[str, Any], *, timeout_ms: int = 3000) -> Any:
        return {"ok": True}


class StreamingStub:
    """Minimal streaming backend; ``generate_until`` must not be reached."""

    model_id = "streaming-stub"

    def __init__(self) -> None:
        self.generate_calls = 0
        self.generate_until_calls = 0

    def generate(self, request: InferRequest, *, draft: str | None = None) -> Generation:
        del request, draft
        self.generate_calls += 1
        return Generation(
            text="端侧完整答案。",
            token_logprobs=[-0.05] * 40,
            model_id=self.model_id,
            latency_ms=1.0,
        )

    def generate_until(
        self,
        request: InferRequest,
        policy: Any,
        **kwargs: Any,
    ) -> Generation:
        del request, policy, kwargs
        self.generate_until_calls += 1
        raise AssertionError("probe 不可用时不应进入流式生成")


def edge_service(
    tmp_path: Path, *, backend: Any = None, cloud: FakeCloud | None = None, **overrides: Any
) -> EdgeService:
    defaults = dict(
        backend={"kind": "mock_edge"},
        cloud_url="http://127.0.0.1:1",
        outbox_path=str(tmp_path / "outbox.jsonl"),
        feedback_path=str(tmp_path / "feedback.jsonl"),
        enable_outbox_flusher=False,
        handoff_mode="probe",
        handoff_threshold=0.44,
        handoff_policy_version="v7",
        max_cost_per_min=2.5,
    )
    defaults.update(overrides)
    return EdgeService(
        EdgeConfig(**defaults),
        backend=backend or MockEdgeBackend(),
        cloud_client=cloud or FakeCloud(),
    )


def handoff_response(request_id: str) -> InferResponse:
    return InferResponse(
        request_id=request_id,
        text="判定：返修。",
        source="edge",
        confidence=0.5,
        latency_ms=5.0,
        model_id="scripted-edge",
        degraded=True,
        edge_text="判定：返修。",
        handoff_index=6,
        edge_tokens_before_handoff=6,
        handoff_reason="low_prefix_trust",
        prefix_trust=0.2,
    )


def test_healthz_and_stats_expose_handoff_policy_and_cost_window(tmp_path: Path) -> None:
    service = edge_service(tmp_path)

    health = service.handle_get("/healthz")
    assert health["status"] == "ok"
    assert health["handoff_mode"] == "probe"
    assert health["handoff_policy_version"] == "v7"
    assert health["handoff_threshold"] == pytest.approx(0.44)

    stats = service.handle_get("/v1/stats")
    assert stats["handoff_mode"] == "probe"
    assert stats["handoff_policy_version"] == "v7"
    assert stats["handoff_threshold"] == pytest.approx(0.44)
    assert stats["budget"]["max_cost_per_min"] == pytest.approx(2.5)
    assert stats["budget"]["window_cost"] == 0.0


def test_admin_config_hot_switches_handoff_mode_and_threshold(tmp_path: Path) -> None:
    service = edge_service(tmp_path)

    result = service.handle_post(
        "/v1/admin/config", {"handoff_mode": "legacy", "handoff_threshold": 0.8}
    )

    assert result["ok"] is True
    assert result["config"]["handoff_mode"] == "legacy"
    assert result["config"]["handoff_threshold"] == pytest.approx(0.8)
    assert service.config.handoff_mode == "legacy"
    assert service.config.handoff_threshold == pytest.approx(0.8)
    assert service.router.config.handoff_mode == "legacy"
    assert service.router.config.handoff_threshold == pytest.approx(0.8)
    health = service.handle_get("/healthz")
    assert health["handoff_mode"] == "legacy"
    assert health["handoff_threshold"] == pytest.approx(0.8)


@pytest.mark.parametrize(
    "payload",
    [
        {"handoff_mode": "auto"},
        {"handoff_threshold": 1.2},
        {"handoff_threshold": -0.1},
        {"handoff_threshold": None},
    ],
)
def test_admin_config_rejects_invalid_handoff_values(
    tmp_path: Path, payload: dict[str, Any]
) -> None:
    service = edge_service(tmp_path)

    with pytest.raises(HttpError) as excinfo:
        service.handle_post("/v1/admin/config", payload)

    assert excinfo.value.status == 400
    assert service.config.handoff_mode == "probe"
    assert service.config.handoff_threshold == pytest.approx(0.44)


def test_registry_publish_carries_the_handoff_policy_version(tmp_path: Path) -> None:
    cloud = FakeCloud()
    service = edge_service(tmp_path, cloud=cloud)

    service.handle_post(
        "/v1/cloud/registry/publish",
        {"version": "v3", "artifact": "runs/adapters/v3", "metrics": {"accuracy": 0.9}},
    )

    path, payload = cloud.posted[0]
    assert path == "/v1/registry/publish"
    assert payload["metrics"]["accuracy"] == pytest.approx(0.9)
    assert payload["metrics"]["handoff_policy_version"] == "v7"
    assert payload["metrics"]["handoff_mode"] == "probe"


def test_outbox_replay_restores_the_enqueue_time_prefix_rewrite_policy(
    tmp_path: Path,
) -> None:
    service = edge_service(
        tmp_path, enable_outbox_flusher=True, allow_prefix_rewrite=True
    )
    request = InferRequest(prompt="焊点空洞 25% 如何判定", max_tokens=64)
    service.outbox.enqueue(request, handoff_response(request.request_id), "offline")

    stored = service.outbox.pending()[0]
    assert stored["edge_response"]["metadata"]["allow_prefix_rewrite"] is True
    assert stored["edge_response"]["metadata"]["handoff_policy_version"] == "v7"
    assert stored["request"]["metadata"]["allow_prefix_rewrite"] is True
    assert stored["request"]["metadata"]["handoff_policy_version"] == "v7"

    assert service.flusher is not None
    assert service.flusher.flush_once() == 1

    sent = service.cloud.calls[0]
    assert sent.partial_trace == "判定：返修。"
    assert sent.draft == ""
    assert sent.handoff_index == 6
    assert sent.edge_tokens_before_handoff == 6
    assert sent.handoff_reason == "low_prefix_trust"
    assert sent.prefix_trust == pytest.approx(0.2)
    assert sent.allow_prefix_rewrite is True
    assert sent.metadata["handoff_policy_version"] == "v7"


def test_outbox_replay_defaults_to_continuation_for_old_entries(tmp_path: Path) -> None:
    service = edge_service(tmp_path, enable_outbox_flusher=True)
    request = InferRequest(prompt="焊点空洞 25% 如何判定", max_tokens=64)
    service.outbox.enqueue(request, handoff_response(request.request_id), "offline")

    assert service.flusher is not None
    assert service.flusher.flush_once() == 1

    sent = service.cloud.calls[0]
    assert sent.allow_prefix_rewrite is False
    assert sent.partial_trace == "判定：返修。"


def test_probe_unavailable_counts_the_fallback_once(tmp_path: Path) -> None:
    backend = StreamingStub()
    service = edge_service(
        tmp_path, backend=backend, probe_path=str(tmp_path / "missing-probe.json")
    )

    response = service.infer(InferRequest(prompt="回流焊峰值温度应该控制在多少？"))

    assert response["source"] == "edge"
    assert backend.generate_calls == 1
    assert backend.generate_until_calls == 0
    counters = service.handle_get("/v1/stats")["metrics"]["counters"]
    assert counters["probe_unavailable_total"] == 1
    assert counters["legacy_fallback_total"] == 1
    assert "streaming_unavailable_total" not in counters
    assert "handoff_total" not in counters


def test_non_streaming_backend_counts_the_fallback_once(tmp_path: Path) -> None:
    service = edge_service(tmp_path)

    service.infer(InferRequest(prompt="回流焊峰值温度应该控制在多少？"))

    counters = service.handle_get("/v1/stats")["metrics"]["counters"]
    assert counters["streaming_unavailable_total"] == 1
    assert counters["legacy_fallback_total"] == 1
    assert "probe_unavailable_total" not in counters
    assert "handoff_total" not in counters
