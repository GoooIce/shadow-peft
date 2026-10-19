from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import pytest

from edge_cloud.backends import MockEdgeBackend
from edge_cloud.cloud_server import CloudService
from edge_cloud.confidence import ConfidenceEstimator
from edge_cloud.config import CloudConfig, EdgeConfig
from edge_cloud.edge_server import EdgeService
from edge_cloud.flywheel import FeedbackStore
from edge_cloud.outbox import Outbox
from edge_cloud.protocol import (
    CloudResponse,
    FeedbackRequest,
    InferRequest,
    ProtocolError,
)
from edge_cloud.registry import AdapterRegistry, RegistryError
from edge_cloud.router import CircuitBreaker, CloudBudget, CloudError, OutboxFlusher, Router

HARD_PROMPT = "IPC-A-610 Class 3 对虚焊和空洞的具体要求是什么？"
EASY_PROMPT = "回流焊峰值温度应该控制在多少？"


class FakeCloud:
    def __init__(self, *, fail: bool = False, confidence: float = 0.95) -> None:
        self.fail = fail
        self.confidence = confidence

    def enhance(self, request, *, timeout_ms=None):
        if self.fail:
            raise CloudError("simulated cloud outage")
        return CloudResponse(
            request_id=request.request_id,
            text="云端增强后的完整答案",
            confidence=self.confidence,
            model_id="fake-cloud",
            latency_ms=18.0,
            confidence_signals={"logprob_geomean": 0.95},
        )

    def feedback(self, payload, *, timeout_ms=3000):
        return {"ok": True}


def edge_config(tmp_path: Path, **overrides) -> EdgeConfig:
    defaults = dict(
        backend={"kind": "mock_edge"},
        cloud_url="http://127.0.0.1:1",
        outbox_path=str(tmp_path / "outbox.jsonl"),
        feedback_path=str(tmp_path / "feedback.jsonl"),
        enable_outbox_flusher=False,
        escalation_threshold=0.62,
    )
    defaults.update(overrides)
    return EdgeConfig(**defaults)


def test_protocol_roundtrip_and_validation() -> None:
    request = InferRequest(prompt="测试问题")
    assert request.request_id
    assert request.to_dict()["prompt"] == "测试问题"
    with pytest.raises(ProtocolError):
        InferRequest(prompt="   ")
    with pytest.raises(ProtocolError):
        InferRequest.from_dict({"prompt": "x", "max_tokens": 99999})
    with pytest.raises(ProtocolError):
        InferRequest.from_dict({"prompt": "x", "temperature": 9})
    bounded = InferRequest.from_dict({"prompt": "x", "max_tokens": 128, "temperature": 0.3})
    assert bounded.max_tokens == 128
    assert bounded.temperature == pytest.approx(0.3)
    feedback = FeedbackRequest.from_dict({"request_id": "r1", "accepted": True})
    assert feedback.request_id == "r1"
    with pytest.raises(ProtocolError):
        FeedbackRequest.from_dict({"request_id": "r1", "accepted": "yes"})


def test_confidence_high_vs_low() -> None:
    estimator = ConfidenceEstimator()
    high = estimator.estimate("完整且可执行的答案" * 3, [-0.05] * 30)
    low = estimator.estimate("不完整的答案", [-3.0] * 20)
    assert high.score > 0.8
    assert low.score < 0.5
    invalid_json = estimator.estimate("not json", [-0.05] * 10, expected_format="json")
    assert invalid_json.signals["format_ok"] == 0.0


def test_confidence_format_validation_is_public_with_a_compat_alias() -> None:
    from edge_cloud.confidence import _format_ok, format_ok

    assert format_ok("{}", "json") == 1.0
    assert format_ok("not json", "json") == 0.0
    assert _format_ok is format_ok


def test_router_keeps_confident_answer_on_edge(tmp_path: Path) -> None:
    router = Router(MockEdgeBackend(), edge_config(tmp_path), cloud_client=FakeCloud())
    response = router.infer(InferRequest(prompt=EASY_PROMPT))
    assert response.source == "edge"
    assert response.escalated is False
    assert response.confidence > 0.62
    assert router.metrics.snapshot()["counters"]["edge_only_total"] == 1


def test_router_escalates_low_confidence_to_cloud(tmp_path: Path) -> None:
    router = Router(MockEdgeBackend(), edge_config(tmp_path), cloud_client=FakeCloud())
    response = router.infer(InferRequest(prompt=HARD_PROMPT))
    assert response.source == "cloud"
    assert response.escalated is True
    assert response.escalation_reason == "low_confidence"
    assert response.edge_text
    assert response.edge_confidence < 0.62
    assert response.text == "云端增强后的完整答案"
    assert response.confidence == pytest.approx(0.95)


def test_router_forces_cloud_for_quality_verdict(tmp_path: Path) -> None:
    router = Router(MockEdgeBackend(), edge_config(tmp_path), cloud_client=FakeCloud())
    response = router.infer(
        InferRequest(prompt=EASY_PROMPT, task="quality_verdict")
    )
    assert response.source == "cloud"
    assert response.escalation_reason == "task_requires_cloud"


def test_router_offline_mode_queues_then_flushes(tmp_path: Path) -> None:
    config = edge_config(tmp_path, offline=True)
    outbox = Outbox(config.outbox_path)
    router = Router(MockEdgeBackend(), config, cloud_client=FakeCloud(), outbox=outbox)
    response = router.infer(InferRequest(prompt=HARD_PROMPT))
    assert response.source == "edge"
    assert response.degraded is True
    assert len(outbox.pending()) == 1

    config.offline = False
    flusher = OutboxFlusher(outbox, FakeCloud(), interval_s=60)
    assert flusher.flush_once() == 1
    assert outbox.pending() == []
    assert outbox.stats()["done"] == 1


def test_router_cloud_failure_degrades_and_queues(tmp_path: Path) -> None:
    config = edge_config(tmp_path)
    outbox = Outbox(config.outbox_path)
    router = Router(MockEdgeBackend(), config, cloud_client=FakeCloud(fail=True), outbox=outbox)
    response = router.infer(InferRequest(prompt=HARD_PROMPT))
    assert response.source == "edge"
    assert response.degraded is True
    assert "云端增强失败" in " ".join(response.trace)
    assert len(outbox.pending()) == 1
    assert router.metrics.snapshot()["counters"]["cloud_failure_total"] == 1


def test_outbox_flusher_marks_corrupt_item_dead(tmp_path: Path) -> None:
    outbox = Outbox(tmp_path / "outbox.jsonl")
    (tmp_path / "outbox.jsonl").write_text(
        json.dumps(
            {
                "item_id": "bad",
                "request": {"prompt": ""},
                "edge_response": {},
                "reason": "low_confidence",
                "status": "pending",
                "attempts": 0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    flusher = OutboxFlusher(outbox, FakeCloud(), interval_s=60)
    assert flusher.flush_once() == 0
    dead = outbox.items("dead")
    assert len(dead) == 1
    assert dead[0]["item_id"] == "bad"


def test_circuit_breaker_opens_and_recovers() -> None:
    breaker = CircuitBreaker(failures=2, cooldown_s=100)
    assert breaker.allow()
    breaker.record_failure()
    assert breaker.allow()
    breaker.record_failure()
    assert not breaker.allow()
    immediate = CircuitBreaker(failures=1, cooldown_s=0)
    immediate.record_failure()
    assert immediate.allow()


def test_cloud_budget_caps_calls_and_ratio() -> None:
    budget = CloudBudget(max_calls_per_window=1, max_ratio=1.0)
    allowed, _ = budget.allow()
    assert allowed
    budget.record(True)
    allowed, reason = budget.allow()
    assert not allowed and reason == "max_calls"

    ratio = CloudBudget(max_calls_per_window=100, max_ratio=0.5, ratio_min_samples=2)
    ratio.record(False)
    ratio.record(False)
    allowed, _ = ratio.allow()
    assert allowed
    ratio.record(True)
    ratio.record(True)
    allowed, reason = ratio.allow()
    assert not allowed and reason == "max_ratio"


def test_feedback_export_deduplicates_and_keeps_hard_cases(tmp_path: Path) -> None:
    store = FeedbackStore(tmp_path / "feedback.jsonl")
    store.record(
        FeedbackRequest(
            request_id="r1",
            accepted=True,
            prompt="问题A",
            task="standard_qa",
            edge_text="端侧A",
            cloud_text="云端A",
            final_text="云端A",
            source="cloud",
            escalation_reason="low_confidence",
        )
    )
    store.record(
        FeedbackRequest(
            request_id="r2",
            accepted=True,
            prompt="问题B",
            task="process_qa",
            edge_text="端侧B",
            final_text="端侧B",
            source="edge",
        )
    )
    store.record(
        FeedbackRequest(
            request_id="r3",
            accepted=True,
            prompt="问题B",
            task="process_qa",
            edge_text="端侧B",
            final_text="端侧B",
            source="edge",
        )
    )
    store.record(
        FeedbackRequest(
            request_id="r4",
            accepted=False,
            prompt="问题C",
            task="process_calc",
            edge_text="端侧C",
            corrected_text="正确C",
            source="edge",
        )
    )
    stats = store.stats()
    assert stats["total"] == 4
    assert stats["hard_cases"] == 2
    assert stats["corrected"] == 1

    result = store.export_sft(tmp_path / "sft.jsonl")
    assert result["exported"] == 2
    lines = [
        json.loads(line)
        for line in (tmp_path / "sft.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    answers = {line["messages"][1]["content"]: line["messages"][2]["content"] for line in lines}
    assert answers["问题A"] == "云端A"
    assert answers["问题C"] == "正确C"


def test_registry_canary_promote_rollback(tmp_path: Path) -> None:
    registry = AdapterRegistry(tmp_path / "registry.json")
    registry.publish("v1", artifact="a/v1")
    assert registry.state()["active"] == "v1"
    registry.publish("v2", artifact="a/v2", canary_ratio=1.0)
    assert registry.resolve("edge-01")["version"] == "v2"
    assert registry.resolve("edge-01")["channel"] == "canary"
    registry.promote("v2")
    assert registry.state()["active"] == "v2"
    registry.rollback()
    assert registry.state()["active"] == "v1"
    with pytest.raises(RegistryError):
        registry.promote("missing")


def _http(url: str, method: str = "GET", payload=None):
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method=method
    )
    with urllib.request.urlopen(request, timeout=8) as response:
        body = response.read().decode("utf-8")
        content_type = response.headers.get("Content-Type", "")
        return response.status, json.loads(body) if "application/json" in content_type else body


def test_edge_cloud_http_end_to_end(tmp_path: Path) -> None:
    cloud = CloudService(
        CloudConfig(
            backend={"kind": "mock_cloud"},
            feedback_path=str(tmp_path / "cloud_feedback.jsonl"),
            registry_path=str(tmp_path / "registry.json"),
            dataset_path=str(tmp_path / "sft.jsonl"),
        )
    )
    cloud_server = cloud.start("127.0.0.1", 0)
    edge = EdgeService(
        edge_config(
            tmp_path,
            cloud_url=cloud_server.url,
            feedback_path=str(tmp_path / "edge_feedback.jsonl"),
        )
    )
    edge_server = edge.start("127.0.0.1", 0)
    try:
        status, page = _http(edge_server.url + "/")
        assert status == 200
        assert "端云协同推理控制台" in page

        status, response = _http(
            edge_server.url + "/v1/infer",
            "POST",
            {"prompt": HARD_PROMPT, "task": "standard_qa"},
        )
        assert status == 200
        assert response["source"] == "cloud"
        assert response["escalated"] is True

        status, feedback = _http(
            edge_server.url + "/v1/feedback",
            "POST",
            {
                "request_id": response["request_id"],
                "accepted": False,
                "prompt": HARD_PROMPT,
                "task": "standard_qa",
                "corrected_text": "人工纠正后的标准答案",
            },
        )
        assert status == 200
        assert feedback["record"]["is_hard_case"] is True

        _, edge_stats = _http(edge_server.url + "/v1/stats")
        counters = edge_stats["metrics"]["counters"]
        assert counters["requests_total"] == 1
        assert counters["escalation_total"] == 1
        assert counters["cloud_success_total"] == 1
        assert counters["feedback_forwarded_total"] == 1

        _, cloud_stats = _http(cloud_server.url + "/v1/stats")
        assert cloud_stats["feedback"]["total"] == 1
        assert cloud_stats["feedback"]["hard_cases"] == 1

        _, exported = _http(
            edge_server.url + "/v1/cloud/dataset/export", "POST", {"hard_only": True}
        )
        assert exported["exported"] == 1

        _, published = _http(
            edge_server.url + "/v1/cloud/registry/publish",
            "POST",
            {"version": "v1", "artifact": "runs/adapters/v1", "canary_ratio": 0.0},
        )
        assert published["active"] == "v1"
        _, resolved = _http(edge_server.url + "/v1/cloud/registry")
        assert resolved["active"] == "v1"

        status, health = _http(edge_server.url + "/healthz")
        assert status == 200
        assert health["status"] == "ok"
    finally:
        edge.stop()
        cloud.stop()


def test_console_asset_exists() -> None:
    path = Path(__file__).resolve().parents[2] / "src" / "edge_cloud" / "web" / "console.html"
    assert path.exists()
    assert "fetch(" in path.read_text(encoding="utf-8")
