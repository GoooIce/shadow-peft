"""Router memory retrieval/reranking wiring — external-behavior tests (issue #39).

Spec: ``docs/specs/jitrl-memory-hotfix.md`` §Implementation Decisions 4 and
Testing Decisions「路由缝」.

The router seam is ``Router.infer``: tests drive a scripted n-best backend, a
real ``ExperienceMemory`` built from ``MemoryEntry`` objects and a fake cloud,
then assert the response contract, cloud calls and metrics.  No private router
method is called.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest

from edge_cloud.backends import Generation
from edge_cloud.config import EdgeConfig
from edge_cloud.memory import ExperienceMemory, MemoryEntry
from edge_cloud.protocol import CloudRequest, CloudResponse, InferRequest, InferResponse
from edge_cloud.router import CloudError, Router

PROMPT = "E-104 真空压力不足"
TASK = "fault_diagnosis"
BASELINE_ANSWER = "检查真空管路是否漏气"
RERANKED_ANSWER = "先泄压再清理吸嘴"


class ScriptedNBestBackend:
    """Deterministic n-best edge model with a whole-answer fallback."""

    model_id = "scripted-nbest"

    def __init__(self, candidates: list[tuple[str, float]], *, prompt_tokens: int = 17) -> None:
        if not candidates:
            raise ValueError("candidates must not be empty")
        self.candidates = list(candidates)
        self.prompt_tokens = prompt_tokens
        self.generate_calls = 0
        self.nbest_calls: list[int] = []

    def _generation(self, text: str, logprob: float) -> Generation:
        return Generation(
            text=text,
            token_logprobs=[logprob] * 20,
            model_id=self.model_id,
            latency_ms=1.0,
            raw={"prompt_tokens": self.prompt_tokens},
        )

    def generate(self, request: InferRequest, *, draft: str | None = None) -> Generation:
        del draft
        self.generate_calls += 1
        text, logprob = self.candidates[0]
        return self._generation(text, logprob)

    def generate_nbest(self, request: InferRequest, n: int = 4) -> list[Generation]:
        self.nbest_calls.append(n)
        return [self._generation(text, logprob) for text, logprob in self.candidates[:n]]


class FakeCloud:
    """Records every CloudRequest and returns a high-confidence answer."""

    def __init__(self, *, fail: bool = False, confidence: float = 0.95) -> None:
        self.fail = fail
        self.confidence = confidence
        self.calls: list[tuple[CloudRequest, int | None]] = []

    def enhance(
        self, request: CloudRequest, *, timeout_ms: int | None = None
    ) -> CloudResponse:
        self.calls.append((request, timeout_ms))
        if self.fail:
            raise CloudError("simulated cloud outage")
        return CloudResponse(
            request_id=request.request_id,
            text="云端增强后的完整答案",
            confidence=self.confidence,
            model_id="fake-cloud",
            latency_ms=18.0,
            confidence_signals={"logprob_geomean": self.confidence},
        )


def edge_config(tmp_path: Path, **overrides: Any) -> EdgeConfig:
    defaults = dict(
        backend={"kind": "mock_edge"},
        cloud_url="http://127.0.0.1:1",
        outbox_path=str(tmp_path / "outbox.jsonl"),
        feedback_path=str(tmp_path / "feedback.jsonl"),
        enable_outbox_flusher=False,
        escalation_threshold=0.62,
        memory_mode="on",
    )
    defaults.update(overrides)
    return EdgeConfig(**defaults)


def build_memory(
    *,
    answer: str = RERANKED_ANSWER,
    prompt: str = PROMPT,
    returns: tuple[float, ...] = (1.0, 0.5),
    task: str = TASK,
    adapter_version: str = "base",
    tenant: str = "default",
) -> ExperienceMemory:
    now = time.time()
    return ExperienceMemory(
        [
            MemoryEntry(
                entry_id=f"mem-{index}",
                prompt=prompt,
                task=task,
                candidates=[answer],
                return_value=return_value,
                accepted=return_value > 0,
                corrected=return_value >= 1.0,
                adapter_version=adapter_version,
                tenant=tenant,
                created_at=now,
            )
            for index, return_value in enumerate(returns)
        ]
    )


def test_memory_hit_high_confidence_returns_edge_without_cloud(tmp_path: Path) -> None:
    backend = ScriptedNBestBackend(
        [(BASELINE_ANSWER, -0.10), (RERANKED_ANSWER, -0.40)]
    )
    cloud = FakeCloud()
    router = Router(
        backend,
        edge_config(tmp_path),
        cloud_client=cloud,
        memory=build_memory(),
    )
    request = InferRequest(prompt=PROMPT, task=TASK, deadline_ms=3000)

    response = router.infer(request)

    assert response.source == "edge"
    assert response.escalated is False
    assert response.text == RERANKED_ANSWER
    assert response.memory_hit is True
    assert response.memory_entry_ids == ["mem-0", "mem-1"]
    assert response.memory_score == pytest.approx(1.5)
    assert response.memory_reranked is True
    assert cloud.calls == []
    assert backend.nbest_calls == [4]
    assert router.budget.snapshot()["cloud_calls"] == 0

    metrics = router.metrics.snapshot()
    counters = metrics["counters"]
    assert counters["memory_hit_total"] == 1
    assert counters["memory_reranked_total"] == 1
    assert "memory_fallback_total" not in counters
    assert counters["edge_only_total"] == 1
    assert metrics["latency_ms"]["memory_latency_ms"]["count"] == 1


def test_memory_off_keeps_the_single_answer_baseline(tmp_path: Path) -> None:
    backend = ScriptedNBestBackend(
        [(BASELINE_ANSWER, -0.10), (RERANKED_ANSWER, -0.40)]
    )
    cloud = FakeCloud()
    router = Router(
        backend,
        edge_config(tmp_path, memory_mode="off"),
        cloud_client=cloud,
        memory=build_memory(),
    )
    request = InferRequest(prompt=PROMPT, task=TASK, deadline_ms=3000)

    response = router.infer(request)

    assert response.source == "edge"
    assert response.text == BASELINE_ANSWER
    assert response.memory_hit is False
    assert response.memory_entry_ids == []
    assert response.memory_score is None
    assert response.memory_reranked is False
    assert backend.generate_calls == 1
    assert backend.nbest_calls == []
    assert cloud.calls == []
    counters = router.metrics.snapshot()["counters"]
    assert not [name for name in counters if name.startswith("memory_")]


def test_reranked_answer_failing_format_falls_back_to_the_original(tmp_path: Path) -> None:
    baseline = '{"answer": "检查真空管路是否漏气"}'
    backend = ScriptedNBestBackend([(baseline, -0.05), (RERANKED_ANSWER, -0.30)])
    cloud = FakeCloud()
    router = Router(
        backend,
        edge_config(tmp_path),
        cloud_client=cloud,
        memory=build_memory(answer=RERANKED_ANSWER),
    )
    request = InferRequest(
        prompt=PROMPT,
        task=TASK,
        deadline_ms=3000,
        metadata={"expected_format": "json"},
    )

    response = router.infer(request)

    assert response.source == "edge"
    assert response.text == baseline
    assert response.memory_hit is True
    assert response.memory_entry_ids == ["mem-0", "mem-1"]
    assert response.memory_reranked is False
    assert cloud.calls == []
    counters = router.metrics.snapshot()["counters"]
    assert counters["memory_hit_total"] == 1
    assert counters["memory_fallback_total"] == 1
    assert "memory_reranked_total" not in counters
    assert "格式校验" in " ".join(response.trace)


def test_unavailable_memory_falls_back_silently_to_the_baseline(tmp_path: Path) -> None:
    backend = ScriptedNBestBackend([(BASELINE_ANSWER, -0.10)])
    cloud = FakeCloud()
    # A directory cannot be read as a JSONL memory file; loading must not fail
    # the request, and the router must keep the single-answer baseline.
    router = Router(
        backend,
        edge_config(tmp_path, memory_path=str(tmp_path)),
        cloud_client=cloud,
    )
    request = InferRequest(prompt=PROMPT, task=TASK, deadline_ms=3000)

    response = router.infer(request)

    assert response.source == "edge"
    assert response.text == BASELINE_ANSWER
    assert response.memory_hit is False
    assert response.memory_reranked is False
    assert backend.generate_calls == 1
    assert backend.nbest_calls == []
    assert cloud.calls == []
    counters = router.metrics.snapshot()["counters"]
    assert counters["memory_fallback_total"] == 1
    assert "memory_hit_total" not in counters
    assert "记忆不可用" in " ".join(response.trace)


def test_reranked_answer_still_escalates_when_confidence_is_low(tmp_path: Path) -> None:
    backend = ScriptedNBestBackend(
        [(BASELINE_ANSWER, -0.05), (RERANKED_ANSWER, -0.70)]
    )
    cloud = FakeCloud()
    router = Router(
        backend,
        edge_config(tmp_path),
        cloud_client=cloud,
        memory=build_memory(),
    )
    request = InferRequest(prompt=PROMPT, task=TASK, deadline_ms=3000)

    response = router.infer(request)

    assert response.source == "cloud"
    assert response.escalated is True
    assert response.escalation_reason == "low_confidence"
    assert len(cloud.calls) == 1
    assert response.memory_hit is True
    assert response.memory_reranked is False
    counters = router.metrics.snapshot()["counters"]
    assert counters["memory_hit_total"] == 1
    assert counters["memory_reranked_total"] == 1
    assert counters["cloud_attempt_total"] == 1


def test_offline_mode_still_uses_the_local_memory(tmp_path: Path) -> None:
    backend = ScriptedNBestBackend(
        [(BASELINE_ANSWER, -0.10), (RERANKED_ANSWER, -0.40)]
    )
    cloud = FakeCloud()
    router = Router(
        backend,
        edge_config(tmp_path, offline=True),
        cloud_client=cloud,
        memory=build_memory(),
    )
    request = InferRequest(prompt=PROMPT, task=TASK, deadline_ms=3000)

    response = router.infer(request)

    assert response.source == "edge"
    assert response.degraded is False
    assert response.text == RERANKED_ANSWER
    assert response.memory_reranked is True
    assert cloud.calls == []
    assert router.budget.snapshot()["cloud_calls"] == 0


@pytest.mark.parametrize("forced_by", ["task", "request"])
def test_forced_cloud_is_never_bypassed_by_memory(
    tmp_path: Path, forced_by: str
) -> None:
    backend = ScriptedNBestBackend(
        [(BASELINE_ANSWER, -0.10), (RERANKED_ANSWER, -0.40)]
    )
    cloud = FakeCloud()
    router = Router(
        backend,
        edge_config(tmp_path),
        cloud_client=cloud,
        memory=build_memory(),
    )
    request = InferRequest(
        prompt=PROMPT,
        task="quality_verdict" if forced_by == "task" else TASK,
        require_cloud=forced_by == "request",
        deadline_ms=3000,
    )

    response = router.infer(request)

    assert response.source == "cloud"
    assert response.escalation_reason == "task_requires_cloud"
    assert len(cloud.calls) == 1
    assert backend.generate_calls == 1
    assert backend.nbest_calls == []
    assert response.memory_hit is False
    assert response.memory_reranked is False


def test_memory_latency_budget_is_a_no_memory_fallback(tmp_path: Path) -> None:
    backend = ScriptedNBestBackend(
        [(BASELINE_ANSWER, -0.10), (RERANKED_ANSWER, -0.40)]
    )
    cloud = FakeCloud()
    router = Router(
        backend,
        edge_config(tmp_path, memory_latency_budget_ms=0.0),
        cloud_client=cloud,
        memory=build_memory(),
    )
    request = InferRequest(prompt=PROMPT, task=TASK, deadline_ms=3000)

    response = router.infer(request)

    assert response.source == "edge"
    assert response.text == BASELINE_ANSWER
    assert response.memory_hit is False
    assert response.memory_reranked is False
    assert backend.nbest_calls == []
    assert cloud.calls == []
    counters = router.metrics.snapshot()["counters"]
    assert counters["memory_fallback_total"] == 1
    assert "memory_hit_total" not in counters
    assert router.metrics.snapshot()["latency_ms"]["memory_latency_ms"]["count"] == 1


def test_memory_miss_keeps_the_single_answer_without_counting_a_fallback(
    tmp_path: Path,
) -> None:
    backend = ScriptedNBestBackend(
        [(BASELINE_ANSWER, -0.10), (RERANKED_ANSWER, -0.40)]
    )
    cloud = FakeCloud()
    router = Router(
        backend,
        edge_config(tmp_path),
        cloud_client=cloud,
        memory=build_memory(),
    )
    request = InferRequest(
        prompt="回流焊峰值温度应该控制在多少？", task=TASK, deadline_ms=3000
    )

    response = router.infer(request)

    assert response.source == "edge"
    assert response.text == BASELINE_ANSWER
    assert response.memory_hit is False
    assert backend.nbest_calls == []
    assert cloud.calls == []
    counters = router.metrics.snapshot()["counters"]
    assert "memory_fallback_total" not in counters
    assert "memory_hit_total" not in counters
    assert router.metrics.snapshot()["latency_ms"]["memory_latency_ms"]["count"] == 1


def test_backend_without_nbest_keeps_the_single_answer_path(tmp_path: Path) -> None:
    class WholeAnswerOnlyBackend:
        model_id = "whole-only"

        def __init__(self) -> None:
            self.generate_calls = 0

        def generate(
            self, request: InferRequest, *, draft: str | None = None
        ) -> Generation:
            del request, draft
            self.generate_calls += 1
            return Generation(
                text=BASELINE_ANSWER,
                token_logprobs=[-0.05] * 20,
                model_id=self.model_id,
                latency_ms=1.0,
            )

    backend = WholeAnswerOnlyBackend()
    router = Router(
        backend,
        edge_config(tmp_path),
        cloud_client=FakeCloud(),
        memory=build_memory(),
    )

    response = router.infer(InferRequest(prompt=PROMPT, task=TASK, deadline_ms=3000))

    assert response.source == "edge"
    assert response.text == BASELINE_ANSWER
    assert response.memory_hit is False
    assert backend.generate_calls == 1
    counters = router.metrics.snapshot()["counters"]
    assert not [name for name in counters if name.startswith("memory_")]


def test_retrieval_failure_falls_back_and_counts_once(tmp_path: Path) -> None:
    class BrokenMemory:
        def retrieve(self, *args: Any, **kwargs: Any) -> list[Any]:
            raise RuntimeError("memory index corrupted")

    backend = ScriptedNBestBackend([(BASELINE_ANSWER, -0.10)])
    router = Router(
        backend,
        edge_config(tmp_path),
        cloud_client=FakeCloud(),
        memory=BrokenMemory(),  # type: ignore[arg-type]
    )

    response = router.infer(InferRequest(prompt=PROMPT, task=TASK, deadline_ms=3000))

    assert response.text == BASELINE_ANSWER
    assert response.memory_hit is False
    assert backend.nbest_calls == []
    counters = router.metrics.snapshot()["counters"]
    assert counters["memory_fallback_total"] == 1
    assert "memory_hit_total" not in counters
    assert "记忆检索失败" in " ".join(response.trace)


def test_memory_config_defaults_and_legacy_roundtrip() -> None:
    config = EdgeConfig()

    assert config.memory_mode == "off"
    assert config.memory_path == ""
    assert config.memory_top_k == 4
    assert config.memory_threshold == pytest.approx(0.45)
    assert config.memory_alpha == pytest.approx(0.5)
    assert config.memory_min_support == 2
    assert config.memory_ttl_s == 30 * 24 * 60 * 60
    assert config.memory_tenant == "default"
    assert config.memory_latency_budget_ms == pytest.approx(50.0)
    assert EdgeConfig.from_dict(config.to_dict()) == config
    loaded = EdgeConfig.from_dict(
        {
            "memory_mode": "on",
            "memory_path": "runs/memory.jsonl",
            "memory_top_k": 8,
            "memory_threshold": 0.75,
            "memory_alpha": 1.0,
            "memory_min_support": 3,
            "memory_ttl_s": 3600,
            "memory_tenant": "line-a",
            "memory_latency_budget_ms": 25,
        }
    )
    assert loaded.memory_mode == "on"
    assert loaded.memory_path == "runs/memory.jsonl"
    assert loaded.memory_top_k == 8
    assert loaded.memory_threshold == pytest.approx(0.75)
    assert loaded.memory_alpha == pytest.approx(1.0)
    assert loaded.memory_min_support == 3
    assert loaded.memory_ttl_s == 3600
    assert loaded.memory_tenant == "line-a"
    assert loaded.memory_latency_budget_ms == pytest.approx(25.0)


def test_memory_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        EdgeConfig(memory_mode="auto")
    with pytest.raises(ValueError):
        EdgeConfig(memory_threshold=1.5)
    with pytest.raises(ValueError):
        EdgeConfig(memory_top_k=-1)
    with pytest.raises(ValueError):
        EdgeConfig(memory_alpha=-0.1)
    with pytest.raises(ValueError):
        EdgeConfig(memory_min_support=-1)
    with pytest.raises(ValueError):
        EdgeConfig(memory_ttl_s=-1)
    with pytest.raises(ValueError):
        EdgeConfig(memory_latency_budget_ms=-1.0)


def test_infer_response_serializes_the_memory_fields() -> None:
    response = InferResponse(
        request_id="r1",
        text="答案",
        source="edge",
        confidence=0.9,
        latency_ms=5.0,
        model_id="m",
    )

    payload = response.to_dict()

    assert payload["memory_hit"] is False
    assert payload["memory_entry_ids"] == []
    assert payload["memory_score"] is None
    assert payload["memory_reranked"] is False
