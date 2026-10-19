from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest

from edge_cloud.backends import Generation, StreamingBackend, _decode_until
from edge_cloud.config import EdgeConfig
from edge_cloud.handoff import HandoffDecision, HandoffStep
from edge_cloud.outbox import Outbox
from edge_cloud.probe import PROBE_VERSION, write_probe
from edge_cloud.protocol import CloudRequest, CloudResponse, InferRequest
from edge_cloud.router import CloudBudget, CloudError, OutboxFlusher, Router

_FIXED_TRAINED_AT = "2026-01-01T00:00:00+00:00"


class FakeCloud:
    """Records every CloudRequest and returns a scripted continuation."""

    def __init__(
        self,
        text: str = "后续。",
        *,
        fail: bool = False,
        confidence: float = 0.9,
        output_tokens: int = 5,
        corrected_prefix: bool = False,
        continued_from_prefix: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.text = text
        self.fail = fail
        self.confidence = confidence
        self.output_tokens = output_tokens
        self.corrected_prefix = corrected_prefix
        self.continued_from_prefix = continued_from_prefix
        self.metadata = dict(metadata or {})
        self.calls: list[tuple[CloudRequest, int | None]] = []

    def enhance(
        self, request: CloudRequest, *, timeout_ms: int | None = None
    ) -> CloudResponse:
        self.calls.append((request, timeout_ms))
        if self.fail:
            raise CloudError("simulated cloud outage")
        return CloudResponse(
            request_id=request.request_id,
            text=self.text,
            confidence=self.confidence,
            model_id="fake-cloud",
            latency_ms=12.0,
            confidence_signals={"logprob_geomean": 0.9},
            continued_from_prefix=self.continued_from_prefix,
            corrected_prefix=self.corrected_prefix,
            output_tokens=self.output_tokens,
            metadata=dict(self.metadata),
        )


class ScriptedStreamingBackend:
    """Deterministic token stream plus a whole-answer fallback path."""

    model_id = "scripted-edge"

    def __init__(
        self,
        tokens: list[str],
        *,
        full_text: str = "端侧完整答案。",
        prompt_tokens: int = 17,
        logprob: float = -0.05,
    ) -> None:
        self.tokens = list(tokens)
        self.full_text = full_text
        self.prompt_tokens = prompt_tokens
        self.logprob = logprob
        self.generate_calls = 0
        self.generate_until_calls = 0
        self.contexts: list[dict[str, Any] | None] = []
        self.boundary_waits: list[int] = []
        self.expected_formats: list[str | None] = []

    def generate(self, request: InferRequest, *, draft: str | None = None) -> Generation:
        del draft
        self.generate_calls += 1
        return Generation(
            text=self.full_text,
            token_logprobs=[self.logprob] * 40,
            model_id=self.model_id,
            latency_ms=1.0,
            raw={"prompt_tokens": self.prompt_tokens},
        )

    def generate_until(
        self,
        request: InferRequest,
        policy: Any,
        *,
        max_boundary_wait_tokens: int = 12,
        expected_format: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> Generation:
        self.generate_until_calls += 1
        self.contexts.append(context)
        self.boundary_waits.append(max_boundary_wait_tokens)
        self.expected_formats.append(expected_format)
        task = request.task if context is None else str(context.get("task", request.task))
        steps = (
            (
                token,
                self.logprob,
                [0.0, 1.0],
                None,
                {"task": task, "format_ok": 1.0, "expected_format": expected_format},
            )
            for token in self.tokens
        )
        outcome = _decode_until(
            steps,
            policy,
            max_tokens=request.max_tokens,
            max_boundary_wait_tokens=max_boundary_wait_tokens,
            expected_format=expected_format,
            context=context,
        )
        raw: dict[str, Any] = {"prompt_tokens": self.prompt_tokens}
        if outcome.handoff is not None:
            raw["handoff_waited_tokens"] = outcome.waited_tokens
        return Generation(
            text=outcome.text,
            token_logprobs=outcome.token_logprobs,
            model_id=self.model_id,
            finish_reason=outcome.finish_reason,
            latency_ms=1.0,
            raw=raw,
            handoff=outcome.handoff,
            output_tokens=outcome.output_tokens,
        )


class AlwaysHandoffPolicy:
    """Test stub: requests a handoff on every observed step."""

    policy_name = "scripted"

    def __init__(self, *, trust: float = 0.2) -> None:
        self.trust = trust
        self.reset_calls = 0
        self.contexts: list[dict[str, Any] | None] = []
        self.steps: list[HandoffStep] = []

    def reset(self, context: dict[str, Any] | None = None) -> None:
        self.reset_calls += 1
        self.contexts.append(context)

    def observe(self, step: HandoffStep) -> HandoffDecision:
        self.steps.append(step)
        return HandoffDecision(
            action="handoff",
            reason="low_prefix_trust",
            boundary=step.features.boundary_kind,
            handoff_index=step.features.prefix_tokens,
            prefix_trust=self.trust,
        )


def assert_streaming(backend: ScriptedStreamingBackend) -> None:
    assert isinstance(backend, StreamingBackend)


def edge_config(tmp_path: Path, **overrides: Any) -> EdgeConfig:
    defaults = dict(
        backend={"kind": "mock_edge"},
        cloud_url="http://127.0.0.1:1",
        outbox_path=str(tmp_path / "outbox.jsonl"),
        feedback_path=str(tmp_path / "feedback.jsonl"),
        enable_outbox_flusher=False,
        escalation_threshold=0.62,
        handoff_mode="probe",
    )
    defaults.update(overrides)
    return EdgeConfig(**defaults)


def write_trigger_probe(
    tmp_path: Path, trigger_prefix_tokens: int, *, name: str = "probe.json"
) -> str:
    """Probe whose trust is high below *trigger_prefix_tokens* and low at/above it."""

    payload = {
        "version": PROBE_VERSION,
        "feature_names": ["prefix_tokens"],
        "means": [0.0],
        "stds": [1.0],
        "coefs": [-10.0],
        "intercept": 10.0 * trigger_prefix_tokens - 5.0,
        "threshold": 0.5,
        "calibration": [],
        "fingerprint": "",
        "required_features": ["prefix_tokens"],
        "trained_at": _FIXED_TRAINED_AT,
    }
    path = tmp_path / name
    write_probe(payload, path)
    return str(path)


def write_feature_probe(
    tmp_path: Path, feature_name: str, *, name: str = "feature-probe.json"
) -> str:
    """Probe that requires a feature some streaming backends cannot provide."""

    payload = {
        "version": PROBE_VERSION,
        "feature_names": [feature_name],
        "means": [0.0],
        "stds": [1.0],
        "coefs": [1.0],
        "intercept": 0.0,
        "threshold": 0.5,
        "calibration": [],
        "fingerprint": "",
        "required_features": [feature_name],
        "trained_at": _FIXED_TRAINED_AT,
    }
    path = tmp_path / name
    write_probe(payload, path)
    return str(path)


def test_probe_confident_prefix_stays_on_edge_with_router_context(
    tmp_path: Path,
) -> None:
    backend = ScriptedStreamingBackend(["安全", "完成。"])
    assert_streaming(backend)
    cloud = FakeCloud()
    config = edge_config(
        tmp_path, probe_path=write_trigger_probe(tmp_path, 999), max_boundary_wait_tokens=7
    )
    router = Router(backend, config, cloud_client=cloud)
    request = InferRequest(prompt="安全操作规范", task="process_qa", deadline_ms=3000)

    response = router.infer(request)

    assert response.source == "edge"
    assert response.escalated is False
    assert response.text == "安全完成。"
    assert backend.generate_until_calls == 1
    assert backend.generate_calls == 0
    context = backend.contexts[0]
    assert context is not None
    assert context["task"] == "process_qa"
    assert context["deadline_ms"] == 3000
    assert context["expected_format"] is None
    assert context["min_remaining_ms"] == 150.0
    assert context["clock"] is time.monotonic
    assert context["deadline_at"] > context["clock"]()
    assert backend.boundary_waits == [7]
    assert cloud.calls == []
    counters = router.metrics.snapshot()["counters"]
    assert counters["edge_only_total"] == 1
    assert "handoff_total" not in counters


def test_probe_handoff_waits_for_sentence_boundary_and_sends_partial_trace(
    tmp_path: Path,
) -> None:
    backend = ScriptedStreamingBackend(["判", "定", "：", "返", "修", "。", "后", "续", "。"])
    cloud = FakeCloud(text="后续。", output_tokens=3)
    config = edge_config(tmp_path, probe_path=write_trigger_probe(tmp_path, 2))
    router = Router(backend, config, cloud_client=cloud)
    request = InferRequest(
        prompt="焊点空洞 25% 如何判定", task="fault_diagnosis", max_tokens=64, deadline_ms=3000
    )

    response = router.infer(request)

    assert backend.generate_until_calls == 1
    assert backend.generate_calls == 0
    assert len(cloud.calls) == 1
    cloud_request, _ = cloud.calls[0]
    assert cloud_request.partial_trace == "判定：返修。"
    assert cloud_request.draft == ""
    assert cloud_request.handoff_index == 6
    assert cloud_request.edge_tokens_before_handoff == 6
    assert cloud_request.handoff_reason == "low_prefix_trust"
    assert cloud_request.prefix_trust is not None
    assert cloud_request.prefix_trust < 0.5
    assert cloud_request.allow_prefix_rewrite is False
    assert cloud_request.max_tokens == 58

    assert response.source == "cloud"
    assert response.escalated is True
    assert response.escalation_reason == "low_prefix_trust"
    assert response.text == "判定：返修。后续。"
    assert response.edge_text == "判定：返修。"
    assert response.handoff_index == 6
    assert response.edge_tokens_before_handoff == 6
    assert response.handoff_reason == "low_prefix_trust"
    assert response.prefix_trust == pytest.approx(cloud_request.prefix_trust)
    assert response.cloud_output_tokens == 3

    metrics = router.metrics.snapshot()
    counters = metrics["counters"]
    assert counters["handoff_total"] == 1
    assert counters["handoff_reason_low_prefix_trust"] == 1
    assert counters["handoff_boundary_sentence"] == 1
    assert counters["handoff_index_bucket_lt_8"] == 1
    assert metrics["latency_ms"]["handoff_index"]["count"] == 1
    assert metrics["latency_ms"]["handoff_index"]["max"] == 6
    assert metrics["latency_ms"]["boundary_wait_tokens"]["max"] == 4


def test_handoff_wait_budget_forces_a_cut_at_the_token_boundary(tmp_path: Path) -> None:
    backend = ScriptedStreamingBackend(["a", "b", "c", "d", "e"])
    cloud = FakeCloud()
    config = edge_config(
        tmp_path,
        handoff_mode="oracle",
        max_boundary_wait_tokens=2,
    )
    router = Router(backend, config, cloud_client=cloud)
    request = InferRequest(
        prompt="无标点长句",
        max_tokens=32,
        deadline_ms=3000,
        metadata={"gold_handoff_index": 1},
    )

    response = router.infer(request)

    assert backend.boundary_waits == [2]
    assert backend.generate_until_calls == 1
    assert len(cloud.calls) == 1
    cloud_request, _ = cloud.calls[0]
    assert cloud_request.partial_trace == "abc"
    assert cloud_request.handoff_index == 3
    assert cloud_request.edge_tokens_before_handoff == 3
    assert response.handoff_index == 3
    counters = router.metrics.snapshot()["counters"]
    assert counters["handoff_boundary_token"] == 1
    assert router.metrics.snapshot()["latency_ms"]["boundary_wait_tokens"]["max"] == 2


@pytest.mark.parametrize("forced_by", ["task", "request"])
def test_task_requires_cloud_escalates_whole_request_without_streaming(
    tmp_path: Path, forced_by: str
) -> None:
    backend = ScriptedStreamingBackend(["判", "定", "："])
    cloud = FakeCloud(text="云端整段答案", output_tokens=4)
    config = edge_config(tmp_path, probe_path=write_trigger_probe(tmp_path, 1))
    router = Router(backend, config, cloud_client=cloud)
    request = InferRequest(
        prompt="这批焊点是否放行",
        task="quality_verdict" if forced_by == "task" else "general",
        require_cloud=forced_by == "request",
        max_tokens=64,
        deadline_ms=3000,
    )

    response = router.infer(request)

    assert backend.generate_until_calls == 0
    assert backend.generate_calls == 1
    assert len(cloud.calls) == 1
    cloud_request, _ = cloud.calls[0]
    assert cloud_request.partial_trace == ""
    assert cloud_request.handoff_index is None
    assert cloud_request.draft == "端侧完整答案。"
    assert response.source == "cloud"
    assert response.escalation_reason == "task_requires_cloud"
    assert response.text == "云端整段答案"
    assert response.handoff_index is None
    assert response.handoff_reason is None
    counters = router.metrics.snapshot()["counters"]
    assert "handoff_total" not in counters
    assert "legacy_fallback_total" not in counters


def test_handoff_skips_cloud_when_the_remaining_deadline_is_too_short(
    tmp_path: Path,
) -> None:
    backend = ScriptedStreamingBackend(["判", "定", "：", "返", "修", "。"])
    cloud = FakeCloud()
    config = edge_config(tmp_path, handoff_mode="oracle", cloud_rtt_ms=500)
    outbox = Outbox(config.outbox_path)
    router = Router(backend, config, cloud_client=cloud, outbox=outbox)
    request = InferRequest(
        prompt="焊点空洞 25%",
        max_tokens=64,
        deadline_ms=50,
        metadata={"gold_handoff_index": 1},
    )

    response = router.infer(request)

    assert response.source == "edge"
    assert response.degraded is True
    assert response.escalation_reason == "deadline"
    # The deadline is already shorter than the cloud RTT, so the streaming loop
    # stops at the first token boundary instead of emitting the full prefix.
    assert response.text == ""
    assert response.handoff_index == 0
    assert cloud.calls == []
    assert len(outbox.pending()) == 1
    counters = router.metrics.snapshot()["counters"]
    assert counters["handoff_total"] == 1
    assert counters["handoff_reason_deadline"] == 1
    assert "cloud_attempt_total" not in counters


def test_cost_budget_denial_degrades_without_a_cloud_call(tmp_path: Path) -> None:
    backend = ScriptedStreamingBackend(["判", "定", "：", "返", "修", "。"])
    cloud = FakeCloud()
    config = edge_config(
        tmp_path,
        handoff_mode="oracle",
        cost_per_cloud_call=1.0,
        max_cost_per_min=0.5,
    )
    outbox = Outbox(config.outbox_path)
    router = Router(backend, config, cloud_client=cloud, outbox=outbox)
    request = InferRequest(
        prompt="焊点空洞 25%",
        max_tokens=64,
        deadline_ms=3000,
        metadata={"gold_handoff_index": 1},
    )

    response = router.infer(request)

    assert response.source == "edge"
    assert response.degraded is True
    assert response.escalation_reason == "budget"
    assert response.text == "判定：返修。"
    assert cloud.calls == []
    assert len(outbox.pending()) == 1
    assert router.budget.snapshot()["window_cost"] == 0.0
    counters = router.metrics.snapshot()["counters"]
    assert counters["handoff_total"] == 1
    assert "cloud_attempt_total" not in counters


def test_cloud_failure_keeps_the_partial_answer_and_counts_the_breaker(
    tmp_path: Path,
) -> None:
    backend = ScriptedStreamingBackend(["判", "定", "：", "返", "修", "。"])
    config = edge_config(tmp_path, handoff_mode="oracle")
    outbox = Outbox(config.outbox_path)
    router = Router(backend, config, cloud_client=FakeCloud(fail=True), outbox=outbox)
    request = InferRequest(
        prompt="焊点空洞 25%",
        max_tokens=64,
        deadline_ms=3000,
        metadata={"gold_handoff_index": 1},
    )

    response = router.infer(request)

    assert response.source == "edge"
    assert response.degraded is True
    assert response.text == "判定：返修。"
    assert response.handoff_index == 6
    assert response.handoff_reason == "low_prefix_trust"
    counters = router.metrics.snapshot()["counters"]
    assert counters["cloud_failure_total"] == 1
    assert counters["handoff_total"] == 1
    assert router.breaker.snapshot()["failures"] == 1
    pending = outbox.pending()
    assert len(pending) == 1
    assert pending[0]["edge_response"]["text"] == "判定：返修。"
    assert pending[0]["edge_response"]["handoff_index"] == 6


def test_offline_handoff_queues_partial_trace_and_replays_it(tmp_path: Path) -> None:
    backend = ScriptedStreamingBackend(["判", "定", "：", "返", "修", "。"])
    config = edge_config(tmp_path, handoff_mode="oracle", offline=True)
    outbox = Outbox(config.outbox_path)
    router = Router(backend, config, cloud_client=FakeCloud(), outbox=outbox)
    request = InferRequest(
        prompt="焊点空洞 25%",
        max_tokens=64,
        deadline_ms=3000,
        metadata={"gold_handoff_index": 1},
    )

    response = router.infer(request)

    assert response.source == "edge"
    assert response.degraded is True
    assert response.escalation_reason == "offline"
    pending = outbox.pending()
    assert len(pending) == 1
    cached = pending[0]["edge_response"]
    assert cached["text"] == "判定：返修。"
    assert cached["handoff_index"] == 6
    assert cached["handoff_reason"] == "low_prefix_trust"

    replay_cloud = FakeCloud(text="后续。", output_tokens=3)
    flusher = OutboxFlusher(outbox, replay_cloud, interval_s=60)
    assert flusher.flush_once() == 1

    assert len(replay_cloud.calls) == 1
    replayed, _ = replay_cloud.calls[0]
    assert replayed.partial_trace == "判定：返修。"
    assert replayed.draft == ""
    assert replayed.handoff_index == 6
    assert replayed.edge_tokens_before_handoff == 6
    done = outbox.items("done")
    assert len(done) == 1
    assert done[0]["cloud_text"] == "判定：返修。后续。"


def test_single_handoff_calls_cloud_once_even_when_the_policy_keeps_triggering(
    tmp_path: Path,
) -> None:
    backend = ScriptedStreamingBackend(["a", "b", "。", "c", "。"])
    cloud = FakeCloud(text="tail", output_tokens=2)
    policy = AlwaysHandoffPolicy()
    config = edge_config(tmp_path, handoff_mode="probe")
    router = Router(
        backend, config, cloud_client=cloud, handoff_policy=policy
    )
    request = InferRequest(prompt="反复触发的策略", max_tokens=32, deadline_ms=3000)

    first = router.infer(request)

    assert policy.reset_calls == 1
    assert backend.generate_until_calls == 1
    assert len(cloud.calls) == 1
    assert first.text == "ab。tail"
    assert first.handoff_index == 3
    assert router.metrics.snapshot()["counters"]["handoff_total"] == 1


def test_probe_unavailable_falls_back_to_the_whole_segment_path(tmp_path: Path) -> None:
    backend = ScriptedStreamingBackend(["判", "定", "："])
    cloud = FakeCloud()
    config = edge_config(tmp_path, probe_path=str(tmp_path / "missing-probe.json"))
    router = Router(backend, config, cloud_client=cloud)
    request = InferRequest(prompt="安全操作规范", deadline_ms=3000)

    response = router.infer(request)

    assert backend.generate_until_calls == 0
    assert backend.generate_calls == 1
    assert cloud.calls == []
    assert response.source == "edge"
    assert response.text == "端侧完整答案。"
    counters = router.metrics.snapshot()["counters"]
    assert counters["probe_unavailable_total"] == 1
    assert counters["legacy_fallback_total"] == 1
    assert "handoff_total" not in counters


def test_non_streaming_backend_falls_back_with_streaming_unavailable_metric(
    tmp_path: Path,
) -> None:
    backend = ScriptedStreamingBackend(["判", "定", "："])

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
                text="整段端侧答案。",
                token_logprobs=[-0.05] * 40,
                model_id=self.model_id,
                latency_ms=1.0,
            )

    whole_only = WholeAnswerOnlyBackend()
    cloud = FakeCloud()
    config = edge_config(tmp_path, probe_path=write_trigger_probe(tmp_path, 1))
    router = Router(whole_only, config, cloud_client=cloud)
    request = InferRequest(prompt="安全操作规范", deadline_ms=3000)

    response = router.infer(request)

    assert backend.generate_until_calls == 0
    assert whole_only.generate_calls == 1
    assert cloud.calls == []
    assert response.source == "edge"
    counters = router.metrics.snapshot()["counters"]
    assert counters["streaming_unavailable_total"] == 1
    assert counters["legacy_fallback_total"] == 1
    assert "probe_unavailable_total" not in counters


@pytest.mark.parametrize("mode", ["legacy", "off"])
def test_legacy_and_off_modes_never_stream_or_send_partial_trace(
    tmp_path: Path, mode: str
) -> None:
    backend = ScriptedStreamingBackend(["判", "定", "：", "返", "修", "。"])
    cloud = FakeCloud(text="云端整段答案", output_tokens=4)
    config = edge_config(tmp_path, handoff_mode=mode)
    router = Router(backend, config, cloud_client=cloud)
    request = InferRequest(
        prompt="这批焊点是否放行", require_cloud=True, max_tokens=64, deadline_ms=3000
    )

    response = router.infer(request)

    assert backend.generate_until_calls == 0
    assert backend.generate_calls == 1
    assert len(cloud.calls) == 1
    cloud_request, _ = cloud.calls[0]
    assert cloud_request.partial_trace == ""
    assert cloud_request.handoff_index is None
    assert cloud_request.draft == "端侧完整答案。"
    assert response.source == "cloud"
    assert response.text == "云端整段答案"
    assert response.handoff_index is None


def test_oracle_without_a_gold_index_never_hands_off(tmp_path: Path) -> None:
    backend = ScriptedStreamingBackend(["安全", "完成。"])
    cloud = FakeCloud()
    config = edge_config(tmp_path, handoff_mode="oracle")
    router = Router(backend, config, cloud_client=cloud)

    response = router.infer(InferRequest(prompt="安全操作规范", deadline_ms=3000))

    assert response.source == "edge"
    assert backend.generate_until_calls == 1
    assert cloud.calls == []
    assert "handoff_total" not in router.metrics.snapshot()["counters"]


def test_handoff_cost_estimate_uses_actual_cloud_output_tokens(tmp_path: Path) -> None:
    backend = ScriptedStreamingBackend(
        ["判", "定", "：", "返", "修", "。", "后"], prompt_tokens=17
    )
    cloud = FakeCloud(text="后续。", output_tokens=5)
    config = edge_config(
        tmp_path,
        handoff_mode="oracle",
        cost_per_edge_token=0.001,
        cost_per_cloud_input_token=0.002,
        cost_per_cloud_output_token=0.01,
        cost_per_cloud_call=0.5,
        cloud_rtt_ms=40.0,
    )
    router = Router(backend, config, cloud_client=cloud)
    request = InferRequest(
        prompt="焊点空洞 25%",
        max_tokens=64,
        deadline_ms=3000,
        metadata={"gold_handoff_index": 2},
    )

    response = router.infer(request)

    assert response.cost_estimate == {
        "edge_tokens": 6,
        "cloud_prefill_tokens": 23,
        "cloud_decode_tokens_expected": 5,
        "cloud_calls": 1,
        "rtt_ms": 40.0,
        "estimated_cost": 0.602,
        "edge_cost": 0.006,
        "cloud_prefill_cost": 0.046,
        "cloud_decode_cost": 0.05,
        "cloud_call_cost": 0.5,
    }
    assert response.cloud_output_tokens == 5
    # The budget gate used the pre-handoff estimate, not the actual response.
    budget = router.budget.snapshot()
    assert budget["cloud_calls"] == 1
    assert budget["window_cost"] == pytest.approx(1.132)


def test_cloud_restating_the_prefix_is_recorded_without_duplication(
    tmp_path: Path,
) -> None:
    backend = ScriptedStreamingBackend(["判", "定", "：", "返", "修", "。"])
    cloud = FakeCloud(
        text="判定：返修。继续。",
        output_tokens=7,
        continued_from_prefix=False,
    )
    config = edge_config(tmp_path, handoff_mode="oracle")
    router = Router(backend, config, cloud_client=cloud)
    request = InferRequest(
        prompt="焊点空洞 25%",
        max_tokens=64,
        deadline_ms=3000,
        metadata={"gold_handoff_index": 1},
    )

    response = router.infer(request)

    assert response.source == "cloud"
    assert response.text == "判定：返修。继续。"
    assert router.metrics.snapshot()["counters"]["continuation_mismatch_total"] == 1


def test_allowed_prefix_rewrite_without_a_declared_correction_is_a_continuation(
    tmp_path: Path,
) -> None:
    backend = ScriptedStreamingBackend(["判", "定", "：", "返", "修", "。"])
    cloud = FakeCloud(
        text="后续。",
        output_tokens=3,
        corrected_prefix=False,
        continued_from_prefix=True,
    )
    config = edge_config(tmp_path, handoff_mode="oracle", allow_prefix_rewrite=True)
    router = Router(backend, config, cloud_client=cloud)
    request = InferRequest(
        prompt="焊点空洞 25%",
        max_tokens=64,
        deadline_ms=3000,
        metadata={"gold_handoff_index": 1},
    )

    response = router.infer(request)

    assert response.source == "cloud"
    assert response.text == "判定：返修。后续。"
    assert "云端未声明纠正，按续写处理" in " ".join(response.trace)
    counters = router.metrics.snapshot()["counters"]
    assert "continuation_mismatch_total" not in counters


def test_undeclared_full_answer_records_a_continuation_mismatch(tmp_path: Path) -> None:
    backend = ScriptedStreamingBackend(["判", "定", "：", "返", "修", "。"])
    cloud = FakeCloud(
        text="云端整段答案。",
        output_tokens=6,
        continued_from_prefix=False,
    )
    config = edge_config(tmp_path, handoff_mode="oracle")
    router = Router(backend, config, cloud_client=cloud)
    request = InferRequest(
        prompt="焊点空洞 25%",
        max_tokens=64,
        deadline_ms=3000,
        metadata={"gold_handoff_index": 1},
    )

    response = router.infer(request)

    assert response.text == "判定：返修。云端整段答案。"
    counters = router.metrics.snapshot()["counters"]
    assert counters["continuation_mismatch_total"] == 1
    assert "未声明续写" in " ".join(response.trace)


def test_probe_losing_a_required_feature_falls_back_to_whole_segment(tmp_path: Path) -> None:
    backend = ScriptedStreamingBackend(["安全", "完成。"])
    cloud = FakeCloud()
    config = edge_config(
        tmp_path,
        probe_path=write_feature_probe(tmp_path, "kl_base_shadow"),
    )
    router = Router(backend, config, cloud_client=cloud)
    request = InferRequest(prompt="安全操作规范", deadline_ms=3000)

    response = router.infer(request)

    assert backend.generate_until_calls == 1
    assert backend.generate_calls == 0
    assert cloud.calls == []
    assert response.source == "edge"
    assert response.text == "安全完成。"
    counters = router.metrics.snapshot()["counters"]
    assert counters["probe_unavailable_total"] == 1
    assert counters["legacy_fallback_total"] == 1
    assert counters["edge_success_total"] == 1
    assert "probe" in " ".join(response.trace)


def test_format_guard_triggers_a_format_invalid_handoff_for_invalid_json(
    tmp_path: Path,
) -> None:
    backend = ScriptedStreamingBackend(['{"a": 1,', ' "b":', " }", "x"])
    cloud = FakeCloud(text="后续", output_tokens=2)
    config = edge_config(tmp_path, handoff_mode="oracle")
    router = Router(backend, config, cloud_client=cloud)
    request = InferRequest(
        prompt="输出 JSON",
        max_tokens=64,
        deadline_ms=3000,
        metadata={"expected_format": "json"},
    )

    response = router.infer(request)

    assert len(cloud.calls) == 1
    cloud_request, _ = cloud.calls[0]
    assert cloud_request.partial_trace == '{"a": 1, "b": }'
    assert cloud_request.handoff_reason == "format_invalid"
    assert response.handoff_reason == "format_invalid"
    counters = router.metrics.snapshot()["counters"]
    assert counters["handoff_reason_format_invalid"] == 1


def test_prefix_rewrite_uses_the_full_cloud_answer_without_mismatch(
    tmp_path: Path,
) -> None:
    backend = ScriptedStreamingBackend(["判", "定", "：", "返", "修", "。"])
    cloud = FakeCloud(
        text="纠正后的完整答案。",
        output_tokens=6,
        corrected_prefix=True,
        continued_from_prefix=False,
    )
    config = edge_config(tmp_path, handoff_mode="oracle", allow_prefix_rewrite=True)
    router = Router(backend, config, cloud_client=cloud)
    request = InferRequest(
        prompt="焊点空洞 25%",
        max_tokens=64,
        deadline_ms=3000,
        metadata={"gold_handoff_index": 1},
    )

    response = router.infer(request)

    assert response.source == "cloud"
    assert response.text == "纠正后的完整答案。"
    assert cloud.calls[0][0].allow_prefix_rewrite is True
    counters = router.metrics.snapshot()["counters"]
    assert "continuation_mismatch_total" not in counters


def test_invalid_final_format_is_flagged_but_the_cloud_answer_is_kept(
    tmp_path: Path,
) -> None:
    backend = ScriptedStreamingBackend(["判", "定", "：", "返", "修", "。"])
    cloud = FakeCloud(text="不是JSON", output_tokens=4)
    config = edge_config(tmp_path, handoff_mode="oracle")
    router = Router(backend, config, cloud_client=cloud)
    request = InferRequest(
        prompt="输出 JSON",
        max_tokens=64,
        deadline_ms=3000,
        metadata={"gold_handoff_index": 1, "expected_format": "json"},
    )

    response = router.infer(request)

    assert response.source == "cloud"
    assert response.text == "判定：返修。不是JSON"
    assert response.degraded is True
    counters = router.metrics.snapshot()["counters"]
    assert counters["final_format_invalid_total"] == 1


def test_legacy_cloud_path_prices_actual_tokens_and_records_the_estimate(
    tmp_path: Path,
) -> None:
    backend = ScriptedStreamingBackend([], full_text="端侧完整答案。", prompt_tokens=17)
    cloud = FakeCloud(text="云端完整答案", output_tokens=5)
    config = edge_config(
        tmp_path,
        handoff_mode="legacy",
        cost_per_edge_token=0.001,
        cost_per_cloud_input_token=0.002,
        cost_per_cloud_output_token=0.01,
        cost_per_cloud_call=0.5,
        cloud_rtt_ms=40.0,
    )
    router = Router(backend, config, cloud_client=cloud)
    request = InferRequest(
        prompt="这批焊点是否放行", require_cloud=True, max_tokens=64, deadline_ms=3000
    )

    response = router.infer(request)

    assert response.cost_estimate == {
        "edge_tokens": 40,
        "cloud_prefill_tokens": 57,
        "cloud_decode_tokens_expected": 5,
        "cloud_calls": 1,
        "rtt_ms": 40.0,
        "estimated_cost": 0.704,
        "edge_cost": 0.04,
        "cloud_prefill_cost": 0.114,
        "cloud_decode_cost": 0.05,
        "cloud_call_cost": 0.5,
    }
    # The budget gate uses the pre-call estimate (max_tokens of cloud decode).
    budget = router.budget.snapshot()
    assert budget["cloud_calls"] == 1
    assert budget["window_cost"] == pytest.approx(1.294)


@pytest.mark.parametrize("mode", ["legacy", "off"])
def test_legacy_edge_only_path_attaches_a_zero_cloud_cost(
    tmp_path: Path, mode: str
) -> None:
    backend = ScriptedStreamingBackend([], full_text="端侧完整答案。")
    config = edge_config(tmp_path, handoff_mode=mode, cost_per_edge_token=0.001)
    router = Router(backend, config, cloud_client=FakeCloud())

    response = router.infer(InferRequest(prompt="安全操作规范", deadline_ms=3000))

    assert response.source == "edge"
    assert response.cost_estimate is not None
    assert response.cost_estimate["cloud_calls"] == 0
    assert response.cost_estimate["rtt_ms"] == 0.0
    assert response.cost_estimate["edge_tokens"] == 40
    assert response.cost_estimate["estimated_cost"] == pytest.approx(0.04)


def test_legacy_cost_budget_denial_skips_the_cloud_call(tmp_path: Path) -> None:
    backend = ScriptedStreamingBackend([], full_text="端侧完整答案。")
    cloud = FakeCloud()
    config = edge_config(
        tmp_path,
        handoff_mode="legacy",
        cost_per_cloud_call=1.0,
        max_cost_per_min=0.5,
    )
    router = Router(backend, config, cloud_client=cloud)

    response = router.infer(
        InferRequest(prompt="这批焊点是否放行", require_cloud=True, deadline_ms=3000)
    )

    assert cloud.calls == []
    assert response.source == "edge"
    assert response.degraded is True
    assert "跳过云端增强：budget_max_cost" in " ".join(response.trace)
    assert router.budget.snapshot()["window_cost"] == 0.0


def test_cloud_budget_cost_window_and_legacy_semantics() -> None:
    budget = CloudBudget(
        max_calls_per_window=100,
        max_ratio=1.0,
        max_cost_per_min=1.0,
    )

    assert budget.allow(now=0.0, estimated_cost=0.4) == (True, "")
    budget.record(True, now=0.0, cost=0.4)
    assert budget.allow(now=1.0, estimated_cost=0.4) == (True, "")
    budget.record(True, now=1.0, cost=0.4)
    assert budget.allow(now=2.0, estimated_cost=0.3) == (False, "max_cost")
    budget.record(True, now=2.0, cost=0.3)

    snapshot = budget.snapshot(now=2.0)
    assert snapshot["cloud_calls"] == 3
    assert snapshot["window_cost"] == pytest.approx(1.1)
    assert snapshot["max_cost_per_min"] == pytest.approx(1.0)

    # After the window slides past the first events, cost is available again.
    assert budget.allow(now=62.0, estimated_cost=0.3) == (True, "")

    disabled = CloudBudget(max_calls_per_window=10, max_ratio=1.0)
    assert disabled.max_cost_per_min == 0.0
    disabled.record(True)
    assert disabled.allow(estimated_cost=1000.0) == (True, "")
