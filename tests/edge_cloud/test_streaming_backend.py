from __future__ import annotations

import math

import pytest

from edge_cloud import knowledge
from edge_cloud.backends import (
    Generation,
    MockEdgeBackend,
    MockStreamingEdgeBackend,
    StreamingBackend,
    TransformersBackend,
    _decode_until,
)
from edge_cloud.cost import CostModel
from edge_cloud.handoff import HandoffDecision, HandoffPolicy, HandoffStep
from edge_cloud.protocol import InferRequest


class _RecordingPolicy:
    def __init__(self, trigger=None, *, estimated_cost=None) -> None:
        self.trigger = trigger
        self.estimated_cost = estimated_cost
        self.context: dict | None = None
        self.reset_calls = 0
        self.steps: list[HandoffStep] = []

    def reset(self, context: dict | None = None) -> None:
        self.reset_calls += 1
        self.context = context

    def observe(self, step: HandoffStep) -> HandoffDecision:
        self.steps.append(step)
        if self.trigger is not None and self.trigger(step):
            return HandoffDecision(
                action="handoff",
                reason="low_prefix_trust",
                boundary=step.features.boundary_kind,
                handoff_index=step.features.prefix_tokens,
                prefix_trust=0.35,
                estimated_cost=self.estimated_cost,
            )
        return HandoffDecision(action="continue")


def test_generation_new_fields_default_to_legacy_values() -> None:
    generation = Generation(text="x", token_logprobs=None, model_id="m")

    assert generation.handoff is None
    assert generation.output_tokens == 0


def test_streaming_protocol_detects_generate_until_backends_only() -> None:
    assert isinstance(MockStreamingEdgeBackend(), StreamingBackend)
    assert not isinstance(MockEdgeBackend(), StreamingBackend)


def test_no_trigger_generates_the_full_answer() -> None:
    backend = MockStreamingEdgeBackend()
    policy: HandoffPolicy = _RecordingPolicy()
    request = InferRequest(prompt="回流焊峰值温度")

    generation = backend.generate_until(request, policy, context={"task": "process_qa"})

    entry = knowledge.lookup(request.prompt)
    assert generation.text == entry.edge_answer
    assert generation.handoff is None
    assert generation.finish_reason == "stop"
    assert generation.output_tokens > 0
    assert generation.token_logprobs is not None
    assert len(generation.token_logprobs) == generation.output_tokens
    assert policy.reset_calls == 1
    assert policy.context == {"task": "process_qa"}
    assert len(policy.steps) == generation.output_tokens


def test_features_use_the_request_task_and_context_task_wins() -> None:
    backend = MockStreamingEdgeBackend()

    request_policy = _RecordingPolicy()
    backend.generate_until(
        InferRequest(prompt="焊点空洞 25%", task="custom_task"), request_policy
    )
    context_policy = _RecordingPolicy()
    backend.generate_until(
        InferRequest(prompt="焊点空洞 25%", task="custom_task"),
        context_policy,
        context={"task": "safety"},
    )

    assert request_policy.steps[0].features.task == "custom_task"
    assert context_policy.steps[0].features.task == "safety"


def test_handoff_mid_sentence_stops_at_the_next_sentence_boundary() -> None:
    backend = MockStreamingEdgeBackend()
    policy = _RecordingPolicy(lambda step: step.token == "定")

    generation = backend.generate_until(InferRequest(prompt="焊点空洞 25%"), policy)

    assert generation.text == "判定：返修。"
    assert generation.finish_reason == "handoff"
    assert generation.output_tokens == 6
    assert generation.handoff is not None
    assert generation.handoff.handed_off is True
    assert generation.handoff.reason == "low_prefix_trust"
    assert generation.handoff.boundary == "sentence"
    assert generation.handoff.handoff_index == generation.output_tokens
    assert generation.handoff.partial_trace == generation.text
    assert generation.handoff.prefix_trust == 0.35
    assert generation.raw["handoff_waited_tokens"] == 5


def test_zero_wait_budget_cuts_at_the_current_token_boundary() -> None:
    backend = MockStreamingEdgeBackend()
    policy = _RecordingPolicy(lambda step: step.token == "定")

    generation = backend.generate_until(
        InferRequest(prompt="焊点空洞 25%"),
        policy,
        max_boundary_wait_tokens=0,
    )

    assert generation.text == "判"
    assert generation.output_tokens == 1
    assert generation.handoff is not None
    assert generation.handoff.boundary == "token"
    assert generation.handoff.handoff_index == 1
    assert generation.raw["handoff_waited_tokens"] == 0


def test_handoff_outcome_carries_the_policy_cost_estimate() -> None:
    cost = CostModel(cost_per_cloud_call=0.01).estimate(cloud_calls=1)
    backend = MockStreamingEdgeBackend()
    policy = _RecordingPolicy(lambda step: step.token == "定", estimated_cost=cost)

    generation = backend.generate_until(InferRequest(prompt="焊点空洞 25%"), policy)

    assert generation.handoff is not None
    assert generation.handoff.estimated_cost is cost


def test_handoff_at_an_existing_boundary_does_not_emit_the_next_token() -> None:
    backend = MockStreamingEdgeBackend()
    policy = _RecordingPolicy(lambda step: step.token == "焊")

    generation = backend.generate_until(InferRequest(prompt="焊点空洞 25%"), policy)

    assert generation.text == "判定：返修。"
    assert generation.output_tokens == 6
    assert policy.steps[-1].features.boundary_kind == "sentence"
    assert generation.handoff is not None
    assert generation.handoff.boundary == "sentence"
    assert generation.handoff.handoff_index == 6
    assert generation.raw["handoff_waited_tokens"] == 0


def test_wait_budget_forces_a_cut_at_the_raw_token_boundary() -> None:
    backend = MockStreamingEdgeBackend()
    policy = _RecordingPolicy(lambda step: step.token == "定")

    generation = backend.generate_until(
        InferRequest(prompt="焊点空洞 25%"),
        policy,
        max_boundary_wait_tokens=2,
    )

    assert generation.text == "判定："
    assert generation.finish_reason == "handoff"
    assert generation.output_tokens == 3
    assert generation.handoff is not None
    assert generation.handoff.boundary == "token"
    assert generation.handoff.handoff_index == generation.output_tokens
    assert generation.raw["handoff_waited_tokens"] == 2


def test_shadow_features_are_none_without_a_shadow_gap() -> None:
    backend = MockStreamingEdgeBackend()
    policy = _RecordingPolicy()

    backend.generate_until(InferRequest(prompt="回流焊峰值温度"), policy)

    assert policy.steps
    for step in policy.steps:
        assert step.features.kl_base_shadow is None
        assert step.features.top1_agreement is None
        assert step.features.shadow_entropy is None
        assert step.features.shadow_margin is None
        assert step.features.base_top1_in_shadow_topk is None
        assert math.isfinite(step.features.base_entropy)


def test_shadow_gap_produces_measurable_disagreement() -> None:
    backend = MockStreamingEdgeBackend(edge_quality=0.9, shadow_gap=0.9)
    policy = _RecordingPolicy()

    backend.generate_until(InferRequest(prompt="回流焊峰值温度"), policy)

    first = policy.steps[0]
    assert first.features.kl_base_shadow is not None
    assert first.features.kl_base_shadow > 0.0
    assert first.features.top1_agreement == 0.0
    assert first.features.shadow_entropy is not None
    assert first.features.shadow_margin is not None
    assert first.features.base_top1_in_shadow_topk is not None


def test_zero_shadow_gap_reports_shadow_features_with_full_agreement() -> None:
    backend = MockStreamingEdgeBackend(shadow_gap=0.0)
    policy = _RecordingPolicy()

    backend.generate_until(InferRequest(prompt="回流焊峰值温度"), policy)

    first = policy.steps[0]
    assert first.features.kl_base_shadow == 0.0
    assert first.features.top1_agreement == 1.0
    assert first.features.shadow_entropy == first.features.base_entropy


def test_mock_streaming_output_is_reproducible() -> None:
    def run() -> tuple[Generation, _RecordingPolicy]:
        policy = _RecordingPolicy()
        generation = MockStreamingEdgeBackend(
            risk_at=4, shadow_gap=0.5
        ).generate_until(InferRequest(prompt="回流焊峰值温度"), policy)
        return generation, policy

    first_generation, first_policy = run()
    second_generation, second_policy = run()

    assert first_generation.text == second_generation.text
    assert first_generation.token_logprobs == second_generation.token_logprobs
    assert [step.features.to_dict() for step in first_policy.steps] == [
        step.features.to_dict() for step in second_policy.steps
    ]


def test_risk_at_drops_the_token_logprob_below_the_low_probability_threshold() -> None:
    backend = MockStreamingEdgeBackend(risk_at=2)
    policy = _RecordingPolicy()

    backend.generate_until(InferRequest(prompt="回流焊峰值温度"), policy)

    assert policy.steps[1].token_logprob is not None
    assert policy.steps[1].token_logprob > -2.0
    assert policy.steps[2].token_logprob == pytest.approx(math.log(0.05))
    assert policy.steps[2].features.low_prob_frac == 0.0
    assert policy.steps[3].features.low_prob_frac > 0.0


def test_output_tokens_respects_the_request_limit() -> None:
    backend = MockStreamingEdgeBackend()
    policy = _RecordingPolicy()

    generation = backend.generate_until(
        InferRequest(prompt="回流焊峰值温度", max_tokens=3), policy
    )

    assert generation.text == "回流焊"
    assert generation.finish_reason == "length"
    assert generation.handoff is None
    assert generation.output_tokens == 3
    assert generation.token_logprobs is not None
    assert len(generation.token_logprobs) == 3


def _step(
    token: str,
    logprob: float | None = -0.1,
    *,
    context: dict | None = None,
    shadow: tuple[float, float] | None = None,
) -> tuple[str, float | None, list[float], list[float] | None, dict | None]:
    shadow_logits = None if shadow is None else list(shadow)
    return token, logprob, [0.0, 1.0], shadow_logits, context


def test_decode_until_waits_for_a_scripted_sentence_boundary() -> None:
    steps = [
        _step("检查", context={"task": "fault_diagnosis", "format_ok": 0.0}),
        _step("完成", context={"task": "fault_diagnosis", "format_ok": 0.0}),
        _step("。"),
        _step("继续"),
        _step("回答"),
    ]
    policy = _RecordingPolicy(lambda step: step.token == "完成")

    outcome = _decode_until(steps, policy, max_tokens=8)

    assert outcome.text == "检查完成。"
    assert outcome.finish_reason == "handoff"
    assert outcome.output_tokens == 3
    assert outcome.waited_tokens == 2
    assert outcome.handoff is not None
    assert outcome.handoff.boundary == "sentence"
    assert outcome.handoff.handoff_index == 3
    assert policy.steps[0].features.task == "fault_diagnosis"
    assert policy.steps[0].features.format_ok == 0.0
    assert policy.steps[0].features.boundary_kind == "token"
    assert policy.steps[1].features.prefix_tokens == 1


def test_pending_handoff_resolves_when_the_step_source_ends() -> None:
    steps = [_step("检查"), _step("完成")]
    policy = _RecordingPolicy(lambda step: step.token == "检查")

    outcome = _decode_until(steps, policy, max_tokens=8)

    assert outcome.text == "检查完成"
    assert outcome.finish_reason == "handoff"
    assert outcome.handoff is not None
    assert outcome.handoff.boundary == "token"
    assert outcome.handoff.handoff_index == 2
    assert outcome.waited_tokens == 2


def test_decode_until_uses_shadow_logits_when_present() -> None:
    steps = [
        _step("a", shadow=(1.0, 0.0)),
        _step("b", shadow=(1.0, 0.0)),
    ]
    policy = _RecordingPolicy()

    outcome = _decode_until(steps, policy, max_tokens=2)

    assert outcome.token_logprobs == [-0.1, -0.1]
    assert policy.steps[0].features.kl_base_shadow is not None
    assert policy.steps[0].features.kl_base_shadow > 0.0
    assert policy.steps[0].features.top1_agreement == 0.0


def test_decode_until_falls_back_to_the_request_task_context() -> None:
    policy = _RecordingPolicy()

    _decode_until([_step("a")], policy, max_tokens=4, context={"task": "safety"})

    assert policy.context == {"task": "safety"}
    assert policy.steps[0].features.task == "safety"


def test_transformers_decode_loop_is_injectable_without_torch() -> None:
    backend = TransformersBackend("unused-model")
    policy = _RecordingPolicy()

    outcome = backend._decode_until([_step("a"), _step("。")], policy, max_tokens=4)

    assert outcome.text == "a。"
    assert outcome.finish_reason == "stop"
    assert outcome.output_tokens == 2
    assert outcome.handoff is None


def test_deadline_forces_a_handoff_at_the_current_token_boundary() -> None:
    policy = _RecordingPolicy()
    times = iter([0.0, 1.0])
    steps = [_step("a"), _step("b"), _step("c")]

    outcome = _decode_until(
        steps,
        policy,
        max_tokens=8,
        context={
            "deadline_at": 0.5,
            "min_remaining_ms": 150,
            "clock": lambda: next(times),
        },
    )

    assert outcome.text == "a"
    assert outcome.finish_reason == "handoff"
    assert outcome.output_tokens == 1
    assert outcome.handoff is not None
    assert outcome.handoff.handed_off is True
    assert outcome.handoff.reason == "deadline"
    assert outcome.handoff.boundary == "token"
    assert outcome.handoff.handoff_index == 1
    assert outcome.handoff.partial_trace == "a"
    assert len(policy.steps) == 1


def test_deadline_leaves_enough_time_when_the_clock_has_not_run_out() -> None:
    policy = _RecordingPolicy()

    outcome = _decode_until(
        [_step("a"), _step("。")],
        policy,
        max_tokens=4,
        context={
            "deadline_at": 10.0,
            "min_remaining_ms": 150,
            "clock": lambda: 0.0,
        },
    )

    assert outcome.finish_reason == "stop"
    assert outcome.handoff is None
    assert len(policy.steps) == 2


def test_decode_until_ignores_a_malformed_deadline_context() -> None:
    policy = _RecordingPolicy()

    outcome = _decode_until(
        [_step("a")],
        policy,
        max_tokens=2,
        context={"deadline_at": "soon", "min_remaining_ms": None},
    )

    assert outcome.text == "a"
    assert outcome.handoff is None


class _FakeChatTokenizer:
    chat_template = "fake-template"

    def __init__(self, *, reject_kwargs: bool = False) -> None:
        self.calls: list[dict[str, object]] = []
        self.reject_kwargs = reject_kwargs

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        **kwargs,
    ) -> str:
        del messages, tokenize, add_generation_prompt
        if self.reject_kwargs and kwargs:
            raise TypeError("template does not accept extra kwargs")
        self.calls.append(dict(kwargs))
        return "formatted"


def test_transformers_backend_passes_enable_thinking_to_chat_template() -> None:
    backend = TransformersBackend("unused", enable_thinking=False)
    tokenizer = _FakeChatTokenizer()
    backend._tokenizer = tokenizer

    assert backend._format_prompt("hello") == "formatted"
    assert tokenizer.calls == [{"enable_thinking": False}]


def test_transformers_backend_omits_enable_thinking_by_default() -> None:
    backend = TransformersBackend("unused")
    tokenizer = _FakeChatTokenizer()
    backend._tokenizer = tokenizer

    assert backend._format_prompt("hello") == "formatted"
    assert tokenizer.calls == [{}]


def test_transformers_backend_retries_without_rejected_chat_kwargs() -> None:
    backend = TransformersBackend("unused", enable_thinking=True)
    tokenizer = _FakeChatTokenizer(reject_kwargs=True)
    backend._tokenizer = tokenizer

    assert backend._format_prompt("hello") == "formatted"
    assert tokenizer.calls == [{}]


def test_build_backend_reads_enable_thinking() -> None:
    from edge_cloud.backends import build_backend

    backend = build_backend(
        {"kind": "transformers", "model_path": "/tmp/unused", "enable_thinking": False}
    )
    assert isinstance(backend, TransformersBackend)
    assert backend.enable_thinking is False


def test_transformers_backend_requires_shadow_ckpt_for_a_shadow_model() -> None:
    with pytest.raises(ValueError):
        TransformersBackend("unused", shadow_model_path="/tmp/shadow")


def test_build_backend_reads_shadow_model_path() -> None:
    from edge_cloud.backends import build_backend

    backend = build_backend(
        {
            "kind": "transformers",
            "model_path": "/tmp/base",
            "shadow_ckpt": "/tmp/ckpt",
            "shadow_model_path": "/tmp/shadow",
        }
    )
    assert isinstance(backend, TransformersBackend)
    assert backend.shadow_ckpt == "/tmp/ckpt"
    assert backend.shadow_model_path == "/tmp/shadow"
