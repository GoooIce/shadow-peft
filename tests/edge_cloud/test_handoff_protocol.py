from __future__ import annotations

import pytest

from edge_cloud.protocol import (
    CloudRequest,
    CloudResponse,
    FeedbackRequest,
    InferResponse,
    ProtocolError,
)


def test_cloud_request_parses_legacy_payload_without_handoff_fields() -> None:
    request = CloudRequest.from_dict({"prompt": "旧客户端问题", "draft": "端侧草稿"})

    assert request.draft == "端侧草稿"
    assert request.partial_trace == ""
    assert request.handoff_index is None
    assert request.edge_tokens_before_handoff is None
    assert request.handoff_reason is None
    assert request.prefix_trust is None
    assert request.allow_prefix_rewrite is False


def test_cloud_request_roundtrips_handoff_fields() -> None:
    request = CloudRequest(
        prompt="问题",
        draft="端侧草稿",
        partial_trace="步骤一已完成。步骤二",
        handoff_index=7,
        edge_tokens_before_handoff=7,
        handoff_reason="low_prefix_trust",
        prefix_trust=0.25,
        allow_prefix_rewrite=True,
    )

    payload = request.to_dict()
    assert payload["draft"] == "端侧草稿"
    assert payload["partial_trace"] == "步骤一已完成。步骤二"
    assert payload["handoff_index"] == 7
    assert payload["edge_tokens_before_handoff"] == 7
    assert payload["handoff_reason"] == "low_prefix_trust"
    assert payload["prefix_trust"] == pytest.approx(0.25)
    assert payload["allow_prefix_rewrite"] is True

    restored = CloudRequest.from_dict(payload)
    assert restored.handoff_index == 7
    assert restored.partial_trace == "步骤一已完成。步骤二"
    assert restored.prefix_trust == pytest.approx(0.25)
    assert restored.handoff_reason == "low_prefix_trust"
    assert restored.allow_prefix_rewrite is True
    assert restored.draft == "端侧草稿"


def test_cloud_request_rejects_invalid_handoff_index() -> None:
    with pytest.raises(ProtocolError):
        CloudRequest(prompt="x", handoff_index=-1)
    with pytest.raises(ProtocolError):
        CloudRequest.from_dict({"prompt": "x", "handoff_index": -3})
    with pytest.raises(ProtocolError):
        CloudRequest.from_dict({"prompt": "x", "handoff_index": "abc"})


@pytest.mark.parametrize("value", [1.5, 3.0, True, False, "3"])
def test_cloud_request_handoff_index_rejects_non_integer_json_values(value: object) -> None:
    with pytest.raises(ProtocolError):
        CloudRequest(prompt="x", handoff_index=value)
    with pytest.raises(ProtocolError):
        CloudRequest.from_dict({"prompt": "x", "handoff_index": value})


def test_cloud_request_edge_token_count_is_strict_too() -> None:
    with pytest.raises(ProtocolError):
        CloudRequest(prompt="x", edge_tokens_before_handoff=2.5)
    with pytest.raises(ProtocolError):
        CloudRequest.from_dict({"prompt": "x", "edge_tokens_before_handoff": True})


def test_cloud_request_rejects_prefix_trust_outside_unit_interval() -> None:
    for value in (-0.01, 1.01):
        with pytest.raises(ProtocolError):
            CloudRequest(prompt="x", prefix_trust=value)
        with pytest.raises(ProtocolError):
            CloudRequest.from_dict({"prompt": "x", "prefix_trust": value})
    with pytest.raises(ProtocolError):
        CloudRequest.from_dict({"prompt": "x", "prefix_trust": "high"})


@pytest.mark.parametrize("value", [True, False, "0.5", float("nan"), float("inf")])
def test_cloud_request_prefix_trust_rejects_non_real_or_non_finite_values(
    value: object,
) -> None:
    with pytest.raises(ProtocolError):
        CloudRequest(prompt="x", prefix_trust=value)
    with pytest.raises(ProtocolError):
        CloudRequest.from_dict({"prompt": "x", "prefix_trust": value})


def test_feedback_request_parses_legacy_payload_with_handoff_defaults() -> None:
    feedback = FeedbackRequest.from_dict({"request_id": "r1", "accepted": True})

    assert feedback.handoff_index is None
    assert feedback.handoff_reason is None
    assert feedback.prefix_trust is None
    assert feedback.continued_from_prefix is False
    assert feedback.corrected_prefix is False
    assert feedback.cost_estimate is None

    restored = FeedbackRequest.from_dict(feedback.to_dict())
    assert restored == feedback

    null_flags = FeedbackRequest.from_dict(
        {
            "request_id": "r1",
            "accepted": True,
            "continued_from_prefix": None,
            "corrected_prefix": None,
        }
    )
    assert null_flags.continued_from_prefix is False
    assert null_flags.corrected_prefix is False


def test_feedback_request_roundtrips_handoff_fields() -> None:
    cost = {
        "edge_tokens": 6,
        "cloud_prefill_tokens": 23,
        "cloud_decode_tokens_expected": 5,
        "cloud_calls": 1,
        "rtt_ms": 40.0,
        "estimated_cost": 0.602,
    }
    feedback = FeedbackRequest.from_dict(
        {
            "request_id": "r2",
            "accepted": True,
            "handoff_index": 6,
            "handoff_reason": "low_prefix_trust",
            "prefix_trust": 0.2,
            "continued_from_prefix": True,
            "corrected_prefix": False,
            "cost_estimate": cost,
        }
    )

    assert feedback.handoff_index == 6
    assert feedback.handoff_reason == "low_prefix_trust"
    assert feedback.prefix_trust == pytest.approx(0.2)
    assert feedback.continued_from_prefix is True
    assert feedback.cost_estimate == cost
    assert FeedbackRequest.from_dict(feedback.to_dict()) == feedback


def test_feedback_request_rejects_invalid_handoff_field_types() -> None:
    with pytest.raises(ProtocolError):
        FeedbackRequest.from_dict(
            {"request_id": "r1", "accepted": True, "handoff_index": "6"}
        )
    with pytest.raises(ProtocolError):
        FeedbackRequest.from_dict(
            {"request_id": "r1", "accepted": True, "prefix_trust": "high"}
        )
    with pytest.raises(ProtocolError):
        FeedbackRequest.from_dict(
            {"request_id": "r1", "accepted": True, "cost_estimate": "cheap"}
        )


def test_cloud_request_accepts_handoff_boundary_values() -> None:
    request = CloudRequest(prompt="x", handoff_index=0, prefix_trust=0.0)
    assert request.handoff_index == 0
    assert request.prefix_trust == 0.0
    assert CloudRequest(prompt="x", prefix_trust=1.0).prefix_trust == 1.0


def test_cloud_response_handoff_fields_default_and_serialize() -> None:
    legacy = CloudResponse(
        request_id="r1", text="答案", confidence=0.9, model_id="m", latency_ms=12.0
    )
    assert legacy.continued_from_prefix is False
    assert legacy.corrected_prefix is False
    assert legacy.output_tokens == 0

    continued = CloudResponse(
        request_id="r2",
        text="续写",
        confidence=0.9,
        model_id="m",
        latency_ms=12.0,
        continued_from_prefix=True,
        corrected_prefix=True,
        output_tokens=17,
    )
    payload = continued.to_dict()
    assert payload["continued_from_prefix"] is True
    assert payload["corrected_prefix"] is True
    assert payload["output_tokens"] == 17


def test_infer_response_handoff_fields_default_and_serialize() -> None:
    legacy = InferResponse(
        request_id="r1",
        text="答案",
        source="edge",
        confidence=0.9,
        latency_ms=5.0,
        model_id="m",
    )
    payload = legacy.to_dict()
    assert payload["handoff_index"] is None
    assert payload["edge_tokens_before_handoff"] is None
    assert payload["handoff_reason"] is None
    assert payload["prefix_trust"] is None
    assert payload["cloud_output_tokens"] == 0
    assert payload["cost_estimate"] is None

    cost_estimate = {
        "edge_tokens": 7,
        "cloud_prefill_tokens": 7,
        "cloud_decode_tokens_expected": 12,
        "cloud_calls": 1,
        "rtt_ms": 40.0,
        "estimated_cost": 0.004,
    }
    handed_off = InferResponse(
        request_id="r2",
        text="答案",
        source="cloud",
        confidence=0.9,
        latency_ms=5.0,
        model_id="m",
        handoff_index=7,
        edge_tokens_before_handoff=7,
        handoff_reason="low_prefix_trust",
        prefix_trust=0.3,
        cloud_output_tokens=12,
        cost_estimate=cost_estimate,
    )
    handed_payload = handed_off.to_dict()
    assert handed_payload["handoff_index"] == 7
    assert handed_payload["edge_tokens_before_handoff"] == 7
    assert handed_payload["handoff_reason"] == "low_prefix_trust"
    assert handed_payload["prefix_trust"] == pytest.approx(0.3)
    assert handed_payload["cloud_output_tokens"] == 12
    assert handed_payload["cost_estimate"] == cost_estimate
