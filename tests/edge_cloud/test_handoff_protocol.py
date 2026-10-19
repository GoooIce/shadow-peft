from __future__ import annotations

import pytest

from edge_cloud.protocol import CloudRequest, CloudResponse, InferResponse, ProtocolError


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


def test_cloud_request_rejects_prefix_trust_outside_unit_interval() -> None:
    for value in (-0.01, 1.01):
        with pytest.raises(ProtocolError):
            CloudRequest(prompt="x", prefix_trust=value)
        with pytest.raises(ProtocolError):
            CloudRequest.from_dict({"prompt": "x", "prefix_trust": value})
    with pytest.raises(ProtocolError):
        CloudRequest.from_dict({"prompt": "x", "prefix_trust": "high"})


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
