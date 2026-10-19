from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from edge_cloud.outbox import Outbox
from edge_cloud.protocol import CloudRequest, CloudResponse, InferRequest, InferResponse
from edge_cloud.router import OutboxFlusher

HANDOFF_TEXT = "判定：返修。"
HANDOFF_REASON = "low_prefix_trust"


class RecordingCloud:
    """Scripted cloud that records every replayed ``CloudRequest``."""

    def __init__(
        self, text: str = "后续。", *, corrected_prefix: bool = False
    ) -> None:
        self.text = text
        self.corrected_prefix = corrected_prefix
        self.calls: list[CloudRequest] = []

    def enhance(
        self, request: CloudRequest, *, timeout_ms: int | None = None
    ) -> CloudResponse:
        self.calls.append(request)
        return CloudResponse(
            request_id=request.request_id,
            text=self.text,
            confidence=0.91,
            model_id="recording-cloud",
            latency_ms=9.0,
            continued_from_prefix=not self.corrected_prefix,
            corrected_prefix=self.corrected_prefix,
            output_tokens=3,
        )


def handoff_response(request_id: str) -> InferResponse:
    return InferResponse(
        request_id=request_id,
        text=HANDOFF_TEXT,
        source="edge",
        confidence=0.5,
        latency_ms=5.0,
        model_id="scripted-edge",
        degraded=True,
        edge_text=HANDOFF_TEXT,
        edge_tokens_before_handoff=6,
        handoff_index=6,
        handoff_reason=HANDOFF_REASON,
        prefix_trust=0.2,
    )


def legacy_response(request_id: str) -> InferResponse:
    return InferResponse(
        request_id=request_id,
        text="端侧完整答案。",
        source="edge",
        confidence=0.3,
        latency_ms=4.0,
        model_id="mock-edge",
        degraded=True,
        edge_text="端侧完整答案。",
    )


def test_replayed_handoff_item_keeps_partial_trace_and_metadata(
    tmp_path: Path,
) -> None:
    outbox = Outbox(tmp_path / "outbox.jsonl")
    request = InferRequest(
        prompt="焊点空洞 25% 如何判定",
        task="fault_diagnosis",
        max_tokens=64,
    )
    outbox.enqueue(request, handoff_response(request.request_id), "offline")
    cloud = RecordingCloud("后续。")

    flushed = OutboxFlusher(outbox, cloud, interval_s=60).flush_once()

    assert flushed == 1
    assert len(cloud.calls) == 1
    sent = cloud.calls[0]
    assert sent.partial_trace == HANDOFF_TEXT
    assert sent.draft == ""
    assert sent.handoff_index == 6
    assert sent.edge_tokens_before_handoff == 6
    assert sent.handoff_reason == HANDOFF_REASON
    assert sent.prefix_trust == pytest.approx(0.2)
    assert sent.max_tokens == 58
    assert sent.request_id == request.request_id
    # router.py's flusher keeps the protocol default; EdgeService restores the
    # enqueue-time policy before the request reaches the network.
    assert sent.allow_prefix_rewrite is False

    done = outbox.items("done")
    assert len(done) == 1
    assert done[0]["cloud_text"] == "判定：返修。后续。"
    assert done[0]["cloud_confidence"] == pytest.approx(0.91)


def test_replayed_corrected_prefix_replaces_the_edge_prefix(tmp_path: Path) -> None:
    outbox = Outbox(tmp_path / "outbox.jsonl")
    request = InferRequest(prompt="焊点空洞 25% 如何判定", max_tokens=64)
    outbox.enqueue(request, handoff_response(request.request_id), "offline")
    cloud = RecordingCloud("纠正后的完整答案。", corrected_prefix=True)

    assert OutboxFlusher(outbox, cloud, interval_s=60).flush_once() == 1

    done = outbox.items("done")
    assert done[0]["cloud_text"] == "纠正后的完整答案。"
    assert cloud.calls[0].partial_trace == HANDOFF_TEXT


def test_replayed_legacy_item_keeps_the_whole_draft_semantics(tmp_path: Path) -> None:
    outbox = Outbox(tmp_path / "outbox.jsonl")
    request = InferRequest(prompt="回流焊峰值温度", max_tokens=32)
    outbox.enqueue(request, legacy_response(request.request_id), "low_confidence")
    cloud = RecordingCloud("云端整段答案")

    assert OutboxFlusher(outbox, cloud, interval_s=60).flush_once() == 1

    sent = cloud.calls[0]
    assert sent.draft == "端侧完整答案。"
    assert sent.partial_trace == ""
    assert sent.handoff_index is None
    assert sent.edge_tokens_before_handoff is None
    assert sent.handoff_reason is None
    assert sent.prefix_trust is None
    assert sent.max_tokens == 32
    assert sent.allow_prefix_rewrite is False
    done = outbox.items("done")
    assert done[0]["cloud_text"] == "云端整段答案"


def test_handoff_metadata_survives_the_outbox_roundtrip(tmp_path: Path) -> None:
    outbox = Outbox(tmp_path / "outbox.jsonl")
    request = InferRequest(prompt="焊点空洞 25% 如何判定", max_tokens=64)
    item: dict[str, Any] = outbox.enqueue(
        request, handoff_response(request.request_id), "offline"
    ).to_dict()

    stored = outbox.pending()[0]
    assert stored["item_id"] == item["item_id"]
    cached = stored["edge_response"]
    assert cached["text"] == HANDOFF_TEXT
    assert cached["edge_text"] == HANDOFF_TEXT
    assert cached["handoff_index"] == 6
    assert cached["edge_tokens_before_handoff"] == 6
    assert cached["handoff_reason"] == HANDOFF_REASON
    assert cached["prefix_trust"] == pytest.approx(0.2)
