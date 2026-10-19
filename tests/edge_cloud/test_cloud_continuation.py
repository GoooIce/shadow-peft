from __future__ import annotations

from edge_cloud.backends import Generation
from edge_cloud.cloud_server import CloudService
from edge_cloud.config import CloudConfig
from edge_cloud.continuation import assemble_continuation, looks_like_duplicate_prefix
from edge_cloud.http_utils import BaseJsonHandler, ServiceServer
from edge_cloud.protocol import CloudRequest
from edge_cloud.router import CloudClient


def test_assemble_continuation_appends_remainder_to_prefix() -> None:
    assert assemble_continuation("步骤一完成。", "步骤二完成。") == "步骤一完成。步骤二完成。"


def test_assemble_continuation_replaces_prefix_when_corrected() -> None:
    assert (
        assemble_continuation("旧前缀。", "纠正后的完整答案。", corrected=True)
        == "纠正后的完整答案。"
    )


def test_assemble_continuation_does_not_duplicate_restated_prefix() -> None:
    assert (
        assemble_continuation("步骤一完成。", "步骤一完成。步骤二完成。")
        == "步骤一完成。步骤二完成。"
    )


def test_assemble_continuation_handles_empty_sides() -> None:
    assert assemble_continuation("", "续写") == "续写"
    assert assemble_continuation("前缀", "") == "前缀"


def test_looks_like_duplicate_prefix_normalizes_whitespace() -> None:
    assert looks_like_duplicate_prefix("步骤一\n已完成。", "步骤一 已完成。 步骤二") is True
    assert looks_like_duplicate_prefix("步骤一 已完成。", "步骤一\n\n已完成。") is True


def test_looks_like_duplicate_prefix_rejects_unrelated_or_empty_prefix() -> None:
    assert looks_like_duplicate_prefix("步骤一完成。", "步骤二完成。") is False
    assert looks_like_duplicate_prefix("", "任意文本") is False
    assert looks_like_duplicate_prefix("比文本更长的前缀", "短") is False


class CapturingBackend:
    model_id = "fake-cloud"

    def __init__(
        self,
        text: str,
        *,
        token_logprobs: list[float] | None = None,
        raw: dict | None = None,
    ) -> None:
        self.text = text
        self.token_logprobs = token_logprobs
        self.raw = raw or {}
        self.calls: list[dict] = []

    def generate(self, request, *, draft=None) -> Generation:
        self.calls.append({"prompt": request.prompt, "draft": draft})
        return Generation(
            text=self.text,
            token_logprobs=self.token_logprobs,
            model_id=self.model_id,
            latency_ms=1.0,
            raw=self.raw,
        )


def make_service(tmp_path, backend: CapturingBackend) -> CloudService:
    config = CloudConfig(
        backend={"kind": "mock_cloud"},
        feedback_path=str(tmp_path / "cloud_feedback.jsonl"),
        registry_path=str(tmp_path / "registry.json"),
        dataset_path=str(tmp_path / "sft.jsonl"),
    )
    return CloudService(config, backend=backend)


def test_cloud_service_continues_from_partial_trace(tmp_path) -> None:
    backend = CapturingBackend("步骤二完成。", token_logprobs=[-0.1] * 5)
    service = make_service(tmp_path, backend)

    payload = service.enhance(
        CloudRequest(
            prompt="制定工艺方案",
            draft="端侧整段草稿",
            partial_trace="步骤一完成。",
        )
    )

    call = backend.calls[0]
    assert call["draft"] is None
    assert "步骤一完成。" in call["prompt"]
    assert "不要重复" in call["prompt"]
    assert "继续" in call["prompt"]
    assert payload["text"] == "步骤二完成。"
    assert payload["continued_from_prefix"] is True
    assert payload["corrected_prefix"] is False
    assert payload["output_tokens"] == 5


def test_cloud_service_prefix_rewrite_returns_full_replacement(tmp_path) -> None:
    backend = CapturingBackend("纠正前缀后的完整答案。")
    service = make_service(tmp_path, backend)

    payload = service.enhance(
        CloudRequest(
            prompt="制定工艺方案",
            draft="端侧整段草稿",
            partial_trace="错误前缀。",
            allow_prefix_rewrite=True,
        )
    )

    call = backend.calls[0]
    assert call["draft"] is None
    assert "错误前缀。" in call["prompt"]
    assert "纠正" in call["prompt"]
    assert payload["text"] == "纠正前缀后的完整答案。"
    assert payload["continued_from_prefix"] is False
    assert payload["corrected_prefix"] is True


def test_cloud_service_legacy_request_keeps_prompt_and_draft(tmp_path) -> None:
    backend = CapturingBackend("旧版云端答案", token_logprobs=[-0.1] * 4)
    service = make_service(tmp_path, backend)

    payload = service.enhance(CloudRequest(prompt="原始问题", draft="端侧旧草稿"))

    assert backend.calls[0] == {"prompt": "原始问题", "draft": "端侧旧草稿"}
    assert payload["text"] == "旧版云端答案"
    assert payload["continued_from_prefix"] is False
    assert payload["corrected_prefix"] is False
    assert payload["output_tokens"] == 4


def test_cloud_service_reports_restated_prefix_as_full_answer(tmp_path) -> None:
    prefix = "步骤一完成。"
    backend = CapturingBackend(prefix + "步骤二完成。")
    service = make_service(tmp_path, backend)

    payload = service.enhance(CloudRequest(prompt="制定工艺方案", partial_trace=prefix))

    assert payload["text"] == prefix + "步骤二完成。"
    assert payload["continued_from_prefix"] is False
    assert payload["metadata"]["continuation_mismatch"] is True
    assert any("重述" in line for line in payload["trace"])


def test_output_tokens_prefers_backend_usage(tmp_path) -> None:
    backend = CapturingBackend(
        "续写内容", token_logprobs=[-0.1] * 3, raw={"usage": {"completion_tokens": 42}}
    )
    service = make_service(tmp_path, backend)

    payload = service.enhance(
        CloudRequest(prompt="问题", partial_trace="前缀。")
    )

    assert payload["output_tokens"] == 42


def test_output_tokens_falls_back_to_deterministic_estimate(tmp_path) -> None:
    backend = CapturingBackend("没有 usage 也没有 logprobs 的续写答案")
    service = make_service(tmp_path, backend)

    first = service.enhance(CloudRequest(prompt="问题", partial_trace="前缀。"))
    second = service.enhance(CloudRequest(prompt="问题", partial_trace="前缀。"))

    assert first["output_tokens"] > 0
    assert first["output_tokens"] == second["output_tokens"]


def _serve_cloud_response(payload: dict) -> tuple[ServiceServer, dict]:
    seen: dict = {}

    class Handler(BaseJsonHandler):
        def route_post(self, path, request):
            seen.update(request)
            return payload

    return ServiceServer(Handler, "127.0.0.1", 0).start(), seen


def test_cloud_client_parses_continuation_response_fields() -> None:
    server, seen = _serve_cloud_response(
        {
            "request_id": "r1",
            "text": "续写部分",
            "confidence": 0.9,
            "model_id": "fake-cloud",
            "latency_ms": 12.0,
            "continued_from_prefix": True,
            "corrected_prefix": False,
            "output_tokens": 9,
        }
    )
    try:
        response = CloudClient(server.url).enhance(
            CloudRequest(
                prompt="问题",
                partial_trace="端侧前缀。",
                handoff_index=5,
                prefix_trust=0.4,
            )
        )
    finally:
        server.stop()

    assert seen["partial_trace"] == "端侧前缀。"
    assert seen["handoff_index"] == 5
    assert response.continued_from_prefix is True
    assert response.corrected_prefix is False
    assert response.output_tokens == 9


def test_cloud_client_defaults_missing_continuation_fields() -> None:
    server, _ = _serve_cloud_response(
        {
            "request_id": "r1",
            "text": "旧版答案",
            "confidence": 0.8,
            "model_id": "legacy-cloud",
            "latency_ms": 5.0,
        }
    )
    try:
        response = CloudClient(server.url).enhance(CloudRequest(prompt="旧问题"))
    finally:
        server.stop()

    assert response.continued_from_prefix is False
    assert response.corrected_prefix is False
    assert response.output_tokens == 0
