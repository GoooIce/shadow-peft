"""Model backends: deterministic mocks plus real HF / OpenAI-compatible clients.

The edge agent only depends on the ``Backend`` protocol. A deployment can swap
the 3B on-premise model for a 0.5B one, or replace the cloud vendor, without
touching routing, confidence, feedback or observability.
"""

from __future__ import annotations

import json
import math
import os
import random
import time
import urllib.error
import urllib.request
import zlib
from dataclasses import dataclass, field
from typing import Any, Protocol

from . import knowledge
from .protocol import InferRequest


class BackendError(RuntimeError):
    """A backend failure that the router may fall back from."""

    def __init__(self, message: str, *, retryable: bool = True, status: int | None = None) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.status = status


@dataclass(slots=True)
class Generation:
    text: str
    token_logprobs: list[float] | None
    model_id: str
    finish_reason: str = "stop"
    latency_ms: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)


class Backend(Protocol):
    model_id: str

    def generate(self, request: InferRequest, *, draft: str | None = None) -> Generation: ...


def _synth_logprobs(text: str, quality: float) -> list[float]:
    """Deterministic logprobs whose geometric mean is close to *quality*."""

    quality = max(0.02, min(0.995, quality))
    count = max(4, len(text) // 3)
    rng = random.Random(zlib.crc32(text.encode("utf-8")))
    center = math.log(quality)
    values = [min(-1e-4, center + rng.uniform(-0.30, 0.18)) for _ in range(count)]
    return values


class MockEdgeBackend:
    """Small on-premise model simulator (production-line knowledge only)."""

    model_id = "mock-edge-0.6b"

    def __init__(self, latency_ms: float = 0.0) -> None:
        self.latency_ms = max(0.0, float(latency_ms))

    def generate(self, request: InferRequest, *, draft: str | None = None) -> Generation:
        started = time.perf_counter()
        if self.latency_ms:
            time.sleep(self.latency_ms / 1000)
        entry = knowledge.lookup(request.prompt)
        text = entry.edge_answer
        quality = entry.edge_quality
        return Generation(
            text=text,
            token_logprobs=_synth_logprobs(text, quality),
            model_id=self.model_id,
            finish_reason="stop",
            latency_ms=(time.perf_counter() - started) * 1000,
            raw={"kb_key": entry.key, "difficulty": entry.difficulty, "edge_quality": quality},
        )


class MockCloudBackend:
    """Large cloud model simulator; can review an edge draft."""

    model_id = "mock-cloud-32b"

    def __init__(self, latency_ms: float = 0.0) -> None:
        self.latency_ms = max(0.0, float(latency_ms))

    def generate(self, request: InferRequest, *, draft: str | None = None) -> Generation:
        started = time.perf_counter()
        if self.latency_ms:
            time.sleep(self.latency_ms / 1000)
        entry = knowledge.lookup(request.prompt)
        text = entry.cloud_answer
        return Generation(
            text=text,
            token_logprobs=_synth_logprobs(text, entry.cloud_quality),
            model_id=self.model_id,
            finish_reason="stop",
            latency_ms=(time.perf_counter() - started) * 1000,
            raw={
                "kb_key": entry.key,
                "difficulty": entry.difficulty,
                "cloud_quality": entry.cloud_quality,
                "draft_reviewed": bool(draft),
            },
        )


class OpenAICompatibleBackend:
    """Cloud vendor adapter (vLLM / Qwen / DeepSeek / OpenAI-compatible)."""

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key: str = "",
        timeout_s: float = 30.0,
        system_prompt: str = "",
        supports_logprobs: bool = True,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.model_id = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.timeout_s = timeout_s
        self.system_prompt = system_prompt
        self.supports_logprobs = supports_logprobs
        self.extra_body = extra_body or {}

    def _endpoint(self) -> str:
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        return f"{self.base_url}/chat/completions"

    def _post(self, payload: dict[str, Any], timeout_s: float) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            self._endpoint(),
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise BackendError(
                f"云端返回 HTTP {exc.code}: {detail}",
                retryable=exc.code >= 500 or exc.code == 429,
                status=exc.code,
            ) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise BackendError(f"云端连接失败: {exc}", retryable=True) from exc

    def generate(self, request: InferRequest, *, draft: str | None = None) -> Generation:
        started = time.perf_counter()
        user_content = request.prompt
        if draft:
            user_content = (
                f"{request.prompt}\n\n端侧小模型草稿（请复核事实与数值，给出更完整、可执行的最终答案）：\n{draft}"
            )
        messages: list[dict[str, str]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": user_content})
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
        }
        if self.supports_logprobs:
            payload["logprobs"] = True
        payload.update(self.extra_body)
        timeout_s = min(self.timeout_s, max(0.05, request.deadline_ms / 1000))

        try:
            data = self._post(payload, timeout_s)
        except BackendError as exc:
            if self.supports_logprobs and exc.status == 400:
                # Some gateways reject the logprobs field; retry once without it.
                payload.pop("logprobs", None)
                self.supports_logprobs = False
                data = self._post(payload, max(0.05, timeout_s - (time.perf_counter() - started)))
            else:
                raise

        try:
            choice = data["choices"][0]
            text = (choice.get("message") or {}).get("content") or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise BackendError(f"云端响应格式不正确: {data}", retryable=False) from exc

        logprobs: list[float] | None = None
        content_logprobs = (choice.get("logprobs") or {}).get("content") or []
        parsed = [
            float(item["logprob"])
            for item in content_logprobs
            if isinstance(item, dict) and isinstance(item.get("logprob"), (int, float))
        ]
        if parsed:
            logprobs = parsed

        finish_reason = str(choice.get("finish_reason") or "stop")
        if finish_reason == "max_tokens":
            finish_reason = "length"
        return Generation(
            text=text,
            token_logprobs=logprobs,
            model_id=str(data.get("model") or self.model),
            finish_reason=finish_reason,
            latency_ms=(time.perf_counter() - started) * 1000,
            raw={"usage": data.get("usage", {})},
        )


class TransformersBackend:
    """Local Hugging Face decoder, with optional ShadowPEFT adapter (端侧影子)."""

    def __init__(
        self,
        model_path: str,
        *,
        device: str = "auto",
        dtype: str = "bfloat16",
        shadow_ckpt: str | None = None,
        system_prompt: str = "",
        max_context_tokens: int = 2048,
        model_id: str | None = None,
    ) -> None:
        self.model_path = os.path.expanduser(model_path)
        self.device = device
        self.dtype = dtype
        self.shadow_ckpt = os.path.expanduser(shadow_ckpt) if shadow_ckpt else None
        self.system_prompt = system_prompt
        self.max_context_tokens = max_context_tokens
        self.model_id = model_id or (self.shadow_ckpt or self.model_path)
        self._model: Any = None
        self._tokenizer: Any = None
        self._torch: Any = None
        self._shadow = False

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        dtype = getattr(torch, self.dtype, torch.bfloat16)
        if self.device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        else:
            device = self.device

        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        try:
            model = AutoModelForCausalLM.from_pretrained(self.model_path, dtype=dtype)
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(self.model_path, torch_dtype=dtype)
        if self.shadow_ckpt:
            from shadow_peft import ShadowPeftModel

            model = ShadowPeftModel.from_pretrained(model, self.shadow_ckpt)
            self._shadow = True
        model.eval().to(device)
        self._tokenizer = tokenizer
        self._model = model

    def _format_prompt(self, prompt: str) -> str:
        tokenizer = self._tokenizer
        messages: list[dict[str, str]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})
        if getattr(tokenizer, "chat_template", None):
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return prompt

    def generate(self, request: InferRequest, *, draft: str | None = None) -> Generation:
        self._ensure_loaded()
        torch = self._torch
        tokenizer = self._tokenizer
        prompt = request.prompt
        if draft:
            prompt = (
                f"{prompt}\n\n端侧小模型草稿（请复核事实与数值）：\n{draft}"
            )
        prompt_text = self._format_prompt(prompt)
        inputs = tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_context_tokens,
        ).to(self._model.device if hasattr(self._model, "device") else "cpu")
        prompt_len = int(inputs["input_ids"].shape[1])
        started = time.perf_counter()

        with torch.inference_mode():
            if self._shadow:
                text, logprobs, finish_reason = self._generate_shadow(inputs, request)
            else:
                text, logprobs, finish_reason = self._generate_plain(inputs, request)

        return Generation(
            text=text.strip(),
            token_logprobs=logprobs,
            model_id=self.model_id,
            finish_reason=finish_reason,
            latency_ms=(time.perf_counter() - started) * 1000,
            raw={"prompt_tokens": prompt_len},
        )

    def _generate_plain(
        self, inputs: Any, request: InferRequest
    ) -> tuple[str, list[float] | None, str]:
        torch = self._torch
        tokenizer = self._tokenizer
        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": request.max_tokens,
            "output_scores": True,
            "return_dict_in_generate": True,
            "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        }
        if request.temperature > 0:
            generate_kwargs.update(
                {"do_sample": True, "temperature": request.temperature, "top_p": 0.95}
            )
        else:
            generate_kwargs.update({"do_sample": False})
        output = self._model.generate(**inputs, **generate_kwargs)
        new_ids = output.sequences[0, inputs["input_ids"].shape[1] :]
        text = tokenizer.decode(new_ids, skip_special_tokens=True)
        logprobs: list[float] = []
        for index, score in enumerate(output.scores):
            if index >= len(new_ids):
                break
            step = torch.log_softmax(score[0].float(), dim=-1)
            logprobs.append(float(step[new_ids[index]].item()))
        finish_reason = "length" if len(new_ids) >= request.max_tokens else "stop"
        return text, (logprobs or None), finish_reason

    def _generate_shadow(
        self, inputs: Any, request: InferRequest
    ) -> tuple[str, list[float] | None, str]:
        """Full-prefix decoding for ShadowPEFT; simple and correct at edge lengths."""

        torch = self._torch
        tokenizer = self._tokenizer
        ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")
        eos_token_id = tokenizer.eos_token_id
        generated: list[int] = []
        logprobs: list[float] = []
        finish_reason = "stop"
        for _ in range(request.max_tokens):
            output = self._model(input_ids=ids, attention_mask=attention_mask, use_cache=False)
            logits = output.logits[0, -1].float()
            step = torch.log_softmax(logits, dim=-1)
            if request.temperature > 0:
                probs = torch.softmax(logits / request.temperature, dim=-1)
                next_id = int(torch.multinomial(probs, 1).item())
            else:
                next_id = int(torch.argmax(logits).item())
            if eos_token_id is not None and next_id == eos_token_id:
                break
            logprobs.append(float(step[next_id].item()))
            generated.append(next_id)
            next_tensor = torch.tensor([[next_id]], device=ids.device, dtype=ids.dtype)
            ids = torch.cat([ids, next_tensor], dim=1)
            if attention_mask is not None:
                attention_mask = torch.cat(
                    [attention_mask, torch.ones((1, 1), device=ids.device, dtype=attention_mask.dtype)],
                    dim=1,
                )
            if len(generated) >= request.max_tokens:
                finish_reason = "length"
                break
        text = tokenizer.decode(generated, skip_special_tokens=True)
        return text, (logprobs or None), finish_reason


def build_backend(spec: dict[str, Any]) -> Backend:
    """Build a backend from a JSON config block."""

    kind = str(spec.get("kind", "")).strip()
    if kind == "mock_edge":
        return MockEdgeBackend(latency_ms=float(spec.get("latency_ms", 0.0)))
    if kind == "mock_cloud":
        return MockCloudBackend(latency_ms=float(spec.get("latency_ms", 0.0)))
    if kind in {"openai", "openai_compatible", "vllm"}:
        return OpenAICompatibleBackend(
            base_url=str(spec.get("base_url", "http://127.0.0.1:8000/v1")),
            model=str(spec.get("model", "")),
            api_key=str(spec.get("api_key", "")) or os.environ.get("OPENAI_API_KEY", ""),
            timeout_s=float(spec.get("timeout_s", 30.0)),
            system_prompt=str(spec.get("system_prompt", "")),
            supports_logprobs=bool(spec.get("supports_logprobs", True)),
            extra_body=spec.get("extra_body") or {},
        )
    if kind == "transformers":
        return TransformersBackend(
            model_path=str(spec.get("model_path", "")),
            device=str(spec.get("device", "auto")),
            dtype=str(spec.get("dtype", "bfloat16")),
            shadow_ckpt=spec.get("shadow_ckpt"),
            system_prompt=str(spec.get("system_prompt", "")),
            max_context_tokens=int(spec.get("max_context_tokens", 2048)),
            model_id=spec.get("model_id"),
        )
    raise ValueError(f"未知后端类型: {kind!r}")
