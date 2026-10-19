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
import re
import time
import urllib.error
import urllib.request
import zlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from itertools import islice
from typing import Any, NamedTuple, Protocol, runtime_checkable

from . import knowledge
from .boundaries import detect_boundary
from .features import compute_handoff_features
from .handoff import HandoffOutcome, HandoffPolicy, HandoffStep
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
    handoff: HandoffOutcome | None = None
    output_tokens: int = 0


class Backend(Protocol):
    model_id: str

    def generate(self, request: InferRequest, *, draft: str | None = None) -> Generation: ...


@runtime_checkable
class StreamingBackend(Protocol):
    """Backend that can pause generation at a semantic handoff boundary."""

    model_id: str

    def generate_until(
        self,
        request: InferRequest,
        policy: HandoffPolicy,
        *,
        max_boundary_wait_tokens: int = 12,
        expected_format: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> Generation: ...


class _DecodeOutcome(NamedTuple):
    text: str
    token_logprobs: list[float] | None
    finish_reason: str
    handoff: HandoffOutcome | None
    output_tokens: int
    waited_tokens: int


StepContext = str | Mapping[str, Any] | None
StepSource = Iterable[
    tuple[str, float | None, Sequence[float], Sequence[float] | None, StepContext]
]


def _context_task(context: Any, fallback: str = "general") -> str:
    if isinstance(context, Mapping):
        task = context.get("task")
        if isinstance(task, str) and task.strip():
            return task.strip()
    return fallback


def _step_context(boundary_context: StepContext, task: str) -> tuple[str, float]:
    format_ok = 1.0
    if isinstance(boundary_context, Mapping):
        raw_task = boundary_context.get("task")
        if isinstance(raw_task, str) and raw_task.strip():
            task = raw_task.strip()
        raw_format = boundary_context.get("format_ok")
        if raw_format is not None:
            try:
                format_ok = float(raw_format)
            except (TypeError, ValueError):
                format_ok = 1.0
    return task, format_ok


def _step_expected_format(
    expected_format: str | None, boundary_context: StepContext
) -> str | None:
    if isinstance(boundary_context, Mapping):
        raw = boundary_context.get("expected_format")
        if isinstance(raw, str) and raw:
            return raw
    return expected_format


def _decode_until(
    step_source: StepSource,
    policy: HandoffPolicy,
    *,
    max_tokens: int,
    max_boundary_wait_tokens: int = 12,
    expected_format: str | None = None,
    context: dict[str, Any] | None = None,
) -> _DecodeOutcome:
    """Run the shared token loop over a backend-specific step source.

    ``step_source`` yields plain Python values
    ``(token_text, token_logprob, base_logits, shadow_logits, boundary_context)``.
    A step is observed *after* it is sampled and *before* it is committed, so a
    policy decides with the current prefix plus the next-token distribution. If
    the current prefix already ends on a qualifying boundary, a handoff request
    cuts immediately and the sampled token is never emitted. Otherwise the
    backend keeps committing tokens until ``detect_boundary`` confirms a
    sentence/newline/step/JSON-field edge, or until
    ``max_boundary_wait_tokens`` tokens have been committed after the request,
    in which case the cut is forced at the raw token boundary. If the source
    ends while a handoff is pending, the outcome resolves at the last committed
    token instead of being dropped.
    """

    if max_tokens < 0:
        raise ValueError("max_tokens 不能为负数")
    if max_boundary_wait_tokens < 0:
        raise ValueError("max_boundary_wait_tokens 不能为负数")

    policy.reset(context)
    default_task = _context_task(context)

    text = ""
    output_logprobs: list[float] = []
    scored_logprobs: list[float] = []
    committed = 0
    waited = 0
    pending = False
    pending_reason = None
    pending_trust: float | None = None
    pending_cost = None
    handoff: HandoffOutcome | None = None
    finish_reason = "stop"

    for token_text, token_logprob, base_logits, shadow_logits, boundary_context in islice(
        step_source, max_tokens
    ):
        token_text = "" if token_text is None else str(token_text)
        step_format = _step_expected_format(expected_format, boundary_context)
        boundary_kind = detect_boundary(text, expected_format=step_format)
        task, format_ok = _step_context(boundary_context, default_task)

        features = compute_handoff_features(
            base_logits,
            shadow_logits,
            token_logprobs=scored_logprobs,
        )
        features = replace(
            features,
            boundary_kind=boundary_kind or "token",
            task=task,
            format_ok=format_ok,
        )

        decision = policy.observe(
            HandoffStep(
                features=features,
                prefix_text=text,
                token=token_text,
                token_logprob=token_logprob,
            )
        )
        if not pending and decision.action == "handoff":
            pending = True
            pending_reason = decision.reason
            pending_trust = decision.prefix_trust
            pending_cost = decision.estimated_cost
            if boundary_kind is not None:
                # The sampled token is risky: stop before emitting it.
                handoff = HandoffOutcome(
                    handed_off=True,
                    reason=pending_reason,
                    boundary=boundary_kind,
                    handoff_index=committed,
                    partial_trace=text,
                    prefix_trust=pending_trust,
                    estimated_cost=pending_cost,
                )
                finish_reason = "handoff"
                break
            if max_boundary_wait_tokens == 0:
                # Zero budget: the sampled token is never emitted either.
                handoff = HandoffOutcome(
                    handed_off=True,
                    reason=pending_reason,
                    boundary="token",
                    handoff_index=committed,
                    partial_trace=text,
                    prefix_trust=pending_trust,
                    estimated_cost=pending_cost,
                )
                finish_reason = "handoff"
                break

        text += token_text
        committed += 1
        if token_logprob is None:
            scored_logprobs.append(float("nan"))
        else:
            logprob = float(token_logprob)
            scored_logprobs.append(logprob)
            if math.isfinite(logprob):
                output_logprobs.append(logprob)

        if not pending:
            continue

        waited += 1
        post_boundary = detect_boundary(text, expected_format=step_format)
        if post_boundary is not None:
            handoff = HandoffOutcome(
                handed_off=True,
                reason=pending_reason,
                boundary=post_boundary,
                handoff_index=committed,
                partial_trace=text,
                prefix_trust=pending_trust,
                estimated_cost=pending_cost,
            )
        elif waited >= max_boundary_wait_tokens:
            handoff = HandoffOutcome(
                handed_off=True,
                reason=pending_reason,
                boundary="token",
                handoff_index=committed,
                partial_trace=text,
                prefix_trust=pending_trust,
                estimated_cost=pending_cost,
            )
        if handoff is not None:
            finish_reason = "handoff"
            break

    if handoff is None and pending:
        # Generation ended before a semantic boundary (max_tokens or EOS): the
        # requested handoff still resolves at the last committed token.
        handoff = HandoffOutcome(
            handed_off=True,
            reason=pending_reason,
            boundary="token",
            handoff_index=committed,
            partial_trace=text,
            prefix_trust=pending_trust,
            estimated_cost=pending_cost,
        )
        finish_reason = "handoff"
    elif handoff is None:
        finish_reason = "length" if max_tokens > 0 and committed >= max_tokens else "stop"

    return _DecodeOutcome(
        text=text,
        token_logprobs=output_logprobs or None,
        finish_reason=finish_reason,
        handoff=handoff,
        output_tokens=committed,
        waited_tokens=waited,
    )


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


_MOCK_TOKEN = re.compile(
    r"[A-Za-z]+(?:['’\-][A-Za-z]+)*|\d+(?:[.,]\d+)*|\s+|.", re.DOTALL
)

# Base confidence used for tokens at or after ``risk_at``; log(0.05) is below
# the -2.0 low-probability threshold shared with the legacy confidence path.
_RISK_QUALITY = 0.05


def _tokenize_answer(text: str) -> list[str]:
    """Split a mock answer into Chinese characters, English words and whitespace."""

    return _MOCK_TOKEN.findall(text)


def _binary_logits(probability: float) -> list[float]:
    """Two-token distribution with an exact softmax probability."""

    p = max(1e-6, min(1.0 - 1e-6, float(probability)))
    return [math.log(p), math.log1p(-p)]


class MockStreamingEdgeBackend(MockEdgeBackend):
    """Deterministic token-streaming edge model for tests and the eval bench.

    The answer comes from the same knowledge base as ``MockEdgeBackend`` and is
    split into Chinese characters, English words and whitespace runs. Each step
    exposes a two-token base distribution whose top probability is
    ``edge_quality`` (or ``_RISK_QUALITY`` from ``risk_at`` onwards). With
    ``shadow_gap`` set, a shadow distribution moves that mass towards the
    competing token, so ``compute_handoff_features`` reports a real KL and
    top-1 agreement; without it the shadow fields stay ``None``.
    """

    model_id = "mock-streaming-edge-0.6b"

    def __init__(
        self,
        latency_ms: float = 0.0,
        *,
        risk_at: int | None = None,
        edge_quality: float | None = None,
        shadow_gap: float | None = None,
    ) -> None:
        super().__init__(latency_ms=latency_ms)
        if risk_at is not None:
            risk_at = int(risk_at)
            if risk_at < 0:
                raise ValueError("risk_at 必须非负")
        self.risk_at = risk_at
        if edge_quality is not None:
            edge_quality = float(edge_quality)
            if not math.isfinite(edge_quality):
                raise ValueError("edge_quality 必须是有限数值")
        self.edge_quality = edge_quality
        if shadow_gap is not None:
            shadow_gap = float(shadow_gap)
            if not math.isfinite(shadow_gap) or shadow_gap < 0.0:
                raise ValueError("shadow_gap 必须是非负有限数值")
        self.shadow_gap = shadow_gap

    def _quality_at(self, index: int, base_quality: float) -> float:
        quality = base_quality
        if self.risk_at is not None and index >= self.risk_at:
            quality = min(quality, _RISK_QUALITY)
        return max(0.02, min(0.995, quality))

    def _steps(
        self,
        tokens: Sequence[str],
        base_quality: float,
        task: str,
    ) -> Iterable[tuple[str, float, list[float], list[float] | None, dict[str, Any]]]:
        for index, token in enumerate(tokens):
            quality = self._quality_at(index, base_quality)
            base_logits = _binary_logits(quality)
            shadow_logits: list[float] | None = None
            if self.shadow_gap is not None:
                shadow_quality = max(0.01, min(0.99, quality - self.shadow_gap))
                shadow_logits = _binary_logits(shadow_quality)
            yield (
                token,
                math.log(quality),
                base_logits,
                shadow_logits,
                {"task": task, "format_ok": 1.0},
            )

    def generate_until(
        self,
        request: InferRequest,
        policy: HandoffPolicy,
        *,
        max_boundary_wait_tokens: int = 12,
        expected_format: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> Generation:
        started = time.perf_counter()
        if self.latency_ms:
            time.sleep(self.latency_ms / 1000)
        entry = knowledge.lookup(request.prompt)
        base_quality = (
            entry.edge_quality if self.edge_quality is None else self.edge_quality
        )
        tokens = _tokenize_answer(entry.edge_answer)
        task = _context_task(context, request.task)
        outcome = _decode_until(
            self._steps(tokens, base_quality, task),
            policy,
            max_tokens=request.max_tokens,
            max_boundary_wait_tokens=max_boundary_wait_tokens,
            expected_format=expected_format,
            context=context,
        )
        raw: dict[str, Any] = {
            "kb_key": entry.key,
            "difficulty": entry.difficulty,
            "edge_quality": base_quality,
            "risk_at": self.risk_at,
            "shadow_gap": self.shadow_gap,
        }
        if outcome.handoff is not None:
            raw["handoff_waited_tokens"] = outcome.waited_tokens
        return Generation(
            text=outcome.text,
            token_logprobs=outcome.token_logprobs,
            model_id=self.model_id,
            finish_reason=outcome.finish_reason,
            latency_ms=(time.perf_counter() - started) * 1000,
            raw=raw,
            handoff=outcome.handoff,
            output_tokens=outcome.output_tokens,
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
            from shadow_peft.task_models import ShadowForCausalLM

            model = ShadowPeftModel.from_pretrained(model, self.shadow_ckpt)
            model = ShadowForCausalLM(model, inference_mode="base_shadow")
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

    def generate_until(
        self,
        request: InferRequest,
        policy: HandoffPolicy,
        *,
        max_boundary_wait_tokens: int = 12,
        expected_format: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> Generation:
        """Token-level decoding that lets *policy* pause at a semantic boundary."""

        self._ensure_loaded()
        torch = self._torch
        tokenizer = self._tokenizer
        prompt_text = self._format_prompt(request.prompt)
        inputs = tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_context_tokens,
        ).to(self._model.device if hasattr(self._model, "device") else "cpu")
        prompt_len = int(inputs["input_ids"].shape[1])
        started = time.perf_counter()

        with torch.inference_mode():
            outcome = _decode_until(
                self._torch_step_source(
                    inputs,
                    request,
                    expected_format=expected_format,
                    context=context,
                ),
                policy,
                max_tokens=request.max_tokens,
                max_boundary_wait_tokens=max_boundary_wait_tokens,
                expected_format=expected_format,
                context=context,
            )

        raw: dict[str, Any] = {
            "prompt_tokens": prompt_len,
            "shadow": self._shadow,
        }
        if outcome.handoff is not None:
            raw["handoff_waited_tokens"] = outcome.waited_tokens
        return Generation(
            text=outcome.text,
            token_logprobs=outcome.token_logprobs,
            model_id=self.model_id,
            finish_reason=outcome.finish_reason,
            latency_ms=(time.perf_counter() - started) * 1000,
            raw=raw,
            handoff=outcome.handoff,
            output_tokens=outcome.output_tokens,
        )

    def _decode_until(
        self,
        step_source: StepSource,
        policy: HandoffPolicy,
        *,
        max_tokens: int,
        max_boundary_wait_tokens: int = 12,
        expected_format: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> _DecodeOutcome:
        """Expose the shared loop for torch-free unit tests of the decoding path."""

        return _decode_until(
            step_source,
            policy,
            max_tokens=max_tokens,
            max_boundary_wait_tokens=max_boundary_wait_tokens,
            expected_format=expected_format,
            context=context,
        )

    def _torch_step_source(
        self,
        inputs: Any,
        request: InferRequest,
        *,
        expected_format: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> Iterable[tuple[str, float, list[float], list[float] | None, dict[str, Any]]]:
        """One full-prefix forward per token, yielding plain Python step values."""

        torch = self._torch
        tokenizer = self._tokenizer
        ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")
        eos_token_id = tokenizer.eos_token_id
        generated: list[int] = []
        decoded = ""
        task = _context_task(context, request.task)

        for _ in range(request.max_tokens):
            output = self._model(input_ids=ids, attention_mask=attention_mask, use_cache=False)
            logits = output.logits[0, -1].float()
            base_logits = logits.tolist()
            raw_shadow = getattr(output, "shadow_logits", None)
            shadow_logits = (
                None if raw_shadow is None else raw_shadow[0, -1].float().tolist()
            )
            step_logprobs = torch.log_softmax(logits, dim=-1)
            if request.temperature > 0:
                probs = torch.softmax(logits / request.temperature, dim=-1)
                next_id = int(torch.multinomial(probs, 1).item())
            else:
                next_id = int(torch.argmax(logits).item())
            if eos_token_id is not None and next_id == eos_token_id:
                break
            logprob = float(step_logprobs[next_id].item())
            generated.append(next_id)
            ids = torch.cat(
                [
                    ids,
                    torch.tensor([[next_id]], device=ids.device, dtype=ids.dtype),
                ],
                dim=1,
            )
            if attention_mask is not None:
                attention_mask = torch.cat(
                    [
                        attention_mask,
                        torch.ones(
                            (1, 1),
                            device=attention_mask.device,
                            dtype=attention_mask.dtype,
                        ),
                    ],
                    dim=1,
                )
            text_so_far = tokenizer.decode(generated, skip_special_tokens=True)
            if text_so_far.startswith(decoded):
                token_text = text_so_far[len(decoded):]
            else:
                token_text = tokenizer.decode([next_id], skip_special_tokens=True)
            decoded = text_so_far
            yield (
                token_text,
                logprob,
                base_logits,
                shadow_logits,
                {"task": task, "format_ok": 1.0, "expected_format": expected_format},
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
