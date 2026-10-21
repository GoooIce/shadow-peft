"""Shadow-only recovery pipeline for exported shadows (issue #46).

The exp7 exported shadow cannot independently follow instructions or emit a
gradable GSM8K answer (see ``docs/exp14-pyrodash-stage0.md``).  This module is
the MVP infrastructure for the recovery route: load base + H4B shadow + exp7
adapter, export a standalone shadow model, verify a save/load round-trip,
scope which parameters are trainable, build chat-template SFT rows, run a
small deterministic SFT loop, and grade GSM8K-style generations.

Heavy imports (torch / transformers / shadow_peft) stay inside the functions
so the pure helpers can be imported and unit-tested without a model runtime.
"""

from __future__ import annotations

import gc
import math
import random
import re
import shutil
import tempfile
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

IGNORE_INDEX = -100
TRAIN_SCOPES = ("backbone", "head", "all", "none")
PROMPT_MASK = IGNORE_INDEX
QA_KEYWORD_PASS_COVERAGE = 0.5
_QA_STRUCTURED_TASKS = ("quality_verdict", "process_calc")

_DTYPE_NAMES = {
    "bfloat16": "bfloat16",
    "bf16": "bfloat16",
    "float16": "float16",
    "fp16": "float16",
    "float32": "float32",
    "fp32": "float32",
}

_GSM8K_ANSWER_LINE = re.compile(r"####\s*([^\n\r]+)")
_GSM8K_NUMBER = re.compile(r"-?\d[\d,]*\.?\d*")
_WORD_TOKEN = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]")

_TOKENIZER_FILE_NAMES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "tokenizer.model",
    "chat_template.jinja",
    "chat_template.json",
)


@dataclass
class ShadowOnlyTrainConfig:
    """Configuration for loading and training an exported shadow only.

    ``train_scope`` is applied to the *exported* model:

    - ``backbone``: train the shadow transformer blocks (and its embeddings),
      freeze the projection and LM head; this is the recovery default.
    - ``head``: train projection + LM head only.
    - ``all``: train every exported parameter.
    - ``none``: freeze everything (diagnose / dry-run).

    ``epochs`` and ``steps`` are alternative ways to size the loop: when
    ``epochs`` is set, the loop runs ``ceil(rows / batch_size) * epochs`` steps
    and ``steps`` is ignored; otherwise it runs exactly ``steps`` steps.
    """

    train_scope: str = "backbone"
    lr: float = 1e-5
    steps: int = 100
    epochs: int | None = None
    batch_size: int = 1
    max_seq_len: int = 1024
    seed: int = 0
    dtype: str = "bfloat16"
    device: str = "cpu"
    grad_clip: float = 1.0
    weight_decay: float = 0.0
    warmup_steps: int = 0
    log_every: int = 10

    def __post_init__(self) -> None:
        if self.train_scope not in TRAIN_SCOPES:
            raise ValueError(
                f"train_scope 必须是 {TRAIN_SCOPES} 之一，收到 {self.train_scope!r}"
            )
        if self.epochs is not None:
            if self.epochs < 1:
                raise ValueError(f"epochs 必须 >= 1 或为 None，收到 {self.epochs}")
        elif self.steps < 1:
            raise ValueError(f"steps 必须 >= 1，收到 {self.steps}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size 必须 >= 1，收到 {self.batch_size}")
        if self.max_seq_len < 2:
            raise ValueError(f"max_seq_len 必须 >= 2，收到 {self.max_seq_len}")
        if self.lr <= 0:
            raise ValueError(f"lr 必须 > 0，收到 {self.lr}")
        if self.grad_clip < 0:
            raise ValueError(f"grad_clip 必须 >= 0，收到 {self.grad_clip}")
        if self.weight_decay < 0:
            raise ValueError(f"weight_decay 必须 >= 0，收到 {self.weight_decay}")
        if self.warmup_steps < 0:
            raise ValueError(f"warmup_steps 必须 >= 0，收到 {self.warmup_steps}")
        if self.epochs is None and self.warmup_steps > self.steps:
            raise ValueError(
                f"warmup_steps 必须在 [0, steps={self.steps}] 内，收到 {self.warmup_steps}"
            )
        if self.log_every < 1:
            raise ValueError(f"log_every 必须 >= 1，收到 {self.log_every}")
        if self.seed < 0:
            raise ValueError(f"seed 必须 >= 0，收到 {self.seed}")
        if self.dtype not in _DTYPE_NAMES:
            raise ValueError(
                f"dtype 必须是 {tuple(sorted(_DTYPE_NAMES))} 之一，收到 {self.dtype!r}"
            )
        if not self.device:
            raise ValueError("device 不能为空")

    @property
    def torch_dtype_name(self) -> str:
        return _DTYPE_NAMES[self.dtype]


@dataclass
class TrainableScopeReport:
    """Parameter counts after applying a trainable scope."""

    scope: str
    trainable_parameters: int
    total_parameters: int
    trainable_modules: list[str] = field(default_factory=list)

    @property
    def trainable_ratio(self) -> float:
        if self.total_parameters == 0:
            return 0.0
        return self.trainable_parameters / self.total_parameters

    def as_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "trainable_parameters": self.trainable_parameters,
            "total_parameters": self.total_parameters,
            "trainable_ratio": self.trainable_ratio,
            "trainable_modules": list(self.trainable_modules),
        }


def resolve_dtype(name: str) -> Any:
    """Resolve a dtype name to a ``torch.dtype`` (lazy torch import)."""
    if name not in _DTYPE_NAMES:
        raise ValueError(
            f"dtype 必须是 {tuple(sorted(_DTYPE_NAMES))} 之一，收到 {name!r}"
        )
    import torch

    return getattr(torch, _DTYPE_NAMES[name])


# ---------------------------------------------------------------------------
# Parameter scoping
# ---------------------------------------------------------------------------


def _backbone_module(model: Any) -> Any | None:
    """Best-effort locate the transformer backbone of an exported model."""
    import torch.nn as nn

    for attribute in ("shadow_model", "model", "transformer"):
        candidate = getattr(model, attribute, None)
        if isinstance(candidate, nn.Module):
            return candidate
    return None


def _head_modules(model: Any) -> list[tuple[str, Any]]:
    """Locate output-side modules: hidden projection and LM head."""
    import torch.nn as nn

    modules: list[tuple[str, Any]] = []
    backbone = _backbone_module(model)
    for attribute in ("shadow_hidden_projection", "lm_head"):
        candidate = getattr(model, attribute, None)
        if isinstance(candidate, nn.Module) and candidate is not backbone:
            modules.append((attribute, candidate))

    output_embeddings = None
    getter = getattr(model, "get_output_embeddings", None)
    if callable(getter):
        try:
            output_embeddings = getter()
        except Exception:  # pragma: no cover - defensive duck typing
            output_embeddings = None
    if isinstance(output_embeddings, nn.Module):
        named = {id(module) for _, module in modules}
        if id(output_embeddings) not in named and output_embeddings is not backbone:
            modules.append(("output_embeddings", output_embeddings))
    return modules


def set_trainable_scope(model: Any, scope: str) -> TrainableScopeReport:
    """Freeze everything, then unfreeze the parameters selected by ``scope``.

    Returns the resulting trainable/total parameter counts and the names of the
    modules that were re-enabled, so a run report can show exactly what moved.
    """
    if scope not in TRAIN_SCOPES:
        raise ValueError(f"scope 必须是 {TRAIN_SCOPES} 之一，收到 {scope!r}")

    for parameter in model.parameters():
        parameter.requires_grad = False

    enabled: list[str] = []
    backbone = _backbone_module(model)
    heads = _head_modules(model)

    def enable(name: str, module: Any) -> None:
        changed = False
        for parameter in module.parameters():
            parameter.requires_grad = True
            changed = True
        if changed:
            enabled.append(name)

    if scope == "all":
        for parameter in model.parameters():
            parameter.requires_grad = True
        enabled = ["all"]
    elif scope == "backbone":
        if backbone is not None:
            enable("backbone", backbone)
        else:
            # No recognizable backbone: keep everything except the heads.
            head_ids = {id(parameter) for _, module in heads for parameter in module.parameters()}
            for parameter in model.parameters():
                if id(parameter) not in head_ids:
                    parameter.requires_grad = True
            enabled = ["backbone"]
    elif scope == "head":
        for name, module in heads:
            enable(name, module)

    trainable, total = count_parameters(model)
    if scope != "none" and trainable == 0:
        raise ValueError(
            f"scope={scope!r} 没有启用任何参数；检查模型结构或改用 scope='all'"
        )
    return TrainableScopeReport(
        scope=scope,
        trainable_parameters=trainable,
        total_parameters=total,
        trainable_modules=enabled,
    )


def count_parameters(model: Any) -> tuple[int, int]:
    """Return ``(trainable, total)`` parameter counts."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


# ---------------------------------------------------------------------------
# Export round-trip
# ---------------------------------------------------------------------------


def _default_roundtrip_loader(path: Path, dtype: Any) -> Any:
    from transformers import AutoModelForCausalLM

    # Importing shadow_peft registers the projection-wrapper config with the
    # Auto classes; without it a saved H4B export is not loadable by name.
    import shadow_peft  # noqa: F401

    try:
        return AutoModelForCausalLM.from_pretrained(str(path), dtype=dtype)
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(str(path), torch_dtype=dtype)


def _roundtrip_report_template(
    *,
    prompt: str,
    atol: float,
    rtol: float,
    device: str,
    dtype: str,
) -> dict[str, Any]:
    return {
        "passed": False,
        "allclose": False,
        "max_abs_diff": None,
        "atol": atol,
        "rtol": rtol,
        "vocab_size": None,
        "n_tokens": 0,
        "prompt": prompt,
        "tokenizer_files": [],
        "model_files": [],
        "save_dir": None,
        "device": device,
        "dtype": dtype,
        "error": None,
    }


def export_roundtrip_check(
    model: Any,
    tokenizer: Any,
    *,
    prompt: str = "The final answer is",
    max_length: int = 64,
    device: str = "cpu",
    dtype: str | None = None,
    atol: float = 1e-4,
    rtol: float = 1e-4,
    save_dir: str | Path | None = None,
    keep_dir: bool = False,
    model_loader: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Save the exported model + tokenizer, reload, and compare fixed logits.

    The report carries the gate inputs (``passed``/``allclose``/
    ``max_abs_diff``), the saved vocabulary size, the tokenizer files that
    travelled with the weights, and any loader error.  ``model_loader`` is the
    reload seam; it defaults to ``AutoModelForCausalLM.from_pretrained`` and is
    injected by tests so no checkpoint is fetched.  ``dtype=None`` reloads in
    the model's own dtype so a bf16 export is not compared against an fp32
    reload.
    """
    torch = _lazy_torch()
    if dtype is None:
        parameter = next(model.parameters(), None)
        torch_dtype = parameter.dtype if parameter is not None else resolve_dtype("float32")
        dtype_name = str(torch_dtype).removeprefix("torch.")
    else:
        torch_dtype = resolve_dtype(dtype)
        dtype_name = dtype
    report = _roundtrip_report_template(
        prompt=prompt, atol=atol, rtol=rtol, device=device, dtype=dtype_name
    )
    directory = (
        Path(save_dir)
        if save_dir is not None
        else Path(tempfile.mkdtemp(prefix="shadow-only-roundtrip-"))
    )
    created_temp = save_dir is None
    was_training = bool(getattr(model, "training", False))

    try:
        directory.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(directory))
        tokenizer.save_pretrained(str(directory))
        report["save_dir"] = str(directory)
        report["tokenizer_files"] = [
            name for name in _TOKENIZER_FILE_NAMES if (directory / name).exists()
        ]
        report["model_files"] = sorted(
            path.name
            for path in directory.iterdir()
            if path.is_file() and path.name not in _TOKENIZER_FILE_NAMES
        )

        loader = model_loader or _default_roundtrip_loader
        reloaded = loader(directory, dtype=torch_dtype)
        if hasattr(reloaded, "eval"):
            reloaded.eval()
        # ``from_pretrained(dtype=...)`` loads parameters in the requested
        # dtype, but non-persistent buffers (e.g. Qwen3's rotary inv_freq) are
        # re-initialized in fp32 during module construction.  Cast them back so
        # a bf16 export is compared against a bf16 reload, not bf16 weights with
        # fp32 RoPE tables.
        reloaded.to(device=device, dtype=torch_dtype)

        encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        token_ids = normalize_token_ids(encoded)
        if not token_ids:
            fallback = getattr(tokenizer, "eos_token_id", None)
            token_ids = [int(fallback if fallback is not None else 0)]
        token_ids = token_ids[:max_length]
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)

        if hasattr(model, "eval"):
            model.eval()
        with torch.inference_mode():
            reference = model(
                input_ids=input_ids, attention_mask=attention_mask
            ).logits.float().cpu()
            reloaded_logits = reloaded(
                input_ids=input_ids, attention_mask=attention_mask
            ).logits.float().cpu()

        report["n_tokens"] = len(token_ids)
        config = getattr(model, "config", None)
        vocab_size = getattr(config, "vocab_size", None)
        report["vocab_size"] = int(vocab_size or reference.shape[-1])
        if reference.shape != reloaded_logits.shape:
            report["error"] = (
                f"logits shape mismatch: {tuple(reference.shape)} vs "
                f"{tuple(reloaded_logits.shape)}"
            )
        else:
            max_abs_diff = float((reference - reloaded_logits).abs().max().item())
            report["max_abs_diff"] = max_abs_diff
            report["allclose"] = bool(
                torch.allclose(reference, reloaded_logits, atol=atol, rtol=rtol)
            )
            report["passed"] = report["allclose"]
    except Exception as exc:  # report the failure instead of losing the diagnosis
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["passed"] = False
        report["allclose"] = False
    finally:
        if hasattr(model, "train"):
            model.train(was_training)
        if created_temp and not keep_dir:
            shutil.rmtree(directory, ignore_errors=True)
            report["save_dir"] = None

    return report


# ---------------------------------------------------------------------------
# Loading the exported shadow
# ---------------------------------------------------------------------------


def load_shadow_tokenizer(path: str | Path) -> Any:
    """Load the tokenizer that ships with the base model, ready for decoding."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(path))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def _load_base_causal_lm(path: str | Path, dtype: Any) -> Any:
    from transformers import AutoModelForCausalLM

    try:
        return AutoModelForCausalLM.from_pretrained(str(path), dtype=dtype)
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(str(path), torch_dtype=dtype)


def load_exported_shadow(
    base_path: str | Path,
    shadow_model_path: str | Path,
    shadow_ckpt: str | Path,
    *,
    device: str = "cpu",
    dtype: str = "bfloat16",
    tokenizer_path: str | Path | None = None,
    tokenizer_loader: Callable[[str | Path], Any] | None = None,
) -> tuple[Any, Any]:
    """Load base + explicit H4B shadow + adapter and export a standalone model.

    The returned model is the one ``export_shadow()`` produced: the shadow
    backbone plus the trained projection and the frozen base ``lm_head``, with
    no live reference to the base model or the adapter.  Those source objects
    are dropped (and CUDA memory is released) before returning, so a caller can
    train the export without holding the 4B teacher/base in memory.

    ``tokenizer_loader`` is an injection seam for tests; production uses
    :func:`load_shadow_tokenizer` on ``tokenizer_path or base_path``.
    """
    for label, path in (
        ("base", base_path),
        ("shadow_model", shadow_model_path),
        ("shadow_ckpt", shadow_ckpt),
    ):
        if not Path(path).exists():
            raise FileNotFoundError(f"找不到 {label} 路径: {path}")

    torch = _lazy_torch()
    torch_dtype = resolve_dtype(dtype)

    from shadow_peft import ShadowForCausalLM
    from shadow_peft.projected_causal_lm import AutoModelForCausalLMWithHiddenProjection

    base = _load_base_causal_lm(base_path, torch_dtype)
    try:
        shadow_model = AutoModelForCausalLMWithHiddenProjection.from_pretrained(
            str(shadow_model_path),
            dtype=torch_dtype,
            freeze_backbone=True,
            freeze_embed_tokens=True,
            freeze_lm_head=True,
        )
    except TypeError:
        shadow_model = AutoModelForCausalLMWithHiddenProjection.from_pretrained(
            str(shadow_model_path),
            torch_dtype=torch_dtype,
            freeze_backbone=True,
            freeze_embed_tokens=True,
            freeze_lm_head=True,
        )
    task = ShadowForCausalLM.from_pretrained(
        base,
        str(shadow_ckpt),
        shadow_model=shadow_model,
        inference_mode="shadow_only",
    )
    task.eval()

    exported = task.peft_model.export_shadow()
    exported.eval()
    exported.to(device)

    loader = tokenizer_loader or load_shadow_tokenizer
    tokenizer = loader(tokenizer_path if tokenizer_path is not None else base_path)

    # The export owns independent copies of the shadow/projection/head.
    del task, shadow_model, base
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return exported, tokenizer


# ---------------------------------------------------------------------------
# Chat-template SFT rows
# ---------------------------------------------------------------------------


def normalize_token_ids(encoded: Any) -> list[int]:
    """Normalize tokenizer output (list/tuple/tensor/BatchEncoding) to ids."""
    if encoded is None:
        return []
    if isinstance(encoded, Mapping):
        encoded = encoded.get("input_ids")
        return normalize_token_ids(encoded)
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if isinstance(encoded, (list, tuple)):
        if len(encoded) == 1 and isinstance(encoded[0], (list, tuple)):
            encoded = encoded[0]
        return [int(token) for token in encoded]
    raise TypeError(f"无法把 {type(encoded)} 归一化为 token ids")


def _chat_template_ids(
    tokenizer: Any,
    messages: Sequence[Mapping[str, str]],
    *,
    add_generation_prompt: bool,
) -> list[int]:
    if getattr(tokenizer, "chat_template", None):
        try:
            encoded = tokenizer.apply_chat_template(
                list(messages),
                tokenize=True,
                add_generation_prompt=add_generation_prompt,
                enable_thinking=False,
            )
        except TypeError:
            encoded = tokenizer.apply_chat_template(
                list(messages),
                tokenize=True,
                add_generation_prompt=add_generation_prompt,
            )
        return normalize_token_ids(encoded)

    text = "\n".join(str(message.get("content") or "") for message in messages)
    if add_generation_prompt:
        text += "\n"
    return normalize_token_ids(tokenizer(text, add_special_tokens=False))


def format_chat_prompt(
    tokenizer: Any, messages: Sequence[Mapping[str, str]]
) -> str:
    """Render a prompt string with the tokenizer's chat template."""
    normalized = [
        {"role": str(m.get("role") or "user"), "content": str(m.get("content") or "")}
        for m in messages
    ]
    if getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template(
                normalized,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                normalized, tokenize=False, add_generation_prompt=True
            )
    return "\n".join(message["content"] for message in normalized) + "\n"


def _common_prefix(left: Sequence[int], right: Sequence[int]) -> int:
    limit = min(len(left), len(right))
    index = 0
    while index < limit and left[index] == right[index]:
        index += 1
    return index


def _normalize_messages(row: Mapping[str, Any], index: int) -> list[dict[str, str]]:
    messages = row.get("messages")
    if messages is not None:
        if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
            raise ValueError(f"SFT row {index} 的 messages 必须是消息列表")
        normalized = [
            {
                "role": str(message.get("role") or ""),
                "content": str(message.get("content") or ""),
            }
            for message in messages
        ]
        if not normalized:
            raise ValueError(f"SFT row {index} 的 messages 为空")
        if normalized[-1]["role"] != "assistant":
            raise ValueError(
                f"SFT row {index} 的 messages 最后一条必须是 assistant answer，"
                f"收到 {normalized[-1]['role']!r}"
            )
        return normalized

    prompt = row.get("prompt", row.get("question"))
    answer = row.get("answer", row.get("solution"))
    if prompt is None or answer is None or not str(answer).strip():
        raise ValueError(
            f"SFT row {index} 需要非空的 prompt/answer 或带 assistant answer 的 messages"
        )
    return [
        {"role": "user", "content": str(prompt)},
        {"role": "assistant", "content": str(answer)},
    ]


def _masked_ids(
    tokenizer: Any, messages: Sequence[Mapping[str, str]]
) -> tuple[list[int], int]:
    """Tokenize a conversation and return ids plus the masked prompt length."""
    full_ids = _chat_template_ids(
        tokenizer, messages, add_generation_prompt=False
    )
    prefix_ids = _chat_template_ids(
        tokenizer, messages[:-1], add_generation_prompt=True
    )
    masked = _common_prefix(prefix_ids, full_ids)

    if masked == 0 and messages[-1]["role"] == "assistant":
        # Some templates render the assistant header separately; fall back to
        # locating the answer text verbatim at the end of the sequence.
        answer_ids = normalize_token_ids(
            tokenizer(messages[-1]["content"], add_special_tokens=False)
        )
        if answer_ids and full_ids[-len(answer_ids) :] == answer_ids:
            masked = len(full_ids) - len(answer_ids)

    if not full_ids or masked >= len(full_ids):
        raise ValueError(
            "SFT row 没有可训练的 assistant token；检查 chat template 或消息角色"
        )
    return full_ids, masked


def build_sft_rows(
    rows: Iterable[Mapping[str, Any]],
    tokenizer: Any,
    max_length: int,
) -> list[dict[str, Any]]:
    """Build chat-template SFT rows with prompt positions masked to ``-100``.

    Each input row is either ``{"prompt": ..., "answer": ...}`` or a
    conversation under ``"messages"`` whose last message is the assistant
    answer.  Oversized sequences are truncated from the **left** so the
    assistant answer (the only supervised span) survives; the returned row
    records ``truncated=True``.
    """
    if max_length < 2:
        raise ValueError(f"max_length 必须 >= 2，收到 {max_length}")
    if tokenizer is None:
        raise ValueError("build_sft_rows 需要 tokenizer")

    built: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"SFT row {index} 必须是 mapping")
        messages = _normalize_messages(row, index)
        full_ids, masked = _masked_ids(tokenizer, messages)

        truncated = len(full_ids) > max_length
        if truncated:
            cut = len(full_ids) - max_length
            input_ids = full_ids[cut:]
            masked = max(0, masked - cut)
        else:
            input_ids = full_ids

        labels = [PROMPT_MASK] * masked + input_ids[masked:]
        if not any(label != PROMPT_MASK for label in labels):
            raise ValueError(
                f"SFT row {index} 截断后没有 assistant token；增大 max_length"
            )
        built.append(
            {
                "input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
                "labels": labels,
                "truncated": truncated,
            }
        )
    return built


def _lazy_torch() -> Any:
    import torch

    return torch


# ---------------------------------------------------------------------------
# GSM8K-style grading (pure)
# ---------------------------------------------------------------------------


def parse_gsm8k_answer(text: str | None) -> str:
    """Return the normalized final answer of a GSM8K-style solution.

    Prefers the ``#### <number>`` line and falls back to the first number in
    the text, matching exp14's grader so numbers stay comparable.
    """
    if text is None:
        return ""
    text = str(text)
    match = _GSM8K_ANSWER_LINE.search(text)
    candidate = match.group(1).strip() if match else text.strip()
    number = _GSM8K_NUMBER.search(candidate)
    if not number:
        return candidate
    return number.group(0).replace(",", "").strip()


def _as_float(value: str | None) -> float | None:
    try:
        return float(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def numeric_answer_match(predicted: str | None, gold: str | None) -> bool:
    """Numeric equality when both sides parse, else normalized strings."""
    if predicted is None or gold is None:
        return False
    pred_value = _as_float(parse_gsm8k_answer(predicted))
    gold_value = _as_float(parse_gsm8k_answer(gold))
    if pred_value is not None and gold_value is not None:
        return math.isclose(pred_value, gold_value, rel_tol=1e-6, abs_tol=1e-9)
    return parse_gsm8k_answer(predicted).casefold() == parse_gsm8k_answer(gold).casefold()


def format_is_valid(text: str | None) -> bool:
    """A GSM8K output is format-valid when ``####`` carries a number."""
    match = _GSM8K_ANSWER_LINE.search("" if text is None else str(text))
    if not match:
        return False
    return bool(_GSM8K_NUMBER.search(match.group(1)))


def _word_tokens(text: str | None) -> list[str]:
    return _WORD_TOKEN.findall(("" if text is None else str(text)).lower())


def distinct_n(text: str | None, n: int = 3) -> float:
    """Word-level distinct-n; CJK characters count as individual tokens."""
    tokens = _word_tokens(text)
    if len(tokens) < n:
        return 0.0
    grams = [tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1)]
    return len(set(grams)) / len(grams)


def _char_ngrams(text: str | None, n: int) -> list[str]:
    cleaned = re.sub(r"\s+", " ", ("" if text is None else str(text)).strip().lower())
    if len(cleaned) < n:
        return []
    return [cleaned[index : index + n] for index in range(len(cleaned) - n + 1)]


def repetition_rate(text: str | None, n: int = 4) -> float:
    """Fraction of repeated character n-grams (1 - distinct-n)."""
    grams = _char_ngrams(text, n)
    if not grams:
        return 0.0
    return 1.0 - len(set(grams)) / len(grams)


def summarize_generations(
    records: Sequence[Mapping[str, Any]], *, max_new_tokens: int | None = None
) -> dict[str, Any]:
    """Aggregate per-prompt generations into the exp14-style gate metrics."""
    total = len(records)
    if total == 0:
        return {
            "n": 0,
            "accuracy": 0.0,
            "format_rate": 0.0,
            "truncation_rate": 0.0,
            "empty_rate": 0.0,
            "mean_distinct3": 0.0,
            "mean_repetition4": 0.0,
            "mean_new_tokens": 0.0,
            "n_errors": 0,
            "max_new_tokens": max_new_tokens,
        }

    correct = 0
    formatted = 0
    truncated = 0
    empty = 0
    errors = 0
    distinct3_sum = 0.0
    repetition_sum = 0.0
    new_tokens_sum = 0
    for record in records:
        output = str(record.get("output") or "")
        answer = record.get("answer")
        correct += int(numeric_answer_match(output, answer))
        formatted += int(format_is_valid(output))
        truncated += int(bool(record.get("truncated", False)))
        empty += int(bool(record.get("empty", not output.strip())))
        errors += int(record.get("error") is not None)
        distinct3_sum += distinct_n(output)
        repetition_sum += repetition_rate(output)
        new_tokens_sum += int(record.get("n_new_tokens") or 0)

    return {
        "n": total,
        "accuracy": correct / total,
        "format_rate": formatted / total,
        "truncation_rate": truncated / total,
        "empty_rate": empty / total,
        "mean_distinct3": distinct3_sum / total,
        "mean_repetition4": repetition_sum / total,
        "mean_new_tokens": new_tokens_sum / total,
        "n_errors": errors,
        "max_new_tokens": max_new_tokens,
    }


def _generation_record(
    index: int, example: Mapping[str, Any], generated: Mapping[str, Any]
) -> dict[str, Any]:
    output = str(generated.get("text") or "")
    answer = example.get("answer", example.get("gold"))
    return {
        "index": index,
        "prompt": str(example.get("prompt") or ""),
        "answer": None if answer is None else str(answer),
        "output": output,
        "correct": numeric_answer_match(output, answer),
        "format_valid": format_is_valid(output),
        "truncated": bool(generated.get("truncated", False)),
        "empty": not output.strip(),
        "n_new_tokens": int(generated.get("n_new_tokens") or 0),
        "distinct3": distinct_n(output),
        "repetition4": repetition_rate(output),
        "error": generated.get("error"),
    }


def generate_shadow_only(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    *,
    max_new_tokens: int = 256,
    batch_size: int = 4,
    device: str = "cpu",
    use_cache: bool = True,
) -> list[dict[str, Any]]:
    """Greedy decode already chat-formatted prompts with an exported shadow."""
    torch = _lazy_torch()
    if not prompts:
        return []
    tokenizer.padding_side = "left"
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(tokenizer, "eos_token_id", None)
    if pad_id is None:
        pad_id = 0
    eos_id = getattr(tokenizer, "eos_token_id", None)

    results: list[dict[str, Any]] = []
    for start in range(0, len(prompts), max(1, batch_size)):
        batch_prompts = list(prompts[start : start + batch_size])
        encoded = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )
        inputs = {
            key: value.to(device)
            for key, value in encoded.items()
            if key in ("input_ids", "attention_mask")
        }
        input_length = int(inputs["input_ids"].shape[1])

        def run(cache: bool, batch_inputs: Mapping[str, Any] = inputs) -> Any:
            with torch.inference_mode():
                return model.generate(
                    **batch_inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    use_cache=cache,
                    pad_token_id=pad_id,
                    eos_token_id=eos_id,
                )

        try:
            output = run(use_cache)
        except Exception as exc:
            message = str(exc).lower()
            if not use_cache or (
                "use_cache" not in message and "past_key_values" not in message
            ):
                raise
            output = run(False)

        for row_index, prompt in enumerate(batch_prompts):
            raw_ids = [int(token) for token in output[row_index, input_length:].tolist()]
            if eos_id is not None and eos_id in raw_ids:
                new_ids = raw_ids[: raw_ids.index(eos_id) + 1]
                hit_eos = True
            else:
                new_ids = raw_ids
                hit_eos = False
            results.append(
                {
                    "prompt": prompt,
                    "text": tokenizer.decode(new_ids, skip_special_tokens=True),
                    "n_new_tokens": len(new_ids),
                    "hit_eos": hit_eos,
                    "truncated": (not hit_eos) and len(new_ids) >= max_new_tokens,
                }
            )
    return results


def evaluate_shadow_only(
    model: Any,
    tokenizer: Any,
    examples: Sequence[Mapping[str, Any]],
    *,
    max_new_tokens: int = 512,
    batch_size: int = 4,
    device: str = "cpu",
    generate_fn: Callable[..., list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Generate and grade GSM8K-style examples.

    ``examples`` are ``{"prompt": <chat-formatted text>, "answer": <gold>}``
    records.  ``generate_fn`` is the decoding seam: it defaults to
    :func:`generate_shadow_only` and lets tests grade canned generations
    without loading a model.
    """
    materialized = list(examples)
    if not materialized:
        return {"summary": summarize_generations([], max_new_tokens=max_new_tokens), "generations": []}
    for index, example in enumerate(materialized):
        if not isinstance(example, Mapping) or "prompt" not in example:
            raise ValueError(f"evaluation example {index} 需要 prompt 字段")
    prompts = [str(example["prompt"]) for example in materialized]
    generator = generate_fn or generate_shadow_only
    generated = generator(
        model,
        tokenizer,
        prompts,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
        device=device,
    )
    if len(generated) != len(materialized):
        raise ValueError(
            f"生成条数不匹配：{len(generated)} 条输出对 {len(materialized)} 条 prompt"
        )
    records = [
        _generation_record(index, example, output)
        for index, (example, output) in enumerate(
            zip(materialized, generated, strict=True)
        )
    ]
    return {
        "summary": summarize_generations(records, max_new_tokens=max_new_tokens),
        "generations": records,
    }


# ---------------------------------------------------------------------------
# Production QA grading (pure, RESTRAIN-aware)
# ---------------------------------------------------------------------------


def keyword_coverage(text: str | None, keywords: Sequence[str]) -> float:
    """Fraction of expected keywords present in ``text`` (case-insensitive)."""
    if not keywords:
        return 0.0
    lowered = ("" if text is None else str(text)).lower()
    hits = sum(1 for keyword in keywords if str(keyword).lower() in lowered)
    return hits / len(keywords)


def _calc_answers_match(
    predicted: str | tuple[float, str] | None,
    gold: str | tuple[float, str] | None,
) -> bool:
    """Compare two ``extract_answer(..., "process_calc")`` results."""
    if not isinstance(predicted, tuple) or not isinstance(gold, tuple):
        return False
    predicted_value, predicted_unit = predicted
    gold_value, gold_unit = gold
    if gold_unit and predicted_unit != gold_unit:
        return False
    return math.isclose(predicted_value, gold_value, rel_tol=1e-6, abs_tol=1e-9)


def grade_qa_output(
    row: Mapping[str, Any], output: str | None
) -> dict[str, Any]:
    """Grade one production-QA generation against its fixture row.

    ``quality_verdict`` / ``process_calc`` rows are graded through
    :func:`shadow_lab.restrain.extract_answer`; any other task is graded by
    keyword coverage and passes at ``QA_KEYWORD_PASS_COVERAGE`` (0.5), matching
    the exp14 Stage 0 fixture convention.  ``format_valid`` means the output
    was parseable by its task's extractor (or non-empty for keyword rows).
    """
    task = str(row.get("task") or "")
    expected = row.get("expected")
    keywords = [str(keyword) for keyword in (row.get("expected_keywords") or ())]
    text = "" if output is None else str(output)
    coverage: float | None = None

    if task in _QA_STRUCTURED_TASKS and expected is not None:
        from shadow_lab.restrain import extract_answer

        predicted = extract_answer(text, task)
        gold = extract_answer(str(expected), task)
        if task == "quality_verdict":
            correct = predicted is not None and predicted == gold
        else:
            correct = _calc_answers_match(predicted, gold)
        format_valid = predicted is not None
    elif keywords:
        coverage = keyword_coverage(text, keywords)
        correct = bool(text.strip()) and coverage >= QA_KEYWORD_PASS_COVERAGE
        format_valid = bool(text.strip())
    else:
        correct = False
        format_valid = bool(text.strip())

    return {
        "task": task,
        "correct": bool(correct),
        "format_valid": bool(format_valid),
        "keyword_coverage": coverage,
        "empty": not text.strip(),
    }


def summarize_qa(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate graded production-QA records into one diagnostic summary.

    ``keyword_coverage`` / ``keyword_full_rate`` cover only rows that carry
    ``expected_keywords`` and are ``None`` when the fixture has none.
    """
    total = len(records)
    if total == 0:
        return {
            "n": 0,
            "accuracy": 0.0,
            "format_rate": 0.0,
            "keyword_coverage": None,
            "keyword_full_rate": None,
            "n_keyword_rows": 0,
            "by_task": {},
        }

    correct = sum(1 for record in records if record.get("correct"))
    formatted = sum(1 for record in records if record.get("format_valid"))
    coverages = [
        float(record["keyword_coverage"])
        for record in records
        if record.get("keyword_coverage") is not None
    ]

    by_task: dict[str, dict[str, Any]] = {}
    for record in records:
        task = str(record.get("task") or "")
        bucket = by_task.setdefault(task, {"n": 0, "correct": 0, "formatted": 0})
        bucket["n"] += 1
        bucket["correct"] += int(bool(record.get("correct")))
        bucket["formatted"] += int(bool(record.get("format_valid")))
    for bucket in by_task.values():
        bucket["accuracy"] = bucket["correct"] / bucket["n"]
        bucket["format_rate"] = bucket["formatted"] / bucket["n"]

    return {
        "n": total,
        "accuracy": correct / total,
        "format_rate": formatted / total,
        "keyword_coverage": (
            sum(coverages) / len(coverages) if coverages else None
        ),
        "keyword_full_rate": (
            sum(1 for value in coverages if value >= 1.0) / len(coverages)
            if coverages
            else None
        ),
        "n_keyword_rows": len(coverages),
        "by_task": by_task,
    }


def evaluate_qa_only(
    model: Any,
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    *,
    max_new_tokens: int = 256,
    batch_size: int = 4,
    device: str = "cpu",
    generate_fn: Callable[..., list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Generate and grade the production-QA fixture with the shadow.

    This is a diagnostic, not a recovery gate input.  ``generate_fn`` is the
    decoding seam (defaults to :func:`generate_shadow_only`).
    """
    materialized = list(rows)
    if not materialized:
        return {"summary": summarize_qa([]), "generations": []}

    prompts: list[str] = []
    for index, row in enumerate(materialized):
        if not isinstance(row, Mapping) or "prompt" not in row:
            raise ValueError(f"QA row {index} 需要 prompt 字段")
        prompts.append(
            format_chat_prompt(
                tokenizer, [{"role": "user", "content": str(row["prompt"])}]
            )
        )

    generator = generate_fn or generate_shadow_only
    generated = generator(
        model,
        tokenizer,
        prompts,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
        device=device,
    )
    if len(generated) != len(materialized):
        raise ValueError(
            f"QA 生成条数不匹配：{len(generated)} 条输出对 {len(materialized)} 条 prompt"
        )

    records: list[dict[str, Any]] = []
    for index, (row, output) in enumerate(
        zip(materialized, generated, strict=True)
    ):
        text = str(output.get("text") or "")
        record = grade_qa_output(row, text)
        record.update(
            {
                "index": index,
                "id": str(row.get("id") or f"qa-{index}"),
                "prompt": str(row.get("prompt") or ""),
                "output": text,
                "truncated": bool(output.get("truncated", False)),
                "n_new_tokens": int(output.get("n_new_tokens") or 0),
                "error": output.get("error"),
            }
        )
        records.append(record)
    return {"summary": summarize_qa(records), "generations": records}


@dataclass
class ShadowOnlyTrainStats:
    """Outcome of a shadow-only SFT run."""

    steps: int
    epochs: int | None
    rows: int
    tokens: int
    loss_mean: float
    loss_first: float
    loss_last: float
    losses: list[float]
    trainable_parameters: int
    total_parameters: int
    train_scope: str
    seconds: float
    lr: float
    batch_size: int
    max_seq_len: int
    seed: int
    device: str
    dtype: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _seed_everything(torch: Any, seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _collate_sft(
    rows: Sequence[Mapping[str, Any]], pad_id: int, device: Any, torch: Any
) -> dict[str, Any]:
    max_len = max(len(row["input_ids"]) for row in rows)
    input_ids = torch.full((len(rows), max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(rows), max_len), dtype=torch.long)
    labels = torch.full((len(rows), max_len), IGNORE_INDEX, dtype=torch.long)
    for index, row in enumerate(rows):
        ids = [int(token) for token in row["input_ids"]]
        length = len(ids)
        input_ids[index, :length] = torch.tensor(ids, dtype=torch.long)
        attention_mask[index, :length] = 1
        labels[index, :length] = torch.tensor(
            [int(label) for label in row["labels"]], dtype=torch.long
        )
    return {
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device),
        "labels": labels.to(device),
    }


def _forward_loss(outputs: Any) -> Any:
    loss = getattr(outputs, "loss", None)
    if loss is None and isinstance(outputs, tuple) and outputs:
        loss = outputs[0]
    return loss


def _resolved_steps(row_count: int, config: ShadowOnlyTrainConfig) -> int:
    """Steps for the SFT loop: ``epochs`` wins, otherwise ``config.steps``."""
    if config.epochs is None:
        return config.steps
    return math.ceil(row_count / config.batch_size) * config.epochs


def train_shadow_only_sft(
    model: Any,
    tokenizer: Any,
    rows: Iterable[Mapping[str, Any]],
    config: ShadowOnlyTrainConfig,
) -> ShadowOnlyTrainStats:
    """Run a small deterministic SFT loop over chat-template rows.

    The exported model's ``forward(labels=...)`` already computes the shifted
    cross-entropy, so the loop only handles batching, AdamW, gradient clipping
    and seeding.  ``config.train_scope`` decides whether the shadow backbone,
    the projection/head, everything, or nothing receives gradients.  The model
    is left in ``eval()`` mode for the subsequent generation pass.
    """
    torch = _lazy_torch()
    materialized = list(rows)
    if not materialized:
        raise ValueError("SFT rows 不能为空")
    if tokenizer is None:
        raise ValueError("train_shadow_only_sft 需要 tokenizer")

    _seed_everything(torch, config.seed)
    scope_report = set_trainable_scope(model, config.train_scope)
    if scope_report.trainable_parameters == 0:
        raise ValueError(
            f"scope={config.train_scope!r} 没有启用任何参数；改用 scope='all' 或检查模型结构"
        )

    dataset = build_sft_rows(materialized, tokenizer, config.max_seq_len)
    steps = _resolved_steps(len(dataset), config)
    if config.warmup_steps > steps:
        raise ValueError(
            f"warmup_steps={config.warmup_steps} 超过实际训练步数 {steps}"
        )
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters, lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = None
    if config.warmup_steps > 0:

        def warmup(step: int) -> float:
            return min(1.0, float(step + 1) / float(config.warmup_steps))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup)

    device = torch.device(config.device)
    model.to(device)
    model.train()

    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "eos_token_id", None)
    if pad_token_id is None:
        raise ValueError("tokenizer 同时缺少 pad_token_id 和 eos_token_id")

    order = list(range(len(dataset)))
    shuffle_rng = random.Random(config.seed)
    shuffle_rng.shuffle(order)
    cursor = 0

    losses: list[float] = []
    tokens = 0
    started = time.perf_counter()
    for step_index in range(steps):
        batch_rows = []
        while len(batch_rows) < config.batch_size:
            if cursor >= len(order):
                shuffle_rng.shuffle(order)
                cursor = 0
            batch_rows.append(dataset[order[cursor]])
            cursor += 1

        batch = _collate_sft(batch_rows, int(pad_token_id), device, torch)
        outputs = model(**batch, use_cache=False)
        loss = _forward_loss(outputs)
        if loss is None:
            raise RuntimeError(
                "模型 forward 没有返回 loss；导出模型必须支持 labels 交叉熵"
            )
        if not bool(torch.isfinite(loss).all()):
            raise RuntimeError(f"训练出现非有限 loss: {float(loss.detach().cpu())}")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(parameters, config.grad_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        loss_value = float(loss.detach().cpu())
        losses.append(loss_value)
        tokens += int((batch["labels"] != IGNORE_INDEX).sum().item())
        step_number = step_index + 1
        if (
            step_number == 1
            or step_number % config.log_every == 0
            or step_number == steps
        ):
            print(
                f"[shadow-only] step {step_number}/{steps} loss={loss_value:.4f}",
                flush=True,
            )

    seconds = time.perf_counter() - started
    model.eval()
    return ShadowOnlyTrainStats(
        steps=steps,
        epochs=config.epochs,
        rows=len(dataset),
        tokens=tokens,
        loss_mean=sum(losses) / len(losses),
        loss_first=losses[0],
        loss_last=losses[-1],
        losses=losses,
        trainable_parameters=scope_report.trainable_parameters,
        total_parameters=scope_report.total_parameters,
        train_scope=config.train_scope,
        seconds=seconds,
        lr=config.lr,
        batch_size=config.batch_size,
        max_seq_len=config.max_seq_len,
        seed=config.seed,
        device=config.device,
        dtype=config.dtype,
    )


__all__ = [
    "IGNORE_INDEX",
    "QA_KEYWORD_PASS_COVERAGE",
    "TRAIN_SCOPES",
    "ShadowOnlyTrainConfig",
    "ShadowOnlyTrainStats",
    "TrainableScopeReport",
    "build_sft_rows",
    "count_parameters",
    "distinct_n",
    "evaluate_qa_only",
    "evaluate_shadow_only",
    "export_roundtrip_check",
    "format_chat_prompt",
    "format_is_valid",
    "generate_shadow_only",
    "grade_qa_output",
    "keyword_coverage",
    "load_exported_shadow",
    "load_shadow_tokenizer",
    "normalize_token_ids",
    "numeric_answer_match",
    "parse_gsm8k_answer",
    "repetition_rate",
    "resolve_dtype",
    "set_trainable_scope",
    "summarize_generations",
    "summarize_qa",
    "train_shadow_only_sft",
]
