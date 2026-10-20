"""S2T-Local diagnostic runner: greedy base generation with shadow top-K capture.

The diagnostic answers one question before selector training: do the shadow
model's top-K candidates cover the base model's greedy next token?  It runs
``ShadowForCausalLM`` in ``base_shadow`` mode, advances generation with the
base top-1 token, and records both distributions at every generated position.

**Shadow hidden caveat:** ``shadow_logits`` come from the *initial* shadow
hidden state produced inside ``forward_with_shadow`` (before shadow injection
and update). They are not the injected ``base_shadow`` deployment path. Every
per-token record carries ``shadow_hidden="initial"`` and the assembled report
carries the same field plus the core module's caveat note.

The module imports torch lazily, so importing it (or running the CLI
``--help``) never loads the model stack.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from shadow_lab.s2t import build_report, to_csv, to_json

SHADOW_HIDDEN = "initial"
TOKEN_SCHEMA_VERSION = "s2t-local-token/v1"
REPORT_FILENAME = "s2t_diagnostic.json"
TOKENS_FILENAME = "s2t_tokens.jsonl"
CSV_FILENAME = "s2t_diagnostic.csv"
SHADOW_HIDDEN_NOTE = (
    "shadow_logits were captured from the initial shadow hidden state; the "
    "injected base_shadow deployment path is not measured by this diagnostic."
)


@dataclass(frozen=True, slots=True)
class PromptSpec:
    """One diagnostic prompt with its stable id and task label."""

    prompt_id: str
    text: str
    task: str


@dataclass(frozen=True, slots=True)
class DiagnosticResult:
    """Report, per-token records, skipped prompt ids and written paths."""

    report: dict[str, Any]
    records: tuple[dict[str, Any], ...]
    skipped_prompts: tuple[str, ...]
    paths: dict[str, str] = field(default_factory=dict)


def _require_positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def _normalize_prompts(
    prompts: Sequence[str | Mapping[str, Any]],
    *,
    task_key: str,
) -> list[PromptSpec]:
    if isinstance(prompts, (str, bytes)) or not isinstance(prompts, Sequence):
        raise TypeError("prompts must be a sequence of strings or mappings")
    specs: list[PromptSpec] = []
    for index, item in enumerate(prompts):
        if isinstance(item, str):
            text = item
            task = "unspecified"
            prompt_id = f"p{index}"
        elif isinstance(item, Mapping):
            raw_text = item.get("prompt")
            if not isinstance(raw_text, str):
                raise ValueError(f"prompt[{index}] needs a string 'prompt' field")
            text = raw_text
            raw_task = item.get(task_key)
            task = "unspecified" if raw_task is None or raw_task == "" else str(raw_task)
            raw_id = item.get("prompt_id")
            prompt_id = f"p{index}" if raw_id is None else str(raw_id)
        else:
            raise TypeError(
                f"prompt[{index}] must be a string or mapping, got {type(item).__name__}"
            )
        specs.append(PromptSpec(prompt_id=prompt_id, text=text, task=task))
    return specs


def _device_of(model: Any) -> Any:
    device = getattr(model, "device", None)
    if device is not None:
        return device
    return next(model.parameters()).device


def _rank_in_shadow(shadow_logits: Any, base_top1: int, torch_module: Any) -> int:
    """1-based rank with the core's lower-token-id tie-break, without a sort."""

    target = shadow_logits[base_top1]
    better = int((shadow_logits > target).sum().item())
    tied_lower = int(
        (
            (shadow_logits == target)
            & (
                torch_module.arange(shadow_logits.shape[-1], device=shadow_logits.device)
                < base_top1
            )
        )
        .sum()
        .item()
    )
    return 1 + better + tied_lower


def _record_step(
    *,
    prompt: PromptSpec,
    position: int,
    base_logits: Any,
    shadow_logits: Any,
    k: int,
    torch_module: Any,
) -> dict[str, Any]:
    base_logits = base_logits.float()
    shadow_logits = shadow_logits.float()
    base_top1 = int(torch_module.argmax(base_logits).item())
    effective_k = min(k, int(shadow_logits.shape[-1]))
    shadow_logprobs = torch_module.log_softmax(shadow_logits, dim=-1)
    shadow_top = torch_module.topk(shadow_logprobs, k=effective_k)
    base_top = torch_module.topk(base_logits, k=effective_k)
    return {
        "schema_version": TOKEN_SCHEMA_VERSION,
        "prompt_id": prompt.prompt_id,
        "task": prompt.task,
        "position": position,
        "base_top1": base_top1,
        "base_topk": [int(token) for token in base_top.indices.tolist()],
        "shadow_topk_ids": [int(token) for token in shadow_top.indices.tolist()],
        "shadow_topk_logprobs": [float(value) for value in shadow_top.values.tolist()],
        "base_top1_rank_in_shadow": _rank_in_shadow(shadow_logits, base_top1, torch_module),
        "shadow_hidden": SHADOW_HIDDEN,
    }


def _iter_prompt_records(
    model: Any,
    tokenizer: Any,
    prompt: PromptSpec,
    *,
    max_new_tokens: int,
    k: int,
    use_cache: bool,
    torch_module: Any,
) -> list[dict[str, Any]]:
    if not prompt.text.strip():
        return []
    batch = tokenizer(prompt.text, return_tensors="pt")
    input_ids = batch["input_ids"]
    attention_mask = batch.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch_module.ones_like(input_ids)
    if int(input_ids.shape[-1]) == 0:
        return []
    device = _device_of(model)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    past_key_values = None
    records: list[dict[str, Any]] = []

    for position in range(max_new_tokens):
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=use_cache,
            past_key_values=past_key_values,
        )
        shadow_logits = getattr(outputs, "shadow_logits", None)
        if shadow_logits is None:
            raise RuntimeError("model output is missing shadow_logits; use base_shadow mode")
        record = _record_step(
            prompt=prompt,
            position=position,
            base_logits=outputs.logits[0, -1],
            shadow_logits=shadow_logits[0, -1],
            k=k,
            torch_module=torch_module,
        )
        records.append(record)
        next_token = record["base_top1"]
        if eos_token_id is not None and next_token == eos_token_id:
            break
        if position + 1 >= max_new_tokens:
            break
        if use_cache:
            past_key_values = outputs.past_key_values
            if past_key_values is None:
                raise RuntimeError("use_cache=True but the model returned no past_key_values")
            input_ids = torch_module.tensor([[next_token]], dtype=torch_module.long, device=device)
        else:
            input_ids = torch_module.cat(
                [
                    input_ids,
                    torch_module.tensor([[next_token]], dtype=input_ids.dtype, device=device),
                ],
                dim=1,
            )
        attention_mask = torch_module.cat(
            [attention_mask, torch_module.ones((1, 1), dtype=attention_mask.dtype, device=device)],
            dim=1,
        )
    return records


def diagnose(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str | Mapping[str, Any]],
    *,
    max_new_tokens: int,
    k: int = 8,
    seed: int = 0,
    task_key: str = "task",
    use_cache: bool = False,
) -> list[dict[str, Any]]:
    """Record base top-1 vs. shadow top-K for every greedy generated token.

    ``model`` must be a ``ShadowForCausalLM`` in ``base_shadow`` mode so a
    single forward returns both ``logits`` (base) and ``shadow_logits``.
    Generation uses the base top-1 as the next input token; the shadow head
    only supplies candidates. Prompts are strings or mappings with ``prompt``
    plus an optional task under ``task_key`` (default ``"task"``).

    ``shadow_logits`` are read from the **initial** shadow hidden state, so
    every record reports ``shadow_hidden="initial"``. This is a diagnostic of
    candidate coverage, not of the injected deployment path.

    Greedy decoding is deterministic; ``seed`` still seeds torch (and CUDA)
    for auditability. Empty/blank prompts are skipped. Generation stops after
    ``max_new_tokens`` or when the base emits ``tokenizer.eos_token_id`` (the
    EOS token is recorded). The caller's train/eval state is restored.
    """

    max_new_tokens = _require_positive_int(max_new_tokens, name="max_new_tokens")
    k = _require_positive_int(k, name="k")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError(f"seed must be an integer, got {seed!r}")
    mode = getattr(model, "inference_mode", None)
    if mode not in (None, "base_shadow"):
        raise ValueError(f"diagnose needs a base_shadow model, got inference_mode={mode!r}")
    specs = _normalize_prompts(prompts, task_key=task_key)

    torch_module = _import_torch()
    torch_module.manual_seed(seed)
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed_all(seed)
    was_training = bool(getattr(model, "training", False))
    model.eval()
    records: list[dict[str, Any]] = []
    try:
        with torch_module.inference_mode():
            for spec in specs:
                records.extend(
                    _iter_prompt_records(
                        model,
                        tokenizer,
                        spec,
                        max_new_tokens=max_new_tokens,
                        k=k,
                        use_cache=use_cache,
                        torch_module=torch_module,
                    )
                )
    finally:
        model.train(was_training)
    return records


def _import_torch() -> Any:
    """Import torch lazily so module import (and CLI --help) stays light."""

    import torch

    return torch


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def run_diagnostic(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str | Mapping[str, Any]],
    *,
    out: str | Path | None = None,
    max_new_tokens: int,
    k: int = 8,
    seed: int = 0,
    task_key: str = "task",
    use_cache: bool = False,
    n_boot: int = 1000,
    metadata: Mapping[str, Any] | None = None,
) -> DiagnosticResult:
    """Run :func:`diagnose`, build the go/no-go report and export JSON/JSONL/CSV.

    ``out`` is a directory (created when missing). It receives
    ``s2t_diagnostic.json``, ``s2t_tokens.jsonl`` and ``s2t_diagnostic.csv``.
    Pass ``out=None`` to skip writing. The returned report carries
    ``shadow_hidden="initial"`` and the sample-size, task and CI provenance.
    """

    records = diagnose(
        model,
        tokenizer,
        prompts,
        max_new_tokens=max_new_tokens,
        k=k,
        seed=seed,
        task_key=task_key,
        use_cache=use_cache,
    )
    if not records:
        raise ValueError("diagnostic produced no tokens; check the prompts and max_new_tokens")

    report = build_report(records, ks=(k,), n_boot=n_boot, seed=seed)
    report["shadow_hidden"] = SHADOW_HIDDEN
    report["shadow_hidden_note"] = SHADOW_HIDDEN_NOTE
    report["candidate_k"] = k
    report["max_new_tokens"] = max_new_tokens
    report["use_cache"] = use_cache
    report["skipped_prompts"] = list(_skipped_prompt_ids(prompts, task_key=task_key))
    report["n_skipped_prompts"] = len(report["skipped_prompts"])
    notes = list(report.get("notes") or [])
    if k < 16:
        notes.append(
            f"Shadow candidates were captured at K={k}; hit_at_16 is capped by this "
            "capture width and equals the Hit@K rate."
        )
    notes.append(
        "Rank percentiles use the captured shadow top-K list, so they are known only "
        "for hits; s2t_tokens.jsonl keeps the exact base_top1_rank_in_shadow per token."
    )
    notes.append(SHADOW_HIDDEN_NOTE)
    report["notes"] = notes
    if metadata:
        report["metadata"] = {str(key): value for key, value in metadata.items()}

    paths: dict[str, str] = {}
    if out is not None:
        out_dir = Path(out).expanduser()
        paths = {
            "json": str(out_dir / REPORT_FILENAME),
            "jsonl": str(out_dir / TOKENS_FILENAME),
            "csv": str(out_dir / CSV_FILENAME),
        }
        _write_text(out_dir / REPORT_FILENAME, to_json(report) + "\n")
        lines = [
            json.dumps(record, ensure_ascii=False, allow_nan=False, sort_keys=True)
            for record in records
        ]
        _write_text(out_dir / TOKENS_FILENAME, "\n".join(lines) + "\n")
        _write_text(out_dir / CSV_FILENAME, to_csv(report))

    return DiagnosticResult(
        report=report,
        records=tuple(records),
        skipped_prompts=tuple(report["skipped_prompts"]),
        paths=paths,
    )


def _skipped_prompt_ids(prompts: Sequence[str | Mapping[str, Any]], *, task_key: str) -> list[str]:
    return [
        spec.prompt_id
        for spec in _normalize_prompts(prompts, task_key=task_key)
        if not spec.text.strip()
    ]


__all__ = [
    "CSV_FILENAME",
    "DiagnosticResult",
    "PromptSpec",
    "REPORT_FILENAME",
    "SHADOW_HIDDEN",
    "SHADOW_HIDDEN_NOTE",
    "TOKENS_FILENAME",
    "TOKEN_SCHEMA_VERSION",
    "diagnose",
    "run_diagnostic",
]
