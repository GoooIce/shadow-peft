"""Real-corpus evaluation suites for issues #2-#5 (issue #47 MVP).

This module is the reusable harness behind ``scripts/exp17_real_eval.py``. It
turns real examples into four uniform suite reports:

- ``gkd``: joint/shadow perplexity on held-out text plus structured task
  accuracy, reusing :func:`shadow_lab.onpolicy_eval.evaluate_ppl` and
  :func:`shadow_lab.onpolicy_eval.grade_task`.
- ``s2t``: candidate coverage diagnostics (Hit@8 / Agree@1), reusing
  :func:`shadow_lab.s2t_diagnose.diagnose` and
  :func:`shadow_lab.s2t.build_report`.
- ``restrain``: structured rollout grading and vote consistency, reusing
  ``shadow_lab.restrain`` and :mod:`shadow_lab.restrain_eval`.
- ``jitrl``: no-memory vs memory-reranked accuracy, false-bias and cloud calls
  over an n-best backend, driven by :func:`edge_cloud.rerank.rerank`.

Design rules
------------
- **No fabricated results.** A missing model, memory, n-best backend or usable
  example yields ``status="data_missing"`` with a note; metrics that cannot be
  measured stay ``None``.
- **Small samples cannot claim gains.** ``claims_gain`` is always ``False`` in
  this harness (it has no baseline arm), and fewer than
  :data:`EVIDENCE_MIN_EXAMPLES` examples leave the evidence label at
  :data:`EVIDENCE_INSUFFICIENT`.
- **No models or datasets at import.** The module reuses the existing
  torch-backed eval helpers but never loads weights; ``datasets`` is imported
  inside :func:`load_gsm8k_examples` (and only there).

The CLI's smoke path uses GSM8K test for GKD/S2T and
``runs/exp12/heldout.jsonl`` for RESTRAIN; JitRL needs a real feedback memory
JSONL and reports ``data_missing`` when it is absent.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from edge_cloud.backends import Generation
from edge_cloud.memory import ExperienceMemory
from edge_cloud.protocol import CloudRequest, InferRequest
from edge_cloud.rerank import Candidate, rerank
from shadow_lab.onpolicy_eval import bootstrap_ci, evaluate_ppl, grade_task
from shadow_lab.restrain import cluster_answers, consistency
from shadow_lab.restrain_eval import (
    answer_entropy,
    grade_structured,
    low_consistency_rate,
    top1_share,
)
from shadow_lab.restrain_eval import (
    build_report as build_restrain_report,
)
from shadow_lab.s2t import build_report as build_s2t_report
from shadow_lab.s2t_diagnose import diagnose

SCHEMA_VERSION = "real-eval/v1"
REPORT_FILENAME = "report.json"
SUMMARY_FILENAME = "summary.csv"
PER_EXAMPLE_FILENAME = "per_example.jsonl"
SUITES: tuple[str, ...] = ("gkd", "s2t", "restrain", "jitrl")

STATUS_OK = "ok"
STATUS_DATA_MISSING = "data_missing"
EVIDENCE_INSUFFICIENT = "evidence_insufficient"
EVIDENCE_COLLECTED = "evidence_collected"
# Mirrors the RESTRAIN ticket's >=100 structured hard-case bar. Below this a
# suite may describe mechanism behaviour but never a gain.
EVIDENCE_MIN_EXAMPLES = 100
DEFAULT_CI_BOOTSTRAP = 200
CLAIM_NOTE = (
    "claims_gain is false: this MVP harness evaluates a single model/path and "
    "does not run a baseline-vs-treatment campaign."
)


# ---------------------------------------------------------------------------
# Example schema and loaders
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EvalExample:
    """One evaluation item shared by every suite.

    ``expected`` is task-specific: a verdict label, a number, a GSM8K final
    answer or ``None`` for open-ended prompts. ``metadata`` keeps provenance
    (dataset split, reference solution, row index) without affecting grading.
    """

    id: str
    prompt: str
    task: str | None = None
    expected: Any = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SuiteReport:
    """Uniform result of one evaluation suite."""

    suite: str
    status: str
    metrics: Mapping[str, Any]
    per_example: tuple[Mapping[str, Any], ...] = ()
    config: Mapping[str, Any] = field(default_factory=dict)
    claims_gain: bool = False
    evidence: str = EVIDENCE_INSUFFICIENT
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "suite": self.suite,
            "status": self.status,
            "metrics": dict(self.metrics),
            "per_example": [dict(row) for row in self.per_example],
            "config": dict(self.config),
            "claims_gain": bool(self.claims_gain),
            "evidence": self.evidence,
            "notes": list(self.notes),
        }


def _evidence_for(count: int) -> str:
    return (
        EVIDENCE_COLLECTED
        if count >= EVIDENCE_MIN_EXAMPLES
        else EVIDENCE_INSUFFICIENT
    )


def _data_missing(
    suite: str,
    reason: str,
    *,
    config: Mapping[str, Any] | None = None,
    examples: int = 0,
) -> SuiteReport:
    return SuiteReport(
        suite=suite,
        status=STATUS_DATA_MISSING,
        metrics={
            "n_examples": int(examples),
            "accuracy": None,
            "joint_ppl": None,
            "shadow_ppl": None,
            "hit_at_8": None,
            "agree_at_1": None,
        },
        config=dict(config or {}),
        claims_gain=False,
        evidence=EVIDENCE_INSUFFICIENT,
        notes=(reason,),
    )


def load_jsonl_examples(path: str | Path) -> list[EvalExample]:
    """Read examples from JSONL.

    Required: a non-empty ``prompt``. Optional: ``id`` (or ``prompt_id``),
    ``task``, ``expected`` (or ``expected_answer``) and a mapping
    ``metadata``. Blank lines are skipped; malformed rows raise ``ValueError``
    with the line number, and duplicate ids are rejected so per-example rows
    never collide.
    """

    source = Path(path).expanduser()
    text = source.read_text(encoding="utf-8")
    examples: list[EvalExample] = []
    seen: set[str] = set()
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{source}: line {line_number}: invalid JSON: {exc}") from exc
        if not isinstance(row, Mapping):
            raise ValueError(f"{source}: line {line_number}: expected a JSON object")
        prompt = row.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"{source}: line {line_number}: missing non-empty 'prompt'")
        raw_id = row.get("id", row.get("prompt_id"))
        example_id = f"ex{len(examples) + 1}" if raw_id in (None, "") else str(raw_id)
        if example_id in seen:
            raise ValueError(f"{source}: line {line_number}: duplicate example id {example_id!r}")
        seen.add(example_id)
        metadata = row.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise ValueError(f"{source}: line {line_number}: 'metadata' must be an object")
        task = row.get("task")
        expected = row.get("expected", row.get("expected_answer"))
        examples.append(
            EvalExample(
                id=example_id,
                prompt=prompt,
                task=None if task is None else str(task),
                expected=expected,
                metadata=dict(metadata),
            )
        )
    return examples


def _final_answer(reference: str) -> str:
    """Extract the GSM8K ``####`` final answer as a grader-friendly string."""

    body = reference.rsplit("####", 1)[-1].strip()
    return body.replace(",", "").replace("$", "").strip()


def load_gsm8k_examples(
    limit: int | None = None,
    seed: int = 0,
    *,
    split: str = "test",
) -> list[EvalExample]:
    """Load real GSM8K rows through ``datasets`` with a seeded shuffle.

    Examples carry the question plus an ``Answer:`` cue, the numeric final
    answer as ``expected`` (for the process_calc numeric grader) and the full
    reference solution in ``metadata["reference"]`` (used for perplexity).
    ``limit`` applies after the shuffle, so ``(limit, seed)`` is reproducible.
    """

    if limit is not None and limit < 1:
        raise ValueError(f"limit must be >= 1 or None, got {limit!r}")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError(f"seed must be an integer, got {seed!r}")
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - optional experiment extra
        raise RuntimeError(
            "load_gsm8k_examples needs the 'datasets' package; install the "
            "experiment extra (pip install 'shadow-peft[experiment]')."
        ) from exc

    dataset = load_dataset("openai/gsm8k", "main", split=split)
    rows = list(dataset)
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    if limit is not None:
        order = order[:limit]

    examples: list[EvalExample] = []
    for row_index in order:
        row = rows[row_index]
        question = str(row.get("question") or "").strip()
        reference = str(row.get("answer") or "").strip()
        if not question or "####" not in reference:
            continue
        examples.append(
            EvalExample(
                id=f"gsm8k-{split}-{row_index}",
                prompt=f"{question}\nAnswer:",
                task="gsm8k",
                expected=_final_answer(reference),
                metadata={
                    "source": "gsm8k",
                    "split": split,
                    "row_index": row_index,
                    "reference": reference,
                    "question": question,
                },
            )
        )
    return examples


def dataset_fingerprint(examples: Iterable[EvalExample]) -> str:
    """Stable sha256 over the example set (order-insensitive, id-keyed)."""

    digest = hashlib.sha256()
    for example in sorted(examples, key=lambda item: item.id):
        payload = json.dumps(
            {
                "id": example.id,
                "prompt": example.prompt,
                "task": example.task,
                "expected": example.expected,
                "reference": example.metadata.get("reference"),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        digest.update(payload.encode("utf-8"))
        digest.update(b"\x00")
    return "sha256:" + digest.hexdigest()


# ---------------------------------------------------------------------------
# Shared generation and grading helpers
# ---------------------------------------------------------------------------


def _import_torch() -> Any:
    import torch

    return torch


def _device_of(model: Any) -> Any:
    device = getattr(model, "device", None)
    if device is not None:
        return device
    return next(model.parameters()).device


def _encode_prompt(tokenizer: Any, prompt: str, torch_module: Any) -> tuple[Any, Any]:
    encoded = tokenizer(prompt, return_tensors="pt")
    if isinstance(encoded, Mapping):
        input_ids = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask")
    else:  # pragma: no cover - tokenizers normally return a mapping here
        input_ids = encoded
        attention_mask = None
    input_ids = torch_module.as_tensor(input_ids, dtype=torch_module.long)
    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)
    if int(input_ids.shape[-1]) == 0:
        raise ValueError("prompt tokenized to zero tokens")
    if attention_mask is None:
        attention_mask = torch_module.ones_like(input_ids)
    else:
        attention_mask = torch_module.as_tensor(attention_mask, dtype=torch_module.long)
        if attention_mask.ndim == 1:
            attention_mask = attention_mask.unsqueeze(0)
    return input_ids, attention_mask


def _generate_response(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    max_new_tokens: int,
    temperature: float = 0.0,
    torch_module: Any | None = None,
) -> str:
    """Greedy/sampled generation on the joint path using the model seam.

    Uses ``model.generate`` when available (real ShadowForCausalLM) and falls
    back to a forward-only loop for test doubles. ``temperature > 0`` samples
    with the process torch RNG, so suites seed torch once per run.
    """

    if max_new_tokens < 1:
        raise ValueError(f"max_new_tokens must be >= 1, got {max_new_tokens!r}")
    torch = torch_module or _import_torch()
    device = _device_of(model)
    input_ids, attention_mask = _encode_prompt(tokenizer, prompt, torch)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    input_length = int(input_ids.shape[-1])
    previous_mode = getattr(model, "inference_mode", None)
    set_mode = getattr(model, "set_inference_mode", None)
    was_training = bool(getattr(model, "training", False))
    if previous_mode is not None and callable(set_mode):
        set_mode("base_shadow")
    if hasattr(model, "eval"):
        model.eval()
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "eos_token_id", None)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    new_tokens: list[int] = []
    try:
        generate = getattr(model, "generate", None)
        if callable(generate) and temperature <= 0.0:
            kwargs: dict[str, Any] = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "max_new_tokens": max_new_tokens,
                "do_sample": False,
                "use_cache": True,
            }
            if pad_token_id is not None:
                kwargs["pad_token_id"] = pad_token_id
            if eos_token_id is not None:
                kwargs["eos_token_id"] = eos_token_id
            with torch.no_grad():
                output = generate(**kwargs)
            new_tokens = [int(token) for token in output[0, input_length:].tolist()]
        else:
            current_ids = input_ids
            current_mask = attention_mask
            with torch.no_grad():
                for _ in range(max_new_tokens):
                    outputs = model(input_ids=current_ids, attention_mask=current_mask)
                    logits = outputs.logits[0, -1].float()
                    if temperature > 0.0:
                        probs = torch.softmax(logits / float(temperature), dim=-1)
                        next_token = int(torch.multinomial(probs, num_samples=1).item())
                    else:
                        next_token = int(torch.argmax(logits).item())
                    if eos_token_id is not None and next_token == eos_token_id:
                        break
                    new_tokens.append(next_token)
                    next_ids = torch.tensor(
                        [[next_token]], dtype=torch.long, device=device
                    )
                    current_ids = torch.cat([current_ids, next_ids], dim=1)
                    current_mask = torch.cat(
                        [
                            current_mask,
                            torch.ones((1, 1), dtype=current_mask.dtype, device=device),
                        ],
                        dim=1,
                    )
    finally:
        if previous_mode is not None and callable(set_mode):
            set_mode(previous_mode)
        if hasattr(model, "train"):
            model.train(was_training)
    decode = getattr(tokenizer, "decode", None)
    if not new_tokens or not callable(decode):
        return ""
    try:
        return decode(new_tokens, skip_special_tokens=True).strip()
    except TypeError:  # pragma: no cover - minimal tokenizers
        return decode(new_tokens).strip()


def _reference_text(example: EvalExample) -> str:
    reference = example.metadata.get("reference", example.expected)
    if reference is None:
        return example.prompt
    return f"{example.prompt}\n{reference}"


def _grade(example: EvalExample, prediction: str | None) -> bool | None:
    """Grade one prediction with the shared structured grader.

    GSM8K final answers use the ``process_calc`` numeric tolerance (1%); other
    tasks go through their own label. Open-ended tasks return ``None``.
    """

    if prediction is None:
        return None
    task = "process_calc" if example.task == "gsm8k" else example.task
    return grade_task(prediction, example.expected, task)


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _ppl_interval(
    losses: Sequence[float],
    window_tokens: Sequence[int] | None,
    *,
    n_boot: int,
    seed: int,
) -> list[float] | None:
    if len(losses) < 2:
        return None
    weights = None if window_tokens is None else [float(count) for count in window_tokens]
    lower, upper = bootstrap_ci(losses, n_boot=n_boot, seed=seed, weights=weights)
    return [math.exp(lower), math.exp(upper)]


def _apply_limit(examples: Sequence[EvalExample], limit: int | None) -> list[EvalExample]:
    if limit is None:
        return list(examples)
    if limit < 1:
        raise ValueError(f"limit must be >= 1 or None, got {limit!r}")
    return list(examples[:limit])


# ---------------------------------------------------------------------------
# GKD suite
# ---------------------------------------------------------------------------


def run_gkd_suite(
    model: Any,
    tokenizer: Any,
    examples: Sequence[EvalExample],
    *,
    shadow_ckpt: str | None = None,
    seq_len: int = 2048,
    max_windows: int | None = None,
    max_new_tokens: int = 128,
    limit: int | None = None,
    n_boot: int = DEFAULT_CI_BOOTSTRAP,
    seed: int = 0,
) -> SuiteReport:
    """Joint/shadow perplexity plus structured task accuracy (issue #2).

    Perplexity is measured on ``prompt + reference`` with
    :func:`evaluate_ppl`; task accuracy comes from greedy/sampled answers
    graded by :func:`grade_task`. A model or tokenizer that cannot score the
    corpus yields ``data_missing`` rather than a made-up number.
    """

    if model is None or tokenizer is None:
        return _data_missing(
            "gkd",
            "gkd needs a loaded model and tokenizer (--model/--shadow-ckpt).",
            examples=len(examples),
        )
    items = _apply_limit(examples, limit)
    if not items:
        return _data_missing("gkd", "gkd received no examples.", examples=0)
    config = {
        "shadow_ckpt": shadow_ckpt,
        "seq_len": int(seq_len),
        "max_windows": max_windows,
        "max_new_tokens": int(max_new_tokens),
        "limit": limit,
        "n_boot": int(n_boot),
        "seed": int(seed),
    }
    notes: list[str] = []
    metrics: dict[str, Any] = {
        "n_examples": len(items),
        "n_graded": 0,
        "n_generation_failures": 0,
        "joint_ppl": None,
        "shadow_ppl": None,
        "joint_ppl_ci": None,
        "shadow_ppl_ci": None,
        "ppl_tokens": 0,
        "ppl_windows": 0,
        "task_accuracy": None,
        "task_accuracy_ci": None,
    }

    texts = [_reference_text(example) for example in items]
    try:
        ppl = evaluate_ppl(
            model,
            tokenizer,
            texts,
            seq_len=seq_len,
            max_windows=max_windows,
        )
    except (TypeError, ValueError) as exc:
        notes.append(f"perplexity unavailable: {exc}")
    else:
        metrics["joint_ppl"] = float(ppl["joint_ppl"])
        metrics["shadow_ppl"] = float(ppl["shadow_ppl"])
        metrics["ppl_tokens"] = int(ppl["tokens"])
        metrics["ppl_windows"] = len(ppl["window_tokens"])
        metrics["joint_ppl_ci"] = _ppl_interval(
            ppl["joint_losses"],
            ppl["window_tokens"],
            n_boot=n_boot,
            seed=seed,
        )
        metrics["shadow_ppl_ci"] = _ppl_interval(
            ppl["shadow_losses"],
            ppl["window_tokens"],
            n_boot=n_boot,
            seed=seed,
        )

    torch = _import_torch()
    torch.manual_seed(seed)
    per_example: list[dict[str, Any]] = []
    graded: list[bool] = []
    for example in items:
        prediction: str | None = None
        error: str | None = None
        try:
            prediction = _generate_response(
                model,
                tokenizer,
                example.prompt,
                max_new_tokens=max_new_tokens,
                torch_module=torch,
            )
        except Exception as exc:  # noqa: BLE001 - one bad prompt must not kill the suite
            error = f"{type(exc).__name__}: {exc}"
            metrics["n_generation_failures"] += 1
        correct = _grade(example, prediction)
        if correct is not None:
            graded.append(bool(correct))
        per_example.append(
            {
                "id": example.id,
                "task": example.task,
                "expected": example.expected,
                "prediction": prediction,
                "correct": correct,
                "error": error,
            }
        )

    outcomes = graded
    if outcomes:
        metrics["n_graded"] = len(outcomes)
        metrics["task_accuracy"] = sum(outcomes) / len(outcomes)
        if len(outcomes) >= 2:
            metrics["task_accuracy_ci"] = list(
                bootstrap_ci(
                    [float(value) for value in outcomes], n_boot=n_boot, seed=seed
                )
            )
    if not outcomes:
        notes.append("no gradable structured answers; task_accuracy stays null.")
    if metrics["joint_ppl"] is None:
        notes.append("no perplexity windows were scored.")
    notes.append(CLAIM_NOTE)

    return SuiteReport(
        suite="gkd",
        status=STATUS_OK,
        metrics=metrics,
        per_example=tuple(per_example),
        config=config,
        claims_gain=False,
        evidence=_evidence_for(len(items)),
        notes=tuple(notes),
    )


# ---------------------------------------------------------------------------
# S2T suite
# ---------------------------------------------------------------------------


def run_s2t_suite(
    model: Any,
    tokenizer: Any,
    examples: Sequence[EvalExample],
    *,
    shadow_ckpt: str | None = None,
    shadow_model_path: str | None = None,
    max_new_tokens: int = 32,
    k: int = 8,
    use_cache: bool = False,
    limit: int | None = None,
    n_boot: int = DEFAULT_CI_BOOTSTRAP,
    seed: int = 0,
) -> SuiteReport:
    """Hit@K / Agree@1 candidate coverage on real prompts (issue #3).

    Reuses the diagnostic runner (initial shadow hidden) and the S2T report
    builder. ``selector_gain`` is not measurable without a selector head and
    stays ``None``.
    """

    if model is None or tokenizer is None:
        return _data_missing(
            "s2t",
            "s2t needs a loaded base_shadow model and tokenizer.",
            examples=len(examples),
        )
    items = _apply_limit(examples, limit)
    if not items:
        return _data_missing("s2t", "s2t received no examples.", examples=0)
    config = {
        "shadow_ckpt": shadow_ckpt,
        "shadow_model_path": shadow_model_path,
        "max_new_tokens": int(max_new_tokens),
        "k": int(k),
        "use_cache": bool(use_cache),
        "limit": limit,
        "n_boot": int(n_boot),
        "seed": int(seed),
    }
    prompts = [
        {
            "prompt_id": example.id,
            "prompt": example.prompt,
            "task": example.task or "unspecified",
        }
        for example in items
    ]
    records = diagnose(
        model,
        tokenizer,
        prompts,
        max_new_tokens=max_new_tokens,
        k=k,
        seed=seed,
        use_cache=use_cache,
    )
    if not records:
        return _data_missing(
            "s2t",
            "s2t diagnostic produced no tokens; check prompts and max_new_tokens.",
            config=config,
            examples=len(items),
        )
    report = build_s2t_report(records, ks=(k,), n_boot=n_boot, seed=seed)

    per_prompt: dict[str, dict[str, Any]] = {}
    for record in records:
        prompt_id = str(record.get("prompt_id"))
        row = per_prompt.setdefault(
            prompt_id,
            {
                "id": prompt_id,
                "task": record.get("task"),
                "n_tokens": 0,
                "hit_at_k": 0,
                "agree_at_1": 0,
            },
        )
        row["n_tokens"] += 1
        row["hit_at_k"] += int(int(record.get("base_top1_rank_in_shadow", k + 1)) <= k)
        row["agree_at_1"] += int(
            int(record.get("base_top1")) == int(record.get("shadow_topk_ids")[0])
            if record.get("shadow_topk_ids")
            else 0
        )
    for row in per_prompt.values():
        tokens = max(1, int(row["n_tokens"]))
        row["hit_at_k"] = row["hit_at_k"] / tokens
        row["agree_at_1"] = row["agree_at_1"] / tokens

    notes = [
        "Hit@K/Agree@1 come from the initial shadow hidden state, not the "
        "injected deployment path.",
        "selector_gain is null: this MVP runs no selector head.",
        CLAIM_NOTE,
    ]
    if k < 8:
        notes.append(f"K={k} < 8, so hit_at_8 is capped at the captured width.")

    metrics = {
        "n_examples": len(items),
        "n_prompts_with_tokens": int(report.get("n_prompts") or 0),
        "n_tokens": int(report.get("n_tokens") or 0),
        "k": int(k),
        "hit_at_8": report.get("hit_at_8"),
        "hit_at_8_ci": report.get("hit_at_8_ci"),
        "agree_at_1": report.get("agree_at_1"),
        "agree_at_1_ci": report.get("agree_at_1_ci"),
        "rank_all_p95": report.get("rank_all_p95"),
        "go_no_go": report.get("go_no_go"),
        "selector_gain": None,
    }
    return SuiteReport(
        suite="s2t",
        status=STATUS_OK,
        metrics=metrics,
        per_example=tuple(per_prompt.values()),
        config=config,
        claims_gain=False,
        evidence=_evidence_for(len(items)),
        notes=tuple(notes),
    )


# ---------------------------------------------------------------------------
# RESTRAIN suite
# ---------------------------------------------------------------------------


def _restrain_rollouts(
    model: Any,
    tokenizer: Any,
    example: EvalExample,
    *,
    n: int,
    max_new_tokens: int,
    temperature: float,
    torch_module: Any,
) -> tuple[list[str], list[str]]:
    answers: list[str] = []
    errors: list[str] = []
    for _ in range(n):
        try:
            answer = _generate_response(
                model,
                tokenizer,
                example.prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                torch_module=torch_module,
            )
        except Exception as exc:  # noqa: BLE001 - record and continue
            errors.append(f"{type(exc).__name__}: {exc}")
            continue
        answers.append(answer)
    return answers, errors


def run_restrain_suite(
    model: Any,
    tokenizer: Any,
    examples: Sequence[EvalExample],
    *,
    adapter: str | None = None,
    n: int = 4,
    max_new_tokens: int = 32,
    temperature: float = 1.0,
    kappa: int = 3,
    sigma: float = 0.5,
    limit: int | None = None,
    n_boot: int = DEFAULT_CI_BOOTSTRAP,
    seed: int = 0,
) -> SuiteReport:
    """Structured rollout accuracy, low-consistency rate and skip rate (#4).

    Each example gets ``n`` rollouts; answers are clustered/weighted with
    ``shadow_lab.restrain`` and graded with ``shadow_lab.restrain_eval``.
    ``skip_rate`` counts examples with no parseable/unclusterable rollouts.
    """

    if model is None or tokenizer is None:
        return _data_missing(
            "restrain",
            "restrain needs a loaded model and tokenizer.",
            examples=len(examples),
        )
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n!r}")
    items = _apply_limit(examples, limit)
    if not items:
        return _data_missing("restrain", "restrain received no examples.", examples=0)
    config = {
        "adapter": adapter,
        "n": int(n),
        "max_new_tokens": int(max_new_tokens),
        "temperature": float(temperature),
        "kappa": int(kappa),
        "sigma": float(sigma),
        "limit": limit,
        "n_boot": int(n_boot),
        "seed": int(seed),
    }

    torch = _import_torch()
    torch.manual_seed(seed)
    records: list[dict[str, Any]] = []
    counts_rows: list[dict[str, Any]] = []
    per_example: list[dict[str, Any]] = []
    entropy_values: list[float] = []
    top1_values: list[float] = []
    skipped = 0
    failures = 0

    for example in items:
        answers, errors = _restrain_rollouts(
            model,
            tokenizer,
            example,
            n=n,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            torch_module=torch,
        )
        failures += len(errors)
        cluster = cluster_answers(answers, example.task)
        is_skipped = cluster.unclusterable or not cluster.counts
        if is_skipped:
            skipped += 1
        counts_rows.append({"counts": cluster.counts})
        entropy = answer_entropy(cluster.counts)
        if entropy is not None:
            entropy_values.append(entropy)
            top1_values.append(top1_share(cluster.counts))
        graded: list[bool | None] = []
        for answer in answers:
            correct = grade_structured(answer, example.expected, example.task)
            graded.append(correct)
            if correct is not None:
                records.append(
                    {
                        "task": example.task,
                        "expected": example.expected,
                        "answer": answer,
                    }
                )
        per_example.append(
            {
                "id": example.id,
                "task": example.task,
                "expected": example.expected,
                "answers": answers,
                "counts": dict(cluster.counts),
                "cluster_method": cluster.cluster_method,
                "consistency": consistency(cluster.counts, sigma) if cluster.counts else 0.0,
                "skipped": bool(is_skipped),
                "correct": graded,
                "errors": errors,
            }
        )

    outcomes = [
        bool(value)
        for value in (
            grade_structured(record["answer"], record["expected"], record["task"])
            for record in records
        )
        if value is not None
    ]
    accuracy = sum(outcomes) / len(outcomes) if outcomes else None
    report = build_restrain_report(
        accuracy=accuracy,
        outcomes=outcomes,
        answer_entropy=_mean(entropy_values),
        top1_share=_mean(top1_values),
        low_consistency_rate=low_consistency_rate(counts_rows, kappa=kappa),
        skip_rate=skipped / len(items),
        config={
            "adapter": adapter,
            "temperature": float(temperature),
            "max_new_tokens": int(max_new_tokens),
        },
        sigma=sigma,
        kappa=kappa,
        n=n,
        n_boot=n_boot,
        ci_seed=seed,
    )
    report.update(
        {
            "n_examples": len(items),
            "n_rollouts": sum(len(row["answers"]) for row in per_example),
            "generation_failures": failures,
            "skipped_examples": skipped,
            "mean_consistency": _mean(
                [float(row["consistency"]) for row in per_example]
            ),
        }
    )
    notes = [
        "low_consistency_rate uses the prompt-weight gate: largest cluster < kappa.",
        "skip_rate counts examples with no parseable or unclusterable rollouts.",
    ]
    if temperature <= 0.0 and n > 1:
        notes.append("temperature=0 makes repeated rollouts identical; consistency is not informative.")
    notes.append(CLAIM_NOTE)
    return SuiteReport(
        suite="restrain",
        status=STATUS_OK,
        metrics=report,
        per_example=tuple(per_example),
        config=config,
        claims_gain=bool(report.get("claims_gain", False)),
        evidence=_evidence_for(len(items)),
        notes=tuple(notes),
    )


# ---------------------------------------------------------------------------
# JitRL suite
# ---------------------------------------------------------------------------


def _coerce_memory(
    memory: ExperienceMemory | str | Path | None,
) -> tuple[ExperienceMemory | None, str | None]:
    if memory is None:
        return None, "jitrl needs a feedback memory (--memory) or an ExperienceMemory."
    if isinstance(memory, ExperienceMemory):
        return memory, None
    if isinstance(memory, (str, Path)):
        path = Path(memory).expanduser()
        if not path.exists():
            return None, f"jitrl memory path not found: {path}"
        loaded = ExperienceMemory.load(path)
        if len(loaded) == 0:
            return None, f"jitrl memory is empty: {path}"
        return loaded, None
    raise TypeError(
        "memory must be an ExperienceMemory, a JSONL path or None, got "
        f"{type(memory).__name__}"
    )


def _sequence_logprob(generation: Generation) -> float:
    """Length-normalized log probability, matching the Router's rerank term."""

    values = [
        float(value)
        for value in (generation.token_logprobs or ())
        if math.isfinite(float(value))
    ]
    return sum(values) / len(values) if values else 0.0


def _best_index(candidates: Sequence[Candidate]) -> int:
    best = 0
    for index in range(1, len(candidates)):
        if candidates[index].logprob > candidates[best].logprob:
            best = index
    return best


def _cloud_text(
    cloud: Any,
    request: InferRequest,
    draft: str,
    *,
    timeout_ms: int = 5000,
) -> str:
    enhance = getattr(cloud, "enhance", None)
    if not callable(enhance):
        raise TypeError("cloud must expose enhance(CloudRequest)")
    cloud_request = CloudRequest(
        prompt=request.prompt,
        task=request.task,
        draft=draft,
    )
    try:
        response = enhance(cloud_request)
    except TypeError:  # pragma: no cover - clients with an explicit timeout kwarg
        response = enhance(cloud_request, timeout_ms=timeout_ms)
    return str(getattr(response, "text", "") or "")


def _escalation_confidence(
    generation: Generation,
    *,
    confidence: Any,
    expected_format: str | None,
) -> float | None:
    if confidence is None:
        return None
    report = confidence.estimate(
        generation.text,
        generation.token_logprobs,
        finish_reason=generation.finish_reason,
        expected_format=expected_format,
    )
    return float(getattr(report, "score", 0.0))


def run_jitrl_suite(
    examples: Sequence[EvalExample],
    *,
    memory: ExperienceMemory | str | Path | None = None,
    backend: Any = None,
    cloud: Any = None,
    n: int = 4,
    alpha: float = 0.5,
    threshold: float = 0.45,
    top_k: int = 4,
    min_support: int = 2,
    min_return: float = 0.0,
    tenant: str = "default",
    adapter_version: str = "base",
    escalation_threshold: float | None = 0.62,
    limit: int | None = None,
    n_boot: int = DEFAULT_CI_BOOTSTRAP,
    seed: int = 0,
) -> SuiteReport:
    """No-memory vs memory-reranked accuracy, false-bias and cloud calls (#5).

    ``backend`` must expose ``generate_nbest(request, n)``. Each example is
    reranked with :func:`edge_cloud.rerank.rerank` after retrieving memory hits
    for the same task/tenant/adapter version. With a cloud client, examples
    whose selected answer is below ``escalation_threshold`` are escalated and
    counted per arm; without one ``cloud_calls`` stays ``None``. Missing
    memory, backend or examples produce ``data_missing``.
    """

    if n < 1:
        raise ValueError(f"n must be >= 1, got {n!r}")
    if top_k < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k!r}")
    if min_support < 1:
        raise ValueError(f"min_support must be >= 1, got {min_support!r}")
    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError(f"threshold must be in [0, 1], got {threshold!r}")
    if not math.isfinite(float(alpha)):
        raise ValueError(f"alpha must be finite, got {alpha!r}")

    config = {
        "n": int(n),
        "alpha": float(alpha),
        "threshold": float(threshold),
        "top_k": int(top_k),
        "min_support": int(min_support),
        "min_return": float(min_return),
        "tenant": tenant,
        "adapter_version": adapter_version,
        "escalation_threshold": escalation_threshold,
        "limit": limit,
        "n_boot": int(n_boot),
        "seed": int(seed),
        "cloud_measured": cloud is not None,
    }
    if backend is None or not callable(getattr(backend, "generate_nbest", None)):
        return _data_missing(
            "jitrl",
            "jitrl needs an n-best backend exposing generate_nbest(request, n).",
            config=config,
            examples=len(examples),
        )
    loaded_memory, memory_error = _coerce_memory(memory)
    if loaded_memory is None:
        return _data_missing(
            "jitrl",
            memory_error or "jitrl memory unavailable.",
            config=config,
            examples=len(examples),
        )
    items = _apply_limit(examples, limit)
    if not items:
        return _data_missing("jitrl", "jitrl received no examples.", config=config)

    from edge_cloud.confidence import ConfidenceEstimator

    confidence = ConfidenceEstimator()
    cloud_calls = {"no_memory": 0, "rerank": 0}
    outcomes = {"no_memory": [], "rerank": []}
    per_example: list[dict[str, Any]] = []
    failures = 0
    memory_hits = 0
    reranked_cases = 0
    false_bias_count = 0
    false_bias_denominator = 0

    for example in items:
        task = example.task or "general"
        request = InferRequest(prompt=example.prompt, task=task)
        try:
            generations = [
                generation
                for generation in backend.generate_nbest(request, n=n)
                if isinstance(generation, Generation) and str(generation.text).strip()
            ]
        except Exception as exc:  # noqa: BLE001 - one backend failure must not kill the suite
            failures += 1
            per_example.append(
                {
                    "id": example.id,
                    "task": example.task,
                    "expected": example.expected,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        if not generations:
            failures += 1
            per_example.append(
                {
                    "id": example.id,
                    "task": example.task,
                    "expected": example.expected,
                    "error": "backend returned no candidates",
                }
            )
            continue

        candidates = [
            Candidate(
                text=generation.text,
                logprob=_sequence_logprob(generation),
                finish_reason=generation.finish_reason,
            )
            for generation in generations
        ]
        baseline_index = _best_index(candidates)
        hits = loaded_memory.retrieve(
            example.prompt,
            task=task,
            tenant=tenant,
            adapter_version=adapter_version,
            k=top_k,
            threshold=threshold,
        )
        result = rerank(
            candidates,
            hits,
            alpha=alpha,
            min_support=min_support,
            min_return=min_return,
        )
        baseline_generation = generations[baseline_index]
        rerank_generation = generations[result.index]
        memory_hits += int(bool(hits))
        reranked = result.reranked and rerank_generation.text != baseline_generation.text
        reranked_cases += int(reranked)

        answers: dict[str, str] = {}
        cloud_used: dict[str, bool] = {"no_memory": False, "rerank": False}
        for arm, generation in (
            ("no_memory", baseline_generation),
            ("rerank", rerank_generation),
        ):
            answer = generation.text
            score = _escalation_confidence(
                generation,
                confidence=confidence,
                expected_format=None,
            )
            if (
                cloud is not None
                and escalation_threshold is not None
                and score is not None
                and score < escalation_threshold
            ):
                try:
                    answer = _cloud_text(cloud, request, answer) or answer
                except Exception:  # noqa: BLE001 - fall back to the local answer
                    answer = generation.text
                else:
                    cloud_used[arm] = True
                    cloud_calls[arm] += 1
            answers[arm] = answer

        baseline_correct = grade_structured(
            answers["no_memory"], example.expected, example.task
        )
        rerank_correct = grade_structured(
            answers["rerank"], example.expected, example.task
        )
        if baseline_correct is not None:
            outcomes["no_memory"].append(bool(baseline_correct))
        if rerank_correct is not None:
            outcomes["rerank"].append(bool(rerank_correct))
        if baseline_correct is not None and rerank_correct is not None:
            false_bias_denominator += int(bool(baseline_correct))
            false_bias_count += int(bool(baseline_correct) and not bool(rerank_correct))
        per_example.append(
            {
                "id": example.id,
                "task": example.task,
                "expected": example.expected,
                "no_memory_answer": answers["no_memory"],
                "rerank_answer": answers["rerank"],
                "no_memory_correct": baseline_correct,
                "rerank_correct": rerank_correct,
                "memory_hits": len(hits),
                "memory_entry_ids": [hit.entry.entry_id for hit in hits],
                "reranked": bool(reranked),
                "rerank_reason": result.reason,
                "cloud_called_no_memory": cloud_used["no_memory"],
                "cloud_called_rerank": cloud_used["rerank"],
                "error": None,
            }
        )

    def arm_accuracy(arm: str) -> float | None:
        values = outcomes[arm]
        return sum(values) / len(values) if values else None

    def arm_ci(arm: str) -> list[float] | None:
        values = outcomes[arm]
        if len(values) < 2:
            return None
        return list(
            bootstrap_ci([float(value) for value in values], n_boot=n_boot, seed=seed)
        )

    no_memory_accuracy = arm_accuracy("no_memory")
    rerank_accuracy = arm_accuracy("rerank")
    metrics = {
        "n_examples": len(items),
        "n_gradable": len(outcomes["no_memory"]),
        "n_generation_failures": failures,
        "no_memory_accuracy": no_memory_accuracy,
        "no_memory_accuracy_ci": arm_ci("no_memory"),
        "rerank_accuracy": rerank_accuracy,
        "rerank_accuracy_ci": arm_ci("rerank"),
        "accuracy_delta": (
            None
            if no_memory_accuracy is None or rerank_accuracy is None
            else rerank_accuracy - no_memory_accuracy
        ),
        "false_bias_count": false_bias_count,
        "false_bias_rate": (
            false_bias_count / false_bias_denominator
            if false_bias_denominator
            else None
        ),
        "false_bias_denominator": false_bias_denominator,
        "cloud_calls_no_memory": cloud_calls["no_memory"] if cloud is not None else None,
        "cloud_calls_rerank": cloud_calls["rerank"] if cloud is not None else None,
        "cloud_calls_delta": (
            cloud_calls["rerank"] - cloud_calls["no_memory"]
            if cloud is not None
            else None
        ),
        "memory_hits": memory_hits,
        "reranked_cases": reranked_cases,
    }
    notes = [
        "accuracy is structured-only (quality_verdict/process_calc); "
        "open-ended examples are ungradable.",
        "cloud_calls are null when no cloud client is supplied.",
        CLAIM_NOTE,
    ]
    if not outcomes["no_memory"]:
        notes.append("no gradable examples; accuracy and false-bias stay null.")
    if memory_hits == 0:
        notes.append(
            "no memory hits; rerank equals the no-memory arm. Check tenant, "
            "task, adapter_version and threshold against the memory file."
        )
    if not any(not row.get("error") for row in per_example):
        return _data_missing(
            "jitrl",
            "n-best backend produced no usable candidates for any example.",
            config=config,
            examples=len(items),
        )
    return SuiteReport(
        suite="jitrl",
        status=STATUS_OK,
        metrics=metrics,
        per_example=tuple(per_example),
        config=config,
        claims_gain=False,
        evidence=_evidence_for(len(items)),
        notes=tuple(notes),
    )


# ---------------------------------------------------------------------------
# Dispatch and unified report
# ---------------------------------------------------------------------------


def run_suites(
    suites: Sequence[str] | str,
    *,
    examples: Sequence[EvalExample],
    model: Any = None,
    tokenizer: Any = None,
    shadow_ckpt: str | None = None,
    shadow_model_path: str | None = None,
    adapter: str | None = None,
    memory: ExperienceMemory | str | Path | None = None,
    backend: Any = None,
    cloud: Any = None,
    gkd_options: Mapping[str, Any] | None = None,
    s2t_options: Mapping[str, Any] | None = None,
    restrain_options: Mapping[str, Any] | None = None,
    jitrl_options: Mapping[str, Any] | None = None,
) -> dict[str, SuiteReport]:
    """Dispatch suites and preserve the canonical ``SUITES`` order.

    ``"all"`` expands to every suite. Missing model/tokenizer for the
    model-backed suites becomes a ``data_missing`` report, so ``--suite all``
    can still run the data-driven JitRL arm.
    """

    requested = [suites] if isinstance(suites, str) else list(suites)
    if "all" in requested:
        requested = list(SUITES)
    unknown = [suite for suite in requested if suite not in SUITES]
    if unknown:
        raise ValueError(f"unknown suite(s): {', '.join(unknown)}; expected {SUITES}")
    selected = [suite for suite in SUITES if suite in requested]
    gkd_options = dict(gkd_options or {})
    s2t_options = dict(s2t_options or {})
    restrain_options = dict(restrain_options or {})
    jitrl_options = dict(jitrl_options or {})

    reports: dict[str, SuiteReport] = {}
    for suite in selected:
        if suite == "gkd":
            reports[suite] = run_gkd_suite(
                model,
                tokenizer,
                examples,
                shadow_ckpt=shadow_ckpt,
                **gkd_options,
            )
        elif suite == "s2t":
            reports[suite] = run_s2t_suite(
                model,
                tokenizer,
                examples,
                shadow_ckpt=shadow_ckpt,
                shadow_model_path=shadow_model_path,
                **s2t_options,
            )
        elif suite == "restrain":
            reports[suite] = run_restrain_suite(
                model,
                tokenizer,
                examples,
                adapter=adapter,
                **restrain_options,
            )
        else:
            reports[suite] = run_jitrl_suite(
                examples,
                memory=memory,
                backend=backend,
                cloud=cloud,
                **jitrl_options,
            )
    return reports


def build_report(
    suites: Mapping[str, SuiteReport] | Sequence[SuiteReport],
    *,
    examples: Sequence[EvalExample] | None = None,
    dataset: Mapping[str, Any] | None = None,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the unified, auditable ``report.json`` payload.

    Includes the dataset fingerprint, per-suite payloads, the claim/evidence
    split, and a global conservative verdict: any ``data_missing`` or
    insufficient suite keeps the whole report at ``evidence_insufficient``.
    """

    report_items = (
        list(suites.values()) if isinstance(suites, Mapping) else list(suites)
    )
    if not report_items:
        raise ValueError("build_report needs at least one suite report")
    payload = {report.suite: report.to_dict() for report in report_items}
    if dataset is None:
        if examples is None:
            dataset_payload: dict[str, Any] = {}
        else:
            dataset_payload = {
                "name": "examples",
                "n_examples": len(examples),
                "fingerprint": dataset_fingerprint(examples),
            }
    else:
        dataset_payload = dict(dataset)
    if examples is not None and "fingerprint" not in dataset_payload:
        dataset_payload["fingerprint"] = dataset_fingerprint(examples)
        dataset_payload.setdefault("n_examples", len(examples))

    claims = [
        {
            "suite": report.suite,
            "status": report.status,
            "claims_gain": bool(report.claims_gain),
            "evidence": report.evidence,
        }
        for report in report_items
    ]
    claims_gain = any(item["claims_gain"] for item in claims)
    evidence = (
        EVIDENCE_INSUFFICIENT
        if any(
            item["status"] != STATUS_OK or item["evidence"] != EVIDENCE_COLLECTED
            for item in claims
        )
        else EVIDENCE_COLLECTED
    )
    notes = [
        "Small smoke samples are evidence_insufficient; they do not extrapolate "
        "to production gain.",
        CLAIM_NOTE,
    ]
    for report in report_items:
        notes.extend(f"{report.suite}: {note}" for note in report.notes)
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset": dataset_payload,
        "config": dict(config or {}),
        "suites": payload,
        "summary": {
            suite: {
                "status": item["status"],
                "evidence": item["evidence"],
                "claims_gain": item["claims_gain"],
            }
            for suite, item in payload.items()
        },
        "claims": claims,
        "claims_gain": claims_gain,
        "evidence": evidence,
        "notes": notes,
    }


_SUMMARY_FIELDS: tuple[str, ...] = (
    "suite",
    "status",
    "evidence",
    "claims_gain",
    "n_examples",
    "joint_ppl",
    "shadow_ppl",
    "task_accuracy",
    "hit_at_8",
    "agree_at_1",
    "accuracy",
    "low_consistency_rate",
    "skip_rate",
    "no_memory_accuracy",
    "rerank_accuracy",
    "false_bias_rate",
)


def _csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def write_report(
    report: Mapping[str, Any],
    out: str | Path,
) -> dict[str, str]:
    """Write ``report.json``, ``summary.csv`` and ``per_example.jsonl``."""

    out_dir = Path(out).expanduser()
    paths = {
        "report": str(out_dir / REPORT_FILENAME),
        "summary": str(out_dir / SUMMARY_FILENAME),
        "per_example": str(out_dir / PER_EXAMPLE_FILENAME),
    }
    _write_text(
        out_dir / REPORT_FILENAME,
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )

    suites = report.get("suites") or {}
    rows: list[dict[str, Any]] = []
    for suite, payload in suites.items():
        metrics = payload.get("metrics") or {}
        metrics = metrics if isinstance(metrics, Mapping) else {}
        row = {
            "suite": suite,
            "status": payload.get("status"),
            "evidence": payload.get("evidence"),
            "claims_gain": payload.get("claims_gain"),
        }
        for field_name in _SUMMARY_FIELDS[4:]:
            row[field_name] = metrics.get(field_name)
        rows.append(row)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(_SUMMARY_FIELDS), extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: _csv_value(row.get(key)) for key in _SUMMARY_FIELDS})
    _write_text(out_dir / SUMMARY_FILENAME, buffer.getvalue())

    lines: list[str] = []
    for suite, payload in suites.items():
        for row in payload.get("per_example") or ():
            lines.append(
                json.dumps(
                    {**dict(row), "suite": suite},
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
    _write_text(
        out_dir / PER_EXAMPLE_FILENAME,
        "\n".join(lines) + ("\n" if lines else ""),
    )
    return paths


__all__ = [
    "CLAIM_NOTE",
    "DEFAULT_CI_BOOTSTRAP",
    "EVIDENCE_COLLECTED",
    "EVIDENCE_INSUFFICIENT",
    "EVIDENCE_MIN_EXAMPLES",
    "EvalExample",
    "PER_EXAMPLE_FILENAME",
    "REPORT_FILENAME",
    "SCHEMA_VERSION",
    "STATUS_DATA_MISSING",
    "STATUS_OK",
    "SUITES",
    "SUMMARY_FILENAME",
    "SuiteReport",
    "build_report",
    "dataset_fingerprint",
    "load_gsm8k_examples",
    "load_jsonl_examples",
    "run_gkd_suite",
    "run_jitrl_suite",
    "run_restrain_suite",
    "run_s2t_suite",
    "run_suites",
    "write_report",
]
