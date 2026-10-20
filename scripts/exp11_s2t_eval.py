#!/usr/bin/env python3
"""exp11: evaluate S2T-Local shadow-only generation with/without a selector.

Modes
-----
``shadow_greedy``
    Shadow-only top-1 generation. No selector head and no base forward run
    during inference.
``shadow_selector``
    Shadow top-K candidates re-ranked by a selector head. Without a head the
    wrapper falls back to the same greedy tokens. A greedy baseline is run for
    the latency comparison unless ``--skip-greedy-baseline`` is set.
``base_upper``
    ``base_shadow`` joint greedy generation (the same upper-bound control the
    diagnostic used). It needs the base model plus the shadow checkpoint and is
    not a deployable shadow-only mode.

For the shadow modes, Agree@1, Hit@K and the ``delta_hit``/``delta_sel``
breakdown need the ``base_shadow`` top-1 of the same context. That reference
is measured by one **separate teacher-forced base_shadow forward** over each
generated prefix after generation; it is not part of the shadow inference path
and can be disabled with ``--skip-base-reference``.

The smoke command for this ticket is ``--help``; real 4B/0.6B runs happen on
the GB10 machines as part of #29. Inputs are JSONL rows with a required
``prompt`` string and optional ``task``/``expected``/``prompt_id`` fields.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

DEFAULT_OUT = "runs/exp11_s2t_eval"
DEFAULT_LATENCY_BUDGET = 0.20
SCHEMA_VERSION = "s2t-local-eval/v1"
REPORT_FILENAME = "s2t_eval.json"
RECORDS_FILENAME = "s2t_eval_prompts.jsonl"
BASE_REFERENCE_NOTE = (
    "Agree@1/Hit@K/delta_hit/delta_sel use the base_shadow top-1 from one "
    "separate teacher-forced base_shadow forward over each generated prefix; "
    "that forward is evaluation-only and is never part of shadow inference."
)
EXP8_REFERENCE = {
    "name": "exp8 logit ensemble beta=0.7",
    "perplexity": 8.238,
    "requires_base": True,
    "source": (
        "issue #3 (docs/specs/s2t-local-shadow-selector.md, "
        "§Implementation Decisions 6)"
    ),
    "note": "Quoted archive number for comparison; this script does not rerun exp8.",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate S2T-Local shadow-only generation: greedy vs selector "
            "re-rank vs the base upper bound."
        )
    )
    parser.add_argument("--model", required=True, help="Base model path or HF id.")
    parser.add_argument(
        "--shadow-ckpt",
        default=None,
        help="ShadowPEFT adapter checkpoint (required for the shadow modes).",
    )
    parser.add_argument(
        "--shadow-model-path",
        default=None,
        help="Explicit shadow backbone (cross-architecture); needs --shadow-ckpt.",
    )
    parser.add_argument(
        "--selector",
        default=None,
        help="Selector directory with selector.safetensors + selector_config.json.",
    )
    parser.add_argument(
        "--prompts",
        required=True,
        help="JSONL rows with {'prompt', 'task', 'expected', 'prompt_id'} (only prompt required).",
    )
    parser.add_argument(
        "--mode",
        choices=("shadow_greedy", "shadow_selector", "base_upper"),
        default="shadow_selector",
        help="Generation path to evaluate.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--k", type=int, default=8, help="Shadow candidate count.")
    parser.add_argument("--limit", type=int, default=None, help="Use only the first N prompts.")
    parser.add_argument("--out", default=DEFAULT_OUT, help="Output directory.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, mps, ...")
    parser.add_argument("--dtype", default="bfloat16", help="Torch dtype name, e.g. bfloat16.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Reuse the shadow backbone KV-cache during decode.",
    )
    parser.add_argument(
        "--skip-greedy-baseline",
        action="store_true",
        help="Do not run the shadow-greedy latency baseline in shadow_selector mode.",
    )
    parser.add_argument(
        "--skip-base-reference",
        action="store_true",
        help="Skip the base top-1 reference (Agree@1/Hit@K/delta metrics stay null).",
    )
    parser.add_argument(
        "--dump-tokens",
        action="store_true",
        help="Include per-token candidates/scores in the prompts JSONL.",
    )
    parser.add_argument(
        "--latency-budget",
        type=float,
        default=DEFAULT_LATENCY_BUDGET,
        help="Maximum tolerated per-token latency overhead before deployable=false.",
    )
    return parser


def load_prompts(path: str | Path) -> list[dict[str, Any]]:
    """Read JSONL prompts, preserving task/expected/prompt_id for grading."""

    source = Path(path).expanduser()
    text = source.read_text(encoding="utf-8")
    prompts: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{source}:{line_number}: invalid JSON ({error})") from error
        if not isinstance(row, dict):
            raise ValueError(f"{source}:{line_number}: expected a JSON object")
        prompt = row.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"{source}:{line_number}: JSON prompt must be a non-empty string")
        raw_task = row.get("task")
        item: dict[str, Any] = {
            "prompt_id": (
                f"p{len(prompts)}" if row.get("prompt_id") is None else str(row["prompt_id"])
            ),
            "prompt": prompt,
            "task": None if raw_task in (None, "") else str(raw_task),
        }
        if "expected" in row:
            item["expected"] = row["expected"]
        prompts.append(item)
    if not prompts:
        raise ValueError(f"{source}: no prompts found")
    return prompts


def _require_positive_int(value: int, *, name: str) -> int:
    if isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def _resolve_device(torch_module: Any, device: str) -> str:
    if device != "auto":
        return device
    if torch_module.cuda.is_available():
        return "cuda"
    mps = getattr(torch_module.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def _resolve_dtype(torch_module: Any, name: str) -> Any:
    dtype = getattr(torch_module, name, None)
    if dtype is None:
        raise ValueError(f"unknown torch dtype: {name!r}")
    return dtype


def _load_base(args: argparse.Namespace) -> tuple[Any, Any, Any, str]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = _resolve_dtype(torch, args.dtype)
    device = _resolve_device(torch, args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    try:
        base = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
    except TypeError:
        base = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    return base.eval().to(device), tokenizer, torch, device


def _load_shadow(args: argparse.Namespace, base: Any, device: str) -> Any:
    from shadow_peft import AutoModelForCausalLMWithHiddenProjection, ShadowForCausalLM

    shadow_model = None
    if args.shadow_model_path:
        shadow_model = AutoModelForCausalLMWithHiddenProjection.from_pretrained(
            args.shadow_model_path
        )
    model = ShadowForCausalLM.from_pretrained(
        base,
        args.shadow_ckpt,
        shadow_model=shadow_model,
        inference_mode="shadow_only",
    )
    return model.eval().to(device)


def _input_ids(tokenizer: Any, prompt: str, torch_module: Any, device: Any) -> tuple[Any, Any]:
    batch = tokenizer(prompt, return_tensors="pt")
    input_ids = batch["input_ids"]
    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)
    if int(input_ids.shape[-1]) == 0:
        raise ValueError("prompt tokenized to zero tokens")
    attention_mask = batch.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch_module.ones_like(input_ids)
    return input_ids.to(device), attention_mask.to(device)


def _base_shadow_generate(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    max_new_tokens: int,
    torch_module: Any,
) -> tuple[tuple[int, ...], str, float]:
    """Greedy generation on the joint ``base_shadow`` path (upper-bound control)."""

    device = next(model.parameters()).device
    input_ids, attention_mask = _input_ids(tokenizer, prompt, torch_module, device)
    kwargs: dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "use_cache": True,
    }
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "eos_token_id", None)
    if pad_token_id is not None:
        kwargs["pad_token_id"] = pad_token_id
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is not None:
        kwargs["eos_token_id"] = eos_token_id
    previous_mode = model.inference_mode
    model.set_inference_mode("base_shadow")
    started = time.perf_counter()
    try:
        with torch_module.inference_mode():
            output = model.generate(**kwargs)
    finally:
        model.set_inference_mode(previous_mode)
    elapsed = time.perf_counter() - started
    token_ids = tuple(int(token) for token in output[0, int(input_ids.shape[1]) :].tolist())
    text = tokenizer.decode(list(token_ids), skip_special_tokens=True)
    return token_ids, str(text), elapsed


def _reference_top1(
    model: Any,
    prompt_ids: tuple[int, ...],
    token_ids: tuple[int, ...],
    *,
    torch_module: Any,
) -> list[int]:
    """Teacher-forced ``base_shadow`` top-1 under each generated prefix.

    This is the same joint reference the diagnostic measured: the base model
    runs with the shadow injection/update path active. It is a separate
    evaluation forward, never part of shadow-only generation.
    """

    if not token_ids:
        return []
    device = next(model.parameters()).device
    full = torch_module.tensor(
        [list(prompt_ids) + list(token_ids)], dtype=torch_module.long, device=device
    )
    attention_mask = torch_module.ones_like(full)
    with torch_module.inference_mode():
        outputs, _shadow_hidden = model.peft_model.forward_with_shadow(
            input_ids=full,
            attention_mask=attention_mask,
            use_cache=False,
        )
    logits = outputs.logits[0]
    start = len(prompt_ids) - 1
    return [int(logits[start + index].argmax()) for index in range(len(token_ids))]


def _aggregate_breakdown(breakdowns: list[Any], breakdown_class: Any) -> Any:
    return breakdown_class(
        n_tokens=sum(item.n_tokens for item in breakdowns),
        hit_count=sum(item.hit_count for item in breakdowns),
        agree_count=sum(item.agree_count for item in breakdowns),
        greedy_agree_count=sum(item.greedy_agree_count for item in breakdowns),
    )


def _task_accuracy(records: list[dict[str, Any]]) -> tuple[float | None, int]:
    outcomes = [
        bool(record["task_outcome"])
        for record in records
        if record.get("task_outcome") is not None
    ]
    if not outcomes:
        return None, 0
    return sum(outcomes) / len(outcomes), len(outcomes)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _evaluate(args: argparse.Namespace) -> dict[str, Any]:
    if not args.shadow_ckpt:
        raise ValueError("--shadow-ckpt is required for every evaluation mode")
    if args.selector and args.mode != "shadow_selector":
        raise ValueError("--selector is only valid with --mode shadow_selector")
    if args.latency_budget < 0:
        raise ValueError("--latency-budget must be non-negative")
    _require_positive_int(args.k, name="--k")
    _require_positive_int(args.max_new_tokens, name="--max-new-tokens")
    if args.limit is not None:
        _require_positive_int(args.limit, name="--limit")

    from shadow_lab.onpolicy_eval import grade_task
    from shadow_lab.s2t_infer import (
        SelectionBreakdown,
        SelectorShadowModel,
        latency_comparison,
        selection_breakdown,
    )

    prompts = load_prompts(args.prompts)
    if args.limit is not None:
        prompts = prompts[: args.limit]
    base, tokenizer, torch_module, device = _load_base(args)
    model = _load_shadow(args, base, device)
    runner = None
    if args.mode != "base_upper":
        runner = SelectorShadowModel(
            model,
            tokenizer,
            selector=args.selector,
            k=args.k,
            use_cache=args.use_cache,
            device=device,
        )
    selector_active = bool(args.selector) and args.mode == "shadow_selector"
    selector_seconds = 0.0
    selector_tokens = 0
    greedy_seconds = 0.0
    greedy_tokens = 0
    records: list[dict[str, Any]] = []
    breakdowns: list[Any] = []

    for prompt in prompts:
        record: dict[str, Any] = {
            "prompt_id": prompt["prompt_id"],
            "task": prompt["task"],
            "prompt": prompt["prompt"],
        }
        if "expected" in prompt:
            record["expected"] = prompt["expected"]
        if args.mode == "base_upper":
            token_ids, text, elapsed = _base_shadow_generate(
                model,
                tokenizer,
                prompt["prompt"],
                max_new_tokens=args.max_new_tokens,
                torch_module=torch_module,
            )
            record.update(
                {
                    "response": text,
                    "token_ids": list(token_ids),
                    "n_tokens": len(token_ids),
                    "elapsed_seconds": elapsed,
                    "base_top1": None,
                    "breakdown": None,
                }
            )
        else:
            generation = runner.generate(
                prompt["prompt"],
                max_new_tokens=args.max_new_tokens,
                use_selector=args.mode == "shadow_selector",
            )
            selector_seconds += generation.elapsed_seconds
            selector_tokens += generation.n_tokens
            base_top1 = None
            breakdown = None
            if not args.skip_base_reference:
                base_top1 = _reference_top1(
                    model,
                    generation.prompt_ids,
                    generation.token_ids,
                    torch_module=torch_module,
                )
                breakdown = selection_breakdown(generation.steps, base_top1)
                breakdowns.append(breakdown)
            record.update(
                {
                    "response": generation.text,
                    "token_ids": list(generation.token_ids),
                    "n_tokens": generation.n_tokens,
                    "elapsed_seconds": generation.elapsed_seconds,
                    "base_top1": base_top1,
                    "breakdown": None if breakdown is None else breakdown.to_dict(),
                }
            )
            if args.dump_tokens:
                record["steps"] = [step.to_dict() for step in generation.steps]
            if (
                args.mode == "shadow_selector"
                and selector_active
                and not args.skip_greedy_baseline
            ):
                greedy_generation = runner.generate(
                    prompt["prompt"],
                    max_new_tokens=args.max_new_tokens,
                    use_selector=False,
                )
                greedy_seconds += greedy_generation.elapsed_seconds
                greedy_tokens += greedy_generation.n_tokens
        record["task_outcome"] = grade_task(
            record["response"], prompt.get("expected"), prompt.get("task")
        )
        records.append(record)

    breakdown = _aggregate_breakdown(breakdowns, SelectionBreakdown) if breakdowns else None
    task_accuracy, n_gradable = _task_accuracy(records)
    mode_seconds = sum(record["elapsed_seconds"] for record in records)
    mode_tokens = sum(record["n_tokens"] for record in records)
    latency: dict[str, Any] = {
        "mode": args.mode,
        "mode_seconds": mode_seconds,
        "mode_seconds_per_token": mode_seconds / mode_tokens if mode_tokens else None,
        "shadow_seconds": selector_seconds,
        "n_shadow_tokens": selector_tokens,
        "shadow_seconds_per_token": (
            selector_seconds / selector_tokens if selector_tokens else None
        ),
        "greedy_seconds": greedy_seconds if greedy_tokens else None,
        "n_greedy_tokens": greedy_tokens,
        "greedy_seconds_per_token": greedy_seconds / greedy_tokens if greedy_tokens else None,
        "selector_over_greedy": None,
        "overhead": None,
        "budget": args.latency_budget,
        "deployable": None,
    }
    deployable = None
    if selector_active and greedy_tokens:
        comparison = latency_comparison(
            selector_seconds=selector_seconds,
            selector_tokens=selector_tokens,
            greedy_seconds=greedy_seconds,
            greedy_tokens=greedy_tokens,
            budget=args.latency_budget,
        )
        latency.update(comparison.to_dict())
        deployable = comparison.deployable

    notes: list[str] = []
    if args.mode == "base_upper":
        notes.append(
            "base_upper runs the base model and is the upper-bound control; it is "
            "not a deployable shadow-only mode."
        )
    elif breakdowns:
        notes.append(BASE_REFERENCE_NOTE)
    if args.skip_base_reference:
        notes.append(
            "Base reference skipped: Agree@1, Hit@K and the delta breakdown are null."
        )
    elif breakdowns:
        notes.append(
            "hit_at_8 is 1 - delta_hit at the configured candidate width "
            f"(--k {args.k}, default 8)."
        )
    if args.mode == "shadow_selector" and not selector_active:
        notes.append(
            "No selector head supplied; shadow_selector fell back to shadow-greedy tokens."
        )
    if selector_active:
        if args.skip_greedy_baseline:
            notes.append(
                "Greedy latency baseline skipped; deployable is null and the "
                "selector/greedy latency ratio is unavailable."
            )
        else:
            notes.append(
                "deployable=false when the selector's per-token latency is more than "
                f"{args.latency_budget:.0%} above shadow-greedy."
            )
    notes.append(
        "The selector head is an additional adapter file; export_shadow's lm_head and "
        "vocabulary are unchanged."
    )

    out_dir = Path(args.out).expanduser()
    paths = {
        "json": str(out_dir / REPORT_FILENAME),
        "prompts_jsonl": str(out_dir / RECORDS_FILENAME),
    }
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "mode": args.mode,
        "model": str(args.model),
        "shadow_ckpt": None if args.shadow_ckpt is None else str(args.shadow_ckpt),
        "shadow_model_path": (
            None if args.shadow_model_path is None else str(args.shadow_model_path)
        ),
        "selector": None if args.selector is None else str(args.selector),
        "selector_active": selector_active,
        "k": args.k,
        "max_new_tokens": args.max_new_tokens,
        "use_cache": bool(args.use_cache),
        "seed": args.seed,
        "n_prompts": len(prompts),
        "n_tokens": sum(record["n_tokens"] for record in records),
        "task_accuracy": task_accuracy,
        "n_gradable_tasks": n_gradable,
        "breakdown": None if breakdown is None else breakdown.to_dict(),
        "hit_at_k": None if breakdown is None else breakdown.hit_at_k,
        "hit_at_8": None if breakdown is None else breakdown.hit_at_k,
        "agree_at_1": None if breakdown is None else breakdown.agree_at_1,
        "greedy_agree_at_1": None if breakdown is None else breakdown.greedy_agree_at_1,
        "delta_hit": None if breakdown is None else breakdown.delta_hit,
        "delta_sel": None if breakdown is None else breakdown.delta_sel,
        "selection_accuracy_on_hits": (
            None if breakdown is None else breakdown.selection_accuracy_on_hits
        ),
        "selector_gain": None if breakdown is None else breakdown.selector_gain,
        "latency_overhead": latency.get("overhead"),
        "latency": latency,
        "deployable": deployable,
        "base_reference": BASE_REFERENCE_NOTE if breakdowns else None,
        "exp8_reference": EXP8_REFERENCE,
        "notes": notes,
        "paths": paths,
    }

    _write_text(
        Path(paths["json"]),
        json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
    )
    lines = [
        json.dumps(record, ensure_ascii=False, allow_nan=False, sort_keys=True)
        for record in records
    ]
    _write_text(Path(paths["prompts_jsonl"]), "\n".join(lines) + "\n")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = _evaluate(args)
    except (OSError, ValueError, RuntimeError, TypeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    accuracy = report["task_accuracy"]
    accuracy_text = "n/a" if accuracy is None else f"{accuracy:.3f}"
    agree = report["agree_at_1"]
    agree_text = "n/a" if agree is None else f"{agree:.3f}"
    print(
        f"mode={report['mode']} n_tokens={report['n_tokens']} "
        f"task_accuracy={accuracy_text} agree@1={agree_text} "
        f"deployable={report['deployable']}"
    )
    for name, path in report["paths"].items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
