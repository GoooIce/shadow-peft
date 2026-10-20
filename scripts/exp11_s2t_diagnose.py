#!/usr/bin/env python3
"""exp11: S2T-Local diagnostic runner (base top-1 vs. shadow top-K).

Examples::

    # Diagnose a shadow adapter on a JSONL prompt set, K=16, 32 new tokens.
    PYTHONPATH=src python scripts/exp11_s2t_diagnose.py \
        --model Qwen/Qwen3-4B \
        --shadow-ckpt runs/exp7/adapter \
        --prompts runs/exp11/prompts.jsonl \
        --max-new-tokens 32 --k 16 --seed 0 \
        --out runs/exp11/diagnostic

    # Same on a plain-text prompt file (one prompt per line), cached decode.
    PYTHONPATH=src python scripts/exp11_s2t_diagnose.py \
        --model Qwen/Qwen3-4B --shadow-ckpt runs/exp7/adapter \
        --prompts prompts.txt --use-cache --out runs/exp11/diagnostic

Outputs (in ``--out``):
- ``s2t_diagnostic.json``: report, Hit@K/Agree@1, bootstrap CI, go/no-go.
- ``s2t_tokens.jsonl``: one record per generated token.
- ``s2t_diagnostic.csv``: overall + task/position-bucket long format.

``shadow_logits`` come from the **initial** shadow hidden state, not the
injected ``base_shadow`` deployment path; the report says so explicitly.
Prompts are JSONL rows ``{"prompt": ..., "task": ...}`` (task optional) or
plain text with one prompt per line. Heavy imports are lazy so ``--help``
stays fast.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the S2T-Local diagnostic: greedy base generation while "
            "recording base top-1 and shadow top-K at every token."
        )
    )
    parser.add_argument("--model", required=True, help="Base model path or HF id.")
    parser.add_argument("--shadow-ckpt", required=True, help="ShadowPEFT adapter checkpoint.")
    parser.add_argument(
        "--shadow-model-path",
        default=None,
        help="Explicit shadow backbone (cross-architecture); needs --shadow-ckpt.",
    )
    parser.add_argument(
        "--prompts",
        required=True,
        help="JSONL rows with {'prompt', 'task'} or a plain-text file (one prompt per line).",
    )
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument(
        "--k",
        type=int,
        default=16,
        help="Shadow candidate capture width (default 16 covers Hit@1/4/8/16).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="runs/exp11_s2t_diagnose", help="Output directory.")
    parser.add_argument("--limit", type=int, default=None, help="Use only the first N prompts.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, mps, ...")
    parser.add_argument("--dtype", default="bfloat16", help="Torch dtype name, e.g. bfloat16.")
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Use the KV cache during decode (logits stay comparable to the full-prefix path).",
    )
    return parser


def load_prompts(path: str | Path) -> list[str | dict[str, Any]]:
    """Read JSONL prompts or a plain-text file with one prompt per line.

    A line starting with ``{`` is parsed as a JSON object with a required
    ``prompt`` string and optional ``task``/``prompt_id``. Other non-blank
    lines are kept verbatim as prompt text.
    """

    source = Path(path).expanduser()
    text = source.read_text(encoding="utf-8")
    prompts: list[str | dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        if line.lstrip().startswith("{"):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{source}:{line_number}: invalid JSON ({error})") from error
            if not isinstance(row, dict):
                raise ValueError(f"{source}:{line_number}: expected a JSON object")
            prompt = row.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"{source}:{line_number}: JSON prompt must be a non-empty string")
            parsed: dict[str, Any] = {"prompt": prompt}
            if row.get("task") not in (None, ""):
                parsed["task"] = str(row["task"])
            if row.get("prompt_id") is not None:
                parsed["prompt_id"] = str(row["prompt_id"])
            prompts.append(parsed)
        else:
            prompts.append(line)
    if not prompts:
        raise ValueError(f"{source}: no prompts found")
    return prompts


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


def _load_model(args: argparse.Namespace) -> tuple[Any, Any, dict[str, Any]]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from shadow_peft import AutoModelForCausalLMWithHiddenProjection, ShadowForCausalLM

    dtype = _resolve_dtype(torch, args.dtype)
    device = _resolve_device(torch, args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    try:
        base = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
    except TypeError:
        base = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    shadow_model = None
    if args.shadow_model_path:
        shadow_model = AutoModelForCausalLMWithHiddenProjection.from_pretrained(
            args.shadow_model_path
        )
    model = ShadowForCausalLM.from_pretrained(
        base,
        args.shadow_ckpt,
        shadow_model=shadow_model,
        inference_mode="base_shadow",
    )
    model.eval().to(device)
    metadata = {
        "model": str(args.model),
        "shadow_ckpt": str(args.shadow_ckpt),
        "shadow_model_path": (
            None if args.shadow_model_path is None else str(args.shadow_model_path)
        ),
        "prompts": str(args.prompts),
        "max_new_tokens": args.max_new_tokens,
        "k": args.k,
        "seed": args.seed,
        "device": device,
        "dtype": str(dtype),
        "use_cache": bool(args.use_cache),
        "limit": args.limit,
    }
    return model, tokenizer, metadata


def _run(args: argparse.Namespace) -> Any:
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be >= 1")
    if args.k < 1:
        raise ValueError("--k must be >= 1")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be >= 1")
    from shadow_lab.s2t_diagnose import run_diagnostic

    prompts = load_prompts(args.prompts)
    if args.limit is not None:
        prompts = prompts[: args.limit]
    model, tokenizer, metadata = _load_model(args)
    return run_diagnostic(
        model,
        tokenizer,
        prompts,
        out=args.out,
        max_new_tokens=args.max_new_tokens,
        k=args.k,
        seed=args.seed,
        use_cache=args.use_cache,
        metadata=metadata,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = _run(args)
    except (OSError, ValueError, RuntimeError, TypeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    report = result.report
    print(
        f"Hit@8={report['hit_at_8']:.3f} Agree@1={report['agree_at_1']:.3f} "
        f"n_tokens={report['n_tokens']} go_no_go={report['go_no_go']}"
    )
    for name, path in result.paths.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
