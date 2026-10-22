#!/usr/bin/env python3
"""exp18: train an S2T selector for the recovered standalone shadow (#47).

The exp11 selector was trained on the exp7 adapter's injected shadow hidden and
lost 0.116-0.161 Agree@1 against shadow-greedy. The recovered exp16 export is a
different (SFT'd, standalone) model, so this script builds a fresh selector on
its post-projection hidden and reports the same four references:

- ``greedy_agree_at_1``: shadow top-1 (candidate slot 0),
- ``random_agree_at_1``: chance-level pick inside top-K (hit/K),
- ``selector_agree_at_1``: the trained head's pick,
- ``oracle_agree_at_1``: coverage (base top-1 anywhere in top-K).

Training prompts come from the GSM8K **train** split; the exp17 S2T evaluation
scores the **test** split, so the two never share a prompt. Collection advances
on the base top-1 exactly like the diagnostic, and the selector sees only the
shadow hidden + candidate embeddings at inference time.

Example (GB10)::

    PYTHONPATH=src .venv/bin/python scripts/exp18_s2t_selector.py \
        --model ~/models/Qwen3-4B \
        --shadow-model-dir ~/exp16-full/runs/all/model \
        --train-limit 100 --max-new-tokens 16 --epochs 50 \
        --out ~/exp18-selector/all

Then evaluate it through exp17::

    PYTHONPATH=src .venv/bin/python scripts/exp17_real_eval.py \
        --suite s2t --dataset gsm8k --model ~/models/Qwen3-4B \
        --shadow-model-dir ~/exp16-full/runs/all/model \
        --selector ~/exp18-selector/all/selector \
        --limit 200 --max-new-tokens 16 --k 8 --out ~/exp18-selector/eval-200
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

EXPERIMENT = "exp18-s2t-selector"
SCHEMA_VERSION = "exp18-s2t-selector/v1"
REPORT_FILENAME = "train_report.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Collect GSM8K-train selector examples from the recovered "
            "standalone shadow, train a SelectorHead, and write it beside the "
            "run metadata."
        )
    )
    parser.add_argument("--model", required=True, help="Base model path (Qwen3-4B).")
    parser.add_argument(
        "--shadow-model-dir",
        required=True,
        help="exp16 exported standalone shadow (e.g. ~/exp16-full/runs/all/model).",
    )
    parser.add_argument(
        "--train-split",
        default="train",
        help="GSM8K split used for selector training (default: train).",
    )
    parser.add_argument("--train-limit", type=int, default=100)
    parser.add_argument("--train-seed", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--mlp-hidden", type=int, default=64)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument(
        "--out",
        default="runs-local/exp18/selector",
        help="Output directory (selector/ child plus train_report.json).",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Tiny collection/training budget for a wiring check (not evidence).",
    )
    return parser


def _resolve_smoke(args: argparse.Namespace) -> None:
    if not args.smoke:
        return
    args.train_limit = min(args.train_limit, 4)
    args.max_new_tokens = min(args.max_new_tokens, 4)
    args.epochs = min(args.epochs, 2)


def _run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM

    from shadow_lab.real_eval import (
        dataset_fingerprint,
        load_gsm8k_examples,
        load_standalone_shadow,
        standalone_s2t_diagnose,
    )
    from shadow_lab.s2t_selector import (
        save_selector,
        selector_agreement,
        split_selector_examples,
        train_selector,
    )
    from shadow_lab.shadow_only import resolve_dtype

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dtype = resolve_dtype(args.dtype)
    device = args.device
    shadow, tokenizer = load_standalone_shadow(
        args.shadow_model_dir, device=device, dtype=dtype
    )
    try:
        base = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
    except TypeError:  # pragma: no cover - older transformers
        base = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    base.eval().to(device)

    examples = load_gsm8k_examples(
        limit=args.train_limit, seed=args.train_seed, split=args.train_split
    )
    if not examples:
        raise SystemExit("no GSM8K prompts loaded; check --train-split/--train-limit")
    prompts = [
        {
            "prompt_id": example.id,
            "prompt": example.prompt,
            "task": example.task or "unspecified",
        }
        for example in examples
    ]

    started = time.time()
    records, collected = standalone_s2t_diagnose(
        base,
        shadow,
        tokenizer,
        prompts,
        k=args.k,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
        capture_selector=True,
    )
    collection_seconds = time.time() - started
    if not collected:
        raise SystemExit("collection produced no selector examples")

    weight = shadow.lm_head.weight.detach()
    started = time.time()
    result = train_selector(
        collected,
        candidate_weight=weight,
        epochs=args.epochs,
        lr=args.lr,
        beta=args.beta,
        seed=args.seed,
        device=device,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        mlp_hidden=args.mlp_hidden,
    )
    training_seconds = time.time() - started

    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    saved = save_selector(result.head, out / "selector", k=args.k)
    all_agreement = selector_agreement(result.head, collected, weight)
    _, val_examples = split_selector_examples(
        collected, val_ratio=args.val_ratio, seed=args.seed
    )
    val_agreement = selector_agreement(result.head, val_examples, weight)

    torch.save(collected, out / "train_examples.pt")
    with (out / "train_prompts.jsonl").open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(
                json.dumps(
                    {
                        "prompt_id": example.id,
                        "task": example.task,
                        "prompt": example.prompt,
                        "expected": example.expected,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )

    report = {
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT,
        "issue": 47,
        "smoke": bool(args.smoke),
        "config": {
            "model": args.model,
            "shadow_model_dir": str(Path(args.shadow_model_dir).expanduser()),
            "train_split": args.train_split,
            "train_limit": args.train_limit,
            "train_seed": args.train_seed,
            "max_new_tokens": args.max_new_tokens,
            "k": args.k,
            "epochs": args.epochs,
            "lr": args.lr,
            "beta": args.beta,
            "mlp_hidden": args.mlp_hidden,
            "val_ratio": args.val_ratio,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "device": args.device,
            "dtype": args.dtype,
        },
        "data": {
            "dataset": "gsm8k",
            "split": args.train_split,
            "n_prompts": len(examples),
            "fingerprint": dataset_fingerprint(examples),
            "prompt_ids": [example.id for example in examples],
            "n_tokens": len(collected),
            "n_hit_tokens": sum(1 for example in collected if example.hit),
        },
        "train": {
            "steps": result.steps,
            "n_train_examples": result.n_train_examples,
            "n_val_examples": result.n_val_examples,
            "n_train_prompts": result.n_train_prompts,
            "n_val_prompts": result.n_val_prompts,
            "val_loss": result.val_loss,
            "loss_first": result.history[0] if result.history else None,
            "loss_last": result.history[-1] if result.history else None,
            "loss_mean": (
                sum(result.history) / len(result.history) if result.history else None
            ),
            "trainable_parameters": sum(
                parameter.numel() for parameter in result.head.parameters()
            ),
            "seconds": training_seconds,
        },
        "collection": {
            "seconds": collection_seconds,
            "records": len(records),
        },
        "agreement": {
            "all_examples": all_agreement,
            "validation_split": val_agreement,
            "note": (
                "all_examples is in-sample and optimistic; validation_split is "
                "grouped by prompt and is the number to quote."
            ),
        },
        "selector": saved,
        "total_seconds": collection_seconds + training_seconds,
    }
    (out / REPORT_FILENAME).write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _resolve_smoke(args)
    report = _run(args)
    print(
        json.dumps(
            {
                "out": str(Path(args.out).expanduser()),
                "n_train_examples": report["train"]["n_train_examples"],
                "n_val_examples": report["train"]["n_val_examples"],
                "val_loss": report["train"]["val_loss"],
                "validation_agreement": report["agreement"]["validation_split"],
                "selector": report["selector"],
            },
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
