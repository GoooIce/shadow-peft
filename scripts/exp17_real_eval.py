#!/usr/bin/env python3
"""exp17: real-corpus re-evaluation harness for issues #2-#5 (issue #47 MVP).

This is the small, reusable entry point — not the full 2-4 day campaign. It
loads real examples (GSM8K test by default, ``runs/exp12/heldout.jsonl`` for
RESTRAIN), dispatches the requested suites through
:func:`shadow_lab.real_eval.run_suite` (the same entry as ``run_suites``) and
writes ``report.json`` / ``summary.csv`` / ``per_example.jsonl``.

Examples::

    # GSM8K test, GKD perplexity + task accuracy (first 50 rows, 8192 ppl tokens)
    PYTHONPATH=src python scripts/exp17_real_eval.py \
        --suite gkd --model /path/Qwen3-0.6B \
        --shadow-ckpt runs/exp7/e4b-seed0/ckpt-final \
        --limit 50 --token-budget 8192 --out runs-local/exp17/gkd

    # S2T coverage on the same real prompts; --selector adds selector_gain
    PYTHONPATH=src python scripts/exp17_real_eval.py \
        --suite s2t --model /path/Qwen3-4B \
        --shadow-ckpt runs/exp7/e4b-seed0/ckpt-final \
        --shadow-model /path/Qwen3-0.6B-H4B \
        --selector runs/exp11/selector --limit 50 --out runs-local/exp17/s2t

    # RESTRAIN structured hard cases (default dataset runs/exp12/heldout.jsonl)
    PYTHONPATH=src python scripts/exp17_real_eval.py \
        --suite restrain --model /path/Qwen3-0.6B \
        --shadow-ckpt runs/exp10/smoke/fresh-adapter \
        --limit 8 --n 4 --out runs-local/exp17/restrain

    # JitRL needs a real feedback memory and an n-best backend config
    PYTHONPATH=src python scripts/exp17_real_eval.py \
        --suite jitrl --dataset runs/exp13/dataset.jsonl \
        --memory runs/exp13/memory.jsonl \
        --backend-config runs/exp13/backend.json --out runs-local/exp17/jitrl

``--smoke`` only shrinks the sample/generation budget; it never invents data.
Missing inputs yield ``status="data_missing"`` in the report.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

DEFAULT_OUT = "runs/exp17-real-eval"
EXPERIMENT = "exp17-real-eval"
SPEC = "GitHub issue #47"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Real-corpus evaluation harness for GKD (#2), S2T (#3), "
            "RESTRAIN (#4) and JitRL (#5)."
        )
    )
    parser.add_argument(
        "--suite",
        choices=("gkd", "s2t", "restrain", "jitrl", "all"),
        default="all",
        help="Suite to run; 'all' runs every suite whose inputs are available.",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help=(
            "JSONL examples, or 'gsm8k' for the real GSM8K test split. "
            "Defaults: gsm8k for gkd/s2t, runs/exp12/heldout.jsonl for restrain; "
            "jitrl requires an explicit dataset."
        ),
    )
    parser.add_argument("--model", default=None, help="Base model path or HF id.")
    parser.add_argument(
        "--shadow-ckpt",
        default=None,
        help="ShadowPEFT adapter checkpoint; required for shadow/joint metrics.",
    )
    parser.add_argument(
        "--shadow-model",
        "--shadow-model-path",
        dest="shadow_model_path",
        default=None,
        help="Explicit shadow backbone (cross-architecture).",
    )
    parser.add_argument(
        "--memory",
        default=None,
        help="JitRL experience-memory JSONL (real feedback only).",
    )
    parser.add_argument(
        "--backend-config",
        default=None,
        help="JitRL n-best backend JSON (edge_cloud.backends.build_backend spec).",
    )
    parser.add_argument("--limit", type=int, default=None, help="Use only N examples.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, mps, ...")
    parser.add_argument("--dtype", default="auto", help="auto, bfloat16, float32, ...")
    parser.add_argument("--out", default=DEFAULT_OUT, help="Output directory.")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Shrink limit/generation budget for a wiring smoke run.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Generation budget (defaults: gkd 128, s2t/restrain 32; smoke 16).",
    )
    parser.add_argument("--k", type=int, default=8, help="S2T candidate width.")
    parser.add_argument(
        "--selector",
        default=None,
        help=(
            "S2T saved selector directory (selector.safetensors + "
            "selector_config.json); adds selector_gain."
        ),
    )
    parser.add_argument(
        "--token-budget",
        type=int,
        default=None,
        help=(
            "GKD perplexity budget in scored target tokens; maps to "
            "ceil(token_budget / seq_len) windows."
        ),
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Reuse the KV cache during S2T diagnostic decoding.",
    )
    parser.add_argument("--n", type=int, default=None, help="Rollouts / n-best size.")
    parser.add_argument("--temperature", type=float, default=1.0, help="RESTRAIN sampling temperature.")
    parser.add_argument("--kappa", type=int, default=3, help="RESTRAIN majority-vote gate.")
    parser.add_argument("--adapter", default=None, help="Adapter label for RESTRAIN/JitRL reports.")
    parser.add_argument(
        "--tenant",
        default="default",
        help="JitRL memory tenant.",
    )
    parser.add_argument(
        "--adapter-version",
        default="base",
        help="JitRL memory adapter-version filter.",
    )
    parser.add_argument("--alpha", type=float, default=0.5, help="JitRL memory rerank weight.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.45,
        help="JitRL memory retrieval similarity threshold.",
    )
    parser.add_argument("--top-k", type=int, default=4, help="JitRL memory hits.")
    parser.add_argument("--min-support", type=int, default=2, help="JitRL rerank support gate.")
    return parser


def _resolve_smoke(args: argparse.Namespace) -> None:
    if not args.smoke:
        return
    if args.limit is None:
        args.limit = 8
    if args.max_new_tokens is None:
        args.max_new_tokens = 16
    if args.n is None:
        args.n = 2


def _resolve_dtype(torch_module: Any, name: str) -> Any:
    if name != "auto":
        dtype = getattr(torch_module, name, None)
        if dtype is None:
            raise ValueError(f"unknown torch dtype: {name!r}")
        return dtype
    return torch_module.bfloat16 if torch_module.cuda.is_available() else torch_module.float32


def _resolve_device(torch_module: Any, name: str) -> Any:
    if name != "auto":
        return torch_module.device(name)
    if torch_module.cuda.is_available():
        return torch_module.device("cuda")
    mps = getattr(torch_module.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch_module.device("mps")
    return torch_module.device("cpu")


def _load_model(args: argparse.Namespace) -> tuple[Any | None, Any | None, str]:
    """Load the base tokenizer/model and, when given, the shadow adapter."""

    if not args.model:
        return None, None, "no --model supplied"
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = _resolve_dtype(torch, args.dtype)
    device = _resolve_device(torch, args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if getattr(tokenizer, "pad_token_id", None) is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    try:
        base = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
    except TypeError:  # pragma: no cover - older transformers
        base = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    if args.shadow_ckpt:
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
            inference_mode="base_shadow",
        )
    else:
        model = base
    model.eval().to(device)
    return model, tokenizer, "loaded"


def _dataset_examples(
    args: argparse.Namespace,
    suite: str,
) -> tuple[list[Any], dict[str, Any], str | None]:
    """Resolve the suite dataset; return ``(examples, metadata, error)``.

    The default source per suite comes from
    :data:`shadow_lab.real_eval.SUITE_DEFAULT_DATASETS`, so adding a suite does
    not require a new branch here.
    """

    from shadow_lab.real_eval import (
        SUITE_DEFAULT_DATASETS,
        load_gsm8k_examples,
        load_jsonl_examples,
    )

    requested = args.dataset
    if requested in (None, ""):
        requested = SUITE_DEFAULT_DATASETS.get(suite)
        if requested is None:
            return [], {"name": None}, f"{suite} requires --dataset (real examples)."
    if str(requested).lower() == "gsm8k":
        examples = load_gsm8k_examples(limit=args.limit, seed=args.seed)
        return (
            examples,
            {"name": "gsm8k", "split": "test", "limit": args.limit, "seed": args.seed},
            None,
        )
    path = Path(str(requested)).expanduser()
    if not path.is_file():
        return [], {"name": str(path)}, f"dataset not found: {path}"
    examples = load_jsonl_examples(path)
    if args.limit is not None:
        examples = examples[: max(1, args.limit)]
    return (
        examples,
        {"name": path.stem, "path": str(path), "n_available": len(examples)},
        None,
    )


def _build_jitrl_backend(args: argparse.Namespace) -> Any:
    if not args.backend_config:
        return None
    from edge_cloud.backends import build_backend

    payload = json.loads(Path(args.backend_config).expanduser().read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("backend"), dict):
        payload = payload["backend"]
    if not isinstance(payload, dict) or not payload.get("kind"):
        raise ValueError(
            f"{args.backend_config}: expected a backend object with a 'kind' field"
        )
    return build_backend(payload)


def _run(args: argparse.Namespace) -> dict[str, Any]:
    from shadow_lab.real_eval import (
        SUITES,
        build_report,
        data_missing_report,
        dataset_fingerprint,
        run_suite,
        write_report,
    )

    selected = list(SUITES) if args.suite == "all" else [args.suite]
    model, tokenizer, load_status = _load_model(args)
    # One flat option mapping feeds the shared dispatch; each suite picks the
    # keys it accepts and falls back to its own defaults.
    options = {
        "max_new_tokens": args.max_new_tokens,
        "token_budget": args.token_budget,
        "k": args.k,
        "use_cache": args.use_cache,
        "selector": args.selector,
        "n": args.n,
        "temperature": args.temperature,
        "kappa": args.kappa,
        "alpha": args.alpha,
        "threshold": args.threshold,
        "top_k": args.top_k,
        "min_support": args.min_support,
        "tenant": args.tenant,
        "adapter_version": args.adapter_version,
        "seed": args.seed,
        "n_boot": 200,
    }

    reports: dict[str, Any] = {}
    datasets: dict[str, Any] = {}
    all_examples: dict[str, Any] = {}
    for suite in selected:
        if suite != "jitrl" and model is None:
            datasets[suite] = {}
            reports[suite] = data_missing_report(suite, load_status)
            continue
        examples, metadata, error = _dataset_examples(args, suite)
        if examples:
            metadata = {
                **metadata,
                "n_examples": len(examples),
                "fingerprint": dataset_fingerprint(examples),
            }
            for example in examples:
                all_examples[example.id] = example
        datasets[suite] = metadata
        if error:
            reports[suite] = data_missing_report(
                suite, error, examples=len(examples)
            )
            continue
        if suite == "s2t" and not args.shadow_ckpt:
            reports[suite] = data_missing_report(
                "s2t",
                "s2t needs --shadow-ckpt for shadow_logits.",
                examples=len(examples),
            )
            continue
        reports[suite] = run_suite(
            suite,
            examples,
            model=model,
            tokenizer=tokenizer,
            shadow_ckpt=args.shadow_ckpt,
            shadow_model_path=args.shadow_model_path,
            adapter=args.adapter or args.shadow_ckpt,
            memory=args.memory,
            backend=_build_jitrl_backend(args) if suite == "jitrl" else None,
            cloud=None,
            options=options,
        )

    dataset_summary = {
        "name": "+".join(
            str(datasets.get(suite, {}).get("name") or suite) for suite in selected
        ),
        "per_suite": datasets,
    }
    report = build_report(
        reports,
        examples=list(all_examples.values()),
        dataset=dataset_summary,
        config={
            "experiment": EXPERIMENT,
            "spec": SPEC,
            "suite": args.suite,
            "model": args.model,
            "shadow_ckpt": args.shadow_ckpt,
            "shadow_model_path": args.shadow_model_path,
            "selector": args.selector,
            "token_budget": args.token_budget,
            "device": args.device,
            "dtype": args.dtype,
            "limit": args.limit,
            "seed": args.seed,
            "smoke": bool(args.smoke),
            "model_load": load_status,
        },
    )
    paths = write_report(report, args.out)
    return {
        "out": str(Path(args.out).expanduser()),
        "paths": paths,
        "evidence": report["evidence"],
        "claims_gain": report["claims_gain"],
        "summary": report["summary"],
        "notes": report["notes"][:8],
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _resolve_smoke(args)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be >= 1")
    result = _run(args)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
