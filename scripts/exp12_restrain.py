#!/usr/bin/env python3
"""exp12: RESTRAIN-lite flywheel — export, weighted SFT, evaluation and report.

Examples::

    # 1) Roll out hard cases with a deployed adapter and export self-train rows.
    PYTHONPATH=src python scripts/exp12_restrain.py export \
        --feedback-store runs/edge_cloud/feedback.jsonl \
        --model Qwen/Qwen3-0.6B --shadow-ckpt runs/exp7/adapter \
        --n 8 --tasks quality_verdict,process_calc --max-new-tokens 64 \
        --out runs/exp12/self_train.jsonl

    # 2) Train weighted SFT against the same-budget best-answer SFT baseline.
    PYTHONPATH=src python scripts/exp12_restrain.py train \
        --self-train runs/exp12/self_train.jsonl \
        --sft runs/exp12/best_answer.jsonl \
        --model Qwen/Qwen3-0.6B --max-steps 200 --token-budget 200000 \
        --out runs/exp12/train

    # 3) Score held-out rollouts offline (prompt/answer/expected/task JSONL).
    PYTHONPATH=src python scripts/exp12_restrain.py eval \
        --predictions runs/exp12/heldout_predictions.jsonl \
        --out runs/exp12/report.json

    # 4) Or generate held-out rollouts with the trained adapter.
    PYTHONPATH=src python scripts/exp12_restrain.py eval \
        --heldout runs/exp12/heldout.jsonl --model Qwen/Qwen3-0.6B \
        --shadow-ckpt runs/exp12/train/weighted --n 8 \
        --baseline runs/exp12/eval_baseline.json --out runs/exp12/report.json

    # 5) Chain everything; --smoke keeps it to two prompts and one step.
    PYTHONPATH=src python scripts/exp12_restrain.py run \
        --feedback-store runs/edge_cloud/feedback.jsonl \
        --model Qwen/Qwen3-0.6B --shadow-ckpt runs/exp7/adapter \
        --heldout runs/exp12/heldout.jsonl --out runs/exp12/smoke --smoke

File formats:
- ``--self-train``: JSONL from ``FeedbackStore.export_self_train`` (``prompt``,
  ``task``, ``answers``, ``counts``, ``weights``, ``prompt_weight``).
- ``--sft``: JSONL from ``FeedbackStore.export_sft`` (``messages``); when it is
  omitted the best-answer baseline is derived from the self-train rows.
- ``--heldout``: JSONL rows with ``prompt`` plus optional ``task``/``expected``
  for generation; ``--predictions`` rows carry ``prompt``/``answer``/``expected``
  /``task`` and are scored without loading a model.

``export --from-jsonl`` passes an existing self-train file through (a fake
backend for tests and for re-scoring exported data). Heavy imports
(torch/transformers/edge_cloud) are lazy so ``--help`` stays fast.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from shadow_lab.restrain_eval import (
    accuracy,
    answer_counts,
    answer_entropy,
    best_answer_sft_to_rows,
    build_report,
    graded_outcomes,
    low_consistency_rate,
    self_train_to_rows,
    top1_share,
    train_weighted_sft,
)

DEFAULT_TASKS = ("quality_verdict", "process_calc")


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON ({error})") from error
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(row)
    return rows


def _write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> str:
    output = Path(path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return str(output)


def _read_json(path: str | Path) -> Any:
    return json.loads(Path(path).expanduser().read_text(encoding="utf-8"))


def _write_json(path: str | Path, payload: Any) -> str:
    output = Path(path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    return str(output)


def _task_list(value: str) -> list[str]:
    tasks = [item.strip() for item in value.split(",") if item.strip()]
    if not tasks:
        raise argparse.ArgumentTypeError("tasks must not be empty")
    return tasks


def _report_path(args: argparse.Namespace) -> Path:
    if getattr(args, "report", None):
        return Path(args.report).expanduser()
    out = Path(args.out).expanduser()
    return out.with_name(out.name + ".report.json")


def _seed_everything(seed: int) -> None:
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a hard dependency
        return
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# Backends (lazy)
# ---------------------------------------------------------------------------


def _resolve_dtype(name: str) -> str:
    if name != "auto":
        return name
    try:
        import torch
    except ImportError:
        return "float32"
    return "bfloat16" if torch.cuda.is_available() else "float32"


def _backend_spec(args: argparse.Namespace) -> dict[str, Any]:
    config = getattr(args, "backend_config", None)
    if config:
        payload = _read_json(config)
        if isinstance(payload, dict) and isinstance(payload.get("backend"), dict):
            payload = payload["backend"]
        if not isinstance(payload, dict) or not payload.get("kind"):
            raise ValueError(f"{config}: expected a backend object with a 'kind' field")
        return dict(payload)
    model = getattr(args, "model", None)
    if not model:
        raise ValueError("provide --backend-config or --model for rollout generation")
    spec: dict[str, Any] = {
        "kind": "transformers",
        "model_path": str(model),
        "dtype": _resolve_dtype(getattr(args, "dtype", "auto")),
    }
    if getattr(args, "device", None):
        spec["device"] = str(args.device)
    if getattr(args, "shadow_ckpt", None):
        spec["shadow_ckpt"] = str(args.shadow_ckpt)
    if getattr(args, "shadow_model_path", None):
        spec["shadow_model_path"] = str(args.shadow_model_path)
    if getattr(args, "system_prompt", None):
        spec["system_prompt"] = str(args.system_prompt)
    return spec


def _build_generate(args: argparse.Namespace):
    from edge_cloud.backends import build_backend
    from edge_cloud.protocol import InferRequest

    backend = build_backend(_backend_spec(args))
    default_tokens = int(getattr(args, "max_new_tokens", 256) or 256)
    temperature = float(getattr(args, "temperature", 0.0) or 0.0)

    def generate(prompt: str, task: str, max_new_tokens: int | None = None) -> str:
        request = InferRequest(
            prompt=prompt,
            task=task or "general",
            max_tokens=int(max_new_tokens or default_tokens),
            temperature=temperature,
        )
        return backend.generate(request).text

    return generate


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def _limited_feedback_store(path: str, limit: int, out_dir: Path):
    """A temporary store with at most *limit* unique hard-case prompts."""

    from edge_cloud.flywheel import FeedbackStore

    records = FeedbackStore(path).records()
    deduped: dict[str, dict[str, Any]] = {}
    for record in records:
        if not record.get("is_hard_case"):
            continue
        prompt = str(record.get("prompt", "")).strip()
        if prompt:
            deduped[prompt] = record
    selected = list(deduped.values())[:limit]
    if not selected:
        raise ValueError(f"no hard-case prompts in {path}")
    limited_path = out_dir / "feedback.limited.jsonl"
    _write_jsonl(limited_path, selected)
    return FeedbackStore(limited_path)


def cmd_export(args: argparse.Namespace) -> dict[str, Any]:
    """Export self-train rows, or pass an existing JSONL through."""

    out = Path(args.out).expanduser()
    if args.from_jsonl:
        rows = _read_jsonl(args.from_jsonl)
        _write_jsonl(out, rows)
        report = {
            "path": str(out),
            "source": str(args.from_jsonl),
            "records": len(rows),
            "selected": len(rows),
            "exported": len(rows),
            "skipped_empty": 0,
            "skipped_unsupported": 0,
            "skipped_unclusterable": 0,
            "rollout_failures": 0,
            "by_task": dict(Counter(str(row.get("task", "general")) for row in rows)),
            "cluster_methods": dict(
                Counter(str(row.get("cluster_method", "none")) for row in rows)
            ),
            "low_consistency_rate": low_consistency_rate(rows, kappa=args.kappa),
            "n": int(args.n),
            "seed": int(args.seed),
            "from_jsonl": True,
        }
    else:
        if not args.feedback_store:
            raise ValueError("provide --feedback-store or --from-jsonl")
        if args.n < 1:
            raise ValueError("--n must be >= 1")
        if args.max_new_tokens < 1:
            raise ValueError("--max-new-tokens must be >= 1")
        from edge_cloud.flywheel import FeedbackStore

        store = FeedbackStore(args.feedback_store)
        if args.limit is not None:
            store = _limited_feedback_store(
                args.feedback_store, max(1, args.limit), out.parent
            )
        _seed_everything(args.seed)
        generate = _build_generate(args)
        report = store.export_self_train(
            out,
            generate=generate,
            n=args.n,
            tasks=args.tasks,
            max_new_tokens=args.max_new_tokens,
            kappa=args.kappa,
            sigma=args.sigma,
        )
        rows = _read_jsonl(out)
        report["source"] = str(args.feedback_store)
        report["low_consistency_rate"] = low_consistency_rate(rows, kappa=args.kappa)
        report["n"] = int(args.n)
        report["seed"] = int(args.seed)
        report["from_jsonl"] = False

    report_path = _report_path(args)
    report["report_path"] = str(report_path)
    _write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return report


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------


def _load_trainable_shadow_model(args: argparse.Namespace, seed: int):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from shadow_peft import ShadowConfig, ShadowForCausalLM, get_shadow_model

    dtype = getattr(torch, _resolve_dtype(getattr(args, "dtype", "auto")), torch.float32)
    try:
        base = AutoModelForCausalLM.from_pretrained(str(args.model), dtype=dtype)
    except TypeError:
        base = AutoModelForCausalLM.from_pretrained(
            str(args.model), torch_dtype=dtype
        )
    tokenizer = AutoTokenizer.from_pretrained(str(args.model))
    if not getattr(tokenizer, "pad_token_id", None) and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    shadow_model = None
    if getattr(args, "shadow_model_path", None):
        from shadow_peft import AutoModelForCausalLMWithHiddenProjection

        shadow_model = AutoModelForCausalLMWithHiddenProjection.from_pretrained(
            str(args.shadow_model_path)
        )
    torch.manual_seed(seed)
    if getattr(args, "shadow_ckpt", None):
        model = ShadowForCausalLM.from_pretrained(
            base,
            str(args.shadow_ckpt),
            is_trainable=True,
            shadow_model=shadow_model,
        )
    else:
        peft = get_shadow_model(base, ShadowConfig(), shadow_model=shadow_model)
        model = ShadowForCausalLM(peft, inference_mode="base_shadow")

    device = _resolve_device(getattr(args, "device", None))
    model.to(device)
    return model, tokenizer


def _resolve_device(name: str | None = None):
    import torch

    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _train_one(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    out_dir: Path,
    *,
    label: str,
) -> dict[str, Any]:
    import torch

    model, tokenizer = _load_trainable_shadow_model(args, args.seed)
    result = train_weighted_sft(
        model,
        tokenizer,
        rows,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        seed=args.seed,
        max_seq_len=args.max_seq_len,
        token_budget=args.token_budget,
        limit=args.limit,
        device=getattr(args, "device", None),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    result["label"] = label
    result["adapter"] = str(out_dir)
    result["model"] = str(args.model)
    result["weights"] = bool(getattr(args, "weights", True)) if label == "weighted" else False
    _write_json(out_dir / "train_stats.json", result)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def cmd_train(args: argparse.Namespace) -> dict[str, Any]:
    """Train weighted SFT and the best-answer SFT baseline with one budget/seed."""

    self_rows = _read_jsonl(args.self_train)
    weighted_rows = self_train_to_rows(self_rows, use_weights=args.weights)
    if args.sft:
        baseline_rows = best_answer_sft_to_rows(_read_jsonl(args.sft))
    else:
        baseline_rows = [
            {**row, "weight": 1.0}
            for row in self_train_to_rows(self_rows, use_weights=False)
        ]
    if not weighted_rows:
        raise ValueError("no weighted training rows after the prompt-weight gate")
    if not baseline_rows:
        raise ValueError("no best-answer SFT rows")

    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    baseline = _train_one(args, baseline_rows, out / "baseline", label="baseline")
    weighted = _train_one(args, weighted_rows, out / "weighted", label="weighted")
    report = {
        "out": str(out),
        "weighted": weighted,
        "baseline": baseline,
        "config": {
            "weights": bool(args.weights),
            "max_steps": int(args.max_steps),
            "token_budget": args.token_budget,
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "max_seq_len": int(args.max_seq_len),
            "seed": int(args.seed),
            "n_self_train_rows": len(self_rows),
            "n_weighted_rows": len(weighted_rows),
            "n_baseline_rows": len(baseline_rows),
        },
    }
    _write_json(out / "train_stats.json", report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return report


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------


def _prompt_counts(records: list[dict[str, Any]]) -> list[dict[str, int]]:
    """One answer distribution per (prompt, task), including empty ones.

    An empty distribution means no rollout parsed into a structured answer;
    keeping the group lets the consistency rate count it as low instead of
    silently dropping the prompt.
    """

    groups: dict[tuple[str, str], list[str]] = {}
    for index, record in enumerate(records):
        answer = record.get("answer", record.get("response"))
        if answer is None:
            continue
        prompt = str(record.get("prompt") or f"#{index}")
        task = str(record.get("task") or "")
        groups.setdefault((prompt, task), []).append(str(answer))
    return [answer_counts(answers, task) for (_, task), answers in groups.items()]


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _generate_records(
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], int]:
    rows = _read_jsonl(args.heldout)
    if args.limit is not None:
        rows = rows[: max(1, args.limit)]
    prompts = [str(row.get("prompt", "")).strip() for row in rows]
    usable = [row for row, prompt in zip(rows, prompts, strict=True) if prompt]
    if not usable:
        raise ValueError(f"no usable prompts in {args.heldout}")

    _seed_everything(args.seed)
    generate = _build_generate(args)
    records: list[dict[str, Any]] = []
    failures = 0
    for row in usable:
        prompt = str(row["prompt"]).strip()
        task = str(row.get("task") or "general")
        expected = row.get("expected")
        for rollout in range(max(1, args.n)):
            try:
                answer = generate(prompt, task)
            except Exception:
                failures += 1
                continue
            if not isinstance(answer, str) or not answer.strip():
                failures += 1
                continue
            records.append(
                {
                    "prompt": prompt,
                    "task": task,
                    "expected": expected,
                    "answer": answer,
                    "rollout": rollout,
                }
            )
    if not records:
        raise ValueError("generation produced no usable answers")
    return records, failures


def _eval_report(args: argparse.Namespace) -> dict[str, Any]:
    if args.predictions:
        records = _read_jsonl(args.predictions)
        generation_failures = 0
    elif args.heldout:
        records, generation_failures = _generate_records(args)
    else:
        raise ValueError("provide --predictions (offline) or --heldout (generate)")

    outcomes = graded_outcomes(records)
    prompt_counts = _prompt_counts(records)
    entropy_values = [
        entropy
        for counts in prompt_counts
        if (entropy := answer_entropy(counts)) is not None
    ]
    top1_values = [
        top1_share(counts) for counts in prompt_counts if counts
    ]

    if getattr(args, "self_train", None):
        consistency_rows = _read_jsonl(args.self_train)
    else:
        consistency_rows = [{"counts": counts} for counts in prompt_counts]

    export_report = (
        _read_json(args.export_report) if getattr(args, "export_report", None) else None
    )
    baseline = _read_json(args.baseline) if getattr(args, "baseline", None) else None

    report = build_report(
        accuracy=accuracy(records),
        outcomes=outcomes,
        answer_entropy=_mean(entropy_values),
        top1_share=_mean(top1_values),
        low_consistency_rate=low_consistency_rate(
            consistency_rows, kappa=args.kappa
        ),
        export_report=export_report,
        escalation_rate=getattr(args, "escalation_rate", None),
        cloud_calls=getattr(args, "cloud_calls", 0),
        baseline=baseline,
        sigma=args.sigma,
        kappa=args.kappa,
        n=args.n,
        delta=args.delta,
        n_boot=args.ci_n_boot,
        ci_seed=args.ci_seed,
    )
    report["records"] = len(records)
    report["graded_records"] = len(outcomes)
    report["prompts"] = len(prompt_counts)
    report["generation_failures"] = generation_failures
    report["seed"] = int(args.seed)
    report["source"] = (
        str(args.predictions) if args.predictions else str(args.heldout)
    )
    return report


def cmd_eval(args: argparse.Namespace) -> dict[str, Any]:
    """Score held-out predictions (offline) or generate and score them."""

    report = _eval_report(args)
    out = _write_json(args.out, report)
    print(json.dumps({**report, "report_path": out}, ensure_ascii=False, sort_keys=True))
    return report


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


def cmd_run(args: argparse.Namespace) -> dict[str, Any]:
    """Chain export -> train -> baseline eval -> weighted eval + report."""

    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    self_train_path = out_dir / "self_train.jsonl"
    export_report_path = out_dir / "export_report.json"
    train_dir = out_dir / "train"
    baseline_report_path = out_dir / "eval_baseline.json"
    weighted_report_path = out_dir / "report.json"

    export_args = argparse.Namespace(**vars(args))
    export_args.out = str(self_train_path)
    export_args.report = str(export_report_path)
    export_summary = cmd_export(export_args)

    train_args = argparse.Namespace(**vars(args))
    train_args.self_train = str(self_train_path)
    train_args.out = str(train_dir)
    train_summary = cmd_train(train_args)

    eval_args = argparse.Namespace(**vars(args))
    eval_args.predictions = None
    eval_args.out = str(baseline_report_path)
    eval_args.shadow_ckpt = train_summary["baseline"]["adapter"]
    eval_args.self_train = str(self_train_path)
    eval_args.export_report = str(export_report_path)
    eval_args.baseline = None
    cmd_eval(eval_args)

    eval_args = argparse.Namespace(**vars(eval_args))
    eval_args.out = str(weighted_report_path)
    eval_args.shadow_ckpt = train_summary["weighted"]["adapter"]
    eval_args.baseline = str(baseline_report_path)
    report = cmd_eval(eval_args)

    summary = {
        "out": str(out_dir),
        "self_train": str(self_train_path),
        "export_report": str(export_report_path),
        "train": str(train_dir),
        "baseline_report": str(baseline_report_path),
        "report": str(weighted_report_path),
        "exported": export_summary["exported"],
        "weighted_tokens": train_summary["weighted"]["tokens_seen"],
        "baseline_tokens": train_summary["baseline"]["tokens_seen"],
        "accuracy": report["accuracy"],
        "baseline_best_answer": report["baseline_best_answer"],
        "claims_gain": report["claims_gain"],
    }
    _write_json(out_dir / "run_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return summary


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def _add(parser: argparse.ArgumentParser, *flags: str, **kwargs: Any) -> None:
    """Add an option unless a shared command group already registered it."""
    if any(flag in parser._option_string_actions for flag in flags):
        return
    parser.add_argument(*flags, **kwargs)


def _add_export_args(parser: argparse.ArgumentParser, *, smoke: bool = False) -> None:
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--feedback-store", help="FeedbackStore JSONL with hard cases"
    )
    source.add_argument(
        "--from-jsonl",
        help="existing self-train JSONL; copied through without a model",
    )
    _add(parser, "--out", required=True, help="self-train JSONL output path")
    _add(parser, "--report", default=None, help="export report JSON path")
    _add(parser, "--model", default=None, help="base model path or HF id")
    _add(parser, "--backend-config", default=None, help="edge_cloud backend JSON")
    _add(parser, "--shadow-ckpt", default=None, help="deployed adapter for rollouts")
    _add(parser, "--shadow-model-path", default=None, help="explicit shadow backbone")
    _add(parser, "--system-prompt", default="", help="optional generation system prompt")
    _add(parser, "--n", type=int, default=8, help="rollouts per prompt")
    _add(
        parser,
        "--tasks",
        type=_task_list,
        default=list(DEFAULT_TASKS),
        help="comma-separated structured tasks",
    )
    _add(parser, "--max-new-tokens", type=int, default=512)
    _add(parser, "--temperature", type=float, default=1.0)
    _add(parser, "--kappa", type=int, default=3, help="prompt-weight vote gate")
    _add(parser, "--sigma", type=float, default=0.5, help="RESTRAIN Gaussian bandwidth")
    _add(parser, "--seed", type=int, default=0)
    _add(parser, "--limit", type=int, default=None, help="cap prompts (debug)")
    _add(
        parser,
        "--dtype",
        choices=("auto", "float32", "bfloat16"),
        default="auto",
    )
    _add(parser, "--device", default=None)
    if smoke:
        _add(parser, "--smoke", action="store_true", help="tiny end-to-end run")


def _add_train_args(
    parser: argparse.ArgumentParser, *, require_self_train: bool = True
) -> None:
    _add(parser, "--out", required=True, help="adapter output directory")
    _add(
        parser,
        "--self-train",
        required=require_self_train,
        help="self-train JSONL from export",
    )
    _add(parser, "--sft", default=None, help="best-answer SFT JSONL (messages)")
    _add(parser, "--model", required=True, help="base model path or HF id")
    _add(parser, "--shadow-ckpt", default=None, help="existing adapter to continue")
    _add(parser, "--shadow-model-path", default=None, help="explicit shadow backbone")
    _add(
        parser,
        "--weights",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="apply prompt/cluster weights (default: on)",
    )
    _add(parser, "--max-steps", type=int, default=100)
    _add(parser, "--token-budget", type=int, default=None)
    _add(parser, "--batch-size", type=int, default=1)
    _add(parser, "--lr", type=float, default=1e-4)
    _add(parser, "--max-seq-len", type=int, default=512)
    _add(parser, "--seed", type=int, default=0)
    _add(parser, "--limit", type=int, default=None, help="cap training rows (debug)")
    _add(
        parser,
        "--dtype",
        choices=("auto", "float32", "bfloat16"),
        default="auto",
    )
    _add(parser, "--device", default=None)


def _add_eval_args(parser: argparse.ArgumentParser) -> None:
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--predictions",
        default=None,
        help="offline JSONL with prompt/answer/expected/task",
    )
    source.add_argument(
        "--heldout",
        default=None,
        help="held-out JSONL with prompt plus optional task/expected",
    )
    _add(parser, "--out", required=True, help="report JSON output path")
    _add(parser, "--model", default=None, help="base model path or HF id")
    _add(parser, "--backend-config", default=None, help="edge_cloud backend JSON")
    _add(parser, "--shadow-ckpt", default=None, help="adapter to evaluate")
    _add(parser, "--shadow-model-path", default=None, help="explicit shadow backbone")
    _add(parser, "--n", type=int, default=8, help="rollouts per held-out prompt")
    _add(parser, "--max-new-tokens", type=int, default=64)
    _add(parser, "--temperature", type=float, default=1.0)
    _add(parser, "--seed", type=int, default=0)
    _add(parser, "--limit", type=int, default=None, help="cap held-out prompts")
    _add(parser, "--self-train", default=None, help="self-train rows for consistency")
    _add(parser, "--export-report", default=None, help="export report JSON")
    _add(parser, "--baseline", default=None, help="best-answer SFT report JSON")
    _add(parser, "--escalation-rate", type=float, default=None)
    _add(parser, "--cloud-calls", type=int, default=0)
    _add(parser, "--sigma", type=float, default=0.5)
    _add(parser, "--kappa", type=int, default=3)
    _add(parser, "--delta", type=float, default=0.01, help="accuracy gain threshold")
    _add(parser, "--ci-n-boot", type=int, default=200)
    _add(parser, "--ci-seed", type=int, default=0)
    _add(
        parser,
        "--dtype",
        choices=("auto", "float32", "bfloat16"),
        default="auto",
    )
    _add(parser, "--device", default=None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="exp12_restrain.py",
        description=(
            "RESTRAIN-lite flywheel: hard-case rollouts, self-train export, "
            "weighted SFT vs best-answer SFT, held-out structured evaluation "
            "and an auditable JSON report."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    export = subparsers.add_parser(
        "export", help="export RESTRAIN-lite self-train rows from a feedback store"
    )
    _add_export_args(export)

    train = subparsers.add_parser(
        "train", help="train weighted SFT and the best-answer SFT baseline"
    )
    _add_train_args(train)

    evaluation = subparsers.add_parser(
        "eval", help="score held-out predictions and emit the report"
    )
    _add_eval_args(evaluation)

    run = subparsers.add_parser(
        "run", help="chain export -> train -> eval and compare to the baseline"
    )
    _add_export_args(run, smoke=True)
    _add_train_args(run, require_self_train=False)
    _add_eval_args(run)
    return parser


def _apply_smoke(args: argparse.Namespace) -> None:
    if not getattr(args, "smoke", False):
        return
    if getattr(args, "limit", None) is None:
        args.limit = 2
    else:
        args.limit = min(args.limit, 2)
    if hasattr(args, "n"):
        args.n = min(args.n, 2)
    if hasattr(args, "max_new_tokens"):
        args.max_new_tokens = min(args.max_new_tokens, 8)
    if hasattr(args, "max_steps"):
        args.max_steps = min(args.max_steps, 1)
    if hasattr(args, "token_budget") and args.token_budget is None:
        args.token_budget = 256
    if hasattr(args, "ci_n_boot"):
        args.ci_n_boot = min(args.ci_n_boot, 50)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _apply_smoke(args)
    handlers = {
        "export": cmd_export,
        "train": cmd_train,
        "eval": cmd_eval,
        "run": cmd_run,
    }
    try:
        handlers[args.command](args)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
