#!/usr/bin/env python
"""exp16 -- shadow-only recovery pipeline (issue #46).

Four commands share one export path:

- ``diagnose``: save/reload round-trip plus a shadow-only generation baseline;
- ``train``: chat-template SFT on GSM8K train (or ``--dataset-jsonl``);
- ``eval``: GSM8K-style generation/grading from a saved export, plus the
  native ``--baseline-model`` comparison and the production-QA diagnostic;
- ``run``: diagnose -> train -> eval in one process.

Note: this is the revision for issue #46.  ``eval``/``run`` now require a
native baseline (e.g. ``~/models/Qwen3-0.6B``) before the accuracy gate can
pass; without ``--baseline-model`` the report says ``status="not_run"`` and
``recovery_ready`` stays false.  ``eval`` reports ``eval_ready`` (post-SFT
format/truncation/empty/accuracy) separately from ``recovery_ready``, which
also requires the round-trip that only ``run`` executes.  ``--epochs`` replaces
``--steps`` when set.

Every command writes ``report.json`` under ``--out``; generation commands also
write ``generations.jsonl`` and training writes ``train_stats.json``.  The
0.6B/4B checkpoints are never loaded by the unit tests; this script is meant to
run on the GB10 hosts.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import random
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

EXPERIMENT = "exp16-shadow-only"
ISSUE = 46

DEFAULT_BASE_MODEL = "~/models/Qwen3-4B"
DEFAULT_SHADOW_MODEL = "~/models/Qwen3-0.6B-H4B"
DEFAULT_SHADOW_CKPT = "runs/exp7/e4b-seed0/ckpt-final"
DEFAULT_QA_JSONL = "data/pyrodash_stage0/production_qa.jsonl"

GSM8K_SYSTEM = (
    "You are a careful math assistant. Solve the problem step by step. "
    "Put the final numeric answer on its own line as '#### <number>'."
)
ROUNDTRIP_PROMPT = "Question: What is 2 + 2?\nAnswer:"

PRODUCT_MIN_FORMAT = 0.80
PRODUCT_MAX_TRUNCATION = 0.20
PRODUCT_MAX_EMPTY = 0.05
ACCURACY_RATIO_MIN = 0.70

RECOVERY_GATE_KEYS = (
    "roundtrip_passed",
    "eval_format_ok",
    "eval_truncation_ok",
    "eval_empty_ok",
    "accuracy_gate_passed",
)
# ``eval`` runs no round-trip, so it can report post-SFT readiness without
# claiming the full recovery gate (which also requires a round-trip).
EVAL_GATE_KEYS = (
    "eval_format_ok",
    "eval_truncation_ok",
    "eval_empty_ok",
    "accuracy_gate_passed",
)


def _expand(path: str) -> str:
    return str(Path(path).expanduser())


def _resolve(value: Any, smoke_value: Any, normal_value: Any, smoke: bool) -> Any:
    if value is not None:
        return value
    return smoke_value if smoke else normal_value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_jsonl_rows(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).expanduser().open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def load_gsm8k_rows(
    *,
    split: str,
    limit: int,
    seed: int,
    for_training: bool,
    system_prompt: str = GSM8K_SYSTEM,
) -> list[dict[str, Any]]:
    """Load a seeded GSM8K subset in prompt/answer or messages form."""
    from datasets import load_dataset

    dataset = load_dataset("openai/gsm8k", "main", split=split)
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    rows: list[dict[str, Any]] = []
    for index in indices[: max(0, limit)]:
        item = dataset[index]
        question = str(item["question"]).strip()
        solution = str(item["answer"]).strip()
        if for_training:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": question})
            messages.append({"role": "assistant", "content": solution})
            rows.append(
                {
                    "id": f"gsm8k-{split}-{index}",
                    "messages": messages,
                    "answer": solution.rsplit("####", 1)[-1].strip(),
                }
            )
        else:
            rows.append(
                {
                    "id": f"gsm8k-{split}-{index}",
                    "prompt": question,
                    "answer": solution.rsplit("####", 1)[-1].strip(),
                }
            )
    return rows


def load_train_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    limit = _resolve(args.train_limit, 8, 2000, args.smoke)
    if args.dataset_jsonl:
        return load_jsonl_rows(args.dataset_jsonl, limit=limit)
    return load_gsm8k_rows(
        split="train",
        limit=limit,
        seed=args.seed,
        for_training=True,
        system_prompt=args.system_prompt,
    )


def _example_prompt(row: Mapping[str, Any], tokenizer: Any, system_prompt: str) -> str:
    from shadow_lab.shadow_only import format_chat_prompt

    messages = row.get("messages")
    if messages:
        # Drop trailing assistant turns so a gold answer can never leak into
        # the evaluation prompt.
        trimmed = list(messages)
        while trimmed and str(trimmed[-1].get("role") or "") == "assistant":
            trimmed.pop()
        if not trimmed:
            raise ValueError("evaluation messages 没有可用的 user prompt")
        return format_chat_prompt(tokenizer, trimmed)
    prompt = row.get("prompt", row.get("question"))
    if prompt is None:
        raise ValueError("evaluation row 需要 prompt/question 或 messages")
    chat = []
    if system_prompt:
        chat.append({"role": "system", "content": system_prompt})
    chat.append({"role": "user", "content": str(prompt)})
    return format_chat_prompt(tokenizer, chat)


def load_eval_examples(args: argparse.Namespace, tokenizer: Any) -> list[dict[str, Any]]:
    limit = _resolve(args.limit, 8, 100, args.smoke)
    if args.dataset_jsonl:
        rows = load_jsonl_rows(args.dataset_jsonl, limit=limit)
    else:
        rows = load_gsm8k_rows(
            split=args.eval_split,
            limit=limit,
            seed=args.seed,
            for_training=False,
            system_prompt=args.system_prompt,
        )
    examples: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        answer = row.get("answer", row.get("gold"))
        if answer is None:
            raise ValueError(f"evaluation row {index} 缺少 answer/gold")
        examples.append(
            {
                "id": row.get("id", f"example-{index}"),
                "prompt": _example_prompt(row, tokenizer, args.system_prompt),
                "answer": str(answer),
            }
        )
    return examples


def qa_path(path: str | Path | None = None) -> Path:
    """Resolve the production-QA fixture (``--qa-jsonl`` wins over the repo)."""
    if path:
        return Path(path).expanduser()
    repo = Path(__file__).resolve().parents[1] / DEFAULT_QA_JSONL
    if repo.is_file():
        return repo
    return Path(DEFAULT_QA_JSONL)


def load_qa_rows(path: str | Path | None = None) -> list[dict[str, Any]]:
    """Load the 16 production-QA rows (``prompt`` / ``task`` / expectations)."""
    source = qa_path(path)
    if not source.is_file():
        raise FileNotFoundError(
            f"找不到产线 QA 数据: {source}；用 --qa-jsonl 指定，或提供 "
            f"{DEFAULT_QA_JSONL}"
        )
    return load_jsonl_rows(source)


# ---------------------------------------------------------------------------
# Model loading / saving
# ---------------------------------------------------------------------------


def load_exported(args: argparse.Namespace) -> tuple[Any, Any]:
    from shadow_lab.shadow_only import load_exported_shadow

    return load_exported_shadow(
        _expand(args.base_model),
        _expand(args.shadow_model),
        _expand(args.shadow_ckpt),
        device=args.device,
        dtype=args.dtype,
    )


def load_saved(model_dir: str | Path, *, device: str, dtype: str) -> tuple[Any, Any]:
    from transformers import AutoModelForCausalLM

    # Registers the projection-wrapper config with AutoModelForCausalLM before
    # loading a saved H4B export in a fresh process.
    import shadow_peft  # noqa: F401
    from shadow_lab.shadow_only import load_shadow_tokenizer, resolve_dtype

    torch_dtype = resolve_dtype(dtype)
    try:
        model = AutoModelForCausalLM.from_pretrained(str(model_dir), dtype=torch_dtype)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            str(model_dir), torch_dtype=torch_dtype
        )
    # Cast non-persistent buffers too; from_pretrained loads parameters in
    # torch_dtype but leaves rotary inv_freq in fp32.
    model.eval().to(device=device, dtype=torch_dtype)
    tokenizer = load_shadow_tokenizer(model_dir)
    return model, tokenizer


def save_model(model: Any, tokenizer: Any, model_dir: str | Path) -> str:
    directory = Path(model_dir)
    directory.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(directory))
    tokenizer.save_pretrained(str(directory))
    return str(directory)


def load_native_model(path: str | Path, *, device: str, dtype: str) -> tuple[Any, Any]:
    """Load a plain HF causal LM + tokenizer for the native baseline."""
    from transformers import AutoModelForCausalLM

    from shadow_lab.shadow_only import load_shadow_tokenizer, resolve_dtype

    torch_dtype = resolve_dtype(dtype)
    try:
        model = AutoModelForCausalLM.from_pretrained(str(path), dtype=torch_dtype)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            str(path), torch_dtype=torch_dtype
        )
    model.eval().to(device=device, dtype=torch_dtype)
    return model, load_shadow_tokenizer(path)


def release_model(model: Any) -> None:
    """Drop a loaded model and return its CUDA cache before the next one."""
    del model
    gc.collect()
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _model_dir(args: argparse.Namespace) -> Path:
    if args.model_dir:
        return Path(args.model_dir).expanduser()
    return Path(args.out).expanduser() / "model"


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def _accuracy_gate(
    evaluation: Mapping[str, Any] | None,
    baseline: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Gate the shadow on a native baseline measured with the same prompts.

    No baseline, or a baseline with zero accuracy, can never pass: the ratio
    would be fabricated.  ``accuracy_ratio`` stays ``None`` in both cases.
    """
    baseline_accuracy = None
    if baseline is not None and baseline.get("status", "ok") == "ok":
        baseline_accuracy = baseline.get("accuracy")
    shadow_accuracy = None if evaluation is None else evaluation.get("accuracy")

    if shadow_accuracy is None or baseline_accuracy is None:
        return {
            "accuracy_status": "not_run",
            "baseline_accuracy": baseline_accuracy,
            "accuracy_ratio": None,
            "accuracy_gate_passed": False,
        }
    if float(baseline_accuracy) <= 0:
        return {
            "accuracy_status": "baseline_zero",
            "baseline_accuracy": float(baseline_accuracy),
            "accuracy_ratio": None,
            "accuracy_gate_passed": False,
        }

    ratio = float(shadow_accuracy) / float(baseline_accuracy)
    passed = ratio >= ACCURACY_RATIO_MIN
    return {
        "accuracy_status": "passed" if passed else "failed",
        "baseline_accuracy": float(baseline_accuracy),
        "accuracy_ratio": ratio,
        "accuracy_gate_passed": passed,
    }


def _gate_summary(
    roundtrip: Mapping[str, Any] | None,
    diagnose: Mapping[str, Any] | None,
    evaluation: Mapping[str, Any] | None,
    baseline: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the exp16 gates.

    ``recovery_ready`` requires *every* key in :data:`RECOVERY_GATE_KEYS`:
    round-trip, eval format, truncation, empty, and a native-baseline accuracy
    ratio of at least 0.70.  ``eval_ready`` is the same without the round-trip,
    so an ``eval``-only run can report post-SFT readiness without claiming the
    round-trip gate it did not execute.  A missing eval key keeps both false
    rather than silently dropping the gate.
    """
    gates: dict[str, Any] = {}
    if roundtrip is not None:
        gates["roundtrip_passed"] = bool(roundtrip.get("passed"))
    if diagnose is not None:
        gates["diagnose_format_ok"] = (
            float(diagnose.get("format_rate", 0.0)) >= PRODUCT_MIN_FORMAT
        )
    if evaluation is not None:
        gates["eval_format_ok"] = (
            float(evaluation.get("format_rate", 0.0)) >= PRODUCT_MIN_FORMAT
        )
        gates["eval_truncation_ok"] = (
            float(evaluation.get("truncation_rate", 1.0)) <= PRODUCT_MAX_TRUNCATION
        )
        gates["eval_empty_ok"] = (
            float(evaluation.get("empty_rate", 1.0)) <= PRODUCT_MAX_EMPTY
        )
    gates.update(_accuracy_gate(evaluation, baseline))
    gates["eval_ready"] = all(
        key in gates and bool(gates[key]) for key in EVAL_GATE_KEYS
    )
    gates["recovery_ready"] = (
        bool(gates.get("roundtrip_passed", False)) and gates["eval_ready"]
    )
    return gates


def _report_preamble(args: argparse.Namespace, command: str) -> dict[str, Any]:
    baseline_model = getattr(args, "baseline_model", None)
    qa_jsonl = getattr(args, "qa_jsonl", None)
    return {
        "experiment": EXPERIMENT,
        "issue": ISSUE,
        "command": command,
        "smoke": bool(args.smoke),
        "config": {
            "base_model": _expand(args.base_model),
            "shadow_model": _expand(args.shadow_model),
            "shadow_ckpt": _expand(args.shadow_ckpt),
            "baseline_model": _expand(baseline_model) if baseline_model else None,
            "qa_jsonl": str(qa_path(qa_jsonl)),
            "device": args.device,
            "dtype": args.dtype,
            "seed": args.seed,
            "dataset_jsonl": args.dataset_jsonl,
            "system_prompt": args.system_prompt,
            "steps": getattr(args, "steps", None),
            "epochs": getattr(args, "epochs", None),
        },
    }


def _evaluate_examples(
    args: argparse.Namespace,
    model: Any,
    tokenizer: Any,
    examples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    from shadow_lab.shadow_only import evaluate_shadow_only

    max_new_tokens = _resolve(args.max_new_tokens, 64, 512, args.smoke)
    batch_size = _resolve(args.eval_batch_size, 2, 4, args.smoke)
    result = evaluate_shadow_only(
        model,
        tokenizer,
        examples,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
        device=args.device,
    )
    for record, example in zip(result["generations"], examples, strict=True):
        record["id"] = example["id"]
    return result


def _evaluate(
    args: argparse.Namespace, model: Any, tokenizer: Any
) -> dict[str, Any]:
    return _evaluate_examples(args, model, tokenizer, load_eval_examples(args, tokenizer))


def _baseline_report(
    args: argparse.Namespace, examples: Sequence[Mapping[str, Any]]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run the native baseline on the exact shadow eval prompts, or not_run."""
    baseline_model = getattr(args, "baseline_model", None)
    if not baseline_model:
        return (
            {
                "status": "not_run",
                "model": None,
                "reason": "--baseline-model 未提供；accuracy gate 不能通过",
                "n": 0,
                "accuracy": None,
            },
            [],
        )

    model, tokenizer = load_native_model(
        _expand(baseline_model), device=args.device, dtype=args.dtype
    )
    try:
        result = _evaluate_examples(args, model, tokenizer, examples)
    finally:
        release_model(model)

    summary = {
        "status": "ok",
        "model": _expand(baseline_model),
        **result["summary"],
    }
    return summary, result["generations"]


def _evaluate_qa(
    args: argparse.Namespace, model: Any, tokenizer: Any
) -> dict[str, Any]:
    """Grade the 16 production-QA rows as a diagnostic (never a gate)."""
    from shadow_lab.shadow_only import evaluate_qa_only

    source = qa_path(getattr(args, "qa_jsonl", None))
    if not source.is_file():
        return {
            "summary": {
                "status": "not_run",
                "path": str(source),
                "reason": f"找不到产线 QA 数据: {source}",
                "n": 0,
                "accuracy": None,
                "format_rate": None,
                "keyword_coverage": None,
            },
            "generations": [],
        }

    rows = load_qa_rows(source)
    max_new_tokens = _resolve(
        getattr(args, "qa_max_new_tokens", None), 64, 256, args.smoke
    )
    batch_size = _resolve(args.eval_batch_size, 2, 4, args.smoke)
    result = evaluate_qa_only(
        model,
        tokenizer,
        rows,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
        device=args.device,
    )
    summary = {"status": "ok", "path": str(source), **result["summary"]}
    return {"summary": summary, "generations": result["generations"]}


def _run_roundtrip(
    args: argparse.Namespace, model: Any, tokenizer: Any
) -> dict[str, Any]:
    from shadow_lab.shadow_only import export_roundtrip_check

    return export_roundtrip_check(
        model,
        tokenizer,
        prompt=ROUNDTRIP_PROMPT,
        max_length=64,
        device=args.device,
        dtype=args.dtype,
        atol=args.roundtrip_atol,
        rtol=args.roundtrip_rtol,
    )


def _train(
    args: argparse.Namespace, model: Any, tokenizer: Any
) -> tuple[Any, dict[str, Any]]:
    from shadow_lab.shadow_only import ShadowOnlyTrainConfig, train_shadow_only_sft

    rows = load_train_rows(args)
    config = ShadowOnlyTrainConfig(
        train_scope=args.train_scope,
        lr=args.lr,
        steps=_resolve(args.steps, 2, 200, args.smoke),
        epochs=args.epochs,
        batch_size=_resolve(args.train_batch_size, 2, 2, args.smoke),
        max_seq_len=_resolve(args.max_seq_len, 256, 1024, args.smoke),
        seed=args.seed,
        dtype=args.dtype,
        device=args.device,
        grad_clip=args.grad_clip,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        log_every=args.log_every,
    )
    stats = train_shadow_only_sft(model, tokenizer, rows, config)
    return model, stats.as_dict()


def cmd_diagnose(args: argparse.Namespace) -> int:
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_exported(args)
    roundtrip = _run_roundtrip(args, model, tokenizer)
    diagnose_args = copy.copy(args)
    diagnose_args.limit = _resolve(args.limit, 4, 32, args.smoke)
    result = _evaluate(diagnose_args, model, tokenizer)
    report = _report_preamble(args, "diagnose")
    report["roundtrip"] = roundtrip
    report["generation"] = result["summary"]
    report["gates"] = _gate_summary(roundtrip, result["summary"], None)
    _write_json(out / "report.json", report)
    _write_jsonl(out / "generations.jsonl", result["generations"])
    print(
        json.dumps(
            {
                "command": "diagnose",
                "roundtrip_passed": roundtrip["passed"],
                "format_rate": result["summary"]["format_rate"],
                "accuracy": result["summary"]["accuracy"],
                "out": str(out),
            },
            ensure_ascii=False,
        )
    )
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_exported(args)
    model, stats = _train(args, model, tokenizer)
    model_dir = save_model(model, tokenizer, _model_dir(args))
    report = _report_preamble(args, "train")
    report["train"] = stats
    report["model_dir"] = model_dir
    _write_json(out / "train_stats.json", stats)
    _write_json(out / "report.json", report)
    print(
        json.dumps(
            {
                "command": "train",
                "steps": stats["steps"],
                "tokens": stats["tokens"],
                "loss_last": stats["loss_last"],
                "model_dir": model_dir,
            },
            ensure_ascii=False,
        )
    )
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    model_dir = _model_dir(args)
    if model_dir.exists():
        model, tokenizer = load_saved(
            model_dir, device=args.device, dtype=args.dtype
        )
    elif args.model_dir:
        raise FileNotFoundError(
            f"--model-dir 指向的目录不存在: {model_dir}；先运行 train 或去掉该参数以现场导出"
        )
    else:
        model, tokenizer = load_exported(args)
    examples = load_eval_examples(args, tokenizer)
    result = _evaluate_examples(args, model, tokenizer, examples)
    evaluation = result["summary"]
    baseline, baseline_generations = _baseline_report(args, examples)
    qa = _evaluate_qa(args, model, tokenizer)

    report = _report_preamble(args, "eval")
    report["generation"] = evaluation
    report["baseline"] = baseline
    report["qa"] = qa["summary"]
    report["gates"] = _gate_summary(None, None, evaluation, baseline)
    report["baseline_accuracy"] = baseline.get("accuracy")
    report["accuracy_ratio"] = report["gates"]["accuracy_ratio"]
    report["accuracy_gate_passed"] = report["gates"]["accuracy_gate_passed"]
    report["model_dir"] = str(model_dir) if model_dir.exists() else "in-memory-export"
    _write_json(out / "report.json", report)
    _write_jsonl(out / "generations.jsonl", result["generations"])
    _write_jsonl(out / "baseline_generations.jsonl", baseline_generations)
    _write_jsonl(out / "qa_generations.jsonl", qa["generations"])
    print(
        json.dumps(
            {
                "command": "eval",
                "format_rate": evaluation["format_rate"],
                "accuracy": evaluation["accuracy"],
                "truncation_rate": evaluation["truncation_rate"],
                "baseline_status": baseline["status"],
                "accuracy_ratio": report["gates"]["accuracy_ratio"],
                "qa_accuracy": qa["summary"]["accuracy"],
                "eval_ready": report["gates"]["eval_ready"],
                "recovery_ready": report["gates"]["recovery_ready"],
                "out": str(out),
            },
            ensure_ascii=False,
        )
    )
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_exported(args)

    roundtrip = _run_roundtrip(args, model, tokenizer)
    diagnose_args = copy.copy(args)
    diagnose_args.limit = _resolve(args.diagnose_limit, 4, 32, args.smoke)
    diagnose = _evaluate(diagnose_args, model, tokenizer)
    model, stats = _train(args, model, tokenizer)
    model_dir = save_model(model, tokenizer, _model_dir(args))
    examples = load_eval_examples(args, tokenizer)
    evaluation_result = _evaluate_examples(args, model, tokenizer, examples)
    evaluation = evaluation_result["summary"]
    baseline, baseline_generations = _baseline_report(args, examples)
    qa = _evaluate_qa(args, model, tokenizer)

    report = _report_preamble(args, "run")
    report["roundtrip"] = roundtrip
    report["diagnose"] = diagnose["summary"]
    report["train"] = stats
    report["eval"] = evaluation
    report["baseline"] = baseline
    report["qa"] = qa["summary"]
    report["gates"] = _gate_summary(
        roundtrip, diagnose["summary"], evaluation, baseline
    )
    report["baseline_accuracy"] = baseline.get("accuracy")
    report["accuracy_ratio"] = report["gates"]["accuracy_ratio"]
    report["accuracy_gate_passed"] = report["gates"]["accuracy_gate_passed"]
    report["model_dir"] = model_dir
    _write_json(out / "report.json", report)
    _write_json(out / "train_stats.json", stats)
    _write_jsonl(out / "generations.jsonl", evaluation_result["generations"])
    _write_jsonl(out / "diagnose_generations.jsonl", diagnose["generations"])
    _write_jsonl(out / "baseline_generations.jsonl", baseline_generations)
    _write_jsonl(out / "qa_generations.jsonl", qa["generations"])
    print(
        json.dumps(
            {
                "command": "run",
                "roundtrip_passed": roundtrip["passed"],
                "diagnose_format_rate": diagnose["summary"]["format_rate"],
                "eval_format_rate": evaluation["format_rate"],
                "eval_accuracy": evaluation["accuracy"],
                "baseline_status": baseline["status"],
                "accuracy_ratio": report["gates"]["accuracy_ratio"],
                "qa_accuracy": qa["summary"]["accuracy"],
                "eval_ready": report["gates"]["eval_ready"],
                "recovery_ready": report["gates"]["recovery_ready"],
                "out": str(out),
            },
            ensure_ascii=False,
        )
    )
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--shadow-model", default=DEFAULT_SHADOW_MODEL)
    parser.add_argument("--shadow-ckpt", default=DEFAULT_SHADOW_CKPT)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--system-prompt", default=GSM8K_SYSTEM)
    parser.add_argument("--dataset-jsonl", default=None)
    parser.add_argument(
        "--smoke", action="store_true", help="4-8 条样本、2 步训练的小规模冒烟"
    )


def _add_diagnose_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--roundtrip-atol", type=float, default=1e-4)
    parser.add_argument("--roundtrip-rtol", type=float, default=1e-4)


def _add_train_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="每轮 ceil(len(rows)/batch_size) 步；设置后忽略 --steps",
    )
    parser.add_argument("--train-batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max-seq-len", type=int, default=None)
    parser.add_argument(
        "--train-scope",
        default="backbone",
        choices=("backbone", "head", "all", "none"),
    )
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--model-dir", default=None)


def _add_baseline_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--baseline-model",
        default=None,
        help="原生 baseline 模型路径（如 ~/models/Qwen3-0.6B）；不提供则 accuracy gate 记 not_run",
    )


def _add_qa_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--qa-jsonl",
        default=None,
        help=f"产线 QA 数据；默认仓库内 {DEFAULT_QA_JSONL}",
    )
    parser.add_argument("--qa-max-new-tokens", type=int, default=None)


def _add_eval_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--model-dir", default=None)
    _add_baseline_args(parser)
    _add_qa_args(parser)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="exp16 shadow-only recovery pipeline (issue #46)"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    diagnose = subparsers.add_parser(
        "diagnose", help="export round-trip + shadow-only generation baseline"
    )
    _add_common(diagnose)
    _add_diagnose_args(diagnose)

    train = subparsers.add_parser("train", help="shadow-only SFT")
    _add_common(train)
    _add_train_args(train)

    evaluate = subparsers.add_parser("eval", help="grade a saved export on GSM8K")
    _add_common(evaluate)
    _add_eval_args(evaluate)

    run = subparsers.add_parser("run", help="diagnose -> train -> eval")
    _add_common(run)
    _add_diagnose_args(run)
    _add_train_args(run)
    _add_baseline_args(run)
    _add_qa_args(run)
    # ``run`` needs the eval limit separate from the diagnose limit.
    run.add_argument("--diagnose-limit", type=int, default=None)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    commands = {
        "diagnose": cmd_diagnose,
        "train": cmd_train,
        "eval": cmd_eval,
        "run": cmd_run,
    }
    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
