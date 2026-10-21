#!/usr/bin/env python
"""exp16 -- shadow-only recovery pipeline (issue #46).

Four commands share one export path:

- ``diagnose``: save/reload round-trip plus a shadow-only generation baseline;
- ``train``: chat-template SFT on GSM8K train (or ``--dataset-jsonl``);
- ``eval``: GSM8K-style generation/grading from a saved export;
- ``run``: diagnose -> train -> eval in one process.

Every command writes ``report.json`` under ``--out``; generation commands also
write ``generations.jsonl`` and training writes ``train_stats.json``.  The
0.6B/4B checkpoints are never loaded by the unit tests; this script is meant to
run on the GB10 hosts.
"""

from __future__ import annotations

import argparse
import copy
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

GSM8K_SYSTEM = (
    "You are a careful math assistant. Solve the problem step by step. "
    "Put the final numeric answer on its own line as '#### <number>'."
)
ROUNDTRIP_PROMPT = "Question: What is 2 + 2?\nAnswer:"

PRODUCT_MIN_FORMAT = 0.80
PRODUCT_MAX_TRUNCATION = 0.20
PRODUCT_MAX_EMPTY = 0.05


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
    model.eval().to(device)
    tokenizer = load_shadow_tokenizer(model_dir)
    return model, tokenizer


def save_model(model: Any, tokenizer: Any, model_dir: str | Path) -> str:
    directory = Path(model_dir)
    directory.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(directory))
    tokenizer.save_pretrained(str(directory))
    return str(directory)


def _model_dir(args: argparse.Namespace) -> Path:
    if args.model_dir:
        return Path(args.model_dir).expanduser()
    return Path(args.out).expanduser() / "model"


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def _gate_summary(
    roundtrip: Mapping[str, Any] | None,
    diagnose: Mapping[str, Any] | None,
    evaluation: Mapping[str, Any] | None,
) -> dict[str, Any]:
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
    gates["accuracy_gate"] = "n/a (needs a native 0.6B baseline report)"
    recovery_keys = [
        key
        for key in ("roundtrip_passed", "eval_format_ok", "eval_truncation_ok", "eval_empty_ok")
        if key in gates
    ]
    gates["recovery_ready"] = bool(recovery_keys) and all(
        bool(gates[key]) for key in recovery_keys
    )
    return gates


def _report_preamble(args: argparse.Namespace, command: str) -> dict[str, Any]:
    return {
        "experiment": EXPERIMENT,
        "issue": ISSUE,
        "command": command,
        "smoke": bool(args.smoke),
        "config": {
            "base_model": _expand(args.base_model),
            "shadow_model": _expand(args.shadow_model),
            "shadow_ckpt": _expand(args.shadow_ckpt),
            "device": args.device,
            "dtype": args.dtype,
            "seed": args.seed,
            "dataset_jsonl": args.dataset_jsonl,
            "system_prompt": args.system_prompt,
        },
    }


def _evaluate(
    args: argparse.Namespace, model: Any, tokenizer: Any
) -> dict[str, Any]:
    from shadow_lab.shadow_only import evaluate_shadow_only

    examples = load_eval_examples(args, tokenizer)
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
        base_model_path=_expand(args.base_model),
        shadow_model_path=_expand(args.shadow_model),
        shadow_ckpt=_expand(args.shadow_ckpt),
        train_scope=args.train_scope,
        lr=args.lr,
        steps=_resolve(args.steps, 2, 200, args.smoke),
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
    result = _evaluate(args, model, tokenizer)
    report = _report_preamble(args, "eval")
    report["generation"] = result["summary"]
    report["gates"] = _gate_summary(None, None, result["summary"])
    report["model_dir"] = str(model_dir) if model_dir.exists() else "in-memory-export"
    _write_json(out / "report.json", report)
    _write_jsonl(out / "generations.jsonl", result["generations"])
    print(
        json.dumps(
            {
                "command": "eval",
                "format_rate": result["summary"]["format_rate"],
                "accuracy": result["summary"]["accuracy"],
                "truncation_rate": result["summary"]["truncation_rate"],
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
    evaluation = _evaluate(args, model, tokenizer)

    report = _report_preamble(args, "run")
    report["roundtrip"] = roundtrip
    report["diagnose"] = diagnose["summary"]
    report["train"] = stats
    report["eval"] = evaluation["summary"]
    report["gates"] = _gate_summary(
        roundtrip, diagnose["summary"], evaluation["summary"]
    )
    report["model_dir"] = model_dir
    _write_json(out / "report.json", report)
    _write_json(out / "train_stats.json", stats)
    _write_jsonl(out / "generations.jsonl", evaluation["generations"])
    _write_jsonl(out / "diagnose_generations.jsonl", diagnose["generations"])
    print(
        json.dumps(
            {
                "command": "run",
                "roundtrip_passed": roundtrip["passed"],
                "diagnose_format_rate": diagnose["summary"]["format_rate"],
                "eval_format_rate": evaluation["summary"]["format_rate"],
                "eval_accuracy": evaluation["summary"]["accuracy"],
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


def _add_eval_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--model-dir", default=None)


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
