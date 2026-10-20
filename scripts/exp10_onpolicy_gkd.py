#!/usr/bin/env python3
"""exp10: GKD-style on-policy flywheel (rollouts -> train -> eval -> report).

Examples::

    # 1) Sample G trajectories from the deployed base_shadow path.
    PYTHONPATH=src python scripts/exp10_onpolicy_gkd.py rollouts \
        --prompts runs/exp10/prompts.jsonl --model Qwen/Qwen3-0.6B \
        --shadow-ckpt runs/exp7/adapter --out runs/exp10/rollouts.jsonl \
        --g 4 --max-new-tokens 512 --seed 0

    # 2) Train the adapter with (1-lambda)*SFT + lambda*on-policy KD.
    PYTHONPATH=src python scripts/exp10_onpolicy_gkd.py train \
        --model Qwen/Qwen3-0.6B --teacher Qwen/Qwen3-4B \
        --shadow-ckpt runs/exp7/adapter --rollouts runs/exp10/rollouts.jsonl \
        --supervised runs/exp7/sft.jsonl --out runs/exp10/adapter \
        --lambda 0.5 --divergence jsd --max-steps 200

    # 3) Evaluate joint and shadow-only ppl plus structured task accuracy.
    PYTHONPATH=src python scripts/exp10_onpolicy_gkd.py eval \
        --model Qwen/Qwen3-0.6B --shadow-ckpt runs/exp10/adapter \
        --heldout runs/exp10/heldout.jsonl --tasks runs/exp10/tasks.jsonl \
        --out runs/exp10/report.json

    # 4) Or chain everything; --smoke keeps it to 1-2 prompts and one step.
    PYTHONPATH=src python scripts/exp10_onpolicy_gkd.py run \
        --prompts runs/exp10/prompts.jsonl --model Qwen/Qwen3-0.6B \
        --shadow-ckpt runs/exp7/adapter --teacher Qwen/Qwen3-4B \
        --heldout runs/exp10/heldout.jsonl --out runs/exp10/smoke --smoke

Prompt/file formats:
- ``--prompts``: JSONL from ``FeedbackStore.export_prompts`` (``prompt`` plus
  optional ``task``/``source``/``confidence``); ``--feedback-store`` exports it.
- ``--rollouts``: JSONL written by the ``rollouts`` command.
- ``--supervised``: JSONL from ``FeedbackStore.export_sft`` (``messages``).
- ``--heldout``: ``.jsonl`` rows with ``text`` (or ``prompt``+``completion``);
  any other suffix is read as one plain-text document.
- ``--tasks``: JSONL rows ``{"task", "expected"}`` with either a ``response``
  to grade or a ``prompt`` to generate first.

``rollouts`` uses ``base_shadow`` by default (the deployment path);
``shadow_only`` is the explicit ablation. ``eval`` always reports both paths,
bootstrap intervals (``--ci-n-boot``/``--ci-seed``) and marks ``no_go`` when
the baseline guard fails. ``train``/``run`` accept ``--token-budget`` (counts
only the loss-participating branch, so λ=0 and λ=1 can share one budget; at
0<λ<1 it keeps at least one row from each branch and alternates, flagging
``token_budget_overflow`` if those guaranteed rows exceed the budget) and
length-bucket rows by default (``--no-sort-by-length`` disables it).

Heavy imports (torch/transformers/trl/datasets) are lazy so ``--help`` stays
fast and the parser can be smoke-tested without loading a model.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

DEFAULT_TEACHER_TEMPERATURE = 1.0


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


def _heldout_texts(path: str | Path) -> list[str]:
    source = Path(path).expanduser()
    if source.suffix.lower() in {".jsonl", ".json", ".ndjson"}:
        texts: list[str] = []
        for row in _read_jsonl(source):
            text = row.get("text") or row.get("content")
            if not text:
                prompt = str(row.get("prompt") or "")
                completion = str(row.get("completion") or "")
                text = f"{prompt}{completion}" if (prompt or completion) else ""
            if str(text).strip():
                texts.append(str(text))
        return texts
    text = source.read_text(encoding="utf-8").strip()
    return [text] if text else []


def _resolve_prompt_path(args: argparse.Namespace, *, default: Path) -> Path:
    """Return the prompt JSONL, exporting ``--feedback-store`` when requested."""
    if getattr(args, "feedback_store", None):
        from edge_cloud.flywheel import FeedbackStore

        target = Path(getattr(args, "prompts_out", None) or default).expanduser()
        result = FeedbackStore(args.feedback_store).export_prompts(target)
        print(
            f"exported {result['exported']} hard-case prompts -> {result['path']}",
            file=sys.stderr,
        )
        return target
    if not getattr(args, "prompts", None):
        raise ValueError("either --prompts or --feedback-store is required")
    return Path(args.prompts).expanduser()


# ---------------------------------------------------------------------------
# Model helpers (lazy imports)
# ---------------------------------------------------------------------------


def _resolve_dtype(name: str):
    import torch

    if name == "auto":
        return torch.bfloat16 if torch.cuda.is_available() else torch.float32
    return getattr(torch, name)


def _resolve_device(name: str | None = None):
    import torch

    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_base_model(path: str, dtype_name: str):
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(
        str(path), dtype=_resolve_dtype(dtype_name)
    )


def _load_shadow_model(args: argparse.Namespace, *, is_trainable: bool):
    """Load the student (base + shadow adapter) and its tokenizer."""
    from transformers import AutoTokenizer

    from shadow_peft import ShadowConfig, ShadowForCausalLM, get_shadow_model

    base = _load_base_model(args.model, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model))
    shadow_model = (
        _load_base_model(args.shadow_model_path, args.dtype)
        if args.shadow_model_path
        else None
    )
    if args.shadow_ckpt:
        model = ShadowForCausalLM.from_pretrained(
            base,
            args.shadow_ckpt,
            is_trainable=is_trainable,
            shadow_model=shadow_model,
        )
    else:
        peft = get_shadow_model(base, ShadowConfig(), shadow_model=shadow_model)
        model = ShadowForCausalLM(peft, inference_mode="base_shadow")
    if not tokenizer.pad_token_id and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    _add_trainer_compat(model)
    return model, tokenizer


def _add_trainer_compat(model: Any) -> None:
    """Bridge the PreTrainedModel methods trl's SFTTrainer assumes.

    ``ShadowForCausalLM`` is a plain ``nn.Module``; trl 1.10 still calls
    ``add_model_tags`` at init and probes ``get_output_embeddings``. The tags
    are cosmetic and the embeddings are only needed by trainer internals this
    CLI does not use (``loss_type="nll"``), so a no-op shim is enough.
    """
    if not hasattr(model, "add_model_tags"):
        model.add_model_tags = lambda *args, **kwargs: None
    if not hasattr(model, "get_output_embeddings"):
        model.get_output_embeddings = lambda: getattr(model, "lm_head", None)
    if not hasattr(model, "get_input_embeddings"):
        model.get_input_embeddings = lambda: model.peft_model.base_model.get_input_embeddings()
    if not hasattr(model, "warnings_issued"):
        model.warnings_issued = {}


def _token_ids(tokenizer: Any, encoded: Any) -> list[int]:
    del tokenizer
    from shadow_lab.onpolicy import normalize_token_ids

    return normalize_token_ids(encoded)


def _encode_prompt(tokenizer: Any, prompt: str) -> list[int]:
    if getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": prompt}]
        try:
            encoded = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            try:
                encoded = tokenizer.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True
                )
            except TypeError:
                encoded = tokenizer.apply_chat_template(messages, tokenize=True)
        return _token_ids(tokenizer, encoded)
    return _token_ids(tokenizer, tokenizer(prompt, add_special_tokens=True))


def _generate_responses(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    *,
    max_new_tokens: int,
) -> list[str]:
    """Greedy ``base_shadow`` generation for task prompts without responses."""
    import torch

    if not prompts:
        return []
    was_training = bool(getattr(model, "training", False))
    previous_mode = getattr(model, "inference_mode", "base_shadow")
    device = getattr(model, "device", None) or next(model.parameters()).device
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    model.eval()
    if hasattr(model, "set_inference_mode"):
        model.set_inference_mode("base_shadow")
    responses: list[str] = []
    try:
        with torch.no_grad():
            for prompt in prompts:
                prompt_ids = _encode_prompt(tokenizer, prompt)
                input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
                generated = model.generate(
                    input_ids=input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=pad_id,
                    use_cache=False,
                )
                if hasattr(generated, "sequences"):
                    generated = generated.sequences
                responses.append(
                    tokenizer.decode(
                        generated[0, len(prompt_ids) :].tolist(),
                        skip_special_tokens=True,
                    )
                )
    finally:
        if hasattr(model, "set_inference_mode"):
            model.set_inference_mode(previous_mode)
        if hasattr(model, "train"):
            model.train(was_training)
    return responses


def _record_has_response(record: dict[str, Any]) -> bool:
    return any(
        record.get(key) is not None
        for key in ("response", "text", "output", "prediction", "answer")
    )


def _fill_task_responses(
    records: list[dict[str, Any]], responses: list[str]
) -> list[dict[str, Any]]:
    """Attach generated responses to the task records that needed one."""
    for record, response in zip(records, responses, strict=True):
        record["response"] = response
    return records


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_rollouts(args: argparse.Namespace) -> dict[str, Any]:
    """Sample rollout trajectories and write them as JSONL."""
    from shadow_lab.onpolicy import (
        RolloutConfig,
        generate_rollouts,
        load_prompts,
        write_rollouts,
    )

    output = Path(args.out).expanduser()
    prompt_path = _resolve_prompt_path(
        args, default=output.with_name(output.name + ".prompts.jsonl")
    )
    prompts = load_prompts(prompt_path)
    if getattr(args, "limit", None):
        prompts = prompts[: args.limit]
    if not prompts:
        raise ValueError(f"no usable prompts in {prompt_path}")

    model, tokenizer = _load_shadow_model(args, is_trainable=False)
    device = _resolve_device(args.device)
    model.to(device)
    config = RolloutConfig(
        rollout_mode=args.rollout_mode,
        num_rollouts=args.num_rollouts,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        max_prompt_tokens=args.max_prompt_tokens,
        max_rollout_tokens=args.max_rollout_tokens,
        adapter_version=args.adapter_version or Path(args.shadow_ckpt or "").name,
        val_ratio=args.val_ratio,
    )
    started = time.perf_counter()
    rows = generate_rollouts(model, tokenizer, prompts, config)
    wall_clock = time.perf_counter() - started
    write_rollouts(rows, output)

    summary = {
        "out": str(output),
        "prompts": str(prompt_path),
        "rollout_mode": args.rollout_mode,
        "num_rollouts": args.num_rollouts,
        "seed": args.seed,
        "rows": len(rows),
        "wall_clock": wall_clock,
        "stats": rows.stats,
    }
    _write_json(output.with_name(output.name + ".stats.json"), summary)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return summary


def cmd_train(args: argparse.Namespace) -> dict[str, Any]:
    """Train with ``(1-lambda)*CE_supervised + lambda*KD_onpolicy``."""
    import torch
    from datasets import Dataset as HFDataset
    from trl import SFTConfig

    from shadow_lab.onpolicy import bucket_rows_by_length, load_rollouts
    from shadow_lab.onpolicy_eval import (
        OnPolicyTrainDataset,
        collate_training_rows,
        count_training_tokens,
        truncate_training_rows_by_token_budget,
    )
    from shadow_lab.trainers import OnPolicyGKDTrainer, validate_teacher_vocab

    model, tokenizer = _load_shadow_model(args, is_trainable=True)
    teacher = _load_base_model(args.teacher, args.dtype)
    validate_teacher_vocab(model, teacher)
    device = _resolve_device(args.device)
    model.to(device)
    teacher.to(device)
    teacher.requires_grad_(False)
    teacher.eval()

    rollout_rows = load_rollouts(args.rollouts)
    train_rows = [row for row in rollout_rows if row.get("split", "train") == "train"]
    if not train_rows:
        raise ValueError(f"no train-split rollouts in {args.rollouts}")
    if getattr(args, "smoke", False):
        train_rows = train_rows[:2]
        if not train_rows:
            raise ValueError("--smoke left no training rows")
    supervised_rows = _read_jsonl(args.supervised) if args.supervised else []
    if getattr(args, "smoke", False):
        supervised_rows = supervised_rows[:1]

    dataset = OnPolicyTrainDataset(
        tokenizer,
        rollout_rows=train_rows,
        supervised_rows=supervised_rows,
        max_seq_len=args.max_seq_len,
        # Apply the token budget in source order first; bucketing reorders
        # the selected rows below without changing the budget accounting.
        sort_by_length=False,
    )
    selection = truncate_training_rows_by_token_budget(
        dataset.to_list(),
        args.lambda_onpolicy,
        getattr(args, "token_budget", None),
    )
    records = list(selection)
    if getattr(args, "sort_by_length", True):
        records = [
            row
            for bucket in bucket_rows_by_length(records, max(1, args.batch_size))
            for row in bucket
        ]
    token_counts = count_training_tokens(records, args.lambda_onpolicy)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    def collator(rows):
        return collate_training_rows(rows, pad_id)

    use_bf16 = _resolve_dtype(args.dtype) == torch.bfloat16 and device.type == "cuda"
    training_args = SFTConfig(
        output_dir=str(Path(args.out).expanduser() / "trainer"),
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=max(1, args.logging_steps),
        save_strategy="no",
        report_to=[],
        remove_unused_columns=False,
        max_length=None,
        packing=False,
        # trl 1.10 defaults to chunked_nll, which patches get_output_embeddings();
        # ShadowForCausalLM does not expose it, and OnPolicyGKDTrainer computes
        # the loss itself, so keep the plain NLL path.
        loss_type="nll",
        dataset_kwargs={"skip_prepare_dataset": True},
        seed=args.seed,
        data_seed=args.seed,
        bf16=use_bf16,
    )
    trainer = OnPolicyGKDTrainer(
        model=model,
        args=training_args,
        train_dataset=HFDataset.from_list(records),
        data_collator=collator,
        processing_class=tokenizer,
        teacher_model=teacher,
        lambda_onpolicy=args.lambda_onpolicy,
        divergence=args.divergence,
        topk=args.topk,
        shadow_aux_weight=args.shadow_aux_weight,
        teacher_temperature=args.teacher_temperature,
        shadow_lr_multiplier=args.shadow_lr_multiplier,
    )
    started = time.perf_counter()
    result = trainer.train()
    wall_clock = time.perf_counter() - started

    output = Path(args.out).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output)
    summary = {
        "out": str(output),
        "steps": int(result.global_step),
        "rows": len(records),
        "onpolicy_rows": sum(1 for row in records if row["source"] == "onpolicy"),
        "supervised_rows": sum(1 for row in records if row["source"] == "supervised"),
        "tokens_seen": token_counts["tokens_seen"],
        "tokens_supervised": token_counts["tokens_supervised"],
        "tokens_onpolicy": token_counts["tokens_onpolicy"],
        "token_budget": getattr(args, "token_budget", None),
        "token_budget_overflow": bool(selection.overflow),
        "sort_by_length": bool(getattr(args, "sort_by_length", True)),
        "wall_clock": wall_clock,
        "lambda": args.lambda_onpolicy,
        "divergence": args.divergence,
        "topk": args.topk,
        "teacher_support": "topk" if args.topk else "full",
        "shadow_aux_weight": args.shadow_aux_weight,
        "teacher": str(args.teacher),
        "discard_rate": getattr(args, "discard_rate", 0.0),
    }
    _write_json(output / "train_stats.json", summary)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return summary


def cmd_eval(args: argparse.Namespace) -> dict[str, Any]:
    """Evaluate held-out ppl (joint + shadow) and structured task accuracy."""
    from shadow_lab.onpolicy_eval import (
        build_report,
        evaluate_ppl,
        task_accuracy,
        task_outcomes,
    )

    model, tokenizer = _load_shadow_model(args, is_trainable=False)
    model.to(_resolve_device(args.device))
    texts = _heldout_texts(args.heldout)
    tasks = _read_jsonl(args.tasks) if args.tasks else []
    if not texts and not tasks:
        raise ValueError("eval needs at least one held-out text or task record")

    started = time.perf_counter()
    pending_records = [
        record
        for record in tasks
        if not _record_has_response(record) and record.get("prompt")
    ]
    pending = [str(record["prompt"]) for record in pending_records]
    responses = _generate_responses(
        model, tokenizer, pending, max_new_tokens=args.generation_max_new_tokens
    )
    _fill_task_responses(pending_records, responses)

    if texts:
        ppl = evaluate_ppl(
            model,
            tokenizer,
            texts,
            seq_len=args.seq_len,
            max_windows=args.max_windows,
        )
    else:
        raise ValueError("--heldout is required to report joint/shadow perplexity")
    accuracy = task_accuracy(tasks)
    outcomes = task_outcomes(tasks)
    wall_clock = float(getattr(args, "wall_clock_base", 0.0) or 0.0) + (
        time.perf_counter() - started
    )

    baseline = _read_json(args.baseline_sft) if args.baseline_sft else None
    if getattr(args, "prompts", None):
        from shadow_lab.onpolicy import load_prompts

        fingerprint_prompts = [
            str(row.get("prompt") or "") for row in load_prompts(args.prompts)
        ]
    else:
        fingerprint_prompts = list(texts) + [
            str(record.get("prompt") or "")
            for record in tasks
            if record.get("prompt")
        ]
    report = build_report(
        joint_ppl=ppl["joint_ppl"],
        shadow_ppl=ppl["shadow_ppl"],
        task_accuracy=accuracy,
        rollout_mode=args.rollout_mode,
        lambda_onpolicy=args.lambda_onpolicy,
        divergence=args.divergence,
        teacher_support=args.teacher_support,
        tokens_seen=args.tokens_seen,
        tokens_supervised=getattr(args, "tokens_supervised", None),
        tokens_onpolicy=getattr(args, "tokens_onpolicy", None),
        token_budget=getattr(args, "token_budget", None),
        token_budget_overflow=bool(getattr(args, "token_budget_overflow", False)),
        wall_clock=wall_clock,
        discard_rate=args.discard_rate,
        prompts=fingerprint_prompts,
        teacher_version=args.teacher_version,
        baseline_sft=baseline,
        joint_losses=ppl["joint_losses"],
        shadow_losses=ppl["shadow_losses"],
        window_tokens=ppl["window_tokens"],
        task_outcomes=outcomes,
        n_boot=getattr(args, "ci_n_boot", 200),
        ci_seed=getattr(args, "ci_seed", 0),
    )
    _write_json(args.out, report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return report


def cmd_run(args: argparse.Namespace) -> dict[str, Any]:
    """Run rollouts -> train -> eval under one output directory."""
    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    prompts_out = out_dir / "prompts.jsonl"
    rollouts_out = out_dir / "rollouts.jsonl"
    adapter_out = out_dir / "adapter"
    report_out = out_dir / "report.json"

    rollout_args = argparse.Namespace(**vars(args))
    rollout_args.out = str(rollouts_out)
    rollout_args.prompts_out = str(prompts_out)
    if getattr(args, "feedback_store", None) and not args.prompts:
        rollout_args.prompts = None
    rollout_summary = cmd_rollouts(rollout_args)

    train_args = argparse.Namespace(**vars(args))
    train_args.rollouts = str(rollouts_out)
    train_args.out = str(adapter_out)
    train_args.discard_rate = rollout_summary["stats"].get("discard_rate", 0.0)
    train_summary = cmd_train(train_args)

    eval_args = argparse.Namespace(**vars(args))
    eval_args.shadow_ckpt = str(adapter_out)
    eval_args.out = str(report_out)
    eval_args.tokens_seen = train_summary["tokens_seen"]
    eval_args.tokens_supervised = train_summary["tokens_supervised"]
    eval_args.tokens_onpolicy = train_summary["tokens_onpolicy"]
    eval_args.token_budget = train_summary["token_budget"]
    eval_args.token_budget_overflow = train_summary["token_budget_overflow"]
    eval_args.discard_rate = train_args.discard_rate
    eval_args.teacher_support = "topk" if args.topk else "full"
    eval_args.wall_clock_base = (
        rollout_summary["wall_clock"] + train_summary["wall_clock"]
    )
    if getattr(args, "feedback_store", None) and not args.prompts:
        eval_args.prompts = str(prompts_out)
    if not eval_args.teacher_version:
        eval_args.teacher_version = str(args.teacher)
    report = cmd_eval(eval_args)

    summary = {
        "out": str(out_dir),
        "rollouts": rollout_summary["rows"],
        "tokens_seen": train_summary["tokens_seen"],
        "tokens_supervised": train_summary["tokens_supervised"],
        "tokens_onpolicy": train_summary["tokens_onpolicy"],
        "token_budget": train_summary["token_budget"],
        "token_budget_overflow": train_summary["token_budget_overflow"],
        "report": str(report_out),
        "no_go": report["no_go"],
        "claims_gain": report["claims_gain"],
    }
    _write_json(out_dir / "run_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return summary


def _read_json(path: str | Path) -> Any:
    return json.loads(Path(path).expanduser().read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def _add(parser: argparse.ArgumentParser, *flags: str, **kwargs: Any) -> None:
    """Add an option unless a shared command group already registered it."""
    if any(flag in parser._option_string_actions for flag in flags):
        return
    parser.add_argument(*flags, **kwargs)


def _add_rollout_args(parser: argparse.ArgumentParser, *, smoke: bool = False) -> None:
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--prompts", help="JSONL prompt set (FeedbackStore.export_prompts)"
    )
    source.add_argument(
        "--feedback-store",
        help="FeedbackStore JSONL; hard-case prompts are exported before sampling",
    )
    _add(
        parser,
        "--out",
        required=True,
        help="output path (rollout JSONL; `run` treats it as a directory)",
    )
    _add(parser, "--model", required=True, help="base model path or HF id")
    _add(
        parser,
        "--shadow-ckpt",
        required=True,
        help="ShadowPEFT adapter directory",
    )
    _add(parser, "--shadow-model-path", help="explicit shadow backbone path")
    _add(
        parser,
        "--rollout-mode",
        choices=("base_shadow", "shadow_only"),
        default="base_shadow",
        help="deployment path by default; shadow_only is the ablation",
    )
    _add(
        parser,
        "--g",
        dest="num_rollouts",
        type=int,
        default=4,
        help="rollouts per prompt",
    )
    _add(parser, "--max-new-tokens", type=int, default=64)
    _add(parser, "--temperature", type=float, default=1.0)
    _add(parser, "--top-p", type=float, default=1.0)
    _add(parser, "--max-prompt-tokens", type=int, default=None)
    _add(parser, "--max-rollout-tokens", type=int, default=None)
    _add(parser, "--val-ratio", type=float, default=0.1)
    _add(parser, "--adapter-version", default="")
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
        _add(
            parser, "--smoke", action="store_true", help="tiny end-to-end run"
        )


def _add_train_args(
    parser: argparse.ArgumentParser, *, require_rollouts: bool = True
) -> None:
    _add(parser, "--out", required=True, help="adapter output directory")
    _add(parser, "--model", required=True, help="base model path or HF id")
    _add(
        parser,
        "--shadow-ckpt",
        default=None,
        help="existing adapter to continue from",
    )
    _add(parser, "--shadow-model-path", help="explicit shadow backbone path")
    _add(parser, "--teacher", required=True, help="same-vocabulary teacher model")
    if require_rollouts:
        _add(parser, "--rollouts", required=True, help="rollout JSONL")
    _add(parser, "--supervised", default=None, help="optional SFT JSONL")
    _add(
        parser,
        "--lambda",
        dest="lambda_onpolicy",
        type=float,
        default=0.5,
        help="weight of the on-policy KD term (0=SFT, 1=pure GKD)",
    )
    _add(
        parser,
        "--divergence",
        choices=("forward_kl", "reverse_kl", "jsd"),
        default="forward_kl",
    )
    _add(parser, "--topk", type=int, default=None)
    _add(parser, "--shadow-aux-weight", type=float, default=0.0)
    _add(
        parser,
        "--teacher-temperature",
        type=float,
        default=DEFAULT_TEACHER_TEMPERATURE,
    )
    _add(parser, "--shadow-lr-multiplier", type=float, default=1.0)
    _add(parser, "--max-steps", type=int, default=100)
    _add(parser, "--batch-size", type=int, default=1)
    _add(parser, "--grad-accum", type=int, default=1)
    _add(parser, "--lr", type=float, default=1e-4)
    _add(parser, "--max-seq-len", type=int, default=512)
    _add(parser, "--logging-steps", type=int, default=1)
    _add(parser, "--seed", type=int, default=0)
    _add(parser, "--discard-rate", type=float, default=0.0)
    _add(
        parser,
        "--token-budget",
        type=int,
        default=None,
        help=(
            "cap on loss-participating tokens (whole rows, active branch only); "
            "mixed λ keeps one row per branch and may record "
            "token_budget_overflow; use the same value for the λ=0 SFT baseline"
        ),
    )
    _add(
        parser,
        "--sort-by-length",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="length-bucket the selected rows before batching (default: on)",
    )
    _add(
        parser,
        "--dtype",
        choices=("auto", "float32", "bfloat16"),
        default="auto",
    )
    _add(parser, "--device", default=None)


def _add_eval_args(parser: argparse.ArgumentParser) -> None:
    _add(parser, "--out", required=True, help="report JSON output path")
    _add(parser, "--model", required=True, help="base model path or HF id")
    _add(parser, "--shadow-ckpt", required=True, help="adapter to evaluate")
    _add(parser, "--shadow-model-path", help="explicit shadow backbone path")
    _add(
        parser,
        "--heldout",
        required=True,
        help="held-out text JSONL ({'text'}) or a plain-text document",
    )
    _add(parser, "--tasks", default=None, help="structured task JSONL")
    _add(
        parser,
        "--prompts",
        default=None,
        help="training prompt JSONL; used for the data fingerprint when given",
    )
    _add(
        parser,
        "--rollout-mode",
        choices=("base_shadow", "shadow_only"),
        default="base_shadow",
    )
    _add(parser, "--seq-len", type=int, default=2048)
    _add(parser, "--max-windows", type=int, default=8)
    _add(parser, "--lambda", dest="lambda_onpolicy", type=float, default=0.5)
    _add(
        parser,
        "--divergence",
        choices=("forward_kl", "reverse_kl", "jsd"),
        default="forward_kl",
    )
    _add(
        parser, "--teacher-support", choices=("full", "topk"), default="full"
    )
    _add(parser, "--teacher-version", default="")
    _add(parser, "--tokens-seen", type=int, default=0)
    _add(parser, "--tokens-supervised", type=int, default=None)
    _add(parser, "--tokens-onpolicy", type=int, default=None)
    _add(
        parser,
        "--token-budget",
        type=int,
        default=None,
        help="training token budget recorded in the report",
    )
    _add(
        parser,
        "--token-budget-overflow",
        action="store_true",
        help=(
            "record that the mixed-branch budget kept one row per branch and "
            "tokens_seen exceeded token_budget"
        ),
    )
    _add(parser, "--ci-n-boot", type=int, default=200)
    _add(parser, "--ci-seed", type=int, default=0)
    _add(parser, "--discard-rate", type=float, default=0.0)
    _add(
        parser,
        "--baseline-sft",
        default=None,
        help="JSON with joint_ppl/shadow_ppl from the same-budget SFT baseline",
    )
    _add(parser, "--generation-max-new-tokens", type=int, default=64)
    _add(
        parser,
        "--dtype",
        choices=("auto", "float32", "bfloat16"),
        default="auto",
    )
    _add(parser, "--device", default=None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="exp10_onpolicy_gkd.py",
        description=(
            "GKD-style on-policy flywheel: rollouts, OnPolicyGKDTrainer, paired "
            "joint/shadow evaluation and an auditable JSON report."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    rollouts = subparsers.add_parser(
        "rollouts", help="sample trajectories from the deployed path"
    )
    _add_rollout_args(rollouts)

    train = subparsers.add_parser(
        "train", help="train the shadow adapter with on-policy distillation"
    )
    _add_train_args(train)

    evaluation = subparsers.add_parser(
        "eval", help="report joint/shadow ppl and structured task accuracy"
    )
    _add_eval_args(evaluation)

    run = subparsers.add_parser("run", help="chain rollouts -> train -> eval")
    _add_rollout_args(run, smoke=True)
    _add_train_args(run, require_rollouts=False)
    _add_eval_args(run)
    return parser


def _apply_smoke(args: argparse.Namespace) -> None:
    if not getattr(args, "smoke", False):
        return
    args.limit = min(2, args.limit) if args.limit else 2
    args.num_rollouts = 1
    args.max_new_tokens = min(args.max_new_tokens, 8)
    args.max_steps = 1
    args.max_windows = 1
    args.seq_len = min(args.seq_len, 64)
    args.generation_max_new_tokens = min(args.generation_max_new_tokens, 8)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _apply_smoke(args)
    handlers = {
        "rollouts": cmd_rollouts,
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
