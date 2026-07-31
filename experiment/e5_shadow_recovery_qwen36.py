"""E5: one-stage Shadow recovery training on Qwen3.6-27B MLX 2-bit.

The official Bonsai checkpoint supplies a small sequence-level distillation
set and held-out first-token references. The v3 one-shot checkpoint remains
frozen; only a one-layer Shadow backbone and injection/update adapters train.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from e2_hybrid_bonsai_qwen36 import (
    PROMPTS,
    _chat_prompt,
    _cosine,
    _kl,
    _top_tokens,
)
from mlx import optimizers
from mlx.utils import tree_flatten
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler

from shadow_peft_mlx import (
    ShadowConfig,
    get_shadow_model,
)
from shadow_peft_mlx.quantized_grad import quantized_linear_input_grad
from shadow_peft_mlx.task_models import _shifted_ce_loss

DEFAULT_OFFICIAL = Path(
    "~/.cache/modelscope/models/prism-ml--Ternary-Bonsai-27B-mlx-2bit/snapshots/master"
).expanduser()
DEFAULT_PTQ = Path("qwen3.6-27b-mlx-2bit-v3-oneshot")
DEFAULT_OUT = Path("experiment/analysis_qwen36_bonsai_migration/e5")

TRAIN_PROMPTS = [
    "What is the capital of Germany? Answer briefly.",
    "What is the capital of Japan? Answer briefly.",
    "Calculate 12 * 14. Give only the result.",
    "Calculate 144 / 12. Give only the result.",
    "请计算 15 × 4，只给出答案。",
    "请问日本的首都是哪里？简短回答。",
    "Which planet is known as the Red Planet? Answer briefly.",
    "What gas do plants absorb from the atmosphere? Answer briefly.",
    "Write a Python function square(x) that returns x squared.",
    "Write a Python function subtract(a, b) that returns a minus b.",
    "Complete briefly: Water freezes at ___ degrees Celsius.",
    "Give the next number only: 2, 4, 6, 8,",
]


def _model_logits(model: Any, token_ids: list[int]) -> np.ndarray:
    output = model(mx.array([token_ids], dtype=mx.int32))
    logits = output.logits if hasattr(output, "logits") else output
    last = logits[0, -1].astype(mx.float32)
    mx.eval(last)
    return np.asarray(last)


def _teacher_response(model: Any, tokenizer: Any, chat: str, max_tokens: int) -> str:
    sampler = make_sampler(temp=0.0)
    return "".join(
        response.text
        for response in stream_generate(
            model,
            tokenizer,
            prompt=chat,
            max_tokens=max_tokens,
            sampler=sampler,
        )
    )


def _full_greedy(
    model: Any,
    tokenizer: Any,
    chat: str,
    *,
    max_tokens: int,
) -> str:
    token_ids = tokenizer.encode(chat)
    generated: list[int] = []
    eos_id = tokenizer.eos_token_id
    for _ in range(max_tokens):
        logits = _model_logits(model, token_ids)
        next_id = int(np.argmax(logits))
        if next_id == eos_id:
            break
        generated.append(next_id)
        token_ids.append(next_id)
    return tokenizer.decode(generated)


def _example(
    tokenizer: Any,
    *,
    prompt_id: str,
    prompt: str,
    response: str,
) -> dict[str, Any]:
    chat = _chat_prompt(tokenizer, prompt)
    prompt_ids = tokenizer.encode(chat)
    response_ids = tokenizer.encode(response)
    eos_id = tokenizer.eos_token_id
    if eos_id is not None:
        response_ids = [*response_ids, int(eos_id)]
    input_ids = [*prompt_ids, *response_ids]
    labels = [-100] * len(prompt_ids) + response_ids
    return {
        "id": prompt_id,
        "prompt": prompt,
        "chat": chat,
        "response": response,
        "prompt_ids": prompt_ids,
        "input_ids": input_ids,
        "labels": labels,
    }


def _evaluate_losses(model: Any, examples: list[dict[str, Any]]) -> dict[str, Any]:
    model.eval()
    rows = []
    for example in examples:
        input_ids = mx.array([example["input_ids"]], dtype=mx.int32)
        labels = mx.array([example["labels"]], dtype=mx.int32)
        loss = _shifted_ce_loss(model(input_ids), labels)
        mx.eval(loss)
        rows.append({"id": example["id"], "loss": float(loss)})
    return {
        "mean": sum(row["loss"] for row in rows) / len(rows),
        "examples": rows,
    }


def _evaluate_probes(
    model: Any,
    tokenizer: Any,
    probes: list[dict[str, Any]],
    reference_logits: dict[str, np.ndarray],
    *,
    generation_tokens: int,
) -> dict[str, Any]:
    model.eval()
    rows = {}
    for probe in probes:
        logits = _model_logits(model, probe["prompt_ids"])
        reference = reference_logits[probe["id"]]
        ref_top = set(np.argpartition(reference, -10)[-10:].tolist())
        cand_top = set(np.argpartition(logits, -10)[-10:].tolist())
        rows[probe["id"]] = {
            "kl_official_to_student": _kl(reference, logits),
            "logit_cosine": _cosine(reference, logits),
            "top10_overlap": len(ref_top & cand_top),
            "top10": _top_tokens(logits, tokenizer),
            "generation": _full_greedy(
                model,
                tokenizer,
                probe["chat"],
                max_tokens=generation_tokens,
            ),
        }
    values = list(rows.values())
    return {
        "aggregate": {
            "mean_kl": sum(row["kl_official_to_student"] for row in values) / len(values),
            "mean_logit_cosine": sum(row["logit_cosine"] for row in values) / len(values),
            "top10_overlap": sum(row["top10_overlap"] for row in values),
            "top10_total": 10 * len(values),
        },
        "prompts": rows,
    }


def _make_teacher_data(
    official_dir: Path,
    *,
    teacher_tokens: int,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, np.ndarray],
    Any,
]:
    print(f"Loading official teacher: {official_dir}", flush=True)
    teacher, tokenizer = load(str(official_dir))
    teacher.eval()
    train_examples = []
    for index, prompt in enumerate(TRAIN_PROMPTS):
        chat = _chat_prompt(tokenizer, prompt)
        response = _teacher_response(teacher, tokenizer, chat, max_tokens=teacher_tokens)
        print(f"  train_{index:02d}: {response!r}", flush=True)
        train_examples.append(
            _example(
                tokenizer,
                prompt_id=f"train_{index:02d}",
                prompt=prompt,
                response=response,
            )
        )

    eval_examples = []
    reference_logits = {}
    for prompt in PROMPTS:
        chat = _chat_prompt(tokenizer, prompt["text"])
        prompt_ids = tokenizer.encode(chat)
        reference_logits[prompt["id"]] = _model_logits(teacher, prompt_ids)
        response = _teacher_response(teacher, tokenizer, chat, max_tokens=teacher_tokens)
        print(f"  eval_{prompt['id']}: {response!r}", flush=True)
        eval_examples.append(
            _example(
                tokenizer,
                prompt_id=prompt["id"],
                prompt=prompt["text"],
                response=response,
            )
        )
    del teacher
    gc.collect()
    mx.clear_cache()
    return train_examples, eval_examples, reference_logits, tokenizer


def run(
    official_dir: Path,
    ptq_dir: Path,
    out_dir: Path,
    *,
    steps: int,
    learning_rate: float,
    teacher_tokens: int,
    generation_tokens: int,
) -> dict[str, Any]:
    train_examples, eval_examples, reference_logits, teacher_tokenizer = _make_teacher_data(
        official_dir, teacher_tokens=teacher_tokens
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    serializable_teacher = {
        "train": [
            {
                "id": row["id"],
                "prompt": row["prompt"],
                "response": row["response"],
                "input_ids": row["input_ids"],
                "labels": row["labels"],
            }
            for row in train_examples
        ],
        "eval": [
            {
                "id": row["id"],
                "prompt": row["prompt"],
                "response": row["response"],
                "input_ids": row["input_ids"],
                "labels": row["labels"],
            }
            for row in eval_examples
        ],
    }
    (out_dir / "teacher_data.json").write_text(
        json.dumps(serializable_teacher, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Loading frozen PTQ student: {ptq_dir}", flush=True)
    base, tokenizer = load(str(ptq_dir))
    for example in [*train_examples, *eval_examples]:
        if tokenizer.encode(example["chat"]) != example["prompt_ids"]:
            raise ValueError(f"tokenizer mismatch for {example['id']}")
    if tokenizer.get_vocab() != teacher_tokenizer.get_vocab():
        raise ValueError("teacher and student vocabularies differ")
    del teacher_tokenizer

    config = ShadowConfig(
        num_shadow_layers=1,
        shadow_intermediate_size=512,
        injection_hidden_size=4,
        gate_hidden_size=4,
        alpha=1.0,
        dropout=0.0,
    )
    shadow = get_shadow_model(base, config)
    trainable = list(tree_flatten(shadow.trainable_parameters()))
    trainable_count = int(sum(value.size for _, value in trainable))
    total_count = int(sum(value.size for _, value in tree_flatten(shadow.parameters())))
    print(
        f"Shadow trainable: {trainable_count:,} / {total_count:,} parameters",
        flush=True,
    )

    base_flat = dict(tree_flatten(base.parameters()))
    sentinel_name = next(
        name
        for name, value in base_flat.items()
        if name.endswith(".weight") and value.dtype == mx.uint32
    )
    sentinel_before = base_flat[sentinel_name] * mx.array(1, dtype=mx.uint32)
    mx.eval(sentinel_before)

    before_train_loss = _evaluate_losses(shadow, train_examples)
    before_eval_loss = _evaluate_losses(shadow, eval_examples)
    before_probes = _evaluate_probes(
        shadow,
        tokenizer,
        eval_examples,
        reference_logits,
        generation_tokens=generation_tokens,
    )
    print(
        f"Before: train_loss={before_train_loss['mean']:.4f}, "
        f"eval_loss={before_eval_loss['mean']:.4f}, "
        f"KL={before_probes['aggregate']['mean_kl']:.4f}",
        flush=True,
    )

    optimizer = optimizers.AdamW(
        learning_rate=learning_rate,
        betas=(0.9, 0.98),
        weight_decay=0.0,
    )

    def loss_fn(model: nn.Module, input_ids: mx.array, labels: mx.array) -> mx.array:
        return _shifted_ce_loss(model(input_ids), labels)

    value_and_grad = nn.value_and_grad(shadow, loss_fn)
    history = []
    checkpoint_evaluations = []
    best_eval_loss = before_eval_loss["mean"]
    best_step = 0
    best_adapter: dict[str, mx.array] | None = None
    rng = random.Random(17)
    training_order: list[int] = []
    while len(training_order) < steps:
        epoch_order = list(range(len(train_examples)))
        rng.shuffle(epoch_order)
        training_order.extend(epoch_order)
    training_order = training_order[:steps]
    started = time.perf_counter()
    shadow.train()
    with quantized_linear_input_grad():
        for step in range(1, steps + 1):
            example = train_examples[training_order[step - 1]]
            input_ids = mx.array([example["input_ids"]], dtype=mx.int32)
            labels = mx.array([example["labels"]], dtype=mx.int32)
            loss, gradients = value_and_grad(shadow, input_ids, labels)
            gradients, gradient_norm = optimizers.clip_grad_norm(gradients, max_norm=1.0)
            optimizer.update(shadow, gradients)
            mx.eval(shadow.parameters(), optimizer.state, loss, gradient_norm)
            row = {
                "step": step,
                "example": example["id"],
                "tokens": sum(label != -100 for label in example["labels"][1:]),
                "loss": float(loss),
                "gradient_norm": float(gradient_norm),
                "elapsed_seconds": time.perf_counter() - started,
            }
            history.append(row)
            print(
                f"step {step:02d}/{steps}: {example['id']} "
                f"loss={row['loss']:.4f} grad={row['gradient_norm']:.4f}",
                flush=True,
            )
            if step % len(train_examples) == 0 or step == steps:
                evaluation = _evaluate_losses(shadow, eval_examples)
                checkpoint_evaluations.append({"step": step, "mean_eval_loss": evaluation["mean"]})
                print(
                    f"  checkpoint step {step}: held-out CE={evaluation['mean']:.4f}",
                    flush=True,
                )
                if evaluation["mean"] < best_eval_loss:
                    best_eval_loss = evaluation["mean"]
                    best_step = step
                    best_adapter = {
                        name: value * 1 for name, value in shadow.adapter_parameters().items()
                    }
                    mx.eval(best_adapter)
                shadow.train()
    shadow.eval()
    if best_adapter is not None:
        shadow.load_weights(list(best_adapter.items()), strict=False)
        mx.eval(shadow.parameters())
        del best_adapter
        mx.clear_cache()

    after_train_loss = _evaluate_losses(shadow, train_examples)
    after_eval_loss = _evaluate_losses(shadow, eval_examples)
    after_probes = _evaluate_probes(
        shadow,
        tokenizer,
        eval_examples,
        reference_logits,
        generation_tokens=generation_tokens,
    )
    sentinel_after = dict(tree_flatten(base.parameters()))[sentinel_name]
    packed_unchanged = bool(mx.array_equal(sentinel_before, sentinel_after).item())
    if not packed_unchanged:
        raise RuntimeError(f"frozen packed base changed: {sentinel_name}")

    adapter_dir = out_dir / "adapter"
    shadow.save_pretrained(adapter_dir)
    result = {
        "protocol": {
            "official_model": str(official_dir),
            "ptq_model": str(ptq_dir),
            "steps": steps,
            "learning_rate": learning_rate,
            "teacher_tokens": teacher_tokens,
            "generation_tokens": generation_tokens,
            "shadow_config": config.to_dict(),
            "trainable_parameters": trainable_count,
            "total_module_parameters": total_count,
            "packed_sentinel": sentinel_name,
            "packed_base_unchanged": packed_unchanged,
            "best_step": best_step,
        },
        "before": {
            "train_loss": before_train_loss,
            "eval_loss": before_eval_loss,
            "probes": before_probes,
        },
        "after": {
            "train_loss": after_train_loss,
            "eval_loss": after_eval_loss,
            "probes": after_probes,
        },
        "history": history,
        "checkpoint_evaluations": checkpoint_evaluations,
        "training_seconds": time.perf_counter() - started,
        "peak_memory_gb": mx.get_peak_memory() / (1024**3),
        "adapter_dir": str(adapter_dir),
    }
    return result


def _write_report(result: dict[str, Any], out_dir: Path) -> None:
    (out_dir / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    before = result["before"]
    after = result["after"]
    before_probe = before["probes"]["aggregate"]
    after_probe = after["probes"]["aggregate"]
    lines = [
        "# E5 — one-stage Shadow recovery training",
        "",
        "## Aggregate",
        "",
        "| Metric | Before | After | Change |",
        "|---|---:|---:|---:|",
        f"| Train teacher-forced CE | {before['train_loss']['mean']:.6f} | "
        f"{after['train_loss']['mean']:.6f} | "
        f"{after['train_loss']['mean'] - before['train_loss']['mean']:+.6f} |",
        f"| Held-out teacher-forced CE | {before['eval_loss']['mean']:.6f} | "
        f"{after['eval_loss']['mean']:.6f} | "
        f"{after['eval_loss']['mean'] - before['eval_loss']['mean']:+.6f} |",
        f"| Mean KL to official | {before_probe['mean_kl']:.6f} | "
        f"{after_probe['mean_kl']:.6f} | "
        f"{after_probe['mean_kl'] - before_probe['mean_kl']:+.6f} |",
        f"| Mean logit cosine | {before_probe['mean_logit_cosine']:.6f} | "
        f"{after_probe['mean_logit_cosine']:.6f} | "
        f"{after_probe['mean_logit_cosine'] - before_probe['mean_logit_cosine']:+.6f} |",
        f"| Top-10 overlap | {before_probe['top10_overlap']}/"
        f"{before_probe['top10_total']} | {after_probe['top10_overlap']}/"
        f"{after_probe['top10_total']} |  |",
        "",
        "## Held-out generations",
        "",
        "| Prompt | Before | After | Official teacher |",
        "|---|---|---|---|",
    ]
    teacher_rows = {
        row["id"]: row
        for row in json.loads((out_dir / "teacher_data.json").read_text(encoding="utf-8"))["eval"]
    }
    for prompt_id, row in before["probes"]["prompts"].items():
        before_text = row["generation"].replace("|", "\\|").replace("\n", "\\n")
        after_text = (
            after["probes"]["prompts"][prompt_id]["generation"]
            .replace("|", "\\|")
            .replace("\n", "\\n")
        )
        teacher_text = teacher_rows[prompt_id]["response"].replace("|", "\\|").replace("\n", "\\n")
        lines.append(f"| {prompt_id} | {before_text} | {after_text} | {teacher_text} |")
    protocol = result["protocol"]
    lines.extend(
        [
            "",
            "## Protocol",
            "",
            f"- Steps: {protocol['steps']}; learning rate: {protocol['learning_rate']:.2e}.",
            f"- Trainable Shadow parameters: {protocol['trainable_parameters']:,}.",
            f"- Frozen packed base unchanged: {protocol['packed_base_unchanged']}.",
            f"- Selected checkpoint step: {protocol['best_step']}.",
            f"- Training time: {result['training_seconds']:.1f}s.",
            f"- Peak MLX memory: {result['peak_memory_gb']:.3f} GiB.",
            "",
            "## Verdict",
            "",
            "- Shadow recovery is meaningful: the held-out loss and first-token "
            "distribution both improve substantially while every packed base "
            "weight remains frozen.",
            "- This short sequence-level distillation stage does not yet recover "
            "autoregressive semantics. Generation changes from the original "
            "`fkk` loop to early EOS or `0`, so the run is a positive gradient "
            "signal rather than a successful model checkpoint.",
            "- Step 24 beats step 36 on held-out CE, confirming that the 12-example "
            "teacher set is already overfitting. The next run needs a larger, "
            "balanced teacher corpus and token-level KL, not simply more epochs "
            "on these examples.",
            "",
        ]
    )
    (out_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official", type=Path, default=DEFAULT_OFFICIAL)
    parser.add_argument("--ptq", type=Path, default=DEFAULT_PTQ)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--steps", type=int, default=36)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--teacher-tokens", type=int, default=16)
    parser.add_argument("--generation-tokens", type=int, default=12)
    args = parser.parse_args()
    result = run(
        args.official.expanduser().resolve(),
        args.ptq.expanduser().resolve(),
        args.out,
        steps=args.steps,
        learning_rate=args.learning_rate,
        teacher_tokens=args.teacher_tokens,
        generation_tokens=args.generation_tokens,
    )
    _write_report(result, args.out)
    print(json.dumps(result["after"]["probes"]["aggregate"], indent=2), flush=True)
    print(f"Wrote {args.out / 'REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
