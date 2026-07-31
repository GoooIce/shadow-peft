"""E5-R2: long-context assistant-only Shadow recovery for Qwen3.6-27B.

The dataset contains up to 4096 tokens of causal context, but gradients are
restricted to short contiguous assistant-token windows.  Frozen prefixes are
materialized into the base and Shadow caches before each microbatch, providing
long context without retaining a 4096-token backward graph.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from e2_hybrid_bonsai_qwen36 import PROMPTS, _chat_prompt
from e5_shadow_recovery_qwen36 import _evaluate_probes, _model_logits
from e5_tulu_shadow_recovery_qwen36 import (
    DEFAULT_DATASET,
    DEFAULT_OFFICIAL,
    DEFAULT_PTQ,
    _priority,
    _quotas,
    _sample_rows,
)
from mlx import optimizers
from mlx.utils import tree_flatten, tree_map
from mlx_lm import load
from mlx_lm.utils import load_tokenizer

from shadow_peft_mlx import (
    ShadowConfig,
    get_shadow_model,
)
from shadow_peft_mlx.model_utils import _get_inner_model, _get_lm_head
from shadow_peft_mlx.quantized_grad import quantized_linear_input_grad

DEFAULT_OUT = Path("experiment/analysis_qwen36_bonsai_migration/e5_tulu_r2")


def _assistant_runs(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    max_length: int,
) -> tuple[list[int], list[list[int]]]:
    template_kwargs = {
        "tokenize": True,
        "enable_thinking": False,
    }
    full_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=False,
        **template_kwargs,
    )
    if full_ids and isinstance(full_ids[0], list):
        full_ids = full_ids[0]
    full_ids = [int(token_id) for token_id in full_ids[:max_length]]

    runs = []
    for index, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        prefix_ids = tokenizer.apply_chat_template(
            messages[:index],
            add_generation_prompt=True,
            **template_kwargs,
        )
        if prefix_ids and isinstance(prefix_ids[0], list):
            prefix_ids = prefix_ids[0]
        content_ids = tokenizer.encode(
            message["content"],
            add_special_tokens=False,
        )
        content_start = len(prefix_ids)
        content_end = min(content_start + len(content_ids), len(full_ids))
        if content_start >= content_end:
            continue
        if full_ids[content_start:content_end] != [
            int(token_id) for token_id in content_ids[: content_end - content_start]
        ]:
            raise ValueError("assistant content does not align with the chat template")
        # A logit at position p predicts the target token at p + 1.
        runs.append(list(range(content_start - 1, content_end - 1)))
    return full_ids, runs


def _choose_assistant_window(
    runs: list[list[int]],
    *,
    row_id: str,
    max_tokens: int,
    seed: int,
) -> list[int]:
    if not runs:
        return []
    # Prefer the latest assistant turn so the selected target sees the longest
    # available causal context.
    run = max(runs, key=lambda values: (values[-1], len(values)))
    if len(run) <= max_tokens:
        return run
    choices = len(run) - max_tokens + 1
    offset = _priority(seed + 11, row_id) % choices
    return run[offset : offset + max_tokens]


def prepare_dataset(
    dataset_dir: Path,
    official_dir: Path,
    out_dir: Path,
    *,
    train_samples: int,
    eval_samples: int,
    max_length: int,
    max_assistant_tokens: int,
    seed: int,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    requested_total = train_samples + eval_samples
    candidates, scan_stats = _sample_rows(
        dataset_dir,
        total=requested_total * 2,
        seed=seed,
    )
    tokenizer = load_tokenizer(official_dir)

    valid_by_bucket: dict[str, list[dict[str, Any]]] = {
        bucket: [] for bucket in _quotas(requested_total)
    }
    rejected = Counter()
    for row in candidates:
        try:
            input_ids, runs = _assistant_runs(
                tokenizer,
                row["messages"],
                max_length=max_length,
            )
        except (TypeError, ValueError) as error:
            rejected[type(error).__name__] += 1
            continue
        assistant_positions = _choose_assistant_window(
            runs,
            row_id=row["id"],
            max_tokens=max_assistant_tokens,
            seed=seed,
        )
        if not assistant_positions:
            rejected["no_assistant_within_context"] += 1
            continue
        valid_by_bucket[row["bucket"]].append(
            {
                "id": row["id"],
                "source": row["source"],
                "bucket": row["bucket"],
                "input_ids": input_ids,
                "assistant_positions": assistant_positions,
            }
        )

    train_quotas = _quotas(train_samples)
    eval_quotas = _quotas(eval_samples)
    records = []
    for bucket in train_quotas:
        rows = sorted(
            valid_by_bucket[bucket],
            key=lambda row: _priority(seed + 13, row["id"]),
        )
        needed = train_quotas[bucket] + eval_quotas[bucket]
        if len(rows) < needed:
            raise ValueError(
                f"bucket {bucket!r} has {len(rows)} valid rows, needs {needed}"
            )
        eval_rows = rows[: eval_quotas[bucket]]
        train_rows = rows[eval_quotas[bucket] : needed]
        records.extend({**row, "split": "eval"} for row in eval_rows)
        records.extend({**row, "split": "train"} for row in train_rows)
    records.sort(key=lambda row: _priority(seed + 17, row["id"]))

    dataset_path = out_dir / "dataset.jsonl"
    with dataset_path.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    digest = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    context_lengths = np.array([len(row["input_ids"]) for row in records])
    assistant_counts = np.array([len(row["assistant_positions"]) for row in records])
    manifest = {
        "dataset_dir": str(dataset_dir),
        "official_tokenizer": str(official_dir),
        "train_samples": train_samples,
        "eval_samples": eval_samples,
        "max_length": max_length,
        "max_assistant_tokens": max_assistant_tokens,
        "seed": seed,
        "sha256": digest,
        "selected_by_split": dict(Counter(row["split"] for row in records)),
        "selected_by_bucket": dict(Counter(row["bucket"] for row in records)),
        "selected_by_source": dict(Counter(row["source"] for row in records)),
        "context_tokens": int(context_lengths.sum()),
        "assistant_tokens": int(assistant_counts.sum()),
        "context_length": {
            "min": int(context_lengths.min()),
            "mean": float(context_lengths.mean()),
            "p50": float(np.percentile(context_lengths, 50)),
            "p90": float(np.percentile(context_lengths, 90)),
            "p99": float(np.percentile(context_lengths, 99)),
            "max": int(context_lengths.max()),
            "at_4096": int((context_lengths == max_length).sum()),
        },
        "assistant_token_count": {
            "min": int(assistant_counts.min()),
            "mean": float(assistant_counts.mean()),
            "max": int(assistant_counts.max()),
        },
        "rejected": dict(rejected),
        "scan": scan_stats,
    }
    (out_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return manifest


def _load_records(out_dir: Path) -> list[dict[str, Any]]:
    path = out_dir / "dataset.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"prepared dataset missing: {path}")
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _topk_tail(logits: mx.array, top_k: int) -> tuple[np.ndarray, ...]:
    logits = logits.astype(mx.float32)
    top_ids = mx.argpartition(logits, kth=-top_k, axis=-1)[..., -top_k:]
    top_logits = mx.take_along_axis(logits, top_ids, axis=-1)
    log_norm = mx.logsumexp(logits, axis=-1, keepdims=True)
    top_log_probs = top_logits - log_norm
    top_probs = mx.exp(top_log_probs)
    top_mass = mx.sum(top_probs, axis=-1)
    tail_prob = mx.maximum(1.0 - top_mass, 1e-7)
    aggregate_norm = top_mass + tail_prob
    top_log_probs = mx.log(top_probs / aggregate_norm[..., None])
    tail_log_probs = mx.log(tail_prob / aggregate_norm)
    mx.eval(top_ids, top_log_probs, tail_log_probs)
    return (
        np.asarray(top_ids, dtype=np.int32),
        np.asarray(top_log_probs, dtype=np.float32),
        np.asarray(tail_log_probs, dtype=np.float32),
    )


def _teacher_distributions(
    model: nn.Module,
    input_ids: list[int],
    positions: list[int],
    *,
    top_k: int,
    prefill_chunk_size: int,
    head_chunk_size: int,
) -> tuple[np.ndarray, ...]:
    inner = _get_inner_model(model)
    head = _get_lm_head(model)
    if head is None:
        head = inner.embed_tokens.as_linear
    cache = model.make_cache()
    position_to_output = {position: index for index, position in enumerate(positions)}
    result_ids = np.zeros((len(positions), top_k), dtype=np.int32)
    result_log_probs = np.zeros((len(positions), top_k), dtype=np.float32)
    result_tail = np.zeros(len(positions), dtype=np.float32)
    stop = positions[-1] + 1
    for chunk_start in range(0, stop, prefill_chunk_size):
        chunk_end = min(chunk_start + prefill_chunk_size, stop)
        ids = mx.array([input_ids[chunk_start:chunk_end]], dtype=mx.int32)
        hidden = inner(ids, cache=cache)
        local_positions = [
            position - chunk_start
            for position in positions
            if chunk_start <= position < chunk_end
        ]
        if not local_positions:
            mx.eval(hidden)
            continue
        for offset in range(0, len(local_positions), head_chunk_size):
            local = local_positions[offset : offset + head_chunk_size]
            selected_hidden = mx.take(
                hidden[0],
                mx.array(local, dtype=mx.int32),
                axis=0,
            )
            top_ids, top_log_probs, tail_log_probs = _topk_tail(
                head(selected_hidden),
                top_k,
            )
            for local_index, local_position in enumerate(local):
                absolute_position = chunk_start + local_position
                output_index = position_to_output[absolute_position]
                result_ids[output_index] = top_ids[local_index]
                result_log_probs[output_index] = top_log_probs[local_index]
                result_tail[output_index] = tail_log_probs[local_index]
        mx.eval(hidden)
    return result_ids, result_log_probs, result_tail


def _open_teacher_arrays(
    cache_dir: Path,
    *,
    count: int,
    max_assistant_tokens: int,
    top_k: int,
    refresh: bool,
) -> dict[str, np.ndarray]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    specs = {
        "top_ids": (np.int32, (count, max_assistant_tokens, top_k)),
        "top_log_probs": (np.float32, (count, max_assistant_tokens, top_k)),
        "tail_log_probs": (np.float32, (count, max_assistant_tokens)),
        "positions": (np.int32, (count, max_assistant_tokens)),
        "counts": (np.int32, (count,)),
        "completed": (np.uint8, (count,)),
    }
    arrays = {}
    for name, (dtype, shape) in specs.items():
        path = cache_dir / f"{name}.npy"
        mode = "w+" if refresh or not path.exists() else "r+"
        arrays[name] = np.lib.format.open_memmap(
            path,
            mode=mode,
            dtype=dtype,
            shape=shape,
        )
        if mode == "w+":
            arrays[name][...] = 0
    return arrays


def build_teacher_cache(
    official_dir: Path,
    out_dir: Path,
    *,
    top_k: int,
    max_assistant_tokens: int,
    prefill_chunk_size: int,
    head_chunk_size: int,
    refresh: bool,
) -> dict[str, Any]:
    records = _load_records(out_dir)
    dataset_manifest = json.loads(
        (out_dir / "dataset_manifest.json").read_text(encoding="utf-8")
    )
    cache_dir = out_dir / "teacher_cache"
    arrays = _open_teacher_arrays(
        cache_dir,
        count=len(records),
        max_assistant_tokens=max_assistant_tokens,
        top_k=top_k,
        refresh=refresh,
    )
    completed_before = int(arrays["completed"].sum())
    print(
        f"Loading official teacher: {official_dir}; "
        f"resume={completed_before}/{len(records)}",
        flush=True,
    )
    teacher, tokenizer = load(str(official_dir))
    teacher.eval()
    started = time.perf_counter()
    processed_tokens = 0
    for index, row in enumerate(records):
        if arrays["completed"][index]:
            continue
        positions = [int(position) for position in row["assistant_positions"]]
        ids, log_probs, tail = _teacher_distributions(
            teacher,
            row["input_ids"],
            positions,
            top_k=top_k,
            prefill_chunk_size=prefill_chunk_size,
            head_chunk_size=head_chunk_size,
        )
        count = len(positions)
        arrays["top_ids"][index, :count] = ids
        arrays["top_log_probs"][index, :count] = log_probs
        arrays["tail_log_probs"][index, :count] = tail
        arrays["positions"][index, :count] = positions
        arrays["counts"][index] = count
        arrays["completed"][index] = 1
        processed_tokens += positions[-1] + 1
        if (index + 1) % 8 == 0 or int(arrays["completed"].sum()) == len(records):
            for array in arrays.values():
                array.flush()
            done = int(arrays["completed"].sum())
            elapsed = time.perf_counter() - started
            print(
                f"teacher {done:04d}/{len(records)}: "
                f"new_context_tokens={processed_tokens:,}, elapsed={elapsed:.1f}s",
                flush=True,
            )
            progress = {
                "dataset_sha256": dataset_manifest["sha256"],
                "completed": done,
                "examples": len(records),
                "top_k": top_k,
                "elapsed_this_run": elapsed,
            }
            (cache_dir / "progress.json").write_text(
                json.dumps(progress, indent=2),
                encoding="utf-8",
            )
        mx.clear_cache()

    probe_logits = {}
    for prompt in PROMPTS:
        chat = _chat_prompt(tokenizer, prompt["text"])
        probe_logits[prompt["id"]] = _model_logits(
            teacher,
            tokenizer.encode(chat),
        )
    with (cache_dir / "probes.npz").open("wb") as handle:
        np.savez_compressed(handle, **probe_logits)

    metadata = {
        "dataset_sha256": dataset_manifest["sha256"],
        "official_model": str(official_dir),
        "top_k": top_k,
        "max_assistant_tokens": max_assistant_tokens,
        "prefill_chunk_size": prefill_chunk_size,
        "head_chunk_size": head_chunk_size,
        "examples": len(records),
        "assistant_tokens": int(arrays["counts"].sum()),
        "teacher_seconds_this_run": time.perf_counter() - started,
        "resumed_examples": completed_before,
    }
    (cache_dir / "manifest.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    del teacher
    gc.collect()
    mx.clear_cache()
    return metadata


def _load_teacher_arrays(out_dir: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    cache_dir = out_dir / "teacher_cache"
    manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
    names = (
        "top_ids",
        "top_log_probs",
        "tail_log_probs",
        "positions",
        "counts",
        "completed",
    )
    arrays = {
        name: np.load(cache_dir / f"{name}.npy", mmap_mode="r")
        for name in names
    }
    if not bool(np.all(arrays["completed"])):
        raise ValueError("teacher cache is incomplete")
    return arrays, manifest


def _assistant_kl_from_hidden(
    model: Any,
    input_ids: mx.array,
    teacher_top_ids: mx.array,
    teacher_top_log_probs: mx.array,
    teacher_tail_log_probs: mx.array,
    *,
    cache: list,
) -> mx.array:
    hidden, _ = model.forward_hidden(input_ids, cache=cache)
    head = _get_lm_head(model.base_model)
    if head is None:
        inner = _get_inner_model(model.base_model)
        student_logits = inner.embed_tokens.as_linear(hidden)
    else:
        student_logits = head(hidden)
    student_logits = student_logits.astype(mx.float32)
    student_log_norm = mx.logsumexp(student_logits, axis=-1, keepdims=True)
    student_top_logits = mx.take_along_axis(
        student_logits,
        teacher_top_ids[None],
        axis=-1,
    )
    student_top_log_probs = student_top_logits - student_log_norm
    student_top_mass = mx.sum(mx.exp(student_top_log_probs), axis=-1)
    student_tail_log_probs = mx.log(mx.maximum(1.0 - student_top_mass, 1e-7))

    teacher_top_log_probs = teacher_top_log_probs[None].astype(mx.float32)
    teacher_tail_log_probs = teacher_tail_log_probs[None].astype(mx.float32)
    teacher_top_probs = mx.exp(teacher_top_log_probs)
    teacher_tail_probs = mx.exp(teacher_tail_log_probs)
    top_kl = mx.sum(
        teacher_top_probs * (teacher_top_log_probs - student_top_log_probs),
        axis=-1,
    )
    tail_kl = teacher_tail_probs * (
        teacher_tail_log_probs - student_tail_log_probs
    )
    return mx.mean(top_kl + tail_kl)


def _prefill(
    model: Any,
    input_ids: list[int],
    *,
    prefix_length: int,
    chunk_size: int,
) -> list:
    cache = model.make_cache()
    model.eval()
    for start in range(0, prefix_length, chunk_size):
        stop = min(start + chunk_size, prefix_length)
        hidden, _ = model.forward_hidden(
            mx.array([input_ids[start:stop]], dtype=mx.int32),
            cache=cache,
        )
        mx.eval(hidden)
    return cache


def _window(
    records: list[dict[str, Any]],
    arrays: dict[str, np.ndarray],
    index: int,
    *,
    target_tokens: int,
    salt: int,
) -> tuple[int, mx.array, mx.array, mx.array, mx.array]:
    count = int(arrays["counts"][index])
    width = min(target_tokens, count)
    choices = count - width + 1
    offset = _priority(salt, records[index]["id"]) % choices
    positions = arrays["positions"][index, offset : offset + width]
    if width > 1 and not bool(np.all(np.diff(positions) == 1)):
        raise ValueError(f"assistant positions are not contiguous for {records[index]['id']}")
    prefix_length = int(positions[0])
    chunk_ids = records[index]["input_ids"][
        prefix_length : int(positions[-1]) + 1
    ]
    return (
        prefix_length,
        mx.array([chunk_ids], dtype=mx.int32),
        mx.array(arrays["top_ids"][index, offset : offset + width], dtype=mx.int32),
        mx.array(
            arrays["top_log_probs"][index, offset : offset + width],
            dtype=mx.float32,
        ),
        mx.array(
            arrays["tail_log_probs"][index, offset : offset + width],
            dtype=mx.float32,
        ),
    )


def _one_loss(
    model: Any,
    records: list[dict[str, Any]],
    arrays: dict[str, np.ndarray],
    index: int,
    *,
    target_tokens: int,
    prefill_chunk_size: int,
    salt: int,
) -> mx.array:
    prefix_length, chunk_ids, top_ids, top_log_probs, tail_log_probs = _window(
        records,
        arrays,
        index,
        target_tokens=target_tokens,
        salt=salt,
    )
    cache = _prefill(
        model,
        records[index]["input_ids"],
        prefix_length=prefix_length,
        chunk_size=prefill_chunk_size,
    )
    model.eval()
    loss = _assistant_kl_from_hidden(
        model,
        chunk_ids,
        top_ids,
        top_log_probs,
        tail_log_probs,
        cache=cache,
    )
    mx.eval(loss)
    return loss


def _evaluate(
    model: Any,
    records: list[dict[str, Any]],
    arrays: dict[str, np.ndarray],
    indices: list[int],
    *,
    target_tokens: int,
    prefill_chunk_size: int,
    salt: int,
) -> dict[str, Any]:
    rows = []
    for number, index in enumerate(indices, start=1):
        loss = _one_loss(
            model,
            records,
            arrays,
            index,
            target_tokens=target_tokens,
            prefill_chunk_size=prefill_chunk_size,
            salt=salt,
        )
        rows.append({"index": index, "id": records[index]["id"], "kl": float(loss)})
        if number % 8 == 0 or number == len(indices):
            print(f"  eval {number:03d}/{len(indices)}", flush=True)
        mx.clear_cache()
    return {
        "mean": sum(row["kl"] for row in rows) / len(rows),
        "examples": rows,
    }


def train_shadow(
    ptq_dir: Path,
    out_dir: Path,
    *,
    steps: int,
    batch_size: int,
    target_tokens: int,
    learning_rate: float,
    eval_every: int,
    checkpoint_eval_samples: int,
    prefill_chunk_size: int,
    generation_tokens: int,
    seed: int,
) -> dict[str, Any]:
    records = _load_records(out_dir)
    arrays, teacher_manifest = _load_teacher_arrays(out_dir)
    dataset_manifest = json.loads(
        (out_dir / "dataset_manifest.json").read_text(encoding="utf-8")
    )
    if teacher_manifest["dataset_sha256"] != dataset_manifest["sha256"]:
        raise ValueError("teacher cache does not match the prepared dataset")
    train_indices = [i for i, row in enumerate(records) if row["split"] == "train"]
    eval_indices = [i for i, row in enumerate(records) if row["split"] == "eval"]
    if steps * batch_size > len(train_indices):
        raise ValueError("steps * batch_size exceeds the number of train samples")

    print(f"Loading frozen PTQ student: {ptq_dir}", flush=True)
    base, tokenizer = load(str(ptq_dir))
    config = ShadowConfig(
        num_shadow_layers=1,
        shadow_intermediate_size=512,
        injection_hidden_size=4,
        gate_hidden_size=4,
        alpha=1.0,
        dropout=0.0,
    )
    shadow = get_shadow_model(base, config)
    trainable_count = int(
        sum(value.size for _, value in tree_flatten(shadow.trainable_parameters()))
    )
    total_count = int(sum(value.size for _, value in tree_flatten(shadow.parameters())))
    base_flat = dict(tree_flatten(base.parameters()))
    sentinel_name = next(
        name
        for name, value in base_flat.items()
        if name.endswith(".weight") and value.dtype == mx.uint32
    )
    sentinel_before = base_flat[sentinel_name] * mx.array(1, dtype=mx.uint32)
    mx.eval(sentinel_before)

    probe_cache = np.load(out_dir / "teacher_cache" / "probes.npz")
    probes = []
    reference_logits = {}
    for prompt in PROMPTS:
        chat = _chat_prompt(tokenizer, prompt["text"])
        probes.append(
            {
                "id": prompt["id"],
                "chat": chat,
                "prompt_ids": tokenizer.encode(chat),
            }
        )
        reference_logits[prompt["id"]] = probe_cache[prompt["id"]]

    print("Evaluating held-out assistant KL before training", flush=True)
    before_eval = _evaluate(
        shadow,
        records,
        arrays,
        eval_indices,
        target_tokens=target_tokens,
        prefill_chunk_size=prefill_chunk_size,
        salt=seed + 101,
    )
    before_probes = _evaluate_probes(
        shadow,
        tokenizer,
        probes,
        reference_logits,
        generation_tokens=generation_tokens,
    )
    print(
        f"Before: eval_KL={before_eval['mean']:.5f}, "
        f"probe_KL={before_probes['aggregate']['mean_kl']:.5f}",
        flush=True,
    )

    optimizer = optimizers.AdamW(
        learning_rate=learning_rate,
        betas=(0.9, 0.98),
        weight_decay=0.0,
    )
    rng = random.Random(seed + 103)
    order = train_indices[:]
    rng.shuffle(order)
    order = order[: steps * batch_size]
    checkpoint_indices = eval_indices[:]
    rng.shuffle(checkpoint_indices)
    checkpoint_indices = checkpoint_indices[:checkpoint_eval_samples]

    history = []
    evaluations = []
    best_eval = before_eval["mean"]
    best_step = 0
    best_adapter: dict[str, mx.array] | None = None
    started = time.perf_counter()
    with quantized_linear_input_grad():
        for step in range(1, steps + 1):
            batch = order[(step - 1) * batch_size : step * batch_size]
            accumulated = None
            micro_rows = []
            for micro, index in enumerate(batch, start=1):
                prefix_length, chunk_ids, top_ids, top_log_probs, tail_log_probs = _window(
                    records,
                    arrays,
                    index,
                    target_tokens=target_tokens,
                    salt=seed + step * 1009 + micro,
                )
                cache = _prefill(
                    shadow,
                    records[index]["input_ids"],
                    prefix_length=prefix_length,
                    chunk_size=prefill_chunk_size,
                )
                shadow.train()

                def loss_fn(
                    model: nn.Module,
                    ids: mx.array,
                    teacher_ids: mx.array,
                    teacher_log_probs: mx.array,
                    teacher_tail: mx.array,
                    _cache: list = cache,
                ) -> mx.array:
                    return _assistant_kl_from_hidden(
                        model,
                        ids,
                        teacher_ids,
                        teacher_log_probs,
                        teacher_tail,
                        cache=_cache,
                    )

                value_and_grad = nn.value_and_grad(shadow, loss_fn)
                loss, gradients = value_and_grad(
                    shadow,
                    chunk_ids,
                    top_ids,
                    top_log_probs,
                    tail_log_probs,
                )
                gradients = tree_map(lambda value: value / batch_size, gradients)
                accumulated = (
                    gradients
                    if accumulated is None
                    else tree_map(lambda left, right: left + right, accumulated, gradients)
                )
                mx.eval(accumulated, loss)
                micro_rows.append(
                    {
                        "index": index,
                        "id": records[index]["id"],
                        "bucket": records[index]["bucket"],
                        "context_tokens": prefix_length,
                        "assistant_tokens": int(chunk_ids.shape[1]),
                        "loss": float(loss),
                    }
                )
                print(
                    f"step {step:03d}/{steps} micro {micro}/{batch_size}: "
                    f"ctx={prefix_length:4d} assistant={int(chunk_ids.shape[1]):2d} "
                    f"KL={float(loss):.5f}",
                    flush=True,
                )
                mx.clear_cache()

            assert accumulated is not None
            accumulated, gradient_norm = optimizers.clip_grad_norm(
                accumulated,
                max_norm=1.0,
            )
            optimizer.update(shadow, accumulated)
            mx.eval(shadow.parameters(), optimizer.state, gradient_norm)
            row = {
                "step": step,
                "mean_loss": sum(item["loss"] for item in micro_rows) / len(micro_rows),
                "gradient_norm": float(gradient_norm),
                "microbatches": micro_rows,
                "elapsed_seconds": time.perf_counter() - started,
            }
            history.append(row)
            print(
                f"step {step:03d}: mean_KL={row['mean_loss']:.5f} "
                f"grad={row['gradient_norm']:.4f}",
                flush=True,
            )
            if step % eval_every == 0 or step == steps:
                evaluation = _evaluate(
                    shadow,
                    records,
                    arrays,
                    checkpoint_indices,
                    target_tokens=target_tokens,
                    prefill_chunk_size=prefill_chunk_size,
                    salt=seed + 107,
                )
                evaluations.append(
                    {"step": step, "mean_eval_kl": evaluation["mean"]}
                )
                print(
                    f"  checkpoint step {step}: eval_KL={evaluation['mean']:.5f}",
                    flush=True,
                )
                if evaluation["mean"] < best_eval:
                    best_eval = evaluation["mean"]
                    best_step = step
                    best_adapter = {
                        name: value * 1
                        for name, value in shadow.adapter_parameters().items()
                    }
                    mx.eval(best_adapter)

    if best_adapter is not None:
        shadow.load_weights(list(best_adapter.items()), strict=False)
        mx.eval(shadow.parameters())
        del best_adapter
        mx.clear_cache()
    shadow.eval()

    print("Evaluating held-out assistant KL after training", flush=True)
    after_eval = _evaluate(
        shadow,
        records,
        arrays,
        eval_indices,
        target_tokens=target_tokens,
        prefill_chunk_size=prefill_chunk_size,
        salt=seed + 101,
    )
    after_probes = _evaluate_probes(
        shadow,
        tokenizer,
        probes,
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
            "dataset": str(dataset_manifest["dataset_dir"]),
            "dataset_sha256": dataset_manifest["sha256"],
            "official_model": teacher_manifest["official_model"],
            "ptq_model": str(ptq_dir),
            "train_samples": len(train_indices),
            "eval_samples": len(eval_indices),
            "max_context_length": dataset_manifest["max_length"],
            "context_length": dataset_manifest["context_length"],
            "cached_assistant_tokens": teacher_manifest["assistant_tokens"],
            "teacher_top_k": teacher_manifest["top_k"],
            "steps": steps,
            "batch_size": batch_size,
            "target_tokens_per_microbatch": target_tokens,
            "optimization_examples": steps * batch_size,
            "optimization_assistant_tokens": sum(
                item["assistant_tokens"]
                for row in history
                for item in row["microbatches"]
            ),
            "learning_rate": learning_rate,
            "shadow_config": config.to_dict(),
            "trainable_parameters": trainable_count,
            "total_module_parameters": total_count,
            "packed_sentinel": sentinel_name,
            "packed_base_unchanged": packed_unchanged,
            "best_step": best_step,
        },
        "before": {"eval_kl": before_eval, "probes": before_probes},
        "after": {"eval_kl": after_eval, "probes": after_probes},
        "history": history,
        "checkpoint_evaluations": evaluations,
        "training_seconds": time.perf_counter() - started,
        "peak_memory_gb": mx.get_peak_memory() / (1024**3),
        "adapter_dir": str(adapter_dir),
    }
    probe_cache.close()
    return result


def write_report(result: dict[str, Any], out_dir: Path) -> None:
    (out_dir / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    protocol = result["protocol"]
    before = result["before"]
    after = result["after"]
    before_probe = before["probes"]["aggregate"]
    after_probe = after["probes"]["aggregate"]
    context = protocol["context_length"]
    lines = [
        "# E5-R2 — 4096-context assistant-only Shadow recovery",
        "",
        "## Aggregate",
        "",
        "| Metric | Before | After | Change |",
        "|---|---:|---:|---:|",
        f"| Held-out assistant KL | {before['eval_kl']['mean']:.6f} | "
        f"{after['eval_kl']['mean']:.6f} | "
        f"{after['eval_kl']['mean'] - before['eval_kl']['mean']:+.6f} |",
        f"| Probe full-vocab KL | {before_probe['mean_kl']:.6f} | "
        f"{after_probe['mean_kl']:.6f} | "
        f"{after_probe['mean_kl'] - before_probe['mean_kl']:+.6f} |",
        f"| Probe logit cosine | {before_probe['mean_logit_cosine']:.6f} | "
        f"{after_probe['mean_logit_cosine']:.6f} | "
        f"{after_probe['mean_logit_cosine'] - before_probe['mean_logit_cosine']:+.6f} |",
        f"| Probe top-10 overlap | {before_probe['top10_overlap']}/"
        f"{before_probe['top10_total']} | {after_probe['top10_overlap']}/"
        f"{after_probe['top10_total']} | |",
        "",
        "## Protocol",
        "",
        f"- Dataset: {protocol['train_samples']} train / "
        f"{protocol['eval_samples']} held out.",
        f"- Maximum context: {protocol['max_context_length']}; mean "
        f"{context['mean']:.1f}; p90 {context['p90']:.1f}; max {context['max']}.",
        f"- Updates: {protocol['steps']} × batch "
        f"{protocol['batch_size']} = {protocol['optimization_examples']} examples.",
        f"- Assistant positions optimized: "
        f"{protocol['optimization_assistant_tokens']}.",
        f"- Truncated-backprop target window: "
        f"{protocol['target_tokens_per_microbatch']} assistant tokens.",
        f"- Trainable Shadow parameters: {protocol['trainable_parameters']:,}.",
        f"- Frozen packed base unchanged: {protocol['packed_base_unchanged']}.",
        f"- Selected checkpoint: {protocol['best_step']}; training time "
        f"{result['training_seconds']:.1f}s; peak MLX memory "
        f"{result['peak_memory_gb']:.3f} GiB.",
        "",
        "## Held-out probe generations",
        "",
        "| Prompt | Before | After |",
        "|---|---|---|",
    ]
    for prompt_id, row in before["probes"]["prompts"].items():
        before_text = row["generation"].replace("|", "\\|").replace("\n", "\\n")
        after_text = (
            after["probes"]["prompts"][prompt_id]["generation"]
            .replace("|", "\\|")
            .replace("\n", "\\n")
        )
        lines.append(f"| {prompt_id} | {before_text} | {after_text} |")
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--official", type=Path, default=DEFAULT_OFFICIAL)
    parser.add_argument("--ptq", type=Path, default=DEFAULT_PTQ)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--train-samples", type=int, default=1024)
    parser.add_argument("--eval-samples", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--max-assistant-tokens", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--target-tokens", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--eval-every", type=int, default=16)
    parser.add_argument("--checkpoint-eval-samples", type=int, default=8)
    parser.add_argument("--prefill-chunk-size", type=int, default=256)
    parser.add_argument("--head-chunk-size", type=int, default=16)
    parser.add_argument("--generation-tokens", type=int, default=12)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--teacher-only", action="store_true")
    parser.add_argument("--refresh-data", action="store_true")
    parser.add_argument("--refresh-teacher-cache", action="store_true")
    args = parser.parse_args()

    dataset_dir = args.dataset.expanduser().resolve()
    official_dir = args.official.expanduser().resolve()
    ptq_dir = args.ptq.expanduser().resolve()
    out_dir = args.out.expanduser().resolve()
    manifest_path = out_dir / "dataset_manifest.json"
    dataset_path = out_dir / "dataset.jsonl"
    if args.refresh_data or not manifest_path.is_file() or not dataset_path.is_file():
        manifest = prepare_dataset(
            dataset_dir,
            official_dir,
            out_dir,
            train_samples=args.train_samples,
            eval_samples=args.eval_samples,
            max_length=args.max_length,
            max_assistant_tokens=args.max_assistant_tokens,
            seed=args.seed,
        )
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)
    if args.prepare_only:
        return

    teacher_manifest_path = out_dir / "teacher_cache" / "manifest.json"
    if args.refresh_teacher_cache or not teacher_manifest_path.is_file():
        teacher_manifest = build_teacher_cache(
            official_dir,
            out_dir,
            top_k=args.top_k,
            max_assistant_tokens=args.max_assistant_tokens,
            prefill_chunk_size=args.prefill_chunk_size,
            head_chunk_size=args.head_chunk_size,
            refresh=args.refresh_teacher_cache,
        )
    else:
        teacher_manifest = json.loads(
            teacher_manifest_path.read_text(encoding="utf-8")
        )
    print(json.dumps(teacher_manifest, indent=2, ensure_ascii=False), flush=True)
    if args.teacher_only:
        return

    result = train_shadow(
        ptq_dir,
        out_dir,
        steps=args.steps,
        batch_size=args.batch_size,
        target_tokens=args.target_tokens,
        learning_rate=args.learning_rate,
        eval_every=args.eval_every,
        checkpoint_eval_samples=args.checkpoint_eval_samples,
        prefill_chunk_size=args.prefill_chunk_size,
        generation_tokens=args.generation_tokens,
        seed=args.seed,
    )
    write_report(result, out_dir)
    print(json.dumps(result["after"]["probes"]["aggregate"], indent=2), flush=True)
    print(f"Wrote {out_dir / 'REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
