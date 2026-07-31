"""E5-R1: token-level Shadow recovery on a deterministic Tulu 3 sample.

The official Bonsai checkpoint and the locally quantized student are loaded
sequentially.  The teacher first writes a compact top-k-plus-tail distribution
for every valid token position; the teacher is then released before the frozen
2-bit student and its trainable Shadow adapter are loaded.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import heapq
import json
import random
import time
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pyarrow.parquet as pq
from e2_hybrid_bonsai_qwen36 import PROMPTS, _chat_prompt
from e5_shadow_recovery_qwen36 import (
    _evaluate_probes,
    _model_logits,
    _teacher_response,
)
from mlx import optimizers
from mlx.utils import tree_flatten
from mlx_lm import load
from mlx_lm.utils import load_tokenizer

from shadow_peft_mlx import (
    ShadowConfig,
    get_shadow_model,
)
from shadow_peft_mlx.quantized_grad import quantized_linear_input_grad

DEFAULT_DATASET = Path(
    "~/.cache/modelscope/datasets/"
    "allenai--tulu-3-sft-mixture/snapshots/master"
).expanduser()
DEFAULT_OFFICIAL = Path(
    "~/.cache/modelscope/models/"
    "prism-ml--Ternary-Bonsai-27B-mlx-2bit/snapshots/master"
).expanduser()
DEFAULT_PTQ = Path("qwen3.6-27b-mlx-2bit-v3-oneshot")
DEFAULT_OUT = Path("experiment/analysis_qwen36_bonsai_migration/e5_tulu_r1")

BUCKET_WEIGHTS = {
    "general": 0.40,
    "multilingual": 0.20,
    "math": 0.20,
    "code": 0.20,
}

EXCLUDED_SOURCE_MARKERS = (
    "wildjailbreak",
    "wildguard",
    "hard_coded_repeated",
)


def _unwrap_logits(output: Any) -> mx.array:
    return output.logits if hasattr(output, "logits") else output


def _source_bucket(source: str) -> str | None:
    lower = source.lower()
    if any(marker in lower for marker in EXCLUDED_SOURCE_MARKERS):
        return None
    if "aya_" in lower:
        return "multilingual"
    if "code" in lower:
        return "code"
    if "math" in lower or "algebra" in lower or "gsm8k" in lower:
        return "math"
    return "general"


def _quotas(total: int) -> dict[str, int]:
    raw = {name: total * weight for name, weight in BUCKET_WEIGHTS.items()}
    quotas = {name: int(value) for name, value in raw.items()}
    remainder = total - sum(quotas.values())
    order = sorted(raw, key=lambda name: (raw[name] - quotas[name], name), reverse=True)
    for name in order[:remainder]:
        quotas[name] += 1
    return quotas


def _priority(seed: int, row_id: str) -> int:
    payload = f"{seed}\0{row_id}".encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def _valid_messages(messages: Any) -> bool:
    if not isinstance(messages, list) or len(messages) < 2:
        return False
    valid_roles = {"system", "user", "assistant", "tool"}
    return all(
        isinstance(message, dict)
        and message.get("role") in valid_roles
        and isinstance(message.get("content"), str)
        and bool(message["content"].strip())
        for message in messages
    )


def _iter_tulu_rows(dataset_dir: Path) -> Iterable[dict[str, Any]]:
    parquet_dir = dataset_dir / "data" if (dataset_dir / "data").is_dir() else dataset_dir
    files = sorted(parquet_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet files found under {parquet_dir}")
    for path in files:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(
            columns=["id", "messages", "source"],
            batch_size=8192,
            use_threads=False,
        ):
            yield from batch.to_pylist()


def _sample_rows(
    dataset_dir: Path,
    *,
    total: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    quotas = _quotas(total)
    heaps: dict[str, list[tuple[int, str, dict[str, Any]]]] = {
        bucket: [] for bucket in quotas
    }
    scanned = 0
    eligible = Counter()
    excluded = Counter()
    for row in _iter_tulu_rows(dataset_dir):
        scanned += 1
        source = str(row.get("source", ""))
        bucket = _source_bucket(source)
        if bucket is None:
            excluded[source] += 1
            continue
        messages = row.get("messages")
        if not _valid_messages(messages):
            excluded["invalid_messages"] += 1
            continue
        eligible[bucket] += 1
        row_id = str(row["id"])
        priority = _priority(seed, row_id)
        candidate = (
            -priority,
            row_id,
            {
                "id": row_id,
                "source": source,
                "bucket": bucket,
                "messages": messages,
            },
        )
        heap = heaps[bucket]
        if len(heap) < quotas[bucket]:
            heapq.heappush(heap, candidate)
        elif priority < -heap[0][0]:
            heapq.heapreplace(heap, candidate)

    selected = []
    for bucket, heap in heaps.items():
        if len(heap) != quotas[bucket]:
            raise ValueError(
                f"bucket {bucket!r} supplied {len(heap)} rows, expected {quotas[bucket]}"
            )
        selected.extend(candidate[2] for candidate in heap)
    selected.sort(key=lambda row: _priority(seed + 1, row["id"]))
    stats = {
        "scanned_rows": scanned,
        "eligible_by_bucket": dict(eligible),
        "excluded_by_source": dict(excluded),
        "selected_by_bucket": dict(Counter(row["bucket"] for row in selected)),
    }
    return selected, stats


def _tokenize_messages(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    max_length: int,
) -> list[int]:
    kwargs = {
        "tokenize": True,
        "add_generation_prompt": False,
        "enable_thinking": False,
    }
    token_ids = tokenizer.apply_chat_template(messages, **kwargs)
    if token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    return [int(token_id) for token_id in token_ids[:max_length]]


def prepare_dataset(
    dataset_dir: Path,
    official_dir: Path,
    out_dir: Path,
    *,
    train_samples: int,
    eval_samples: int,
    max_length: int,
    seed: int,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    total = train_samples + eval_samples
    selected, stats = _sample_rows(dataset_dir, total=total, seed=seed)
    tokenizer = load_tokenizer(official_dir)

    records = []
    rejected_short = 0
    for index, row in enumerate(selected):
        input_ids = _tokenize_messages(
            tokenizer,
            row["messages"],
            max_length=max_length,
        )
        if len(input_ids) < 2:
            rejected_short += 1
            continue
        split = "eval" if index < eval_samples else "train"
        records.append({**row, "split": split, "input_ids": input_ids})
    if rejected_short:
        raise ValueError(
            f"{rejected_short} sampled conversations tokenize to fewer than two tokens"
        )
    if sum(row["split"] == "train" for row in records) != train_samples:
        raise ValueError("prepared train sample count differs from requested count")

    dataset_path = out_dir / "dataset.jsonl"
    with dataset_path.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    digest = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    manifest = {
        "dataset_dir": str(dataset_dir),
        "official_tokenizer": str(official_dir),
        "train_samples": train_samples,
        "eval_samples": eval_samples,
        "max_length": max_length,
        "seed": seed,
        "sha256": digest,
        "tokens": sum(len(row["input_ids"]) - 1 for row in records),
        "tokens_by_split": {
            split: sum(
                len(row["input_ids"]) - 1 for row in records if row["split"] == split
            )
            for split in ("train", "eval")
        },
        "selected_by_split": dict(Counter(row["split"] for row in records)),
        "selected_by_bucket": dict(Counter(row["bucket"] for row in records)),
        "selected_by_source": dict(Counter(row["source"] for row in records)),
        **stats,
    }
    (out_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return manifest


def _load_records(out_dir: Path) -> list[dict[str, Any]]:
    dataset_path = out_dir / "dataset.jsonl"
    if not dataset_path.is_file():
        raise FileNotFoundError(f"prepared dataset missing: {dataset_path}")
    return [
        json.loads(line)
        for line in dataset_path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def build_teacher_cache(
    official_dir: Path,
    out_dir: Path,
    *,
    top_k: int,
    generation_tokens: int,
) -> dict[str, Any]:
    records = _load_records(out_dir)
    manifest = json.loads((out_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    max_length = int(manifest["max_length"])
    count = len(records)

    input_ids = np.zeros((count, max_length), dtype=np.int32)
    lengths = np.zeros(count, dtype=np.int32)
    top_ids = np.zeros((count, max_length - 1, top_k), dtype=np.int32)
    top_log_probs = np.zeros((count, max_length - 1, top_k), dtype=np.float32)
    tail_log_probs = np.zeros((count, max_length - 1), dtype=np.float32)
    mask = np.zeros((count, max_length - 1), dtype=np.uint8)

    print(f"Loading official teacher: {official_dir}", flush=True)
    teacher, tokenizer = load(str(official_dir))
    teacher.eval()
    started = time.perf_counter()
    for index, row in enumerate(records):
        ids = row["input_ids"]
        length = len(ids)
        input_ids[index, :length] = ids
        lengths[index] = length
        mask[index, : length - 1] = 1

        output = _unwrap_logits(teacher(mx.array([ids], dtype=mx.int32)))
        logits = output[0, : length - 1].astype(mx.float32)
        selected_ids = mx.argpartition(logits, kth=-top_k, axis=-1)[..., -top_k:]
        selected_logits = mx.take_along_axis(logits, selected_ids, axis=-1)
        log_norm = mx.logsumexp(logits, axis=-1, keepdims=True)
        selected_log_probs = selected_logits - log_norm
        selected_probs = mx.exp(selected_log_probs)
        selected_mass = mx.sum(selected_probs, axis=-1)
        tail_prob = mx.maximum(1.0 - selected_mass, 1e-7)
        aggregate_norm = selected_mass + tail_prob
        selected_log_probs = mx.log(selected_probs / aggregate_norm[..., None])
        tail_log_prob = mx.log(tail_prob / aggregate_norm)
        mx.eval(selected_ids, selected_log_probs, tail_log_prob)
        top_ids[index, : length - 1] = np.asarray(selected_ids, dtype=np.int32)
        top_log_probs[index, : length - 1] = np.asarray(
            selected_log_probs, dtype=np.float32
        )
        tail_log_probs[index, : length - 1] = np.asarray(
            tail_log_prob, dtype=np.float32
        )
        del output, logits, selected_ids, selected_logits, selected_log_probs
        if (index + 1) % 8 == 0 or index + 1 == count:
            mx.clear_cache()
            elapsed = time.perf_counter() - started
            print(
                f"teacher {index + 1:04d}/{count}: "
                f"{manifest['tokens'] / max(count, 1):.1f} avg tokens, "
                f"{elapsed:.1f}s elapsed",
                flush=True,
            )

    probe_logits = {}
    teacher_generations = {}
    for prompt in PROMPTS:
        chat = _chat_prompt(tokenizer, prompt["text"])
        prompt_ids = tokenizer.encode(chat)
        probe_logits[prompt["id"]] = _model_logits(teacher, prompt_ids)
        teacher_generations[prompt["id"]] = _teacher_response(
            teacher,
            tokenizer,
            chat,
            max_tokens=generation_tokens,
        )

    cache_metadata = {
        "dataset_sha256": manifest["sha256"],
        "official_model": str(official_dir),
        "top_k": top_k,
        "max_length": max_length,
        "examples": count,
        "tokens": int(mask.sum()),
        "teacher_seconds": time.perf_counter() - started,
        "teacher_generations": teacher_generations,
    }
    arrays = {
        "input_ids": input_ids,
        "lengths": lengths,
        "top_ids": top_ids,
        "top_log_probs": top_log_probs,
        "tail_log_probs": tail_log_probs,
        "mask": mask,
        "metadata": np.array(json.dumps(cache_metadata, ensure_ascii=False)),
    }
    arrays.update(
        {f"probe_logits_{name}": values for name, values in probe_logits.items()}
    )
    cache_path = out_dir / "teacher_topk.npz"
    with cache_path.open("wb") as handle:
        np.savez_compressed(handle, **arrays)

    del teacher
    gc.collect()
    mx.clear_cache()
    (out_dir / "teacher_cache_manifest.json").write_text(
        json.dumps(cache_metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return cache_metadata


def _topk_tail_kl(
    model: nn.Module,
    input_ids: mx.array,
    teacher_top_ids: mx.array,
    teacher_top_log_probs: mx.array,
    teacher_tail_log_probs: mx.array,
) -> mx.array:
    output = _unwrap_logits(model(input_ids))
    student_logits = output[:, :-1].astype(mx.float32)
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


def _cache_example(
    cache: Any,
    index: int,
    *,
    max_length: int | None = None,
) -> tuple[mx.array, ...]:
    length = int(cache["lengths"][index])
    if max_length is not None:
        length = min(length, max_length)
    return (
        mx.array(cache["input_ids"][index : index + 1, :length], dtype=mx.int32),
        mx.array(cache["top_ids"][index, : length - 1], dtype=mx.int32),
        mx.array(cache["top_log_probs"][index, : length - 1], dtype=mx.float32),
        mx.array(cache["tail_log_probs"][index, : length - 1], dtype=mx.float32),
    )


def _evaluate_cache_kl(
    model: nn.Module,
    cache: Any,
    indices: list[int],
    *,
    max_length: int | None = None,
) -> dict[str, Any]:
    model.eval()
    rows = []
    for index in indices:
        loss = _topk_tail_kl(
            model,
            *_cache_example(cache, index, max_length=max_length),
        )
        mx.eval(loss)
        rows.append({"index": index, "kl": float(loss)})
    return {
        "mean": sum(row["kl"] for row in rows) / len(rows),
        "examples": rows,
    }


def train_shadow(
    ptq_dir: Path,
    out_dir: Path,
    *,
    steps: int,
    learning_rate: float,
    eval_every: int,
    generation_tokens: int,
    train_max_length: int,
    seed: int,
) -> dict[str, Any]:
    records = _load_records(out_dir)
    cache = np.load(out_dir / "teacher_topk.npz", allow_pickle=False)
    cache_metadata = json.loads(str(cache["metadata"].item()))
    dataset_manifest = json.loads(
        (out_dir / "dataset_manifest.json").read_text(encoding="utf-8")
    )
    if cache_metadata["dataset_sha256"] != dataset_manifest["sha256"]:
        raise ValueError("teacher cache does not match prepared dataset")

    train_indices = [i for i, row in enumerate(records) if row["split"] == "train"]
    eval_indices = [i for i, row in enumerate(records) if row["split"] == "eval"]
    if not train_indices or not eval_indices:
        raise ValueError("both train and eval records are required")

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
        reference_logits[prompt["id"]] = cache[f"probe_logits_{prompt['id']}"]

    before_train = _evaluate_cache_kl(
        shadow,
        cache,
        train_indices,
        max_length=train_max_length,
    )
    before_eval = _evaluate_cache_kl(
        shadow,
        cache,
        eval_indices,
        max_length=train_max_length,
    )
    before_probes = _evaluate_probes(
        shadow,
        tokenizer,
        probes,
        reference_logits,
        generation_tokens=generation_tokens,
    )
    print(
        f"Before: train_KL={before_train['mean']:.5f}, "
        f"eval_KL={before_eval['mean']:.5f}, "
        f"probe_KL={before_probes['aggregate']['mean_kl']:.5f}",
        flush=True,
    )

    optimizer = optimizers.AdamW(
        learning_rate=learning_rate,
        betas=(0.9, 0.98),
        weight_decay=0.0,
    )
    value_and_grad = nn.value_and_grad(shadow, _topk_tail_kl)
    order = []
    rng = random.Random(seed + 2)
    while len(order) < steps:
        epoch = train_indices[:]
        rng.shuffle(epoch)
        order.extend(epoch)
    order = order[:steps]

    history = []
    evaluations = []
    best_eval = before_eval["mean"]
    best_step = 0
    best_adapter: dict[str, mx.array] | None = None
    started = time.perf_counter()
    shadow.train()
    with quantized_linear_input_grad():
        for step, index in enumerate(order, start=1):
            loss, gradients = value_and_grad(
                shadow,
                *_cache_example(cache, index, max_length=train_max_length),
            )
            gradients, gradient_norm = optimizers.clip_grad_norm(
                gradients,
                max_norm=1.0,
            )
            optimizer.update(shadow, gradients)
            mx.eval(shadow.parameters(), optimizer.state, loss, gradient_norm)
            row = {
                "step": step,
                "index": index,
                "id": records[index]["id"],
                "bucket": records[index]["bucket"],
                "tokens": min(int(cache["lengths"][index]), train_max_length) - 1,
                "loss": float(loss),
                "gradient_norm": float(gradient_norm),
                "elapsed_seconds": time.perf_counter() - started,
            }
            history.append(row)
            print(
                f"step {step:04d}/{steps}: {row['bucket']:<12} "
                f"tokens={row['tokens']:3d} KL={row['loss']:.5f} "
                f"grad={row['gradient_norm']:.4f}",
                flush=True,
            )
            if step % eval_every == 0 or step == steps:
                evaluation = _evaluate_cache_kl(
                    shadow,
                    cache,
                    eval_indices,
                    max_length=train_max_length,
                )
                evaluations.append({"step": step, "mean_eval_kl": evaluation["mean"]})
                print(
                    f"  checkpoint step {step}: held-out KL={evaluation['mean']:.5f}",
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
                shadow.train()

    shadow.eval()
    if best_adapter is not None:
        shadow.load_weights(list(best_adapter.items()), strict=False)
        mx.eval(shadow.parameters())
        del best_adapter
        mx.clear_cache()

    after_train = _evaluate_cache_kl(
        shadow,
        cache,
        train_indices,
        max_length=train_max_length,
    )
    after_eval = _evaluate_cache_kl(
        shadow,
        cache,
        eval_indices,
        max_length=train_max_length,
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
    windowed_train_tokens = sum(
        min(int(cache["lengths"][index]), train_max_length) - 1
        for index in train_indices
    )
    windowed_eval_tokens = sum(
        min(int(cache["lengths"][index]), train_max_length) - 1
        for index in eval_indices
    )
    result = {
        "protocol": {
            "dataset": str(dataset_manifest["dataset_dir"]),
            "dataset_sha256": dataset_manifest["sha256"],
            "official_model": cache_metadata["official_model"],
            "ptq_model": str(ptq_dir),
            "teacher_top_k": cache_metadata["top_k"],
            "train_samples": len(train_indices),
            "eval_samples": len(eval_indices),
            "cached_train_tokens": dataset_manifest["tokens_by_split"]["train"],
            "cached_eval_tokens": dataset_manifest["tokens_by_split"]["eval"],
            "windowed_train_tokens": windowed_train_tokens,
            "windowed_eval_tokens": windowed_eval_tokens,
            "optimization_tokens": sum(row["tokens"] for row in history),
            "steps": steps,
            "learning_rate": learning_rate,
            "train_max_length": train_max_length,
            "shadow_config": config.to_dict(),
            "trainable_parameters": trainable_count,
            "total_module_parameters": total_count,
            "packed_sentinel": sentinel_name,
            "packed_base_unchanged": packed_unchanged,
            "best_step": best_step,
        },
        "before": {
            "train_kl": before_train,
            "eval_kl": before_eval,
            "probes": before_probes,
        },
        "after": {
            "train_kl": after_train,
            "eval_kl": after_eval,
            "probes": after_probes,
        },
        "history": history,
        "checkpoint_evaluations": evaluations,
        "training_seconds": time.perf_counter() - started,
        "peak_memory_gb": mx.get_peak_memory() / (1024**3),
        "adapter_dir": str(adapter_dir),
    }
    cache.close()
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
    lines = [
        "# E5-R1 — Tulu token-level Shadow recovery",
        "",
        "## Aggregate",
        "",
        "| Metric | Before | After | Change |",
        "|---|---:|---:|---:|",
        f"| Train top-k+tail KL | {before['train_kl']['mean']:.6f} | "
        f"{after['train_kl']['mean']:.6f} | "
        f"{after['train_kl']['mean'] - before['train_kl']['mean']:+.6f} |",
        f"| Held-out top-k+tail KL | {before['eval_kl']['mean']:.6f} | "
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
        f"- Tulu samples: {protocol['train_samples']} train / "
        f"{protocol['eval_samples']} held out.",
        f"- Cached teacher tokens: {protocol['cached_train_tokens']} train / "
        f"{protocol['cached_eval_tokens']} held out.",
        f"- Windowed KL coverage: {protocol['windowed_train_tokens']} train / "
        f"{protocol['windowed_eval_tokens']} held out; optimization consumed "
        f"{protocol['optimization_tokens']} token positions.",
        f"- Teacher approximation: exact top-{protocol['teacher_top_k']} tokens "
        "plus one aggregated tail bucket at every position.",
        f"- Steps: {protocol['steps']}; learning rate: "
        f"{protocol['learning_rate']:.2e}; selected checkpoint: "
        f"{protocol['best_step']}.",
        f"- Training/evaluation token window: {protocol['train_max_length']}.",
        f"- Trainable Shadow parameters: {protocol['trainable_parameters']:,}.",
        f"- Frozen packed base unchanged: {protocol['packed_base_unchanged']}.",
        f"- Training time: {result['training_seconds']:.1f}s; peak MLX memory: "
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
    parser.add_argument("--train-samples", type=int, default=64)
    parser.add_argument("--eval-samples", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--train-max-length", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--eval-every", type=int, default=16)
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
    if args.refresh_data or not manifest_path.is_file():
        manifest = prepare_dataset(
            dataset_dir,
            official_dir,
            out_dir,
            train_samples=args.train_samples,
            eval_samples=args.eval_samples,
            max_length=args.max_length,
            seed=args.seed,
        )
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)
    if args.prepare_only:
        return

    cache_path = out_dir / "teacher_topk.npz"
    if args.refresh_teacher_cache or not cache_path.is_file():
        cache_manifest = build_teacher_cache(
            official_dir,
            out_dir,
            top_k=args.top_k,
            generation_tokens=args.generation_tokens,
        )
    else:
        cache_manifest = json.loads(
            (out_dir / "teacher_cache_manifest.json").read_text(encoding="utf-8")
        )
    print(json.dumps(cache_manifest, indent=2, ensure_ascii=False), flush=True)
    if args.teacher_only:
        return

    result = train_shadow(
        ptq_dir,
        out_dir,
        steps=args.steps,
        learning_rate=args.learning_rate,
        eval_every=args.eval_every,
        generation_tokens=args.generation_tokens,
        train_max_length=args.train_max_length,
        seed=args.seed,
    )
    write_report(result, out_dir)
    print(json.dumps(result["after"]["probes"]["aggregate"], indent=2), flush=True)
    print(f"Wrote {out_dir / 'REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
