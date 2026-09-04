#!/usr/bin/env python3
"""Run held-out GenEval-compatible diagnostics with local ModelScope assets."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import socket
import statistics
import time
from collections import defaultdict
from pathlib import Path

from shadow_lab.geneval import GENEVAL_TASKS, summarize_results
from shadow_lab.geneval_models import TransformersGenEvalEvaluator
from shadow_lab.image_eval import load_generated_reward_inputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-jsonl", required=True, type=Path)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--detector-path", required=True, type=Path)
    parser.add_argument("--clip-checkpoint", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--determinism-check-count", type=int, default=2)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="permit a smoke subset that does not contain all six GenEval tasks",
    )
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def partial_summary(records: list[dict]) -> dict:
    grouped = defaultdict(list)
    for record in records:
        grouped[record["tag"]].append(float(record["correct"]))
    return {
        "images": len(records),
        "task_scores": {
            task: statistics.fmean(grouped[task]) for task in GENEVAL_TASKS if grouped[task]
        },
        "overall_equal_task_mean": None,
        "warning": "partial smoke only; not a held-out GenEval-48 score",
    }


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.determinism_check_count < 0:
        raise ValueError("batch size must be positive and determinism count non-negative")
    samples_path = args.samples_jsonl.expanduser().resolve()
    image_root = args.image_root.expanduser().resolve()
    detector_path = args.detector_path.expanduser().resolve()
    clip_checkpoint = args.clip_checkpoint.expanduser().resolve()
    out_dir = args.out.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if any(out_dir.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {out_dir}")

    inputs = load_generated_reward_inputs(
        samples_path,
        image_root,
        offset=args.offset,
        limit=args.limit,
    )
    started_at = time.time()
    load_started = time.perf_counter()
    evaluator = TransformersGenEvalEvaluator(
        detector_path,
        clip_checkpoint,
        batch_size=args.batch_size,
    )
    load_seconds = time.perf_counter() - load_started
    score_started = time.perf_counter()
    records = evaluator.evaluate_batch(inputs)
    score_seconds = time.perf_counter() - score_started

    check_count = min(args.determinism_check_count, len(inputs))
    repeated = evaluator.evaluate_batch(inputs[:check_count])
    deterministic = repeated == records[:check_count]
    if not deterministic:
        raise RuntimeError("GenEval-compatible evaluator was not exactly deterministic")

    present_tasks = {record["tag"] for record in records}
    if args.allow_partial:
        summary = partial_summary(records)
    else:
        missing = sorted(set(GENEVAL_TASKS) - present_tasks)
        if missing:
            raise ValueError(f"held-out GenEval run is missing tasks: {missing}")
        summary = summarize_results(records)

    results_path = out_dir / "results.jsonl"
    results_path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )
    import torch

    run = {
        "schema_version": 1,
        "metric_name": "GenEval-compatible Transformers/ModelScope port",
        "leaderboard_comparable": False,
        "started_at_unix": started_at,
        "finished_at_unix": time.time(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "device": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "transformers": package_version("transformers"),
            "open_clip_torch": package_version("open_clip_torch"),
        },
        "input": {
            "samples_jsonl": str(samples_path),
            "samples_sha256": file_sha256(samples_path),
            "image_root": str(image_root),
            "offset": args.offset,
            "count": len(inputs),
        },
        "assets": {
            "detector_path": str(detector_path),
            "detector_checkpoint_sha256": file_sha256(
                detector_path / "model.safetensors"
            ),
            "detector_config_sha256": file_sha256(detector_path / "config.json"),
            "clip_checkpoint": str(clip_checkpoint),
            "clip_checkpoint_sha256": file_sha256(clip_checkpoint),
        },
        "checkpoint_loading_info": evaluator.loading_info,
        "timing_seconds": {"model_load": load_seconds, "score": score_seconds},
        "determinism": {"sample_count": check_count, "exact": deterministic},
        "summary": summary,
        "results": {
            "path": results_path.name,
            "rows": len(records),
            "sha256": file_sha256(results_path),
        },
    }
    run_path = out_dir / "run.json"
    run_path.write_text(
        json.dumps(run, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"run": str(run_path), **summary}, ensure_ascii=False))


if __name__ == "__main__":
    main()
