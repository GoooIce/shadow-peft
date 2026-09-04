#!/usr/bin/env python3
"""Score a frozen Mage image batch with local ModelScope HPS/CLIP assets."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import resource
import socket
import time
from pathlib import Path

from shadow_lab.image_eval import (
    audit_reward_determinism,
    fit_reward_calibration,
    load_generated_reward_inputs,
    score_reward_batch,
)
from shadow_lab.image_reward_models import HPSv21Scorer, OpenCLIPH14Scorer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-jsonl", required=True, type=Path)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--hps-checkpoint", required=True, type=Path)
    parser.add_argument("--clip-checkpoint", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--determinism-check-count", type=int, default=4)
    parser.add_argument(
        "--fit-calibration",
        action="store_true",
        help="fit BF16 mean/std; use only on the preregistered calibration split",
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


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.offset < 0:
        raise ValueError("--batch-size must be positive and --offset non-negative")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    if args.determinism_check_count < 0:
        raise ValueError("--determinism-check-count must be non-negative")
    samples_path = args.samples_jsonl.expanduser().resolve()
    image_root = args.image_root.expanduser().resolve()
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
    hps = HPSv21Scorer(args.hps_checkpoint, batch_size=args.batch_size)
    clip = OpenCLIPH14Scorer(args.clip_checkpoint, batch_size=args.batch_size)
    scorers = [hps, clip]
    load_seconds = time.perf_counter() - load_started

    score_started = time.perf_counter()
    score_records = score_reward_batch(inputs, scorers)
    score_seconds = time.perf_counter() - score_started
    for item, record in zip(inputs, score_records, strict=True):
        record.update(
            {
                "source": item.sample.source,
                "source_id": item.sample.source_id,
                "split": item.sample.split,
                "stratum": item.sample.stratum,
                "seed": item.sample.seed,
            }
        )
    scores_path = out_dir / "scores.jsonl"
    scores_path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            for record in score_records
        ),
        encoding="utf-8",
    )

    check_count = min(args.determinism_check_count, len(inputs))
    determinism = audit_reward_determinism(inputs[:check_count], scorers)
    calibration = None
    if args.fit_calibration:
        fitted = fit_reward_calibration(score_records)
        calibration_path = out_dir / "calibration.json"
        calibration_path.write_text(
            json.dumps(fitted.as_record(), ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        calibration = {
            "path": calibration_path.name,
            "sha256": file_sha256(calibration_path),
            "sample_baselines": len(fitted.sample_baselines),
        }

    import torch

    run = {
        "schema_version": 1,
        "started_at_unix": started_at,
        "finished_at_unix": time.time(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "device": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "hpsv2": package_version("hpsv2"),
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
            "hps_v2_1": {
                "path": str(args.hps_checkpoint.expanduser().resolve()),
                "sha256": file_sha256(args.hps_checkpoint.expanduser().resolve()),
            },
            "clip_h14": {
                "path": str(args.clip_checkpoint.expanduser().resolve()),
                "sha256": file_sha256(args.clip_checkpoint.expanduser().resolve()),
            },
        },
        "timing_seconds": {"model_load": load_seconds, "score": score_seconds},
        "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "determinism": {"sample_count": check_count, **determinism},
        "calibration": calibration,
        "scores": {
            "path": scores_path.name,
            "sha256": file_sha256(scores_path),
            "rows": len(score_records),
        },
        "privacy_contract": "scorers received image bytes plus FrozenEvalSample; no arm name",
    }
    run_path = out_dir / "run.json"
    run_path.write_text(
        json.dumps(run, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "run": str(run_path),
                "rows": len(score_records),
                "deterministic": determinism["deterministic"],
                "score_seconds": score_seconds,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
