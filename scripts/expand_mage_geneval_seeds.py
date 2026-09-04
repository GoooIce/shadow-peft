#!/usr/bin/env python3
"""Expand frozen GenEval prompt rows into the four-seed held-out protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from shadow_lab.geneval import expand_geneval_samples
from shadow_lab.image_eval import load_frozen_eval_samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-jsonl", required=True, type=Path)
    parser.add_argument("--replicas", type=int, default=4)
    parser.add_argument("--seed-stride", type=int, default=100_000_000)
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_samples(path: Path):
    samples = [sample for sample in load_frozen_eval_samples(path) if sample.source == "geneval"]
    if not samples:
        raise ValueError("pool selection has no GenEval rows")
    return samples


def main() -> None:
    args = parse_args()
    source_path = args.pool_jsonl.expanduser().resolve()
    out_dir = args.out.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if any(out_dir.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {out_dir}")
    expanded = expand_geneval_samples(
        load_samples(source_path),
        replicas=args.replicas,
        seed_stride=args.seed_stride,
    )
    output_path = out_dir / "geneval-expanded.jsonl"
    output_path.write_text(
        "".join(
            json.dumps(sample.as_record(), ensure_ascii=False, sort_keys=True) + "\n"
            for sample in expanded
        ),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "source": {
            "path": str(source_path),
            "sha256": file_sha256(source_path),
        },
        "policy": {
            "replicas": args.replicas,
            "seed_stride": args.seed_stride,
            "cluster_unit": "parent GenEval prompt",
        },
        "output": {
            "path": output_path.name,
            "rows": len(expanded),
            "sha256": file_sha256(output_path),
        },
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"manifest": str(manifest_path), **manifest["output"]}))


if __name__ == "__main__":
    main()
