#!/usr/bin/env python3
"""exp19: prepare GSM8K prompts/heldout for dynamic-sparse GKD (issue #54).

The GKD trainer consumes prompt JSONL (``prompt`` + optional task/expected)
and a heldout text JSONL for perplexity. This script materializes both from
GSM8K with a fixed seed and records a data fingerprint, so the fixed and
dynamic teacher-support arms train and evaluate on exactly the same rows.

Example (GB10)::

    PYTHONPATH=src .venv/bin/python scripts/exp19_prepare_gkd_data.py \
        --out-dir ~/exp19-gkd/data --train-limit 1000 --heldout-limit 200

Outputs: ``prompts.jsonl``, ``heldout.jsonl``, ``data_report.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

EXPERIMENT = "exp19-dynamic-topk-gkd"
SCHEMA_VERSION = "exp19-data/v1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Materialize GSM8K train prompts and test references for exp19."
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--train-limit", type=int, default=1000)
    parser.add_argument("--heldout-limit", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from shadow_lab.real_eval import dataset_fingerprint, load_gsm8k_examples

    out = Path(args.out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    train = load_gsm8k_examples(
        limit=args.train_limit, seed=args.seed, split="train"
    )
    heldout = load_gsm8k_examples(
        limit=args.heldout_limit, seed=args.seed, split="test"
    )
    if not train or not heldout:
        raise SystemExit("GSM8K loader returned no rows")

    _write_jsonl(
        out / "prompts.jsonl",
        [
            {
                "prompt_id": example.id,
                "prompt": example.prompt,
                "task": example.task,
                "expected": example.expected,
            }
            for example in train
        ],
    )
    _write_jsonl(
        out / "heldout.jsonl",
        [
            {
                "id": example.id,
                "text": example.metadata["reference"],
            }
            for example in heldout
        ],
    )

    report = {
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT,
        "issue": 54,
        "seed": args.seed,
        "train": {
            "split": "train",
            "n": len(train),
            "fingerprint": dataset_fingerprint(train),
            "path": str(out / "prompts.jsonl"),
        },
        "heldout": {
            "split": "test",
            "n": len(heldout),
            "fingerprint": dataset_fingerprint(heldout),
            "path": str(out / "heldout.jsonl"),
        },
    }
    (out / "data_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
