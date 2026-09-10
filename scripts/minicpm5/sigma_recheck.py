#!/usr/bin/env python3
"""Offline re-check of ticket 17's episode-level pairing sigma (CPU, no GPU).

Recomputes the three registered arm pairs from the frozen artifact files
``scores_{q0,cand_search,bf16}_dev2.jsonl`` and prints the pairing sigma both
for the binary axis (TF accuracy) and for the continuous ``gold_prob``, so the
pilot can quote 17's 0.204 / 0.227 / 0.269 against a fresh recomputation.

Usage:
    python sigma_recheck.py --artifacts <minicpm5-confirm-power-20260910 dir> \
        --out <json>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ARMS = ("q0", "cand_search", "bf16")
PAIRS = (("bf16", "cand_search"), ("q0", "cand_search"), ("q0", "bf16"))


def load(artifacts: Path, arm: str, episodes: dict[str, str]) -> dict[str, dict]:
    rows = [json.loads(line) for line in (artifacts / f"scores_{arm}_dev2.jsonl").open()]
    for row in rows:
        row["episode"] = episodes[row["id"]]
    return {row["id"]: row for row in rows}


def episode_axis(rows: dict[str, dict], key: str) -> dict[str, float]:
    buckets: dict[str, list[float]] = {}
    for row in rows.values():
        if key == "tf_correct":
            value = 1.0 if row["tf_correct"] else 0.0
        elif key == "gold_prob":
            value = float(np.exp(row["tf_gold_logprob"])) if row.get("tf_gold_logprob") is not None else None
        else:
            value = row.get(key)
        if value is None:
            continue
        buckets.setdefault(row["episode"], []).append(value)
    return {episode: float(np.mean(values)) for episode, values in buckets.items()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", required=True)
    parser.add_argument("--taskset", required=True, help="frozen taskset with the episode field")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    artifacts = Path(args.artifacts).expanduser()
    episodes = {
        item["id"]: item["episode"]
        for item in (json.loads(line) for line in Path(args.taskset).expanduser().open())
    }
    table = {arm: load(artifacts, arm, episodes) for arm in ARMS}
    report = {"artifacts": str(artifacts), "arms": {}, "pairs": {}}

    for arm, rows in table.items():
        hits = [1.0 if row["tf_correct"] else 0.0 for row in rows.values()]
        logprobs = [row["tf_gold_logprob"] for row in rows.values() if row.get("tf_gold_logprob") is not None]
        report["arms"][arm] = {
            "items": len(rows),
            "tf_acc": float(np.mean(hits)),
            "gold_prob": float(np.exp(np.mean(logprobs))),
            "episodes": len({row["episode"] for row in rows.values()}),
        }

    for left, right in PAIRS:
        entry = {}
        for key in ("tf_correct", "gold_prob"):
            left_axis = episode_axis(table[left], key)
            right_axis = episode_axis(table[right], key)
            episodes = sorted(set(left_axis) & set(right_axis))
            diffs = np.asarray([left_axis[e] - right_axis[e] for e in episodes], dtype=np.float64)
            entry[key] = {
                "episodes": len(episodes),
                "mean": float(diffs.mean()),
                "sd": float(diffs.std(ddof=1)),
                "zero_fraction": float((diffs == 0).mean()),
                "min": float(diffs.min()),
                "max": float(diffs.max()),
            }
        report["pairs"][f"{left}-{right}"] = entry
    Path(args.out).expanduser().write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    print(json.dumps(report, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
