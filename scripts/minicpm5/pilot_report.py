#!/usr/bin/env python3
"""Summarise a pilot run: numbers for the note and the 50-candidate extrapolation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, help="pilot output dir")
    parser.add_argument("--json")
    args = parser.parse_args()
    out = Path(args.out).expanduser()

    history = [json.loads(line) for line in (out / "history.jsonl").open()]
    records = {}
    for path in sorted((out / "records").glob("gen*.json")):
        record = json.loads(path.read_text())
        records[record["generation"]] = record

    report = {"history": history, "generations": []}
    for generation, record in sorted(records.items()):
        candidates = record["candidates"]
        ledger = record["residual_ledger"]
        resolution = record["resolution"]
        entry = {
            "generation": generation,
            "group": record["group_id"],
            "candidates": len(candidates),
            "edits_per_candidate": record["edits_per_candidate"],
            "centre_acc": record["center_fitness"]["acc"],
            "centre_gold_prob": record["center_fitness"]["gold_prob"],
            "candidate_acc": [c["fitness"]["acc"] for c in candidates],
            "candidate_gold_prob": [c["fitness"]["gold_prob"] for c in candidates],
            "candidate_delta_center": [c["delta_center"] for c in candidates],
            "candidate_sd": [c["sd"] for c in candidates],
            "aggregation": record["aggregation"],
            "ledger": {key: ledger[key] for key in
                       ("touched", "moved", "rejected_beyond_grid", "residual_clamped",
                        "unmoved_below_projection", "realised_gap_mean", "realised_gap_median",
                        "u_abs_max", "residual_abs_max")},
            "moved_coordinates": len(ledger["moved_coordinates"]),
            "resolution": {key: resolution[key] for key in
                           ("episodes", "candidate_sd_mean", "candidate_sd_min", "candidate_sd_max",
                            "mirror_sd_mean", "mirror_mean_abs", "all_pairs_n",
                            "all_pairs_sd_mean", "all_pairs_sd_median", "all_pairs_sd_min",
                            "all_pairs_sd_max")},
            "confirm": record["confirm"]["fitness"]["acc"] if record.get("confirm") else None,
            "kl": record.get("kl"),
            "hard_gate": {key: record["hard_gate"].get(key) for key in
                          ("tensors", "mismatches", "all_bitwise_equal", "seconds")},
            "centre_sha256_before": record["center_sha256_before"],
            "centre_sha256_after": record["center_sha256_after"],
            "wall_seconds": record["wall_seconds"],
            "timings": record["timings"],
        }
        report["generations"].append(entry)

    # -- candidate-level paired-difference distribution ----------------------
    all_deltas = [d for entry in report["generations"] for d in entry["candidate_delta_center"]]
    report["paired_diff_distribution"] = {
        "n": len(all_deltas),
        "mean": float(np.mean(all_deltas)),
        "sd": float(np.std(all_deltas, ddof=1)),
        "min": float(np.min(all_deltas)),
        "max": float(np.max(all_deltas)),
        "quantiles": {str(q): float(np.quantile(all_deltas, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)},
        "sd_of_candidate_sd": float(np.mean([entry["resolution"]["candidate_sd_mean"]
                                             for entry in report["generations"]])),
    }

    # -- 50-candidate extrapolation ------------------------------------------
    score_times, refit_times = [], []
    for entry in report["generations"]:
        timings = records[entry["generation"]]["timings"]
        n = len(records[entry["generation"]]["candidates"])
        score_times.append(timings["candidate_score_total"] / n)
        refit_times.append(timings["candidate_patch_total"] / max(1, n - 1))
    per_gen = [entry["wall_seconds"] for entry in report["generations"]]
    report["cost"] = {
        "candidate_tf_seconds_mean": float(np.mean(score_times)),
        "candidate_refit_seconds_mean": float(np.mean(refit_times)),
        "candidate_refit_seconds_min": float(np.min(refit_times)),
        "generation_wall_seconds_mean": float(np.mean(per_gen)) if per_gen else None,
        "generation_wall_seconds": per_gen,
        "gate_seconds": [entry["hard_gate"]["seconds"] for entry in report["generations"]],
        "confirm_seconds": [entry["confirm"] for entry in report["generations"]],
        "candidates_per_generation": [entry["candidates"] for entry in report["generations"]],
    }
    print(json.dumps({k: v for k, v in report.items() if k != "generations"}, indent=1))
    for entry in report["generations"]:
        print(json.dumps({
            "gen": entry["generation"], "group": entry["group"],
            "centre_acc": entry["centre_acc"], "centre_gp": entry["centre_gold_prob"],
            "moved": entry["ledger"]["moved"], "rejected": entry["ledger"]["rejected_beyond_grid"],
            "unmoved": entry["ledger"]["unmoved_below_projection"],
            "qualified_pairs": entry["aggregation"]["qualified_pairs"],
            "dropped": entry["aggregation"]["dropped_pairs"],
            "u_max": entry["ledger"]["u_abs_max"],
            "residual_max": entry["ledger"]["residual_abs_max"],
            "mirror_sd": entry["resolution"]["mirror_sd_mean"],
            "all_pairs_sd": entry["resolution"]["all_pairs_sd_mean"],
            "gate": entry["hard_gate"], "kl": entry["kl"] and entry["kl"]["generation_kl_error"],
            "wall": entry["wall_seconds"],
        }, ensure_ascii=False))
    if args.json:
        Path(args.json).expanduser().write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
