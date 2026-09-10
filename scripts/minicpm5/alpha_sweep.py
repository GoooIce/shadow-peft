#!/usr/bin/env python3
"""Freeze the centre-step gain alpha from the pilot's own recorded generations.

Ticket 09 froze the updater with ``u = alpha * g_hat + gamma * e`` and left
``alpha`` to be calibrated.  The pilot ran 5 generations at ``alpha = 1.0`` and
moved 26/11/0/0/0 coordinates over a touch set of 55/215/359/533/536 -- the
"small-step stagnation" of ticket 08's counter-example: the aggregated gradient
is a mean over the mirror pairs, so its per-coordinate size is small compared
with the E2M1 grid step, and most of the update is absorbed by quantisation
(frozen into the residual) instead of moving the centre.

This script replays the **recorded** generations exactly (populations, code
edits, realised displacements and scored fitness all come from the run's own
records) and re-accounts the centre step for a ladder of alphas:

* ``alpha = 1.0`` must reproduce the recorded ledgers -- that is the check that
  the replay is faithful, not a re-simulation;
* every other alpha is the same gradient with a different gain, so the
  reported cross-grid rate answers "does the update become a real numerical
  change at this gain?" without spending GPU time.

CPU only, no engine, no scoring: it reads the payload bytes and the records.

Usage:
    python alpha_sweep.py --model <q0 payload> --records <out/records> --out <json> \
        [--alphas 0.5,1,2,3,4,6,8,12,16]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import pilot_lib as pl


def _realized(blob: dict) -> dict[tuple[str, int], float]:
    """``"layer|flat" -> displacement`` records back into coordinate keys."""
    out = {}
    for key, value in blob.items():
        layer, flat = key.split("|")
        out[(layer, int(flat))] = float(value)
    return out


def load_generation(records_dir: Path, seed: int, groups, masks) -> list[dict]:
    """Rebuild every recorded generation: population, edits, fitness, pairs."""
    generations = []
    for path in sorted(records_dir.glob("gen*.json")):
        record = json.loads(path.read_text())
        generation = int(record["generation"])
        group_index = int(record["group_index"])
        pair_records: dict[int, dict[int, dict]] = {}
        for cand_path in sorted(records_dir.glob(f"cand_gen{generation:02d}_*.json")):
            blob = json.loads(cand_path.read_text())
            candidate = blob["candidate"]
            pair_records.setdefault(int(candidate["pair"]), {})[int(candidate["side"])] = candidate
        pairs = []
        realized_by_index: dict[int, dict[tuple[str, int], float]] = {}
        for pair_index in sorted(pair_records):
            sides = pair_records[pair_index]
            plus, minus = sides[0], sides[1]
            realized_by_index[int(plus["index"])] = _realized(plus["realized"])
            realized_by_index[int(minus["index"])] = _realized(minus["realized"])
            pairs.append({
                "pair": pair_index,
                "plus": {"index": int(plus["index"]), "fitness": float(plus["fitness"]["acc"]),
                         "delta_center": float(plus["delta_center"])},
                "minus": {"index": int(minus["index"]), "fitness": float(minus["fitness"]["acc"]),
                          "delta_center": float(minus["delta_center"])},
            })
        # check the recorded population is what the frozen proposal reproduces
        population = pl.propose_generation(
            groups, masks, generation=generation, group_index=group_index,
            pairs=int(record["pairs"]), edits_per_candidate=int(record["edits_per_candidate"]),
            seed=seed,
        )
        mine = {(candidate.pair, candidate.side): tuple(candidate.coordinates) for candidate in population}
        theirs = {
            (int(item["pair"]), int(item["side"])): tuple(tuple(entry) for entry in item["coordinates"])
            for item in record["population"]
        }
        generations.append({
            "generation": generation,
            "group_id": record["group_id"],
            "replay_population_matches": mine == theirs,
            "pairs": pairs,
            "realized": realized_by_index,
            "recorded_moved": {
                (item["layer"], int(item["flat"]))
                for item in record.get("residual_ledger", {}).get("moved_coordinates", [])
            },
            "recorded_ledger": {
                key: record["residual_ledger"].get(key)
                for key in ("moved", "touched", "rejected_beyond_grid",
                            "residual_clamped", "u_abs_max", "residual_abs_max")
            },
            "recorded_center_sha256_after": record.get("center_sha256_after"),
        })
    return generations


def replay(generations: list[dict], reader, alphas: list[float],
           base_hash_of) -> dict:
    """Full 5-generation replay for every alpha; returns ledgers per alpha."""
    report = {"alphas": {}, "alpha_one_reproduces_records": None}
    for alpha in alphas:
        state = pl.CentreState()
        rows = []
        faithful = True
        for entry in generations:
            aggregation = pl.aggregate(entry["pairs"], [entry["realized"][index]
                                                        for index in sorted(entry["realized"])])
            gradient = aggregation["gradient"]
            ledger = pl.apply_step(state, reader, gradient, alpha)
            moved = {(item["layer"], int(item["flat"])) for item in ledger["moved_coordinates"]}
            entry["replay_moved_match"] = moved == entry["recorded_moved"]
            if alpha == 1.0:
                faithful &= bool(entry["replay_moved_match"])
            gaps = np.asarray(ledger["realised_gaps"], dtype=np.float64)
            rows.append({
                "generation": entry["generation"],
                "group_id": entry["group_id"],
                "gradient_support": len(gradient),
                "touched": ledger["touched"],
                "moved": ledger["moved"],
                "cross_grid_rate": (ledger["moved"] / ledger["touched"]) if ledger["touched"] else None,
                "unmoved_below_projection": ledger["unmoved_below_projection"],
                "rejected_beyond_grid": ledger["rejected_beyond_grid"],
                "residual_clamped": ledger["residual_clamped"],
                "u_abs_max": ledger["u_abs_max"],
                "residual_abs_max": ledger["residual_abs_max"],
                "residual_coordinates": sum(len(bucket) for bucket in state.residual.values()),
                "realised_gap_mean": ledger["realised_gap_mean"],
                "realised_gap_median": ledger["realised_gap_median"],
                "realised_gaps": [round(float(gap), 4) for gap in gaps.tolist()],
                "multi_step_moves": int((gaps >= 4.0).sum()) if gaps.size else 0,
                "replay_moved_match": entry.get("replay_moved_match"),
                "center_sha256_after": state.sha256(base_hash_of),
            })
        touched = sum(row["touched"] for row in rows)
        moved = sum(row["moved"] for row in rows)
        all_gaps = [gap for row in rows for gap in row["realised_gaps"]]
        report["alphas"][f"{alpha:g}"] = {
            "alpha": alpha,
            "touched_total": touched,
            "moved_total": moved,
            "cross_grid_rate": (moved / touched) if touched else None,
            "moved_per_generation": [row["moved"] for row in rows],
            "cross_grid_rate_per_generation": [row["cross_grid_rate"] for row in rows],
            "rejected_beyond_grid": sum(row["rejected_beyond_grid"] for row in rows),
            "residual_clamped": sum(row["residual_clamped"] for row in rows),
            "u_abs_max": max(row["u_abs_max"] for row in rows),
            "residual_abs_max": max(row["residual_abs_max"] for row in rows),
            "realised_gaps": all_gaps,
            "multi_step_moves": sum(row["multi_step_moves"] for row in rows),
            "replay_moved_match_all": all(row["replay_moved_match"] for row in rows),
            "rows": rows,
        }
        if alpha == 1.0:
            report["alpha_one_reproduces_records"] = bool(faithful)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--records", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--alphas", default="0.5,1,2,3,4,6,8,12,16")
    parser.add_argument("--recorded-alpha", type=float, default=None,
                        help="the alpha the recorded run actually used (witness row)")
    parser.add_argument("--seed", type=int, default=20260910)
    args = parser.parse_args()

    started = time.time()
    reader = pl.PayloadReader(Path(args.model).expanduser())
    layers = [key[: -len(".weight_packed")] for key in sorted(reader._keys)
              if key.endswith(".weight_packed")]
    groups = pl.build_groups(layers)
    masks, census = pl.group_census(groups, reader)

    def base_hash_of(layer: str) -> str:
        import hashlib

        packed = reader._handle.get_tensor(f"{layer}.weight_packed")
        return hashlib.sha256(packed.reshape(-1).numpy().tobytes()).hexdigest()[:24]

    generations = load_generation(Path(args.records).expanduser(), args.seed, groups, masks)
    alphas = [float(item) for item in args.alphas.split(",")]
    report = replay(generations, reader, alphas, base_hash_of)
    # the run recorded its ledgers at its own alpha: the replay of that alpha
    # must reproduce them, and that row is the faithfulness witness
    recorded_alpha = str(float(args.recorded_alpha)).rstrip("0").rstrip(".") if args.recorded_alpha else None
    witness = recorded_alpha if recorded_alpha in report["alphas"] else next(iter(report["alphas"]))
    report.update({
        "model": args.model,
        "records": args.records,
        "generations": len(generations),
        "recorded_alpha": recorded_alpha,
        "replay_witness_alpha": witness,
        "population_replay_all_match": all(g["replay_population_matches"] for g in generations),
        "recorded_centers": [g["recorded_center_sha256_after"] for g in generations],
        "witness_centers": [row["center_sha256_after"]
                            for row in report["alphas"][witness]["rows"]],
        "recorded_ledgers": [g["recorded_ledger"] for g in generations],
        "seconds": round(time.time() - started, 2),
        "protocol": pl.PROTOCOL,
        "group_protocol": pl.GROUP_PROTOCOL,
    })
    Path(args.out).expanduser().write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n"
    )
    print(json.dumps({
        "population_replay_all_match": report["population_replay_all_match"],
        "alpha_one_reproduces_records": report["alpha_one_reproduces_records"],
        "witness_alpha": witness,
        "witness_moved": report["alphas"][witness]["moved_per_generation"],
        "recorded_moved": [g["recorded_ledger"]["moved"] for g in generations],
        "table": {key: {"moved": blob["moved_total"], "touched": blob["touched_total"],
                        "rate": round(blob["cross_grid_rate"], 3) if blob["cross_grid_rate"] else None,
                        "rejects": blob["rejected_beyond_grid"],
                        "clamps": blob["residual_clamped"],
                        "u_max": round(blob["u_abs_max"], 3),
                        "multi_step": blob["multi_step_moves"]}
                  for key, blob in report["alphas"].items()},
        "seconds": report["seconds"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
