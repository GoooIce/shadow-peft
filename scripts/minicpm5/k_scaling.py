#!/usr/bin/env python3
"""How does the aggregated centre step scale with the sparse edit count k?

Ticket 18 freezes the per-group sparse edit count, so the relation between ``k``
(the number of rank edits per candidate) and the *size* of the aggregated
direction ``g_hat`` has to be measured rather than assumed.  The recorded pilot
generations carry everything needed -- realised displacements per candidate and
the scored fitness per pair -- so this is a records-only CPU job: for each
generation the same aggregation rule (rank shape reward, unqualified mirror
pairs dropped whole, zero-sum coefficients, ``sum(c*delta)/sum|c|``) is replayed
on prefixes of the recorded coordinate set of length k, and the resulting
per-coordinate gradient is summarised.

The empirical exponent ``p`` in ``|g_hat| ~ k^p`` (``p = 0.5`` for independent
signs) is what lets the frozen alpha be extrapolated to a group whose edit count
differs from the recorded one.

Usage:
    python k_scaling.py --records <out/records> --out <json> [--ks 8,16,24,32]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pilot_lib as pl


def load_pairs(records_dir: Path) -> list[dict]:
    generations = []
    for path in sorted(records_dir.glob("gen*.json")):
        record = json.loads(path.read_text())
        generation = int(record["generation"])
        pair_records: dict[int, dict[int, dict]] = {}
        for cand_path in sorted(records_dir.glob(f"cand_gen{generation:02d}_*.json")):
            candidate = json.loads(cand_path.read_text())["candidate"]
            pair_records.setdefault(int(candidate["pair"]), {})[int(candidate["side"])] = candidate
        pairs = []
        for pair_index in sorted(pair_records):
            sides = pair_records[pair_index]
            entry = {"pair": pair_index}
            for side, key in ((0, "plus"), (1, "minus")):
                candidate = sides[side]
                entry[key] = {
                    "index": int(candidate["index"]),
                    "fitness": float(candidate["fitness"]["acc"]),
                    "delta_center": float(candidate["delta_center"]),
                    "coordinates": [tuple(item) for item in candidate["coordinates"]],
                    "realized": {
                        tuple(item[0].split("|")): float(item[1])
                        for item in candidate["realized"].items()
                    },
                }
            pairs.append(entry)
        generations.append({
            "generation": generation,
            "group_id": record["group_id"],
            "tau": float(record.get("tau", 0.02)),
            "pairs": pairs,
            "recorded_qualified": int(
                record.get("aggregation", {}).get("qualified_pairs", 0)
            ),
        })
    return generations


def gradient_for(generation: dict, k: int) -> tuple[dict, dict]:
    """Aggregate the qualified pairs over the first ``k`` coordinates of each."""
    qualified, dropped = [], []
    for pair in generation["pairs"]:
        ok = all(
            pair[side]["delta_center"] >= -generation["tau"] for side in ("plus", "minus")
        )
        (qualified if ok else dropped).append(pair)
    diffs = [pair["plus"]["fitness"] - pair["minus"]["fitness"] for pair in qualified]
    rewards = pl.rank_shape_reward(diffs)
    gradient: dict[tuple[str, int], float] = {}
    weight_total = 0.0
    for pair, reward in zip(qualified, rewards):
        weight_total += 2.0 * abs(reward)
        for side, coefficient in (("plus", reward), ("minus", -reward)):
            coordinates = pair[side]["coordinates"][:k]
            for coordinate in coordinates:
                layer, flat = coordinate
                value = pair[side]["realized"].get((layer, str(flat)))
                if value is None:
                    continue
                gradient[(layer, int(flat))] = (
                    gradient.get((layer, int(flat)), 0.0) + coefficient * value
                )
    if weight_total:
        gradient = {key: value / weight_total for key, value in gradient.items()}
    return gradient, {"qualified": len(qualified), "dropped": len(dropped),
                      "weight_total": weight_total}


def summarise(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    ordered = sorted(abs(value) for value in values)
    def quantile(q):
        return ordered[min(len(ordered) - 1, int(q * len(ordered)))]
    return {
        "n": len(ordered),
        "abs_mean": math.fsum(ordered) / len(ordered),
        "abs_median": quantile(0.5),
        "abs_p90": quantile(0.9),
        "abs_max": ordered[-1],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--ks", default="8,16,24,32")
    args = parser.parse_args()

    ks = [int(item) for item in args.ks.split(",")]
    generations = load_pairs(Path(args.records).expanduser())
    report = {"generations": [], "recorded": args.records, "ks": ks}
    for generation in generations:
        row = {"generation": generation["generation"], "group_id": generation["group_id"],
               "recorded_qualified_pairs": generation["recorded_qualified"], "by_k": {}}
        for k in ks:
            gradient, info = gradient_for(generation, k)
            row["by_k"][str(k)] = {
                "qualified_pairs": info["qualified"],
                "dropped_pairs": info["dropped"],
                "weight_total": info["weight_total"],
                "support": len(gradient),
                "gradient": summarise(list(gradient.values())),
            }
        # empirical exponent between the two smallest k rungs
        small, large = ks[0], ks[-1]
        a = row["by_k"][str(small)]["gradient"].get("abs_mean")
        b = row["by_k"][str(large)]["gradient"].get("abs_mean")
        row["exponent_abs_mean"] = (
            math.log(b / a) / math.log(large / small) if a and b else None
        )
        report["generations"].append(row)
    Path(args.out).expanduser().write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({
        "ks": ks,
        "per_generation": [{
            "generation": row["generation"],
            "group": row["group_id"],
            "support": {k: row["by_k"][k]["support"] for k in row["by_k"]},
            "abs_mean": {k: round(row["by_k"][k]["gradient"].get("abs_mean", 0), 5)
                         for k in row["by_k"]},
            "abs_max": {k: round(row["by_k"][k]["gradient"].get("abs_max", 0), 4)
                        for k in row["by_k"]},
            "exponent": row["exponent_abs_mean"],
        } for row in report["generations"]],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
