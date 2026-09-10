#!/usr/bin/env python3
"""Per-group sparse-edit calibration for the ticket-14 pilot.

Answers "how many sparse rank edits does a candidate need before the edit turns
into an **actual numerical change**?"  For every one of the 13 layer groups and
a ladder of edit counts it reports, from the real q0 payload:

* ``code_changes``        -- requested == applied nibble changes;
* ``numerical_changes``   -- changes whose E2M1 rank differs (the only channel
  that can move a dequantised weight); ``0 <-> 8`` equal-value rewrites are
  counted separately and are *not* numerical changes (ticket 08);
* ``payload_bytes``       -- packed bytes whose byte value really changed;
* ``realized_*``          -- |delta| in code-value units and in weight units;
* ``relative_weight_change`` -- ||delta w||_2 / ||W||_F of the touched layers;
* ``endpoint_steps``      -- coordinates sitting on the +/-6.0 grid endpoint
  (the rank step is clamped inward there: candidate-side endpoint saturation).

CPU only, no GPU, no engine.

Usage:
    python calibrate_edits.py --payload <q0 dir> --out <json> \
        [--ladder 4,8,16,32,64,128] [--seed 20260910]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import pilot_lib as pl


def _frobenius_sq(reader: pl.PayloadReader, layer: str) -> float:
    codes = reader.codes(layer).astype(np.int64)
    values = pl.VALUE[codes]
    scale = reader.block_scales(layer).astype(np.float64)
    global_scale = reader.global_scale(layer)
    out, in_features = codes.shape
    blocks = values.reshape(out, in_features // pl.BLOCK, pl.BLOCK)
    energy = (blocks**2).sum(axis=2)
    return float(((energy * (scale * global_scale) ** 2).sum()))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--ladder", default="4,8,16,32,64,128")
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--pairs", type=int, default=2, help="mirror pairs per ladder rung")
    args = parser.parse_args()

    started = time.time()
    reader = pl.PayloadReader(Path(args.payload).expanduser())
    payload_layers = [key[: -len(".weight_packed")] for key in sorted(reader._keys) if key.endswith(".weight_packed")]
    groups = pl.build_groups(payload_layers)
    masks, census = pl.group_census(groups, reader)

    # --- payload-wide facts that gate the 0<->8 channel ---------------------
    zero_blocks = 0
    for layer in payload_layers:
        zero_blocks += int((reader.block_scales(layer) <= 0).sum())
    # count code 8 *before* canonicalisation by re-reading the packed nibbles
    code8_total = 0
    for layer in payload_layers:
        packed = reader._handle.get_tensor(f"{layer}.weight_packed").numpy()
        code8_total += int(((packed >> 4) == 8).sum()) + int(((packed & 0x0F) == 8).sum())

    report = {
        "payload": str(args.payload),
        "protocol": pl.PROTOCOL,
        "group_protocol": pl.GROUP_PROTOCOL,
        "seed": args.seed,
        "ladder": [int(item) for item in args.ladder.split(",")],
        "payload_facts": {
            "layers": len(payload_layers),
            "code8_nibbles_raw": code8_total,
            "zero_scale_blocks": zero_blocks,
            "coordinates": census["totals"]["coordinates"],
            "movable": census["totals"]["movable"],
        },
        "groups": census["groups"],
        "rows": [],
    }

    frobenius_cache: dict[str, float] = {}

    def fro(layer: str) -> float:
        if layer not in frobenius_cache:
            frobenius_cache[layer] = _frobenius_sq(reader, layer)
        return frobenius_cache[layer]

    for group_index, group in enumerate(groups):
        for rung in report["ladder"]:
            population = pl.propose_generation(
                groups, masks, generation=0, group_index=group_index,
                pairs=args.pairs, edits_per_candidate=rung, seed=args.seed,
            )
            candidate = population[0]
            edits = candidate.edits(pl.CentreState(), reader)
            numerical = 0
            equal_value = 0
            endpoint = 0
            payload_bytes = 0
            code_abs = []
            weight_abs = []
            touched: dict[str, float] = {}
            for edit in edits:
                layer, flat = edit["layer"], edit["flat"]
                codes = reader.codes(layer).reshape(-1)
                old, new = int(codes[flat]), edit["new_code"]
                rank_move = abs(pl.rank_of(old) - pl.rank_of(new))
                value_move = pl.VALUE[new] - pl.VALUE[old]
                effective = float(reader.block_scales(layer).reshape(-1)[flat // pl.BLOCK]) * reader.global_scale(layer)
                if rank_move == 1 and value_move != 0:
                    numerical += 1
                elif value_move == 0:
                    equal_value += 1
                if abs(pl.VALUE[old]) == pl.MAX_VALUE:
                    endpoint += 1
                payload_bytes += 1
                code_abs.append(abs(float(value_move)))
                delta_weight = float(value_move) * effective
                weight_abs.append(abs(delta_weight))
                touched[layer] = touched.get(layer, 0.0) + delta_weight**2
            norm_sq = sum(fro(layer) for layer in touched)
            report["rows"].append(
                {
                    "group_id": group.group_id,
                    "requested_edits": rung,
                    "candidates_proposed": len(population),
                    "code_changes": len(edits),
                    "numerical_changes": numerical,
                    "equal_value_rewrites": equal_value,
                    "endpoint_steps": endpoint,
                    "payload_bytes": payload_bytes,
                    "realized_code_abs_mean": float(np.mean(code_abs)) if code_abs else None,
                    "realized_code_abs_max": float(np.max(code_abs)) if code_abs else None,
                    "realized_weight_abs_mean": float(np.mean(weight_abs)) if weight_abs else None,
                    "relative_weight_change": float(np.sqrt(sum(touched.values()) / norm_sq)) if norm_sq else None,
                    "touched_layers": len(touched),
                }
            )
    report["seconds"] = round(time.time() - started, 2)
    pl.json_dump(Path(args.out).expanduser(), report)
    print(json.dumps({key: report[key] for key in ("payload_facts", "seconds")}, indent=1))
    for row in report["rows"][:6]:
        print(json.dumps(row, ensure_ascii=False))
    print(f"[calibrate] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
