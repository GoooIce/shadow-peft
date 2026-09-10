#!/usr/bin/env python3
"""Attribute every hard-gate difference to a centre coordinate or to "outside".

Ticket 18 requires that the re-verification leaves **no unattributed item**: a
difference reported by :mod:`compare_kernel_values` must be classified as either

* a coordinate of the centre delta of that generation (a patched coordinate the
  export wrote and the engine did not, or the other way round),
* a coordinate of a candidate of that generation (a rollback that did not land),
* a coordinate of an *earlier* generation's candidate set (a stale patch), or
* ``outside`` -- which is a finding, not a rounding error.

The engine layout is inverted here (``engine (module, row, col) -> checkpoint
(layer, flat)``) with the same :func:`pilot_lib.engine_layout` the runner uses,
so the attribution cannot drift from the patch path.

Usage:
    python attribute_gate_diffs.py --model <q0 payload> --out <json> \
        --compare <compare.json> [--compare ...] [--records <out/records>]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pilot_lib as pl


def inverse_layout(layout: dict[str, tuple[str, int]], shapes: dict[str, tuple[int, int]]):
    """``module -> [(layer, row_offset, rows, in_features, out_features), ...]``."""
    table: dict[str, list[tuple[str, int, int, int, int]]] = {}
    for layer, (module, offset) in layout.items():
        out_features, in_features = shapes[layer]
        table.setdefault(module, []).append((layer, offset, out_features, in_features, out_features))
    return table


def coordinate_of(module_table, module: str, position: int):
    for layer, offset, rows, in_features, _out in module_table.get(module, ()):
        row, col = divmod(position, in_features)
        if offset <= row < offset + rows:
            return layer, (row - offset) * in_features + col
    return None, None


def load_records(records_dir: Path) -> dict:
    """Per-generation candidate coordinate sets and moved-coordinate sets."""
    out = {"generations": {}, "candidates": {}, "all_candidate_coordinates": set(),
           "all_moved_coordinates": set()}
    for path in sorted(records_dir.glob("gen*.json")):
        record = json.loads(path.read_text())
        generation = int(record["generation"])
        moved = {
            (item["layer"], int(item["flat"]))
            for item in record.get("residual_ledger", {}).get("moved_coordinates", [])
        }
        out["generations"][generation] = {
            "group_id": record["group_id"],
            "moved": moved,
            "center_sha256_after": record.get("center_sha256_after"),
        }
        out["all_moved_coordinates"] |= moved
    for path in sorted(records_dir.glob("cand_gen*.json")):
        blob = json.loads(path.read_text())
        generation = int(blob["generation"])
        coordinate_set = {
            (layer, int(flat)) for layer, flat in blob["candidate"]["coordinates"]
        }
        out["candidates"][(generation, blob["candidate"]["index"])] = coordinate_set
        out["all_candidate_coordinates"] |= coordinate_set
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="q0 payload (layout source)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--compare", action="append", required=True,
                        help="compare_kernel_values JSON (repeatable)")
    parser.add_argument("--records", help="<out>/records directory of the run")
    args = parser.parse_args()

    reader = pl.PayloadReader(Path(args.model).expanduser())
    layers = [key[: -len(".weight_packed")] for key in sorted(reader._keys)
              if key.endswith(".weight_packed")]
    shapes = {layer: reader.shape(layer) for layer in layers}
    layout = pl.engine_layout(layers, shapes)
    module_table = inverse_layout(layout, shapes)
    records = load_records(Path(args.records).expanduser()) if args.records else None

    report = {"compares": [], "totals": Counter(), "unattributed": []}
    for compare_path in args.compare:
        blob = json.loads(Path(compare_path).expanduser().read_text())
        name = Path(compare_path).name
        generation = None
        for token in name.replace(".json", "").split("_"):
            if token.startswith("gen") and token[3:].isdigit():
                generation = int(token[3:])
        tally = Counter()
        examples = []
        for row in blob["rows"]:
            key = row["key"]
            if row.get("status") in ("missing", "structural") or not row.get("raw_code_diffs"):
                continue
            module = key.rsplit(".", 1)[0]
            for diff in row.get("diff_examples", []):
                layer, flat = coordinate_of(module_table, module, int(diff["flat"]))
                klass = "outside"
                if layer is not None:
                    coordinate = (layer, int(flat))
                    if records is not None:
                        if generation is not None and coordinate in records["generations"].get(
                            generation, {}
                        ).get("moved", set()):
                            klass = "generation_moved"
                        elif coordinate in records["all_candidate_coordinates"]:
                            klass = "candidate_coordinate"
                        elif coordinate in records["all_moved_coordinates"]:
                            klass = "earlier_generation_moved"
                        else:
                            klass = "outside"
                    else:
                        klass = "unknown_no_records"
                tally[diff["class"]] += 1
                tally[klass] += 1
                if klass == "outside" and len(report["unattributed"]) < 200:
                    report["unattributed"].append({
                        "compare": name, "key": key, "flat": diff["flat"],
                        "left_code": diff["left_code"], "right_code": diff["right_code"],
                        "class": diff["class"], "layer": layer,
                    })
                if len(examples) < 40:
                    examples.append({
                        "key": key, "flat": diff["flat"], "layer": layer,
                        "flat_in_layer": flat, "left_code": diff["left_code"],
                        "right_code": diff["right_code"], "class": diff["class"],
                        "attribution": klass,
                    })
        report["compares"].append({
            "compare": name,
            "generation": generation,
            "tally": dict(tally),
            "tensors_with_diffs": sum(
                1 for row in blob["rows"] if row.get("raw_code_diffs")
            ),
            "examples": examples,
            "pass_value": blob.get("pass_value"),
        })
        report["totals"].update(tally)
    report["totals"] = dict(report["totals"])
    report["records_dir"] = str(args.records)
    Path(args.out).expanduser().write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n"
    )
    print(json.dumps({"totals": report["totals"],
                      "unattributed": len(report["unattributed"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
