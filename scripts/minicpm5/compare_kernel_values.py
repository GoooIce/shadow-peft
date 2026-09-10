#!/usr/bin/env python3
"""Value-semantics kernel comparison for the ticket-05/11 hard gate (ticket 18).

The pilot (ticket 14, 2026-09-10) showed that a **code-position** comparison is
the wrong hard gate for NVFP4: the E2M1 grid has two codes for zero, and the
external ``compressed-tensors`` payload really emits code 8 (negative zero) --
69,669,613 nibbles = 3.5% of q0.  Two payloads that decode to the same weights
can therefore differ in code positions forever.

This tool reports three calibres over the same file pair, keeping them apart:

``raw_code_diffs``
    differing nibbles / bytes, exactly what :mod:`compare_kernel_files` calls a
    mismatch (the old gate).
``value_diffs``
    coordinates whose **decoded E2M1 value** differs (``VALUE[a] != VALUE[b]``),
    which for the E2M1 grid is the same as ``canonical_code_diffs`` below.  For
    the float side tensors (``weight_scale``, the global scales, ``alpha``) a
    value difference is any non-bitwise-equal element.
``canonical_code_diffs``
    code positions after folding code 8 onto code 0 on **both** sides; reported
    as a separate, independently computed count.
``zero8_diffs``
    the subset of raw differences that is exactly the ``0 <-> 8`` representation
    channel -- numerically empty, and the reason the old gate failed.

The compare is byte level and only decodes the bytes that actually differ, so
it stays cheap on the 2.04 GiB kernel payloads (1008 tensors).

Usage:
    python compare_kernel_values.py --left <engine dump> --right <offline export> \
        --out <json> [--max-diffs 20000] [--tol 0.0]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

#: E2M1 code -> value (codes 0 and 8 are both zero: negative zero)
E2M1_VALUES = (
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
)
PACKED_SUFFIXES = (".weight", ".weight_packed")


def _bytes(tensor: torch.Tensor) -> np.ndarray:
    """Flat uint8 view (scalars included: a float32 scalar is four bytes)."""
    cpu = tensor.detach().cpu().contiguous().reshape(-1)
    return cpu.view(torch.uint8).numpy()


def _is_packed(key: str) -> bool:
    return key.endswith(PACKED_SUFFIXES)


def nibbles(array: np.ndarray) -> np.ndarray:
    """Flat nibble view of a packed uint8 array (even index = low nibble)."""
    out = np.empty(array.size * 2, dtype=np.uint8)
    out[0::2] = array & 0x0F
    out[1::2] = array >> 4
    return out


def canonical_code_diffs(left: np.ndarray, right: np.ndarray) -> int:
    """Third calibre: compare code positions after folding 8 -> 0 on both sides.

    Computed directly from the nibble arrays (not inferred from the value
    calibre), so the identity ``canonical == value`` is an empirical result of
    this tool rather than an assumption baked into it.
    """
    a = nibbles(left)
    b = nibbles(right)
    a = np.where(a == 8, 0, a).astype(np.uint8)
    b = np.where(b == 8, 0, b).astype(np.uint8)
    return int((a != b).sum())


def compare_tensor(key: str, left: torch.Tensor, right: torch.Tensor,
                   max_diffs: int, tol: float) -> dict:
    row = {
        "key": key,
        "same_shape": tuple(left.shape) == tuple(right.shape),
        "same_dtype": left.dtype == right.dtype,
        "dtype": str(left.dtype),
        "elements": int(left.numel()),
    }
    if not (row["same_shape"] and row["same_dtype"]):
        row["status"] = "structural"
        return row
    a, b = _bytes(left), _bytes(right)
    raw_mask = a != b
    raw_diffs = int(raw_mask.sum())
    row["raw_byte_diffs"] = raw_diffs
    row["raw_code_diffs"] = None
    row["value_diffs"] = None
    row["canonical_code_diffs"] = None
    row["zero8_diffs"] = 0
    row["equal_value_diffs"] = 0
    row["status"] = "equal" if raw_diffs == 0 else "differs"
    if raw_diffs == 0:
        # every calibre is zero by construction
        row["raw_code_diffs"] = 0
        row["value_diffs"] = 0
        row["canonical_code_diffs"] = 0
        return row

    if _is_packed(key):
        # nibble layout: even flat index = low nibble (compressed-tensors)
        positions = np.flatnonzero(raw_mask)
        zero8 = value = 0
        examples = []
        for position in positions.tolist():
            left_byte, right_byte = int(a[position]), int(b[position])
            for shift, side in ((0, "low"), (4, "high")):
                x = (left_byte >> shift) & 0x0F
                y = (right_byte >> shift) & 0x0F
                if x == y:
                    continue
                flat_index = 2 * position + (1 if shift else 0)
                vx, vy = E2M1_VALUES[x], E2M1_VALUES[y]
                if vx == vy:
                    zero8 += 1
                    klass = "zero8"
                else:
                    value += 1
                    klass = "value"
                if len(examples) < max_diffs:
                    examples.append({
                        "flat": int(flat_index),
                        "byte": int(position),
                        "side": side,
                        "left_code": x,
                        "right_code": y,
                        "left_value": vx,
                        "right_value": vy,
                        "class": klass,
                    })
        row["raw_code_diffs"] = zero8 + value
        row["value_diffs"] = value
        row["zero8_diffs"] = zero8
        row["equal_value_diffs"] = zero8
        row["canonical_code_diffs"] = canonical_code_diffs(a, b)
        row["diff_examples"] = examples
        row["diff_examples_truncated"] = (zero8 + value) > len(examples)
    else:
        # float side tensors: bitwise equality is the right calibre; a value
        # difference is any bitwise-inequal element (with an optional tolerance)
        a_f = left.detach().cpu().contiguous().float().reshape(-1).numpy()
        b_f = right.detach().cpu().contiguous().float().reshape(-1).numpy()
        width = max(1, a.size // max(1, left.numel()))
        differing_elements = int(
            (a.reshape(-1, width) != b.reshape(-1, width)).any(axis=1).sum()
        )
        row["raw_code_diffs"] = differing_elements
        delta = np.abs(a_f - b_f)
        value = int((delta > tol).sum())
        row["value_diffs"] = value
        row["zero8_diffs"] = 0
        # float-side analogue of the 0<->8 channel: elements that differ bitwise
        # (e.g. +0.0 against -0.0) without differing in value
        row["equal_value_diffs"] = differing_elements - value
        row["canonical_code_diffs"] = value
        row["max_abs_diff"] = float(delta.max()) if delta.size else 0.0
        if value:
            worst = int(np.argmax(delta))
            row["diff_examples"] = [{
                "flat": worst,
                "left": float(a_f[worst]),
                "right": float(b_f[worst]),
                "abs_diff": float(delta[worst]),
                "class": "value",
            }]
            row["diff_examples_truncated"] = value > 1
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", required=True, help="engine dump")
    parser.add_argument("--right", required=True, help="offline export")
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-diffs", type=int, default=20000)
    parser.add_argument("--tol", type=float, default=0.0)
    args = parser.parse_args()

    rows = []
    totals = {
        "raw_code_diffs": 0,
        "value_diffs": 0,
        "canonical_code_diffs": 0,
        "zero8_diffs": 0,
        "equal_value_diffs": 0,
    }
    structural = 0
    raw_tensors = 0
    value_tensors = 0
    notes = []
    with safe_open(args.left, framework="pt", device="cpu") as left, safe_open(
        args.right, framework="pt", device="cpu"
    ) as right:
        left_keys, right_keys = set(left.keys()), set(right.keys())
        for key in sorted(left_keys | right_keys):
            if key not in left_keys or key not in right_keys:
                rows.append({"key": key, "status": "missing"})
                structural += 1
                continue
            row = compare_tensor(key, left.get_tensor(key), right.get_tensor(key),
                                 args.max_diffs, args.tol)
            if row["status"] == "structural":
                structural += 1
                notes.append(f"{key}: shape/dtype mismatch")
            else:
                if row["raw_code_diffs"]:
                    raw_tensors += 1
                if row["value_diffs"]:
                    value_tensors += 1
                    notes.append(
                        f"{key}: {row['value_diffs']} value diffs "
                        f"({row['zero8_diffs']} 0<->8 in the same tensor)"
                    )
                for name in totals:
                    totals[name] += int(row.get(name) or 0)
            rows.append(row)

    summary = {
        "left": args.left,
        "right": args.right,
        "tensors": len(rows),
        "structural_mismatches": structural,
        "tensors_with_raw_diffs": raw_tensors,
        "tensors_with_value_diffs": value_tensors,
        "notes": notes[:50],
        "rows": rows,
        "e2m1_values": list(E2M1_VALUES),
        "calibres": {
            "raw_code_diffs": "code positions as stored (the pre-ticket-18 gate)",
            "value_diffs": "decoded E2M1 value differs per coordinate (the frozen gate)",
            "canonical_code_diffs": "code positions after folding 8->0 on both sides",
            "zero8_diffs": "raw differences that are the numerically empty 0<->8 channel",
        },
    }
    summary.update(totals)
    summary["pass_value"] = structural == 0 and totals["value_diffs"] == 0
    summary["pass_canonical"] = structural == 0 and totals["canonical_code_diffs"] == 0
    summary["pass_raw"] = structural == 0 and totals["raw_code_diffs"] == 0
    Path(args.out).expanduser().write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({
        "tensors": summary["tensors"],
        "structural": structural,
        "raw_code_diffs": totals["raw_code_diffs"],
        "zero8_diffs": totals["zero8_diffs"],
        "value_diffs": totals["value_diffs"],
        "canonical_code_diffs": totals["canonical_code_diffs"],
        "tensors_with_value_diffs": value_tensors,
        "pass_value": summary["pass_value"],
        "pass_raw": summary["pass_raw"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
