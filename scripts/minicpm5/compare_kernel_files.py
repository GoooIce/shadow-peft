#!/usr/bin/env python3
"""Bitwise comparison of two kernel-format NVFP4 safetensors files.

Used for the ticket-11 weight-path readback: one file is dumped from a live
engine (``refit_worker.dump_kernel_params``), the other is exported offline from
the payload by ``build_kernel_payload.py``.

Usage:
    python compare_kernel_files.py --left <engine dump> --right <offline export> \
        --out <json> [--tol 0.0]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open


def _sha256(tensor: torch.Tensor) -> str:
    cpu = tensor.detach().contiguous()
    if cpu.dim() == 0:
        cpu = cpu.reshape(1)
    return hashlib.sha256(cpu.view(torch.uint8).numpy().tobytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tol", type=float, default=0.0)
    args = parser.parse_args()

    rows = []
    mismatch = 0
    with safe_open(args.left, framework="pt", device="cpu") as left, safe_open(
        args.right, framework="pt", device="cpu"
    ) as right:
        left_keys = set(left.keys())
        right_keys = set(right.keys())
        for key in sorted(left_keys | right_keys):
            if key not in left_keys or key not in right_keys:
                rows.append({"key": key, "status": "missing"})
                mismatch += 1
                continue
            a = left.get_tensor(key)
            b = right.get_tensor(key)
            same_shape = tuple(a.shape) == tuple(b.shape)
            same_dtype = a.dtype == b.dtype
            equal = bool(same_shape and same_dtype and torch.equal(a, b))
            max_abs = None
            if same_shape and a.dtype.is_floating_point and b.dtype.is_floating_point:
                max_abs = float((a.float() - b.float()).abs().max()) if a.numel() else 0.0
            if not equal and max_abs is not None and max_abs <= args.tol:
                equal = True
            if not equal:
                mismatch += 1
            rows.append(
                {
                    "key": key,
                    "equal": equal,
                    "same_shape": same_shape,
                    "same_dtype": same_dtype,
                    "left_dtype": str(a.dtype),
                    "right_dtype": str(b.dtype),
                    "left_sha256": _sha256(a),
                    "right_sha256": _sha256(b),
                    "max_abs_diff": max_abs,
                }
            )
    summary = {
        "left": args.left,
        "right": args.right,
        "tensors": len(rows),
        "mismatches": mismatch,
        "all_bitwise_equal": mismatch == 0,
        "rows": rows,
    }
    Path(args.out).expanduser().write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(
        {key: summary[key] for key in ("left", "right", "tensors", "mismatches", "all_bitwise_equal")},
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
