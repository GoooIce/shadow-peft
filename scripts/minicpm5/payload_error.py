#!/usr/bin/env python3
"""Reconstruction error of one NVFP4 payload vs the BF16 source weights.

Same quantity as QUASAR's ``weighted_error`` (sum of squared error per layer,
importance = 1) plus the relative Frobenius norm, so candidates can be ranked
without starting an engine.

Usage:
    python payload_error.py --payload <dir> --source <bf16 dir> --out <json>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open

from shadow_lab.quasar import e2m1_decode, unpack_e2m1

BLOCK = 16


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    payload_dir = Path(args.payload).expanduser()
    source_dir = Path(args.source).expanduser()

    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(source_dir, dtype=torch.bfloat16)
    state = model.state_dict()

    rows = []
    with safe_open(payload_dir / "model.safetensors", framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        prefixes = sorted({key.rsplit(".", 1)[0] for key in keys})
        quant = [prefix for prefix in prefixes if f"{prefix}.weight_packed" in keys]
        if args.limit:
            quant = quant[: args.limit]
        for prefix in quant:
            packed = handle.get_tensor(f"{prefix}.weight_packed")
            scale = handle.get_tensor(f"{prefix}.weight_scale").float()
            stored = handle.get_tensor(f"{prefix}.weight_global_scale").float().reshape(())
            global_scale = 1.0 / stored
            codes = unpack_e2m1(packed)
            out_features, in_features = codes.shape
            values = e2m1_decode(codes).reshape(out_features, in_features)
            effective = (scale * global_scale).repeat_interleave(BLOCK, dim=1)
            reconstructed = values * effective
            source = state[f"{prefix}.weight"].float()
            residual = reconstructed - source
            rows.append(
                {
                    "layer": prefix,
                    "sse": float(residual.square().sum()),
                    "relative_frobenius": float(
                        residual.square().sum().sqrt() / source.square().sum().sqrt()
                    ),
                }
            )

    sse_total = sum(row["sse"] for row in rows)
    summary = {
        "payload": str(payload_dir),
        "layers": len(rows),
        "sse_total": sse_total,
        "relative_frobenius_mean": sum(row["relative_frobenius"] for row in rows)
        / len(rows),
        "rows": rows,
    }
    Path(args.out).expanduser().write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(
        {
            "payload": summary["payload"],
            "layers": summary["layers"],
            "sse_total": summary["sse_total"],
            "relative_frobenius_mean": summary["relative_frobenius_mean"],
        },
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
