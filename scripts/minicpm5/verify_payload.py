#!/usr/bin/env python3
"""Independent reconstruction check of an NVFP4 payload.

Reads a compressed-tensors ``nvfp4-pack-quantized`` checkpoint with the QUASAR
primitives (no vLLM, no compressed-tensors loader), dequantises every quantised
Linear, and reports:

* per-layer relative Frobenius reconstruction error vs the BF16 source weight;
* how many E2M1 codes are the ``8`` (-0.0) representation of zero;
* whether the payload's own codes match a fresh RTN encode of the source weight
  (i.e. the payload is reproducible from the source + our encoder).

Usage:
    python verify_payload.py --payload <dir> --model <bf16 dir> --out <json>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open

from shadow_lab.quasar import e2m1_decode, e2m1_encode, unpack_e2m1

BLOCK = 16


def _load_layer(handle, prefix: str) -> dict[str, torch.Tensor]:
    tensors = {}
    for key in handle.keys():
        if key.startswith(prefix + "."):
            tensors[key[len(prefix) + 1 :]] = handle.get_tensor(key)
    return tensors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-layers", type=int, default=0)
    args = parser.parse_args()

    payload_dir = Path(args.payload).expanduser()
    model_dir = Path(args.model).expanduser()

    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16)
    state = model.state_dict()

    weight_file = payload_dir / "model.safetensors"
    if not weight_file.exists():
        candidates = sorted(payload_dir.glob("*.safetensors"))
        if not candidates:
            raise SystemExit(f"no safetensors in {payload_dir}")
        weight_file = candidates[0]

    results = []
    with safe_open(weight_file, framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        prefixes = sorted({key.rsplit(".", 1)[0] for key in keys})
        quantised = [
            prefix
            for prefix in prefixes
            if f"{prefix}.weight_packed" in keys
        ]
        if args.max_layers:
            quantised = quantised[: args.max_layers]
        print(f"[verify] {len(quantised)} quantised layers in {weight_file.name}")

        for prefix in quantised:
            tensors = _load_layer(handle, prefix)
            packed = tensors["weight_packed"]
            scale = tensors["weight_scale"].float()
            global_scale_stored = tensors["weight_global_scale"].float().reshape(())
            # compressed-tensors stores the *divisor*: actual = 1 / stored.
            global_scale = 1.0 / global_scale_stored

            codes = unpack_e2m1(packed)
            out_features, in_features = codes.shape[0], codes.shape[1]
            values = e2m1_decode(codes).reshape(out_features, in_features)
            scale = scale.reshape(out_features, in_features // BLOCK)
            effective = (scale * global_scale).repeat_interleave(BLOCK, dim=1)
            reconstructed = values * effective

            source = state[f"{prefix}.weight"].float()
            if tuple(source.shape) != tuple(reconstructed.shape):
                raise SystemExit(
                    f"{prefix}: shape {tuple(reconstructed.shape)} != source "
                    f"{tuple(source.shape)}"
                )
            error = (reconstructed - source).square().sum().sqrt()
            norm = source.square().sum().sqrt()
            relative = float(error / norm) if norm > 0 else None

            # Re-encode the source with the payload's own scales to test
            # reproducibility (RTN with the same block/global scales).
            reencoded = e2m1_encode(source / effective)
            code_mismatch = int((reencoded != codes).sum())
            zero_as_eight = int((codes == 8).sum())

            results.append(
                {
                    "layer": prefix,
                    "shape": [out_features, in_features],
                    "relative_frobenius_error": relative,
                    "max_abs_error": float((reconstructed - source).abs().max()),
                    "code_mismatch_vs_rtn": code_mismatch,
                    "code8_count": zero_as_eight,
                    "code_total": int(codes.numel()),
                    "weight_global_scale_stored": float(global_scale_stored),
                    "weight_global_scale_actual": float(global_scale),
                    "input_global_scale_stored": float(
                        tensors.get("input_global_scale", torch.tensor(float("nan")))
                        .float()
                        .reshape(())
                    ),
                    "block_scale_min": float(scale.min()),
                    "block_scale_max": float(scale.max()),
                }
            )

    relative_errors = [item["relative_frobenius_error"] for item in results]
    summary = {
        "payload": str(payload_dir),
        "source_model": str(model_dir),
        "quantised_layers": len(results),
        "weight_file": weight_file.name,
        "relative_frobenius_error": {
            "min": min(relative_errors),
            "max": max(relative_errors),
            "mean": sum(relative_errors) / len(relative_errors),
        },
        "code_mismatch_total": sum(item["code_mismatch_vs_rtn"] for item in results),
        "code8_total": sum(item["code8_count"] for item in results),
        "code_total": sum(item["code_total"] for item in results),
        "layers": results,
    }
    Path(args.out).expanduser().write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(
        {
            "quantised_layers": summary["quantised_layers"],
            "relative_frobenius_error": summary["relative_frobenius_error"],
            "code_mismatch_total": summary["code_mismatch_total"],
            "code8_total": summary["code8_total"],
            "code_total": summary["code_total"],
        },
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
