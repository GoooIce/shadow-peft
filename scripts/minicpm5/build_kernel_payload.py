#!/usr/bin/env python3
"""Export a compressed-tensors NVFP4 payload to the engine's kernel layout.

This is the ES-side exporter the refit gate must validate: it turns the
checkpoint tensors (per-shard ``weight_packed`` / ``weight_scale`` /
``*_global_scale``) into the exact parameter set a running vLLM engine holds
after ``process_weights_after_loading``:

* fused q/k/v -> ``qkv_proj`` and gate/up -> ``gate_up_proj`` (row concat);
* ``weight`` padded with ``pad_nvfp4_weight_for_cutlass``;
* ``weight_scale`` swizzled with ``swizzle_blockscale`` (128x4 layout);
* ``weight_global_scale`` / ``input_global_scale`` reciprocated (the payload
  stores divisors) and reduced with ``max`` over fused shards, exactly as
  ``CompressedTensorsW4A4Fp4.process_weights_after_loading`` does;
* ``alpha = input_global_scale * weight_global_scale``,
  ``input_global_scale_inv = 1 / input_global_scale``.

Usage:
    python build_kernel_payload.py --payload <dir> --out <kernel.safetensors> \
        [--manifest <json>]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
    pad_nvfp4_weight_for_cutlass,
    swizzle_blockscale,
)

FUSED_ORDER = {
    "qkv_proj": ("q_proj", "k_proj", "v_proj"),
    "gate_up_proj": ("gate_proj", "up_proj"),
}


def _engine_layer(prefix: str) -> tuple[str, str]:
    """Map a checkpoint prefix to (engine layer prefix, shard kind)."""
    parts = prefix.split(".")
    leaf = parts[-1]
    if leaf in ("q_proj", "k_proj", "v_proj"):
        return ".".join(parts[:-1] + ["qkv_proj"]), leaf
    if leaf in ("gate_proj", "up_proj"):
        return ".".join(parts[:-1] + ["gate_up_proj"]), leaf
    return prefix, leaf


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--manifest")
    args = parser.parse_args()

    payload_dir = Path(args.payload).expanduser()
    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with safe_open(payload_dir / "model.safetensors", framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        prefixes = sorted({key.rsplit(".", 1)[0] for key in keys})
        int8 = [prefix for prefix in prefixes
                if f"{prefix}.weight_scale" in keys
                and f"{prefix}.weight_packed" not in keys]
        if int8:
            # W8A8 int8 kernel format: fuse row-wise (q/k/v, gate/up), then
            # transpose to the live [in, out] layout the triton kernel keeps
            # (verified bitwise on 2026-09-10: live weight == payload.T).
            # Scales upcast to the live fp32 [out, 1] per-channel form.
            groups: dict[str, list[str]] = {}
            for prefix in int8:
                engine, _ = _engine_layer(prefix)
                groups.setdefault(engine, []).append(prefix)
            tensors: dict[str, torch.Tensor] = {}
            rows = []
            for engine, shards in sorted(groups.items()):
                leaf = engine.split(".")[-1]
                order = FUSED_ORDER.get(leaf)
                if order:
                    shards = sorted(
                        shards, key=lambda name: order.index(name.split(".")[-1])
                    )
                weight_parts = [handle.get_tensor(f"{shard}.weight") for shard in shards]
                scale_parts = [handle.get_tensor(f"{shard}.weight_scale").float()
                               for shard in shards]
                fused = torch.cat(weight_parts, dim=0)          # [out_fused, in]
                live = fused.t().contiguous()                    # [in, out]
                scale = torch.cat(scale_parts, dim=0).reshape(-1, 1).contiguous()
                tensors[f"{engine}.weight"] = live
                tensors[f"{engine}.weight_scale"] = scale
                rows.append(
                    {
                        "engine_layer": engine,
                        "shards": shards,
                        # logical payload-order shape: verify_layout tiles row
                        # offsets against weight_shape[0], which counts OUT rows
                        "weight_shape": [int(fused.shape[0]), int(fused.shape[1])],
                        "live_weight_shape": list(live.shape),
                        "weight_scale_shape": list(scale.shape),
                        "grid": "int8-uniform",
                    }
                )
            save_file(tensors, str(out_path))
            manifest = {
                "payload": str(payload_dir),
                "kind": "int8-w8a8",
                "tensors": len(tensors),
                "rows": rows,
            }
            if args.manifest:
                Path(args.manifest).expanduser().write_text(
                    json.dumps(manifest, indent=1, sort_keys=True) + "\n")
            print(f"int8 kernel payload: {len(tensors)} tensors -> {out_path}")
            return 0

        quant = [prefix for prefix in prefixes if f"{prefix}.weight_packed" in keys]

        groups: dict[str, list[str]] = {}
        for prefix in quant:
            engine, _ = _engine_layer(prefix)
            groups.setdefault(engine, []).append(prefix)

        tensors: dict[str, torch.Tensor] = {}
        rows = []
        for engine, shards in sorted(groups.items()):
            leaf = engine.split(".")[-1]
            order = FUSED_ORDER.get(leaf)
            if order:
                shards = sorted(
                    shards, key=lambda name: order.index(name.split(".")[-1])
                )
            packed_parts = []
            scale_parts = []
            stored_wgs = []
            stored_igs = []
            for shard in shards:
                packed_parts.append(handle.get_tensor(f"{shard}.weight_packed"))
                scale_parts.append(handle.get_tensor(f"{shard}.weight_scale"))
                stored_wgs.append(
                    float(handle.get_tensor(f"{shard}.weight_global_scale").float().reshape(()))
                )
                stored_igs.append(
                    float(handle.get_tensor(f"{shard}.input_global_scale").float().reshape(()))
                )
            packed = torch.cat(packed_parts, dim=0)
            scale = torch.cat(scale_parts, dim=0)
            weight, padding_cols = pad_nvfp4_weight_for_cutlass(packed)
            swizzled = swizzle_blockscale(scale).cpu()
            # Reciprocal-scale chain must be computed in float32 *tensor*
            # arithmetic: vLLM does ``1.0 / tensor`` (fp32 division) and then
            # ``igs * wgs`` (fp32 multiply).  Computing the same expression in
            # Python doubles and casting at the end differs in the last ulp.
            wgs_stored_t = torch.tensor(max(stored_wgs), dtype=torch.float32)
            igs_stored_t = torch.tensor(max(stored_igs), dtype=torch.float32)
            weight_global_scale = 1.0 / wgs_stored_t
            input_global_scale = 1.0 / igs_stored_t
            tensors[f"{engine}.weight"] = weight.contiguous()
            tensors[f"{engine}.weight_scale"] = swizzled.contiguous()
            tensors[f"{engine}.weight_global_scale"] = weight_global_scale
            tensors[f"{engine}.input_global_scale"] = input_global_scale
            tensors[f"{engine}.input_global_scale_inv"] = igs_stored_t
            tensors[f"{engine}.alpha"] = input_global_scale * weight_global_scale
            rows.append(
                {
                    "engine_layer": engine,
                    "shards": shards,
                    "padding_cols": int(padding_cols),
                    "weight_shape": list(weight.shape),
                    "weight_scale_shape": list(swizzled.shape),
                    "weight_global_scale": float(weight_global_scale),
                    "input_global_scale": float(input_global_scale),
                    "alpha": float(input_global_scale * weight_global_scale),
                    "stored_weight_global_scale": float(wgs_stored_t),
                    "stored_input_global_scale": float(igs_stored_t),
                }
            )

    save_file(tensors, str(out_path))
    manifest = {
        "payload": str(payload_dir),
        "kernel_file": str(out_path),
        "layers": len(rows),
        "tensors": len(tensors),
        "bytes": out_path.stat().st_size,
        "rows": rows,
    }
    if args.manifest:
        Path(args.manifest).expanduser().write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
    print(json.dumps(
        {key: manifest[key] for key in ("payload", "kernel_file", "layers", "tensors", "bytes")},
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
