#!/usr/bin/env python3
"""Build a candidate NVFP4 payload (compressed-tensors ``nvfp4-pack-quantized``)
from the BF16 source plus a QUASAR clip genome.

The output directory is a drop-in replacement for the delivered q0 payload:
same config.json/tokenizer, same BF16 tensors, re-quantised Linear weights.
That is what the ES search would hand to the engine, so the refit gate can be
measured on *real* candidates instead of a no-op reload of q0.

Usage:
    python make_candidate_payload.py --q0 <payload dir> --source <bf16 dir> \
        --out <dir> --kind clip --factor 0.60 --label clip060
    python make_candidate_payload.py ... --kind search --factors 0.55,0.7,0.85,1.0
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from shadow_lab.quasar import (
    QuasarNVFP4Config,
    reconstruct_nvfp4,
    reconstruct_nvfp4_with_clip,
)

FUSED_GROUPS = (
    ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"),
    ("mlp.gate_proj", "mlp.up_proj"),
)


def _fp8_scale(value: torch.Tensor) -> torch.Tensor:
    """Round to the E4M3 block-scale grid (same convention as quasar)."""
    finfo = torch.finfo(torch.float8_e4m3fn)
    return value.float().clamp(min=finfo.tiny, max=finfo.max).to(torch.float8_e4m3fn)

COPY_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--q0", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--kind", required=True, choices=("clip", "search"))
    parser.add_argument("--factor", type=float, default=1.0)
    parser.add_argument("--factors", default="0.55,0.7,0.85,1.0")
    parser.add_argument("--label", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    q0_dir = Path(args.q0).expanduser()
    source_dir = Path(args.source).expanduser()
    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in COPY_FILES:
        shutil.copy2(q0_dir / name, out_dir / name)

    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(source_dir, dtype=torch.bfloat16)
    state = model.state_dict()

    device = torch.device(args.device)
    quantised: dict[str, dict[str, torch.Tensor]] = {}
    bf16_tensors: dict[str, torch.Tensor] = {}
    rows = []
    started = time.perf_counter()
    with safe_open(q0_dir / "model.safetensors", framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        prefixes = sorted({key.rsplit(".", 1)[0] for key in keys})
        quant_prefixes = [
            prefix for prefix in prefixes if f"{prefix}.weight_packed" in keys
        ]
        if args.limit:
            quant_prefixes = quant_prefixes[: args.limit]
        for prefix in prefixes:
            if prefix in quant_prefixes:
                continue
            for key in keys:
                if key.startswith(prefix + "."):
                    bf16_tensors[key] = handle.get_tensor(key)
        print(f"[candidate] {len(quant_prefixes)} quantised layers", flush=True)

        # Pass 1: reconstruct every layer, keeping the *effective* (absolute)
        # block scale so fused groups can be harmonised afterwards.
        effective: dict[str, torch.Tensor] = {}
        tensor_scales: dict[str, torch.Tensor] = {}
        for index, prefix in enumerate(quant_prefixes):
            source = state[f"{prefix}.weight"].to(device=device, dtype=torch.float32)
            input_global_scale = handle.get_tensor(f"{prefix}.input_global_scale")
            if args.kind == "clip":
                result = reconstruct_nvfp4_with_clip(source, float(args.factor))
            else:
                factors = tuple(float(item) for item in args.factors.split(","))
                result = reconstruct_nvfp4(
                    source,
                    config=QuasarNVFP4Config(
                        clip_factors=factors, reconstruction_group_size=16
                    ),
                )
            # NVFP4Weight keeps the block structure in ``codes``; the
            # properties reshape/pack with the compressed-tensors conventions.
            packed = result.weight_packed.cpu()
            scale = result.weight_scale.cpu().float()
            tensor_scale = result.tensor_scale.reshape(1).cpu().float()
            effective[prefix] = scale * tensor_scale
            tensor_scales[prefix] = tensor_scale
            quantised[prefix] = {
                "weight_packed": packed,
                "input_global_scale": input_global_scale,
            }
            rows.append(
                {
                    "layer": prefix,
                    "weighted_error": float(result.weighted_error),
                    "tensor_scale": float(tensor_scale),
                }
            )
            if (index + 1) % 42 == 0:
                print(f"[candidate] {index + 1}/{len(quant_prefixes)} layers "
                      f"({time.perf_counter() - started:.1f}s)", flush=True)
            del source

        # Pass 2: vLLM's compressed-tensors PWAL reduces the per-shard global
        # scales with ``max`` (i.e. takes the *smallest* tensor scale) and does
        # NOT rescale the other shards' block scales.  A checkpoint must
        # therefore share one global scale per fused layer (llmcompressor
        # harmonises q/k/v and gate/up).  Re-derive each shard's E4M3 block
        # scale against the group's tensor scale; the dequantised weights are
        # unchanged, only the representation is.
        group_scale: dict[str, float] = {}
        members: dict[str, list[str]] = {}
        for prefix in quantised:
            parts = prefix.split(".")
            members.setdefault(".".join(parts[:3]), []).append(prefix)
        for layer, prefixes in members.items():
            for group in FUSED_GROUPS:
                group_prefixes = [
                    prefix
                    for prefix in prefixes
                    if ".".join(prefix.split(".")[3:]) in group
                ]
                if not group_prefixes:
                    continue
                fused_scale = max(
                    float(tensor_scales[prefix].reshape(()))
                    for prefix in group_prefixes
                )
                for prefix in group_prefixes:
                    group_scale[prefix] = fused_scale
        clamped = 0
        scale_stats = {}
        for prefix, payload in quantised.items():
            fused_scale = group_scale.get(prefix, float(tensor_scales[prefix].reshape(())))
            stored = _fp8_scale(effective[prefix] / fused_scale)
            clamped += int(
                (stored.float() <= torch.finfo(torch.float8_e4m3fn).tiny).sum()
            )
            payload["weight_scale"] = stored
            payload["weight_global_scale"] = torch.tensor(
                1.0 / fused_scale, dtype=torch.float32
            ).reshape(1)
            scale_stats[prefix] = {
                "tensor_scale": fused_scale,
                "stored_global_scale": 1.0 / fused_scale,
                "stored_scale_min": float(stored.float().min()),
                "stored_scale_max": float(stored.float().max()),
            }
        for row in rows:
            row.update(scale_stats[row["layer"]])

    tensors: dict[str, torch.Tensor] = dict(bf16_tensors)
    for prefix, payload in quantised.items():
        for name, tensor in payload.items():
            tensors[f"{prefix}.{name}"] = tensor.contiguous()
    out_file = out_dir / "model.safetensors"
    save_file(tensors, str(out_file))
    manifest = {
        "label": args.label,
        "kind": args.kind,
        "factor": args.factor,
        "factors": args.factors,
        "q0": str(q0_dir),
        "source": str(source_dir),
        "quantised_layers": len(quantised),
        "tensors": len(tensors),
        "bytes": out_file.stat().st_size,
        "seconds": round(time.perf_counter() - started, 2),
        "fused_harmonised_layers": len(group_scale),
        "block_scales_clamped_to_e4m3_min": clamped,
        "layers": rows,
    }
    (out_dir / "candidate_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(
        {key: manifest[key] for key in
         ("label", "kind", "quantised_layers", "tensors", "bytes", "seconds")},
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
