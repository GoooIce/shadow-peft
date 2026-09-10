#!/usr/bin/env python3
"""Which exact RTN recipe reproduces the llmcompressor INT8 payload codes?

CPU-only forensic probe.  For a handful of representative layers it computes the
int8 codes under several plausible recipes and reports the fraction of codes
that differ from the payload:

  R1  fp32 scale from fp32 source, fp32 divide, round
  R2  the *stored* (bf16) scale, fp32 divide, round
  R3  everything in bfloat16: scale = amax(bf16)/127 in bf16, bf16 divide, bf16 round
  R4  R3 but with scale taken from the file (cast to bf16)
  R5  fp32 scale, fp32 divide, clamp to [-127, 127] instead of [-128, 127]

The winner is the recipe that the note can state as "payload = source + this
recipe"; anything still differing is a tie-break or clamp artifact.

Usage:
    python int8_recipe_forensics.py --payload <dir> --model <bf16 dir> [--layers 12]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open


def codes_diff(a: torch.Tensor, b: torch.Tensor) -> tuple[int, int]:
    d = int((a != b).sum().item())
    return d, a.numel()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--layers", type=int, default=12)
    args = parser.parse_args()

    payload = Path(args.payload).expanduser()
    model = Path(args.model).expanduser()

    source: dict[str, torch.Tensor] = {}
    for path in sorted(model.glob("*.safetensors")):
        with safe_open(str(path), framework="pt") as handle:
            for key in handle.keys():
                if key.endswith(".weight"):
                    source[key] = handle.get_tensor(key)

    with safe_open(str(payload / "model.safetensors"), framework="pt") as handle:
        keys = [k for k in handle.keys() if k.endswith(".weight_scale")]
        step = max(1, len(keys) // args.layers)
        chosen = keys[::step][: args.layers]
        results = []
        for scale_key in chosen:
            base = scale_key[: -len(".weight_scale")]
            w_key = f"{base}.weight"
            w_q = handle.get_tensor(w_key)
            w_s_stored = handle.get_tensor(scale_key)
            src = source[w_key]
            s_fp32 = (src.to(torch.float32).abs().amax(dim=1, keepdim=True) / 127.0)

            def frac(a: torch.Tensor) -> float:
                d, n = codes_diff(a, w_q)
                return d / n

            r = {"layer": base, "codes": w_q.numel()}
            r["stored_scale_dtype"] = str(w_s_stored.dtype)

            # R1 fp32/fp32
            c1 = (src.to(torch.float32) / s_fp32).round().clamp(-127, 127).to(torch.int8)
            r["R1_fp32_scale_fp32_math"] = frac(c1)
            # R2 stored scale, fp32 math
            c2 = (src.to(torch.float32) / w_s_stored.to(torch.float32)).round() \
                .clamp(-127, 127).to(torch.int8)
            r["R2_stored_scale_fp32_math"] = frac(c2)
            # R3 bf16 end to end
            s_bf16 = (src.abs().amax(dim=1, keepdim=True) / 127.0).to(torch.bfloat16)
            c3 = torch.round(src / s_bf16).clamp(-127, 127).to(torch.int8)
            r["R3_bf16_math_bf16_scale"] = frac(c3)
            # R4 bf16 math with the stored scale
            c4 = torch.round(src / w_s_stored.to(torch.bfloat16)) \
                .clamp(-127, 127).to(torch.int8)
            r["R4_bf16_math_stored_scale"] = frac(c4)
            # R5 fp32 math, symmetric clamp at -127
            c5 = (src.to(torch.float32) / s_fp32).round().clamp(-127.0, 127.0) \
                .to(torch.int8)
            r["R5_fp32_math_clamp127"] = frac(c5)
            # R6 bf16 math, no clamp at all before cast
            c6 = torch.round(src / s_bf16).to(torch.int8)
            r["R6_bf16_math_no_clamp"] = frac(c6)
            results.append(r)
            print(json.dumps({k: (round(v, 6) if isinstance(v, float) else v)
                              for k, v in r.items()}), flush=True)

    summary: dict[str, float] = {}
    for key in results[0]:
        if key.startswith("R"):
            summary[key] = sum(r[key] * r["codes"] for r in results) / sum(
                r["codes"] for r in results
            )
    print()
    print("weighted mean differing fraction per recipe:")
    for key, value in sorted(summary.items(), key=lambda kv: kv[1]):
        print(f"  {key:<32} {value:.8f}")
    out = Path(payload) / ".." / "int8_recipe_forensics.json"
    out.write_text(json.dumps({"per_layer": results, "weighted": summary},
                              indent=2, sort_keys=True) + "\n")
    print(f"[forensics] wrote {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
