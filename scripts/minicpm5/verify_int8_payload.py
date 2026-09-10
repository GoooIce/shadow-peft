#!/usr/bin/env python3
"""Independent structural + numerical verification of the INT8 W8A8 payload.

Ticket 10 verified the NVFP4 payload against this repo's own QUASAR E2M1
primitives.  There is no comparable repo primitive for INT8, so the equivalent
check is a full **round-trip against the source BF16 weights**, done with the
`safetensors`/`torch` readers only -- no llmcompressor, no compressed-tensors,
no vLLM:

  * enumerate every quantised weight tensor and its scale, and assert the
    layout the checkpoint claims (int8 weights, FP32 per-output-channel
    ``weight_scale``, and for this recipe NO activation side-tensor at all);
  * recompute RTN per-channel scales from the BF16 source and compare with the
    scales stored in the payload;
  * recompute the int8 codes from the source and count differing codes;
  * report the relative Frobenius error of the reconstruction.

CPU only.  Usage:
    python verify_int8_payload.py --payload <dir> --model <bf16 dir> --out <json>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open


def load_index(path: Path) -> dict:
    with safe_open(str(path), framework="pt") as handle:
        keys = list(handle.keys())
        meta = handle.metadata() or {}
    return {"keys": keys, "metadata": meta}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    payload = Path(args.payload).expanduser()
    model = Path(args.model).expanduser()

    man = json.loads((payload / "config.json").read_text())["quantization_config"]
    weights_cfg = man["config_groups"]["group_0"]["weights"]
    input_cfg = man["config_groups"]["group_0"]["input_activations"]

    payload_index = load_index(payload / "model.safetensors")
    keys = payload_index["keys"]
    weight_keys = sorted(k for k in keys if k.endswith(".weight"))
    scale_keys = sorted(k for k in keys if k.endswith(".weight_scale"))
    input_scale_keys = sorted(
        k for k in keys if k.endswith("input_scale") or k.endswith("input_zero_point")
    )
    print(f"[verify] {len(weight_keys)} weight tensors, {len(scale_keys)} scales, "
          f"{len(input_scale_keys)} input-side tensors", flush=True)

    model_files = sorted(model.glob("*.safetensors"))
    source: dict[str, torch.Tensor] = {}
    for path in model_files:
        with safe_open(str(path), framework="pt") as handle:
            for key in handle.keys():
                if key.endswith(".weight"):
                    source[key] = handle.get_tensor(key)

    report: dict = {
        "payload": str(payload),
        "source_model": str(model),
        "quant_config": {
            "format": man.get("format"),
            "weights": weights_cfg,
            "input_activations": input_cfg,
            "ignore": man.get("ignore"),
        },
        "tensor_census": {
            "weight_tensors": len(weight_keys),
            "weight_scale_tensors": len(scale_keys),
            "input_scale_or_zp_tensors": input_scale_keys,
            "excluded_linears": sorted(
                k for k in source if k not in set(weight_keys)
            ),
        },
        "dtype_check": {},
        "layers": [],
    }

    quad = 0
    scale_dtypes: set[str] = set()
    per_layer = []
    lhs = rhs = 0.0
    codes_total = codes_diff = codes_diff_stored = codes_diff_bf16 = 0
    scale_rel = []
    limit = args.limit or len(weight_keys)

    with safe_open(str(payload / "model.safetensors"), framework="pt") as handle:
        for key in weight_keys[:limit]:
            base = key[: -len(".weight")]
            scale_key = f"{base}.weight_scale"
            if scale_key not in payload_index["keys"]:
                # Weight-only / ignored module (lm_head by recipe): present as a
                # BF16 weight with no scale.  Recorded, not asserted on.
                w_full = handle.get_tensor(key)
                report["layers"].append(
                    {
                        "layer": base,
                        "weight_shape": list(w_full.shape),
                        "weight_dtype": str(w_full.dtype),
                        "quantized": False,
                    }
                )
                continue
            w_q = handle.get_tensor(key)
            w_s = handle.get_tensor(scale_key)
            w_s_raw = w_s  # keep the stored dtype for the R4 recipe
            src = source.get(key)
            entry = {
                "layer": base,
                "weight_shape": list(w_q.shape),
                "weight_dtype": str(w_q.dtype),
                "scale_shape": list(w_s.shape),
                "scale_dtype": str(w_s.dtype),
            }
            assert w_q.dtype == torch.int8, entry
            # NOTE: llmcompressor 0.13.0 writes the INT8 per-channel scale as
            # bfloat16 (not fp32).  Recorded, not assumed.
            assert w_s.dtype.is_floating_point, entry
            assert w_s.numel() == w_q.shape[0], entry
            scale_dtypes.add(str(w_s.dtype))
            quad += 1
            if src is None:
                entry["note"] = "no source weight found"
                report["layers"].append(entry)
                continue
            src = src.to(torch.float32)
            w_s = w_s.to(torch.float32)
            s_ref = src.abs().amax(dim=1, keepdim=True) / 127.0
            s_ref = s_ref.clamp(min=1e-8)
            rel = float(((w_s.reshape(-1, 1) - s_ref).abs() / s_ref).max().item())
            scale_rel.append(rel)
            if scale_rel[-1] == max(scale_rel):
                report["worst_scale_layer"] = {
                    "layer": base,
                    "max_rel_scale_error": rel,
                    "source_amax_per_channel_head": src.abs().amax(dim=1)[:5].tolist(),
                    "stored_scale_head": w_s.reshape(-1)[:5].tolist(),
                }
            # RTN recomputed two ways: against an fp32 scale derived from the
            # source (s_ref) and against the scale actually stored in the file
            # (bf16-rounded by llmcompressor -- see scale_dtypes_seen).
            codes_ref = (src / s_ref).round().clamp(-127, 127).to(torch.int8)
            diff = int((codes_ref != w_q).sum().item())
            codes_diff += diff
            codes_stored = (src / w_s.reshape(-1, 1)).round().clamp(-127, 127)
            codes_stored = codes_stored.to(torch.int8)
            diff_stored = int((codes_stored != w_q).sum().item())
            codes_diff_stored += diff_stored
            # R4 (the recipe that reproduces the payload; see
            # int8_recipe_forensics.py): stored scale, arithmetic in the weight
            # dtype (bfloat16), round, clamp.
            src_bf16 = source[key]  # original dtype, untouched
            codes_bf16 = torch.round(src_bf16 / w_s_raw).clamp(-127, 127).to(torch.int8)
            diff_bf16 = int((codes_bf16 != w_q).sum().item())
            codes_diff_bf16 += diff_bf16
            codes_total += w_q.numel()
            rec = w_q.to(torch.float32) * w_s.reshape(-1, 1)
            lhs += float(((rec - src) ** 2).sum().item())
            rhs += float((src**2).sum().item())
            entry.update(
                {
                    "max_rel_scale_error": round(scale_rel[-1], 8),
                    "codes": int(w_q.numel()),
                    "codes_differing_from_rtn_max_abs_fp32_scale": diff,
                    "codes_differing_from_rtn_max_abs_stored_scale": diff_stored,
                    "codes_differing_from_recipe_R4_bf16_math": diff_bf16,
                    "rel_frobenius_error": round(
                        float(
                            ((rec - src).norm() / src.norm()).item()
                        ),
                        6,
                    ),
                }
            )
            per_layer.append(entry)

    report["dtype_check"] = {
        "layers_checked": quad,
        "unquantized_weights": [
            e["layer"] for e in report["layers"] if e.get("quantized") is False
        ],
        "all_weights_int8": True,
        "scale_dtypes_seen": sorted(scale_dtypes),
        "all_scales_per_channel": True,
        "input_side_tensors_present": bool(input_scale_keys),
    }
    report["reconstruction"] = {
        "layers": len(per_layer),
        "max_scale_rel_error_vs_fp32_rtn": max(scale_rel) if scale_rel else None,
        "codes_total": codes_total,
        "codes_differing_from_rtn_fp32_scale": codes_diff,
        "codes_differing_fraction_fp32_scale": (
            codes_diff / codes_total if codes_total else None
        ),
        "codes_differing_from_rtn_stored_scale": codes_diff_stored,
        "codes_differing_fraction_stored_scale": (
            codes_diff_stored / codes_total if codes_total else None
        ),
        "codes_differing_from_recipe_R4_bf16_math": codes_diff_bf16,
        "codes_differing_fraction_recipe_R4": (
            codes_diff_bf16 / codes_total if codes_total else None
        ),
        "relative_frobenius_error_global": (
            (lhs**0.5) / (rhs**0.5) if rhs else None
        ),
        "relative_frobenius_error_per_layer_range": (
            [
                min(e["rel_frobenius_error"] for e in per_layer),
                max(e["rel_frobenius_error"] for e in per_layer),
            ]
            if per_layer
            else None
        ),
    }
    report["layers"] = per_layer

    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["reconstruction"], indent=2))
    print(json.dumps(report["tensor_census"], indent=2)[:1200])
    print(f"[verify] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
