"""
Layerwise analysis: ternary 2-bit PTQ (ours) vs fp16 original.

For every quantized module (Qwen3-1.7B architecture, group_size=128, ternary
2-bit {-1,0,+1}), computes:

  - code distribution (fraction of {-1, 0, +1} codes, plus code-3 leakage check)
  - scale/bias statistics
  - dequantized-weight stats vs fp16 original (std ratio)
  - relative L2 error and cosine similarity to fp16

Optionally compares against a Bonsai ternary checkpoint (requires MLX to load
the MLX-format safetensors; pass --bonsai-path).

Outputs CSV + markdown summary under experiment/analysis_2bit/.

Run: .venv/bin/python experiment/analyze_2bit_ternary.py
     .venv/bin/python experiment/analyze_2bit_ternary.py --model /path/to/local/Qwen3-1.7B
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM

from shadow_peft.quantization import (
    dequantize_2bit_ternary,
    quantize_2bit_ternary,
)

GROUP_SIZE = 128
OUT_DIR = Path(__file__).parent / "analysis_2bit"
DTYPE = torch.float16

PROJ_TYPES = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
]


def layer_metrics(name: str, w_fp16: torch.Tensor, refine_iters: int) -> dict:
    """Compute ternary 2-bit metrics for one module."""
    packed, scales, biases = quantize_2bit_ternary(
        w_fp16, group_size=GROUP_SIZE, refine_iters=refine_iters
    )
    deq = dequantize_2bit_ternary(packed, scales, biases, group_size=GROUP_SIZE)

    # Unpack codes to compute distribution
    shifts = 2 * torch.arange(16, dtype=torch.int32)
    codes = ((packed.unsqueeze(-1) >> shifts) & 0b11).reshape(-1)
    code_counts = torch.bincount(codes, minlength=4).float()
    total = code_counts.sum()
    code_fracs = (code_counts / total).tolist()

    w32 = w_fp16.float()
    d32 = deq.float()

    rel_l2 = (w32 - d32).norm() / w32.norm()
    cos = (w32.flatten() @ d32.flatten()) / (w32.norm() * d32.norm())

    return {
        "name": name,
        "shape": "x".join(map(str, w_fp16.shape)),
        "refine_iters": refine_iters,
        "code_neg1_frac": code_fracs[0],   # code 0 -> -s_g
        "code_zero_frac": code_fracs[1],   # code 1 -> 0
        "code_pos1_frac": code_fracs[2],   # code 2 -> +s_g
        "code_3_leakage": code_fracs[3],   # should always be 0
        "scale_mean": scales.float().mean().item(),
        "scale_std": scales.float().std().item(),
        "fp16_weight_std": w32.std().item(),
        "deq_std": d32.std().item(),
        "deq_std_ratio": (d32.std() / w32.std()).item(),
        "rel_l2": rel_l2.item(),
        "cosine_fp16": cos.item(),
    }


def layer_sort_key(name: str) -> tuple:
    parts = name.split(".")
    if "layers" in parts:
        idx = int(parts[parts.index("layers") + 1])
        sub = ".".join(parts[parts.index("layers") + 2 :])
        return (1, idx, sub)
    return (0, 0, name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B", help="HF model id or local path")
    ap.add_argument("--refine-iters", type=int, default=8)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"loading {args.model} fp16 ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=DTYPE)

    # Collect all Linear/Embedding weights
    targets: list[tuple[str, torch.Tensor]] = []
    for name, module in model.named_modules():
        if hasattr(module, "weight") and module.weight is not None:
            w = module.weight
            if w.ndim == 2 and w.shape[-1] % GROUP_SIZE == 0:
                targets.append((name, w.detach()))

    targets.sort(key=lambda x: layer_sort_key(x[0]))
    print(f"{len(targets)} quantizable modules\n", flush=True)

    rows: list[dict] = []
    for name, w in targets:
        rows.append(layer_metrics(name, w, args.refine_iters))
        print(f"  {name}: cos={rows[-1]['cosine_fp16']:.4f} rel_l2={rows[-1]['rel_l2']:.4f}", flush=True)

    # ---- per-layer CSV ----
    fieldnames = list(rows[0].keys())
    with open(OUT_DIR / "per_layer_metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # ---- aggregated table by module type ----
    groups: dict[str, list[dict]] = {"embed_tokens": []}
    for r in rows:
        if "embed" in r["name"]:
            groups["embed_tokens"].append(r)
        else:
            for pt in PROJ_TYPES:
                if r["name"].endswith(pt):
                    groups.setdefault(pt, []).append(r)

    metric_keys = [
        k for k in fieldnames if k not in ("name", "shape", "refine_iters")
    ]
    lines = ["| module | " + " | ".join(metric_keys) + " |"]
    lines.append("|" + "---|" * (len(metric_keys) + 1))
    for gname, grows in groups.items():
        if not grows:
            continue
        means = {k: float(np.mean([r[k] for r in grows])) for k in metric_keys}
        lines.append(
            f"| {gname} ({len(grows)}) | "
            + " | ".join(f"{means[k]:.4g}" for k in metric_keys)
            + " |"
        )
    summary_md = "\n".join(lines)
    print("\n" + summary_md)
    (OUT_DIR / "summary_by_module_type.md").write_text(summary_md + "\n")

    # ---- global code distribution ----
    global_codes = {
        "code_neg1": float(np.mean([r["code_neg1_frac"] for r in rows])),
        "code_zero": float(np.mean([r["code_zero_frac"] for r in rows])),
        "code_pos1": float(np.mean([r["code_pos1_frac"] for r in rows])),
        "code_3_leak": float(np.max([r["code_3_leakage"] for r in rows])),
    }
    print(f"\nglobal code distribution: {global_codes}")

    print(f"\noutputs written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
