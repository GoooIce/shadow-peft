"""
Bitwise & layerwise analysis: naive 1-bit PTQ vs Bonsai trained 1-bit.

Compares, for every one of the 197 quantized modules shared by both models
(Qwen3-1.7B architecture, group_size=128, bits=1, uint32 LSB-first packing):

  - bit density (fraction of 1-bits), per bit position 0..31
  - degenerate groups (all-0 / all-1 bit groups, zero-scale groups)
  - scale/bias statistics
  - dequantized-weight statistics vs the fp16 original
  - relative L2 error and cosine similarity to the fp16 original

Outputs CSVs + PNG figures + stdout summary under experiment/analysis_1bit/.

Run: .venv/bin/python experiment/analyze_1bit_layers.py
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlx.core as mx
import numpy as np
from huggingface_hub import snapshot_download

GROUP_SIZE = 128
OUT_DIR = Path(__file__).parent / "analysis_1bit"

PROJ_TYPES = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
]


def load_weights(repo: str) -> dict[str, mx.array]:
    snap = Path(snapshot_download(repo))
    weights: dict[str, mx.array] = {}
    for shard in sorted(snap.glob("model*.safetensors")):
        weights.update(mx.load(str(shard)))
    return weights


def unpack_bits(packed: mx.array) -> mx.array:
    """[..., words] uint32 -> [..., words * 32] uint8 bits, LSB-first order."""
    words = packed.astype(mx.int32)
    shifts = mx.arange(32, dtype=mx.int32)
    bits = (words[..., None] >> shifts) & 1
    out_shape = bits.shape[:-2] + (bits.shape[-2] * 32,)
    return bits.reshape(out_shape).astype(mx.uint8)


def layer_metrics(name: str, w_fp16: mx.array, bonsai: dict[str, mx.array]) -> dict:
    """Compute all metrics for one quantized module; returns a flat dict."""
    w16 = w_fp16.astype(mx.float16)
    o_packed, o_s, o_b = mx.quantize(w16, group_size=GROUP_SIZE, bits=1)
    o_deq = mx.dequantize(o_packed, o_s, o_b, group_size=GROUP_SIZE, bits=1)

    b_packed, b_s, b_b = (
        bonsai[f"{name}.weight"],
        bonsai[f"{name}.scales"],
        bonsai[f"{name}.biases"],
    )
    b_deq = mx.dequantize(b_packed, b_s, b_b, group_size=GROUP_SIZE, bits=1)

    o_bits, b_bits = unpack_bits(o_packed), unpack_bits(b_packed)
    w32 = w_fp16.astype(mx.float32)
    o32, b32 = o_deq.astype(mx.float32), b_deq.astype(mx.float32)

    def group_stats(bits: mx.array) -> tuple[float, float]:
        """Fraction of all-0 and all-1 groups."""
        g = bits.reshape(bits.shape[0], -1, GROUP_SIZE)
        counts = g.sum(axis=-1)
        return (
            (counts == 0).astype(mx.float32).mean(),
            (counts == GROUP_SIZE).astype(mx.float32).mean(),
        )

    o_g0, o_g1 = group_stats(o_bits)
    b_g0, b_g1 = group_stats(b_bits)

    shifts = mx.arange(32, dtype=mx.int32)

    # Bonsai trimmed the vocab (151669 vs Qwen's 151936); compare on the
    # shared prefix whenever fp16-relative metrics involve its tensors.
    if b32.shape != w32.shape:
        rows = min(b32.shape[0], w32.shape[0])
        b_cmp, w_cmp_b = b32[:rows], w32[:rows]
    else:
        b_cmp, w_cmp_b = b32, w32

    def rel_l2(a: mx.array, ref: mx.array) -> mx.array:
        return mx.sqrt(((a - ref) ** 2).sum()) / mx.sqrt((ref**2).sum())

    def cos_sim(a: mx.array, ref: mx.array) -> mx.array:
        return (a * ref).sum() / (mx.sqrt((a**2).sum()) * mx.sqrt((ref**2).sum()))

    o_bp_arr = ((o_packed.astype(mx.int32)[..., None] >> shifts) & 1).astype(
        mx.float32
    ).mean(axis=tuple(range(o_packed.ndim)))
    b_bp_arr = ((b_packed.astype(mx.int32)[..., None] >> shifts) & 1).astype(
        mx.float32
    ).mean(axis=tuple(range(b_packed.ndim)))

    vals = [
        o_bits.astype(mx.float32).mean(),
        b_bits.astype(mx.float32).mean(),
        (o_s == 0).astype(mx.float32).mean(),
        (b_s == 0).astype(mx.float32).mean(),
        o_s.astype(mx.float32).mean(),
        o_s.astype(mx.float32).std(),
        b_s.astype(mx.float32).mean(),
        b_s.astype(mx.float32).std(),
        o_g0,
        o_g1,
        b_g0,
        b_g1,
        w32.std(),
        o32.std(),
        b32.std(),
        rel_l2(o32, w32),
        rel_l2(b_cmp, w_cmp_b),
        cos_sim(o32, w32),
        cos_sim(b_cmp, w_cmp_b),
        o_bp_arr,
        b_bp_arr,
    ]
    mx.eval(*vals)
    (
        o_density,
        b_density,
        o_zeros,
        b_zeros,
        o_smean,
        o_sstd,
        b_smean,
        b_sstd,
        o_g0,
        o_g1,
        b_g0,
        b_g1,
        w_std,
        o_std,
        b_std,
        o_rell2,
        b_rell2,
        o_cos,
        b_cos,
        o_bp_arr,
        b_bp_arr,
    ) = [float(v) if v.ndim == 0 else v for v in vals]
    o_bp, b_bp = np.array(o_bp_arr), np.array(b_bp_arr)

    nbits = int(o_bits.size)

    return {
        "name": name,
        "shape": "x".join(map(str, w_fp16.shape)),
        "nbits": nbits,
        "ours_bit_density": o_density,
        "bonsai_bit_density": b_density,
        "ours_zero_scale_frac": o_zeros,
        "bonsai_zero_scale_frac": b_zeros,
        "ours_scale_mean": o_smean,
        "ours_scale_std": o_sstd,
        "bonsai_scale_mean": b_smean,
        "bonsai_scale_std": b_sstd,
        "ours_all0_group_frac": o_g0,
        "ours_all1_group_frac": o_g1,
        "bonsai_all0_group_frac": b_g0,
        "bonsai_all1_group_frac": b_g1,
        "fp16_weight_std": w_std,
        "ours_deq_std": o_std,
        "bonsai_deq_std": b_std,
        "ours_rel_l2": o_rell2,
        "bonsai_rel_l2": b_rell2,
        "ours_cos_fp16": o_cos,
        "bonsai_cos_fp16": b_cos,
        "_ours_bitpos": o_bp,
        "_bonsai_bitpos": b_bp,
    }


def layer_sort_key(name: str) -> tuple:
    parts = name.split(".")
    if "layers" in parts:
        idx = int(parts[parts.index("layers") + 1])
        sub = ".".join(parts[parts.index("layers") + 2 :])
        return (1, idx, sub)
    return (0, 0, name)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("loading Qwen3-1.7B fp16 ...", flush=True)
    qwen = load_weights("Qwen/Qwen3-1.7B")
    print("loading Bonsai-1.7B-mlx-1bit ...", flush=True)
    bonsai = load_weights("prism-ml/Bonsai-1.7B-mlx-1bit")

    prefixes = sorted(
        (k[: -len(".scales")] for k in bonsai if k.endswith(".scales")),
        key=layer_sort_key,
    )
    print(f"{len(prefixes)} quantized modules\n", flush=True)

    rows: list[dict] = []
    for name in prefixes:
        rows.append(layer_metrics(name, qwen[f"{name}.weight"], bonsai))
        print(f"  {name}", flush=True)

    # ---- per-layer CSV ----
    fieldnames = [k for k in rows[0] if not k.startswith("_")]
    with open(OUT_DIR / "per_layer_metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r[k] for k in fieldnames})

    # ---- global per-bit-position density (weighted by layer size) ----
    total = sum(r["nbits"] for r in rows)
    o_bitpos = sum(r["_ours_bitpos"] * r["nbits"] for r in rows) / total
    b_bitpos = sum(r["_bonsai_bitpos"] * r["nbits"] for r in rows) / total
    with open(OUT_DIR / "bit_position_density.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["bit_position", "ours", "bonsai"])
        for i in range(32):
            writer.writerow([i, float(o_bitpos[i]), float(b_bitpos[i])])

    # ---- aggregated table by module type ----
    groups: dict[str, list[dict]] = {"embed_tokens": []}
    for r in rows:
        if "embed" in r["name"]:
            groups["embed_tokens"].append(r)
        else:
            for pt in PROJ_TYPES:
                if r["name"].endswith(pt):
                    groups.setdefault(pt, []).append(r)

    metric_keys = [k for k in fieldnames if k not in ("name", "shape", "nbits")]
    lines = ["| module | " + " | ".join(metric_keys) + " |"]
    lines.append("|" + "---|" * (len(metric_keys) + 1))
    for gname, grows in groups.items():
        means = {
            k: float(np.mean([r[k] for r in grows])) for k in metric_keys
        }
        lines.append(
            f"| {gname} ({len(grows)}) | "
            + " | ".join(f"{means[k]:.4g}" for k in metric_keys)
            + " |"
        )
    summary_md = "\n".join(lines)
    print("\n" + summary_md)
    (OUT_DIR / "summary_by_module_type.md").write_text(summary_md + "\n")

    # ---- figures ----
    def layer_series(proj: str, key: str) -> tuple[list[int], list[float]]:
        pts = [
            (int(r["name"].split(".")[2]), r[key])
            for r in rows
            if r["name"].endswith(proj)
        ]
        pts.sort()
        return [p[0] for p in pts], [p[1] for p in pts]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0][0]
    for proj in ("self_attn.q_proj", "mlp.down_proj"):
        for key, marker in (("ours_bit_density", "o--"), ("bonsai_bit_density", "x-")):
            xs, ys = layer_series(proj, key)
            ax.plot(xs, ys, marker, ms=3, label=f"{proj.split('.')[-1]} {key.split('_')[0]}")
    ax.set_title("bit density per layer")
    ax.set_xlabel("layer idx")
    ax.legend(fontsize=7)

    ax = axes[0][1]
    for proj in PROJ_TYPES:
        xs, ys = layer_series(proj, "bonsai_rel_l2")
        ax.plot(xs, ys, "-", lw=1, label=f"bonsai {proj.split('.')[-1]}")
        xs, ys = layer_series(proj, "ours_rel_l2")
        ax.plot(xs, ys, "--", lw=1, label=f"ours {proj.split('.')[-1]}")
    ax.set_title("relative L2 error to fp16")
    ax.set_xlabel("layer idx")
    ax.set_yscale("log")
    ax.legend(fontsize=6, ncol=2)

    ax = axes[1][0]
    ax.bar(np.arange(32) - 0.2, o_bitpos, width=0.4, label="ours")
    ax.bar(np.arange(32) + 0.2, b_bitpos, width=0.4, label="bonsai")
    ax.set_title("bit-1 density per bit position (LSB-first)")
    ax.set_xlabel("bit position in uint32 word")
    ax.legend()

    ax = axes[1][1]
    o_scales = np.concatenate(
        [np.asarray(r["ours_scale_mean"]).reshape(1) for r in rows]
    )
    b_scales = np.concatenate(
        [np.asarray(r["bonsai_scale_mean"]).reshape(1) for r in rows]
    )
    ax.hist(o_scales, bins=40, alpha=0.6, label="ours", density=True)
    ax.hist(b_scales, bins=40, alpha=0.6, label="bonsai", density=True)
    ax.set_title("per-layer mean scale distribution")
    ax.legend()

    fig.tight_layout()
    fig.savefig(OUT_DIR / "overview.png", dpi=140)
    print(f"\noutputs written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
