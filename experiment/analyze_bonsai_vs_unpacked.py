#!/usr/bin/env python3
"""Real-data cross-validation: dequantized Bonsai packed weights vs unpacked FP16.

This is the verification step from ``deep-research-report.md`` §改进建议 that was
**not** done by the existing ``analyze_container_format.py`` (which only
self-checks the repo's own packer). Here we load the *real* Bonsai checkpoints:

  - ``prism-ml/Ternary-Bonsai-27B-mlx-2bit``    packed (group_size=128, bits=2)
  - ``prism-ml/Ternary-Bonsai-27B-unpacked``     FP16 reference

For each of the 498 quantized triplets (weight + scales + biases) we unpack the
2-bit codes, dequantize with the container codebook, and compare against the
unpacked FP16 same-name layer. We then brute-force the 2-bit -> ternary codebook
on a few small layers to confirm the mapping is the unique optimum.

Outputs (``experiment/analysis_bonsai_vs_unpacked/``)::

    layer_metrics.csv       498 rows: mse / mae / max_abs / cosine / rel_l2
    bruteforce_results.csv  per-layer 24 mappings + best
    alignment_check.json    name-normalisation alignment report
    REPORT.md               human-readable summary

Usage::

    python experiment/analyze_bonsai_vs_unpacked.py \
        --packed-dir /path/to/Ternary-Bonsai-27B-mlx-2bit \
        --unpacked-dir /path/to/Ternary-Bonsai-27B-unpacked

    # auto-download via modelscope if missing
    python experiment/analyze_bonsai_vs_unpacked.py --download

    # only a subset of layers (substring match, comma-separated)
    python experiment/analyze_bonsai_vs_unpacked.py --layers layers.0,in_proj_a
"""

from __future__ import annotations

import argparse
import csv
import gc
import itertools
import json
import os
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

from shadow_peft.quantization import dequantize_2bit_ternary

# --------------------------------------------------------------------------- #
# Constants                                                                   #
# --------------------------------------------------------------------------- #
GROUP_SIZE = 128
OUT_DIR = Path(__file__).parent / "analysis_bonsai_vs_unpacked"

# Default ModelScope snapshot locations (overridable via CLI).
_MS_BASE = os.path.expanduser("~/.cache/modelscope")
DEFAULT_PACKED_DIR = os.path.join(_MS_BASE, "prism-ml/Ternary-Bonsai-27B-mlx-2bit")
DEFAULT_UNPACKED_DIR = os.path.join(_MS_BASE, "prism-ml/Ternary-Bonsai-27B-unpacked")

PACKED_REPO = "prism-ml/Ternary-Bonsai-27B-mlx-2bit"
UNPACKED_REPO = "prism-ml/Ternary-Bonsai-27B-unpacked"

# Layer types to pick small representatives from for brute-force.
BRUTEFORCE_TYPE_TAGS = [
    "in_proj_a",
    "in_proj_b",
    "out_proj",
    "linear_fc1",
    "linear_fc2",
    "embed_tokens",
    "lm_head",
]


# --------------------------------------------------------------------------- #
# Name normalisation                                                          #
# --------------------------------------------------------------------------- #
def normalize_layer_name(name: str) -> str:
    """Collapse the packed vs unpacked naming divergence to a common key.

    packed:   ``language_model.model.layers.0.in_proj_a``
    unpacked: ``model.language_model.layers.0.in_proj_a``
    both  ->  ``layers.0.in_proj_a``

    The known divergent prefixes are stripped in order. Anything that does not
    match a known prefix is returned unchanged so norm/lm_head/embed still work.
    """
    for prefix in (
        "language_model.model.",
        "model.language_model.",
        "language_model.",
        "model.",
    ):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


# --------------------------------------------------------------------------- #
# Index loading + alignment                                                   #
# --------------------------------------------------------------------------- #
def _load_index(model_dir: str) -> dict[str, str]:
    """Return ``{tensor_name: shard_file}`` from a safetensors index.json."""
    idx_path = Path(model_dir) / "model.safetensors.index.json"
    if not idx_path.exists():
        raise FileNotFoundError(f"index not found: {idx_path}")
    return json.loads(idx_path.read_text())["weight_map"]


def _triplet_bases(packed_index: dict[str, str]) -> list[str]:
    """Return the base names of every (weight, scales, biases) triplet."""
    bases = []
    seen = set()
    for key in packed_index:
        if key.endswith(".scales"):
            base = key[: -len(".scales")]
            if (
                base + ".weight" in packed_index
                and base + ".biases" in packed_index
                and base not in seen
            ):
                bases.append(base)
                seen.add(base)
    return bases


def load_and_align(packed_dir: str, unpacked_dir: str) -> tuple[list[dict], dict[str, str]]:
    """Load both indexes and align triplets to unpacked weights by normalised name.

    Returns ``(triplets, unpacked_index)`` where each triplet is::

        {
            "packed_base":   "language_model.model.layers.0.in_proj_a",
            "norm_key":      "layers.0.in_proj_a",
            "unpacked_key":  "model.language_model.layers.0.in_proj_a.weight",
            "shard":         "model-00001-of-00012.safetensors",  # unpacked shard
        }

    ``unpacked_key`` / ``shard`` are ``None`` if no unpacked match was found.
    """
    packed_index = _load_index(packed_dir)
    unpacked_index = _load_index(unpacked_dir)

    # Build normalised -> unpacked key map (only weights).
    norm_to_unpacked: dict[str, str] = {}
    for key in unpacked_index:
        if key.endswith(".weight"):
            norm_to_unpacked[normalize_layer_name(key[: -len(".weight")])] = key

    triplets = []
    for base in _triplet_bases(packed_index):
        norm_key = normalize_layer_name(base)
        unpacked_key = norm_to_unpacked.get(norm_key)
        triplets.append(
            {
                "packed_base": base,
                "norm_key": norm_key,
                "unpacked_key": unpacked_key,
                "shard": unpacked_index[unpacked_key] if unpacked_key else None,
            }
        )
    return triplets, unpacked_index


# --------------------------------------------------------------------------- #
# Per-layer loading helpers (streaming, low memory)                           #
# --------------------------------------------------------------------------- #
def _load_packed_triplet(
    packed_dir: str, base: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load (weight, scales, biases) for one packed triplet from the single file."""
    path = str(Path(packed_dir) / "model.safetensors")
    with safe_open(path, framework="pt") as f:
        weight = f.get_tensor(f"{base}.weight")
        scales = f.get_tensor(f"{base}.scales")
        biases = f.get_tensor(f"{base}.biases")
    # MLX stores packed weights as uint32; torch may read them as int32 or
    # uint32-ish. dequantize_2bit_ternary expects int32 bit-patterns.
    if weight.dtype != torch.int32:
        weight = weight.view(torch.int32)
    return weight, scales.to(torch.float16), biases.to(torch.float16)


def _load_unpacked_weight(unpacked_dir: str, shard: str, key: str) -> torch.Tensor:
    """Load one FP16 weight tensor from an unpacked shard."""
    path = str(Path(unpacked_dir) / shard)
    with safe_open(path, framework="pt") as f:
        return f.get_tensor(key).to(torch.float16)


# --------------------------------------------------------------------------- #
# Metrics                                                                     #
# --------------------------------------------------------------------------- #
def _metrics(deq: torch.Tensor, ref: torch.Tensor) -> dict[str, float]:
    # Use double precision for the norm/dot products: for large fp16 tensors the
    # fp32 sum-of-squares accumulates enough error that d@d can exceed |d||d|,
    # yielding a cosine slightly above 1.0 even when deq == ref bit-exactly.
    diff = deq.double().reshape(-1) - ref.double().reshape(-1)
    mse = float((diff * diff).mean().item())
    mae = float(diff.abs().mean().item())
    max_abs = float(diff.abs().max().item())
    d = deq.double().reshape(-1)
    r = ref.double().reshape(-1)
    denom = d.norm() * r.norm()
    cosine = float((d @ r / denom).item()) if denom > 0 else 0.0
    cosine = max(-1.0, min(1.0, cosine))  # clamp fp round-off into [-1, 1]
    ref_norm = r.norm()
    rel_l2 = float((diff.norm() / ref_norm).item()) if ref_norm > 0 else 0.0
    return {"mse": mse, "mae": mae, "max_abs": max_abs, "cosine": cosine, "rel_l2": rel_l2}


def compare_layer(
    triplet: dict, packed_dir: str, unpacked_dir: str, unpacked_index: dict
) -> dict[str, object]:
    """Dequantize one packed layer and compare to the unpacked FP16 reference."""
    base = triplet["packed_base"]
    packed, scales, biases = _load_packed_triplet(packed_dir, base)
    deq = dequantize_2bit_ternary(packed, scales, biases, group_size=GROUP_SIZE)

    result: dict[str, object] = {
        "norm_key": triplet["norm_key"],
        "shape": "x".join(map(str, deq.shape)),
        "matched": triplet["unpacked_key"] is not None,
    }
    if triplet["unpacked_key"] is None:
        for k in ("mse", "mae", "max_abs", "cosine", "rel_l2"):
            result[k] = float("nan")
    else:
        ref = _load_unpacked_weight(unpacked_dir, triplet["shard"], triplet["unpacked_key"])
        if ref.shape != deq.shape:
            result["shape_mismatch"] = f"deq={list(deq.shape)} ref={list(ref.shape)}"
            for k in ("mse", "mae", "max_abs", "cosine", "rel_l2"):
                result[k] = float("nan")
        else:
            result.update(_metrics(deq, ref))
        del ref
    del packed, scales, biases, deq
    gc.collect()
    return result


# --------------------------------------------------------------------------- #
# Codeword brute-force                                                        #
# --------------------------------------------------------------------------- #
def enumerate_code_mappings() -> list[tuple[int, int, int, int]]:
    """All 24 assignments that pick one of the 4 codes as "unused" and give the
    remaining three the values ``{-1, 0, +1}`` in every order.

    Each mapping is a 4-tuple ``(c0, c1, c2, c3)`` giving the ternary value that
    code index *i* decodes to. Exactly 3 of the 4 positions carry the distinct
    values ``{-1, 0, +1}``; the 4th (the unused slot) gets whatever value the
    unused code would decode to — it never appears in the data so it does not
    affect MSE, but keeping the 4-tuples distinct preserves which code is unused.

    Count: 4 choices for the unused code × 3! permutations = 24.
    """
    mappings = []
    ternary_vals = (-1, 0, 1)
    for unused in range(4):  # which code index is the unused slot
        active = [c for c in range(4) if c != unused]
        for ternary_perm in itertools.permutations(ternary_vals):
            m = [0, 0, 0, 0]
            for code, val in zip(active, ternary_perm, strict=True):
                m[code] = val
            mappings.append(tuple(m))
    return mappings


def _unpack_codes_full(packed: torch.Tensor) -> torch.Tensor:
    """Unpack all 2-bit codes from packed int32 [out, in/16] -> [out, in]."""
    out_features, num_words = packed.shape
    shifts = 2 * torch.arange(16, dtype=torch.int32, device=packed.device)
    codes = ((packed.unsqueeze(-1) >> shifts) & 0b11)  # [out, in/16, 16]
    return codes.reshape(out_features, num_words * 16)


def bruteforce_layer(
    triplet: dict, packed_dir: str, unpacked_dir: str, unpacked_index: dict
) -> dict[str, object]:
    """Brute-force the 2-bit -> ternary codebook on one small layer.

    For each of the 24 candidate mappings, decode every weight as
    ``ternary_value * s_g`` (per group) and compute MSE vs the unpacked FP16
    reference. Returns the best mapping + full per-mapping scores.
    """
    packed, scales, biases = _load_packed_triplet(packed_dir, triplet["packed_base"])
    codes = _unpack_codes_full(packed).to(torch.int64)  # [out, in], values 0..3
    out_features, in_features = codes.shape

    ref = _load_unpacked_weight(
        unpacked_dir, triplet["shard"], triplet["unpacked_key"]
    ).float()  # [out, in]

    # s_g per element: expand scales [out, groups] -> [out, in]
    s_g_expanded = scales.float().repeat_interleave(GROUP_SIZE, dim=1)

    mappings = enumerate_code_mappings()
    rows = []
    best_mse = float("inf")
    best_mapping = None
    for m in mappings:
        lut = torch.tensor(m, dtype=torch.float32)  # [4]
        ternary_decoded = lut[codes]  # [out, in]
        decoded = ternary_decoded * s_g_expanded
        diff = decoded - ref
        mse = float((diff * diff).mean().item())
        rows.append({"mapping": list(m), "mse": mse})
        if mse < best_mse:
            best_mse = mse
            best_mapping = list(m)

    del packed, scales, biases, codes, ref, s_g_expanded
    gc.collect()
    return {
        "norm_key": triplet["norm_key"],
        "n_mappings": len(rows),
        "best_mapping": best_mapping,
        "best_mse": best_mse,
        "all_mappings": rows,
    }


# --------------------------------------------------------------------------- #
# Small-layer selection for brute-force                                       #
# --------------------------------------------------------------------------- #
def _layer_type(norm_key: str) -> str:
    for tag in BRUTEFORCE_TYPE_TAGS:
        if tag in norm_key:
            return tag
    return "other"


def _packed_layer_nelem(triplet: dict, packed_dir: str) -> int:
    """Element count of a packed layer's logical weight for ordering by size."""
    base = triplet["packed_base"]
    path = str(Path(packed_dir) / "model.safetensors")
    with safe_open(path, framework="pt") as f:
        w = f.get_slice(f"{base}.weight")
        shape = w.get_shape()  # [out, in/16]
        return int(shape[0]) * int(shape[1]) * 16  # logical dequantised elements


def select_bruteforce_layers(
    triplets: list[dict], packed_dir: str, per_type: int = 1
) -> list[dict]:
    """Pick ``per_type`` smallest matched layer per type tag."""
    matched = [t for t in triplets if t["unpacked_key"] is not None]
    by_type: dict[str, list[dict]] = {}
    for t in matched:
        t["_nelem"] = _packed_layer_nelem(t, packed_dir)
        by_type.setdefault(_layer_type(t["norm_key"]), []).append(t)
    chosen = []
    for _, layers in sorted(by_type.items()):
        layers.sort(key=lambda t: t["_nelem"])
        chosen.extend(layers[:per_type])
    return chosen


# --------------------------------------------------------------------------- #
# Report writer                                                               #
# --------------------------------------------------------------------------- #
def write_report(
    layer_rows: list[dict],
    bf_results: list[dict],
    n_total: int,
    n_matched: int,
    out_dir: Path,
) -> None:
    lines: list[str] = []
    lines.append("# 真实 Bonsai packed vs unpacked 逐层比对报告")
    lines.append("")
    lines.append(f"> 数据来源：`{PACKED_REPO}`（packed）+ `{UNPACKED_REPO}`（unpacked FP16）")
    lines.append(f"> 量化层三元组总数：{n_total}，匹配到 unpacked：{n_matched}")
    lines.append("")

    valid = [r for r in layer_rows if r.get("matched") and not _is_nan(r.get("cosine"))]
    if valid:
        cosines = [r["cosine"] for r in valid]
        mses = [r["mse"] for r in valid]
        rel_l2s = [r["rel_l2"] for r in valid]
        lines.append("## 1. 全局统计（逐层指标汇总）")
        lines.append("")
        lines.append("| 指标 | 均值 | 中位数 | 最大 |")
        lines.append("|---|---:|---:|---:|")
        lines.append(
            f"| cosine | {np.mean(cosines):.6f} | {np.median(cosines):.6f} | {np.max(cosines):.6f} |"
        )
        lines.append(
            f"| MSE | {np.mean(mses):.6e} | {np.median(mses):.6e} | {np.max(mses):.6e} |"
        )
        lines.append(
            f"| rel-L2 | {np.mean(rel_l2s):.6f} | {np.median(rel_l2s):.6f} | {np.max(rel_l2s):.6f} |"
        )
        lines.append("")

    if bf_results:
        lines.append("## 2. 小层 codeword brute-force 结果")
        lines.append("")
        lines.append("对每个小层穷举全部 24 种 2-bit code → ternary 映射，取 MSE 最小者：")
        lines.append("")
        lines.append("| 层 | 最优映射 (code0,1,2,3 → ternary) | 最优 MSE | 与理论一致 |")
        lines.append("|---|---|---:|:---:|")
        for r in bf_results:
            bm = r["best_mapping"]
            agree = "✅" if bm[0] == -1 and bm[1] == 0 and bm[2] == 1 else "❌"
            lines.append(f"| {r['norm_key']} | {bm} | {r['best_mse']:.6e} | {agree} |")
        lines.append("")
        lines.append(
            "> 理论映射：code {0,1,2} → ternary {-1,0,+1}，code 3 未使用。"
            " 若所有小层最优映射一致即证实该 codebook 是唯一解。"
        )
        lines.append("")

    lines.append("## 3. 结论")
    lines.append("")
    lines.append(
        "- 若逐层 cosine ≈ 1.0 且 MSE ≈ 0，则「MLX 容器 codebook = "
        "{0,1,2}→{-s_g,0,+s_g}, scale=s_g, bias=-s_g」在**真实 Bonsai 数据**上得到直接证实。"
    )
    lines.append(
        "- 若 brute-force 最优映射在所有小层上一致且为 {-1,0,+1,unused}，"
        "则该 codebook 是唯一最优映射，报告 §改进建议的这一步验证完成。"
    )
    lines.append("")

    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _is_nan(v) -> bool:
    try:
        return v != v  # NaN is not equal to itself
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
def _maybe_download(cache_dir: str | None, which: str) -> str:
    from modelscope import snapshot_download

    repo = PACKED_REPO if which == "packed" else UNPACKED_REPO
    print(f"[download] {repo} ...", flush=True)
    return snapshot_download(model_id=repo, local_dir=cache_dir)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--packed-dir", default=DEFAULT_PACKED_DIR)
    ap.add_argument("--unpacked-dir", default=DEFAULT_UNPACKED_DIR)
    ap.add_argument(
        "--download", action="store_true", help="auto-download missing repos via modelscope"
    )
    ap.add_argument("--cache-dir", default=None, help="modelscope cache root for --download")
    ap.add_argument(
        "--layers", default=None, help="comma-separated substring filter on layer names"
    )
    ap.add_argument(
        "--brute-force-types",
        default="1",
        help="per-type layer count for brute-force (0 to skip brute-force)",
    )
    ap.add_argument(
        "--brute-force-layers",
        default=None,
        help="explicit comma-separated layer substrings for brute-force",
    )
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    packed_dir = args.packed_dir
    unpacked_dir = args.unpacked_dir
    if args.download:
        if not Path(packed_dir, "model.safetensors").exists():
            packed_dir = _maybe_download(args.cache_dir, "packed")
        if not Path(unpacked_dir, "model.safetensors.index.json").exists():
            unpacked_dir = _maybe_download(args.cache_dir, "unpacked")

    print(f"packed_dir:   {packed_dir}", flush=True)
    print(f"unpacked_dir: {unpacked_dir}", flush=True)

    # ---- align ----
    triplets, unpacked_index = load_and_align(packed_dir, unpacked_dir)
    n_total = len(triplets)
    n_matched = sum(1 for t in triplets if t["unpacked_key"])
    print(f"triplets: {n_total}, matched to unpacked: {n_matched}", flush=True)

    if args.layers:
        subs = [s.strip() for s in args.layers.split(",") if s.strip()]
        triplets = [t for t in triplets if any(s in t["norm_key"] for s in subs)]
        print(f"after --layers filter: {len(triplets)} layers", flush=True)

    (OUT_DIR / "alignment_check.json").write_text(
        json.dumps(
            {
                "n_triplets": n_total,
                "n_matched": n_matched,
                "n_unmatched": n_total - n_matched,
                "unmatched_examples": [
                    t["packed_base"] for t in triplets if t["unpacked_key"] is None
                ][:20],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # ---- per-layer compare ----
    layer_rows: list[dict] = []
    for i, t in enumerate(triplets):
        row = compare_layer(t, packed_dir, unpacked_dir, unpacked_index)
        layer_rows.append(row)
        if (i + 1) % 25 == 0 or i + 1 == len(triplets):
            print(f"  [{i + 1}/{len(triplets)}] {t['norm_key']}", flush=True)

    if layer_rows:
        fieldnames = ["norm_key", "shape", "matched", "mse", "mae", "max_abs", "cosine", "rel_l2"]
        with open(OUT_DIR / "layer_metrics.csv", "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=fieldnames)
            wr.writeheader()
            for r in layer_rows:
                wr.writerow({k: r.get(k, "") for k in fieldnames})

    # ---- brute-force ----
    bf_results: list[dict] = []
    bf_count = int(args.brute_force_types)
    if bf_count > 0:
        if args.brute_force_layers:
            subs = [s.strip() for s in args.brute_force_layers.split(",") if s.strip()]
            candidates = [t for t in triplets if any(s in t["norm_key"] for s in subs)]
        else:
            candidates = select_bruteforce_layers(triplets, packed_dir, per_type=bf_count)
        print(f"\nbrute-forcing {len(candidates)} small layers ...", flush=True)
        for t in candidates:
            r = bruteforce_layer(t, packed_dir, unpacked_dir, unpacked_index)
            bf_results.append(r)
            print(
                f"  {r['norm_key']}: best={r['best_mapping']} mse={r['best_mse']:.3e}",
                flush=True,
            )
        with open(OUT_DIR / "bruteforce_results.csv", "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["norm_key", "mapping", "mse"])
            for r in bf_results:
                for m in r["all_mappings"]:
                    wr.writerow([r["norm_key"], m["mapping"], m["mse"]])

    write_report(layer_rows, bf_results, n_total, n_matched, OUT_DIR)
    print(f"\noutputs written to {OUT_DIR}/", flush=True)


if __name__ == "__main__":
    main()
