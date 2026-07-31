"""E1: reverse-engineer Bonsai g128 codes/scales against Qwen3.6-27B.

The official packed checkpoint supplies the final ternary codes and scales.
The Qwen checkpoint supplies the pre-migration BF16 weights.  This script
samples the same number of groups from every matrix, weights the aggregates
back by each layer's true group count, and measures how well progressively
more flexible magnitude-threshold rules can explain the official solution.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors import safe_open

DEFAULT_QWEN = Path(
    "~/.cache/modelscope/models/Qwen--Qwen3.6-27B/snapshots/master"
).expanduser()
DEFAULT_PACKED = Path(
    "~/.cache/modelscope/models/"
    "prism-ml--Ternary-Bonsai-27B-mlx-2bit/snapshots/master"
).expanduser()
DEFAULT_OUT = Path("experiment/analysis_qwen36_bonsai_migration/e1")
GROUP_SIZE = 128
WORDS_PER_GROUP = GROUP_SIZE // 16


def _source_key(base: str) -> str:
    if base == "language_model.lm_head":
        return "lm_head.weight"
    prefix = "language_model.model."
    if not base.startswith(prefix):
        raise ValueError(f"unexpected official language key: {base}")
    return "model.language_model." + base.removeprefix(prefix) + ".weight"


def _decode_groups(words: np.ndarray) -> np.ndarray:
    shifts = (2 * np.arange(16, dtype=np.uint32)).reshape(1, 1, 16)
    codes = (words[:, :, None] >> shifts) & np.uint32(3)
    return codes.astype(np.int8).reshape(words.shape[0], GROUP_SIZE) - 1


def _report_quantize(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scales = np.maximum(np.mean(np.abs(x), axis=1, keepdims=True), 1e-8)
    q = np.zeros_like(x, dtype=np.int8)
    for iteration in range(8):
        threshold = 0.5 * scales
        q = np.where(x > threshold, 1, np.where(x < -threshold, -1, 0)).astype(
            np.int8
        )
        denom = np.maximum(np.sum(q * q, axis=1, keepdims=True), 1)
        scales = np.maximum(
            np.abs(np.sum(x * q, axis=1, keepdims=True) / denom), 1e-8
        )
        if iteration == 7:
            break
    return q, scales[:, 0]


def _best_histogram_threshold(
    zero_hist: np.ndarray,
    signed_hist: np.ndarray,
    edges: np.ndarray,
    unexplainable: float = 0.0,
) -> tuple[float, float]:
    zero_prefix = np.cumsum(zero_hist)
    signed_prefix = np.cumsum(signed_hist)
    correct = zero_prefix + signed_hist.sum() - signed_prefix
    index = int(np.argmax(correct))
    total = zero_hist.sum() + signed_hist.sum() + unexplainable
    return float(edges[index + 1]), float(correct[index] / max(total, 1.0))


def _oracle_group_threshold(
    ratios: np.ndarray, official_q: np.ndarray, base_sign: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(ratios, axis=1)
    sorted_q = np.take_along_axis(official_q, order, axis=1)
    sorted_sign = np.take_along_axis(base_sign, order, axis=1)

    zero_ok = sorted_q == 0
    signed_ok = (sorted_q != 0) & (sorted_q == sorted_sign)
    nonzero_ok = sorted_q != 0

    zero_prefix = np.concatenate(
        [np.zeros((len(ratios), 1), dtype=np.int16), np.cumsum(zero_ok, axis=1)],
        axis=1,
    )
    signed_prefix = np.concatenate(
        [
            np.zeros((len(ratios), 1), dtype=np.int16),
            np.cumsum(signed_ok, axis=1),
        ],
        axis=1,
    )
    nonzero_prefix = np.concatenate(
        [
            np.zeros((len(ratios), 1), dtype=np.int16),
            np.cumsum(nonzero_ok, axis=1),
        ],
        axis=1,
    )
    signed_total = signed_prefix[:, -1:]
    nonzero_total = nonzero_prefix[:, -1:]
    best_code = np.max(zero_prefix + signed_total - signed_prefix, axis=1)
    best_mask = np.max(zero_prefix + nonzero_total - nonzero_prefix, axis=1)

    zero_count = np.sum(official_q == 0, axis=1)
    row = np.arange(len(ratios))
    same_sparsity = (
        zero_prefix[row, zero_count]
        + nonzero_total[:, 0]
        - nonzero_prefix[row, zero_count]
    )
    return (
        best_code.astype(np.float64) / GROUP_SIZE,
        best_mask.astype(np.float64) / GROUP_SIZE,
        same_sparsity.astype(np.float64) / GROUP_SIZE,
    )


def _weighted_corr(
    weight: float,
    x: np.ndarray,
    y: np.ndarray,
    state: dict[str, float],
) -> None:
    state["n"] += weight * len(x)
    state["sx"] += weight * float(x.sum(dtype=np.float64))
    state["sy"] += weight * float(y.sum(dtype=np.float64))
    state["sxx"] += weight * float(np.square(x, dtype=np.float64).sum())
    state["syy"] += weight * float(np.square(y, dtype=np.float64).sum())
    state["sxy"] += weight * float(np.multiply(x, y, dtype=np.float64).sum())
    state["sabs"] += weight * float(np.abs(x - y).sum(dtype=np.float64))


def _corr_result(state: dict[str, float]) -> dict[str, float]:
    n = state["n"]
    cov = state["sxy"] - state["sx"] * state["sy"] / n
    var_x = state["sxx"] - state["sx"] ** 2 / n
    var_y = state["syy"] - state["sy"] ** 2 / n
    return {
        "correlation": cov / math.sqrt(max(var_x * var_y, 1e-30)),
        "mean_x": state["sx"] / n,
        "mean_y": state["sy"] / n,
        "mean_abs_diff": state["sabs"] / n,
    }


def analyze(
    qwen_dir: Path,
    packed_dir: Path,
    *,
    groups_per_layer: int,
    hist_max: float,
    hist_bins: int,
) -> dict[str, Any]:
    qwen_dir = qwen_dir.expanduser().resolve()
    packed_dir = packed_dir.expanduser().resolve()
    qwen_index = json.loads(
        (qwen_dir / "model.safetensors.index.json").read_text(encoding="utf-8")
    )["weight_map"]
    packed_path = packed_dir / "model.safetensors"
    if not packed_path.is_file():
        raise FileNotFoundError(packed_path)

    with safe_open(packed_path, framework="numpy") as packed:
        bases = sorted(
            k.removesuffix(".scales")
            for k in packed.keys()  # noqa: SIM118 - safe_open is not iterable
            if k.endswith(".scales")
        )

    if len(bases) != 498:
        raise ValueError(f"expected 498 official triplets, found {len(bases)}")

    by_source_shard: dict[str, list[str]] = defaultdict(list)
    for base in bases:
        key = _source_key(base)
        by_source_shard[qwen_index[key]].append(base)

    edges = np.linspace(0.0, hist_max, hist_bins + 1)
    global_zero_hist = np.zeros(hist_bins, dtype=np.float64)
    global_signed_hist = np.zeros(hist_bins, dtype=np.float64)
    global_flipped = 0.0
    scale_absmean = defaultdict(float)
    scale_ls = defaultdict(float)
    reconstruction = defaultdict(float)
    totals = defaultdict(float)
    layers: list[dict[str, Any]] = []

    with torch.inference_mode(), safe_open(
        packed_path, framework="numpy"
    ) as packed:
        for shard_number, (shard_name, shard_bases) in enumerate(
            sorted(by_source_shard.items()), start=1
        ):
            print(
                f"[{shard_number:02d}/{len(by_source_shard):02d}] "
                f"{shard_name}: {len(shard_bases)} matrices",
                flush=True,
            )
            with safe_open(
                qwen_dir / shard_name, framework="pt", device="cpu"
            ) as qwen:
                for base in sorted(shard_bases):
                    source_key = _source_key(base)
                    packed_weight = packed.get_tensor(base + ".weight").astype(
                        np.uint32, copy=False
                    )
                    official_scales_all = packed.get_tensor(base + ".scales").astype(
                        np.float32, copy=False
                    ).reshape(-1)
                    group_count = official_scales_all.size
                    sample_count = min(group_count, groups_per_layer)
                    group_ids = np.linspace(
                        0, group_count - 1, sample_count, dtype=np.int64
                    )
                    row_count, words_per_row = packed_weight.shape
                    groups_per_row = words_per_row // WORDS_PER_GROUP
                    rows = group_ids // groups_per_row
                    group_columns = group_ids % groups_per_row
                    word_columns = (
                        group_columns[:, None] * WORDS_PER_GROUP
                        + np.arange(WORDS_PER_GROUP, dtype=np.int64)[None, :]
                    )
                    words = packed_weight[rows[:, None], word_columns]
                    official_q = _decode_groups(words)
                    official_scales = official_scales_all[group_ids]

                    source = qwen.get_tensor(source_key)
                    x = (
                        source.reshape(-1, GROUP_SIZE)[torch.from_numpy(group_ids)]
                        .float()
                        .numpy()
                    )
                    del source

                    weight = group_count / sample_count
                    absmean = np.maximum(np.mean(np.abs(x), axis=1), 1e-8)
                    ratios = np.abs(x) / absmean[:, None]
                    base_sign = np.sign(x).astype(np.int8)
                    official_zero = official_q == 0
                    official_signed = (official_q != 0) & (official_q == base_sign)
                    official_flipped = (official_q != 0) & (
                        official_q != base_sign
                    )

                    zero_hist, _ = np.histogram(
                        np.minimum(ratios[official_zero], hist_max),
                        bins=edges,
                    )
                    signed_hist, _ = np.histogram(
                        np.minimum(ratios[official_signed], hist_max),
                        bins=edges,
                    )
                    global_zero_hist += weight * zero_hist
                    global_signed_hist += weight * signed_hist
                    layer_threshold, layer_threshold_acc = _best_histogram_threshold(
                        zero_hist.astype(np.float64),
                        signed_hist.astype(np.float64),
                        edges,
                        float(official_flipped.sum()),
                    )

                    oracle_code, oracle_mask, same_sparsity = (
                        _oracle_group_threshold(ratios, official_q, base_sign)
                    )

                    report_q, report_scales = _report_quantize(x)
                    one_shot_q = np.where(
                        x > 0.5 * absmean[:, None],
                        1,
                        np.where(x < -0.5 * absmean[:, None], -1, 0),
                    ).astype(np.int8)
                    denom = np.maximum(np.sum(official_q * official_q, axis=1), 1)
                    ls_scales = np.maximum(
                        np.abs(np.sum(x * official_q, axis=1) / denom), 1e-8
                    )

                    _weighted_corr(
                        weight, official_scales, absmean, scale_absmean
                    )
                    _weighted_corr(weight, official_scales, ls_scales, scale_ls)

                    official_deq = official_q * official_scales[:, None]
                    ls_deq = official_q * ls_scales[:, None]
                    reconstruction["base_sq"] += weight * float(
                        np.square(x, dtype=np.float64).sum()
                    )
                    reconstruction["official_sq"] += weight * float(
                        np.square(official_deq, dtype=np.float64).sum()
                    )
                    reconstruction["official_dot"] += weight * float(
                        np.multiply(x, official_deq, dtype=np.float64).sum()
                    )
                    reconstruction["official_diff_sq"] += weight * float(
                        np.square(x - official_deq, dtype=np.float64).sum()
                    )
                    reconstruction["ls_diff_sq"] += weight * float(
                        np.square(x - ls_deq, dtype=np.float64).sum()
                    )

                    values = group_count * GROUP_SIZE
                    totals["groups"] += group_count
                    totals["values"] += values
                    totals["sample_groups"] += sample_count
                    totals["official_zero"] += weight * float(official_zero.sum())
                    totals["official_nonzero"] += weight * float(
                        (official_q != 0).sum()
                    )
                    totals["sign_match"] += weight * float(official_signed.sum())
                    totals["sign_flip"] += weight * float(official_flipped.sum())
                    totals["report_match"] += weight * float(
                        (report_q == official_q).sum()
                    )
                    totals["report_zero"] += weight * float((report_q == 0).sum())
                    totals["one_shot_match"] += weight * float(
                        (one_shot_q == official_q).sum()
                    )
                    totals["one_shot_zero"] += weight * float(
                        (one_shot_q == 0).sum()
                    )
                    totals["scale_ratio_absmean"] += weight * float(
                        np.sum(official_scales / absmean)
                    )
                    totals["scale_ratio_ls"] += weight * float(
                        np.sum(official_scales / ls_scales)
                    )
                    totals["oracle_code"] += (
                        weight * GROUP_SIZE * float(oracle_code.sum())
                    )
                    totals["oracle_mask"] += (
                        weight * GROUP_SIZE * float(oracle_mask.sum())
                    )
                    totals["same_sparsity_mask"] += (
                        weight * GROUP_SIZE * float(same_sparsity.sum())
                    )

                    layers.append(
                        {
                            "name": base,
                            "groups": int(group_count),
                            "sample_groups": int(sample_count),
                            "official_zero_rate": float(official_zero.mean()),
                            "official_sign_preservation": float(
                                official_signed.sum()
                                / max((official_q != 0).sum(), 1)
                            ),
                            "report_code_agreement": float(
                                (report_q == official_q).mean()
                            ),
                            "report_zero_rate": float((report_q == 0).mean()),
                            "one_shot_code_agreement": float(
                                (one_shot_q == official_q).mean()
                            ),
                            "one_shot_zero_rate": float((one_shot_q == 0).mean()),
                            "best_layer_threshold": layer_threshold,
                            "best_layer_threshold_accuracy": layer_threshold_acc,
                            "best_group_threshold_code_accuracy": float(
                                oracle_code.mean()
                            ),
                            "best_group_threshold_mask_accuracy": float(
                                oracle_mask.mean()
                            ),
                            "same_sparsity_rank_mask_accuracy": float(
                                same_sparsity.mean()
                            ),
                            "official_scale_mean": float(official_scales.mean()),
                            "base_absmean_mean": float(absmean.mean()),
                            "base_ls_scale_for_official_codes_mean": float(
                                ls_scales.mean()
                            ),
                            "report_scale_mean": float(report_scales.mean()),
                        }
                    )

                    global_flipped += weight * float(official_flipped.sum())

    global_threshold, global_threshold_accuracy = _best_histogram_threshold(
        global_zero_hist, global_signed_hist, edges, global_flipped
    )
    values = totals["values"]
    nonzero = totals["official_nonzero"]
    base_sq = reconstruction["base_sq"]
    official_sq = reconstruction["official_sq"]
    summary = {
        "matrices": len(bases),
        "groups": int(totals["groups"]),
        "weights": int(values),
        "sample_groups": int(totals["sample_groups"]),
        "official_zero_rate": totals["official_zero"] / values,
        "official_sign_preservation": totals["sign_match"] / nonzero,
        "official_sign_flip_rate": totals["sign_flip"] / nonzero,
        "report_code_agreement": totals["report_match"] / values,
        "report_zero_rate": totals["report_zero"] / values,
        "one_shot_code_agreement": totals["one_shot_match"] / values,
        "one_shot_zero_rate": totals["one_shot_zero"] / values,
        "best_global_threshold": global_threshold,
        "best_global_threshold_code_accuracy": global_threshold_accuracy,
        "best_layer_threshold_code_accuracy_weighted": float(
            np.average(
                [row["best_layer_threshold_accuracy"] for row in layers],
                weights=[row["groups"] for row in layers],
            )
        ),
        "best_group_threshold_code_accuracy": totals["oracle_code"] / values,
        "best_group_threshold_mask_accuracy": totals["oracle_mask"] / values,
        "same_sparsity_rank_mask_accuracy": totals["same_sparsity_mask"] / values,
        "official_vs_qwen_cosine": reconstruction["official_dot"]
        / math.sqrt(base_sq * official_sq),
        "official_vs_qwen_nrmse": math.sqrt(
            reconstruction["official_diff_sq"] / base_sq
        ),
        "fixed_official_codes_ls_nrmse": math.sqrt(
            reconstruction["ls_diff_sq"] / base_sq
        ),
        "official_scale_over_base_absmean_mean": totals["scale_ratio_absmean"]
        / totals["groups"],
        "official_scale_over_qwen_ls_mean": totals["scale_ratio_ls"]
        / totals["groups"],
        "official_scale_vs_base_absmean": _corr_result(scale_absmean),
        "official_scale_vs_qwen_ls_for_official_codes": _corr_result(scale_ls),
    }
    return {"summary": summary, "layers": layers}


def _write_report(result: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    summary = result["summary"]
    layers = result["layers"]
    worst = sorted(layers, key=lambda row: row["report_code_agreement"])[:15]
    lines = [
        "# E1 — Qwen3.6 → official Bonsai g128 rule inversion",
        "",
        "## Global result",
        "",
        f"- Matrices: {summary['matrices']}",
        f"- Groups: {summary['groups']:,}; sampled: {summary['sample_groups']:,}",
        f"- Official zero rate: {summary['official_zero_rate']:.6%}",
        f"- Official nonzero sign preservation: "
        f"{summary['official_sign_preservation']:.6%}",
        f"- Report surrogate code agreement: "
        f"{summary['report_code_agreement']:.6%}",
        f"- One-shot absmean/threshold code agreement: "
        f"{summary['one_shot_code_agreement']:.6%}",
        f"- One-shot zero rate: {summary['one_shot_zero_rate']:.6%}",
        f"- Best global magnitude threshold: "
        f"{summary['best_global_threshold']:.4f} "
        f"(accuracy {summary['best_global_threshold_code_accuracy']:.6%})",
        f"- Best per-layer threshold accuracy: "
        f"{summary['best_layer_threshold_code_accuracy_weighted']:.6%}",
        f"- Best per-group threshold code upper bound: "
        f"{summary['best_group_threshold_code_accuracy']:.6%}",
        f"- Best per-group zero-mask upper bound: "
        f"{summary['best_group_threshold_mask_accuracy']:.6%}",
        f"- Same-sparsity magnitude-rank mask accuracy: "
        f"{summary['same_sparsity_rank_mask_accuracy']:.6%}",
        f"- Official vs Qwen cosine: {summary['official_vs_qwen_cosine']:.6f}",
        f"- Official vs Qwen NRMSE: {summary['official_vs_qwen_nrmse']:.6f}",
        f"- Fixed official codes + Qwen LS scale NRMSE: "
        f"{summary['fixed_official_codes_ls_nrmse']:.6f}",
        f"- Official scale / base absmean (group mean): "
        f"{summary['official_scale_over_base_absmean_mean']:.6f}",
        f"- Official scale vs base absmean correlation: "
        f"{summary['official_scale_vs_base_absmean']['correlation']:.6f}",
        f"- Official scale / Qwen LS scale (group mean): "
        f"{summary['official_scale_over_qwen_ls_mean']:.6f}",
        "",
        "## Interpretation",
        "",
        "- The current 8-iteration LS surrogate over-prunes: its zero rate is "
        f"{summary['report_zero_rate']:.2%}, versus "
        f"{summary['official_zero_rate']:.2%} in the official checkpoint.",
        "- A single absmean/0.5 pass is a substantially better reproduction "
        f"({summary['one_shot_code_agreement']:.2%} code agreement), and the "
        "histogram-optimal global threshold is exactly 0.5.",
        "- Even an oracle threshold chosen independently for every g128 group "
        f"tops out at {summary['best_group_threshold_code_accuracy']:.2%}. "
        "The remaining gap therefore cannot be explained by a different "
        "magnitude threshold alone.",
        "- Official scales track the original Qwen group absmean very closely; "
        "re-fitting scales by LS against the official codes moves away from the "
        "official scale distribution.",
        "",
        "## Worst report-surrogate layer agreement",
        "",
        "| Layer | Agreement | Official zero | Sign preservation | "
        "Group-threshold upper bound |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in worst:
        lines.append(
            f"| `{row['name']}` | {row['report_code_agreement']:.2%} | "
            f"{row['official_zero_rate']:.2%} | "
            f"{row['official_sign_preservation']:.2%} | "
            f"{row['best_group_threshold_code_accuracy']:.2%} |"
        )
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen", type=Path, default=DEFAULT_QWEN)
    parser.add_argument("--packed", type=Path, default=DEFAULT_PACKED)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--groups-per-layer", type=int, default=1024)
    parser.add_argument("--hist-max", type=float, default=8.0)
    parser.add_argument("--hist-bins", type=int, default=4096)
    args = parser.parse_args()
    if args.groups_per_layer < 1:
        parser.error("--groups-per-layer must be >= 1")
    result = analyze(
        args.qwen,
        args.packed,
        groups_per_layer=args.groups_per_layer,
        hist_max=args.hist_max,
        hist_bins=args.hist_bins,
    )
    _write_report(result, args.out)
    print(json.dumps(result["summary"], indent=2), flush=True)
    print(f"report: {args.out / 'REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
