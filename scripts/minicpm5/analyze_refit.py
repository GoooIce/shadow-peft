#!/usr/bin/env python3
"""Offline analysis of the ticket-11 refit campaign.

Consumes the ``refit_gate.py`` run JSONs plus the layer captures and prints the
numbers the note needs:

* A1/A4  reload wall clock per path (checkpoint format vs kernel format);
* A2     pointer/graph census deltas, reciprocal-scale refresh evidence;
* A3     hot-reloaded engine vs fresh-loaded engine, token by token;
* B3     Generation KL Error (BF16 reference) and per-layer ||dY||_2;
* B4     candidate ranking under the refit engine vs fresh engines.

Usage:
    python analyze_refit.py --dir <out/refit> --out <analysis.json>
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch


def load(path: Path):
    return json.loads(path.read_text())


def snapshot_of(step):
    return step.get("snapshot", step)


def step_map(report):
    return {step["tag"]: step for step in report["steps"]}


def decode_compare(left: dict, right: dict) -> dict:
    per_prompt = []
    max_abs = 0.0
    for lprompt, rprompt in zip(left["decode"]["prompts"], right["decode"]["prompts"]):
        identical = list(lprompt["token_ids"]) == list(rprompt["token_ids"])
        diffs = [
            abs((la if la is not None else float("nan"))
                - (ra if ra is not None else float("nan")))
            for la, ra in zip(lprompt["logprobs"], rprompt["logprobs"])
        ]
        finite = [value for value in diffs if value == value]
        if finite:
            max_abs = max(max_abs, max(finite))
        per_prompt.append(
            {
                "tokens_identical": identical,
                "first_divergence": next(
                    (
                        index
                        for index, (la, ra) in enumerate(
                            zip(lprompt["token_ids"], rprompt["token_ids"])
                        )
                        if la != ra
                    ),
                    None,
                ),
                "max_abs_logprob_diff": max(finite) if finite else None,
            }
        )
    return {
        "all_tokens_identical": all(item["tokens_identical"] for item in per_prompt),
        "max_abs_logprob_diff": max_abs,
        "per_prompt": per_prompt,
    }


def scalar_refresh(engine_dump: Path, built: Path) -> dict:
    """Which kernel scalars differ between two kernel-format files."""
    from safetensors import safe_open

    rows = []
    with safe_open(engine_dump, framework="pt", device="cpu") as left, safe_open(
        built, framework="pt", device="cpu"
    ) as right:
        keys = [key for key in left.keys() if key.rsplit(".", 1)[-1] in (
            "weight_global_scale", "input_global_scale", "input_global_scale_inv", "alpha"
        )]
        for key in sorted(keys):
            a = float(left.get_tensor(key))
            b = float(right.get_tensor(key))
            rows.append(
                {
                    "key": key,
                    "engine": a,
                    "built": b,
                    "equal": a == b,
                    "rel_diff": abs(a - b) / abs(b) if b else None,
                }
            )
    return {
        "tensors": len(rows),
        "mismatches": sum(0 if row["equal"] else 1 for row in rows),
        "rows": rows,
    }


def layer_delta(left_path: Path, right_path: Path) -> dict:
    left = torch.load(left_path, map_location="cpu", weights_only=True)
    right = torch.load(right_path, map_location="cpu", weights_only=True)
    left_inputs = left.get("inputs") if isinstance(left, dict) else None
    right_inputs = right.get("inputs") if isinstance(right, dict) else None
    left = left["layers"] if isinstance(left, dict) and "layers" in left else left
    right = right["layers"] if isinstance(right, dict) and "layers" in right else right
    inputs_match = None
    if left_inputs and right_inputs:
        inputs_match = bool(
            torch.equal(left_inputs[-1].long(), right_inputs[-1].long())
        )
    rows = []
    for name in sorted(set(left) & set(right)):
        a = left[name].float()
        b = right[name].float()
        delta = a - b
        rows.append(
            {
                "layer": name,
                "relative_frobenius": float(
                    delta.square().sum().sqrt() / a.square().sum().sqrt()
                ),
                "mean_abs": float(delta.abs().mean()),
                "max_abs": float(delta.abs().max()),
                "reference_norm": float(a.square().sum().sqrt()),
            }
        )
    rel = [row["relative_frobenius"] for row in rows]
    return {
        "layers": len(rows),
        "inputs_match": inputs_match,
        "relative_frobenius_mean": sum(rel) / len(rel) if rel else None,
        "relative_frobenius_max": max(rel) if rel else None,
        "rows": rows,
    }


def kl_against(bf16: dict, other: dict) -> dict:
    if "kl" not in bf16 or "kl" not in other:
        return {"tokens": 0, "error": "missing kl probe"}
    deltas = []
    for lseq, rseq in zip(
        bf16["kl"]["continuation_logprobs"], other["kl"]["continuation_logprobs"]
    ):
        for lv, rv in zip(lseq["logprobs"], rseq["logprobs"]):
            if lv is None or rv is None:
                continue
            deltas.append(abs(lv - rv))
    if not deltas:
        return {"tokens": 0}
    return {
        "tokens": len(deltas),
        "mean_abs_delta": sum(deltas) / len(deltas),
        "max_abs_delta": max(deltas),
        "generation_kl_error": sum(math.exp(value) for value in deltas) / len(deltas),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--engine", default="engineA2")
    parser.add_argument("--bf16", default="bf16.json")
    args = parser.parse_args()
    root = Path(args.dir).expanduser()

    report: dict = {"dir": str(root), "engine": args.engine}
    engine_a = load(root / f"{args.engine}.json")
    fresh = {
        "q0": load(root / "fresh_q0.json"),
        "clip060": load(root / "fresh_clip060.json"),
        "search": load(root / "fresh_search.json"),
    }
    bf16 = load(root / args.bf16)

    # ---- A1/A4 timings -------------------------------------------------- #
    timings = []
    for step in engine_a["steps"]:
        if "timing" not in step:
            continue
        entry = {"tag": step["tag"], "kind": step["step"]["kind"]}
        timing = step["timing"]
        for key in ("seconds", "load_seconds", "median", "min", "max", "tensors", "bytes"):
            if key in timing:
                entry[key] = timing[key]
        timings.append(entry)
    report["timings"] = {
        "engine_load_seconds": {
            "engineA": engine_a["engine_load_seconds"],
            "bf16": bf16["engine_load_seconds"],
            **{key: value["engine_load_seconds"] for key, value in fresh.items()},
        },
        "chain": timings,
    }

    # ---- A2 censuses ---------------------------------------------------- #
    a_steps = step_map(engine_a)
    base = snapshot_of(a_steps["baseline"])
    census_rows = []
    for tag in a_steps:
        snap = snapshot_of(a_steps[tag])
        pointer_changes = 0
        hash_changes = 0
        for name, row in snap["census"]["layers"].items():
            reference = base["census"]["layers"][name]
            for param, entry in row.items():
                if entry["ptr"] != reference[param]["ptr"]:
                    pointer_changes += 1
            if row["weight"]["sha256"] != reference["weight"]["sha256"]:
                hash_changes += 1
        census_rows.append(
            {
                "tag": tag,
                "pointer_changes": pointer_changes,
                "weight_hash_changes": hash_changes,
                "graphs": {
                    name: {
                        "num_graphs": value["num_graphs"],
                        "graph_ids": value["graph_ids"],
                        "graphs_captured": value["graphs_captured"],
                    }
                    for name, value in snap["graph_census"].items()
                },
                "kernel": snap["kernel_census"]["counts"],
                "ppl": snap["score"]["ppl"],
            }
        )
    report["censuses"] = census_rows

    # ---- A3 hot vs fresh ------------------------------------------------ #
    pairs = {
        "q0": ("baseline", fresh["q0"]),
        "clip060": ("cp_cand_clip060", fresh["clip060"]),
        "search": ("cp_cand_search", fresh["search"]),
        "search_kf_vs_cp": ("kf_built_cand_search", None),
        "q0_restore": ("cp_restore_q0", fresh["q0"]),
        "q0_kf_restore": ("kf_restore_q0", fresh["q0"]),
    }
    report["hot_vs_fresh"] = {}
    for label, (tag, fresh_report) in pairs.items():
        if tag not in a_steps:
            continue
        left = snapshot_of(a_steps[tag])
        right = snapshot_of(a_steps["cp_cand_search"]) if fresh_report is None else snapshot_of(fresh_report["steps"][0])
        report["hot_vs_fresh"][label] = decode_compare(left, right)

    # ---- B3 KL ---------------------------------------------------------- #
    bf16_snapshot = snapshot_of(bf16["steps"][0])
    report["kl"] = {}
    for label, tag in (("q0", "baseline"), ("clip060", "cp_cand_clip060"), ("search", "cp_cand_search")):
        if tag in a_steps:
            report["kl"][label] = kl_against(bf16_snapshot, snapshot_of(a_steps[tag]))

    # ---- B4 ranking ----------------------------------------------------- #
    def score_of(snapshot):
        return snapshot["score"]["mean_nll"]

    hot_scores = {}
    fresh_scores = {}
    for label, tag in (("q0", "baseline"), ("clip060", "cp_cand_clip060"), ("search", "cp_cand_search")):
        if tag in a_steps:
            hot_scores[label] = score_of(snapshot_of(a_steps[tag]))
    for label, value in fresh.items():
        fresh_scores[label] = score_of(snapshot_of(value["steps"][0]))
    hot_rank = sorted(hot_scores, key=lambda key: hot_scores[key])
    fresh_rank = sorted(fresh_scores, key=lambda key: fresh_scores[key])
    common = [label for label in hot_rank if label in fresh_rank]
    report["ranking"] = {
        "hot_scores": hot_scores,
        "fresh_scores": fresh_scores,
        "hot_rank": hot_rank,
        "fresh_rank": fresh_rank,
        "common_rank_equal": common == [label for label in fresh_rank if label in hot_scores],
        "score_deltas": {
            label: hot_scores[label] - fresh_scores[label]
            for label in common
        },
    }

    # ---- B3b layer deltas ----------------------------------------------- #
    layers_bf16 = root / "layers_bf16.pt"
    layers_nvfp4 = root / "layers_nvfp4_q0.pt"
    if layers_bf16.exists() and layers_nvfp4.exists():
        report["layer_delta_bf16_vs_nvfp4"] = layer_delta(layers_bf16, layers_nvfp4)

    # ---- kernel file comparisons ---------------------------------------- #
    comparisons = {}
    for name in ("compare_q0_engine_vs_built", "compare_cand_search_engine_vs_built"):
        path = root / f"{name}.json"
        if path.exists():
            data = load(path)
            comparisons[name] = {
                "tensors": data["tensors"],
                "mismatches": data["mismatches"],
                "all_bitwise_equal": data["all_bitwise_equal"],
            }
    report["kernel_comparisons"] = comparisons

    Path(args.out).expanduser().write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(
        {
            "timings": report["timings"],
            "hot_vs_fresh": {
                key: {
                    "identical": value["all_tokens_identical"],
                    "max_abs_logprob_diff": value["max_abs_logprob_diff"],
                }
                for key, value in report["hot_vs_fresh"].items()
            },
            "kl": report["kl"],
            "ranking": report["ranking"],
            "kernel_comparisons": comparisons,
        },
        indent=2,
    )[:6000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
