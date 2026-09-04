#!/usr/bin/env python3
"""Freeze Phase-2 Mage ES hyperparameters from preregistered GB10 artifacts."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
from pathlib import Path

REWARD_STD_MIN = 0.02
PAIR_ABS_DIFF_MIN = 0.02
RECONSTRUCTION_RATIO_MAX = 1.05
PERTURBATION_RMS_MAX = 0.03
CLIP_STEP_RMS_MAX = 0.02


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("calibration_root", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict]:
    values = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if not values or not all(isinstance(value, dict) for value in values):
        raise ValueError(f"expected non-empty JSONL objects: {path}")
    return values


def evaluated_alphas(alpha_report: dict, center_evaluation: dict) -> list[float]:
    comparison_by_label = {
        comparison["label"]: comparison
        for comparison in center_evaluation["comparisons"]
    }
    record_by_label = {
        record["label"]: record for record in center_evaluation["records"]
    }
    eligible = []
    for alpha_record in alpha_report["records"]:
        if float(alpha_record["clip_step_rms"]) > CLIP_STEP_RMS_MAX:
            continue
        matches = [
            label
            for label, record in record_by_label.items()
            if record["clip_sha256"] == alpha_record["center_clip_sha256"]
        ]
        if len(matches) != 1:
            raise ValueError("step-eligible alpha must have exactly one center evaluation")
        if comparison_by_label[matches[0]]["non_degrading"] is True:
            eligible.append(float(alpha_record["alpha"]))
    return eligible


def select_alpha(
    primary_report: dict,
    primary_evaluation: dict,
    fallback_report: dict,
    fallback_evaluation: dict,
) -> tuple[float | None, str]:
    primary = evaluated_alphas(primary_report, primary_evaluation)
    if primary:
        return max(primary), "primary"
    fallback = evaluated_alphas(fallback_report, fallback_evaluation)
    return (max(fallback), "fallback") if fallback else (None, "failed")


def confirmation_metrics(rows: list[dict], *, quasar_error: float) -> dict:
    rewards = [float(row["reward"]) for row in rows if row.get("reward") is not None]
    if not rewards:
        raise ValueError("confirmation has no valid rewards")
    failures = len(rows) - len(rewards)
    pairs: dict[int, dict[int, float]] = {}
    for row in rows:
        if row.get("reward") is None:
            continue
        spec = row["spec"]
        pairs.setdefault(int(spec["pair_index"]), {})[int(spec["sign"])] = float(
            row["reward"]
        )
    if any(set(pair) != {-1, 1} for pair in pairs.values()):
        raise ValueError("confirmation does not contain complete antithetic pairs")
    pair_differences = [pair[1] - pair[-1] for _, pair in sorted(pairs.items())]
    perturbations = [
        float(row["clip_statistics"]["perturbation_rms"])
        for row in rows
        if row.get("reward") is not None
    ]
    reconstruction_errors = [
        float(row["weighted_reconstruction_error"])
        for row in rows
        if row.get("reward") is not None
    ]
    audits = [row["replay_audit"] for row in rows if "replay_audit" in row]
    replay_exact = bool(audits) and all(
        audit.get("exact") is True and all(audit.get("matches", {}).values())
        for audit in audits
    )
    metrics = {
        "candidate_count": len(rows),
        "failure_count": failures,
        "reward_mean": statistics.fmean(rewards),
        "reward_std": statistics.pstdev(rewards),
        "pair_mean_abs_difference": statistics.fmean(
            abs(value) for value in pair_differences
        ),
        "pair_reward_differences": pair_differences,
        "perturbation_rms_max": max(perturbations),
        "reconstruction_error_ratio": statistics.fmean(reconstruction_errors)
        / quasar_error,
        "replay_exact": replay_exact,
    }
    metrics["passes"] = (
        failures == 0
        and metrics["reward_std"] >= REWARD_STD_MIN
        and metrics["pair_mean_abs_difference"] >= PAIR_ABS_DIFF_MIN
        and metrics["perturbation_rms_max"] <= PERTURBATION_RMS_MAX
        and metrics["reconstruction_error_ratio"] <= RECONSTRUCTION_RATIO_MAX
        and replay_exact
    )
    return metrics


def center_non_degrading(evaluation: dict, expected_sha256: str) -> tuple[bool, dict]:
    records = [
        record
        for record in evaluation["records"]
        if record["clip_sha256"] == expected_sha256
    ]
    if len(records) != 1:
        raise ValueError("N=30 center evaluation does not identify the replayed center")
    label = records[0]["label"]
    comparisons = [
        comparison
        for comparison in evaluation["comparisons"]
        if comparison["label"] == label
    ]
    if len(comparisons) != 1:
        raise ValueError("N=30 center comparison is missing or ambiguous")
    return comparisons[0]["non_degrading"] is True, comparisons[0]


def finalize(root: Path) -> dict:
    root = root.expanduser().resolve()
    preflight = load_json(root / "summary.json")
    selected_sigma = preflight["selected_sigma_preflight"]
    primary_alpha = load_json(root / "alpha-replay-pre-registered.json")
    primary_eval = load_json(root / "center-eval-next-batch.json")
    fallback_alpha = load_json(root / "alpha-replay-fallback.json")
    fallback_eval = load_json(root / "center-eval-fallback-episode2.json")
    selected_alpha, alpha_stage = select_alpha(
        primary_alpha,
        primary_eval,
        fallback_alpha,
        fallback_eval,
    )
    if selected_sigma is None or selected_alpha is None:
        raise ValueError("preflight did not select both sigma and alpha")

    confirmation_dir = root / "sigma-0p003-n30-p8-confirmation"
    run = load_json(confirmation_dir / "run.json")
    initialization = load_json(confirmation_dir / "initialization.json")
    generation = load_json(confirmation_dir / "generation-0000-summary.json")
    rows = load_jsonl(confirmation_dir / "generation-0000-candidates.jsonl")
    config = run["es_config"]
    if not (
        config["population_size"] == 30
        and math.isclose(config["sigma"], selected_sigma, rel_tol=0.0, abs_tol=1e-15)
        and math.isclose(
            config["learning_rate"], selected_alpha, rel_tol=0.0, abs_tol=1e-15
        )
        and config["seed"] == 20260905
        and run["generation"]["episode_size"] == 8
        and run["generation"]["schedule_seed"] == 20260904
        and run["completed_generations"] == 1
    ):
        raise ValueError("N=30 confirmation config differs from the frozen protocol")
    metrics = confirmation_metrics(
        rows,
        quasar_error=float(
            initialization["quasar"]["weighted_reconstruction_error"]
        ),
    )
    validation = load_json(root / "n30-validation.json")
    validation_passes = (
        validation.get("valid") is True
        and validation.get("candidate_count") == 30
        and validation.get("failure_count") == 0
        and validation.get("replay_audit_count", 0) >= 1
    )
    n30_alpha = load_json(root / "n30-alpha-replay.json")
    replay_records = [
        record
        for record in n30_alpha["records"]
        if math.isclose(
            float(record["alpha"]), selected_alpha, rel_tol=0.0, abs_tol=1e-15
        )
    ]
    if len(replay_records) != 1:
        raise ValueError("N=30 alpha replay does not contain the selected alpha")
    replay_record = replay_records[0]
    center_hash_matches = (
        replay_record["center_clip_sha256"] == generation["center_clip_sha256"]
    )
    clip_step_passes = float(replay_record["clip_step_rms"]) <= CLIP_STEP_RMS_MAX
    n30_eval = load_json(root / "n30-center-eval-episode3.json")
    non_degrading, center_comparison = center_non_degrading(
        n30_eval,
        replay_record["center_clip_sha256"],
    )
    passes = (
        metrics["passes"]
        and validation_passes
        and center_hash_matches
        and clip_step_passes
        and non_degrading
    )
    candidate_config = {
        "sigma": selected_sigma,
        "alpha": selected_alpha,
        "alpha_selection_stage": alpha_stage,
        "population_size": 30,
        "episode_size": 8,
        "es_seed": 20260905,
        "schedule_seed": 20260904,
        "dimension": config["dimension"],
        "shaping": config["shaping"],
        "clip_min": config["clip_min"],
        "clip_max": config["clip_max"],
    }
    return {
        "schema_version": 1,
        "passes": passes,
        "candidate_config": candidate_config,
        "frozen_config": candidate_config if passes else None,
        "confirmation": {
            **metrics,
            "validation_passes": validation_passes,
            "center_hash_matches": center_hash_matches,
            "clip_step_rms": replay_record["clip_step_rms"],
            "clip_step_passes": clip_step_passes,
            "center_reward": center_comparison,
        },
        "artifact_paths": {
            "preflight": str(root / "summary.json"),
            "primary_alpha": str(root / "alpha-replay-pre-registered.json"),
            "primary_center_evaluation": str(root / "center-eval-next-batch.json"),
            "fallback_alpha": str(root / "alpha-replay-fallback.json"),
            "fallback_center_evaluation": str(
                root / "center-eval-fallback-episode2.json"
            ),
            "n30_run": str(confirmation_dir / "run.json"),
            "n30_validation": str(root / "n30-validation.json"),
            "n30_alpha": str(root / "n30-alpha-replay.json"),
            "n30_center_evaluation": str(root / "n30-center-eval-episode3.json"),
        },
    }


def main() -> None:
    args = parse_args()
    out_path = args.out.expanduser().resolve()
    if out_path.exists():
        raise FileExistsError(f"refusing existing calibration output: {out_path}")
    report = finalize(args.calibration_root)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = out_path.with_name(f".{out_path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, out_path)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
