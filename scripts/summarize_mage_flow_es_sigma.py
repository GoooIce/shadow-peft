#!/usr/bin/env python3
"""Summarize preregistered Mage ES sigma preflight arms."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
from pathlib import Path

SIGMAS = (0.003, 0.01, 0.03, 0.1)
REWARD_STD_MIN = 0.02
PAIR_ABS_DIFF_MIN = 0.02
RECONSTRUCTION_RATIO_MAX = 1.05
PERTURBATION_RMS_MAX = 0.03


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sweep_root", type=Path)
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


def arm_tag(sigma: float) -> str:
    return f"sigma-{sigma:g}".replace(".", "p")


def arm_metrics(rows: list[dict], *, quasar_error: float) -> dict:
    rewards = [float(row["reward"]) for row in rows if row.get("reward") is not None]
    failures = len(rows) - len(rewards)
    if not rewards:
        raise ValueError("sigma arm has no valid rewards")
    pairs: dict[int, dict[int, float]] = {}
    for row in rows:
        reward = row.get("reward")
        if reward is None:
            continue
        spec = row["spec"]
        pairs.setdefault(int(spec["pair_index"]), {})[int(spec["sign"])] = float(
            reward
        )
    if any(set(pair) != {-1, 1} for pair in pairs.values()):
        raise ValueError("sigma arm does not contain complete antithetic pairs")
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
    replay_audits = [row["replay_audit"] for row in rows if "replay_audit" in row]
    replay_exact = bool(replay_audits) and all(
        audit.get("exact") is True and all(audit.get("matches", {}).values())
        for audit in replay_audits
    )
    metrics = {
        "candidate_count": len(rows),
        "failure_count": failures,
        "reward_mean": statistics.fmean(rewards),
        "reward_std": statistics.pstdev(rewards),
        "reward_min": min(rewards),
        "reward_max": max(rewards),
        "pair_reward_differences": pair_differences,
        "pair_mean_abs_difference": statistics.fmean(
            abs(value) for value in pair_differences
        ),
        "perturbation_rms_mean": statistics.fmean(perturbations),
        "perturbation_rms_max": max(perturbations),
        "reconstruction_error_mean": statistics.fmean(reconstruction_errors),
        "reconstruction_error_ratio": statistics.fmean(reconstruction_errors)
        / quasar_error,
        "replay_exact": replay_exact,
    }
    metrics["passes_preflight"] = (
        failures == 0
        and metrics["reward_std"] >= REWARD_STD_MIN
        and metrics["pair_mean_abs_difference"] >= PAIR_ABS_DIFF_MIN
        and metrics["reconstruction_error_ratio"] <= RECONSTRUCTION_RATIO_MAX
        and metrics["perturbation_rms_max"] <= PERTURBATION_RMS_MAX
        and replay_exact
    )
    return metrics


def comparable_identity(experiment: dict) -> dict:
    identity = experiment["identity"]
    search = dict(identity["search"])
    search.pop("sigma")
    return {
        "implementation": identity["implementation"],
        "model": identity["model"],
        "assets": identity["assets"],
        "initialization_source": identity.get("initialization_source"),
        "search_except_sigma": search,
        "generation": identity["generation"],
        "reward_batch_size": identity["reward_batch_size"],
    }


def summarize(sweep_root: Path) -> dict:
    sweep_root = sweep_root.expanduser().resolve()
    arms = []
    reference_identity = None
    reference_specs = None
    reference_prompts = None
    reference_initial_hash = None
    for sigma in SIGMAS:
        arm_dir = sweep_root / arm_tag(sigma)
        run = load_json(arm_dir / "run.json")
        experiment = load_json(arm_dir / "experiment.json")
        initialization = load_json(arm_dir / "initialization.json")
        summary = load_json(arm_dir / "generation-0000-summary.json")
        rows = load_jsonl(arm_dir / "generation-0000-candidates.jsonl")
        if not math.isclose(
            float(run["es_config"]["sigma"]), sigma, rel_tol=0.0, abs_tol=1e-15
        ):
            raise ValueError(f"sigma differs from arm name: {arm_dir}")
        identity = comparable_identity(experiment)
        specs = [row["spec"] for row in rows]
        prompts = (summary["prompt_sample_ids"], summary["prompt_seeds"])
        initial_hash = initialization["clip_sha256"]
        if reference_identity is None:
            reference_identity = identity
            reference_specs = specs
            reference_prompts = prompts
            reference_initial_hash = initial_hash
        elif not (
            identity == reference_identity
            and specs == reference_specs
            and prompts == reference_prompts
            and initial_hash == reference_initial_hash
        ):
            raise ValueError("sigma arms do not share identity/specs/prompts/initialization")
        quasar_error = float(
            initialization["quasar"]["weighted_reconstruction_error"]
        )
        arms.append(
            {
                "sigma": sigma,
                "path": str(arm_dir),
                **arm_metrics(rows, quasar_error=quasar_error),
            }
        )
    passing = [arm["sigma"] for arm in arms if arm["passes_preflight"]]
    return {
        "schema_version": 1,
        "thresholds": {
            "reward_std_min": REWARD_STD_MIN,
            "pair_mean_abs_difference_min": PAIR_ABS_DIFF_MIN,
            "reconstruction_error_ratio_max": RECONSTRUCTION_RATIO_MAX,
            "perturbation_rms_max": PERTURBATION_RMS_MAX,
            "failure_count": 0,
            "replay_exact": True,
        },
        "arms": arms,
        "selected_sigma_preflight": min(passing) if passing else None,
    }


def main() -> None:
    args = parse_args()
    out_path = args.out.expanduser().resolve()
    if out_path.exists():
        raise FileExistsError(f"refusing existing sigma summary: {out_path}")
    report = summarize(args.sweep_root)
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
