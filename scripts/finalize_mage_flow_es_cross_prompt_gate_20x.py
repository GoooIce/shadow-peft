#!/usr/bin/env python3
"""Apply the preregistered EXP-007 20x cross-prompt center gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
from pathlib import Path

from shadow_lab.mage_es_gate import (
    batch_ids,
    comparison_for_clip,
    exact_replay,
    source_run_hashes,
)

OBSERVATION_EPISODES = list(range(24))
SELECTION_EPISODES = list(range(24, 32))
CONFIRMATION_EPISODES = list(range(32, 40))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def generation_budget() -> dict[str, int]:
    budget = {
        "gradient_candidates": 24 * 80 * 8,
        "gradient_replay_audits": 24 * 8,
        "selection": 8 * 4 * 8,
        "confirmation": 8 * 3 * 8,
    }
    budget["total"] = sum(budget.values())
    return budget


def _validate_protocol(gradient_report: dict) -> dict:
    protocol = gradient_report["protocol"]
    if not (
        protocol["observation_episodes"] == OBSERVATION_EPISODES
        and protocol["gradient_batch_count"] == 24
        and protocol["population_size"] == 80
        and protocol["episode_size"] == 8
        and math.isclose(protocol["sigma"], 0.003, rel_tol=0.0, abs_tol=1e-15)
        and protocol["es_seed"] == 20260907
        and protocol["schedule_seed"] == 20260904
        and protocol["shaping"] == "zscore"
        and protocol["alphas"] == [0.00003, 0.0001]
        and math.isclose(
            protocol["clip_step_rms_max"], 0.02, rel_tol=0.0, abs_tol=1e-15
        )
    ):
        raise ValueError("gradient report differs from the frozen EXP-007 protocol")
    sources = gradient_report["sources"]
    if len(sources) != 24 or any(
        source["failure_count"] != 0
        or source["reward_std"] < 0.02
        or source["replay_exact"] is not True
        for source in sources
    ):
        raise ValueError("gradient sources fail the frozen EXP-007 observation gates")
    observed_ids: set[str] = set()
    for expected_episode, source in zip(
        OBSERVATION_EPISODES, sources, strict=True
    ):
        if source["episode_index"] != expected_episode:
            raise ValueError("gradient source episode order differs")
        prompt_ids = [str(value) for value in source["prompt_sample_ids"]]
        if len(prompt_ids) != len(set(prompt_ids)) or len(prompt_ids) != 8:
            raise ValueError("gradient source prompt batch is invalid")
        overlap = observed_ids.intersection(prompt_ids)
        if overlap:
            raise ValueError(f"gradient source prompt batches overlap: {sorted(overlap)}")
        observed_ids.update(prompt_ids)
    candidates = gradient_report["candidates"]
    if [candidate["alpha"] for candidate in candidates] != protocol["alphas"]:
        raise ValueError("gradient candidates differ from the frozen alpha order")
    if any(
        candidate["clip_step_rms"] > protocol["clip_step_rms_max"]
        for candidate in candidates
    ):
        raise ValueError("average-gradient center exceeds the clip-step gate")
    return protocol


def _evaluate_batches(
    reports: list[dict],
    *,
    expected_episodes: list[int],
    expected_source_hashes: list[str],
    initial_clip_sha256: str,
    candidate_clip_sha256s: list[str],
    used_prompt_ids: set[str],
    schedule_seed: int,
    episode_size: int,
) -> tuple[dict[str, list[dict]], set[str]]:
    if len(reports) != len(expected_episodes):
        raise ValueError("center evaluation batch count differs from frozen protocol")
    comparisons = {clip_sha256: [] for clip_sha256 in candidate_clip_sha256s}
    expected_hashes = {initial_clip_sha256, *candidate_clip_sha256s}
    batch_prompt_ids: set[str] = set()
    for report, expected_episode in zip(reports, expected_episodes, strict=True):
        if source_run_hashes(report) != expected_source_hashes:
            raise ValueError("center evaluation source runs differ from gradient sources")
        if report["records"][0]["clip_sha256"] != initial_clip_sha256:
            raise ValueError("center evaluation baseline is not the initial center")
        if not exact_replay(report):
            raise ValueError("center evaluation initial replay is not exact")
        actual_hashes = {record["clip_sha256"] for record in report["records"]}
        if actual_hashes != expected_hashes or len(report["records"]) != len(
            expected_hashes
        ):
            raise ValueError("center evaluation contains the wrong centers")
        current_ids = batch_ids(
            report,
            expected_episode=expected_episode,
            expected_size=episode_size,
            expected_schedule_seed=schedule_seed,
        )
        overlap = current_ids.intersection(used_prompt_ids | batch_prompt_ids)
        if overlap:
            raise ValueError(f"center evaluation prompt batches overlap: {sorted(overlap)}")
        batch_prompt_ids.update(current_ids)
        for clip_sha256 in candidate_clip_sha256s:
            comparisons[clip_sha256].append(
                comparison_for_clip(report, clip_sha256)
            )
    return comparisons, batch_prompt_ids


def _candidate_summary(candidate: dict, comparisons: list[dict]) -> dict:
    deltas = [float(row["reward_delta_vs_baseline"]) for row in comparisons]
    non_degrading_batches = sum(row["non_degrading"] is True for row in comparisons)
    mean_delta = statistics.fmean(deltas)
    return {
        "alpha": candidate["alpha"],
        "center_clip_sha256": candidate["center_clip_sha256"],
        "reward_deltas": deltas,
        "mean_reward_delta": mean_delta,
        "non_degrading_batches": non_degrading_batches,
        "batch_count": len(comparisons),
        "passes": mean_delta > 0 and non_degrading_batches >= 6,
    }


def summarize_gate(
    gradient_report: dict,
    selection_evaluations: list[dict],
    confirmation_evaluations: list[dict] | None = None,
) -> dict:
    protocol = _validate_protocol(gradient_report)
    sources = gradient_report["sources"]
    expected_source_hashes = [source["run_sha256"] for source in sources]
    observed_ids = {
        str(sample_id)
        for source in sources
        for sample_id in source["prompt_sample_ids"]
    }
    initial = gradient_report["initial_effective_center"]
    candidates = gradient_report["candidates"]
    selection_comparisons, selection_ids = _evaluate_batches(
        selection_evaluations,
        expected_episodes=SELECTION_EPISODES,
        expected_source_hashes=expected_source_hashes,
        initial_clip_sha256=initial["clip_sha256"],
        candidate_clip_sha256s=[
            candidate["center_clip_sha256"] for candidate in candidates
        ],
        used_prompt_ids=observed_ids,
        schedule_seed=protocol["schedule_seed"],
        episode_size=protocol["episode_size"],
    )
    selection_summaries = [
        _candidate_summary(
            candidate,
            selection_comparisons[candidate["center_clip_sha256"]],
        )
        for candidate in candidates
    ]
    eligible = [
        (candidate, summary)
        for candidate, summary in zip(candidates, selection_summaries, strict=True)
        if summary["passes"]
    ]
    selection_passes = bool(eligible)
    if eligible:
        selected, selected_summary = max(
            eligible, key=lambda pair: pair[0]["alpha"]
        )
        selected_role = "eligible"
    else:
        selected, selected_summary = max(
            zip(candidates, selection_summaries, strict=True),
            key=lambda pair: (pair[1]["mean_reward_delta"], pair[0]["alpha"]),
        )
        selected_role = "diagnostic"

    base = {
        "schema_version": 1,
        "generation_budget": generation_budget(),
        "selection_passes": selection_passes,
        "selected_candidate": selected,
        "selected_candidate_role": selected_role,
        "selection": {
            "episodes": SELECTION_EPISODES,
            "candidates": selection_summaries,
            "selected_summary": selected_summary,
        },
    }
    if confirmation_evaluations is None:
        return {
            **base,
            "confirmation": None,
            "status": "awaiting-confirmation",
            "passes": False,
            "frozen_config": None,
        }

    confirmation_comparisons, confirmation_ids = _evaluate_batches(
        confirmation_evaluations,
        expected_episodes=CONFIRMATION_EPISODES,
        expected_source_hashes=expected_source_hashes,
        initial_clip_sha256=initial["clip_sha256"],
        candidate_clip_sha256s=[selected["center_clip_sha256"]],
        used_prompt_ids=observed_ids | selection_ids,
        schedule_seed=protocol["schedule_seed"],
        episode_size=protocol["episode_size"],
    )
    if len(observed_ids | selection_ids | confirmation_ids) != 320:
        raise ValueError("EXP-007 must consume every es_train prompt exactly once")
    confirmation_summary = _candidate_summary(
        selected,
        confirmation_comparisons[selected["center_clip_sha256"]],
    )
    passes = selection_passes and confirmation_summary["passes"]
    status = (
        "passed"
        if passes
        else "stopped-selection"
        if not selection_passes
        else "stopped-confirmation"
    )
    frozen_config = None
    if passes:
        frozen_config = {
            "sigma": protocol["sigma"],
            "alpha": selected["alpha"],
            "population_size": protocol["population_size"],
            "episode_size": protocol["episode_size"],
            "gradient_batches": protocol["gradient_batch_count"],
            "es_seed": protocol["es_seed"],
            "schedule_seed": protocol["schedule_seed"],
            "shaping": protocol["shaping"],
        }
    return {
        **base,
        "confirmation": {
            "episodes": CONFIRMATION_EPISODES,
            **confirmation_summary,
        },
        "status": status,
        "passes": passes,
        "frozen_config": frozen_config,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gradient-report", required=True, type=Path)
    parser.add_argument(
        "--selection-evaluation", action="append", required=True, type=Path
    )
    parser.add_argument(
        "--confirmation-evaluation", action="append", type=Path
    )
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gradient_path = args.gradient_report.expanduser().resolve()
    selection_paths = [path.expanduser().resolve() for path in args.selection_evaluation]
    confirmation_paths = (
        [path.expanduser().resolve() for path in args.confirmation_evaluation]
        if args.confirmation_evaluation is not None
        else None
    )
    out_path = args.out.expanduser().resolve()
    if out_path.exists():
        raise FileExistsError(f"refusing existing EXP-007 gate output: {out_path}")
    result = summarize_gate(
        load_json(gradient_path),
        [load_json(path) for path in selection_paths],
        (
            [load_json(path) for path in confirmation_paths]
            if confirmation_paths is not None
            else None
        ),
    )
    result["artifacts"] = {
        "gradient_report": {
            "path": str(gradient_path),
            "sha256": sha256_file(gradient_path),
        },
        "selection_evaluations": [
            {"path": str(path), "sha256": sha256_file(path)}
            for path in selection_paths
        ],
    }
    if confirmation_paths is not None:
        result["artifacts"]["confirmation_evaluations"] = [
            {"path": str(path), "sha256": sha256_file(path)}
            for path in confirmation_paths
        ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = out_path.with_name(f".{out_path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, out_path)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
