"""Integrity checks shared by Mage ES center-gate protocols."""

from __future__ import annotations

import math


def _finite_number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"center evaluation {field} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"center evaluation {field} must be a finite number")
    return result


def exact_replay(evaluation: dict) -> bool:
    audit = evaluation["records"][0].get("replay_audit", {})
    matches = audit.get("matches")
    return (
        audit.get("exact") is True
        and isinstance(matches, dict)
        and bool(matches)
        and all(matches.values())
    )


def comparison_for_clip(evaluation: dict, clip_sha256: str) -> dict:
    """Recompute a paired comparison and reject stale derived fields."""
    records = [
        record
        for record in evaluation["records"]
        if record["clip_sha256"] == clip_sha256
    ]
    if len(records) != 1:
        raise ValueError("center evaluation does not identify a clip exactly once")
    comparisons = [
        comparison
        for comparison in evaluation["comparisons"]
        if comparison["label"] == records[0]["label"]
    ]
    if len(comparisons) != 1:
        raise ValueError("center evaluation comparison is missing or ambiguous")
    baseline_reward = _finite_number(
        evaluation["records"][0].get("reward"), field="baseline reward"
    )
    candidate_reward = _finite_number(
        records[0].get("reward"), field="candidate reward"
    )
    reward_delta = candidate_reward - baseline_reward
    non_degrading = candidate_reward >= baseline_reward
    reported_delta = _finite_number(
        comparisons[0].get("reward_delta_vs_baseline"),
        field="comparison reward delta",
    )
    if not math.isclose(reported_delta, reward_delta, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("center evaluation comparison reward delta is inconsistent")
    if comparisons[0].get("non_degrading") is not non_degrading:
        raise ValueError("center evaluation comparison non-degrading flag is inconsistent")
    return {
        **comparisons[0],
        "reward_delta_vs_baseline": reward_delta,
        "non_degrading": non_degrading,
    }


def source_run_hashes(evaluation: dict) -> list[str]:
    source = evaluation["source"]
    return [
        source["run_sha256"],
        *[
            record["run_sha256"]
            for record in source.get("additional_source_runs", [])
        ],
    ]


def batch_ids(
    report: dict,
    *,
    expected_episode: int,
    expected_size: int,
    expected_schedule_seed: int,
) -> set[str]:
    episode = report["episode"]
    if (
        episode["index"] != expected_episode
        or episode["schedule_seed"] != expected_schedule_seed
    ):
        raise ValueError(f"center evaluation must use episode {expected_episode}")
    sample_ids = [str(value) for value in episode["sample_ids"]]
    if len(sample_ids) != len(set(sample_ids)) or len(sample_ids) != expected_size:
        raise ValueError("center evaluation prompt batch is invalid")
    return set(sample_ids)
