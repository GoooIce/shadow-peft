#!/usr/bin/env python3
"""Apply the preregistered EXP-006 cross-prompt center gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path


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


def exact_replay(evaluation: dict) -> bool:
    audit = evaluation["records"][0].get("replay_audit", {})
    matches = audit.get("matches")
    return (
        audit.get("exact") is True
        and isinstance(matches, dict)
        and bool(matches)
        and all(matches.values())
    )


def _finite_number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"center evaluation {field} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"center evaluation {field} must be a finite number")
    return result


def comparison_for_clip(evaluation: dict, clip_sha256: str) -> dict:
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


def generation_budget(protocol: dict) -> dict[str, int]:
    """Return the maximum frozen image-generation budget, including replay audits."""
    episode_size = int(protocol["episode_size"])
    observation_count = len(protocol["observation_episodes"])
    population_size = int(protocol["population_size"])
    candidate_count = len(protocol["alphas"])
    budget = {
        "gradient_candidates": observation_count * population_size * episode_size,
        "gradient_replay_audits": observation_count * episode_size,
        "episode_7_centers": (1 + candidate_count) * episode_size,
        "episode_7_replay_audit": episode_size,
        "episode_8_centers": 2 * episode_size,
        "episode_8_replay_audit": episode_size,
    }
    budget["total"] = sum(budget.values())
    return budget


def _validate_protocol(gradient_report: dict) -> dict:
    protocol = gradient_report["protocol"]
    if not (
        protocol["observation_episodes"] == [4, 5, 6]
        and protocol["population_size"] == 30
        and protocol["episode_size"] == 8
        and math.isclose(protocol["sigma"], 0.003, rel_tol=0.0, abs_tol=1e-15)
        and protocol["es_seed"] == 20260906
        and protocol["schedule_seed"] == 20260904
        and protocol["alphas"] == [0.00003, 0.0001]
        and math.isclose(
            protocol["clip_step_rms_max"], 0.02, rel_tol=0.0, abs_tol=1e-15
        )
    ):
        raise ValueError("gradient report differs from the frozen EXP-006 protocol")
    sources = gradient_report["sources"]
    if len(sources) != 3 or any(
        source["failure_count"] != 0
        or source["reward_std"] < 0.02
        or source["replay_exact"] is not True
        for source in sources
    ):
        raise ValueError("gradient sources fail the frozen observation gates")
    observed_ids: set[str] = set()
    for source in sources:
        prompt_ids = [str(value) for value in source["prompt_sample_ids"]]
        if len(prompt_ids) != len(set(prompt_ids)) or len(prompt_ids) != protocol[
            "episode_size"
        ]:
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


def _batch_ids(report: dict, *, expected_episode: int, expected_size: int) -> set[str]:
    episode = report["episode"]
    if episode["index"] != expected_episode or episode["schedule_seed"] != 20260904:
        raise ValueError(f"center evaluation must use episode {expected_episode}")
    sample_ids = [str(value) for value in episode["sample_ids"]]
    if len(sample_ids) != len(set(sample_ids)) or len(sample_ids) != expected_size:
        raise ValueError("center evaluation prompt batch is invalid")
    return set(sample_ids)


def source_run_hashes(evaluation: dict) -> list[str]:
    source = evaluation["source"]
    return [
        source["run_sha256"],
        *[
            record["run_sha256"]
            for record in source.get("additional_source_runs", [])
        ],
    ]


def summarize_gate(
    gradient_report: dict,
    gate_evaluation: dict,
    confirmation: dict | None = None,
) -> dict:
    protocol = _validate_protocol(gradient_report)
    expected_source_hashes = [
        source["run_sha256"] for source in gradient_report["sources"]
    ]
    if source_run_hashes(gate_evaluation) != expected_source_hashes:
        raise ValueError("episode 7 evaluation source runs differ from gradient sources")
    initial = gradient_report["initial_effective_center"]
    if gate_evaluation["records"][0]["clip_sha256"] != initial["clip_sha256"]:
        raise ValueError("episode 7 baseline is not the initial effective center")
    if not exact_replay(gate_evaluation):
        raise ValueError("episode 7 initial center replay is not exact")
    expected_gate_hashes = {
        initial["clip_sha256"],
        *[
            candidate["center_clip_sha256"]
            for candidate in gradient_report["candidates"]
        ],
    }
    actual_gate_hashes = {
        record["clip_sha256"] for record in gate_evaluation["records"]
    }
    if actual_gate_hashes != expected_gate_hashes or len(gate_evaluation["records"]) != 3:
        raise ValueError("episode 7 must evaluate only the initial and two candidate centers")
    gate_ids = _batch_ids(
        gate_evaluation,
        expected_episode=7,
        expected_size=protocol["episode_size"],
    )
    observed_ids = {
        str(sample_id)
        for source in gradient_report["sources"]
        for sample_id in source["prompt_sample_ids"]
    }
    if gate_ids.intersection(observed_ids):
        raise ValueError("episode 7 overlaps a gradient observation batch")

    passing = []
    gate_comparisons = []
    for candidate in gradient_report["candidates"]:
        comparison = comparison_for_clip(
            gate_evaluation, candidate["center_clip_sha256"]
        )
        gate_comparisons.append(
            {"alpha": candidate["alpha"], **comparison}
        )
        if comparison["non_degrading"] is True:
            passing.append(candidate)
    selected = max(passing, key=lambda candidate: candidate["alpha"]) if passing else None
    base = {
        "schema_version": 1,
        "generation_budget": generation_budget(protocol),
        "gate_passes": selected is not None,
        "selected_candidate": selected,
        "gate": {
            "episode": 7,
            "replay_exact": True,
            "comparisons": gate_comparisons,
        },
    }
    if selected is None:
        return {
            **base,
            "status": "stopped-gate",
            "passes": False,
            "frozen_config": None,
        }
    if confirmation is None:
        return {
            **base,
            "status": "awaiting-confirmation",
            "passes": False,
            "frozen_config": None,
        }

    if confirmation["records"][0]["clip_sha256"] != initial["clip_sha256"]:
        raise ValueError("episode 8 baseline is not the initial effective center")
    if source_run_hashes(confirmation) != expected_source_hashes:
        raise ValueError("episode 8 evaluation source runs differ from gradient sources")
    if not exact_replay(confirmation):
        raise ValueError("episode 8 initial center replay is not exact")
    expected_confirmation_hashes = {
        initial["clip_sha256"],
        selected["center_clip_sha256"],
    }
    actual_confirmation_hashes = {
        record["clip_sha256"] for record in confirmation["records"]
    }
    if (
        actual_confirmation_hashes != expected_confirmation_hashes
        or len(confirmation["records"]) != 2
    ):
        raise ValueError("episode 8 must evaluate only the initial and selected centers")
    confirmation_ids = _batch_ids(
        confirmation,
        expected_episode=8,
        expected_size=protocol["episode_size"],
    )
    if confirmation_ids.intersection(observed_ids | gate_ids):
        raise ValueError("episode 8 overlaps an earlier EXP-006 prompt batch")
    confirmation_comparison = comparison_for_clip(
        confirmation, selected["center_clip_sha256"]
    )
    passes = confirmation_comparison["non_degrading"] is True
    frozen_config = None
    if passes:
        frozen_config = {
            "sigma": protocol["sigma"],
            "alpha": selected["alpha"],
            "population_size": protocol["population_size"],
            "episode_size": protocol["episode_size"],
            "gradient_batches": len(protocol["observation_episodes"]),
            "es_seed": protocol["es_seed"],
            "schedule_seed": protocol["schedule_seed"],
            "shaping": protocol.get("shaping", "zscore"),
        }
    return {
        **base,
        "status": "passed" if passes else "stopped-confirmation",
        "passes": passes,
        "frozen_config": frozen_config,
        "confirmation": {
            "episode": 8,
            "replay_exact": True,
            "comparison": confirmation_comparison,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gradient-report", required=True, type=Path)
    parser.add_argument("--gate-evaluation", required=True, type=Path)
    parser.add_argument("--confirmation", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gradient_path = args.gradient_report.expanduser().resolve()
    gate_path = args.gate_evaluation.expanduser().resolve()
    confirmation_path = (
        args.confirmation.expanduser().resolve() if args.confirmation is not None else None
    )
    out_path = args.out.expanduser().resolve()
    if out_path.exists():
        raise FileExistsError(f"refusing existing cross-prompt gate output: {out_path}")
    result = summarize_gate(
        load_json(gradient_path),
        load_json(gate_path),
        load_json(confirmation_path) if confirmation_path is not None else None,
    )
    result["artifacts"] = {
        "gradient_report": {
            "path": str(gradient_path),
            "sha256": sha256_file(gradient_path),
        },
        "gate_evaluation": {
            "path": str(gate_path),
            "sha256": sha256_file(gate_path),
        },
    }
    if confirmation_path is not None:
        result["artifacts"]["confirmation"] = {
            "path": str(confirmation_path),
            "sha256": sha256_file(confirmation_path),
        }
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
