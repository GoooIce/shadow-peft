"""Deterministic quality and preference summaries over annotation exports."""

from __future__ import annotations

import math
import random
import statistics
from collections import defaultdict


def _current_answers(export: dict) -> list[dict]:
    return [
        answer
        for answer in export["answers"]
        if answer["event_type"] == "answer" and not answer["is_superseded"]
    ]


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _pairwise_agreement(answers: list[dict]) -> dict:
    by_comparison: dict[str, list[str]] = defaultdict(list)
    for answer in answers:
        by_comparison[answer["comparison_id"]].append(answer["canonical_decision"])
    agreements = 0
    pairs = 0
    overlapping = 0
    for decisions in by_comparison.values():
        if len(decisions) < 2:
            continue
        overlapping += 1
        for left_index, left in enumerate(decisions):
            for right in decisions[left_index + 1 :]:
                pairs += 1
                agreements += left == right
    return {
        "overlap_comparisons": overlapping,
        "pair_count": pairs,
        "exact_agreement": agreements / pairs if pairs else None,
    }


def _repeat_consistency(answers: list[dict]) -> dict:
    groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for answer in answers:
        if answer["qc_kind"] == "repeat" and answer["qc_group_id"]:
            groups[(answer["annotator_id"], answer["qc_group_id"])].append(
                answer["canonical_decision"]
            )
    comparable = [decisions for decisions in groups.values() if len(decisions) >= 2]
    consistent = sum(len(set(decisions)) == 1 for decisions in comparable)
    return {
        "repeat_groups": len(comparable),
        "consistent_groups": consistent,
        "consistency": consistent / len(comparable) if comparable else None,
    }


def _bradley_terry(answers: list[dict]) -> dict[str, float]:
    arms = sorted(
        {
            arm
            for answer in answers
            for arm in (
                answer["reveal"]["left_arm_id"],
                answer["reveal"]["right_arm_id"],
            )
        }
    )
    if len(arms) < 2:
        return {arm: 0.0 for arm in arms}
    wins = {arm: 0.0 for arm in arms}
    matches: dict[tuple[str, str], int] = defaultdict(int)
    for answer in answers:
        if answer["presentation_decision"] == "skip":
            continue
        left = answer["reveal"]["left_arm_id"]
        right = answer["reveal"]["right_arm_id"]
        matches[tuple(sorted((left, right)))] += 1
        if answer["presentation_decision"] == "left":
            wins[left] += 1
        elif answer["presentation_decision"] == "right":
            wins[right] += 1
        else:
            wins[left] += 0.5
            wins[right] += 0.5
    strengths = {arm: 1.0 for arm in arms}
    for _ in range(200):
        updated = {}
        for arm in arms:
            denominator = 0.0
            for other in arms:
                if other == arm:
                    continue
                count = matches.get(tuple(sorted((arm, other))), 0)
                denominator += count / (strengths[arm] + strengths[other])
            updated[arm] = max(wins[arm] / denominator, 1e-12) if denominator else 1.0
        geometric_mean = math.exp(statistics.fmean(math.log(value) for value in updated.values()))
        updated = {arm: value / geometric_mean for arm, value in updated.items()}
        if max(abs(math.log(updated[arm] / strengths[arm])) for arm in arms) < 1e-10:
            strengths = updated
            break
        strengths = updated
    return {arm: math.log(value) for arm, value in strengths.items()}


def _arm_scores(answers: list[dict]) -> dict[str, float]:
    totals: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    for answer in answers:
        decision = answer["presentation_decision"]
        if decision == "skip":
            continue
        left = answer["reveal"]["left_arm_id"]
        right = answer["reveal"]["right_arm_id"]
        counts[left] += 1
        counts[right] += 1
        if decision == "left":
            totals[left] += 1
        elif decision == "right":
            totals[right] += 1
        else:
            totals[left] += 0.5
            totals[right] += 0.5
    return {arm: totals[arm] / count for arm, count in counts.items()}


def _bootstrap_arm_scores(answers: list[dict], *, samples: int, seed: int) -> dict[str, dict]:
    if samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    by_comparison: dict[str, list[dict]] = defaultdict(list)
    for answer in answers:
        by_comparison[answer["comparison_id"]].append(answer)
    groups = list(by_comparison.values())
    if not groups:
        return {}
    observed = _arm_scores(answers)
    replicates: dict[str, list[float]] = defaultdict(list)
    generator = random.Random(seed)
    for _ in range(samples):
        sampled = [generator.choice(groups) for _ in groups]
        scores = _arm_scores([answer for group in sampled for answer in group])
        for arm, score in scores.items():
            replicates[arm].append(score)
    return {
        arm: {
            "score": score,
            "paired_bootstrap_ci95": [
                _quantile(replicates[arm], 0.025),
                _quantile(replicates[arm], 0.975),
            ],
        }
        for arm, score in observed.items()
    }


def summarize_annotations(
    export: dict,
    *,
    bootstrap_samples: int = 2000,
    bootstrap_seed: int = 20260904,
) -> dict:
    answers = _current_answers(export)
    decisions = {decision: 0 for decision in ("left", "tie", "right", "skip")}
    for answer in answers:
        decisions[answer["presentation_decision"]] += 1
    decisive = decisions["left"] + decisions["right"]
    by_annotator: dict[str, dict[str, int]] = defaultdict(
        lambda: {decision: 0 for decision in decisions}
    )
    for answer in answers:
        by_annotator[answer["annotator_id"]][answer["presentation_decision"]] += 1
    report = {
        "schema_version": "shadow-ab-analysis/0.1",
        "export_mode": export["export_mode"],
        "current_answer_count": len(answers),
        "decision_counts": decisions,
        "skip_rate": decisions["skip"] / len(answers) if answers else None,
        "tie_rate": decisions["tie"] / len(answers) if answers else None,
        "left_choice_rate": decisions["left"] / decisive if decisive else None,
        "agreement": _pairwise_agreement(answers),
        "repeat_consistency": _repeat_consistency(answers),
        "by_annotator": dict(by_annotator),
    }
    if export["export_mode"] == "revealed":
        report["bradley_terry_log_strength"] = _bradley_terry(answers)
        report["arm_scores"] = _bootstrap_arm_scores(
            answers,
            samples=bootstrap_samples,
            seed=bootstrap_seed,
        )
    return report
