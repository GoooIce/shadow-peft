"""Pure GenEval-compatible decision logic for the GB10 evaluation port.

The original project couples this logic to MMDetection 2.x.  Keeping the
decision layer free of detector/runtime dependencies lets us test it exactly
while using a current, local ModelScope detector on SM121.
"""

from __future__ import annotations

import math
import random
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .image_eval import FrozenEvalSample, canonical_json, sha256_text

GENEVAL_TASKS = (
    "single_object",
    "two_object",
    "counting",
    "colors",
    "position",
    "color_attr",
)


@dataclass(frozen=True)
class Detection:
    class_name: str
    score: float
    bbox: tuple[float, float, float, float]
    color: str | None = None

    def __post_init__(self) -> None:
        if not self.class_name or not math.isfinite(self.score):
            raise ValueError("detection class and score must be valid")
        if len(self.bbox) != 4 or not all(math.isfinite(value) for value in self.bbox):
            raise ValueError("detection bbox must contain four finite coordinates")


@dataclass(frozen=True)
class GenEvalDecision:
    correct: bool
    reasons: tuple[str, ...]


def _iou(left: Detection, right: Detection) -> float:
    left_area = max(left.bbox[2] - left.bbox[0] + 1, 0) * max(
        left.bbox[3] - left.bbox[1] + 1, 0
    )
    right_area = max(right.bbox[2] - right.bbox[0] + 1, 0) * max(
        right.bbox[3] - right.bbox[1] + 1, 0
    )
    intersection = max(min(left.bbox[2], right.bbox[2]) - max(left.bbox[0], right.bbox[0]) + 1, 0) * max(
        min(left.bbox[3], right.bbox[3]) - max(left.bbox[1], right.bbox[1]) + 1,
        0,
    )
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def filter_detections(
    detections: Sequence[Detection],
    *,
    tag: str,
    threshold: float = 0.3,
    counting_threshold: float = 0.9,
    max_objects: int = 16,
    max_overlap: float = 1.0,
) -> dict[str, list[Detection]]:
    """Apply the original GenEval score, ordering, cap and optional NMS policy."""

    if max_objects <= 0 or not 0 <= max_overlap <= 1:
        raise ValueError("invalid GenEval filtering policy")
    confidence = counting_threshold if tag == "counting" else threshold
    by_class: dict[str, list[Detection]] = defaultdict(list)
    for detection in detections:
        if detection.score > confidence:
            by_class[detection.class_name].append(detection)
    filtered: dict[str, list[Detection]] = {}
    for class_name, values in by_class.items():
        ordered = sorted(values, key=lambda item: item.score, reverse=True)
        kept: list[Detection] = []
        for detection in ordered:
            if max_overlap == 1.0 or all(
                _iou(detection, previous) < max_overlap for previous in kept
            ):
                kept.append(detection)
            if len(kept) == max_objects:
                break
        if kept:
            filtered[class_name] = kept
    return filtered


def relative_positions(
    left: Detection, right: Detection, *, position_threshold: float = 0.1
) -> set[str]:
    """Return the original GenEval coarse relations of ``left`` to ``right``."""

    if position_threshold < 0:
        raise ValueError("position_threshold must be non-negative")
    left_center = (
        (left.bbox[0] + left.bbox[2]) / 2,
        (left.bbox[1] + left.bbox[3]) / 2,
    )
    right_center = (
        (right.bbox[0] + right.bbox[2]) / 2,
        (right.bbox[1] + right.bbox[3]) / 2,
    )
    left_dim = (abs(left.bbox[2] - left.bbox[0]), abs(left.bbox[3] - left.bbox[1]))
    right_dim = (
        abs(right.bbox[2] - right.bbox[0]),
        abs(right.bbox[3] - right.bbox[1]),
    )
    offset = (left_center[0] - right_center[0], left_center[1] - right_center[1])
    revised = tuple(
        math.copysign(
            max(abs(value) - position_threshold * (left_size + right_size), 0.0),
            value,
        )
        if value
        else 0.0
        for value, left_size, right_size in zip(offset, left_dim, right_dim, strict=True)
    )
    if all(abs(value) < 1e-3 for value in revised):
        return set()
    norm = math.hypot(*offset)
    dx, dy = revised[0] / norm, revised[1] / norm
    relations = set()
    if dx < -0.5:
        relations.add("left of")
    if dx > 0.5:
        relations.add("right of")
    if dy < -0.5:
        relations.add("above")
    if dy > 0.5:
        relations.add("below")
    return relations


def evaluate_metadata(
    metadata: Mapping,
    objects: Mapping[str, Sequence[Detection]],
    *,
    position_threshold: float = 0.1,
) -> GenEvalDecision:
    """Evaluate one image using the rule order from official GenEval."""

    reasons: list[str] = []
    matched_groups: list[Sequence[Detection] | None] = []
    for requirement in metadata.get("include", []):
        class_name = requirement["class"]
        count = int(requirement["count"])
        matched = True
        found = list(objects.get(class_name, ()))[:count]
        if len(found) < count:
            matched = False
            reasons.append(f"expected {class_name}>={count}, found {len(found)}")
        else:
            expected_color = requirement.get("color")
            if expected_color is not None:
                colors = [item.color for item in found]
                actual = colors.count(expected_color)
                if actual < count:
                    matched = False
                    reasons.append(
                        f"expected {expected_color} {class_name}>={count}, found {actual}"
                    )
            if "position" in requirement and matched:
                expected_relation, target_group = requirement["position"]
                target = matched_groups[int(target_group)]
                if target is None:
                    matched = False
                    reasons.append(
                        f"no target for {class_name} to be {expected_relation}"
                    )
                else:
                    for item in found:
                        for target_item in target:
                            actual_relations = relative_positions(
                                item,
                                target_item,
                                position_threshold=position_threshold,
                            )
                            if expected_relation not in actual_relations:
                                matched = False
                                actual = " and ".join(sorted(actual_relations))
                                reasons.append(
                                    f"expected {class_name} {expected_relation} target, "
                                    f"found {actual} target"
                                )
                                break
                        if not matched:
                            break
        matched_groups.append(found if matched else None)

    for requirement in metadata.get("exclude", []):
        class_name = requirement["class"]
        count = int(requirement["count"])
        found_count = len(objects.get(class_name, ()))
        if found_count >= count:
            reasons.append(f"expected {class_name}<{count}, found {found_count}")
    return GenEvalDecision(not reasons, tuple(reasons))


def summarize_results(records: Sequence[Mapping]) -> dict:
    """Compute official-style per-task means and the equal-task overall."""

    grouped: dict[str, list[float]] = defaultdict(list)
    for record in records:
        tag = str(record.get("tag", ""))
        if tag not in GENEVAL_TASKS:
            raise ValueError(f"unknown GenEval task: {tag!r}")
        correct = record.get("correct")
        if not isinstance(correct, bool):
            raise ValueError("GenEval result correct field must be boolean")
        grouped[tag].append(float(correct))
    missing = [task for task in GENEVAL_TASKS if not grouped[task]]
    if missing:
        raise ValueError(f"GenEval summary is missing tasks: {missing}")
    task_scores = {task: statistics.fmean(grouped[task]) for task in GENEVAL_TASKS}
    return {
        "images": len(records),
        "task_scores": task_scores,
        "overall_equal_task_mean": statistics.fmean(task_scores.values()),
    }


def expand_geneval_samples(
    samples: Sequence[FrozenEvalSample],
    *,
    replicas: int = 4,
    seed_stride: int = 100_000_000,
) -> list[FrozenEvalSample]:
    """Expand prompt rows into deterministic, explicitly clustered image seeds."""

    if replicas <= 0 or seed_stride <= 0:
        raise ValueError("replicas and seed_stride must be positive")
    expanded = []
    for sample in samples:
        if sample.source != "geneval":
            raise ValueError(f"cannot expand non-GenEval sample: {sample.sample_id}")
        for replica in range(replicas):
            metadata = sample.metadata
            metadata["_geneval_protocol"] = {
                "parent_sample_id": sample.sample_id,
                "replica": replica,
                "replicas": replicas,
                "seed_stride": seed_stride,
            }
            expanded.append(
                FrozenEvalSample(
                    sample_id=sha256_text(f"{sample.sample_id}\0geneval\0{replica}")[:24],
                    source=sample.source,
                    source_revision=sample.source_revision,
                    source_id=sample.source_id,
                    prompt=sample.prompt,
                    stratum=sample.stratum,
                    split=sample.split,
                    seed=sample.seed + replica * seed_stride,
                    metadata_json=canonical_json(metadata),
                )
            )
    if len({sample.sample_id for sample in expanded}) != len(expanded):
        raise RuntimeError("derived GenEval sample IDs collided")
    if len({sample.seed for sample in expanded}) != len(expanded):
        raise RuntimeError("derived GenEval seeds collided")
    return expanded


def paired_clustered_bootstrap(
    candidate: Sequence[Mapping],
    reference: Sequence[Mapping],
    *,
    resamples: int = 10_000,
    seed: int = 20260904,
    confidence: float = 0.95,
) -> dict:
    """Paired bootstrap carrying a prompt's replicas as one task-stratified cluster."""

    if resamples <= 0 or not 0 < confidence < 1:
        raise ValueError("invalid bootstrap configuration")

    def indexed(records: Sequence[Mapping]) -> dict[tuple[str, int], Mapping]:
        result = {}
        for record in records:
            key = (str(record.get("sample_id", "")), int(record.get("seed", -1)))
            if not key[0] or key[1] < 0 or key in result:
                raise ValueError("GenEval pair identities must be unique and valid")
            result[key] = record
        return result

    candidate_index = indexed(candidate)
    reference_index = indexed(reference)
    if set(candidate_index) != set(reference_index) or not candidate_index:
        raise ValueError("candidate and reference GenEval pairs differ")

    task_clusters: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for key, candidate_record in candidate_index.items():
        reference_record = reference_index[key]
        tag = str(candidate_record.get("tag", ""))
        cluster_id = str(candidate_record.get("cluster_id", ""))
        if (
            tag not in GENEVAL_TASKS
            or tag != reference_record.get("tag")
            or not cluster_id
            or cluster_id != reference_record.get("cluster_id")
        ):
            raise ValueError("paired GenEval task or cluster identity differs")
        left = candidate_record.get("correct")
        right = reference_record.get("correct")
        if not isinstance(left, bool) or not isinstance(right, bool):
            raise ValueError("paired GenEval correct fields must be boolean")
        task_clusters[tag][cluster_id].append(float(left) - float(right))
    missing = [task for task in GENEVAL_TASKS if not task_clusters[task]]
    if missing:
        raise ValueError(f"paired GenEval bootstrap is missing tasks: {missing}")

    cluster_means = {
        task: [statistics.fmean(values) for values in clusters.values()]
        for task, clusters in task_clusters.items()
    }
    observed_by_task = {
        task: statistics.fmean(values) for task, values in cluster_means.items()
    }
    observed = statistics.fmean(observed_by_task.values())
    rng = random.Random(seed)
    bootstrapped = []
    for _ in range(resamples):
        task_draws = []
        for task in GENEVAL_TASKS:
            values = cluster_means[task]
            task_draws.append(
                statistics.fmean(values[rng.randrange(len(values))] for _ in values)
            )
        bootstrapped.append(statistics.fmean(task_draws))
    bootstrapped.sort()
    tail = (1 - confidence) / 2
    lower_index = max(0, math.floor(tail * (resamples - 1)))
    upper_index = min(resamples - 1, math.ceil((1 - tail) * (resamples - 1)))
    return {
        "mean_difference": observed,
        "task_mean_differences": observed_by_task,
        "confidence": confidence,
        "lower": bootstrapped[lower_index],
        "upper": bootstrapped[upper_index],
        "resamples": resamples,
        "seed": seed,
        "cluster_counts": {
            task: len(task_clusters[task]) for task in GENEVAL_TASKS
        },
    }
