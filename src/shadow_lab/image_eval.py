"""Frozen image-evaluation protocol shared by ES and offline validation.

The module is deliberately model-agnostic.  Reward models receive image bytes
and immutable sample metadata, never an experiment-arm or candidate name.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
from collections import defaultdict, deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

EVAL_SPLITS = ("es_train", "checkpoint_val", "auto_test", "human_dev", "human_test")
EVAL_SPLIT_COUNTS = {
    "es_train": 320,
    "checkpoint_val": 60,
    "auto_test": 60,
    "human_dev": 30,
    "human_test": 30,
}
SOURCE_SPLIT_COUNTS = {
    "geneval": {
        "es_train": 256,
        "checkpoint_val": 48,
        "auto_test": 48,
        "human_dev": 24,
        "human_test": 24,
    },
    "text_rendering": {
        "es_train": 64,
        "checkpoint_val": 12,
        "auto_test": 12,
        "human_dev": 6,
        "human_test": 6,
    },
}
PRIMARY_REWARD_AXES = ("hps_v2_1", "clip_h14")


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SourcePrompt:
    source: str
    revision: str
    source_id: str
    prompt: str
    stratum: str
    metadata_json: str = "{}"

    def __post_init__(self) -> None:
        if not all((self.source, self.revision, self.source_id, self.prompt, self.stratum)):
            raise ValueError("source prompt identity fields must be non-empty")
        metadata = json.loads(self.metadata_json)
        if not isinstance(metadata, dict):
            raise ValueError("metadata_json must encode an object")
        if canonical_json(metadata) != self.metadata_json:
            raise ValueError("metadata_json must use canonical_json")


@dataclass(frozen=True)
class FrozenEvalSample:
    sample_id: str
    source: str
    source_revision: str
    source_id: str
    prompt: str
    stratum: str
    split: str
    seed: int
    metadata_json: str

    def __post_init__(self) -> None:
        if self.split not in EVAL_SPLITS:
            raise ValueError(f"unknown evaluation split: {self.split}")
        if self.seed < 0:
            raise ValueError("sample seed must be non-negative")

    @property
    def metadata(self) -> dict:
        return json.loads(self.metadata_json)

    def as_record(self) -> dict:
        return {
            "sample_id": self.sample_id,
            "source": self.source,
            "source_revision": self.source_revision,
            "source_id": self.source_id,
            "prompt": self.prompt,
            "stratum": self.stratum,
            "split": self.split,
            "seed": self.seed,
            "metadata": self.metadata,
        }


def _stable_rank(row: SourcePrompt, namespace: str, seed: int) -> str:
    return sha256_text(
        "\0".join((str(seed), namespace, row.source, row.source_id, row.prompt))
    )


def _allocate_by_capacity(capacities: Mapping[str, int], total: int) -> dict[str, int]:
    if total < 0 or total > sum(capacities.values()):
        raise ValueError("requested selection exceeds stratum capacity")
    if total == 0:
        return {key: 0 for key in capacities}
    capacity_total = sum(capacities.values())
    ideals = {key: total * value / capacity_total for key, value in capacities.items()}
    result = {key: min(value, math.floor(ideals[key])) for key, value in capacities.items()}
    remaining = total - sum(result.values())
    order = sorted(
        capacities,
        key=lambda key: (ideals[key] - math.floor(ideals[key]), capacities[key], key),
        reverse=True,
    )
    while remaining:
        progressed = False
        for key in order:
            if result[key] < capacities[key]:
                result[key] += 1
                remaining -= 1
                progressed = True
                if not remaining:
                    break
        if not progressed:  # pragma: no cover - guarded by total capacity check
            raise RuntimeError("failed to allocate prompt strata")
    return result


def _stratified_selection(
    rows: Sequence[SourcePrompt], count: int, *, namespace: str, seed: int
) -> list[SourcePrompt]:
    groups: dict[str, list[SourcePrompt]] = defaultdict(list)
    for row in rows:
        groups[row.stratum].append(row)
    quotas = _allocate_by_capacity({key: len(value) for key, value in groups.items()}, count)
    selected: dict[str, deque[SourcePrompt]] = {}
    for key, values in groups.items():
        ordered = sorted(values, key=lambda row: _stable_rank(row, namespace, seed))
        selected[key] = deque(ordered[: quotas[key]])

    interleaved: list[SourcePrompt] = []
    keys = sorted(selected)
    while len(interleaved) < count:
        for key in keys:
            if selected[key]:
                interleaved.append(selected[key].popleft())
    return interleaved


def build_frozen_eval_pool(
    rows: Iterable[SourcePrompt],
    *,
    selection_seed: int = 20260904,
    sample_seed_base: int = 7_300_000,
) -> list[FrozenEvalSample]:
    """Build the preregistered 400 GenEval + 100 text-rendering pool."""

    materialized = list(rows)
    identities = [(row.source, row.source_id) for row in materialized]
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate source/source_id in candidate prompts")
    by_source: dict[str, list[SourcePrompt]] = defaultdict(list)
    for row in materialized:
        by_source[row.source].append(row)
    unknown = sorted(set(by_source) - set(SOURCE_SPLIT_COUNTS))
    if unknown:
        raise ValueError(f"unknown prompt sources: {unknown}")

    assigned: list[tuple[SourcePrompt, str]] = []
    for source, split_counts in SOURCE_SPLIT_COUNTS.items():
        remaining = list(by_source.get(source, []))
        for split in EVAL_SPLITS:
            selected = _stratified_selection(
                remaining,
                split_counts[split],
                namespace=f"select:{source}:{split}",
                seed=selection_seed,
            )
            assigned.extend((row, split) for row in selected)
            selected_ids = {(row.source, row.source_id) for row in selected}
            remaining = [
                row
                for row in remaining
                if (row.source, row.source_id) not in selected_ids
            ]

    assigned.sort(
        key=lambda item: (
            EVAL_SPLITS.index(item[1]),
            _stable_rank(item[0], f"order:{item[1]}", selection_seed),
        )
    )
    samples: list[FrozenEvalSample] = []
    for index, (row, split) in enumerate(assigned):
        identity = canonical_json(
            {
                "source": row.source,
                "revision": row.revision,
                "source_id": row.source_id,
                "prompt": row.prompt,
            }
        )
        samples.append(
            FrozenEvalSample(
                sample_id=sha256_text(identity)[:24],
                source=row.source,
                source_revision=row.revision,
                source_id=row.source_id,
                prompt=row.prompt,
                stratum=row.stratum,
                split=split,
                seed=sample_seed_base + index,
                metadata_json=row.metadata_json,
            )
        )
    validate_frozen_eval_pool(samples)
    return samples


def validate_frozen_eval_pool(samples: Sequence[FrozenEvalSample]) -> None:
    if len(samples) != sum(EVAL_SPLIT_COUNTS.values()):
        raise ValueError(f"expected 500 samples, got {len(samples)}")
    if len({sample.sample_id for sample in samples}) != len(samples):
        raise ValueError("sample IDs overlap")
    if len({sample.seed for sample in samples}) != len(samples):
        raise ValueError("sample seeds overlap")
    prompt_splits: dict[str, set[str]] = defaultdict(set)
    actual_counts = defaultdict(int)
    source_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for sample in samples:
        prompt_splits[sample.prompt].add(sample.split)
        actual_counts[sample.split] += 1
        source_counts[sample.source][sample.split] += 1
    leaked = sorted(prompt for prompt, splits in prompt_splits.items() if len(splits) > 1)
    if leaked:
        raise ValueError(f"prompts leak across splits: {leaked[:3]}")
    if dict(actual_counts) != EVAL_SPLIT_COUNTS:
        raise ValueError(f"split counts differ: {dict(actual_counts)}")
    normalized_source_counts = {
        source: dict(counts) for source, counts in source_counts.items()
    }
    if normalized_source_counts != SOURCE_SPLIT_COUNTS:
        raise ValueError(f"source split counts differ: {normalized_source_counts}")


def episode_prompt_batch(
    samples: Sequence[FrozenEvalSample],
    generation: int,
    *,
    batch_size: int = 32,
    schedule_seed: int = 20260904,
) -> list[FrozenEvalSample]:
    """Return one deterministic batch, exhausting an epoch before reshuffling."""

    if generation < 0 or batch_size <= 0 or schedule_seed < 0:
        raise ValueError("generation/seed must be non-negative and batch size positive")
    if not samples or any(sample.split != "es_train" for sample in samples):
        raise ValueError("episode scheduler accepts only a non-empty es_train split")
    if len(samples) % batch_size:
        raise ValueError("es_train size must be divisible by episode batch size")
    batches_per_epoch = len(samples) // batch_size
    epoch, batch_index = divmod(generation, batches_per_epoch)
    ordered = sorted(
        samples,
        key=lambda sample: sha256_text(
            f"{schedule_seed}\0episode-epoch\0{epoch}\0{sample.sample_id}"
        ),
    )
    start = batch_index * batch_size
    return ordered[start : start + batch_size]


@dataclass(frozen=True)
class RewardInput:
    image_bytes: bytes
    sample: FrozenEvalSample

    @property
    def image_sha256(self) -> str:
        return hashlib.sha256(self.image_bytes).hexdigest()


def frozen_sample_from_generated_record(row: Mapping) -> FrozenEvalSample:
    metadata = row.get("prompt_metadata", row)
    if not isinstance(metadata, Mapping):
        raise ValueError("sample prompt_metadata must be an object")
    required = (
        "sample_id",
        "source",
        "source_revision",
        "source_id",
        "stratum",
        "split",
    )
    missing = [field for field in required if field not in metadata]
    if missing:
        raise ValueError(f"generated sample is missing frozen metadata: {missing}")
    return FrozenEvalSample(
        sample_id=str(metadata["sample_id"]),
        source=str(metadata["source"]),
        source_revision=str(metadata["source_revision"]),
        source_id=str(metadata["source_id"]),
        prompt=str(row["prompt"]),
        stratum=str(metadata["stratum"]),
        split=str(metadata["split"]),
        seed=int(row["seed"]),
        metadata_json=canonical_json(metadata.get("metadata", {})),
    )


def load_generated_reward_inputs(
    path: Path, image_root: Path, *, offset: int = 0, limit: int | None = None
) -> list[RewardInput]:
    """Load images only from a controlled root and restore frozen sample identity."""

    path = path.expanduser().resolve()
    image_root = image_root.expanduser().resolve()
    if offset < 0 or (limit is not None and limit <= 0):
        raise ValueError("offset must be non-negative and limit positive")
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    selected = rows[offset:] if limit is None else rows[offset : offset + limit]
    if not selected:
        raise ValueError("sample selection is empty")
    inputs = []
    for row in selected:
        image_name = row.get("image")
        if not isinstance(image_name, str):
            raise ValueError("generated sample has no image path")
        image_path = (image_root / image_name).resolve()
        if not image_path.is_file() or image_root not in image_path.parents:
            raise ValueError(f"image is outside the controlled root or missing: {image_name}")
        inputs.append(
            RewardInput(
                image_path.read_bytes(),
                frozen_sample_from_generated_record(row),
            )
        )
    return inputs


def load_frozen_eval_samples(path: Path) -> list[FrozenEvalSample]:
    path = path.expanduser().resolve()
    samples = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line:
            continue
        row = json.loads(line)
        try:
            samples.append(
                FrozenEvalSample(
                    sample_id=str(row["sample_id"]),
                    source=str(row["source"]),
                    source_revision=str(row["source_revision"]),
                    source_id=str(row["source_id"]),
                    prompt=str(row["prompt"]),
                    stratum=str(row["stratum"]),
                    split=str(row["split"]),
                    seed=int(row["seed"]),
                    metadata_json=canonical_json(row.get("metadata", {})),
                )
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{path}:{line_number}: invalid frozen sample") from error
    if not samples:
        raise ValueError("frozen evaluation sample file is empty")
    if len({sample.sample_id for sample in samples}) != len(samples):
        raise ValueError("frozen evaluation sample IDs are duplicated")
    return samples


class RewardScorer(Protocol):
    name: str

    def score_batch(self, inputs: Sequence[RewardInput]) -> list[float]: ...


@dataclass(frozen=True)
class RewardCalibration:
    means: Mapping[str, float]
    stds: Mapping[str, float]
    weights: Mapping[str, float]
    sample_baselines: Mapping[str, Mapping[str, float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        expected = set(PRIMARY_REWARD_AXES)
        if set(self.means) != expected or set(self.stds) != expected or set(self.weights) != expected:
            raise ValueError(f"calibration axes must be exactly {sorted(expected)}")
        values = [*self.means.values(), *self.stds.values(), *self.weights.values()]
        if not all(math.isfinite(value) for value in values):
            raise ValueError("calibration values must be finite")
        if any(value <= 0 for value in self.stds.values()):
            raise ValueError("calibration standard deviations must be positive")
        if sum(abs(value) for value in self.weights.values()) == 0:
            raise ValueError("at least one reward weight must be non-zero")
        for sample_id, scores in self.sample_baselines.items():
            if not sample_id or set(scores) != expected:
                raise ValueError("sample baseline must have an ID and both primary axes")
            if not all(math.isfinite(value) for value in scores.values()):
                raise ValueError("sample baseline scores must be finite")

    def as_record(self) -> dict:
        return {
            "means": dict(self.means),
            "stds": dict(self.stds),
            "weights": dict(self.weights),
            "sample_baselines": {
                sample_id: dict(scores)
                for sample_id, scores in self.sample_baselines.items()
            },
        }

    @classmethod
    def from_record(cls, record: Mapping) -> RewardCalibration:
        required = ("means", "stds", "weights", "sample_baselines")
        if any(not isinstance(record.get(key), Mapping) for key in required):
            raise ValueError("invalid serialized reward calibration")
        return cls(
            means={str(key): float(value) for key, value in record["means"].items()},
            stds={str(key): float(value) for key, value in record["stds"].items()},
            weights={str(key): float(value) for key, value in record["weights"].items()},
            sample_baselines={
                str(sample_id): {
                    str(axis): float(value) for axis, value in scores.items()
                }
                for sample_id, scores in record["sample_baselines"].items()
            },
        )


@dataclass(frozen=True)
class AggregateReward:
    value: float | None
    z_scores: Mapping[str, float]
    valid: bool
    failure_reason: str | None


@dataclass(frozen=True)
class EpisodeReward:
    value: float | None
    valid: bool
    sample_rewards: tuple[Mapping, ...]
    failure_reason: str | None


def aggregate_primary_reward(
    scores: Mapping[str, float | None],
    calibration: RewardCalibration,
    *,
    sample_id: str | None = None,
) -> AggregateReward:
    """Aggregate only frozen HPS/CLIP axes; GenEval remains a diagnostic."""

    if set(scores) != set(PRIMARY_REWARD_AXES):
        return AggregateReward(None, {}, False, "missing or unknown primary reward axis")
    if any(value is None or not math.isfinite(value) for value in scores.values()):
        return AggregateReward(None, {}, False, "non-finite primary reward score")
    if calibration.sample_baselines:
        baseline = calibration.sample_baselines.get(sample_id or "")
        if baseline is None:
            return AggregateReward(None, {}, False, "missing paired BF16 sample baseline")
    else:
        baseline = calibration.means
    z_scores = {
        axis: (float(scores[axis]) - baseline[axis]) / calibration.stds[axis]
        for axis in PRIMARY_REWARD_AXES
    }
    value = sum(calibration.weights[axis] * z_scores[axis] for axis in PRIMARY_REWARD_AXES)
    return AggregateReward(value, z_scores, True, None)


def aggregate_episode_reward(
    score_records: Sequence[Mapping], calibration: RewardCalibration
) -> EpisodeReward:
    """Aggregate a complete paired episode, failing atomically on any bad sample."""

    if not score_records:
        return EpisodeReward(None, False, (), "empty episode")
    sample_rewards = []
    seen = set()
    for record in score_records:
        sample_id = str(record.get("sample_id", ""))
        scores = record.get("scores")
        if not sample_id or sample_id in seen or not isinstance(scores, Mapping):
            return EpisodeReward(
                None,
                False,
                tuple(sample_rewards),
                "invalid or duplicate episode sample",
            )
        seen.add(sample_id)
        outcome = aggregate_primary_reward(scores, calibration, sample_id=sample_id)
        sample_rewards.append(
            {
                "sample_id": sample_id,
                "value": outcome.value,
                "z_scores": dict(outcome.z_scores),
                "valid": outcome.valid,
                "failure_reason": outcome.failure_reason,
            }
        )
        if not outcome.valid:
            return EpisodeReward(
                None,
                False,
                tuple(sample_rewards),
                f"sample {sample_id}: {outcome.failure_reason}",
            )
    return EpisodeReward(
        statistics.fmean(float(item["value"]) for item in sample_rewards),
        True,
        tuple(sample_rewards),
        None,
    )


def score_reward_batch(
    inputs: Sequence[RewardInput], scorers: Sequence[RewardScorer]
) -> list[dict]:
    names = [scorer.name for scorer in scorers]
    if len(names) != len(set(names)):
        raise ValueError("reward scorer names must be unique")
    columns: dict[str, list[float]] = {}
    for scorer in scorers:
        values = scorer.score_batch(inputs)
        if len(values) != len(inputs):
            raise ValueError(f"{scorer.name} returned {len(values)} values for {len(inputs)} inputs")
        columns[scorer.name] = [float(value) for value in values]
    return [
        {
            "sample_id": item.sample.sample_id,
            "image_sha256": item.image_sha256,
            "scores": {name: columns[name][index] for name in names},
        }
        for index, item in enumerate(inputs)
    ]


def audit_reward_determinism(
    inputs: Sequence[RewardInput],
    scorers: Sequence[RewardScorer],
    *,
    absolute_tolerance: float = 0.0,
) -> dict:
    if absolute_tolerance < 0:
        raise ValueError("absolute_tolerance must be non-negative")
    first = score_reward_batch(inputs, scorers)
    second = score_reward_batch(inputs, scorers)
    axes = [scorer.name for scorer in scorers]
    maximums = {
        axis: max(
            (abs(left["scores"][axis] - right["scores"][axis]) for left, right in zip(first, second, strict=True)),
            default=0.0,
        )
        for axis in axes
    }
    return {
        "absolute_tolerance": absolute_tolerance,
        "max_abs_difference": maximums,
        "deterministic": all(value <= absolute_tolerance for value in maximums.values()),
    }


def fit_reward_calibration(
    score_records: Sequence[Mapping],
    *,
    weights: Mapping[str, float] | None = None,
) -> RewardCalibration:
    """Fit frozen BF16 mean/sample-std constants for the primary axes."""

    weights = weights or {axis: 0.5 for axis in PRIMARY_REWARD_AXES}
    columns: dict[str, list[float]] = {axis: [] for axis in PRIMARY_REWARD_AXES}
    for record in score_records:
        scores = record.get("scores")
        if not isinstance(scores, Mapping):
            raise ValueError("score record has no scores mapping")
        for axis in PRIMARY_REWARD_AXES:
            value = scores.get(axis)
            if value is None or not math.isfinite(float(value)):
                raise ValueError(f"non-finite or missing {axis} score in calibration data")
            columns[axis].append(float(value))
    if any(len(values) < 2 for values in columns.values()):
        raise ValueError("reward calibration requires at least two samples")
    means = {axis: statistics.fmean(values) for axis, values in columns.items()}
    stds = {axis: statistics.stdev(values) for axis, values in columns.items()}
    baselines = {}
    for record in score_records:
        sample_id = record.get("sample_id")
        if sample_id is None:
            continue
        if str(sample_id) in baselines:
            raise ValueError(f"duplicate calibration sample_id: {sample_id}")
        baselines[str(sample_id)] = {
            axis: float(record["scores"][axis]) for axis in PRIMARY_REWARD_AXES
        }
    return RewardCalibration(
        means=means,
        stds=stds,
        weights=dict(weights),
        sample_baselines=baselines,
    )


def paired_bootstrap_mean_difference(
    candidate: Sequence[float],
    reference: Sequence[float],
    *,
    resamples: int = 10_000,
    seed: int = 20260904,
    confidence: float = 0.95,
) -> dict:
    """Deterministic paired-bootstrap CI for candidate minus reference."""

    if len(candidate) != len(reference) or len(candidate) < 2:
        raise ValueError("paired samples must have equal length of at least two")
    if resamples <= 0 or not 0.0 < confidence < 1.0:
        raise ValueError("invalid bootstrap configuration")
    differences = [float(left) - float(right) for left, right in zip(candidate, reference, strict=True)]
    if not all(math.isfinite(value) for value in differences):
        raise ValueError("paired samples must be finite")
    rng = random.Random(seed)
    count = len(differences)
    bootstrapped = sorted(
        statistics.fmean(differences[rng.randrange(count)] for _ in range(count))
        for _ in range(resamples)
    )
    tail = (1.0 - confidence) / 2.0
    lower_index = max(0, math.floor(tail * (resamples - 1)))
    upper_index = min(resamples - 1, math.ceil((1.0 - tail) * (resamples - 1)))
    return {
        "mean_difference": statistics.fmean(differences),
        "confidence": confidence,
        "lower": bootstrapped[lower_index],
        "upper": bootstrapped[upper_index],
        "resamples": resamples,
        "seed": seed,
    }
