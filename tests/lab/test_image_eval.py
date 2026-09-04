"""Contracts for the frozen Mage image-evaluation protocol."""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import pytest

from shadow_lab.image_eval import (
    EVAL_SPLIT_COUNTS,
    SOURCE_SPLIT_COUNTS,
    FrozenEvalSample,
    RewardCalibration,
    RewardInput,
    SourcePrompt,
    aggregate_episode_reward,
    aggregate_primary_reward,
    audit_reward_determinism,
    build_frozen_eval_pool,
    canonical_json,
    episode_prompt_batch,
    fit_reward_calibration,
    load_generated_reward_inputs,
    paired_bootstrap_mean_difference,
    score_reward_batch,
)


def _candidate(source: str, index: int, stratum: str) -> SourcePrompt:
    return SourcePrompt(
        source=source,
        revision="fixed@abc",
        source_id=str(index),
        prompt=f"{source} prompt {index}",
        stratum=stratum,
        metadata_json=canonical_json({"index": index}),
    )


def _pool_candidates() -> list[SourcePrompt]:
    geneval = [_candidate("geneval", index, f"tag-{index % 6}") for index in range(553)]
    text = [
        _candidate("text_rendering", index, f"facet-{index % 4}") for index in range(140)
    ]
    return geneval + text


def test_frozen_pool_is_deterministic_disjoint_and_exactly_allocated():
    candidates = _pool_candidates()
    first = build_frozen_eval_pool(candidates)
    second = build_frozen_eval_pool(reversed(candidates))

    assert first == second
    assert len(first) == 500
    assert len({sample.sample_id for sample in first}) == 500
    assert len({sample.prompt for sample in first}) == 500
    assert len({sample.seed for sample in first}) == 500
    for split, expected in EVAL_SPLIT_COUNTS.items():
        assert sum(sample.split == split for sample in first) == expected
    for source, split_counts in SOURCE_SPLIT_COUNTS.items():
        for split, expected in split_counts.items():
            assert sum(
                sample.source == source and sample.split == split for sample in first
            ) == expected
    for split in EVAL_SPLIT_COUNTS:
        assert {
            sample.stratum
            for sample in first
            if sample.source == "geneval" and sample.split == split
        } == {f"tag-{index}" for index in range(6)}


def test_pool_rejects_prompt_leakage_across_splits():
    from shadow_lab.image_eval import validate_frozen_eval_pool

    pool = build_frozen_eval_pool(_pool_candidates())
    train = next(sample for sample in pool if sample.split == "es_train")
    test_index = next(
        index for index, sample in enumerate(pool) if sample.split == "human_test"
    )
    pool[test_index] = replace(pool[test_index], prompt=train.prompt)

    with pytest.raises(ValueError, match="prompts leak across splits"):
        validate_frozen_eval_pool(pool)


def test_episode_batches_exhaust_train_pool_before_deterministic_reshuffle():
    train = [sample for sample in build_frozen_eval_pool(_pool_candidates()) if sample.split == "es_train"]

    first_epoch = [episode_prompt_batch(train, generation) for generation in range(10)]

    assert all(len(batch) == 32 for batch in first_epoch)
    assert len({sample.sample_id for batch in first_epoch for sample in batch}) == 320
    assert episode_prompt_batch(train, 0) == episode_prompt_batch(train, 0)
    assert episode_prompt_batch(train, 10) != episode_prompt_batch(train, 0)


def test_primary_reward_uses_frozen_baseline_zscores_and_never_geneval():
    calibration = RewardCalibration(
        means={"hps_v2_1": 0.25, "clip_h14": 0.30},
        stds={"hps_v2_1": 0.02, "clip_h14": 0.05},
        weights={"hps_v2_1": 0.6, "clip_h14": 0.4},
    )
    outcome = aggregate_primary_reward(
        {"hps_v2_1": 0.27, "clip_h14": 0.25}, calibration
    )

    assert outcome.valid
    assert outcome.z_scores == pytest.approx({"hps_v2_1": 1.0, "clip_h14": -1.0})
    assert outcome.value == pytest.approx(0.2)
    assert not aggregate_primary_reward(
        {"hps_v2_1": 0.27, "clip_h14": 0.25, "geneval": 0.0}, calibration
    ).valid
    assert not aggregate_primary_reward(
        {"hps_v2_1": math.nan, "clip_h14": 0.25}, calibration
    ).valid


def test_reward_scorer_receives_only_image_bytes_and_frozen_sample():
    sample = FrozenEvalSample(
        sample_id="sample",
        source="geneval",
        source_revision="fixed",
        source_id="1",
        prompt="a photo of a cat",
        stratum="single_object",
        split="es_train",
        seed=4,
        metadata_json="{}",
    )
    item = RewardInput(b"image payload", sample)

    class FakeScorer:
        name = "metric"

        def score_batch(self, inputs):
            assert inputs == [item]
            return [0.75]

    records = score_reward_batch([item], [FakeScorer()])
    assert records[0]["sample_id"] == "sample"
    assert records[0]["scores"] == {"metric": 0.75}
    assert "candidate" not in records[0]


def test_determinism_audit_and_calibration_are_explicit():
    sample = FrozenEvalSample(
        sample_id="sample",
        source="geneval",
        source_revision="fixed",
        source_id="1",
        prompt="a photo of a cat",
        stratum="single_object",
        split="es_train",
        seed=4,
        metadata_json="{}",
    )
    inputs = [RewardInput(b"image", sample)]

    class StableScorer:
        name = "stable"

        def score_batch(self, items):
            return [0.5 for _ in items]

    assert audit_reward_determinism(inputs, [StableScorer()])["deterministic"]

    records = [
        {"sample_id": "a", "scores": {"hps_v2_1": 0.20, "clip_h14": 0.30}},
        {"sample_id": "b", "scores": {"hps_v2_1": 0.24, "clip_h14": 0.40}},
    ]
    calibration = fit_reward_calibration(records)
    assert RewardCalibration.from_record(calibration.as_record()) == calibration
    assert calibration.means == pytest.approx({"hps_v2_1": 0.22, "clip_h14": 0.35})
    assert calibration.stds["hps_v2_1"] > 0
    paired = aggregate_primary_reward(
        {"hps_v2_1": 0.22, "clip_h14": 0.35}, calibration, sample_id="a"
    )
    assert paired.z_scores["hps_v2_1"] > 0
    assert not aggregate_primary_reward(
        {"hps_v2_1": 0.22, "clip_h14": 0.35}, calibration, sample_id="unknown"
    ).valid


def test_paired_bootstrap_is_replayable_and_preserves_pairing():
    candidate = [0.2, 0.4, 0.6, 0.8]
    reference = [0.1, 0.3, 0.5, 0.7]
    first = paired_bootstrap_mean_difference(candidate, reference, resamples=500)
    second = paired_bootstrap_mean_difference(candidate, reference, resamples=500)

    assert first == second
    assert first["mean_difference"] == pytest.approx(0.1)
    assert first["lower"] == pytest.approx(0.1)
    assert first["upper"] == pytest.approx(0.1)


def test_generated_inputs_restore_identity_and_confine_image_path(tmp_path: Path):
    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"png")
    row = {
        "prompt": "a photo of a cat",
        "seed": 4,
        "image": image_path.name,
        "prompt_metadata": {
            "sample_id": "sample",
            "source": "geneval",
            "source_revision": "fixed",
            "source_id": "1",
            "stratum": "single_object",
            "split": "es_train",
            "metadata": {"tag": "single_object"},
        },
    }
    samples_path = tmp_path / "samples.jsonl"
    samples_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    loaded = load_generated_reward_inputs(samples_path, tmp_path)

    assert loaded[0].image_bytes == b"png"
    assert loaded[0].sample.sample_id == "sample"
    assert loaded[0].sample.metadata == {"tag": "single_object"}

    row["image"] = "../escape.png"
    samples_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="controlled root"):
        load_generated_reward_inputs(samples_path, tmp_path)


def test_episode_reward_is_paired_and_fails_atomically() -> None:
    calibration = RewardCalibration(
        means={"hps_v2_1": 0.2, "clip_h14": 0.3},
        stds={"hps_v2_1": 0.1, "clip_h14": 0.2},
        weights={"hps_v2_1": 0.5, "clip_h14": 0.5},
        sample_baselines={
            "a": {"hps_v2_1": 0.2, "clip_h14": 0.3},
            "b": {"hps_v2_1": 0.3, "clip_h14": 0.4},
        },
    )
    records = [
        {"sample_id": "a", "scores": {"hps_v2_1": 0.3, "clip_h14": 0.5}},
        {"sample_id": "b", "scores": {"hps_v2_1": 0.3, "clip_h14": 0.4}},
    ]

    outcome = aggregate_episode_reward(records, calibration)

    assert outcome.valid
    assert outcome.value == pytest.approx(0.5)
    assert not aggregate_episode_reward(records[:1] + [records[0]], calibration).valid
    assert not aggregate_episode_reward(
        records[:1]
        + [{"sample_id": "missing", "scores": records[1]["scores"]}],
        calibration,
    ).valid
