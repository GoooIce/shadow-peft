"""Tests for the detector-independent GenEval decision layer."""

from __future__ import annotations

import pytest

from shadow_lab.geneval import (
    Detection,
    evaluate_metadata,
    expand_geneval_samples,
    filter_detections,
    paired_clustered_bootstrap,
    relative_positions,
    summarize_results,
)
from shadow_lab.image_eval import FrozenEvalSample, canonical_json


def _detection(
    class_name: str,
    score: float,
    bbox: tuple[float, float, float, float],
    color: str | None = None,
) -> Detection:
    return Detection(class_name, score, bbox, color)


def test_counting_uses_stricter_threshold_and_exclusion_rule() -> None:
    detections = [
        _detection("cat", 0.95, (0, 0, 10, 10)),
        _detection("cat", 0.89, (20, 0, 30, 10)),
    ]
    objects = filter_detections(detections, tag="counting")
    decision = evaluate_metadata(
        {
            "include": [{"class": "cat", "count": 1}],
            "exclude": [{"class": "cat", "count": 2}],
        },
        objects,
    )

    assert decision.correct
    assert len(objects["cat"]) == 1


def test_position_and_color_requirements_match_official_direction() -> None:
    red_cat = _detection("cat", 0.99, (0, 0, 10, 10), "red")
    dog = _detection("dog", 0.99, (100, 0, 110, 10))
    assert relative_positions(red_cat, dog) == {"left of"}

    decision = evaluate_metadata(
        {
            "include": [
                {"class": "dog", "count": 1},
                {
                    "class": "cat",
                    "count": 1,
                    "color": "red",
                    "position": ["left of", 0],
                },
            ]
        },
        {"cat": [red_cat], "dog": [dog]},
    )

    assert decision.correct


def test_summary_is_equal_task_mean_not_image_weighted_mean() -> None:
    records = [
        {"tag": "single_object", "correct": True},
        {"tag": "single_object", "correct": True},
        {"tag": "two_object", "correct": False},
        {"tag": "counting", "correct": False},
        {"tag": "colors", "correct": False},
        {"tag": "position", "correct": False},
        {"tag": "color_attr", "correct": False},
    ]

    summary = summarize_results(records)

    assert summary["overall_equal_task_mean"] == pytest.approx(1 / 6)
    assert summary["overall_equal_task_mean"] != pytest.approx(2 / 7)


def test_four_seed_expansion_is_deterministic_and_keeps_prompt_cluster() -> None:
    sample = FrozenEvalSample(
        sample_id="parent",
        source="geneval",
        source_revision="fixed",
        source_id="1",
        prompt="a cat",
        stratum="single_object",
        split="auto_test",
        seed=7_300_000,
        metadata_json=canonical_json({"tag": "single_object"}),
    )

    expanded = expand_geneval_samples([sample])

    assert expanded == expand_geneval_samples([sample])
    assert len(expanded) == 4
    assert len({item.seed for item in expanded}) == 4
    assert {
        item.metadata["_geneval_protocol"]["parent_sample_id"] for item in expanded
    } == {"parent"}


def test_clustered_bootstrap_carries_replicas_and_equal_weights_tasks() -> None:
    candidate = []
    reference = []
    for task_index, task in enumerate(
        ("single_object", "two_object", "counting", "colors", "position", "color_attr")
    ):
        for replica in range(2):
            common = {
                "sample_id": f"{task}-{replica}",
                "cluster_id": f"cluster-{task}",
                "seed": task_index * 10 + replica,
                "tag": task,
            }
            candidate.append({**common, "correct": task == "single_object"})
            reference.append({**common, "correct": False})

    result = paired_clustered_bootstrap(candidate, reference, resamples=100)

    assert result["mean_difference"] == pytest.approx(1 / 6)
    assert result["lower"] == pytest.approx(1 / 6)
    assert result["upper"] == pytest.approx(1 / 6)
    assert result["cluster_counts"] == {
        "single_object": 1,
        "two_object": 1,
        "counting": 1,
        "colors": 1,
        "position": 1,
        "color_attr": 1,
    }
