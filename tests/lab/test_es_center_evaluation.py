from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).parents[2] / "scripts" / "evaluate_mage_flow_es_centers.py"
SPEC = importlib.util.spec_from_file_location("evaluate_mage_flow_es_centers", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
EVALUATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVALUATE)


def test_parse_candidates_requires_unique_safe_labels(tmp_path: Path) -> None:
    first = tmp_path / "first.pt"
    second = tmp_path / "second.pt"
    first.touch()
    second.touch()

    candidates = EVALUATE.parse_candidates(
        [f"initial={first}", f"alpha-0p001={second}"]
    )

    assert [label for label, _ in candidates] == ["initial", "alpha-0p001"]
    with pytest.raises(ValueError, match="unique"):
        EVALUATE.parse_candidates([f"same={first}", f"same={second}"])
    with pytest.raises(ValueError, match="LABEL=PATH"):
        EVALUATE.parse_candidates([str(first)])


def test_reward_deltas_use_first_center_as_paired_baseline() -> None:
    comparisons = EVALUATE.reward_deltas(
        [
            {"label": "initial", "reward": 1.0},
            {"label": "up", "reward": 1.25},
            {"label": "down", "reward": 0.75},
        ]
    )

    assert comparisons == [
        {"label": "initial", "reward_delta_vs_baseline": 0.0, "non_degrading": True},
        {"label": "up", "reward_delta_vs_baseline": 0.25, "non_degrading": True},
        {"label": "down", "reward_delta_vs_baseline": -0.25, "non_degrading": False},
    ]


def test_source_run_requires_the_next_disjoint_prompt_batch(tmp_path: Path) -> None:
    assets = {
        key: {"sha256": f"sha-{key}"}
        for key in ("prompts", "calibration", "hps_v2_1", "clip_h14")
    }
    (tmp_path / "run.json").write_text(
        json.dumps(
            {
                "completed_generations": 1,
                "generation": {"schedule_seed": 7},
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "experiment.json").write_text(
        json.dumps(
            {
                "identity": {
                    "model": {"metadata_and_size_sha256": "model-sha"},
                    "assets": assets,
                }
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "generation-0000-summary.json").write_text(
        json.dumps({"prompt_sample_ids": ["seen"]}),
        encoding="utf-8",
    )

    source = EVALUATE.validate_source_run(
        tmp_path,
        assets=assets,
        model_identity={"metadata_and_size_sha256": "model-sha"},
        episode_index=1,
        episode_sample_ids=["unseen"],
        schedule_seed=7,
    )

    assert source["source_completed_generations"] == 1
    with pytest.raises(ValueError, match="unseen episode"):
        EVALUATE.validate_source_run(
            tmp_path,
            assets=assets,
            model_identity={"metadata_and_size_sha256": "model-sha"},
            episode_index=0,
            episode_sample_ids=["unseen"],
            schedule_seed=7,
        )
    later = EVALUATE.validate_source_run(
        tmp_path,
        assets=assets,
        model_identity={"metadata_and_size_sha256": "model-sha"},
        episode_index=2,
        episode_sample_ids=["later"],
        schedule_seed=7,
        excluded_sample_ids={"unseen"},
    )
    assert later["source_completed_generations"] == 1
    with pytest.raises(ValueError, match="reuses source prompts"):
        EVALUATE.validate_source_run(
            tmp_path,
            assets=assets,
            model_identity={"metadata_and_size_sha256": "model-sha"},
            episode_index=1,
            episode_sample_ids=["seen"],
            schedule_seed=7,
        )


def _write_source_run(
    path: Path,
    *,
    assets: dict,
    episode_offset: int,
    prompt_ids: list[str],
) -> Path:
    path.mkdir()
    (path / "run.json").write_text(
        json.dumps(
            {
                "completed_generations": 1,
                "generation": {
                    "schedule_seed": 7,
                    "episode_offset": episode_offset,
                },
            }
        ),
        encoding="utf-8",
    )
    (path / "experiment.json").write_text(
        json.dumps(
            {
                "identity": {
                    "model": {"metadata_and_size_sha256": "model-sha"},
                    "assets": assets,
                }
            }
        ),
        encoding="utf-8",
    )
    (path / "generation-0000-summary.json").write_text(
        json.dumps(
            {
                "prompt_episode_index": episode_offset,
                "prompt_sample_ids": prompt_ids,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_center_evaluation_accepts_multiple_disjoint_gradient_sources(
    tmp_path: Path,
) -> None:
    assets = {
        key: {"sha256": f"sha-{key}"}
        for key in ("prompts", "calibration", "hps_v2_1", "clip_h14")
    }
    first = _write_source_run(
        tmp_path / "episode-4",
        assets=assets,
        episode_offset=4,
        prompt_ids=["four"],
    )
    second = _write_source_run(
        tmp_path / "episode-5",
        assets=assets,
        episode_offset=5,
        prompt_ids=["five"],
    )

    identities = EVALUATE.validate_source_runs(
        [first, second],
        assets=assets,
        model_identity={"metadata_and_size_sha256": "model-sha"},
        episode_index=7,
        episode_sample_ids=["seven"],
        schedule_seed=7,
    )

    assert len(identities) == 2
    summary_path = second / "generation-0000-summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["prompt_sample_ids"] = ["four"]
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError, match="overlap"):
        EVALUATE.validate_source_runs(
            [first, second],
            assets=assets,
            model_identity={"metadata_and_size_sha256": "model-sha"},
            episode_index=7,
            episode_sample_ids=["seven"],
            schedule_seed=7,
        )
