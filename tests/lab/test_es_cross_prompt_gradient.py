from __future__ import annotations

import importlib.util
import json
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from shadow_lab.es import (
    CandidateObservation,
    ESConfig,
    candidate_specs,
    initialize_es,
    latent_to_clips,
)

SCRIPT_PATH = (
    Path(__file__).parents[2] / "scripts" / "aggregate_mage_flow_es_gradients.py"
)
SPEC = importlib.util.spec_from_file_location(
    "aggregate_mage_flow_es_gradients", SCRIPT_PATH
)
assert SPEC is not None and SPEC.loader is not None
AGGREGATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AGGREGATE)


def test_gradient_consistency_reports_pairwise_and_unanimous_signs() -> None:
    gradients = [
        torch.tensor([1.0, 1.0, -1.0, 0.0]),
        torch.tensor([1.0, -1.0, -1.0, 0.0]),
        torch.tensor([1.0, 1.0, 1.0, 0.0]),
    ]

    report, mean = AGGREGATE.gradient_consistency(
        ["episode-4", "episode-5", "episode-6"], gradients
    )

    assert torch.equal(mean, torch.tensor([1.0, 1 / 3, -1 / 3, 0.0]))
    pairs = {
        (row["left"], row["right"]): row for row in report["pairwise"]
    }
    assert pairs[("episode-4", "episode-5")]["cosine"] == pytest.approx(1 / 3)
    assert pairs[("episode-4", "episode-5")][
        "sign_agreement_fraction"
    ] == pytest.approx(0.75)
    assert pairs[("episode-5", "episode-6")]["cosine"] == pytest.approx(-1 / 3)
    assert pairs[("episode-5", "episode-6")][
        "sign_agreement_fraction"
    ] == pytest.approx(0.5)
    assert report["unanimous_sign_agreement_fraction"] == pytest.approx(0.5)
    assert report["unanimous_nonzero_sign_fraction"] == pytest.approx(0.25)
    assert report["all_zero_fraction"] == pytest.approx(0.25)


def test_average_gradient_candidates_apply_only_the_registered_steps() -> None:
    initial_clips = torch.tensor([0.4, 0.5, 0.6, 0.7])
    config = ESConfig(dimension=4, population_size=4, sigma=0.003)
    gradient = torch.tensor([1.0, -2.0, 3.0, -4.0])

    candidates = AGGREGATE.average_gradient_candidates(
        initial_clips,
        config,
        gradient,
        [0.00003, 0.0001],
    )

    state = initialize_es(initial_clips, config)
    assert [record["alpha"] for record, _ in candidates] == [0.00003, 0.0001]
    assert torch.equal(
        candidates[1][1],
        latent_to_clips(state.center + 0.0001 * gradient, config),
    )
    assert candidates[1][0]["clip_step_rms"] > candidates[0][0]["clip_step_rms"]
    assert candidates[0][0]["center_clip_sha256"] != candidates[1][0][
        "center_clip_sha256"
    ]
    with pytest.raises(ValueError, match="unique"):
        AGGREGATE.average_gradient_candidates(
            initial_clips, config, gradient, [0.0001, 0.0001]
        )


def _write_gradient_source(
    root: Path,
    *,
    episode: int,
    prompt_ids: list[str],
    reward_scale: float,
) -> Path:
    root.mkdir()
    clips = torch.tensor([0.4, 0.5, 0.6, 0.7])
    config = ESConfig(
        dimension=4,
        population_size=4,
        sigma=0.003,
        learning_rate=0.0001,
        seed=20260906,
    )
    state = initialize_es(clips, config)
    torch.save({"clips": clips}, root / "initial-clips.pt")
    (root / "initialization.json").write_text(
        json.dumps({"clip_sha256": AGGREGATE.sha256_tensor(clips)}),
        encoding="utf-8",
    )
    (root / "run.json").write_text(
        json.dumps(
            {
                "completed_generations": 1,
                "es_config": asdict(config),
                "generation": {
                    "episode_size": len(prompt_ids),
                    "schedule_seed": 20260904,
                    "episode_offset": episode,
                },
            }
        ),
        encoding="utf-8",
    )
    (root / "experiment.json").write_text(
        json.dumps(
            {
                "identity": {
                    "implementation": {"sha256": "same"},
                    "model": {"sha256": "same"},
                    "assets": {"sha256": "same"},
                    "initialization_source": {"sha256": "same"},
                    "search": {"episode_offset": episode},
                    "generation": {"steps": 4},
                    "reward_batch_size": 8,
                }
            }
        ),
        encoding="utf-8",
    )
    (root / "generation-0000-summary.json").write_text(
        json.dumps(
            {
                "generation": 0,
                "prompt_episode_index": episode,
                "prompt_sample_ids": prompt_ids,
                "prompt_seeds": list(range(episode * 10, episode * 10 + len(prompt_ids))),
            }
        ),
        encoding="utf-8",
    )
    rows = []
    for spec in candidate_specs(state):
        observation = CandidateObservation(spec, reward_scale * spec.sign)
        row = {
            "spec": asdict(spec),
            "reward": observation.reward,
            "failure_reason": None,
        }
        if spec.population_index == 0:
            row["replay_audit"] = {"exact": True, "matches": {"reward": True}}
        rows.append(row)
    (root / "generation-0000-candidates.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    return root


def test_load_gradient_sources_enforces_common_noise_and_disjoint_prompts(
    tmp_path: Path,
) -> None:
    sources = [
        _write_gradient_source(
            tmp_path / f"episode-{episode}",
            episode=episode,
            prompt_ids=[f"p{episode}-a", f"p{episode}-b"],
            reward_scale=float(index + 1),
        )
        for index, episode in enumerate((4, 5, 6))
    ]

    loaded = AGGREGATE.load_gradient_sources(
        sources,
        expected_episode_offsets=[4, 5, 6],
        expected_population_size=4,
        expected_episode_size=2,
    )

    assert [source["episode_index"] for source in loaded] == [4, 5, 6]
    assert all(source["replay_exact"] for source in loaded)
    assert all(source["estimate"].reward_std > 0.02 for source in loaded)
    assert len({source["specs"] for source in loaded}) == 1

    report, candidates = AGGREGATE.aggregate_gradient_sources(
        loaded, [0.00003, 0.0001]
    )
    assert report["protocol"]["observation_episodes"] == [4, 5, 6]
    assert report["protocol"]["population_size"] == 4
    assert report["consistency"]["pairwise"][0]["cosine"] == pytest.approx(1.0)
    assert [record["alpha"] for record in report["candidates"]] == [
        0.00003,
        0.0001,
    ]
    assert len(candidates) == 2

    summary_path = sources[2] / "generation-0000-summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["prompt_sample_ids"][0] = "p4-a"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError, match="overlap"):
        AGGREGATE.load_gradient_sources(
            sources,
            expected_episode_offsets=[4, 5, 6],
            expected_population_size=4,
            expected_episode_size=2,
        )


def test_gradient_aggregation_accepts_a_preregistered_larger_batch_count(
    tmp_path: Path,
) -> None:
    sources = [
        _write_gradient_source(
            tmp_path / f"episode-{episode}",
            episode=episode,
            prompt_ids=[f"p{episode}-a", f"p{episode}-b"],
            reward_scale=float(episode + 1),
        )
        for episode in range(4)
    ]

    loaded = AGGREGATE.load_gradient_sources(
        sources,
        expected_episode_offsets=list(range(4)),
        expected_population_size=4,
        expected_episode_size=2,
    )
    report, _ = AGGREGATE.aggregate_gradient_sources(
        loaded,
        [0.00003, 0.0001],
    )

    assert report["protocol"]["observation_episodes"] == [0, 1, 2, 3]
    assert report["protocol"]["gradient_batch_count"] == 4
    assert len(report["consistency"]["pairwise"]) == 6
