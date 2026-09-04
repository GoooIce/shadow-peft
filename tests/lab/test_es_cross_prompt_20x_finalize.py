from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT_PATH = (
    Path(__file__).parents[2]
    / "scripts"
    / "finalize_mage_flow_es_cross_prompt_gate_20x.py"
)
SPEC = importlib.util.spec_from_file_location(
    "finalize_mage_flow_es_cross_prompt_gate_20x", SCRIPT_PATH
)
assert SPEC is not None and SPEC.loader is not None
FINALIZE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FINALIZE)


def _gradient_report() -> dict:
    return {
        "protocol": {
            "observation_episodes": list(range(24)),
            "gradient_batch_count": 24,
            "population_size": 80,
            "episode_size": 8,
            "sigma": 0.003,
            "es_seed": 20260907,
            "schedule_seed": 20260904,
            "shaping": "zscore",
            "alphas": [0.00003, 0.0001],
            "clip_step_rms_max": 0.02,
        },
        "sources": [
            {
                "run_sha256": f"run-{episode}",
                "episode_index": episode,
                "failure_count": 0,
                "reward_std": 0.03,
                "replay_exact": True,
                "prompt_sample_ids": [
                    f"obs-{episode}-{index}" for index in range(8)
                ],
            }
            for episode in range(24)
        ],
        "initial_effective_center": {
            "clip_sha256": "initial",
            "path": "/centers/initial.pt",
        },
        "candidates": [
            {
                "alpha": 0.00003,
                "clip_step_rms": 0.002,
                "center_clip_sha256": "small",
                "center_clip_path": "/centers/small.pt",
            },
            {
                "alpha": 0.0001,
                "clip_step_rms": 0.006,
                "center_clip_sha256": "large",
                "center_clip_path": "/centers/large.pt",
            },
        ],
    }


def _evaluation(episode: int, deltas: dict[str, float]) -> dict:
    rows = [("initial", "initial", 1.0), *[
        (label, label, 1.0 + delta) for label, delta in deltas.items()
    ]]
    return {
        "episode": {
            "index": episode,
            "schedule_seed": 20260904,
            "sample_ids": [f"eval-{episode}-{index}" for index in range(8)],
        },
        "source": {
            "run_sha256": "run-0",
            "additional_source_runs": [
                {"run_sha256": f"run-{episode_index}"}
                for episode_index in range(1, 24)
            ],
        },
        "records": [
            {
                "label": label,
                "clip_sha256": clip_hash,
                "reward": reward,
                **(
                    {"replay_audit": {"exact": True, "matches": {"reward": True}}}
                    if label == "initial"
                    else {}
                ),
            }
            for label, clip_hash, reward in rows
        ],
        "comparisons": [
            {
                "label": label,
                "reward_delta_vs_baseline": reward - 1.0,
                "non_degrading": reward >= 1.0,
            }
            for label, _, reward in rows
        ],
    }


def test_budget_is_exactly_twenty_times_exp006() -> None:
    assert FINALIZE.generation_budget() == {
        "gradient_candidates": 15360,
        "gradient_replay_audits": 192,
        "selection": 256,
        "confirmation": 192,
        "total": 16000,
    }


def test_gate_uses_eight_selection_and_confirmation_batches() -> None:
    selection = [
        _evaluation(episode, {"small": 0.01, "large": 0.02})
        for episode in range(24, 32)
    ]
    confirmation = [
        _evaluation(episode, {"large": 0.01 if episode < 38 else -0.001})
        for episode in range(32, 40)
    ]

    result = FINALIZE.summarize_gate(
        _gradient_report(),
        selection,
        confirmation,
    )

    assert result["status"] == "passed"
    assert result["passes"] is True
    assert result["selected_candidate"]["alpha"] == 0.0001
    assert result["selection"]["candidates"][1]["non_degrading_batches"] == 8
    assert result["confirmation"]["non_degrading_batches"] == 6
    assert result["frozen_config"]["gradient_batches"] == 24


def test_selection_phase_freezes_the_candidate_before_confirmation() -> None:
    selection = [
        _evaluation(episode, {"small": 0.01, "large": -0.01})
        for episode in range(24, 32)
    ]

    result = FINALIZE.summarize_gate(_gradient_report(), selection)

    assert result["status"] == "awaiting-confirmation"
    assert result["selected_candidate"]["alpha"] == 0.00003
    assert result["selected_candidate_role"] == "eligible"
    assert result["passes"] is False


def test_failed_selection_still_names_a_diagnostic_candidate_but_cannot_unlock() -> None:
    selection = [
        _evaluation(
            episode,
            {
                "small": 0.01 if episode < 29 else -0.03,
                "large": -0.01,
            },
        )
        for episode in range(24, 32)
    ]
    confirmation = [
        _evaluation(episode, {"small": 0.02}) for episode in range(32, 40)
    ]

    result = FINALIZE.summarize_gate(
        _gradient_report(),
        selection,
        confirmation,
    )

    assert result["selection_passes"] is False
    assert result["selected_candidate"]["alpha"] == 0.00003
    assert result["confirmation"]["passes"] is True
    assert result["status"] == "stopped-selection"
    assert result["passes"] is False
    assert result["frozen_config"] is None


def test_gate_rejects_prompt_reuse_across_center_batches() -> None:
    selection = [
        _evaluation(episode, {"small": 0.01, "large": -0.01})
        for episode in range(24, 32)
    ]
    selection[1]["episode"]["sample_ids"] = selection[0]["episode"]["sample_ids"]
    confirmation = [
        _evaluation(episode, {"small": 0.01}) for episode in range(32, 40)
    ]

    with pytest.raises(ValueError, match="overlap"):
        FINALIZE.summarize_gate(_gradient_report(), selection, confirmation)
