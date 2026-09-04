from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT_PATH = (
    Path(__file__).parents[2]
    / "scripts"
    / "finalize_mage_flow_es_cross_prompt_gate.py"
)
SPEC = importlib.util.spec_from_file_location(
    "finalize_mage_flow_es_cross_prompt_gate", SCRIPT_PATH
)
assert SPEC is not None and SPEC.loader is not None
FINALIZE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FINALIZE)


def _gradient_report() -> dict:
    return {
        "protocol": {
            "observation_episodes": [4, 5, 6],
            "population_size": 30,
            "episode_size": 8,
            "sigma": 0.003,
            "es_seed": 20260906,
            "schedule_seed": 20260904,
            "alphas": [0.00003, 0.0001],
            "clip_step_rms_max": 0.02,
        },
        "sources": [
            {
                "run_sha256": f"run-{episode}",
                "failure_count": 0,
                "reward_std": 0.03,
                "replay_exact": True,
                "prompt_sample_ids": [f"obs-{episode}-{index}" for index in range(8)],
            }
            for episode in (4, 5, 6)
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


def _evaluation(episode: int, small: bool, large: bool | None) -> dict:
    hashes = [
        ("initial", "initial", 1.0, True),
        ("small", "small", 1.1 if small else 0.9, small),
    ]
    if large is not None:
        hashes.append(("large", "large", 1.1 if large else 0.9, large))
    return {
        "episode": {
            "index": episode,
            "schedule_seed": 20260904,
            "sample_ids": [f"gate-{episode}-{index}" for index in range(8)],
        },
        "source": {
            "run_sha256": "run-4",
            "additional_source_runs": [
                {"run_sha256": "run-5"},
                {"run_sha256": "run-6"},
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
            for label, clip_hash, reward, _ in hashes
        ],
        "comparisons": [
            {
                "label": label,
                "reward_delta_vs_baseline": reward - 1.0,
                "non_degrading": non_degrading,
            }
            for label, _, reward, non_degrading in hashes
        ],
    }


def test_gate_selects_larger_step_when_both_average_centers_do_not_degrade() -> None:
    result = FINALIZE.summarize_gate(
        _gradient_report(), _evaluation(7, small=True, large=True)
    )

    assert result["status"] == "awaiting-confirmation"
    assert result["selected_candidate"]["alpha"] == 0.0001
    assert result["gate_passes"] is True
    assert result["passes"] is False


def test_final_confirmation_requires_the_selected_center_to_hold_again() -> None:
    gradient = _gradient_report()
    gate = _evaluation(7, small=True, large=False)

    passed = FINALIZE.summarize_gate(
        gradient, gate, _evaluation(8, small=True, large=None)
    )
    failed = FINALIZE.summarize_gate(
        gradient, gate, _evaluation(8, small=False, large=None)
    )

    assert passed["selected_candidate"]["alpha"] == 0.00003
    assert passed["passes"] is True
    assert passed["frozen_config"]["alpha"] == 0.00003
    assert failed["passes"] is False
    assert failed["status"] == "stopped-confirmation"


def test_gate_rejects_an_evaluation_from_different_gradient_sources() -> None:
    evaluation = _evaluation(7, small=True, large=True)
    evaluation["source"]["additional_source_runs"][1]["run_sha256"] = "other"

    try:
        FINALIZE.summarize_gate(_gradient_report(), evaluation)
    except ValueError as error:
        assert "source runs" in str(error)
    else:  # pragma: no cover - the assertion gives a clearer failure than pytest alone
        raise AssertionError("mismatched source runs were accepted")


def test_gate_recomputes_rewards_and_rejects_stale_comparison_fields() -> None:
    evaluation = _evaluation(7, small=True, large=False)
    evaluation["comparisons"][1]["non_degrading"] = False

    try:
        FINALIZE.summarize_gate(_gradient_report(), evaluation)
    except ValueError as error:
        assert "comparison" in str(error)
    else:  # pragma: no cover - the assertion gives a clearer failure than pytest alone
        raise AssertionError("stale non-degrading flag was accepted")

    evaluation = _evaluation(7, small=True, large=False)
    evaluation["comparisons"][1]["reward_delta_vs_baseline"] = -123.0
    try:
        FINALIZE.summarize_gate(_gradient_report(), evaluation)
    except ValueError as error:
        assert "comparison" in str(error)
    else:  # pragma: no cover - the assertion gives a clearer failure than pytest alone
        raise AssertionError("stale reward delta was accepted")


def test_frozen_generation_budget_includes_all_replay_audits() -> None:
    assert FINALIZE.generation_budget(_gradient_report()["protocol"]) == {
        "gradient_candidates": 720,
        "gradient_replay_audits": 24,
        "episode_7_centers": 24,
        "episode_7_replay_audit": 8,
        "episode_8_centers": 16,
        "episode_8_replay_audit": 8,
        "total": 800,
    }
