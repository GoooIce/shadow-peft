from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT_PATH = Path(__file__).parents[2] / "scripts" / "summarize_mage_flow_es_sigma.py"
SPEC = importlib.util.spec_from_file_location("summarize_mage_flow_es_sigma", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
SUMMARY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUMMARY)


def _row(pair: int, sign: int, reward: float) -> dict:
    return {
        "spec": {"pair_index": pair, "sign": sign},
        "reward": reward,
        "failure_reason": None,
        "clip_statistics": {"perturbation_rms": 0.01},
        "weighted_reconstruction_error": 101.0,
    }


def test_arm_metrics_apply_preregistered_thresholds() -> None:
    rows = [
        _row(0, 1, 0.1),
        _row(0, -1, 0.0),
        _row(1, 1, 0.05),
        _row(1, -1, -0.05),
    ]
    rows[0]["replay_audit"] = {"exact": True, "matches": {"reward": True}}

    metrics = SUMMARY.arm_metrics(rows, quasar_error=100.0)

    assert metrics["failure_count"] == 0
    assert metrics["pair_reward_differences"] == [0.1, 0.1]
    assert metrics["reconstruction_error_ratio"] == 1.01
    assert metrics["passes_preflight"]


def test_arm_metrics_reject_weak_reward_signal() -> None:
    rows = [_row(0, 1, 0.001), _row(0, -1, 0.0)]
    rows[0]["replay_audit"] = {"exact": True, "matches": {"reward": True}}

    assert not SUMMARY.arm_metrics(rows, quasar_error=100.0)["passes_preflight"]
