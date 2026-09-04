from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).parents[2] / "scripts" / "finalize_mage_flow_es_calibration.py"
SPEC = importlib.util.spec_from_file_location("finalize_mage_flow_es_calibration", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
FINALIZE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FINALIZE)


def _alpha_report(*records: tuple[float, float, str]) -> dict:
    return {
        "records": [
            {
                "alpha": alpha,
                "clip_step_rms": step,
                "center_clip_sha256": clip_hash,
            }
            for alpha, step, clip_hash in records
        ]
    }


def _evaluation(*records: tuple[str, str, bool]) -> dict:
    return {
        "records": [
            {"label": label, "clip_sha256": clip_hash}
            for label, clip_hash, _ in records
        ],
        "comparisons": [
            {"label": label, "non_degrading": non_degrading}
            for label, _, non_degrading in records
        ],
    }


def test_alpha_selection_uses_fallback_and_largest_non_degrading_step() -> None:
    selected, stage = FINALIZE.select_alpha(
        _alpha_report((0.0003, 0.017, "primary"), (0.001, 0.08, "too-large")),
        _evaluation(("primary", "primary", False)),
        _alpha_report((0.00003, 0.002, "small"), (0.0001, 0.005, "large")),
        _evaluation(("small", "small", True), ("large", "large", True)),
    )

    assert selected == 0.0001
    assert stage == "fallback"


def test_step_eligible_alpha_requires_a_center_evaluation() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        FINALIZE.evaluated_alphas(
            _alpha_report((0.0001, 0.005, "missing")),
            _evaluation(("other", "other", True)),
        )


def test_confirmation_metrics_apply_pre_registered_gates() -> None:
    rows = []
    for index, reward in enumerate((1.0, 0.8, 1.1, 0.7)):
        rows.append(
            {
                "reward": reward,
                "spec": {"pair_index": index // 2, "sign": 1 if index % 2 == 0 else -1},
                "clip_statistics": {"perturbation_rms": 0.001},
                "weighted_reconstruction_error": 101.0,
                **(
                    {"replay_audit": {"exact": True, "matches": {"reward": True}}}
                    if index == 0
                    else {}
                ),
            }
        )

    metrics = FINALIZE.confirmation_metrics(rows, quasar_error=100.0)

    assert metrics["reward_std"] > 0.02
    assert metrics["pair_mean_abs_difference"] == pytest.approx(0.3)
    assert metrics["reconstruction_error_ratio"] == pytest.approx(1.01)
    assert metrics["passes"] is True
