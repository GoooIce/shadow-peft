from __future__ import annotations

import json
from dataclasses import asdict

import pytest

from edge_cloud.cost_policy import (
    CostPolicy,
    CostPolicyFit,
    CostPolicyReport,
    CostSample,
    evaluate_cost_policy,
    fit_cost_policy,
)


def sample(
    task: str,
    score: float,
    *,
    edge_correct: bool,
    cloud_correct: bool,
    edge_cost: float = 0.1,
    escalated_cost: float = 0.5,
    cloud_only_cost: float = 1.0,
    split: str = "train",
    sample_id: str | None = None,
) -> CostSample:
    return CostSample(
        task=task,
        score=score,
        edge_correct=edge_correct,
        cloud_correct=cloud_correct,
        edge_cost=edge_cost,
        escalated_cost=escalated_cost,
        cloud_only_cost=cloud_only_cost,
        split=split,
        sample_id=sample_id,
    )


def two_task_samples() -> list[CostSample]:
    """Task A is best at 0.30, task B at 0.70 under lambda=0.1."""

    return [
        sample("task_a", 0.2, edge_correct=False, cloud_correct=True),
        sample("task_a", 0.3, edge_correct=True, cloud_correct=True),
        sample("task_b", 0.5, edge_correct=False, cloud_correct=True),
        sample("task_b", 0.7, edge_correct=True, cloud_correct=True),
    ]


def test_fit_finds_known_per_task_thresholds() -> None:
    fit = fit_cost_policy(two_task_samples(), cost_weight=0.1, min_samples=2)

    assert isinstance(fit, CostPolicyFit)
    assert fit.thresholds_by_task == {
        "task_a": pytest.approx(0.30),
        "task_b": pytest.approx(0.70),
    }
    assert fit.global_threshold in fit.threshold_grid
    assert fit.cost_weight == pytest.approx(0.1)
    assert fit.min_samples == 2
    assert fit.always_cloud_tasks == ()
    assert fit.per_task["task_a"].n == 2
    assert fit.per_task["task_a"].fallback_reason is None
    assert fit.per_task["task_b"].n == 2


def test_cost_weight_zero_maximises_accuracy_and_high_weight_prefers_low_cost() -> None:
    samples = [
        sample("task", 0.2, edge_correct=False, cloud_correct=True, escalated_cost=0.9),
        sample("task", 0.8, edge_correct=True, cloud_correct=False, escalated_cost=0.9),
    ]

    accuracy_only = fit_cost_policy(samples, cost_weight=0.0, min_samples=2)
    cost_averse = fit_cost_policy(samples, cost_weight=10.0, min_samples=2)

    assert accuracy_only.global_threshold == pytest.approx(0.80)
    assert accuracy_only.global_report.accuracy == pytest.approx(1.0)
    assert cost_averse.global_threshold == pytest.approx(0.20)
    assert cost_averse.global_report.cost < accuracy_only.global_report.cost


def test_objective_ties_pick_the_higher_threshold_and_record_the_tie_break() -> None:
    samples = [
        sample("task", 0.4, edge_correct=True, cloud_correct=True, edge_cost=0.5),
        sample("task", 0.4, edge_correct=True, cloud_correct=True, edge_cost=0.5),
    ]

    fit = fit_cost_policy(samples, cost_weight=0.1, min_samples=2)

    assert fit.global_threshold == pytest.approx(1.0)
    assert fit.per_task["task"].threshold == pytest.approx(1.0)
    assert fit.per_task["task"].tie_break == "higher_threshold"


def test_sparse_task_falls_back_to_the_global_threshold() -> None:
    samples = [
        sample("task_a", 0.2, edge_correct=False, cloud_correct=True),
        sample("task_a", 0.3, edge_correct=True, cloud_correct=True),
        sample("rare_task", 0.9, edge_correct=False, cloud_correct=True),
    ]

    fit = fit_cost_policy(samples, cost_weight=0.1, min_samples=2)

    assert "rare_task" not in fit.thresholds_by_task
    assert fit.policy.threshold_for("rare_task") == pytest.approx(fit.global_threshold)
    assert fit.per_task["rare_task"].fallback_reason == "insufficient_samples"
    assert fit.per_task["rare_task"].threshold == pytest.approx(fit.global_threshold)
    assert fit.per_task["rare_task"].n == 1


def test_require_cloud_tasks_are_never_fitted() -> None:
    samples = [
        sample("task_a", 0.2, edge_correct=False, cloud_correct=True),
        sample("task_a", 0.3, edge_correct=True, cloud_correct=True),
        sample("quality_verdict", 0.95, edge_correct=False, cloud_correct=True),
    ]

    fit = fit_cost_policy(
        samples,
        cost_weight=0.1,
        min_samples=1,
        require_cloud_tasks=("quality_verdict",),
    )

    assert fit.always_cloud_tasks == ("quality_verdict",)
    assert "quality_verdict" not in fit.thresholds_by_task
    report = fit.per_task["quality_verdict"]
    assert report.always_cloud is True
    assert report.threshold is None
    assert report.escalation_rate == pytest.approx(1.0)


def test_evaluate_reports_hand_computed_metrics() -> None:
    policy = CostPolicy(global_threshold=0.5, cost_weight=0.2)
    samples = [
        sample(
            "task",
            0.4,
            edge_correct=False,
            cloud_correct=True,
            edge_cost=0.1,
            escalated_cost=0.7,
            split="val",
        ),
        sample(
            "task",
            0.6,
            edge_correct=True,
            cloud_correct=False,
            edge_cost=0.2,
            escalated_cost=0.8,
            split="val",
        ),
    ]

    report = evaluate_cost_policy(policy, samples, split="val")

    assert isinstance(report, CostPolicyReport)
    assert report.n == 2
    assert report.accuracy == pytest.approx(1.0)
    assert report.cost == pytest.approx(0.45)
    assert report.cost_norm == pytest.approx(0.45)
    assert report.objective == pytest.approx(0.91)
    assert report.escalation_rate == pytest.approx(0.5)
    assert report.per_task["task"].accuracy == pytest.approx(1.0)
    assert report.per_task["task"].cost == pytest.approx(0.45)


def test_fit_report_records_cost_norm_and_objective() -> None:
    samples = [
        sample(
            "task",
            0.4,
            edge_correct=False,
            cloud_correct=True,
            edge_cost=0.1,
            escalated_cost=0.7,
        ),
        sample(
            "task",
            0.6,
            edge_correct=True,
            cloud_correct=False,
            edge_cost=0.2,
            escalated_cost=0.8,
        ),
    ]

    fit = fit_cost_policy(samples, cost_weight=0.2, threshold_grid=[0.5], min_samples=1)

    assert fit.global_report.threshold == pytest.approx(0.5)
    assert fit.global_report.accuracy == pytest.approx(1.0)
    assert fit.global_report.cost == pytest.approx(0.45)
    assert fit.global_report.cost_norm == pytest.approx(0.45)
    assert fit.global_report.objective == pytest.approx(0.91)
    assert fit.per_task["task"].objective == pytest.approx(0.91)


def test_cost_norm_may_exceed_one() -> None:
    policy = CostPolicy(global_threshold=0.5, cost_weight=0.0)
    samples = [
        sample(
            "task",
            0.2,
            edge_correct=True,
            cloud_correct=True,
            escalated_cost=3.0,
            cloud_only_cost=1.0,
            split="val",
        )
    ]

    report = evaluate_cost_policy(policy, samples)

    assert report.cost_norm == pytest.approx(3.0)


def test_evaluate_uses_per_task_thresholds_and_always_cloud() -> None:
    policy = CostPolicy(
        global_threshold=0.5,
        thresholds_by_task={"task_a": 0.2, "task_b": 0.8},
        cost_weight=0.0,
        always_cloud_tasks=("quality_verdict",),
    )
    samples = [
        sample("task_a", 0.5, edge_correct=True, cloud_correct=False, split="val"),
        sample("task_b", 0.5, edge_correct=True, cloud_correct=False, split="val"),
        sample(
            "quality_verdict",
            1.0,
            edge_correct=False,
            cloud_correct=True,
            split="val",
        ),
        sample("unknown_task", 0.5, edge_correct=True, cloud_correct=False, split="val"),
    ]

    report = evaluate_cost_policy(policy, samples)

    assert report.per_task["task_a"].escalation_rate == pytest.approx(0.0)
    assert report.per_task["task_a"].threshold == pytest.approx(0.2)
    assert report.per_task["task_b"].escalation_rate == pytest.approx(1.0)
    assert report.per_task["quality_verdict"].escalation_rate == pytest.approx(1.0)
    assert report.per_task["quality_verdict"].always_cloud is True
    assert report.per_task["quality_verdict"].threshold is None
    assert report.per_task["unknown_task"].escalation_rate == pytest.approx(0.0)
    assert report.per_task["unknown_task"].threshold == pytest.approx(0.5)
    assert report.escalation_rate == pytest.approx(0.5)
    assert report.cloud_calls == 2
    assert report.accuracy == pytest.approx(0.75)


def test_evaluate_accepts_a_fit_and_is_deterministic() -> None:
    train = two_task_samples()
    val = [
        sample("task_a", 0.3, edge_correct=True, cloud_correct=True, split="val"),
        sample("task_b", 0.5, edge_correct=False, cloud_correct=True, split="val"),
    ]
    fit = fit_cost_policy(train, cost_weight=0.1, min_samples=2)

    first = evaluate_cost_policy(fit, val, n_boot=50, seed=3)
    second = evaluate_cost_policy(fit.policy, val, n_boot=50, seed=3)

    assert first == second
    assert first.to_dict() == second.to_dict()
    assert first.accuracy_ci["samples"] == 50
    assert first.accuracy_ci["low"] <= first.accuracy_ci["point"] <= first.accuracy_ci["high"]


def test_policy_rejects_invalid_thresholds() -> None:
    with pytest.raises(ValueError):
        CostPolicy(global_threshold=1.1)
    with pytest.raises(ValueError):
        CostPolicy(global_threshold=0.5, thresholds_by_task={"task": -0.1})
    with pytest.raises(ValueError):
        CostPolicy(global_threshold=0.5, cost_weight=-0.1)


def test_evaluate_rejects_an_empty_or_unknown_split() -> None:
    policy = CostPolicy(global_threshold=0.5)
    samples = [sample("task", 0.5, edge_correct=True, cloud_correct=True, split="train")]

    with pytest.raises(ValueError):
        evaluate_cost_policy(policy, samples, split="val")
    with pytest.raises(ValueError):
        evaluate_cost_policy(policy, samples, split="test")


def test_fit_is_deterministic_and_to_config_round_trips() -> None:
    samples = two_task_samples()

    first = fit_cost_policy(samples, cost_weight=0.1, min_samples=2)
    second = fit_cost_policy(samples, cost_weight=0.1, min_samples=2)

    assert first == second
    assert json.dumps(asdict(first), sort_keys=True) == json.dumps(asdict(second), sort_keys=True)

    config = first.policy.to_config()
    assert set(config) == {"escalation_threshold", "escalation_threshold_by_task"}
    assert config["escalation_threshold"] == pytest.approx(first.global_threshold)
    assert config["escalation_threshold_by_task"] == first.thresholds_by_task
    assert first.to_config() == config

    restored = CostPolicy.from_config(config)
    assert restored.global_threshold == pytest.approx(first.global_threshold)
    assert restored.thresholds_by_task == first.thresholds_by_task


def test_empty_train_data_is_rejected() -> None:
    with pytest.raises(ValueError):
        fit_cost_policy([], cost_weight=0.1)


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -0.1, 1.1])
def test_invalid_score_is_rejected(score: float) -> None:
    with pytest.raises(ValueError):
        sample("task", score, edge_correct=True, cloud_correct=True)


@pytest.mark.parametrize("field", ["edge_cost", "escalated_cost", "cloud_only_cost"])
def test_negative_or_non_finite_cost_is_rejected(field: str) -> None:
    for value in (-0.1, float("nan"), float("inf")):
        kwargs = {
            "task": "task",
            "score": 0.5,
            "edge_correct": True,
            "cloud_correct": True,
            "edge_cost": 0.1,
            "escalated_cost": 0.5,
            "cloud_only_cost": 1.0,
            field: value,
        }
        with pytest.raises(ValueError):
            CostSample(**kwargs)


def test_invalid_split_is_rejected() -> None:
    with pytest.raises(ValueError):
        sample("task", 0.5, edge_correct=True, cloud_correct=True, split="test")


def test_invalid_fit_parameters_are_rejected() -> None:
    samples = two_task_samples()

    with pytest.raises(ValueError):
        fit_cost_policy(samples, cost_weight=-0.1)
    with pytest.raises(ValueError):
        fit_cost_policy(samples, cost_weight=float("nan"))
    with pytest.raises(ValueError):
        fit_cost_policy(samples, cost_weight=0.1, threshold_grid=[])
    with pytest.raises(ValueError):
        fit_cost_policy(samples, cost_weight=0.1, threshold_grid=[-0.1])
    with pytest.raises(ValueError):
        fit_cost_policy(samples, cost_weight=0.1, threshold_grid=[1.1])
    with pytest.raises(ValueError):
        fit_cost_policy(samples, cost_weight=0.1, min_samples=-1)
