"""RES-003 contracts: pre-registered ES inference, power simulation and verdict wording."""

from __future__ import annotations

import json
import math

import pytest
import torch

from shadow_lab.es_inference import (
    GAIN_EXCLUDED,
    GAIN_SUPPORTED,
    INSUFFICIENT_EVIDENCE,
    StatisticalPlan,
    budget_verdict,
    classify_outcome,
    cluster_bootstrap_interval,
    consistency_checkpoint_power,
    cosine_episode_flip_null,
    delta_support_table,
    episode_pair_cosines,
    episode_sign_flip_test,
    fixed_sequence_family,
    holm_family,
    joint_gate_power,
    preference_power,
    sequential_checkpoint_power,
)

# -- episode-level sign-flip null ---------------------------------------------


def test_sign_flip_exact_p_is_symmetric_and_bounded() -> None:
    # k=4 -> 16 exact sign patterns; all-positive deltas have one-sided p = 1/16.
    result = episode_sign_flip_test([1.0, 2.0, 3.0, 4.0])
    assert result["method"] == "exact"
    assert result["pattern_count"] == 16
    assert result["p_greater"] == pytest.approx(1 / 16)
    assert result["mean"] == pytest.approx(2.5)

    centered = [1.0, -1.0, 1.0, -1.0]
    null_result = episode_sign_flip_test(centered)
    # Over the 16 patterns the flipped mean is negative 5 times, zero 6 times and
    # positive 5 times, so the greater-sided p-value is 11/16 and the two-sided
    # p-value saturates at 1.
    assert null_result["p_greater"] == pytest.approx(11 / 16)
    assert null_result["p_two_sided"] == pytest.approx(1.0)


def test_sign_flip_monte_carlo_is_seeded_and_reproducible() -> None:
    deltas = [0.4, -0.1, 0.9, 0.2, 0.6, -0.3, 0.8, 0.1]
    first = episode_sign_flip_test(deltas, resamples=100, seed=11)
    second = episode_sign_flip_test(deltas, resamples=100, seed=11)
    assert first == second
    assert first["method"] == "monte-carlo"
    assert first["pattern_count"] == 100
    assert 0.0 < first["p_greater"] <= 1.0


def test_sign_flip_rejects_invalid_input() -> None:
    with pytest.raises(ValueError):
        episode_sign_flip_test([1.0, float("nan")])
    with pytest.raises(ValueError):
        episode_sign_flip_test([1.0])


# -- prompt-cluster bootstrap --------------------------------------------------


def test_cluster_bootstrap_carries_whole_clusters_and_is_deterministic() -> None:
    # Two clusters of four seeds each; resampling must move clusters, never split them.
    clusters = {
        "prompt-0": [0.30, 0.32, 0.28, 0.31],
        "prompt-1": [-0.10, -0.12, -0.08, -0.11],
    }
    first = cluster_bootstrap_interval(clusters, resamples=2_000, seed=20260904)
    second = cluster_bootstrap_interval(clusters, resamples=2_000, seed=20260904)
    assert first == second
    assert first["mean"] == pytest.approx(0.1)
    assert first["cluster_count"] == 2
    assert first["resamples"] == 2_000
    assert first["lower"] <= first["mean"] <= first["upper"]
    # Only two clusters -> some resamples are all-prompt-0 or all-prompt-1, so the
    # interval must be at least as wide as the between-prompt spread.
    assert first["lower"] < -0.05 and first["upper"] > 0.25


def test_cluster_bootstrap_rejects_unpaired_or_empty_clusters() -> None:
    with pytest.raises(ValueError):
        cluster_bootstrap_interval({})
    with pytest.raises(ValueError):
        cluster_bootstrap_interval({"a": [0.1], "b": [float("nan")]})


# -- cosine hierarchy null ------------------------------------------------------


def _three_vector_gram() -> list[torch.Tensor]:
    """Vectors with pairwise cosines (0.5, 0.5, 0.0) sharing episode 1."""
    v1 = torch.tensor([1.0, 0.0, 0.0])
    v2 = torch.tensor([0.5, math.sqrt(0.75), 0.0])
    v3 = torch.tensor([0.5, -0.25 / math.sqrt(0.75), math.sqrt(0.75 - 0.25**2 / 0.75)])
    return [v1, v2, v3]


def test_cosine_flip_null_detects_naive_independence_overstatement() -> None:
    vectors = _three_vector_gram()
    result = cosine_episode_flip_null(vectors)
    assert result["mean_cosine"] == pytest.approx(1 / 3)
    assert result["pair_count"] == 3
    assert result["method"] == "exact"
    # Pairs share episode 1, so the naive independent-pair sd (pop sd 0.2357 over
    # sqrt(3) pairs = 0.136) is anticonservative: the flip sd is sqrt(0.5)/3 = 0.2357.
    assert result["naive_sd"] == pytest.approx(0.2357 / math.sqrt(3), rel=1e-3)
    assert result["flip_sd"] == pytest.approx(math.sqrt(0.5) / 3, rel=1e-3)
    assert result["flip_p_greater"] == pytest.approx(2 / 8)
    assert result["naive_z"] == pytest.approx((1 / 3) / (0.2357 / math.sqrt(3)), rel=1e-3)
    assert result["naive_z"] > (1 / 3) / result["flip_sd"]
    assert result["precision_ratio"] == pytest.approx(
        result["naive_sd"] / result["flip_sd"]
    )


def test_cosine_flip_null_monte_carlo_reproducible_and_validates() -> None:
    generator = torch.Generator().manual_seed(5)
    vectors = [torch.randn(40, generator=generator) for _ in range(24)]
    first = cosine_episode_flip_null(vectors, resamples=2_000, seed=7)
    second = cosine_episode_flip_null(vectors, resamples=2_000, seed=7)
    assert first == second
    assert first["method"] == "monte-carlo"
    # For 24 independent random directions the pair cosines are nearly uncorrelated,
    # so the naive sd sits near the flip sd; the exact shared-episode test above is
    # what exposes the anticonservative direction.
    assert abs(first["precision_ratio"] - 1.0) < 0.25
    assert 0.0 <= first["flip_p_greater"] <= 1.0
    with pytest.raises(ValueError):
        cosine_episode_flip_null([torch.zeros(4), torch.randn(4)])
    with pytest.raises(ValueError):
        cosine_episode_flip_null([torch.randn(4)])


# -- joint selection/confirmation gate -----------------------------------------


def test_joint_gate_null_fp_near_analytic_and_power_monotone() -> None:
    null_row = joint_gate_power(
        episodes=8,
        min_non_degrading=6,
        effects=[0.0, 0.5, 2.0],
        n_sims=4_000,
        seed=20260904,
    )
    by_effect = {row["effect"]: row for row in null_row}
    # Review example: sign-model P(>=6 of 8) = 37/256; the joint gate sits between
    # the win rule alone and the mean rule alone.
    assert 0.10 < by_effect[0.0]["pass_rate"] < 0.20
    assert by_effect[0.0]["mean_only_rate"] > by_effect[0.0]["pass_rate"]
    assert by_effect[0.5]["pass_rate"] > by_effect[0.0]["pass_rate"]
    assert by_effect[2.0]["pass_rate"] > by_effect[0.5]["pass_rate"]
    assert by_effect[2.0]["pass_rate"] > 0.95
    # Determinism.
    repeat = joint_gate_power(
        episodes=8, min_non_degrading=6, effects=[0.0], n_sims=4_000, seed=20260904
    )
    assert repeat[0]["pass_rate"] == by_effect[0.0]["pass_rate"]


def test_joint_gate_k16_is_tighter_at_null() -> None:
    k8 = joint_gate_power(episodes=8, min_non_degrading=6, effects=[0.0], n_sims=4_000, seed=1)
    k16 = joint_gate_power(episodes=16, min_non_degrading=12, effects=[0.0], n_sims=4_000, seed=1)
    assert k16[0]["pass_rate"] < k8[0]["pass_rate"]


# -- consistency checkpoint rule ------------------------------------------------


def test_checkpoint_rule_fp_near_zero_and_power_rises_with_true_cosine() -> None:
    rows = consistency_checkpoint_power(
        episode_counts=[8],
        rho_grid=[0.0, 0.10, 0.30],
        dimension=40,
        n_sims=2_000,
        seed=20260904,
    )
    by_rho = {row["rho_true"]: row for row in rows}
    assert by_rho[0.0]["pass_rate"] < 0.01
    assert by_rho[0.10]["pass_rate"] > by_rho[0.0]["pass_rate"]
    assert by_rho[0.30]["pass_rate"] > by_rho[0.10]["pass_rate"]
    assert by_rho[0.30]["pass_rate"] > 0.5


def test_sequential_rule_survival_matches_whole_rule_calibration() -> None:
    rows = sequential_checkpoint_power(
        gradient_episodes=16,
        checkpoints=[(8, 0.10, 0.70), (16, 0.10, 0.70)],
        rho_grid=[0.0, 0.15],
        dimension=40,
        n_sims=2_000,
        seed=20260904,
    )
    by_rho = {row["rho_true"]: row for row in rows}
    null = by_rho[0.0]
    signal = by_rho[0.15]
    # Under the null the sequential procedure almost surely stops (the survival
    # rate is the whole-rule false-positive rate); at rho=0.15 most campaigns pass
    # both checkpoints.
    assert null["survival_rate"] < 0.01
    assert null["stopped_rate"] == pytest.approx(1 - null["survival_rate"])
    assert null["stop_rate_at"]["8"] >= null["stop_rate_at"]["16"]
    assert signal["survival_rate"] > 0.5
    with pytest.raises(ValueError):
        sequential_checkpoint_power(
            gradient_episodes=8,
            checkpoints=[(8, 0.1, 0.7), (16, 0.1, 0.7)],
            rho_grid=[0.0],
            n_sims=100,
            seed=1,
        )


def test_episode_pair_cosines_matches_manual_gram() -> None:
    directions = torch.tensor(
        [
            [[3.0, 0.0], [0.0, 4.0], [3.0, 4.0]],
        ]
    )
    cosines = episode_pair_cosines(directions)[0]
    assert cosines.shape == (3,)
    assert cosines[0] == pytest.approx(0.0)
    assert cosines[1] == pytest.approx(0.6)
    assert cosines[2] == pytest.approx(0.8)
    with pytest.raises(ValueError):
        episode_pair_cosines(torch.tensor([[[1.0, 0.0]]]))


# -- delta support / budget -----------------------------------------------------


def test_delta_support_table_verdicts_and_exclusion_power() -> None:
    rows = delta_support_table(
        episode_counts=[8],
        sigma_over_delta=[0.5, 2.0],
        effects=[0.0, 2.0],
        n_sims=1_000,
        bootstrap_resamples=500,
        seed=20260904,
    )
    by_cell = {(row["episodes"], row["sigma_over_delta"], row["effect"]): row for row in rows}
    tight_null = by_cell[(8, 0.5, 0.0)]
    wide_null = by_cell[(8, 2.0, 0.0)]
    strong_effect = by_cell[(8, 0.5, 2.0)]
    # With noise at half the useful-gain scale, a null study almost always lands its
    # 95% interval below delta: the budget can honestly exclude. At twice the gain
    # scale, exclusion collapses and "insufficient evidence" dominates.
    assert tight_null["gain_excluded_rate"] > 0.95
    assert tight_null["gain_excluded_rate"] > wide_null["gain_excluded_rate"]
    assert wide_null["insufficient_rate"] > 0.5
    assert strong_effect["gain_supported_rate"] > 0.9
    for row in rows:
        assert (
            row["gain_supported_rate"]
            + row["gain_excluded_rate"]
            + row["insufficient_rate"]
            == pytest.approx(1.0)
        )


def test_budget_verdict_thresholds() -> None:
    rows = delta_support_table(
        episode_counts=[8],
        sigma_over_delta=[0.5, 1.25, 2.0],
        effects=[0.0],
        n_sims=1_000,
        bootstrap_resamples=500,
        seed=20260904,
    )
    assert budget_verdict(rows, episodes=8, sigma_over_delta=0.5) == "budget-supports-delta"
    assert budget_verdict(rows, episodes=8, sigma_over_delta=1.25) == "interval-only"
    assert budget_verdict(rows, episodes=8, sigma_over_delta=2.0) == "redesign-required"
    with pytest.raises(KeyError):
        budget_verdict(rows, episodes=16, sigma_over_delta=0.5)


# -- preference power (EXP-011 style) -------------------------------------------


def test_preference_power_fp_controlled_and_monotone() -> None:
    rows = preference_power(
        prompts=30,
        judgments_per_prompt=6,
        q_grid=[0.5, 0.7, 0.9],
        decisive_rate=0.8,
        n_sims=1_000,
        bootstrap_resamples=500,
        seed=20260904,
    )
    by_q = {row["q"]: row for row in rows}
    assert by_q[0.5]["gain_supported_rate"] < 0.10
    assert by_q[0.7]["gain_supported_rate"] > by_q[0.5]["gain_supported_rate"]
    assert by_q[0.9]["gain_supported_rate"] > by_q[0.7]["gain_supported_rate"]
    for row in rows:
        assert 0.0 <= row["gain_excluded_rate"] <= 1.0


# -- verdict wording ------------------------------------------------------------


def test_classify_outcome_maps_the_three_pre_registered_wordings() -> None:
    supported = classify_outcome(mean=0.30, lower=0.12, upper=0.48, delta=0.25)
    assert supported["verdict"] == GAIN_SUPPORTED
    assert "可继续" in supported["wording"]

    excluded = classify_outcome(mean=0.05, lower=-0.02, upper=0.11, delta=0.25)
    assert excluded["verdict"] == GAIN_EXCLUDED
    assert "已排除" in excluded["wording"]

    insufficient = classify_outcome(mean=0.10, lower=-0.05, upper=0.25, delta=0.25)
    assert insufficient["verdict"] == INSUFFICIENT_EVIDENCE
    assert "证据不足" in insufficient["wording"]

    positive_but_small = classify_outcome(mean=0.05, lower=0.01, upper=0.09, delta=0.25)
    assert positive_but_small["verdict"] == GAIN_EXCLUDED
    assert positive_but_small["positive_point_estimate"] is True

    with pytest.raises(ValueError):
        classify_outcome(mean=0.1, lower=0.2, upper=0.05, delta=0.25)


# -- multiplicity ----------------------------------------------------------------


def test_fixed_sequence_stops_at_first_failure() -> None:
    results = fixed_sequence_family(
        [
            {"family": "noise-group-a", "p_value": 0.01},
            {"family": "noise-group-b", "p_value": 0.2},
            {"family": "noise-group-c", "p_value": 0.001},
        ],
        alpha=0.05,
    )
    assert [row["verdict"] for row in results] == [
        "supported",
        "not-supported",
        "not-tested",
    ]
    assert results[2]["reason"] == "fixed-sequence-stopped"


def test_holm_family_adjusts_in_order() -> None:
    results = holm_family(
        [
            {"family": "c", "p_value": 0.04},
            {"family": "b", "p_value": 0.03},
            {"family": "a", "p_value": 0.01},
        ],
        alpha=0.05,
    )
    by_name = {row["family"]: row for row in results}
    # Holm step-down with m=3: 0.01*3=0.03, 0.03*2=0.06, and the raw 0.04 is lifted
    # to the running max 0.06 so adjusted p-values stay monotone.
    assert by_name["a"]["p_holm"] == pytest.approx(0.03)
    assert by_name["b"]["p_holm"] == pytest.approx(0.06)
    assert by_name["c"]["p_holm"] == pytest.approx(0.06)
    assert by_name["a"]["verdict"] == "supported"
    assert by_name["b"]["verdict"] == "not-supported"
    assert by_name["c"]["verdict"] == "not-supported"


# -- frozen plan -----------------------------------------------------------------


def test_statistical_plan_freezes_to_stable_json_and_hash() -> None:
    plan = StatisticalPlan.default()
    payload = plan.canonical_json()
    parsed = json.loads(payload)
    assert parsed["delta_primary"] == plan.delta_primary
    rebuilt = StatisticalPlan.from_json(payload)
    assert rebuilt == plan
    assert rebuilt.content_hash() == plan.content_hash()
    assert len(rebuilt.content_hash()) == 64
    # JSON round-trip (lists) must hash identically to the Python-built plan.
    assert StatisticalPlan.from_json(json.dumps(parsed)).content_hash() == plan.content_hash()


def test_statistical_plan_rejects_inconsistent_settings() -> None:
    with pytest.raises(ValueError):
        StatisticalPlan.default(delta_primary=0.0)
    with pytest.raises(ValueError):
        StatisticalPlan.default(alpha_selection=0.05, alpha_confirmation=0.05)
    with pytest.raises(ValueError):
        StatisticalPlan.default(min_power_support=1.1)
