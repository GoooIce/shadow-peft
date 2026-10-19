from __future__ import annotations

import math

import pytest

from edge_cloud.cost import CostModel
from edge_cloud.features import compute_handoff_features
from edge_cloud.handoff import (
    HandoffDecision,
    HandoffFeatures,
    HandoffOutcome,
    HandoffPolicy,
    HandoffStep,
)


def test_missing_shadow_features_stay_none_not_zero() -> None:
    features = HandoffFeatures()

    assert features.kl_base_shadow is None
    assert features.top1_agreement is None
    assert features.shadow_entropy is None
    assert features.shadow_margin is None
    assert features.base_top1_in_shadow_topk is None
    assert features.base_entropy == 0.0
    assert features.base_margin == 0.0
    assert features.low_prob_frac == 0.0
    assert features.prefix_tokens == 0


class _ContinuePolicy:
    def __init__(self) -> None:
        self.context: dict | None = None
        self.steps: list[HandoffStep] = []

    def reset(self, context: dict | None = None) -> None:
        self.context = context

    def observe(self, step: HandoffStep) -> HandoffDecision:
        self.steps.append(step)
        return HandoffDecision(action="continue")


def test_handoff_policy_protocol_accepts_a_deterministic_stub() -> None:
    policy: HandoffPolicy = _ContinuePolicy()
    step = HandoffStep(
        features=HandoffFeatures(prefix_tokens=3),
        prefix_text="检查完成",
        token="。",
        token_logprob=-0.2,
    )

    assert isinstance(policy, HandoffPolicy)
    policy.reset({"task": "diagnosis"})
    decision = policy.observe(step)

    assert policy.context == {"task": "diagnosis"}
    assert decision.action == "continue"
    assert decision.reason is None
    assert decision.handoff_index == 0


def test_handoff_decision_serializes_nested_cost() -> None:
    cost = CostModel(cost_per_cloud_call=0.01).estimate(cloud_calls=1)
    decision = HandoffDecision(
        action="handoff",
        reason="low_prefix_trust",
        boundary="sentence",
        handoff_index=42,
        prefix_trust=0.31,
        estimated_cost=cost,
    )

    payload = decision.to_dict()

    assert payload["action"] == "handoff"
    assert payload["reason"] == "low_prefix_trust"
    assert payload["boundary"] == "sentence"
    assert payload["handoff_index"] == 42
    assert payload["prefix_trust"] == 0.31
    assert payload["estimated_cost"]["estimated_cost"] == pytest.approx(0.01)


def test_handoff_outcome_defaults_to_no_handoff() -> None:
    outcome = HandoffOutcome()

    assert outcome.handed_off is False
    assert outcome.reason is None
    assert outcome.boundary is None
    assert outcome.handoff_index is None
    assert outcome.partial_trace == ""
    assert outcome.prefix_trust is None
    assert outcome.estimated_cost is None


def test_identical_logits_have_zero_kl_and_full_agreement() -> None:
    logits = [0.5, 1.5, -0.5, 2.0]

    features = compute_handoff_features(logits, logits)

    assert features.kl_base_shadow == 0.0
    assert features.top1_agreement == 1.0
    assert features.base_entropy == features.shadow_entropy
    assert features.base_margin == features.shadow_margin
    assert features.base_top1_in_shadow_topk is True


def test_kl_against_a_hand_worked_example() -> None:
    base = [math.log(0.9), math.log(0.1)]
    shadow = [math.log(0.5), math.log(0.5)]

    features = compute_handoff_features(base, shadow)

    # 0.9*ln(1.8) + 0.1*ln(0.2) = 0.368064...
    assert features.kl_base_shadow == pytest.approx(0.368064, abs=1e-6)
    assert features.top1_agreement == 1.0


def test_shadow_topk_membership_and_disagreement() -> None:
    base = [0.0, 4.0, 1.0, 0.5]  # base top-1 is index 1
    shadow = [0.0, 2.0, 1.5, 2.5]  # shadow top-2 is {3, 1}

    in_top2 = compute_handoff_features(base, shadow, topk=2)
    in_top1 = compute_handoff_features(base, shadow, topk=1)

    assert in_top2.top1_agreement == 0.0
    assert in_top2.base_top1_in_shadow_topk is True
    assert in_top1.base_top1_in_shadow_topk is False


def test_entropy_and_margin_from_known_probabilities() -> None:
    logits = [math.log(0.75), math.log(0.25)]

    features = compute_handoff_features(logits)

    assert features.base_entropy == pytest.approx(0.562335, abs=1e-6)
    assert features.base_margin == pytest.approx(0.5, abs=1e-6)


def test_uniform_two_way_distribution_has_ln2_entropy() -> None:
    features = compute_handoff_features([0.0, 0.0])

    assert features.base_entropy == pytest.approx(0.693147, abs=1e-6)
    assert features.base_margin == pytest.approx(0.0, abs=1e-6)


def test_no_shadow_degrades_shadow_features_to_none() -> None:
    features = compute_handoff_features([1.0, 0.0], None)

    assert features.kl_base_shadow is None
    assert features.top1_agreement is None
    assert features.shadow_entropy is None
    assert features.shadow_margin is None
    assert features.base_top1_in_shadow_topk is None
    # Base-side features are still real numbers, not placeholders.
    assert features.base_entropy > 0.0
    assert features.base_margin > 0.0


def test_low_prob_fraction_and_prefix_length() -> None:
    features = compute_handoff_features(
        [0.0, 1.0], token_logprobs=(-0.1, -3.0, -1.5, -2.5)
    )

    assert features.low_prob_frac == 0.5
    assert features.prefix_tokens == 4


def test_low_prob_fraction_ignores_non_finite_logprobs() -> None:
    features = compute_handoff_features(
        [0.0, 1.0],
        token_logprobs=[-0.5, float("-inf"), float("nan"), -3.0],
    )

    assert features.low_prob_frac == 0.5
    assert features.prefix_tokens == 4


def test_empty_token_logprobs_give_zero_fraction() -> None:
    features = compute_handoff_features([0.0, 1.0])

    assert features.low_prob_frac == 0.0
    assert features.prefix_tokens == 0


def test_context_fields_have_documented_neutral_defaults() -> None:
    features = compute_handoff_features([0.0, 1.0])

    assert features.boundary_kind == "token"
    assert features.task == "general"
    assert features.format_ok == 1.0


def test_extreme_logits_stay_finite_and_saturated() -> None:
    features = compute_handoff_features([-1000.0, 1000.0])

    assert math.isfinite(features.base_entropy)
    assert features.base_entropy == pytest.approx(0.0, abs=1e-9)
    assert features.base_margin == pytest.approx(1.0, abs=1e-9)


def test_kl_stays_finite_when_shadow_probability_underflows() -> None:
    features = compute_handoff_features([0.0, 1.0], [1000.0, -1000.0])

    assert features.kl_base_shadow is not None
    assert math.isfinite(features.kl_base_shadow)
    assert features.kl_base_shadow > 0.0


def test_topk_larger_than_vocabulary_is_clamped() -> None:
    features = compute_handoff_features([0.0, 1.0], [1.0, 0.0], topk=10)

    assert features.base_top1_in_shadow_topk is True


def test_empty_base_logits_are_rejected() -> None:
    with pytest.raises(ValueError):
        compute_handoff_features([])


def test_mismatched_logit_lengths_are_rejected() -> None:
    with pytest.raises(ValueError):
        compute_handoff_features([0.0], [0.0, 1.0])


def test_non_positive_topk_is_rejected() -> None:
    with pytest.raises(ValueError):
        compute_handoff_features([0.0, 1.0], topk=0)


def test_non_finite_logits_are_rejected() -> None:
    with pytest.raises(ValueError):
        compute_handoff_features([0.0, float("nan")])
