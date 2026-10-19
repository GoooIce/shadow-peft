from __future__ import annotations

import pytest

from edge_cloud.cost import CostEstimate, CostModel


def test_components_are_priced_separately() -> None:
    model = CostModel(
        cost_per_edge_token=0.000001,
        cost_per_cloud_input_token=0.000002,
        cost_per_cloud_output_token=0.000004,
        cost_per_cloud_call=0.01,
        cloud_rtt_ms=120.0,
    )

    estimate = model.estimate(
        edge_tokens=100,
        cloud_prefill_tokens=200,
        cloud_decode_tokens_expected=50,
        cloud_calls=1,
    )

    assert isinstance(estimate, CostEstimate)
    assert estimate.edge_tokens == 100
    assert estimate.cloud_prefill_tokens == 200
    assert estimate.cloud_decode_tokens_expected == 50
    assert estimate.cloud_calls == 1
    assert estimate.rtt_ms == 120.0
    assert estimate.edge_cost == pytest.approx(0.0001)
    assert estimate.cloud_prefill_cost == pytest.approx(0.0004)
    assert estimate.cloud_decode_cost == pytest.approx(0.0002)
    assert estimate.cloud_call_cost == pytest.approx(0.01)
    assert estimate.estimated_cost == pytest.approx(0.0107)


def test_zero_token_prices_fall_back_to_per_call_pricing() -> None:
    model = CostModel(cost_per_cloud_call=0.02)

    estimate = model.estimate(
        edge_tokens=500,
        cloud_prefill_tokens=800,
        cloud_decode_tokens_expected=120,
        cloud_calls=2,
    )

    assert estimate.edge_cost == 0.0
    assert estimate.cloud_prefill_cost == 0.0
    assert estimate.cloud_decode_cost == 0.0
    assert estimate.cloud_call_cost == pytest.approx(0.04)
    assert estimate.estimated_cost == pytest.approx(0.04)


def test_all_zero_prices_cost_nothing() -> None:
    estimate = CostModel().estimate(
        edge_tokens=10,
        cloud_prefill_tokens=20,
        cloud_decode_tokens_expected=30,
        cloud_calls=1,
    )

    assert estimate.estimated_cost == 0.0


def test_empty_estimate_is_zero() -> None:
    estimate = CostModel(cost_per_cloud_call=0.5).estimate()

    assert estimate.cloud_calls == 0
    assert estimate.estimated_cost == 0.0


def test_rtt_is_reported_and_can_be_overridden() -> None:
    model = CostModel(cloud_rtt_ms=85.0)

    assert model.estimate().rtt_ms == 85.0
    assert model.estimate(rtt_ms=40.0).rtt_ms == 40.0


def test_to_dict_round_trips_every_field() -> None:
    estimate = CostModel(cost_per_cloud_call=0.01).estimate(cloud_calls=1, rtt_ms=15.0)

    payload = estimate.to_dict()

    assert payload["cloud_calls"] == 1
    assert payload["rtt_ms"] == 15.0
    assert payload["estimated_cost"] == pytest.approx(0.01)
    assert set(payload) == {
        "edge_tokens",
        "cloud_prefill_tokens",
        "cloud_decode_tokens_expected",
        "cloud_calls",
        "rtt_ms",
        "estimated_cost",
        "edge_cost",
        "cloud_prefill_cost",
        "cloud_decode_cost",
        "cloud_call_cost",
    }


def test_estimation_is_deterministic() -> None:
    model = CostModel(
        cost_per_edge_token=0.3,
        cost_per_cloud_input_token=0.7,
        cost_per_cloud_output_token=1.1,
        cost_per_cloud_call=0.05,
        cloud_rtt_ms=30.0,
    )
    first = model.estimate(edge_tokens=7, cloud_prefill_tokens=11, cloud_decode_tokens_expected=13, cloud_calls=2)
    second = model.estimate(edge_tokens=7, cloud_prefill_tokens=11, cloud_decode_tokens_expected=13, cloud_calls=2)

    assert first == second


def test_negative_prices_are_rejected() -> None:
    with pytest.raises(ValueError):
        CostModel(cost_per_cloud_output_token=-0.1)


def test_negative_counts_are_rejected() -> None:
    model = CostModel()

    with pytest.raises(ValueError):
        model.estimate(edge_tokens=-1)
    with pytest.raises(ValueError):
        model.estimate(cloud_calls=-1)
    with pytest.raises(ValueError):
        model.estimate(rtt_ms=-5.0)
