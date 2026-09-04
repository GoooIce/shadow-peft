"""Contracts for replayable antithetic ES over NVFP4 clip factors."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from shadow_lab.es import (
    CandidateObservation,
    ESConfig,
    NoRewardSignalError,
    candidate_clips,
    candidate_latent,
    candidate_specs,
    estimate_es_gradient,
    initialization_projection_report,
    initialize_es,
    load_es_checkpoint,
    replay_noise,
    save_es_checkpoint,
    shape_rewards,
    update_es,
)


def _state(shaping: str = "zscore"):
    config = ESConfig(
        dimension=8,
        population_size=4,
        sigma=0.03,
        learning_rate=0.01,
        shaping=shaping,
    )
    return initialize_es(torch.tensor([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]), config)


def test_candidate_noise_is_antithetic_replayable_and_bounded() -> None:
    state = _state()
    plus, minus = candidate_specs(state)[:2]

    assert plus.noise_seed == minus.noise_seed
    assert plus.sign == 1 and minus.sign == -1
    expected_noise = replay_noise(state.config, plus.noise_seed)
    assert torch.allclose(
        candidate_latent(state, plus) - state.center,
        state.config.sigma * expected_noise,
    )
    assert torch.allclose(
        candidate_latent(state, minus) - state.center,
        -state.config.sigma * expected_noise,
    )
    assert torch.equal(candidate_clips(state, plus), candidate_clips(state, plus))
    assert torch.all(candidate_clips(state, plus) >= 0.3)
    assert torch.all(candidate_clips(state, plus) <= 1.0)


def test_zero_perturbation_is_the_initialized_baseline() -> None:
    state = _state()
    baseline = candidate_clips(
        state,
        candidate_specs(state)[0],
    )
    zero_sigma_config = ESConfig(
        dimension=state.config.dimension,
        population_size=state.config.population_size,
        sigma=1e-30,
    )
    zero_state = initialize_es(
        torch.tensor([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
        zero_sigma_config,
    )
    zero = candidate_clips(zero_state, candidate_specs(zero_state)[0])

    assert not torch.equal(baseline, zero)
    assert torch.allclose(
        zero,
        torch.tensor([0.301, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.999]),
        atol=2e-6,
    )
    report = initialization_projection_report(
        torch.tensor([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
        zero_sigma_config,
    )
    assert report["lower_endpoint_count"] == 1
    assert report["upper_endpoint_count"] == 1
    assert report["max_abs_projection"] == pytest.approx(0.001, abs=2e-6)


def test_update_moves_toward_rewards_and_replays_only_seeds() -> None:
    state = _state()
    specs = candidate_specs(state)
    observations = [
        CandidateObservation(spec, float(spec.sign)) for spec in specs
    ]

    first = update_es(state, observations)
    second = update_es(state, observations)

    assert torch.equal(first.state.center, second.state.center)
    assert first.state.generation == 1
    assert first.gradient_norm > 0
    assert first.gradient_rms == pytest.approx(
        first.gradient_norm / state.config.dimension**0.5,
        rel=1e-6,
    )
    assert first.latent_step_rms == pytest.approx(
        state.config.learning_rate * first.gradient_rms
    )
    assert first.failure_count == 0


def test_gradient_estimate_is_the_public_replayable_update_direction() -> None:
    state = _state()
    observations = [
        CandidateObservation(spec, float(spec.sign * (spec.pair_index + 1)))
        for spec in candidate_specs(state)
    ]

    estimate = estimate_es_gradient(state, observations)
    update = update_es(state, observations)

    assert torch.equal(
        update.state.center,
        state.center + state.config.learning_rate * estimate.vector,
    )
    assert estimate.raw_rewards == update.raw_rewards
    assert estimate.shaped_rewards == update.shaped_rewards
    assert estimate.reward_mean == update.reward_mean
    assert estimate.reward_std == update.reward_std
    assert estimate.norm == update.gradient_norm
    assert estimate.rms == update.gradient_rms


def test_failures_are_penalized_but_do_not_abort_generation() -> None:
    state = _state("rank")
    specs = candidate_specs(state)
    observations = [
        CandidateObservation(specs[0], 1.0),
        CandidateObservation(specs[1], None, "generation OOM"),
        CandidateObservation(specs[2], 0.5),
        CandidateObservation(specs[3], 0.0),
    ]

    update = update_es(state, observations)

    assert update.failure_count == 1
    assert update.raw_rewards[1] < min(1.0, 0.5, 0.0)


def test_identical_rewards_have_no_search_signal() -> None:
    with pytest.raises(NoRewardSignalError):
        shape_rewards([1.0, 1.0], "zscore")


def test_checkpoint_replays_the_exact_next_population(tmp_path: Path) -> None:
    state = _state()
    path = tmp_path / "es-state.pt"

    save_es_checkpoint(state, path)
    restored = load_es_checkpoint(path)

    assert restored.config == state.config
    assert restored.generation == state.generation
    assert torch.equal(restored.center, state.center)
    assert candidate_specs(restored) == candidate_specs(state)
    assert torch.equal(
        candidate_clips(restored, candidate_specs(restored)[0]),
        candidate_clips(state, candidate_specs(state)[0]),
    )
