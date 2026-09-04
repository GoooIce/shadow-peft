"""Replayable antithetic evolution strategies for NVFP4 clip genomes."""

from __future__ import annotations

import hashlib
import math
import os
import statistics
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import torch


class NoRewardSignalError(RuntimeError):
    """Raised when a complete generation contains no usable search signal."""


@dataclass(frozen=True)
class ESConfig:
    dimension: int
    population_size: int = 30
    sigma: float = 0.03
    learning_rate: float = 0.015
    shaping: str = "zscore"
    seed: int = 20260904
    clip_min: float = 0.30
    clip_max: float = 1.00
    transform_temperature: float = 0.10
    boundary_margin: float = 0.001

    def __post_init__(self) -> None:
        if self.dimension <= 0:
            raise ValueError("ES dimension must be positive")
        if self.population_size <= 0 or self.population_size % 2:
            raise ValueError("antithetic population size must be positive and even")
        if self.sigma <= 0 or self.learning_rate <= 0:
            raise ValueError("sigma and learning rate must be positive")
        if self.shaping not in {"zscore", "rank"}:
            raise ValueError("shaping must be zscore or rank")
        if self.seed < 0 or not self.clip_min < self.clip_max:
            raise ValueError("invalid seed or clip interval")
        if self.transform_temperature <= 0:
            raise ValueError("transform temperature must be positive")
        if not 0 < self.boundary_margin < (self.clip_max - self.clip_min) / 2:
            raise ValueError("boundary margin must fit strictly inside the clip interval")


@dataclass(frozen=True)
class CandidateSpec:
    generation: int
    population_index: int
    pair_index: int
    sign: int
    noise_seed: int


@dataclass(frozen=True)
class CandidateObservation:
    spec: CandidateSpec
    reward: float | None
    failure_reason: str | None = None

    def __post_init__(self) -> None:
        if self.reward is not None and not math.isfinite(self.reward):
            raise ValueError("candidate reward must be finite or None")
        if (self.reward is None) != (self.failure_reason is not None):
            raise ValueError("failed candidates require a reason and valid candidates must not")
        if self.failure_reason is not None and (
            not isinstance(self.failure_reason, str) or not self.failure_reason.strip()
        ):
            raise ValueError("candidate failure reason must be a non-empty string")


@dataclass(frozen=True)
class ESState:
    config: ESConfig
    center: torch.Tensor
    generation: int = 0

    def __post_init__(self) -> None:
        if self.center.device.type != "cpu" or self.center.dtype != torch.float32:
            raise ValueError("ES center must be a CPU float32 tensor")
        if self.center.ndim != 1 or self.center.numel() != self.config.dimension:
            raise ValueError("ES center shape differs from configured dimension")
        if not torch.isfinite(self.center).all():
            raise ValueError("ES center must be finite")
        if self.generation < 0:
            raise ValueError("generation must be non-negative")


@dataclass(frozen=True)
class ESGradientEstimate:
    vector: torch.Tensor
    raw_rewards: tuple[float, ...]
    shaped_rewards: tuple[float, ...]
    failure_count: int
    reward_mean: float
    reward_std: float
    norm: float
    rms: float


@dataclass(frozen=True)
class ESUpdate:
    state: ESState
    raw_rewards: tuple[float, ...]
    shaped_rewards: tuple[float, ...]
    failure_count: int
    reward_mean: float
    reward_std: float
    gradient_norm: float
    gradient_rms: float
    latent_step_rms: float


def clips_to_latent(clips: torch.Tensor, config: ESConfig) -> torch.Tensor:
    """Encode clips into the unconstrained search space with an explicit margin."""

    values = clips.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
    if values.numel() != config.dimension or not torch.isfinite(values).all():
        raise ValueError("initial clip vector shape/value mismatch")
    if (values < config.clip_min).any() or (values > config.clip_max).any():
        raise ValueError("initial clips are outside the configured interval")
    lower = config.clip_min + config.boundary_margin
    upper = config.clip_max - config.boundary_margin
    normalized = (values.clamp(lower, upper) - config.clip_min) / (
        config.clip_max - config.clip_min
    )
    return (
        config.transform_temperature
        * torch.log(normalized / (1.0 - normalized))
    ).contiguous()


def latent_to_clips(latent: torch.Tensor, config: ESConfig) -> torch.Tensor:
    if latent.numel() != config.dimension or not torch.isfinite(latent).all():
        raise ValueError("latent vector shape/value mismatch")
    normalized = torch.sigmoid(
        latent.to(device="cpu", dtype=torch.float32).reshape(-1)
        / config.transform_temperature
    )
    return (
        config.clip_min + (config.clip_max - config.clip_min) * normalized
    ).contiguous()


def initialize_es(initial_clips: torch.Tensor, config: ESConfig) -> ESState:
    return ESState(config=config, center=clips_to_latent(initial_clips, config))


def initialization_projection_report(
    initial_clips: torch.Tensor, config: ESConfig
) -> dict:
    """Make endpoint movement from QUASAR's inclusive box explicit in manifests."""

    original = initial_clips.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
    effective = latent_to_clips(clips_to_latent(original, config), config)
    return {
        "dimension": config.dimension,
        "boundary_margin": config.boundary_margin,
        "lower_endpoint_count": int((original == config.clip_min).sum()),
        "upper_endpoint_count": int((original == config.clip_max).sum()),
        "original_min": float(original.min()),
        "original_max": float(original.max()),
        "effective_min": float(effective.min()),
        "effective_max": float(effective.max()),
        "max_abs_projection": float((effective - original).abs().max()),
    }


def _noise_seed(base_seed: int, generation: int, pair_index: int) -> int:
    payload = f"{base_seed}\0{generation}\0{pair_index}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**63 - 1)


def candidate_specs(state: ESState) -> tuple[CandidateSpec, ...]:
    specs = []
    for pair_index in range(state.config.population_size // 2):
        seed = _noise_seed(state.config.seed, state.generation, pair_index)
        for sign in (1, -1):
            specs.append(
                CandidateSpec(
                    generation=state.generation,
                    population_index=len(specs),
                    pair_index=pair_index,
                    sign=sign,
                    noise_seed=seed,
                )
            )
    return tuple(specs)


def replay_noise(config: ESConfig, noise_seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(noise_seed)
    return torch.randn(config.dimension, generator=generator, dtype=torch.float32)


def candidate_latent(state: ESState, spec: CandidateSpec) -> torch.Tensor:
    expected = candidate_specs(state)[spec.population_index]
    if spec != expected:
        raise ValueError("candidate spec does not belong to the current ES generation")
    return state.center + spec.sign * state.config.sigma * replay_noise(
        state.config,
        spec.noise_seed,
    )


def candidate_clips(state: ESState, spec: CandidateSpec) -> torch.Tensor:
    return latent_to_clips(candidate_latent(state, spec), state.config)


def _replace_failures(observations: tuple[CandidateObservation, ...]) -> tuple[list[float], int]:
    valid = [item.reward for item in observations if item.reward is not None]
    if not valid:
        raise NoRewardSignalError("every candidate in the generation failed")
    valid_values = [float(value) for value in valid]
    scale = statistics.pstdev(valid_values) if len(valid_values) > 1 else 0.0
    if scale <= 1e-12:
        scale = max(abs(valid_values[0]), 1.0) * 0.1
    penalty = min(valid_values) - scale
    return [penalty if item.reward is None else item.reward for item in observations], len(
        observations
    ) - len(valid_values)


def _average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        stop = cursor + 1
        while stop < len(order) and values[order[stop]] == values[order[cursor]]:
            stop += 1
        rank = (cursor + stop - 1) / 2
        for index in order[cursor:stop]:
            ranks[index] = rank
        cursor = stop
    return ranks


def shape_rewards(values: list[float], method: str) -> list[float]:
    if len(values) < 2 or not all(math.isfinite(value) for value in values):
        raise ValueError("reward shaping requires at least two finite values")
    if method == "zscore":
        mean = statistics.fmean(values)
        scale = statistics.pstdev(values)
        if scale <= 1e-12:
            raise NoRewardSignalError("candidate rewards are all identical")
        return [(value - mean) / scale for value in values]
    if method == "rank":
        ranks = _average_ranks(values)
        if max(ranks) == min(ranks):
            raise NoRewardSignalError("candidate rewards are all tied")
        return [rank / (len(values) - 1) - 0.5 for rank in ranks]
    raise ValueError(f"unknown reward shaping method: {method}")


def update_es(
    state: ESState, observations: list[CandidateObservation]
) -> ESUpdate:
    estimate = estimate_es_gradient(state, observations)
    updated = replace(
        state,
        center=(
            state.center + state.config.learning_rate * estimate.vector
        ).contiguous(),
        generation=state.generation + 1,
    )
    return ESUpdate(
        state=updated,
        raw_rewards=estimate.raw_rewards,
        shaped_rewards=estimate.shaped_rewards,
        failure_count=estimate.failure_count,
        reward_mean=estimate.reward_mean,
        reward_std=estimate.reward_std,
        gradient_norm=estimate.norm,
        gradient_rms=estimate.rms,
        latent_step_rms=state.config.learning_rate * estimate.rms,
    )


def estimate_es_gradient(
    state: ESState, observations: list[CandidateObservation]
) -> ESGradientEstimate:
    """Replay one population into its update direction without moving the center."""

    expected = candidate_specs(state)
    if tuple(item.spec for item in observations) != expected:
        raise ValueError("observations must contain the exact current population in order")
    raw_rewards, failures = _replace_failures(tuple(observations))
    shaped = shape_rewards(raw_rewards, state.config.shaping)
    gradient = torch.zeros_like(state.center)
    for item, utility in zip(observations, shaped, strict=True):
        gradient.add_(
            replay_noise(state.config, item.spec.noise_seed),
            alpha=utility * item.spec.sign,
        )
    gradient.div_(state.config.population_size * state.config.sigma)
    gradient_norm = float(torch.linalg.vector_norm(gradient))
    gradient_rms = float(torch.sqrt(torch.mean(gradient.square())))
    return ESGradientEstimate(
        vector=gradient,
        raw_rewards=tuple(raw_rewards),
        shaped_rewards=tuple(shaped),
        failure_count=failures,
        reward_mean=statistics.fmean(raw_rewards),
        reward_std=statistics.pstdev(raw_rewards),
        norm=gradient_norm,
        rms=gradient_rms,
    )


def save_es_checkpoint(state: ESState, path: str | Path) -> None:
    """Atomically save only the center, config and generation cursor."""

    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    torch.save(
        {
            "schema_version": 1,
            "config": asdict(state.config),
            "generation": state.generation,
            "center": state.center,
        },
        temporary,
    )
    os.replace(temporary, path)


def load_es_checkpoint(path: str | Path) -> ESState:
    path = Path(path).expanduser().resolve()
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("unsupported ES checkpoint schema")
    config_record = payload.get("config")
    if not isinstance(config_record, dict):
        raise ValueError("ES checkpoint has no config")
    return ESState(
        config=ESConfig(**config_record),
        center=payload["center"],
        generation=int(payload["generation"]),
    )
