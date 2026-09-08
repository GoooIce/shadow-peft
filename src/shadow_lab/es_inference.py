"""Pre-registered statistical inference and power machinery for ES campaigns (RES-003).

Every decision layer that EXP-010/011/013 consume is calibrated here before any new
outcome data exists: episode-level sign-flip nulls for paired effects, prompt-cluster
bootstrap intervals that carry generation seeds whole, episode-hierarchy nulls for
mean pairwise gradient cosines (28/276 cosine pairs share episodes and are not
independent), Monte Carlo false-positive/power rates for the registered joint
selection/confirmation gate, budget-exclusion power for the frozen minimum useful
gain delta, and the three-wording verdict mapping.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import random
from dataclasses import asdict, dataclass, fields

import torch

FLIP_NULL_MODEL = (
    "episode-exchangeable sign symmetry: under H0 each episode's paired delta is "
    "equally likely to be flipped, which respects the shared episode/noise structure "
    "instead of treating episodes, coordinates or cosine pairs as independent"
)

# Review F02 worked example: under an independent, fair, tie-free sign model the
# probability of at least 6 of 8 batch wins is 37/256 ~ 14.45%; a win-count gate is
# not a significance proof.
SIGN_MODEL_PROBABILITY_6_OF_8 = 37 / 256

GAIN_SUPPORTED = "gain-supported"
GAIN_EXCLUDED = "practical-gain-excluded"
INSUFFICIENT_EVIDENCE = "insufficient-evidence"

_WORDING = {
    GAIN_SUPPORTED: "可继续：配对效应区间下界高于 0 且点估计达到冻结的最小有用增益 δ。",
    GAIN_EXCLUDED: "已排除实用增益：区间上界低于冻结的最小有用增益 δ。",
    INSUFFICIENT_EVIDENCE: "证据不足：区间覆盖 0 或未达到冻结的最小有用增益 δ，只能给区间结论。",
}


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _validate_finite(values, label: str) -> list[float]:
    numbers = [float(value) for value in values]
    if len(numbers) < 2 or not all(math.isfinite(number) for number in numbers):
        raise ValueError(f"{label} requires at least two finite values")
    return numbers


def _flip_patterns(count: int, limit: int, seed: int) -> list[tuple[int, ...]]:
    """Exact +/-1 patterns when they fit the budget, otherwise seeded Monte Carlo."""

    total = 2**count
    if total <= limit:
        return [
            tuple(1 if (mask >> index) & 1 else -1 for index in range(count))
            for mask in range(total)
        ]
    rng = random.Random(seed)
    return [tuple(1 if rng.getrandbits(1) else -1 for _ in range(count)) for _ in range(limit)]


def _percentile_indices(resamples: int, confidence: float) -> tuple[int, int]:
    """Order-statistic indices matching the frozen ENG-003 bootstrap convention."""

    tail = (1.0 - confidence) / 2.0
    lower_index = max(0, math.floor(tail * (resamples - 1)))
    upper_index = min(resamples - 1, math.ceil((1.0 - tail) * (resamples - 1)))
    return lower_index, upper_index


def _flip_null_p_values(
    stats: list[float],
    observed: float,
    *,
    exact: bool,
) -> tuple[float, float]:
    """Greater- and two-sided p-values of an observed statistic under flip nulls."""

    if exact:
        p_greater = sum(stat >= observed for stat in stats) / len(stats)
        p_two_sided = sum(abs(stat) >= abs(observed) for stat in stats) / len(stats)
    else:
        p_greater = (sum(stat >= observed for stat in stats) + 1) / (len(stats) + 1)
        p_two_sided = (sum(abs(stat) >= abs(observed) for stat in stats) + 1) / (len(stats) + 1)
    return p_greater, p_two_sided


def _tally_verdicts(means, lower, upper, *, delta: float) -> dict[str, int]:
    """Classify paired studies with the frozen three-wording rule and count verdicts."""

    tally = {GAIN_SUPPORTED: 0, GAIN_EXCLUDED: 0, INSUFFICIENT_EVIDENCE: 0}
    for mean, low, high in zip(means, lower, upper, strict=True):
        verdict = classify_outcome(
            mean=float(mean), lower=float(low), upper=float(high), delta=delta
        )["verdict"]
        tally[verdict] += 1
    return tally


def episode_pair_cosines(directions: torch.Tensor) -> torch.Tensor:
    """Per-simulation pairwise cosines from a ``(n_sims, episodes, dimension)`` batch."""

    if directions.ndim != 3 or directions.shape[1] < 2 or directions.shape[2] < 2:
        raise ValueError("directions must have shape (n_sims, episodes >= 2, dimension >= 2)")
    if not torch.isfinite(directions).all():
        raise ValueError("direction vectors must be finite")
    normalized = directions / torch.linalg.vector_norm(directions, dim=2, keepdim=True)
    grams = torch.bmm(normalized, normalized.transpose(1, 2))
    triu = torch.triu(
        torch.ones(normalized.shape[1], normalized.shape[1], dtype=torch.bool), diagonal=1
    )
    return grams[:, triu]


def episode_sign_flip_test(
    deltas,
    *,
    seed: int = 20260904,
    resamples: int = 10_000,
) -> dict:
    """One-sided sign-flip null for the mean of per-episode paired deltas.

    With k episodes there are 2**k sign patterns; when that fits in ``resamples`` the
    null distribution is enumerated exactly, otherwise it is sampled with a seeded
    generator. The paired deltas must already be cluster-level (one value per
    episode/prompt block), so replays and extra generation seeds never inflate k.
    """

    values = _validate_finite(deltas, "episode deltas")
    patterns = _flip_patterns(len(values), resamples, seed)
    observed = math.fsum(values) / len(values)
    flip_means = [
        math.fsum(sign * value for sign, value in zip(pattern, values, strict=True))
        / len(values)
        for pattern in patterns
    ]
    method = "exact" if len(patterns) == 2 ** len(values) else "monte-carlo"
    p_greater, p_two_sided = _flip_null_p_values(flip_means, observed, exact=method == "exact")
    return {
        "mean": observed,
        "episodes": len(values),
        "p_greater": p_greater,
        "p_two_sided": p_two_sided,
        "method": method,
        "pattern_count": len(patterns),
        "resamples": resamples,
        "seed": seed,
        "null_model": FLIP_NULL_MODEL,
    }


def cluster_bootstrap_interval(
    cluster_deltas,
    *,
    seed: int = 20260904,
    resamples: int = 10_000,
    confidence: float = 0.95,
) -> dict:
    """Paired bootstrap that resamples prompt clusters and carries their seeds whole.

    ``cluster_deltas`` maps a cluster identity (the prompt) to its paired
    differences (one per generation seed, replay or blind repeat). Replays and
    repeated blind judgments share the cluster, so they widen the cluster mean's
    stability without increasing the effective sample size.
    """

    if not cluster_deltas:
        raise ValueError("cluster bootstrap requires at least one cluster")
    ordered = sorted(cluster_deltas)
    cluster_means = [_validate_finite(cluster_deltas[key], f"cluster {key}") for key in ordered]
    means = [math.fsum(values) / len(values) for values in cluster_means]
    observed = math.fsum(means) / len(means)
    rng = random.Random(seed)
    count = len(means)
    bootstrapped = sorted(
        math.fsum(means[rng.randrange(count)] for _ in range(count)) for _ in range(resamples)
    )
    lower_index, upper_index = _percentile_indices(resamples, confidence)
    return {
        "mean": observed,
        "lower": bootstrapped[lower_index],
        "upper": bootstrapped[upper_index],
        "confidence": confidence,
        "cluster_count": count,
        "observation_count": sum(len(values) for values in cluster_means),
        "resamples": resamples,
        "seed": seed,
    }


def cosine_episode_flip_null(
    vectors,
    *,
    seed: int = 20260904,
    resamples: int = 10_000,
) -> dict:
    """Episode-hierarchy null for the mean pairwise cosine of per-episode gradients.

    The C(k,2) pairwise cosines all share episodes, so treating them as independent
    samples is uncontrolled and can be anticonservative. Flipping a whole episode's
    sign flips every cosine it participates in (the pair (i, j) picks up s_i * s_j),
    which yields the dependence-corrected null distribution of the mean pairwise
    cosine without ever resampling pairs as if they were independent.
    """

    matrix = torch.stack(
        [
            vector.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
            for vector in vectors
        ]
    )
    episodes, dimension = matrix.shape
    if episodes < 2 or dimension < 2:
        raise ValueError("cosine hierarchy needs at least two gradients of dimension >= 2")
    if not torch.isfinite(matrix).all():
        raise ValueError("gradient vectors must be finite")
    norms = torch.linalg.vector_norm(matrix, dim=1)
    if (norms == 0).any():
        raise ValueError("gradient cosine is undefined for a zero vector")
    matrix = matrix / norms[:, None]
    gram = matrix @ matrix.T
    pairs = list(itertools.combinations(range(episodes), 2))
    pair_cosines = [float(gram[left, right]) for left, right in pairs]
    observed = math.fsum(pair_cosines) / len(pairs)
    pair_sd = math.sqrt(
        math.fsum((value - observed) ** 2 for value in pair_cosines) / len(pair_cosines)
    )
    naive_sd = pair_sd / math.sqrt(len(pairs))
    naive_z = observed / naive_sd if naive_sd > 0 else 0.0

    patterns = _flip_patterns(episodes, resamples, seed)
    flip_stats = []
    for pattern in patterns:
        stat = math.fsum(
            pattern[left] * pattern[right] * value
            for (left, right), value in zip(pairs, pair_cosines, strict=True)
        ) / len(pairs)
        flip_stats.append(stat)
    method = "exact" if len(patterns) == 2**episodes else "monte-carlo"
    p_greater, p_two_sided = _flip_null_p_values(flip_stats, observed, exact=method == "exact")
    flip_mean = math.fsum(flip_stats) / len(flip_stats)
    flip_sd = math.sqrt(
        math.fsum((stat - flip_mean) ** 2 for stat in flip_stats) / len(flip_stats)
    )
    precision_ratio = (
        naive_sd / flip_sd if flip_sd > 0 else (1.0 if naive_sd == 0 else math.inf)
    )
    return {
        "mean_cosine": observed,
        "pair_count": len(pairs),
        "naive_sd": naive_sd,
        "naive_z": naive_z,
        "flip_sd": flip_sd,
        "flip_p_greater": p_greater,
        "flip_p_two_sided": p_two_sided,
        "precision_ratio": precision_ratio,
        "method": method,
        "pattern_count": len(patterns),
        "episodes": episodes,
        "resamples": resamples,
        "seed": seed,
        "null_model": FLIP_NULL_MODEL,
    }


def _normal_draws(generator: torch.Generator, rows: int, columns: int) -> torch.Tensor:
    return torch.randn(rows, columns, generator=generator, dtype=torch.float32)


def joint_gate_power(
    *,
    episodes: int,
    min_non_degrading: int,
    effects,
    n_sims: int = 10_000,
    seed: int = 20260904,
    batch_gate_pass_probability: float = 1.0,
) -> list[dict]:
    """False-positive/power rates of the registered joint selection/confirmation gate.

    Each simulated episode contributes one paired reward delta ~ Normal(effect, 1)
    in units of the episode-level paired sd; the registered gate requires
    mean(delta) > 0, at least ``min_non_degrading`` episodes with delta >= 0, and
    every episode's hard quality gate to pass.
    """

    if not 1 <= min_non_degrading <= episodes:
        raise ValueError("non-degrading threshold does not fit the episode budget")
    if not 0.0 < batch_gate_pass_probability <= 1.0:
        raise ValueError("batch gate pass probability must be in (0, 1]")
    rows = []
    for effect in effects:
        generator = torch.Generator()
        generator.manual_seed(seed)
        deltas = _normal_draws(generator, n_sims, episodes) + float(effect)
        gate_mask = torch.rand(n_sims, episodes, generator=generator) < batch_gate_pass_probability
        mean_ok = deltas.mean(dim=1) > 0
        non_degrading = (deltas >= 0).sum(dim=1) >= min_non_degrading
        gates_ok = gate_mask.all(dim=1)
        passed = (mean_ok & non_degrading & gates_ok).sum().item()
        rows.append(
            {
                "effect": float(effect),
                "episodes": episodes,
                "min_non_degrading": min_non_degrading,
                "n_sims": n_sims,
                "seed": seed,
                "batch_gate_pass_probability": batch_gate_pass_probability,
                "pass_rate": passed / n_sims,
                "mean_only_rate": float(mean_ok.sum().item()) / n_sims,
                "non_degrading_only_rate": float(non_degrading.sum().item()) / n_sims,
            }
        )
    return rows


def _simulated_gradient_directions(
    generator: torch.Generator,
    *,
    episodes: int,
    rho: float,
    dimension: int,
    n_sims: int,
) -> torch.Tensor:
    """Directions shaped as sqrt(rho) * true direction + sqrt(1 - rho) * noise."""

    true_direction = _normal_draws(generator, 1, dimension)
    true_direction = true_direction / torch.linalg.vector_norm(true_direction)
    noise = _normal_draws(generator, n_sims * episodes, dimension)
    noise = noise / torch.linalg.vector_norm(noise, dim=1, keepdim=True)
    directions = math.sqrt(rho) * true_direction + math.sqrt(1.0 - rho) * noise
    return directions.view(n_sims, episodes, dimension)


def _checkpoint_pass(cosines: torch.Tensor, mean_threshold, positive_fraction_threshold):
    mean_cosine = cosines.mean(dim=1)
    positive_fraction = (cosines > 0).float().mean(dim=1)
    return (
        mean_cosine >= mean_threshold,
        positive_fraction >= positive_fraction_threshold,
        mean_cosine,
        positive_fraction,
    )


def consistency_checkpoint_power(
    *,
    episode_counts,
    rho_grid,
    dimension: int = 40,
    n_sims: int = 10_000,
    seed: int = 20260904,
    mean_threshold: float = 0.10,
    positive_fraction_threshold: float = 0.70,
) -> list[dict]:
    """False-positive/power of the registered k-episode cosine consistency checkpoint.

    Each episode's gradient direction is modelled as sqrt(rho) * true direction plus
    sqrt(1 - rho) independent noise, so E[cosine] = rho_true. The checkpoint passes
    when the mean pairwise cosine and the positive fraction both clear their
    registered thresholds; at rho_true = 0 the pass rate is the rule's false-positive
    rate, and at small positive rho it quantifies how often a real but weak signal
    is stopped as "no-gradient-signal". This calibrates each checkpoint size in
    isolation; :func:`sequential_checkpoint_power` calibrates the whole registered
    two-checkpoint stop rule as one unit.
    """

    if dimension < 2:
        raise ValueError("checkpoint simulation needs dimension >= 2")
    rows = []
    for episodes in episode_counts:
        if episodes < 2:
            raise ValueError("checkpoint simulation needs at least two episodes")
        for rho in rho_grid:
            if not 0.0 <= rho < 1.0:
                raise ValueError("rho_true must be in [0, 1)")
            generator = torch.Generator()
            generator.manual_seed(seed)
            passed = 0
            mean_cosine_total = 0.0
            positive_fraction_total = 0.0
            chunk = max(1, min(n_sims, 4_000_000 // max(1, episodes * dimension)))
            completed = 0
            while completed < n_sims:
                size = min(chunk, n_sims - completed)
                directions = _simulated_gradient_directions(
                    generator, episodes=episodes, rho=rho, dimension=dimension, n_sims=size
                )
                cosines = episode_pair_cosines(directions)
                mean_ok, fraction_ok, mean_cosine, positive_fraction = _checkpoint_pass(
                    cosines, mean_threshold, positive_fraction_threshold
                )
                passed += (mean_ok & fraction_ok).sum().item()
                mean_cosine_total += float(mean_cosine.sum().item())
                positive_fraction_total += float(positive_fraction.sum().item())
                completed += size
            rows.append(
                {
                    "episodes": episodes,
                    "rho_true": float(rho),
                    "dimension": dimension,
                    "n_sims": n_sims,
                    "seed": seed,
                    "mean_threshold": mean_threshold,
                    "positive_fraction_threshold": positive_fraction_threshold,
                    "pass_rate": passed / n_sims,
                    "mean_cosine_mean": mean_cosine_total / n_sims,
                    "positive_fraction_mean": positive_fraction_total / n_sims,
                }
            )
    return rows


def sequential_checkpoint_power(
    *,
    gradient_episodes: int,
    checkpoints,
    rho_grid,
    dimension: int = 40,
    n_sims: int = 10_000,
    seed: int = 20260904,
) -> list[dict]:
    """False-positive/power of the WHOLE registered two-checkpoint stop rule.

    The campaign is simulated over ``gradient_episodes`` batches; at each registered
    checkpoint (episode count, mean cosine floor, positive-fraction floor) the rule
    stops the campaign as "no-gradient-signal" on the first failure and otherwise
    continues. Rows report per-rho stop and survival rates for the complete
    sequential procedure, which is what F02 asked to see calibrated as one unit
    rather than as isolated checkpoint sizes.
    """

    if dimension < 2:
        raise ValueError("checkpoint simulation needs dimension >= 2")
    if not checkpoints or sorted(count for count, _, _ in checkpoints) != [
        count for count, _, _ in checkpoints
    ]:
        raise ValueError("checkpoints need strictly increasing episode counts")
    if not min(count for count, _, _ in checkpoints) >= 2 or gradient_episodes < max(
        count for count, _, _ in checkpoints
    ):
        raise ValueError("checkpoints must fit inside the gradient episode budget")
    rows = []
    for rho in rho_grid:
        if not 0.0 <= rho < 1.0:
            raise ValueError("rho_true must be in [0, 1)")
        generator = torch.Generator()
        generator.manual_seed(seed)
        stop_at: dict[int, int] = {count: 0 for count, _, _ in checkpoints}
        survived = 0
        chunk = max(1, min(n_sims, 4_000_000 // max(1, gradient_episodes * dimension)))
        completed = 0
        while completed < n_sims:
            size = min(chunk, n_sims - completed)
            directions = _simulated_gradient_directions(
                generator,
                episodes=gradient_episodes,
                rho=rho,
                dimension=dimension,
                n_sims=size,
            )
            alive = torch.ones(size, dtype=torch.bool)
            for count, mean_threshold, fraction_threshold in checkpoints:
                cosines = episode_pair_cosines(directions[:, :count, :])
                mean_ok, fraction_ok, _, _ = _checkpoint_pass(
                    cosines, mean_threshold, fraction_threshold
                )
                check_ok = mean_ok & fraction_ok
                stopped = alive & ~check_ok
                stop_at[count] += int(stopped.sum().item())
                alive = alive & check_ok
            survived += int(alive.sum().item())
            completed += size
        rows.append(
            {
                "gradient_episodes": gradient_episodes,
                "checkpoints": [list(checkpoint) for checkpoint in checkpoints],
                "rho_true": float(rho),
                "dimension": dimension,
                "n_sims": n_sims,
                "seed": seed,
                "stop_rate_at": {str(count): rate / n_sims for count, rate in stop_at.items()},
                "survival_rate": survived / n_sims,
                "stopped_rate": 1.0 - survived / n_sims,
            }
        )
    return rows


def _bootstrap_ci_quantiles(
    deltas: torch.Tensor,
    *,
    bootstrap_resamples: int,
    confidence: float,
    generator: torch.Generator,
    chunk_sims: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-simulation percentile bootstrap CI, vectorized over chunked simulations.

    Interval indices follow the frozen order-statistic convention of
    ``image_eval.paired_bootstrap_mean_difference`` so the simulated decision rule
    is exactly the registered interval algorithm.
    """

    n_sims, episodes = deltas.shape
    indices = torch.randint(
        0, episodes, (bootstrap_resamples, episodes), generator=generator
    )
    lower_index, upper_index = _percentile_indices(bootstrap_resamples, confidence)
    lower = torch.empty(n_sims, dtype=torch.float32)
    upper = torch.empty(n_sims, dtype=torch.float32)
    for start in range(0, n_sims, chunk_sims):
        block = deltas[start : start + chunk_sims]
        means = block[:, indices].mean(dim=2)
        ordered = means.sort(dim=1).values
        lower[start : start + block.shape[0]] = ordered[:, lower_index]
        upper[start : start + block.shape[0]] = ordered[:, upper_index]
    return lower, upper


def delta_support_table(
    *,
    episode_counts,
    sigma_over_delta,
    effects,
    n_sims: int = 10_000,
    bootstrap_resamples: int = 10_000,
    seed: int = 20260904,
    confidence: float = 0.95,
    delta: float = 1.0,
) -> list[dict]:
    """Budget-exclusion power: verdict rates of the three-wording rule per scenario.

    Everything is simulated in units of the frozen minimum useful gain delta: each
    episode contributes Normal(effect * delta, sigma_over_delta * delta) and each
    simulated study is classified with :func:`classify_outcome` on its bootstrap CI.
    The gain-excluded rate under effect=0 answers how often the budget can honestly
    write "已排除实用增益"; at effect=delta the same rate is the type-I error of that
    claim, measured at 5.6% (k=8) and 4.2% (k=16) in the registered grid — the
    order-statistic percentile bootstrap at these tiny episode counts is mildly
    anticonservative versus the 2.5% one-sided ideal; recorded, not re-tuned.
    """

    if not 0.0 < confidence < 1.0 or delta <= 0:
        raise ValueError("invalid confidence or delta scale")
    rows = []
    for episodes in episode_counts:
        for ratio in sigma_over_delta:
            for effect in effects:
                generator = torch.Generator()
                generator.manual_seed(seed)
                deltas = (
                    float(effect) * delta
                    + float(ratio) * delta * _normal_draws(generator, n_sims, episodes)
                )
                lower, upper = _bootstrap_ci_quantiles(
                    deltas,
                    bootstrap_resamples=bootstrap_resamples,
                    confidence=confidence,
                    generator=generator,
                )
                means = deltas.mean(dim=1)
                degenerate = int((~torch.isfinite(deltas).all(dim=1)).sum().item())
                tally = _tally_verdicts(means.tolist(), lower.tolist(), upper.tolist(), delta=delta)
                rows.append(
                    {
                        "episodes": episodes,
                        "sigma_over_delta": float(ratio),
                        "effect": float(effect),
                        "delta": delta,
                        "n_sims": n_sims,
                        "bootstrap_resamples": bootstrap_resamples,
                        "seed": seed,
                        "confidence": confidence,
                        "gain_supported_rate": tally[GAIN_SUPPORTED] / n_sims,
                        "gain_excluded_rate": tally[GAIN_EXCLUDED] / n_sims,
                        "insufficient_rate": tally[INSUFFICIENT_EVIDENCE] / n_sims,
                        "mean_ci_lower": float(lower.mean().item()),
                        "mean_ci_upper": float(upper.mean().item()),
                        "degenerate_sim_count": degenerate,
                    }
                )
    return rows


def budget_verdict(
    support_rows,
    *,
    episodes: int,
    sigma_over_delta: float,
    min_power_support: float = 0.8,
    min_power_interval: float = 0.5,
) -> str:
    """Frozen mapping from the exclusion-power table to the pre-run budget wording."""

    if not 0.0 < min_power_interval <= min_power_support <= 1.0:
        raise ValueError("power thresholds must satisfy 0 < interval <= support <= 1")
    for row in support_rows:
        if row["episodes"] == episodes and row["sigma_over_delta"] == sigma_over_delta:
            power = row["gain_excluded_rate"]
            if power >= min_power_support:
                return "budget-supports-delta"
            if power >= min_power_interval:
                return "interval-only"
            return "redesign-required"
    raise KeyError(f"no support row for episodes={episodes}, sigma_over_delta={sigma_over_delta}")


def preference_power(
    *,
    prompts: int,
    judgments_per_prompt: int,
    q_grid,
    decisive_rate: float,
    n_sims: int = 10_000,
    bootstrap_resamples: int = 10_000,
    seed: int = 20260904,
    confidence: float = 0.95,
    delta: float = 0.10,
) -> list[dict]:
    """Human-preference power for a fixed prompt pool (EXP-011 style).

    Every judgment is decisive with probability ``decisive_rate``; a decisive
    judgment prefers the ES arm with probability ``q``. The per-prompt net preference
    (fraction preferring E minus fraction preferring H, ties contributing zero) is
    averaged over prompts and wrapped in a prompt-cluster bootstrap; verdicts follow
    the frozen three-wording rule at ``delta``. Ties and repeats never add prompts.
    """

    if prompts < 2 or judgments_per_prompt < 1:
        raise ValueError("preference power needs at least two prompts and one judgment")
    if not 0.0 < decisive_rate <= 1.0:
        raise ValueError("decisive rate must be in (0, 1]")
    rows = []
    for q in q_grid:
        if not 0.0 <= q <= 1.0:
            raise ValueError("q must be in [0, 1]")
        generator = torch.Generator()
        generator.manual_seed(seed)
        decisive = (
            torch.rand(n_sims, prompts, judgments_per_prompt, generator=generator)
            < decisive_rate
        )
        prefers_e = (
            torch.rand(n_sims, prompts, judgments_per_prompt, generator=generator) < q
        )
        e_fraction = (decisive & prefers_e).float().mean(dim=2)
        h_fraction = (decisive & ~prefers_e).float().mean(dim=2)
        prompt_nets = e_fraction - h_fraction
        means = prompt_nets.mean(dim=1)
        lower, upper = _bootstrap_ci_quantiles(
            prompt_nets,
            bootstrap_resamples=bootstrap_resamples,
            confidence=confidence,
            generator=generator,
        )
        tally = _tally_verdicts(means.tolist(), lower.tolist(), upper.tolist(), delta=delta)
        rows.append(
            {
                "prompts": prompts,
                "judgments_per_prompt": judgments_per_prompt,
                "decisive_rate": decisive_rate,
                "q": float(q),
                "delta": delta,
                "n_sims": n_sims,
                "bootstrap_resamples": bootstrap_resamples,
                "seed": seed,
                "gain_supported_rate": tally[GAIN_SUPPORTED] / n_sims,
                "gain_excluded_rate": tally[GAIN_EXCLUDED] / n_sims,
                "insufficient_rate": tally[INSUFFICIENT_EVIDENCE] / n_sims,
                "mean_net_preference": float(means.mean().item()),
            }
        )
    return rows


def classify_outcome(*, mean: float, lower: float, upper: float, delta: float) -> dict:
    """Frozen three-wording mapping from a paired effect and its interval.

    "已排除实用增益" fires whenever the interval's upper bound sits below the frozen
    minimum useful gain delta (even when the point estimate is positive but small);
    "可继续" requires both a positive lower bound and a point estimate at least as
    large as delta; everything else is reported as "证据不足" with the interval only.
    """

    values = (mean, lower, upper)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("effect and interval bounds must be finite")
    if not lower <= upper:
        raise ValueError("interval bounds must satisfy lower <= upper")
    if delta <= 0:
        raise ValueError("delta must be positive")
    positive_point_estimate = mean > 0
    if upper < delta:
        verdict = GAIN_EXCLUDED
    elif lower > 0 and mean >= delta:
        verdict = GAIN_SUPPORTED
    else:
        verdict = INSUFFICIENT_EVIDENCE
    return {
        "verdict": verdict,
        "wording": _WORDING[verdict],
        "delta": delta,
        "positive_point_estimate": positive_point_estimate,
        "mean": mean,
        "lower": lower,
        "upper": upper,
    }


def fixed_sequence_family(tests, *, alpha: float) -> list[dict]:
    """Pre-ordered family tested at full alpha until the first failure (FWER control)."""

    results = []
    stopped = False
    for index, test in enumerate(tests):
        row = dict(test)
        if stopped:
            row.update(verdict="not-tested", reason="fixed-sequence-stopped")
        elif row["p_value"] < alpha:
            row.update(verdict="supported", reason=f"p < alpha at position {index}")
        else:
            row.update(verdict="not-supported", reason=f"p >= alpha at position {index}")
            stopped = True
        results.append(row)
    return results


def holm_family(tests, *, alpha: float) -> list[dict]:
    """Holm step-down adjustment for exploratory families (e.g. the low-dim arm)."""

    ordered = sorted(tests, key=lambda test: test["p_value"])
    count = len(ordered)
    running_max = 0.0
    rejected = True
    results = []
    for index, test in enumerate(ordered):
        row = dict(test)
        adjusted = max(running_max, row["p_value"] * (count - index))
        running_max = adjusted
        row["p_holm"] = min(1.0, adjusted)
        if rejected and row["p_holm"] < alpha:
            row["verdict"] = "supported"
        else:
            rejected = False
            row["verdict"] = "not-supported"
        results.append(row)
    return results


@dataclass(frozen=True)
class StatisticalPlan:
    """Frozen RES-003 statistical plan; every field is part of the freeze identity."""

    plan_id: str = "res003"
    created_date: str = "2026-09-08"
    primary_endpoint: str = "paired_mean_reward_z_e_minus_h_unseen_prompts"
    mechanism_arms: tuple[str, ...] = ("e_vs_p", "p_vs_h")
    delta_primary: float = 0.25
    axis_noninferiority_z: float = 0.10
    alpha_selection: float = 0.05
    alpha_confirmation: float = 0.01
    confidence: float = 0.95
    bootstrap_seed: int = 20260904
    bootstrap_resamples: int = 10_000
    flip_resamples: int = 10_000
    power_sims_per_cell: int = 10_000
    selection_episodes: int = 8
    confirmation_episodes: int = 8
    min_non_degrading_batches: int = 6
    gradient_episodes: int = 24
    gate_episodes: tuple[int, ...] = (8, 16)
    gate_effect_grid: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0)
    sigma_over_delta_grid: tuple[float, ...] = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0)
    support_effect_grid: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0)
    checkpoint_episode_counts: tuple[int, ...] = (8, 16, 24)
    checkpoint_rho_grid: tuple[float, ...] = (0.0, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30)
    checkpoint_dimension: int = 40
    checkpoint_mean_threshold: float = 0.10
    checkpoint_positive_fraction_threshold: float = 0.70
    min_power_support: float = 0.8
    min_power_interval: float = 0.5
    preference_prompts: int = 30
    preference_judgments_per_prompt: int = 6
    preference_q_grid: tuple[float, ...] = (0.5, 0.6, 0.7, 0.8, 0.9)
    preference_decisive_rate_grid: tuple[float, ...] = (0.6, 0.8)
    preference_delta: float = 0.10
    multiplicity: str = "fixed-sequence-by-noise-group-seed-order"
    exploratory_family_method: str = "holm"
    analysis_plan: str = (
        "one selection analysis after all selection episodes, one confirmation "
        "analysis after all confirmation episodes; registered consistency "
        "checkpoints stop exploration only and never add samples"
    )

    def __post_init__(self) -> None:
        tuple_fields = {
            "mechanism_arms",
            "gate_episodes",
            "gate_effect_grid",
            "sigma_over_delta_grid",
            "support_effect_grid",
            "checkpoint_episode_counts",
            "checkpoint_rho_grid",
            "preference_q_grid",
            "preference_decisive_rate_grid",
        }
        for name in tuple_fields:
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if self.delta_primary <= 0 or not math.isfinite(self.delta_primary):
            raise ValueError("delta_primary must be positive and finite")
        if self.axis_noninferiority_z <= 0:
            raise ValueError("axis non-inferiority margin must be positive")
        if not (0.0 < self.alpha_selection < 1.0 and 0.0 < self.alpha_confirmation < 1.0):
            raise ValueError("alpha levels must be in (0, 1)")
        if self.alpha_selection == self.alpha_confirmation:
            raise ValueError("the two pre-specified alpha levels must differ")
        if not 0.0 < self.min_power_interval <= self.min_power_support <= 1.0:
            raise ValueError("power thresholds must satisfy 0 < interval <= support <= 1")
        if self.bootstrap_resamples < 1 or self.flip_resamples < 1 or self.power_sims_per_cell < 1:
            raise ValueError("resample/simulation budgets must be positive")
        if self.selection_episodes < 2 or self.confirmation_episodes < 2:
            raise ValueError("selection/confirmation episode budgets must cover >= 2 episodes")
        if not 1 <= self.min_non_degrading_batches <= min(
            self.selection_episodes, self.confirmation_episodes
        ):
            raise ValueError("non-degrading rule does not fit the episode budgets")
        if self.gradient_episodes < max(self.checkpoint_episode_counts):
            raise ValueError("gradient episode budget must cover every checkpoint size")

    def canonical_json(self) -> str:
        return canonical_json(asdict(self))

    def content_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    @classmethod
    def default(cls, **overrides) -> StatisticalPlan:
        return cls(**overrides)

    @classmethod
    def from_json(cls, payload: str) -> StatisticalPlan:
        record = json.loads(payload)
        known = {field.name for field in fields(cls)}
        unknown = set(record) - known
        if unknown:
            raise ValueError(f"unknown statistical plan fields: {sorted(unknown)}")
        return cls(**record)
