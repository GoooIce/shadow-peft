#!/usr/bin/env python3
"""Freeze the confirmation gate and the minimum useful gain delta (ticket 13/18).

Ticket 17 delivered the frozen confirm pool (7,200 items = 900 episodes) and its
resolution: with an episode-level paired SD of ``sigma ~ 0.17-0.23`` the pool can
separate about **1.9-2.1 accuracy points** at 80% power on the binary axis
(1.1-1.7 points on the ``gold_prob`` continuous statistic).  Ticket 13 left the
delta tier, the non-inferiority tolerance and the wording boundaries to be
frozen here, reusing the registered three-wording mapping of
:mod:`shadow_lab.es_inference` (``classify_outcome``).

This script is pure standard library (no torch/numpy) so the numbers can be
re-derived on the control machine:

* the verdict rule is a verbatim port of ``classify_outcome`` plus the
  order-statistic percentile-bootstrap convention of ``_percentile_indices``
  (both quoted from ``src/shadow_lab/es_inference.py``, RES-003);
* the resolution table ``delta_min = (z_0.975 + z_0.8) * sigma / sqrt(n)``
  reproduces ticket 17's numbers on its sigma grid;
* the decision boundaries of the three wordings are computed analytically for
  each sigma (the interval half-width decides them, so no simulation is needed);
* a Monte Carlo pass measures the operating characteristics of the frozen rule
  (false support / excluded / insufficient rates at effect 0, delta/2, delta and
  -delta), with the interval taken from the frozen bootstrap on a spot check.

Usage:
    python confirm_gate.py --out <json> [--episodes 900] [--delta 0.02]
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from pathlib import Path

# --------------------------------------------------------------------------- #
# ports from src/shadow_lab/es_inference.py (RES-003) -- keep the wording exact
# --------------------------------------------------------------------------- #

GAIN_SUPPORTED = "gain-supported"
GAIN_EXCLUDED = "practical-gain-excluded"
INSUFFICIENT_EVIDENCE = "insufficient-evidence"

_WORDING = {
    GAIN_SUPPORTED: "可继续：配对效应区间下界高于 0 且点估计达到冻结的最小有用增益 δ。",
    GAIN_EXCLUDED: "已排除实用增益：区间上界低于冻结的最小有用增益 δ。",
    INSUFFICIENT_EVIDENCE: "证据不足：区间覆盖 0 或未达到冻结的最小有用增益 δ，只能给区间结论。",
}


def classify_outcome(*, mean: float, lower: float, upper: float, delta: float) -> dict:
    """Verbatim port of ``shadow_lab.es_inference.classify_outcome``."""
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


def percentile_indices(resamples: int, confidence: float) -> tuple[int, int]:
    """Port of ``es_inference._percentile_indices`` (frozen ENG-003 convention)."""
    tail = (1.0 - confidence) / 2.0
    lower_index = max(0, math.floor(tail * (resamples - 1)))
    upper_index = min(resamples - 1, math.ceil((1.0 - tail) * (resamples - 1)))
    return lower_index, upper_index


def bootstrap_interval(deltas, *, seed: int, resamples: int, confidence: float) -> tuple[float, float]:
    """Episode-level percentile bootstrap of the mean paired difference."""
    rng = random.Random(seed)
    count = len(deltas)
    boot = sorted(statistics.fmean(deltas[rng.randrange(count)] for _ in range(count))
                  for _ in range(resamples))
    lower_index, upper_index = percentile_indices(resamples, confidence)
    return boot[lower_index], boot[upper_index]


def sign_flip_p(deltas, *, seed: int, resamples: int) -> dict:
    """One-sided episode sign-flip null (Monte Carlo when 2**k exceeds budget)."""
    rng = random.Random(seed)
    observed = statistics.fmean(deltas)
    exact = 2 ** len(deltas) <= resamples
    patterns = []
    if exact:
        for mask in range(1, 2 ** len(deltas)):
            pat = tuple(1 if (mask >> bit) & 1 else -1 for bit in range(len(deltas)))
            patterns.append(pat)
    else:
        for _ in range(resamples):
            patterns.append(tuple(1 if rng.getrandbits(1) else -1 for _ in deltas))
    greater = sum(
        1 for pat in patterns
        if statistics.fmean(sign * value for sign, value in zip(pat, deltas)) >= observed
    )
    total = len(patterns)
    p_greater = greater / total if exact else (greater + 1) / (total + 1)
    return {"p_greater": p_greater, "method": "exact" if exact else "monte-carlo",
            "patterns": total, "observed": observed}


Z_975 = statistics.NormalDist().inv_cdf(0.975)
Z_80 = statistics.NormalDist().inv_cdf(0.80)


def delta_min(sigma: float, episodes: int, power: float = 0.8, alpha: float = 0.05) -> float:
    z_alpha = statistics.NormalDist().inv_cdf(1.0 - alpha / 2.0)
    z_power = statistics.NormalDist().inv_cdf(power)
    return (z_alpha + z_power) * sigma / math.sqrt(episodes)


def boundaries(sigma: float, episodes: int, delta: float, confidence: float) -> dict:
    """Where the three wordings switch, given a fixed interval half-width."""
    half_width = Z_975 * sigma / math.sqrt(episodes)
    # support: mean >= delta and mean - half_width > 0
    support_floor = max(delta, half_width)
    # excluded: mean + half_width < delta
    excluded_ceiling = delta - half_width
    return {
        "interval_half_width": half_width,
        "support_requires_mean_at_least": support_floor,
        "excluded_requires_mean_at_most": excluded_ceiling,
        "insufficient_band": [excluded_ceiling, support_floor],
        "band_is_empty": support_floor <= excluded_ceiling,
        "non_inferior_requires_lower_above": -delta,
    }


def simulate(sigma: float, episodes: int, delta: float, effect: float, *, sims: int,
             seed: int, confidence: float) -> dict:
    """Operating characteristics with the frozen rule and a normal interval."""
    rng = random.Random(seed)
    tally = {GAIN_SUPPORTED: 0, GAIN_EXCLUDED: 0, INSUFFICIENT_EVIDENCE: 0}
    non_inferior = 0
    for _ in range(sims):
        deltas = [effect + sigma * rng.gauss(0.0, 1.0) for _ in range(episodes)]
        mean = statistics.fmean(deltas)
        sd = statistics.stdev(deltas)
        half_width = Z_975 * sd / math.sqrt(episodes)
        verdict = classify_outcome(mean=mean, lower=mean - half_width,
                                   upper=mean + half_width, delta=delta)["verdict"]
        tally[verdict] += 1
        non_inferior += int(mean - half_width > -delta)
    return {
        "sigma": sigma, "effect": effect, "sims": sims,
        "support_rate": tally[GAIN_SUPPORTED] / sims,
        "excluded_rate": tally[GAIN_EXCLUDED] / sims,
        "insufficient_rate": tally[INSUFFICIENT_EVIDENCE] / sims,
        "non_inferior_rate": non_inferior / sims,
    }


def bootstrap_check(sigma: float, episodes: int, delta: float, effect: float, *,
                    sims: int, seed: int, resamples: int, confidence: float) -> dict:
    """Spot-check that the analytic interval reproduces the frozen bootstrap rule."""
    rng = random.Random(seed + 1)
    tally = {GAIN_SUPPORTED: 0, GAIN_EXCLUDED: 0, INSUFFICIENT_EVIDENCE: 0}
    for _ in range(sims):
        deltas = [effect + sigma * rng.gauss(0.0, 1.0) for _ in range(episodes)]
        mean = statistics.fmean(deltas)
        lower, upper = bootstrap_interval(deltas, seed=seed + 2, resamples=resamples,
                                          confidence=confidence)
        verdict = classify_outcome(mean=mean, lower=lower, upper=upper, delta=delta)["verdict"]
        tally[verdict] += 1
    return {
        "sigma": sigma, "effect": effect, "sims": sims, "resamples": resamples,
        "support_rate": tally[GAIN_SUPPORTED] / sims,
        "excluded_rate": tally[GAIN_EXCLUDED] / sims,
        "insufficient_rate": tally[INSUFFICIENT_EVIDENCE] / sims,
    }


def measure(arms_path: str, delta: float, confidence: float, resamples: int, seed: int,
            looks: int) -> dict:
    """Terminal wording from a ``pilot_search.py --confirm-only`` arms dump.

    The arms carry per-item rows for q0 and for the centre; the statistic is the
    episode-level paired difference of TF accuracy over the **same** 8-item
    episode clusters, exactly the frozen statistical unit.
    """
    blob = json.loads(Path(arms_path).expanduser().read_text())
    arms = blob["arms"]

    def episodes(rows):
        buckets: dict[str, list[float]] = {}
        for row in rows:
            buckets.setdefault(row["episode"], []).append(1.0 if row["tf_correct"] else 0.0)
        return {key: statistics.fmean(values) for key, values in buckets.items()}

    q0, centre = episodes(arms["q0"]["rows"]), episodes(arms["center"]["rows"])
    keys = sorted(set(q0) & set(centre))
    deltas = [centre[key] - q0[key] for key in keys]
    mean = statistics.fmean(deltas)
    lower, upper = bootstrap_interval(deltas, seed=seed, resamples=resamples,
                                      confidence=confidence)
    flip = sign_flip_p(deltas, seed=seed, resamples=resamples)
    verdict = classify_outcome(mean=mean, lower=lower, upper=upper, delta=delta)
    wording_supported = bool(verdict["verdict"] == GAIN_SUPPORTED
                             and flip["p_greater"] <= 0.05 / max(1, looks))
    return {
        "arms": arms_path,
        "centre_sha256": blob.get("center_sha256"),
        "episodes": len(deltas),
        "centre_acc": arms["center"]["fitness"]["acc"],
        "q0_acc": arms["q0"]["fitness"]["acc"],
        "mean": mean,
        "lower": lower,
        "upper": upper,
        "delta": delta,
        "confidence": confidence,
        "resamples": resamples,
        "sign_flip": flip,
        "looks": looks,
        "verdict": verdict["verdict"],
        "wording": verdict["wording"],
        "support_condition_with_multiplicity": {
            "p_le_alpha_over_looks": flip["p_greater"] <= 0.05 / max(1, looks),
            "wording_stands": wording_supported,
        },
        "non_inferior": bool(lower > -delta),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--measure", help="confirm arms dump from pilot_search --confirm-only")
    parser.add_argument("--episodes", type=int, default=900)
    parser.add_argument("--final-episodes", type=int, default=150)
    parser.add_argument("--delta", type=float, default=0.02)
    parser.add_argument("--delta-tiers", default="0.01,0.02,0.04")
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--sims", type=int, default=600)
    parser.add_argument("--check-sims", type=int, default=12)
    parser.add_argument("--check-resamples", type=int, default=1500)
    parser.add_argument("--looks", type=int, default=2,
                        help="how many confirm opens carry the decision")
    args = parser.parse_args()

    if args.measure:
        measured = measure(args.measure, args.delta, args.confidence, args.resamples,
                           args.seed, args.looks)
        destination = Path(args.out).expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(measured, indent=2, sort_keys=True) + "\n")
        print(json.dumps(measured, indent=2, sort_keys=True))
        return 0

    # sigma grid: the pilot's own near-neighbour noise floor and ticket 17's arms
    sigma_grid = [0.169, 0.204, 0.227, 0.269]
    tiers = [float(item) for item in args.delta_tiers.split(",")]
    delta = args.delta

    resolution = []
    for sigma in sigma_grid:
        resolution.append({
            "sigma": sigma,
            "delta_min_900": delta_min(sigma, args.episodes),
            "delta_min_900_pct_points": 100 * delta_min(sigma, args.episodes),
            "delta_min_final_150": delta_min(sigma, args.final_episodes),
            "delta_min_if_noise_upper_0.24": delta_min(0.24, args.episodes),
            "episodes_needed_delta_2pct": (Z_975 + Z_80) ** 2 * sigma ** 2 / (delta ** 2),
            "boundaries": boundaries(sigma, args.episodes, delta, args.confidence),
        })

    effects = [-delta, 0.0, delta / 2, delta, 2 * delta]
    operating = [simulate(sigma, args.episodes, delta, effect, sims=args.sims,
                          seed=args.seed, confidence=args.confidence)
                 for sigma in (0.169, 0.227) for effect in effects]
    checks = [bootstrap_check(sigma, args.episodes, delta, effect, sims=args.check_sims,
                              seed=args.seed, resamples=args.check_resamples,
                              confidence=args.confidence)
              for sigma in (0.169, 0.227) for effect in (0.0, delta)]

    report = {
        "protocol": "minicpm5-confirm-gate/v1",
        "frozen": {
            "confirm_episodes": args.episodes,
            "final_episodes": args.final_episodes,
            "delta_main": delta,
            "delta_tiers": tiers,
            "non_inferiority_tolerance": delta,
            "pair_gate_tau": delta,
            "confidence": args.confidence,
            "bootstrap_resamples": args.resamples,
            "bootstrap_unit": "episode (8 items), order-statistic percentile",
            "sign_flip": "episode sign symmetry, one-sided, seed %d" % args.seed,
            "verdict_rule": "shadow_lab.es_inference.classify_outcome (ported)",
            "decision_looks": args.looks,
            "multiplicity": (
                "the terminal wording is issued once; if confirm was opened G times, the "
                "'支持正收益' wording additionally requires the one-sided sign-flip "
                "p <= 0.05/G over the looks actually taken"
            ),
        },
        "resolution": resolution,
        "operating_characteristics": operating,
        "bootstrap_check": checks,
        "sign_flip_example": sign_flip_p(
            [0.02 + 0.227 * random.Random(args.seed).gauss(0, 1) for _ in range(args.episodes)],
            seed=args.seed, resamples=2000),
        "narrative": {
            "main_axis": "teacher-forced TF accuracy, episode-level paired difference vs q0",
            "secondary_axis": "gold_prob (exp of the mean gold logprob), same scorer",
            "excluded": "已排除实用增益 requires the whole interval below δ",
            "supported": "支持正收益 requires lower > 0 and mean >= δ",
            "insufficient": "everything else is reported as 证据不足 with the interval only",
        },
    }
    Path(args.out).expanduser().write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({
        "delta": delta, "episodes": args.episodes,
        "sigma": {row["sigma"]: round(100 * row["delta_min_900"], 2) for row in resolution},
        "boundaries_0.227": resolution[2]["boundaries"],
        "operating": [{"sigma": row["sigma"], "effect": row["effect"],
                       "support": round(row["support_rate"], 3),
                       "excluded": round(row["excluded_rate"], 3),
                       "insufficient": round(row["insufficient_rate"], 3),
                       "non_inferior": round(row["non_inferior_rate"], 3)}
                      for row in operating],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
