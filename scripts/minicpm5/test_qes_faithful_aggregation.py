"""QES-original (ungated) aggregation: unit tests for the 2026-09-10 pivot.

Preregistration: docs/experiments/minicpm5-qes-faithful-prereg-20260910.md.

Covers the preregistered predictions:
  P1  every mirror pair participates -- ``qualified_pairs == len(pairs)`` and
      ``dropped_pairs == []`` -- including pairs whose sides sit far below the
      centre (the old per-side non-inferiority gate dropped exactly those);
  P2  the gradient is non-empty whenever at least one pair exists, so the
      ``moved=0`` starvation halt cannot be caused by an empty aggregation;
  symmetry -- each pair contributes ``+reward`` on the plus side and
      ``-reward`` on the minus side of its realised displacements, and the
      total weight matches ``2 * sum(|reward|)``;
  protocol bump -- ``PROTOCOL`` is the ungated v2 string (centre identity
      hashes must not mix protocols).

Usage:
  cd $(git rev-parse --show-toplevel)/scripts/minicpm5 && \
      python test_qes_faithful_aggregation.py
"""
from __future__ import annotations

import inspect
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pilot_lib as pl  # noqa: E402


def check(name, cond, detail=""):
    if not cond:
        raise AssertionError(f"{name}: {detail}")
    print(f"  ok  {name}")


def make_pair(index: int, plus_fitness: float, minus_fitness: float,
              plus_delta: float = 0.0, minus_delta: float = 0.0) -> dict:
    return {
        "pair": index,
        "plus": {"index": 2 * index, "fitness": plus_fitness,
                 "delta_center": plus_delta},
        "minus": {"index": 2 * index + 1, "fitness": minus_fitness,
                  "delta_center": minus_delta},
    }


def test_all_pairs_aggregate_regardless_of_delta():
    """P1: the gate is gone; catastrophically negative sides still vote."""
    pairs = [
        make_pair(0, 0.50, 0.48, plus_delta=0.02, minus_delta=0.00),
        make_pair(1, 0.47, 0.46, plus_delta=-0.036, minus_delta=-0.046),
        make_pair(2, 0.52, 0.44, plus_delta=-0.05, minus_delta=-0.05),
        make_pair(3, 0.49, 0.49, plus_delta=-0.0792, minus_delta=-0.0083),
    ]
    realized = [
        {("l0", 0): 1.0}, {("l0", 0): -1.0},
        {("l0", 1): 1.0}, {("l0", 1): -1.0},
        {("l1", 2): 1.0}, {("l1", 2): -1.0},
        {("l1", 3): 1.0}, {("l1", 3): -1.0},
    ]
    out = pl.aggregate(pairs, realized)
    check("P1: qualified_pairs == all pairs", out["qualified_pairs"] == 4, out)
    check("P1: dropped_pairs is empty", out["dropped_pairs"] == [], out)
    check("P1: pair_diffs over every pair",
          len(out["pair_diffs"]) == 4 and abs(out["pair_diffs"][2] - 0.08) < 1e-12, out)
    check("P2: gradient non-empty with a single pair alive",
          bool(pl.aggregate(pairs[:1], realized[:2])["gradient"]))


def test_symmetric_coefficients_and_weight():
    """The QES mirror update: +reward on plus, -reward on minus, normalised."""
    # pair 0 has the larger diff -> reward +1; pair 1 -> reward -1 (M == 2)
    pairs = [make_pair(0, 0.55, 0.45), make_pair(1, 0.45, 0.55)]
    realized = [
        {("l0", 7): 2.0},    # pair 0 plus
        {("l0", 7): -2.0},   # pair 0 minus (mirror-symmetric displacement)
        {("l0", 9): 1.0},    # pair 1 plus
        {("l0", 9): 1.0},    # pair 1 minus (identical displacement)
    ]
    out = pl.aggregate(pairs, realized)
    rewards = out["pair_rewards"]
    check("rewards are the rank shape over all pairs",
          abs(rewards[0] - 1.0) < 1e-12 and abs(rewards[1] + 1.0) < 1e-12, rewards)
    check("weight_total is 2 * sum(|reward|)",
          abs(out["weight_total"] - 2.0 * sum(abs(r) for r in rewards)) < 1e-12, out)
    # pair 0: reward*2 + (-reward)*(-2) = 4, normalised by 4 -> exactly +1
    check("mirror-symmetric displacements add with the pair sign",
          abs(out["gradient"][("l0", 7)] - 1.0) < 1e-12, out["gradient"])
    # pair 1: (-reward)*1 + (+reward)*1 = 0 -> identical displacements cancel
    check("identical displacements cancel to zero",
          abs(out["gradient"].get(("l0", 9), 0.0)) < 1e-12, out["gradient"])


def test_signature_and_protocol():
    """The tau parameter is gone and the protocol string is the ungated v2."""
    params = list(inspect.signature(pl.aggregate).parameters)
    check("aggregate takes (pairs, realized) only", params == ["pairs", "realized"], params)
    check("PROTOCOL is the ungated v2", pl.PROTOCOL == "minicpm5-rank-mirror-ungated/v2",
          pl.PROTOCOL)
    help_text = inspect.getdoc(pl.aggregate) or ""
    check("docstring states the ungated QES-original semantics",
          "no per-side qualification gate" in help_text, help_text)


def test_edge_cases():
    """Single pair and empty population behave deterministically."""
    pairs = [make_pair(0, 0.5, 0.5)]
    realized = [{("l0", 0): 1.0}, {("l0", 0): -1.0}]
    out = pl.aggregate(pairs, realized)
    check("single pair: rank shape of M==1 is 0 -> all-zero gradient",
          out["pair_rewards"] == [0.0] and not any(out["gradient"].values())
          and out["weight_total"] == 0.0, out)
    empty = pl.aggregate([], [])
    check("empty population: empty gradient, zero weight",
          empty["gradient"] == {} and empty["weight_total"] == 0.0
          and empty["dropped_pairs"] == [], empty)


def main():
    test_all_pairs_aggregate_regardless_of_delta()
    test_symmetric_coefficients_and_weight()
    test_signature_and_protocol()
    test_edge_cases()
    print("qes-faithful aggregation: all checks passed")


if __name__ == "__main__":
    main()
