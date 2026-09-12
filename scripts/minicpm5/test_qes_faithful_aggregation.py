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
    check("aggregate's required parameters stay (pairs, realized)",
          params[:2] == ["pairs", "realized"]
          and params[2:] == ["coef_transform", "coef_normalize", "coeff_clip"], params)
    check("PROTOCOL is the ungated v2", pl.PROTOCOL == "minicpm5-rank-mirror-ungated/v2",
          pl.PROTOCOL)
    help_text = inspect.getdoc(pl.aggregate) or ""
    check("docstring states the ungated QES-original semantics",
          "no per-side qualification gate" in help_text, help_text)


def test_zscore_coefficients():
    """Paper-default coefficients: clip((d - mean)/std, +/-clip), symmetric."""
    diffs = [0.010, 0.020, -0.010, 0.030, 0.000, 0.005, 0.015, -0.005]
    coeffs = pl.zscore_reward(diffs, clip=3.0)
    mean = sum(diffs) / len(diffs)
    var = sum((d - mean) ** 2 for d in diffs) / len(diffs)
    std = var ** 0.5
    manual = [min(3.0, max(-3.0, (d - mean) / (std + 1e-8))) for d in diffs]
    check("zscore matches the paper formula elementwise",
          all(abs(a - b) < 1e-12 for a, b in zip(coeffs, manual)), (coeffs, manual))
    check("zscore is centred", abs(sum(coeffs)) < 1e-9, sum(coeffs))
    check("clip binds on outliers",
          pl.zscore_reward([0.0] * 19 + [10.0], clip=3.0)[-1] == 3.0)
    pairs = [make_pair(i, 0.5 + d, 0.5) for i, d in enumerate(diffs)]
    realized = [{("l0", i): 1.0} for i in range(len(diffs))] + \
               [{("l0", i): -1.0} for i in range(len(diffs))]
    out = pl.aggregate(pairs, realized, coef_transform="zscore", coef_normalize="none")
    check("zscore mode reports its transform",
          out["coef_transform"] == "zscore" and out["coef_normalize"] == "none", out)


def test_no_normalization_keeps_scale():
    """coef_normalize=none must not divide the gradient (paper convention)."""
    pairs = [make_pair(0, 0.60, 0.50), make_pair(1, 0.52, 0.50)]
    realized = [{("l0", 0): 1.0}, {("l0", 0): -1.0},
                {("l0", 1): 1.0}, {("l0", 1): -1.0}]
    raw = pl.aggregate(pairs, realized, coef_transform="rank", coef_normalize="none")
    norm = pl.aggregate(pairs, realized, coef_transform="rank", coef_normalize="sum-abs")
    rewards = raw["pair_rewards"]
    # coordinate l0:0 gets 2*reward_0 (plus +1 with +r, minus -1 with -r)
    check("no-normalization keeps the raw coefficient sum",
          abs(raw["gradient"][("l0", 0)] - 2.0 * rewards[0]) < 1e-12,
          (raw["gradient"], rewards))
    check("sum-abs divides by weight_total as before",
          abs(norm["gradient"][("l0", 0)] - 2.0 * rewards[0] / norm["weight_total"]) < 1e-12,
          norm["gradient"])
    check("the two normalizations differ whenever weight_total != 1",
          abs(norm["weight_total"] - 1.0) > 0.5, norm["weight_total"])


def test_aggregation_variant_in_identity():
    """The identity hash must namespace update semantics."""
    state = pl.CentreState(generation=1)
    state.set_code("l0", 1, 5, base=0)
    pl.set_aggregation_variant("rank-sumab")
    first = state.sha256(lambda layer: "b")
    pl.set_aggregation_variant("zscore-none")
    second = state.sha256(lambda layer: "b")
    check("same weights under different update rules hash differently",
          first != second)
    pl.set_aggregation_variant("rank-sumab")
    check("variant round-trips", state.sha256(lambda layer: "b") == first)


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


def test_sha256_ignores_empty_layer_buckets():
    """delta_map drops empty buckets, so the identity hash must too.

    set_code returning the last coordinate of a layer to base leaves an empty
    ``{}`` bucket behind; before the fix the same centre hashed differently
    before/after a state.json round-trip (cross-process resume mismatch).
    """
    with_bucket = pl.CentreState(generation=1)
    with_bucket.set_code("l0", 1, 5, base=0)
    with_bucket.set_code("l1", 2, 3, base=0)
    with_bucket.codes["l2"] = {}                       # the empty-bucket artefact
    round_tripped = pl.CentreState.from_dict(with_bucket.to_dict())
    check("sha256 is invariant to empty layer buckets",
          with_bucket.sha256(lambda layer: f"base-{layer}")
          == round_tripped.sha256(lambda layer: f"base-{layer}"),
          (with_bucket.delta_map(), round_tripped.delta_map()))
    check("...and delta maps agree", with_bucket.delta_map() == round_tripped.delta_map())
    emptied = pl.CentreState(generation=1)
    emptied.set_code("l0", 1, 5, base=0)
    emptied.set_code("l1", 2, 0, base=0)               # back to base -> empty bucket
    no_artefact = pl.CentreState(generation=1)
    no_artefact.set_code("l0", 1, 5, base=0)           # same delta, no l1 key at all
    check("a coordinate returned to base does not change the identity",
          emptied.sha256(lambda layer: f"base-{layer}")
          == no_artefact.sha256(lambda layer: f"base-{layer}"))


def main():
    test_all_pairs_aggregate_regardless_of_delta()
    test_symmetric_coefficients_and_weight()
    test_signature_and_protocol()
    test_zscore_coefficients()
    test_no_normalization_keeps_scale()
    test_aggregation_variant_in_identity()
    test_edge_cases()
    test_sha256_ignores_empty_layer_buckets()
    print("qes-faithful aggregation: all checks passed")


if __name__ == "__main__":
    main()
