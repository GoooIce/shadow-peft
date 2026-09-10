#!/usr/bin/env python3
"""Round-2 retention rules: per-generation rollback, terminal safety retention,
harness-swap gate and the run's own feedback noise band.

Three results from the ByteDance Seed closed-loop RSI trilogy map onto this
loop, and each one becomes a *pre-registered* rule here rather than a knob:

* **HarnessDev (arXiv 2609.01437)** -- same-commit re-evaluation noise is about
  ±4.75 points, visible-feedback and held-out verdicts agree only 53.1% of the
  time, and the only behaviour that correlated with the downstream score was the
  *targeted revision loop* ("read the failure -> make a targeted fix ->
  re-verify"; Spearman 0.57).  §1 below is that loop in our protocol: after the
  centre step is applied, re-read the feedback channel, decompose the change per
  family, and roll the step back when a family drops further than the run's own
  measured feedback noise band (§4a) -- with the residual re-anchored per ticket
  09 §4 so the rolled-back direction cannot return.
* **Aspire (arXiv 2608.31111)** -- with a sealed evaluation set, only 1/12
  final-only runs beat baseline and 21/24 triggered a safety rollback; the
  protocol's own rule is "keep the checkpoint only if it strictly beats
  baseline, otherwise roll back".  §2 (terminal safety retention) is that rule.
* **S3Gym (arXiv 2608.31100)** -- judgment quality and improvement magnitude are
  uncorrelated (rho ~ -0.01), so a self-reported "this looked useful" is not a
  gain.  Hence every number this module emits is a measured, paired, per-family
  quantity with the threshold stated against a *measured* band, and the
  deliverable rule keys off the sealed pool rather than the feedback channel.

The module is pure CPU (no torch / no vLLM), so every rule is unit-testable on
synthetic rows and re-derivable on the control machine.
"""

from __future__ import annotations

import math
from statistics import fmean, stdev

PROTOCOL = "minicpm5-retention/v1"

#: A centre-level score, decomposed per family, is the unit every rule below
#: compares.  The fixed "held constant" band a remembered constant would use
#: (e.g. HarnessDev's +-4.75) is deliberately *not* used.
BAND_TERMS = ("harness", "sampling", "max")

FEEDBACK_BAND_NOTE = (
    "band = max(scale x term_value, quantisation floor 1/n_bucket). The floor is the smallest "
    "non-zero change the channel can express, so the rule is never 'drop > 0'. The term is "
    "measured on this run: 'harness' = standard error of repeated measurements of the identical "
    "centre through the same pipeline (patch -> refit -> flush -> score), 'sampling' = the "
    "cluster-corrected standard error of the pool score, 'max' = the larger of the two."
)
RETENTION_RULE_NOTE = (
    "pre-registered: roll the generation's centre step back if ANY family drops by more "
    "than the run's own feedback noise band for that family, relative to the previous centre"
)
TERMINAL_RULE_NOTE = (
    "pre-registered: the deliverable is the centre only if it strictly beats q0 on the "
    "sealed pool (episode-paired mean > 0); otherwise the deliverable is q0 itself and the "
    "record says no_gain_delivered_baseline"
)
SWAP_RULE_NOTE = (
    "pre-registered: a positive claim requires the gain not to vanish under the harness "
    "swap (gain > 0 on both the frozen-suffix form and the plain chat-template form)"
)


# --------------------------------------------------------------------------- #
# score blocks
# --------------------------------------------------------------------------- #


def episode_accuracies(rows: list[dict], key: str = "tf_correct") -> dict[str, float]:
    """Episode -> mean hit rate (17's statistical unit: 8-item stratum clusters)."""
    buckets: dict[str, list[float]] = {}
    for row in rows:
        buckets.setdefault(row["episode"], []).append(1.0 if row.get(key) else 0.0)
    return {episode: fmean(values) for episode, values in sorted(buckets.items())}


def _block(rows: list[dict], key: str = "tf_correct") -> dict:
    n = len(rows)
    acc = fmean([1.0 if r.get(key) else 0.0 for r in rows]) if n else None
    episodes = episode_accuracies(rows, key)
    episode_acc = fmean(list(episodes.values())) if episodes else None
    return {"n": n, "acc": acc, "episodes": len(episodes), "episode_acc": episode_acc}


def score_block(rows: list[dict], key: str = "tf_correct") -> dict:
    """Overall + per-family score block for one arm reading (centre or candidate)."""
    per_family: dict[str, list[dict]] = {}
    for row in rows:
        per_family.setdefault(str(row.get("family")), []).append(row)
    return {
        "overall": _block(rows, key),
        "per_family": {family: _block(members, key)
                       for family, members in sorted(per_family.items())},
        "unit": "accuracy on the feedback channel, item-level; per-family = same statistic "
                "restricted to one family",
    }


def _rows_of(arm):
    """Arms arrive either as raw score rows (library callers, unit fixtures)
    or as the runner's arm dicts (``confirm_arms``: {"fitness", "rows", ...})."""
    if isinstance(arm, dict) and "rows" in arm:
        return arm["rows"]
    return arm


def paired_mean(left_rows: list[dict], right_rows: list[dict], key: str = "tf_correct") -> dict:
    """Episode-level paired difference ``left - right`` (the frozen statistic)."""
    left = episode_accuracies(left_rows, key)
    right = episode_accuracies(right_rows, key)
    keys = sorted(set(left) & set(right))
    diffs = [left[episode] - right[episode] for episode in keys]
    if not diffs:
        return {"episodes": 0, "mean": None, "sd": None}
    return {
        "episodes": len(diffs),
        "mean": fmean(diffs),
        "sd": stdev(diffs) if len(diffs) > 1 else None,
    }


def item_delta(left_rows: list[dict], right_rows: list[dict], key: str = "tf_correct") -> float | None:
    """Item-level accuracy difference on the *common* items only."""
    right = {row["id"]: row for row in right_rows}
    pairs = [(row, right[row["id"]]) for row in left_rows if row["id"] in right]
    if not pairs:
        return None
    return fmean([1.0 if a.get(key) else 0.0 for a, _ in pairs]) - fmean(
        [1.0 if b.get(key) else 0.0 for _, b in pairs])


# --------------------------------------------------------------------------- #
# §4a  the run's own feedback noise band
# --------------------------------------------------------------------------- #


def _sd_se(values: list[float]) -> tuple[float | None, float | None]:
    clean = [float(value) for value in values if value is not None]
    if len(clean) < 2:
        return None, None
    sd = stdev(clean)
    return sd, sd / math.sqrt(len(clean))


def harness_se(replicates: list[dict]) -> dict:
    """Dispersion of repeated readings of the **same** centre on the feedback channel.

    ``replicates`` are :func:`score_block` outputs of the identical centre, taken
    through the full pipeline (patch -> refit -> flush -> score), so this is the
    same-commit re-evaluation noise HarnessDev measured -- at our precision, on
    this run, not a remembered constant.
    """
    centre = [rep["overall"]["acc"] for rep in replicates if rep["overall"]["acc"] is not None]
    episodes = [rep["overall"]["episode_acc"] for rep in replicates
                if rep["overall"]["episode_acc"] is not None]
    families = sorted({family for rep in replicates for family in rep["per_family"]})
    centre_sd, centre_se = _sd_se(centre)
    episode_sd, episode_se = _sd_se(episodes)
    per_family: dict[str, dict] = {}
    for family in families:
        values = [rep["per_family"][family]["acc"] for rep in replicates
                  if family in rep["per_family"] and rep["per_family"][family]["acc"] is not None]
        sd, se = _sd_se(values)
        per_family[family] = {"replicates": len(values), "sd": sd, "se": se}
    return {
        "kind": "harness_se",
        "replicates": len(replicates),
        "centre_sd": centre_sd,
        "centre_se": centre_se,
        "episode_sd": episode_sd,
        "episode_se": episode_se,
        "per_family": per_family,
        "deterministic": bool(centre_sd == 0.0 and all(
            block["sd"] == 0.0 for block in per_family.values() if block["sd"] is not None)),
        "metric": ("standard error of a repeated measurement of the identical centre through "
                   "the same pipeline; 0 means the harness is bit-deterministic at this "
                   "precision, in which case the band falls back to the quantisation floor"),
    }


def sampling_se(rows: list[dict], key: str = "tf_correct") -> dict:
    """Cluster-corrected standard error of a score on this pool (episode level).

    Reported alongside the harness term: it is the right reference for a *claim*
    about the family/pool, while the harness term is the right reference for the
    retention *decision* (which asks whether the measured drop is real on the
    channel, not whether the channel represents the world).
    """
    episodes = episode_accuracies(rows, key)
    per_family: dict[str, dict] = {}
    families: dict[str, list[dict]] = {}
    for row in rows:
        families.setdefault(str(row.get("family")), []).append(row)
    for family, members in sorted(families.items()):
        values = list(episode_accuracies(members, key).values())
        if len(values) < 2:
            per_family[family] = {"episodes": len(values), "sd": None, "se": None}
            continue
        sd = stdev(values)
        per_family[family] = {"episodes": len(values), "sd": sd, "se": sd / math.sqrt(len(values))}
    overall = list(episodes.values())
    overall_sd = stdev(overall) if len(overall) > 1 else None
    return {
        "kind": "sampling_se",
        "centre_se": (overall_sd / math.sqrt(len(overall))) if overall_sd is not None else None,
        "episode_sd": overall_sd,
        "episodes": len(overall),
        "per_family": per_family,
        "metric": "cluster-corrected (episode-level) standard error of the pool score",
    }


def band_replicates(startup: list[dict], inrun: list[dict]) -> tuple[list[dict], str]:
    """Deterministic selection of the replicate set for the harness term.

    Preference order (frozen): the startup no-op re-scores when there are at least
    two of them; otherwise the largest group of in-run readings that share one
    centre identity (``sha``), which is what "repeated centre measurements across
    generations" gives for free when the step was rolled back; otherwise no
    harness term at all (the band is then the quantisation floor alone).
    """
    if len(startup) >= 2:
        return list(startup), "startup_noop"
    groups: dict[str, list[dict]] = {}
    for entry in inrun or []:
        groups.setdefault(str(entry.get("sha")), []).append(entry.get("block"))
    if groups:
        sha, blocks = max(sorted(groups.items()), key=lambda kv: (len(kv[1]), kv[0]))
        if len(blocks) >= 2:
            return list(blocks), f"generation_repeats:{sha[:16]}"
    return [], "none"


def noise_band(replicates: list[dict], rows: list[dict], scale: float = 1.0,
               apply_floor: bool = True, key: str = "tf_correct",
               source: str = "startup_noop", term: str = "harness") -> dict:
    """The run's own feedback noise band, per bucket (centre level and per family).

    ``band = max(scale * term_value, quantisation floor)`` with the floor equal to
    the smallest non-zero change the bucket can express (``1 / n``).  The floor is
    named explicitly because a bit-deterministic pipeline produces a harness SE of
    exactly 0, and ``drop > 0`` is not a rule -- it is a coin flip on a single
    item.

    ``term`` names which measured standard error the threshold is stated against:

    * ``harness`` (the brief's literal reading): the standard error of repeated
      measurements of the identical centre through the same pipeline.  On a
      bit-deterministic scorer this is 0 and the band is the floor.
    * ``sampling``: the cluster-corrected standard error of the pool score
      (episode-level dispersion).  This is the term a *claim* has to be stated
      against, and the one that describes how much a per-family reading moves
      between generations on a pool of this size.
    * ``max``: the larger of the two.

    Which term was used is written into every decision record, and the multiplier
    (``scale``) defaults to the pre-registered 1.0.
    """
    if term not in BAND_TERMS:
        raise ValueError(f"unknown band term {term!r}; want one of {BAND_TERMS}")
    harness = harness_se(replicates)
    sampling = sampling_se(rows, key) if rows else {"kind": "sampling_se", "per_family": {}}
    items = len(rows)
    families = sorted({str(row.get("family")) for row in rows})
    counts = {family: sum(1 for row in rows if str(row.get("family")) == family)
              for family in families}

    def pick(harness_value, sampling_value) -> float:
        harness_value = harness_value or 0.0
        sampling_value = sampling_value or 0.0
        if term == "harness":
            return harness_value
        if term == "sampling":
            return sampling_value
        return max(harness_value, sampling_value)

    centre_term = pick(harness["centre_se"], sampling.get("centre_se"))
    centre_floor = (1.0 / items) if (apply_floor and items) else 0.0
    per_family: dict[str, dict] = {}
    for family in families:
        harness_value = (harness["per_family"].get(family, {}) or {}).get("se")
        sampling_value = (sampling["per_family"].get(family, {}) or {}).get("se")
        value = pick(harness_value, sampling_value)
        floor = (1.0 / counts[family]) if (apply_floor and counts[family]) else 0.0
        band = max(scale * value, floor)
        per_family[family] = {
            "band": band,
            "term": term,
            "term_value": value,
            "harness_se": harness_value or 0.0,
            "sampling_se": sampling_value,
            "quantisation_floor": floor,
            "band_items": band * counts[family],
            "n_items": counts[family],
            "driven_by": "quantisation_floor" if (band == floor and floor >= scale * value)
                         else f"{term}_se",
        }
    centre_band = max(scale * centre_term, centre_floor)
    return {
        "protocol": PROTOCOL,
        "kind": "feedback_noise_band",
        "rule": FEEDBACK_BAND_NOTE,
        "source": source,
        "term": term,
        "scale": scale,
        "apply_floor": apply_floor,
        "replicates": harness["replicates"],
        "n_items": items,
        "centre": {
            "band": centre_band,
            "term": term,
            "term_value": centre_term,
            "harness_se": harness["centre_se"] or 0.0,
            "sampling_se": sampling.get("centre_se"),
            "quantisation_floor": centre_floor,
            "band_items": centre_band * items,
            "episode_sd": sampling.get("episode_sd"),
            "episodes": sampling.get("episodes"),
            "driven_by": "quantisation_floor" if (centre_band == centre_floor
                                                  and centre_floor >= scale * centre_term)
                         else f"{term}_se",
        },
        "per_family": per_family,
        "harness": harness,
        "sampling": sampling,
        "items_per_family": counts,
    }


# --------------------------------------------------------------------------- #
# §1  per-generation retention (targeted revision)
# --------------------------------------------------------------------------- #


def retention_decision(rows_before: list[dict], rows_after: list[dict], band: dict,
                       key: str = "tf_correct") -> dict:
    """Pre-registered rule: roll back the step if any family drops past its band.

    ``rows_before`` is the *previous* centre's reading on the feedback channel and
    ``rows_after`` the reading of the centre the generation just produced.  The
    comparison is per family, one-sided, and strictly greater than the band, so a
    family that merely fails to improve never triggers a rollback.

    The comparison is made in **item counts** (integers): the band is converted to
    "how many items is that on this bucket" and the drop must exceed it.  A float
    comparison at the boundary would let arithmetic dust decide a pre-registered
    rule -- a drop of exactly one item must not breach when the band is exactly
    one item.
    """
    before = score_block(rows_before, key)
    after = score_block(rows_after, key)
    families = sorted(set(before["per_family"]) & set(after["per_family"]))
    per_family: dict[str, dict] = {}
    breaches: list[dict] = []
    worst_family, worst_drop = None, None
    for family in families:
        b = before["per_family"][family]
        a = after["per_family"][family]
        entry = band["per_family"].get(family) or band["centre"]
        threshold = entry["band"]
        n = min(b["n"], a["n"])
        b_items = round(b["acc"] * b["n"]) if b["acc"] is not None else None
        a_items = round(a["acc"] * a["n"]) if a["acc"] is not None else None
        band_items = threshold * n
        drop_items = (b_items - a_items) if (b_items is not None and a_items is not None) else None
        delta = (a["acc"] - b["acc"]) if (a["acc"] is not None and b["acc"] is not None) else None
        breach = bool(drop_items is not None and drop_items > band_items + 1e-9)
        per_family[family] = {
            "n_items": a["n"],
            "before": b["acc"],
            "after": a["acc"],
            "before_items": b_items,
            "after_items": a_items,
            "delta": delta,
            "drop_items": drop_items,
            "band": threshold,
            "band_items": band_items,
            "band_source": entry["driven_by"],
            "breach": breach,
        }
        if delta is not None and (worst_drop is None or delta < worst_drop):
            worst_drop, worst_family = delta, family
        if breach:
            breaches.append({"family": family, "delta": delta, "band": threshold,
                             "drop_items": drop_items, "band_items": band_items})
    centre_delta = (after["overall"]["acc"] - before["overall"]["acc"]) if (
        after["overall"]["acc"] is not None and before["overall"]["acc"] is not None) else None
    centre_items = (round(before["overall"]["acc"] * before["overall"]["n"])
                    - round(after["overall"]["acc"] * after["overall"]["n"])) if (
        after["overall"]["acc"] is not None and before["overall"]["acc"] is not None) else None
    centre_band_items = band["centre"]["band"] * min(before["overall"]["n"], after["overall"]["n"])
    return {
        "protocol": PROTOCOL,
        "kind": "retention_decision",
        "rule": RETENTION_RULE_NOTE,
        "channel": "feedback (dev pool), item-level accuracy per family, paired reading of "
                   "the previous centre vs the centre the step produced",
        "comparison_unit": "items (integers): the band is converted to items on the bucket and "
                           "the drop must exceed it, so the boundary is not decided by float dust",
        "rollback": bool(breaches),
        "breaches": breaches,
        "worst_family": worst_family,
        "worst_delta": worst_drop,
        "centre_delta": centre_delta,
        "centre_drop_items": centre_items,
        "centre_band": band["centre"]["band"],
        "centre_band_items": centre_band_items,
        "centre_breach": bool(centre_items is not None
                              and centre_items > centre_band_items + 1e-9),
        "per_family": per_family,
        "band_used": {
            "protocol": band.get("protocol"),
            "source": band.get("source"),
            "scale": band.get("scale"),
            "replicates": band.get("replicates"),
            "centre": band["centre"],
            "per_family": {family: band["per_family"].get(family) for family in families},
        },
        "before": before["overall"],
        "after": after["overall"],
    }


# --------------------------------------------------------------------------- #
# §2 / §3  terminal safety retention and the harness-swap gate
# --------------------------------------------------------------------------- #


def terminal_retention(arms_by_variant: dict, headline: str = "frozen",
                       delta: float = 0.02, keys: dict | None = None) -> dict:
    """§2 + §3 at the terminal step: deliverable rule and the swap robustness gate.

    ``arms_by_variant`` maps a prompt variant name to ``{"q0": rows, "center": rows}``
    measured on the **sealed** pool in the same process.  The headline variant
    decides the deliverable (pre-registered: centre only if it strictly beats q0);
    the other variant(s) are the harness-swap robustness gate (HarnessDev's
    executor swap: a gain that only exists under the frozen ``Answer:`` suffix may
    not transfer to deployment).
    """
    keys = keys or {}
    variants: dict[str, dict] = {}
    for variant, arms in arms_by_variant.items():
        if "q0" not in arms or "center" not in arms:
            continue
        centre_rows, q0_rows = _rows_of(arms["center"]), _rows_of(arms["q0"])
        paired = paired_mean(centre_rows, q0_rows)
        variants[variant] = {
            "q0_acc": _block(q0_rows)["acc"],
            "center_acc": _block(centre_rows)["acc"],
            "episode_paired": paired,
            "item_paired": item_delta(centre_rows, q0_rows),
            "q0_episodes": _block(q0_rows)["episodes"],
            "center_episodes": _block(centre_rows)["episodes"],
            "gain_mean": paired["mean"],
            "gain_positive": bool(paired["mean"] is not None and paired["mean"] > 0.0),
            "gain_reaches_delta": bool(paired["mean"] is not None and paired["mean"] >= delta),
        }
    if headline not in variants:
        return {
            "protocol": PROTOCOL, "kind": "terminal_retention",
            "ok": False, "reason": f"headline variant {headline!r} has no complete arm pair",
            "variants": variants, "rule": TERMINAL_RULE_NOTE,
        }
    head = variants[headline]
    others = {name: block for name, block in variants.items() if name != headline}
    deliver_center = bool(head["gain_positive"])
    # §3 gate: the gain must not vanish under the swap
    swap: dict = {"rule": SWAP_RULE_NOTE, "headline": headline, "comparisons": {}}
    survives: list[bool] = []
    for name, block in others.items():
        heads_both = bool(head["gain_positive"] and block["gain_positive"])
        survives.append(heads_both)
        swap["comparisons"][name] = {
            "gain_mean": block["gain_mean"],
            "gain_positive": block["gain_positive"],
            "gain_lost": (head["gain_mean"] - block["gain_mean"])
            if (head["gain_mean"] is not None and block["gain_mean"] is not None) else None,
            "gate_passes": heads_both,
        }
    swap["gate_passes"] = all(survives) if survives else None
    swap["ran"] = bool(others)
    swap["detail"] = ("no swap arm measured (--no-terminal-swap): the robustness gate is "
                      "not evaluated and no positive claim may be made from this record"
                      if not others else
                      "positive claim requires gain > 0 on the headline form and on every "
                      "swap form")
    record = {
        "protocol": PROTOCOL,
        "kind": "terminal_retention",
        "ok": True,
        "rule": TERMINAL_RULE_NOTE,
        "delta_tier": delta,
        "headline_variant": headline,
        "deliverable": "center" if deliver_center else "q0",
        "record": "gain_delivered_centre" if deliver_center else "no_gain_delivered_baseline",
        "decision_basis": ("episode-paired mean of the headline form on the sealed pool, "
                           "strictly greater than 0"),
        "headline": head,
        "variants": variants,
        "swap": swap,
        "claim": {
            "gain_claimable": bool(deliver_center and swap["gate_passes"] is True),
            "requires": ["deliverable == centre (strictly beats q0)",
                         "the gain does not vanish under the harness swap"],
            "wording_authority": ("the wording itself stays with confirm_gate.py / "
                                  "classify_outcome; this record only gates the deliverable "
                                  "and the swap robustness clause"),
        },
    }
    if keys:
        record["keys"] = keys
    return record
