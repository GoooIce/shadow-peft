#!/usr/bin/env python3
"""Dead-mechanism audit: did every mechanism in the protocol actually fire, and
did firing change anything (§6).

HarnessDev's retrospective counts 124 dead features and 18 components that never
triggered; our own ledger already contains one instance of the expensive class:
ticket 14's KL **trend** clause fired at gen6 and gen14 as a *report-only*
`trend_breach` and the run continued for ten more generations.  That is not
cosmetic -- it cost the run its planned length and it changed what the delivered
centre is.  This script makes the audit routine: every mechanism in the protocol
gets a status derived from the ledger, with the evidence that decides it.

Statuses
--------
``alive``            fired at least once, and the effect is visible downstream.
``never_fired``      wired and watched, but the trigger never occurred.
``fired_no_effect``  fired, and nothing downstream changed (the report-only
                     class: the guard that did not guard).
``not_wired``        the run has no evidence the mechanism was even switched on.
``no_data``          the run directory does not carry the files this needs.

Usage
-----
    # 1) local run directory
    python mechanism_audit.py --run <out_dir> --out mechanism_audit.json

    # 2) a live run, read-only over ssh (the script is copied to /tmp on the host;
    #    only the small ledgers are read, nothing under the run dir is written)
    python mechanism_audit.py --ssh gooo:~/minicpm5-run300/out/seed20260912 \
        --out mechanism_audit_gooo.json

The projection mode (``--project``) is pure standard library and reads only the
ledgers, so it is safe to run on a machine that is busy with GPU work.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

PROTOCOL = "minicpm5-mechanism-audit/v1"

#: top-level files the projection reads (all small)
TOP_FILES = (
    "summary.json", "state.json", "history.jsonl", "stop_ledger.jsonl", "budget.jsonl",
    "stop.json", "census.json", "feedback_noise.json", "terminal_retention.json",
    "prompt_swap.json", "reference_check.json",
)
RECORDS_RE = re.compile(
    r"^(gen\d+\.json|centre_gen\d+\.json|q0_reference\.json|"
    r"confirm_arms\.json|confirm_gen\d+\.json|noop_rescore.*\.json|retention_gen\d+\.json|"
    r"halt_record\.json)$")


# --------------------------------------------------------------------------- #
# projection (pure stdlib; safe to run against a live run, read-only)
# --------------------------------------------------------------------------- #


def project(run_dir: Path) -> dict:
    """Read the ledgers and emit the compact bundle the audit analyses."""
    out: dict = {"run_dir": str(run_dir), "protocol": PROTOCOL, "files": {}}
    for name in TOP_FILES:
        path = run_dir / name
        out["files"][name] = path.exists()
    summary = _json(run_dir / "summary.json")
    if isinstance(summary, dict):
        out["summary"] = {
            "protocol": summary.get("protocol"),
            "generations": summary.get("generations"),
            "candidates_per_generation": summary.get("candidates_per_generation"),
            "alpha": summary.get("alpha"),
            "tau": summary.get("tau"),
            "seed": summary.get("seed"),
            "wall_seconds": summary.get("wall_seconds"),
            "final_centre_sha256": summary.get("final_centre_sha256"),
            "stop": summary.get("stop"),
            "round2": summary.get("round2"),
            "layout_check": summary.get("layout_check"),
            "engine_load_seconds": summary.get("engine_load_seconds"),
        }
    state = _json(run_dir / "state.json")
    if isinstance(state, dict):
        out["state"] = {
            "generation": state.get("generation"),
            "protocol": state.get("protocol"),
            "delta_coords": sum(len(b) for b in (state.get("codes") or {}).values()),
            "residual_coords": sum(len(b) for b in (state.get("residual") or {}).values()),
            "delta_layers": len(state.get("codes") or {}),
            "residual_layers": len(state.get("residual") or {}),
            "gamma": state.get("gamma"),
            "clamp": state.get("clamp"),
        }
    out["history"] = _jsonl(run_dir / "history.jsonl")
    out["stop_ledger"] = _jsonl(run_dir / "stop_ledger.jsonl")
    out["budget_intervals"] = _jsonl(run_dir / "budget.jsonl")
    out["stop"] = _json(run_dir / "stop.json")
    for name, target in (("feedback_noise.json", "feedback_noise"),
                         ("terminal_retention.json", "terminal_retention"),
                         ("prompt_swap.json", "prompt_swap")):
        blob = _json(run_dir / name)
        if blob is not None:
            out[target] = blob
    q0 = _json(run_dir / "records" / "q0_reference.json")
    if isinstance(q0, dict):
        out["q0_reference"] = {
            "taskset": q0.get("taskset"),
            "fitness": q0.get("fitness"),
            "no_answer": {"overall": (q0.get("no_answer") or {}).get("overall")},
        }
    confirm = _json(run_dir / "records" / "confirm_arms.json")
    if isinstance(confirm, dict):
        out["confirm_arms"] = {
            "center_sha256": confirm.get("center_sha256"),
            "center_generation": confirm.get("center_generation"),
            "prompt_variant": confirm.get("prompt_variant"),
            "variants": list((confirm.get("arms_by_variant") or {}).keys())
            or ([confirm.get("variant")] if confirm.get("variant") else ["frozen"]),
            "arms": {name: {"fitness": (arm or {}).get("fitness")}
                     for name, arm in (confirm.get("arms") or {}).items()
                     if isinstance(arm, dict)},
            "attribution": (confirm.get("arms") or {}).get("attribution_center_vs_q0"),
            "terminal_retention": confirm.get("terminal_retention"),
        }
    records = []
    for path in sorted((run_dir / "records").glob("*.json")):
        if not RECORDS_RE.match(path.name):
            continue
        blob = _json(path)
        if not isinstance(blob, dict):
            continue
        records.append({"file": path.name, **_project_record(blob)})
    out["records"] = records
    kernel_values = []
    for path in sorted(run_dir.glob("kernel_values_gen*.json")):
        blob = _json(path)
        if isinstance(blob, dict):
            kernel_values.append({
                "file": path.name,
                "tensors": blob.get("tensors"),
                "value_diffs": blob.get("value_diffs"),
                "canonical_code_diffs": blob.get("canonical_code_diffs"),
                "raw_code_diffs": blob.get("raw_code_diffs"),
                "zero8_diffs": blob.get("zero8_diffs"),
                "structural_mismatches": blob.get("structural_mismatches"),
                "tensors_with_value_diffs": blob.get("tensors_with_value_diffs"),
                "pass_value": blob.get("pass_value"),
            })
    out["kernel_values"] = kernel_values
    return out


def _project_record(blob: dict) -> dict:
    ledger = blob.get("residual_ledger") or {}
    aggregation = blob.get("aggregation") or {}
    gate = (blob.get("hard_gate") or {}).get("gate") or {}
    readback = blob.get("live_readback") or {}
    retention = blob.get("retention") or {}
    channels = retention.get("channels") or {}
    feedback = channels.get("feedback") or {}
    entry = {
        "generation": blob.get("generation"),
        "kind": blob.get("kind"),
        "group_id": blob.get("group_id"),
        "alpha": blob.get("alpha"),
        "center_sha256_before": blob.get("center_sha256_before"),
        "center_sha256_after": blob.get("center_sha256_after"),
        "centre_acc": (blob.get("center_fitness") or {}).get("acc"),
        "centre_post_acc": (blob.get("centre_post_fitness") or {}).get("acc"),
        "moved": ledger.get("moved"),
        "touched": ledger.get("touched"),
        "rejected_beyond_grid": ledger.get("rejected_beyond_grid"),
        "residual_clamped": ledger.get("residual_clamped"),
        "unmoved_below_projection": ledger.get("unmoved_below_projection"),
        "residual_abs_max": ledger.get("residual_abs_max"),
        "u_abs_max": ledger.get("u_abs_max"),
        "qualified_pairs": aggregation.get("qualified_pairs"),
        "dropped_pairs": aggregation.get("dropped_pairs"),
        "pair_rewards": aggregation.get("pair_rewards"),
        "gradient_support": aggregation.get("gradient_support"),
        "gate_pass": gate.get("pass"),
        "gate_tensors": gate.get("tensors"),
        "gate_value_diffs": gate.get("value_diffs"),
        "gate_zero8_diffs": gate.get("zero8_diffs"),
        "gate_raw_diffs": gate.get("raw_code_diffs"),
        "gate_structural": gate.get("structural_mismatches"),
        "readback_coordinates": readback.get("coordinates"),
        "readback_value_mismatches": readback.get("value_mismatches"),
        "readback_engine_cache_raw": readback.get("engine_cache_raw_mismatches"),
        "readback_zero8": readback.get("zero8_representation"),
        "kl": (_kl(blob)),
        "confirm": (((blob.get("confirm") or {}).get("fitness") or {}).get("acc")),
        "retention_rollback": retention.get("rollback"),
        "retention_kind": retention.get("kind"),
        "retention_breaches": [b.get("family") for b in (feedback.get("breaches") or [])],
        "retention_band_source": (retention.get("band") or {}).get("source"),
        "retention_slice_role": (retention.get("feedback_slice") or {}).get("role"),
        "retention_slice_rollback": ((retention.get("feedback_slice") or {}).get("gate") or {}
                                     ).get("rollback"),
        "retention_sha_restored": retention.get("sha_restored_exactly"),
        "retention_reanchor_coords": (retention.get("reanchor") or {}).get("coordinates"),
    }
    if blob.get("kind") == "centre" or blob.get("kind") == "centre_baseline":
        entry["centre_acc"] = (blob.get("fitness") or {}).get("acc")
    return entry


def _kl(blob: dict):
    """KL is a probe dict in generation records and a bare float in guard records."""
    value = blob.get("kl")
    if isinstance(value, dict):
        return value.get("generation_kl_error")
    return value


def _json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _jsonl(path: Path) -> list:
    rows = []
    try:
        for line in path.read_text().splitlines():
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []
    return rows


# --------------------------------------------------------------------------- #
# mechanisms
# --------------------------------------------------------------------------- #


def _gens(ledger: dict) -> list[dict]:
    rows = []
    for entry in ledger.get("stop_ledger") or []:
        rows.append(entry)
    return rows


def _record_series(ledger: dict, key: str) -> list[tuple[int, object]]:
    """Generation -> value, from the per-generation records (one entry per generation)."""
    latest: dict[int, object] = {}
    for entry in ledger.get("records") or []:
        if entry.get("kind") in ("centre", "centre_baseline"):
            continue
        if entry.get("generation") is None:
            continue
        generation = int(entry["generation"])
        if entry.get(key) is not None or generation not in latest:
            latest[generation] = entry.get(key)
    return sorted(latest.items())


def _history(ledger: dict) -> list[dict]:
    return ledger.get("history") or []


def _status(fired: int, effect: bool, wired: bool = True) -> str:
    if not wired:
        return "not_wired"
    if not fired:
        return "never_fired"
    return "alive" if effect else "fired_no_effect"


def probe_hard_gate(ledger: dict) -> dict:
    series = _record_series(ledger, "gate_pass")
    values = ledger.get("kernel_values") or []
    if not series and not values:
        return _entry("hard_gate_value", "no_data", "abort the generation when the kernel dump "
                      "and the offline export disagree on the decoded values")
    fails = [gen for gen, passed in series if passed is False]
    tensors = sorted({entry["tensors"] for entry in values if entry.get("tensors")})
    return _entry(
        "hard_gate_value", _status(len(series), not fails),
        "abort the generation when the engine dump and the offline export disagree on values",
        fired=len(series), first=series[0][0] if series else None,
        last=series[-1][0] if series else None,
        effect=("every generation passed the value calibre" if not fails else
                f"FAILED at generations {fails}"),
        evidence={"generations": len(series), "failures": fails,
                  "tensors_per_generation": tensors,
                  "value_diffs_total": sum(int(e.get("value_diffs") or 0) for e in values),
                  "value_diffs_nonzero_generations":
                      [e["file"] for e in values if int(e.get("value_diffs") or 0) > 0]},
    )


def probe_readback(ledger: dict) -> dict:
    series = _record_series(ledger, "readback_value_mismatches")
    coordinates = [c for _g, c in _record_series(ledger, "readback_coordinates") if c is not None]
    if not series:
        return _entry("readback", "no_data",
                      "read every live coordinate back from the engine each generation")
    mismatches = [gen for gen, count in series if count]
    return _entry(
        "readback", _status(len(series), not mismatches),
        "read every live coordinate back from the engine and compare (leak detector)",
        fired=len(series), first=series[0][0] if series else None,
        last=series[-1][0] if series else None,
        effect=("0 value mismatches on every generation" if not mismatches else
                f"mismatches at {mismatches}"),
        evidence={"generations": len(series), "mismatch_generations": mismatches,
                  "coordinates_min": min(coordinates) if coordinates else None,
                  "coordinates_max": max(coordinates) if coordinates else None},
    )


def probe_kl_absolute(ledger: dict, threshold: float) -> dict:
    kls = [(gen, kl) for gen, kl in _record_series(ledger, "kl") if kl is not None]
    ledger_rows = _gens(ledger)
    halts = [row.get("generation") for row in ledger_rows
             if (row.get("halt") or {}).get("rule") == "kl_over_threshold"]
    over = [gen for gen, kl in kls if kl > threshold]
    peak = max((kl for _g, kl in kls), default=None)
    return _entry(
        "kl_absolute_halt", _status(len(over), bool(halts)),
        f"halt when the centre KL exceeds the absolute threshold ({threshold})",
        fired=len(over), first=over[0] if over else None, last=over[-1] if over else None,
        effect=("never approached" if not over else f"over threshold at {over}; halts {halts}"),
        evidence={"generations_measured": len(kls), "peak_kl": peak, "threshold": threshold,
                  "over_threshold_generations": over, "halts": halts},
    )


def probe_kl_trend(ledger: dict) -> dict:
    rows = _gens(ledger)
    breaches = [row.get("generation") for row in rows if row.get("trend_breach")]
    material = [row.get("generation") for row in rows if row.get("trend_material")]
    halts = [row.get("generation") for row in rows
             if (row.get("halt") or {}).get("rule") == "kl_trend_breach"]
    floors = sorted({row.get("materiality_floor") for row in rows
                     if row.get("materiality_floor") is not None})
    if not rows:
        return _entry("kl_trend_halt", "no_data",
                      "halt on two consecutive rises that are materially above the KL baseline")
    effect = bool(halts) or (not material and not breaches)
    status = _status(len(rows), effect)
    if breaches and not halts:
        status = "fired_no_effect"
    return _entry(
        "kl_trend_halt", status,
        "halt on two consecutive KL rises that are both materially above the baseline "
        "(ticket 14's trend clause, corrected by the materiality band)",
        fired=len(breaches), first=breaches[0] if breaches else None,
        last=breaches[-1] if breaches else None,
        effect=("no breach occurred" if not breaches else
                ("breach(es) halted the run" if halts else
                 "breach(es) recorded but NOT acted on: report-only")),
        evidence={"generations": len(rows), "trend_breach_generations": breaches,
                  "trend_material_generations": material, "halts": halts,
                  "materiality_floors": floors,
                  "kl_series_tail": [row.get("kl") for row in rows[-8:]]},
    )


def probe_alpha_ladder(ledger: dict, enabled: bool | None = None) -> dict:
    rows = _gens(ledger)
    switches = [{"generation": row.get("generation"), **(row.get("switch") or {})}
                for row in rows if row.get("switch")]
    alphas = sorted({row.get("alpha") for row in rows if row.get("alpha") is not None})
    highs = [s for s in switches if s.get("to") == 16.0]
    lows = [s for s in switches if s.get("to") == 8.0]
    # the ledger does not record --alpha-ladder, so "wired" is inferred: a recorded
    # switch, or more than one alpha value in the series, or the caller's override
    wired = enabled if enabled is not None else bool(switches or len(alphas) > 1)
    evidence = {"switches": switches, "alphas_seen": alphas,
                "high_switches": len(highs), "low_switches": len(lows),
                "generations": len(rows)}
    return {
        "high": _entry("alpha_ladder_high", _status(len(highs), bool(highs), wired=True),
                       "raise alpha to 16 after two generations of cross-grid < 0.40 with M >= 8",
                       fired=len(highs),
                       first=highs[0]["generation"] if highs else None,
                       last=highs[-1]["generation"] if highs else None,
                       effect=("alpha raised" if highs else "never triggered"),
                       evidence=evidence),
        "low": _entry("alpha_ladder_low", _status(len(lows), bool(lows), wired=True),
                      "lower alpha to 8 after two generations of endpoint rejection > 5% with M <= 8",
                      fired=len(lows),
                      first=lows[0]["generation"] if lows else None,
                      last=lows[-1]["generation"] if lows else None,
                      effect=("alpha lowered" if lows else "never triggered"),
                      evidence=evidence),
        "enabled_guess": wired,
    }


def probe_no_accept(ledger: dict) -> dict:
    rows = _gens(ledger)
    halts = [row.get("generation") for row in rows
             if (row.get("halt") or {}).get("rule") == "no_accept"]
    zero = [row.get("generation") for row in rows if row.get("moved") == 0]
    longest, current = 0, 0
    for row in rows:
        current = current + 1 if row.get("moved") == 0 else 0
        longest = max(longest, current)
    effect = bool(halts) or longest < 2
    return _entry("no_accept_halt", _status(longest, effect),
                  "halt after two consecutive generations without an accepted update",
                  fired=longest, first=zero[0] if zero else None, last=zero[-1] if zero else None,
                  effect=("no two consecutive empty generations" if longest < 2 else
                          ("halted" if halts else "the run continued past the trigger")),
                  evidence={"generations": len(rows), "moved_zero_generations": zero,
                            "longest_empty_streak": longest, "halts": halts})


def probe_residual_clamp(ledger: dict) -> dict:
    clamps = [(gen, value) for gen, value in _record_series(ledger, "residual_clamped")
              if value is not None]
    rejected = [(gen, value) for gen, value in _record_series(ledger, "rejected_beyond_grid")
                if value is not None]
    unmoved = [(gen, value) for gen, value in _record_series(ledger, "unmoved_below_projection")
               if value is not None]
    total_clamp = sum(value for _g, value in clamps)
    total_reject = sum(value for _g, value in rejected)
    residual_max = [value for _g, value in _record_series(ledger, "residual_abs_max")
                    if value is not None]
    return {
        "clamp": _entry("residual_clamp", _status(1 if total_clamp else 0, total_clamp > 0),
                        "clamp the numeric residual at +/-1.5 (ticket 09's gamma=1 bookkeeping)",
                        fired=total_clamp,
                        effect=("the clamp never bit" if total_clamp == 0 else "clamp engaged"),
                        evidence={"generations_watched": len(clamps), "clamped_total": total_clamp,
                                  "residual_abs_max_peak": max(residual_max) if residual_max else None}),
        "reject": _entry("endpoint_rejection", _status(1 if total_reject else 0, total_reject > 0),
                         "reject a whole step that projects beyond the E2M1 endpoints",
                         fired=total_reject,
                         effect=("never triggered" if total_reject == 0 else "steps rejected"),
                         evidence={"generations_watched": len(rejected),
                                   "rejected_total": total_reject,
                                   "rejected_max": max((v for _g, v in rejected), default=None)}),
        "unmoved": _entry("unmoved_below_projection",
                          _status(1 if sum(v for _g, v in unmoved) else 0,
                                  sum(v for _g, v in unmoved) > 0),
                          "count coordinates whose projection lands back on the same code "
                          "(the residual keeps the float step)",
                          fired=sum(v for _g, v in unmoved),
                          effect=("never" if sum(v for _g, v in unmoved) == 0
                                  else "coordinates held by the residual"),
                          evidence={"generations_watched": len(unmoved),
                                    "unmoved_total": sum(v for _g, v in unmoved)}),
    }


def probe_zero8(ledger: dict) -> dict:
    values = ledger.get("kernel_values") or []
    zero8 = [int(entry.get("zero8_diffs") or 0) for entry in values]
    raw = [int(entry.get("raw_code_diffs") or 0) for entry in values]
    entry_gens = [entry["file"] for entry in values if int(entry.get("zero8_diffs") or 0) > 0]
    raw_gens = [entry["file"] for entry in values if int(entry.get("raw_code_diffs") or 0) > 0]
    fired = len({*entry_gens, *raw_gens})
    return _entry(
        "zero8_accounting", _status(fired, fired > 0),
        "keep the numerically empty 0<->8 code channel on its own line of account "
        "(never folded into the value calibre)",
        fired=fired,
        effect=("nothing to keep apart (every calibre empty)" if not fired else
                "differences observed and reported apart from the value calibre"),
        evidence={"generations": len(values), "zero8_total": sum(zero8),
                  "zero8_generations": entry_gens, "raw_diffs_total": sum(raw),
                  "raw_nonzero_generations": raw_gens},
    )


def probe_q0_reference(ledger: dict) -> dict:
    ref = ledger.get("q0_reference")
    baseline_rows = [row for row in _history(ledger) if row.get("kind") == "centre_baseline"]
    if not ref and not baseline_rows:
        return _entry("q0_reference", "not_wired",
                      "measure q0 on the run's own dev pool at startup and compare every "
                      "generation against it")
    return _entry(
        "q0_reference", "alive" if ref else "fired_no_effect",
        "measure q0 on the run's own dev pool at startup and compare every generation "
        "against it instead of a remembered constant",
        fired=1 + len(baseline_rows),
        effect=("baseline measured in-process" if ref else "only the baseline history row exists"),
        evidence={"q0_fitness": (ref or {}).get("fitness"),
                  "q0_no_answer": (ref or {}).get("no_answer"),
                  "baseline_rows": len(baseline_rows)},
    )


def probe_no_answer_flag(ledger: dict) -> dict:
    rows = _history(ledger)
    flagged = [row for row in rows if row.get("no_answer_flagged")]
    with_rate = [row for row in rows if row.get("centre_no_answer") is not None]
    if not with_rate:
        return _entry("no_answer_flag", "not_wired",
                      "flag a generation whose arms stop answering above the q0 baseline margin")
    return _entry(
        "no_answer_flag", _status(len(with_rate), bool(flagged), wired=True),
        "flag a generation whose centre or best candidate stops answering more than the "
        "q0 baseline by --no-answer-margin",
        fired=len(flagged), first=flagged[0]["generation"] if flagged else None,
        last=flagged[-1]["generation"] if flagged else None,
        effect=(f"flagged on {len(flagged)} generation(s)" if flagged else
                "watched every generation, never exceeded the margin"),
        evidence={"generations_watched": len(with_rate), "flagged_generations":
                  [row["generation"] for row in flagged],
                  "centre_no_answer_max": max((row.get("centre_no_answer") or 0.0)
                                              for row in with_rate),
                  "flags": [row["no_answer_flagged"] for row in flagged][:5]},
    )


def probe_retention(ledger: dict) -> dict:
    rows = _history(ledger)
    rollbacks = [row for row in rows if row.get("retention_rollback")]
    enabled = any(row.get("retention_band_source") is not None for row in rows) \
        or bool((ledger.get("summary") or {}).get("round2"))
    if not enabled:
        return _entry("retention", "not_wired",
                      "roll the centre step back when a family drops past the run's own "
                      "feedback noise band (round 2 §1)")
    restored = [row for row in _record_series(ledger, "retention_sha_restored") if row[1] is not None]
    faithful = [gen for gen, value in restored if value is False]
    return _entry(
        "retention", _status(len(rows), bool(rollbacks), wired=True),
        "roll the centre step back when a family drops past the run's own feedback noise band, "
        "re-anchoring the residual per ticket 09 §4",
        fired=len(rollbacks),
        first=rollbacks[0]["generation"] if rollbacks else None,
        last=rollbacks[-1]["generation"] if rollbacks else None,
        effect=(f"rolled back {len(rollbacks)} generation(s)" if rollbacks else
                "watched every generation, never beyond the band"),
        evidence={"generations_watched": len(rows),
                  "rollback_generations": [row["generation"] for row in rollbacks],
                  "breaches": [row.get("retention_breaches") for row in rollbacks][:10],
                  "band_sources": sorted({row.get("retention_band_source") for row in rows
                                          if row.get("retention_band_source")}),
                  "sha_restore_failures": faithful},
    )


def probe_feedback_slice(ledger: dict) -> dict:
    rows = _history(ledger)
    scored = [row for row in rows if row.get("feedback_slice_acc") is not None]
    if not scored:
        return _entry("feedback_slice", "not_wired",
                      "score a centre-only feedback slice once per generation (monitoring)")
    gated = [row for row in rows if row.get("retention_slice_role")]
    return _entry(
        "feedback_slice", "alive",
        "score a centre-only feedback slice once per generation (monitoring by default; "
        "only --feedback-slice-gate wires it into the retention decision)",
        fired=len(scored), first=scored[0]["generation"], last=scored[-1]["generation"],
        effect="monitoring channel fed on every generation",
        evidence={"generations_scored": len(scored),
                  "acc_min": min(row["feedback_slice_acc"] for row in scored),
                  "acc_max": max(row["feedback_slice_acc"] for row in scored),
                  "gate_role_recorded": len(gated)},
    )


def probe_budget(ledger: dict) -> dict:
    intervals = ledger.get("budget_intervals") or []
    rows = _gens(ledger)
    halts = [row.get("generation") for row in rows
             if (row.get("halt") or {}).get("rule") == "budget_exhausted"]
    if not intervals:
        return _entry("budget", "not_wired", "stop when the cumulative device budget is spent")
    return _entry(
        "budget", _status(len(intervals), bool(halts), wired=True),
        "stop when the cumulative device budget is spent (interval log survives restarts)",
        fired=len(intervals),
        effect=("budget never exhausted the run" if not halts else f"halted at {halts}"),
        evidence={"intervals": len(intervals), "halts": halts,
                  "last_wall_seconds": rows[-1].get("wall_seconds") if rows else None},
    )


def probe_stop_policy(ledger: dict) -> dict:
    rows = _gens(ledger)
    halts = [{"generation": row.get("generation"), "rule": (row.get("halt") or {}).get("rule")}
             for row in rows if row.get("halt")]
    stop_file = ledger.get("stop")
    if not rows:
        return _entry("stop_policy", "not_wired",
                      "apply ticket 13/18 §6.5 stop rules between generations")
    return _entry(
        "stop_policy", _status(len(rows), bool(halts) or bool(stop_file), wired=True),
        "apply the ticket 13/18 §6.5 stop rules between generations and write stop_ledger.jsonl",
        fired=len(rows), first=rows[0].get("generation"), last=rows[-1].get("generation"),
        effect=("halts: " + str(halts) if halts else
                "the in-process guard never halted: " +
                ("stop.json exists (external halt)" if stop_file else "stopped externally")),
        evidence={"ledger_rows": len(rows), "halts": halts, "stop_file": bool(stop_file)},
    )


def probe_mirror_gate(ledger: dict) -> dict:
    dropped = [(gen, value) for gen, value in _record_series(ledger, "dropped_pairs")
               if value is not None]
    rewards = [value for _g, value in _record_series(ledger, "pair_rewards") if value]
    with_drops = [gen for gen, value in dropped if len(value) > 0]
    partial = 0
    for block in rewards:
        if any(abs(float(reward)) < 1.0 for reward in block):
            partial += 1
    support = [(gen, value) for gen, value in _record_series(ledger, "gradient_support")
               if value is not None]
    return {
        "pair_gate": _entry("mirror_pair_gate", _status(len(dropped), bool(with_drops)),
                            "drop a mirror pair as a whole when either side breaches the "
                            "non-inferiority gate (tau)",
                            fired=sum(len(value) for _g, value in dropped),
                            first=with_drops[0] if with_drops else None,
                            last=with_drops[-1] if with_drops else None,
                            effect=(f"dropped pairs on {len(with_drops)} generation(s)"
                                    if with_drops else "no pair was ever dropped"),
                            evidence={"generations": len(dropped), "drop_generations":
                                      with_drops,
                                      "dropped_total": sum(len(value) for _g, value in dropped)}),
        "rank_reward": _entry("rank_shape_reward", _status(len(rewards), partial > 0),
                              "weight mirror pairs by the rank shape reward 1 - 2*rank/(M-1)",
                              fired=len(rewards),
                              effect=(f"{partial} generation(s) had a non-degenerate reward spread"
                                      if partial else "every generation was degenerate (M == 1?)"),
                              evidence={"generations": len(rewards),
                                        "degenerate_generations": len(rewards) - partial}),
        "gradient_support": _entry("gradient_support", _status(len(support),
                                  any(value for _g, value in support)),
                                   "the aggregation's support set is non-empty",
                                   fired=sum(1 for _g, value in support if value),
                                   effect="alive",
                                   evidence={"support_min": min((v for _g, v in support),
                                                                default=None),
                                             "support_max": max((v for _g, v in support),
                                                                default=None)}),
    }


def probe_centre_step(ledger: dict) -> dict:
    moved = [(gen, value) for gen, value in _record_series(ledger, "moved") if value is not None]
    empty = [gen for gen, value in moved if not value]
    return _entry(
        "centre_step", _status(len(moved), any(value for _g, value in moved)),
        "move the centre by alpha*g + gamma*e and book-keep the realised displacement",
        fired=sum(int(value) for _g, value in moved),
        first=moved[0][0] if moved else None, last=moved[-1][0] if moved else None,
        effect=f"{len(empty)} generation(s) moved nothing",
        evidence={"generations": len(moved), "moved_total": sum(int(v) for _g, v in moved),
                  "empty_generations": empty},
    )


def probe_confirm(ledger: dict) -> dict:
    confirm = ledger.get("confirm_arms")
    terminal = ledger.get("terminal_retention")
    if not confirm and not terminal:
        return _entry("terminal_arms", "never_fired",
                      "measure q0 and the centre on the sealed pool at the terminal step")
    return _entry(
        "terminal_arms", "alive",
        "measure q0 and the centre on the sealed pool in one process (confirm_gate consumes "
        "the headline arm pair)",
        fired=1,
        effect=("centre beats q0" if (terminal or {}).get("deliverable") == "center"
                else "no gain: baseline delivered" if terminal else "arms measured"),
        evidence={"prompt_variant": (confirm or {}).get("prompt_variant"),
                  "variants": (confirm or {}).get("variants"),
                  "arms": (confirm or {}).get("arms"),
                  "attribution": (confirm or {}).get("attribution"),
                  "terminal": terminal},
    )


def probe_prompt_swap(ledger: dict) -> dict:
    swap = ledger.get("prompt_swap")
    terminal = ledger.get("terminal_retention") or {}
    variants = ((ledger.get("confirm_arms") or {}).get("variants") or [])
    if not swap and len(variants) < 2:
        return _entry("prompt_swap_gate", "not_wired",
                      "measure every arm under both prompt forms on the sealed pool and state "
                      "whether the gain survives the swap")
    return _entry(
        "prompt_swap_gate", "alive",
        "measure every arm under both prompt forms on the sealed pool; a positive claim "
        "requires the gain not to vanish under the swap",
        fired=1,
        effect=str((swap or (terminal.get("swap") or {})).get("gate_passes")),
        evidence={"variants_measured": variants, "swap": swap or terminal.get("swap")},
    )


def probe_feedback_noise(ledger: dict) -> dict:
    blob = ledger.get("feedback_noise") or {}
    band = ((blob.get("feedback_channel") or {}).get("band")) or \
        ((ledger.get("summary") or {}).get("round2") or {}).get("feedback_noise")
    if not band:
        return _entry("feedback_noise_band", "not_wired",
                      "measure the run's own feedback noise band empirically and state the "
                      "retention threshold and claim rule against it")
    harness = band.get("harness") or {}
    return _entry(
        "feedback_noise_band", "alive",
        "measure the run's own feedback noise band empirically (repeated readings of the "
        "identical centre) and state the retention threshold against it",
        fired=int(band.get("replicates") or 0),
        effect=("harness deterministic at this precision; band = quantisation floor"
                if harness.get("deterministic") else "harness SE measured"),
        evidence={"replicates": band.get("replicates"), "source": band.get("source"),
                  "centre": band.get("centre"), "per_family": band.get("per_family"),
                  "harness": {"centre_se": harness.get("centre_se"),
                              "deterministic": harness.get("deterministic")}},
    )


def _entry(mechanism: str, status: str, intended_effect: str, fired: int = 0, first=None,
           last=None, effect: str | None = None, evidence: dict | None = None) -> dict:
    return {
        "mechanism": mechanism,
        "status": status,
        "intended_effect": intended_effect,
        "fired": fired,
        "first_generation": first,
        "last_generation": last,
        "effect": effect,
        "evidence": evidence or {},
    }


def audit(ledger: dict, kl_halt: float = 6.4, alpha_ladder_enabled: bool | None = None) -> dict:
    mechanisms = []
    mechanisms.append(probe_hard_gate(ledger))
    mechanisms.append(probe_readback(ledger))
    mechanisms.append(probe_kl_absolute(ledger, kl_halt))
    mechanisms.append(probe_kl_trend(ledger))
    ladder = probe_alpha_ladder(ledger, alpha_ladder_enabled)
    mechanisms.append(ladder["high"])
    mechanisms.append(ladder["low"])
    mechanisms.append(probe_no_accept(ledger))
    residuals = probe_residual_clamp(ledger)
    mechanisms.extend([residuals["clamp"], residuals["reject"], residuals["unmoved"]])
    mechanisms.append(probe_zero8(ledger))
    mechanisms.append(probe_q0_reference(ledger))
    mechanisms.append(probe_no_answer_flag(ledger))
    mechanisms.append(probe_retention(ledger))
    mechanisms.append(probe_feedback_slice(ledger))
    mechanisms.append(probe_feedback_noise(ledger))
    mechanisms.append(probe_budget(ledger))
    mechanisms.append(probe_stop_policy(ledger))
    mirror = probe_mirror_gate(ledger)
    mechanisms.extend([mirror["pair_gate"], mirror["rank_reward"], mirror["gradient_support"]])
    mechanisms.append(probe_centre_step(ledger))
    mechanisms.append(probe_confirm(ledger))
    mechanisms.append(probe_prompt_swap(ledger))
    counts: dict[str, int] = {}
    for entry in mechanisms:
        counts[entry["status"]] = counts.get(entry["status"], 0) + 1
    return {
        "protocol": PROTOCOL,
        "run_dir": ledger.get("run_dir"),
        "generations": len(_gens(ledger)),
        "history_rows": len(_history(ledger)),
        "mechanisms": mechanisms,
        "status_counts": counts,
        "dead_or_inert": [
            {"mechanism": entry["mechanism"], "status": entry["status"],
             "intended_effect": entry["intended_effect"], "effect": entry["effect"]}
            for entry in mechanisms
            if entry["status"] in ("fired_no_effect", "not_wired")
        ],
        "never_fired": [entry["mechanism"] for entry in mechanisms
                        if entry["status"] == "never_fired"],
    }


# --------------------------------------------------------------------------- #
# transport
# --------------------------------------------------------------------------- #


def run_projection_over_ssh(target: str, timeout: int = 180) -> dict:
    """Copy this script to /tmp on the host and run its projection over ssh.

    Read-only with respect to the run directory: the only remote writes are the
    script and the projection bundle, both in ``/tmp``.  Nothing under the run
    directory is touched, no GPU work is started, and only the small ledger files
    are read (the per-candidate and kernel-dump files are never opened).
    """
    host, remote = target.split(":", 1)
    if not re.fullmatch(r"[A-Za-z0-9_./~\-]+", remote):
        raise ValueError(f"unsafe remote path {remote!r}: only [A-Za-z0-9_./~-] is allowed")
    stamp = str(os.getpid())
    remote_script = f"/tmp/mechanism_audit_{stamp}.py"
    remote_bundle = f"/tmp/mechanism_audit_{stamp}.json"
    subprocess.run(["scp", "-q", str(Path(__file__).resolve()), f"{host}:{remote_script}"],
                   check=True)
    # the run path is left unquoted so the remote shell expands '~' (validated above)
    cmd = (f"python3 {remote_script} --project --run {remote} "
           f"--out {remote_bundle} >/dev/null 2>/tmp/mechanism_audit_{stamp}.err; rc=$?; "
           f"cat {remote_bundle} 2>/dev/null; "
           f"rm -f {remote_script} {remote_bundle} /tmp/mechanism_audit_{stamp}.err; exit $rc")
    result = subprocess.run(["ssh", "-o", "BatchMode=yes", host, cmd],
                            capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(f"remote projection failed: {result.stderr.strip()[:400]}")
    # stdout carries only the projection bundle
    text = result.stdout.strip()
    if not text:
        raise RuntimeError("remote projection produced no bundle")
    return json.loads(text)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", help="run output directory (local)")
    parser.add_argument("--ssh", help="HOST:RUN_DIR -- project read-only over ssh")
    parser.add_argument("--out", required=True)
    parser.add_argument("--project", action="store_true",
                        help="only project the ledgers (pure stdlib, read-only)")
    parser.add_argument("--kl-halt", type=float, default=6.4)
    parser.add_argument("--alpha-ladder", dest="alpha_ladder",
                        action=argparse.BooleanOptionalAction, default=None,
                        help="override whether the alpha ladder was enabled")
    parser.add_argument("--print", action="store_true", help="print the table")
    args = parser.parse_args()

    if args.project:
        if not args.run:
            raise SystemExit("--project needs --run")
        bundle = project(Path(args.run).expanduser())
        Path(args.out).expanduser().write_text(
            json.dumps(bundle, ensure_ascii=False, sort_keys=True) + "\n")
        print(json.dumps({"projected": args.run, "records": len(bundle.get("records") or []),
                          "history": len(bundle.get("history") or []),
                          "stop_ledger": len(bundle.get("stop_ledger") or [])}))
        return 0

    if args.ssh:
        ledger = run_projection_over_ssh(args.ssh)
    elif args.run:
        ledger = project(Path(args.run).expanduser())
    else:
        raise SystemExit("need --run or --ssh")

    report = audit(ledger, kl_halt=args.kl_halt, alpha_ladder_enabled=args.alpha_ladder)
    Path(args.out).expanduser().write_text(
        json.dumps(report, ensure_ascii=False, indent=1, sort_keys=True) + "\n")
    width = max(len(entry["mechanism"]) for entry in report["mechanisms"])
    for entry in report["mechanisms"]:
        print(f"{entry['mechanism']:<{width}}  {entry['status']:<17} fired={entry['fired']:<8}"
              f" {entry['effect']}")
    print(json.dumps({"status_counts": report["status_counts"],
                      "dead_or_inert": report["dead_or_inert"],
                      "never_fired": report["never_fired"]}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
