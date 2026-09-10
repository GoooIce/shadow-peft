"""§6 dead-mechanism audit: unit tests on synthetic run ledgers (no GPU, no ssh).

The fixtures are small run directories that reproduce the two shapes that matter:

* ``report_only`` -- a run whose KL trend clause fired but was not acted on (the
  ticket-14 shape), with the residual clamp never biting, the alpha ladder firing
  one way only and no retention at all;
* ``retention_rollback`` -- a run where round-2 retention fired, restored the
  centre identity and re-anchored the residual, plus the slice monitoring channel.

Covers the projection (``--project`` reads only ledgers, so it can run against a
live run), the status vocabulary and the evidence each probe reports.

Usage:
  cd $(git rev-parse --show-toplevel)/scripts/minicpm5 && python test_mechanism_audit.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mechanism_audit as ma  # noqa: E402


def check(name, cond, detail=""):
    if not cond:
        raise AssertionError(f"{name}: {detail}")
    print(f"  ok  {name}")


def write(path: Path, blob) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(blob, ensure_ascii=False) + "\n")


def write_lines(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def gen_record(generation, *, moved=12, clamped=0, rejected=0, dropped_pairs=None,
               trend=False, gate_pass=True, zero8=0, raw_diffs=0, value_diffs=0,
               readback=0, retention=None, unmoved=0, rewards=None, support=100):
    return {
        "generation": generation,
        "group_id": f"band_{generation % 13:02d}",
        "alpha": 12.0,
        "residual_ledger": {
            "moved": moved, "touched": support, "rejected_beyond_grid": rejected,
            "residual_clamped": clamped, "unmoved_below_projection": unmoved,
            "residual_abs_max": 1.2, "u_abs_max": 3.0,
        },
        "aggregation": {
            "qualified_pairs": 12, "dropped_pairs": dropped_pairs or [],
            "pair_rewards": rewards if rewards is not None else [1.0, 0.5, 0.0, -0.5, -1.0],
            "gradient_support": support,
        },
        "hard_gate": {"gate": {"pass": gate_pass, "tensors": 1008, "value_diffs": value_diffs,
                               "zero8_diffs": zero8, "raw_code_diffs": raw_diffs,
                               "structural_mismatches": 0}},
        "live_readback": {"coordinates": 400, "value_mismatches": readback,
                          "engine_cache_raw_mismatches": 0, "zero8_representation": zero8},
        "kl": {"generation_kl_error": 2.0 + 0.01 * generation},
        "center_fitness": {"acc": 0.5},
        "retention": retention,
    }


def stop_row(generation, *, moved=12, trend=False, material=False, halt=None, alpha=12.0,
             switch=None, kl=2.0):
    return {
        "generation": generation, "moved": moved, "gradient_support": 480,
        "qualified_pairs": 12, "rejected_beyond_grid": 0, "cross_grid": 0.5,
        "reject_rate": 0.02, "kl": kl, "trend_breach": trend, "trend_material": material,
        "materiality_floor": 3.2019, "readback_value_mismatches": 0, "wall_seconds": 90.0,
        "alpha": alpha, "switch": switch, "halt": halt,
    }


def build_report_only_run(root: Path) -> Path:
    """Ticket-14 shape: trend fired twice, nothing was halted, clamp never bit."""
    run = root / "report_only"
    write(run / "summary.json", {"protocol": "minicpm5-rank-mirror/v1", "generations": 4,
                                 "alpha": 12.0, "tau": 0.02, "seed": 1, "wall_seconds": 400.0,
                                 "final_centre_sha256": "abc", "stop": None})
    write(run / "state.json", {"generation": 4, "gamma": 1.0, "clamp": 1.5,
                               "codes": {"l0": {"1": 2}}, "residual": {"l0": {"2": 0.5}}})
    history = [{"generation": 0, "kind": "centre_baseline", "center_acc": 0.5,
                "centre_no_answer": 0.01, "no_answer_flagged": []}]
    for generation in range(4):
        history.append({"generation": generation, "center_acc": 0.5,
                        "centre_no_answer": 0.02, "no_answer_flagged": []})
    write_lines(run / "history.jsonl", history)
    write_lines(run / "stop_ledger.jsonl",
                [stop_row(0), stop_row(1, trend=True), stop_row(2, trend=True),
                 stop_row(3, alpha=16.0, switch={"from": 12.0, "to": 16.0,
                                                 "trigger": "cross_grid x2"})])
    write_lines(run / "budget.jsonl", [{"pid": 1, "start": 0.0, "end": 400.0}])
    write(run / "records" / "q0_reference.json", {"taskset": "dev.jsonl",
                                                  "fitness": {"acc": 0.5},
                                                  "no_answer": {"overall": {"rate": 0.0}}})
    for generation in range(4):
        write(run / "records" / f"gen{generation:02d}.json",
              gen_record(generation, trend=generation in (1, 2), dropped_pairs=[3],
                         rewards=[1.0, -1.0]))
        write(run / f"kernel_values_gen{generation:02d}.json",
              {"tensors": 1008, "value_diffs": 0, "canonical_code_diffs": 5, "raw_code_diffs": 9,
               "zero8_diffs": 4, "structural_mismatches": 0, "pass_value": True,
               "tensors_with_value_diffs": 0})
    return run


def build_retention_run(root: Path) -> Path:
    """Round-2 shape: one rollback with a faithful restore, plus the slice channel."""
    run = root / "retention"
    write(run / "summary.json", {"protocol": "minicpm5-rank-mirror/v1", "generations": 3,
                                 "alpha": 8.0, "tau": 0.02, "seed": 2,
                                 "round2": {"protocol": "minicpm5-retention/v1",
                                            "feedback_noise": {"replicates": 3}}})
    write(run / "state.json", {"generation": 3, "codes": {}, "residual": {}})
    history = []
    for generation in range(3):
        history.append({
            "generation": generation, "center_acc": 0.5, "centre_no_answer": 0.01,
            "no_answer_flagged": [] if generation != 1 else [{"arm": "centre"}],
            "retention_rollback": generation == 1,
            "retention_breaches": ([{"family": "logic", "delta": -0.05, "band": 0.02}]
                                   if generation == 1 else []),
            "retention_worst_family": "logic" if generation == 1 else "arith",
            "retention_worst_delta": -0.05 if generation == 1 else 0.0,
            "retention_band_source": "startup_noop",
            "feedback_slice_acc": 0.51 + 0.001 * generation,
        })
    write_lines(run / "history.jsonl", history)
    write_lines(run / "stop_ledger.jsonl",
                [stop_row(0, switch={"from": 12.0, "to": 8.0, "trigger": "reject x2"}),
                 stop_row(1, alpha=8.0, moved=0), stop_row(2, alpha=8.0)])
    write(run / "records" / "q0_reference.json", {"taskset": "dev.jsonl",
                                                  "fitness": {"acc": 0.5}})
    retention = {
        "protocol": "minicpm5-retention/v1", "kind": "retention_rollback",
        "rollback": True, "breached_channels": ["feedback"],
        "channels": {"feedback": {"rollback": True, "worst_family": "logic",
                                  "worst_delta": -0.05,
                                  "breaches": [{"family": "logic", "delta": -0.05,
                                                "band": 0.02}]}},
        "band": {"source": "startup_noop"},
        "centre_post_acc": 0.45, "sha_restored_exactly": True,
        "reanchor": {"coordinates": 320, "rule": "ticket 09 §4"},
        "feedback_slice": {"role": "monitoring only", "gate": {"rollback": False}},
    }
    write(run / "records" / "gen00.json", gen_record(0, retention=None))
    write(run / "records" / "gen01.json", gen_record(1, retention=retention))
    write(run / "records" / "gen02.json",
          gen_record(2, retention={"protocol": "minicpm5-retention/v1",
                                   "kind": "retention_keep", "rollback": False,
                                   "channels": {"feedback": {"rollback": False,
                                                             "worst_family": "arith",
                                                             "worst_delta": 0.01,
                                                             "breaches": []}},
                                   "band": {"source": "startup_noop"},
                                   "feedback_slice": {"role": "monitoring only",
                                                      "gate": {"rollback": False}}}))
    for generation in range(3):
        write(run / f"kernel_values_gen{generation:02d}.json",
              {"tensors": 1008, "value_diffs": 0, "canonical_code_diffs": 0, "raw_code_diffs": 0,
               "zero8_diffs": 0, "structural_mismatches": 0, "pass_value": True,
               "tensors_with_value_diffs": 0})
    write(run / "feedback_noise.json", {"feedback_channel": {"band": {
        "replicates": 3, "source": "startup_noop",
        "centre": {"band": 0.003, "harness_se": 0.0, "quantisation_floor": 0.002},
        "per_family": {"logic": {"band": 0.02, "harness_se": 0.0,
                                 "quantisation_floor": 0.02}},
        "harness": {"centre_se": 0.0, "deterministic": True}}}})
    return run


def probe(report: dict, mechanism: str) -> dict:
    for entry in report["mechanisms"]:
        if entry["mechanism"] == mechanism:
            return entry
    raise AssertionError(f"mechanism {mechanism} not in the report")


# --------------------------------------------------------------------------- #
def test_projection_and_gate():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        run = build_report_only_run(root)
        ledger = ma.project(run)
        check("projection reads the ledgers", len(ledger["history"]) == 5
              and len(ledger["stop_ledger"]) == 4 and len(ledger["records"]) >= 4,
              (len(ledger["history"]), len(ledger["stop_ledger"]), len(ledger["records"])))
        check("projection records which files existed", ledger["files"]["summary.json"] is True
              and ledger["files"]["terminal_retention.json"] is False)
        check("projection is json-serialisable", bool(json.dumps(ledger)))
        check("projection keeps the state census",
              ledger["state"]["delta_coords"] == 1 and ledger["state"]["residual_coords"] == 1)
        report = ma.audit(ledger)
        gate = probe(report, "hard_gate_value")
        check("hard gate alive over every generation", gate["status"] == "alive"
              and gate["fired"] == 4 and gate["evidence"]["value_diffs_total"] == 0, gate)
        readback = probe(report, "readback")
        check("readback alive with 0 mismatches", readback["status"] == "alive"
              and readback["evidence"]["mismatch_generations"] == [], readback)
        mirror = probe(report, "mirror_pair_gate")
        check("mirror pair gate fired (pairs dropped)", mirror["status"] == "alive"
              and mirror["fired"] == 4 and mirror["evidence"]["drop_generations"] == [0, 1, 2, 3],
              mirror)


def test_report_only_is_flagged():
    with tempfile.TemporaryDirectory() as tmp:
        run = build_report_only_run(Path(tmp))
        report = ma.audit(ma.project(run))
        trend = probe(report, "kl_trend_halt")
        check("the report-only trend clause is fired_no_effect",
              trend["status"] == "fired_no_effect", trend)
        check("...and the evidence names the generations", trend["first_generation"] == 1
              and trend["last_generation"] == 2, trend)
        check("report-only mechanisms are surfaced", any(
            entry["mechanism"] == "kl_trend_halt" for entry in report["dead_or_inert"]))
        clamp = probe(report, "residual_clamp")
        check("the clamp never bit -> never_fired", clamp["status"] == "never_fired"
              and clamp["evidence"]["clamped_total"] == 0, clamp)
        zero8 = probe(report, "zero8_accounting")
        check("0<->8 channel observed and kept apart", zero8["status"] == "alive"
              and zero8["evidence"]["zero8_total"] == 16
              and zero8["evidence"]["raw_diffs_total"] > 0, zero8)
        high = probe(report, "alpha_ladder_high")
        check("ladder high fired", high["status"] == "alive" and high["fired"] == 1, high)
        low = probe(report, "alpha_ladder_low")
        check("ladder low never fired", low["status"] == "never_fired", low)
        keep = probe(report, "no_accept_halt")
        check("no-accept never triggered", keep["status"] == "never_fired"
              and keep["evidence"]["longest_empty_streak"] == 0, keep)
        confirm = probe(report, "terminal_arms")
        check("terminal arms never measured in this shape", confirm["status"] == "never_fired")
        retention = probe(report, "retention")
        check("retention reported as not wired", retention["status"] == "not_wired", retention)
        check("status counts add up", sum(report["status_counts"].values())
              == len(report["mechanisms"]))
        check("terminal arms and retention in the never_fired list",
              "terminal_arms" in report["never_fired"] and
              "retention" not in report["never_fired"])


def test_retention_run():
    with tempfile.TemporaryDirectory() as tmp:
        run = build_retention_run(Path(tmp))
        report = ma.audit(ma.project(run))
        retention = probe(report, "retention")
        check("retention alive", retention["status"] == "alive"
              and retention["fired"] == 1, retention)
        check("rollback generation recorded", retention["evidence"]["rollback_generations"] == [1])
        check("band source recorded", retention["evidence"]["band_sources"] == ["startup_noop"])
        check("no identity-restore failures", retention["evidence"]["sha_restore_failures"] == [])
        slice_entry = probe(report, "feedback_slice")
        check("feedback slice alive as a monitoring channel", slice_entry["status"] == "alive"
              and slice_entry["fired"] == 3, slice_entry)
        band = probe(report, "feedback_noise_band")
        check("noise band alive and deterministic-flagged", band["status"] == "alive"
              and band["evidence"]["harness"]["deterministic"] is True, band)
        no_answer = probe(report, "no_answer_flag")
        check("no-answer flag fired once", no_answer["status"] == "alive"
              and no_answer["fired"] == 1, no_answer)
        ladder_low = probe(report, "alpha_ladder_low")
        check("ladder low fired in this run", ladder_low["status"] == "alive", ladder_low)
        gate = probe(report, "hard_gate_value")
        check("gate alive", gate["status"] == "alive")
        check("nothing dead_or_inert from round 2",
              not any(entry["mechanism"] in ("retention", "feedback_slice", "feedback_noise_band")
                      for entry in report["dead_or_inert"]), report["dead_or_inert"])


def test_unfaithful_rollback_is_caught():
    with tempfile.TemporaryDirectory() as tmp:
        run = build_retention_run(Path(tmp))
        record = json.loads((run / "records" / "gen01.json").read_text())
        record["retention"]["sha_restored_exactly"] = False
        write(run / "records" / "gen01.json", record)
        report = ma.audit(ma.project(run))
        retention = probe(report, "retention")
        check("a rollback that did not restore the identity is reported",
              retention["evidence"]["sha_restore_failures"] == [1], retention)


def test_guard_record_shapes_tolerated():
    """The projection must survive the real ledger's two KL shapes and extra records."""
    with tempfile.TemporaryDirectory() as tmp:
        run = build_report_only_run(Path(tmp))
        # guard records carry `kl` as a bare float and repeat the generation's counters
        write(run / "records" / "stop_policy_gen00.json",
              {"generation": 0, "moved": 12, "kl": 2.0, "halt": None, "alpha": 12.0})
        write(run / "records" / "halt_record.json",
              {"rule": "kl_trend_breach", "generations": [1, 2]})
        ledger = ma.project(run)
        report = ma.audit(ledger)
        kl_series = [entry["kl"] for entry in ledger["records"]]
        check("guard-record KL floats are tolerated", all(isinstance(v, (int, float, type(None)))
                                                         for v in kl_series), kl_series)
        gate = probe(report, "hard_gate_value")
        check("guard records do not double-count generations", gate["fired"] == 4, gate)


def test_empty_run_degrades():
    with tempfile.TemporaryDirectory() as tmp:
        run = Path(tmp) / "empty"
        run.mkdir(parents=True)
        report = ma.audit(ma.project(run))
        check("an empty run produces a report instead of crashing",
              len(report["mechanisms"]) >= 15 and report["status_counts"].get("no_data", 0) >= 1,
              report["status_counts"])
        check("no_data mechanisms are not counted as dead_or_inert",
              all(entry["status"] != "no_data" for entry in report["dead_or_inert"]))


def test_alpha_ladder_override():
    with tempfile.TemporaryDirectory() as tmp:
        run = build_report_only_run(Path(tmp))
        report = ma.audit(ma.project(run), alpha_ladder_enabled=False)
        low = probe(report, "alpha_ladder_low")
        check("the override is accepted (status still derived from the ledger)",
              low["status"] == "never_fired")


def main():
    print("== 1. projection + live gates"); test_projection_and_gate()
    print("== 2. report-only mechanisms are flagged"); test_report_only_is_flagged()
    print("== 3. retention run"); test_retention_run()
    test_unfaithful_rollback_is_caught()
    print("== 4. degenerate inputs"); test_guard_record_shapes_tolerated()
    test_empty_run_degrades()
    test_alpha_ladder_override()
    print("ALL TESTS PASS")


if __name__ == "__main__":
    main()
