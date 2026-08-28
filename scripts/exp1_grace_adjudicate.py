"""exp1-grace ticket 05/06: ladder adjudication + criteria summary verdict.

Reads the arm verdicts written by exp1_grace_arm.py and applies the frozen
experiment-spec decision rules, writing one machine-readable adjudication
verdict (the sole judgement source, per run-conventions):

  recipe line    — a B1/B2/B3 arm qualifies iff ppl improves >=1.5% over B0
                   AND KLD improves in the same direction (kld_shadow < B0's).
                   Winner = lowest ppl among qualifiers. No qualifier ->
                   negative result (gate_degraded flags distinguish "recipe
                   ineffective" from "gate collapsed").
  native-4b line — Q1 must not lose to the same-structure PTQ-swap nvfp4
                   anchor (eval_shadow_precision.py measurement) and must
                   degrade <=2% vs the same-recipe bf16 shadow arm.
  30M confirm    — optional: the winning combo's 30M rerun must keep the
                   ordering vs B0, else the verdict is downgraded to
                   "insufficient signal".

    ssh <host> 'cd ~/Code/shadow-peft && .venv/bin/python \
      scripts/exp1_grace_adjudicate.py --runs-root runs/exp1 \
      --ptq-swap runs/exp1/grace_shadow_precision.json \
      --rerun runs/exp1/grace_Q1_30m/verdict.json'
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

LADDER = ("B1", "B2", "B3")
RECIPE_MIN_GAIN = 0.015  # -1.5% ppl vs B0
NATIVE_MAX_DEGRADATION = 0.02  # <=2% vs same-recipe bf16 shadow


def _load_arm(runs_root: Path, arm: str) -> dict:
    path = runs_root / f"grace_{arm}" / "verdict.json"
    if not path.exists():
        raise SystemExit(f"missing arm verdict: {path} (run exp1_grace_arm.py --arm {arm} first)")
    return json.loads(path.read_text())


def _load_ptq_swap(spec: str) -> float:
    """A bare number, or a path to eval_shadow_precision.py output (a list of
    {ckpt, mode, ppl} rows — the nvfp4 row on the B0 checkpoint is the anchor)."""
    try:
        return float(spec)
    except ValueError:
        rows = json.loads(Path(spec).read_text())
        cands = [r["ppl"] for r in rows if r.get("mode") == "nvfp4"]
        if not cands:
            raise SystemExit(f"no nvfp4 row found in {spec}") from None
        return float(cands[0])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default="runs/exp1")
    ap.add_argument("--ptq-swap", default=None,
                    help="same-structure PTQ-swap nvfp4 ppl: number or eval_shadow_precision.json path")
    ap.add_argument("--rerun", default=None, help="30M rerun verdict of the winning combo")
    ap.add_argument("--out", default=None, help="default <runs-root>/grace_adjudication/verdict.json")
    args = ap.parse_args()

    runs_root = Path(args.runs_root)
    arms = {arm: _load_arm(runs_root, arm) for arm in ("B0", *LADDER, "Q0", "Q1")}
    b0 = arms["B0"]

    ladder_rows = []
    for arm in LADDER:
        v = arms[arm]
        ppl_gain = (b0["ppl_shadow"] - v["ppl_shadow"]) / b0["ppl_shadow"]
        kld_gain = b0["kld_shadow"] - v["kld_shadow"]
        ladder_rows.append(
            {
                "arm": arm,
                "loss_mode": v["loss_mode"],
                "ppl_shadow": v["ppl_shadow"],
                "kld_shadow": v["kld_shadow"],
                "ppl_gain_vs_b0": round(ppl_gain, 4),
                "kld_gain_vs_b0": round(kld_gain, 4),
                "gate_degraded": v.get("gate_degraded"),
                "meets_ppl_line": ppl_gain >= RECIPE_MIN_GAIN,
                "kld_same_direction": kld_gain > 0,
                "qualified": ppl_gain >= RECIPE_MIN_GAIN and kld_gain > 0,
            }
        )
    qualified = [r for r in ladder_rows if r["qualified"]]
    winner = min(qualified, key=lambda r: r["ppl_shadow"]) if qualified else None

    verdict = {
        "b0": {"ppl_shadow": b0["ppl_shadow"], "kld_shadow": b0["kld_shadow"],
               "loss_mode": b0["loss_mode"]},
        "ladder": ladder_rows,
        "recipe_valid": winner is not None,
        "winner": winner["arm"] if winner else None,
        "negative_result": winner is None,
        "criteria": {
            "ppl_line": ">=1.5% ppl improvement vs B0",
            "kld_line": "KLD(bf16||quant+shadow) improves in the same direction",
        },
    }
    if winner is None:
        degraded = [r["arm"] for r in ladder_rows if r["gate_degraded"]]
        verdict["negative_result_note"] = (
            "gate collapsed on " + ", ".join(degraded) if degraded
            else "gate healthy on all arms — recipe ineffective in this setting"
        )

    native = None
    if winner is not None and args.ptq_swap:
        ptq_ppl = _load_ptq_swap(args.ptq_swap)
        q1 = arms["Q1"]
        bf16_ref = arms[winner["arm"]]["ppl_shadow"]
        degradation = (q1["ppl_shadow"] - bf16_ref) / bf16_ref
        beats_ptq = q1["ppl_shadow"] <= ptq_ppl
        native = {
            "ptq_swap_ppl": ptq_ppl,
            "q1_ppl": q1["ppl_shadow"],
            "q1_recipe": q1["loss_mode"],
            "q1_vs_bf16_same_recipe": round(degradation, 4),
            "q1_beats_ptq_swap": beats_ptq,
            "degradation_within_2pct": degradation <= NATIVE_MAX_DEGRADATION,
            "native_4b_valid": beats_ptq and degradation <= NATIVE_MAX_DEGRADATION,
            "q0_ppl": arms["Q0"]["ppl_shadow"],  # precision-effect attribution arm
        }
        verdict["native_4b"] = native

    if args.rerun:
        rerun = json.loads(Path(args.rerun).read_text())
        still_better = rerun["ppl_shadow"] < b0["ppl_shadow"]
        verdict["rerun_30m"] = {
            "path": args.rerun,
            "ppl_shadow": rerun["ppl_shadow"],
            "ordering_kept": still_better,
            "note": None if still_better else "ordering flipped at 30M — downgrade to insufficient signal",
        }

    out = Path(args.out or runs_root / "grace_adjudication" / "verdict.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(verdict, indent=2) + "\n")
    print(json.dumps(verdict, indent=2), flush=True)
    print(f"written {out}", flush=True)


if __name__ == "__main__":
    main()
