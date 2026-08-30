"""exp1-rl1b adjudication: lr gate (100M A/B) + final scale-curve verdict.

Pure JSON logic — no GPU, runs on the Mac. Two stages:

  gate100 — the four 100M arms re-evaluated on axis2 (the PRIMARY axis for the
            rl line; the wiki axis inverted the B0/Q0 ordering on this model,
            so the lr decision must NOT trust it). Per family (B0/Q0): lower
            axis2 ppl wins; a difference <0.1% is a tie -> keep 1e-5. Families
            disagreeing -> the Q0 family (champion line) decides.
  final   — the gain-vs-budget curve {10M, 30M, 100M, 1B} (Q0 vs same-budget
            B0, axis2), the upper-bound read, hypothesis-2 (pool repetition vs
            budget-intrinsic overfit: B0_1b vs B0_100m with a 1B pool = no
            repetition), the 1B confirm triangle (+ PTQ-swap anchor) and the
            best operating point.

    .venv/bin/python scripts/exp1_rl1b_adjudicate.py --stage gate100 \
        --runs-root runs-local/exp1-rl1b
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

LR_TIE_EPS = 0.001  # <0.1% axis2 ppl difference = tie -> keep 1e-5


def _load(root: Path, name: str) -> dict:
    p = root / name / "verdict.json"
    if not p.exists():
        raise SystemExit(f"missing verdict: {p}")
    return json.loads(p.read_text())


def _gain(b0: float, arm: float) -> float:
    return (b0 - arm) / b0


def _gate100(root: Path) -> dict:
    fams = {}
    for fam in ("B0", "Q0"):
        lo = _load(root, f"rl1b-axis2_{fam}_100m")         # lr 1e-5
        hi = _load(root, f"rl1b-axis2_{fam}_100m_lr3")     # lr 3e-5
        diff = (lo["ppl_shadow"] - hi["ppl_shadow"]) / lo["ppl_shadow"]
        if abs(diff) < LR_TIE_EPS:
            verdict, win_lr = "tie", "1e-5"
        elif diff > 0:  # hi (3e-5) is lower -> better
            verdict, win_lr = "3e-5", "3e-5"
        else:
            verdict, win_lr = "1e-5", "1e-5"
        fams[fam] = {
            "ppl_lr1e-5": lo["ppl_shadow"],
            "ppl_lr3e-5": hi["ppl_shadow"],
            "rel_diff": round(diff, 4),
            "verdict": verdict,
            "win_lr": win_lr,
        }
    recommended = fams["Q0"]["win_lr"]
    if fams["B0"]["win_lr"] != recommended:
        note = "families disagree — Q0 (champion line) decides"
    else:
        note = "families agree"
    out = {
        "stage": "gate100",
        "axis": "held-out self-distilled pool (axis2, primary for the rl line)",
        "families": fams,
        "recommended_lr": recommended,
        "note": note,
        "1b_launch": f"PHASE=final1b FAMILY=B0 LR={recommended} (gooo); FAMILY=Q0 LR={recommended} (dc4a)",
    }
    return out


def _final(root: Path) -> dict:
    refs = {
        "10m": (_load(root, "grace-rl-axis2_B0"), _load(root, "grace-rl-axis2_Q0")),
        "30m": (_load(root, "n4b-rl-axis2_B0_30m"), _load(root, "n4b-rl-axis2_Q0_30m")),
    }
    gate = json.loads((root / "rl1b_adjudication" / "gate100.json").read_text())
    lr_tag = "" if gate["recommended_lr"] == "1e-5" else "_lr3"
    refs["100m"] = (
        _load(root, f"rl1b-axis2_B0_100m{lr_tag}"),
        _load(root, f"rl1b-axis2_Q0_100m{lr_tag}"),
    )
    refs["1b"] = (_load(root, "rl1b-axis2_B0_1b"), _load(root, "rl1b-axis2_Q0_1b"))
    ptq_1b = _load(root, "rl1b-axis2_B0_1b_ptq")

    curve = []
    for budget, (b0, q0) in refs.items():
        curve.append({
            "budget": budget,
            "b0_ppl": b0["ppl_shadow"],
            "q0_ppl": q0["ppl_shadow"],
            "q0_gain": round(_gain(b0["ppl_shadow"], q0["ppl_shadow"]), 4),
            "kld_b0": b0.get("kld_shadow"),
            "kld_q0": q0.get("kld_shadow"),
        })
    gains = {c["budget"]: c["q0_gain"] for c in curve}
    if gains["1b"] >= gains["100m"]:
        upper, upper_note = "not_capped", "gain at 1B >= 100M — next entry: scale further"
    elif gains["100m"] >= gains["30m"]:
        upper, upper_note = "peak_at_100m", "100M is the gain peak; declines by 1B"
    else:
        upper, upper_note = "peak_at_or_below_30m", "gain already declining at 100M"

    b0_1b, b0_100m = refs["1b"][0]["ppl_shadow"], refs["100m"][0]["ppl_shadow"]
    hyp2 = "repetition_was_culprit" if b0_1b <= b0_100m else "budget_intrinsic_overfit"
    q0_1b = refs["1b"][1]["ppl_shadow"]
    confirm = {
        "q0_1b_ppl": q0_1b,
        "b0_1b_ppl": b0_1b,
        "ordering_kept": q0_1b < b0_1b,
        "ptq_swap_1b_ppl": ptq_1b["ppl_shadow"],
        "q0_beats_ptq": q0_1b <= ptq_1b["ppl_shadow"],
    }
    confirm["confirmed"] = confirm["ordering_kept"] and confirm["q0_beats_ptq"]

    best = min(
        ({"budget": c["budget"], "q0_ppl": c["q0_ppl"]} for c in curve),
        key=lambda r: r["q0_ppl"],
    )
    # the 10M all-time champion so far (grace Q0) for continuity
    all_100m = [
        {"budget": "100m_lr1e-5", "q0_ppl": _load(root, "rl1b-axis2_Q0_100m")["ppl_shadow"]},
        {"budget": "100m_lr3e-5", "q0_ppl": _load(root, "rl1b-axis2_Q0_100m_lr3")["ppl_shadow"]},
    ]
    return {
        "stage": "final",
        "axis": "held-out self-distilled pool (axis2, primary)",
        "lr_selected": gate["recommended_lr"],
        "curve": curve,
        "upper_bound": upper,
        "upper_bound_note": upper_note,
        "hypothesis2_overfit": {
            "b0_100m": b0_100m,
            "b0_1b": b0_1b,
            "verdict": hyp2,
            "note": "1B pool has no repetition — a B0_1b regression vs B0_100m "
            "means the budget itself overfits, not the pool repetition",
        },
        "confirm_1b": confirm,
        "best_q0_point": best,
        "q0_100m_both_lrs": all_100m,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["gate100", "final"])
    ap.add_argument("--runs-root", default="runs-local/exp1-rl1b")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = Path(args.runs_root)
    verdict = _gate100(root) if args.stage == "gate100" else _final(root)
    out = Path(args.out or root / "rl1b_adjudication" / f"{args.stage}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(verdict, indent=2) + "\n")
    print(json.dumps(verdict, indent=2), flush=True)
    print(f"written {out}", flush=True)


if __name__ == "__main__":
    main()
