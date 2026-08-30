"""exp1-solo adjudication: dual-role trade-off verdict (pure JSON, runs on
the Mac).

Rows per pass: the lambda=0.05 refs (eval-only solo verdicts + their copied
recovery numbers) and the trained S3 (0.3) / S10 (1.0) arms. Recovery role
for the base pass = the arm's own wiki verdict; for the rl pass = the axis2
re-eval (primary axis), referenced against grace-rl-axis2_B0. Solo role is
always the runner's solo_* fields.

  best arm  — lowest solo_ppl among arms whose recovery degradation vs the
              10M same-pass B0 reference is <=2%
  verdict   — dual_role_viable | solo_at_recovery_cost | solo_not_learnable

    .venv/bin/python scripts/exp1_solo_adjudicate.py --pass base \
        --runs-root runs-local/exp1-solo
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REC_MAX_DEGRADATION = 0.02
SOLO_GAIN_MEANINGFUL = 0.30  # 30% solo ppl improvement over the 0.05 ref


def _load(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"missing verdict: {path}")
    return json.loads(path.read_text())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pass", dest="which_pass", required=True, choices=["base", "rl"])
    ap.add_argument("--runs-root", default="runs-local/exp1-solo")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = Path(args.runs_root)
    rl = args.which_pass == "rl"
    prefix = "solo-rl" if rl else "solo"
    ref_src = "grace-rl_B0" if rl else "grace_B0"

    # recovery reference: 10M same-pass B0 (base: wiki verdict; rl: axis2)
    if rl:
        rec_ref = _load(root / "grace-rl-axis2_B0" / "verdict.json")["ppl_shadow"]
    else:
        rec_ref = _load(root / f"{prefix}_ref_{ref_src}" / "verdict.json")["ppl_shadow"]
    solo_ref = _load(root / f"{prefix}_ref_{ref_src}" / "verdict.json")["solo_ppl"]

    rows = []
    for arm, lam in (("S3", 0.3), ("S10", 1.0)):
        v = _load(root / f"{prefix}_{arm}" / "verdict.json")
        if rl:
            rec = _load(root / f"solo-rl-axis2_{arm}" / "verdict.json")["ppl_shadow"]
            rec_axis = "axis2"
        else:
            rec = v["ppl_shadow"]
            rec_axis = "wiki"
        deg = (rec - rec_ref) / rec_ref
        gain = (solo_ref - v["solo_ppl"]) / solo_ref
        rows.append({
            "arm": arm,
            "solo_weight": lam,
            "recovery_ppl": rec,
            "recovery_axis": rec_axis,
            "recovery_degradation": round(deg, 4),
            "solo_ppl": v["solo_ppl"],
            "solo_kld": v["solo_kld"],
            "solo_gain_vs_005": round(gain, 4),
            "solo_sample": v.get("solo_sample", "")[:120],
        })

    # context: the 0.05 refs' solo numbers (all existing checkpoints)
    ctx = []
    for name in (ref_src, "grace-rl_Q0" if rl else "grace_Q0",
                 "n4b-rl_B0_30m" if rl else "n4b_B0_30m",
                 "n4b-rl_Q0_30m" if rl else "n4b_Q0_30m"):
        p = root / f"{prefix}_ref_{name}" / "verdict.json"
        if p.exists():
            r = _load(p)
            ctx.append({"ckpt": name, "solo_ppl": r["solo_ppl"], "solo_kld": r["solo_kld"]})

    eligible = [r for r in rows if r["recovery_degradation"] <= REC_MAX_DEGRADATION]
    best = min(eligible, key=lambda r: r["solo_ppl"]) if eligible else None
    if best is not None:
        verdict, best_arm = "dual_role_viable", best["arm"]
    elif any(r["solo_gain_vs_005"] >= SOLO_GAIN_MEANINGFUL for r in rows):
        verdict = "solo_at_recovery_cost"
        best_arm = min(rows, key=lambda r: r["solo_ppl"])["arm"]
    else:
        verdict, best_arm = "solo_not_learnable", None

    out = {
        "campaign": f"exp1-solo/{args.which_pass}",
        "recovery_ref": {"src": ref_src, "ppl": rec_ref,
                         "axis": "axis2" if rl else "wiki (despaced)"},
        "solo_ref_005": solo_ref,
        "arms": rows,
        "refs_context": ctx,
        "best_arm": best_arm,
        "verdict": verdict,
        "criteria": {
            "recovery_line": f"recovery degradation <= {REC_MAX_DEGRADATION:.0%} vs the 10M B0 ref",
            "best": "lowest solo_ppl among eligible arms",
            "solo_at_recovery_cost": f"solo gain >= {SOLO_GAIN_MEANINGFUL:.0%} but every arm "
            "degrades recovery beyond the line — trade-off curve delivered",
            "solo_not_learnable": f"no arm reaches {SOLO_GAIN_MEANINGFUL:.0%} solo gain — "
            "representation/head bottleneck, next entry: unfreeze the shadow head or two-phase",
        },
    }
    path = Path(args.out or root / f"{prefix}_adjudication" / "solo_verdict.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(out, indent=2, ensure_ascii=False), flush=True)
    print(f"written {path}", flush=True)


if __name__ == "__main__":
    main()
