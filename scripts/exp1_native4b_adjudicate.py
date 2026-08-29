"""exp1-native4b adjudication: 30M confirmation + mechanism attribution.

Reads the arm verdicts (n4b namespace) plus the REUSED exp1-grace 10M
references and applies the frozen experiment-spec decision rules, writing one
machine-readable adjudication verdict (the sole judgement source, per
run-conventions). Pure JSON logic — no GPU, runs on the Mac.

  confirm-30m  — Q0_30m must beat B0_30m (ordering kept) and the 30M
                 PTQ-swap anchor; gain30/gain10 ratio = the saturation read
                 (<0.5 fading, 0.5-1 partial, >=1 unsaturated/grown).
  mechanism    — noise_share = Δ_N0 / Δ_Q0 at 10M on the pass's primary
                 axis (>=0.5 noise-dominant, <=0.25 grid-dominant, else
                 mixed; N0/N1 both flat while Δ_Q0 stands -> undecided).
  rl pass      — primary axis is axis2 (held-out self-distilled pool): 10M
                 refs from grace-rl-axis2_*, 30M/noise from n4b-rl-axis2_*.

    .venv/bin/python scripts/exp1_native4b_adjudicate.py --pass base --runs-root runs/exp1
    .venv/bin/python scripts/exp1_native4b_adjudicate.py --pass rl --runs-root runs-local/exp1-native4b
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

NOISE_SHARE_NOISE = 0.5
NOISE_SHARE_GRID = 0.25
FLAT_EPS = 0.001  # ppl gain below this reads as flat (0.1%)


def _load(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"missing verdict: {path}")
    return json.loads(path.read_text())


def _ptq_ppl(spec) -> float:
    """A bare number, an eval_shadow_precision list ({ckpt, mode, ppl} rows,
    the nvfp4 row is the anchor), or an axis2 PTQ-swap verdict dict."""
    if isinstance(spec, (int, float)):
        return float(spec)
    data = spec if isinstance(spec, dict) else json.loads(Path(spec).read_text())
    if isinstance(data, dict) and "ppl_shadow" in data:
        return float(data["ppl_shadow"])
    cands = [r["ppl"] for r in data if r.get("mode") == "nvfp4"]
    if not cands:
        raise SystemExit(f"no nvfp4 row found in {spec}")
    return float(cands[0])


def _gain(b0_ppl: float, arm_ppl: float) -> float:
    return (b0_ppl - arm_ppl) / b0_ppl


def _saturation_label(ratio: float) -> str:
    if ratio < 0.5:
        return "fading (gain halves or worse by 30M — regularization saturates with budget)"
    if ratio < 1.0:
        return "partial (gain persists at reduced relative size)"
    return "unsaturated/grown (gain holds or expands at 30M)"


def _mechanism(b0_10m: dict, q0_10m: dict, n0: dict | None, n1: dict | None) -> dict:
    q0_gain = _gain(b0_10m["ppl_shadow"], q0_10m["ppl_shadow"])
    out = {
        "b0_10m": b0_10m["ppl_shadow"],
        "q0_10m": q0_10m["ppl_shadow"],
        "q0_gain": round(q0_gain, 4),
        "criteria": {
            "noise_dominant": f"noise_share >= {NOISE_SHARE_NOISE}",
            "grid_dominant": f"noise_share <= {NOISE_SHARE_GRID} (a flat N0 IS the grid verdict: "
            "zero noise contribution, the grid explains the whole gain)",
            "mixed": "in between",
        },
    }
    if q0_gain < FLAT_EPS:
        out["attribution"] = "no_q0_signal"
        return out
    if n0 is None:
        out["attribution"] = "n0_missing"
        return out
    n0_gain = _gain(b0_10m["ppl_shadow"], n0["ppl_shadow"])
    out["n0"] = n0["ppl_shadow"]
    out["n0_gain"] = round(n0_gain, 4)
    out["noise_share"] = round(n0_gain / q0_gain, 4)
    if n1 is not None:
        n1_gain = _gain(b0_10m["ppl_shadow"], n1["ppl_shadow"])
        out["n1"] = n1["ppl_shadow"]
        out["n1_gain"] = round(n1_gain, 4)
        out["dose_direction"] = (
            "amplifies" if n1_gain > n0_gain + FLAT_EPS
            else "flat" if abs(n1_gain - n0_gain) <= FLAT_EPS
            else "degrades (matched level is the sweet spot — consistent with a noise optimum)"
        )
    if out["noise_share"] >= NOISE_SHARE_NOISE:
        out["attribution"] = "noise_dominant"
    elif out["noise_share"] <= NOISE_SHARE_GRID:
        out["attribution"] = "grid_dominant"
    else:
        out["attribution"] = "mixed"
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pass", dest="which_pass", required=True, choices=["base", "rl"])
    ap.add_argument("--runs-root", default="runs/exp1")
    ap.add_argument("--out", default=None,
                    help="default <runs-root>/n4b[_-rl]_adjudication/native4b_verdict.json")
    args = ap.parse_args()

    root = Path(args.runs_root)
    if args.which_pass == "base":
        b0_10m = _load(root / "grace_B0" / "verdict.json")            # wiki axis
        q0_10m = _load(root / "grace_Q0" / "verdict.json")
        b0_30m = _load(root / "n4b_B0_30m" / "verdict.json")
        q0_30m = _load(root / "n4b_Q0_30m" / "verdict.json")
        n0 = _load(root / "n4b_N0" / "verdict.json")
        n1_path = root / "n4b_N1" / "verdict.json"
        n1 = _load(n1_path) if n1_path.exists() else None
        ptq_10m = _ptq_ppl(str(root / "grace_shadow_precision.json"))
        ptq_30m = _ptq_ppl(str(root / "n4b_shadow_precision_30m.json"))
        axis = "despaced wikitext (primary, base pass)"
        prefix = "n4b"
    else:
        b0_10m = _load(root / "grace-rl-axis2_B0" / "verdict.json")   # axis2 = primary
        q0_10m = _load(root / "grace-rl-axis2_Q0" / "verdict.json")
        b0_30m = _load(root / "n4b-rl-axis2_B0_30m" / "verdict.json")
        q0_30m = _load(root / "n4b-rl-axis2_Q0_30m" / "verdict.json")
        n0 = _load(root / "n4b-rl-axis2_N0" / "verdict.json")
        n1 = None
        ptq_10m = None  # 10M PTQ anchor exists on the (distorted) wiki axis only
        ptq_30m = _ptq_ppl(str(root / "n4b-rl-axis2_B0_30m_ptq" / "verdict.json"))
        axis = "held-out self-distilled pool, axis2 (primary, rl pass)"
        prefix = "n4b-rl"

    gain_10m = _gain(b0_10m["ppl_shadow"], q0_10m["ppl_shadow"])
    gain_30m = _gain(b0_30m["ppl_shadow"], q0_30m["ppl_shadow"])
    ratio = gain_30m / gain_10m if gain_10m > 0 else float("inf")

    confirm = {
        "b0_30m": b0_30m["ppl_shadow"],
        "q0_30m": q0_30m["ppl_shadow"],
        "kld": {"b0_30m": b0_30m.get("kld_shadow"), "q0_30m": q0_30m.get("kld_shadow")},
        "q0_gain_10m": round(gain_10m, 4),
        "q0_gain_30m": round(gain_30m, 4),
        "saturation_ratio": round(ratio, 4) if ratio != float("inf") else None,
        "saturation": _saturation_label(ratio) if gain_10m > 0 else "no 10M gain to ratio against",
        "ordering_kept": q0_30m["ppl_shadow"] < b0_30m["ppl_shadow"],
        "ptq_swap_30m": ptq_30m,
        "q0_beats_ptq_30m": q0_30m["ppl_shadow"] <= ptq_30m,
    }
    confirm["confirmed"] = confirm["ordering_kept"] and confirm["q0_beats_ptq_30m"]
    if ptq_10m is not None:
        confirm["ptq_swap_10m"] = ptq_10m
        confirm["q0_beats_ptq_10m"] = q0_10m["ppl_shadow"] <= ptq_10m

    verdict = {
        "campaign": f"exp1-native4b/{args.which_pass}",
        "axis": axis,
        "confirm_30m": confirm,
        "mechanism": _mechanism(b0_10m, q0_10m, n0, n1),
        "reused_refs": {
            "b0_10m": b0_10m.get("arm", "B0"),
            "q0_10m": q0_10m.get("arm", "Q0"),
            "note": "10M references reused from exp1-grace (no re-run, per spec)",
        },
    }

    out = Path(args.out or root / f"{prefix}_adjudication" / "native4b_verdict.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(verdict, indent=2) + "\n")
    print(json.dumps(verdict, indent=2), flush=True)
    print(f"written {out}", flush=True)


if __name__ == "__main__":
    main()
