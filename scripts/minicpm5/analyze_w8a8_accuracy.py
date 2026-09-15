#!/usr/bin/env python3
"""Three-arm accuracy table for the W8A8 ticket on the frozen v3 dev pool.

Inputs (all produced by the frozen scorer, `--arm tf`, on
`taskset_dev.jsonl`, sha256 e7a7d29a0b5cb90a...):

  scores_<arm>_dev3.jsonl            frozen-protocol run (this ticket)
  scores_<arm>_dev3_summary.json     its summary
  instr_scores_<arm>_dev3.jsonl      same protocol, scorer with the two
                                     additive diagnostics (tf_first_argmax,
                                     no-answer rate); used ONLY to decompose
                                     the early-termination artifact

The script asserts that `tf_correct` is item-for-item identical between the two
scorer versions -- that is what licenses reading the second as "the same
protocol plus instrumentation" rather than a second protocol.

Outputs a JSON report and a markdown table.

Usage:
    python analyze_w8a8_accuracy.py --dir <where the score files live> \
        --out-json <report.json> --out-md <table.md>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

IM_END = 130073
ARMS = ("bf16", "w4a4", "w8a8")


def load(path: Path) -> dict:
    return {json.loads(line)["id"]: json.loads(line) for line in path.open()}


def acc(rows: dict, ids: list[str], key: str = "tf_correct") -> float:
    return sum(int(bool(rows[i].get(key))) for i in ids) / len(ids) if ids else None


def slice_ids(items: list[dict], field: str, value: str) -> list[str]:
    return [it["id"] for it in items if it[field] == value]


def is_no_answer(row: dict) -> bool | None:
    """19-ticket definition; None when the field is absent (frozen scorer)."""
    if row.get("tf_first_argmax") is None:
        return None
    return int(row["tf_first_argmax"]) == IM_END


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    parser.add_argument("--prefix", default="scores_")
    parser.add_argument("--instr-prefix", default="instr_scores_")
    parser.add_argument("--instr-dir", default=None,
                        help="directory holding the instrumented score files "
                             "(default: same as --dir)")
    args = parser.parse_args()

    root = Path(args.dir).expanduser()
    items = [json.loads(line) for line in (root / "taskset_dev.jsonl").open()]
    ids = [it["id"] for it in items]

    frozen = {arm: load(root / f"{args.prefix}{arm}_dev3.jsonl") for arm in ARMS}
    instr_root = Path(args.instr_dir).expanduser() if args.instr_dir else root
    instrumented = {
        arm: load(instr_root / f"{args.instr_prefix}{arm}_dev3.jsonl")
        for arm in ARMS
    }

    report: dict = {
        "pool": str(root / "taskset_dev.jsonl"),
        "n_items": len(items),
        "frozen_dir": str(root),
        "instrumented_dir": str(instr_root),
    }

    # --- agreement between the two scorer versions --------------------------
    agreement = {}
    for arm in ARMS:
        diffs = [i for i in ids if bool(frozen[arm][i]["tf_correct"])
                 != bool(instrumented[arm][i].get("tf_correct"))]
        hits = [i for i in ids
                if frozen[arm][i].get("tf_token_hits") != instrumented[arm][i].get("tf_token_hits")]
        agreement[arm] = {
            "tf_correct_mismatches": len(diffs),
            "tf_token_hits_mismatches": len(hits),
            "identical": not diffs and not hits,
        }
    report["scorer_version_agreement"] = agreement

    # --- headline table -----------------------------------------------------
    table = {}
    for arm in ARMS:
        table[arm] = {
            "tf_acc": acc(frozen[arm], ids),
            "tf_acc_instrumented": acc(instrumented[arm], ids),
            "gen_acc": (
                acc(frozen[arm], ids, "gen_correct")
                if any("gen_correct" in frozen[arm][i] for i in ids) else None
            ),
            "tf_format_im_end_rate": sum(
                1 for i in ids if frozen[arm][i].get("tf_next_is_im_end")
            ) / len(ids),
            "gold_logprob_mean": (
                sum(r["tf_gold_logprob"] for r in frozen[arm].values()
                    if r.get("tf_gold_logprob") is not None)
                / max(1, sum(1 for r in frozen[arm].values()
                             if r.get("tf_gold_logprob") is not None))
            ),
        }
    report["table"] = table

    gaps = {}
    for a, b in (("bf16", "w8a8"), ("bf16", "w4a4"), ("w8a8", "w4a4")):
        gaps[f"{a}_minus_{b}"] = {
            "tf_points": (table[a]["tf_acc"] - table[b]["tf_acc"]) * 100,
        }
    report["gaps"] = gaps

    # --- per family / per stratum ------------------------------------------
    for field in ("family", "stratum", "tier"):
        per: dict = {}
        for value in sorted({it[field] for it in items}):
            sub = slice_ids(items, field, value)
            per[value] = {"n": len(sub)}
            for arm in ARMS:
                per[value][arm] = round(acc(frozen[arm], sub), 4)
            per[value]["bf16_minus_w8a8_pts"] = round(
                (acc(frozen["bf16"], sub) - acc(frozen["w8a8"], sub)) * 100, 2
            )
            per[value]["bf16_minus_w4a4_pts"] = round(
                (acc(frozen["bf16"], sub) - acc(frozen["w4a4"], sub)) * 100, 2
            )
        report[f"per_{field}"] = per

    # --- paired tests between arms (same 480 items, McNemar exact) ----------
    try:
        from math import comb

        def mcnemar(a_rows, b_rows, subset=None):
            sub = subset if subset is not None else ids
            b_win = sum(1 for i in sub
                        if bool(a_rows[i].get("tf_correct"))
                        and not bool(b_rows[i].get("tf_correct")))
            c_win = sum(1 for i in sub
                        if not bool(a_rows[i].get("tf_correct"))
                        and bool(b_rows[i].get("tf_correct")))
            n = b_win + c_win
            if n == 0:
                p = 1.0
            else:
                k = min(b_win, c_win)
                p = sum(comb(n, j) for j in range(k + 1)) / (2 ** n) * 2
                p = min(1.0, p)
            return {"discordant_a_only": b_win, "discordant_b_only": c_win,
                    "n_discordant": n, "mcnemar_exact_two_sided_p": round(p, 4)}

        paired = {}
        for a, b in (("bf16", "w8a8"), ("bf16", "w4a4"), ("w8a8", "w4a4")):
            paired[f"{a}_vs_{b}"] = mcnemar(frozen[a], frozen[b])
        report["paired_mcnemar_tf"] = paired
    except Exception as error:  # pragma: no cover
        report["paired_mcnemar_tf_error"] = f"{type(error).__name__}: {error}"

    # --- no-answer decomposition (instrumented run only) --------------------
    na = {}
    for arm in ARMS:
        rows = instrumented[arm]
        na[arm] = {
            "overall": round(sum(1 for i in ids if is_no_answer(rows[i]))
                             / len(ids), 4),
        }
        for field in ("family", "tier"):
            na[arm][f"per_{field}"] = {}
            for value in sorted({it[field] for it in items}):
                sub = slice_ids(items, field, value)
                na[arm][f"per_{field}"][value] = round(
                    sum(1 for i in sub if is_no_answer(rows[i])) / len(sub), 4
                )
    report["no_answer_rate"] = na

    # gaps restricted to items where both arms actually answered
    both_answered = {}
    for a, b in (("bf16", "w8a8"), ("bf16", "w4a4")):
        subset = [
            i for i in ids
            if is_no_answer(instrumented[a][i]) is False
            and is_no_answer(instrumented[b][i]) is False
        ]
        both_answered[f"{a}_vs_{b}"] = {
            "n": len(subset),
            f"acc_{a}": round(acc(instrumented[a], subset), 4),
            f"acc_{b}": round(acc(instrumented[b], subset), 4),
            "gap_pts": round(
                (acc(instrumented[a], subset) - acc(instrumented[b], subset)) * 100, 2
            ),
        }
    report["both_answered_subset"] = both_answered

    # bf16 with its no-answer items recovered at the rate it answers the rest
    bf = instrumented["bf16"]
    answered = [i for i in ids if is_no_answer(bf[i]) is False]
    na_items = [i for i in ids if is_no_answer(bf[i])]
    report["bf16_artifact_arithmetic"] = {
        "no_answer_items": len(na_items),
        "acc_on_answered_items": round(acc(bf, answered), 4),
        "acc_on_no_answer_items": round(acc(bf, na_items), 4),
        "bf16_acc_if_no_answer_items_scored_at_answered_rate": round(
            (acc(bf, answered) * len(answered)
             + acc(bf, answered) * len(na_items)) / len(ids), 4
        ),
    }

    # --- item difficulty of the items one arm refused to answer ------------
    # The pool's headline numbers are contaminated by the frozen `Answer:`
    # suffix: an arm that ends the turn there scores 0 on that item.  For the
    # fenced arm, the *other* arms' accuracy on exactly those items is an
    # observed difficulty proxy, which bounds what the fenced arm would have
    # scored had it answered.
    proxy = {}
    for fenced in ARMS:
        rows = instrumented[fenced]
        silent = [i for i in ids if is_no_answer(rows[i])]
        proxy[fenced] = {"n_silent": len(silent)}
        for arm in ARMS:
            proxy[fenced][f"acc_{arm}_on_silent_items"] = round(
                acc(instrumented[arm], silent), 4
            ) if silent else None
        if silent:
            others = [a for a in ARMS if a != fenced]
            for arm in others:
                answered = [i for i in ids if is_no_answer(rows[i]) is False]
                recovered = (
                    acc(instrumented[fenced], answered) * len(answered)
                    + acc(instrumented[arm], silent) * len(silent)
                ) / len(ids)
                proxy[fenced][f"{fenced}_acc_if_silent_scored_at_{arm}_rate"] = round(
                    recovered, 4
                )
    report["refusal_difficulty_proxy"] = proxy

    out_json = Path(args.out_json).expanduser()
    out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    lines = []
    lines.append("| arm | TF acc | GEN acc | im_end-after-gold | no-answer rate |")
    lines.append("| --- | --- | --- | --- | --- |")
    for arm in ARMS:
        gen = table[arm]["gen_acc"]
        gen_text = "n/a" if gen is None else f"{gen:.4f}"
        lines.append(
            f"| {arm} | {table[arm]['tf_acc']:.4f} | {gen_text} | "
            f"{table[arm]['tf_format_im_end_rate']:.4f} | {na[arm]['overall']:.4f} |"
        )
    lines.append("")
    lines.append("| family | n | bf16 | w8a8 | w4a4 | bf16-w8a8 (pts) | bf16-w4a4 (pts) |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for value, row in report["per_family"].items():
        lines.append(
            f"| {value} | {row['n']} | {row['bf16']:.4f} | {row['w8a8']:.4f} | "
            f"{row['w4a4']:.4f} | {row['bf16_minus_w8a8_pts']:+.2f} | "
            f"{row['bf16_minus_w4a4_pts']:+.2f} |"
        )
    lines.append("")
    lines.append("| stratum | n | bf16 | w8a8 | w4a4 | bf16-w8a8 (pts) | bf16-w4a4 (pts) |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for value, row in report["per_stratum"].items():
        lines.append(
            f"| {value} | {row['n']} | {row['bf16']:.4f} | {row['w8a8']:.4f} | "
            f"{row['w4a4']:.4f} | {row['bf16_minus_w8a8_pts']:+.2f} | "
            f"{row['bf16_minus_w4a4_pts']:+.2f} |"
        )
    lines.append("")
    lines.append("| pair | A-only wins | B-only wins | McNemar exact p |")
    lines.append("| --- | --- | --- | --- |")
    for pair, row in report.get("paired_mcnemar_tf", {}).items():
        lines.append(
            f"| {pair} | {row['discordant_a_only']} | {row['discordant_b_only']} | "
            f"{row['mcnemar_exact_two_sided_p']} |"
        )
    lines.append("")
    for pair, row in report["both_answered_subset"].items():
        lines.append(
            f"- both-answered {pair}: n={row['n']}, gap={row['gap_pts']:+.2f} pts"
        )
    lines.append(
        f"- bf16 artifact arithmetic: {json.dumps(report['bf16_artifact_arithmetic'])}"
    )
    md = "\n".join(lines) + "\n"
    Path(args.out_md).expanduser().write_text(md)
    print(md)
    print(f"[analyze] wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
