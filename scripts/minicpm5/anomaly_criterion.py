"""How much of the BF16 - q0 gap survives changes to the criterion / to the strata?

Four accounts of the same 85,815-item generation pool:
  A  as-scored (frozen criterion: every gold token must hit argmax)
  B  no-answer items excluded (a model that emits nothing is not scored on that item)
  C  no-answer items counted as wrong for BOTH arms (i.e. a "answered" rate folded in --
     identical to A for q0, so this shows the ceiling of the artifact)
  D  the balanced (confirm-style) reweighting actually used by the frozen confirm pool:
     0.5 * P(correct | q0 correct) + 0.5 * P(correct | q0 wrong)  -- reproduces T4
"""
from __future__ import annotations

import argparse
import collections
import json


def load(p):
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", default="/tmp/minicpm5-sanity-data/full.jsonl")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = list(load(a.full))

    def acc(sub, key):
        return sum(1 for r in sub if r[key]) / len(sub) if sub else None

    def table(sub):
        q = acc(sub, "tf_correct")
        b = acc(sub, "bf16_tf_correct")
        both = [r for r in sub if r["gen_n_tokens"] != 1 and r["bf16_gen_n_tokens"] != 1]
        qok = [r for r in sub if r["tf_correct"]]
        qbad = [r for r in sub if not r["tf_correct"]]
        bal = 0.5 * acc(qok, "bf16_tf_correct") + 0.5 * acc(qbad, "bf16_tf_correct")
        return dict(n=len(sub), q0=q, bf16=b, gap=b - q,
                    gap_answered_only=(acc(both, "bf16_tf_correct") - acc(both, "tf_correct")),
                    n_answered_only=len(both),
                    p_bf16_given_q0ok=acc(qok, "bf16_tf_correct"),
                    p_bf16_given_q0bad=acc(qbad, "bf16_tf_correct"),
                    balanced_prediction=bal, balanced_gap=bal - 0.5,
                    q0_noanswer=sum(1 for r in sub if r["gen_n_tokens"] == 1) / len(sub),
                    bf16_noanswer=sum(1 for r in sub if r["bf16_gen_n_tokens"] == 1) / len(sub))

    out = {}
    strata = sorted({r["stratum"] for r in rows})
    out["per_stratum"] = {s: table([r for r in rows if r["stratum"] == s]) for s in strata}
    out["all"] = table(rows)
    hard = [r for r in rows if r["stratum"] in ("logic/hard", "ordering/hard")]
    out["logic_hard+ordering_hard"] = table(hard)
    out["all_except_two_hard"] = table([r for r in rows
                                        if r["stratum"] not in ("logic/hard", "ordering/hard")])
    for fam in sorted({r["family"] for r in rows}):
        out[f"family:{fam}"] = table([r for r in rows if r["family"] == fam])

    # stratum contribution to the overall gap (item-weighted, as in the generation pool)
    n = len(rows)
    base = out["all"]
    contrib = {}
    for s in strata:
        t = out["per_stratum"][s]
        contrib[s] = dict(share=t["n"] / n, gap=t["gap"],
                          contribution_to_overall=t["gap"] * t["n"] / n)
    out["gap_decomposition"] = contrib
    out["gap_decomposition_sum"] = sum(v["contribution_to_overall"] for v in contrib.values())

    json.dump(out, open(a.out, "w"), indent=2, ensure_ascii=False, sort_keys=True)
    print(json.dumps({k: v for k, v in out.items() if k != "per_stratum"},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
