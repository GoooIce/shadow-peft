"""Build the probe item file for the BF16 early-termination check.

Groups (200 items each, deterministic stride sample out of the 85,815-item pool):
  logic_hard_as_is       -- reproducibility of the recorded empty-answer behaviour
  ordering_hard_as_is    -- idem
  arith_easy_as_is       -- control: BF16 answers these 100% of the time today
  arith_easy_filler1     -- same task, ~+100 neutral prompt tokens
  arith_easy_filler2     -- same task, ~+170 neutral prompt tokens
  arith_easy_filler1_note-- filler + an explicit "answer with a number" reminder
  logic_hard_noanswerpfx -- same items, template prompt WITHOUT the "Answer:" prefix
  ordering_hard_noanswerpfx -- idem

The frozen pool files are only read; every variant is built in this scratch file.
"""
from __future__ import annotations

import argparse
import json

FILLER = (" Note: the sentence that follows carries no information about the problem above and "
          "may be ignored entirely. It is included only to make this message longer, and it "
          "contains no numbers, no names, and nothing that changes the question.")

REMINDER = " Remember: you must state the final answer as a single number."


def load(p):
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def stride_pick(rows, n, stride):
    """deterministic sample: take every `stride`-th row, then trim/pad to n."""
    out = rows[::stride]
    return out[:n] if len(out) >= n else rows[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="/tmp/minicpm5-sanity-data/pool_v3_scaled.jsonl")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    by_stratum = {}
    for r in load(a.pool):
        by_stratum.setdefault(r["stratum"], []).append(r)
    for k in by_stratum:
        by_stratum[k].sort(key=lambda r: r["id"])

    def take(stratum, n, stride):
        rows = by_stratum[stratum]
        out = rows[::stride][:n]
        assert len(out) == n, (stratum, len(out), len(rows))
        return out

    groups = {}
    lh = take("logic/hard", a.n, 33)
    oh = take("ordering/hard", a.n, 42)
    ae = take("arith/easy", a.n, 34)
    groups["logic_hard_as_is"] = (lh, {})
    groups["ordering_hard_as_is"] = (oh, {})
    groups["arith_easy_as_is"] = (ae, {})
    groups["arith_easy_filler1"] = (ae, dict(filler=1, reminder=False))
    groups["arith_easy_filler2"] = (ae, dict(filler=3, reminder=False))
    groups["arith_easy_filler1_note"] = (ae, dict(filler=1, reminder=True))
    groups["logic_hard_noanswerpfx"] = (lh, dict(strip_answer_prefix=True))
    groups["ordering_hard_noanswerpfx"] = (oh, dict(strip_answer_prefix=True))

    out = []
    for gname, (rows, opt) in groups.items():
        for r in rows:
            it = dict(r)
            q = r["question"]
            if opt.get("filler"):
                q = q + FILLER * opt["filler"] + ("\n\n" if opt.get("reminder") else "")
            if opt.get("reminder"):
                q = q + REMINDER.strip()
            it["question"] = q
            it["_group"] = gname
            it["_opt"] = {k: bool(v) for k, v in opt.items()}
            out.append(it)

    with open(a.out, "w") as f:
        for it in out:
            f.write(json.dumps(it, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({"n_items": len(out),
                      "groups": {g: len(v[0]) for g, v in groups.items()}}, indent=2))


if __name__ == "__main__":
    main()
