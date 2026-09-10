"""BF16 anomaly check: rebuild the full per-item q0/BF16 score table for the v3 85,815-item pool.

The v3 ticket scored the 23,708-item first version of the pool in one pass and only the
62,107-item delta afterwards; the two parts are joined here by QUESTION (identical rule to
make_delta.py / merge_pool_scores.py: same question == same item, score depends only on the
prompt).

Usage:
  python3 anomaly_join.py --data /tmp/minicpm5-sanity-data --out /tmp/minicpm5-sanity-data/full.jsonl
"""
from __future__ import annotations

import argparse
import json
import os


def load(p):
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/tmp/minicpm5-sanity-data")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = a.data

    new = {it["id"]: it for it in load(os.path.join(d, "pool_v3_scaled.jsonl"))}
    old = {it["id"]: it for it in load(os.path.join(d, "pool_v3.jsonl"))}
    old_q2id = {it["question"]: it["id"] for it in old.values()}

    pass1 = {}
    for arm in ("q0", "bf16"):
        pass1[arm] = {r["id"]: r for r in load(os.path.join(d, f"scores_{arm}_pool_v3.jsonl"))}
    delta = {}
    for arm in ("q0", "bf16"):
        delta[arm] = {r["id"]: r for r in load(os.path.join(d, f"scores_{arm}_delta.jsonl"))}

    stats = {"new_n": len(new), "old_n": len(old), "reuse": 0, "delta": 0, "moved": 0,
             "missing": []}
    rows = []
    for iid, it in new.items():
        oid = old_q2id.get(it["question"])
        src = {}
        if oid is not None and oid in pass1["q0"]:
            stats["reuse"] += 1
            old_it = old[oid]
            if any(old_it.get(k) != it.get(k) for k in ("family", "tier", "stratum")):
                stats["moved"] += 1
            for arm in ("q0", "bf16"):
                src[arm] = pass1[arm][oid]
        elif iid in delta["q0"]:
            stats["delta"] += 1
            for arm in ("q0", "bf16"):
                src[arm] = delta[arm][iid]
        else:
            stats["missing"].append(iid)
            continue
        row = dict(id=iid, stratum=it["stratum"], family=it["family"], tier=it["tier"],
                   question=it["question"], answer=it["answer"],
                   meta=it.get("meta"), verify=it.get("verify"), source=it.get("source"))
        for arm in ("q0", "bf16"):
            s = src[arm]
            for k in ("tf_correct", "tf_token_hits", "tf_tokens", "tf_gold_logprob",
                      "tf_next_token", "tf_next_is_im_end", "gen_correct", "gen_text",
                      "gen_parsed", "gen_reason", "gen_finish", "gen_n_tokens",
                      "n_prompt_tokens"):
                row[f"{arm}_{k}" if arm != "q0" else k] = s.get(k)
        rows.append(row)

    rows.sort(key=lambda r: r["id"])
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")

    # reproduce the ticket's summaries as a check
    per = {}
    for arm, key in (("q0", "tf_correct"), ("bf16", "bf16_tf_correct")):
        n = len(rows)
        acc = sum(1 for r in rows if r[key]) / n
        per[arm] = acc
    stats["tf_acc_q0"] = per["q0"]
    stats["tf_acc_bf16"] = per["bf16"]
    stats["out"] = a.out
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
