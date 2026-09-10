"""把"上一版池的分数（按题干复用）"与"delta 的分数"合并成扩容池的完整逐题分数。

复用规则与 make_delta.py 一致：同一题干 = 同一道题（模型得分只取决于题面）；条目在
stratum 间移动时，以新池的 family/tier/stratum 标签为准写回。

用法：
  python3 merge_pool_scores.py --new-pool pool_v3.jsonl --map delta_map_full.json \
      --old-scores scores_q0_pool_v3.jsonl --delta-scores scores_q0_delta.jsonl \
      --out scores_q0_pool_v3_full.jsonl \
      [--summary scores_q0_pool_v3_full_summary.json]
"""
from __future__ import annotations

import argparse
import collections
import json


def load(p):
    return [json.loads(l) for l in open(p)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new-pool", required=True)
    ap.add_argument("--map", required=True, help="make_delta.py 的 *_full.json（新 id -> 旧 id）")
    ap.add_argument("--old-scores", required=True)
    ap.add_argument("--delta-scores", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--summary", default=None)
    a = ap.parse_args()

    new = {it["id"]: it for it in load(a.new_pool)}
    remap = json.load(open(a.map))
    old = {r["id"]: r for r in load(a.old_scores)}
    delta = {r["id"]: r for r in load(a.delta_scores)}

    out_rows, n_reuse, n_delta, n_moved = [], 0, 0, 0
    for iid, it in new.items():
        src = None
        oid = remap.get(iid)
        if oid and oid in old:
            src = old[oid]
            n_reuse += 1
            if any(src.get(k) != it.get(k) for k in ("family", "tier", "stratum")):
                n_moved += 1
        elif iid in delta:
            src = delta[iid]
            n_delta += 1
        else:
            raise SystemExit(f"missing score for {iid}")
        r = dict(src)
        r["id"] = iid
        r["family"], r["tier"], r["stratum"] = it["family"], it["tier"], it["stratum"]
        out_rows.append(r)
    out_rows.sort(key=lambda r: r["id"])
    with open(a.out, "w") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")

    per_stratum = collections.defaultdict(lambda: [0, 0])
    for r in out_rows:
        d = per_stratum[r["stratum"]]
        d[0] += 1
        d[1] += int(bool(r.get("tf_correct")))
    summary = dict(n=len(out_rows), reused=n_reuse, scored_delta=n_delta,
                   stratum_relabelled=n_moved,
                   tf_acc=(sum(v[1] for v in per_stratum.values()) / len(out_rows)),
                   per_stratum={s: dict(n=v[0], acc=v[1] / v[0]) for s, v in sorted(per_stratum.items())},
                   note="q0/BF16 分数按题干复用 + delta 新评；stratum 标签以新池为准")
    if a.summary:
        json.dump(summary, open(a.summary, "w"), indent=2, ensure_ascii=False, sort_keys=True)
    print(json.dumps({k: v for k, v in summary.items() if k != "per_stratum"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
