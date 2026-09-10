"""逐 stratum 的**题面容量探针**（v3 轮扩容前）：给定 attempt 预算，能造出多少条唯一题面？

用途（扩容的绑定约束）：confirm 放大到 ≈4,000 episode 后，每 stratum 需要
4×(4+confirm_ep+final_ep) 道**少数侧**题（q0 答对/答错各半），因此池容量必须支持
    N_s ≥ margin * 4*E_s / min(p_s, 1-p_s)。
本脚本给出每个 stratum 在 attempt 预算内可产出的唯一题干数（并报告撞到排除集/重复的次数）。

用法：
  python3 capacity_probe.py --out capacity_probe.json --attempts 600000 \
      --exclude-questions <v2 三池 jsonl ...>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_pool_v3 import gen_item_v3  # noqa: E402
from taskset_lib import FAMILIES, TIERS, verify  # noqa: E402


def probe_stratum(fam, tier, exclude, attempts, existing_questions=None):
    seen = set(existing_questions or ())
    n_new, dup, excl, bad = 0, 0, 0, 0
    t0 = time.time()
    for attempt in range(attempts):
        try:
            it, seed, src = gen_item_v3(fam, tier, attempt)
        except RuntimeError:
            continue
        q = it["question"]
        if q in exclude:
            excl += 1
            continue
        if q in seen:
            dup += 1
            continue
        ok, _, _ = verify(it, it["answer"])
        if not ok:
            bad += 1
            continue
        seen.add(q)
        n_new += 1
    return dict(unique_questions=n_new, attempts=attempts, collisions_within=dup,
                collisions_with_exclusion=excl, verifier_rejects=bad,
                seconds=round(time.time() - t0, 1),
                exhausted=(n_new + dup + excl + bad) < attempts * 0.02)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--attempts", type=int, default=600000)
    ap.add_argument("--exclude-questions", nargs="*", default=[])
    ap.add_argument("--plan-from", default=None, help="只探这些 stratum（JSON 文件 list）")
    a = ap.parse_args()

    exclude = set()
    for p in a.exclude_questions:
        if not p or not os.path.exists(p):
            continue
        with open(p) as f:
            for line in f:
                try:
                    exclude.add(json.loads(line)["question"])
                except Exception:
                    continue
    targets = json.load(open(a.plan_from)) if a.plan_from else [f"{f}/{t}" for f in FAMILIES for t in TIERS]

    out = {"attempts_per_stratum": a.attempts, "n_excluded_questions": len(exclude),
           "excluded_sources": a.exclude_questions, "per_stratum": {}}
    for s in targets:
        fam, tier = s.split("/")
        r = probe_stratum(fam, tier, exclude, a.attempts)
        out["per_stratum"][s] = r
        print(f"{s:16s} unique={r['unique_questions']:7d} attempts={r['attempts']:7d} "
              f"dup={r['collisions_within']:6d} excl={r['collisions_with_exclusion']:6d} "
              f"t={r['seconds']:6.1f}s", flush=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, sort_keys=True)
    print("total unique:", sum(v["unique_questions"] for v in out["per_stratum"].values()))


if __name__ == "__main__":
    main()
