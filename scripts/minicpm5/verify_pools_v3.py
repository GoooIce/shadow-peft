"""v3 冻结件的**交付前自检**（把设计承诺逐条变成可复算的断言）。

检查项：
  1. 三池互斥（item 级与 episode 级）、池内无重复 id；
  2. 每个 episode 恰好 4 题 q0 对 + 4 题 q0 错（`episode_ok_count_span == [4,4]`）；
  3. 每个 stratum 的 item 级 q0 准确率 = 0.5000（零填充、两侧等量）；
  4. 三池逐题都有 q0 与 BF16 两臂的分数；
  5. 与 v2 三池 + v2 生成池 + v1 三池**零题干重叠**（"新鲜条目"的硬承诺）；
  6. decontamination 命中题**不在**任何冻结池里；
  7. sha256 与清单一致。
用法：python3 verify_pools_v3.py --dir <artifacts 目录> [--v2-gen pool_v2.jsonl --v2-pools ... --v1-pools ...]
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os

POOLS = ("dev", "confirm", "final")


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def load_questions(paths):
    q, ids = set(), set()
    for p in paths or []:
        if not p or not os.path.exists(p):
            continue
        for line in open(p):
            try:
                r = json.loads(line)
            except Exception:
                continue
            q.add(r["question"])
            ids.add(r["id"])
    return q, ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--v2-gen", default=None)
    ap.add_argument("--v2-pools", nargs="*", default=[])
    ap.add_argument("--v1-pools", nargs="*", default=[])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    man = json.load(open(os.path.join(a.dir, "pools_manifest.json")))
    q0 = {json.loads(l)["id"]: json.loads(l)
          for l in open(os.path.join(a.dir, "scores_q0_pool_v3_full.jsonl"))}
    bf = {json.loads(l)["id"]: json.loads(l)
          for l in open(os.path.join(a.dir, "scores_bf16_pool_v3_full.jsonl"))}
    rep = {"checks": {}, "ok": True}

    def check(name, ok, detail=None):
        rep["checks"][name] = dict(ok=bool(ok), detail=detail)
        if not ok:
            rep["ok"] = False

    pools = {}
    for p in POOLS:
        rows = [json.loads(l) for l in open(os.path.join(a.dir, f"taskset_{p}.jsonl"))]
        pools[p] = rows
        h = sha256_file(os.path.join(a.dir, f"taskset_{p}.jsonl"))
        check(f"sha256_{p}", h == man["pools"][p]["sha256"], h)
        check(f"n_items_{p}", len(rows) == man["pools"][p]["n_items"], len(rows))
        ids = [r["id"] for r in rows]
        check(f"unique_ids_{p}", len(set(ids)) == len(ids))
        check(f"scored_both_arms_{p}", all(i in q0 and i in bf for i in ids))
        # episode 组成：每簇 4 对 4 错
        span = collections.Counter()
        for ep in {r["episode"] for r in rows}:
            k = sum(1 for r in rows if r["episode"] == ep and q0[r["id"]]["tf_correct"])
            span[k] += 1
        check(f"episode_4ok_4bad_{p}", set(span) == {4}, dict(span))
        per_strat = collections.defaultdict(lambda: [0, 0])
        for r in rows:
            d = per_strat[r["stratum"]]
            d[0] += 1
            d[1] += int(bool(q0[r["id"]]["tf_correct"]))
        accs = {s: v[1] / v[0] for s, v in per_strat.items()}
        check(f"q0_balance_0p5_{p}", all(abs(x - 0.5) < 1e-9 for x in accs.values()),
              dict(min=min(accs.values()), max=max(accs.values()), n_strata=len(accs)))
    # 池间互斥
    idsets = {p: {r["id"] for r in pools[p]} for p in POOLS}
    epsets = {p: {r["episode"] for r in pools[p]} for p in POOLS}
    for i, p1 in enumerate(POOLS):
        for p2 in POOLS[i + 1:]:
            check(f"disjoint_ids_{p1}_{p2}", not (idsets[p1] & idsets[p2]),
                  len(idsets[p1] & idsets[p2]))
            check(f"disjoint_episodes_{p1}_{p2}", not (epsets[p1] & epsets[p2]),
                  len(epsets[p1] & epsets[p2]))
    # 与上一轮产物零题干重叠
    if a.v2_gen or a.v2_pools or a.v1_pools:
        prev_q, prev_ids = load_questions([a.v2_gen] + list(a.v2_pools) + list(a.v1_pools))
        for p in POOLS:
            ov = [r["id"] for r in pools[p] if r["question"] in prev_q]
            check(f"zero_question_overlap_with_prev_{p}", not ov, ov[:5])
    # 去污染命中不在冻结池里
    drep = os.path.join(a.dir, "decontamination_report_v3_scaled.json")
    if os.path.exists(drep):
        hits = set(json.load(open(drep))["hit_item_ids"])
        for p in POOLS:
            inter = [r["id"] for r in pools[p] if r["id"] in hits]
            check(f"no_decontam_hit_{p}", not inter, inter[:5])
        rl = set(json.load(open(drep))["rl2609_layer_report"]["hit_ids_in_rl2609_layer"])
        surv = [r["id"] for p in POOLS for r in pools[p] if r["id"] in rl]
        check("no_rl2609_layer_hit_in_frozen_pools", not surv, surv)
    # 与清单里的 per-stratum 组成一致（BF16 也核）
    for p in POOLS:
        rows = pools[p]
        for s, v in man["pools"][p]["per_stratum"].items():
            rs = [r for r in rows if r["stratum"] == s]
            q = sum(1 for r in rs if q0[r["id"]]["tf_correct"]) / len(rs)
            b = sum(1 for r in rs if bf[r["id"]]["tf_correct"]) / len(rs)
            check(f"{p}_{s}_manifest_acc_consistent",
                  abs(q - v["q0_tf_acc"]) < 1e-9 and abs(b - v["bf16_tf_acc"]) < 1e-9,
                  dict(manifest=[v["q0_tf_acc"], v["bf16_tf_acc"]], recomputed=[q, b]))
    if a.out:
        json.dump(rep, open(a.out, "w"), indent=2, ensure_ascii=False, sort_keys=True)
    fails = [k for k, v in rep["checks"].items() if not v["ok"]]
    print(f"checks: {len(rep['checks'])}, failed: {len(fails)}")
    for k in fails:
        print("  FAIL", k, rep["checks"][k]["detail"])
    print("ALL OK" if rep["ok"] else "SOME CHECKS FAILED")


if __name__ == "__main__":
    main()
