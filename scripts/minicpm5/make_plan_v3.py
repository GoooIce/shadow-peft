"""v3 扩容用的**规模计划**（可复算）：由实测 q0 难度 + 目标 episode 结构推出逐 stratum 生成量。

目标（协调者口径，2026-09-10 扩容指令）：
  * confirm ≈ 4,000 episode（每 stratum 267，共 4,005）→ 二值主轴 δ_min ≈ 1.0 点（σ≈0.227）；
  * final 按比例放大（v2 是 60→10，即 1/6）：267/6 ≈ 45（每 stratum）；
  * dev 不动（4 episode / stratum，480 题）——它是每候选成本。

绑定约束 = **少数侧容量**：每个 episode 需要 4 道 q0 答对 + 4 道 q0 答错，所以每 stratum 的
池子里**两侧各需** 4×(4+267+45) = 1,264 题，即
    N_s ≥ margin * 4*E_s / min(p_s, 1 - p_s)，  p_s = 该 stratum 的 q0 准确率（实测）。

用法：
  python3 make_plan_v3.py --scores scores_q0_pool_v3.jsonl --out plan_v3_scaled.json
      [--dev 4 --confirm 267 --final 45 --margin 1.25]
"""
from __future__ import annotations

import argparse
import collections
import json
import math

CAPACITY = {   # 400k attempt 容量探针（capacity_probe.py，排除 v2 三池题干）
    "arith/easy": 95243, "arith/medium": 132404, "arith/hard": 393748,
    "pct/easy": 42471, "pct/medium": 5540, "pct/hard": 60573,
    "logic/easy": 322282, "logic/medium": 399885, "logic/hard": 399995,
    "ordering/easy": 398060, "ordering/medium": 399997, "ordering/hard": 400000,
    "modulo/easy": 41526, "modulo/medium": 56182, "modulo/hard": 383176,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True, help="q0 在现有池上的逐题分数")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dev", type=int, default=4)
    ap.add_argument("--confirm", type=int, default=267)
    ap.add_argument("--final", type=int, default=45)
    ap.add_argument("--margin", type=float, default=1.25)
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.scores)]
    per = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        d = per[r["stratum"]]
        d[0] += 1
        d[1] += int(bool(r["tf_correct"]))
    E = a.dev + a.confirm + a.final
    per_side = 4 * E
    plan, table = {}, {}
    for s in sorted(per):
        n, k = per[s]
        p = k / n
        need = math.ceil(a.margin * per_side / min(p, 1 - p))
        cap = CAPACITY.get(s)
        plan_s = min(need, cap) if cap else need
        plan[s] = plan_s
        table[s] = dict(
            measured_pool_n=n, measured_q0_tf_acc=round(p, 4),
            per_side_needed=per_side, need_from_formula=need, capacity=cap,
            capacity_limited=bool(cap and plan_s < need), plan=plan_s,
            planned_majority_side=int(plan_s * max(p, 1 - p)),
            planned_minority_side=int(plan_s * min(p, 1 - p)),
            min_side_margin=(plan_s * min(p, 1 - p)) / per_side,
            supported_episodes_at_margin=math.floor(plan_s * min(p, 1 - p) / (4 * a.margin)),
        )
    out = dict(target=dict(dev=a.dev, confirm=a.confirm, final=a.final, episodes_per_stratum=E,
                           per_side_needed=per_side),
               margin=a.margin, formula="N_s = ceil(margin * 4*E / min(p, 1-p))，cap = 容量探针值",
               capacity_source="capacity_probe.py（400k attempts/stratum，排除 v2 三池题干）",
               plan=plan, per_stratum=table,
               total_plan=sum(plan.values()),
               capacity_limited_strata=[s for s, v in table.items() if v["capacity_limited"]])
    json.dump(out, open(a.out, "w"), indent=2, ensure_ascii=False, sort_keys=True)
    print(json.dumps({k: v for k, v in out.items() if k != "per_stratum"}, indent=2, ensure_ascii=False))
    print(f"{'stratum':16s} {'p':>6s} {'need':>7s} {'cap':>7s} {'plan':>7s} {'min_side':>8s} {'margin':>7s} {'ep_supp':>7s}")
    for s, v in table.items():
        print(f"{s:16s} {v['measured_q0_tf_acc']:6.3f} {v['need_from_formula']:7d} "
              f"{str(v['capacity']):>7s} {v['plan']:7d} {v['planned_minority_side']:8d} "
              f"{v['min_side_margin']:7.2f} {v['supported_episodes_at_margin']:7d}")


if __name__ == "__main__":
    main()
