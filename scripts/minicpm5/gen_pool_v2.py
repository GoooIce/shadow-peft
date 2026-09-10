"""v2 生成池（17 票）：扩池到 ~23.5k 题，重设计 3 个"饱和/地板 + 容量不足"的 stratum。

相对 16 票的口径变化（其余 12 个 stratum 完全沿用 v1 生成器与 v1 种子流，
因此 v2 池在这 12 个 stratum 上是 v1 池的**前缀超集**，可用脚本自检）：

  1. `pct/easy`：v1 题面容量 508、q0 TF 准确率 0.996（天花板）——饱和题不携带信息。
     重设计为 4 个子模式的**难度谱**（v1-easy 10% / v1-medium 20% / 非 5 倍数百分比 40% /
     分数 30%），容量 >10 万，q0 落在判别带内。
  2. `modulo/easy`：v1 容量 556（不足以给 confirm 提供 480 题）、q0 0.904。
     重设计为 5 个子模式（2 位÷2..9 20% / 3 位÷2..9 30% / 小幂末位 15% / 2 位 gcd 15% /
     3 位÷11..19 20%）。
  3. `pct/hard`：v1 容量 2220 且 q0 0.067（地板）——少数侧只有 18/270，无法为 900 episode
     规模的 confirm 提供 296 道 q0-错题。重设计为 5 个子模式的难度谱，把多数侧拉回判别带。

  其余 12 个 stratum：`GENERATORS` 与种子派生与 v1 逐字节一致（同一 attempt 流），
  只是取更多条目 → v1 池是 v2 池的前缀。

用法（Mac / GB10 均可，纯标准库）：
  python3 gen_pool_v2.py gen --out pool_v2.jsonl [--plan plan.json] [--scale 1.0]
  python3 gen_pool_v2.py selfcheck --pool pool_v2.jsonl --v1-pool <v1 pool.jsonl>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_fitness_taskset import GENERATORS as V1_GENERATORS  # noqa: E402
from taskset_lib import (  # noqa: E402
    BASE_SEED, FAMILIES, NAME_BANK, TIERS, eval_int_expr, self_check, sha12, stable_seed, verify,
)

# --------------------------------------------------------------- 逐 stratum 生成量
# 目标：每 stratum 需要 74 episode = 592 题（dev 4 + confirm 60 + final 10），
# 且 q0 对/错两侧各需 >= 296 题（50/50 平衡，否则该 stratum 的 confirm 分片会被"多数侧"填充）。
#
# PLAN_ROUND1 是第一轮（按 16 票 v1 实测准确率估的规模 + 3 个重设计 stratum 的目标 p≈0.5）；
# 第一轮 q0 实测（ROUND1_Q0_TF_ACC）后按
#     N_s = clip(1.30 * 296 / min(p_measured, 1 - p_measured), 1200, 3500)
# 重算得到 PLAN（交付用）。生成器按 attempt 序号确定性产出，故加量只是**追加**，
# 第一轮已评分的条目 id 与其分数完全复用。
PLAN_ROUND1 = {
    "arith/easy": 1200, "arith/medium": 1200, "arith/hard": 2800,
    "logic/easy": 1200, "logic/medium": 1450, "logic/hard": 1600,
    "ordering/easy": 1200, "ordering/medium": 1200, "ordering/hard": 2100,
    "modulo/easy": 1200, "modulo/medium": 1200, "modulo/hard": 1300,
    "pct/easy": 1200, "pct/medium": 1200, "pct/hard": 1200,
}
ROUND1_Q0_TF_ACC = {  # 21250 题第一轮 q0 TF 实测（q0 = NVFP4 W4A4 起点）
    "arith/easy": 0.7742, "arith/medium": 0.6700, "arith/hard": 0.1007,
    "logic/easy": 0.4250, "logic/medium": 0.2724, "logic/hard": 0.2331,
    "ordering/easy": 0.5750, "ordering/medium": 0.3933, "ordering/hard": 0.1695,
    "modulo/easy": 0.6633, "modulo/medium": 0.5633, "modulo/hard": 0.3431,
    "pct/easy": 0.4517, "pct/medium": 0.7883, "pct/hard": 0.2200,
}
V1_ACC = {  # 16 票 v1 生成池 q0 TF 准确率（第一轮估规模的依据）
    "arith/easy": 0.752, "arith/medium": 0.678, "arith/hard": 0.122,
    "logic/easy": 0.444, "logic/medium": 0.278, "logic/hard": 0.248,
    "ordering/easy": 0.607, "ordering/medium": 0.333, "ordering/hard": 0.189,
    "modulo/easy": 0.904, "modulo/medium": 0.563, "modulo/hard": 0.311,
    "pct/easy": 0.996, "pct/medium": 0.781, "pct/hard": 0.067,
}
PLAN = {  # 交付规模（= recompute_plan(ROUND1_Q0_TF_ACC)，由脚本内 assert 保证一致）
    "arith/easy": 1705, "arith/medium": 1200, "arith/hard": 3500,
    "logic/easy": 1200, "logic/medium": 1413, "logic/hard": 1651,
    "ordering/easy": 1200, "ordering/medium": 1200, "ordering/hard": 2271,
    "modulo/easy": 1200, "modulo/medium": 1200, "modulo/hard": 1200,
    "pct/easy": 1200, "pct/medium": 1818, "pct/hard": 1750,
}


def recompute_plan(acc, need=296, margin=1.30, lo=1200, hi=3500):
    """N_s = clip(margin * need / min(p, 1-p), lo, hi)。"""
    return {s: max(lo, min(hi, math.ceil(margin * need / min(p, 1 - p))))
            for s, p in acc.items()}


assert recompute_plan(ROUND1_Q0_TF_ACC) == PLAN, (
    "PLAN 必须等于 recompute_plan(ROUND1_Q0_TF_ACC)：先跑 recompute_plan 再抄进来")
assert set(PLAN) == {f"{f}/{t}" for f in FAMILIES for t in TIERS}
REDESIGNED = ("pct/easy", "modulo/easy", "pct/hard")

_PCT_T = ("What is {x} of {n}?", "Compute {x} of {n}.", "Find {x} of {n}.",
          "Calculate {x} of {n}.", "Determine {x} of {n}.")


# ---------------------------------------------------------------- pct/easy v2
# 难度谱（容量 >10 万）：所有子模式都保持"整数答案 + 可程序化验证 + 短答案"。
_PCT_EASY_MODES = (
    ("pct_round", 10),   # v1 easy：整百分比 + 4 的倍数（唯一保留的天花板子模式，占比压低）
    ("pct_odd", 55),     # 非 5 倍数百分比（12/17/23/…）× 使答案为整数的 n
    ("frac_big", 35),    # 大分母分数 a/b × n（b ∈ {16,20,25,40,50}，与 pct/medium 的 b 集不交）
)
_PCT_EASY_ODD_P = (12, 13, 17, 18, 23, 27, 28, 32, 33, 38, 42, 47, 56, 64, 72, 88, 92, 95)


def _pick(rng, modes):
    tot = sum(w for _, w in modes)
    r = rng.randrange(tot)
    for name, w in modes:
        if r < w:
            return name
        r -= w
    return modes[-1][0]


def gen_pct_easy_v2(rng, tier, idx):
    for _ in range(400):
        mode = _pick(rng, _PCT_EASY_MODES)
        if mode == "pct_round":
            p, n = rng.choice([10, 20, 25, 50, 75]), rng.randrange(4, 41) * 4
            if (n * p) % 100:
                continue
            return dict(question=rng.choice(_PCT_T).format(x=f"{p}%", n=n),
                        answer=str(n * p // 100), verify=dict(kind="pct_of", n=n, p=p),
                        meta=dict(sub=mode, p=p, n=n))
        if mode == "pct_odd":
            p = rng.choice(_PCT_EASY_ODD_P)
            step = 100 // math.gcd(100, p)
            n = rng.randrange(2, 41) * step
            if (n * p) % 100 or n * p // 100 > 9999:
                continue
            return dict(question=rng.choice(_PCT_T).format(x=f"{p}%", n=n),
                        answer=str(n * p // 100), verify=dict(kind="pct_of", n=n, p=p),
                        meta=dict(sub=mode, p=p, n=n))
        b = rng.choice([16, 20, 25, 40, 50])
        a = rng.randrange(1, b)
        n = rng.randrange(3, 60) * b
        if (n * a) % b or n * a // b > 9999:
            continue
        return dict(question=rng.choice(_PCT_T).format(x=f"{a}/{b}", n=n),
                    answer=str(n * a // b), verify=dict(kind="frac_of", n=n, a=a, b=b),
                    meta=dict(sub=mode, a=a, b=b, n=n))
    raise RuntimeError("pct/easy v2 generation failed")


# ---------------------------------------------------------------- modulo/easy v2
# 4 个子模式的除数/位数区间与 modulo/medium（4 位÷7..13、末位、2 位 gcd）、
# modulo/hard（6 位÷17..31、大幂末位、4 位 gcd）都不重叠，避免跨 stratum 抢题。
_MOD_EASY_MODES = (
    ("rem2", 25),     # v1 easy：2 位 ÷ 2..9
    ("rem3", 30),     # 3 位 ÷ 2..9
    ("rem3b", 25),    # 3 位 ÷ 11..19
    ("rem4small", 20),  # 4 位 ÷ 2..6（medium 用 7..13，不重叠）
)


def gen_modulo_easy_v2(rng, tier, idx):
    for _ in range(400):
        mode = _pick(rng, _MOD_EASY_MODES)
        if mode == "rem2":
            a, b = rng.randint(10, 99), rng.randint(2, 9)
        elif mode == "rem3":
            a, b = rng.randint(100, 999), rng.randint(2, 9)
        elif mode == "rem3b":
            a, b = rng.randint(100, 999), rng.randint(11, 19)
        else:
            a, b = rng.randint(1000, 9999), rng.randint(2, 6)
        if a % b == 0:
            continue
        return dict(question=f"What is the remainder when {a} is divided by {b}?",
                    answer=str(a % b), verify=dict(kind="modulo_rem", a=a, b=b),
                    meta=dict(sub=mode))
    raise RuntimeError("modulo/easy v2 generation failed")


# ---------------------------------------------------------------- pct/hard v2
_PCT_HARD_MODES = (
    ("reverse_big", 30),    # v1 hard：复合百分比反推（大 v）
    ("reverse_small", 20),  # 反推，小 v
    ("net", 25),            # 复合增减（基数范围比 v1 hard 宽）
    ("frac_big", 25),       # 分数 × 大 n（m ∈ [61,600]，与 pct/medium 的 [3,60] 不交）
)


def gen_pct_hard_v2(rng, tier, idx):
    for _ in range(400):
        mode = _pick(rng, _PCT_HARD_MODES)
        if mode == "reverse_big":
            p = rng.choice([20, 25, 40, 50, 60, 75])
            o = rng.randrange(4, 60) * (100 // math.gcd(100, p))
            v = o * (100 + p) // 100
            if o * (100 + p) % 100 or v > 99999:
                continue
            return dict(question=rng.choice([
                f"After an increase of {p}%, a value becomes {v}. What was the original value?",
                f"A number grows by {p}% and the result is {v}. What was the number?",
                f"After a {p}% raise, a price is {v}. What was the price before the raise?"]),
                answer=str(o), verify=dict(kind="pct_original", v=v, p=p), meta=dict(sub=mode))
        if mode == "reverse_small":
            p = rng.choice([10, 20, 25, 50])
            step = 100 // math.gcd(100, p)
            o = rng.randrange(2, 13) * step
            v = o * (100 + p) // 100
            if o * (100 + p) % 100 or v > 9999:
                continue
            return dict(question=rng.choice([
                f"After an increase of {p}%, a value becomes {v}. What was the original value?",
                f"A number grows by {p}% and the result is {v}. What was the number?"]),
                answer=str(o), verify=dict(kind="pct_original", v=v, p=p), meta=dict(sub=mode))
        if mode == "net":
            p = rng.choice([10, 20, 25, 30, 40])
            qd = rng.choice([10, 20, 25, 50])
            o = rng.randrange(2, 501) * 20
            num = o * (100 + p) * (100 - qd)
            if num % 10000 or num // 10000 > 99999:
                continue
            return dict(question=rng.choice([
                f"A quantity starts at {o}. It rises by {p}% and then falls by {qd}%. "
                f"What is the final value?",
                f"Start with {o}. Increase it by {p}%, then decrease the result by {qd}%. "
                f"What number do you end with?"]),
                answer=str(num // 10000), verify=dict(kind="pct_net", o=o, p=p, q=qd),
                meta=dict(sub=mode))
        b = rng.choice([4, 5, 8, 10])
        a = rng.randrange(1, b)
        n = rng.randrange(61, 601) * b
        if (n * a) % b or n * a // b > 99999:
            continue
        return dict(question=rng.choice(_PCT_T).format(x=f"{a}/{b}", n=n),
                    answer=str(n * a // b), verify=dict(kind="frac_of", n=n, a=a, b=b),
                    meta=dict(sub=mode))
    raise RuntimeError("pct/hard v2 generation failed")


V2_GENERATORS = {
    "pct/easy": gen_pct_easy_v2,
    "modulo/easy": gen_modulo_easy_v2,
    "pct/hard": gen_pct_hard_v2,
}


def gen_item(fam, tier, attempt):
    """返回 (item, seed)。重设计的 stratum 走 v2 生成器，其余与 v1 逐字节一致。"""
    seed = stable_seed(BASE_SEED, fam, tier, attempt)
    rng = random.Random(seed)
    s = f"{fam}/{tier}"
    if s in V2_GENERATORS:
        return V2_GENERATORS[s](rng, tier, attempt), seed, "v2"
    return V1_GENERATORS[fam](rng, tier, attempt), seed, "v1"


def build_pool(plan, id_prefix="mini17"):
    items, seen = [], set()
    per_stratum_stats = {}
    for fam in FAMILIES:
        for tier in TIERS:
            stratum = f"{fam}/{tier}"
            want = plan[stratum]
            got, attempt, gen_v1, gen_v2 = 0, 0, 0, 0
            guard = 4000 * want
            while got < want:
                if attempt > guard:
                    raise RuntimeError(f"stratum {stratum}: only {got}/{want} after {attempt} attempts")
                try:
                    it, seed, src = gen_item(fam, tier, attempt)
                except RuntimeError:
                    attempt += 1
                    continue
                attempt += 1
                if it["question"] in seen:
                    continue
                ok, _, _ = verify(it, it["answer"])
                if not ok:
                    continue
                seen.add(it["question"])
                it.update(family=fam, tier=tier, stratum=stratum, seed=seed, source=src,
                          id=f"{id_prefix}-{fam}-{tier}-{got:04d}-{sha12(it['question'] + '|' + it['answer'])}")
                items.append(it)
                got += 1
                gen_v1 += (src == "v1")
                gen_v2 += (src == "v2")
            per_stratum_stats[stratum] = dict(want=want, got=got, attempts=attempt,
                                              from_v1_generator=gen_v1, from_v2_generator=gen_v2,
                                              v1_pool_q0_tf_acc=V1_ACC.get(stratum),
                                              redesigned=stratum in REDESIGNED,
                                              per_sub_mode=_sub_mode_counts(
                                                  [x for x in items if x["stratum"] == stratum]))
    items.sort(key=lambda x: x["id"])
    return items, per_stratum_stats


def _sub_mode_counts(rows):
    out = {}
    for r in rows:
        m = (r.get("meta") or {}).get("sub") or (r.get("meta") or {}).get("kind") or "-"
        out[m] = out.get(m, 0) + 1
    return dict(sorted(out.items()))


def check_superset(items, v1_path):
    """v2 池在 12 个未重设计的 stratum 上应是 v1 池的前缀超集（同种子流）。

    v1/v2 的 id 前缀与序号不同（mini16/mini17），故按 **题干集合** 逐 stratum 比对；
    未重设计的 stratum 应 100% 覆盖 v1，重设计的 3 个 stratum 允许不同。
    """
    if not v1_path or not os.path.exists(v1_path):
        return None
    v1 = [json.loads(l) for l in open(v1_path)]
    by_strat_v2 = {}
    for it in items:
        by_strat_v2.setdefault(it["stratum"], set()).add(it["question"])
    per_stratum, missing_by_stratum = {}, {}
    for it in v1:
        s = it["stratum"]
        d = per_stratum.setdefault(s, dict(v1_n=0, present=0))
        d["v1_n"] += 1
        if it["question"] in by_strat_v2.get(s, ()):
            d["present"] += 1
    for s, d in per_stratum.items():
        d["missing"] = d["v1_n"] - d["present"]
        d["fully_covered"] = d["missing"] == 0
        d["redesigned"] = s in REDESIGNED
        if d["missing"]:
            missing_by_stratum[s] = d["missing"]
    return dict(v1_n=len(v1),
                uncovered_on_unredesigned=[s for s, d in per_stratum.items()
                                           if not d["redesigned"] and not d["fully_covered"]],
                missing_by_stratum=dict(sorted(missing_by_stratum.items())),
                superset_on_unredesigned=all(d["fully_covered"] for s, d in per_stratum.items()
                                             if not d["redesigned"]),
                per_stratum={s: dict(v1_n=d["v1_n"], present=d["present"], missing=d["missing"],
                                     fully_covered=d["fully_covered"], redesigned=d["redesigned"])
                             for s, d in sorted(per_stratum.items())})


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--out", required=True)
    g.add_argument("--plan", default=None, help="JSON 覆盖 PLAN（stratum -> N）")
    g.add_argument("--v1-pool", default=None)
    g.add_argument("--id-prefix", default="mini17")
    s = sub.add_parser("selfcheck")
    s.add_argument("--pool", required=True)
    s.add_argument("--v1-pool", default=None)
    a = ap.parse_args()

    if a.cmd == "gen":
        plan = dict(PLAN)
        if a.plan:
            plan.update(json.load(open(a.plan)))
        items, stats = build_pool(plan, a.id_prefix)
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False, sort_keys=True) + "\n")
        sup = check_superset(items, a.v1_pool)
        chk = self_check(items)
        h = hashlib.sha256(open(a.out, "rb").read()).hexdigest()
        manifest = dict(schema="mini17-taskset-v2", base_seed=BASE_SEED, plan=plan,
                        plan_round1=PLAN_ROUND1, round1_q0_tf_acc=ROUND1_Q0_TF_ACC,
                        families=list(FAMILIES), tiers=list(TIERS),
                        redesigned_strata=list(REDESIGNED), n_items=len(items), sha256=h,
                        self_check=chk, per_stratum=stats, superset_vs_v1=sup,
                        path=os.path.basename(a.out))
        with open(a.out.replace(".jsonl", "_manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False, sort_keys=True)
        print(json.dumps({k: v for k, v in manifest.items() if k != "per_stratum"},
                         indent=2, ensure_ascii=False))
        print(f"--- per stratum (n, v1-generator/v2-generator) ---")
        for s, v in stats.items():
            print(f"  {s:16s} {v['got']:5d}  v1={v['from_v1_generator']:5d} v2={v['from_v2_generator']:5d} "
                  f"subs={v['per_sub_mode']}")
    else:
        items = [json.loads(l) for l in open(a.pool)]
        print(json.dumps(dict(self_check=self_check(items),
                              superset_vs_v1=check_superset(items, a.v1_pool)),
                         indent=2, ensure_ascii=False)[:3000])


if __name__ == "__main__":
    main()
