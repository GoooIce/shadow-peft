"""v3 生成池（v3 轮，2026-09-10）：**新种子、全新条目、v2 设计不变**。

为什么需要 v3：
  v2 的三池（dev 480 / confirm 7,200 / final 1,200）已在 14/15 两票各开箱一次而**耗尽**
  （confirm 用后退出、final 整轮只开一次）。下一轮（含 BF16 参照臂的量化代价测量与
  逐 family 门）需要**全新题干**，不能复用 v2 的 spent 条目。

与 v2 的关系（本脚本刻意只改两处）：
  1. **种子：** BASE_SEED_V3 = 20260911（v2 = 20260910）——所有 `stable_seed` 派生（生成器
     attempt 流）全部改变，因此题干集合与 v2 不同；
  2. **排除：** `--exclude-questions` 指定的题干集合（默认 = v2 三池 + v2 生成池）在生成期
     被跳过，保证 v3 条目不与 v2 任何已测量条目重合。

其余全部沿用 v2：15 个 stratum（5 family × 3 tier）、每个 stratum 的**子模式与权重**、
同样的短答案 + 唯一整数 + 程序化验证、同样的 PLAN 规模（15 stratum × 74 episode = 592 题需求）。

用法（Mac / GB10 均可，纯标准库）：
  python3 gen_pool_v3.py gen --out pool_v3.jsonl --v2-pool pool_v2.jsonl \
      --v2-pools taskset_dev.jsonl taskset_confirm.jsonl taskset_final.jsonl
  python3 gen_pool_v3.py selfcheck --pool pool_v3.jsonl --v2-pool pool_v2.jsonl
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_fitness_taskset import GENERATORS as V1_GENERATORS  # noqa: E402
from gen_pool_v2 import PLAN as PLAN_V2  # noqa: E402
from gen_pool_v2 import (  # noqa: E402
    REDESIGNED, V2_GENERATORS, V1_ACC, ROUND1_Q0_TF_ACC, recompute_plan,
)
from taskset_lib import FAMILIES, TIERS, self_check, sha12, stable_seed, verify  # noqa: E402

BASE_SEED_V3 = 20260911          # 新种子：v2 = 20260910，v3 = 20260911
PLAN_V3 = dict(PLAN_V2)          # 规模沿用 v2（15 stratum × 74 episode 的需求不变）
ID_PREFIX_V3 = "mini18"

assert set(PLAN_V3) == {f"{f}/{t}" for f in FAMILIES for t in TIERS}


def gen_item_v3(fam, tier, attempt):
    """返回 (item, seed, generator)。与 gen_pool_v2.gen_item 同结构，只换种子。"""
    seed = stable_seed(BASE_SEED_V3, fam, tier, attempt)
    rng = random.Random(seed)
    s = f"{fam}/{tier}"
    if s in V2_GENERATORS:
        return V2_GENERATORS[s](rng, tier, attempt), seed, "v2"
    return V1_GENERATORS[fam](rng, tier, attempt), seed, "v1"


def build_pool(plan, excluded_questions, id_prefix=ID_PREFIX_V3,
               prev_questions=None, allow_shortfall=False):
    """逐 stratum 生成；`excluded_questions` 里的题干一律跳过（并计入诊断）。

    prev_questions：上一版（小 plan）的题干集合——用于**前缀超集自检**：同一种子流下，放大 plan
      只是追加，前 `got` 条必须与上一版逐条相同（否则已测过的分数不能复用）。
    allow_shortfall：容量受限 stratum 允许产出 < want（记 shortfall），不抛异常。
    """
    items, seen = [], set()
    per_stratum_stats = {}
    prefix_ok = True
    for fam in FAMILIES:
        for tier in TIERS:
            stratum = f"{fam}/{tier}"
            want = plan[stratum]
            got, attempt, gen_v1, gen_v2 = 0, 0, 0, 0
            excluded_hits = 0
            # attempt 预算：正常 stratum 的产出率 ≈1.0–1.3 attempt/题；容量受限的 stratum 用
            # 固定的 400k attempt 上限（容量探针同口径），避免 4000*want 的天文数字空转。
            guard = max(400_000, 20 * want)
            while got < want:
                if attempt > guard:
                    if allow_shortfall:
                        break
                    raise RuntimeError(f"stratum {stratum}: only {got}/{want} after {attempt} attempts")
                try:
                    it, seed, src = gen_item_v3(fam, tier, attempt)
                except RuntimeError:
                    attempt += 1
                    continue
                attempt += 1
                if it["question"] in excluded_questions:
                    excluded_hits += 1
                    continue
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
            prefix_missing = None
            if prev_questions is not None:
                have = {x["question"] for x in items if x["stratum"] == stratum}
                prefix_missing = len(prev_questions.get(stratum, set()) - have)
                prefix_ok = prefix_ok and (prefix_missing == 0)
            per_stratum_stats[stratum] = dict(
                want=want, got=got, attempts=attempt, shortfall=want - got,
                capacity_limited=got < want,
                excluded_question_hits=excluded_hits,
                from_v1_generator=gen_v1, from_v2_generator=gen_v2,
                prefix_missing_vs_prev=prefix_missing,
                v2_plan_same=PLAN_V2[stratum] == want,
                v2_pool_q0_tf_acc=ROUND1_Q0_TF_ACC.get(stratum),
                v1_pool_q0_tf_acc=V1_ACC.get(stratum),
                redesigned=stratum in REDESIGNED,
                per_sub_mode=_sub_mode_counts([x for x in items if x["stratum"] == stratum]))
    items.sort(key=lambda x: x["id"])
    return items, per_stratum_stats, prefix_ok


def _sub_mode_counts(rows):
    out = {}
    for r in rows:
        m = (r.get("meta") or {}).get("sub") or (r.get("meta") or {}).get("kind") or "-"
        out[m] = out.get(m, 0) + 1
    return dict(sorted(out.items()))


def load_questions(paths):
    out = set()
    per_file = {}
    for p in paths or []:
        if not p or not os.path.exists(p):
            continue
        qs = set()
        with open(p) as f:
            for line in f:
                try:
                    qs.add(json.loads(line)["question"])
                except Exception:
                    continue
        per_file[os.path.basename(p)] = len(qs)
        out |= qs
    return out, per_file


def overlap_vs_v2(items, v2_pool_path, v2_pool_files):
    """v3 与 v2 的重叠自检：题干集合 / id 集合（都应为 0，因为已排除）。"""
    rep = {}
    q_v3 = {it["question"] for it in items}
    id_v3 = {it["id"] for it in items}
    if v2_pool_path and os.path.exists(v2_pool_path):
        v2 = [json.loads(l) for l in open(v2_pool_path)]
        q_v2 = {it["question"] for it in v2}
        id_v2 = {it["id"] for it in v2}
        rep["vs_v2_generation_pool"] = dict(v2_n=len(v2), question_overlap=len(q_v3 & q_v2),
                                            id_overlap=len(id_v3 & id_v2))
    for p in v2_pool_files or []:
        if not p or not os.path.exists(p):
            continue
        rows = [json.loads(l) for l in open(p)]
        q = {r["question"] for r in rows}
        i = {r["id"] for r in rows}
        rep[f"vs_{os.path.basename(p)}"] = dict(v2_n=len(rows), question_overlap=len(q_v3 & q),
                                               id_overlap=len(id_v3 & i))
    rep["v3_unique_questions"] = len(q_v3)
    rep["v3_n_items"] = len(items)
    rep["v3_dup_questions_within"] = len(items) - len(q_v3)
    return rep


def mix_compare(items, v2_pool_path):
    """子模式分布对照（v2 vs v3），用于如实登记"同设计"这句话。"""
    if not v2_pool_path or not os.path.exists(v2_pool_path):
        return None
    v2 = [json.loads(l) for l in open(v2_pool_path)]

    def mix(rows):
        per = collections.defaultdict(collections.Counter)
        for r in rows:
            m = (r.get("meta") or {}).get("sub") or (r.get("meta") or {}).get("kind") or "-"
            per[r["stratum"]][m] += 1
        return per

    m2, m3 = mix(v2), mix(items)
    out = {}
    for s in sorted(set(m2) | set(m3)):
        c2, c3 = m2.get(s, collections.Counter()), m3.get(s, collections.Counter())
        n2, n3 = sum(c2.values()) or 1, sum(c3.values()) or 1
        out[s] = dict(
            v2={k: dict(n=v, share=round(v / n2, 4)) for k, v in sorted(c2.items())},
            v3={k: dict(n=v, share=round(v / n3, 4)) for k, v in sorted(c3.items())},
            max_share_shift=max((abs(c3.get(k, 0) / n3 - c2.get(k, 0) / n2)
                                 for k in set(c2) | set(c3)), default=0.0),
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--out", required=True)
    g.add_argument("--plan", default=None, help="JSON 覆盖 PLAN_V3（stratum -> N）")
    g.add_argument("--exclude-questions", nargs="*", default=[],
                   help="题干排除源（JSONL，可有多次/多个文件）")
    g.add_argument("--v2-pool", default=None, help="v2 生成池（用于重叠/混合比对）")
    g.add_argument("--v2-pools", nargs="*", default=[], help="v2 三池文件（用于重叠自检）")
    g.add_argument("--prefix-check-against", default=None,
                   help="上一版（小 plan）池 JSONL：自检'放大 = 追加'")
    g.add_argument("--allow-shortfall", action="store_true",
                   help="容量受限 stratum 允许产出 < plan（记 shortfall）")
    g.add_argument("--id-prefix", default=ID_PREFIX_V3)
    s = sub.add_parser("selfcheck")
    s.add_argument("--pool", required=True)
    s.add_argument("--v2-pool", default=None)
    s.add_argument("--v2-pools", nargs="*", default=[])
    a = ap.parse_args()

    if a.cmd == "gen":
        plan = dict(PLAN_V3)
        if a.plan:
            override = json.load(open(a.plan))
            if isinstance(override.get("plan"), dict):
                override = override["plan"]     # make_plan_v3.py 的输出（含 rationale）
            plan.update(override)
        excl, excl_files = load_questions(a.exclude_questions)
        print(f"excluded questions: {len(excl)} from {excl_files}", flush=True)
        prev_questions = None
        if a.prefix_check_against and os.path.exists(a.prefix_check_against):
            prev = [json.loads(l) for l in open(a.prefix_check_against)]
            prev_questions = {}
            for it in prev:
                prev_questions.setdefault(it["stratum"], set()).add(it["question"])
            print(f"prefix check against {a.prefix_check_against}: "
                  f"{sum(len(v) for v in prev_questions.values())} questions", flush=True)
        items, stats, prefix_ok = build_pool(plan, excl, a.id_prefix,
                                             prev_questions=prev_questions,
                                             allow_shortfall=a.allow_shortfall)
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False, sort_keys=True) + "\n")
        chk = self_check(items)
        overlap = overlap_vs_v2(items, a.v2_pool, a.v2_pools)
        mix = mix_compare(items, a.v2_pool)
        # 前缀超集自检不适用（新种子 → 与 v2 无语义前缀关系，只有"重叠为 0"这一条）
        manifest = dict(
            schema="mini18-taskset-v3", base_seed=BASE_SEED_V3,
            base_seed_v2=20260910, plan=plan, plan_v2=PLAN_V2,
            plan_recomputed_from_round1=recompute_plan(ROUND1_Q0_TF_ACC),
            families=list(FAMILIES), tiers=list(TIERS),
            redesigned_strata=list(REDESIGNED),
            n_items=len(items), sha256=hashlib.sha256(open(a.out, "rb").read()).hexdigest(),
            self_check=chk, per_stratum=stats,
            excluded_question_sources=a.exclude_questions, n_excluded_questions=len(excl),
            overlap_vs_v2=overlap, sub_mode_mix_v2_vs_v3=mix,
            prefix_superset_vs_prev=prefix_ok,
            allow_shortfall=a.allow_shortfall,
            capacity_limited_strata=[s for s, v in stats.items() if v["capacity_limited"]],
            total_shortfall=sum(v["shortfall"] for v in stats.values()),
            path=os.path.basename(a.out))
        with open(a.out.replace(".jsonl", "_manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False, sort_keys=True)
        print(json.dumps({k: v for k, v in manifest.items()
                          if k not in ("per_stratum", "sub_mode_mix_v2_vs_v3")},
                         indent=2, ensure_ascii=False))
        print("--- per stratum ---")
        for st, v in stats.items():
            print(f"  {st:16s} {v['got']:5d} attempts={v['attempts']:6d} "
                  f"excluded_hits={v['excluded_question_hits']:6d} subs={v['per_sub_mode']}")
        print("--- mix shift (max share shift per stratum) ---")
        for st, v in sorted((mix or {}).items()):
            print(f"  {st:16s} v2={v['v2']}\n  {'':16s} v3={v['v3']} max_shift={v['max_share_shift']:.3f}")
    else:
        items = [json.loads(l) for l in open(a.pool)]
        print(json.dumps(dict(
            self_check=self_check(items),
            overlap_vs_v2=overlap_vs_v2(items, a.v2_pool, a.v2_pools),
            mix_compare=mix_compare(items, a.v2_pool)), indent=2, ensure_ascii=False)[:4000])


if __name__ == "__main__":
    main()
