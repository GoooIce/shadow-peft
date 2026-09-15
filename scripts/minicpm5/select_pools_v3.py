"""v3 三池划分（v3 轮）：在 17 票冻结器上加入 **BF16 参照臂** 与"与 v2 全量不重叠"的排除。

相对 `select_pools_v2.py` 的变化（选择规则本身不变）：
  * **多臂分数**：`--scores q0=... --scores bf16=...`；平衡仍**只按 q0**（难度锚点 = 起点），
    清单里同时登记每个池的 **q0 与 BF16 准确率**（逐 stratum / 逐 family）+ **BF16 − q0 间距**，
    这就是"4-bit 量化代价 / 可回收余量"的度量；
  * **新鲜度**：`--exclude-questions-from` 把 v2 生成池、v1 池的题干一并排除 → v3 冻结池与
    v2 的任何产物（三池 + 生成池）**零题干重叠**（生成期的硬排除已覆盖 v2 三池）；
  * `--sigma`/`--sigma-block`：把实测 σ 与逐 pool 的 δ 网格写进 `power` 块；
  * 逐池 token 量（由逐题 `n_prompt_tokens` 与生成 token 求和）写进清单，供成本口径复用。

用法：
  python3 select_pools_v3.py --pool pool_v3.jsonl \
      --scores q0=scores_q0_pool_v3.jsonl --scores bf16=scores_bf16_pool_v3.jsonl \
      --exclude-from decontamination_report_v3.json \
      --exclude-questions-from pool_v2.jsonl taskset_v2_dev.jsonl ... \
      --episodes dev=4 confirm=60 final=10 --out-dir . --scoring-protocol scoring_protocol.json
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from taskset_lib import ANSWER_PREFIX, GOLD_PREFIX, INSTRUCTION, stable_seed  # noqa: E402

EPISODE_SIZE = 8
K_POWER = 7.8489          # (z_.975 + z_.8)^2
BASE_SEED_V3 = 20260911   # 与 v3 生成同种子；选择用的派生种子随之为新的（不再复用 v2 的划分）


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_kv(spec_list):
    out = {}
    for s in spec_list or []:
        k, v = s.split("=", 1)
        out[k] = v
    return out


def parse_episodes(spec_list):
    return {k: int(v) for k, v in parse_kv(spec_list).items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--scores", action="append", required=True,
                    help="arm=path.jsonl（第一个 arm 是平衡锚点，应为 q0）")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--episode-size", type=int, default=EPISODE_SIZE)
    ap.add_argument("--episodes", nargs="*", default=["dev=4", "confirm=60", "final=10"])
    ap.add_argument("--exclude-from", default=None, help="去污染报告（hit_item_ids）")
    ap.add_argument("--exclude-questions-from", nargs="*", default=[],
                    help="题干排除源（JSONL）：v2 生成池 / v1 池 等")
    ap.add_argument("--scoring-protocol", default=None)
    ap.add_argument("--sigma", nargs="*", default=[], help="pair=σ（写进 power 块）")
    ap.add_argument("--sigma-block", default=None, help="JSON 文件：power 块的 σ 来源详情")
    ap.add_argument("--schema", default="mini18-taskset-pools-v3")
    a = ap.parse_args()

    episodes_spec = parse_episodes(a.episodes)
    total_ep = sum(episodes_spec.values())
    per_side = a.episode_size * total_ep // 2
    arm_paths = parse_kv(a.scores)
    anchor_arm = next(iter(arm_paths))

    items = [json.loads(l) for l in open(a.pool)]
    n_pool0 = len(items)
    excluded = set()
    if a.exclude_from:
        rep = json.load(open(a.exclude_from))
        excluded = set(rep.get("hit_item_ids", []))
        items = [it for it in items if it["id"] not in excluded]
        print(f"decontamination exclusion: {n_pool0} -> {len(items)} ({n_pool0 - len(items)} removed)")

    # 题干排除（v2 生成池 / v1 池）：保证 v3 冻结池与上一轮产物零题干重叠
    excl_q, excl_q_files = set(), {}
    for p in a.exclude_questions_from:
        if not p or not os.path.exists(p):
            continue
        qs = set()
        with open(p) as f:
            for line in f:
                try:
                    qs.add(json.loads(line)["question"])
                except Exception:
                    continue
        excl_q_files[os.path.basename(p)] = len(qs)
        excl_q |= qs
    n_before_q = len(items)
    items = [it for it in items if it["question"] not in excl_q]
    print(f"question exclusion: {n_before_q} -> {len(items)} ({n_before_q - len(items)} removed) "
          f"from {excl_q_files}")

    arms = {}
    for arm, path in arm_paths.items():
        arms[arm] = {json.loads(l)["id"]: json.loads(l) for l in open(path)}
    missing = [it["id"] for it in items if it["id"] not in arms[anchor_arm]]
    if missing:
        raise SystemExit(f"{len(missing)} items missing {anchor_arm} scores, e.g. {missing[:3]}")

    strata = sorted({it["stratum"] for it in items})
    selected, report = [], {}
    for stratum in strata:
        pool_s = [it for it in items if it["stratum"] == stratum]
        ok = sorted([it for it in pool_s if arms[anchor_arm][it["id"]].get("tf_correct")],
                    key=lambda x: x["id"])
        bad = sorted([it for it in pool_s if not arms[anchor_arm][it["id"]].get("tf_correct")],
                     key=lambda x: x["id"])
        rng = random.Random(stable_seed(BASE_SEED_V3, "select-v3", stratum))
        rng.shuffle(ok)
        rng.shuffle(bad)
        # 每 stratum 的**实际** episode 数：受少数侧容量约束（零填充，不混入不平衡 episode）。
        n_ep_avail = (2 * min(len(ok), len(bad))) // a.episode_size
        n_ep_target = total_ep
        n_ep_s = min(n_ep_target, n_ep_avail)
        per_side_s = a.episode_size * n_ep_s // 2
        capacity_limited = n_ep_s < n_ep_target
        take_ok, take_bad = ok[:per_side_s], bad[:per_side_s]
        padded_from_ok = padded_from_bad = 0
        sel = take_ok + take_bad
        rng.shuffle(sel)
        ok_s = [x for x in sel if arms[anchor_arm][x["id"]].get("tf_correct")]
        bad_s = [x for x in sel if not arms[anchor_arm][x["id"]].get("tf_correct")]
        arranged = []
        while ok_s or bad_s:
            if ok_s:
                arranged.append(ok_s.pop())
            if bad_s:
                arranged.append(bad_s.pop())
        n_ep = len(arranged) // a.episode_size
        episodes = [arranged[i * a.episode_size:(i + 1) * a.episode_size] for i in range(n_ep)]
        order = list(range(n_ep))
        rng.shuffle(order)
        assign, cursor = {}, 0
        for pool_name, k in episodes_spec.items():
            for _ in range(min(k, n_ep - cursor)):
                assign[order[cursor]] = pool_name
                cursor += 1
        for i, ep in enumerate(episodes):
            pool_name = assign.get(i, "unused")
            ep_id = f"{stratum.replace('/', '-')}-ep{i:02d}"
            for it in ep:
                rec = dict(it)
                rec["episode"] = ep_id
                rec["pool"] = pool_name
                selected.append((pool_name, rec))
        ep_ok = sorted({sum(1 for it in ep if arms[anchor_arm][it["id"]].get("tf_correct"))
                        for ep in episodes})
        report[stratum] = dict(
            pool_n=len(pool_s),
            q0_ok_pool=len(ok), q0_bad_pool=len(bad), q0_acc_pool=len(ok) / len(pool_s),
            per_side_target=per_side, per_side_used=per_side_s,
            episodes_target=n_ep_target, episodes_available=n_ep_avail, episodes_used=n_ep_s,
            capacity_limited=capacity_limited,
            max_episodes_supported_by_minority_side=n_ep_avail,
            q0_ok_used=len(take_ok), q0_bad_used=len(take_bad),
            padded_from_ok=padded_from_ok, padded_from_bad=padded_from_bad,
            selected_n=len(arranged), episodes=n_ep,
            q0_acc_selected=(sum(1 for x in arranged if arms[anchor_arm][x["id"]].get("tf_correct"))
                             / len(arranged)),
            episode_ok_count_span=[ep_ok[0], ep_ok[-1]] if ep_ok else None,
            assign={k: sum(1 for p in assign.values() if p == k) for k in episodes_spec},
            sub_modes=collections.Counter(
                (it.get("meta") or {}).get("sub") or (it.get("meta") or {}).get("kind") or "-"
                for it in pool_s).most_common(),
        )

    os.makedirs(a.out_dir, exist_ok=True)
    protocol = json.load(open(a.scoring_protocol)) if a.scoring_protocol else {}

    def arm_acc(rows, arm, key="tf_correct"):
        sc = arms.get(arm, {})
        vals = [sc[r["id"]].get(key) for r in rows if r["id"] in sc]
        return (sum(1 for v in vals if v) / len(vals)) if vals else None

    def episode_level(rows, arm, key="tf_correct"):
        sc = arms.get(arm, {})
        eps = sorted({r["episode"] for r in rows})
        out = []
        for e in eps:
            rs = [r for r in rows if r["episode"] == e and r["id"] in sc]
            if rs:
                out.append(sum(1 for r in rs if sc[r["id"]].get(key)) / len(rs))
        return (sum(out) / len(out)) if out else None

    # 每 pool 的实际 episode 数（容量受限 stratum 会少给 confirm/final）
    pool_ep_achieved = {}
    for pool_name in episodes_spec:
        rows_p = [r for p, r in selected if p == pool_name]
        pool_ep_achieved[pool_name] = len({r["episode"] for r in rows_p})

    manifest = dict(
        schema=a.schema, frozen_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        base_seed_v3=BASE_SEED_V3,
        episode_size=a.episode_size, episodes_per_stratum_target=episodes_spec,
        episodes_per_stratum_target_total=sum(episodes_spec.values()) * len(strata),
        episodes_per_stratum_achieved={s: v["episodes_used"] for s, v in report.items()},
        episodes_achieved_total=sum(v["episodes_used"] for v in report.values()),
        capacity_limited_strata={s: dict(episodes_supported=v["max_episodes_supported_by_minority_side"],
                                         episodes_target=v["episodes_target"], pool_n=v["pool_n"],
                                         q0_acc_pool=v["q0_acc_pool"], q0_bad_pool=v["q0_bad_pool"])
                                 for s, v in report.items() if v["capacity_limited"]},
        n_strata=len(strata),
        pool_file=os.path.basename(a.pool),
        pool_items_after_exclusions=len(items),
        decontamination_excluded_n=len(excluded),
        decontamination_report=(os.path.basename(a.exclude_from) if a.exclude_from else None),
        question_exclusion_files=excl_q_files,
        question_exclusion_n=len(excl_q),
        scores_files={k: os.path.basename(v) for k, v in arm_paths.items()},
        balance_anchor_arm=anchor_arm,
        arms_measured_for_composition=[k for k in arm_paths],
        selection=("within-stratum q0 teacher-forced 50/50 balance (per-side cap "
                   f"{per_side}), seeded shuffle (seed={BASE_SEED_V3}:select-v3), ok/bad "
                   "interleaved episodes, seeded episode->pool assignment"),
        scoring_protocol=protocol, per_stratum=report, pools={})
    for pool_name in episodes_spec:
        rows = sorted([r for p, r in selected if p == pool_name],
                      key=lambda x: (x["stratum"], x["id"]))
        path = os.path.join(a.out_dir, f"taskset_{pool_name}.jsonl")
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
        eps = sorted({r["episode"] for r in rows})
        per_stratum, gaps = {}, {}
        for s in sorted({r["stratum"] for r in rows}):
            rs = [r for r in rows if r["stratum"] == s]
            q0 = arm_acc(rs, "q0")
            bf = arm_acc(rs, "bf16")
            per_stratum[s] = dict(n=len(rs), q0_tf_acc=q0, bf16_tf_acc=bf,
                                  gap_bf16_minus_q0=(bf - q0 if (q0 is not None and bf is not None) else None))
            gaps[s] = per_stratum[s]["gap_bf16_minus_q0"]
        per_family = {}
        for f_ in sorted({r["family"] for r in rows}):
            rs = [r for r in rows if r["family"] == f_]
            q0 = arm_acc(rs, "q0")
            bf = arm_acc(rs, "bf16")
            per_family[f_] = dict(n=len(rs), q0_tf_acc=q0, bf16_tf_acc=bf,
                                  gap_bf16_minus_q0=(bf - q0 if (q0 is not None and bf is not None) else None))
        all_q0, all_bf = arm_acc(rows, "q0"), arm_acc(rows, "bf16")
        prompt_tok = sum(arms["q0"][r["id"]].get("n_prompt_tokens", 0)
                         for r in rows if r["id"] in arms["q0"])
        gold_tok = sum(arms["q0"][r["id"]].get("tf_tokens", 0)
                       for r in rows if r["id"] in arms["q0"])
        manifest["pools"][pool_name] = dict(
            path=os.path.basename(path), n_items=len(rows), n_episodes=len(eps),
            sha256=sha256_file(path),
            q0_tf_acc=all_q0, bf16_tf_acc=all_bf,
            gap_bf16_minus_q0=(all_bf - all_q0 if (all_q0 is not None and all_bf is not None) else None),
            q0_tf_acc_episode_level=episode_level(rows, "q0"),
            bf16_tf_acc_episode_level=episode_level(rows, "bf16"),
            gap_bf16_minus_q0_episode_level=(
                (episode_level(rows, "bf16") - episode_level(rows, "q0"))
                if episode_level(rows, "q0") is not None else None),
            tf_prompt_tokens=prompt_tok, tf_gold_tokens=gold_tok,
            per_stratum=per_stratum, per_family=per_family, episode_ids=eps,
        )

    # 全池（生成池）的 BF16−q0 间距：本票的主产出
    strata_all = sorted({it["stratum"] for it in items})
    overall = {}
    for s in strata_all:
        rs = [it for it in items if it["stratum"] == s]
        q0 = arm_acc(rs, "q0")
        bf = arm_acc(rs, "bf16")
        overall[s] = dict(n=len(rs), q0_tf_acc=q0, bf16_tf_acc=bf,
                          gap_bf16_minus_q0=(bf - q0 if (q0 is not None and bf is not None) else None))
    fam_overall = {}
    for f_ in sorted({it["family"] for it in items}):
        rs = [it for it in items if it["family"] == f_]
        q0 = arm_acc(rs, "q0")
        bf = arm_acc(rs, "bf16")
        fam_overall[f_] = dict(n=len(rs), q0_tf_acc=q0, bf16_tf_acc=bf,
                               gap_bf16_minus_q0=(bf - q0 if (q0 is not None and bf is not None) else None))
    manifest["bf16_vs_q0_gap_generation_pool"] = dict(
        n_items=len(items), q0_tf_acc=arm_acc(items, "q0"), bf16_tf_acc=arm_acc(items, "bf16"),
        gap_bf16_minus_q0=(arm_acc(items, "bf16") - arm_acc(items, "q0")),
        per_stratum=overall, per_family=fam_overall)
    manifest["bf16_vs_q0_gap_confirm_pool_derived"] = manifest["pools"]["confirm"].get("gap_bf16_minus_q0")

    # power 块
    sigmas = parse_kv(a.sigma) if a.sigma else {}
    sigmas = {k: float(v) for k, v in sigmas.items()}
    power = dict(
        formula="delta_min = (z_.975 + z_.8) * sigma / sqrt(n_ep)；n_ep = K sigma^2 / delta^2, K=7.8489",
        alpha=0.05, power_required=0.80, delta_frozen=0.02,
        sigma_measured=sigmas, sigma_source=json.load(open(a.sigma_block)) if a.sigma_block else None,
        per_pool={})
    for pool_name in episodes_spec:
        n_ep = pool_ep_achieved[pool_name]                       # 实测（容量受限 stratum 更少）
        n_ep_target = episodes_spec[pool_name] * len(strata)
        row = dict(n_episodes=n_ep, n_episodes_target=n_ep_target,
                   items=n_ep * a.episode_size, delta_grid={})
        for d in (0.01, 0.015, 0.02, 0.03, 0.05):
            row["delta_grid"][f"{d}"] = dict(
                delta=d, n_episodes_needed=math.ceil(K_POWER * (0.2274 ** 2) / d ** 2),
                items_needed=math.ceil(K_POWER * (0.2274 ** 2) / d ** 2) * a.episode_size,
                note="n_ep 用 σ=0.2274（v2 近邻配对，本票在同规模 v3 dev 池上复测）估算；实测 σ 见 sigma_measured")
        for name, sg in sigmas.items():
            row[f"delta_min_80pct_{name}"] = (K_POWER ** 0.5) * sg / (n_ep ** 0.5)
        if sigmas:
            sg = sigmas.get("q0-cand_search") or min(sigmas.values())
            row["delta_min_80pct"] = (K_POWER ** 0.5) * sg / (n_ep ** 0.5)
            row["sigma_used_for_delta_min"] = sg
            # item 级统计量的 na ive 与 cluster-corrected 口径
            row["item_level_note"] = ("item 级配对均值的点估计与 episode 级完全相同（等簇大小）；"
                                      "naive 口径用 σ_item/√N 会低估 SE —— 见 per_family_resolution.json")
        power["per_pool"][pool_name] = row
    manifest["power"] = power

    with open(os.path.join(a.out_dir, "pools_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False, sort_keys=True)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk not in
                          ("episode_ids", "per_stratum", "per_family")}
                      for k, v in manifest["pools"].items()}, indent=2, ensure_ascii=False))
    print("--- per-stratum (pool-wide q0/bf16 from generation pool) ---")
    for s, v in manifest["bf16_vs_q0_gap_generation_pool"]["per_stratum"].items():
        print(f"  {s:16s} n={v['n']:5d} q0={v['q0_tf_acc']:.3f} bf16={v['bf16_tf_acc']:.3f} "
              f"gap={v['gap_bf16_minus_q0']:+.3f}")
    print("--- per-family ---")
    for s, v in manifest["bf16_vs_q0_gap_generation_pool"]["per_family"].items():
        print(f"  {s:16s} n={v['n']:5d} q0={v['q0_tf_acc']:.3f} bf16={v['bf16_tf_acc']:.3f} "
              f"gap={v['gap_bf16_minus_q0']:+.3f}")


if __name__ == "__main__":
    main()
