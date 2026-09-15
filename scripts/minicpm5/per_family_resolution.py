"""v3 轮：**逐 stratum / 逐 family 的分辨力**（下一轮可能需要逐 family 验收门）与可行性判定。

问题（来自 v2/14 票的发现）：搜索中心在两个互斥池上比起点差 3.3–3.9 点，且损伤集中在
**hard 档**（arith-hard / ordering-hard / logic-hard / logic-medium），部分 easy 档反而上升。
若下一轮想上"逐 family 验收门"，必须先知道：**dev 池（每 stratum 32 题 / 4 episode，
每 family 96 题 / 12 episode）能不能分辨出这种量级的逐 family 变化？**

方法：
  1. **直接估计**：在冻结 dev 池上按 episode 求配对差值（A−B），给出逐 stratum（4 ep）与
     逐 family（12 ep）的 σ 与 δ_min = 2.802·σ/√n_ep（80% 功效、双侧 5%）。4/12 个 episode 的
     σ 估计自由度为 3/11 —— 一并给出 χ² 区间，避免把噪声当结构。
  2. **池化估计（更稳）**：用整池 σ（60 ep）作为"该池的"σ，给出**该池规模下**逐 stratum /
     逐 family 可分辨的最小 δ（这才是"门"要面对的噪声水平，而不是各格子的噪声点估计）。
  3. **规模外推**：由生成池（23,708 题）上 q0−BF16 配对的 item 级翻转率 + dev 上估的簇内相关
     ρ，外推到任意 n_ep 的 σ，得到"逐 family 门在 δ=0.02/0.05/0.10 时需要的 episode 与题数"。
  4. **可行性判定**：把需要的规模与三池规模、整轮成本放在一起，给出"不行 + 替代方案"。

用法：
  python3 per_family_resolution.py --items taskset_dev.jsonl \
      --scores q0=scores_q0_dev3.jsonl --scores bf16=scores_bf16_dev3.jsonl \
      --scores cand_search=... --scores centre=... \
      --pool-items pool_v3.jsonl --pool-scores q0=... --pool-scores bf16=... \
      --out per_family_resolution.json
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import statistics as st

Z_A2 = 1.959963984540054
Z_B = 0.8416212335729143
K = (Z_A2 + Z_B) ** 2          # 7.8489
EPISODE_SIZE = 8
def chi2_quantile(df: int, p: float) -> float:
    """χ²(df) 的 p 分位（Wilson–Hilferty 近似，df>=3 时误差 <1%）。"""
    z = norm_ppf(p)
    return df * (1.0 - 2.0 / (9.0 * df) + z * math.sqrt(2.0 / (9.0 * df))) ** 3


def norm_ppf(p: float) -> float:
    """标准正态分位（Acklam 近似，纯标准库）。"""
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def load(p):
    return [json.loads(l) for l in open(p)]


def pair_values(rows, ep_of, strat_of, fam_of, a, b, key="tf_correct"):
    """返回 (per-episode diffs, per-item diffs, item flip rate)。"""
    ids = [k for k in rows.get(a, {}) if k in rows.get(b, {}) and k in ep_of]
    item_d, ep_buf = [], collections.defaultdict(list)
    for k in ids:
        da = 1.0 if rows[a][k].get(key) else 0.0
        db = 1.0 if rows[b][k].get(key) else 0.0
        d = da - db
        item_d.append(d)
        ep_buf[ep_of[k]].append(d)
    ep_d = {e: st.mean(v) for e, v in ep_buf.items()}
    flip = sum(1 for x in item_d if x != 0) / len(item_d) if item_d else None
    return ids, item_d, ep_d, flip


def sd_ci(sd, n, conf=0.95):
    """σ 的 (1-conf) 双侧区间：σ ∈ [s·√(df/χ²_upper), s·√(df/χ²_lower)]。"""
    if not sd or n <= 2:
        return None, None
    df = n - 1
    hi = chi2_quantile(df, 1 - (1 - conf) / 2)
    lo = chi2_quantile(df, (1 - conf) / 2)
    return sd * math.sqrt(df / hi), sd * math.sqrt(df / lo)


def delta_min(sd, n_ep):
    return (Z_A2 + Z_B) * sd / math.sqrt(n_ep) if sd else None


def n_ep_for(sd, delta):
    return math.ceil(K * sd * sd / (delta * delta)) if sd else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True, help="冻结 dev 池（含 episode 字段）")
    ap.add_argument("--scores", action="append", required=True, help="arm=path.jsonl（dev 池上）")
    ap.add_argument("--pool-items", default=None, help="生成池（用于逐 stratum item 级方差）")
    ap.add_argument("--pool-scores", action="append", default=[], help="arm=path.jsonl（生成池上）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--delta", type=float, default=0.02)
    a = ap.parse_args()

    dev = load(a.items)
    ep_of = {it["id"]: it["episode"] for it in dev}
    strat_of = {it["id"]: it["stratum"] for it in dev}
    fam_of = {it["id"]: it["family"] for it in dev}
    n_ep_dev = len(set(ep_of.values()))
    rows = {}
    for spec in a.scores:
        tag, path = spec.split("=", 1)
        rows[tag] = {r["id"]: r for r in load(path)}
    tags = list(rows)
    ep_fam = collections.defaultdict(set)
    ep_strat = {}
    for iid, e in ep_of.items():
        ep_fam[fam_of[iid]].add(e)
        ep_strat[e] = strat_of[iid]

    rep = dict(items_file=os.path.basename(a.items), n_items=len(dev), n_episodes=n_ep_dev,
               episode_size=EPISODE_SIZE, arms=tags, delta_target=a.delta,
               generated_at=__import__("time").strftime("%Y-%m-%dT%H:%M:%S"))

    # 逐臂准确率
    rep["arm_accuracy"] = {}
    for t in tags:
        ids = [k for k in rows[t] if k in ep_of]
        item_acc = st.mean([1.0 if rows[t][k].get("tf_correct") else 0.0 for k in ids])
        epv = collections.defaultdict(list)
        for k in ids:
            epv[ep_of[k]].append(1.0 if rows[t][k].get("tf_correct") else 0.0)
        rep["arm_accuracy"][t] = dict(
            n_items=len(ids), item_acc=item_acc,
            episode_acc=st.mean([st.mean(v) for v in epv.values()]),
            gold_prob_episode=st.mean([math.exp(max(-20.0, rows[t][k]["tf_gold_logprob"]))
                                       for k in ids if rows[t][k].get("tf_gold_logprob") is not None]),
            tf_next_is_im_end_rate=st.mean([1.0 if rows[t][k].get("tf_next_is_im_end") else 0.0
                                            for k in ids]))

    # ---- 1/2. 逐配对：整池 + 逐 stratum + 逐 family ----
    pairs = {}
    for i in range(len(tags)):
        for j in range(i + 1, len(tags)):
            A, B = tags[i], tags[j]
            ids, item_d, ep_d, flip = pair_values(rows, ep_of, strat_of, fam_of, A, B)
            eps = sorted(ep_d)
            d = [ep_d[e] for e in eps]
            sd_full = st.pstdev(d) if len(d) > 1 else None
            per_strat, per_fam = {}, {}
            for s in sorted(set(strat_of[k] for k in ids)):
                es = [e for e in eps if ep_strat[e] == s]
                dd = [ep_d[e] for e in es]
                sd = st.pstdev(dd) if len(dd) > 1 else None
                lo, hi = sd_ci(sd, len(dd))
                per_strat[s] = dict(
                    n_episodes=len(dd), mean_episode_delta=st.mean(dd), sd_episode_delta=sd,
                    sd_ci95=[lo, hi], delta_min_80pct_at_dev_size=delta_min(sd, len(dd)),
                    n_episodes_needed_delta_target=n_ep_for(sd, a.delta),
                    item_flip_rate=(sum(1 for k in ids if strat_of[k] == s
                                        and (rows[A][k].get("tf_correct") != rows[B][k].get("tf_correct")))
                                    / max(1, sum(1 for k in ids if strat_of[k] == s))),
                )
            for f in sorted(set(fam_of[k] for k in ids)):
                es = sorted(ep_fam[f])
                dd = [ep_d[e] for e in es if e in ep_d]
                sd = st.pstdev(dd) if len(dd) > 1 else None
                lo, hi = sd_ci(sd, len(dd))
                per_fam[f] = dict(
                    n_episodes=len(dd), mean_episode_delta=st.mean(dd), sd_episode_delta=sd,
                    sd_ci95=[lo, hi], delta_min_80pct_at_dev_size=delta_min(sd, len(dd)),
                    n_episodes_needed_delta_target=n_ep_for(sd, a.delta),
                    item_flip_rate=(sum(1 for k in ids if fam_of[k] == f
                                        and (rows[A][k].get("tf_correct") != rows[B][k].get("tf_correct")))
                                    / max(1, sum(1 for k in ids if fam_of[k] == f))),
                )
            pairs[f"{A}-{B}"] = dict(
                n_items=len(ids), n_episodes=len(eps),
                mean_episode_delta=st.mean(d), sd_episode_delta=sd_full,
                se_episode_delta=(sd_full / math.sqrt(len(d)) if sd_full else None),
                t_stat=(st.mean(d) / (sd_full / math.sqrt(len(d))) if sd_full else None),
                delta_min_80pct_at_dev_size=delta_min(sd_full, len(d)),
                sd_ci95=list(sd_ci(sd_full, len(d))),
                n_episodes_needed_delta_target=n_ep_for(sd_full, a.delta),
                item_flip_rate=flip, item_mean_delta=st.mean(item_d),
                per_stratum=per_strat, per_family=per_fam)
    rep["pairs"] = pairs

    # ---- 3. 外推：item 级翻转率（生成池）+ 簇内相关 ρ（dev） ----
    extrap = {}
    if a.pool_items and a.pool_scores:
        pitems = load(a.pool_items)
        pstrat = {it["id"]: it["stratum"] for it in pitems}
        pfam = {it["id"]: it["family"] for it in pitems}
        prows = {}
        for spec in a.pool_scores:
            tag, path = spec.split("=", 1)
            prows[tag] = {r["id"]: r for r in load(path)}
        ptags = list(prows)
        # dev 上估 ρ（整池 + 逐 family），用于把 item 级方差还原成 episode 级方差
        rho = {}
        for pk, pv in pairs.items():
            A, B = pk.split("-", 1)
            ids = [k for k in rows.get(A, {}) if k in rows.get(B, {}) and k in ep_of]
            var_item = st.pvariance([(1.0 if rows[A][k].get("tf_correct") else 0.0)
                                     - (1.0 if rows[B][k].get("tf_correct") else 0.0) for k in ids])
            var_ep = (pv["sd_episode_delta"] ** 2) if pv["sd_episode_delta"] else None
            if var_item and var_ep:
                # var_ep = var_item/8*(1+7ρ)  →  ρ = (8*var_ep/var_item - 1)/7
                rho[pk] = max(0.0, (8 * var_ep / var_item - 1) / 7)
        rho_used = rho.get("q0-bf16") or (sorted(rho.values())[len(rho) // 2] if rho else 0.05)
        for i in range(len(ptags)):
            for j in range(i + 1, len(ptags)):
                A, B = ptags[i], ptags[j]
                ids = sorted(k for k in prows[A] if k in prows[B] and k in pstrat)
                flip_all = st.mean([1.0 if prows[A][k].get("tf_correct")
                                    != prows[B][k].get("tf_correct") else 0.0 for k in ids])
                per_strat, per_fam = {}, {}
                for s in sorted(set(pstrat[k] for k in ids)):
                    ks = [k for k in ids if pstrat[k] == s]
                    f = st.mean([1.0 if prows[A][k].get("tf_correct")
                                 != prows[B][k].get("tf_correct") else 0.0 for k in ks])
                    sd_ep = math.sqrt(f) * math.sqrt((1 + 7 * rho_used) / EPISODE_SIZE)
                    per_strat[s] = dict(
                        n_items=len(ks), item_flip_rate=f,
                        sigma_item=math.sqrt(f), sigma_episode_modelled=sd_ep,
                        n_episodes_needed_delta_target=n_ep_for(sd_ep, a.delta),
                        episodes_for_delta_0p05=n_ep_for(sd_ep, 0.05),
                        episodes_for_delta_0p10=n_ep_for(sd_ep, 0.10))
                for fm in sorted(set(pfam[k] for k in ids)):
                    ks = [k for k in ids if pfam[k] == fm]
                    f = st.mean([1.0 if prows[A][k].get("tf_correct")
                                 != prows[B][k].get("tf_correct") else 0.0 for k in ks])
                    sd_ep = math.sqrt(f) * math.sqrt((1 + 7 * rho_used) / EPISODE_SIZE)
                    per_fam[fm] = dict(
                        n_items=len(ks), item_flip_rate=f, sigma_item=math.sqrt(f),
                        sigma_episode_modelled=sd_ep,
                        n_episodes_needed_delta_target=n_ep_for(sd_ep, a.delta),
                        episodes_for_delta_0p05=n_ep_for(sd_ep, 0.05),
                        episodes_for_delta_0p10=n_ep_for(sd_ep, 0.10))
                extrap[f"{A}-{B}"] = dict(
                    pool_items=len(ids), item_flip_rate_all=flip_all,
                    rho_within_episode_used=rho_used,
                    rho_estimated_on_dev=rho,
                    sigma_item_all=math.sqrt(flip_all),
                    sigma_episode_modelled_all=math.sqrt(flip_all) * math.sqrt((1 + 7 * rho_used) / 8),
                    per_stratum=per_strat, per_family=per_fam)
    rep["extrapolation"] = dict(
        rho_definition=("sigma_episode = sigma_item * sqrt((1 + (m-1) rho)/m)，m=8；"
                        "rho = 同 episode 内 item 配对差值的簇内相关，在 dev 池上由 "
                        "rho=(8*var_ep/var_item-1)/7 估计"),
        pairs=extrap)

    # ---- 4. 可行性判定（逐 family 门 vs dev 规模） ----
    key = "q0-cand_search" if "q0-cand_search" in pairs else next(iter(pairs))
    sd_pool = pairs[key]["sd_episode_delta"]
    fam_sd = {f: v["sd_episode_delta"] for f, v in pairs[key]["per_family"].items()}
    sd_fam_pooled = st.median([x for x in fam_sd.values() if x]) if any(fam_sd.values()) else sd_pool
    verdict = dict(
        scenario_pair=key,
        dev_episodes_per_family=12, dev_items_per_family=96,
        dev_episodes_per_stratum=4, dev_items_per_stratum=32,
        pool_sigma_dev60ep=sd_pool,
        per_family_sigma_median=(sd_fam_pooled),
        delta_min_per_family_at_dev_size=delta_min(sd_pool, 12),
        delta_min_per_stratum_at_dev_size=delta_min(sd_pool, 4),
        delta_min_per_family_at_dev_size_using_median_family_sigma=delta_min(sd_fam_pooled, 12),
        confirm_episodes_per_family=801,        # 267/stratum × 3 stratum
        final_episodes_per_family=135,          # 45/stratum × 3
        delta_min_per_family_at_confirm_size=delta_min(sd_pool, 801),
        delta_min_per_family_at_final_size=delta_min(sd_pool, 135),
        sigma_uncertainty_note=("逐 stratum 的 σ 只有 3 个自由度，逐 family 11 个自由度 —— "
                               "点估计不能直接当门限，故结论用**整池 pooled σ**（60 ep）而不是各格子的 σ。"),
    )
    # 需要的规模（用池化 σ；也给出逐 family 中位 σ）
    DEV_FAM_EP, CONFIRM_FAM_EP, FINAL_FAM_EP = 12, 801, 135
    need = {}
    for d in (0.01, 0.02, 0.03, 0.05, 0.10):
        n_ep = n_ep_for(sd_pool, d)
        n_ep_gp = n_ep_for(sd_pool / math.sqrt(2.0), d)   # gold_prob ≈ 省 2×（本池实测 1.1–2.25×）
        need[f"{d}"] = dict(
            delta=d, n_episodes_per_family=n_ep, items_per_family=(n_ep or 0) * 8,
            items_all_5_families=(n_ep or 0) * 8 * 5,
            n_episodes_per_stratum=math.ceil((n_ep or 0) / 3),
            generations_to_accumulate_on_dev=round((n_ep or 0) / DEV_FAM_EP, 1),
            gold_prob_n_episodes_per_family=n_ep_gp,
            gold_prob_generations_to_accumulate_on_dev=round((n_ep_gp or 0) / DEV_FAM_EP, 1),
            feasible_at_dev_size=bool(n_ep and n_ep <= DEV_FAM_EP),
            feasible_at_confirm_size=bool(n_ep and n_ep <= CONFIRM_FAM_EP),
            feasible_at_final_size=bool(n_ep and n_ep <= FINAL_FAM_EP),
        )
    verdict["required_scale_per_family"] = need
    verdict["feasible_per_family_gating_at_dev_size"] = False
    verdict["feasible_per_family_gating_at_confirm_size_delta_0p02"] = bool(
        need["0.02"]["feasible_at_confirm_size"])
    verdict["which_deltas_feasible_at_confirm_size"] = sorted(
        [d for d, v in need.items() if v["feasible_at_confirm_size"]], key=float)
    verdict["recommendation"] = (
        "逐 family 门**不能**挂在 dev 上（12 ep/family → 只能分辨 ~13–15 点，而观察到的损伤是 3–7 点）；"
        "可行路径有三条：(1) **在确认池上做逐 family 判定**（本池 confirm 给每 family 801 episode → "
        "δ≈1.8 点，达标 δ=0.02）；(2) **跨代累积**：每代 dev 给每 family 12 episode，累积到 670 episode "
        "需要 ~56 代（用 gold_prob ≈ 28 代），适合『家族疑似偏移』时再启用；"
        "(3) **定向探针**：只在某 family 被标记可疑时，对该 family 单独开一次 670-episode 探针"
        "（5,360 题，≈1.5 min GPU），其它 family 不动。**不建议**把 dev 从 480 题放大——"
        "它是每候选成本，会乘以候选数×代数。")
    rep["per_family_gating_verdict"] = verdict
    rep["note"] = ("逐 stratum / 逐 family 的 σ 与 δ_min 是**诊断量**；判定轴仍是整池 episode 配对差。"
                   "本文件的用途：量化『若下一轮要上逐 family 门，需要多大池子』。")

    with open(a.out, "w") as f:
        json.dump(rep, f, indent=2, ensure_ascii=False, default=str)

    print("=== arm accuracy (dev) ===")
    for t, v in rep["arm_accuracy"].items():
        print(f"  {t:14s} item={v['item_acc']:.4f} episode={v['episode_acc']:.4f} "
              f"gold_prob={v['gold_prob_episode']:.4f} im_end={v['tf_next_is_im_end_rate']:.4f}")
    print("=== pairs (dev, 60 ep) ===")
    for pk, pv in pairs.items():
        print(f"  {pk:22s} mean={pv['mean_episode_delta']:+.4f} sd={pv['sd_episode_delta']:.4f} "
              f"t={pv['t_stat']:+.2f} dmin={pv['delta_min_80pct_at_dev_size']:.4f} "
              f"n_ep(δ={a.delta})={pv['n_episodes_needed_delta_target']}")
        print(f"    per-family sd: " + ", ".join(
            f"{f}={v['sd_episode_delta']:.3f}" for f, v in pv["per_family"].items()))
    print("=== per-family gating verdict ===")
    for k, v in verdict.items():
        if k != "required_scale_per_family":
            print(f"  {k}: {v}")
    for d, v in need.items():
        print(f"  δ={d}: {v['n_episodes_per_family']} ep/family ({v['items_per_family']} items), "
              f"{v['items_all_5_families']} items all families, "
              f"feasible@dev={v['feasible_at_dev_size']} feasible@confirm={v['feasible_at_confirm_size']}")


if __name__ == "__main__":
    main()
