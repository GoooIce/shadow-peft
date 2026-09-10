"""17 票：确认池统计功效分析（排序信号方差、逐 stratum 信息量、样本量）。

输入（全部来自 16 票产物，无需 GPU，纯标准库）：
  * 多臂在同一池上的逐题结果（score_taskset.py 输出，带 tf_correct / tf_gold_logprob /
    tf_token_hits / tf_tokens）；
  * 池 JSONL（带 stratum / episode）。

方法（避免 logistic 在二值信号上退化）：
  * **item 级判别力**：每个排序信号对 tf_correct 的 AUC / 点二列相关（信号能否把对题排在错题前面）；
  * **跨信号换算**：以一条"效应已被确认"的配对（默认 bf16_dev-cand_search_dev，p≈0.0014、
    CI 不含 0）为**锚**，用 dS/dacc = m_S / m_acc 得到"每 1 个准确率点，该信号动多少"，
    再用 episode 级配对差的 σ 得到 n_ep(δ) = K σ_S²/(δ·dS/dacc)²，K=(z_.975+z_.8)²=7.8489；
  * **校验**：用锚定斜率预测另一条（近邻、效应小得多）配对的信号差，与实测比较——
    实测/预测 > 1 说明该信号对"小幅变化"相对二值有放大响应。

输出：item 级信号诊断、逐配对逐信号功效表、逐 stratum 信息量与留一分析、样本量表。

用法：
  python3 pool_power_analysis.py \
      --scores q0_dev=scores_q0_dev.jsonl --scores bf16_dev=scores_bf16_dev.jsonl \
             --scores cand_search_dev=scores_cand_search_dev.jsonl \
      --items taskset_dev.jsonl --out pool_power_dev.json --anchor bf16_dev-cand_search_dev
"""
from __future__ import annotations

import argparse
import bisect
import json
import math
import os
import statistics as st
import sys

Z_A2 = 1.959963984540054   # 双侧 alpha=0.05
Z_B = 0.8416212335729143   # 80% 功效
K = (Z_A2 + Z_B) ** 2      # 7.8489
LP_FLOOR = -20.0


def load_jsonl(p):
    return [json.loads(l) for l in open(p)]


def _lp(r):
    v = r.get("tf_gold_logprob")
    return None if v is None else float(v)


# 排序信号：name -> 逐题取值函数（越大越好）
SIGNALS = {
    "acc": lambda r: (1.0 if r.get("tf_correct") else 0.0),
    "tok_ratio": lambda r: (r["tf_token_hits"] / r["tf_tokens"] if r.get("tf_tokens") else None),
    "gold_prob": lambda r: (None if _lp(r) is None else math.exp(max(LP_FLOOR, _lp(r)))),
    "lp_mean": _lp,
    "lp_clip5": lambda r: (None if _lp(r) is None else max(-5.0, _lp(r))),
    "lp_clip10": lambda r: (None if _lp(r) is None else max(-10.0, _lp(r))),
}
SIGNAL_NOTE = {
    "acc": "二值正确率 tf_correct（16 票登记主轴）",
    "tok_ratio": "gold token 命中比例 tf_token_hits/tf_tokens ∈[0,1]（gold 多为 1-2 token，取值退化为 {0,.5,1}）",
    "gold_prob": "exp(mean gold logprob)（floor -20）∈[0,1]：logprob 的概率化",
    "lp_mean": "mean gold logprob（原始无界连续量，16 票已逐题记录）",
    "lp_clip5": "per-item clip 到 [-5,0] 后平均",
    "lp_clip10": "per-item clip 到 [-10,0] 后平均",
}


def auc(scores, labels):
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return None
    neg_sorted = sorted(neg)
    tot = 0.0
    for s in pos:
        lo = bisect.bisect_left(neg_sorted, s)
        hi = bisect.bisect_right(neg_sorted, s)
        tot += lo + 0.5 * (hi - lo)
    return tot / (len(pos) * len(neg))


def point_biserial(scores, labels):
    n = len(scores)
    if n < 3:
        return None
    sd = st.pstdev(scores)
    if sd == 0:
        return None
    p = sum(labels) / n
    if p in (0.0, 1.0):
        return None
    m1 = st.mean([s for s, y in zip(scores, labels) if y == 1])
    m0 = st.mean([s for s, y in zip(scores, labels) if y == 0])
    return (m1 - m0) / sd * math.sqrt(p * (1 - p))


def n_ep_for(sd, delta):
    if not sd or not delta:
        return None
    return math.ceil(K * sd * sd / (delta * delta))


def delta_min(sd, n_ep):
    if not sd or not n_ep:
        return None
    return (Z_A2 + Z_B) * sd / math.sqrt(n_ep)


def episode_mean(per_item, ep_of, ids):
    buf = {}
    for k in ids:
        v = per_item.get(k)
        if v is None:
            continue
        buf.setdefault(ep_of[k], []).append(v)
    return {e: st.mean(v) for e, v in buf.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", action="append", required=True, help="tag=path.jsonl")
    ap.add_argument("--items", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--delta", type=float, default=0.02)
    ap.add_argument("--episode-size", type=int, default=8)
    ap.add_argument("--signals", nargs="*", default=list(SIGNALS))
    ap.add_argument("--anchor", default=None, help="用作换算锚的配对（tagA-tagB）；默认取二值 t 最大的配对")
    a = ap.parse_args()

    items = load_jsonl(a.items)
    ep_of = {it["id"]: it["episode"] for it in items if "episode" in it}
    strat_of = {it["id"]: it["stratum"] for it in items}
    arms = {}
    for spec in a.scores:
        tag, path = spec.split("=", 1)
        arms[tag] = {r["id"]: r for r in load_jsonl(path)}
    tags = list(arms)

    rep = dict(items_file=os.path.basename(a.items), n_items=len(items),
               n_episodes=len(set(ep_of.values())), episode_size=a.episode_size,
               delta_default=a.delta, signals={k: SIGNAL_NOTE[k] for k in a.signals},
               generated_at=__import__("time").strftime("%Y-%m-%dT%H:%M:%S"),
               arms={}, item_level_ranking_signal={}, pairs={}, per_stratum={}, power_table={})

    for tag, rows in arms.items():
        ids = [k for k in rows if k in ep_of]
        rep["arms"][tag] = dict(
            n=len(ids), tf_acc=st.mean([1 if rows[k].get("tf_correct") else 0 for k in ids]),
            gen_acc=(st.mean([1 if rows[k].get("gen_correct") else 0 for k in ids])
                     if all("gen_correct" in rows[k] for k in ids) else None))

    # ---- 1. item 级信号诊断（三臂池化） ----
    for name in a.signals:
        f = SIGNALS[name]
        vals, labels = {}, {}
        for tag, rows in arms.items():
            for k, r in rows.items():
                if k not in ep_of:
                    continue
                v = f(r)
                if v is None:
                    continue
                vals[k] = v
                labels[k] = 1 if r.get("tf_correct") else 0
        xs = list(vals.values())
        ys = [labels[k] for k in vals]
        rep["item_level_ranking_signal"][name] = dict(
            n=len(xs), mean=st.mean(xs), sd=st.pstdev(xs), min=min(xs), max=max(xs),
            auc_vs_tf_correct=auc(xs, ys), point_biserial_r=point_biserial(xs, ys),
            note=SIGNAL_NOTE[name])

    # ---- 2. 逐配对：逐信号的 episode 级配对差 ----
    for i in range(len(tags)):
        for j in range(i + 1, len(tags)):
            A, B = tags[i], tags[j]
            ids = sorted(k for k in arms[A] if k in arms[B] and k in ep_of)
            pair = dict(n_items=len(ids), n_episodes=len({ep_of[k] for k in ids}),
                        item_flip_rate_acc=(sum(
                            1 for k in ids
                            if bool(arms[A][k].get("tf_correct")) != bool(arms[B][k].get("tf_correct")))
                            / len(ids)),
                        acc_delta_item=(sum(1 for k in ids if arms[A][k].get("tf_correct"))
                                        - sum(1 for k in ids if arms[B][k].get("tf_correct"))) / len(ids),
                        signals={})
            for name in a.signals:
                f = SIGNALS[name]
                va = {k: f(arms[A][k]) for k in ids if f(arms[A][k]) is not None}
                vb = {k: f(arms[B][k]) for k in ids if f(arms[B][k]) is not None}
                common = sorted(set(va) & set(vb))
                ea = episode_mean(va, ep_of, common)
                eb = episode_mean(vb, ep_of, common)
                eps = sorted(set(ea) & set(eb))
                d = [ea[e] - eb[e] for e in eps]
                mean, sd = st.mean(d), (st.pstdev(d) if len(d) > 1 else 0.0)
                se = sd / math.sqrt(len(d)) if d else None
                pair["signals"][name] = dict(
                    mean_episode_delta=mean, sd_episode_delta=sd, se_episode_delta=se,
                    t_stat=(mean / se if se else None),
                    sign_consistency=(sum(1 for x in d if x > 0) / len(d)),
                )
            rep["pairs"][f"{A}-{B}"] = pair

    # ---- 3. 锚定换算：dS/dacc + n_ep(δ) ----
    anchor = a.anchor
    if anchor is None:
        anchor = max(rep["pairs"], key=lambda k: abs(rep["pairs"][k]["signals"]["acc"]["t_stat"] or 0))
    ap_ = rep["pairs"][anchor]
    m_acc_anchor = ap_["signals"]["acc"]["mean_episode_delta"]
    transfer = {}
    for name in a.signals:
        dS_dacc = (ap_["signals"][name]["mean_episode_delta"] / m_acc_anchor) if m_acc_anchor else None
        sd = ap_["signals"][name]["sd_episode_delta"]
        n_ep = n_ep_for(sd, a.delta * dS_dacc) if dS_dacc else None
        transfer[name] = dict(
            anchor_pair=anchor, dS_dacc=dS_dacc, sd_episode_delta=sd,
            n_episodes_needed_for_delta=dict(),
        )
        for dlt in (0.01, 0.015, 0.02, 0.03, 0.05):
            transfer[name]["n_episodes_needed_for_delta"][f"{dlt}"] = dict(
                delta_acc=dlt, delta_signal=(dlt * dS_dacc if dS_dacc else None),
                n_episodes_needed=n_ep_for(sd, dlt * dS_dacc) if dS_dacc else None,
                items_needed=(n_ep_for(sd, dlt * dS_dacc) * a.episode_size
                              if (dS_dacc and n_ep_for(sd, dlt * dS_dacc)) else None))
    transfer["_meta"] = dict(
        anchor_pair=anchor, anchor_acc_delta=m_acc_anchor,
        method=("dS/dacc = m_S(anchor)/m_acc(anchor)（锚配对是效应已被确认的配对）；"
                "n_ep(δ) = K * sigma_S^2 / (δ * dS/dacc)^2，K=7.8489"),
        note="n_ep 的比值（acc vs 其它信号）与锚配对的绝对效应无关，是纯粹的 SNR 比值")
    rep["signal_transfer"] = transfer

    # 校验：用锚斜率预测其它配对的信号差
    validation = {}
    for pk, pv in rep["pairs"].items():
        if pk == anchor:
            continue
        m_acc = pv["signals"]["acc"]["mean_episode_delta"]
        row = {}
        for name in a.signals:
            if name == "acc":
                continue
            pred = m_acc * transfer[name]["dS_dacc"] if transfer[name]["dS_dacc"] else None
            obs = pv["signals"][name]["mean_episode_delta"]
            row[name] = dict(predicted=pred, observed=obs,
                             observed_over_predicted=(obs / pred if pred else None))
        validation[pk] = row
    rep["transfer_validation"] = validation

    # ---- 4. 逐 stratum 信息量 + 留一（二值主轴） ----
    for i in range(len(tags)):
        for j in range(i + 1, len(tags)):
            A, B = tags[i], tags[j]
            ids = [k for k in arms[A] if k in arms[B] and k in ep_of]
            per, dd = {}, {}
            for k in ids:
                d = per.setdefault(strat_of[k], dict(n=0, flip=0, okA=0, okB=0))
                d["n"] += 1
                a_ok, b_ok = bool(arms[A][k].get("tf_correct")), bool(arms[B][k].get("tf_correct"))
                d["flip"] += int(a_ok != b_ok)
                d["okA"] += int(a_ok)
                d["okB"] += int(b_ok)
                dd.setdefault(ep_of[k], [[], []])
                dd[ep_of[k]][0].append(1.0 if a_ok else 0.0)
                dd[ep_of[k]][1].append(1.0 if b_ok else 0.0)
            dd = {e: st.mean(v[0]) - st.mean(v[1]) for e, v in dd.items()}
            ep_strat = {}
            for k in ids:
                ep_strat[ep_of[k]] = strat_of[k]
            sd_strat = {}
            for s in per:
                vals = [dd[e] for e in dd if ep_strat[e] == s]
                sd_strat[s] = vals
            for s, d in per.items():
                d["flip_rate"] = d["flip"] / d["n"]
                d["item_acc_delta"] = (d["okA"] - d["okB"]) / d["n"]
                d["stratum_episode_delta_mean"] = st.mean(sd_strat[s])
                d["stratum_episode_delta_sd"] = (st.pstdev(sd_strat[s])
                                                 if len(sd_strat[s]) > 1 else 0.0)
                d["n_episodes"] = len(sd_strat[s])
                d["delta_zero_share"] = sum(1 for x in sd_strat[s] if x == 0) / len(sd_strat[s])
            loo = {}
            for s in per:
                keep = [dd[e] for e in dd if ep_strat[e] != s]
                sd2 = st.pstdev(keep) if len(keep) > 1 else None
                loo[s] = dict(n_episodes=len(keep), mean=st.mean(keep), sd_episode_delta=sd2,
                              n_episodes_needed_for_delta=n_ep_for(sd2, a.delta))
            full_sd = st.pstdev(list(dd.values()))
            rep["per_stratum"][f"{A}-{B}"] = dict(
                full_sd_episode_delta=full_sd,
                full_n_episodes_needed=n_ep_for(full_sd, a.delta),
                per_stratum={s: d for s, d in sorted(per.items())},
                leave_one_stratum_out=dict(sorted(loo.items())))

    # ---- 5. 样本量表 ----
    rep["power_table"] = dict(
        formula="n_ep = (z_{1-a/2} + z_{1-beta})^2 * sigma^2 / delta^2；items = n_ep * episode_size",
        by_sigma={str(sg): {f"delta_{d}": dict(delta=d, n_episodes_needed=n_ep_for(sg, d),
                                               items_needed=n_ep_for(sg, d) * a.episode_size)
                            for d in (0.01, 0.015, 0.02, 0.03, 0.05)}
                  for sg in (0.16, 0.18, 0.20, 0.213, 0.24, 0.285)},
    )

    with open(a.out, "w") as f:
        json.dump(rep, f, indent=2, ensure_ascii=False, default=str)

    print("=== item-level signal diagnostic ===")
    for k, v in rep["item_level_ranking_signal"].items():
        print(f"  {k:10s} mean={v['mean']:+.4f} sd={v['sd']:.4f} range=[{v['min']:.3f},{v['max']:.3f}] "
              f"AUC={v['auc_vs_tf_correct']:.4f} r_pb={v['point_biserial_r']:.4f}")
    print(f"=== pairs (anchor={anchor}) ===")
    for pk, pv in rep["pairs"].items():
        print(f"  {pk} (n_ep {pv['n_episodes']}, acc_delta {pv['acc_delta_item']:+.4f}, "
              f"flip {pv['item_flip_rate_acc']:.3f})")
        for sk, sv in pv["signals"].items():
            print(f"     {sk:10s} mean={sv['mean_episode_delta']:+.4f} sd={sv['sd_episode_delta']:.4f} "
                  f"t={sv['t_stat']:+.2f} n_ep(2pct)={transfer[sk]['n_episodes_needed_for_delta']['0.02']['n_episodes_needed']}")
    print("=== signal transfer (n_ep for delta=0.02 accuracy) ===")
    for name in a.signals:
        t = transfer[name]["n_episodes_needed_for_delta"]["0.02"]
        print(f"  {name:10s} dS/dacc={transfer[name]['dS_dacc']:.4f} "
              f"n_ep={t['n_episodes_needed']} items={t['items_needed']}")
    print("=== transfer validation (observed / predicted) ===")
    for pk, row in rep["transfer_validation"].items():
        print(f"  {pk}: " + ", ".join(f"{k}={v['observed_over_predicted']:.2f}" for k, v in row.items()))


if __name__ == "__main__":
    main()
