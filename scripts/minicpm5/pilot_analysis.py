"""先导分析：episode 级配对统计、σ 与样本量输入（16 票交付给 13/14）。

输入 = score_taskset.py 的逐题 JSONL（多个候选/多个口径）+ 三池 JSONL（带 episode 字段）。
输出：
  * 每臂准确率（TF 主轴 / GEN 辅助），总量与逐家族；
  * episode 级配对差值（d_e = acc_A(ep) - acc_B(ep)）：均值、σ（episode 级）、SE、符号翻转检验、
    bootstrap 置信区间；
  * 三措辞判定（复用本仓 es_inference.classify_outcome，可用时；否则用文档化的等价实现）；
  * 样本量输入：n_ep = (z_{1-a/2}+z_{1-b})^2 σ^2 / δ^2，对 δ 网格给出；
  * TF 与 GEN 的一致性（逐题与 episode 级）；
  * 每候选评分耗时（从 summary JSON 读取，若给出）。

用法（gooo）：
  python3 pilot_analysis.py --scores q0_dev=scores_q0_dev.jsonl --scores bf16_dev=... \
      --items taskset_dev.jsonl --out pilot_dev.json [--delta 0.02] [--summaries ...]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics as st
import sys

Z_ALPHA2 = 1.959963984540054   # 双侧 0.05
Z_BETA = 0.8416212335729143    # 80% 功效


def load_jsonl(p):
    return [json.loads(l) for l in open(p)]


def episode_acc(rows, key, ep_of):
    """返回 {episode: accuracy}（episode 必须在 ep_of 内）。"""
    acc, n = {}, {}
    for r in rows:
        if r["id"] not in ep_of:
            continue
        e = ep_of[r["id"]]
        v = 1.0 if r.get(key) else 0.0
        acc[e] = acc.get(e, 0.0) + v
        n[e] = n.get(e, 0) + 1
    return {e: acc[e] / n[e] for e in acc}, n


def sign_flip_test(deltas, seed=20260910, mc=200000):
    """episode 符号翻转零假设（精确枚举 <=20 个 episode，否则 Monte Carlo）。"""
    n = len(deltas)
    obs = abs(sum(deltas) / n)
    if n <= 20:
        from itertools import product
        cnt = tot = 0
        for signs in product((1, -1), repeat=n):
            m = abs(sum(s * d for s, d in zip(signs, deltas)) / n)
            tot += 1
            cnt += (m >= obs - 1e-12)
        return dict(method="exact", p=cnt / tot, n=n)
    rng = random.Random(seed)
    cnt = 0
    for _ in range(mc):
        m = abs(sum((1 if rng.random() < 0.5 else -1) * d for d in deltas) / n)
        cnt += (m >= obs - 1e-12)
    return dict(method="monte-carlo", p=(cnt + 1) / (mc + 1), n=n, draws=mc)


def bootstrap_ci(deltas, seed=20260910, draws=10000, alpha=0.05):
    rng = random.Random(seed)
    n = len(deltas)
    means = []
    for _ in range(draws):
        means.append(sum(deltas[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    lo = means[int(alpha / 2 * draws)]
    hi = means[int((1 - alpha / 2) * draws) - 1]
    return lo, hi


def classify(mean, lo, hi, delta):
    """es_inference.classify_outcome 的等价实现（可用时直接用仓库模块）。"""
    try:
        sys.path.insert(0, os.path.expanduser("~/Code/shadow-peft/src"))
        from shadow_lab.es_inference import classify_outcome  # type: ignore
        return classify_outcome(mean=mean, lower=lo, upper=hi, delta=delta), "es_inference"
    except Exception:
        if hi < delta:
            v = "practical-gain-excluded"
        elif lo > 0 and mean >= delta:
            v = "gain-supported"
        else:
            v = "insufficient-evidence"
        return dict(verdict=v, mean=mean, lower=lo, upper=hi, delta=delta), "replica"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", action="append", required=True, help="tag=path.jsonl")
    ap.add_argument("--items", required=True, help="池 JSONL（含 episode/pool）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--delta", type=float, default=0.02)
    ap.add_argument("--summaries", nargs="*", default=[])
    a = ap.parse_args()

    items = load_jsonl(a.items)
    ep_of = {it["id"]: it["episode"] for it in items if "episode" in it}
    fam_of = {it["id"]: it["family"] for it in items}

    arms = {}
    for spec in a.scores:
        tag, path = spec.split("=", 1)
        rows = load_jsonl(path)
        arms[tag] = {r["id"]: r for r in rows}

    report = dict(items_file=os.path.basename(a.items), n_items=len(items),
                  n_episodes=len(set(ep_of.values())), arms={}, pairs={}, power={},
                  delta_default=a.delta, generated_at=__import__("time").strftime("%Y-%m-%dT%H:%M:%S"))

    for tag, rows in arms.items():
        ids = [i for i in rows if i in ep_of]
        per = {}
        for i in ids:
            f = fam_of[i]
            d = per.setdefault(f, [0, 0])
            d[0] += 1
            d[1] += int(bool(rows[i].get("tf_correct")))
        report["arms"][tag] = dict(
            n=len(ids),
            tf_acc=sum(1 for i in ids if rows[i].get("tf_correct")) / len(ids),
            gen_acc=(sum(1 for i in ids if rows[i].get("gen_correct")) /
                     max(1, sum(1 for i in ids if "gen_correct" in rows[i]))),
            n_gen=sum(1 for i in ids if "gen_correct" in rows[i]),
            per_family={k: dict(n=v[0], tf_acc=v[1] / v[0]) for k, v in sorted(per.items())},
        )
        acc, nn = episode_acc([rows[i] for i in ids], "tf_correct", ep_of)
        report["arms"][tag]["episode_acc"] = {k: round(v, 4) for k, v in sorted(acc.items())}
        report["arms"][tag]["episode_acc_mean"] = st.mean(acc.values())
        report["arms"][tag]["episode_acc_sd"] = st.pstdev(acc.values())

    tags = list(arms)
    for i in range(len(tags)):
        for j in range(i + 1, len(tags)):
            A, B = tags[i], tags[j]
            ids = [k for k in arms[A] if k in arms[B] and k in ep_of]
            accA, _ = episode_acc([arms[A][k] for k in ids], "tf_correct", ep_of)
            accB, _ = episode_acc([arms[B][k] for k in ids], "tf_correct", ep_of)
            eps = sorted(set(accA) & set(accB))
            d = [accA[e] - accB[e] for e in eps]
            mean = st.mean(d)
            sd = st.pstdev(d) if len(d) > 1 else 0.0
            lo, hi = bootstrap_ci(d)
            verdict, vmod = classify(mean, lo, hi, a.delta)
            report["pairs"][f"{A}-{B}"] = dict(
                n_episodes=len(eps), mean_episode_delta=mean, sd_episode_delta=sd,
                se_episode_delta=(sd / math.sqrt(len(eps))) if eps else None,
                item_level_acc_delta=(sum(1 for k in ids if arms[A][k].get("tf_correct")) -
                                      sum(1 for k in ids if arms[B][k].get("tf_correct"))) / len(ids),
                flip_rate_item=(sum(1 for k in ids
                                    if bool(arms[A][k].get("tf_correct")) != bool(arms[B][k].get("tf_correct")))
                                / len(ids)),
                sign_flip_test=sign_flip_test(d), bootstrap_ci=[lo, hi],
                three_wording=verdict, verdict_module=vmod,
                wins=sum(1 for x in d if x > 0), losses=sum(1 for x in d if x < 0),
                ties=sum(1 for x in d if x == 0),
            )

    # 样本量输入（配对、episode 为单位）
    for name, p in report["pairs"].items():
        sd = p["sd_episode_delta"]
        report["power"][name] = {
            f"delta_{d}": dict(delta=d, n_episodes_needed=math.ceil(
                ((Z_ALPHA2 + Z_BETA) ** 2) * (sd ** 2) / (d ** 2)) if sd else None,
                items_needed=int(math.ceil(((Z_ALPHA2 + Z_BETA) ** 2) * (sd ** 2) / (d ** 2)) * 8)
                if sd else None)
            for d in (0.005, 0.01, 0.02, 0.03, 0.05)}
        report["power"][name]["sd_used"] = sd
        report["power"][name]["note"] = ("n_ep = (z_.975+z_.8)^2 * sd^2 / delta^2；"
                                        "items = n_ep * 8（本题集 episode=8 题）")

    # TF vs GEN 一致性
    for tag, rows in arms.items():
        both = [r for r in rows.values() if "tf_correct" in r and "gen_correct" in r]
        if both:
            agree = sum(1 for r in both if bool(r["tf_correct"]) == bool(r["gen_correct"])) / len(both)
            p0 = sum(1 for r in both if r["tf_correct"]) / len(both)
            p1 = sum(1 for r in both if r["gen_correct"]) / len(both)
            pe = p0 * p1 + (1 - p0) * (1 - p1)
            report["arms"][tag]["tf_gen_item_agreement"] = agree
            report["arms"][tag]["tf_gen_cohen_kappa"] = ((agree - pe) / (1 - pe)) if pe < 1 else None
            ea, _ = episode_acc([r for r in both], "tf_correct", ep_of)
            eb, _ = episode_acc([r for r in both], "gen_correct", ep_of)
            ks = sorted(set(ea) & set(eb))
            if len(ks) > 2:
                ma, mb = st.mean([ea[k] for k in ks]), st.mean([eb[k] for k in ks])
                cov = sum((ea[k] - ma) * (eb[k] - mb) for k in ks) / len(ks)
                sa = math.sqrt(sum((ea[k] - ma) ** 2 for k in ks) / len(ks))
                sb = math.sqrt(sum((eb[k] - mb) ** 2 for k in ks) / len(ks))
                report["arms"][tag]["tf_gen_episode_pearson_r"] = cov / (sa * sb) if sa and sb else None

    for sp in a.summaries:
        tag, path = sp.split("=", 1)
        try:
            s = json.load(open(path))
            report.setdefault("timing", {})[tag] = dict(
                engine_load_s=s.get("engine_load_s"), tf_seconds=s.get("tf_timing", {}).get("tf_seconds"),
                gen_seconds=s.get("gen_timing", {}).get("gen_seconds"),
                tf_tok_s=s.get("tf_timing", {}).get("tf_tokens_per_s"),
                gen_tok_s=s.get("gen_timing", {}).get("gen_tokens_per_s"),
                n_items=s.get("n_items"))
        except Exception as e:
            report.setdefault("timing", {})[tag] = f"err {e}"

    with open(a.out, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    print(json.dumps({k: report[k] for k in ("arms", "pairs", "power") if k in report},
                     indent=2, ensure_ascii=False, default=str)[:4000])


if __name__ == "__main__":
    main()
