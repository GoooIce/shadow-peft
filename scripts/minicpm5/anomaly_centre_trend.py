"""19 票 Part 1（补）：上次搜索有没有把中心推向"不作答"？

上游 run 的逐代记录只存了 `tf_correct` / `tf_gold_logprob`（没有首个 gold 位的 argmax），
所以"逐代的未作答率"只能用一个**代理量**：gold token 的平均 logprob。做法：

 1. 用本票刚重评过的 v2 dev 行（q0 与 centre gen17 两臂共 960 条，**有精确的
    `tf_first_argmax`**）标定代理量的阈值（最大化 Youden J），并报 AUC 与误判率；
 2. 把标定好的阈值套到 `records/centre_gen*.json` 与 `records/cand_gen*_*.json`
    （885 个候选）上，得到逐代的"未作答率代理"趋势（总体 + logic/ordering 两族）。

用法（gooo，纯 CPU）：
  python3 anomaly_centre_trend.py --exact out_v2/records --records ~/minicpm5-search/out/formal/records --out trend.json
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os


def bail_exact(row):
    return row.get("tf_first_argmax") == 130073


def auc(scores, labels):
    """rank-based AUC（scores 越大越像正类）。"""
    pairs = sorted(zip(scores, labels))
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if not n_pos or not n_neg:
        return None
    rank_sum, i = 0.0, 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + j) / 2 + 1
        for _, lab in pairs[i:j + 1]:
            if lab:
                rank_sum += avg_rank
        i = j + 1
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exact", required=True, help="本票重评的 records 目录（有 tf_first_argmax）")
    ap.add_argument("--records", required=True, help="上游 run 的 records 目录")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    # ---------------- 1. 标定 -------------------------------------------------
    cal = []
    for name in ("q0_reference.json", "centre_gen17.json"):
        p = os.path.join(a.exact, name)
        if not os.path.exists(p):
            continue
        for r in json.load(open(p))["rows"]:
            if r.get("tf_gold_logprob") is not None:
                cal.append((r["tf_gold_logprob"], bail_exact(r)))
    scores = [s for s, _ in cal]
    labels = [1 if b else 0 for _, b in cal]
    a_roc = auc(scores, labels)              # 代理量 = logprob，越小越像未作答 → AUC 按 -score 报
    a_roc = 1 - a_roc if a_roc is not None else None
    grid = sorted({round(s, 2) for s in scores})
    best = None
    for t in grid:
        tp = sum(1 for s, l in zip(scores, labels) if l and s <= t)
        fn = sum(1 for s, l in zip(scores, labels) if l and s > t)
        fp = sum(1 for s, l in zip(scores, labels) if not l and s <= t)
        tn = sum(1 for s, l in zip(scores, labels) if not l and s > t)
        j = tp / (tp + fn) + tn / (tn + fp) - 1 if (tp + fn) and (tn + fp) else -1
        if best is None or j > best["youden"]:
            best = dict(threshold=t, youden=j, tp=tp, fn=fn, fp=fp, tn=tn,
                        tpr=tp / (tp + fn), fpr=fp / (fp + tn))
    report = {"calibration_rows": len(cal), "n_true_bail": sum(labels),
              "auc_logprob_as_bail_score": a_roc, "best_threshold": best}

    # ---------------- 2. 套到上游 run 的逐代记录 -------------------------------
    def proxy_rate(rows, thr, fam=None):
        rs = [r for r in rows if r.get("tf_gold_logprob") is not None
              and (fam is None or r.get("family") == fam)]
        if not rs:
            return None
        return sum(1 for r in rs if r["tf_gold_logprob"] <= thr) / len(rs)

    thr = best["threshold"]
    per_gen = {}
    for gen in range(17):
        cand = sorted(glob.glob(os.path.join(a.records, f"cand_gen{gen:02d}_*.json")))
        rates, logic, ordering = [], [], []
        for p in cand:
            d = json.load(open(p))
            rows = d["rows"]
            rates.append(proxy_rate(rows, thr))
            logic.append(proxy_rate(rows, thr, "logic"))
            ordering.append(proxy_rate(rows, thr, "ordering"))
        if not rates:
            continue
        rates = [r for r in rates if r is not None]
        per_gen[gen] = {
            "n_candidates": len(rates),
            "proxy_bail_rate_mean": sum(rates) / len(rates),
            "proxy_bail_rate_median": sorted(rates)[len(rates) // 2],
            "proxy_bail_rate_min": min(rates), "proxy_bail_rate_max": max(rates),
            "proxy_bail_logic_mean": sum(x for x in logic if x is not None) / max(1, len([x for x in logic if x is not None])),
            "proxy_bail_ordering_mean": sum(x for x in ordering if x is not None) / max(1, len([x for x in ordering if x is not None])),
        }
    report["per_generation_candidate_proxy"] = per_gen

    # gen00 的中心 = q0：代理值应当≈实测的 1.25%
    c0 = os.path.join(a.records, "centre_gen00.json")
    if os.path.exists(c0):
        rows = json.load(open(c0))["rows"]
        report["centre_gen00_proxy_rate"] = proxy_rate(rows, thr)
        report["centre_gen00_proxy_by_family"] = {
            f: proxy_rate(rows, thr, f) for f in ("logic", "ordering")}
    report["exact"] = {
        "note": "本票重评的精确值（首个 gold 位 argmax == 130073）",
    }
    json.dump(report, open(a.out, "w"), indent=2, ensure_ascii=False, sort_keys=True)
    print(json.dumps(report, indent=2, ensure_ascii=False)[:4000])


if __name__ == "__main__":
    main()
