"""v3 去污染（v3 轮）：**四路方法与阈值逐字沿用 16/17 票**，新增两项报告。

新增（本票要求）：
  1. `hits_by_layer` / `per_layer_hit_items`：每一路的命中按**语料层**归因（v2 只记了总量）；
  2. `rl2609_layer_report`：**单独报告与 UltraData-RL-2609 层的重叠**（Math / Knowledge /
     Long_Context 全量 + Code 抽样），因为"下一轮在该层上训练"是一个现实选项，需要显式知道
     题集与该层是否相交——与 SFT-2605 / SFT-Agent 层的重叠分开统计。

阈值与实现从 `decontaminate.py` 直接 import，保证与 16/17 票**逐字一致**（不复制常量）。

用法（goooinext）：
  python3 decontaminate_v3.py --items pool_v3.jsonl --data-root ~/data/ultradata \
      --out decontamination_report_v3.json
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import decontaminate as D  # noqa: E402  四路常量与工具函数复用（阈值不在本文件里重定义）

RL2609_PREFIX = "rl2609"          # UltraData-RL-2609 层（Math / Knowledge / Long_Context）
RL2609_CODE_PREFIX = "rl2609_code_samples"   # UltraData-RL-2609/Code 抽样
SFT2605_PREFIX = "sft2605"        # UltraData-SFT-2605 抽样
SFTAGENT_PREFIX = "sft_agent2609"  # UltraData-SFT-Agent-2609 抽样


def layer_group(layer: str) -> str:
    """把语料层归到 4 个来源组之一（用于 RL-2609 的单独报告）。"""
    if layer.startswith(RL2609_PREFIX + "/") or layer == RL2609_PREFIX:
        return "ultradata_rl_2609"
    if layer.startswith(RL2609_CODE_PREFIX):
        return "ultradata_rl_2609_code_sample"
    if layer.startswith(SFT2605_PREFIX):
        return "ultradata_sft_2605_sample"
    if layer.startswith(SFTAGENT_PREFIX):
        return "ultradata_sft_agent_2609_sample"
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sample", type=int, default=0)
    a = ap.parse_args()

    items = [json.loads(l) for l in open(a.items)]
    print(f"items: {len(items)}", flush=True)

    # ---- 条目侧索引（与 decontaminate.py 逐字同口径） ----
    item_q_sha, item_sig, item_ngrams, item_anchors, item_qn = {}, {}, {}, {}, {}
    n_digits_ok = 0
    for it in items:
        q, ans = it["question"], it["answer"]
        qn = D.norm_text(q)
        q_toks = qn.split()
        item_qn[it["id"]] = qn
        item_q_sha.setdefault(D.sha1(qn), []).append(it["id"])
        nums_q = D.item_numbers(q)
        if len(set(nums_q)) >= D.SIG_MIN_NUMBERS and D.signature_applicable(nums_q) \
                and __import__("re").fullmatch(r"-?\d+", ans.strip()):
            item_sig.setdefault((tuple(sorted(nums_q)), int(ans)), []).append(it["id"])
            n_digits_ok += 1
        n = min(13, len(q_toks))
        if n >= 5:
            for i in range(len(q_toks) - n + 1):
                item_ngrams.setdefault(tuple(q_toks[i:i + n]), []).append(it["id"])
        for x in set(nums_q):
            if len(str(x)) >= 3:
                item_anchors.setdefault(str(x), []).append(it["id"])
    print(f"items with numeric signature: {n_digits_ok}/{len(items)}; ngram keys: {len(item_ngrams)}; "
          f"anchors: {len(item_anchors)}", flush=True)

    stats = {"rows": 0, "bytes": 0, "tokens": 0, "rows_ngram_checked": 0,
             "rows_blocked_substring": 0, "long_rows_skipped_ngram": 0}
    checks = ("A_signature", "A2_exact_question", "B_ngram", "C_blocked_substring")
    hits = {k: {} for k in checks}          # check -> item_id -> [locations]
    hit_layers = {k: {} for k in checks}     # check -> item_id -> [layer]
    per_layer = {}
    t0 = time.time()
    files_done = 0
    for layer, path in D.iter_corpus(a.data_root):
        pl = per_layer.setdefault(layer, {"files": 0, "rows": 0, "bytes": 0, "hits": 0,
                                          "group": layer_group(layer)})
        pl["files"] += 1
        files_done += 1
        if a.sample and files_done > a.sample:
            break
        with open(path, "r", errors="replace") as fh:
            for line in fh:
                line = line.rstrip("\n")
                if len(line) < 3:
                    continue
                stats["rows"] += 1
                stats["bytes"] += len(line)
                pl["rows"] += 1
                pl["bytes"] += len(line)
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                user, answer = D.user_and_answer_text(row)
                un = D.norm_text(user)
                toks = un.split()
                stats["tokens"] += len(toks)
                row_hit = False

                def record(check, ids):
                    nonlocal row_hit
                    row_hit = True
                    for iid in ids:
                        if iid not in hit_layers[check]:
                            hit_layers[check][iid] = []
                        if layer not in hit_layers[check][iid]:
                            hit_layers[check][iid].append(layer)
                        hits[check].setdefault(iid, []).append(f"{layer}:{stats['rows']}")

                # A2 精确题干
                h = D.sha1(un)
                if h in item_q_sha:
                    record("A2_exact_question", item_q_sha[h])

                # A 签名
                unums = D.item_numbers(user)
                if D.SIG_MIN_NUMBERS <= len(unums) <= D.SIG_MAX_NUMBERS \
                        and D.signature_applicable(unums):
                    rans = D.canonical_row_answer(answer)
                    if rans is not None:
                        key = (tuple(sorted(unums)), rans)
                        if key in item_sig:
                            record("A_signature", item_sig[key])

                # B n-gram
                if len(toks) <= D.MAX_NGRAM_ROWS_TOKENS:
                    stats["rows_ngram_checked"] += 1
                    n = min(13, len(toks))
                    if n >= 5:
                        for i in range(len(toks) - n + 1):
                            g = tuple(toks[i:i + n])
                            if g in item_ngrams:
                                record("B_ngram", item_ngrams[g])
                else:
                    stats["long_rows_skipped_ngram"] += 1
                    cand = set()
                    for x in set(unums):
                        s = str(x)
                        if s in item_anchors:
                            cand.update(item_anchors[s])
                    if cand:
                        stats["rows_blocked_substring"] += 1
                        for iid in cand:
                            if item_qn[iid] and item_qn[iid] in un:
                                record("C_blocked_substring", [iid])
                if row_hit:
                    pl["hits"] += 1
        if stats["rows"] % 20000 < 500:
            print(f"  [{layer}] rows={stats['rows']} t={time.time()-t0:.0f}s", flush=True)

    hit_ids = sorted(set().union(*[set(v) for v in hits.values()]) if hits else set())
    fam_of = {i["id"]: i["family"] for i in items}
    strat_of = {i["id"]: i["stratum"] for i in items}

    # 命中按来源组归因（一个条目可能同时落在多个组；这里记"是否与 RL-2609 层相交"）
    rl2609_ids = sorted({iid for chk in hits for iid in hits[chk]
                         if any(layer_group(l) == "ultradata_rl_2609" for l in hit_layers[chk][iid])})
    rl2609_code_ids = sorted({iid for chk in hits for iid in hits[chk]
                              if any(layer_group(l) == "ultradata_rl_2609_code_sample"
                                     for l in hit_layers[chk][iid])})
    sft2605_ids = sorted({iid for chk in hits for iid in hits[chk]
                          if any(layer_group(l) == "ultradata_sft_2605_sample"
                                 for l in hit_layers[chk][iid])})
    sftagent_ids = sorted({iid for chk in hits for iid in hits[chk]
                           if any(layer_group(l) == "ultradata_sft_agent_2609_sample"
                                  for l in hit_layers[chk][iid])})
    layer_totals = collections.Counter()
    for layer, pl in per_layer.items():
        layer_totals[pl["group"]] += pl["rows"]

    report = {
        "schema": "minicpm5-decontamination-v3",
        "method": {
            "A_signature": (f"题干整数多重集(剥列表编号,含重复,排序)精确相等 且 答案整数精确相等；"
                            f"仅对含 >= {D.SIG_BIG} 的数的条目生效（小数字多重集撞车率过高）"),
            "A2_exact_question": "sha1(lowercase alnum-folded question) == sha1(row user text); 全语料",
            "B_ngram": f"n=min(13,#question tokens)>=5 词 n-gram 交集；对 user 侧 <= {D.MAX_NGRAM_ROWS_TOKENS} 词的行",
            "C_blocked_substring": "长行(>3000 词)且含条目 >=3 位数字锚：规范化题干子串包含",
            "unchanged_from": "16 票（2026-09-10）定义、17 票原样重跑；本脚本从 decontaminate.py import 常量与函数，未重定义阈值",
        },
        "corpus_root": a.data_root,
        "stats": stats,
        "per_layer": per_layer,
        "layer_rows_by_group": dict(layer_totals),
        "n_items": len(items),
        "n_items_hit": len(hit_ids),
        "hit_rate": len(hit_ids) / len(items),
        "hits_by_check": {k: len(v) for k, v in hits.items()},
        "hit_item_ids": hit_ids,
        "hits_by_family": dict(collections.Counter(fam_of[x] for x in hit_ids)),
        "hits_by_stratum": dict(collections.Counter(strat_of[x] for x in hit_ids)),
        "hits_by_check_and_layer": {
            chk: dict(collections.Counter(l for iid in hits[chk] for l in hit_layers[chk][iid]))
            for chk in checks},
        "hit_item_layers": {iid: {chk: hit_layers[chk][iid] for chk in checks if iid in hit_layers[chk]}
                            for iid in hit_ids},
        "hit_examples": {k: dict(list(v.items())[:10]) for k, v in hits.items()},
        "rl2609_layer_report": {
            "question": "题集是否与 UltraData-RL-2609 层相交（下一轮的候选训练语料）？",
            "ultradata_rl_2609_full_scan": True,
            "ultradata_rl_2609_layers": sorted([l for l in per_layer
                                                if layer_group(l) == "ultradata_rl_2609"]),
            "ultradata_rl_2609_code_sample_only": True,
            "n_hits_in_rl2609_layer": len(rl2609_ids),
            "hit_ids_in_rl2609_layer": rl2609_ids,
            "n_hits_in_rl2609_code_sample": len(rl2609_code_ids),
            "hit_ids_in_rl2609_code_sample": rl2609_code_ids,
            "n_hits_in_sft2605_sample": len(sft2605_ids),
            "n_hits_in_sft_agent_sample": len(sftagent_ids),
            "hit_ids_in_sft2605_sample": sft2605_ids,
            "hit_ids_in_sft_agent_sample": sftagent_ids,
            "rows_scanned_by_group": dict(layer_totals),
            "coverage_note": ("RL-2609 的 Math/Knowledge/Long_Context 为全量扫描；RL-2609/Code 与 "
                              "SFT-2605/SFT-Agent 为抽样（16/17 票同口径），抽样覆盖见 per_layer。"),
        },
        "elapsed_s": round(time.time() - t0, 1),
    }
    with open(a.out, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps({k: report[k] for k in
                      ("stats", "n_items_hit", "hit_rate", "hits_by_check",
                       "hits_by_check_and_layer", "elapsed_s")}, indent=2, ensure_ascii=False))
    print("RL-2609: hits=%d (code sample %d), SFT-2605=%d, SFT-Agent=%d" %
          (len(rl2609_ids), len(rl2609_code_ids), len(sft2605_ids), len(sftagent_ids)))


if __name__ == "__main__":
    main()
