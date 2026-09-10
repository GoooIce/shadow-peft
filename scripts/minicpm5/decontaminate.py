"""题集去污染：与 UltraData 各层做重叠检查（16 票，07 污染控制条款）。

方法（三层，全部记录方法/阈值/命中率）：
  A. 签名比对（全语料）：`(题干整数多重集, 答案整数)` — 同一道题（同数字同答案）无论措辞
     都会命中；对 >2 个数字的条目生效。
  A'. 规范化题干精确相等（全语料）：小写化 + 非字母数字折叠成空格后的 sha1 相等。
  B. n-gram 重叠（全语料，n = min(13, 题干词数)）：题干词 n-gram 集合与语料 user 侧文本
     的 n-gram 交集非空即命中（GPT-3 式 13-gram 口径）；仅对 user 侧文本 <= 3000 词的行
     计算（覆盖 Math/Knowledge/Code/SFT/Agent 的全部 user 侧；Long-Context 长文走 C）。
  C. 阻塞式子串比对（超长行）：只在行内含条目"数字锚"（>=3 位整数）时才做规范化子串包含检查。

语料（goooinext:~/data/ultradata，见 manifest.tsv）：
  RL-2609 Math(4) + Knowledge(2) + Long_Context(2) 全量；Code 12 shard 抽样；
  SFT-2605 no_think/Math 6 shard + Knowledge 4 shard；SFT-Agent Tool_Use/General/Search 抽样。
用法：python3 decontaminate.py --items pool.jsonl --data-root ~/data/ultradata --out report.json
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

NUM_RE = re.compile(r"\d+")
MAX_INT_DIGITS = 18
LIST_MARKER = re.compile(r"(?m)^\s*\d+\s*[).]\s*")   # 去掉题面里的编号 "1)" "2."，否则它们会被当成题目数字
SIG_BIG = 100          # 签名检查的适用门槛：题干里至少有一个 >= 100 的数（小数字多重集撞车率太高）          # 超长数字串（语料里有 7800 位的）不当整数处理，直接截断比较
NON_ALNUM = re.compile(r"[^a-z0-9]+")
MAX_NGRAM_ROWS_TOKENS = 3000
SIG_MIN_NUMBERS = 2
SIG_MAX_NUMBERS = 40


def norm_text(t: str) -> str:
    return NON_ALNUM.sub(" ", t.lower()).strip()


def item_numbers(text: str):
    """题干侧数字：先剥掉列表编号。"""
    return numbers(LIST_MARKER.sub(" ", text))


def signature_applicable(nums) -> bool:
    return max(nums) >= SIG_BIG


def numbers(text: str, cap=SIG_MAX_NUMBERS):
    out = []
    for x in NUM_RE.findall(text.replace(",", "")):
        if len(x) <= MAX_INT_DIGITS:
            out.append(int(x))
        if len(out) >= cap:
            break
    return out


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode()).hexdigest()


ANS_PATTERNS = (r"####\s*(-?\d+)", r"\\boxed\{(-?\d+)\}", r"\*\*\s*(-?\d+)\s*\*\*",
                r"(?m)^\s*(-?\d+)\s*\.?\s*$")


def canonical_row_answer(answer_text: str):
    """把语料侧参考答案收敛成一个整数；收敛不出来返回 None（宁缺勿滥，避免弱签名假命中）。"""
    s = (answer_text or "").strip()
    if not s:
        return None
    if len(s) <= 24 and re.fullmatch(r"\$?\s*-?\d+\s*\.?\s*\$?", s):
        return int(re.sub(r"[^0-9-]", "", s))
    for pat in ANS_PATTERNS:
        ms = re.findall(pat, s)
        if ms:
            try:
                return int(ms[-1])
            except ValueError:
                return None
    return None


def user_and_answer_text(row: dict):
    """返回 (user_side_text, answer_side_text)。"""
    if "query" in row:  # UltraData-RL-2609
        return str(row["query"]), str(row.get("ground_truth", ""))
    msgs = row.get("messages") or []
    user_parts, ans_parts = [], []
    for m in msgs:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        c = m.get("content")
        if isinstance(c, list):
            c = " ".join(str(p.get("text", "")) if isinstance(p, dict) else str(p) for p in c)
        c = str(c or "")
        if role == "user":
            user_parts.append(c)
        elif role == "assistant":
            ans_parts.append(c)
    return "\n".join(user_parts), "\n".join(ans_parts)


def iter_corpus(data_root, pats=None):
    pats = pats or ["rl2609/data/*/*.jsonl", "rl2609_code_samples/*", "sft2605/data/*/*/*.jsonl",
                    "sft_agent2609/data/*/*.jsonl"]
    for pat in pats:
        for path in sorted(glob.glob(os.path.join(data_root, pat))):
            rel = os.path.relpath(path, data_root)
            layer = os.path.dirname(rel) or rel
            yield layer, path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sample", type=int, default=0, help="调试：只跑前 N 行/文件")
    ap.add_argument("--patterns", nargs="*", default=None, help="调试：覆盖语料 glob")
    a = ap.parse_args()

    items = [json.loads(l) for l in open(a.items)]
    print(f"items: {len(items)}", flush=True)

    # ---- 条目侧索引 ----
    item_q_sha = {}          # sha1(norm(question)) -> [item ids]
    item_sig = {}            # (numbers, answer_int) -> [item ids]
    item_ngrams = {}         # n-gram tuple -> [item ids]
    item_anchors = {}        # "4823" -> [item ids]  (>=3 位数字锚)
    item_qn = {}
    n_digits_ok = 0
    for it in items:
        q, ans = it["question"], it["answer"]
        qn = norm_text(q)
        q_toks = qn.split()
        item_qn[it["id"]] = qn
        item_q_sha.setdefault(sha1(qn), []).append(it["id"])
        nums_q = item_numbers(q)
        if len(set(nums_q)) >= SIG_MIN_NUMBERS and signature_applicable(nums_q) \
                and re.fullmatch(r"-?\d+", ans.strip()):
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

    # ---- 语料侧扫描 ----
    stats = {"rows": 0, "bytes": 0, "tokens": 0, "rows_ngram_checked": 0,
             "rows_blocked_substring": 0, "long_rows_skipped_ngram": 0}
    hits = {"A_signature": {}, "A2_exact_question": {}, "B_ngram": {}, "C_blocked_substring": {}}
    per_layer = {}
    t0 = time.time()
    files_done = 0
    for layer, path in iter_corpus(a.data_root, a.patterns):
        pl = per_layer.setdefault(layer, {"files": 0, "rows": 0, "bytes": 0, "hits": 0})
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
                user, answer = user_and_answer_text(row)
                un = norm_text(user)
                toks = un.split()
                stats["tokens"] += len(toks)
                row_hit = False

                # A2 精确题干
                h = sha1(un)
                if h in item_q_sha:
                    row_hit = True
                    for iid in item_q_sha[h]:
                        hits["A2_exact_question"].setdefault(iid, []).append(f"{layer}:{stats['rows']}")

                # A 签名（仅对 user 侧数字合理、行不太长的行）
                unums = item_numbers(user)
                if SIG_MIN_NUMBERS <= len(unums) <= SIG_MAX_NUMBERS \
                        and signature_applicable(unums):
                    nums_t = tuple(sorted(unums))
                    rans = canonical_row_answer(answer)
                    if rans is not None:
                        key = (nums_t, rans)
                        if key in item_sig:
                            row_hit = True
                            for iid in item_sig[key]:
                                hits["A_signature"].setdefault(iid, []).append(f"{layer}:{stats['rows']}")

                # B n-gram
                if len(toks) <= MAX_NGRAM_ROWS_TOKENS:
                    stats["rows_ngram_checked"] += 1
                    for n in {min(13, len(toks))}:
                        if n >= 5:
                            for i in range(len(toks) - n + 1):
                                g = tuple(toks[i:i + n])
                                if g in item_ngrams:
                                    row_hit = True
                                    for iid in item_ngrams[g]:
                                        hits["B_ngram"].setdefault(iid, []).append(f"{layer}:{stats['rows']}")
                else:
                    stats["long_rows_skipped_ngram"] += 1
                    # C 阻塞式子串（长行）
                    cand = set()
                    for x in set(unums):
                        s = str(x)
                        if s in item_anchors:
                            cand.update(item_anchors[s])
                    if cand:
                        stats["rows_blocked_substring"] += 1
                        for iid in cand:
                            if item_qn[iid] and item_qn[iid] in un:
                                row_hit = True
                                hits["C_blocked_substring"].setdefault(iid, []).append(
                                    f"{layer}:{stats['rows']}")
                if row_hit:
                    pl["hits"] += 1
        if stats["rows"] % 20000 < 500:
            print(f"  [{layer}] rows={stats['rows']} t={time.time()-t0:.0f}s", flush=True)

    hit_ids = sorted(set().union(*[set(v) for v in hits.values()]) if hits else set())
    report = {
        "method": {
            "A_signature": (f"题干整数多重集(剥列表编号,含重复,排序)精确相等 且 答案整数精确相等；"
                            f"仅对含 >= {SIG_BIG} 的数的条目生效（小数字多重集撞车率过高）"),
            "A2_exact_question": "sha1(lowercase alnum-folded question) == sha1(row user text); 全语料",
            "B_ngram": f"n=min(13,#question tokens)>=5 词 n-gram 交集；对 user 侧 <= {MAX_NGRAM_ROWS_TOKENS} 词的行",
            "C_blocked_substring": "长行(>3000 词)且含条目 >=3 位数字锚：规范化题干子串包含",
        },
        "corpus_root": a.data_root,
        "stats": stats,
        "per_layer": per_layer,
        "n_items": len(items),
        "n_items_hit": len(hit_ids),
        "hit_rate": len(hit_ids) / len(items),
        "hits_by_check": {k: len(v) for k, v in hits.items()},
        "hit_item_ids": hit_ids,
        "hits_by_family": __import__("collections").Counter(
            next(i["family"] for i in items if i["id"] == x) for x in hit_ids),
        "hits_by_stratum": __import__("collections").Counter(
            next(i["stratum"] for i in items if i["id"] == x) for x in hit_ids),
        "hit_examples": {k: dict(list(v.items())[:10]) for k, v in hits.items()},
        "elapsed_s": round(time.time() - t0, 1),
    }
    with open(a.out, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps({k: report[k] for k in
                      ("stats", "n_items_hit", "hit_rate", "hits_by_check", "elapsed_s")},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
