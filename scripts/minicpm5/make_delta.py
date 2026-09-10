"""扩容后只对**新增条目**评分：按"题干"把上一版分数映射到新版池，产出待评分的 delta.jsonl。

背景：v3 池是同一种子流下"放大 = 追加"，但**全局去重集合**使先处理的 stratum 计划变化会让
后处理的 stratum 的序号整体平移（实测 1/23,708 条题干在 stratum 间移动，导致 arith/hard 的
id 从某处起全部平移）。因此不能按 id 复用分数，必须按**题干**复用（同一题干 = 同一道题，
模型的得分只取决于题面）。

输出：
  delta.jsonl   —— 新版池里"题干从未被测过"的条目（需要 GPU 评分的部分）
  merge_map.json —— 新版 id -> 旧 id（题干可复用旧分数）；以及旧池里已消失的题干
用法：
  python3 make_delta.py --new-pool pool_v3.jsonl --old-pool pool_v3_prev.jsonl \
      --old-scores scores_q0_pool_v3.jsonl [--old-scores scores_bf16_pool_v3.jsonl] \
      --delta-out delta.jsonl --map-out merge_map.json
"""
from __future__ import annotations

import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new-pool", required=True)
    ap.add_argument("--old-pool", required=True)
    ap.add_argument("--old-scores", action="append", required=True)
    ap.add_argument("--delta-out", required=True)
    ap.add_argument("--map-out", required=True)
    a = ap.parse_args()

    new = [json.loads(l) for l in open(a.new_pool)]
    old = [json.loads(l) for l in open(a.old_pool)]
    old_ids = {it["id"] for it in old}
    old_q_to_id = {it["question"]: it["id"] for it in old}
    scored = set()
    for p in a.old_scores:
        scored |= {json.loads(l)["id"] for l in open(p)}
    scored &= old_ids

    delta, reuse, missing_prev = [], {}, []
    for it in new:
        oid = old_q_to_id.get(it["question"])
        if oid and oid in scored:
            reuse[it["id"]] = oid
        else:
            delta.append(it)
    for it in old:
        if it["id"] in scored:
            pass
    new_q = {it["question"] for it in new}
    for it in old:
        if it["question"] not in new_q:
            missing_prev.append(it["id"])

    with open(a.delta_out, "w") as f:
        for it in delta:
            f.write(json.dumps(it, ensure_ascii=False, sort_keys=True) + "\n")
    json.dump(dict(new_n=len(new), old_n=len(old), scored_old_n=len(scored),
                   delta_n=len(delta), reuse_n=len(reuse),
                   reuse_rate=len(reuse) / len(new),
                   old_items_absent_from_new=missing_prev,
                   id_remap={k: v for k, v in list(reuse.items())[:0]},  # 全量映射另存
                   old_scores_files=a.old_scores,
                   note=("按题干复用旧分数；同一题干在不同 stratum 间移动时，以新池的 stratum 标签为准"
                         "（模型得分只取决于题面）")),
              open(a.map_out, "w"), indent=2, ensure_ascii=False, sort_keys=True)
    with open(a.map_out.replace(".json", "_full.json"), "w") as f:
        json.dump(reuse, f, ensure_ascii=False, sort_keys=True)
    print(json.dumps(dict(new_n=len(new), old_n=len(old), scored_old_n=len(scored),
                          delta_n=len(delta), reuse_n=len(reuse),
                          old_items_absent_from_new=len(missing_prev)), indent=2))


if __name__ == "__main__":
    main()
