"""v4 评分口径文件：**逐字复制** v3 的 ``scoring_protocol.json``，只追加 v4 的池登记块。

纪律：``arms`` / ``engine`` / ``model_prompt`` / ``statistics`` 四个块必须与 v3 **逐字节相同**
（本脚本写文件前用 json 深比较断言），因为下一轮的评分协议不得被本票改动。只新增：

  * ``schema`` → ``minicpm5-taskset-scoring-protocol-v4``；
  * ``pools_protocol.confirm_v4`` / ``pools_protocol.centre_v4``：两个新池的用途与开箱纪律；
  * ``v4_note``：v4 两池的 q0/bf16 组成来自 v3 在**生成池**上的既有全量评分（取子集），
    不构成对冻结池的额外开箱；冻结后任何新评分只作为**封条复核**（seal check）登记。

用法：python3 make_scoring_protocol_v4.py --v3-protocol <v3 scoring_protocol.json> --out <v4 文件>
"""
from __future__ import annotations

import argparse
import copy
import json

FROZEN_BLOCKS = ("arms", "engine", "model_prompt", "statistics")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v3-protocol", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    v3 = json.load(open(a.v3_protocol))
    v4 = copy.deepcopy(v3)
    for b in FROZEN_BLOCKS:
        assert json.dumps(v3[b], sort_keys=True, ensure_ascii=False) == \
               json.dumps(v4[b], sort_keys=True, ensure_ascii=False), f"block {b} changed"
    v4["schema"] = "minicpm5-taskset-scoring-protocol-v4"
    v4["registered_by"] = v3["registered_by"] + "；pool4 轮 (2026-09-10) 原样复制并追加 v4 池登记"
    v4["pools_protocol"]["confirm_v4"] = (
        "**下一轮主判定池**（v4 confirm）：8,344 题 / 1,043 episode，15 stratum 每层 q0 精确 0.5000、"
        "每 episode 恰 4 对 4 错；**零开箱**继承自 v3 未消费余量。每代只对当代优胜者开一次；"
        "开箱后不得再用于选择。")
    v4["pools_protocol"]["centre_v4"] = (
        "**中心反馈切片**（v4 centre）：2,040 题 / 255 episode，同样的 15 stratum 平衡设计；"
        "**每代只对当代中心评分一次**，作为监测/自适应反馈通道，不参与主判定、不用于选择。"
        "它的逐代统计量是中心在该切片上的 episode 级准确率（每代标准误见 pool_manifest_v4.json "
        "resolution.centre_v4.se_per_generation）。")
    v4["pools_protocol"]["v4_note"] = (
        "v4 两池的 q0 与 bf16 逐题结果沿用 v3 在**生成池**（85,815 题，三池的父集）上的既有全量评分"
        "（按 id 取子集）——与 16/17/v3 同口径，属冻结前的难度标定与组成度量，**不是**对冻结池的"
        "开箱。冻结后的新评分一律登记为 seal check（对冻结字节复算 q0 准确率），不得用于挑选条目。")
    v4["v4_pools"] = {
        "confirm_v4": {"file": "taskset_confirm_v4.jsonl", "n_items": 8344, "n_episodes": 1043,
                       "delta_min_80pct_sigma_0_1847": 0.016024},
        "centre_v4": {"file": "taskset_centre_v4.jsonl", "n_items": 2040, "n_episodes": 255,
                      "se_per_generation_sigma_0_1677": 0.010502},
    }
    with open(a.out, "w") as f:
        json.dump(v4, f, indent=2, ensure_ascii=False, sort_keys=True)
    print(f"wrote {a.out}: frozen blocks {FROZEN_BLOCKS} byte-identical to v3")


if __name__ == "__main__":
    main()
