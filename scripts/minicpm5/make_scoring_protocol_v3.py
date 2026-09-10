"""生成 v3 评分口径 JSON（在 17 票冻结副本上只加"BF16 参照臂"登记，其它字段逐字不变）。"""
import json
import os

SRC = "/Users/devel0per/Code/shadow-peft/docs/experiments/artifacts/minicpm5-confirm-power-20260910/scoring_protocol.json"
OUT = "docs/experiments/artifacts/minicpm5-pools-v3-20260910/scoring_protocol.json"

p = json.load(open(SRC))
p["schema"] = "minicpm5-taskset-scoring-protocol-v3"
p["registered_by"] = ("16 (2026-09-10)，17 (2026-09-10) 原样重跑，v3 轮 (2026-09-10) 再次原样冻结")
p["note"] = (
    "v3 轮未改动任何评分口径（prompt 渲染 / 双臂定义 / 引擎参数 / 统计口径逐字不变），"
    "只把池换成新种子生成的 v3 条目，并首次把 BF16 参照臂登记为池组成的一部分："
    "v3 生成池（23,708 题）同时用 q0（NVFP4 W4A4）与 BF16（~/models/MiniCPM5-2B）评分，"
    "gap_bf16_minus_q0 = acc_bf16 - acc_q0 即 4-bit 量化的可回收余量。"
)
p["arms"]["bf16_reference"] = {
    "role": "参照臂（recoverable headroom，不是候选）",
    "model": "/home/goooice/models/MiniCPM5-2B（BF16，权重 sha256 14fb8e7f0a18d53d…）",
    "invocation": "--model ~/models/MiniCPM5-2B --quantization= （显式关闭 compressed-tensors）",
    "mechanism": "与 tf/gen 两臂完全相同（同一 prompt 渲染、同一 TF SamplingParams、同一停止条件、同一 prefix-cache 关闭口径）",
    "note": "v2 池从未测过 BF16（唯一 BF16 数字是 0.6000，测在 v1 的非平衡 dev 池上）→ v2 无法度量量化代价；v3 把 BF16 与 q0 在同一份题面上配对测量。",
}
p["pools_protocol"] = {
    "dev": "每候选适应度（50 候选/代都用它）；480 题 / 60 episode",
    "confirm": "配对确认；用后退出（每代只对当代优胜者开一次）",
    "final": "终检，整轮只开一次",
    "v3_note": (
        "v3 的 q0 与 BF16 都在生成池（23,708 题，三池的父集）上评分——这是冻结前的难度标定与组成"
        "度量，与 16/17 票同口径；三池的逐 stratum / 逐 family 数字由该评分**取子集**得到，"
        "不构成对冻结池的额外开箱。搜索期不得再对 confirm/final 评分。"
    ),
}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(p, open(OUT, "w"), indent=2, ensure_ascii=False, sort_keys=True)
print("written", OUT)
