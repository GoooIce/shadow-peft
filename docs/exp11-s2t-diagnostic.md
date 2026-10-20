# exp11 S2T-Local 影子选择器诊断（spec #3）

一句话收束：**在 `goooinext` 上用 exp7 的真实 `Qwen3-4B + H4B` ShadowPEFT checkpoint 做 S2T-Local 诊断：56 条 prompt × 32 token = 1792 token 上，shadow top-8 对 base top-1 的 Hit@8 = 0.898（95% CI [0.884, 0.912]），Agree@1 = 0.721（[0.700, 0.743]）→ 达到 go 门槛（≥0.85），进入 selector 训练。**

状态：**诊断已结题（2026-10-20），结论 go**；selector 训练/评估见后续 ticket #27–#29。

## 实验设置

| 项 | 值 |
|---|---|
| 机器 | `goooinext`（aitopatom-dc4a，192.168.1.177，GB10） |
| 接入 | `ssh -J gooo goooinext` |
| 代码 | `feat/s2t-local` @ `63067f7`（#24/#25 合并后） |
| 主干 | `~/models/Qwen3-4B`（冻结） |
| 影子 | exp7 `runs/exp7/e4b-seed0/ckpt-final` + `~/models/Qwen3-0.6B-H4B` |
| 诊断 | `base_shadow` greedy，逐 token 同时取 `logits` 与 `shadow_logits`；K=8；max_new_tokens=32；`--use-cache` |
| prompt | 7 个 task × 8 条 = 56 条（KB 产线问题模板） |
| 样本量 | 1792 token（每 task 256） |
| bootstrap | 1000 次，seed 0 |
| 耗时 | 2m56s（含 4B+0.6B 加载） |

## 命令

```bash
PYTHONPATH=src .venv/bin/python scripts/exp11_s2t_diagnose.py \
  --model /home/goooice/models/Qwen3-4B \
  --shadow-ckpt /home/goooice/Code/shadow-peft/runs/exp7/e4b-seed0/ckpt-final \
  --shadow-model-path /home/goooice/models/Qwen3-0.6B-H4B \
  --prompts runs/exp11/prompts_large.jsonl \
  --max-new-tokens 32 --k 8 --seed 0 \
  --out runs/exp11/diagnostic_large --limit 56 \
  --device cuda --dtype bfloat16 --use-cache
```

产物：`runs-local/exp11/diagnostic_large/{s2t_diagnostic.json,s2t_tokens.jsonl,s2t_diagnostic.csv}`。

## 结果（1792 token）

| 指标 | 值 | 95% CI |
|---|---|---|
| Hit@1 | 0.721 | — |
| Hit@4 | 0.858 | — |
| **Hit@8** | **0.898** | [0.884, 0.912] |
| **Agree@1** | **0.721** | [0.700, 0.743] |
| base top-1 rank p50 / p90 / p95 / p99 | 1 / 4.1 / 5.0 / 6.0 | — |

按 task：

| task | n | Hit@8 | Agree@1 |
|---|---|---|---|
| quality_verdict | 256 | 0.934 | 0.797 |
| fault_diagnosis | 256 | 0.922 | 0.754 |
| standard_qa | 256 | 0.914 | 0.773 |
| safety | 256 | 0.895 | 0.664 |
| process_calc | 256 | 0.879 | 0.746 |
| improvement | 256 | 0.875 | 0.676 |
| process_qa | 256 | 0.871 | 0.637 |

按 position bucket：0 token 位置没有可评分 token（所有 prompt 的第 0 步只作为上下文）；1–3、4–7、8–15 的 Hit@8 分别约 0.79 / 0.78 / 0.875（小样本、仅 8–16 个 prompt 贡献），细节见 JSON。

## Go / no-go

- 门槛：Hit@8 ≥ 0.85 → go；0.70–0.85 → pilot；< 0.70 → no-go。
- 本次 **Hit@8=0.898，CI 下界 0.884 > 0.85 → go**。
- 先在 8 条 prompt（128 token）上跑过一次：Hit@8=0.781、Agree@1=0.477、CI [0.703, 0.852]，落在 pilot 区间；样本太小，扩到 56 条后结论稳定为 go。小样本结果保留在 `runs-local/exp11/diagnostic/`，不作为决策依据。
- Agree@1 0.721 说明还有约 28% 的位置 shadow top-1 与 base 不同，但绝大多数 base top-1 已在 shadow top-8 内——这正是 selector 可以尝试的空间。

## 重要说明与限制

- **shadow_logits 来自 initial shadow hidden**，不是注入后的 `base_shadow` hidden；诊断衡量的是 shadow-only 路径的候选覆盖，不是 joint 部署路径。报告字段 `shadow_hidden="initial"` 已标注。
- prompt 是 KB 模板扩写的 56 条，不是甲方真实流量；task 分布均匀（每 task 256 token），不代表产线频率。
- 只测了一个 checkpoint（exp7 e4b）和一组解码设置（greedy、32 token）；换 checkpoint/温度/长度可能变化。
- rank 分位只对 top-K 命中位置已知；每 token 的精确 rank 在 `s2t_tokens.jsonl`。
- 没有人工标注；Agree@1 只衡量与 base 的一致性，不衡量答案正确性。

## 结论与下一步

**go**：Hit@8 0.898 足够支持 selector 训练。下一步按 #27–#29：
1. selector head（K=8，`MLP([h_t; e_c])`，`L_sel + 0.1 L_reg`）训练；
2. shadow-only + selector 推理/评估，断言不调用 base；
3. 与 shadow-only greedy、`base_shadow` 上界、exp8 ensemble β=0.7 对比，报告准确率/Agree@1/延迟与 δ_hit/δ_sel。

如果 selector 不能把 Agree@1 拉到 ≥50% 或准确率 +1pp，则按 spec 判 no-go，不部署。
