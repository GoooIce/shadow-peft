# exp11 S2T-Local 影子选择器诊断（spec #3）

一句话收束：**在 `goooinext` 上用 exp7 的真实 `Qwen3-4B + H4B` ShadowPEFT checkpoint 做 S2T-Local 诊断：56 条 prompt × 32 token = 1792 token 上，shadow top-8 对 base top-1 的 Hit@8 = 0.898（95% CI [0.884, 0.912]），Agree@1 = 0.721（[0.700, 0.743]）→ 达到 go 门槛（≥0.85），进入 selector 训练。这是 base-prefix 上的诊断；部署 prefix 上 selector 评估的 Hit@8 只有 0.72–0.78（见 `docs/exp11-s2t-selector.md`），覆盖与选择都有失败。**

状态：**诊断已结题（2026-10-20），base-prefix 结论 go**；K=16 / 全量 rank 的修复重跑见下方「待重跑」占位，selector 训练/评估结论见 `docs/exp11-s2t-selector.md`（no-go）。

## 实验设置

| 项 | 值 |
|---|---|
| 机器 | `goooinext`（aitopatom-dc4a，192.168.1.177，GB10） |
| 接入 | `ssh -J gooo goooinext` |
| 代码 | `feat/s2t-local`（#30 修复：`--k` 默认 16、rank_all 用逐 token 精确 rank） |
| 主干 | `~/models/Qwen3-4B`（冻结） |
| 影子 | exp7 `runs/exp7/e4b-seed0/ckpt-final` + `~/models/Qwen3-0.6B-H4B` |
| 诊断 | `base_shadow` greedy，逐 token 同时取 `logits` 与 `shadow_logits`；K=16；max_new_tokens=32；`--use-cache` |
| prompt | 7 个 task × 8 条 = 56 条（KB 产线问题模板） |
| 样本量 | 1792 token（每 task 256） |
| bootstrap | 1000 次，seed 0 |
| 耗时 | 2m56s（旧 K=8 运行，含 4B+0.6B 加载）；K=16 重跑耗时待补 |

## 命令

```bash
PYTHONPATH=src .venv/bin/python scripts/exp11_s2t_diagnose.py \
  --model /home/goooice/models/Qwen3-4B \
  --shadow-ckpt /home/goooice/Code/shadow-peft/runs/exp7/e4b-seed0/ckpt-final \
  --shadow-model-path /home/goooice/models/Qwen3-0.6B-H4B \
  --prompts runs/exp11/prompts_large.jsonl \
  --max-new-tokens 32 --k 16 --seed 0 \
  --out runs/exp11/diagnostic_large_k16 --limit 56 \
  --device cuda --dtype bfloat16 --use-cache
```

产物：`runs-local/exp11/diagnostic_large_k16/{s2t_diagnostic.json,s2t_tokens.jsonl,s2t_diagnostic.csv}`。旧 K=8 档案保留在 `runs-local/exp11/diagnostic_large/`，不覆盖。

## 结果（1792 token）

| 指标 | 值 | 95% CI |
|---|---|---|
| Hit@1 | 0.721 | — |
| Hit@4 | 0.858 | — |
| **Hit@8** | **0.898** | [0.884, 0.912] |
| **Hit@16** | **待重跑**（K=16） | — |
| **Agree@1** | **0.721** | [0.700, 0.743] |
| rank_all p50 / p90 / p95 / p99 | 待重跑（全量精确 `base_top1_rank_in_shadow`） | — |
| rank_hit p50 / p90 / p95 / p99（旧命中位置口径） | 1 / 4.1 / 5.0 / 6.0 | — |

- Hit@1/4/8 预期与旧 K=8 运行一致（扩大捕获宽度只增加尾部候选，重跑后核对）；Hit@16 必须用 K=16 捕获重跑，旧值不可用。
- `rank_all_*` 用每条 token 的精确 `base_top1_rank_in_shadow`（JSONL 已记录），包括 top-16 之外的 rank；`rank_hit_*` 是旧的「只对命中位置已知」口径，报告里 `rank_p50` 等旧键是它的别名。
- K=16 重跑后，`hit_at_16` 与 `rank_all_*` 一起由协调者填入；本文件不预填估计值。

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

按 position bucket（主结果 1792 token；Hit@8 为 K=16 重跑待填，n 由 56 prompt × 32 token 推出）：

| bucket | n | Hit@8 |
|---|---|---|
| 0 | 56 | 待重跑 |
| 1-3 | 168 | 待重跑 |
| 4-7 | 224 | 待重跑 |
| 8-15 | 448 | 待重跑 |
| 16+ | 896 | 待重跑 |
| 合计 | 1792 | 0.898（旧 K=8 运行） |

旧档案里 1–3、4–7、8–15 的约 0.79 / 0.78 / 0.875 与「position 0 没有可评分 token」的说法不再采用：position 0 每 prompt 都有记录，且旧值没有按主结果的完整 bucket 统计。

## Go / no-go

- 门槛：Hit@8 ≥ 0.85 → go；0.70–0.85 → pilot；< 0.70 → no-go。
- 本次 **Hit@8=0.898，CI 下界 0.884 > 0.85 → go（base-prefix）**。
- 先在 8 条 prompt（128 token）上跑过一次：Hit@8=0.781、Agree@1=0.477、CI [0.703, 0.852]，落在 pilot 区间；样本太小，扩到 56 条后结论稳定为 go。小样本结果保留在 `runs-local/exp11/diagnostic/`，不作为决策依据。
- Agree@1 0.721 说明还有约 28% 的位置 shadow top-1 与 base 不同。在 base-prefix 上多数 base top-1 已在 shadow top-8 内，但部署 prefix 上的 eval Hit@8 只有 0.72–0.78，覆盖并没有在部署路径上保持。

## 重要说明与限制

- **shadow_logits 来自 initial shadow hidden**，不是注入后的 `base_shadow` hidden；诊断衡量的是 shadow-only 路径的候选覆盖，不是 joint 部署路径。报告字段 `shadow_hidden="initial"` 已标注。
- **诊断的 prefix 是 base-greedy prefix**；selector 评估在 shadow-only 自己生成的部署 prefix 上测，Hit@8 只有 0.72–0.78（greedy 0.719、selector 0.763/0.777）。两个数值口径不同，不能把 0.898 当作部署覆盖率。
- prompt 是 KB 模板扩写的 56 条，不是甲方真实流量；task 分布均匀（每 task 256 token），不代表产线频率。
- 只测了一个 checkpoint（exp7 e4b）和一组解码设置（greedy、32 token）；换 checkpoint/温度/长度可能变化。
- rank 分位有两个口径：`rank_all_*` 用逐 token 精确 rank（含 top-16 之外），`rank_hit_*` 只覆盖捕获列表内的命中位置；旧 K=8 档案里只有后者。
- 没有人工标注；Agree@1 只衡量与 base 的一致性，不衡量答案正确性。

## 结论与下一步

- **base-prefix 上 go 仍成立**：Hit@8 0.898 足够支持 selector 训练。
- **部署路径上覆盖与选择都有失败**：部署 prefix 上 Hit@8 0.72–0.78，selector 又把 Agree@1 从同 prefix greedy 的 0.473/0.513 降到 0.353（`selector_gain` −0.121/−0.161）。完整结论见 `docs/exp11-s2t-selector.md`：selector no-go，不部署、不扩 K。
- 后续资源回到 **#2 GKD/表征改进**；若再开 S2T，应先扩训练轨迹并复核 initial shadow hidden 与部署 hidden 的对齐。
