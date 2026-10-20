# exp12 RESTRAIN-lite 结构化输出飞轮 smoke（spec #4）

一句话收束：**在 `goooinext` 上用 Qwen3-0.6B + 新建 shadow adapter 跑通了 RESTRAIN-lite 全链路（hard-case → n=4 rollout → 结构化聚类/权重 → `export_self_train` → weighted SFT vs best-answer SFT → held-out 评估），但数据分布不适合：8 条 hard-case 只有 5 条可聚类、低一致性比例 60%（spec 门槛 <30%），weighted 与 baseline 的 held-out 结构化准确率都是 0。按首轮验收口径判 no-go；这是 8 条 prompt/2 步 smoke，不是 RESTRAIN 在大数据上的结论。**

状态：**smoke 已跑通（2026-10-20），结论 no-go（当前数据/规模）**；500 prompt、n=8、更长训练与 κ 敏感性未跑。

## 实验设置

| 项 | 值 |
|---|---|
| 机器 | `goooinext`（aitopatom-dc4a，192.168.1.177，GB10） |
| 代码 | `feat/restrain-lite` @ `36dba55`（#31–#34 合并后） |
| 模型 | `Qwen3-0.6B` + 新建隐式 shadow adapter（`runs/exp10/smoke/fresh-adapter`） |
| 数据 | 手工 8 条 hard-case：4 条 `quality_verdict` + 4 条 `process_calc`（KB 答案为 best_answer） |
| Rollout | n=4，temperature 1.0，max_new_tokens=32，seed 7，`enable_thinking=False` |
| 训练 | weighted：κ=3、σ=0.5；2 步、batch 2、max_seq_len 256、lr=1e-4 |
| 评估 | 同一 8 条 held-out，`--n 1`、temperature 0、max_new_tokens 32；结构化 verdict/number 容差 |

## 命令

```bash
# 1) 导出 self-train
PYTHONPATH=src .venv/bin/python scripts/exp12_restrain.py export \
  --feedback-store runs/exp12/feedback.jsonl \
  --backend-config runs/exp12/backend.json \
  --out runs/exp12/self_train.jsonl --report runs/exp12/export_report.json \
  --n 4 --tasks quality_verdict,process_calc \
  --max-new-tokens 32 --temperature 1.0 --seed 7 --limit 8

# 2) weighted vs best-answer SFT（同模型/seed/步数）
PYTHONPATH=src .venv/bin/python scripts/exp12_restrain.py train \
  --out runs/exp12/train --self-train runs/exp12/self_train.jsonl \
  --sft runs/exp12/baseline.jsonl \
  --model /home/goooice/models/Qwen3-0.6B \
  --shadow-ckpt runs/exp10/smoke/fresh-adapter \
  --max-steps 2 --batch-size 2 --max-seq-len 256 --seed 7 \
  --device cuda --dtype bfloat16

# 3) 两个 adapter 的 held-out 评估
for arm in weighted baseline; do
  PYTHONPATH=src .venv/bin/python scripts/exp12_restrain.py eval \
    --heldout runs/exp12/heldout.jsonl --model /home/goooice/models/Qwen3-0.6B \
    --shadow-ckpt runs/exp12/train/$arm --out runs/exp12/eval_$arm.json \
    --max-new-tokens 32 --n 1 --temperature 0 --seed 7 \
    --self-train runs/exp12/self_train.jsonl --export-report runs/exp12/export_report.json \
    --device cuda --dtype bfloat16
done
```

产物：`runs-local/exp12/{feedback.jsonl,self_train.jsonl,export_report.json,train/,eval_weighted.json,eval_baseline.json}`。

## 结果

### 导出

| 指标 | 值 |
|---|---|
| hard-case records | 8 |
| exported | **5** |
| skipped_unclusterable | **3** |
| skipped_empty / unsupported | 0 / 0 |
| rollout_failures | 0 |
| cluster_methods | verdict_label 3 / number_tolerance 2 |
| **low_consistency_rate** | **0.60**（6/10 可评 prompt，包括被跳过的 unclusterable） |

- 低一致性 60% 远超 spec 的 `<30%` 门槛：同一 prompt 的 4 条答案多数不一致，说明 0.6B 在结构化 verdict/number 上本身不稳定，RESTRAIN 的“票数”信号很弱。
- 3 条 unclusterable 被跳过；如果强行训练，会把噪声当伪标签。

### 训练

| 臂 | 行数 | tokens_seen | loss | 说明 |
|---|---|---|---|---|
| weighted | 2 | 4 | 23.25 | κ=3 下只有 2 条 prompt 达到多数票门槛；样本极少 |
| baseline best-answer | 8 | 85 | 6.47 | 8 条固定答案 |

- weighted 与 baseline 不是同 token 预算（4 vs 85 token），loss 不可直接比较；这是 smoke 的显著限制。
- weighted loss 高主要因为只有 4 个 token 参与，且目标/权重分布极端。

### Held-out 评估

| 臂 | accuracy | accuracy CI | answer_entropy | top1_share | low_consistency | skip_rate | claims_gain |
|---|---|---|---|---|---|---|---|
| weighted | 0.0 | [0, 0] | 0.0 | 1.0 | 0.60 | 0.375 | false |
| baseline | 0.0 | [0, 0] | 0.0 | 1.0 | 0.60 | 0.375 | false |

- 两者结构化准确率都是 0：0.6B 的生成没有稳定给出可评的 verdict/number；`--n 1` 的 entropy 为 0 是定义上的（只有一个样本），不是模型塌缩证据。
- `skip_rate=0.375` 来自 3/8 unclusterable；低一致性 0.60 来自导出报告。
- 升级率/云端调用数未接 edge-cloud，标 `null`；不伪造。

## 首轮验收口径

| 口径 | 结论 | 证据 |
|---|---|---|
| 同 token 预算 weighted SFT 准确率 +1pp | **未通过** | 两者 accuracy 0；且预算未对齐（4 vs 85 token） |
| 答案熵 ≥ baseline 80% | **未验证** | n=1 的 entropy=0；需要 n≥4 |
| 低一致性比例 <30% | **未通过** | 0.60 |
| 升级率/云端调用不上升 | **未验证** | 未接 edge-cloud |
| 权重全 1 退化为现有 SFT | **通过（单测）** | `WeightedSFTTrainer` 测试 |

## 结论

- **当前数据/规模下 no-go**：低一致性 60%、可聚类率 5/8、准确率 0；继续在这个数据上加权训练没有意义。
- 这不是 RESTRAIN 方法的终审：只有 8 条 prompt、n=4、2 步训练、0.6B 学生；spec 建议 500 prompt、n=8、κ=2/3/5 敏感性。要重试，先换更强的结构化模型/教师或更干净的任务分层，把低一致性降到 30% 以下，再谈 weighted SFT。
- 本次交付的价值是**管线与可审计性**：`export_self_train`、聚类/权重、`WeightedSFTTrainer`、评测 CLI 都已跑通；结论是数据不满足前提。
- 下一步优先级：回到 #2 GKD 提升 shadow 表征，或换结构化任务数据；不要用更大 n/κ 硬救。

## 已知限制

- 8 条手工 prompt、n=4、2 步训练；样本量和预算都远低于 spec 的 500 prompt/n=8。
- weighted 与 baseline 的 token 预算未对齐（4 vs 85 token），不能做严格对照。
- 0.6B 的 verdict/number 生成质量差，accuracy 0；不能据此判断 RESTRAIN 在正常模型上的效果。
- 评估 entropy 用 `--n 1`，恒为 0；真实熵应使用 n≥4 的 rollout 分布。
- 升级率/云端调用数未接 edge-cloud；`null` 表示未测。
- **Gaussian 中心**：实现为 `g(f)=exp(-(1-f)²/(2σ²))`（向多数票倾斜），与 spec 字面 `exp(-f²/(2σ²))` 相反；理由是该字面式会给少数票更高权重，与「多数票倾斜、不是赢者通吃」的验收矛盾。已在代码/测试中固定，若上游要求字面式需改口径。
- `WeightedSFTTrainer` 的均匀权重快路径使用全局 token 均值；不等权时才严格按逐样本均值公式。加权步不注入 teacher KD。
