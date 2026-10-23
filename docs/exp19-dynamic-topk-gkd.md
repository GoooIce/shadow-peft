# exp19 GKD 动态稀疏 teacher support（issue #54）

一句话收束：**4 个训练 seed × mass 网格 {0.5, 0.9, 0.99, 0.999} 的结果是：固定 `k=2` 的 task accuracy 在所有 mass 上都不差于动态 support。** 200 条上，mass 0.5 / 0.9 显著更差（固定 − 动态 **+9.25 / +5.88 pp**，95% CI 分别 [+5.37,+13.25] / [+2.25,+9.63]），0.99 是 +2.13 pp（CI [−1.25,+5.50]），0.999 基本持平（+0.38 pp，CI [−2.38,+3.13]）。1,000 条上 mass 0.99 的 2-seed 平均是 **+2.05 pp**（CI **[+0.1,+4.0]**），与 200 条网格一致。ppl 方面，0.5/0.9 大幅变差（7.30/5.60 vs 4.69），0.99/0.999 更好（4.42/4.50 vs 4.69）。结论：**动态 support 是一个随稀疏度恶化的 ppl/accuracy trade-off；没有哪个 mass 给出 accuracy 增益，阶段 2 松弛不开启。****

状态：**结题（2026-10-23）**。4 seeds × 4 masses × 200 条 + 2 seeds × 1,000 条（mass 0.99）+ 配对 bootstrap；越稀疏（mass 0.5/0.9）accuracy 和 ppl 都显著更差，越接近固定（0.999）越接近持平，最好也只是 neutral。

## 动机

- 现有 GKD 的 `teacher_support=topk` 对所有位置用同一个 `k`。teacher 在不同位置的熵差异很大；按输入分配 support 是一个自然的稀疏化方向。
- MiTA（arXiv 2602.01219）只提供了“动态稀疏 support”的设计类比；它不是 GKD 论文，也没有离散松弛或 LLM 实验。
- 目标：在相同/更低平均 support 下，用更少的 teacher token 保留或提升 KD 信号。

## 实现

- `_shifted_kd_loss` / `onpolicy_gkd_loss` / `OnPolicyGKDTrainer` 新增 `topk_mass`、`topk_min`：
  - `topk` 是最大 k；每个位置取最小的 `k >= topk_min`，使 teacher top-k 概率质量 ≥ `topk_mass`；
  - support 仍是 student top-k 与 teacher top-k 的并集（与固定 top-k 同语义）；
  - `support_stats` 记录 mean/min/max k、达到 max k 的比例、参与 token 数。
- CLI：`--topk-mass`、`--topk-min`；`train_stats.json` 报告 `teacher_support=dynamic_topk` 和 `support_stats`。
- 测试：mass=1.0 与固定 top-k 数值一致；峰值/平坦 teacher 的 k 自适应；非法参数报错；trainer 统计累积。

## 数据与设置

| 项 | 值 |
|---|---|
| 机器 | `goooinext`（训练/固定臂评测）+ `gooo`（动态臂评测），GB10 |
| 学生 | `Qwen3-0.6B` + fresh 1-layer implicit shadow adapter（exp10 smoke 起点） |
| Teacher | `Qwen3-4B`（冻结，同词表） |
| 数据 | GSM8K **train** 1,000 prompts（fingerprint `sha256:d541bc07…`）；heldout 为 GSM8K test 1,000 条 reference |
| Rollout | `base_shadow`，G=4，32 token，temperature 1.0，top-p 1.0，seed 0 → 4,000 行；train split 3,600 行 |
| 训练 | λ=1（纯 on-policy JSD），JSD，1000 steps，batch 4，max_seq_len 256，lr 1e-4；训练 seed 0 / 1，两臂共用同一份 rollouts |
| 固定臂 | `--topk 2` |
| 动态臂 | `--topk 32 --topk-mass 0.99 --topk-min 1` |
| 评测 | exp17 GKD：GSM8K test 1,000 prompts，greedy 256 token，joint ppl + task accuracy |

Pilot（25 steps，batch 4，3,200 loss token）：

| mass | k_mean | k_min | k_max |
|---|---|---|---|
| 0.9 | 1.224 | 1 | 6 |
| **0.99** | **1.523** | 1 | 11 |

teacher 在 GSM8K 上非常 peaked，0.99 质量平均只需 1.5 个 token；正式动态臂在 128k token 上统计到 mean **1.546–1.548**、max **32**（达到 max 的比例约 0.01%）。这是本轮实验的核心限制：动态范围天然很窄。

## 结果

### 1,000 条 test（2 个训练 seed）

| seed | 臂 | joint ppl [95% CI] | task acc [95% CI] | support |
|---|---|---|---|---|
| 0 | 固定 k=2 | **4.744** [4.643, 4.832] | **57.5%** [54.4, 60.5] | k=2 |
| 0 | 动态 mass=0.99 | **4.468** [4.377, 4.547] | **53.9%** [50.7, 56.6] | mean 1.55 / max 32 |
| 1 | 固定 k=2 | **4.686** [4.586, 4.773] | **57.7%** [54.5, 60.3] | k=2 |
| 1 | 动态 mass=0.99 | **4.423** [4.336, 4.502] | **57.2%** [54.4, 60.2] | mean 1.55 / max 32 |

- **ppl：动态在两个 seed 上都显著更低。** 两个 seed 的固定/动态 95% CI 都不重叠（seed 0：4.643 > 4.547；seed 1：4.586 > 4.502）。
- **accuracy：固定平均更高。** 配对 bootstrap（同一批 1,000 条，固定 − 动态）：
  - seed 0：**+3.6 pp**，95% CI **[+1.0, +6.2]**，110 胜 / 74 负；
  - seed 1：**+0.5 pp**，95% CI **[−1.9, +3.0]**，82 胜 / 77 负；
  - 两个 seed 的 per-example 平均：**+2.05 pp**，95% CI **[+0.1, +4.0]**，141 胜 / 107 负。
- 平均 CI 只是勉强排除 0，而且 seed 0 贡献了大部分效应；这是**名义 95%**、未做多重比较校正的结果。更准确的说法是“有可分辨的、约 2 pp 的 accuracy 减益，但证据强度中等”。

### 4-seed mass 网格（200 条）

固定臂始终 `k=2`；每个 mass 4 个训练 seed（0–3），共用同一份 rollouts；配对 bootstrap（固定 − 动态），正数表示固定更准。

| mass | mean k | 固定 − 动态 acc | 95% CI | 固定 ppl | 动态 ppl |
|---|---|---|---|---|---|
| 0.5 | 1.02 | **+9.25 pp** | **[+5.37, +13.25]** | 4.691 | **7.301** |
| 0.9 | 1.24 | **+5.88 pp** | **[+2.25, +9.63]** | 4.691 | **5.603** |
| 0.99 | 1.55 | +2.13 pp | [−1.25, +5.50] | 4.691 | **4.415** |
| 0.999 | 1.94 | +0.38 pp | [−2.38, +3.13] | 4.691 | **4.497** |

- **accuracy 随稀疏度单调恶化。** mass 0.5（平均 1.02 个 token）比固定 k=2 低 **9.25 pp**，0.9 低 **5.88 pp**，两者的 CI 都排除 0；0.99 低 2.13 pp（200 条上不显著，1,000 条上 +2.05 pp CI [+0.1,+4.0]）；0.999 基本持平（+0.38 pp）。
- **ppl 不是单调的。** 过度稀疏（0.5/0.9）把 ppl 从 4.69 抬到 7.30/5.60，属于 accuracy 和 ppl 双输；0.99/0.999 把 ppl 降到 4.42/4.50，但没有转化为 accuracy。
- **最好的 mass 也只是 neutral。** 0.999 的 mean k 1.94 已经接近固定 k=2，既没有 accuracy 增益，也没有明显的 support 节省。
- 结论：**没有任何 mass 给出可分辨的 accuracy 增益；0.5/0.9 是可分辨的减益，0.99 是小幅减益，0.999 是持平。** 动态 support 不是免费增益。

### 200 条筛查 vs 1,000 条（为什么不能停在 200）

| seed | 固定 acc (200) | 动态 acc (200) | 固定 − 动态 |
|---|---|---|---|
| 0 | 56.5% | 49.0% | +7.5 pp（配对 CI [+1.0, +14.0]） |
| 1 | 56.0% | 59.5% | −3.5 pp（配对 CI [−8.5, +1.5]） |

200 条上 seed 0 的 −7.5 pp 看起来“显著”，seed 1 又反向；扩到 1,000 条后 seed 0 缩到 −3.6 pp、seed 1 变成 −0.5 pp。**200 条筛查的结论不可靠，必须用 1,000 条 + 多 seed。**

### baseline（仅 200 条）

fresh adapter、未训练：joint ppl 4.615 [4.418, 4.844]，task acc 52.0% [46.0, 58.5]。它没有 1,000 条数字；表里不参与 1,000 条的主比较。

## 解释

1. **动态 support 是一个 ppl/accuracy trade-off，不是免费增益。** 它把 teacher support 从固定 2 个 token 压到平均 1.55 个，显著改善了 reference 文本上的 joint ppl；但生成任务的 accuracy 平均下降约 2 pp，且 seed 0 的下降更大。ppl 不是部署指标，accuracy 才是。
2. **seed 方差大于平均效应。** 动态臂两个 seed 的 accuracy 是 53.9% / 57.2%，固定是 57.5% / 57.7%。动态 support 让优化轨迹更不稳定；固定 k=2 在两个 seed 上都很稳。
3. **teacher 太 peaked，动态范围太窄。** mass=0.99 平均 k 1.55，固定 k=2 已覆盖几乎相同的 head；动态主要在极少数平坦位置扩到 32。这个设置下“动态”与“固定”的差别本来就被压缩。
4. **MiTA 没有提供直接证据。** 本轮没有 Gumbel/straight-through/soft top-k，也没有 attention 算子；MiTA 论文本身没有 LLM/GKD 实验。

## 结论

- **可分辨减益**：4-seed 网格上，mass 0.5 比固定 k=2 低 **9.25 pp**（CI [+5.37,+13.25]），mass 0.9 低 **5.88 pp**（CI [+2.25,+9.63]）；mass 0.99 在 1,000 条上低 **2.05 pp**（CI [+0.1,+4.0]）。
- **没有可分辨增益**：mass 0.999 只是持平（+0.38 pp，CI [−2.38,+3.13]），而且 mean k 1.94 已接近固定 k=2，没有明显的 support 节省。
- **ppl 不是替代指标**：0.99/0.999 的 joint ppl 显著更低，但没有转化为 accuracy；0.5/0.9 则是 accuracy 和 ppl 双输。
- **阶段 2 不开启**：Gumbel/ST 松弛只有在动态 hard-k 已经出现 accuracy 增益时才值得做；本轮是反向的。当前不建议把 dynamic top-k 设为默认，也不建议先移植 MiTA attention。

如果还要继续，方向不是再加评测条数，而是换一个真正能改变选择的机制（例如可学习 selector 或 token-level 蒸馏），而不是只调 mass。

## 限制

- 4 个训练 seed（0–3）、1,000 steps、32-token rollout；200 条网格的 CI 是名义 95%，未做多重比较校正；1,000 条只做了 mass 0.99。
- 网格包含 mass 0.5/0.9/0.99/0.999；mass 0.999 的 mean k 1.94，接近固定 k=2，因此它更像是“固定 support 的噪声版本”，不是真正的稀疏化。
- 固定臂 k=2 与动态臂的 mean k 只是近似匹配；实际 union support 大小没有逐位置统计。
- 只有 Qwen3-0.6B/4B 一组模型；没有 SFT（λ=0）或 full-vocab JSD 对照。
- 没有端到端生产 QA 数据；GSM8K task accuracy 是 256-token 生成，不是产线指标。

## 命令

```bash
# 数据
PYTHONPATH=src python scripts/exp19_prepare_gkd_data.py \
  --out-dir ~/exp19-gkd/data --train-limit 1000 --heldout-limit 1000 --seed 0

# rollouts（一次生成，两臂/两 seed 复用）
PYTHONPATH=src python scripts/exp10_onpolicy_gkd.py rollouts \
  --prompts ~/exp19-gkd/data/prompts.jsonl --model ~/models/Qwen3-0.6B \
  --shadow-ckpt runs/exp10/smoke/fresh-adapter --out ~/exp19-gkd/rollouts.jsonl \
  --g 4 --max-new-tokens 32 --val-ratio 0.1 --seed 0 --device cuda --dtype bfloat16

# 固定臂 / 动态臂（seed 0；seed 1 只换 --seed）
PYTHONPATH=src python scripts/exp10_onpolicy_gkd.py train \
  --model ~/models/Qwen3-0.6B --shadow-ckpt runs/exp10/smoke/fresh-adapter \
  --teacher ~/models/Qwen3-4B --rollouts ~/exp19-gkd/rollouts.jsonl \
  --lambda 1.0 --divergence jsd --topk 2 --max-steps 1000 --batch-size 4 \
  --max-seq-len 256 --seed 0 --device cuda --dtype bfloat16 \
  --out ~/exp19-gkd/arms/fixed-k2

PYTHONPATH=src python scripts/exp10_onpolicy_gkd.py train \
  --model ~/models/Qwen3-0.6B --shadow-ckpt runs/exp10/smoke/fresh-adapter \
  --teacher ~/models/Qwen3-4B --rollouts ~/exp19-gkd/rollouts.jsonl \
  --lambda 1.0 --divergence jsd --topk 32 --topk-mass 0.99 --topk-min 1 \
  --max-steps 1000 --batch-size 4 --max-seq-len 256 --seed 0 \
  --device cuda --dtype bfloat16 --out ~/exp19-gkd/arms/dynamic-m099-k32

# 1,000 条评测
PYTHONPATH=src python scripts/exp17_real_eval.py --suite gkd --dataset gsm8k \
  --model ~/models/Qwen3-0.6B --shadow-ckpt ~/exp19-gkd/arms/<arm> \
  --limit 1000 --max-new-tokens 256 --device cuda --dtype bfloat16 \
  --out ~/exp19-gkd/arms/<arm>/gkd-1000
```

## 产物

- 代码：`src/shadow_peft/task_models.py`（dynamic union mask/stats）、`src/shadow_lab/trainers.py`（`topk_mass`/`topk_min`、support 统计）、`scripts/exp10_onpolicy_gkd.py`（CLI/summary）、`scripts/exp19_prepare_gkd_data.py`、`scripts/exp19_analyze_grid.py`（配对 bootstrap 分析）
- 测试：`tests/lab/test_onpolicy_trainer.py`（动态 support 用例）
- 本地：`runs-local/exp19/{data,pilot-m0.9,pilot-m0.99,baseline,arms}/`（`arms/*/gkd-200/report.json`、`arms/*/gkd-1000/report.json`、`train_stats.json`、`grid_200.json`）
- 远端：`goooinext:~/exp19-gkd/`、`gooo:~/exp19-gkd/arms/`
- 相关：`docs/exp10-onpolicy-gkd.md`、`docs/specs/gkd-on-policy-flywheel.md`
