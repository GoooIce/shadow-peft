# exp19 GKD 动态稀疏 teacher support（issue #54）

一句话收束：**在 GSM8K 1k prompts 上，把 GKD 的固定 top-k teacher support 换成按 teacher 概率质量自适应的 per-position k（`mass=0.99`，平均 k **1.55**，max 32），1000 步、λ=1 JSD、与固定 `k=2` 同数据同 token 预算对照：动态臂 joint ppl **4.446**（固定 4.741、baseline 4.615），但 GSM8K task accuracy **49.0%**（固定 56.5%、baseline 52.0%）；三项的 95% CI 全部重叠，没有任何指标出现可分辨增益。结论：**动态 mass-k 在本轮配置下 no-go，不进入阶段 2 的 Gumbel/straight-through 松弛。****

状态：**结题（2026-10-23），no-go**。MiTA（arXiv 2602.01219）只提供了“动态稀疏 support”的设计类比；它不是 GKD 论文，也没有离散松弛或 LLM 实验。本实验只验证了无松弛的 mass-adaptive k。

## 动机

- 现有 GKD 的 `teacher_support=topk` 对所有位置用同一个 `k`。teacher 在不同位置的熵差异很大；按输入分配 support 是一个自然的稀疏化方向。
- MiTA 的 dynamic top-k/routing 是视觉 attention 算子，不能直接搬到 GKD；这里只借“同一平均预算下按位置调整 support 宽度”的原则。
- 目标不是提升 attention 速度，而是用更少的 teacher support 保留/提升 KD 信号。

## 实现

- `_shifted_kd_loss` / `onpolicy_gkd_loss` / `OnPolicyGKDTrainer` 新增 `topk_mass`、`topk_min`：
  - `topk` 是最大 k；每个位置取最小的 `k >= topk_min`，使 teacher top-k 概率质量 ≥ `topk_mass`；
  - support 仍是 student top-k 与 teacher top-k 的并集（与固定 top-k 同语义）；
  - `support_stats` 记录 mean/min/max k、达到 max k 的比例、参与 token 数。
- CLI：`--topk-mass`、`--topk-min`（`--topk` 为 max k，默认 64）。
- 训练记录 `teacher_support=dynamic_topk` 和 `support_stats`；eval report 也接受该口径。
- 测试：mass=1.0 与固定 top-k 数值一致；峰值/平坦 teacher 的 k 自适应；非法参数报错；trainer 统计累积。

## 数据与设置

| 项 | 值 |
|---|---|
| 机器 | `goooinext`（训练）+ `gooo`（动态臂评测），GB10 |
| 学生 | `Qwen3-0.6B` + fresh 1-layer implicit shadow adapter（exp10 smoke 起点） |
| Teacher | `Qwen3-4B`（冻结，同词表） |
| 数据 | GSM8K **train** 1,000 prompts（fingerprint `sha256:d541bc07…`）；heldout 为 GSM8K test 200 条 reference |
| Rollout | `base_shadow`，G=4，32 token，temperature 1.0，top-p 1.0，seed 0 → 4,000 行；train split 3,600 行 |
| 训练 | λ=1（纯 on-policy JSD），JSD，1000 steps，batch 4，max_seq_len 256，lr 1e-4，seed 0 |
| 固定臂 | `--topk 2`（与动态臂平均 k 1.55 最接近的整数预算） |
| 动态臂 | `--topk 32 --topk-mass 0.99 --topk-min 1` |
| 评测 | exp10 heldout ppl（joint/shadow）；exp17 GKD 200（joint ppl + GSM8K task accuracy，256 token） |

Pilot（25 steps，batch 4，3,200 个 loss token）：

| mass | k_mean | k_min | k_max |
|---|---|---|---|
| 0.9 | 1.224 | 1 | 6 |
| **0.99** | **1.523** | 1 | 11 |

teacher 在 GSM8K 上非常 peaked，0.9 质量平均只需 1.2 个 token；0.99 也只要 1.5 个。动态范围天然很窄，这是本轮实验的主要限制。正式动态臂在 128k token 上统计到 mean **1.548**、max **32**（达到 max 的比例 0.0086%）。

## 结果

### exp17 GKD 200（同一批 GSM8K test）

| 臂 | joint ppl [95% CI] | shadow ppl | task acc [95% CI] | support |
|---|---|---|---|---|
| baseline（fresh adapter，未训练） | **4.615** [4.418, 4.844] | 142,421 | **52.0%** [46.0, 58.5] | — |
| 固定 top-k=2 | 4.741 [4.536, 4.987] | 221,429 | **56.5%** [50.5, 63.0] | k=2 |
| 动态 mass=0.99 | **4.446** [4.263, 4.655] | 143,474 | **49.0%** [41.5, 56.5] | mean 1.55 / max 32 |

- **动态 vs 固定**：动态 joint ppl 低 0.295（约 −6.2%），但 task accuracy 低 7.5 pp；两个方向的 CI 都重叠。没有“动态更好”的一致信号。
- **动态 vs baseline**：ppl 低 0.169（约 −3.7%），accuracy 低 3.0 pp；同样不显著。
- **固定 vs baseline**：accuracy 高 4.5 pp，但 ppl 差 0.126；也不能声称收益。
- shadow-only ppl 在 14–22 万量级，是 1 层 adapter 的未训练 standalone 路径，只作诊断，不作为主指标。

### exp10 heldout（200 条 reference 文本）

| 臂 | joint ppl [95% CI] |
|---|---|
| baseline | 7.996 [5.835, 11.109] |
| 固定 k=2 | 7.836 [5.659, 11.079] |
| 动态 mass=0.99 | 7.909 [5.859, 10.914] |

这组 heldout 与 exp17 GKD 的 200 条 test 不同；三臂差异都在 CI 内，和主表的方向也不完全一致（动态在 exp17 ppl 更好，在这里略差于固定）。这正是不能用单一 ppl 选点的原因。

## 解释

1. **没有可分辨收益。** 主指标 task accuracy 上固定臂最高（56.5%），动态臂低于 baseline（49.0%）；ppl 上动态最好。两者方向冲突、CI 重叠，按 ticket 的验收口径只能判 no-go。
2. **teacher 太 peaked，动态范围太窄。** mass=0.99 平均 k 1.55，固定 k=2 已经覆盖了几乎相同的 head；动态主要在极少数平坦位置扩到 32。这个设置下“动态”与“固定”的差别本来就被压缩了。
3. **不是 MiTA 的验证。** 本轮没有 Gumbel/straight-through/soft top-k，也没有 attention 算子；它只验证了 mass-adaptive hard top-k。MiTA 的论文本身也没有 LLM/GKD 实验。
4. **固定臂的 +4.5 pp 只是提示。** 与 baseline 的 accuracy CI 重叠，且 ppl 变差；不能作为 GKD 收益结论。

## 限制

- 1 个训练 seed、1,000 steps、32-token rollout；task accuracy 的 CI 约 ±6–8 pp，无法分辨小于 7 pp 的差异。
- 只测了 mass 0.99（主臂）和 0.9（pilot）；没有 mass 0.5/0.999 的网格。
- 固定臂 k=2 与动态臂平均 k=1.55 只是近似匹配；实际 union support 大小没有逐位置统计。
- 只有 Qwen3-0.6B/4B 一组模型；没有 SFT 对照（λ=0）或 full-vocab JSD 对照。
- 没有端到端生产 QA 数据；GSM8K 的 task accuracy 是 256-token 生成，不是产线指标。

## 后续

- 如果继续这个方向：先做 mass ∈ {0.5, 0.9, 0.99, 0.999} × 2 seeds 的网格，并在固定臂上匹配实际 union support；只有当某个 mass 在 task accuracy 上出现 CI 可分辨的增益，才开阶段 2 的 Gumbel/straight-through 松弛。
- 当前不建议把 dynamic top-k 设为默认，也不建议先移植 MiTA attention 到 shadow adapter。

## 命令

```bash
# 数据
PYTHONPATH=src python scripts/exp19_prepare_gkd_data.py \
  --out-dir ~/exp19-gkd/data --train-limit 1000 --heldout-limit 200 --seed 0

# rollouts（一次生成，两臂复用）
PYTHONPATH=src python scripts/exp10_onpolicy_gkd.py rollouts \
  --prompts ~/exp19-gkd/data/prompts.jsonl --model ~/models/Qwen3-0.6B \
  --shadow-ckpt runs/exp10/smoke/fresh-adapter --out ~/exp19-gkd/rollouts.jsonl \
  --g 4 --max-new-tokens 32 --val-ratio 0.1 --seed 0 --device cuda --dtype bfloat16

# 固定臂
PYTHONPATH=src python scripts/exp10_onpolicy_gkd.py train \
  --model ~/models/Qwen3-0.6B --shadow-ckpt runs/exp10/smoke/fresh-adapter \
  --teacher ~/models/Qwen3-4B --rollouts ~/exp19-gkd/rollouts.jsonl \
  --lambda 1.0 --divergence jsd --topk 2 --max-steps 1000 --batch-size 4 \
  --max-seq-len 256 --seed 0 --device cuda --dtype bfloat16 \
  --out ~/exp19-gkd/arms/fixed-k2

# 动态臂
PYTHONPATH=src python scripts/exp10_onpolicy_gkd.py train \
  --model ~/models/Qwen3-0.6B --shadow-ckpt runs/exp10/smoke/fresh-adapter \
  --teacher ~/models/Qwen3-4B --rollouts ~/exp19-gkd/rollouts.jsonl \
  --lambda 1.0 --divergence jsd --topk 32 --topk-mass 0.99 --topk-min 1 \
  --max-steps 1000 --batch-size 4 --max-seq-len 256 --seed 0 \
  --device cuda --dtype bfloat16 --out ~/exp19-gkd/arms/dynamic-m099-k32

# 评测
PYTHONPATH=src python scripts/exp17_real_eval.py --suite gkd --dataset gsm8k \
  --model ~/models/Qwen3-0.6B --shadow-ckpt ~/exp19-gkd/arms/<arm> \
  --limit 200 --max-new-tokens 256 --device cuda --dtype bfloat16 \
  --out ~/exp19-gkd/arms/<arm>/gkd-200
```

## 产物

- 代码：`src/shadow_peft/task_models.py`（dynamic union mask/stats）、`src/shadow_lab/trainers.py`（`topk_mass`/`topk_min`、support 统计）、`scripts/exp10_onpolicy_gkd.py`（CLI/summary）、`scripts/exp19_prepare_gkd_data.py`
- 测试：`tests/lab/test_onpolicy_trainer.py`（动态 support 用例）
- 本地：`runs-local/exp19/{data,pilot-m0.9,pilot-m0.99,baseline,fixed-k2,dynamic-m099-k32}/`
- 远端：`goooinext:~/exp19-gkd/`（数据、rollouts、固定臂、baseline）、`gooo:~/exp19-gkd/arms/dynamic-m099-k32/`
- 相关：`docs/exp10-onpolicy-gkd.md`、`docs/specs/gkd-on-policy-flywheel.md`
