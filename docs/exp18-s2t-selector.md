# exp18 S2T selector on the recovered standalone shadow（issue #47）

一句话收束：**在 exp16 恢复出的独立 shadow 上重训 S2T selector（GSM8K train 100 prompts / 1,600 token，5120→64→1，327,809 参数），3 个训练 seed 在 200 条 GSM8K test prompt / 3,200 token 上全部为负：selector Agree@1 0.543 / 0.525 / 0.521，对比 shadow-greedy 0.562，gain −0.0188 / −0.0366 / −0.0409（均值 −0.0321）。训练集 +0.151 是过拟合；S2T 选择瓶颈仍未解决，selector 不部署。**

状态：**结题（2026-10-22），no-go**；exp17 的 S2T 主结果仍是 Hit@8 0.923 / Agree@1 0.562，`selector_gain=null` 是未传 selector 时的默认值。

## 动机

- exp11 的 selector 训练在 exp7 adapter 的**注入前 shadow hidden** 上，held-out 已判 no-go（gain −0.116 到 −0.161）。
- exp16 恢复出的 standalone shadow 经过 2k GSM8K 指令 SFT，hidden 分布已经不同；直接复用 exp11 selector 不合理，需要在恢复后的模型上重新采集/训练。
- exp17 的 standalone S2T 只测了覆盖和 greedy Agree@1；本实验补上 selector 与三个基线（greedy、random、oracle/coverage）的对照。

## 设置

| 项 | 值 |
|---|---|
| 机器 | `goooinext`（dc4a，GB10），CUDA bf16 |
| base | `Qwen3-4B` |
| shadow | exp16 `all` 臂导出：`~/exp16-full/runs/all/model`（28 层、shadow hidden 1024、projection 到 2560、冻结 4B head） |
| 训练数据 | GSM8K **train** split，100 prompts × 16 token = **1,600** token 样本（1,480 hit）；fingerprint `sha256:68b2ae0e2ac9222532f69c49ec1b45442d66f5590a40ef78dbe86d3400867c5d` |
| 验证 | 按 prompt 切 80/20：1,280 / 320 token，80 / 20 prompts |
| 评测 | GSM8K **test** split，200 prompts × 16 token = **3,200** token；与训练 prompt 无重叠 |
| Selector | `MLP([h_t; e_c])`：5120→64→1，**327,809** 参数；K=8；β=0.1；Adam lr 5e-4；50 epoch；batch 256 |
| hidden | 导出模型的 **post-projection hidden**（`shadow_hidden_projection` 之后、`lm_head` 之前），宽度 2560；不是 1024 的 backbone hidden |
| seed | 0 / 1 / 2（模型初始化、数据顺序、train/val 切分都由 seed 决定） |

采集轨迹与诊断完全一致：每个位置用 base top-1 前进，shadow 只提供 top-K 候选和 hidden；selector 推理时只看到 shadow hidden + 候选 embedding，不运行 base。

## 命令

```bash
# 采集 + 训练 + 保存 selector（seed 0）
PYTHONPATH=src .venv/bin/python scripts/exp18_s2t_selector.py \
  --model ~/models/Qwen3-4B \
  --shadow-model-dir ~/exp16-full/runs/all/model \
  --train-limit 100 --max-new-tokens 16 --epochs 50 \
  --seed 0 --device cuda --dtype bfloat16 \
  --out ~/exp18-selector/all

# 200 条 held-out 评测（seed 0；seed 1/2 只换 --selector 目录）
PYTHONPATH=src .venv/bin/python scripts/exp17_real_eval.py \
  --suite s2t --dataset gsm8k \
  --model ~/models/Qwen3-4B \
  --shadow-model-dir ~/exp16-full/runs/all/model \
  --selector ~/exp18-selector/all/selector \
  --limit 200 --max-new-tokens 16 --k 8 \
  --device cuda --dtype bfloat16 \
  --out ~/exp18-selector/eval-200-all
```

`exp18_s2t_selector.py` 的产物：`train_report.json`、`train_examples.pt`、`train_prompts.jsonl`、`selector/`（`selector.safetensors` + `selector_config.json`）。seed 1/2 的 selector 用同一份 `train_examples.pt` 以 `train_selector(seed=1/2)` 重训后保存为 `selector-seed1/`、`selector-seed2/`。

## 结果

### 训练与验证

| 数据 | selector Agree@1 | greedy Agree@1 | gain |
|---|---|---|---|
| 训练集（in-sample，1,600 token） | **0.7038** | 0.5531 | **+0.1506** |
| 验证 split（320 token，seed 0） | 0.5188 | 0.5469 | **−0.0281** |

训练 loss 从 2.287 降到 0.857（mean 1.497），val loss 1.435；train 1,280 / val 320 token（80/20 prompts）。in-sample 的正增益没有泛化到验证 split。

### 200 条 held-out（3,200 token）

| selector seed | selector Agree@1 | greedy Agree@1 | gain | 说明 |
|---|---|---|---|---|
| 0 | **0.5428** | 0.5616 | **−0.0188** | 50 epoch，lr 5e-4 |
| 1 | **0.5250** | 0.5616 | **−0.0366** | 同配置，不同 seed |
| 2 | **0.5206** | 0.5616 | **−0.0409** | 同配置，不同 seed |
| 均值 ± 样本 std | 0.5295 ± 0.0119 | 0.5616 | **−0.0321 ± 0.0118** | 3 seeds |

三个 seed 全部为负，且 greedy 在同一批 token 上是 0.5616。参考基线：

| 基线 | Agree@1 | 含义 |
|---|---|---|
| shadow-greedy | **0.5616** | 当前部署形态的 top-1 |
| random（候选内均匀） | **0.1154** | 机会水平（hit/K 加权） |
| oracle / coverage | **0.9231** | base top-1 在 shadow top-8 内的比例（Hit@8） |

selector 高于机会水平，但低于 greedy；它把“覆盖到的位置”选对的概率反而拉低了。greedy 已经用掉覆盖率的 60.8%（0.5616 / 0.9231）。

### epoch 观察

在同一份 1,600 样本上做 epoch sweep（3 个 seed，20-prompt 验证 split），50 epoch 的 val gain 均值为 +0.088（seed 0 −0.028、seed 1 +0.166、seed 2 +0.125），但同一批 selector 在 200 条 held-out 上是 −0.019 / −0.037 / −0.041。**20-prompt 验证 split 不能预测 200-prompt held-out**，所以不能用它做模型选择；本实验固定采用 50 epoch，而不是按 val 挑一个“看起来更好”的 epoch。

## 解释

1. **过拟合 100 prompts。** 327k 参数的 selector 在 1,280 个训练 token 上把 in-sample Agree@1 从 0.553 拉到 0.704，但 held-out 反而下降。exp11 在 adapter hidden 上也是同样的模式（in-sample/val 与 held-out 脱节）。
2. **选择瓶颈不是靠一个小 MLP 能解决的。** Hit@8 稳定在 0.923，说明候选覆盖足够；Agree@1 只有 0.56，selector 没有学会在候选内挑 base 偏好的 token。更大的训练集、不同特征（例如候选 logprob、base-prefix 统计）可能有用，但那是新实验，不应从本轮数据外推。
3. **不应部署。** 3/3 seed 为负，均值 −0.032，且没有 seed 接近 0；S2T 主结果继续按 `selector_gain=null` 的 greedy 口径报告。

## 限制

- 训练集只有 100 条 GSM8K train prompt（1,600 token）；selector 容量 327k，样本量明显不足。不能据此断言“任何 selector 都失败”，只能断言本轮配置在 held-out 上失败。
- 评测是 GSM8K test prompt 上的 next-token Agree@1，不是生产 QA 的端到端正确率；selector 对生成质量的影响未测。
- selector 只能重排 shadow top-8；Hit@8 0.923 是硬上限，且 16 token 的轨迹很短。
- 3 个 seed 只覆盖训练随机性；base/shadow 模型本身没有换 seed。
- `claims_gain=false`：这是单一模型路径上的机制评测，不是 baseline-vs-treatment 的生产收益 campaign。

## 产物

- 模块：`src/shadow_lab/s2t_selector.py`（`selector_agreement`）、`src/shadow_lab/real_eval.py`（standalone selector 路径）
- 脚本：`scripts/exp18_s2t_selector.py`、`scripts/exp17_real_eval.py`
- 本地：`runs-local/exp18/selector/`（`train_report.json`、3 个 selector 目录）、`runs-local/exp18/eval-200-seed{0,1,2}/`
- 远端：`~/exp18-selector/`（goooinext）
