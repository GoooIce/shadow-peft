# exp18 S2T selector on the recovered standalone shadow（issue #47）

一句话收束：**在 exp16 恢复出的独立 shadow 上重训 S2T selector（5120→64→1，327,809 参数），训练数据规模决定成败：GSM8K train 100 prompts（1,600 token）在 200 条 test 上 gain **−0.0321**；扩到 **1,000 prompts（16,000 token）** 后 3 个 seed 在 1,000 条 test 上 gain **+0.0860 ± 0.0021**；扩到 **2,000 prompts（32,000 token）** 后 gain **+0.1027**（seed 0）。同一套 selector 在 exp7 的 **2,000 条异质 QA 语料**上 gain **−0.2343 ± 0.0041**（500 条 held-out），且在 GSM8K 上也不迁移（−0.2294）。结论：**同质 in-domain 数据 ≥1k prompts 可用；单纯换更大但异质的语料不行；产线 QA 仍缺同分布数据。****

状态：**结题（2026-10-23）**。旧结论“100-prompt selector no-go”仍然成立，但不再是 selector 路线的终点；1k/2k GSM8K 上已翻正，2k 异质语料仍 no-go。`claims_gain=false` 不变（没有端到端 task accuracy campaign）。

## 动机

- exp11 的 selector 训练在 exp7 adapter 的**注入前 shadow hidden** 上，held-out 已判 no-go（gain −0.116 到 −0.161）。
- exp16 恢复出的 standalone shadow 经过 2k GSM8K 指令 SFT，hidden 分布已经不同；直接复用 exp11 selector 不合理，需要在恢复后的模型上重新采集/训练。
- 第一轮只用 GSM8K train 100 prompts（1,600 token），训练集 +0.151 但 held-out 3/3 为负；怀疑是训练量不足。本轮把训练集扩到 1k/2k，并换到 2k 条异质 QA 语料做对照。

## 设置

| 项 | 值 |
|---|---|
| 机器 | `goooinext`（dc4a，GB10）/ `gooo`（GB10），CUDA bf16 |
| base | `Qwen3-4B` |
| shadow | exp16 `all` 臂导出：`~/exp16-full/runs/all/model`（28 层、shadow hidden 1024、projection 到 2560、冻结 4B head） |
| Selector | `MLP([h_t; e_c])`：5120→64→1，**327,809** 参数；K=8；β=0.1；Adam lr 5e-4；50 epoch；batch 256 |
| hidden | 导出模型的 **post-projection hidden**（`shadow_hidden_projection` 之后、`lm_head` 之前），宽度 2560 |
| 训练/验证 | 按 prompt 切 80/20；每个训练 seed 的初始化、shuffle、train/val 切分都由 seed 决定 |
| 评测 | GSМ8K **test** 或 exp7-corpus **held-out 文档**；与训练 prompt 无重叠；greedy base-prefix，16 token |

采集轨迹与诊断完全一致：每个位置用 base top-1 前进，shadow 只提供 top-K 候选和 hidden；selector 推理时只看到 shadow hidden + 候选 embedding，不运行 base。

### 数据规模

| 实验 | 数据源 | 训练 prompts | 训练 token | 验证 prompts | held-out | held-out token |
|---|---|---|---|---|---|---|
| E1（旧） | GSM8K train | 100 | 1,600 | 20 | GSM8K test 200 | 3,200 |
| E2 | GSM8K train | **1,000** | **16,000** | 200 | GSM8K test 1,000 | 16,000 |
| E2b | 同上 | 1,000 | 16,000 | 200 | GSM8K test 200（E1 同一批） | 3,200 |
| E3 | GSM8K train | **2,000** | **32,000** | 400 | GSM8K test 1,000 | 16,000 |
| E4 | exp7-corpus `en_qa,zh_qa` | **2,000** | **32,000** | 400 | 同语料 held-out 500 | 8,000 |

E4 的 prompt 是每篇文档截断到 **64 token** 的前缀；GSM8K 的 prompt 是完整题目 + `Answer:`。

## 结果

### E1：100 prompts（旧结论，no-go）

GSM8K train 100 prompts × 16 token = 1,600 token；3 个训练 seed 在 200 条 test / 3,200 token 上：

| selector seed | selector Agree@1 | greedy Agree@1 | gain |
|---|---|---|---|
| 0 | 0.5428 | 0.5616 | −0.0188 |
| 1 | 0.5250 | 0.5616 | −0.0366 |
| 2 | 0.5206 | 0.5616 | −0.0409 |
| 均值 ± std | 0.5295 ± 0.0119 | 0.5616 | **−0.0321 ± 0.0118** |

训练集 in-sample +0.151，验证 split −0.028；典型过拟合。details 保留在下方“E1 归档”小节。

### E2：1,000 prompts（in-domain 翻正）

GSM8K train 1,000 prompts × 16 token = 16,000 token（14,731 hit）；3 个 seed 各训一个 selector，在 1,000 条 test / 16,000 token 上：

| selector seed | selector Agree@1 | greedy Agree@1 | gain |
|---|---|---|---|
| 0 | 0.6485 | 0.5645 | +0.0840 |
| 1 | 0.6505 | 0.5645 | +0.0860 |
| 2 | 0.6526 | 0.5645 | +0.0881 |
| 均值 ± std | 0.6505 ± 0.0020 | 0.5645 | **+0.0860 ± 0.0021** |

- **同一批 200 条 test prompt（E1 用的那批）**，seed 0：selector **0.6566** vs greedy **0.5616**，gain **+0.0950**。所以 E1 → E2 不是评测集差异，是训练量。
- 训练集 in-sample +0.285（selector 0.847 vs greedy 0.561），验证 split seed 0 +0.079（0.642 vs 0.563）；test +0.084，val 与 test 一致。
- 参考基线：random 0.115、coverage 0.9225；greedy 用掉覆盖率的 61.2%，selector 用掉 70.5%。

### E3：2,000 prompts（继续上升）

GSM8K train 2,000 prompts × 16 token = 32,000 token（29,474 hit）；seed 0，在 1,000 条 test / 16,000 token 上：

| 指标 | 值 |
|---|---|
| selector Agree@1 | **0.6672** |
| greedy Agree@1 | 0.5645 |
| gain | **+0.1027** |
| 验证 split（400 prompts / 6,400 token） | +0.0947（0.646 vs 0.552） |
| in-sample | +0.274 |

相对 1k 的 +0.086，2k 再高约 1.7 pp；仍是 seed 0，未做 3-seed 重复。

### E4：2,000 条异质 QA 语料（no-go）

exp7-corpus `en_qa + zh_qa` 2,000 prompts × 16 token = 32,000 token（28,346 hit）；3 个 seed 在 500 条 held-out 文档 / 8,000 token 上：

| selector seed | selector Agree@1 | greedy Agree@1 | gain |
|---|---|---|---|
| 0 | 0.3420 | 0.5806 | −0.2386 |
| 1 | 0.3501 | 0.5806 | −0.2305 |
| 2 | 0.3468 | 0.5806 | −0.2339 |
| 均值 ± std | 0.3463 ± 0.0041 | 0.5806 | **−0.2343 ± 0.0041** |

- 同一 selector（seed 0）迁到 GSM8K 1,000 test：selector **0.3351** vs greedy 0.5645，gain **−0.2294**；完全不迁移。
- 验证 split 极具误导性：seed 0 是 −0.226，seed 1/2 是 +0.184/+0.195（均值 +0.051），但三个 seed 的 held-out 全部 −0.23 左右。**不能用 400-prompt val 选择 corpus selector。**
- 训练集 in-sample +0.205；50 epoch 明显在记文档特定的 hidden 模式。
- epoch sweep（3 seeds，400-prompt val）：2/5/10/20/50 epoch 的 val gain 均值是 −0.294 / −0.235 / −0.154 / −0.033 / +0.051；没有哪个配置在 500 held-out 上翻正。

### 规模曲线

| 训练 prompts | 域 | held-out | gain |
|---|---|---|---|
| 100 | GSM8K | 200 test | **−0.032**（3 seeds） |
| 1,000 | GSM8K | 1,000 test | **+0.086**（3 seeds） |
| 2,000 | GSM8K | 1,000 test | **+0.103**（seed 0） |
| 2,000 | exp7-corpus | 500 held-out | **−0.234**（3 seeds） |

## 解释

1. **同域数据量是第一瓶颈。** 100 → 1k prompts 把增益从 −3.2 pp 翻到 +8.6 pp，2k 到 +10.3 pp；val 与 test 在 1k/2k 上一致，说明这不是评测运气。selector 容量（327k）在 1k prompts / 16k token 上才够学到一个可迁移的重排函数。
2. **“更大的数据集”不等于“更同质的数据集”。** 同样是 32k token，2k 条 QA 文档的 selector held-out −23.4 pp，而 2k 条 GSM8K 题 +10.3 pp。异质语料让 selector 记住文档特定的 hidden 模式；400-prompt val 也选不出可靠配置。
3. **部署含义。** 对 GSM8K 这类同质任务，1k 条同分布 prompt 训练的 selector 是一个有希望的工作点：Agree@1 从 0.565 提到 0.65–0.67，只增加 327k 参数，推理不跑 base。对 PyroDash 产线 QA，必须先拿到 ≥1k 条同分布 prompt；现有 16 条 fixture 远远不够，不能用 GSM8K 的 selector 直接迁移。
4. **这不是端到端收益。** Agree@1 只衡量候选内选择与 base top-1 的一致率；任务正确率、长生成、延迟都未测。`claims_gain=false` 保持。

## 限制

- Agree@1 是 next-token 指标，不是最终答案正确率；selector 选对下一个 token 不等于把题做对。
- 1k 做了 3 个 seed；2k 只有 seed 0；corpus 2k 做了 3 个 seed。
- 只用一个 shadow（exp16 `all` seed 0）和一个 base（Qwen3-4B）；其他 shadow/base 组合未测。
- GSM8K 用 16-token 轨迹；更长生成、不同 K、不同温度未测。
- corpus 的 64-token 截断会产生不自然的文档边界；它衡量的是“异质文档前缀”下的选择，不是产线指令 QA。
- selector 只能重排 top-8；coverage 0.9225（GSM8K）/ 0.8955（corpus）是硬上限。
- 1k/2k 的 3 个 seed 中，train/val 切分随 seed 变化；test 指标不受影响。
- 没有 SFT baseline、没有端到端任务准确率、没有真实产线数据，因此不能声称生产收益。

## 命令

```bash
# E1：100 prompts
PYTHONPATH=src .venv/bin/python scripts/exp18_s2t_selector.py \
  --model ~/models/Qwen3-4B --shadow-model-dir ~/exp16-full/runs/all/model \
  --source gsm8k --train-limit 100 --max-new-tokens 16 --epochs 50 \
  --out ~/exp18-selector/all

# E2：1,000 prompts
PYTHONPATH=src .venv/bin/python scripts/exp18_s2t_selector.py \
  --model ~/models/Qwen3-4B --shadow-model-dir ~/exp16-full/runs/all/model \
  --source gsm8k --train-limit 1000 --max-new-tokens 16 --epochs 50 \
  --out ~/exp18-selector/gsm8k-1k

# E3：2,000 prompts
PYTHONPATH=src .venv/bin/python scripts/exp18_s2t_selector.py \
  --model ~/models/Qwen3-4B --shadow-model-dir ~/exp16-full/runs/all/model \
  --source gsm8k --train-limit 2000 --max-new-tokens 16 --epochs 50 \
  --out ~/exp18-selector/gsm8k-2k

# E4：exp7-corpus 2,000 prompts + 500 held-out（写 eval_prompts.jsonl）
PYTHONPATH=src .venv/bin/python scripts/exp18_s2t_selector.py \
  --model ~/models/Qwen3-4B --shadow-model-dir ~/exp16-full/runs/all/model \
  --source exp7-corpus --corpus-root ~/data/exp7-corpus/main \
  --corpus-subsets en_qa,zh_qa --prompt-tokens 64 \
  --train-limit 2000 --eval-limit 500 --max-new-tokens 16 --epochs 50 \
  --out ~/exp18-selector/corpus-2k

# 评测（E2/E3 用 GSM8K test；E4 用 eval_prompts.jsonl）
PYTHONPATH=src .venv/bin/python scripts/exp17_real_eval.py \
  --suite s2t --dataset gsm8k --model ~/models/Qwen3-4B \
  --shadow-model-dir ~/exp16-full/runs/all/model \
  --selector ~/exp18-selector/gsm8k-1k/selector \
  --limit 1000 --max-new-tokens 16 --k 8 --out ~/exp18-selector/gsm8k-1k/eval-1000

PYTHONPATH=src .venv/bin/python scripts/exp17_real_eval.py \
  --suite s2t --dataset ~/exp18-selector/corpus-2k/eval_prompts.jsonl \
  --model ~/models/Qwen3-4B --shadow-model-dir ~/exp16-full/runs/all/model \
  --selector ~/exp18-selector/corpus-2k/selector \
  --limit 500 --max-new-tokens 16 --k 8 --out ~/exp18-selector/corpus-2k/eval-500-corpus
```

seed 1/2 的训练：加载同一份 `train_examples.pt`，用 `train_selector(seed=1/2)` 重训后 `save_selector` 到 `selector-seed1/2`（见 `runs-local/exp18/*/train_report.json` 的 `data` 元数据）。

## 归档：E1 细节

E1 的 1,600 样本、1,480 hit；验证 80/20（1,280/320 token）。训练 loss 2.287 → 0.857（mean 1.497），val loss 1.435；in-sample +0.151，val −0.028，test 3 seeds 全负。参考 random 0.115、coverage 0.923。

## 产物

- 模块：`src/shadow_lab/s2t_selector.py`（`selector_agreement`）、`src/shadow_lab/real_eval.py`（standalone selector 路径）
- 脚本：`scripts/exp18_s2t_selector.py`（`--source gsm8k|exp7-corpus`）、`scripts/exp17_real_eval.py`
- 本地：`runs-local/exp18/{all,gsm8k-1k,gsm8k-2k,corpus-2k}/`（`train_report.json`、selector 目录、各 held-out `report.json`）
- 远端：`~/exp18-selector/`（goooinext + gooo）
- 相关档案：[docs/exp17-real-eval.md](exp17-real-eval.md)
