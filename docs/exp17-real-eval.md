# exp17 真实语料/真实模型复评 harness（issue #47）

一句话收束：**real-eval harness 支持 adapter（`ShadowForCausalLM`）和 standalone（直接加载 exp16 导出的 recovered shadow）。达到证据门槛的评测结果：GKD 200 条 GSM8K、`all` 臂 shadow ppl **3.524** / 准确率 **41.0%**（`backbone` 3.594 / 39.0%，旧 adapter shadow ppl 167,148）；S2T 200 prompts / 3200 token，Hit@8 **0.923** / Agree@1 **0.562**；RESTRAIN 100 条 GSM8K 结构化题 ×3 seeds，accuracy **1.75%** / low consistency **98–100%** / skip **73–84%**。GKD/S2T/RESTRAIN 均 `evidence_collected`，但 `claims_gain=false`（无完整 baseline-vs-treatment），JitRL 仍 `data_missing`。**

状态：**adapter/standalone 路径 + 证据门槛规模评测已完成（2026-10-22）**；GKD/S2T 各 200、RESTRAIN 100×3 seeds。仍缺 JitRL 真实反馈日志、S2T selector/baseline 对照、3-seed 训练侧重复；#47 保持 ready-for-human。

## 修复后的评测口径（2026-10-22 code review）

| 项 | 口径 |
|---|---|
| evidence 阈值 | 按 suite 判定（`EVIDENCE_MIN_EXAMPLES_BY_SUITE`）：GKD/S2T/JitRL ≥ **200** 条，RESTRAIN ≥ **100** 条；低于阈值一律 `evidence_insufficient` |
| S2T 指标名 | `hit_at_k` + `k` 字段；仅当 `k == 8` 额外输出 `hit_at_8`/`hit_at_8_ci` 作为**弃用兼容别名**。K≠8 时不再给出会被误读的 Hit@8 |
| GKD token budget | `--token-budget N` 映射为 `ceil(N / seq_len)` 个 ppl 窗口，并与 `max_windows` 取更紧者；报告实际 `ppl_tokens`/`ppl_windows`。不传时维持“全部窗口”现状 |
| S2T selector_gain | 已实现：`--selector <dir>` 通过 `shadow_lab.s2t_selector.load_selector` 加载 selector，用 `collect_selector_examples` 采集初始 shadow hidden，计算 `selector_gain = selector Agree@1 − shadow-greedy Agree@1`；不传 selector 时保持 `null`。selector 模式忽略 `--use-cache`（`collect_selector_examples` 每步重算 prefix），note 会写明 |
| JitRL latency | 每个示例记录 `memory_latency_ms`（memory retrieve + rerank），summary/report 给出 `memory_latency_ms_p50/p95/max`；无 memory 时仍为 `data_missing`，延迟字段为 `null` |
| RESTRAIN 判分 | 每条 rollout 只调用一次 `grade_structured`，同一份 outcomes 同时供 per-example 与 accuracy/consistency 聚合；口径不变 |
| 统一入口 | CLI 与 `run_suites` 共用 `shadow_lab.real_eval.run_suite` 与 `data_missing_report`；新增 suite 只需改 `real_eval`（`SUITES`/默认值/选项键/数据集映射/一个分支），CLI 不再维护第二套分派 |

修复后已用同一批真实数据在 goooinext 重跑 GKD/S2T/RESTRAIN/JitRL；指标与修复前一致，报告现在包含 `hit_at_k`、`ppl_tokens/ppl_windows`、`memory_latency_ms_*`（JitRL data_missing 时为 null）。下表是修复后产物；S2T 的 `hit_at_8` 仅在 k=8 时作为弃用别名保留。

## Standalone recovered shadow（2026-10-22）

`--shadow-model-dir` 直接加载 exp16 导出的独立 shadow（例如 `~/exp16-full/runs/all/model`），不需要 repack 成 adapter。`--model` 仍是 base（GKD/S2T 用）；RESTRAIN 只需要 shadow。

### 命令

```bash
# GKD：base 4B + recovered shadow
PYTHONPATH=src python scripts/exp17_real_eval.py --suite gkd --dataset gsm8k \
  --model ~/models/Qwen3-4B \
  --shadow-model-dir ~/exp16-full/runs/all/model \
  --limit 8 --smoke --device cuda --dtype bfloat16 \
  --out runs-local/exp17/gkd-standalone

# S2T：base 4B + recovered shadow
PYTHONPATH=src python scripts/exp17_real_eval.py --suite s2t --dataset gsm8k \
  --model ~/models/Qwen3-4B \
  --shadow-model-dir ~/exp16-full/runs/all/model \
  --limit 4 --smoke --k 8 --device cuda --dtype bfloat16 \
  --out runs-local/exp17/s2t-standalone

# RESTRAIN：只需要 recovered shadow
PYTHONPATH=src python scripts/exp17_real_eval.py --suite restrain \
  --dataset runs/exp12/heldout.jsonl \
  --shadow-model-dir ~/exp16-full/runs/all/model \
  --limit 8 --smoke --n 2 --device cuda --dtype bfloat16 \
  --out runs-local/exp17/restrain-standalone
```

### 结果（goooinext，16 token 冒烟）

| suite | adapter 冒烟 | recovered standalone | 说明 |
|---|---|---|---|
| GKD | joint 5.567 / shadow 167,148 / acc 0/8 | base ppl **4.891**、shadow ppl **3.771**、joint **5.222**、acc 0/8 | shadow ppl 从 167k 降到 3.77；acc 仍 0 是 16-token 预算太短 |
| S2T | Hit@8 0.922 / Agree@1 0.500 | Hit@8 **0.938** / Agree@1 **0.516** | 64 token；提升很小，仍需 selector |
| RESTRAIN | acc 0 / low consistency 1.0 / skip 0.875 | acc 0 / low consistency 1.0 / skip 0.875 | 16 token + 8 条，未改善 |
| JitRL | data_missing | 未跑 | 仍缺真实反馈/记忆/n-best 后端 |

- **GKD joint 口径不同**：standalone 的 `joint_ppl = exp(token-weighted mean(base CE + 0.05 * shadow CE)) = base_ppl * shadow_ppl ** 0.05`，base 是纯 4B，不经过 injection/update；不要和 adapter 路径的 `base_shadow` joint 直接横比。`shadow_ppl` 才是 recovered shadow 的关键指标。
- **S2T 仍是 base-prefix 诊断**：Hit@K 高、Agree@1 约 0.5，和 exp11 的结论一致；recovered shadow 让覆盖略升，但没有解决选择问题。
- **RESTRAIN 没有改善**：production QA 的 16 条 fixture 与 GSM8K 指令 SFT 不同分布；16 token 下 87.5% examples 仍不可聚类。
- 所有结果仍是 `claims_gain=false`、`evidence=insufficient`；这是 smoke，不是 2–4 天 campaign。

### 结果（中等规模：GKD 100 / S2T 20 / RESTRAIN 8×4）

| suite | 规模 | 指标 |
|---|---|---|
| GKD | GSM8K test 100 条，18,360 ppl token / 100 窗口，256 token 生成 | base ppl **4.523**、shadow ppl **3.600**、joint **4.820**、task accuracy **36.0%** [26.0, 44.0]，n_graded=100 |
| S2T | 20 prompts / 320 token，K=8 | Hit@8 **0.941** [0.909, 0.966]、Agree@1 **0.569** [0.516, 0.622]、`selector_gain=null` |
| RESTRAIN | exp12 held-out 8 条 ×4 rollouts，32 token | accuracy **0.0**、low consistency **1.0**、skip rate **1.0**、mean consistency 0.162、answer entropy 2.0、top1 share 0.25 |
| JitRL | 未跑 | 仍缺真实反馈/记忆/n-best 后端 |

- GKD 的 36.0% 是 **256 token** 预算；exp16 eval 的 40% 是 **512 token**，两者不能直接横比。shadow ppl 3.600 仍明显低于 base ppl 4.523，说明 recovered shadow 已拟合 GSM8K 分布。
- S2T 从 4 prompts 的 0.938/0.516 变成 20 prompts 的 0.941/0.569；覆盖稳定在 0.94，Agree@1 仍只有约 0.57，选择问题没有解决。
- RESTRAIN 在 32 token 下 **8/8 不可聚类**，skip rate 从 0.875 升到 1.0；16 条产线 QA 仍然不能靠这个 shadow 修复。
- 证据门槛：GKD 100 < 200、S2T 20 < 200、RESTRAIN 8 < 100，所以全部 `evidence=insufficient`；没有 3 seeds、baseline-vs-treatment 或真实反馈日志。

### 结果（证据门槛规模：GKD 200 / S2T 200 / RESTRAIN 100×3 seeds）

| suite | 规模 | 结果 | evidence |
|---|---|---|---|
| GKD `all` | GSM8K test 200，37,863 ppl token / 200 窗口，256 token | base ppl **4.475**、shadow ppl **3.524**、joint **4.765**、task accuracy **41.0%** [34.5, 48.0] | collected |
| GKD `backbone` | 同上 | base ppl 4.475、shadow ppl **3.594**、joint 4.771、task accuracy **39.0%** [32.5, 45.0] | collected |
| S2T `all` | 200 prompts / 3,200 token，K=8 | Hit@8 **0.923** [0.914, 0.931]、Agree@1 **0.562** [0.548, 0.578]、`selector_gain=null` | collected |
| RESTRAIN seed 0 | 100 条 GSM8K 结构化题 ×4 rollouts，64 token | accuracy **1.75%**、low consistency 0.99、skip 0.80、consistency 0.407 | collected |
| RESTRAIN seed 1 | 同上 | accuracy **1.75%**、low consistency 1.00、skip 0.84、consistency 0.410 | collected |
| RESTRAIN seed 2 | 同上 | accuracy **1.75%**、low consistency 0.98、skip 0.73、consistency 0.426 | collected |

- **GKD**：`all` 比 `backbone` 的 shadow ppl 低 0.070（约 −1.9%），任务准确率高 2.0 pp（41% vs 39%）；但 95% CI 重叠，不能声称显著。两臂都远好于旧 adapter 的 shadow ppl 167,148。
- **S2T**：200 prompts 上 Hit@8 0.923、Agree@1 0.562，和 20/4-prompt 冒烟一致；覆盖稳定，选择仍是瓶颈。
- **RESTRAIN**：在 100 条 GSM8K 结构化题上 accuracy 只有 1.75%，low consistency 98–100%，skip 73–84%；这是明确的 no-go。影子是在 GSM8K solution 上 SFT 的，但 64 token 内很少产出可解析的最终数字，`process_calc` 的 first-number 判分也会放大误差。生产 QA 16 条仍是 0% / 1.0 / 0.875，两个域都没有改善。
- 证据门槛已满足（GKD/S2T 200、RESTRAIN 100），但 `claims_gain` 仍为 false：没有完整 baseline-vs-treatment campaign；JitRL 仍缺真实反馈日志/记忆/n-best 后端。

## 实验设置

| 项 | 值 |
|---|---|
| 机器 | `goooinext`（dc4a，GB10），CUDA bf16 |
| 代码 | 集成分支 `feat/46-47-research`，模块 `shadow_lab.real_eval`，脚本 `scripts/exp17_real_eval.py` |
| GKD | base `Qwen3-0.6B` + `runs/exp10/smoke/fresh-adapter`；GSM8K test 8 条 |
| S2T | base `Qwen3-4B` + `runs/exp7/e4b-seed0/ckpt-final` + `Qwen3-0.6B-H4B`；GSM8K test 4 条，K=8 |
| RESTRAIN | base `Qwen3-0.6B` + `runs/exp12/train/weighted`；`runs/exp12/heldout.jsonl` 8 条，n=2，temp=1.0 |
| JitRL | 无 `--dataset`/`--memory`/`--backend-config`，预期 `data_missing` |
| 生成 | smoke 预算 16 token；greedy/采样按 suite 默认 |
| 产物 | `runs-local/exp17/{gkd,restrain,s2t,jitrl}/` |

### 命令

```bash
# GKD
PYTHONPATH=/home/goooice/p4647-run/src \
  /home/goooice/Code/shadow-peft/.venv/bin/python \
  /home/goooice/p4647-run/scripts/exp17_real_eval.py \
  --suite gkd --dataset gsm8k \
  --model /home/goooice/models/Qwen3-0.6B \
  --shadow-ckpt /home/goooice/Code/shadow-peft/runs/exp10/smoke/fresh-adapter \
  --limit 8 --smoke --device cuda --dtype bfloat16 \
  --out /home/goooice/p4647-run/exp17-gkd-smoke

# S2T（--selector 可选；指向 exp11 保存的 selector 目录才会计算 selector_gain）
PYTHONPATH=/home/goooice/p4647-run/src \
  /home/goooice/Code/shadow-peft/.venv/bin/python \
  /home/goooice/p4647-run/scripts/exp17_real_eval.py \
  --suite s2t --dataset gsm8k \
  --model /home/goooice/models/Qwen3-4B \
  --shadow-ckpt /home/goooice/Code/shadow-peft/runs/exp7/e4b-seed0/ckpt-final \
  --shadow-model /home/goooice/models/Qwen3-0.6B-H4B \
  --limit 4 --smoke --k 8 --use-cache --device cuda --dtype bfloat16 \
  --out /home/goooice/p4647-run/exp17-s2t-smoke

# RESTRAIN
PYTHONPATH=/home/goooice/p4647-run/src \
  /home/goooice/Code/shadow-peft/.venv/bin/python \
  /home/goooice/p4647-run/scripts/exp17_real_eval.py \
  --suite restrain \
  --dataset /home/goooice/Code/shadow-peft/runs/exp12/heldout.jsonl \
  --model /home/goooice/models/Qwen3-0.6B \
  --shadow-ckpt /home/goooice/Code/shadow-peft/runs/exp12/train/weighted \
  --limit 8 --smoke --n 2 --device cuda --dtype bfloat16 \
  --out /home/goooice/p4647-run/exp17-restrain-smoke

# JitRL 数据缺失路径
PYTHONPATH=/home/goooice/p4647-run/src \
  /home/goooice/Code/shadow-peft/.venv/bin/python \
  /home/goooice/p4647-run/scripts/exp17_real_eval.py \
  --suite jitrl --limit 4 --smoke \
  --out /home/goooice/p4647-run/exp17-jitrl-smoke
```

新参数（需重跑才会出现在产物中）：

```bash
# GKD：最多评 8192 个 ppl target token（seq_len=2048 时即 4 个窗口）
... --suite gkd ... --token-budget 8192

# S2T：加载 exp11 selector，报告 selector_gain
... --suite s2t ... --selector runs/exp11/s2t-selector
```

## Standalone metric semantics and adapter comparison

`--shadow-model-dir` 与 `--shadow-ckpt` 互斥（argparse 直接拒绝）；`--selector` 和
`--shadow-model` 只适用于 adapter 路径，standalone 下会报参数错误。JitRL 仍走
`--backend-config`/`--memory`，缺数据时 `data_missing`。报告 `config` 记录
`mode`（`standalone`/`adapter`）和 `shadow_model_dir`。

### Metric 语义

- `base_ppl`：base LM 的 perplexity，单独前向计算，**不经过 shadow injection 或 update**。
- `shadow_ppl`：recovered shadow（shadow backbone + hidden projection + 冻结的 base head）的 perplexity。
- `joint_ppl`：逐窗口 `joint_loss = base_loss + 0.05 * shadow_loss`，对 target token 加权平均后取 `exp`；等价于 `base_ppl * shadow_ppl ** 0.05`（加权几何平均），不是混合分布。窗口切分复用 `onpolicy_eval._ppl_windows`，`--token-budget`/`--max-windows` 语义不变。
- GKD 的 `task_accuracy` 用 standalone shadow 生成（不再是 adapter joint 模型）；metrics 额外给出 `mode="standalone"`、`base_ppl`、`base_ppl_ci`。
- S2T 每个位置分别前向 base 与 shadow：base top-1 决定生成轨迹，shadow 提供 top-K 候选、`base_top1_rank_in_shadow` 与 `agree_at_1`。standalone 模式没有 KV cache（每步重算 prefix），`--use-cache` 会被忽略并写入 note；metrics 额外给出 `mode="standalone"`。
- RESTRAIN 直接把 standalone shadow 当生成模型（`adapter=None`），判分逻辑不变。

### 与 adapter 路径的区别

| 项 | adapter 路径（`--shadow-ckpt`） | standalone 路径（`--shadow-model-dir`） |
|---|---|---|
| 加载 | base + `ShadowForCausalLM`（`base_shadow` 注入） | base 4B + exp16 导出的 `AutoModelForCausalLMWithHiddenProjection` |
| GKD ppl | `evaluate_ppl`：joint 走 `base_shadow` 前向，shadow 走 `shadow_only` 前向 | `evaluate_standalone_ppl`：base/shadow 两次独立前向，joint 为加权和 |
| metrics | 保持原样（无 `mode`/`base_ppl`） | 额外 `mode="standalone"`、`base_ppl`（+ `base_ppl_ci`） |
| S2T | `diagnose`：一次前向同时取 `logits` 与初始 `shadow_logits` | `standalone_s2t_diagnose`：base/shadow 独立前向 |
| 显存 | base + shadow 包装 | base 4B 与 shadow 4B 同时驻留，单卡需放得下两份权重 |
| 吞吐 | 单次前向 | S2T 无 KV cache，每 token 两次前向；200 prompt × 32 token 明显慢于 adapter 路径，先用 `--limit`/`--max-new-tokens` 冒烟 |

## 结果

| suite | 数据/规模 | 关键指标 | 状态 |
|---|---|---|---|
| GKD | GSM8K test 8 条，1077 ppl token | joint ppl **5.567** [4.573, 7.145]；shadow ppl **167,148** [144,310, 206,846]；task accuracy **0/8** | `ok`，evidence insufficient |
| S2T | 4 prompts，64 token，K=8 | Hit@k（k=8，`hit_at_8` 别名）**0.922** [0.844, 0.969]；Agree@1 **0.500** [0.359, 0.610]；`selector_gain=null`（未传 `--selector`） | `ok`，evidence insufficient |
| RESTRAIN | exp12 held-out 8 条，16 rollouts | accuracy **0.0**；low consistency **1.0**；mean consistency 0.277；skip rate **0.875**；top1 share 0.667 | `ok`，evidence insufficient |
| JitRL | 无数据/记忆/后端 | —（`memory_latency_ms_p50/p95/max` 均为 null） | `data_missing` |

所有报告：`claims_gain=false`、`evidence=insufficient`。S2T 的 `go_no_go=go` 是 S2T 自身的局部诊断（Hit@8 阈值），不代表本轮 64 token 冒烟有统计意义；在 summary 里该数值以 `hit_at_k`（k=8）呈现。

## 解释与后续

1. **Harness 可用**：四个 suite 的统一入口、数据指纹、per-example、summary/report 都能跑；缺数据时不会伪造结果。
2. **recovered standalone 的结论**：GKD 200 条上 `all` 3.524 ppl / 41.0%、`backbone` 3.594 / 39.0%，都远好于旧 adapter 的 167k；S2T 200 prompts 上 Hit@8 0.923 / Agree@1 0.562；RESTRAIN 100 条 GSM8K 结构化题 ×3 seeds 只有 1.75% accuracy、98–100% low consistency、73–84% skip。**GKD/S2T 可用，RESTRAIN no-go。**
3. **GKD 两臂对比**：`all` 比 `backbone` 的 shadow ppl 低约 1.9%、准确率高 2.0 pp，但 CI 重叠；不能声称显著。`all` 训练了 4B head，成本更高；如果只需要格式/成本，`backbone` 仍是合理选择。
4. **S2T 仍是选择瓶颈**：Hit@8 稳定在 0.92，Agree@1 只有 0.56；和 exp11 的 selector no-go 一致。下一步应重训/评估 selector，而不是继续加 K。
5. **仍未做**：JitRL 缺真实反馈日志/记忆/n-best 后端（`data_missing`）；S2T selector/baseline 对照；GKD/S2T 的 3-seed 训练侧重复（eval 是 greedy，seed 不改变结果）；`claims_gain=false` 仍然成立。

## 已知限制

- GKD/S2T 各 200 条、RESTRAIN 100 条；GKD 生成预算 256 token、S2T 16 token、RESTRAIN 64 token。RESTRAIN 有 3 seeds，GKD/S2T 是 greedy，seed 不改变结果。
- GKD 只用数值容差评最终答案，不校验推理过程；分数/小数可能误判。
- RESTRAIN 的 100 条是 **GSM8K 派生的 `process_calc` 结构化题**，不是产线 QA；`process_calc` 的 first-number 判分可能放大误差。生产 QA 16 条仍是 0% / 1.0 / 0.875。
- `--token-budget` 只限制 ppl target token，不限制生成 token；实际窗口数受语料长度影响，报告会给出真实 `ppl_tokens/ppl_windows`。
- S2T selector_gain 只在显式传 `--selector` 时计算；selector 模式走 `collect_selector_examples`，忽略 `--use-cache`，且 metrics 基于初始 shadow hidden（不是注入部署路径）。
- JitRL 的 cloud 升级是简化阈值，不是完整生产 Router；本轮没有真实记忆/后端，直接 `data_missing`，latency 字段为 null。
- `claims_gain=false` 是设计使然：harness 没有 baseline-vs-treatment 对照臂；GKD `all` vs `backbone` 是手工补充对比，不是完整 campaign。
- 没有跑 3-seed 训练侧重复、S2T selector/baseline 或 JitRL 真实数据，因此不能声称 #2–#5 的真实收益或失败。

## 产物

- 模块：`src/shadow_lab/real_eval.py`
- 脚本：`scripts/exp17_real_eval.py`
- 测试：`tests/lab/test_real_eval.py`（44 个，含 standalone/projected tiny-model 用例）
- 冒烟产物：`runs-local/exp17/{gkd,restrain,s2t,jitrl}/`（`report.json`、`summary.csv`、`per_example.jsonl`）
- standalone 产物：`runs-local/exp17/standalone/`
- 证据门槛规模产物：`runs-local/exp17/large/{gkd-200-all,gkd-200-backbone,s2t-200-all,restrain-gsm8k-seed0,restrain-gsm8k-seed1,restrain-gsm8k-seed2}/`
