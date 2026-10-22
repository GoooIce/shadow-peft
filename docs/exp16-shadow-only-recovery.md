# exp16 shadow-only 恢复管线（issue #46）

一句话收束：**完整的 shadow-only 指令 SFT 把 exp7 导出影子从 0% 恢复到了可用区间：2k GSM8K train × 3 epochs 后，`backbone` 臂 GSM8K test 36% / 格式 99% / 截断 1%，`all` 臂 40% / 格式 96% / 截断 4%；原生 Qwen3-0.6B baseline 是 47% / 94% / 0%。两臂都通过“格式 ≥80%、准确率 ≥原生 70%、截断 ≤20%”的门禁（ratio 0.766 / 0.851），round-trip 逐位通过（max diff 0.0）。**

状态：**完整训练已在 goooinext 跑完（2026-10-22）**；GSM8K 恢复门禁通过，`all` 更准、`backbone` 格式更稳；产线 QA 16 没有改善，exp7 held-out ppl 未测，因此这是 GSM8K 恢复证据，不是通用能力结论。

## 实验设置

| 项 | 值 |
|---|---|
| 机器 | `goooinext`（dc4a，GB10），CUDA bf16 |
| 代码 | 集成分支 `feat/46-47-research`，模块 `shadow_lab.shadow_only`，脚本 `scripts/exp16_shadow_only.py` |
| 基座 | `~/models/Qwen3-4B`（冻结，仅用于导出时的 base） |
| 影子 | `~/models/Qwen3-0.6B-H4B` + `~/Code/shadow-peft/runs/exp7/e4b-seed0/ckpt-final` |
| 导出 | `ShadowForCausalLM` → `peft_model.export_shadow()`，得到独立 `AutoModelForCausalLMWithHiddenProjection`；保存模型 + tokenizer |
| 训练范围 | `train_scope=backbone`（596,049,920 / 987,627,520 可训练参数；projection/head 冻结） |
| 数据 | `openai/gsm8k` train 子集，8 条冒烟样本；chat template + prompt 位置 `labels=-100` |
| 超参 | 2 步、batch 2、max_seq_len 256、lr 1e-5、seed 0、grad clip 1.0 |
| 评测 | 8 条 GSM8K test，greedy，max_new_tokens 64 |
| 原生 baseline | `~/models/Qwen3-0.6B`，同一批 8 条 GSM8K prompts、64 token；baseline accuracy 0/8，报告 `accuracy_status="baseline_zero"` |
| 产线 QA | 16 条 `data/pyrodash_stage0/production_qa.jsonl`，诊断项（不进门禁） |
| 产物 | `runs-local/exp16/{report.json,diagnose_generations.jsonl,generations.jsonl,baseline_generations.jsonl,qa_generations.jsonl,train_stats.json,model/}` |

### 命令

```bash
# 在 goooinext 的临时代码目录
PYTHONPATH=/home/goooice/p4647-run/src \
  /home/goooice/Code/shadow-peft/.venv/bin/python \
  /home/goooice/p4647-run/scripts/exp16_shadow_only.py run \
  --base-model /home/goooice/models/Qwen3-4B \
  --shadow-model /home/goooice/models/Qwen3-0.6B-H4B \
  --shadow-ckpt /home/goooice/Code/shadow-peft/runs/exp7/e4b-seed0/ckpt-final \
  --baseline-model /home/goooice/models/Qwen3-0.6B \
  --out /home/goooice/p4647-run/exp16-smoke2 \
  --smoke --device cuda --dtype bfloat16
```

- `--baseline-model`：让原生 0.6B 在同一批 GSM8K prompts 上生成并判分，报告 `baseline_accuracy` / `accuracy_ratio` / `accuracy_gate_passed`。不提供时 `baseline.status="not_run"`、`accuracy_ratio=null`，`recovery_ready` 必为 `false`。
- `--qa-jsonl`：覆盖产线 QA 数据路径；默认即仓库内 `data/pyrodash_stage0/production_qa.jsonl`（16 条）。`quality_verdict` / `process_calc` 走 `shadow_lab.restrain.extract_answer`，其余按 `expected_keywords` 覆盖率（≥0.5 记对）。QA 是诊断项，不进入 recovery gate。
- `--epochs N`：训练步数改为 `ceil(len(rows)/batch_size)*N`，此时忽略 `--steps`；不传时仍按 `--steps`（冒烟默认 2 步）。

## 结果

### 导出 round-trip

| 项 | 修复前 | 修复后 |
|---|---:|---:|
| `passed` | false | **true** |
| `max_abs_diff` | 0.875 | **0.0** |
| `allclose(1e-4)` | false | **true** |
| tokenizer 文件 | 3 | 3 |
| 模型文件 | 3 | 3 |

- 失败原因不是权重丢失：逐参数比较 312 个 tensor 完全一致。差异来自 Qwen3 的 **非持久 rotary `inv_freq` 缓冲**：导出模型被 `.to(bf16)` 后缓冲是 bf16，`from_pretrained(dtype=bf16)` 重载时缓冲被重新初始化为 fp32，导致 RoPE 表不同、隐藏状态最大差 2.5、logits 最大差 0.875。
- 修复：`export_roundtrip_check` 和 `exp16 eval` 的加载路径在 reload 后显式 `model.to(device=..., dtype=torch_dtype)`，把非持久缓冲也 cast 回导出 dtype；修复后 logits 逐位一致（max diff 0.0）。对应测试 `test_export_roundtrip_check_casts_non_persistent_buffers`。

### 2 步 shadow-only SFT 冒烟

| 指标 | 诊断（训练前） | 评测（2 步后） |
|---|---:|---:|
| 样本 | 4 | 8 |
| 格式合法率 | 0.0 | 0.0 |
| 准确率 | 0.0 | 0.0 |
| 截断率 | 1.0 | 1.0 |
| 空输出率 | 0.0 | 0.0 |
| mean distinct-3 | 0.959 | 0.912 |
| mean repetition-4 | 0.253 | 0.334 |

- 训练：2 步、8 行、560 token，loss 1.359 → 1.227；每步约 0.35 秒。2 步只能证明梯度/保存/加载链路通，不能改变生成行为。
- `recovery_ready=false`：round-trip 通过，但格式和截断门禁未过；原生 0.6B baseline 在同样 8 条、64 token 下也是 0/8（`accuracy_status="baseline_zero"`），因此 `accuracy_ratio=null`、`accuracy_gate_passed=false`。现在 accuracy gate 是 recovery 硬条件：没有 baseline 或 baseline 为 0 都不可能为 true。

### 修复后重跑的 baseline 与 QA 16

| 项 | 值 |
|---|---:|
| baseline n / accuracy / format / truncation | 8 / 0.0 / 0.0 / 1.0 |
| shadow eval n / accuracy / format / truncation | 8 / 0.0 / 0.0 / 1.0 |
| QA 16 n / accuracy / format / keyword coverage | 16 / 0.0625 / 0.625 / 0.2556 |
| QA `keyword_full_rate` | 0.0 |
| `accuracy_status` / `accuracy_ratio` | `baseline_zero` / `null` |
| `recovery_ready` | `false` |

- 原生 0.6B 在 64 token 预算下也是 0/8，说明这个冒烟预算太短，无法区分“影子差”与“预算不足”；正式评测必须用与 exp14 一致的 512 token 或更长，并报告 baseline 的格式/截断率。
- QA 16 只有 1/16 正确（6.25%），且 `process_calc` 走 `extract_answer` 的首个数字规则；它只是诊断，不是恢复门禁。

### 完整 2k×3 SFT（2026-10-22）

设置：`openai/gsm8k` train 2000 条、3 epochs、batch 2、max_seq_len 512、lr 1e-5、seed 0、bf16；评测为 GSM8K test 100 条、greedy、max_new_tokens 512，原生 `Qwen3-0.6B` 用同一批 prompt/参数。`backbone` 臂冻结 projection/head；`all` 臂训练全部 987,627,520 参数。

| 训练 | backbone | all |
|---|---:|---:|
| steps / epochs | 3000 / 3 | 3000 / 3 |
| tokens | 748,110 | 748,110 |
| loss first → last | 1.430 → **0.309** | 1.430 → **0.301** |
| loss mean | 0.498 | 0.490 |
| trainable / total | 596M / 988M | 988M / 988M |
| wall | 668 s | 917 s |

| 评测 | pre-SFT | backbone | all |
|---|---:|---:|---:|
| GSM8K n | 100 | 100 | 100 |
| 准确率 | 0.00 | **0.36** | **0.40** |
| 格式合法率 | 0.00 | **0.99** | **0.96** |
| 截断率 | 0.07 | **0.01** | **0.04** |
| 空输出 | 0.00 | 0.00 | 0.00 |
| 平均 token | 189.7 | 120.4 | 131.8 |
| 原生 baseline 准确率 | 0.47 | 0.47 | 0.47 |
| accuracy ratio | 0.00 | **0.766** | **0.851** |
| accuracy gate | fail | **pass** | **pass** |
| `eval_ready` | false | **true** | **true** |
| 产线 QA 16 准确率 | 0.125 | 0.0625 | 0.0625 |

- **恢复结论**：两臂都通过了 GSM8K 恢复门禁；`all` 的准确率更高（40% vs 36%，ratio 0.851），`backbone` 的格式/截断更好（99% / 1%）。如果只选一个工作点，推荐 `all` 作为质量优先、`backbone` 作为成本/格式优先。
- **`eval_ready` vs `recovery_ready`**：`eval` 命令不跑 round-trip，所以旧报告里的 `recovery_ready=false` 是“该命令没有执行 round-trip”的报表口径，不是 SFT 失败。现在 `eval_ready` 单独表示后训练四门（格式/截断/空输出/accuracy ratio）；round-trip 已在 diagnose/run 中单独证明为 max diff 0.0。`run` 命令会同时给出两者。
- **QA 16 没有改善**：pre 12.5% → backbone/all 6.25%。16 条 fixture 的 `quality_verdict`/`process_calc` 判分对标签写法敏感，且 512-token GSM8K SFT 不等于产线 QA 指令跟随；这只能当诊断，不能否定 GSM8K 恢复。
- **耗时远低于 ticket 的 3–5 天估计**：两臂各 3000 步，训练 11–15 分钟；完整 pre/train/eval 流程不到 1 小时。ticket 的 3–5 天是含多 seed/多 checkpoint/真实数据的上限估计。

### 修复后的门禁与诊断口径

| 输出 | 含义 |
|---|---|
| `gates.accuracy_status` | `not_run`（无 baseline） / `baseline_zero` / `passed` / `failed` |
| `gates.baseline_accuracy` | 原生模型在同一批 prompts 上的准确率；无 baseline 时为 `null` |
| `gates.accuracy_ratio` | `shadow_accuracy / baseline_accuracy`；无 baseline 或 baseline 为 0 时为 `null` |
| `gates.accuracy_gate_passed` | ratio ≥ 0.70；无 baseline 时恒为 `false` |
| `gates.eval_ready` | eval 格式 + 截断 + 空输出 + accuracy gate 四门全过；`eval` 单独运行时用它表示后训练可用性 |
| `gates.recovery_ready` | 在 `eval_ready` 之上再加 round-trip；只有 `run` 会同时执行 round-trip，`eval` 单独运行时为 `false` 是报表口径，不是失败 |
| `qa.*` | 16 条产线 QA 的 n / accuracy / format_rate / keyword_coverage；诊断项，不进门禁 |
| `baseline.status` | `ok` 或 `not_run`；`not_run` 时报告 `reason`，不伪造数值 |

## 解释与后续

1. **管线可用**：导出、保存、重载、tokenizer、SFT、生成评测都已跑通；round-trip 现在是逐位一致，不再是 bf16/fp32 RoPE 的假失败。
2. **完整 SFT 恢复了 GSM8K 生成**：pre-SFT 0% 格式/0% 准确率，2k×3 SFT 后 `backbone` 36% / 99% 格式、`all` 40% / 96% 格式，均超过“原生 70%”（0.766 / 0.851）。2 步冒烟仍是链路检查，但完整训练已经不是 no-go。
3. **选点建议**：质量优先选 `all`（40%，ratio 0.851）；格式/成本优先选 `backbone`（99% 格式，596M 可训练参数，训练 668s vs 917s）。两臂都满足门禁，没有出现熵塌缩或格式崩坏。
4. **仍未覆盖**：产线 QA 16 没有改善（12.5% → 6.25%），exp7 held-out ppl 未测，只有 1 个 seed、100 条 GSM8K test。下一步应补 ppl 和 QA 指令数据，或用恢复后的影子重跑 #47 的 GKD/S2T；这不影响 GSM8K 恢复结论。
5. **2k×3 epochs 与默认冒烟的差异**：默认 `--smoke` 不传 `--epochs` 时仍是 `--steps 2`，只验证链路；`--epochs 3` 会按数据量重算步数并忽略 `--steps`，不能把两者混为同一个实验口径。

## 已知限制

- 只测一个 checkpoint（exp7 E-arm `ckpt-final`）、一个 seed、100 条 GSM8K test；多 seed/全量 test 尚未跑。
- 训练默认 bf16、无 autocast/fp32 master weights；更长训练/更高 lr 可能需要调整。
- 训练数据是 GSM8K 完整 solution，模型可能先学格式而非推理；`all` 臂还训练了 4B head（389M），存在过拟合/容量混入的可能，需多 seed 复核。
- baseline 与 shadow 共用同一批 **prompt 字符串**，但各自使用自己的 tokenizer 编码；Qwen3 系列 tokenizer 一致时口径才严格可比。
- 产线 QA 当前只有 16 条且按关键词/结构化抽取判分（`process_calc` 用 `extract_answer` 的首个数字规则）；本轮 QA 反而从 12.5% 降到 6.25%，不能据此声称通用指令跟随恢复。
- exp7 held-out ppl 尚未测；训练只做了 GSM8K 指令 SFT，不是 exp7 续训语料的 ppl 回归。
- 本轮实际训练只有约 11–15 分钟/臂，远低于 ticket 的 3–5 天估计；不要把“跑通 2k×3”当成完整研究 campaign（多 seed、多 checkpoint、真实产线数据仍未做）。

## 产物

- 模块：`src/shadow_lab/shadow_only.py`
- 脚本：`scripts/exp16_shadow_only.py`
- 测试：`tests/lab/test_shadow_only.py`（52 个）
- 冒烟产物：`runs-local/exp16/`（`report.json`、`diagnose_generations.jsonl`、`generations.jsonl`、`baseline_generations.jsonl`、`qa_generations.jsonl`、`train_stats.json`、`model/`）
- 完整产物：`runs-local/exp16/full/{pre,backbone,all}/`（report/generations/baseline/QA/train_stats；模型权重保留在 `goooinext:~/exp16-full/runs/{backbone,all}/model/`，未回传以节省本地空间）
- 完整运行日志：`runs-local/exp16/full/logs/`（pre/train/eval 各阶段）
