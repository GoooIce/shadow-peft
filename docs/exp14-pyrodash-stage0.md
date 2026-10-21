# exp14 PyroDash Stage 0 — 影子 shadow-only 能力诊断（spec #6 / issue #41）

一句话收束：**在 GSM8K test 的 100 条 prompt 上，原生 `Qwen3-0.6B` 准确率 45.0%、格式合法率 94.0%，`Qwen3-4B-Instruct-2507` 准确率 73.0%（73 个正确答案全部带 `####`），而 exp7 导出的显式影子（`Qwen3-0.6B-H4B` + `runs/exp7/e4b-seed0/ckpt-final`）准确率 0.0%、格式合法率 0.0%；Stage 0 判定 `mechanism_only`——任务与判分可用、机制可以在有能力的 SLM 上测，但 exp7 的导出影子不能生成可判分的推理轨迹，因此不进入 Stage 1 的词表/embedding 训练。**

状态：**Stage 0 已在 `goooinext`（GB10）跑完（2026-10-21）**；同日按 no-go 停止控制 token 路线，Stage 1–3 未启动；没有扩词表、没有训练控制 token、没有改库行为。

## 实验设置

| 项 | 值 |
|---|---|
| 机器 | `goooinext`（dc4a，GB10，128GB 统一内存），Linux 6.17 aarch64 |
| 代码 | 分支 `feat/pyrodash-stage0`，脚本 `scripts/exp14_pyrodash_stage0.py` |
| 运行时 | torch 2.12.1+cu130、transformers 5.15.1、datasets 5.0.1、bf16 |
| 原生臂 | `~/models/Qwen3-0.6B`（base） |
| 影子臂 | `~/models/Qwen3-0.6B-H4B` + `runs/exp7/e4b-seed0/ckpt-final` + 冻结 `~/models/Qwen3-4B`，经 `export_shadow()` 导出为独立模型 |
| 对照臂 | `~/models/Qwen3-4B-Instruct-2507` |
| 数据 | GSM8K test 100 条（seed 0 抽样；`openai/gsm8k` main split）+ 产线 QA 16 条（8 条 exp12 held-out + 8 条仓库知识库） |
| 解码 | greedy、batch 4；GSM8K `max_new_tokens=512`，产线 QA `max_new_tokens=128` |
| seed | 0；bootstrap 1000 次 |
| 数据指纹 | `prompts.jsonl` 见 `runs/exp14/stage0-full/`；产线 QA 固定文件 `data/pyrodash_stage0/production_qa.jsonl` |

### 判定口径

- **GSM8K**：格式合法 = 输出含可解析的 `#### <number>`；正确 = 与 gold 数值相等（1e-6 相对容差）。
- **产线 QA**：`quality_verdict` 复用 `shadow_lab.restrain.extract_answer`；`process_calc` 取输出中**最后一个**数字（模型常先写公式再写结果）；其余任务按关键词覆盖率 ≥0.5 判对。
- **产品臂门禁**（原生 0.6B、导出影子）：格式 ≥80%、准确率 10–90%、词级 distinct-3 ≥0.5、截断 ≤20%、空输出 ≤5%；教师可解 hard ≥50%；影子解决率 ≥原生 0.6B 的 70%、影子格式 ≥原生 0.6B 的 80%。
- **对照臂**：准确率 ≥50%，且没有正确答案因格式无法解析；计划中的 0.90 严格格式门禁只作诊断（见“门禁偏差”）。
- **判定树**：对照失败 → `no_go_control_failed`；影子全过 → `go`；原生过、影子不过 → `mechanism_only`；两者都不过 → `no_go`。

## 命令

```bash
# 完整三臂 Stage 0（goooinext）
cd ~/Code/shadow-peft
PYTHONPATH=src .venv/bin/python scripts/exp14_pyrodash_stage0.py \
  --out runs/exp14/stage0-full \
  --gsm8k-limit 100 --qa-limit 16 \
  --max-new-tokens-gsm8k 512 --max-new-tokens-qa 128 \
  --batch-size 4 --device cuda --dtype bfloat16 --seed 0 --n-boot 1000

# 不重新加载模型，用当前判分器从已有 generations.jsonl 重建报告
PYTHONPATH=src .venv/bin/python scripts/exp14_pyrodash_stage0.py \
  --report-from runs/exp14/stage0-full --out runs/exp14/stage0-full --regrade

# 本地纯函数测试
PYTHONPATH=src /Users/devel0per/Code/shadow-peft/.venv/bin/python \
  -m pytest tests/lab/test_pyrodash_stage0.py -q
/Users/devel0per/Code/shadow-peft/.venv/bin/ruff check \
  scripts/exp14_pyrodash_stage0.py tests/lab/test_pyrodash_stage0.py
```

本次结果：`26 passed`；ruff `All checks passed!`；完整三臂墙钟约 13 分钟（raw 2 分钟、shadow 4 分钟、control 7 分钟）。

## 结果

### GSM8K（100 条 test）

| 臂 | 准确率 | 95% CI | 格式合法率 | 截断率 | 空输出 | distinct-3（词） | distinct-3（字） | 平均 token |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| `raw06` | **45.0%** | [35.0%, 54.0%] | **94.0%** | 1.0% | 0.0% | 0.932 | 0.581 | 130.9 |
| `shadow` | **0.0%** | [0.0%, 0.0%] | **0.0%** | 7.0% | 0.0% | 0.810 | 0.463 | 189.7 |
| `ctl4b` | **73.0%** | [65.0%, 81.0%] | 79.0% | 8.0% | 0.0% | 0.920 | 0.486 | 248.4 |

- 影子的 100 条输出里 **0 条**包含 `####`；95 条能抽到某个数字，但没有任何一条与 gold 相等。失败不是判分器把它“误判”了：它根本没有生成最终答案行。
- 影子最常见的失败模式是复读和半截推理：例如产线 QA 的输出出现“焊点间距与焊点间距的比值”连续重复，GSM8K 输出在 512 token 内没有收束到 `####`。这与 exp7 的 shadow-only 漂移、exp10 的 shadow ppl 285,524 一致。
- 原生 0.6B 用同一个 chat template 能解 45%，说明 GSM8K 对 0.6B 级别并非不可做；瓶颈是 exp7 导出影子，而不是任务或 prompt 格式。
- 4B-Instruct 的 73 个正确答案全部带 `####`；21 条未带 `####` 的输出全部是错答。它的严格格式率 79% 是“错答不写格式”，不是判分器丢正确答案。

### 产线 QA（16 条 smoke）

| 臂 | 准确率 | 95% CI | 格式合法率 | 关键词覆盖均值 |
|---|---:|---|---:|---:|
| `raw06` | 6.25% | [0.0%, 18.75%] | 87.5% | 0.256 |
| `shadow` | 6.25% | [0.0%, 18.75%] | 81.2% | 0.256 |
| `ctl4b` | **50.0%** | [25.0%, 75.0%] | 93.8% | 0.533 |

逐任务看，4B-Instruct 的分数主要来自 `process_calc` 4/5、`fault_diagnosis` 2/2、`safety` 1/1；`quality_verdict` 只有 1/5，`process_qa`/`standard_qa`/`improvement` 各 0/1。产线 QA 只有 16 条、`quality_verdict` 的 expected 只是“一种可接受写法”（例如“返修”与“不合格”的边界），**不能用来给三臂排序**；它的作用是确认端到端格式和判分链路能跑通。

### 门禁

| 门禁 | `raw06` | `shadow` | `ctl4b` |
|---|---|---|---|
| 格式 ≥80% | 通过（94.0%） | **失败（0.0%）** | 诊断 79.0% |
| 准确率 10–90% | 通过（45.0%） | **失败（0.0%）** | — |
| 词级 distinct-3 ≥0.5 | 通过（0.932） | 通过（0.810） | — |
| 截断 ≤20% | 通过（1.0%） | 通过（7.0%） | — |
| 空输出 ≤5% | 通过（0.0%） | 通过（0.0%） | — |
| 教师可解 hard ≥50% | 通过（33/55 = 60.0%） | 通过（73/100 = 73.0%） | — |
| 影子 ≥ 原生 70% | — | **失败（0.0 / 45.0 = 0.0）** | — |
| 对照准确率 ≥50%、无正确答案丢失 | — | — | **通过（73/100，73/73 带格式）** |

- **最终判定：`mechanism_only`**。原生 0.6B 通过产品门禁，4B-Instruct 对照通过；导出影子在准确率和格式两个核心门禁上都是 0。
- 影子相对原生的解决率比是 0.0，格式比也是 0.0；这不是“略差”，是“不能作为 PyroDash 的 SLM 使用”。
- 教师可解 hard 比例 73% 说明“如果影子能发出控制 token，云端至少能救回 73% 的失败样本”；问题在影子侧，不在 handoff 上限。

### 门禁偏差（必须记录）

- 计划中的对照臂门禁是“严格 `####` 格式 ≥90%”；实测 4B-Instruct 是 79.0%。交叉表显示：73 个正确答案 73/73 带 `####`，21 个未带 `####` 的全部是错答。0.90 格式门禁会把一个准确率 73% 的对照臂误判为“评测口径坏了”。因此对照臂改为“准确率 ≥50% 且没有正确答案因格式丢失”，严格格式率只作诊断。
- 产品臂的格式门禁仍是 ≥80%，没有放宽；影子 0.0% 的失败与这个偏差无关。
- 词级 distinct-3 是门禁指标；字符级 distinct-3 只作诊断，因为数学/LaTeX 输出会合法地重复字符（4B-Instruct 字符级只有 0.486，但输出并不退化）。

## 解释与决定

1. **影子不能做 PyroDash 的 SLM。** PyroDash 的控制 token 必须由 SLM 在能力边界自己发出；这要求 SLM 至少能生成可判分的推理轨迹。exp7 的导出影子在 GSM8K 上 0/100、0 条 `####`、7% 截断，且在产线 QA 上大量复读。让它学 `<|offload|>` 只会得到“遇到困难就 handoff”或“到处 handoff”，无法验证“能力边界”假设。
2. **不要在这个 checkpoint 上扩词表。** 扩词表、三阶段 SFT/GRPO 都会把成本投在一个 shadow-only 生成已经失败的模型上；即使 Stage 2 的断点重合率达标，也无法区分“学会了边界”与“学会了逃避生成”。
3. **机制本身仍可在有能力的 SLM 上测。** 原生 0.6B 45%、4B-Instruct 73%，说明控制 token 的 Stage 1–2 可以在这两个模型上做“机制验证”；但那是**非 ShadowPEFT** 的 PyroDash 实验，应另开规格/工单，不应伪装成“导出影子”路线。
4. **保留 issue #1 的 probe。** 在线 handoff 仍走 `docs/specs/token-level-edge-cloud-handoff.md` 的 probe/边界策略；本次没有改 Router、没有改任何生产默认值。

## 已知限制

- **单 checkpoint、单 seed、100 条 GSM8K**：影子只测了 exp7 E-arm 的 `ckpt-final`（0.5B token）；0.125B/0.25B/0.375B checkpoint 未测。原生臂的 95% CI 约 ±5 pp，对照臂约 ±8 pp；这不是完整 GSM8K 结论。
- **导出只在内存中完成**：Stage 0 用 `ShadowPeftModel.export_shadow()` 导出后直接生成，没有保存/重载导出模型或 tokenizer；导出 round-trip 是 Stage 1 的测试项。
- **产线 QA 是 smoke**：16 条、expected 写法单一、自由文本用关键词覆盖率，不能当能力排名；`quality_verdict` 的 1/5 主要反映标签歧义。
- **只测单轮、无云端续写**：Stage 0 不触发 handoff，不评估 partial trace 的云端续写质量；`mechanism_only` 只说明“值得在别的 SLM 上测”，不等于控制 token 一定能带来收益。
- **prompt 格式**：GSM8K 使用 system message + `####` 指令；base 模型不一定遵守，原生 0.6B 的 94% 格式率说明这个 prompt 对 0.6B 是可执行的。
- **没有 AutoDL**：全部在 `goooinext` 本地完成；没有租用云端实例。

## 产物与复现

- 脚本：`scripts/exp14_pyrodash_stage0.py`（只依赖仓库现有模块；重模型导入是懒加载）。
- 测试：`tests/lab/test_pyrodash_stage0.py`（GSM8K 解析/数值匹配、词级与字符级 distinct、RESTRAIN 判分、多步 DPPM 取最后一个数字、汇总、门禁树、`--report-from --regrade`）。
- 数据：`data/pyrodash_stage0/production_qa.jsonl`（16 条固定产线 QA）。
- 运行产物（`goooinext:~/Code/shadow-peft/runs/exp14/stage0-full/`，`runs/` 已被 git 忽略）：
  - `report.json`：配置、逐臂汇总、逐任务、门禁、判定；
  - `report.md`：同口径简表；
  - `summary.csv`：长表；
  - `generations.jsonl`：3 臂 × 116 条逐 prompt 输出；
  - `prompts.jsonl`：116 条输入与 gold。
- 复现：先按上表同步脚本/数据，再执行“命令”一节的两条命令；`--report-from --regrade` 可在不重新加载模型的情况下重算报告。

## 后续（2026-10-21 决定）

- **issue #41**：已以本报告关闭。
- **issue #6**：按 no-go 关闭；控制 token 路线停止，不转向“原生 0.6B/4B-Instruct 的机制验证”。如果未来要做机制验证，另开规格，不在本路线内继续。
- **issue #42/#43/#44**：以 `wontfix` 关闭；Stage 0 的“影子可用”前置条件未满足。
- **issue #1 probe**：继续作为在线 handoff 的默认路线；本次没有改它的行为。
