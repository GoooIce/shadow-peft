# exp16 shadow-only 恢复管线（issue #46）

一句话收束：**导出的 exp7 影子（`Qwen3-0.6B-H4B` + `runs/exp7/e4b-seed0/ckpt-final`）现在可以保存/重载并做 shadow-only 指令 SFT；round-trip 在修复 rotary 缓冲 dtype 后逐位通过（logits max diff 0.0），但 8 条 GSM8K、2 步 SFT 的冒烟仍格式 0%、准确率 0%、截断 100%。这只是管线可用，不是恢复成功；完整 3–5 天训练未运行。**

状态：**MVP 管线 + goooinext 冒烟已跑通（2026-10-22）**；完整 shadow-only SFT 恢复实验（约 3–5 个 GB10 天）未启动。

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
| 产物 | `runs-local/exp16/{report.json,diagnose_generations.jsonl,generations.jsonl,train_stats.json,model/}` |

### 命令

```bash
# 在 goooinext 的临时代码目录
PYTHONPATH=/home/goooice/p4647-run/src \
  /home/goooice/Code/shadow-peft/.venv/bin/python \
  /home/goooice/p4647-run/scripts/exp16_shadow_only.py run \
  --base-model /home/goooice/models/Qwen3-4B \
  --shadow-model /home/goooice/models/Qwen3-0.6B-H4B \
  --shadow-ckpt /home/goooice/Code/shadow-peft/runs/exp7/e4b-seed0/ckpt-final \
  --out /home/goooice/p4647-run/exp16-smoke2 \
  --smoke --device cuda --dtype bfloat16
```

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
- `recovery_ready=false`：round-trip 通过，但格式和截断门禁未过；`accuracy_gate=n/a`，因为还没有原生 0.6B 同口径基线。

## 解释与后续

1. **管线可用**：导出、保存、重载、tokenizer、SFT、生成评测都已跑通；round-trip 现在是逐位一致，不再是 bf16/fp32 RoPE 的假失败。
2. **2 步不是恢复**：0% 格式/0% 准确率与 exp14 Stage 0 的 0/100 一致；要判断 SFT 是否有效，需要完整训练（ticket 估计 3–5 个 GB10 天）并在 GSM8K test 上与原生 0.6B 同口径比较。
3. **下一步**：用 2k prompts × 3 epochs 的 shadow-only 指令/CoT SFT（`exp16 train`），再跑 `exp16 eval`；门禁仍是 shadow-only 格式 ≥80%、准确率 ≥原生 0.6B 的 70%、截断 ≤20%。完整训练需要单独预算/审批。

## 已知限制

- 只测一个 checkpoint（exp7 E-arm `ckpt-final`）和一个 seed；2 步冒烟不代表 SFT 的收敛行为。
- 训练默认 bf16、无 autocast/fp32 master weights；更长训练/更高 lr 可能需要调整。
- 训练数据是 GSM8K 完整 solution，模型可能先学格式而非推理；需要 test 集和原生 0.6B 基线。
- 评测只覆盖 GSM8K 数值题，不覆盖产线 QA 自由文本。
- 没有跑完整 3–5 天训练，因此不能声称 shadow-only 恢复成功或失败。

## 产物

- 模块：`src/shadow_lab/shadow_only.py`
- 脚本：`scripts/exp16_shadow_only.py`
- 测试：`tests/lab/test_shadow_only.py`（41 个）
- 冒烟产物：`runs-local/exp16/`（`report.json`、`diagnose_generations.jsonl`、`generations.jsonl`、`train_stats.json`、`model/`）
