# exp10 GKD on-policy 飞轮 smoke（spec #2）

一句话收束：**在 `goooinext` 上用 Qwen3-0.6B 学生（新建隐式 shadow adapter）+ Qwen3-4B teacher 跑通了 GKD 全链路：16 条 base_shadow rollout、2 步 OnPolicyGKD 训练、adapter 保存/重载、joint/shadow ppl 与结构化任务报告；但 2 步 smoke 上 on-policy（λ=1, JSD）joint ppl 38.736 略差于固定标签 SFT（λ=0）38.221，任务准确率 0，不能宣称收益。λ=0.5 与 λ=0 的 joint ppl 完全相同，作为未解异常记录。**

状态：**smoke 已跑通（2026-10-20）**；结论级实验（1k prompts、更长训练、λ/divergence 网格、同 token 预算 SFT 对照）未跑。

## 实验设置

| 项 | 值 |
|---|---|
| 机器 | `goooinext`（`aitopatom-dc4a`，192.168.1.177，GB10） |
| 接入 | `ssh -J gooo goooinext` |
| 代码 | `feat/onpolicy-gkd` @ `6ede07a`（#17–#21 + 两个 CLI 修复） |
| 学生 | `~/models/Qwen3-0.6B` + 新建隐式 shadow adapter（1 层，injection 16 / gate 8 / α=0.1 / dropout=0.1） |
| Teacher | `~/models/Qwen3-4B`（同词表，冻结） |
| Rollout | `base_shadow`，G=2，`max_new_tokens=32`，温度 1.0，top-p 1.0 |
| 训练 | 2 optimizer steps，batch=2，max_seq_len=256，lr=1e-4，λ∈{0, 0.5, 1}，JSD |
| 评测 | heldout 2×512-token 窗口；2 条结构化任务（verdict / DPPM） |
| 数据 | 8 条产线 prompt；heldout 为 KB 云端答案拼接；SFT 对照用 KB 云端答案构造 messages |

## 命令

```bash
# 1) 新建学生 adapter
PYTHONPATH=src .venv/bin/python - <<'PY'
import torch
from transformers import AutoModelForCausalLM
from shadow_peft import ShadowConfig, get_shadow_model
base = AutoModelForCausalLM.from_pretrained("/home/goooice/models/Qwen3-0.6B", dtype=torch.bfloat16)
cfg = ShadowConfig(num_shadow_layers=1, injection_hidden_size=16, gate_hidden_size=8,
                   alpha=0.1, dropout=0.1)
get_shadow_model(base, cfg).save_pretrained("runs/exp10/smoke/fresh-adapter")
PY

# 2) 三次 run：λ=1 纯 on-policy；λ=0 固定标签 SFT；λ=0.5 混合
#    λ=1（无 --supervised）
PYTHONPATH=src .venv/bin/python scripts/exp10_onpolicy_gkd.py run \
  --prompts runs/exp10/smoke/prompts.jsonl --out runs/exp10/smoke/run \
  --model /home/goooice/models/Qwen3-0.6B \
  --shadow-ckpt runs/exp10/smoke/fresh-adapter \
  --teacher /home/goooice/models/Qwen3-4B \
  --heldout runs/exp10/smoke/heldout.txt --tasks runs/exp10/smoke/tasks.jsonl \
  --lambda 1.0 --divergence jsd --rollout-mode base_shadow \
  --g 2 --max-new-tokens 32 --max-steps 2 --batch-size 2 \
  --max-seq-len 256 --seq-len 512 --max-windows 2 \
  --seed 7 --limit 8 --device cuda --dtype bfloat16 \
  --teacher-support full --teacher-version Qwen3-4B

#    λ=0（--supervised，固定标签 SFT 对照）
PYTHONPATH=src .venv/bin/python scripts/exp10_onpolicy_gkd.py run \
  --prompts runs/exp10/smoke/prompts.jsonl --out runs/exp10/smoke/run_sft \
  --model /home/goooice/models/Qwen3-0.6B \
  --shadow-ckpt runs/exp10/smoke/fresh-adapter \
  --teacher /home/goooice/models/Qwen3-4B \
  --heldout runs/exp10/smoke/heldout.txt --tasks runs/exp10/smoke/tasks.jsonl \
  --supervised runs/exp10/smoke/supervised.jsonl \
  --lambda 0.0 --divergence jsd --rollout-mode base_shadow \
  --g 2 --max-new-tokens 32 --max-steps 2 --batch-size 2 \
  --max-seq-len 256 --seq-len 512 --max-windows 2 \
  --seed 7 --limit 8 --device cuda --dtype bfloat16 \
  --teacher-support full --teacher-version Qwen3-4B

#    λ=0.5（--supervised + baseline 对照）
PYTHONPATH=src .venv/bin/python scripts/exp10_onpolicy_gkd.py run \
  --prompts runs/exp10/smoke/prompts.jsonl --out runs/exp10/smoke/run_mix \
  --model /home/goooice/models/Qwen3-0.6B \
  --shadow-ckpt runs/exp10/smoke/fresh-adapter \
  --teacher /home/goooice/models/Qwen3-4B \
  --heldout runs/exp10/smoke/heldout.txt --tasks runs/exp10/smoke/tasks.jsonl \
  --supervised runs/exp10/smoke/supervised.jsonl \
  --baseline-sft runs/exp10/smoke/run_sft/report.json \
  --lambda 0.5 --divergence jsd --rollout-mode base_shadow \
  --g 2 --max-new-tokens 32 --max-steps 2 --batch-size 2 \
  --max-seq-len 256 --seq-len 512 --max-windows 2 \
  --seed 7 --limit 8 --device cuda --dtype bfloat16 \
  --teacher-support full --teacher-version Qwen3-4B
```

产物目录：`runs-local/exp10/smoke/{run,run_sft,run_mix}/`（本机），`runs/exp10/smoke/`（goooinext）。

## 结果

### Rollout

| 指标 | 值 |
|---|---|
| prompt | 8 |
| G | 2 |
| 保留轨迹 | 16 |
| 丢弃 | 0（empty / immediate EOS / too long 均为 0） |
| rollout_mode | `base_shadow` |
| wall clock | 9.2s |

### 训练与评测

| λ | divergence | train loss | joint ppl | shadow ppl | task acc | tokens_seen | no_go |
|---|---|---|---|---|---|---|---|
| 0.0 | —（纯 SFT） | 3.407 → 0 | **38.221** | 285524.164 | 0.0 | 1095 | false |
| 0.5 | JSD | 1.755 → 0.0805 | **38.221** | 285524.164 | 0.0 | 1095 | false |
| 1.0 | JSD | 0.191 → 0.167 | **38.736** | 285524.164 | 0.0 | 448 | false |

- 训练各自 2 步，adapter 均保存到 `<out>/adapter` 并被 eval 重新加载（三个 adapter 的 sha256 不同）。
- `shadow_ppl` 三组完全相同：joint KD 默认只训练注入/更新路径，shadow-only 路径未更新（`shadow_aux_weight=0`），符合 spec「shadow-only 仅作辅助/诊断」的设计。
- λ=0.5 与 λ=0 的 `joint_ppl` 精确相同，作为**未解异常**保留：可能是 2 步训练对 joint 路径的改动不足以改变 ppl，也可能是 adapter 保存/加载有未覆盖的参数；需要更长训练与参数 diff 复核，不能据此下结论。
- `task_accuracy=0`：Qwen3-0.6B 没有按 `quality_verdict`/`process_calc` 的期望格式回答；不代表训练失败，只说明 smoke 的评测任务太严格/学生太小。

### 耗时

| 阶段 | λ=1 | λ=0 | λ=0.5 |
|---|---|---|---|
| 总 wall clock | 1m18s | 1m35s | 1m21s |
| rollout | 9.2s | 9.2s | 9.2s |
| train | 0.65s | 0.45s | 0.68s |
| eval | 11.96s | 11.53s | 11.41s |

包含模型加载；训练本身在 4B teacher 前向下仅 0.5–0.7s（2 步）。

## 首轮验收口径

| 口径 | 结论 | 证据 |
|---|---|---|
| 同 token 预算下 joint ppl 不劣于固定标签 SFT，且任务准确率 +2pp | **未通过** | λ=1 joint 38.736 > λ=0 38.221；准确率 0 |
| shadow-only ppl −3% 且 joint 不退化 | **未通过** | shadow ppl 三组相同；joint 未改善 |
| 至少一个 λ/divergence 组合稳定优于 baseline | **未验证** | 只跑 JSD、2 步；λ=0.5 异常待查 |
| 训练产物能通过现有 adapter 加载/灰度 | **通过** | 三个 adapter 均保存且被 eval 重新加载 |
| 报告字段/数据指纹可审计 | **通过** | `report.json` 含 joint/shadow ppl、λ、divergence、teacher_support、tokens_seen、discard_rate、data_fingerprint、baseline_sft、no_go |

## 已知限制

- 8 条 prompt、2 步训练、0.6B 学生 + 全新隐式 shadow：只能证明管线，不可外推到 4B+H4B 或 1k prompt。
- λ=0 的 `tokens_seen=1095` 与 λ=1 的 448 不同（SFT 的 messages 更长），不是严格的同 token 预算对照；需要按 token 对齐重跑。
- 未接真实 `FeedbackStore` 难例（用了手工 prompt JSONL）；`export_prompts` 路径只在单元测试里覆盖。
- 未跑 λ/divergence 2×2、未跑 bootstrap CI、未做真实模型的人工抽检。
- λ=0.5 与 λ=0 的 joint ppl 完全相同，未解；下一轮必须加参数 diff 与更长训练确认。

## 后续

1. 按 spec 建议扩到 1k prompts、G=4–8、512 token、0.25B 训练 token；同 token 预算 SFT 对照。
2. 学生换成 exp7 `Qwen3-4B + H4B`，teacher 用 Qwen3-4B/8B，跑 λ/divergence 2×2。
3. 用真实 `FeedbackStore` 难例导出 prompt 集；结构化任务加人工抽检。
4. 先查清 λ=0.5/λ=0 joint ppl 相同的异常，再谈收益。

## 冒烟中发现并修复的 CLI bug

- `501d326`：`_token_ids` 用 `isinstance(encoded, dict)`，transformers 的 `BatchEncoding` 是 UserDict，被当 key 迭代 → `int('input_ids')`；改为 `collections.abc.Mapping`。
- `6ede07a`：`cmd_eval` 把 `pending`（字符串列表）当 task dict 赋值；改为对 `pending_records` 赋值并加 `_fill_task_responses` 测试。
