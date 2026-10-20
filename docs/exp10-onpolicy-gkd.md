# exp10 GKD on-policy 飞轮 smoke（spec #2）

一句话收束：**在 `goooinext` 上用 Qwen3-0.6B 学生（新建隐式 shadow adapter）+ Qwen3-4B teacher 跑通了 GKD 全链路：16 条 base_shadow rollout、2 步 OnPolicyGKD 训练、adapter 保存/重载、joint/shadow ppl + bootstrap CI + 结构化任务报告。按 `--token-budget 448` 对齐后，λ=0（纯 SFT）joint ppl 38.221、λ=1（纯 on-policy JSD）38.736、λ=0.5（混训 302 supervised + 128 on-policy）38.221；三者 CI 大幅重叠（[35.3, 61.9] vs [35.8, 61.9]），任务准确率均为 0，2 步 smoke 没有信号，不能宣称收益。**

状态：**smoke 管线已跑通（2026-10-20）；2 步 smoke 无信号/未验证**；结论级实验（1k prompts、更长训练、λ/divergence 网格、真实 FeedbackStore）未跑。`--token-budget 448` 的 λ=0/0.5/1 已按 #23 新口径重跑，数字见下表。

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

## 同 token 预算重跑（待重跑）

`#23` 新增 `--token-budget`：按**实际参与 loss** 的 token 计数（λ=0 只算 supervised，λ=1 只算 onpolicy，0<λ<1 两者都算），保留完整行，累计到预算为止（至少一行）。0<λ<1 且两种 source 都存在时，选行按 supervised/onpolicy **交替（round-robin）**，并优先各保留一行，避免先出现的分支吃满预算；若这两行本身就超预算，允许略超并记录 `token_budget_overflow=true`。λ=0 基线与 λ=1 应使用同一个 448 token 预算：

```bash
# λ=0 固定标签 SFT 基线，与 λ=1 的 448 loss-token 预算对齐
PYTHONPATH=src .venv/bin/python scripts/exp10_onpolicy_gkd.py run \
  --prompts runs/exp10/smoke/prompts.jsonl --out runs/exp10/smoke/run_sft_448 \
  --model /home/goooice/models/Qwen3-0.6B \
  --shadow-ckpt runs/exp10/smoke/fresh-adapter \
  --teacher /home/goooice/models/Qwen3-4B \
  --heldout runs/exp10/smoke/heldout.txt --tasks runs/exp10/smoke/tasks.jsonl \
  --supervised runs/exp10/smoke/supervised.jsonl \
  --lambda 0.0 --divergence jsd --token-budget 448 \
  --g 2 --max-new-tokens 32 --max-steps 2 --batch-size 2 \
  --max-seq-len 256 --seq-len 512 --max-windows 2 \
  --seed 7 --limit 8 --device cuda --dtype bfloat16 \
  --teacher-support full --teacher-version Qwen3-4B
```

**本档案未执行该重跑**；下表中的 λ=0 数字仍是旧口径（1095 tokens，含未参与 loss 的分支），不能与 λ=1 的 448 直接比较。重跑后应同时更新 `tokens_seen`、`tokens_supervised`/`tokens_onpolicy`、`token_budget`/`token_budget_overflow`、`joint_ppl_ci`/`shadow_ppl_ci`/`task_accuracy_ci` 与 `claims_gain`。

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

### 训练与评测（`--token-budget 448` 同预算）

| λ | divergence | train loss | joint ppl（95% CI） | shadow ppl（95% CI） | task acc | tokens（sup/onp） | no_go | claims_gain |
|---|---|---|---|---|---|---|---|---|
| 0.0 | —（纯 SFT） | 3.486 → 2.539 | **38.221** [35.251, 61.868] | 285524.164 [196320, 304066] | 0.0 [0, 0] | 384（384/0） | false | false |
| 0.5 | JSD | 1.778 → 1.829 | **38.221** [35.251, 61.868] | 285524.164 [196320, 304066] | 0.0 [0, 0] | 430（302/128） | true | false |
| 1.0 | JSD | 0.191 → 0.167 | **38.736** [35.806, 61.868] | 285524.164 [196320, 304066] | 0.0 [0, 0] | 448（0/448） | true | false |

- `tokens_seen` 只统计当前 λ 下**实际参与 loss** 的 token（#23 修正）；λ=0 取到 5 条完整 supervised 行共 384 token（再加一行会超 448）；λ=0.5 按 round-robin 保留 4 条 supervised + 4 条 on-policy，共 430 token；无 overflow。
- `no_go` 对 λ=0.5/1 为 true：相对 λ=0 baseline，任务准确率没有 +2pp（两者都是 0）；λ=1 的 joint ppl 还略高。`claims_gain` 全部 false。
- **joint ppl 差异在 2 步/bf16 分辨率内**：零初始化 `injection_ups` 经 2 步 Adam 只更新约 ±1.5e-4，低于 bf16 1 ULP；λ=0.5 与 λ=0 的 joint ppl 逐位相同，λ=1 的 38.736 也只差 0.5（量化噪声），不能解读为 on-policy 回归。
- **shadow-only 路径并非未更新**：adapter 的 `shadow_model`/`shadow_injection_model`/`shadow_update_model` 张量确实发生变化；三组 `shadow_ppl` 相同是 bf16 + 仅 2 步更新的分辨率问题。`shadow_aux_weight=0` 只说明 shadow-only CE 未进入目标函数，不影响 joint KD 通过注入/更新路径回传。
- `task_accuracy=0`：Qwen3-0.6B 没有按 `quality_verdict`/`process_calc` 的期望格式回答；不代表训练失败，只说明 smoke 的评测任务太严格/学生太小。
- 两个 ppl 的 CI 大幅重叠；bootstrap 200 次，seed 0，按窗口 token 加权。

### 耗时

| 阶段 | λ=1 | λ=0 | λ=0.5 |
|---|---|---|---|
| 总 wall clock | 1m20s | 1m36s | 1m38s |
| rollout | 9.2s | 9.2s | 9.2s |
| train | 0.63s | 0.53s | 0.64s |
| eval | 11.32s | 11.82s | 11.82s |

包含模型加载；训练本身在 4B teacher 前向下仅 0.5–0.7s（2 步）。

## 首轮验收口径

2 步 smoke 尺度下这些口径**没有信号**；下面的「未通过」只表示该 smoke 没达到目标，不代表方法在正常训练预算下无效。

| 口径 | 结论 | 证据 |
|---|---|---|
| 同 token 预算下 joint ppl 不劣于固定标签 SFT，且任务准确率 +2pp | **未通过** | λ=1 joint 38.736 > λ=0 38.221；task acc 0 vs 0，无 +2pp；CI 重叠 |
| shadow-only ppl −3% 且 joint 不退化 | **未通过** | shadow ppl 三组相同（bf16/2 步分辨率）；joint 未改善 |
| 至少一个 λ/divergence 组合稳定优于 baseline | **未验证** | 2 步/bf16 无信号；只跑 JSD，未跑 2×2 |
| 训练产物能通过现有 adapter 加载/灰度 | **通过** | 三个 adapter 均保存且被 eval 重新加载 |
| 报告字段/数据指纹/CI 可审计 | **通过** | `report.json` 含 joint/shadow ppl + CI、task acc + CI、λ、divergence、teacher_support、token split、discard_rate、data_fingerprint、baseline_sft、no_go、claims_gain、token_budget_overflow |
| no_go / claims_gain | **通过（保守）** | λ=0.5/1 相对 λ=0 均 `no_go=true`、`claims_gain=false`；λ=0 无 baseline 时二者为 false |

## 已知限制

- 8 条 prompt、2 步训练、0.6B 学生 + 全新隐式 shadow：只能证明管线，不可外推到 4B+H4B 或 1k prompt。
- 2 步 bf16 smoke 的 adapter 更新幅度低于 1 ULP，joint/shadow ppl 差异全部落在量化噪声内；**没有可解读的 on-policy 信号**。
- 同预算对照是近似的：λ=0 只取完整行，实际 384 token；λ=1 448；λ=0.5 430（302 supervised + 128 on-policy，round-robin，无 overflow）。要严格对齐需按 token 截断行或补行。
- 未接真实 `FeedbackStore` 难例（用了手工 prompt JSONL）；`export_prompts` 路径只在单元测试里覆盖。
- 未跑 λ/divergence 2×2、未做真实模型的人工抽检。
- 本档案数字来自修复后的 `feat/onpolicy-gkd` @ `2a9a041`（含 #23 的 token split、CI、no_go/claims_gain、teacher 词表校验、eval `source` 剥离、shadow LR 分组、checkpoint round-trip、长度分桶）。

## 后续

1. 扩到 1k prompts、G=4–8、512 token；严格按 token 预算对齐 SFT 与 on-policy。
2. 学生换成 exp7 `Qwen3-4B + H4B`，teacher 用 Qwen3-4B/8B，跑 λ/divergence 2×2；用 `joint_ppl_ci`/`shadow_ppl_ci`/`task_accuracy_ci` 判定。
3. 用真实 `FeedbackStore` 难例导出 prompt 集；结构化任务加人工抽检。
4. 训练步数增加后用参数 diff 与更长训练确认 joint 路径确有可测更新；在此之前不宣称 on-policy 收益，报告保持 `claims_gain=false`。

## 冒烟中发现并修复的 CLI bug

- `501d326`：`_token_ids` 用 `isinstance(encoded, dict)`，transformers 的 `BatchEncoding` 是 UserDict，被当 key 迭代 → `int('input_ids')`；改为 `collections.abc.Mapping`。
- `6ede07a`：`cmd_eval` 把 `pending`（字符串列表）当 task dict 赋值；改为对 `pending_records` 赋值并加 `_fill_task_responses` 测试。
