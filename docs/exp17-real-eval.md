# exp17 真实语料/真实模型复评 harness（issue #47）

一句话收束：**统一的 real-eval harness 已能在真实 GSM8K test 上跑 GKD/S2T、在 exp12 held-out 上跑 RESTRAIN，并在缺数据时对 JitRL 明确返回 `data_missing`；但本轮是 8/4 条样本、16 token、2-step/exp10 adapter 的冒烟，全部 `claims_gain=false`、`evidence=insufficient`，不能当作真实收益结论。完整 2–4 天复评未运行。**

状态：**MVP harness + goooinext 冒烟已跑通（2026-10-22）**；#2–#5 的完整真实复评（3 seeds、≥200 held-out、≥100 hard-case、真实反馈日志）未启动。

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

# S2T
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

## 结果

| suite | 数据/规模 | 关键指标 | 状态 |
|---|---|---|---|
| GKD | GSM8K test 8 条，1077 ppl token | joint ppl **5.567** [4.573, 7.145]；shadow ppl **167,148** [144,310, 206,846]；task accuracy **0/8** | `ok`，evidence insufficient |
| S2T | 4 prompts，64 token，K=8 | Hit@8 **0.922** [0.844, 0.969]；Agree@1 **0.500** [0.359, 0.610]；`selector_gain=null` | `ok`，evidence insufficient |
| RESTRAIN | exp12 held-out 8 条，16 rollouts | accuracy **0.0**；low consistency **1.0**；mean consistency 0.277；skip rate **0.875**；top1 share 0.667 | `ok`，evidence insufficient |
| JitRL | 无数据/记忆/后端 | — | `data_missing` |

所有报告：`claims_gain=false`、`evidence=insufficient`。S2T 的 `go_no_go=go` 是 S2T 自身的局部诊断（Hit@8 阈值），不代表本轮 64 token 冒烟有统计意义。

## 解释与后续

1. **Harness 可用**：四个 suite 的统一入口、数据指纹、per-example、summary/report 都能跑；缺数据时不会伪造结果。
2. **2-step/exp10 adapter 不是结论**：GKD 的 shadow ppl 167k、RESTRAIN 的 accuracy 0/low consistency 1.0，与 exp10 smoke/exp14 Stage 0 的 shadow-only 失败一致；S2T 的 0.922/0.5 只是 64 token 的局部诊断，不能外推。
3. **完整复评需要**：GKD 固定 token 预算 + 3 seeds；S2T ≥200 条 held-out + 64-hidden selector 重训；RESTRAIN ≥100 条结构化 hard-case；JitRL 真实反馈日志/记忆/n-best 后端。总计约 2–4 个 GB10 天 + 人工抽检，需要单独预算/审批。
4. **下一步**：优先用 #46 的 shadow-only SFT 恢复管线训练一个可用的 shadow，再跑 GKD/S2T 的真实复评；否则真实复评只会重复确认 shadow-only 不可用。

## 已知限制

- 样本极小（8/4/8）、生成预算 16 token，所有置信区间都很宽；不能做显著性结论。
- GKD/RESTRAIN 使用 2-step/exp12 adapter，不是完整训练模型；adapter 选择会影响结果。
- GKD 只用数值容差评最终答案，不校验推理过程；分数/小数可能误判。
- JitRL 的 cloud 升级是简化阈值，不是完整生产 Router；本轮没有真实记忆/后端，直接 `data_missing`。
- `claims_gain=false` 是设计使然：MVP harness 没有 baseline-vs-treatment 对照臂。
- 没有跑完整 2–4 天 campaign，因此不能声称 #2–#5 的真实收益或失败。

## 产物

- 模块：`src/shadow_lab/real_eval.py`
- 脚本：`scripts/exp17_real_eval.py`
- 测试：`tests/lab/test_real_eval.py`（23 个）
- 冒烟产物：`runs-local/exp17/{gkd,restrain,s2t,jitrl}/`（`report.json`、`summary.csv`、`per_example.jsonl`）
