# exp11 S2T-Local selector 训练与评估（spec #3）

一句话收束：**诊断在 base-prefix 上 go（Hit@8=0.898）后，用 exp7 `Qwen3-4B + H4B` 训练了 K=8 的 selector head；但在 14 条 held-out prompt（224 token）上，shadow-only Agree@1 从同 prefix greedy 的 0.473/0.513 降到 selector 的 0.353（`selector_gain` −0.121/−0.161），部署 prefix 的 Hit@8 只有 0.719–0.777。按 spec 的验收口径，selector 侧判 no-go：不部署，不用更大 K 硬救；S2T-Local 在部署 prefix 上覆盖（72–78%）与选择都失败。**

状态：**selector 训练/评估已结题（2026-10-20），结论 no-go（不部署）**；诊断 go 保留（仅 base-prefix），后续回到 #2 GKD/表征改进，而不是继续调 K。`mlp_hidden=64`、`val_ratio=0.2` 的修复重跑数字待协调者填充（见「待重跑」）。

## 实验设置

| 项 | 值 |
|---|---|
| 机器 | `goooinext`（aitopatom-dc4a，192.168.1.177，GB10） |
| 代码 | `feat/s2t-local`（#30 修复后重跑；旧结果来自 `24dca92`） |
| 主干/影子 | `Qwen3-4B` + exp7 `runs/exp7/e4b-seed0/ckpt-final` + `Qwen3-0.6B-H4B` |
| 候选 | K=8；`h_t` = initial shadow hidden；`e_c` = `shadow_lm_head.weight[c]` |
| Selector | `MLP([h_t; e_c])`，5120→64→1，约 327k 参数（#30 起默认 `mlp_hidden=64`；旧产物是 5120→256→1 ≈1.3M） |
| 训练 | 42 条 prompt（7 task × 6），`max_new_tokens=8`，336 个 token 样本；按 prompt 切 34/8（272/64）；`val_ratio=0.2`；β=0.1；Adam |
| 评测 | 14 条 held-out prompt（7 task × 2），`max_new_tokens=16`，224 token；greedy vs selector；base 参考用 `base_shadow` |
| 诊断 | 56 prompt × 32 token = 1792 token；base-prefix Hit@8=0.898，Agree@1=0.721（K=16 重跑见 `docs/exp11-s2t-diagnostic.md`） |
| exp8 对照 | 引用档案数字 ppl 8.238（β=0.7），本轮**未重跑**，需 base 在场 |

## 命令

```bash
# 1) 收集 selector 样本（42 train prompts，8 token）
PYTHONPATH=src .venv/bin/python - <<'PY'
import json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from shadow_peft import ShadowForCausalLM, AutoModelForCausalLMWithHiddenProjection
from shadow_lab.s2t_selector import collect_selector_examples

base = "/home/goooice/models/Qwen3-4B"
tok = AutoTokenizer.from_pretrained(base)
raw = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16)
shadow = AutoModelForCausalLMWithHiddenProjection.from_pretrained(
    "/home/goooice/models/Qwen3-0.6B-H4B"
)
model = ShadowForCausalLM.from_pretrained(
    raw, "runs/exp7/e4b-seed0/ckpt-final",
    shadow_model=shadow, inference_mode="base_shadow",
).to("cuda").eval()
prompts = [json.loads(line) for line in open("runs/exp11/prompts_large.jsonl")]
examples = collect_selector_examples(
    model, tok, prompts[:42], max_new_tokens=8, k=8, seed=0
)
torch.save(examples, "runs/exp11/selector_run/examples.pt")
torch.save(
    model.shadow_lm_head.weight.detach().cpu(),
    "runs/exp11/selector_run/shadow_lm_head.pt",
)
PY

# 2) 训练 selector（修复后配置：mlp_hidden=64, val_ratio=0.2；10 / 50 epoch 都跑）
PYTHONPATH=src .venv/bin/python - <<'PY'
import torch
from shadow_lab.s2t_selector import save_selector, train_selector

examples = torch.load("runs/exp11/selector_run/examples.pt", weights_only=False)
weight = torch.load("runs/exp11/selector_run/shadow_lm_head.pt", weights_only=True)
for epochs, lr, out in ((10, 1e-3, "selector"), (50, 5e-4, "selector_50")):
    result = train_selector(
        examples,
        candidate_weight=weight,
        epochs=epochs, lr=lr, beta=0.1, seed=0,
        device="cuda", batch_size=256, val_ratio=0.2, mlp_hidden=64,
    )
    print(out, result.val_loss, result.n_train_examples, result.n_val_examples)
    save_selector(result.head, f"runs/exp11/{out}", k=8)
PY

# 3) held-out 评估
PYTHONPATH=src .venv/bin/python scripts/exp11_s2t_eval.py \
  --model /home/goooice/models/Qwen3-4B \
  --shadow-ckpt runs/exp7/e4b-seed0/ckpt-final \
  --shadow-model-path /home/goooice/models/Qwen3-0.6B-H4B \
  --prompts runs/exp11/eval_prompts.jsonl \
  --mode shadow_greedy --max-new-tokens 16 --k 8 \
  --out runs/exp11/eval_greedy --device cuda --dtype bfloat16 --use-cache
#    selector 模式加 --selector runs/exp11/selector（或 selector_50） --mode shadow_selector
#    base_upper 对照加 --mode base_upper；exp8 不在本脚本内重跑
```

> 旧产物（`selector`、`selector_50`）用的是 `mlp_hidden=256`；34/8 的 prompt 切分与
> `val_ratio=0.2` 一致。#30 修复后必须用上面的 64/0.2 配置重训，结果填「待重跑」。

产物：`runs-local/exp11/{eval_greedy,eval_selector,eval_selector50,selector,selector_50}/`（重跑目录名可加 `_k16fix` 区分）。

## 结果

### 训练

旧配置（`mlp_hidden=256`；`val_ratio=0.2`，34/8 切分）：

| 配置 | val_loss | 说明 |
|---|---|---|
| 10 epoch，lr=1e-3 | 1.645 | 336 样本，272/64 train/val by prompt |
| 50 epoch，lr=5e-4 | 0.995 | 同一批样本；训练 loss 尾部 0.75–0.92 |

修复后配置（`mlp_hidden=64`, `val_ratio=0.2`）：val_loss 与训练曲线**待重跑**。

两者都保存了独立 selector（`selector.safetensors` + `selector_config.json`），未改 shadow backbone/lm_head。

### Held-out（14 prompt，224 token）

旧配置（256 hidden）的完整结果；同 prefix 基线是 selector 运行中记录的 shadow top-1，用于隔离选择效果：

| 模式 | Agree@1 | 同 prefix greedy_agree@1 | selector_gain | hit@8 (1−δ_hit) | hit 位置选择准确率 | δ_hit | δ_sel | 延迟 overhead |
|---|---|---|---|---|---|---|---|---|
| shadow greedy（自身 prefix） | **0.482** | — | — | 0.719 | **0.671** | 0.281 | 0.237 | 基线 |
| selector 10 epoch | 0.353 | 0.473 | **−0.121** | 0.763 | 0.462 | 0.237 | 0.411 | **0.1946（19.5%）** |
| selector 50 epoch | 0.353 | 0.513 | **−0.161** | 0.777 | 0.454 | 0.223 | 0.424 | **0.1878（18.8%）** |
| base_upper 上界 | 待重跑 | — | — | 待重跑 | — | — | — | 不适用（跑 base） |

- `selector_gain = selector_agree_at_1 − 同 prefix greedy_agree_at_1`；同 prefix greedy 分别是 0.473 / 0.513，selector 都是 0.353，所以增益为负 −0.121 / −0.161。不能用 0.482 做差（那是 greedy 自己在自己 prefix 上的 Agree@1）。
- δ_hit = base top-1 不在 shadow top-8（候选覆盖失败）；δ_sel = 在候选内但选错。部署 prefix 上 Hit@8 只有 0.719（greedy）/0.763/0.777（selector），selector 又把 δ_sel 放大到 0.41–0.42：覆盖与选择都失败，selector 没有把覆盖转成正确选择。
- 50 epoch 的 val_loss 从 1.645 降到 0.995，但 held-out Agree 完全不变（0.3527），说明在当前特征/标签/数据规模下继续训练没有泛化收益。
- 延迟 overhead 实际值是 0.1946 / 0.1878（约 19.5% / 18.8%），低于 0.20 预算但**擦线通过**；这是单次测量，没有重复计时或置信区间，不能当作稳定的延迟结论。`deployable=True` 仅表示这次测量没有超过 20% 预算。
- 修复后 `mlp_hidden=64`、`val_ratio=0.2` 的 selector、`base_upper` 上界与全部指标**待重跑**；重跑后直接替换本表数字，不得沿用旧配置的 256-hidden 结果作为最终值。

## 首轮验收口径

| 口径 | 结论 | 证据 |
|---|---|---|
| shadow-only 任务准确率 +1pp | **未验证** | held-out prompt 没有 expected 标签；本轮只测 Agree@1/选择准确率 |
| Agree@1 ≥50% | **未通过** | selector 0.353 < 0.50，且低于同 prefix greedy 0.473/0.513 |
| 每 token 延迟 <20% | **擦线通过（单次测量）** | overhead 0.1946 / 0.1878 ≈ 19.5% / 18.8% |
| hit/selection failure 分解 | **已报告** | 部署 prefix Hit@8 0.719–0.777，δ_hit/δ_sel 见上表 |
| no-go 时不部署/不用更大 K | **执行** | 不部署 selector；不扩 K |

## 结论

- **诊断 go 仅限 base-prefix**：shadow top-8 覆盖 base top-1 的 89.8%；但部署 prefix 上 eval Hit@8 只有 0.72–0.78，覆盖不是免费的。
- **selector 在当前预算下 no-go**：336 个训练 token、7 个 task 混训、14 条 held-out prompt；selector 没有把覆盖转成正确选择，Agree@1 反而下降（同 prefix `selector_gain` −0.121/−0.161）。按 spec，不部署、不扩 K。
- 这不是「S2T-Local 研究路线已死」的终审：训练样本只有 336 token，远低于 spec 建议的 1M token；失败点更像 selector 的泛化/特征对齐，而不是候选覆盖本身。若要继续，应先扩到 2k 轨迹 × 512 token，并检查 h_t/candidate 特征与标签对齐；在此之前不回主线。
- 下一步优先级回到 **#2 GKD**：提升 shadow 表征/候选质量，而不是继续调 selector 或 K。

## 已知限制

- held-out 只有 14 条 prompt（224 token），Agree@1 的置信区间很宽；不能据此给出精确的泛化估计。
- 没有 expected 标签，任务准确率未测；只有 Agree@1/选择准确率。
- 延迟 overhead 是单次测量（0.1946/0.1878），刚好低于 20% 预算；没有重复计时，结论只能算「擦线通过」。
- 只测了一个 checkpoint（exp7 e4b）、一个 K=8、一个 β=0.1；`mlp_hidden=64`、`val_ratio=0.2` 的重训结果待重跑。
- selector 只在 shadow 候选内重排；不引入词表外 token，export/lm_head 未改。

## 待重跑清单（#30 合并后）

1. 诊断：K=16 的 `hit_at_16`、`rank_all_p50/p90/p95/p99`、`n_rank_known_all`、5 个 position bucket 的 n/Hit@8。
2. selector：`mlp_hidden=64`、`val_ratio=0.2` 的 10/50 epoch 重训与 held-out 评估；`selector_gain`、`hit_at_8`、`latency_overhead` 用报告顶层字段。
3. `base_upper` 上界跑一次并填入上表。
4. exp8 β=0.7 仍是引用档案数字（ppl 8.238），本轮不重跑；如要对比需明确标注「引用」。
