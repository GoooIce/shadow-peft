# exp11 S2T-Local selector 训练与评估（spec #3）

一句话收束：**诊断在 base-prefix 上 go（Hit@8=0.898）后，用 exp7 `Qwen3-4B + H4B` 训练了 K=8 的 selector head；但在 14 条 held-out prompt（224 token）上，shadow-only Agree@1 从同 prefix greedy 的 0.473/0.513 降到 selector 的 0.353（`selector_gain` −0.121/−0.161），部署 prefix 的 Hit@8 只有 0.719–0.777。按 spec 的验收口径，selector 侧判 no-go：不部署，不用更大 K 硬救；S2T-Local 在部署 prefix 上覆盖（72–78%）与选择都失败。**

状态：**selector 训练/评估已结题（2026-10-20），结论 no-go（不部署）**；诊断 go 保留（仅 base-prefix），后续回到 #2 GKD/表征改进，而不是继续调 K。`mlp_hidden=64`、`val_ratio=0.2` 的重训与 held-out 结果已补入下表。

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
> `val_ratio=0.2` 一致。`mlp_hidden=64` 的最终重训产物是 `runs/exp11/selector_64`
> （`val_loss=1.431`），held-out 结果见上表；`base_upper` 产物是 `runs/exp11/eval_base_upper`。

产物：`runs-local/exp11/{eval_greedy,eval_selector,eval_selector50,eval_selector64,eval_base_upper,selector,selector_50,selector_64}/`；诊断 K=16 产物在 `runs-local/exp11/diagnostic_k16/`。

## 结果

### 训练

旧配置（`mlp_hidden=256`；`val_ratio=0.2`，34/8 切分）：

| 配置 | val_loss | 说明 |
|---|---|---|
| 10 epoch，lr=1e-3 | 1.645 | 336 样本，272/64 train/val by prompt |
| 50 epoch，lr=5e-4 | 0.995 | 同一批样本；训练 loss 尾部 0.75–0.92 |

修复后配置（`mlp_hidden=64`，`val_ratio=0.2`，50 epoch，lr=5e-4）：**val_loss=1.431**，参数量 **327,809**（5120→64→1）。产物 `runs-local/exp11/selector_64/`。

以上 selector 都独立保存（`selector.safetensors` + `selector_config.json`），未改 shadow backbone/lm_head。

### Held-out（14 prompt，224 token）

同 prefix 基线是 selector 运行中记录的 shadow top-1，用于隔离选择效果：

| 模式 | Agree@1 | 同 prefix greedy_agree@1 | selector_gain | hit@8 (1−δ_hit) | hit 位置选择准确率 | δ_hit | δ_sel | 延迟 overhead |
|---|---|---|---|---|---|---|---|---|
| shadow greedy（自身 prefix） | **0.482** | — | — | 0.719 | **0.671** | 0.281 | 0.237 | 基线 |
| selector 256-hidden, 10 epoch | 0.353 | 0.473 | **−0.121** | 0.763 | 0.462 | 0.237 | 0.411 | 0.1946（19.5%） |
| selector 256-hidden, 50 epoch | 0.353 | 0.513 | **−0.161** | 0.777 | 0.454 | 0.223 | 0.424 | 0.1878（18.8%） |
| **selector 64-hidden, 50 epoch** | **0.384** | **0.500** | **−0.116** | **0.786** | **0.489** | 0.214 | 0.402 | **0.1922（19.2%）** |
| base_upper 上界 | 未测 | — | — | — | — | — | — | 已跑通生成，但无 expected 标签 |

- `selector_gain = selector_agree_at_1 − 同 prefix greedy_agree_at_1`；64-hidden 的 0.384 vs 同 prefix greedy 0.500，增益 −0.116，仍然为负。
- δ_hit = base top-1 不在 shadow top-8；δ_sel = 在候选内但选错。64-hidden 把 δ_sel 从 greedy 的 0.237 放大到 0.402，hit 位置选择准确率 0.489 < greedy 0.671：覆盖与选择都失败。
- `base_upper` 模式已跑（`runs-local/exp11/eval_base_upper/`），但 held-out 没有 expected 标签，报告里 `agree_at_1=n/a`，只能确认 base 路径可生成；**质量上界不可测**，需要带 expected 的评测集。
- 延迟 overhead：64-hidden 0.1922（19.2%）、256-hidden 0.1946/0.1878；低于 0.20 预算但擦线，单次测量，不是稳定结论。`deployable=True` 只表示本次测量没超预算。
- exp8 β=0.7 是引用档案数字（ppl 8.238），本轮没有重跑。

## 首轮验收口径

| 口径 | 结论 | 证据 |
|---|---|---|
| shadow-only 任务准确率 +1pp | **未验证** | held-out prompt 没有 expected 标签；本轮只测 Agree@1/选择准确率 |
| Agree@1 ≥50% | **未通过** | 最好的 selector（64-hidden）0.384 < 0.50，且低于同 prefix greedy 0.500 |
| 每 token 延迟 <20% | **擦线通过（单次测量）** | 64-hidden overhead 0.1922 ≈ 19.2% |
| hit/selection failure 分解 | **已报告** | 部署 prefix Hit@8 0.72–0.79，δ_hit/δ_sel 见上表 |
| no-go 时不部署/不用更大 K | **执行** | 不部署 selector；不扩 K |

## 结论

- **诊断 go 仅限 base-prefix**：shadow top-8 覆盖 base top-1 的 89.6%（K=16 重跑；Hit@16 0.925）；但部署 prefix 上 eval Hit@8 只有 0.72–0.79，覆盖不是免费的。
- **selector 在当前预算下 no-go**：336 个训练 token、7 个 task 混训、14 条 held-out prompt；64-hidden selector 把 Agree 从 0.353 提到 0.384，但仍低于 greedy 0.482/同 prefix 0.500，选择准确率 0.489 < 0.671。按 spec，不部署、不扩 K。
- 这不是「S2T-Local 研究路线已死」的终审：训练样本只有 336 token，远低于 spec 建议的 1M token；失败点更像 selector 的泛化/特征对齐，而不是候选覆盖本身。若要继续，应先扩到 2k 轨迹 × 512 token，并检查 h_t/candidate 特征与标签对齐；在此之前不回主线。
- 下一步优先级回到 **#2 GKD**：提升 shadow 表征/候选质量，而不是继续调 selector 或 K。

## 已知限制

- held-out 只有 14 条 prompt（224 token），Agree@1 的置信区间很宽；不能据此给出精确的泛化估计。
- 没有 expected 标签，任务准确率未测；base_upper 也只验证了生成路径。
- 延迟 overhead 是单次测量（64-hidden 0.1922；256-hidden 0.1946/0.1878），刚好低于 20% 预算；没有重复计时。
- 只测了一个 checkpoint（exp7 e4b）、一个 K=8、一个 β=0.1；`mlp_hidden=64` 的结果已补入上表。
- selector 只在 shadow 候选内重排；不引入词表外 token，export/lm_head 未改。
- 诊断的 K=16 / 全量 rank 已重跑，见 `docs/exp11-s2t-diagnostic.md`；exp8 β=0.7 仅引用，不重跑。
