# exp11 S2T-Local selector 训练与评估（spec #3）

一句话收束：**诊断 go（Hit@8=0.898）后，用 exp7 `Qwen3-4B + H4B` 训练了 K=8 的 selector head；但在 14 条 held-out prompt（224 token）上，shadow-only Agree@1 从 greedy 的 0.482 降到 selector 的 0.353，hit 位置的选择准确率从 0.671 降到 0.46，10/50 epoch 都一样。按 spec 的验收口径，selector 侧判 no-go：不部署，不用更大 K 硬救；S2T-Local 的候选覆盖成立，但当前小预算 selector 没有把覆盖转成正确选择。**

状态：**selector 训练/评估已结题（2026-10-20），结论 no-go（不部署）**；诊断 go 保留，后续回到 #2 GKD/表征改进，而不是继续调 K。

## 实验设置

| 项 | 值 |
|---|---|
| 机器 | `goooinext`（aitopatom-dc4a，192.168.1.177，GB10） |
| 代码 | `feat/s2t-local` @ `24dca92`（#24–#28 合并后） |
| 主干/影子 | `Qwen3-4B` + exp7 `runs/exp7/e4b-seed0/ckpt-final` + `Qwen3-0.6B-H4B` |
| 候选 | K=8；`h_t` = initial shadow hidden；`e_c` = `shadow_lm_head.weight[c]` |
| Selector | `MLP([h_t; e_c])`，5120→256→1，约 1.3M 参数 |
| 训练 | 42 条 prompt（7 task × 6），`max_new_tokens=8`，336 个 token 样本；按 prompt 切 34/8（272/64）；β=0.1；Adam |
| 评测 | 14 条 held-out prompt（7 task × 2），`max_new_tokens=16`，224 token；greedy vs selector；base 参考用 `base_shadow` |
| 诊断 | 56 prompt × 32 token = 1792 token，Hit@8=0.898，Agree@1=0.721（见 `docs/exp11-s2t-diagnostic.md`） |

## 命令

```bash
# 1) 收集 selector 样本（42 train prompts，8 token）
PYTHONPATH=src .venv/bin/python - <<'PY'
import json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from shadow_peft import ShadowForCausalLM, AutoModelForCausalLMWithHiddenProjection
from shadow_lab.s2t_selector import collect_selector_examples
base="/home/goooice/models/Qwen3-4B"
tok=AutoTokenizer.from_pretrained(base)
raw=AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16)
shadow=AutoModelForCausalLMWithHiddenProjection.from_pretrained("/home/goooice/models/Qwen3-0.6B-H4B")
model=ShadowForCausalLM.from_pretrained(raw, "runs/exp7/e4b-seed0/ckpt-final",
    shadow_model=shadow, inference_mode="base_shadow").to("cuda").eval()
prompts=[json.loads(l) for l in open("runs/exp11/prompts_large.jsonl")]
examples=collect_selector_examples(model, tok, prompts[:42], max_new_tokens=8, k=8, seed=0)
torch.save(examples, "runs/exp11/selector_run/examples.pt")
PY

# 2) 训练 selector（10 / 50 epoch 都试过）
#    见下方结果；最终产物 runs/exp11/selector（10ep）与 selector_50

# 3) held-out 评估
PYTHONPATH=src .venv/bin/python scripts/exp11_s2t_eval.py \
  --model /home/goooice/models/Qwen3-4B \
  --shadow-ckpt runs/exp7/e4b-seed0/ckpt-final \
  --shadow-model-path /home/goooice/models/Qwen3-0.6B-H4B \
  --prompts runs/exp11/eval_prompts.jsonl \
  --mode shadow_greedy --max-new-tokens 16 --k 8 \
  --out runs/exp11/eval_greedy --device cuda --dtype bfloat16 --use-cache
#    selector 模式加 --selector runs/exp11/selector（或 selector_50） --mode shadow_selector
```

产物：`runs-local/exp11/{eval_greedy,eval_selector,eval_selector50,selector,selector_50}/`。

## 结果

### 训练

| 配置 | val_loss | 说明 |
|---|---|---|
| 10 epoch，lr=1e-3 | 1.645 | 336 样本，272/64 train/val by prompt |
| 50 epoch，lr=5e-4 | 0.995 | 同一批样本；训练 loss 尾部 0.75–0.92 |

两者都保存了独立 selector（`selector.safetensors` + `selector_config.json`），未改 shadow backbone/lm_head。

### Held-out（14 prompt，224 token）

| 模式 | Agree@1 | hit 位置选择准确率 | δ_hit | δ_sel | 延迟/可部署 |
|---|---|---|---|---|---|
| shadow greedy | **0.482** | **0.671** | 0.281 | 0.237 | 基线 |
| selector 10 epoch | 0.353 | 0.462 | 0.237 | 0.411 | `deployable=True`，但 overhead 未记录 |
| selector 50 epoch | 0.353 | 0.454 | 0.223 | 0.424 | 同上 |

- δ_hit = base top-1 不在 shadow top-8（候选覆盖失败）；δ_sel = 在候选内但选错。greedy 的 δ_hit=0.281、δ_sel=0.237；selector 把 δ_sel 放大到 0.41–0.42，说明它没有学会选 base top-1，反而把原本能命中的位置选错。
- 50 epoch 的 val_loss 从 1.645 降到 0.995，但 held-out Agree 完全不变（0.3527），说明在当前特征/标签/数据规模下继续训练没有泛化收益。
- `selector_overhead` 在本轮报告里是 null（`--use-cache` 路径没有记录 latency），所以「延迟 <20%」**未验证**；`deployable=True` 只表示脚本没有判定超预算，不能当作延迟通过。

## 首轮验收口径

| 口径 | 结论 | 证据 |
|---|---|---|
| shadow-only 任务准确率 +1pp | **未验证** | held-out prompt 没有 expected 标签；本轮只测 Agree@1/选择准确率 |
| Agree@1 ≥50% | **未通过** | selector 0.353 < 0.50，且低于 greedy 0.482 |
| 每 token 延迟 <20% | **未验证** | 报告 overhead=null |
| hit/selection failure 分解 | **已报告** | δ_hit/δ_sel 见上表 |
| no-go 时不部署/不用更大 K | **执行** | 不部署 selector；不扩 K |

## 结论

- **诊断 go 成立**：shadow top-8 覆盖 base top-1 的 89.8%，候选覆盖不是瓶颈。
- **selector 在当前预算下 no-go**：336 个训练 token、7 个 task 混训、14 条 held-out prompt；selector 没有把覆盖转成正确选择，Agree@1 反而下降。按 spec，不部署、不扩 K。
- 这不是「S2T-Local 研究路线已死」的终审：训练样本只有 336 token，远低于 spec 建议的 1M token；失败点更像 selector 的泛化/特征对齐，而不是候选覆盖。若要继续，应先扩到 2k 轨迹 × 512 token，并检查 h_t/candidate 特征与标签对齐；在此之前不回主线。
- 下一步优先级回到 **#2 GKD**：提升 shadow 表征/候选质量，而不是继续调 selector 或 K。

## 已知限制

- held-out 只有 14 条 prompt（224 token），Agree@1 的置信区间很宽；不能据此给出精确的泛化估计。
- 没有 expected 标签，任务准确率未测；只有 Agree@1/选择准确率。
- 延迟字段为 null（`--use-cache` 路径未记录），部署延迟未验证。
- 只测了一个 checkpoint（exp7 e4b）、一个 K=8、一个 β=0.1。
- selector 只在 shadow 候选内重排；不引入词表外 token，export/lm_head 未改。
