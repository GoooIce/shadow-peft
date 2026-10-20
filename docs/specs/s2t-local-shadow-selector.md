---
status: ready-for-agent
triage: ready-for-agent
date: 2026-10-20
source: docs/research-pyrodash-collab.md
scope: 选择侧 P1-4（先诊断后训练）
---

# 规格：S2T-Local 影子选择器（先诊断后训练）

## Problem Statement

ShadowPEFT 的部署路径是 `base_shadow`，但 `shadow_only` 是一个更便宜、可以独立服务的路径。exp7 的证据显示 shadow 单独生成会从“保守复读”漂移成“流畅幻觉”，所以不能直接把 shadow-only 当成可靠的小模型。

S2T 的“局部充分性”给出了另一种可能：大模型偏好的 token 往往已经在小模型的 top-K 候选里，只是没有成为 top-1。1.5B 小模型的 top-8 对 32B teacher 的命中率是 95%，而它自己的 Hit@1 只有 30%。如果 ShadowPEFT 的 shadow top-K 也能覆盖 base 的 top-1，那么一个轻量 selector 就能在**不调用 base、不调用云端**的情况下把 shadow-only 拉近 base 的选择；如果覆盖不了，说明 shadow 只是一个 correction 网络，不是独立候选生成器，整条 S2T-Local 路线应当停止。

仓库目前没有这个诊断：`ShadowForCausalLM.base_shadow` 同时有 base 与 shadow 两路 logits，但没有脚本统计 Hit@K/Agree@1；也没有 selector head。更麻烦的是，`export_shadow` 会把 base 的 `lm_head` 深拷贝进导出模型，直接加 reserved token 会破坏词表/投影对齐，所以不能照搬 S2T 的 reserved-token 方案。

## Solution

分两阶段，先诊断后训练：

1. **诊断（不训练）**：在固定评测集上逐 token 记录 base top-1、shadow top-K、位置和 task，计算 Hit@K（K=1/4/8/16）与 Agree@1，并按任务/位置分层。给出明确的 go/no-go。
2. **条件训练**：只有在 Hit@8 达到门槛时，才训练一个**独立的 selector head**，对 shadow 的 top-K 候选重新打分；训练目标是“base 的 top-1 作为候选内标签”，加一个保持 shadow 原偏好的正则项。推理时 shadow-only，不跑 base；selector head 作为 adapter 的附加模块保存，`export_shadow` 与 base `lm_head` 保持不变。

这条路径不改变 ShadowPEFT 的注入算子，也不改变部署时的 `base_shadow` 路径；它只决定 shadow-only 是否值得作为独立服务形态继续投入。

## User Stories

1. 作为研究者，我想在不训练任何模型的情况下测出 shadow top-K 对 base top-1 的命中率，以便在投入训练前判断 S2T-Local 是否成立。
2. 作为研究者，我想按 task（故障诊断、工艺计算、质检判定、标准问答）分层看 Hit@K，以便发现某些任务适合选择器、某些不适合。
3. 作为研究者，我想按 token 位置分层看命中率，以便判断命中率是否只在句首/格式 token 上虚高。
4. 作为研究者，我想看到 Agree@1（shadow top-1 与 base top-1 的一致率），以便把“候选覆盖”和“选择错误”分开。
5. 作为研究者，我想看到 base top-1 在 shadow 分布中的 rank 分布，以便知道 K=8 是否足够。
6. 作为项目负责人，我想在 Hit@8 < 70% 时得到明确的 no-go 结论，以便及时止损，转回 GKD 训练而不是继续加 selector。
7. 作为项目负责人，我想在 Hit@8 ≥ 85% 时得到 go 结论，以便安排 selector 训练。
8. 作为训练工程师，我想用 base 的 top-1 作为候选内标签训练 selector，以便把大模型的选择能力蒸馏进 shadow-only。
9. 作为训练工程师，我想只在“base top-1 确实在候选集里”的位置计算选择损失，以便不把 hit failure 错误地算成 selection failure。
10. 作为训练工程师，我想加一个保持 shadow 原偏好的正则项，以便 selector 不会在没有 base 信号的普通位置过度改写。
11. 作为训练工程师，我想让 selector head 独立于 `shadow_lm_head` 和 base `lm_head`，以便不破坏导出模型的词表与投影对齐。
12. 作为训练工程师，我想把 selector head 作为 adapter 的附加模块保存，以便 `export_shadow` 的现有语义不变。
13. 作为训练工程师，我想用 K=8 的固定候选集，以便与 S2T 的固定 K 结论对齐；dynamic-K 只作为消融。
14. 作为评估工程师，我想看到 shadow-only greedy 与 shadow-only + selector 的准确率差，以便判断选择器是否真的有用。
15. 作为评估工程师，我想看到每 token 的额外延迟，以便判断 selector 是否值得部署。
16. 作为评估工程师，我想把结果与 exp8 的 logit ensemble β=0.7 对比，以便明确“shadow-only + selector”与“必须让大模型在场”的质量/成本差异。
17. 作为评估工程师，我想看到 hit failure 与 selection failure 的分解，以便知道下一步该扩 K、改表征还是改选择器。
18. 作为运维人员，我想在没有 selector head 或候选集为空时回退到 greedy，以便不阻塞现有 shadow-only 服务。
19. 作为安全与合规负责人，我想让 selector 只在 shadow 自己的候选集内重排，不引入词表外 token，以便输出仍然受 shadow 词表约束。
20. 作为后续规格的维护者，我想让本规格只负责 shadow-only 选择器，不改变 joint base_shadow 路径，以便与 handoff / GKD 规格解耦。

## Implementation Decisions

### 1. 诊断接口

- 诊断函数接收一个 `ShadowForCausalLM`（`base_shadow` 模式）、prompt 集和 max_new_tokens，逐 token 记录：
  - `base_top1`、`base_topk`（用于对照）
  - `shadow_topk_ids`、`shadow_topk_logprobs`
  - `base_top1_rank_in_shadow`（base top-1 在 shadow 分布中的 rank）
  - `position`、`task`、`prompt_id`
- 指标：`Hit@K`（base top-1 ∈ shadow top-K）、`Agree@1`（shadow top-1 == base top-1）、`rank` 的分位数。
- 注意 `shadow_logits` 来自 **initial** shadow hidden，不是注入后的 hidden；诊断报告中必须注明这一点，避免把“初始影子”误当成“注入后的部署路径”。
- 诊断只读，不训练；输出 JSON + CSV，字段与现有 `docs/exp*.md` 的单一事实源风格一致。

### 2. Go / no-go 门槛

- `Hit@8 ≥ 85%`：go，进入 selector 训练。
- `70% ≤ Hit@8 < 85%`：pilot，只在一个任务上训练 selector，先看 Agree@1 是否提升。
- `Hit@8 < 70%`：no-go，停止 S2T-Local，回到 GKD 规格（提升 shadow 表征）或接受 shadow-only 不作为独立服务。
- 门槛不是论文结论，是本项目的工程判据；报告中必须写明样本量、任务分布和置信区间。

### 3. Selector 结构

- 候选集 `C_t = shadow_topk(ids, K=8)`；不做 dynamic-K（作为消融）。
- 候选向量用 shadow 的 `shadow_lm_head.weight[c]`（或导出模型的 `lm_head.weight[c]`）；不要用 input embedding 代替输出词表向量。
- 打分函数：共享小 MLP `score(c) = MLP([h_t ; e_c])`，输出标量；参数量控制在几十万级。
- `h_t` 用 shadow hidden（selector 训练时可用 `forward_with_shadow` 的 initial hidden；推理时用 shadow-only 的 hidden）。
- 不做 reserved token，不改 `shadow_lm_head` / base `lm_head` 的形状。

### 4. 训练目标

- `L_sel`：候选内交叉熵，target = base top-1；只在 `base_top1 ∈ C_t` 的位置计算。
- `L_reg`：候选内交叉熵，target = shadow top-1；在所有非 padding 位置计算，权重 β 默认 0.1。目的是让 selector 在没有 base 信号时保持 shadow 原偏好。
- 总损失：`L = L_sel + β·L_reg`。
- 训练数据只来自诊断集/训练集，按 prompt 分组切分；验证集按 prompt 隔离。
- 如果某个 task 的候选覆盖率为 0，跳过该 task 并在报告中标记。

### 5. 推理与导出

- `shadow_only` 推理：每步先算 shadow logits，取 top-K；有 selector head 时用 selector 打分并 argmax；没有 head 或候选为空时回退 greedy。
- selector head 作为 adapter 的附加模块保存（`modules_to_save` 或独立 `selector.safetensors`）；`export_shadow` 仍然只导出 shadow backbone + 原有 lm_head，不改变其签名。
- 提供一个轻量推理包装器，把 `shadow_only` + selector 组合成可服务的模型；不改变 `ShadowForCausalLM` 的默认行为。
- 评估时必须断言：shadow-only + selector 的推理过程中没有 base forward（用计数器或 mock 验证）。

### 6. 评估与对照

- 主指标：shadow-only 任务准确率、Agree@1、Hit@8、每 token 延迟。
- 对照：shadow-only greedy；shadow-only + selector；`base_shadow` 上界；exp8 的 logit ensemble β=0.7（ppl 8.238，需 base 在场）。
- 报告 hit failure / selection failure 分解：`δ_hit`（base top-1 不在候选）与 `δ_sel`（在候选但选错）。
- 任何 selector 收益都必须同时报告延迟与显存；如果延迟增加 >20%，默认不部署。

### 7. 成本

- 2k 条轨迹 × 512 token ≈ 1M token；base+shadow 前向约 15–30 分钟；selector 训练 1–2 小时；评测 1 小时；合计约 1 个 GB10 天。
- 诊断本身可在 CPU/单卡完成；如果 Hit@8 不达标，总成本小于半天。

## Testing Decisions

好的测试只断言外部行为：给定合成 logits 或 tiny 模型，输出确定的 Hit@K/Agree@1/selector 选择；不直接断言私有缓存或内部调用顺序。

### 诊断缝

- 合成 logits：构造 base top-1 在/不在 shadow top-K 的例子，断言 Hit@K 与 rank 计算正确。
- K=1 时 Hit@1 等于 Agree@1（当 base 与 shadow 分布同一 top-1 时）。
- 空候选/全 -inf logits 不产生 NaN。
- 按 task/position 分层的计数与总数一致。

### Selector 缝

- 候选内 CE：target 在候选内时 loss 有限且随正确 logit 增大而下降。
- `L_sel` 只对 hit 位置生效：把 target 移出候选集后，该位置不贡献 loss。
- `L_reg` 在 target=shadow top-1 时，把 selector 拉回 shadow 偏好；β=0 时退化为纯 `L_sel`。
- selector 输出只在候选集内 argmax，不会产生词表外 token。
- 固定 seed 下训练与推理确定。

### 导出与推理缝

- 加载/保存 selector head 后，打分逐位一致。
- `export_shadow` 的 lm_head/vocab 未改变；导出模型仍可按原方式加载。
- shadow-only + selector 推理不调用 base（计数器为 0）。
- 没有 selector head 时回退 greedy，输出与 baseline 一致。

### 现有先例

- `tests/test_shadow_peft.py`：adapter 保存/加载与推理模式。
- `tests/test_distill.py`：logits/loss 对齐。
- `tests/lab/test_solo_own_head.py`：独立 head 的训练/评估写法。
- `tests/lab/test_specdec.py`：逐 token 生成与接受/拒绝的测试写法，可参考确定性构造。

## Out of Scope

- **云端在线 selector**：S2T 的协作版需要云端对候选打分，本规格只做 S2T-Local，不引入在线云端调用。
- **reserved-token 方案**：`export_shadow` 共享 base `lm_head`，首版不改词表；reserved token 只在独立词表的导出模型上另立规格。
- **改善 joint `base_shadow` 路径**：selector 只作用于 shadow-only；joint 路径的改进属于 GKD/注入规格。
- **改变 `shadow_lm_head` / base `lm_head` 的形状或权重共享方式**。
- **多模态/工具调用**：首版只做文本 next-token 选择。
- **生产灰度和在线 A/B**：本规格只产出离线诊断与训练产物。

## Further Notes

### 与已有研究的关系

本规格是 [PyroDash 等五篇论文 × ShadowPEFT 机制桥接](../research-pyrodash-collab.md) 中 P1-4 的落地版本。S2T 的关键事实（local sufficiency、top-K 选择、hit/selection 误差分解）以该文档引用为准；本规格只引用结论。

### 为什么先诊断

ShadowPEFT 的 shadow 是 correction 网络，不是独立小模型；它的 top-K 未必覆盖 base 的 top-1。S2T 论文的 95% 命中率来自独立的 1.5B SLM，不能直接外推到 shadow。诊断成本小于半天，却能否掉整条路线，必须先做。

### 已知风险

- `shadow_logits` 来自 initial shadow hidden；如果注入后的 hidden 分布不同，诊断可能高估或低估覆盖率。报告中要标注，并在第二阶段用 `forward_with_shadow` 复核。
- selector 只改善 shadow-only，不影响 joint 路径；如果产品最终只部署 `base_shadow`，这条线的价值有限。
- K=8 的并行打分在长序列上有额外延迟；先在 1% 触发点/短序列上验证。

### 建议的首轮验收口径

- 诊断：给出 Hit@8 与 Agree@1 的点估计 + bootstrap 置信区间；明确 go/no-go。
- 若 go：shadow-only 任务准确率相对 greedy +1 pp，Agree@1 ≥50%，每 token 延迟增加 <20%。
- 若 no-go：记录原因（shadow 候选不覆盖 base），并把资源转回 GKD 规格；不要用更大的 K 硬救。
