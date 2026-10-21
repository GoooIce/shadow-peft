---
status: ready-for-human
triage: ready-for-human
date: 2026-10-20
source: docs/research-pyrodash-collab.md
scope: 训练+运行时 P2-8（高风控，需人工把关）
---

# 规格：PyroDash 控制 token（导出影子）

> 本规格标记为 `ready-for-human`：涉及扩展词表、三阶段训练、GRPO 与导出模型兼容性，默认不由 AFK agent 直接执行；每个阶段都需要人工检查点与 go/no-go。

## Problem Statement

token-level handoff 规格（issue #1）用一个外部 probe 判断“何时交接”。PyroDash 提供了另一种做法：把交接决策内化到小模型自己的生成轨迹里，让小模型学会在能力边界发出控制 token `τ_off`，Collaborate Engine 检测到后把 query + partial trace 单次交给冻结的大模型。它的好处是不需要独立 router、不需要大模型 logits、不需要重训大模型；代价是需要改词表、三阶段训练和成本感知的 GRPO。

ShadowPEFT 的约束让这条路更窄：

- 隐式 shadow 默认共享 base 的 `embed_tokens`，新增控制 token 的输入 embedding 必须改冻结 base，直接违反 ShadowPEFT 的冻结约束；只有**显式 shadow**（`AutoModelForCausalLMWithHiddenProjection`，自带 embedding）才适合做 Stage 1。
- `ShadowForCausalLM.shadow_lm_head` 默认冻结；新增 token 行必须放进 `ShadowConfig.modules_to_save` 才能训练/保存。
- exp7 已经显示 shadow-only 生成会漂移；如果学生本身不能稳定生成，控制 token 可能学成“遇到困难就 handoff”，而不是“在能力边界 handoff”。
- 仓库没有控制 token 的训练、成本感知 GRPO 或 mid-generation 交接的完整链路；issue #1 提供了 partial-trace 协议，但控制 token 检测与训练是另一层。

因此本规格不是默认路线，而是一个**带阶段门禁的可行性规格**：先在显式 shadow 上验证控制 token 能否稳定触发，再决定是否投入完整三阶段训练。

## Solution

在显式 shadow（导出模型）上实现 PyroDash 的最小可行版本：

1. **Stage 1：控制 token embedding 学习**。扩展 shadow 词表，加入 `<|offload|>`；用自然断点 anchor token（句号、换行、EOS）的均值加高斯噪声初始化；只训练该 token 的输入/输出 embedding 行。
2. **Stage 2：offloading-oriented SFT**。Corpus A 不含 offload prompt/控制 token；Corpus B 含 prompt，easy 样本不含控制 token，hard 样本在推理步骤之间插入 1–4 个控制 token；联合最小化 NLL。
3. **Stage 3：cost-aware GRPO**。冻结大模型；对完整协作 rollout 计算 `Acc − λ·Cost/Cost_base` 奖励；只有 SLM token 进入梯度。
4. **推理集成**：shadow-only 生成中检测 `τ_off`，停止并调用 issue #1 的 partial-trace 单次 handoff；不回到 SLM。
5. **阶段门禁**：每个阶段结束都评估 shadow-only 生成质量、控制 token 触发位置分布、LLM token ratio、准确率与成本；任一阶段不达标就停止，保留 issue #1 的 probe 路线。

## User Stories

1. 作为研究者，我想先在显式 shadow 上验证控制 token 能否被稳定生成，以便在投入三阶段训练前排除“词表/导出不兼容”的风险。
2. 作为研究者，我想确认隐式 shadow 不能被用于本规格，以便不会误改冻结 base 的 embedding。
3. 作为训练工程师，我想用 anchor token 均值加噪声初始化控制 token，以便它一开始就落在自然断点附近。
4. 作为训练工程师，我想只训练控制 token 的输入/输出 embedding 行，以便 Stage 1 不破坏 shadow 原有语言能力。
5. 作为训练工程师，我想在 Stage 2 联合训练“无 offload”和“有 offload”两套语料，以便模型既会正常回答，也会在困难处求助。
6. 作为训练工程师，我想让 easy 样本不含控制 token、hard 样本在推理步骤间插入 1–4 个控制 token，以便控制 token 学到“能力边界”而不是固定位置。
7. 作为训练工程师，我想在 Stage 3 用 `Acc − λ·Cost/Cost_base` 的奖励做 GRPO，以便准确率与成本同时进入目标。
8. 作为训练工程师，我想冻结大模型、只让 SLM token 进入梯度，以便复现 PyroDash 的训练约束。
9. 作为训练工程师，我想让成本模型复用 issue #1 的 prefill/decode/edge token 计价，以便训练目标与在线计价一致。
10. 作为推理工程师，我想在 shadow-only 生成中检测控制 token 并停止，以便把 partial trace 交给云端。
11. 作为推理工程师，我想让 handoff 最多发生一次、不回到 SLM，以便成本上界可证明。
12. 作为推理工程师，我想在控制 token 未出现时走普通 shadow-only 输出，以便不强制所有请求都升级。
13. 作为评估工程师，我想看到控制 token 的位置分布，以便判断它是否学在句号/步骤边界而不是随机位置。
14. 作为评估工程师，我想看到 LLM token ratio、LLM calls/example、准确率与成本，以便与 RouteLLM/固定阈值/issue #1 的 probe 对照。
15. 作为评估工程师，我想在 shadow-only 质量不达标时得到明确 no-go，以便不把控制 token 学成“逃避生成”。
16. 作为评审者，我想看到 Stage 1/2/3 的分阶段结果，以便定位问题来自词表、SFT 还是 RL。
17. 作为评审者，我想看到控制 token 触发后的云端续写质量，以便确认 partial trace 没有把错误前缀放大。
18. 作为运维人员，我想一键关闭控制 token 路径、回到 issue #1 的 probe，以便灰度出问题时回滚。
19. 作为安全与合规负责人，我想让控制 token 不进普通用户可见输出，以便不暴露内部协议。
20. 作为后续规格的维护者，我想让本规格只覆盖显式 shadow 与单轮 handoff，不碰多轮/隐式 shadow，以便控制风险。

## Implementation Decisions

### 1. 适用范围

- 只支持显式 shadow（`AutoModelForCausalLMWithHiddenProjection` 或自带 embedding/lm_head 的导出模型）。
- 隐式 shadow（共享 base `embed_tokens`）直接拒绝，错误信息说明“新增 token 会改冻结 base”。
- 只做单轮 QA/推理；多轮、工具调用、agent 轨迹不在本规格。

### 2. 控制 token 与词表

- `ShadowConfig` 新增 `control_token: str = "<|offload|>"`、`control_token_init: str = "anchor_mean"`、`anchor_tokens: list[str]`（默认句号、换行、EOS）。
- 扩展 shadow 的 `embed_tokens` 与 `shadow_lm_head` 各一行；base 权重不变。
- 初始化：`e_tau = mean(anchor embeddings) + N(0, 0.1²)`；输入与输出 embedding 可以共享初始化，也可以独立，首版独立并记录。
- 新 token 的输入/输出行必须进入 `modules_to_save`，否则保存/加载会丢。
- `export_shadow` 的导出模型必须保留扩展后的词表配置与控制 token；导出/加载 round-trip 要测。

### 3. Stage 1：embedding 学习

- 冻结 shadow 其余参数，只训练控制 token 的输入/输出 embedding 行（可加一个临时 LoRA，训练后丢弃，只保留 token 行）。
- 数据：自然断点 anchor 句子/段落，目标是让控制 token 在断点处的 likelihood 合理；不需要 handoff 标签。
- 验收：控制 token 能在断点位置被生成；不会在所有位置被生成；shadow 原有 ppl 不明显退化。

### 4. Stage 2：offloading-oriented SFT

- Corpus A：普通 QA/推理语料，不含 offload prompt、不含控制 token；目标是保持正常生成。
- Corpus B：含 offload prompt；easy 样本不含控制 token；hard 样本在推理步骤之间插入 1–4 个控制 token（位置来自人工/规则标注，不在 token 级随机插）。
- 联合 NLL：`L = L_A + L_B`；控制 token 位置参与 loss，其他位置照常。
- 验收：hard 样本上控制 token 位置与人工断点重合率；easy 样本误触发率；shadow-only 生成质量。

### 5. Stage 3：cost-aware GRPO

- 奖励：`R_total = R_acc − λ·R_eff`，`R_eff = Cost_act / Cost_base`；成本项复用 issue #1 的 `edge_tokens`、`cloud_prefill_tokens`、`cloud_decode_tokens`、`cloud_calls`、`rtt_ms`。
- `Cost_base` 是“同一 query 直接交给大模型”的估算成本；`Cost_act` 是“SLM 生成到控制 token + 云端续写”的成本。
- GRPO：组内标准化优势、clipped loss；大模型冻结、只让 SLM token 进梯度。
- λ 网格：0.05 / 0.1 / 0.6（参考 PyroDash 的工作点），但本项目用自己的成本口径，不直接套美元数字。
- 验收：准确率、LLM token ratio、LLM calls/example、成本；与固定阈值/issue #1 probe 对照。

### 6. 推理集成

- shadow-only 生成逐 token 检测控制 token；检测到后停止，去掉控制 token，构造 `partial_trace`。
- 调用 issue #1 的 `CloudRequest(partial_trace=..., handoff_index=..., prefix_trust=...)`；云端只续写。
- 每个请求最多一次 handoff；不回到 SLM。
- 控制 token 不进入最终用户可见文本；trace 中记录 `handoff_index` 与 `handoff_reason=control_token`。
- 没有云端/离线时，控制 token 被忽略，返回已生成的 shadow-only 文本，并标记 `degraded=true`。

### 7. 阶段门禁

- Stage 1 后：控制 token 可在断点生成，且 shadow ppl 退化 <2%；否则停止。
- Stage 2 后：hard 样本控制 token 与人工断点重合率 ≥70%，easy 误触发率 ≤5%，shadow-only 任务准确率不低于 baseline 的 95%；否则停止。
- Stage 3 后：LLM token ratio <5% 且准确率损失 ≤1 pp，或同准确率下成本 −50%；否则标记 no-go，保留 issue #1 probe。
- 任一阶段失败都不得进入生产灰度。

### 8. 成本与运行

- Stage 1–2：2k prompts × 3 epochs ≈ 3M token，约 1 小时。
- Stage 3：1k prompts × 8 rollouts × 512 token ≈ 4M token/epoch；10 epochs ≈ 40M token，按 1k tok/s 约 11 小时；加调试/评测 3–5 个 GB10 天。
- 训练与评测沿用 GB10 双机约定；不在 Mac 上跑。

## Testing Decisions

好的测试只断言外部行为：tiny 显式 shadow 上的控制 token 生成/停止、词表扩展、loss 分支与导出 round-trip；不直接断言私有层索引。

### 词表缝

- 扩展后 shadow 的 `embed_tokens`/`shadow_lm_head` 各多一行；base 权重逐位不变。
- 控制 token 的 id 在 tokenizer 与模型配置中一致。
- 隐式 shadow 调用本规格入口时抛错，错误信息说明共享 embedding 的约束。

### Stage 1 缝

- 初始化 `e_tau` 等于 anchor 均值 + 噪声（固定 seed 下确定）。
- 只训练控制 token 行：其他参数 `requires_grad=False` 或梯度为 0。
- 控制 token 在断点句子上可被生成；在随机位置的概率不异常高。

### Stage 2 缝

- Corpus A/B 的 loss 分支正确；easy 样本不含控制 token 标签。
- 控制 token 位置只在标注断点上参与 loss；其他位置不被误标。
- 固定 seed 下训练确定。

### Stage 3 缝

- 奖励计算：`R_total = R_acc − λ·R_eff`；`Cost_act/Cost_base` 在构造样例上正确。
- 组内优势标准化不产生 NaN；全相同奖励时优势为 0。
- 只有 SLM token 进入梯度；大模型参数梯度为 None/0。

### 推理与导出缝

- shadow-only 生成遇到控制 token 停止，`partial_trace` 不含控制 token。
- 控制 token 触发一次 handoff；重复控制 token 不会二次调用云端。
- 离线模式下控制 token 被忽略，返回局部文本并标记 degraded。
- 导出/加载 round-trip 保留控制 token 与扩展词表。
- `export_shadow` 的原有测试仍通过（无控制 token 时不改变行为）。

### 现有先例

- `tests/test_shadow_peft.py`：adapter 保存/加载、显式/隐式 shadow 路径。
- `tests/lab/test_solo_own_head.py`：独立 head/自有词表的训练写法。
- `tests/lab/test_specdec.py`：逐 token 生成与停止条件的测试写法。
- `tests/edge_cloud/test_edge_cloud.py`：partial-trace handoff 的路由测试（issue #1 落地后）。

## Out of Scope

- **隐式 shadow 的控制 token**：共享 base embedding，首版不支持。
- **多轮对话/agent 轨迹的控制 token**：首版只做单轮。
- **重训/微调大模型**：大模型始终冻结；本规格只训 SLM。
- **替代 issue #1 的 probe**：控制 token 是并行路线；失败时回退 probe。
- **生产灰度与全量上线**：本规格只到离线阶段门禁；上线另走发布流程。
- **控制 token 的对外 API/协议标准化**：trace 内部字段，不承诺给第三方。

## Further Notes

### 与已有研究的关系

本规格是 [PyroDash 等五篇论文 × ShadowPEFT 机制桥接](../research-pyrodash-collab.md) 中 P2-8 的落地版本。PyroDash 的三阶段、控制 token 初始化、成本归一化 GRPO 与单次 handoff 以该文档引用为准；本规格只引用结论，并把适用范围收窄到显式 shadow。

### 为什么标记 ready-for-human

- 需要改词表与导出模型，任何一步出错都会破坏 base/shadow 的对齐；
- 三阶段训练 + GRPO 的调参空间大，失败模式（控制 token 学成逃避生成）需要人工判断；
- 成本/奖励口径要与项目计价对齐，不能直接套论文美元数字；
- 与 issue #1 的协议耦合，需要人工决定何时切换到控制 token 路线。

### 已知风险

- exp7 的 shadow-only 漂移是最大预警：如果学生不能稳定生成，控制 token 可能学成“遇到困难就 handoff”，而不是“在能力边界 handoff”。
- 控制 token 可能被滥用为“提前结束”以降低成本，导致准确率下降；Stage 3 的奖励必须把准确率放在主项。
- 扩展词表会改变导出模型的 `vocab_size`，旧的 adapter/服务可能不兼容；必须做版本化与 round-trip。
- 单次 handoff 的收益上限在成本，不在准确率；如果云端已经远强于边缘，整段升级可能更准。

### 建议的首轮验收口径

- Stage 1：控制 token 可在断点生成，shadow ppl 退化 <2%。
- Stage 2：hard 样本控制 token 与人工断点重合率 ≥70%，easy 误触发 ≤5%，shadow-only 准确率 ≥baseline 95%。
- Stage 3：LLM token ratio <5% 且准确率损失 ≤1 pp；或同准确率下成本 −50%。
- 任一阶段不达标：停止，保留 issue #1 的 probe 路线，不进入生产。
