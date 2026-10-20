---
status: ready-for-agent
triage: ready-for-agent
date: 2026-10-20
source: docs/research-pyrodash-collab.md
scope: 运行时 P2-7（候选重排，不做动作空间 RL）
---

# 规格：JitRL 式记忆热修（候选重排）

## Problem Statement

产线里有大量重复出现的故障码、工艺参数和标准条款问题。`FeedbackStore` 已经记录了这些问题的端侧答案、云端答案、人工纠正、采纳状态和升级原因，但这些记录只用于离线 SFT；在同一个故障码以不同问法再次出现时，系统仍然从零推理，不能利用“上次人工已经纠正过”的经验。

JitRL 给出了一个免梯度的运行时改进思路：维护一个经验记忆，检索相似状态，估计动作优势，并把优势以加性 logit 更新写回模型。但它的前提是 agent 有明确的动作空间；单轮产线问答没有这样的动作集合，直接套用会退化成“答案检索”。因此本规格把 JitRL 降级为一个更务实的形态：**检索历史纠正案例，对边缘模型的 n-best 候选做重排**，不训练、不更新权重。

## Solution

新增 `ExperienceMemory`：

1. 从 `FeedbackStore` 构建记忆条目：prompt、task、候选答案、回报（采纳/纠正/拒绝）、adapter 版本、时间、租户/产线。
2. 推理时检索 top-k 相似历史案例；如果相似度和记忆置信度都过阈值，就用记忆中的 corrected answer 对边缘模型的 n-best 候选重排。
3. 记忆命中时可以避免一次云端升级；未命中时保持现有路由行为。
4. 用 TTL、adapter 版本、租户隔离和 false-bias 检查做护栏；所有操作在毫秒级完成，不引入新的重型模型。

首版只做候选重排，不做 logit processor、不做动作空间 RL、不替代 shadow 训练。

## User Stories

1. 作为产线运维人员，我想让重复出现的故障码问题复用上次的人工纠正，以便不必每次都升级到云端。
2. 作为产线运维人员，我想看到这次答案是否来自记忆重排、命中了哪条历史案例，以便判断是否可信。
3. 作为产线运维人员，我想在记忆命中但置信度不足时仍然升级，以便不让旧案例误导新问题。
4. 作为数据工程师，我想从 `FeedbackStore` 构建记忆，以便复用已有反馈而不是另建标注系统。
5. 作为数据工程师，我想把 corrected/accepted/rejected 映射成回报，以便记忆知道哪些答案值得复用。
6. 作为数据工程师，我想给记忆加 TTL 和 adapter 版本，以便工艺参数变更后旧案例自动失效。
7. 作为数据工程师，我想按租户/产线隔离记忆，以便 A 线的纠正不会泄漏到 B 线。
8. 作为推理工程师，我想对边缘模型生成 n-best 候选，以便记忆有可重排的对象。
9. 作为推理工程师，我想用轻量编码器（TF-IDF/字符 n-gram）检索，以便不引入新的 GPU 模型。
10. 作为推理工程师，我想在相似度低于阈值时完全不使用记忆，以便保持基线行为。
11. 作为推理工程师，我想把记忆分数与模型 logprob 组合成重排分数，以便不是纯检索、也不是纯模型。
12. 作为推理工程师，我想让重排只在 top-k 记忆的回报足够一致时生效，以便避免“表面相似但根因不同”的误导。
13. 作为质量负责人，我想看到 false-bias rate（记忆把正确答案改错的比率），以便设置安全的阈值。
14. 作为质量负责人，我想在记忆命中时仍然跑格式校验，以便结构化输出不被破坏。
15. 作为评估工程师，我想在重复故障码的改写集上对比 no-memory / rerank，以便量化收益。
16. 作为评估工程师，我想看到额外延迟，以便确认 <50ms 的预算。
17. 作为隐私负责人，我想让记忆只存必要的 prompt/答案与回报，不存敏感工单号/人员，以便合规。
18. 作为运维人员，我想一键关闭记忆重排，以便灰度出问题时回滚。
19. 作为后续规格的维护者，我想让本规格只做运行时重排，不碰 shadow 训练与 GKD 飞轮，以便解耦。
20. 作为项目负责人，我想在收益不足时明确 no-go，以便不把资源投入一个昂贵的 RAG 变体。

## Implementation Decisions

### 1. 记忆条目

```text
MemoryEntry = {
  entry_id: str,
  prompt: str,
  task: str,
  candidates: list[str],
  return_value: float,
  accepted: bool,
  corrected: bool,
  adapter_version: str,
  tenant: str,
  created_at: float,
  ttl_s: int,
  embedding: list[float] | null,
}
```

- 回报：`corrected=+1.0`、`accepted=+0.5`、`rejected=-1.0`；同一记录同时有 corrected 与 accepted 时取 corrected。
- 只从 `FeedbackRecord` 的 `prompt`、`task`、`edge_text`、`cloud_text`、`final_text`、`corrected_text`、`accepted`、`adapter_version` 构建；不新增采集字段。
- 租户默认取 `metadata.line`/`metadata.station`，缺省为 `default`。

### 2. 编码与检索

- 首版用字符 n-gram + TF-IDF（或 hashing）向量，避免引入 sentence encoder；中文按字符、英文/数字按词切分。
- 检索 cosine top-k（k=4–8），相似度阈值默认 0.75，可配置。
- 只检索同一 task、同一租户、未过期、adapter 版本兼容的条目。
- 检索结果为空或最高相似度低于阈值时，返回 `no_memory`，不做任何重排。

### 3. n-best 与重排

- 边缘后端在 `temperature > 0` 时采样 n 条答案，或在贪心解码下用 beam/多温度采样构造 n-best；n 默认 4。
- 重排分数：`score = logprob(candidate) + α · memory_score(candidate)`；α 默认 0.5，网格 0.25/0.5/1.0。
- `memory_score(candidate)` 是 top-k 记忆中与该候选归一化后相同/相近答案的回报加权和；不同任务用不同归一化（verdict/number 复用 RESTRAIN-lite 的抽取器，自由文本用精确/归一化字符串）。
- 只有当 top-k 记忆中至少 2 条支持同一候选、且平均回报 >0 时才允许改写 top-1；否则保持模型原选择。
- 首版不做 logit processor/首 token 偏置；如果后续要做，另立规格。

### 4. 与 Router 的集成

- 在 `Router.infer` 的边缘生成之后、升级判断之前调用记忆重排；重排后的答案仍然走现有置信度评估与格式校验。
- 如果重排把答案改成了高置信度且格式合规，则可以避免云端升级；响应中增加 `memory_hit`、`memory_entry_ids`、`memory_score`、`memory_reranked` 可选字段。
- 如果记忆不可用/检索失败，静默回退到现有路径，并增加 `memory_fallback_total` 指标。
- 离线模式仍然可用：记忆在本地，不依赖云端。

### 5. 护栏

- **TTL**：默认 30 天，可配置；过期条目在检索时过滤，后台可清理。
- **adapter 版本**：只使用与当前 `adapter_version` 相同或兼容的条目；版本变化后旧记忆降级为只读/过期。
- **租户隔离**：检索必须带 tenant 过滤；跨租户条目不可见。
- **false-bias**：在 held-out 上统计“记忆把原本正确的答案改错”的比例；超过 2% 时默认关闭重排。
- **隐私**：不存工单号、人员、原始日志；prompt 中的敏感字段按现有脱敏规则处理。
- **延迟**：检索+重排的 P95 必须 <50ms；超过则降级为 `no_memory`。

### 6. 评估

- 构造重复故障码改写集：同一 `knowledge.py` 条目的不同问法（同义词、语序、补充上下文），一半有历史纠正，一半没有。
- 对照：no-memory / rerank / （可选）first-token bias。
- 指标：准确率、false-bias rate、memory hit rate、额外延迟、云端调用数。
- 报告必须区分“记忆命中且正确”“记忆命中但错误”“未命中”三类。

### 7. 成本

- 无 LM 训练；2k 条记忆的编码 + 检索在 CPU/单卡分钟级；1k prompts 评测约 1 小时；1–2 个工程日。
- 如果引入 sentence encoder，成本和复杂度上升，另立规格。

## Testing Decisions

好的测试只断言外部行为：给定合成记忆与候选，输出确定的检索/重排结果；不直接断言私有索引结构。

### 记忆缝

- 从 `FeedbackRecord` 构建条目：corrected/accepted/rejected 的回报映射正确。
- TTL 过期条目不被检索；未过期条目可检索。
- adapter 版本不兼容的条目被过滤。
- 租户隔离：A 线条目不会出现在 B 线的检索结果中。
- 记忆持久化 round-trip（JSONL/JSON）。

### 检索缝

- 同义改写（“E-104 真空压力不足” vs “E104 吸嘴真空报警”）能检索到同一条目；无关 prompt 不命中。
- 相似度阈值以下的查询返回 `no_memory`。
- top-k 排序稳定，k 小于条目数时返回前 k。

### 重排缝

- 无记忆时输出与 baseline 逐位一致。
- 记忆支持同一候选且平均回报 >0 时，top-1 被改写；支持不足时保持原选择。
- 重排后仍走格式校验；格式不合规时回退到模型原答案。
- false-bias 用例：记忆指向错误答案时，护栏（支持数/回报阈值）阻止改写。
- 固定 seed 下重排确定。

### 路由缝

- 记忆命中且置信度高：`source=edge`，云端零调用，`memory_reranked=true`。
- 记忆命中但格式不合规：回退原答案/升级，行为与 baseline 一致。
- 记忆不可用：静默回退，`memory_fallback_total` 增加。
- 离线模式：记忆仍可用，云端零调用。

### 现有先例

- `tests/edge_cloud/test_edge_cloud.py`：`FeedbackStore`、Router、离线队列的现有测试。
- `tests/edge_cloud/test_edge_cloud.py::test_router_escalates_low_confidence_to_cloud`：升级路径的写法参考。
- `tests/lab/test_annotation_store.py`：本地存储/检索的测试写法参考。

## Out of Scope

- **替代 shadow 训练/GKD 飞轮**：记忆只做运行时重排，不改变模型权重。
- **动作空间 RL / JitRL 的完整 advantage 更新**：单轮 QA 没有动作集合，首版只做候选重排。
- **多轮对话/agent 轨迹记忆**：首版只做单轮。
- **生产数据迁移/历史全量回填**：首版只从现有 `FeedbackStore` 增量构建。
- **sentence encoder / 向量数据库**：首版用轻量 TF-IDF；引入新依赖另立规格。
- **跨租户/跨产线共享记忆**：首版严格隔离。

## Further Notes

### 与已有研究的关系

本规格是 [PyroDash 等五篇论文 × ShadowPEFT 机制桥接](../research-pyrodash-collab.md) 中 P2-7 的落地版本。JitRL 的记忆/优势/logit 更新机制以该文档引用为准；本规格明确把它降级为候选重排，不声称复现 JitRL 的 agent 结果。

### 为什么不做 logit 加法

JitRL 的 `z' = z + β·Â` 需要动作级的 Q/A 估计；单轮问答的“动作”是整段答案，首 token 偏置只能影响开头，无法稳定改变整段答案。候选重排更贴近单轮 QA 的实际决策点，也更容易做 false-bias 护栏。

### 已知风险

- 表面相似但根因不同的案例会误导重排；必须用支持数/回报阈值 + false-bias 监控。
- 记忆陈旧：工艺参数变更后旧案例可能有害；TTL 与 adapter 版本是必需字段。
- 隐私：prompt/答案可能含工艺参数；按现有脱敏规则处理，不存工单号/人员。
- 收益可能不如直接 RAG；如果 false-bias 或延迟不达标，标记 no-go，不要继续加检索复杂度。

### 建议的首轮验收口径

- 重复故障码改写集上准确率 +3 pp，false-bias <2%，额外延迟 P95 <50ms。
- 云端调用数不上升；记忆命中且高置信度的请求可以本地返回。
- 关闭记忆后行为与 baseline 逐位一致（可回滚）。
