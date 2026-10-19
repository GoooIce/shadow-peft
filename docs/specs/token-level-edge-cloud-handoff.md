---
status: ready-for-agent
triage: ready-for-agent
date: 2026-10-20
source: docs/research-pyrodash-collab.md
scope: EdgeCloud-CI 运行时协作路径（P0-1 + P0-3 + 评测台）
---

# 规格：成本感知的 token/span 级端云交接

## Problem Statement

产线端云协同目前是**整条回答级**的：端侧小模型先把整段答案生成完，置信度评估器再用一个全局标量阈值（默认 0.62）判断是否升级；一旦升级，云端拿到的是完整草稿并重新生成整段增强答案。

这带来三个具体问题：

1. **粒度太粗**：模型往往在推理中途才暴露困难，但系统必须等整段回答生成完才升级。端侧在困难段上继续解码的时间、云端重答整段答案的 decode token、以及端到端时延都被浪费。
2. **触发信号太弱**：置信度只来自端侧自己的 token logprob 和格式/重复启发式。ShadowPEFT 明明在同一次前向里同时有 base 与 shadow 两路 logits（`base_shadow` 模式），却没有被用来判断“当前前缀是否可信”。一个全局阈值也无法按任务（故障诊断、工艺计算、质检判定）分别校准。
3. **计价与评估缺失**：预算只数云端调用次数和升级比例，不区分边缘 token、云端 prefill、云端 decode 和网络 RTT；验收指标只有升级率和 P95，没有 risk–coverage 与 cost–accuracy 曲线，无法回答“同一准确率下能省多少”或“同一成本下能准多少”。

结果是：该早点求助的长尾问题没有得到更便宜的协商路径，不该求助的简单问题可能被统一阈值误升级，项目也无法用可解释的曲线向甲方证明升级策略的价值。

## Solution

在 EdgeCloud-CI 中加入一条**成本感知的 token/span 级协作路径**，保留现有整段升级路径作为默认与回退：

1. **流式边缘生成与逐步特征**：端侧生成时按 token 暴露当前步的 base/shadow 分歧特征（KL、top-1 一致、双路熵、margin、base top-1 是否落在 shadow top-K 内）以及 token logprob 特征。
2. **可替换的升级策略**：新增 `HandoffPolicy`，把“何时升级”从 `ConfidenceEstimator` 的固定加权里拆出来。首版提供 `legacy`（现有阈值）、`probe`（离线校准的轻量模型）、`oracle`（仅离线评测）三种模式；配置默认为 `legacy`，行为不变。
3. **span 边界上的单次交接**：当策略判定“下一个 span 不可信”时，端侧继续生成到最近的语义边界（句号、换行、推理步骤结束、结构化字段边界），然后停止，把 prompt + partial trace + 交接位置 + 前缀可信度打包给云端；云端只续写剩余部分，不回到端侧，每个请求最多一次 handoff。
4. **成本与风险的统一评测台**：一个可重放的 harness，对同一批 prompt 跑 `edge_only`、`legacy`、`probe`、`oracle` 四种策略，输出 risk–coverage、cost–accuracy、handoff 位置分布与 P95；mock 后端和真实后端共用同一套产物 schema。

这条路径不改变 ShadowPEFT 的逐层注入算子，也不把大模型塞进 shadow；它只把“端侧小模型 / 云端大模型”之间的接口做细、做可计价、做可学习。

## User Stories

1. 作为产线端侧运维人员，我想在响应里看到交接发生在第几个 token、属于哪种边界、触发原因是什么，以便判断这次升级是否合理。
2. 作为产线端侧运维人员，我想在配置里一键关闭 token 级交接、回到整段升级，以便在灰度出问题时立刻回滚。
3. 作为产线端侧运维人员，我想在断网时仍然拿到端侧的部分答案，并让 partial trace 进入离线队列，以便网络恢复后继续同步。
4. 作为产线端侧运维人员，我想让强制云端任务（如质检判定、安全操作）继续走整段云端路径，而不是被交接策略改写语义。
5. 作为产线端侧运维人员，我想在端侧生成超时或云端剩余时延不足时自动放弃交接并降级，以便不把请求拖过截止时间。
6. 作为产线端侧运维人员，我想看到每次请求的 cost estimate（边缘 token、云端 prefill、云端 decode、RTT），以便向甲方解释成本构成。
7. 作为项目负责人，我想按任务配置不同的前缀可信度阈值和升级预算，以便故障诊断与工艺计算使用不同的风险偏好。
8. 作为项目负责人，我想按 token 单价而不是只按调用次数限制云端预算，以便预算与真实账单对齐。
9. 作为项目负责人，我想在评测台上看到 risk–coverage 曲线，以便选择满足“错误率 ≤ 目标值”的最低升级率。
10. 作为项目负责人，我想在评测台上看到 cost–accuracy 曲线，以便回答“同准确率下省多少”与“同成本下准多少”。
11. 作为项目负责人，我想让评测台同时报告 shadow-only、joint base_shadow 和交接后最终答案的指标，以便避免“shadow 单独变好、部署路径变差”的假收益。
12. 作为质量负责人，我想让最终答案仍然经过现有的格式校验（JSON、number、verdict、label），以便交接不会绕过产线的结构化约束。
13. 作为质量负责人，我想在交接前看到“前缀可信度”信号，以便系统不会把一个已经跑偏的推理前缀交给云端继续放大。
14. 作为质量负责人，我想让云端可以被配置为“只续写”或“允许纠正前缀”，以便在不同供应商能力下选择质量与成本。
15. 作为云端服务开发者，我想在 `POST /v1/enhance` 里收到 `partial_trace`、`handoff_index`、`prefix_trust` 和 `allow_prefix_rewrite`，以便只生成剩余部分而不是重述整段答案。
16. 作为云端服务开发者，我想在响应里返回 `continued_from_prefix` 与 `corrected_prefix`，以便端侧能审计云端是否遵守了续写契约。
17. 作为端侧后端开发者，我想在不支持流式回调的模型后端上自动回退到整段升级，以便新策略不会阻塞旧部署。
18. 作为端侧后端开发者，我想在 `shadow_ckpt` 存在时拿到 base 与 shadow 两路 logits，以便 probe 使用真正的分歧特征；没有 shadow 时则退化为 token-logprob 特征。
19. 作为端侧后端开发者，我想用一个可注入的 `HandoffPolicy` 替换策略，而不修改 `Router` 的控制流，以便并行试验多套触发规则。
20. 作为端侧后端开发者，我想让 `HandoffDecision` 明确区分“现在交接”和“到下一个边界再交接”，以便策略不需要自己处理句子边界。
21. 作为端侧后端开发者，我想让策略看到 `task`、`prefix_tokens`、`boundary_kind`、`format_ok` 等上下文，以便不同任务使用不同规则。
22. 作为端侧后端开发者，我想保证每个请求最多调用一次云端，以便成本上界可证明、trace 可审计。
23. 作为端侧后端开发者，我想让 `partial_trace` 与云端续写在文本层面拼接且不重复前缀，以便客户端拿到的仍是完整答案。
24. 作为端侧后端开发者，我想让旧版客户端继续使用原有请求/响应字段，新字段全部可选，以便协议升级不破坏兼容性。
25. 作为数据飞轮维护者，我想把交接位置、前缀可信度、probe 特征、云端是否纠正前缀写入反馈记录，以便后续训练 on-policy 蒸馏和新的 probe。
26. 作为数据飞轮维护者，我想让离线队列重放时保留 partial trace 与 handoff 元数据，以便云端在恢复后仍能按续写语义处理。
27. 作为可观测性维护者，我想看到 handoff 次数、handoff 原因、handoff 位置直方图、边界等待 token 数、云端 prefill/decode token、成本估算和 legacy 回退次数。
28. 作为可观测性维护者，我想在 probe 文件缺失或特征不完整时看到明确的 fallback 指标，以便区分“策略没触发”和“策略不可用”。
29. 作为评测工程师，我想用同一批 prompt 重放四种策略，并固定随机种子与后端，以便结果可复现。
30. 作为评测工程师，我想用 bootstrap 置信区间报告准确率与成本差异，以便小样本结论不被噪声误导。
31. 作为评测工程师，我想让 mock 后端与真实 `transformers` / OpenAI-compatible 后端共用同一套产物 schema，以便先跑通管线再接真模型。
32. 作为评测工程师，我想用 `oracle` 策略给出“每个位置都知道云端是否会更好”的上界，以便判断 probe 还有多少提升空间。
33. 作为发布负责人，我想按“评测台 → 离线校准 probe → 小流量 probe → 小流量交接 → 默认”的顺序灰度，以便每一步都有回滚点。
34. 作为发布负责人，我想让适配器注册表的指标字段记录 handoff 策略版本，以便区分“模型变好”和“策略变好”。
35. 作为安全与合规负责人，我想让 partial trace 的日志记录遵循与完整答案相同的脱敏规则，以便不因为新增字段泄露工艺参数或人员信息。

## Implementation Decisions

### 1. 两层协议，不改现有 `Backend` 的必需接口

- 保留 `Backend.generate(request) -> Generation` 作为兼容路径。
- 新增可选的 `StreamingBackend` 协议：`generate_until(request, policy) -> Generation`，在逐 token 循环里调用 `policy.observe(...)`；`Router` 通过能力探测选择该路径，缺失时回退到 `generate` + 现有 `ConfidenceEstimator`。
- 新增 `HandoffPolicy` 协议：`observe(step) -> HandoffDecision`，`reset()` 每个请求调用一次。策略不直接调用云端，只做决策；云端调用仍由 `Router` 统一负责预算、熔断、截止时间和离线队列。

### 2. 决策类型形状（设计决策，非原型代码）

```text
HandoffDecision = {
  action: "continue" | "handoff",
  reason: "legacy_threshold" | "low_prefix_trust" | "format_invalid"
        | "task_requires_cloud" | "deadline" | "budget",
  boundary: "token" | "sentence" | "newline" | "step" | "json_field",
  handoff_index: int,
  prefix_trust: float,
  estimated_cost: CostEstimate | null,
}
```

`action="handoff"` 表示“到下一个合格边界再交接”，不是“立刻截断”。`Router` 维护 `handoff_pending` 状态，并在边界检测器确认边界后执行；超过 `max_boundary_wait_tokens` 或截止时间不足时，允许在 token 边界强制交接。

```text
HandoffFeatures = {
  kl_base_shadow: float | null,
  top1_agreement: float | null,
  base_entropy: float,
  shadow_entropy: float | null,
  base_margin: float,
  shadow_margin: float | null,
  base_top1_in_shadow_topk: bool | null,
  low_prob_frac: float,
  prefix_tokens: int,
  boundary_kind: str,
  task: str,
  format_ok: float,
}
```

`kl_base_shadow`、`top1_agreement`、`base_top1_in_shadow_topk` 仅在 base/shadow 双路 logits 可用时存在；缺失时策略必须显式退化，不得把缺失值当成 0。

### 3. 策略组合与优先级

`CompositePolicy` 按以下顺序判定，先命中先返回：

1. 端侧不可用或没有生成结果：沿用现有 `edge_unavailable` 升级路径。
2. `task_requires_cloud`：不做 mid-generation 交接，直接整段云端，保持现有语义。
3. 格式校验失败：标记 `format_invalid`，到最近边界交接。
4. probe 的 `prefix_trust < threshold`：标记 `low_prefix_trust`，到最近边界交接。
5. 剩余时延不足以覆盖 `RTT + prefill + 预期 decode`：不交接，降级并写入离线队列。
6. 成本预算不足：不交接，降级并标记 `budget`。
7. 已经交接一次：不再回到端侧，后续只由云端完成。

### 4. 成本模型

- 配置新增四项单价：`cost_per_edge_token`、`cost_per_cloud_input_token`、`cost_per_cloud_output_token`、`cost_per_cloud_call`；另加 `cloud_rtt_ms` 估算。
- `CostEstimate` 至少包含 `edge_tokens`、`cloud_prefill_tokens`、`cloud_decode_tokens_expected`、`cloud_calls`、`rtt_ms`、`estimated_cost`。
- 预算从“调用次数 + 升级比例”升级为“调用次数 + 升级比例 + 单位时间成本上限”；旧配置字段继续生效。
- 成本只在“是否交接”的门槛判断中使用，不改变最终响应文本；估算值写入响应与指标，便于与真实账单对照。

### 5. 协议扩展

- `CloudRequest` 新增可选字段：`partial_trace`、`handoff_index`、`edge_tokens_before_handoff`、`handoff_reason`、`prefix_trust`、`allow_prefix_rewrite`。`draft` 保留，语义仍是“端侧草稿”；当 `partial_trace` 存在时，云端必须优先按续写处理。
- `CloudResponse` 新增可选字段：`continued_from_prefix`、`corrected_prefix`、`output_tokens`。
- `InferResponse` 新增可选字段：`handoff_index`、`edge_tokens_before_handoff`、`handoff_reason`、`prefix_trust`、`cloud_output_tokens`、`cost_estimate`。
- 所有新字段缺省时行为与旧版完全一致；旧客户端忽略未知字段即可。

### 6. 云端续写契约

- 当 `partial_trace` 非空且 `allow_prefix_rewrite=false` 时，云端提示词要求“从该前缀继续，不要重复已有步骤”；返回文本被视为续写，端侧拼接为 `partial_trace + continuation`。
- 当 `allow_prefix_rewrite=true` 时，云端可以先纠正前缀再继续；响应必须设置 `corrected_prefix=true`，端侧用云端返回的完整文本替换，而不是简单拼接。
- 云端不支持续写或返回内容明显重复前缀时，端侧记录 `continuation_mismatch`，回退为整段云端答案或端侧答案，并计入指标。
- `POST /v1/enhance` 的请求/响应契约保持 JSON 对象；字段校验沿用现有 `ProtocolError` 风格。

### 7. 流式生成与特征提取

- `transformers` 后端在 `shadow_ckpt` 存在时，用 `ShadowForCausalLM` 的 `base_shadow` 模式加载（或把现有 `ShadowPeftModel` 包装成该任务模型），以便同时拿到 base logits 与 `shadow_logits`；没有 shadow 时只暴露 base logits 与 token logprob 特征。
- 逐 token 循环在采样/argmax 之后、拼接下一个 token 之前调用策略；策略看到的是“当前前缀 + 下一步分布”，因此可以在**发出风险 token 之前**决定交接。
- 首版沿用现有全前缀重算（`use_cache=False`）的 `_generate_shadow`，不引入 KV cache 变量；性能优化另行立项。
- 边界检测器是纯函数：句号/问号/感叹号/换行/编号步骤/JSON 字段结束；中文与英文标点都要覆盖。边界检测只看已生成文本，不依赖 tokenizer 的特殊 token。

### 8. Probe 的离线校准

- probe 是一个轻量模型（首版 logistic regression，特征标准化后线性打分），产物是 JSON：特征名、均值/方差、系数、截距、校准表、训练数据指纹、版本号。
- 训练标签来自评测台：对每个候选交接位置，比较“端侧继续”与“云端继续”的最终质量（规则/人工/云端一致性），得到 `prefix_should_handoff` 标签；同时记录 `prefix_trust` 的校准分箱。
- 服务路径不训练模型；probe 文件缺失、版本不匹配或特征不完整时，记录 `probe_unavailable` 并回退到 `legacy`。
- `oracle` 策略只用于离线评测，不进入生产配置；它按标签选择最优交接位置，作为上界。

### 9. 评测台

- 新增独立评测脚本（建议命名 `exp9_handoff_eval`），输入 prompt 集、后端配置、策略列表、随机种子；输出 JSON + CSV，字段与现有 `docs/exp*.md` 的单一事实源风格一致。
- 四种策略：`edge_only`、`legacy`、`probe`、`oracle`。每种策略报告：准确率、升级率、handoff 率、平均/分位 handoff 位置、云端 prefill/decode token、总成本、P50/P95 延迟、degraded 比例。
- 报告 risk–coverage 与 cost–accuracy 曲线；用 bootstrap 给准确率与成本的置信区间。
- mock 后端用于管线冒烟；真实后端用于结论。两者产物 schema 相同，报告中必须标注后端与模型版本。

### 10. 兼容与灰度

- `EdgeConfig` 新增 `handoff_mode`，默认 `legacy`；可选 `off`（完全禁用 token 级路径）、`legacy`、`probe`、`oracle`（仅评测）。
- 新配置字段全部有默认值；旧配置文件原样加载。未知字段继续被忽略（保持现有 `from_dict` 行为）。
- 建议灰度顺序：评测台 → 离线校准 probe → 5% 设备 `probe`（只观察不交接或只交接小流量）→ 25% 设备交接 → 全量。`AdapterRegistry` 的指标字段记录 `handoff_policy_version`，以便区分模型与策略的贡献。

## Testing Decisions

好的测试只断言**外部行为**：HTTP 响应字段、云端收到的请求、离线队列内容、指标计数；不直接断言私有状态、内部方法调用顺序或日志文本。测试必须确定性：注入固定时钟/预算，不使用真实网络与真实 sleep。

### 最高测试缝

- **HTTP 缝**：`POST /v1/infer` → `EdgeService` → `Router` → 脚本化后端 + 假云端。用一个 `ScriptedStreamingBackend` 按预设 token/特征序列产出，假云端记录收到的 `CloudRequest`。断言最终 `InferResponse` 的 `source`、`text`、`escalated`、`handoff_index`、`cloud_output_tokens`、`degraded`、`trace`，以及云端确实收到 `partial_trace` 而不是完整 `draft`。
- 这是现有 `test_edge_cloud_http_end_to_end` 的同级缝，优先复用。

### 策略与路由缝

在 `Router.infer` 直接注入确定性的 `HandoffPolicy` stub，覆盖：

- 前缀可信且未触发：保持 edge-only，云端零调用。
- 触发于句中：等到最近句号/换行/步骤边界才交接，`handoff_index` 等于边界位置。
- 触发但 `max_boundary_wait_tokens` 用尽：在 token 边界强制交接。
- `task_requires_cloud`：不生成 partial trace，直接整段云端，原因为 `task_requires_cloud`。
- 剩余时延不足：不调用云端，`degraded=true`，原因 `deadline`。
- 成本预算不足：不调用云端，`degraded=true`，原因 `budget`。
- 云端失败：保留端侧完整/部分答案，写入离线队列，熔断计数增加。
- 离线模式：不调用云端，partial trace 入队；恢复后重放保留 `partial_trace` 与 `handoff_index`。
- 单次交接：策略反复触发也只调用云端一次，后续 token 不再进入策略。
- probe 文件缺失/特征缺失：回退 `legacy`，指标 `probe_unavailable` 增加。

### 纯函数与协议缝

- 特征提取：相同 logits 的 KL=0、top-1 一致=1；shadow top-K 排序正确；缺失 shadow 时相关特征为 `None` 而非 0。
- 边界检测：中英文句号、问号、感叹号、换行、编号步骤、JSON 字段结束；半句不产生边界。
- 成本估算：prefill/decode/边缘 token 分开计费；单价为 0 时退化为调用次数。
- 协议：新增字段 round-trip；旧 payload 不带新字段仍可解析；非法 `handoff_index`/`prefix_trust` 报 `ProtocolError`。
- 配置：旧配置加载后 `handoff_mode=legacy`；非法 mode/boundary 被拒绝；默认值不改变现有行为。
- 指标：handoff 原因、位置直方图、prefill/decode token、fallback 计数在对应路径上各加一次。

### 评测台缝

- 同一 prompt 集、同一 seed、同一脚本化后端，重复运行产物逐位一致。
- 四种策略的产物 schema 一致；`oracle` 的准确率不低于 `probe` 与 `legacy`（在标签完整时）。
- bootstrap 置信区间在样本量不足时给出更宽的区间，而不是假装显著。
- mock 与真实后端的产物都能被同一读取器解析。

### 现有先例

- `test_edge_cloud_http_end_to_end`：HTTP 端到端与响应字段断言。
- `test_router_escalates_low_confidence_to_cloud` / `test_router_keeps_confident_answer_on_edge`：Router 级升级/不升级行为。
- `test_confidence_high_vs_low`：置信度信号的纯函数测试。
- `test_protocol_roundtrip_and_validation`：协议 round-trip 与非法输入。
- `test_router_offline_mode_queues_then_flushes` / `test_router_cloud_failure_degrades_and_queues`：离线队列与降级。
- `test_cloud_budget_caps_calls_and_ratio`：预算行为，可扩展为成本预算测试。

## Out of Scope

- **GKD 式 on-policy 飞轮蒸馏**：属于训练侧工作流，另立规格；本规格只负责把 handoff 位置、前缀可信度、云端续写结果写入反馈记录，供后续训练消费。
- **RESTRAIN 的无标签自惩罚训练**：另立规格，且仅建议用于结构化输出任务。
- **JitRL 式免训练记忆热修**：另立规格。
- **PyroDash 的控制 token 训练**（control-token embedding、offloading SFT、cost-aware GRPO）：需要改词表/训练管线，风险与成本都高于本规格。
- **跨 tokenizer 或跨架构的 token 级 KV/hidden 交接**：本规格只做文本 partial trace；不传输 KV cache、hidden state 或模型权重。
- **多轮会话级 handoff**：本规格按单请求单次交接设计；多轮需要会话状态、累计预算与 handoff 计数，另行设计。
- **真实云厂商账单接入**：成本单价由配置提供，本规格不拉取账单、不做结算。
- **ShadowPEFT 注入算子与 base 模型改动**：本规格不改变逐层注入、影子网络结构或基础模型。
- **生产流量影子复制（shadow traffic）**：评测台的 `oracle` 只在离线 prompt 集上跑，不复制生产流量。
- **probe 的在线自动重训**：首版只加载离线校准产物；在线学习另行立项。

## Further Notes

### 与已有研究的关系

本规格是 [PyroDash 等五篇论文 × ShadowPEFT 机制桥接](../research-pyrodash-collab.md) 中 P0-1 + P0-3 的落地版本；GKD 训练侧（P0-2）与 S2T 选择器（P1-4）不在本规格内。论文事实与数字以该文档的引用为准，本规格只引用其结论，不重复论文细节。

### 交接成立的经济学条件

设云端输入/输出单价为 `c_pre/c_dec`、边缘每 token 成本 `c_s`、handoff 位置占比 `ρ`，则相对整段升级的收益约为：

```text
(c_dec - c_pre) * ρT - c_s * (1 - ρ)T - ΔRTT
```

因此 token 级交接并非“越早越好”：只有当云端 output 明显贵于 input、且交接位置足够靠后时，节省的 decode 才能覆盖边缘已付出的计算与 RTT。评测台必须同时扫 `ρ` 与前缀可信度，不能只扫固定比例。

### 前缀锚定是主要质量风险

云端续写一个已经跑偏的前缀，会把错误放大。缓解手段有三层：只在语义边界交接；用 `prefix_trust` 而不是“当前 token 低概率”做触发；允许配置 `allow_prefix_rewrite` 让云端纠正前缀。三层都要在评测台上单独消融，不能假设“续写一定比重答好”。

### 与 `shadow_only` 的关系

本规格不把 `shadow_only` 当 drafter，也不依赖它的 KV cache。`shadow_only` 当前每次重算完整 shadow hidden；如果后续要做 shadow→base 的投机解码，需要先给该分支接上增量缓存，那是另一个规格。

### 发布阻塞

本仓库的 GitHub Issues 处于关闭状态，当前没有可发布的 issue tracker，因此本规格暂以 `docs/specs/` 下的本地文件交付，并在 frontmatter 标记 `ready-for-agent`。需要先运行 `/setup-matt-pocock-skills` 配置 tracker 与 triage 词汇，之后可把本文件原样发布并应用 `ready-for-agent` 标签。

### 建议的首轮验收口径

- 同准确率下，云端输出 token 减少 ≥30%，端到端 P95 不劣化超过 10%。
- 同成本下，任务准确率提升 ≥2 pp；或同升级率下 AURC 相对提升 ≥5%。
- 单次交接约束在 100% 请求上成立；离线模式下 100% 请求可本地返回并入队。
- 旧配置、旧客户端、无 shadow 后端三类兼容场景全部通过现有测试与新增回归。
