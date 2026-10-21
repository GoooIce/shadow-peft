Source: https://github.com/GoooIce/shadow-peft/issues/48

## Problem Statement

边缘 Router 目前只有一个全局升级阈值 `escalation_threshold`（默认 0.62）和一个硬预算；不同任务（故障诊断、DPPM 计算、质量判定、标准问答）的边缘/云端质量、token 成本和升级收益差异很大，一个全局阈值会把便宜任务过度升级、把昂贵任务升级不足。

仓库已经有 `CostModel`（edge/cloud prefill/cloud decode/call/RTT 分项计价）和 `handoff_eval`（风险-覆盖、成本-准确率曲线），但没有“按任务选阈值”的策略，也没有把 `accuracy − λ·cost` 变成可拟合、可复核的决策。现有 `handoff_threshold_by_task` 只服务 probe 的 prefix-trust 路径，legacy 整段升级仍用全局值；评测报告也只有全局指标，看不到 per-task 的升级率、成本和准确率。

## Solution

1. 在 `EdgeConfig` 增加 `escalation_threshold_by_task`，Router 的 legacy/fallback 升级按 `request.task` 查阈值，缺省回退全局值。
2. 新增纯函数成本策略模块：从带 counterfactual 的 replay 样本拟合 per-task 阈值，目标是 `accuracy − λ·cost_norm`；样本不足的任务回退全局阈值，`require_cloud_tasks` 保持永远升级。
3. 扩展 `handoff_eval`：`run_evaluation` 接受 per-task 阈值，报告增加 per-task 指标、per-task 风险-覆盖/成本-准确率曲线，schema 升到 v2；全局指标保持不变。
4. 新增 exp15 成本策略实验：synthetic KB 只用于跑通管线并标 `claims_gain=false`；真实 replay 集（500–1000 条、每任务 ≥30 条）才允许 go/no-go，并把拟合结果输出为可直接粘贴的 config 片段。

三个测试缝：纯 `cost_policy` 拟合器；`Router`/`EdgeConfig` 的按任务阈值行为；`handoff_eval` 的 per-task 指标与曲线。

## User Stories

1. 作为边缘运维人员，我想给不同任务设不同的升级阈值，以便便宜任务尽量留在端侧。
2. 作为成本负责人，我想指定一个 λ，让策略在准确率和云端成本之间按我的偏好选点。
3. 作为部署工程师，我想让没有 per-task 配置的任务自动回退全局阈值，以便旧配置不失效。
4. 作为安全/合规负责人，我想让 `require_cloud_tasks`（如质量判定）继续永远升级，不受 per-task 阈值影响。
5. 作为评测工程师，我想看到每个任务的升级率、准确率、单请求成本，以便定位哪个任务被过度升级。
6. 作为评测工程师，我想看到每个任务的风险-覆盖和成本-准确率曲线，以便和全局阈值对照。
7. 作为研究者，我想先用 synthetic KB 跑通“拟合→配置→评测”管线，以便在真实数据到位前验证代码。
8. 作为研究者，我想让 synthetic 结果明确标注为“非证据”，以便不会把 mock 数字当成真实收益。
9. 作为评审者，我想让 baseline 也在同一 train split 上拟合同一个 λ 的全局阈值，以便区分“per-task 的收益”和“只是调了阈值”。
10. 作为评审者，我想看到 train/val 的样本数和 bootstrap CI，以便判断收益是否稳定。
11. 作为数据工程师，我想有一个明确的 replay 样本 schema（task/score/labels/costs/split），以便从真实日志导出训练数据。
12. 作为数据工程师，我想在样本不足时看到 fallback 原因，以便知道哪些任务还需要补数据。
13. 作为训练工程师，我想让阈值拟合是确定性的，以便同一输入产出 byte-identical 的报告。
14. 作为训练工程师，我想让拟合器拒绝负成本、非法阈值和空数据，以便错误在入口暴露。
15. 作为维护者，我想让 probe 的 `handoff_threshold_by_task` 保持原语义，不被新字段覆盖。
16. 作为维护者，我想让空 `escalation_threshold_by_task` 的行为与今天完全一致，以便零风险上线。
17. 作为运维人员，我想在 trace 里看到本次实际使用的任务阈值，以便排查“为什么升级/没升级”。
18. 作为成本负责人，我想让每个任务的升级率不超过预算（默认 0.6），以便不会为了降本把某个任务全部推给云端。
19. 作为研究者，我想在 per-task 收益在 val 上不稳定时得到 no-go，以便不把过拟合的阈值带进生产。
20. 作为评审者，我想看到 edge-only、cloud-only、全局阈值、per-task 阈值四个参照点，以便理解成本-准确率的端点。
21. 作为后续维护者，我想让逻辑回归/决策树/contextual bandit 明确留在 out of scope，以便首版只做 2–4 个参数的 per-task 阈值。
22. 作为发布负责人，我想让 exp15 输出一个 `policy.json` 配置片段，而不是自动改生产配置，以便人工审核后再上线。

## Implementation Decisions

### 1. 配置与阈值语义

- `EdgeConfig` 新增 `escalation_threshold_by_task: dict[str, float] = {}`。
- 语义与现有 legacy 路径一致：当 `report.score < threshold` 时升级；阈值越高，升级越多。
- 查找顺序：`require_cloud_tasks` → `format_invalid`/`edge_unavailable`/`no_edge_result` → `escalation_threshold_by_task.get(task, escalation_threshold)`。
- `require_cloud_tasks` 仍然最优先，per-task 阈值不改变它。
- 新字段只服务 legacy/fallback 整段升级；probe 的 `handoff_threshold_by_task` 保持原语义，不合并、不重命名。
- 校验：必须是对象；key 转字符串；value 必须在 [0,1]，非法值抛 `ValueError`；空对象等价于不配置。
- `EdgeConfig` 的序列化/反序列化 round-trip 必须保留该字段。
- Router trace 记录本次实际使用的阈值和 task，例如 `触发升级：low_confidence（task=process_calc 阈值 0.35）`。

### 2. Replay 样本 schema

`CostSample` 是拟合器的输入，字段：

- `task: str` — 任务名，未知任务按 fallback 处理。
- `score: float` — 端侧置信度/probe score，必须在 [0,1]。
- `edge_correct: bool` — 端侧完整答案是否正确。
- `cloud_correct: bool` — 云端完整答案是否正确（counterfactual 标签）。
- `edge_cost: float` — 端侧完整答案的估计成本，≥0。
- `escalated_cost: float` — 整段升级的实际成本：端侧完整答案 token + 云端 prefill（prompt + 端侧答案）+ 云端 decode + call，≥0。
- `cloud_only_cost: float` — 同一 query 直接交给云端的成本（prompt prefill + decode + call），≥0。
- `split: "train" | "val"` — 显式切分；拟合器不自行切分。
- 可选 `sample_id: str`、`edge_latency_ms: float`、`escalated_latency_ms: float`，只用于报告/P95。

成本由 `CostModel` 计算，不引入新的价格字段；真实 replay 的 token 数必须记录来源。没有 `cloud_correct` 的样本不能进入训练（只能进入“证据不足”报告）。

### 3. 纯成本策略拟合器

新增 torch-free 的 `cost_policy` 模块（不依赖 Router/后端），提供：

- `fit_cost_policy(samples, *, cost_weight, threshold_grid=None, min_samples=30, require_cloud_tasks=()) -> CostPolicyFit`
- `evaluate_cost_policy(policy, samples, *, split="val", n_boot=1000, seed=0) -> CostPolicyReport`
- `CostPolicy.to_config()` 返回 `{"escalation_threshold": ..., "escalation_threshold_by_task": {...}}`

决策：

- 目标函数：对每个候选阈值，`chosen_correct = edge_correct if score >= threshold else cloud_correct`，`chosen_cost = edge_cost if score >= threshold else escalated_cost`；`cost_norm = mean(chosen_cost) / mean(cloud_only_cost)`；`objective = accuracy − cost_weight * cost_norm`。
- `cost_weight` 必须 ≥0；`cost_norm` 以 cloud-only 为分母，落在 [0,1] 附近（允许 >1，报告要显示）。
- 默认阈值网格 0.00–1.00、步长 0.05；允许显式覆盖。目标相同时选更低的阈值（更少升级、更低成本；`score >= threshold` 留在端侧，所以低阈值才更少升级），并在报告中记录 tie-break。
- 先在全量 train 上拟合全局阈值，再对每个 task 单独拟合；task 的 train 样本数 < `min_samples` 时回退全局阈值，`fallback_reason="insufficient_samples"`。
- `require_cloud_tasks` 中的任务不拟合阈值，固定为永远升级；报告中标 `always_cloud=true`。
- 空 train 数据、非有限 score、负成本、非法 split 直接抛 `ValueError`。
- 输出 `CostPolicyFit`：`global_threshold`、`thresholds_by_task`、`cost_weight`、`min_samples`、`always_cloud_tasks`、per-task fit report（n、threshold、accuracy、cost、cost_norm、objective、fallback_reason）。
- 纯函数、固定输入固定输出；不读文件、不联网、不引入随机性（bootstrap 只在 evaluator 里，用显式 seed）。

### 4. Router 集成

- `Router` 的 legacy 升级路径使用 per-task 阈值；probe/oracle 路径不变。
- 当 probe 不可用回退 legacy 时，新的 per-task 阈值生效。
- 空 `escalation_threshold_by_task` 时行为与当前版本逐位一致。
- 不改变 `require_cloud_tasks`、预算、circuit breaker、outbox、memory 的现有行为。
- 不新增自动加载 `policy.json` 的逻辑；exp15 只输出配置片段，部署由人工完成。

### 5. handoff_eval 扩展（第三缝）

- `run_evaluation` 新增 `escalation_threshold_by_task: Mapping[str, float] | None`。
- `legacy` 策略按 `sample.task` 查阈值，缺省回退 `escalation_threshold`；`probe`/`oracle` 不改。
- `strategy_metrics` 增加 `per_task: dict[task, {n, accuracy, escalation_rate, total_cost, cost_per_request, cloud_calls, latency_p95}]`；全局字段保持不变。
- `curves` 增加 `per_task: dict[task, {"risk_coverage": [...], "cost_accuracy": [...]}]`；全局曲线保持不变。
- `results["config"]` 记录 `escalation_threshold_by_task`；`SCHEMA_VERSION` 升到 `exp9-handoff-eval/v2`。
- 所有 per-task 指标从同一批 records 聚合，per-task 的 n 之和等于全局 n；空 task 不产生 NaN。
- 阈值映射在入口校验：key 字符串、value [0,1]，非法抛 `ValueError`。
- `synthetic_dataset`/`load_dataset` 现有 task 字段直接复用，不要求新数据格式。

### 6. exp15 成本策略实验

- `exp15` 支持两种数据源：
  - `--synthetic`：从 `knowledge.ENTRIES` 和确定性 prompt 变体生成 `CostSample`；`score=edge_quality`，`edge_correct=edge_quality>=0.5`，`cloud_correct=cloud_quality>=0.5`，成本用 `count_tokens` + `CostModel`。只用于管线/确定性检查，报告 `dataset_source=synthetic`、`claims_gain=false`。
  - `--dataset <jsonl>`：真实 replay 集，必须带 `split`；报告 `dataset_source=replay`。
- 脚本先按 λ 网格（默认 0.0 / 0.05 / 0.1 / 0.2 / 0.5 / 1.0）拟合 per-task 策略，再在 val 上评估；同时拟合同 λ 的全局阈值 baseline，并给出 edge-only、cloud-only、当前默认 0.62 的参照点。
- 输出：`policy.json`、`report.json`、`summary.csv`、`curves.csv`、`dataset.jsonl`；`policy.json` 可直接粘贴到 `EdgeConfig`。
- `claims_gain=true` 需要同时满足：`dataset_source=replay`、总样本 ≥500、每个被拟合 task 的 train/val 均 ≥`min_samples`；否则 `claims_gain=false`，`evidence="insufficient"`。
- 真实 replay 的 counterfactual 缺失时，报告必须写明“只有被升级的请求有云端标签”，不得把 replay 估计当作无偏因果收益。

### 7. 指标与 go/no-go

- 主指标：val accuracy、cost per request、cost_norm、每个 task 的 escalation rate、cloud calls/example、P95（有 latency 字段时）。
- Baseline：同 train split、同 λ、同阈值网格拟合出的全局阈值；这才是 per-task 策略的公平对照。
- Go 条件（满足其一）：
  - accuracy 不低于 baseline −0.5 pp，且 cost ≤ baseline × 0.8；或
  - cost ≤ baseline × 1.05，且 accuracy ≥ baseline +1 pp。
- 附加条件：没有任何 task 的 val escalation rate > 0.6（可配置）；被拟合 task 的 val 样本 ≥`min_samples`；per-task 的 val objective 不出现“train 提升、val 下降”的方向反转。
- 任一附加条件不满足 → `no_go` 或 `evidence_insufficient`，并在报告里列出原因。
- synthetic 模式永远不输出 go/no-go，只输出管线状态。

### 8. 兼容与回滚

- `escalation_threshold_by_task` 默认为空；空配置下 Router 行为与当前版本一致。
- probe 的 `handoff_threshold_by_task` 不被新字段读取或覆盖。
- exp15 不修改生产配置、不写 Router 状态、不联网；`policy.json` 是人工审核产物。
- 回滚只需清空 `escalation_threshold_by_task`（或恢复旧 config），不需要改代码。

## Testing Decisions

好的测试只断言外部行为，不依赖私有方法或内部网格实现。

### 纯 `cost_policy` 缝

- 构造两个 task 的已知样本：task A 最优阈值 0.3、task B 0.7，拟合结果在网格上落在预期点附近。
- `cost_weight=0` 时目标退化为 accuracy；高 λ 时目标偏向低成本。
- 目标相同时选更低阈值（更少升级）。
- task 样本数 < `min_samples` 时回退全局阈值，`fallback_reason` 正确。
- `require_cloud_tasks` 固定升级，不参与阈值拟合。
- `cost_norm`/`objective` 在手工样本上逐项相等；负成本、非法 score、空数据抛 `ValueError`。
- 同一输入重复拟合 byte-identical；`to_config()` round-trip。
- 先例：`test_cost_model.py` 的分项计价测试、`test_handoff_config.py` 的校验测试。

### Router 缝

- `EdgeConfig` 新字段的默认值、round-trip、非法 key/value 测试。
- 两个 task 同 score、不同阈值：一个升级、一个不升级；断言响应 `escalated`/`escalation_reason` 和 trace 里的有效阈值。
- 未知 task 回退全局阈值；空字典与旧行为一致。
- `require_cloud_tasks` 仍然永远升级，即使 per-task 阈值更高。
- probe 不可用回退 legacy 时，per-task 阈值生效；probe 可用时 `handoff_threshold_by_task` 不被新字段影响。
- 先例：`test_router_handoff.py` 的 mock streaming backend 与 `test_edge_cloud.py` 的 Router 集成测试。

### `handoff_eval` 缝

- `run_evaluation` 传 per-task 阈值后，两个 task 的升级率按各自阈值变化；全局 metrics 与不传时一致（当阈值相同）。
- `per_task` 的 n 之和等于全局 n；accuracy/cost/escalation_rate 与手工聚合一致。
- `per_task` 的 risk-coverage/cost-accuracy 曲线存在且单调性/端点正确；空 task 不产生 NaN。
- schema 升到 v2；确定性输出 byte-identical。
- 先例：`test_handoff_eval.py` 的确定性、曲线与 per-prompt 断言。

### exp15 缝

- `--synthetic --smoke` 写出全部产物，`claims_gain=false`，`policy.json` 能被 `EdgeConfig.from_dict` 接受。
- 小 replay fixture（<500 条）得到 `claims_gain=false`、`evidence=insufficient`。
- 确定性：同 seed/同输入重复运行输出一致。
- 先例：`test_jitrl_eval.py` 的 CLI 产物与 schema 测试。

## Out of Scope

- probe/prefix-trust 阈值的自动拟合；`handoff_threshold_by_task` 仍由人工配置。
- 逻辑回归、决策树、contextual bandit、强化学习；首版只做 2–4 个 per-task 阈值参数。
- 记忆重排（#5）与 per-task 阈值的交互；exp15 在 `memory_mode=off` 下评测。
- 生产日志挖掘、随机探索流量、在线 A/B；spec 只定义 replay schema，真实数据由人工提供。
- 自动部署/热加载 `policy.json`；配置变更仍需人工审核。
- 多轮会话、agent 轨迹、单请求多次 handoff。
- 真实云厂商账单/合同价格；继续用 `CostModel` 的配置价格。
- 把 synthetic KB 的 `edge_quality`/`cloud_quality` 当作真实准确率证据。

## Further Notes

- 研究来源：研究文档的 P1-6（成本感知阈值/任务分层）；它与 #1 的 `CostModel`/handoff 评测、#5 的 memory 是互补关系，不替代 #1 的 probe 或 #5 的重排。
- 公平 baseline 很重要：如果只和“当前默认 0.62”比，per-task 的收益会混入“阈值重新调参”的收益；必须和同 train split、同 λ 拟合的全局阈值比。
- Counterfactual 偏差：真实日志里只有被升级的请求有云端答案；没有随机探索时，replay 会高估/低估策略收益。报告必须显式写偏差方向，不能声称因果收益。
- 样本量：synthetic 只验证管线；真实结论至少 500 条、每任务 ≥30 条 train/val。低于此阈值只能算 `evidence_insufficient`。
- 成本口径：`CostModel` 的价格是配置值，不是真实账单；go/no-go 同时报告 cost_norm 和原始 token/call 数，避免价格假设主导结论。
- 安全边界：`require_cloud_tasks` 不受 per-task 阈值影响；任何 per-task 配置都不能绕过它。
- 首轮验收口径：同准确率成本 −20%，或同成本准确率 +1 pp；无任务升级率 >0.6；val 上方向不反转。任一不满足就停在报告，不进入生产。


