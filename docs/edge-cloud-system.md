# 产线端云协同推理系统（EdgeCloud-CI）设计与实施方案

- 文档版本：v0.1（参考实现阶段）
- 适用范围：生产线端侧小模型 + 云端大模型增强的定制开发项目
- 代码位置：`src/edge_cloud/`，测试：`tests/edge_cloud/`
- 需求基线：`docs/edge-cloud-srs.md`

## 1. 背景与目标

产线现场同时存在三类互相冲突的要求：

1. **低时延**：工位问答、报警处置、质检判定需要在数百毫秒内给出结果；
2. **高准确率**：长尾工艺问题、标准条款、跨工序诊断超出小模型知识边界；
3. **可离线**：车间网络抖动、云端故障或数据不出厂时，产线不能停。

本系统采用「端侧小模型本地直出 + 云端大模型按需增强」的双端架构：

- 端侧负责绝大多数常规请求，保证时延与断网可用；
- 云端只在端侧置信度不足、任务要求复核或格式不合规时介入；
- 云端把产线反馈沉淀为难例数据集，训练轻量适配器（ShadowPEFT / LoRA）后灰度下发回端侧，形成「越用越准」的闭环。

目标不是让云端接管所有请求，而是把云端调用控制在**可预算、可解释、可回滚**的比例内。

## 2. 名词约定

| 名词 | 定义 |
| --- | --- |
| 端侧小模型 | 部署在产线边缘盒子/工控机上的 0.5B–3B 级模型，负责本地推理 |
| 云端大模型 | 部署在云端或厂区数据中心的 14B–72B 级模型，负责复核与增强 |
| 升级（Escalation） | 端侧无法可靠回答时，把请求连同端侧草稿发往云端 |
| 增强（Enhancement） | 云端在端侧草稿基础上给出更完整、可执行的最终答案 |
| 难例（Hard Case） | 被升级、被人工纠正或被判为不可接受的样本 |
| 适配器（Adapter） | 由难例训练出的端侧增量权重，独立于基座模型版本化 |
| 数据飞轮 | 反馈采集 → 难例导出 → 云端训练 → 灰度下发 → 效果回收的闭环 |
| 灰度（Canary） | 按设备稳定哈希分配新适配器版本，先小比例验证再全量 |

## 3. 总体架构

```
                         ┌──────────────────────── 云端（厂区数据中心 / 公有云） ───────────────────────┐
                         │                                                                              │
 产线工位 / MES / Andon │   ┌──────────────┐   升级请求     ┌─────────────────┐   难例     ┌──────────────┐  │
 ────────────────────────┼──▶│  端侧代理     │──────────────▶│  云端增强服务    │──────────▶│  反馈库       │  │
                         │   │ Edge Agent   │◀──────────────│  Cloud Enhancer │           │ FeedbackStore│  │
                         │   │              │   增强结果     │                 │           └──────┬───────┘  │
                         │   │ · 小模型推理  │                │ · 大模型推理     │                  │          │
                         │   │ · 置信度评估  │                │ · 草稿复核       │           ┌──────▼───────┐  │
                         │   │ · 升级路由    │                │ · 结果校验       │           │ 训练流水线    │  │
                         │   │ · 熔断/预算   │                │ · 审计日志       │           │ SFT/Distill  │  │
                         │   │ · 离线队列    │                └─────────────────┘           └──────┬───────┘  │
                         │   │ · 反馈采集    │                                                   │          │
                         │   └──────┬───────┘                                          ┌────────▼────────┐ │
                         │          │ 拉取适配器版本                                     │ 适配器注册表     │ │
                         │          └───────────────────────────────────────────────────│ AdapterRegistry │ │
                         │                                                              └─────────────────┘ │
                         └──────────────────────────────────────────────────────────────────────────────┘
```

端侧代理是产线唯一依赖，必须能在**无外网**时独立工作；云端是可选的增强面和管理面。

## 4. 三种协同模式

### 4.1 本地直出（默认路径）

端侧小模型先答，输出 token 级对数概率。置信度达到任务阈值时直接返回，不产生云端调用。

适用：常规工艺参数、已知故障码、标准作业指导、安全操作。

### 4.2 在线升级与增强（低置信度 / 强制复核）

触发条件满足任一：

- 置信度低于任务阈值；
- 任务被配置为强制云端（如质检判定、安全操作）；
- 结构化输出格式不合规（如 verdict 缺少结论、JSON 解析失败）；
- 端侧模型不可用。

云端收到请求时会带上**端侧草稿**，提示词要求复核事实与数值，输出更完整的最终答案。云端返回自己的置信度与信号，端侧记录两侧结果与决策链路。

### 4.3 离线增强闭环（数据飞轮）

产线反馈（采纳 / 纠正）写入本地 JSONL；云端可用时同步到云端反馈库。云端按需导出难例 SFT 数据集，训练适配器，发布灰度版本，端侧按设备哈希拉取新版本。任一环节失败都不阻塞在线推理。

### 4.4 跨端投机解码（可选，默认关闭）

端侧生成草稿 token，云端批量验证，可进一步降低云端输出成本。该模式对网络往返敏感，仅在厂区数据中心与产线之间 RTT < 20ms 且吞吐需求高时启用；当前参考实现未启用。

## 5. 关键机制

### 5.1 置信度评估

`ConfidenceEstimator` 组合两类信号：

| 信号 | 含义 | 默认权重 |
| --- | --- | --- |
| `logprob_geomean` | 生成 token 对数概率的几何平均 | 0.72 |
| `logprob_min` | 最低 token 概率 | 0.16 |
| `low_prob_ok` | 低于 `-2.0` 的 token 占比的补数 | 0.07 |
| `repetition_ok` | 4-gram 重复惩罚 | 0.02 |
| `length_ok` | 空输出/过短输出惩罚 | 0.02 |
| `format_ok` | JSON / verdict / number 等格式校验 | 0.01 |

设计原则：**token 概率承担校准，启发式只负责识别退化输出，不能用“格式正确”抬高一个低概率答案**。没有 logprob 的云端供应商退化为纯启发式，需在项目期用标注集重新标定。

上线前必须做阈值标定：在 500–1000 条产线评测集上画 risk–coverage 曲线，按任务选择满足「错误率 ≤ 目标值」的最低升级率。默认 0.62 只是演示值。

### 5.2 升级路由与预算

`Router` 的顺序：

1. 端侧推理（失败则记录 `edge_unavailable`）；
2. 置信度评估；
3. 判断升级原因；
4. 离线模式 → 保留端侧结果并入队；
5. 熔断打开 → 跳过云端；
6. 滑动窗口预算（绝对调用数 + 升级比例）→ 跳过云端；
7. 剩余时延不足 → 跳过云端；
8. 云端增强；失败则保留端侧结果、写入离线队列，并累计熔断失败数。

预算默认：60 次/分钟、升级比例 ≤ 60%（前 20 个请求不做比例限制，避免冷启动误杀）。预算是成本护栏，不是准确率策略；超预算时响应会标记 `degraded=true`，产线应展示「建议人工复核」。

### 5.3 云端草稿复核

云端请求包含 `draft`。云端提示词要求：

- 复核草稿中的事实、数值、标准条款；
- 给出结论、依据、可执行步骤；
- 数值计算写明口径；
- 不确定时明确要求补充上下文，而不是编造。

云端返回 `confidence` 与 `confidence_signals`，端侧原样记录，避免二次估算丢失信息。

### 5.4 离线队列

`Outbox` 是追加式 JSONL：

- 升级被跳过、云端失败或断网时写入；
- 后台 `OutboxFlusher` 按固定间隔重放，成功后标记 `done`；
- 连续失败累计 `attempts`，超过 10 次标记 `dead`，保留现场供人工处理；
- 重放使用原 `request_id`，云端可按幂等键去重。

### 5.5 反馈与难例

`FeedbackStore` 记录：

- 是否采纳、人工纠正文本、最终答案；
- 端侧/云端两侧答案、置信度、升级原因、适配器版本；
- 任务类型、来源、备注。

难例定义：未采纳、有人工纠正、或发生过升级。导出时按 prompt 去重，输出 `messages` 格式的 SFT JSONL，可直接接 TRL / 自研训练脚本。

### 5.6 适配器灰度与回滚

`AdapterRegistry` 保存版本、产物路径、指标、灰度比例与历史：

- `publish`：注册新版本；已有稳定版时进入灰度；
- `resolve(device_id)`：按 `crc32(device_id + version)` 稳定分桶，同一设备不会在版本间抖动；
- `promote`：灰度转正，记录上一个稳定版；
- `rollback`：回到上一个稳定版。

端侧启动时与转正后拉取当前版本；生产环境应增加定时轮询与签名校验。

### 5.7 token/span 级交接

整段升级路径之外，系统提供一条**成本感知的 token/span 级交接**路径：端侧流式生成时按 token 观察前缀可信度，判定不可信后等到最近的语义边界停止，把 prompt + partial trace + 交接位置交给云端，云端只续写剩余部分，每个请求最多一次交接。整段升级路径仍是默认与回退。

#### 模式

`edge.handoff_mode` 控制路径，默认 `legacy`：

| 模式 | 行为 | 适用 |
| --- | --- | --- |
| `off` | 完全禁用 token 级路径，按整段升级处理 | 应急关闭 |
| `legacy` | 现有整段升级：端侧生成完整答案，`ConfidenceEstimator` 阈值决定是否把整段草稿送云端 | 默认、回滚 |
| `probe` | 逐 token 流式生成，`ProbePolicy` 用离线校准的 probe 打分；可信度低于阈值时到边界交接 | 灰度、生产候选 |
| `oracle` | 与 `probe` 相同的控制流，但用请求 metadata 里的 `gold_handoff_index` 驱动 | 仅离线评测 |

`off` 与 `legacy` 都走整段路径（`off` 用于显式声明关闭）。`probe` 要求端侧后端支持流式生成（`generate_until`）；后端不支持、probe 文件缺失/版本不匹配/特征不可用时，自动回退整段路径，并分别累计 `streaming_unavailable_total`、`probe_unavailable_total`、`legacy_fallback_total`。`probe_path` 按字面路径读取（不做 `~` 展开），生产配置应使用绝对路径。`require_cloud=true` 或任务在 `require_cloud_tasks` 中时，跳过流式路径，保持整段云端语义。

#### HandoffPolicy 与单次交接

`HandoffPolicy` 是 Router 可注入的策略接口：

- `reset(context)`：每个请求调用一次，context 包含 task、deadline、expected_format；`oracle` 另读 `gold_handoff_index`。
- `observe(step) -> HandoffDecision`：每生成一个 token 调用一次。`step` 包含 `HandoffFeatures`（base/shadow 分歧、熵、margin、top-1 一致、低概率 token 占比、已生成 token 数、边界类型、任务、格式状态）与当前 token/logprob。
- `HandoffDecision.action="handoff"` 的语义是「到下一个合格边界再交接」，不是立刻截断；`reason` 取 `legacy_threshold`、`low_prefix_trust`、`format_invalid`、`task_requires_cloud`、`deadline`、`budget` 之一。

没有 shadow checkpoint 时，`kl_base_shadow`、`top1_agreement`、`shadow_entropy`、`shadow_margin`、`base_top1_in_shadow_topk` 为 `None` 而非 0；probe 若把这些列为必需特征，会在流式开始前标记不可用并回退 `legacy`。

策略只负责决策，不直接调用云端；云端调用、预算、熔断、截止时间、离线队列仍由 `Router` 统一负责。因此同一个 Router 可以替换 `probe`、`oracle` 策略或回退 `legacy` 路径，而不改控制流。

单次交接由 Router 强制执行：策略反复触发也只调用云端一次，之后由云端完成剩余生成；`InferResponse` 的 `handoff_index` 是最终切分位置，`edge_tokens_before_handoff` 是端侧已提交 token 数。交接后云端失败、剩余时延不足、预算不足、熔断打开或离线时，保留端侧前缀并标记 `degraded=true`，按原离线队列语义入队。

#### 边界等待

`max_boundary_wait_tokens`（默认 12）限制策略触发后最多再生成多少 token 去找语义边界。边界检测是纯函数，按优先级识别：

1. `json_field`：`expected_format="json"` 且文本以 `,`、`}`、`]` 结尾；
2. `step`：最后一行只是编号/项目符号（`1.`、`2、`、`-` 等）；
3. `sentence`：中英文句号、问号、感叹号、省略号（可跟收尾引号/括号）；
4. `newline`：换行。

等待预算用尽或截止时间不足时，允许在 `token` 边界强制切分。边界检测只读已生成文本，不依赖 tokenizer 特殊 token；半句不产生边界。

#### 协议字段

新增字段全部可选，缺省时与旧版完全一致：

| 方向 | 字段 | 含义 |
| --- | --- | --- |
| `CloudRequest` | `partial_trace` | 交接前缀全文；非空时云端优先按续写处理，不再看 `draft` |
| `CloudRequest` | `handoff_index` / `edge_tokens_before_handoff` | 交接 token 位置 / 端侧已生成 token 数 |
| `CloudRequest` | `handoff_reason` / `prefix_trust` | 触发原因 / 前缀可信度 |
| `CloudRequest` | `allow_prefix_rewrite` | 是否允许云端先纠正前缀再续写 |
| `CloudResponse` | `continued_from_prefix` / `corrected_prefix` | 云端是否按续写返回 / 是否纠正了前缀 |
| `CloudResponse` | `output_tokens` | 云端实际输出 token 数，用于成本估算 |
| `InferResponse` | `handoff_index` / `edge_tokens_before_handoff` / `handoff_reason` / `prefix_trust` | 端侧可见的交接元数据 |
| `InferResponse` | `cloud_output_tokens` / `cost_estimate` | 云端输出 token / 成本分解 |

`allow_prefix_rewrite=false` 时，云端返回文本按续写拼接为 `partial_trace + continuation`；云端重述前缀会被识别为 `continuation_mismatch`，按安全拼接处理。`allow_prefix_rewrite=true` 时以云端完整文本替换前缀，并标记 `corrected_prefix`。旧客户端忽略未知字段即可；旧 payload 不带新字段仍可解析。

#### 成本口径

配置新增四项单价与 RTT 估算：

- `cost_per_edge_token`
- `cost_per_cloud_input_token`
- `cost_per_cloud_output_token`
- `cost_per_cloud_call`
- `cloud_rtt_ms`

`CostEstimate` 记录 `edge_tokens`、`cloud_prefill_tokens`、`cloud_decode_tokens_expected`、`cloud_calls`、`rtt_ms`、`estimated_cost`，并给出 `edge_cost` / `cloud_prefill_cost` / `cloud_decode_cost` / `cloud_call_cost` 分解。成本只用于预算与评测门槛，不改变最终响应文本；单价全为 0 时退化为按调用次数计价，与旧预算一致。预算窗口在原有「绝对调用数 + 升级比例」之上增加 `max_cost_per_min`（滑动窗口成本上限，窗口长度由 `cloud_window_s` 决定，默认 60s；0 表示不启用）。

#### 回滚到 legacy

- 配置回滚：把 `handoff_mode` 设为 `"legacy"`（或 `"off"`），或将 `probe_path` 清空；旧配置文件不带任何 handoff 字段时默认就是 `legacy`。
- 自动回退：端侧后端不支持流式、probe 不可用、特征缺失时，Router 自动走整段路径，不需要改配置。
- 生产灰度建议：评测台 → 离线校准 probe → 小流量 `probe` → 逐步扩大；每一步都保留 `legacy` 回滚点。

#### 评测与校准入口

```bash
# 四策略评测（合成/mock 或 --dataset 指定 prompt 集）
PYTHONPATH=src .venv/bin/python scripts/exp9_handoff_eval.py \
  --backend synthetic --n 1000 --seed 7 --bootstrap 1000 \
  --out runs/exp9/handoff_eval \
  --export-candidates runs/exp9/handoff_eval/candidates.jsonl

# 离线拟合 probe（候选 label 1 = should_handoff，产物 score 是 prefix trust）
PYTHONPATH=src .venv/bin/python scripts/exp9_fit_probe.py \
  runs/exp9/handoff_eval/candidates.jsonl \
  --out runs/exp9/handoff_eval/probe.json --threshold 0.5

# 带 probe 复跑
PYTHONPATH=src .venv/bin/python scripts/exp9_handoff_eval.py \
  --backend synthetic --n 1000 --seed 7 --bootstrap 1000 \
  --probe runs/exp9/handoff_eval/probe.json \
  --out runs/exp9/handoff_eval/probe_run
```

评测产物（`results.json`、`summary.csv`、`curves.csv`、`per_prompt.jsonl`）的 schema 与 mock/真实后端共用；真实后端需用 `--backend transformers|openai --backend-config <json>` 并额外提供 `--dataset`，云端答案仍由数据集脚本提供，真实模型数字不能由 synthetic 结果替代。已完成的合成轮结论与限制见 [exp9 交接评测档案](exp9-handoff-eval.md)。

## 6. 接口

### 6.1 端侧

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/v1/infer` | 协同推理主入口 |
| POST | `/v1/feedback` | 采纳/纠正反馈 |
| GET | `/v1/stats` | 指标、队列、预算、熔断、反馈统计 |
| GET | `/v1/outbox` | 最近离线队列条目 |
| POST | `/v1/outbox/flush` | 手动触发队列重放 |
| GET | `/metrics` | Prometheus 文本指标 |
| GET | `/healthz` | 健康检查 |
| POST | `/v1/admin/config` | 调整阈值/离线模式/预算（管理面） |
| POST | `/v1/admin/refresh-adapter` | 重新拉取适配器版本 |

`POST /v1/infer` 请求：

```json
{
  "prompt": "设备报 E-217 主轴温度异常，应该怎么排查？",
  "task": "fault_diagnosis",
  "max_tokens": 256,
  "temperature": 0.0,
  "deadline_ms": 8000,
  "require_cloud": false,
  "metadata": {"line": "A2", "station": "S07"}
}
```

响应关键字段：

```json
{
  "request_id": "f75e5e5c7cf2...",
  "text": "E-217 表示主轴冷却液流量不足……",
  "source": "cloud",
  "confidence": 0.88,
  "escalated": true,
  "escalation_reason": "low_confidence",
  "edge_text": "E-217 建议先重启设备，观察是否恢复。",
  "edge_confidence": 0.40,
  "edge_latency_ms": 35.1,
  "cloud_latency_ms": 522.7,
  "degraded": false,
  "trace": [
    "端侧 mock-edge-0.6b 用时 35.1ms，置信度 0.40（logprob+heuristic）",
    "触发升级：low_confidence（阈值 0.62）",
    "云端 mock-cloud-32b 用时 522.7ms，置信度 0.88",
    "最终采用云端增强结果"
  ]
}
```

### 6.2 云端

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/v1/enhance` | 大模型增强（含端侧草稿） |
| POST | `/v1/feedback` | 接收端侧转发的反馈 |
| POST | `/v1/dataset/export` | 导出难例 SFT 数据集 |
| GET | `/v1/registry` | 适配器注册表 |
| GET | `/v1/registry/current/{device_id}` | 设备应运行的版本 |
| POST | `/v1/registry/publish` | 发布新版本（灰度） |
| POST | `/v1/registry/promote` | 灰度转正 |
| POST | `/v1/registry/rollback` | 回滚 |
| GET | `/v1/stats` | 云端指标与反馈统计 |
| GET | `/metrics` | Prometheus 文本指标 |

## 7. 部署

### 7.1 端侧

- 硬件：工控机/边缘盒子，CPU 可运行 0.5B–1.5B 量化模型；有 GPU 时可上 3B–7B。
- 进程：`edge-cloud serve-edge --config configs/edge-cloud/example.json`
- 建议 systemd 托管，失败自动重启；模型与适配器目录只读挂载。
- 网络：仅需出站访问云端 443；不开放入站端口给产线外网。
- 本地端口仅绑定 `127.0.0.1` 或产线内网，由网关做 mTLS 与鉴权。

### 7.2 云端

- 推理：vLLM / SGLang 暴露 OpenAI 兼容接口；`kind: "openai"` 直接接入。
- 服务：`edge-cloud serve-cloud --config ...` 或容器化后由 K8s 托管。
- 存储：反馈 JSONL 落对象存储；注册表落数据库（当前为 JSON 文件，单写者）。
- 训练：消费导出的 SFT JSONL，产出 ShadowPEFT / LoRA 适配器，登记到注册表。

### 7.3 配置

示例见 `configs/edge-cloud/example.json`。密钥不写入配置文件，云端适配器从 `OPENAI_API_KEY` 读取；生产环境应改为密钥管理服务。

## 8. 可靠性与安全

已实现：

- 端侧本地优先、离线可用；
- 云端超时、熔断、预算、截止时间保护；
- 离线队列持久化与重放；
- 反馈本地留存、云端转发失败不阻塞；
- 适配器版本化、灰度、回滚；
- 请求链路可解释（`trace`）与 Prometheus 指标。

项目期必须补齐：

- mTLS / API 鉴权 / 租户隔离；
- 反馈与日志脱敏（工单号、人员、工艺参数按甲方规则）；
- 多实例部署下的共享存储与幂等（当前 JSONL 为单写者）；
- 适配器签名与完整性校验；
- 审计日志留存与导出；
- 云端调用重试与指数退避（当前由熔断与队列兜底）；
- 真实模型的超时/显存/并发压测。

## 9. 观测与验收指标

| 指标 | 目标（建议） | 测量方式 |
| --- | --- | --- |
| 端侧直出 P95 | ≤ 300 ms | `/metrics` 的 `edge_latency_ms` |
| 端到端 P95 | ≤ 3 s（升级请求） | `request_latency_ms` |
| 升级率 | 10%–30%（按任务） | `escalation_total / requests_total` |
| 云端成功率 | ≥ 99% | `cloud_success_total / cloud_attempt_total` |
| 反馈采纳率 | ≥ 85% | 云端反馈统计 |
| 断网可用性 | 100% 本地可答 | 断网演练 |
| 恢复同步 | 断网恢复后 5 min 内清空队列 | `outbox.pending` |
| 适配器回滚 | ≤ 5 min | 注册表操作演练 |
| 难例闭环 | 每月至少一次训练+灰度 | 训练记录与注册表历史 |

具体目标需在需求确认阶段与甲方按工位、任务类型分别签字确认。

## 10. 六个月实施计划

| 阶段 | 时间 | 主要工作 | 交付物 |
| --- | --- | --- | --- |
| M1 需求与基线 | 第 1 月 | 工位调研、任务清单、评测集 500–1000 条、指标基线 | 《需求规格说明书》、评测集、基线报告 |
| M2 双端骨架 | 第 2 月 | 端侧代理、云端增强、协议、配置、日志 | 可运行的双端服务、接口文档 |
| M3 协同策略 | 第 3 月 | 置信度标定、升级策略、预算/熔断、离线队列 | 策略配置、压测报告 |
| M4 数据飞轮 | 第 4 月 | 反馈采集、难例导出、SFT/ShadowPEFT 训练、注册表 | 训练流水线、首个适配器 |
| M5 产线试点 | 第 5 月 | 单工位试点、灰度、时延与准确率评估、安全加固 | 试点报告、灰度记录 |
| M6 验收与移交 | 第 6 月 | 全量部署、培训、运维手册、源代码与文档移交 | 验收报告、培训材料、移交清单 |

工作量建议：需求与数据 15%、端侧 20%、云端 20%、协同策略 15%、飞轮与训练 15%、测试部署与文档 15%。

## 11. 交付物清单

- 可运行的端侧代理与云端服务（源代码、构建脚本、配置模板）；
- 《需求规格说明书》《系统设计说明书》《接口文档》《部署运维手册》《测试报告》；
- 产线评测集与基线报告；
- 至少一个由难例训练的适配器及灰度/回滚记录；
- 培训材料与操作视频（按合同约定）；
- 知识产权按合同第五条执行。

## 12. 风险与对策

| 风险 | 影响 | 对策 |
| --- | --- | --- |
| 端侧模型能力不足，升级率过高 | 云端成本超预算 | 先做任务分层；对高频任务蒸馏/微调；设置预算与降级提示 |
| 云端网络不稳定 | 时延抖动、请求失败 | 本地优先、熔断、离线队列、厂区数据中心部署 |
| 置信度校准偏差 | 该升级不升级/过度升级 | 用产线标注集做阈值标定；按任务独立阈值；持续回收反馈 |
| 数据不合规 | 无法上线 | 最小化上传、字段脱敏、保留策略、甲方审批 |
| 适配器退化 | 端侧效果下降 | 灰度、离线评测、自动回滚、版本冻结 |
| 真实模型与模拟差异 | 指标不达预期 | M2 起接入真实模型；压测与影子流量；预留硬件余量 |

## 13. 代码地图与运行

```text
src/edge_cloud/
  protocol.py       # 请求/响应/反馈协议
  confidence.py     # 置信度评估
  backends.py       # 模拟模型 / HF / OpenAI 兼容后端（含流式交接）
  router.py         # 升级路由、预算、熔断、离线队列重放
  handoff.py        # 交接原语：特征 / 决策 / 结果
  policies.py       # probe / oracle 交接策略
  probe.py          # probe 产物加载、拟合与打分
  features.py       # 逐 token base/shadow 特征提取
  boundaries.py     # 语义边界检测
  continuation.py   # 续写拼接与重复前缀检测
  cost.py           # 端侧 / 云端 token 与调用成本估算
  handoff_eval.py   # exp9 四策略评测台与曲线
  outbox.py         # 断网续传
  flywheel.py       # 反馈库与 SFT 导出
  registry.py       # 适配器灰度与回滚
  edge_server.py    # 端侧 HTTP 服务
  cloud_server.py   # 云端增强服务
  metrics.py        # 指标与 Prometheus 文本
  web/console.html  # 产线控制台
  __main__.py       # CLI

scripts/
  exp9_handoff_eval.py  # exp9 四策略评测与候选导出
  exp9_fit_probe.py     # 离线校准 probe
```

运行演示（无需 GPU，使用模拟模型）：

```bash
PYTHONPATH=src .venv/bin/python -m edge_cloud demo --port 8080 --cloud-port 9100 --reset
# 打开 http://127.0.0.1:8080/
```

运行测试：

```bash
.venv/bin/python -m pytest tests/edge_cloud -q
.venv/bin/python -m ruff check src/edge_cloud tests/edge_cloud
```

接入真实端侧模型：把 `edge.backend.kind` 改为 `transformers`，填写 `model_path`，可选 `shadow_ckpt` 指向 ShadowPEFT 适配器目录。接入真实云端：把 `cloud.backend.kind` 改为 `openai`，填写 `base_url` 与 `model`，密钥走环境变量。

## 14. 与 ShadowPEFT 的关系

ShadowPEFT 提供「小影子网络挂在大基座上」的参数高效适配能力。在本系统中它有两个落点：

1. **端侧适配器**：用云端难例训练 ShadowPEFT 适配器，挂在端侧小模型上，按版本灰度下发；
2. **云端训练侧**：用大模型作为教师，蒸馏端侧小模型/影子网络，降低端侧升级率。

本参考实现只依赖 `ShadowPeftModel.from_pretrained` 的加载接口，训练流程沿用仓库既有的 `scripts/exp*` 与 `shadow_lab`，不在本次代码范围内重复实现。
