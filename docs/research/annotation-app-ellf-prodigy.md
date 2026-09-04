# 图像生成 A/B 盲测标注应用：Ellf 与 Prodigy 产品研究

- 对应票据：`RES-001`
- 研究日期 / 来源访问日期：2026-09-04
- 研究范围：仅 Ellf 与 Prodigy 官方网站、官方文档和官方在线演示
- 结论状态：Research Spike 完成，仍须人工评审；评审通过前不得解除 `ENG-005` 的阻塞

## 结论

推荐首期自建一个小而专用的本地标注应用，同时提供“Prodigy 风格 JSONL”导入导出适配层，但不依赖、不内嵌也不复刻 Prodigy 或 Ellf。

真正值得采用的是两者共同体现的工作流原则：任务与会话是一等对象、一次只做一个明确判断、键盘优先、答案分批后台提交且可显式保存、允许跳过、多人覆盖率可配置、原始答案与复核结论分离、数据可导出、自托管数据面。Ellf 进一步展示了任务分配、实时进度、不可原地改写的审计记录，以及按角色限制访问的协作层。[Ellf 平台](https://beta.ellf.ai/docs/platform)；[Ellf 任务与会话](https://beta.ellf.ai/docs/tasks-actions-agents)；[Prodigy 数据流](https://prodi.gy/docs#recipes-and-data-flow)；[Prodigy 多人路由](https://prodi.gy/docs/task-routing)（均访问于 2026-09-04）

但两者都没有给出可直接满足本项目全部要求的现成方案。Prodigy 的 `compare` 虽可比较图像、随机左右语义并保存映射，却已被官方标记为将弃用，且官方 `review` 不支持 `compare`；新项目应使用通用 `choice` 或自定义界面。Ellf 目前仍是 private beta，功能与 API 可能变化。因此，图像身份脱敏、同 prompt/seed 配对、稳定随机左右、多轴原因、质检重复题、冻结后去盲和统计导出都应由本项目自己定义。[Prodigy interfaces：`compare` 与 `review`](https://prodi.gy/docs/api-interfaces#compare)；[Ellf private beta 声明](https://beta.ellf.ai/docs)（访问于 2026-09-04）

本文使用以下标签严格区分证据与设计判断：

- **来源事实**：官方资料明确写出或官方演示明确展示。
- **项目推断**：结合 Mage-Flow NVFP4 实验目标作出的设计选择，不表示 Ellf/Prodigy 已实现或推荐该细节。
- **待验证**：官方公开资料未覆盖，实施或上线前仍需验证。

## 官方能力与本项目映射

| 主题 | 来源事实 | 本项目推断 |
|---|---|---|
| 任务 / 配方 | Prodigy recipe 是定义输入 stream、界面、存储和回调的 Python 工作流；Ellf task 是 recipe 的具体配置实例，可启动、停止、分配人员并监控。[Prodigy 数据流](https://prodi.gy/docs#recipes-and-data-flow)；[Ellf Tasks](https://beta.ellf.ai/docs/tasks-actions-agents)（访问于 2026-09-04） | 将 A/B 协议冻结为 `experiment -> task -> comparison -> assignment -> answer_event`。生成作业和标注任务是两个系统，标注服务不得加载 Mage 模型。 |
| 流式队列 | Prodigy 的 stream 是 Python generator；Web 应用按批拉取问题并在后台回传答案。Ellf 会为打开标注 UI 的用户创建 session，超时后释放未完成队列。[Prodigy 数据流](https://prodi.gy/docs#recipes-and-data-flow)；[Ellf 会话](https://beta.ellf.ai/docs/tasks-actions-agents)（访问于 2026-09-04） | 使用数据库原子租约领取，而不是进程内 generator 作为真相源。租约到期只能释放未提交 assignment；已提交答案永不重新计为新票。 |
| 单题、跳过 | Prodigy 以 accept/reject/ignore 的单题流工作；Ellf 建议不确定时使用 `ignore`，随后筛选这些样本分析边界情况。[Prodigy 101](https://prodi.gy/docs)；[Ellf 会话](https://beta.ellf.ai/docs/tasks-actions-agents)（访问于 2026-09-04） | `平局` 是有效偏好结果，`跳过` 是无法可靠判断或样本损坏，两者不得合并。默认一屏一对图，确认提交后立即进入下一对。 |
| 快捷键 | Prodigy 官方默认键包括 accept=`a`、reject=`x`、ignore=`space`、undo=`backspace/delete`、save=`Cmd/Ctrl+S`，choice 选项可用数字键且主操作键可配置。[Prodigy Web App](https://prodi.gy/docs/api-web-app#key-shortcuts)（访问于 2026-09-04） | 首期同时支持按钮和 `1=左胜`、`2=平局`、`3=右胜`、`0=跳过`、`Backspace=撤销`、`Cmd/Ctrl+S=保存`。数字提示常驻显示；输入焦点进入备注框时禁用决策快捷键。 |
| 撤销 / 保存 | Prodigy 将刚回答的题放在客户端 outbox，可从历史或 Backspace 撤销；答案按批后台保存，也可显式保存。Ellf 周期性后台保存，最近答案仅留客户端以支持撤销，不写浏览器 local storage；断网恢复后可再次提交。[Prodigy outbox](https://prodi.gy/docs)；[Ellf 保存与恢复](https://beta.ellf.ai/docs/tasks-actions-agents)（访问于 2026-09-04） | UI 显示 `待提交 / 已保存 / 保存失败`。撤销已保存答案时追加一个 superseding event，不原地 UPDATE。MVP 不承诺关闭标签页后的离线恢复；离开时如有待提交答案必须警告。 |
| 图像 A/B | Prodigy `choice` 选项可渲染图像，多图选项会排列为列；`compare` 支持图像并随机分配“assumed correct”语义、保留映射，但官方建议新项目改用 `choice`/`diff`。[Prodigy Computer Vision](https://prodi.gy/docs/computer-vision#classification-choice-images)；[Prodigy `compare`](https://prodi.gy/docs/api-interfaces#compare)（访问于 2026-09-04） | 用专用双栏组件或自建 `choice` 等价物。左右交换按 `experiment_id + comparison_id + annotator_id + presentation_version` 确定，服务端可复现；客户端永远收不到实验臂、模型名或原始文件路径。 |
| 自定义界面 | Prodigy 可用 blocks 组合已有界面并加入 HTML/CSS/JavaScript；Ellf 可运行自定义 recipe，也可把自有浏览器应用作为 cluster service。[Prodigy Custom Interfaces](https://prodi.gy/docs/custom-interfaces)；[Ellf Platform](https://beta.ellf.ai/docs/platform)（访问于 2026-09-04） | “借鉴可组合工作流”不等于复制其 DOM、CSS、JS、按钮形状、配色、截图或布局。应用使用本项目自己的设计系统和组件代码。 |
| 多人路由 | Prodigy 支持零、全量或部分重叠，并允许自定义 task router；若要求精确人数，官方建议预先声明 allowed sessions。任务哈希可用于稳定分配。[Prodigy Task Routing](https://prodi.gy/docs/task-routing)（访问于 2026-09-04） | 每对图预注册 `target_votes`，assignment 对 `(comparison_id, annotator_id)` 唯一。正常题禁止同一标注者重复计票；隐藏重复题使用不同 `qc_group_id` 且不进入主结果。 |
| 数据集 / 会话 | Prodigy dataset 是命名任务集合，session 是一次标注会话；同一 dataset 可有多次或命名多人 session。Ellf 任务页可查看正在进行和已完成的 session 与标注数据。[Prodigy 术语](https://prodi.gy/docs#recipes-and-data-flow)；[Ellf Annotation](https://beta.ellf.ai/docs/platform)（访问于 2026-09-04） | dataset 表示一次冻结协议下的样本集合，session 表示登录到退出/超时的工作时段，二者不能互相替代。答题者使用不可逆 pseudonymous ID 导出。 |
| 标识与去重 | Prodigy 为输入和问题分别建立 `_input_hash`、`_task_hash`，用于区分相同输入上的不同问题、去重与复核聚合。[Prodigy Loaders](https://prodi.gy/docs/api-loaders#hashing)；[Prodigy Review](https://prodi.gy/docs/review)（访问于 2026-09-04） | 本项目用稳定 UUID 与 SHA-256 作为规范标识；Prodigy 整数 hash 只在适配导出时生成，不能作为长期主键。pair identity 对候选顺序不敏感，presentation identity 对左右顺序敏感。 |
| 复核 | Prodigy `review` 汇集多人的版本并生成一个 master annotation；可自动接受无冲突项，但不支持 `compare`。Ellf 内置 review task，并且编辑会复制新记录、禁用原记录而非原地改写。[Prodigy Review](https://prodi.gy/docs/review)；[Ellf 审计式编辑](https://beta.ellf.ai/docs/tasks-actions-agents)（访问于 2026-09-04） | 人类偏好不是单一“真值”：review 只能产生独立的 `review_event`（保留 / 排除 / 需重采 / 仲裁偏好），不得覆盖或替代原始投票。复核者在提交自己的初判前不看模型身份或他人票数。 |
| 导出 | Prodigy `db-out` 输出 JSONL；Ellf dataset 也以 Prodigy JSONL 导出，并可连资产一起备份。[Prodigy `db-out`](https://prodi.gy/docs/recipes#db-out)；[Ellf dataset export](https://beta.ellf.ai/docs/assets-files)（访问于 2026-09-04） | 提供两个不可混淆的导出：盲态分析包和冻结后的去盲分析包。CSV 仅为扁平统计便利格式，JSONL 是完整审计格式。 |
| 质检 | Prodigy 提供 IAA 指标，并建议在早期使用有代表性的重叠子集、让标注者独立作答、结合上下文解释指标；其图像选择示例还建议混入已知不应选择的选项以评估标注质量。[Prodigy Metrics](https://prodi.gy/docs/metrics)；[Prodigy image choices](https://prodi.gy/docs/computer-vision#classification-choice-images)（访问于 2026-09-04） | MVP 支持覆盖率、左右偏置、跳过率、重复题自洽和冲突队列。不得用单个阈值自动剔除标注者；所有排除规则在看结果前预注册，并保留包含/排除两套敏感性分析。 |
| 隐私部署 | Prodigy 可完全本地/内网运行且不“phone home”，可 air-gap；公开 URL 需认证。Ellf 的实际资产、模型、代码、密钥、标注数据库在用户自管 cluster，但其托管端仍保存资源名、ID、路径、统计与用户/组织元数据。[Prodigy Deployment](https://prodi.gy/docs/deployment)；[Ellf Architecture](https://beta.ellf.ai/docs#architecture)（访问于 2026-09-04） | MVP 默认只监听 localhost；多人使用时部署在受控内网，强制 TLS/认证。图片 URL 使用不含身份信息的 opaque ID，禁止第三方字体、分析脚本或外链媒体。 |

## 三种实现路径

### A. 自建应用（推荐）

**项目推断**：这是 `ENG-005` 的默认路径。

优点是可以原生实现冻结 manifest、服务端盲态、同 prompt/seed 约束、双图同步缩放、append-only 决策、独立复核、两阶段导出和完全本地部署；也没有闭源运行时或开发者 seat 依赖。代价是要自行实现并测试并发租约、身份认证、备份和可访问性。

自建不意味着重做通用标注平台。MVP 只实现本实验的 A/B 图像协议，不实现 recipe SDK、模型在环、通用 NER/框选、agent 标注或云项目管理。

### B. 合法接入 Prodigy（条件可行，不作为默认依赖）

**来源事实**：Prodigy 是付费、非开源、自托管软件。个人或公司开发者需要适当 license；条款禁止重新分发 Prodigy、修改版或其部分，禁止把含其源代码的应用提供给第三方，也限制 public-facing hosting。OIDC SSO 仅为 company license 功能。[Prodigy 101](https://prodi.gy/docs)；[Prodigy Terms](https://prodi.gy/docs/terms)；[Prodigy SSO](https://prodi.gy/docs/sso)（访问于 2026-09-04）

**项目推断**：只有在组织已经取得适当 license、法务/采购确认部署及人员范围后，才可另建私有适配包：用 `choice` + 自定义 blocks/recipe 展示双图，以 named sessions 和 task router 分配，用独立 review recipe 复核。不得把 Prodigy wheel、源代码、前端 bundle、截图或衍生实现提交到本仓库。由于官方 `compare` 将弃用且 review 不支持它，不应以 `compare` 作为长期合同。

### C. 兼容 Prodigy 风格的数据协议（推荐作为可选适配器）

**项目推断**：规范合同以本项目 JSON Schema 为准，同时提供无 Prodigy 依赖的 JSONL adapter。它可映射常见字段，如 `_input_hash`、`_task_hash`、`_view_id="choice"`、`options`、`answer`、`_session_id`，便于用户在有合法 Prodigy 环境时自行导入；这不等于承诺与所有 Prodigy 版本双向无损兼容。

adapter 不使用 Prodigy 商标作为本产品名称，不复制官方示例数据、前端资产或专有实现。是否发布“兼容 Prodigy”描述应在发布前再做许可/商标审查。

## 角色和权限

以下均为**项目推断**。Ellf 的 Administrator / Developer / Annotator 三角色证明“标注者只访问已分配任务”的产品模式可行，但本项目按实验职责重命名并收紧权限。[Ellf Roles and Permissions](https://beta.ellf.ai/docs/platform#user-management)（访问于 2026-09-04）

| 角色 | 可以 | 不可以 |
|---|---|---|
| 标注者 `annotator` | 阅读冻结版指南；领取已分配题；盲态查看 prompt 和 A/B 图；选择左/平/右/跳过及少量原因；撤销本会话近期答案；查看自己的保存状态与剩余量 | 查看模型名、实验臂、真实文件路径、他人答案、臂级分数；修改 manifest；去盲；删除原始答案 |
| 复核者 `reviewer` | 领取冲突、跳过、质检异常和人工 flag 队列；在盲态下给出独立初判；随后查看匿名理由并追加复核处置 | 覆盖原始投票；查看未冻结实验的身份映射；改变覆盖率和排除规则；把仲裁票混作普通票 |
| 实验负责人 `experiment_owner` | 导入并校验 manifest；冻结协议；配置覆盖率、路由和质检规则；分配人员；查看非去盲进度；冻结答案；导出盲态数据；冻结后执行一次性去盲导出 | 修改已冻结 manifest；原地编辑答案；在收集中查看臂级胜率后择机停止；以管理员身份补投普通票 |

至少还要有系统层面的 append-only 审计主体 `service`，但它不是人类角色。

## 标注协议决策

以下清单逐项标记事实与推断，实施时可直接转为产品验收项。

1. **来源事实**：Prodigy 的 `choice` 可以显示图像选项，数字键可快速选择；Ellf/Prodigy 都支持单题流与 skip/ignore。[Prodigy Choice](https://prodi.gy/docs/api-interfaces#choice)；[Ellf 会话](https://beta.ellf.ai/docs/tasks-actions-agents)（访问于 2026-09-04）
2. **项目推断**：默认决策必须是四态：`left / tie / right / skip`。`tie` 表示两图同等，`skip` 表示没有形成可用判断，例如图像损坏、加载失败、提示词无法理解或两图都不可评。
3. **项目推断**：每题固定显示同一 prompt、seed、尺寸和生成协议的两张图；一旦任一生成参数不匹配，manifest 校验失败，不进入队列。
4. **项目推断**：左右位置在 assignment 层稳定随机，而不是 comparison 层全局固定。不同标注者可见不同顺序；刷新和断线重连不改变同一 assignment 的顺序。
5. **项目推断**：先选择主结果，再可选原因，最后把两者作为一个 answer event 原子提交；`skip` 可直接提交。MVP 原因码限定为 `prompt_following`、`visual_quality`、`artifact_or_structure`、`text_accuracy`、`other`，并带 `reason_target=left/right/both/none`。没有文字目标时隐藏 `text_accuracy`。是否允许“无原因的一键快速提交”留给 pilot 决定。
6. **来源事实**：Prodigy 官方提醒自由文本容易造成不一致，宜只在无法预定义标签时使用。[Prodigy Text Input](https://prodi.gy/docs/api-interfaces#text_input)（访问于 2026-09-04）
7. **项目推断**：普通题不要求备注；只有 `other`、flag 或 reviewer 处置才显示可选短备注。备注必须经过长度限制和导出转义。
8. **来源事实**：Prodigy 提供帮助弹窗、进度/历史、flag、可配置快捷键；Ellf 展示 task、session、annotator 和 annotation 计数。[Prodigy Web App](https://prodi.gy/docs/api-web-app)；[Ellf Tasks](https://beta.ellf.ai/docs/tasks-actions-agents)（访问于 2026-09-04）
9. **项目推断**：标注页只显示本人的总体进度，不显示实时胜率或其他标注者结果。指南一页可读、随时按 `H` 打开，版本号写入每条 answer event。
10. **项目推断**：双图必须同尺寸容器、同缩放策略，支持点击原图和同步 zoom/pan；移动端不是 MVP，桌面端最小宽度不足时阻止答题而不是上下堆叠，以免产生位置/尺度混杂。
11. **待验证**：官方资料未证明 Ellf/Prodigy 自带本项目需要的左右偏置分析、重复题内一致性、paired bootstrap CI、Bradley–Terry tie 处理、冻结后去盲、WCAG 合规或中文 A/B 指南。不得把这些写成其现成功能。

## 低保真流程图

```text
实验负责人
  │ 导入 prompts + generation records + 图像 SHA-256
  ▼
[Manifest 校验]
  ├─ prompt / seed / size / sampler 不一致 ──> 拒绝并给出清单
  └─ 通过
       │ 生成 comparison、覆盖率、QC 计划和密封 blind map
       ▼
    [冻结实验]
       │
       ├──────────────> [盲态进度面板：只显示覆盖率/保存失败/队列健康]
       │
       ▼
标注者登录 ──> 领取有时限 assignment ──> 单屏同 prompt A/B
                                      │
                  1 左胜 / 2 平局 / 3 右胜 / 0 跳过
                                      │
                                      ▼
                         可选原因 / flag / Enter 确认
                                      │
                            后台保存 + 显式保存
                                      │
                         Backspace 追加 superseding event
                                      │
                                      └────────> 下一题

覆盖完成 ──> 冲突 / skip / QC 异常队列 ──> 复核者盲态初判
                                             │
                                             ▼
                                      追加 review event
                                             │
                          ┌──────────────────┴───────────────┐
                          ▼                                  ▼
                   盲态冻结导出                      实验负责人确认冻结
                                                             │
                                                             ▼
                                                     一次性去盲导出
                                                             │
                                                             ▼
                                         胜/平/跳过、偏置、IAA、BT、paired CI
```

### 标注页线框

```text
┌──────────────────────────────────────────────────────────────────────┐
│ 进度 124/500    ● 已保存                         H 指南   Ctrl+S 保存 │
├──────────────────────────────────────────────────────────────────────┤
│ Prompt: “……”                                                     │
├───────────────────────────────┬──────────────────────────────────────┤
│                               │                                      │
│          [ 图像 LEFT ]        │             [ 图像 RIGHT ]           │
│          点击放大 / 同步缩放   │             点击放大 / 同步缩放       │
│                               │                                      │
├───────────────────────────────┴──────────────────────────────────────┤
│ [1 左胜]  [2 平局]  [3 右胜]  [0 跳过]  [Enter 确认]  [⌫ 撤销]       │
│ 可选原因：□提示遵循 □视觉质量 □伪影/结构 □文字正确性 □其他            │
└──────────────────────────────────────────────────────────────────────┘
```

## 数据合同草案

### 设计原则

以下均为**项目推断**：

- 规范记录使用有版本的 JSON Schema；导入后生成稳定 UUID，不能依赖文件名或数据库自增 ID。
- manifest、answer 和 review 是不可变事件；更正通过 `supersedes_event_id` 表达。
- client payload 与 server-only blind map 分离。客户端媒体 URL、DOM、日志和错误信息都不得出现 arm、model、checkpoint、quant method 或真实路径。
- 所有时间使用 UTC RFC 3339；所有图片记录 SHA-256、MIME、像素尺寸和字节数。
- `comparison_id` 表示无序候选对；`assignment_id` 表示某标注者看到的一次有序呈现。
- 原始投票、QC 判定、review 处置和最终统计分别保存，任何导出都不可静默丢弃原始投票。

### 规范实体

```json
{
  "schema_version": "shadow-ab/0.1",
  "experiment": {
    "experiment_id": "exp_uuid",
    "protocol_sha256": "...",
    "dataset_id": "mage-flow-human-test-v1",
    "split": "test-human",
    "guideline_version": "1.0.0",
    "target_votes": 3,
    "created_at": "2026-09-04T00:00:00Z",
    "frozen_at": "2026-09-04T00:10:00Z"
  },
  "comparison": {
    "comparison_id": "cmp_uuid",
    "prompt_id": "prompt_uuid",
    "prompt_text": "opaque-to-export-if-required",
    "seed": 42,
    "width": 1024,
    "height": 1024,
    "generation_protocol_sha256": "...",
    "candidate_asset_ids_unordered": ["asset_uuid_1", "asset_uuid_2"],
    "qc_kind": "none",
    "qc_group_id": null
  },
  "asset": {
    "asset_id": "asset_uuid_1",
    "sha256": "...",
    "mime_type": "image/png",
    "width": 1024,
    "height": 1024,
    "byte_length": 1234567,
    "storage_key": "server-only/opaque-key"
  },
  "assignment": {
    "assignment_id": "asn_uuid",
    "comparison_id": "cmp_uuid",
    "annotator_id": "pseudonymous_uuid",
    "presentation_version": 1,
    "left_asset_id": "asset_uuid_2",
    "right_asset_id": "asset_uuid_1",
    "lease_token_sha256": "...",
    "leased_at": "2026-09-04T01:00:00Z",
    "lease_expires_at": "2026-09-04T01:15:00Z"
  },
  "answer_event": {
    "event_id": "evt_uuid",
    "idempotency_key": "client-generated-uuid",
    "assignment_id": "asn_uuid",
    "session_id": "session_uuid",
    "annotator_id": "pseudonymous_uuid",
    "decision": "left",
    "reason_codes": ["prompt_following"],
    "reason_target": "right",
    "comment": null,
    "flagged": false,
    "guideline_version": "1.0.0",
    "client_version": "git-sha-or-release",
    "created_at": "2026-09-04T01:02:03Z",
    "received_at": "2026-09-04T01:02:04Z",
    "supersedes_event_id": null
  },
  "review_event": {
    "review_event_id": "rev_uuid",
    "comparison_id": "cmp_uuid",
    "reviewer_id": "pseudonymous_uuid",
    "independent_decision": "tie",
    "disposition": "keep_raw_votes",
    "reason_code": "legitimate_disagreement",
    "created_at": "2026-09-04T02:00:00Z",
    "supersedes_review_event_id": null
  }
}
```

`blind_map` 必须是单独的 server-only 记录：

```json
{
  "experiment_id": "exp_uuid",
  "asset_id": "asset_uuid_1",
  "arm_id": "encrypted-or-access-controlled-arm-id",
  "generation_record_id": "gen_uuid",
  "revealed_at": null,
  "reveal_audit_event_id": null
}
```

### 状态和约束

- `experiment.status`: `draft -> frozen -> collecting -> answers_frozen -> revealed -> archived`，不得反向迁移。
- 唯一约束：正常题 `(experiment_id, comparison_id, annotator_id, qc_kind=none)` 最多一个有效 answer 链。
- 同一 `idempotency_key` 重试必须返回同一事件；payload 不同则报冲突。
- `decision` 只能为 `left | tie | right | skip`；`skip` 不计入 preference 分母但必须单独报告。
- 左右随机必须在服务端写入 assignment；根据 assignment 将 `left/right` 还原到 asset 和 arm，不能从客户端传回“模型 A/B”。
- 回答提交后即使租约过期也可用 idempotency key 安全重试；新 assignment 只有在确认没有已接收事件后才能领取。
- review 不改变 answer 的有效链；任何排除由显式 `disposition` 和预注册规则在分析阶段应用。

### Prodigy 风格 JSONL 适配草案

**来源事实**：Prodigy 的 `choice` 任务使用 `options`，提交后在 `accept` 中保存所选 option ID，并通过 `answer` 区分接受、拒绝或忽略；review 需要 input/task hashes、session/annotator ID 和 view ID。[Prodigy Choice](https://prodi.gy/docs/api-interfaces#choice)；[Prodigy Review](https://prodi.gy/docs/review)（访问于 2026-09-04）

**项目推断**：盲态导出可映射为：

```json
{
  "_input_hash": 123456789,
  "_task_hash": 987654321,
  "_view_id": "choice",
  "_session_id": "mage-ab-pseudonymous-user",
  "_annotator_id": "mage-ab-pseudonymous-user",
  "text": "prompt text",
  "options": [
    {"id": "LEFT", "image": "/media/opaque-left-token"},
    {"id": "TIE", "text": "Tie"},
    {"id": "RIGHT", "image": "/media/opaque-right-token"}
  ],
  "accept": ["LEFT"],
  "answer": "accept",
  "meta": {
    "schema_version": "shadow-ab/0.1",
    "experiment_id": "exp_uuid",
    "comparison_id": "cmp_uuid",
    "assignment_id": "asn_uuid",
    "seed": 42,
    "resolution": [1024, 1024]
  }
}
```

`skip` 映射为 `answer="ignore"`；自建合同中的 reason/revision/review 字段保留在命名空间对象中。`arm_id` 和真实路径绝不进入盲态 JSONL。

## MVP 与后续界线

### MVP：`ENG-005` 必须交付

- 冻结、校验和版本化 manifest；只接受同 prompt/seed/尺寸/生成协议的 pair。
- 本地 SQLite；受控本地资产根目录；opaque 媒体 URL；PNG/JPEG 的 SHA-256、MIME、尺寸校验。
- 三角色最小权限和认证；默认 localhost，内网部署文档化。
- 数据库原子租约、可配置精确覆盖人数、同人去重、稳定左右随机。
- 单题双栏、等尺寸、原图查看/同步缩放、四态选择、少量可选原因、键盘操作和帮助页。
- in-memory outbox、后台提交、显式保存、清晰保存状态、幂等重试、append-only 撤销、待提交离开警告。
- 冲突/skip/flag/QC 复核队列；review event 不覆盖原始答案。
- 隐藏重复题和 sentinel 的数据支持；报告完成率、保存失败、覆盖率、skip、左右偏置、重复题自洽和标注者重叠。
- JSONL 完整导入导出、CSV 扁平导出、盲态/去盲两阶段导出和 reveal 审计事件。
- 单元与并发测试覆盖脱敏、路径穿越、租约竞争、幂等、撤销、刷新后已确认答案恢复、导出可复现。

### 后续能力：另开票

- OIDC/SSO、邀请邮件、细粒度组织/项目管理、移动端和离线持久化。
- 主动学习/基于不确定性的动态路由、agent 预标注、在线模型在环；本项目人工分数仍不得反馈给 ES。
- 通用 recipe/plugin SDK、非 A/B 标注界面、框选/分割/视频/文本任务。
- 高级可访问性审计、国际化、多租户隔离、远程对象存储、审计日志归档策略。
- 实时 Bradley–Terry 看板。正式 test 收集过程中应禁止臂级实时结果，以免人为提前停止；冻结后再运行最终统计。
- 与合法 Prodigy 安装的运行时连接器。MVP 只需无依赖 JSONL adapter。

## 质检与分析边界

**来源事实**：Prodigy 官方强调 IAA 应在早期以代表性重叠样本迭代指南，标注者要独立作答，指标只是一种信号而不应成为目标；其 review 适合聚焦冲突。[Prodigy Metrics](https://prodi.gy/docs/metrics)；[Prodigy Review](https://prodi.gy/docs/review)（访问于 2026-09-04）

以下是**项目推断**：

- 在正式 `test-human` 前先建 `pilot-human`，用小批量检查说明是否能让标注者稳定区分 tie 与 skip、原因码是否足够、同步缩放是否可用。
- 每个正式 comparison 目标至少 3 个独立普通票；10% 左右隐藏重复题用于 intra-rater consistency，但重复题不增加该 comparison 的正式票数。
- sentinel 必须由实验负责人预注册且不依赖某个待比较模型必然更好。它用于发现没有加载、乱点或明显协议错误，不能自动判定标注者“无效”。
- 报告原始与排除后的 paired 结果、标注者 cluster bootstrap、tie 和 skip 的独立比例、左右位置偏置、按 prompt 类别分层结果；Bradley–Terry 的 tie 处理方式必须预注册。
- 复核只解决样本是否损坏、协议是否违规、skip 是否需要重采等数据治理问题。对于真实审美分歧，保留全部独立投票比强行制造 master label 更合适。

## 隐私、安全和许可边界

### 来源事实

- Prodigy 软件本身完全 self-hosted，官方称不会连接 Explosion 或第三方服务器，并可 air-gap；暴露到公共 URL 时官方建议启用认证。[Prodigy Deployment](https://prodi.gy/docs/deployment)；[Prodigy Terms / Privacy](https://prodi.gy/docs/terms)（访问于 2026-09-04）
- Ellf 是混合架构：资产、模型、代码、密钥和标注数据库位于用户 cluster；其服务保存协调所需的名称、ID、路径、统计和账户/组织元数据。隐私政策还列出匿名 usage telemetry 以及特定 subprocessors。[Ellf Architecture](https://beta.ellf.ai/docs#architecture)；[Ellf Privacy](https://beta.ellf.ai/privacy)（访问于 2026-09-04）
- Ellf 条款称用户保留自己数据、标注、模型和代码的权利，但 Ellf 服务的软件、模型、知识库、算法、文档和界面归 Explosion，并禁止逆向、提取源码以及用服务开发竞争产品。[Ellf Terms](https://beta.ellf.ai/terms)（访问于 2026-09-04）
- Prodigy 条款明确其许可为闭源、不可转让的个人/公司 seat；禁止重新分发、转授、提供修改版或其部分，以及把使用其源代码的应用交付第三方。[Prodigy Terms](https://prodi.gy/docs/terms)（访问于 2026-09-04）

### 项目约束

- 本研究只阅读公开官方资料，不安装、不购买、不逆向任何产品。
- 不复制 Ellf/Prodigy 的代码、bundle、图标、logo、截图、文案、配色、DOM 或视觉资产；不使用其品牌命名本应用。
- 只采用不可专有化的高层工作流思想，并以本项目自己的需求、schema、交互和实现表达。
- 若未来接入 Prodigy，必须在单独票中记录有效 license、开发者 seat、annotator 访问范围、部署边界和法务确认；本仓库不得分发 Prodigy。
- 服务图像前剥离或忽略 EXIF/文本块等可能暴露生成工具的元数据；响应使用 opaque URL、严格内容类型、`nosniff`、CSP 和路径根校验。
- 不把 prompt、图像、答案、注释或人员信息发给第三方分析、CDN、字体或错误追踪服务。备份与导出继承同一访问控制。

这不是法律意见。任何对外发布、商标兼容性声明或商业部署都需要项目所有者另行审查当时有效条款。

## 对 `ENG-005` 的具体修改建议

以下均为**项目推断**，供人工评审后改票；本 Research Spike 不直接修改票据。

### 目标与接口

1. 把“Prodigy 风格”从 UI 目标改为数据适配目标：规范接口是 `shadow-ab/0.1` JSON Schema，另提供可选 Prodigy-style JSONL adapter。
2. 明确生成器接口只接收冻结 manifest 和本地图片，不 import diffusers/Mage，也不请求 GB10 GPU。
3. 增加 `validate/freeze/reveal` 三个实验负责人动作；reveal 只能在 answers frozen 后执行，并写不可变审计事件。
4. 将领取接口定义为 lease：原子领取、到期回收、提交幂等、同人去重、目标覆盖人数明确。
5. 将答案接口定义为 append-only event；undo 是 supersede，不是 SQL UPDATE/DELETE。
6. 增加 blind client contract：客户端响应和媒体 URL 不含 arm/model/method/checkpoint/path；映射只存在 server-only blind map。
7. 复核接口必须产生独立 review event，保留全部 raw votes；复核者提交初判前保持盲态。

### 验收标准

建议在现有验收项上增加：

- manifest 若 prompt、seed、尺寸或生成协议不一致会拒绝入队，并输出可定位差异。
- 对同一 assignment 重复请求顺序恒定；跨 annotator 的左右分布在预设容差内；选择可准确还原到 asset/arm。
- 客户端 HTML、JSON、URL、日志、错误页和下载名均通过身份泄漏测试。
- 两个并发 worker 不能领取同一 `(comparison, annotator)`；过期租约、迟到提交与幂等重试有确定语义。
- 后台保存失败在 UI 可见；已确认答案刷新可恢复；未确认离开会警告；撤销后原事件仍可审计。
- 正常题、隐藏重复题和 sentinel 分开计数；同人重复普通票不进入主分析。
- raw、blind、revealed 三种导出均有 schema version、协议 hash、资产 hash 和导出 hash；同一冻结数据库重复导出字节稳定或语义稳定。
- review 无法修改原始 answer；分析可分别运行全量票和预注册排除规则。
- 标注者不能看到他人答案或臂级指标；实验负责人 collecting 期间也看不到臂级胜率。
- 本地默认部署无外部网络请求；路径穿越、symlink escape、恶意 MIME、超大图片和注释注入测试通过。
- 完成一个 `pilot-human` 可用性记录：新标注者只读一页指南即可解释 tie/skip 差别并完成指定任务。

### 非目标

建议补充：

- 不实现或复制 Ellf/Prodigy recipe API、通用标注平台、前端资产或品牌。
- 不承诺无合法 Prodigy license 的运行时兼容或安装。
- 不在 MVP 做实时臂级排名、主动路由、agent 预标注、移动端、离线持久化、SSO 或在线生成。
- review 不是把主观偏好强制压成唯一 ground truth；人工答案不进入 ES 奖励。

## 人工评审时必须决定

1. 原因码是否保持五项，还是首期只保留四态主选择以降低认知负担。
2. 正式题每对的 `target_votes`、隐藏重复比例和 sentinel 设计。
3. reviewer 是只做数据有效性处置，还是也给独立偏好；两种票如何在统计中分开。
4. 桌面最小分辨率、原图查看和同步 zoom/pan 的具体可访问性要求。
5. 去盲授权是否只给单一负责人，是否需要双人确认。
6. 是否需要发布 Prodigy-style adapter；若需要，发布前完成许可/商标复核。

## 官方来源索引

所有来源均于 2026-09-04 访问。

### Ellf

- [Introduction / Architecture](https://beta.ellf.ai/docs)
- [Platform、Annotation、Roles and Permissions](https://beta.ellf.ai/docs/platform)
- [Tasks、Annotation sessions、保存、审计式编辑](https://beta.ellf.ai/docs/tasks-actions-agents)
- [Recipes](https://beta.ellf.ai/docs/recipes)
- [Recipe API](https://beta.ellf.ai/docs/recipes-api)
- [Assets、datasets 与 JSONL 导出](https://beta.ellf.ai/docs/assets-files)
- [Modules（官方项目规划示例）](https://beta.ellf.ai/docs/modules)
- [Privacy Policy](https://beta.ellf.ai/privacy)
- [Terms and Conditions](https://beta.ellf.ai/terms)

### Prodigy

- [Prodigy 101、recipes/data flow、outbox、dataset/session](https://prodi.gy/docs)
- [Web Application、shortcuts、save、undo、history、instructions](https://prodi.gy/docs/api-web-app)
- [Task Routing 与 partial overlap](https://prodi.gy/docs/task-routing)
- [Annotation Interfaces：choice、compare、review、text input](https://prodi.gy/docs/api-interfaces)
- [Custom Interfaces：blocks、HTML/CSS/JavaScript](https://prodi.gy/docs/custom-interfaces)
- [Computer Vision：image choices 与数据存储](https://prodi.gy/docs/computer-vision)
- [Review](https://prodi.gy/docs/review)
- [Built-in Recipes：A/B、review、db-out](https://prodi.gy/docs/recipes)
- [Loaders、hashing 与 deduplication](https://prodi.gy/docs/api-loaders)
- [Database](https://prodi.gy/docs/api-database)
- [Annotation Metrics / IAA](https://prodi.gy/docs/metrics)
- [Deployment 与本地/air-gap](https://prodi.gy/docs/deployment)
- [OIDC SSO 许可边界](https://prodi.gy/docs/sso)
- [Terms、EULA 与 Privacy](https://prodi.gy/docs/terms)
