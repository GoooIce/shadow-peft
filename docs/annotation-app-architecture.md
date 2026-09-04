# Shadow A/B 标注应用架构

对应票据：`ENG-005`。

## 模块与 seam

核心是 `AnnotationStore` 深模块。其 interface 只有导入/冻结、身份认证、标注与复核的领取、
提交、撤销、进度、媒体读取和导出；SQLite 事务、稳定左右随机、租约、幂等、append-only 事件、路径约束和
盲态都隐藏在 implementation 内。HTTP 是薄 adapter，只接收/返回盲态 DTO，不能直接执行 SQL，
也不能接触 `arm_id` 或真实文件路径。

```text
manifest / CLI ─┐
                ├─> AnnotationStore interface ─> SQLite + controlled asset root
browser / HTTP ─┘               │
                                └─> blind DTO / opaque media bytes
```

唯一外部 seam：

- `import_manifest` / `freeze_experiment`
- `create_user` / `authenticate`
- `claim_next`
- `submit_answer` / `undo_answer`
- `progress`
- `read_media`
- `claim_review` / `submit_review` / `undo_review`
- `review_progress` / `read_review_media`
- `export_experiment`

所有时间由 caller 注入或由 interface 入口统一取得，方便并发和租约测试。manifest 与答案冻结后
只能追加事件；更正使用 superseding event。正常标注 adapter 永远拿不到去盲映射。

## 状态

```text
draft -> frozen -> collecting -> answers_frozen -> revealed -> archived
```

MVP 的标注领取只允许 `frozen/collecting`；首次领取把 `frozen` 原子推进到 `collecting`。
复核领取只允许 `answers_frozen`，队列仅包含被标记或当前有效答案分歧的 comparison。队列未清空
时状态机拒绝进入 `revealed`。去盲导出只允许 `revealed/archived`；盲态导出可提前用于审计。

## 安全不变量

- SQLite、manifest 和媒体文件只能位于显式配置的位置；`storage_key` 必须是资产根目录内的
  规范相对路径，拒绝符号链接逃逸和 `..`。
- 浏览器 DTO、opaque 媒体 URL、错误和盲态导出不含 arm/model/checkpoint/quant method、原始
  文件名或磁盘路径。
- assignment 的左右位置由 experiment/comparison/annotator/presentation 的稳定 SHA-256 决定，
  写入数据库后刷新不变。
- `idempotency_key` 相同且 payload 相同返回原 receipt；payload 不同报冲突。
- 撤销追加 `withdraw` event；任何答案事件不 UPDATE/DELETE。
- 复核者看不到原始投票和实验臂；复核裁决与撤销写入独立 append-only `review_events`。
- 正常题同一标注者只产生一个 assignment；QC 重复由不同 comparison 表达。
