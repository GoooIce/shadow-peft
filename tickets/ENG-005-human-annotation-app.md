---
id: ENG-005
type: engineering
status: in-progress
priority: P1
depends_on: [RES-001, ENG-003]
hardware: GB10
---

# 人类友好的盲测标注应用

## 目标

实现本地、自托管、可审计的图像生成 A/B 盲测应用。产品理念参考 ellf.ai 与 Prodigy，但最终接口和边界以 RES-001 的人工评审结论为准。

## MVP 范围

- 同 prompt、同 seed 图片并排，服务端稳定随机左右位置并隐藏实验身份。
- `左胜 / 平局 / 右胜 / 跳过` 主决策，按研究结论提供少量原因标签。
- 键盘优先、撤销、显式保存状态、断点恢复、进度和图片等比放大。
- 标注者、复核者和实验负责人三种角色；任务分配避免同人重复计票。
- task/input hash、不可变 manifest、session、append-only answer 和审计字段。
- SQLite 本地默认存储，JSONL/CSV 导入导出；图片仅从受控本地资产目录提供。
- 复核队列、重复题/质检题、一致性报告和去盲导出。

## 验收标准

- 自动测试覆盖左右随机化、身份脱敏、并发领取、幂等保存、撤销、恢复和导出。
- 关闭或刷新浏览器不会丢失已确认答案。
- 标注页面不显示路径、文件名、模型名或可推断实验臂的元数据。
- 预生成图片的标注服务不加载 Mage 模型，也不占用 GB10 GPU 执行生成。
- 试标人员能只看一页说明完成任务；试标反馈和修订记录留档。
- 输出可直接计算胜率、平局率、Bradley–Terry 排名、标注者一致性和 paired bootstrap CI。

## 后续能力

- 主动路由、agent 预标注和高级项目管理仅在 RES-001 证明有价值后另开票。

## 非目标

- 不复刻 ellf.ai/Prodigy 的代码、品牌或受限视觉资产。
- 不在应用内在线生成图片。
- 不把人工答案反馈给 ES。

## 证据

- 2026-09-04：用户在 RES-001 完成且收到进展后确认“继续”，据此通过人工评审门禁并领取
  本票；采用研究推荐的自建本地应用 + Prodigy 风格 JSONL 适配路线。
- 2026-09-04：完成首个可运行纵切：`AnnotationStore` SQLite 深模块、manifest/媒体校验、
  localhost HTTP adapter、Bearer 身份、稳定盲态左右、并发租约、幂等答案、append-only 撤销、
  盲态/去盲 JSON/JSONL/CSV 导出，以及键盘优先双图 UI（同步 zoom/pan、帮助、保存状态）。
  核心与 HTTP 自动测试通过，Chrome 实际渲染检查无控制台错误；标注包导入不加载 Torch/Mage。
- 2026-09-04：补充 blind/revealed 分析 adapter：skip/tie/左右偏置、重叠题 exact agreement、
  隐藏 repeat 自洽、Bradley–Terry log strength 和 comparison-level paired bootstrap CI；wheel
  构建验证包含全部静态资产，demo access 文件权限为 0600。
- 2026-09-04：完成复核纵切：冻结答案后自动生成“被标记/有效答案分歧”队列，reviewer 独立
  盲看并裁决；租约、幂等、append-only 撤销、盲态/去盲导出和 HTTP/UI 均贯通。未清空复核
  队列时禁止进入 revealed；`--review-ready` 可一条命令创建本地复核 demo。
- 2026-09-04：全仓回归 `376 passed, 18 skipped`，ruff、JavaScript 语法、lockfile 和 wheel
  构建检查通过；wheel 已验证包含 HTML/JS/CSS。试标协议见 `docs/annotation-pilot-protocol.md`。
- 尚未完成：真实人员 pilot 与反馈修订记录，因此本票保持 in-progress。
