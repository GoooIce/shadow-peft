---
id: ENG-003
type: engineering
status: done
priority: P0
depends_on: [ENG-001]
hardware: GB10
---

# 自动奖励与固定评测协议

## 目标

定义 ES 可消费、可重放且与最终人工验收隔离的自动奖励和数据划分。

## 工作范围

- 从 ModelScope 固定 prompt 池、奖励模型和评测资产的 revision 与许可证。
- 把 prompt 划分为 ES 训练轮换池、checkpoint 验证池、自动测试池、人工开发池和一次性人工测试池。
- 评估 HPS、CLIP、GenEval 及文字渲染指标的尺度、相关性、方差和失败模式。
- 定义奖励归一化、缺失/异常值处理、GenEval 使用位置以及 paired seed 协议。
- 预注册主指标、辅助指标、统计检验和等效区间。

## 验收标准

- 同一批图像重复评分结果满足确定性或已量化噪声界线。
- 不把 GenEval 的离散结果直接作为小批次逐 episode 硬门，除非有实验证据支持。
- 训练池和最终测试池在 prompt、seed 和派生样本上无泄漏。
- 奖励接口只接收图像与冻结元数据，不依赖候选名称。

## 非目标

- 不实现 ES 更新。
- 不收集人工评分。

## 证据

- 2026-09-04：ENG-002 验收后领取本票；从已通过 Mage 内容门控的 32 条
  quality golden dev set 开始标定接口与确定性，不把该开发集误作最终留出集。
- 2026-09-04：冻结 500 条协议为 `320 es_train / 60 checkpoint_val / 60 auto_test /
  30 human_dev / 30 human_test`，来源配额为 400 GenEval + 100 Qwen-Image-Bench
  Text Rendering；SHA 排序、prompt/seed 去泄漏和 paired-BF16 聚合已有单元测试。
- 2026-09-04：ModelScope HPS-v2.1、OpenCLIP H/14、Qwen-Image-Bench 资产已在
  goooinext 逐文件通过 SHA-256；gooo 上同一 BF16 图的 HPS/CLIP 重复评分最大差值均为 0。
- 2026-09-04：实现 `GenEval-compatible Transformers/ModelScope port`，保留官方判定
  规则但明确 `leaderboard_comparable=false`。gooo 的 Mask2Former `dog` 冒烟检测通过，
  两次完整判定完全一致；已记录 2 个已知 missing LayerNorm key、24 个可重建
  relative-position buffer，不接受未知 checkpoint 转换缺口。
- 2026-09-04：本地相关检查达到 30 passed / 2 CUDA-only skipped；两台 GB10 的前一轮
  共享核心测试均为 23 passed。751 条候选的全量 Qwen3-VL 内容门控仍在 goooinext
  运行，完成并冻结最终池及 BF16 数值标定后才能关闭本票。
- 2026-09-04：内容门控完成 751/751，747 通过、4 拒绝，accepted/audit SHA 分别为
  `b6c8af4a...365a` / `e72562a3...7db6`。池构建器会强制核验完整 run、候选、accepted
  和 audit，不能从未筛完的前缀建池。
- 2026-09-04：审计拒收了一个仅保证总配额、但 `human_dev/human_test` 未覆盖全部
  GenEval task 的 v1 池。修复为逐 split 分层选择后冻结 v2：500 个唯一 sample/seed，
  pool SHA `13620c73...07e29`，五路 split 均覆盖六类 GenEval。失败 v1 仅保留审计。
- 2026-09-04：32 图端到端奖励 smoke 在 gooo 通过：模型加载 39.20 秒、评分 2.50 秒，
  4 图重复 HPS/CLIP 最大绝对差均为 0；临时 calibration SHA `0c87308c...bd9a`。
  完整 320 图 BF16 生成仍在 goooinext 运行，临时常量不用于 ES。
- 2026-09-04：正式 320 图 BF16 标定完成。生成在 goooinext 耗时 1051.78 秒（load
  95.02 秒），首图重复 PNG 哈希一致；gooo 奖励模型 load 38.08 秒、320 图评分
  21.24 秒，8 图重复评分 HPS/CLIP 最大绝对差均为 0。冻结常量：HPS mean/std =
  `0.2800148512 / 0.0381578632`，CLIP mean/std =
  `0.3800367487 / 0.0586977619`；320 个 paired baseline 的 calibration SHA 为
  `7f3d4a26...e437b`。
- 2026-09-04：本票验收完成。文字 OCR 保持非奖励辅助诊断：必须先在 `human_dev`
  与人工文字正确性校准，且不得冒充 Qwen-Image-Bench 官方分数。
