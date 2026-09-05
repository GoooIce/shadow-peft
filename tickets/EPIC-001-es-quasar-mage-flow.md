---
id: EPIC-001
type: epic
status: open
priority: P0
depends_on: []
hardware: GB10
---

# Mage-Flow-Turbo NVFP4 ES×QUASAR

## 目标

把 QUASAR 的 NVFP4 重建与进化策略（Evolution Strategies, ES）结合，冻结模型权重并直接搜索量化重建参数，以 Mage-Flow-Turbo 的真实生成质量为目标，在 GB10 上完成实验和验收。

## 固定边界

- 首期仅使用 NVFP4，不实现 INT4 或 2-bit。
- 模型与数据资产从 ModelScope 下载。
- GB10 是开发实验和正式验收硬件；Mac 只做单元测试与轻量冒烟。
- 主实验先量化 MMDiT。Qwen3-VL 和 VAE 是独立扩展票，不阻塞主实验。
- Mage QUASAR QAD 可在独立第一阶段更新 MMDiT latent 权重；通过阶段门并交接为 healed
  master 后权重冻结，第二阶段 ES 只更新量化重建参数。
- 人工评分只用于离线验证，不进入 ES 训练奖励。
- 不把 fake-quant、4-bit 存储或 BF16 反量化 GEMM 误报成原生 NVFP4 执行。

## 成功条件

- MMDiT 的 NVFP4 RTN、QUASAR 网格和 ES×QUASAR 可在同一冻结协议下比较。
- Mage QUASAR QAD 必须先达到绝对自动质量门并通过人工视觉门，ES 才能从 healed master
  启动；相对 Reward 提升不能替代绝对可用性。
- GB10 profiler 证据能区分原生 NVFP4 内核与模拟路径。
- 自动指标和盲测人工指标都有预注册的留出集、统计方法与结果。
- Qwen3-VL 与 VAE 的结论分别报告，不能用 MMDiT 结论代替。

## 子票

以 [tickets/README.md](README.md) 的票据图为准。

## 证据

- 2026-09-04：ModelScope 资产、BF16 基线与 MMDiT NVFP4 RTN/QUASAR-grid
  基线完成；自动奖励协议（ENG-003）进行中。
- 2026-09-04：自动奖励、ES 核心、sigma/alpha 校准与跨 prompt 梯度门禁均已完成；两次
  泛化门禁最终都为负，未冻结主实验配置，EXP-001 按停止规则保持 blocked。
- 2026-09-05：用户要求把跨 prompt 门禁从 800 张扩为精确 16,000 张；EXP-007 已按完整
  `es_train` 单轮 40 个互斥 episode 领取，主实验继续等待该高功效复核。
- 2026-09-05：EXP-007 在确认阶段被人工视觉否决并停止；三轮 triage 决定以 EXP-008 建立
  Mage QUASAR QAD → healed master → ES clip refinement 的两阶段替代流程。
