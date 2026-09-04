---
id: ENG-004
type: engineering
status: done
priority: P0
depends_on: [ENG-002, ENG-003]
hardware: GB10
phase: 1
---

# Phase 1：ES×QUASAR 集成

## 目标

冻结模型权重，由 ES 搜索 MMDiT 的 NVFP4 clip 重建参数，以真实生成奖励更新搜索中心。

## 工作范围

- 以无约束 latent 表示 clip，经有界映射落到 `[0.30, 1.00]`，避免边界 clamp 偏置。
- 默认实现 antithetic `+ε/-ε`、seed 重放、只存扰动 seed 与标量奖励。
- 每个 episode 物化完整候选量化模型后生成；层间流式只用于构建，生成前不能丢弃已量化层。
- 支持 z-score 与 rank shaping，可配置种群、prompt 批次、sigma、alpha 和 checkpoint。
- 候选失败可恢复，主进程能从 checkpoint 精确继续。
- Phase 1 内先完成三代冒烟，再在独立子阶段扫描 sigma。

## 验收标准

- 相同 genome/seed 得到相同 clip、量化状态和生成输入。
- 零扰动与记录的 ES 有效初始化完全等价；若 QUASAR 最优值落在有界映射端点，须单独
  报告端点投影数量、最大位移及其相对原始 grid 基线的差异，不得把二者混称为完全等价。
- antithetic 配对、聚合方向和边界映射均有单元测试。
- 三代冒烟中个体奖励不是全同，且无模型权重梯度或更新。
- GB10 峰值统一内存不依赖同时常驻整个种群。

## 非目标

- 不把人工评分接入 ES。
- 不搜索 Qwen3-VL 或 VAE。
- 不宣称质量改善，质量结论属于 EXP-001。

## 证据

- 2026-09-04（依赖解锁前预备）：纯 CPU 接缝已实现并测试，包括 bounded latent、
  antithetic seed replay、z-score/rank shaping、候选失败惩罚、原子 checkpoint、31M 维
  flat genome 到逐层 group-128 clip 的映射、QUASAR-grid 初始化以及 CPU BF16 master
  的逐候选恢复。该工作不代表整模票已领取；必须等 ENG-003 的 320 图 calibration 冻结后
  才能启动 GB10 三代 smoke。
- 2026-09-04：ENG-003 已冻结 320-sample paired calibration（SHA
  `7f3d4a26...e437b`），依赖解除，正式领取本票。两台 GB10 上 ES/量化接缝各
  18 tests passed；下一步为整模 driver 与小种群三代 smoke。
- 2026-09-04：`gooo` GB10 整模 smoke 完成。配置为 NVFP4 W4A4、group=128、
  31,145,408 维、antithetic N=4、P=4 prompts、3 代；12/12 候选成功，三代 reward
  std 为 0.10972/0.05004/0.02218，0 failures，首候选的 clip/重构误差/PNG hash/
  HPS/CLIP/聚合奖励重放全部逐位一致。独立验证器返回 `valid=true`，最终 checkpoint
  SHA `1bdbcbf0...a4012`；进程峰值 RSS 31,139,069,952 bytes。完整证据见
  `docs/experiments/mage-flow-es-nvfp4-smoke-gb10-20260904.md`。
- 2026-09-04：本票只验收集成，不验收质量。smoke 还证明 `alpha=0.015` 对 N=4 过激：
  平均重构误差约 38k→72k→120k，必须在主实验前另做 sigma/alpha 校准。
