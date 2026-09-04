---
id: EXP-004
type: experiment
status: blocked
priority: P1
depends_on: [EXP-001]
hardware: GB10
phase: 4
---

# 消融、泛化与报告

## 目标

解释 ES×QUASAR 的收益来源、泛化性和失败边界，并形成可复现报告。

## 工作范围

- antithetic 与独立采样、rank shaping 与 z-score。
- clip 共享粒度、逐通道退路、仅 attention 子空间。
- 未见 prompt、未训分辨率和文字渲染子集迁移。
- clip 在层、模块和深度上的分布分析。
- 自动奖励与人工偏好分叉、重复标注一致性和奖励 hacking 审计。
- 汇总 Qwen3-VL/VAE 扩展；未完成扩展不阻塞主结论，但必须标明状态。

## 验收标准

- 每个消融只改变一个主因素，并复用相同 prompt/seed 配对。
- 正负结果都保留，报告搜索成本和选择偏差。
- 发布材料能从锁定 manifest 重建表格与图。

## 证据

尚未开始。
