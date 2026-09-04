---
id: EXP-002
type: experiment
status: blocked
priority: P1
depends_on: [RES-002, EXP-001]
hardware: GB10
phase: 3
---

# Qwen3-VL NVFP4 扩展

## 目标

评估 Mage-Flow-Turbo 的 Qwen3-VL 文本编码器 Linear 在 NVFP4 下采用 QUASAR 或独立 ES 子基因组的收益与风险。

## 工作范围

- Linear 为量化候选；embedding、norm 和非矩阵算子默认保持 BF16。
- 先比较冻结 MMDiT ES 最优配置下的 Qwen RTN 与 QUASAR，再决定是否启动联合 ES。
- Qwen clip 参数使用独立子基因组和消融，避免无法归因的全栈联合搜索。
- 检查文本条件、文字渲染、语义对齐和内容筛查路径的变化。

## 验收标准

- 单独报告 MMDiT-only 与 MMDiT+Qwen 的质量、延迟和内存差异。
- 确认内容筛查输出未因量化被悄然改变；若改变，作为独立风险报告。
- 原生 NVFP4 结论有 GB10 kernel 证据。

## 证据

尚未开始。
