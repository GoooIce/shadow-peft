---
id: EXP-003
type: experiment
status: blocked
priority: P2
depends_on: [RES-002, EXP-001]
hardware: GB10
phase: 3
---

# VAE NVFP4 可行性与扩展

## 目标

评估 VAE 卷积权重的 NVFP4 QUASAR/ES 重建是否值得进入全栈量化，并严格区分模拟精度与 GB10 原生执行。

## 工作范围

- 先做 VAE-only 重建：用相同 latent 比较 BF16 与量化解码输出。
- 使用像素/感知/结构指标与人工伪影检查，不能只看最终 prompt 奖励。
- 根据 RES-002 的 Conv2d 能力矩阵决定部署分支：原生内核、仅模拟研究或另开内核研发票。
- 只有 VAE-only 达标后，才比较 MMDiT+Qwen+VAE 全栈组合。

## 验收标准

- VAE 重建误差、感知质量、延迟和内存分别报告。
- 没有原生 Conv2d NVFP4 时，结果明确标为 fake-quant，不计入原生全栈验收。
- 量化目标层与保留 BF16 层有完整清单。

## 证据

尚未开始。
