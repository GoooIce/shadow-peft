---
id: RES-002
type: research
status: done
priority: P0
depends_on: []
blocks: [ENG-002, EXP-002, EXP-003]
hardware: GB10
---

# GB10 原生 NVFP4 兼容性研究

## 问题

GB10（`sm_121`）当前软件栈能否对 Mage-Flow-Turbo 的 MMDiT Linear、Qwen3-VL Linear 和 VAE Conv2d 执行原生 NVFP4，而非模拟量化或 BF16 反量化计算？

## 工作范围

- 记录两台 GB10 的驱动、CUDA、PyTorch、架构、内存和可用编译工具链。
- 以 NVIDIA 官方文档、库源码和 profiler 结果确认 NVFP4 block-16 E2M1/E4M3 路径。
- 分别验证 Linear、批形状不规则的 Linear、Conv2d；每种路径记录正确性、内核名、吞吐和回退行为。
- 核对 Mage-Flow 自定义 diffusers 组件及 ARM64 依赖的可安装性。
- 给出“原生执行”“模拟精度”“不可用”三态能力矩阵。

## 产物

- `docs/research/gb10-mage-nvfp4-kernel-matrix.md`
- 可重复运行的最小探针命令或脚本路径。
- ENG-002、EXP-002、EXP-003 可采用的后端建议与版本锁定。

## 完成标准

- 原生结论必须有 profiler/kernel 证据，不能只依据 dtype、checkpoint 大小或营销规格。
- fake-quant 参考路径与部署路径分别命名和计时。
- VAE Conv2d 若无原生路径，明确停在模拟实验还是另开内核工作，不能静默反量化。
- 研究结论经人工评审后，才解除依赖票阻塞。

## 非目标

- 不量化完整模型。
- 不跑主实验或质量验收。

## 证据

- 2026-09-04：用户批准与下载工作并行启动。
- 2026-09-04：研究产物为 `docs/research/gb10-mage-nvfp4-kernel-matrix.md`；记录 `gooo` 环境、组件能力矩阵、profiler kernel、正确性、回退路径与可重放探针。
- 2026-09-04：TorchAO W4A4 Linear 命中 GB10 原生 `block_scaled_ue4m3xe2m1` kernel；W4A16 与当前 QUASAR runtime 均为反量化 BF16 GEMM；Conv2d 保持 BF16。主 agent 已完整复核。
