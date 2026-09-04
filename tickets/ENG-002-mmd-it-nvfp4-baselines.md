---
id: ENG-002
type: engineering
status: done
priority: P0
depends_on: [RES-002, ENG-001]
hardware: GB10
---

# MMDiT NVFP4 RTN / QUASAR 基线

## 目标

在冻结 MMDiT 上建立可比较的 NVFP4 RTN 与 QUASAR 网格重建基线，为 ES 提供稳定量化接缝。

## 工作范围

- 复用 `shadow_lab.quasar` 的 NVFP4 E2M1、block-16 E4M3 scale 和 tensor scale 语义。
- 建立 Mage MMDiT Linear 目标清单及显式排除清单。
- 区分 NVFP4 原生 block=16 与 ES clip 参数的共享粒度；初始方案为每 128 个权重共享一个 clip，并广播到 8 个 NVFP4 block。
- 支持外部 clip 向量直接重建，使网格、随机和 ES 使用同一量化器。
- 实现 `c=1` RTN 与 QUASAR `0.30..1.00`、步长 `0.05` 的基线。
- 记录量化、恢复、物化和重复执行的确定性。

## 验收标准

- 模型权重在搜索协议中保持冻结。
- RTN 与 QUASAR 的差异只来自 clip/重建策略。
- 单层和整模型有数值测试、seed 重放测试和 materialize parity 测试。
- GB10 报告原生与模拟路径，且主结果不混用两者。
- BF16、RTN 和 QUASAR 都能按 ENG-001 协议完成 4 步生成。

## 非目标

- 不集成 ES。
- 不量化 Qwen3-VL 或 VAE。

## 证据

- 2026-09-04：RES-002 与 ENG-001 均已完成；用户指示继续后领取本票。
- 2026-09-04：新增 `shadow_lab.nvfp4_runtime`，把 QUASAR/外部 clip 的
  E2M1 code、E4M3 block scale 和 tensor scale 无损桥接为 TorchAO
  `NVFP4Tensor`；原生 kernel 由 profiler 严格判定。
- 2026-09-04：本地测试 `15 passed, 2 skipped`；GB10 CUDA 测试 `17 passed`，
  覆盖 group=128、外部 clip、确定性、Q/DQ parity、全 root 冻结和内核分类。
- 2026-09-04：GB10 全模型 RTN 与 QUASAR-grid 均完成 4-step 生成、同 seed
  字节级重放和原生 NVFP4 gate。174/174 个 Linear、4,114,612,224 个权重；
  精确 ES 维度 32,145,408。详见
  `docs/experiments/mage-flow-nvfp4-baselines-gb10-20260904.md`。
