# Mage-Flow-Turbo MMDiT NVFP4 基线（GB10，2026-09-04）

## 结论

Mage-Flow-Turbo 的全部 174 个 MMDiT `Linear` 已在 NVIDIA GB10 上以 TorchAO
NVFP4 W4A4 跑通 RTN 和 QUASAR 15 点 clip 网格重建。两条路径都通过严格 profiler
门槛：运行中存在 SM120 block-scaled E2M1×E2M1 CUTLASS kernel，不能仅凭 4-bit
存储格式宣称原生计算。

单 prompt 结果显示 RTN 已造成显著视觉退化，QUASAR 网格只小幅降低权重重建误差和
像素误差，仍未恢复 BF16 的主体结构。因此 W4A4 有真实优化空间，但单图不能支持质量
排序；排序结论必须等待 ENG-003 的冻结奖励协议和完整留出集。

## 固定协议

- 模型：ModelScope `microsoft/Mage-Flow-Turbo` 的本项目固定快照。
- 量化范围：只量化 `pipeline.model.transformer`；Qwen3-VL 与 VAE 保持 BF16。
- 格式：NVFP4 E2M1 权重与激活、E4M3 block scale、block size 16。
- QUASAR 重建粒度：每个输出行内连续 128 个权重共享一个 clip；候选为
  `0.30, 0.35, ..., 1.00`，随后独立拟合其 8 个 block-16 scale。
- 生成：512×512、4 steps、CFG=1、prompt 0、seed 525200。
- prompt 池 SHA-256：
  `830a12b082d3c1576b31287e85455f2cfbca3fbcaa2eacd0542d4df0f301d58a`。

## MMDiT 清点

| 项目 | 实测值 |
|---|---:|
| `Linear` 总数 / NVFP4 eligible | 174 / 174 |
| 被量化权重 | 4,114,612,224 |
| 保持 BF16 的 Linear bias | 1,124,480 |
| group=128 的 clip 搜索维度 | 32,145,408 |

维度来自每层 `out_features × ceil(in_features / 128)` 的精确求和，而不是把包含 bias
的模型参数量直接除以 128。

## RTN 与 QUASAR 结果

| 指标 | NVFP4 RTN | NVFP4 QUASAR-grid |
|---|---:|---:|
| 全模型重建时间 | 7.60 s | 80.23 s |
| 权重平方重建误差 | 40,895.47 | 37,662.89 |
| 同 BF16 图像 pixel MSE | 1,848.63 | 1,789.75 |
| 同 BF16 图像 PSNR | 15.46 dB | 15.60 dB |
| warm 4-step 生成 | 6.73 s | 5.04 s |
| profiler 4-step 生成 | 7.55 s | 5.98 s |
| profiler `_scaled_mm` 调用 | 696 | 696 |
| 原生 NVFP4 strict gate | pass | pass |
| 重复 PNG 是否字节一致 | 是 | 是 |
| 进程峰值 RSS | 30.93 GB | 31.15 GB |

QUASAR 的重建误差比 RTN 低约 7.9%，但其逐层 15 候选搜索耗时约为 RTN 的 10.6 倍。
32,145,408 个 group 中，最终候选只落在 `0.80..1.00`：`0.80/0.85/0.90/0.95/1.00`
的计数分别为 `105 / 57,084 / 1,841,022 / 11,957,855 / 18,289,342`，中位数为
`1.00`。这说明首期 ES 中心从 `c=1` 启动有数据依据，但不能据此预设最优点。

## 原生内核证据

两条路径都捕获以下 kernel 家族（另含 stream-k 变体）：

```text
cutlass3x_sm120_bstensorop_s16864gemm_block_scaled_
ue4m3xe2m1_ue4m3xe2m1_f32_bf16_bf16_128x128x256_1x1x1_0_tnn_align32_o_vs16_bias_bf16_relu
```

运行环境为 GB10 compute capability 12.1、PyTorch 2.13.0+cu130、TorchAO 0.18.0。
`shadow_lab.nvfp4_runtime.native_kernel_evidence` 同时要求 SM120/121、`block_scaled`
和两个 E2M1 operand 标记；普通 BF16 或仅有 `_scaled_mm` 不会通过。

## 证据位置

- 本地 RTN：[run.json](../../runs-local/mage-flow/nvfp4-rtn-smoke-20260904/run.json)
- 本地 QUASAR-grid：
  [run.json](../../runs-local/mage-flow/nvfp4-quasar-grid-smoke-20260904/run.json)
- 远端 RTN：`goooinext:~/runs/mage-flow/nvfp4-rtn-smoke-20260904/`
- 远端 QUASAR-grid：
  `goooinext:~/runs/mage-flow/nvfp4-quasar-grid-smoke-20260904/`
- 本地/GB10 测试：本地 `15 passed, 2 skipped`；GB10 `17 passed`。

完整物化器会冻结 MMDiT root 的全部参数；CUDA 合同测试验证物化前后 trainable
参数从非零变为 0、外部 clip 直接桥接 TorchAO 后无 Q/DQ 漂移。该冻结标记只影响
autograd，不改变上述已完成的 eval 数值。

## 重放

```bash
nohup scripts/launch_mage_flow_nvfp4_smoke_gb10.sh \
  rtn ~/runs/mage-flow/nvfp4-rtn-smoke-YYYYMMDD \
  >~/logs/launch-nvfp4-rtn.log 2>&1 < /dev/null &

nohup scripts/launch_mage_flow_nvfp4_smoke_gb10.sh \
  quasar-grid ~/runs/mage-flow/nvfp4-quasar-grid-smoke-YYYYMMDD \
  >~/logs/launch-nvfp4-quasar-grid.log 2>&1 < /dev/null &
```

运行器会拒绝非空输出目录，并把 prompt 来源、SHA-256、版本、量化清点、逐层误差、
内核名、时间、内存与输出图哈希写入 `run.json`。
