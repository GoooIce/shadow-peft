# GB10 上 Mage-Flow-Turbo 的 NVFP4 原生内核矩阵

- 研究票：`RES-002`
- 核验日期：2026-09-04
- 核验主机：`gooo`（NVIDIA GB10 / compute capability 12.1）
- 范围：只做环境清点、官方源码核对和最小矩阵/卷积探针；未下载完整 Mage 模型，未运行生成或质量实验

## 结论

1. **GB10 上原生 NVFP4 Linear 已经可用；本研究当前软件栈验证通过的计算路径是 W4A4。** `gooo` 当前的 PyTorch 2.12.1 + CUDA 13.0 + TorchAO 0.18.0 能把 BF16 activation 动态量化为 E2M1，并以 E4M3 block scale 调用 `torch._scaled_mm`。Profiler 捕获到的实际 GPU kernel 是：

   ```text
   cutlass3x_sm120_bstensorop_s16864gemm_block_scaled_
   ue4m3xe2m1_ue4m3xe2m1_f32_bf16_bf16_128x128x256_1x1x1_0_
   tnn_align32_o_vs16_bias_bf16_relu[_stream_k]
   ```

   Kernel 名中的两侧 `ue4m3xe2m1`、`block_scaled` 和本机执行时的 `aten::_scaled_mm` 共同构成原生证据；仅有 FP4 dtype、压缩 checkpoint 或 QDQ 数值不算原生。NVIDIA 将 GB10 列为 compute capability 12.1；CUTLASS 官方也提供 SM120 的 NVFP4×NVFP4→BF16 block-scaled GEMM。[NVIDIA compute capability 表](https://developer.nvidia.com/cuda/gpus)；[CUTLASS 3.9+ 变更记录](https://github.com/NVIDIA/cutlass/blob/main/CHANGELOG.md)；[CUTLASS NVFP4/BF16 示例](https://github.com/NVIDIA/cutlass/blob/v4.5.2/examples/79_blackwell_geforce_gemm/79a_blackwell_geforce_nvfp4_bf16_gemm.cu)

2. **MMDiT Linear 和 Qwen3-VL Linear 的原生 primitive 路径通过。** 代表真实层宽度、且使用不规则 token 行数 `M=257` 的 8 个矩阵形状全部命中上述内核。尚未完成整模替换、真实 packed sequence 和 4 步生成验证，因此能力状态是“原生 primitive 可用，整模集成待解锁”，不是“整模已验收”。

3. **NVFP4 weight-only 不是原生 FP4 计算。** TorchAO 0.18.0 的 `NVFP4WeightOnlyConfig` 会在每次 forward 先把权重反量化为原始 dtype，再调用高精度 `linear`；本机 profiler 实际捕获到 BF16 `cutlass_80_wmma_tensorop_bf16...`。它只能算 NVFP4 存储压缩（W4A16），不能作为原生验收路径。[TorchAO 0.18.0 的 dispatch 源码](https://github.com/pytorch/ao/blob/v0.18.0/torchao/prototype/mx_formats/nvfp4_tensor.py)

4. **当前仓库的 QUASAR 路径也不是原生执行。** `QuasarNVFP4Linear.forward` 做 activation/weight QDQ 后调用 `F.linear`；`PackedNVFP4Linear.forward` 虽保存 packed uint8 权重，但每次 forward 都执行 `dequantize_weight(value.dtype)` 后再调用 `F.linear`。因此前者是 fake-quant，后者是“NVFP4 存储 + BF16 反量化 GEMM”。`materialize_quasar_nvfp4` 中的 `native` 命名不能当成 kernel 证据。

5. **Mage-VAE 不能纳入“全栈原生 NVFP4”验收。** TorchAO 的该配置只注册 `torch.nn.Linear` handler；对 `nn.Conv2d` 调用 `quantize_` 后，权重仍是 BF16 `Parameter`。本机 384→384、3×3 Conv2d 的 profiler 命中 `cutlass_tensorop_bf16_s16816fprop...`。TensorRT 11.2.1 的 Convolution operator 类型表也只列 FP32/FP16/BF16/INT8/FP8，不含 FP4。[TensorRT Convolution 类型表](https://docs.nvidia.com/deeplearning/tensorrt/latest/_static/operators/Convolution.html)

所以首期原生验收边界应是：**MMDiT Linear W4A4 NVFP4 on GB10**；Qwen3-VL Linear 是随后可接入的同类后端；VAE Conv2d 只允许作为明确命名的模拟质量实验，不能计入原生覆盖率。

## 判定口径

| 名称 | 必须满足 | 本研究中的例子 |
| --- | --- | --- |
| 原生 NVFP4 | A/B 操作数实际为 E2M1、block scale 为 E4M3、进入 block-scaled Tensor Core GEMM；必须有 profiler kernel 证据 | TorchAO dynamic activation + NVFP4 weight，经 `torch._scaled_mm` 命中 `...block_scaled_ue4m3xe2m1...` |
| fake-quant / 模拟精度 | 先 QDQ 逼近 NVFP4 数值，但 GEMM 操作数已恢复为 BF16/FP32 | `QuasarNVFP4Linear` |
| 存储压缩 | checkpoint/常驻权重是 packed 4-bit，但 forward 反量化后做 BF16 GEMM | `PackedNVFP4Linear`、TorchAO `NVFP4WeightOnlyConfig` |
| 不可用 | 目标 op 没有转换/内核，继续以原 dtype 执行 | 当前 TorchAO 对 VAE `Conv2d` |

NVIDIA 的 NVFP4 定义为 E2M1 数据、每 16 个连续元素共享 E4M3 scale，并另有 FP32 tensor scale；即 `x = x_e2m1 × s_block × s_global`。[Transformer Engine NVFP4 文档](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/nvfp4/nvfp4.html) TorchAO 0.18.0 的 native dispatch 同样把 packed data view 成 `torch.float4_e2m1fn_x2`、把 scale view 成 `torch.float8_e4m3fn`，再调用 `torch._scaled_mm`。[TorchAO 0.18.0 NVFP4 tensor 源码](https://github.com/pytorch/ao/blob/v0.18.0/torchao/prototype/mx_formats/nvfp4_tensor.py)

## `gooo` 环境与占用

探针前后均先检查 `nvidia-smi`。GPU utilization 为 0%，但 PID 3993671 的 MinerU 服务已经常驻约 5385 MiB；研究过程没有终止、重启或改动该进程，也没有改动任何远端模型/数据资产。

| 项目 | `gooo` 实测 |
| --- | --- |
| OS / CPU ABI | Linux 6.17.0-1029-nvidia, `aarch64` / `linux-aarch64` |
| GPU | NVIDIA GB10, compute capability `(12, 1)`, 48 SM |
| Driver | 580.173.02（`nvidia-smi` 运行时值） |
| 统一内存 | `free`: 121 GiB；PyTorch device property: 130,596,184,064 bytes；探针前 available 83 GiB |
| Python | 3.12.3, GCC 13.3.0 |
| PyTorch | 2.12.1+cu130；`torch.version.cuda == 13.0` |
| CUDA toolkit | `/usr/local/cuda-13.0`, manifest 13.0.3；NVCC 13.0.88 |
| cuBLAS / cuDNN | Python wheel `nvidia-cublas` 13.1.1.3、`nvidia-cudnn-cu13` 9.20.0.48；运行时从当前 venv 的 `nvidia/cu13/lib` 加载 |
| TorchAO / Triton | TorchAO 0.18.0、Triton 3.7.1 |
| Profiler | Nsight Systems 2025.3.2；Nsight Compute 2025.3.1；本研究用 `torch.profiler` 保留具体 CUDA kernel 名 |
| 未安装 | `mage_flow`, `diffusers`, `flash_attn`, `accelerate`, `einops`, `modelscope`, `transformer_engine`, `mslk` |

NVIDIA 当前 DGX Spark 软件发布线也是 CUDA 13.0、580 系驱动和 6.17 内核；伙伴机更新时间可能与 Founder Edition 不同，因此这里以实测运行时值为准。[DGX Spark release notes](https://docs.nvidia.com/dgx/dgx-spark/release-notes.html)

ARM64 本身不是 NVFP4 Linear blocker：上表中的 aarch64 PyTorch/TorchAO/cuBLASLt 组合已经执行原生 kernel。真正的整模 blocker 是 Mage 软件依赖尚未装入该 venv。

## 组件能力矩阵

### MMDiT Linear

Microsoft 官方 config 给出 `in_channels=128`、`context_in_dim=2560`、`hidden_size=3072`、`mlp_ratio=4`、12 层；官方实现中的 image/text projections、attention projections、modulation MLP、FFN 和 output projection 都是 `nn.Linear`。[Mage-Flow transformer config](https://huggingface.co/microsoft/Mage-Flow-Turbo/blob/main/transformer/config.json)；[MageFlow 模型源码](https://github.com/microsoft/Mage/blob/main/mage_flow/models/mage_flow.py)；[Mage 层源码](https://github.com/microsoft/Mage/blob/main/mage_flow/models/modules/mage_layers.py)

| 代表层 | 探针 `(M,K,N)` | 结果 | Kernel 后缀 |
| --- | ---: | --- | --- |
| `img_in` | `(257,128,3072)` | 原生 NVFP4 | `...bias_bf16_relu` |
| attention projection | `(257,3072,3072)` | 原生 NVFP4 | `...bias_bf16_relu_stream_k` |
| FFN expand | `(257,3072,12288)` | 原生 NVFP4 | `...bias_bf16_relu` |
| `proj_out` | `(257,3072,128)` | 原生 NVFP4 | `...bias_bf16_relu` |

所有被测 `K,N` 均能满足 TorchAO 的“两维可被 16 整除”约束；`M=257` 证明 packed/不规则 token 行数本身不阻止 cuBLASLt 选到 SM120 block-scaled kernel。TorchAO 的约束及 native config 定义见[官方 0.18.0 workflow 源码](https://github.com/pytorch/ao/blob/v0.18.0/torchao/prototype/mx_formats/inference_workflow.py)。

状态：**原生 primitive 可用；整模待验证**。在完整模型加载后仍需做逐层 census，明确任何非 16 对齐层的排除策略，并 profile 一次真实 4-step forward。

### Qwen3-VL Linear

Mage checkpoint 中的 Qwen3-VL text config 是 36 层、hidden 2560、intermediate 9728、32 个 query heads、8 个 KV heads、head dim 128；这些主要 projection 宽度均为 16 的倍数。[Microsoft Mage-Flow-Turbo text encoder config](https://huggingface.co/microsoft/Mage-Flow-Turbo/blob/main/text_encoder/config.json)

| 代表层 | 探针 `(M,K,N)` | 结果 | Kernel 后缀 |
| --- | ---: | --- | --- |
| Q projection | `(257,2560,4096)` | 原生 NVFP4 | `...bias_bf16_relu` |
| K/V projection | `(257,2560,1024)` | 原生 NVFP4 | `...bias_bf16_relu` |
| MLP gate/up | `(257,2560,9728)` | 原生 NVFP4 | `...bias_bf16_relu` |
| MLP down | `(257,9728,2560)` | 原生 NVFP4 | `...bias_bf16_relu_stream_k` |

状态：**原生 primitive 可用；整模待验证**。Embedding、RMSNorm、RoPE、attention 和非 Linear op 不属于这项 NVFP4 GEMM 结论；首期应继续保持 BF16。完整 text encoder 还承担 conditioning 与 prompt screening，因此 Qwen 量化应是单独实验臂，不和 MMDiT 首期验收绑定。

### Mage-VAE Conv2d

Mage 官方说明 VAE 是 one-step、fully-convolutional codec；源码包含大量普通、1×1、3×3 和 depthwise `nn.Conv2d`，另有少量 Linear。[Mage-Flow README](https://github.com/microsoft/Mage/blob/main/mage_flow/README.md)；[Mage-VAE 源码](https://github.com/microsoft/Mage/blob/main/mage_flow/models/modules/mage_vae.py)

本机最小探针：

- 输入 `BF16 [1,384,32,32]`，weight `BF16 [384,384,3,3]`。
- `quantize_(conv, NVFP4DynamicActivationNVFP4WeightConfig(...))` 前后 module/weight 均是 `Conv2d` + BF16 `Parameter`，没有 `qdata`。
- Forward 命中 `cutlass_tensorop_bf16_s16816fprop_optimized_bf16_...`，没有 `_scaled_mm` 或 E2M1 kernel。

状态：**当前后端不可用**。VAE 中个别 16 对齐 Linear 可复用 Linear 后端，但这不等于 VAE 原生 NVFP4。`NerfFinalLayer` 等输出宽度为 3 的层也不满足两维都被 16 整除的条件。首期只能：

- 保持 VAE BF16；或
- 以 `sim_nvfp4_conv` 明确命名做 QDQ 质量实验，不汇报 kernel 加速/原生覆盖；或
- 另开内核工程票，把 1×1 Conv 显式 lowering 成 GEMM，并分别解决 3×3/depthwise Conv。该路线完成前不得静默反量化。

## 回退路径、正确性和微基准

统一形状为 `(M,K,N)=(257,3072,3072)`；同一 BF16 权重/输入，8 次 warm-up、40 次 CUDA event 计时。该数字只用于识别路径和数量级，不是整模性能承诺：未锁频，且机器有常驻 MinerU 服务。

| 路径 | 实际计算 | 平均 ms | 与 BF16 MAE / max error | 判定 |
| --- | --- | ---: | ---: | --- |
| BF16 baseline | BF16 GEMM | 0.0927 | — | 基线 |
| TorchAO W4A4 dynamic E2E | activation quantization + 原生 FP4 GEMM | 0.4207 | 0.06396 / 0.42969 | 原生，但当前 unfused quantization 开销较大 |
| TorchAO W4A4 pre-quantized | 原生 FP4 GEMM | 0.04765 | 与 dynamic 输出 0 / 0 | 原生 raw kernel；数值逐位等同同一量化输入 |
| TorchAO W4A16 weight-only | 每次反量化 + BF16 GEMM | 5.0337 | 0.04688 / 0.28516 | 存储压缩，不原生 |

所有输出均 finite。W4A4 dynamic 与显式预量化输入的输出逐位相等，说明 native dispatch 与该量化输入路径一致；相对 BF16 的误差是量化误差，而不是 fallback 判定依据。

一个 `768×512` 权重的内部存储实测为：packed qdata 196,608 bytes + E4M3 scales 24,576 bytes（另有一个很小的 FP32 global scale），而 BF16 为 786,432 bytes。即该表示约 0.5625 byte/weight、相对 BF16 约 3.56×，不是忽略 scale 后宣称的 4×。

### 关键可用性陷阱

TorchAO 0.18.0 的 `NVFP4DynamicActivationNVFP4WeightConfig` 默认 `use_triton_kernel=True`，而官方源码注明它需要 MSLK。本机没有 `mslk`，默认配置的第一轮 forward 会直接报：

```text
AssertionError: mslk is required for NVFP4 triton quantization.
```

本研究用 `use_triton_kernel=False` 走 unfused Torch activation quantization；这不会改变后续 GEMM 的原生性，但使 E2E 比 raw kernel 慢。下一阶段有两种明确选择：

1. 先锁 `use_triton_kernel=False`，以正确性和 native-kernel gate 完成首期功能；
2. 在独立环境验证 MSLK/ARM64 后启用 fused activation quantization，再做性能验收。

不能捕获异常后自动切回 BF16 并继续计为“NVFP4 成功”。

## Mage / ARM64 依赖阻塞

Microsoft 当前 Mage package 声明 `torch>=2.13.0`、`diffusers>=0.37.0`、`transformers>=5.3,<5.6`，并要求 Flash Attention 2.8.3 在其余依赖之后以 `--no-build-isolation` 单独构建。[Mage `pyproject.toml`](https://github.com/microsoft/Mage/blob/main/mage_flow/pyproject.toml)；[Mage-Flow 安装说明](https://github.com/microsoft/Mage/blob/main/mage_flow/README.md)

`gooo` 当前是 torch 2.12.1、transformers 5.15.1，且 Mage/diffusers/flash-attn/accelerate/einops/ModelScope 均未安装。因此：

- 当前 venv **不能直接加载 Mage-Flow-Turbo**；这不是 NVFP4 kernel blocker，而是整模 integration blocker。
- 不应为满足 Mage 约束就地降级/升级现有实验环境。建议为 ENG-002 建独立 ARM64 venv，按 Mage 官方版本范围锁定，再安装 ModelScope 下载客户端。
- Transformer Engine 也未安装。其官方文档宣称 NVFP4 inference 支持 SM100+，但本研究没有在 `gooo` 验证 TE 的 aarch64 wheel/build，因此不能把 TE 当作已可用后端。[Transformer Engine NVFP4 supported devices](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/nvfp4/nvfp4.html)
- 已实测可行的最小 backend 是当前 PyTorch + TorchAO + cuBLASLt，而不是 TE。

## 对后续票的建议

### ENG-002：原生 runtime seam

- 将“搜索/重构”和“执行后端”分开：QUASAR/ES 产生 packed E2M1 code、E4M3 block scale、FP32 tensor scale；native adapter 负责构造 TorchAO `NVFP4Tensor` 的 swizzled scale 布局和 activation quantization metadata。
- 第一个集成目标只做 MMDiT `nn.Linear` W4A4。每个被替换层必须通过三重 gate：权重二维均可被 16 整除、forward 命中 `_scaled_mm`、CUDA kernel 名包含 `block_scaled_ue4m3xe2m1`。
- 为自研 QUASAR pack 与 TorchAO pack 增加 code/scale/dequant parity 测试，再增加 native output 与显式 dequant reference 的容差测试。未经这一步，不能假设当前 nibble/scale layout 可直接喂给 cuBLASLt。
- 删除或改名任何仅凭 materialization 就宣称 `native` 的字段；报告必须分别输出 `quant_representation`、`compute_kernel`、`fallback_reason`。
- 初版固定 TorchAO 0.18.0、PyTorch 2.12.1+cu130、cuBLAS 13.1.1.3，并显式传 `use_triton_kernel=False`。若独立 Mage venv 必须升级 torch，则要重新跑本文所有 kernel gate，不能沿用结论。

### EXP-002：MMDiT / Qwen

- 首个正式 arm：`MMDiT Linear W4A4 native NVFP4 + BF16 Qwen/VAE`。
- 真实模型加载后先输出 layer census 和排除列表，再用一个 prompt、一个 seed、四步生成做 profiler smoke；不要先跑 500-prompt 实验。
- Qwen3-VL 作为第二个独立 arm。保持 embedding、norm、attention/non-Linear op 为 BF16，并单独核查 screening 结果是否受量化影响。
- 吞吐报告同时给出 end-to-end 和 pre-quantized GEMM；不能用 raw kernel 的 0.04765 ms 代替模型端到端收益。

### EXP-003：VAE

- 保持 BF16 作为默认和原生验收基线。
- 若研究质量敏感性，可以加入 `sim_nvfp4_conv`，但结果页必须标注 “QDQ simulation / BF16 convolution”。
- 原生 VAE 量化需要另开 Conv2d kernel/lowering 工程票；在 profiler 出现真实 FP4 Conv/GEMM kernel 前，RES-002 不为全栈 NVFP4 背书。

## 可重放探针

先做占用检查；只有 utilization 为 0 或确认不会干扰现有任务时才继续：

```bash
ssh gooo 'nvidia-smi --query-gpu=name,compute_cap,utilization.gpu,driver_version --format=csv,noheader && nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader'
```

最小原生/weight-only/Conv 判别脚本（不创建远端文件）：

```bash
ssh gooo 'cd ~/Code/shadow-peft && .venv/bin/python - <<"PY"
import torch
from torch.profiler import ProfilerActivity, profile
from torchao.quantization import quantize_
from torchao.prototype.mx_formats import (
    NVFP4DynamicActivationNVFP4WeightConfig,
    NVFP4WeightOnlyConfig,
)

def kernels(fn):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
        y = fn()
        torch.cuda.synchronize()
    names = [e.name for e in p.events() if e.device_type == torch.autograd.DeviceType.CUDA]
    return y, [n for n in names if "block_scaled" in n or "bf16" in n]

x = torch.randn(257, 3072, device="cuda", dtype=torch.bfloat16)
for label, cfg in (
    ("w4a4", NVFP4DynamicActivationNVFP4WeightConfig(use_triton_kernel=False)),
    ("w4a16", NVFP4WeightOnlyConfig()),
):
    layer = torch.nn.Linear(3072, 3072, device="cuda", dtype=torch.bfloat16).eval()
    quantize_(layer, cfg)
    y, names = kernels(lambda: layer(x))
    print(label, type(layer.weight).__name__, torch.isfinite(y).all().item(), names)

conv = torch.nn.Conv2d(384, 384, 3, padding=1, device="cuda", dtype=torch.bfloat16).eval()
quantize_(conv, NVFP4DynamicActivationNVFP4WeightConfig(use_triton_kernel=False))
cx = torch.randn(1, 384, 32, 32, device="cuda", dtype=torch.bfloat16)
cy, cnames = kernels(lambda: conv(cx))
print("conv", type(conv.weight).__name__, conv.weight.dtype, hasattr(conv.weight, "qdata"), cnames)
PY'
```

## 未决事项与证据边界

- `gooo` 上没有 Mage-Flow-Turbo 权重和运行依赖，所以尚无真实层 census、整模 native coverage、4-step 生成正确性或端到端速度。
- 只按任务要求探测了 `gooo`，没有把结论外推为 `goooinext` 的环境状态。
- 没有验证 Transformer Engine 或 MSLK 的 ARM64 安装路径。
- 没有原生 VAE Conv2d 路径；这是明确的技术阻塞，不是“待调参”。
- 微基准用于辨认执行路径；最终性能验收必须在独立 Mage venv、固定软件版本和真实 sequence/image shapes 上重跑。
- 研究结论仍需人工评审；在评审前不应解除 RES-002 对后续实验票的阻塞。
