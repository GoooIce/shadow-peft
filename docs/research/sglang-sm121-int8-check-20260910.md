# SGLang 能否解决 SM121 上没有 CUTLASS INT8 的问题——不能（现状），但有替代数字

日期：2026-09-10 深夜｜机器：`goooinext`（`sglang 0.5.19` + `sglang-kernel 0.4.6.post1`，torch 2.13.0+cu130，SM121）
触发：用户指示「SM121 上没有 CUTLASS INT8 这个问题可以看看 sglang」。

## 结论

1. **SGLang 现状不能解决**。per-channel W8A8 INT8 的 CUDA 路径只有一条：`W8A8Int8LinerMethod.apply` → `sgl_kernel.int8_scaled_mm`（[w8a8_int8.py](https://github.com/sgl-project/sglang)，**无回退分支**）。该算子（CUTLASS INT8）在本 wheel 里只编译了 **sm90**（Hopper TMA warp-specialized 实例，二进制符号确认）与 sm100 装载路径；SM121（cap 12.1）上实测 `NotImplementedError: No implemented int8_scaled_mm for current compute capability`（布局检查过后触发）。**0.4.6.post1 已是 PyPI 最新**。⇒ SGLang 在本机**连 serve 我们的 W8A8 载荷都会在首个前向崩掉**（源码级推断；算子级已实证）。
2. **社区证据一致**：GB10/SM121 的 sgl-kernel 适配仍在进行中——[common_ops 加载失败 #29317](https://github.com/sgl-project/sglang/issues/29317)、[sm121a 需 cu130 wheel #18203](https://github.com/sgl-project/sglang/issues/18203)、[GB10 跟踪票 #11658](https://github.com/sgl-project/sglang/issues/11658)、[能力门导致 GB10 测试失败 #31365](https://github.com/sgl-project/sglang/issues/31365)。即便 cu130 wheel 补了 sm121a cubin，INT8 scaled mm 仍是 Hopper/数据中心限定。
3. **但 cuBLASLt INT8 原生可用**：`torch._int_mm` 在 SM121 上存在且精确（int32 输出、误差 0），限制 M>16（解码需填充到 ≥17）。
4. **同形 GEMM 基准**（K=2048→N=12288，单层；triton 数字含 per-token 量化，cuBLASLt 仅裸 GEMM）：

| M | vLLM triton（现用路径） | cuBLASLt `_int_mm` |
|---|---|---|
| 4（解码） | **0.077 ms** | 0.084 ms（填充到 17） |
| 32 | **0.087 ms** | 0.095 ms |
| 256 | 0.260 ms | **0.184 ms** |
| 2048（prefill） | 1.916 ms | **0.900 ms** |

   **解码端 triton 没有输**（还多干了量化）；prefill 端 cuBLASLt 裸 GEMM 快 1.4–2.1×，但真实路径还要加量化 + int32→scale epilogue + M 填充，差距会收窄。

## 对本栈的含义

- **W8A8 战役不阻塞**：搜索与部署都用 vLLM + triton INT8（重拟合契约已验证）。GEMM 不是瓶颈。
- **部署侧维持 vLLM**：SGLang 对 W8A8 INT8 在 SM121 上当前不可用；且 11 票已记录 SGLang refit 不可用（搜索侧本就只剩 vLLM 一条路）。
- **prefill 吞吐优化项（非阻塞）**：若日后要榨 prefill，可做 `quant + _int_mm(M 填充) + scale epilogue` 的自定义路径，预期收益 ≈1.5×（prefill 段）。
- 复查时机：SGLang 若发布含 SM121 INT8 的 sgl-kernel（盯 [#11658](https://github.com/sgl-project/sglang/issues/11658)/[#29317](https://github.com/sgl-project/sglang/issues/29317)），只需重跑本文件的算子级测试（`goooinext:/tmp/test_sgl_int8.py`）即可判别。

## 复现指针

`goooinext:/tmp/test_sgl_int8.py`（算子级，含两段报错证据）、`/tmp/test_intmm.py`、`/tmp/bench_int8_gemm.py`（基准）；安装现场 `~/venvs/minicpm5-sglang`。
