# W8A8 引擎重拟合契约（SM121 / vLLM 0.29.0 / TritonInt8ScaledMM）

日期：2026-09-10 深夜｜机器：`gooo`（探针脚本 `~/probe_int8_*.py`，产物 `~/minicpm5-int8probe/out/`）
对象：`MiniCPM5-2B-INT8-W8A8` 载荷（`compressed-tensors` int-quantized，294 层 int8 per-channel 对称静态权重 + per-token 动态激活）在 `--linear-backend triton` 下的活体布局。

## 一手事实（全部带探针证据）

1. **内核选择**：`Selected TritonInt8ScaledMMLinearKernel for CompressedTensorsW8A8Int8`（gooo 与 goooinext 双机一致）。SM121 上原生 INT8 scaled GEMM 存在，无需社区补丁。注意 `--linear-backend triton`（cutlass 在此路径非所用）。
2. **模块几何**：168 个 int8 模块 = 42 层 × {qkv_proj, o_proj, gate_up_proj, down_proj}；q/k/v 与 gate/up 全部融合，无未融合模块。
3. **活体权重**：`module.weight` 是 int8 Parameter，形状 `[in, out]`，**非连续转置视图**（stride `[1, N]`），其底层连续存储按 `[out, in]` 行-major 排布 = **载荷张量按融合行拼接后逐位相同**（5 个模块 bitwise 验证：qkv/gate_up/o/down/layer20 全 match）。融合行序：qkv = q|k|v（0..2047|2048..2303|2304..2559），gate_up = gate|up（0..6143|6144..12287）。
4. **weight_scale**：fp32 `[out, 1]` per-output-channel（载荷 bf16 尺度的无损升精度）。
5. **重拟合原语（关键坑）**：活体权重上任何 `reshape(-1)` / `flatten` 都会**物化副本**（非连续张量）——写入落到副本上、引擎不读。正确原语是
   ```python
   flat = module.weight.data.t().reshape(-1)   # [out,in] 连续视图，零拷贝
   flat[flat_index] = int8_code                # flat_index = row * in_features + col
   ```
   `.t()` 后张量连续，`reshape` 是视图；`flat.data_ptr() == weight.data_ptr()` 已断言。这个索引约定与 NVFP4 试点（`search_patch` 文档串）**完全一致**。
6. **两条写入路径均到达执行**（生成差分测试，4 提示 × 12 token 贪心）：直接 2D 视图改码（8192 处 +1）改变输出；cache→patch→`reload_weights(is_checkpoint_format=False)`（336 张量全集，约 80 ms）也改变输出。`copy_` 对跨步目标正确落值。
7. **`_get_parameter_for_reload` 的别名陷阱**：同一 `data_ptr` 的两个句柄可看到不同值（一个读到物化副本）——**读回验证必须走 `.t().reshape(-1)` 视图**，任何依赖 `reshape` 物化的读回都是假的。

## 对运行时适配的映射

- `search_patch`（int8 分支）：`flat = tensors[key].weight.t().reshape(-1)`；逐坐标写 int8 码。无 nibble 打包、无 0↔8 折叠（canonical 恒等）、格点均匀（±1 码步）。
- `search_readback`（engine 源）：同一视图。
- `build_kernel_payload`（int8）：导出 = 载荷融合拼接（就是活体底层存储原样）；无需 swizzle/pad/倒数链。
- `engine_layout`：行偏移沿用 NVFP4 概念（q/k/v = 2048/256/256、gate/up = 6144/6144 行），作用于 payload 行序（= flat 视图行序）。
- KL 探针：`bf16_kl.json` 参照可复用（语义是引擎 vs BF16 参照），**基线数值须按 W8A8 载荷重测**。
- 硬门口径：值级=逐字节 int8 相等（比 NVFP4 三口径简单：无 canonical 折叠、无 0↔8 单列）。

## 已证伪的备选路径（别再走）

- `module.weight` 上 `reshape(-1)` 直写/读回：物化副本，引擎不可见（探针 4/5 的 B、C 臂）。
- 只重载单模块参数子集：`reload_weights` 不报错但同样不生效的假象来自（a）部分集（b）副本物化；全集 + 正确视图才成立。

## 产物指针

`gooo:~/minicpm5-int8probe/out/{int8_live_probe,int8_roundtrip,int8_reload_probe,int8_direct_probe,int8_execution_probe,int8_final_contract}.json`；探针脚本 `gooo:~/probe_int8_{live,roundtrip,reload,direct,effect,final}.py`（本仓同步副本：`scripts/minicpm5/probes_int8/`）。
