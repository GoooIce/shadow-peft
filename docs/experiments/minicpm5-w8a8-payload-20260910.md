# MiniCPM5-2B INT8 W8A8 起点载荷与 SM121 原生内核取证（GB10）

- 上游：`10`（NVFP4 W4A4 起点载荷与三重内核口径）、`01`（原生内核判定口径）、`QES 一手核查`（论文的 headline 设置就是 **W8A8**，不是 W4A4）
- 执行日期：2026-09-10（21:27–22:03）；机器：**`goooinext`**（`aitopatom-dc4a`，GB10 / SM 12.1，全部 GPU 工作）；本地 Mac（脚本、判定、文档）。**`gooo` 全程未使用**（它当时在跑另一条实时诊断；本票未在其上跑任何 GPU 作业，也未改 `~/minicpm5-run300/`）
- 判定：**载荷可服务、且 SM121 上确实存在原生 INT8 W8A8 内核——但不是 vLLM `auto`/`cutlass` 选中的那一个。** 默认选择会在第一个前向直接抛错；`--linear-backend triton` 选中 `TritonInt8ScaledMMLinearKernel`，profiler 证明它跑的是本机编译的真实 INT8 tensor-core MMA（PTX `mma.sync…s32.s8.s8.s32` on `sm_121a`），不是反量化/模拟
- 质量（v3 dev，480 题，冻结口径）：BF16 **0.6021** / W8A8 **0.6208** / W4A4 **0.5000** ⇒ **BF16−W8A8 = −1.88 点**（配对 McNemar p=0.188，不显著），**BF16−W4A4 = +10.21 点**；中性语料 PPL：**W8A8 +0.65%** vs W4A4 +25.0%

## 0. 一句话结论

MiniCPM5-2B 的 INT8 W8A8 起点载荷已生成：与 10 票同一把工具（llmcompressor 0.13.0 / compressed-tensors 0.18.0）、同一 BF16 源权重（sha256 `14fb8e7f…`）、同一份中性校准语料（wikitext-103 train，sha256 `e0d7ad54…`），只把 scheme 从 `NVFP4` 换成 `W8A8`——294 个 Linear 全量量化（`lm_head` 策略排除），`compressed-tensors/int-quantized` 载荷 2.84 GiB，sha256 `d639c738…`。它**能被 vLLM 0.29.0 原生服务**，但**必须显式 `--linear-backend triton`**：SM121 上不存在任何 CUTLASS INT8 内核，vLLM 自己会拒绝（见 §2）。质量代价比 4-bit 低一个数量级：中性留出 PPL 相对 BF16 只涨 **+0.65% / +0.59%**（W4A4 是 +25.0% / +25.4%）。

## 1. 载荷

### 1.1 层 census（沿用 10 票的两维 16 整除约束与排除规则）

模型 `LlamaForCausalLM`：42 层、hidden 2048、intermediate 6144、GQA 16:2、head_dim 128、vocab 130,560。`nn.Linear` 295 个，其中 **294 个进入量化**（权重参数 1,981,808,640），排除 `lm_head`（策略决定，非形状原因）。与 10 票**逐层一致**：q/k/v/o/gate/up/down 各 42 个，两维都是 16 的倍数，因此**没有形状不整除层**。

未量化的 87 个权重张量（= 381 − 294）：`lm_head`、`model.embed_tokens`、85 个 RMSNorm。这一点在本票被独立复核（§1.4），不是抄 10 票的结论。

### 1.2 配方，以及与 10 票（NVFP4 W4A4）的**每一处偏离**

| 项 | 10 票（W4A4） | 本票（W8A8） | 是否偏离 |
| --- | --- | --- | --- |
| 工具链 | llmcompressor 0.13.0 / compressed-tensors 0.18.0 / torch 2.13.0+cu130 / transformers 5.14.1 | 同（独立 venv `~/venvs/minicpm5-quant`，本票在 `goooinext` 新建，安装 65 s） | 无 |
| 修改器 | `QuantizationModifier(targets="Linear", scheme="NVFP4", ignore=["lm_head"])` | `… scheme="W8A8", ignore=["lm_head"]` | **scheme（本票的目的）** |
| oneshot 参数 | `num_calibration_samples=512, max_seq_length=2048, save_compressed=True` | 同 | 无 |
| 校准语料 | wiki103-raw train，512 段 × 2048 token，sha256 `e0d7ad54…` | 同一文件、同一 sha256 | 无 |
| 源权重 | `~/models/MiniCPM5-2B`，sha256 `14fb8e7f0a18d53d…` | 同一目录、同一 sha256 | 无 |
| 落盘格式 | `nvfp4-pack-quantized`（E2M1 + per-16 E4M3 块缩放 + FP32 全局缩放，含 `input_global_scale`） | `int-quantized`（**int8 权重 + per-output-channel 缩放**；**无任何激活侧缩放张量**） | **格式（scheme 的必然后果）** |
| 权重粒度 | 每 16 元素一个 E4M3 块缩放 | 每**输出通道**一个缩放（`strategy: channel`） | 偏离 |
| 激活粒度 | per-tensor FP32 静态 `input_global_scale`（由校准得出）+ 每 16 元素动态 E4M3 | **per-token 动态 int8**（`strategy: token, dynamic: true`），文件里不存激活缩放 | **偏离：这是本票最重要的一条** |
| 缩放张量 dtype | E4M3 块缩放 + FP32 全局缩放 | **bfloat16**（llmcompressor 自己的选择，见 §1.4） | 偏离 |
| 载荷墙钟 | 385.7 s | **62.3 s** | 附带观察 |

**交付的配方（逐字）**：

```python
QuantizationModifier(targets="Linear", scheme="W8A8", ignore=["lm_head"])
oneshot(model=..., tokenizer=..., recipe=..., dataset=<同 10 票 calib jsonl>,
        num_calibration_samples=512, max_seq_length=2048, save_compressed=True)
```

生成的 `recipe.yaml` 还记着一行值得注意的事实：`requires_calibration_data: false`——llmcompressor 自己声明 W8A8 **不需要**校准数据。

**校准数据对 W8A8 是惰性的（已实测，不是推断）**：用同一脚本、同一模型、把 `--num-samples` 从 512 降到 **8** 重新量化一遍，产出的 `model.safetensors` sha256 **逐位相同**（`d639c7389fc4c72d467a295daf55d8855cc7cb558ea9701303d7c8d1766ae905`）。原因是格式：权重缩放逐通道、激活缩放运行时逐 token 计算，文件里没有任何由校准决定的量。（该 8 样本载荷验证后已删除，磁盘上只留主载荷。）

这意味着「校准语料用中性公开语料、不用 UltraData」这条纪律在 W8A8 上**天然成立且更强**——但也意味着**不能**把 W8A8 的低代价归因于「校准得好」。

### 1.3 载荷身份

| 文件 | 字节 | sha256 |
| --- | --- | --- |
| `model.safetensors` | 3,053,374,152 | `d639c7389fc4c72d467a295daf55d8855cc7cb558ea9701303d7c8d1766ae905` |
| `config.json` | 2,047 | `55d84cfa642e25599842e4a311db60e0f3e3b3893723145746d0fde2166bde2d` |
| `payload_manifest.json` | 3,222 | 见文件内 |
| 目录合计 | **3,063,394,463 B（2.85 GiB）** | — |

- 位置（留在机器上、不入 git）：`goooinext:~/minicpm5-payload/out/MiniCPM5-2B-INT8-W8A8/`。
- 体积对照：BF16 权重 5,033,557,096 B → 载荷 **60.9%**；W4A4 载荷 2,194,830,690 B（43.6%）→ W8A8 比 W4A4 大 39%。理论构成：1.98 GB int8 权重 + 3.0 MB 通道缩放 + 1.07 GB 仍是 BF16 的 `lm_head`/`embed_tokens`。
- `config.json` 的 `quantization_config`：`format: int-quantized`、`quant_method: compressed-tensors`、`ignore: ["lm_head"]`、weights `{num_bits: 8, type: int, symmetric: true, strategy: channel, dynamic: false}`、input_activations `{num_bits: 8, type: int, symmetric: true, strategy: token, dynamic: true}`。

### 1.4 独立重建校验（不经 vLLM / compressed-tensors）

`verify_int8_payload.py` 只用 `safetensors` + `torch` 读文件，对**全部 294 层**做结构断言 + 从 BF16 源权重重算 RTN 并逐码比较（`out/verify_int8_payload.json`）：

| 指标 | 值 |
| --- | --- |
| 结构断言 | 294/294 通过：权重 int8、缩放与输出通道一一对应、**缩放 dtype 是 `torch.bfloat16`**、文件内**零个**激活侧张量（`input_scale`/`input_zero_point` 皆无） |
| 未量化权重 | 87 个（`lm_head`、`embed_tokens`、85 个 RMSNorm），符合 `ignore` 策略 |
| 复算不一致的码 | **158,002 / 1,981,808,640 = 0.008%** |
| 全模型相对 Frobenius 误差 | **1.07%**（逐层 0.89% – 3.06%） |

**「载荷 = 源权重 + 哪条配方」是实测出来的，不是假设**（`int8_recipe_forensics.py`，8 层抽样、6 条候选配方）：

| 候选配方 | 不一致码占比 |
| --- | --- |
| **R4：文件里的缩放 + 在权重 dtype（bf16）里做除法/取整/截断** | **0.0077%** |
| R3：全程 bf16 但缩放由 amax 现算 | 2.61% |
| R2：文件里的缩放 + 全程 fp32 算术 | 6.67% |
| R1：fp32 缩放 + fp32 算术（最朴素的写法） | 7.05% |

结论：**载荷 = 源 BF16 权重 + 逐通道对称缩放（`amax/127`，以 bf16 存储）+ bf16 算术下的取整**；剩下的 0.008% 是平局/零行退化。两条实用的推论：

1. 用 fp32 复算会得到 7% 的「不一致」，那不是载荷错了，是**算术精度不同**——比 10 票的平局约定（0.007%，同为 0.008% 量级）更容易误判，值得登记。
2. 有 **2 层含全零权重行**（如 `model.layers.0.mlp.up_proj`）：llmcompressor 给这些通道写缩放 0。复算时必须处理 `0/0`，否则「最大相对缩放误差」会被这一行顶到 1.0（本票第一次跑就撞上，已在报告里单列）。

## 2. 原生内核证据：SM121 到底有没有 INT8 W8A8 内核

**有，但不是默认路径。** 而且 10 票的三重口径在这里**不能照搬**——它的第一折是「启动日志里出现内核名」，而 INT8 的日志行在**内核根本不能执行**时也照样打印（见下）。本票把第一折换成**见证折**：引擎必须真的服务一个请求。

### 2.1 静态事实：这个 wheel 里编了哪些 INT8 内核

`vllm/_C_stable_libtorch.abi3.so` 里的符号与源码路径（`strings`）：

- INT8：`cutlass_scaled_mm_sm75 / sm80 / sm89 / sm90`、`cutlass_scaled_mm_azp_sm75/80/89/90`，源文件 `…/w8a8/cutlass/scaled_mm_c2x_sm89_int8_dispatch.cuh`、`…/c3x/scaled_mm_sm90_int8.cu`。**没有 `…sm100_int8`、没有 `…sm120_int8`。**
- FP8：`cutlass_scaled_mm_sm100_fp8`、`cutlass_scaled_mm_sm120_fp8`、`…blockwise_sm100/sm120_fp8` 齐全；`scaled_mm_sm120_fp8.cu` 里只有 `float_e4m3_t` 这一条类型路径。
- 因此 INT8 走的是「sm120 入口 → 内部按类型分派 → 没有 int8」。

**每个合法 `--linear-backend` 会选中什么**（`int8_kernel_evidence_summary.json` 的 `static_matrix`，由 vLLM 自己的注册表算出来，零 GPU）：

| backend | 过滤后的 INT8 候选 | 实际效果 |
| --- | --- | --- |
| `auto` | 不过滤（优先级 Cutlass → Triton → Humming） | 选中 `CutlassInt8ScaledMMLinearKernel` |
| `cutlass` | `CutlassInt8ScaledMMLinearKernel` | 选中 Cutlass → **运行时报错** |
| `triton` | `TritonInt8ScaledMMLinearKernel` | 选中 Triton → **可服务** |
| `humming` | `HummingInt8ScaledMMLinearKernel` | 选中 Humming（本机未装 humming 包） |
| 其余 17 个（`marlin`、`flashinfer_*`、`b12x`、`torch`、`deep_gemm`、`aiter`、`machete`、`fbgemm`、`conch`、`exllama`、`emulation`、`xpu*`） | **不过滤**（交集为空） | **不报错**，改发一条 WARNING 后按 auto 选择 |

最后一行是本节最容易被误解的地方：`--linear-backend marlin` 在一个 W8A8 模型上**不会**变成 W4A16 回退，它只是被忽略。

### 2.2 见证折（fold 1）：谁真的能服务

三个 backend 的真实启动（`int8_kernel_evidence.py`，每次启动新引擎、发一个 1602-token 提示词）：

| backend | 选中（日志） | 健康？ | 结果 |
| --- | --- | --- | --- |
| `cutlass` | `CutlassInt8ScaledMMLinearKernel` | **否**（48 s 即死） | `Int8 not supported on SM121…`，`Engine core initialization failed` |
| `marlin` | `CutlassInt8ScaledMMLinearKernel` | **否**（同上） | 先 WARNING「marlin 没有该层类型的内核」，随后同样报错 |
| `triton` | `TritonInt8ScaledMMLinearKernel` | **是**（99 s） | 完成 1602-token prefill + 16 token 生成 |

**默认 backend 的完整报错（原文）**：

```text
(EngineCore pid=291140) INFO 09-10 21:38:06 [__init__.py:761] Selected CutlassInt8ScaledMMLinearKernel for CompressedTensorsW8A8Int8
(EngineCore pid=291140) ERROR 09-10 21:38:06 [core.py:1374] RuntimeError: dispatch_scaled_mm, /workspace/csrc/libtorch_stable/quantization/w8a8/cutlass/c3x/scaled_mm_helper.hpp:34, Int8 not supported on SM121. Use FP8 quantization instead, or run on older arch (SM < 100).
(APIServer pid=290987)     raise RuntimeError(
(APIServer pid=290987) RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}
```

**反例 backend（`marlin`）的原文**：

```text
(EngineCore pid=292090) WARNING 09-10 21:40:08 [__init__.py:349] --linear-backend=marlin has no kernel for this linear layer type; using automatic selection for that layer type.
(EngineCore pid=292090) ERROR 09-10 21:40:43 [core.py:1374] RuntimeError: dispatch_scaled_mm, … Int8 not supported on SM121. …
```

**能跑的那个（`triton`）**：

```text
(EngineCore pid=291523) INFO 09-10 21:38:22 [__init__.py:761] Selected TritonInt8ScaledMMLinearKernel for CompressedTensorsW8A8Int8
```

**给 05/11/后续票的硬结论**：`Selected <Kernel> for …` 这一行**不能**作为 INT8 的原生证据。它在 `cutlass` 上打印了、也是完整的类名，但引擎 48 秒后就死。日志折必须与见证折合读。

### 2.3 profiler 折（fold 3）与 PTX 归属折（fold 4）

`triton` 臂一次 prefill（1602 token）的 CUDA kernel 分类（`kernel_evidence_int8-triton.json`）：

| 类别 | 调用数 | 时长占比 | 代表 kernel |
| --- | --- | --- | --- |
| `int8_candidate` | 924 | **72.16%** | 见下 |
| `other` | 252 | 19.31% | attention / 采样 / 调度 |
| `other_gemm` | 126 | 8.52% | FlashAttention（bf16，符合预期） |
| `dequantization` | **0** | **0** | — |

`int8_candidate` 明细：

- `scaled_mm_kernel`（**336 次**，21.6 ms）——≥ 294 = 42 层 × 7 个 Linear，与「所有被量化的 Linear 都走这个 kernel」一致；
- `triton_poi_fused_dynamic_scaled_int8_quant_mul_silu_slice_1`（84 次）、`triton_red_fused_dynamic_scaled_int8_quant_fused_add_rms_norm_0`（84 次）、`triton_red_fused__to_copy_dynamic_scaled_int8_quant_embedding_rms_norm_0`（2 次）——**融合的动态 INT8 激活量化**（RMSNorm/SiLU + 量化），与 §1.2 的 `strategy: token, dynamic: true` 一一对应。

**名字本身不足以证明「这是 int8」**（`scaled_mm_kernel` 这个名字 fp8 也会用），所以本票用**独立 Triton cache**（`TRITON_CACHE_DIR` 指向本次运行专属目录）把本次运行编译出的 kernel 抓下来归属：

```text
/home/goooice/minicpm5-payload/out/w8a8/triton_cache_triton/6IJ6AD…/scaled_mm_kernel.ptx
  .target sm_121a
  mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32   × 128
```

TTGIR 同址可查 `tt.dot %a, %b, %acc … tensor<128x128xi8>`。即：**本机为 sm_121a 编译出的操作数是 int8、累加是 s32、指令是 `mma.sync` INT8 tensor-core MMA**——原生，不是反量化、不是 bf16 wmma。微基准旁证：同一形状（512×2048×2048）上 Triton INT8 52.9 TFLOP/s vs cuBLAS BF16 67.8 TFLOP/s（78%）；如果是「先反量化再 bf16 GEMM」，这里应当**慢于** BF16，而不是接近它。

### 2.4 判定（三重口径的 W8A8 版本）

| 口径 | 要求 | 本票实测 |
| --- | --- | --- |
| 见证 | 引擎真的服务请求 | `triton` 通过（1602 token prefill + 16 token）；`cutlass`/`marlin` 不通过 |
| 日志 | 启动日志出现内核名 | 三者都出现（**因此这条单独不充分**） |
| profiler | 出现原生路径、不得被反量化主导 | `int8_candidate` 72.16%、`dequantization` 0、bf16 GEMM 只有 attention |
| 归属（新增） | 本次运行编译的 kernel 是 int8 MMA | `.target sm_121a` + `mma.sync…s32.s8.s8.s32` |
| 反例 | 该被拒绝的会被拒绝 | `cutlass` / `marlin` → `Int8 not supported on SM121`；`marlin` 另附「无该层类型内核、回落 auto」警告 |

**结论：SM121 上有原生 INT8 W8A8 内核（Triton 路径），vLLM 的 CUTLASS INT8 路径不存在且 vLLM 自己明说要用 FP8 代替。**

## 3. 质量差

### 3.1 题集三臂（v3 dev，480 题，冻结口径，单次同机）

口径**逐字沿用冻结协议**：`score_taskset.py`（用 **run300 的那一份**，md5 `ca2b0a1519d4f0490bd6b7ddb1e23fd7`，即产出 v3 池归档数字的同一脚本）+ `--arm` TF/GEN + `enable_thinking=False` + 候选池 `taskset_dev.jsonl` sha256 `e7a7d29a0b5cb90a…` + `max_model_len=1024`/`gpu_mem=0.30`/关 prefix cache。三臂只有 checkpoint 与 backend 不同。

| arm | TF 准确率 | GEN 准确率 | gold 平均 logprob | 「不回答」率 |
| --- | --- | --- | --- | --- |
| **BF16**（`~/models/MiniCPM5-2B`） | **0.6021** | 0.6000 | −0.9521 | 10.00% |
| **W8A8**（本票载荷，`--linear-backend triton`） | **0.6208** | 0.6125 | −0.9173 | 7.92% |
| **W4A4**（10 票 q0 载荷，`--linear-backend cutlass`） | **0.5000** | 0.4875 | −1.1701 | 0.63% |

| 差值 | TF 点数 | 配对 McNemar（480 题） | 「双方都作答」子集 |
| --- | --- | --- | --- |
| BF16 − W4A4 | **+10.21** | 90 vs 41，p ≈ 0（显著） | n=431，**+15.31** |
| BF16 − W8A8 | **−1.88** | 14 vs 23，**p=0.188（不显著）** | n=421，**−0.95** |
| W8A8 − W4A4 | **+12.08** | 94 vs 36，p ≈ 0（显著） | — |

逐 family（TF）：

| family | n | BF16 | W8A8 | W4A4 | BF16−W8A8 | BF16−W4A4 |
| --- | --- | --- | --- | --- | --- | --- |
| arith | 96 | 0.7396 | 0.7292 | 0.5000 | +1.04 | +23.96 |
| logic | 96 | 0.4583 | 0.4896 | 0.5000 | −3.12 | −4.17 |
| modulo | 96 | 0.5312 | 0.5521 | 0.5000 | −2.08 | +3.12 |
| ordering | 96 | 0.4583 | 0.5312 | 0.5000 | −7.29 | −4.17 |
| pct | 96 | 0.8229 | 0.8021 | 0.5000 | +2.08 | +32.29 |

逐 stratum 的完整表在 `w8a8_accuracy_table.md`。两处必须与结论同读：

1. **BF16 与 W8A8 都是「难族退化」的形状**（logic/hard：BF16 0.1562 / W8A8 0.3125；ordering/hard：BF16 0.3125 / W8A8 0.3750），而 W4A4 是**被配平成 0.5000** 的（池按 q0 配平，所以它在每个 stratum 都是 0.5）。这不是能力差，是池的构造。
2. **「不回答」伪影在 BF16 与 W8A8 上同时存在、在 W4A4 上几乎不存在**：BF16 10.0%（48/480，logic 21.9% / ordering 28.1%）、W8A8 7.9%（38/480，logic 22.9% / ordering 16.7%）、W4A4 0.63%（3/480）。BF16 在这 48 题上**一个都没作答**，因此它的 0.6021 里含 4.8 个百分点的「因拒绝作答而丢分」。

**把这个伪影按两种方式各归一化一次，得到头寸数的区间**（`refusal_difficulty_proxy`，同一批题、同一口径）：

| 归一化 | 结果 | BF16 − W8A8 |
| --- | --- | --- |
| 不归一化（冻结口径的原始读数） | BF16 0.6021 / W8A8 0.6208 | **−1.88 点** |
| 把 BF16 沉默的 48 题按 **W8A8 在这些题上的实际得分**（0.250）计入 | BF16 → 0.6271 | **+0.62 点** |
| 把 W8A8 沉默的 38 题按 **BF16 在这些题上的实际得分**（0.184）计入 | W8A8 → 0.6354 | **−3.33 点** |

**因此本票的头条数字是：BF16 − W8A8 = −1.88 点（区间 −3.33 ~ +0.62），配对检验不显著，dev 池无法把 8-bit 的量化代价与 0 分开；对照的 4-bit 代价是 +10.21 点（区间同向且显著）。** 注意这三个数**都**是同一批 480 题上的区间刻画，不是三次独立实验；选择哪一个当「真值」取决于你怎么处理那个被 10/19 票定性的后缀伪影。

**三条独立的复算/复现（本票最强的可靠性证据）**：

1. **归档数字在 TF 轴上逐位复现**：BF16 = 0.6021、W4A4 = 0.5000，与 v3 池归档的 `scores_bf16_dev3.jsonl` / `scores_q0_dev3.jsonl` **逐题一致（480/480，零处不同）**，逐 stratum 也完全一致（BF16 的 15 个 stratum、W4A4 的全部 0.5）。
2. **评分器确定性**：同一协议跑两遍（冻结脚本 vs 带诊断字段的脚本），三臂 × 480 题的 `tf_correct` 与 `tf_token_hits` **零处不同**。
3. **engines 与载荷未被污染**：全部三臂的 `n_unstable_prefix = 0`、self-test 全过（`tf_firstgold_top1_eq_greedy` 8/8、`prompt_has_closed_think` 8/8）。

**一处必须如实登记的差异（GEN 轴，不是 TF）**：与归档的 dev3 行逐题比较，**TF 480/480 一致**，但 GEN 臂在 BF16 上有 **1/480**、在 W4A4 上有 **9/480** 项不同（全部在 `logic` 族，且是**生成文本不同**，如归档 `'2'` vs 本票 `'3'`）。机制是批组成：归档的 dev3 行是从**整池 23,708 题**的一次评分里切出来的子集，本票是 480 题的独立运行，贪心解码在近似平局处会因归约顺序不同而分叉；TF 是「拼上 gold 一次前向取 argmax」，对批组成远不敏感，所以零差异。**后果：GEN 列只能在同一 session、同一池大小的三臂之间比较（本票的三臂满足），不能与归档子集数字直接相减**；这也是冻结协议把 TF 定为主轴的原因。

### 3.2 中性留出语料的 PPL（与 10 票同口径，同机同 venv）

| 语料 | BF16 PPL | W8A8 PPL | ΔPPL | 相对 | W4A4 PPL | 相对 |
| --- | --- | --- | --- | --- | --- | --- |
| wikitext-103 test | 13.2891 | **13.3759** | +0.0868 | **+0.65%** | 16.6119 | +25.0% |
| wikitext-103 validation | 13.8146 | **13.8961** | +0.0815 | **+0.59%** | 17.3234 | +25.4% |

BF16（13.2891 / 13.8146）与 W4A4（16.6119 / 17.3234）**逐位复现 10 票的归档数字**，所以同一张表里的 W8A8 读数是可比口径。8-bit 的代价是 4-bit 的 **1/40**。

### 3.3 怎么读

- 「量化代价」在两个轴上都成立且同向：**PPL 上 W8A8 只贵 0.6%，W4A4 贵 25%；题集上 W8A8 与 BF16 不可区分，W4A4 便宜 10 点以上**（且配对显著）。
- 题集轴**不能**支持「W8A8 涨点」的主张：它只是「测不出代价」，而 dev 池本身的分辨率（每族 12–15 点、每候选 SE ≈1.9 点，见 17/19 票）远大于 2 点量级。要判定 W8A8 是否真的无损，必须用 confirm 尺度的池子（17 票：confirm 每族 801 episode → 1.8 点）。
- 顺带得到一条对**整个 W4A4 自改进栈**有用的刻度：把 §3.1 的「双方都作答」子集与伪影归一化合起来看，**4-bit 的真实代价在 dev 上是 10–15 点量级**，而不是被伪影压低后的 10.21。

## 4. 机时记账（`goooinext`，单机；`gooo` 全程未使用）

不重叠占用区间（`goooinext` 本地时间；区间按并集合并，CPU-only 的分析/校验单列）：

| 区间 | 作业 | 时长 |
| --- | --- | --- |
| 21:27:31 – 21:28:36 | 新建 `~/venvs/minicpm5-quant`（llmcompressor 0.13.0，CPU 为主） | 1.1 min |
| 21:29:20 – 21:34:01 | INT8 内核探针 v1/v2/v3（含微基准；v1/v2 是本票自己的布局错误，见坑 2）+ **W8A8 量化（62.3 s）** | 4.7 min |
| 21:37:21 – 21:45:10 | 内核证据三 backend 启动（cutlass/triton/marlin）+ 冻结口径三臂评分 | 7.8 min |
| 21:49:10 – 22:02:10 | 确定性复跑三臂 + **PPL 六次测量** + GEN 臂三臂 + 8 样本惰性验证（后三者互相重叠） | 13.0 min |
| **设备机时合计（并集）** | | **≈ 26.6 min ≈ 0.44 h** |

- 上限 **2.0 h**，用掉 **≈22%**；无失败重跑（除下列两次**有意的**失败：`cutlass`/`marlin` 反例启动各一次，均在 1 min 内被脚本自行终止）。
- 另有 CPU-only 工作（不入设备机时）：llmcompressor 安装 65 s、294 层独立重建校验 ≈2 min、配方取证 ≈20 s、分析与文档。
- 内存纪律：全程 `free -b` 可用 ≥112 GiB（该机无常驻 `mineru-api`）；峰值来自 PPL/评分引擎（`gpu-memory-utilization 0.30` ≈ 36 GiB 上限）与量化的 BF16 加载（≈10 GB），**未触发任何内存告警**；结束后复核 0 个残留进程。

## 5. 坑与失败（可复用）

1. **`Selected <Kernel> for <Scheme>` 不是内核证据。** 见 §2.2：`cutlass` 上这行完整打印、48 s 后引擎就死了。INT8 的第一折必须换成「真的服务了一个请求」。
2. **vLLM 的 stable-ABI 报错会把消息丢掉**，只剩 `op_name, file:line,`。本票第一次的探针调用把 `b` 传成了 `.t().contiguous()`（行主序），于是卡在 `scaled_mm_entry.cu:209` = `STD_TORCH_CHECK(b.stride(0) == 1)`（列主序布局检查）——报错文本是空的，看上去**像**架构不支持的硬失败。差一点就把「我调用错了」写成「SM121 不支持」。（正确写法：`b = w.t()`，不 `.contiguous()`；vLLM 的 int8 kernel 装的就是这个转置视图。）
3. **INT8 的 `--linear-backend` 大多数取值是被静默忽略的**：交集为空时不报错，只 WARNING「has no kernel for this linear layer type; using automatic selection」，然后按 auto 选。不要以为 `marlin` 会给出一个 W4A16 回退臂——它给出的是同一个 Cutlass 错误。
4. **`weight_scale` 是 bf16 存储的**，且量化算术在 bf16 里做。用 fp32 复算会得到 7% 的「码不一致」（见 §1.4 表）——这是复算口径问题，不是载荷问题。
5. **全零权重行**：2 层有整行 0，缩存放 0；复算时 `0/0` 会把「最大相对缩放误差」顶到 1.0。
6. **评分进程在收尾时 `terminate called without an active exception`（exit 134）**：vLLM 0.29.0 的 teardown 崩溃，发生在**结果与 summary 都已落盘之后**（bf16/w4a4 臂各一次，w8a8/GEN 臂没有）。判定依据是不能只看退出码：本票所有 exit 134 的臂，其 480 行结果与归档数字逐位一致。
7. **`ssh '<cmd> &'` 里 `cd` 只影响后台子壳**：`cd X && nohup … & sleep 3; tail logs/y` 的 `tail` 仍在 `$HOME` 下，会「找不到文件」。用绝对路径或拆成两条命令。
8. 环境修复照旧（06/10 票）：`PATH` 里 venv `bin` 在前、`CPATH` 指向 `~/.cache/minicpm5/python-dev/usr/include/python3.12`、`VLLM_WORKER_MULTIPROC_METHOD=spawn`。**未**使用 `flashinfer_cutlass`（10 票的停摆事故纪律），全程只用 `cutlass`（W4A4/BF16）与 `triton`（W8A8）。

## 6. 证据产物

随笔记入库的紧凑件在 [`artifacts/minicpm5-w8a8-payload-20260910/`](artifacts/minicpm5-w8a8-payload-20260910/)：

| 文件 | 内容 |
| --- | --- |
| `payload_manifest.json` | 载荷文件清单、哈希、配方、校准记录、工具版本 |
| `verify_int8_payload.json` | 294 层结构断言 + RTN 重建（含逐层表、未量化清单、全零行） |
| `int8_recipe_forensics.json` | 6 条候选配方的逐层/加权不一致率（R1–R6） |
| `int8_kernel_probe.json` | 逐算子探针（vLLM cutlass int8/fp8、triton int8、`torch._scaled_mm`） |
| `int8_kernel_evidence_summary.json` | **21 个合法 `--linear-backend` 覆盖到哪些 INT8 内核类** + 三 backend 汇总 |
| `kernel_evidence_int8-cutlass.json` / `-marlin.json` | 两条**被拒绝**路径：选中行、完整报错、`Engine core initialization failed` |
| `kernel_evidence_int8-triton.json` | 正例：选中行、见证请求、profiler 分类、Triton cache 的 PTX 归属 |
| `scores_{bf16,w8a8,w4a4}_dev3.jsonl` + `_summary.json` | 三臂逐题结果与摘要（冻结协议，含 GEN 臂） |
| `w8a8_accuracy_report.json` / `w8a8_accuracy_table.md` | 统计汇总：逐 family/stratum、配对 McNemar、双方作答子集、伪影归一化 |
| `ppl_{bf16,nvfp4_w4a4,int8_w8a8}_wikitext103_{test,validation}.json` | 6 次 PPL（含逐序列明细） |

**留在机器上、不入库**：载荷本体（`goooinext:~/minicpm5-payload/out/MiniCPM5-2B-INT8-W8A8/`，2.85 GiB）、三 backend 的 serve 原始日志（`~/minicpm5-payload/out/w8a8/serve_int8-*.log`）、profiler trace 与专属 Triton cache（`~/minicpm5-payload/out/w8a8/`）。

脚本（本工作区分支 `codex/wayfinder-minicpm5-w8a8`，源自 `codex/wayfinder-minicpm5-improve`）：`quantize_int8_w8a8_llmcompressor.py`（与 10 票脚本的 diff 只有 docstring / `--scheme` 默认值 / 格式串三处实质改动）、`run_quantize_w8a8.sh`、`int8_kernel_probe.py`、`int8_kernel_evidence.py`、`verify_int8_payload.py`、`int8_recipe_forensics.py`、`run_accuracy_w8a8.sh`（冻结评分器，`--arm tf`）、`run_accuracy_instrumented.sh`（同协议 + 诊断字段的复跑）、`run_accuracy_both.sh`（同协议 + GEN 臂）、`run_quant_calib8.sh`（8 样本惰性验证）、`analyze_w8a8_accuracy.py`、`run_ppl_arms.sh`、`ppl_vllm.py`（只加 `--linear-backend`，默认 None，10 票调用逐字不变）。

## 7. 残余风险与不要过度声称的地方

1. **`--linear-backend triton` 是本票的被迫偏离**，不是选择：冻结协议写的是 `linear_backend=cutlass`，而 SM121 上 INT8 + cutlass **无法启动**（§2.2）。因此**本票的三臂不是同一 backend**（W4A4/BF16 走 cutlass，W8A8 走 triton）。这不影响质量读数（质量由权重与激活量化格式决定），但**任何未来的训推一致性/吞吐对齐必须单独处理**：Triton INT8 的数值路径与 CUTLASS 家族不同，11 票的 refit 契约、CUDA graph、prefix cache 交互都**没有**在 triton 上验过。
2. **本机没有 FP8 对照臂**。vLLM 的报错自己建议「Use FP8 quantization instead」，且 wheel 里 sm120 的 FP8 内核齐全（探针实测 `cutlass_scaled_mm` FP8 在 SM121 上**能跑**，kernel 名是 `enable_sm120_family … MainloopSm120TmaWarpSpecialized … float_e4m3_t … SM120_16x8x32_TN`）。若「SM121 上要一个非 Triton 的原生 8-bit 路径」，FP8 W8A8 是现成候选——但那是**另一个格式**（FP8 ≠ INT8），与 QES 的 W8A8 设置不同，本票**没有**测它的质量。
3. **题集轴的读数受池子分辨率限制**：dev 480 题、每族 32 题，BF16−W8A8 的差异落在 ±2 点量级，**本票无法给出「W8A8 无损」的正面结论**，只能给出「测不出代价」。要正面主张必须换 confirm 尺度的池子（16/17 票口径）。
4. **伪影归一化是三种读法，不是三个实验**：§3.1 的 −1.88 / +0.62 / −3.33 都来自同一批 480 题。**它们不能被当作误差棒平均掉**；正确说法是「区间 −3.33 ~ +0.62 点」。
5. **PPL 只在 wikitext（同分布、中性）上测**：与 10 票同样的污染声明——它给的是同分布下的下界式读数，不能外推到指令/对话流量；也没有测 KV cache 精度、长上下文或吞吐。
6. **载荷未做端到端「训练-推理一致」验证**；本票只证明它能被服务、且服务路径是本机编译的 int8 MMA。`lm_head`/`embed_tokens` 保持 BF16 仍是策略选择（与 10 票一致），未做纳入量化的对照。
7. **只测了一台机器**（`goooinext`）。同型号 `gooo` 未参与，因为它在跑另一条实验且本票被要求不要碰它；载荷哈希与内核选择在另一台同型号机器上未复核。
8. **GEN 轴对批组成敏感**（§3.1）：同一模型同一提示，换池子大小/换 session 会在 ~0.2–1.9% 的 item 上给出不同的贪心文本。凡是用 GEN 数字做的跨轮比较（例如与归档 dev3 子集）都有这个量级的不可复现性；TF 不受影响。
