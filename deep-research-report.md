# 还原 Hugging Face 仓库 prism-ml Ternary-Bonsai-27B-mlx-2bit 的模型量化过程分析报告

## 执行摘要

这份仓库里的“2bit”并不等同于传统意义上的“每个权重真正在信息论上只占 2 bit 的对称/非对称整型量化”，而更准确地说，是一种**以三值权重 `{−1, 0, +1}` 为核心、按 128 个权重分组共享缩放因子的 ternary g128 表示**，再为了适配 MLX 现有 grouped affine quantization 容器，把三值权重装进 2-bit slot 中，并额外为每个组存储一个 bias。公开材料明确写到：原生 Bonsai 表示每组只需要一个 FP16 scale，理想位宽约为 `log2(3) + 16/128 ≈ 1.71 bpw`；但当前部署内核按 2-bit slot 存放三值码，因此原生部署格式是 `2 + 16/128 = 2.125 bpw`；而这个 **MLX safetensors 仓库**由于 MLX 还要求“每组 scale + bias”双参数，所以落盘有效位宽进一步变成 `2 + 32/128 = 2.25 bpw`。换句话说，**仓库名里的“2bit”更接近“runtime slot width / 容器位宽”，而不是原始 ternary 表示的真实平均信息位宽**。citeturn13view1turn14view0turn25view0turn25view1

从仓库结构看，这个 MLX 包是一个**单文件 safetensors** 模型目录：`config.json` 同时在 `text_config.quantization` 与顶层 `quantization` 写明 `group_size=128, bits=2`；`model.safetensors.index.json` 的 `weight_map` 显示，大量语言模型线性层都以 `weight / scales / biases` 三元组出现，例如 `...self_attn.q_proj.weight`、`...self_attn.q_proj.scales`、`...self_attn.q_proj.biases`，而 RMSNorm、部分卷积或其他小张量则保持普通高精度参数名，不带 `scales/biases`。这强烈表明：**语言主干的主要线性权重是 weight-only 的分组低比特量化；scale/bias 按组存储；“尾部”参数与归一化参数保留高精度。**citeturn5view0turn3view0turn4view0turn14view0

就“量化算法”本身而言，公开仓库**没有披露**完整训练/量化脚本，因此无法严谨地断言它是 GPTQ、AWQ、LSQ、HQQ 或某个标准论文实现的直接复现。相反，Prism ML 的模型卡与白皮书摘要都把 Bonsai 描述成一种**专有的 binary/ternary 表示框架**，其目标是把现成的 Qwen3.6-27B “迁移”到 binary/ternary 表示，而不是像 BitNet 那样从零训练专门的 BitLinear 架构；同时，模型卡还明确指出**视觉塔使用的是 HQQ 4-bit**，这反而说明**语言模型权重并不是 HQQ 4-bit**。因此，最稳妥的判断是：**这个仓库落地的语言权重格式是“Bonsai ternary g128 + MLX affine 容器适配”，而不是公开标准 GPTQ/AWQ/HQQ 文件格式。**citeturn13view1turn42view0turn11search2turn11search8turn32view0

在二进制层面，本仓库虽然无法直接在线读取 8.49 GB 权重全量内容，但已经能从文件树、safetensors index、LFS/Xet 指针与 safetensors 规范中重建出高可信的容器结构：`model.safetensors.index.json` 的 `metadata.total_size` 是 `8,490,506,720` 字节，而 LFS 指针页给出的真实文件长度是 `8,490,785,104` 字节，两者相差 `278,384` 字节；结合 safetensors 规范“前 8 字节为 header 长度、随后为 JSON header、其后为连续 tensor payload”的定义，可以高概率推断该文件的 JSON header 长约 `278,376` 字节，外加 8 字节长度前缀，总共正好补齐差值。这个推断对本地复现解析尤其重要。citeturn3view0turn28view0turn29calculator0turn30view0turn10search2

## 证据基础与仓库清单

本报告使用的主要证据包括：该 Hugging Face 仓库的模型卡、文件树、`config.json`、`model.safetensors.index.json`、`model.safetensors` 指针页与 SHA256/Xet 哈希、PrismML 官方文档、Bonsai demo 仓库、MLX 官方量化文档、以及 GPTQ、AWQ、LSQ、BitNet / BitNet b1.58 等原始论文或官方文档。由于本地权重文件未提供，且在线环境未直接抓取完整 8.49 GB 权重体，因此**报告对“算法训练细节”和“逐 tensor 精确 `data_offsets`”采用“仓库公开证据 + 可本地验证步骤”的方式还原**。citeturn44view0turn3view0turn27view0turn23view0turn24view0turn25view0turn25view1turn12search0turn12search1turn12search2turn43search0

下表整理了当前仓库中可直接确认的核心文件与尺寸；其中 `model.safetensors` 是单文件权重主体，`model.safetensors.index.json` 把所有 tensor 都映射到这个单一 safetensors 文件。citeturn44view0turn27view0turn28view0

| 文件 | 作用 | 大小 |
|---|---|---:|
| `README.md` | 模型卡与方法说明 | 23.2 kB |
| `config.json` | 模型与量化配置 | 3.79 kB |
| `model.safetensors` | 单文件权重主体 | 8.49 GB |
| `model.safetensors.index.json` | tensor 到 safetensors 的索引映射 | 186 kB |
| `tokenizer.json` | tokenizer 主文件 | 20 MB |
| `merges.txt` | BPE merges | 3.35 MB |
| `vocab.json` | 词表 | 6.72 MB |
| `chat_template.jinja` | 对话模板 | 7.76 kB |
| `processor_config.json` | 多模态 processor 配置 | 44 B |
| `preprocessor_config.json` | 图像预处理配置 | 390 B |
| `video_preprocessor_config.json` | 视频预处理配置 | 385 B |
| `LICENSE.txt` / `NOTICE.txt` | 许可与声明 | 10.2 kB / 411 B |

进一步看配置文件，可以确认这是一个**Qwen3.5 系的 27B 混合注意力多模态模型**：文本部分 64 层，`full_attention_interval=4`，`num_hidden_layers=64`，`hidden_size=5120`，`num_attention_heads=24`，`num_key_value_heads=4`，上下文长度 `262144`；`vision_config.depth=27`。`full_attention_interval=4` 也与模型卡“64 层中只有 16 层需要 full-attention KV cache”的描述互相印证。配置文件同时把量化参数写成 `group_size=128, bits=2`。citeturn5view3turn5view0turn14view0

索引文件则暴露了非常关键的张量命名模式。以语言模型为例，`embed_tokens`、`lm_head`、`self_attn.q_proj/k_proj/v_proj/o_proj`、`mlp.up_proj/gate_proj/down_proj` 等重权重层，普遍都有 `.weight + .scales + .biases` 三元组；而 `input_layernorm.weight`、`post_attention_layernorm.weight`、`q_norm.weight`、若干卷积权重等则没有对应的 `.scales/.biases`。这说明仓库至少在**权重覆盖范围**上采取了“主矩阵低比特、归一化与少量结构参数高精度留存”的策略，而不是全参数无差别 2-bit 量化。citeturn3view0turn4view0turn14view0

还需要特别指出一个**文档层面的不一致点**。PrismML 文档页写着“所有 27B 仓库都带一个 `mmproj` 文件”；但这个具体的 MLX 仓库文件树中**没有单独的 `mmproj` 文件**，同时模型卡又明确写着“MLX pack measures 8.49 GB on disk，vision tower bundled in the same file”。再结合索引文件中可见的大量 `vision_tower.*` 张量名，可以高度可信地判断：**对于这个 MLX 仓库，视觉相关权重已经并入 `model.safetensors` 单文件中。**citeturn23view0turn14view0turn44view0turn4view1

最后，权重文件本身还能确认两个校验信息：Xet hash 为 `f1fda971...7ae47c3`，LFS 指针中的 SHA256 为 `8acd4597...e45846b`，真实大小是 `8,490,785,104` 字节。这些值对于本地校验下载完整性非常有用。citeturn27view0turn28view0

## 量化流程还原与算法判断

### 最可信的公开还原结论

基于模型卡、配置与索引，当前最可信的流程并不是“对现有 FP16 权重做一次普通 int2 PTQ 并导出通用格式”，而是下面这条链路：

```mermaid
flowchart TD
    A[Qwen3.6-27B 基座] --> B[迁移到 Bonsai ternary 表示]
    B --> C[按最后一维分组 g=128]
    C --> D[每组权重投影到 {-1,0,+1}]
    D --> E[为每组求 scale]
    E --> F[得到原生 ternary g128 表示]
    F --> G[部署侧用 2-bit slot 打包三值码]
    G --> H[GGUF Q2_0 / 自定义 kernel 直接消费]
    F --> I[为适配 MLX affine 容器补上 bias]
    I --> J[生成 weight/scales/biases safetensors]
    J --> K[MLX stock runtime 直接加载]
```

这条还原链路背后的核心证据有四条。第一，模型卡直接把语言模型表示定义为 **“Ternary g128: {−1,0,+1} weights with FP16 group-wise scaling”**，并给出理想位宽 1.71 bpw 与当前 2-bit slot 部署位宽 2.125 bpw。第二，配置明确写出 `group_size=128, bits=2`。第三，索引文件把主要语言线性层全部组织成 `weight/scales/biases` 三元组，这与 MLX grouped affine quantization 的张量组织方式完全一致。第四，MLX 官方文档说明 `quantized_matmul` / `dequantize` 都是“packed unsigned 32-bit integers + per-group scale and bias”的 grouped affine 接口，而 Bonsai 模型卡又明确说它是通过 `s_mlx = 2·s_g` 与 `bias = -s_g` 把**原生只含 scale 的 ternary 表示**嵌入到 MLX 容器中。

> **⚠️ 修正注记（2026-07-29 实验验证，已用真实 Bonsai 数据证实）**：上述 `s_mlx = 2·s_g, bias = -s_g` 的映射对于三元 {-1,0,+1} 三个电平是**数学上不可能的**。穷举验证表明，在 2-bit affine 容器（code 0–3，dequant = scale·q + bias）中表达三值 {-s_g, 0, +s_g} 的唯一有效映射是 **`scale = s_g, bias = -s_g, codes {0,1,2}`**（code 0→-s_g, code 1→0, code 2→+s_g, code 3 不用）。`s_mlx = 2·s_g` 只能产生两个电平 {-s_g, +s_g}（即二值 ±d），无法表示三元中的 0。详见 `src/shadow_peft/quantization/affine_2bit.py` 的实现与 `tests/test_quantization.py::TestQuantize2BitTernary` 的验证。报告其余关于格式、容器、张量组织的结论不受此修正影响。

> **✅ 真实数据证实（2026-07-29）**：上述修正注记已从「数学穷举推断」升级为「直接证实」。下载真实的 `prism-ml/Ternary-Bonsai-27B-mlx-2bit`（packed）与 `prism-ml/Ternary-Bonsai-27B-unpacked`（FP16 基准），对全部 **498 个量化三元组逐层比对**：dequantized packed 权重与 unpacked FP16 权重的 **MSE 中位数 = 0、cosine = 1.000000**（bit-exact 一致，仅 embed_tokens 有 2.88e-20 级 fp16 舍入噪声）。对 6 个不同类型小层（embed_tokens / in_proj_a / in_proj_b / out_proj / k_proj / lm_head）穷举全部 24 种 code→ternary 映射，**最优映射全部为 `[-1,0,1,0]`（MSE=0）**，证实 code 3 在真实数据中确实未被使用，`scale=s_g, bias=-s_g` 是唯一解。复现脚本与产物见 `experiment/analyze_bonsai_vs_unpacked.py` 与 `experiment/analysis_bonsai_vs_unpacked/REPORT.md`。citeturn13view1turn5view0turn14view0turn25view0turn25view1

因此，对用户最重要的结论是：**这个仓库里的“2bit”本质上是一个“ternary-like、group-wise、weight-only、runtime-packed、MLX-containerized”的表示。** 它不是标准 GPTQ int2，不是 AWQ int2，也不是 HQQ int2，更不是常规对称 int2 的“scale + zero_point”导出。真正被仓库公开出来的，是**结果格式**与**runtime 可消费的数据结构**，而不是完整训练配方。citeturn13view1turn24view0turn32view0

### 与常见量化方法的对比判断

下表总结了常见候选方法与本仓库证据的匹配度。这里最关键的一列不是“谁最先进”，而是“谁最符合这个仓库实际暴露出来的文件结构与模型卡描述”。citeturn12search0turn12search1turn12search2turn11search2turn11search14turn43search0

| 方法 | 典型特征 | 优点 | 局限 | 与本仓库匹配度 |
|---|---|---|---|---|
| **Bonsai ternary g128** | `{−1,0,+1}`，组大小 128，组级 scale，当前部署按 2-bit slot 存三值码 | 极低位宽下仍保留较强推理；文件/内核设计直接围绕 ternary runtime 展开 | 公开训练细节不足；需要专门 kernel 或容器适配 | **最高**。模型卡、配置、索引、runtime 文档都直接支持。citeturn13view1turn14view0turn24view0 |
| GPTQ | 基于近似二阶信息的一次性 PTQ，常见 3/4 bit，也报告过 2-bit/ternary 实验 | 推理精度高，适合权重量化 | 公开导出格式通常不是这个 `weight/scales/biases` + ternary 品牌格式 | **低到中**。可作为“可能参考思想”，但不是仓库暴露的直接格式。citeturn12search0turn12search8 |
| AWQ | 激活感知的权重保护与 per-channel scaling，常见 4-bit | 工程成熟、硬件友好 | 主要面向 4-bit；公开证据与本仓库 ternary g128 不一致 | **低**。更像对照组，不像实际实现。citeturn12search1turn11search5 |
| HQQ | Half-Quadratic Quantization，可无校准量化，多 bit-width | 不依赖校准数据，支持快后端 | 官方文档也承认 `<4-bit` 精度退化风险明显 | **仅视觉塔匹配**。模型卡明确 HQQ 4-bit 用于 vision tower。citeturn11search2turn11search8turn11search14turn13view1 |
| LSQ / LSQ+ | 训练时学习 step size / offset 的 QAT 方法 | 低 bit 下训练效果通常优于很多 PTQ | 需要训练介入；仓库未公开此类训练图 | **概念上可能相关，证据不足**。若 Bonsai 做了再优化，LSQ 类思想比 GPTQ/AWQ 更接近“学习式量化”。citeturn12search2turn12search6 |
| k-means / codebook | 用聚类中心替代实值权重 | 可用非均匀 codebook 表示 | 文件中通常会出现 codebook / centroid 概念；本仓库未见 | **低**。仓库公开结构不支持 codebook 解释。citeturn12search3turn12search11turn12search15 |
| BitNet b1.58 | 从训练阶段就把线性层做成 ternary / 1.58-bit 体系 | 从架构到训练都为 ternary 优化 | 一般是“从零或长程训练”的新模型，不是迁移现有模型 | **中等的“概念亲缘性”**。Bonsai 明确说自己与 BitNet 路线相反：它从现成预训练模型迁移到 binary/ternary 表示。citeturn43search0turn43search1turn42view0 |

### 标量、向量、逐层、逐通道策略判断

从公开证据看，最好把它理解成**逐组 group-wise weight-only quantization**，而不是逐层单一 scale，也不是逐标量独立量化。组大小被固定为 128，并且 MLX 的 grouped quantization 是沿最后一维打包，这意味着对一个二维线性层权重 `W[out, in]` 而言，scale/bias 是在 `in` 维上每 128 个值共享一次。MLX 源码示例还给出了从 packed tensor shape 反推 `bits` 与 `group_size` 的公式：`bits = (packed_last_dim * 32) // dims`，`group_size = dims // scales.shape[-1]`。citeturn5view0turn25view0turn9search14

这意味着它既不是“逐层一个 scale”，也不是经典 AWQ 那种强调 per-channel activation-aware scaling 的公开文件导出模式；更准确地说，是一种**逐输出行、沿输入维分块的分组量化**。如果写成工程语言，就是：

- **量化对象**：主要是 embeddings、attention projections、MLP projections、LM head。citeturn13view1
- **量化粒度**：按最后一维每 128 个元素一组共享参数。citeturn5view0turn9search14
- **量化值域**：原生目标是 ternary `{−1,0,+1}`。citeturn13view1turn14view0
- **缩放因子**：原生每组一个 FP16 scale；MLX 容器额外放一个 bias。citeturn14view0
- **零点/偏置**：原生 format 不需要独立 zero-point；MLX 容器层用 `bias = -s_g` 充当 affine offset。citeturn14view0turn25view1turn39search0
- **剪枝/稀疏化**：没有独立的 mask 或结构化稀疏元数据；但 ternary 的“0”态天然引入值稀疏。citeturn3view0turn43search6

### 关于 PTQ、校准、微调与误差度量的判断

公开仓库**没有**发布校准集、Hessian、激活统计、训练脚本、QAT 日志或误差优化器，因此不能证明它是某种已知 PTQ/QAT 工具的直接产物。严格说，本仓库只公开了**结果表示**，没有公开**求解器**。但结合 Bonsai 的描述——“从现成模型迁移到 binary/ternary 表示”“不是一堆 heuristics”“在 sub-4-bit 仍尽量留住 reasoning / tool use”——可以合理判断，**它大概率不是简单 RTN，也不像未经再优化的普通 GPTQ/AWQ 输出**。更像是：先把大权重层投影到 ternary 组表示，再通过某种额外优化或蒸馏/重构，尽量缩小 logits、隐藏状态或下游能力上的误差。只是这个优化阶段并未公开。citeturn42view0turn13view1turn32view0

如果在本地做“最大程度接近公开结果的可复现还原”，我建议把误差评价分成三层：

1. **张量层误差**：MSE、MAE、max error、cosine similarity，用来衡量 dequantized 权重对 unpacked 权重的逼近。  
2. **网络层误差**：逐层隐藏状态 MSE / cosine；逐 token logits KL 散度。  
3. **任务层误差**：困惑度、少量标准基准、工具调用 JSON 解析成功率、长上下文稳定性。  

这套分层指标也能覆盖 GPTQ/AWQ/LSQ/HQQ 等方法常见关注点：GPTQ偏重近似二阶残差控制，AWQ偏重激活保护，LSQ偏重训练时可学习 step size，HQQ强调无需校准但在超低 bit 下易掉点。citeturn12search0turn12search1turn12search2turn11search8turn11search14

## 数据结构、文件格式与二进制布局

### 仓库里已经能确认的权重组织

`model.safetensors.index.json` 的 `weight_map` 明确告诉我们：这个仓库不是多 shard safetensors，而是**所有 tensor 都落在单个 `model.safetensors`** 中。索引里的 `metadata.total_size = 8490506720` 是所有 tensor payload 字节数之和。与此同时，LFS 指针页给出的真实文件大小是 `8490785104` 字节，相差 `278384` 字节。根据 safetensors 规范，文件前 8 字节是 little-endian 的 header 长度，随后是一段 UTF-8 JSON header，再之后才是连续的 tensor byte ranges；因此这里可以高概率反推出：**JSON header 长约 `278376` 字节，外加 8 字节长度前缀，总计 278,384 字节。**citeturn3view0turn28view0turn29calculator0turn30view0turn10search2

这件事很重要，因为它意味着即使不下载整文件，本地也可以只读头部就拿到每个 tensor 的：

- `dtype`
- `shape`
- `data_offsets`

而这些信息正是反向解析 packed weight 布局所需的全部骨架。Hugging Face 官方还专门给出了“用 HTTP range request 先取前 8 字节、再取 JSON header”的 metadata parsing 示例。citeturn30view0

### 文件级二进制布局

下表给出当前能够**直接确认或高可信推断**的 safetensors 文件级布局。citeturn28view0turn29calculator0turn30view0turn10search2

| 偏移 | 字节长度 | 数据类型 | 含义 | 证据状态 |
|---|---:|---|---|---|
| `0x00000000` | 8 | `uint64_le` | JSON header 长度 `N` | **高可信推断**：应为 `278376` |
| `0x00000008` | `N` | UTF-8 JSON | safetensors header，包含每个 tensor 的 `dtype/shape/data_offsets` 与可选 `__metadata__` | **高可信推断** |
| `0x00000008 + N` | `8490506720` | 原始 tensor bytes | 所有 tensor payload 连续区域 | **已与 `total_size` 对齐** |
| 文件结尾 | — | — | 应恰好到最后一个 tensor 的 end offset | safetensors 规范要求 |

### tensor 级布局与形状关系

对于主要的量化线性层，本仓库最值得关注的不是单个 offset，而是**三元组组织**。若某层原始权重是 `W[out_features, in_features]`，那么在 MLX quantized safetensors 中，它通常会表现成：

| 张量名 | 预期 dtype | 预期 shape 关系 | 说明 |
|---|---|---|---|
| `...weight` | `uint32` | `[out_features, in_features * bits / 32]` | packed codes，2-bit 时每个 `uint32` 承载 16 个值 |
| `...scales` | `float16` | `[out_features, in_features / group_size]` | 每 128 个权重一个 scale |
| `...biases` | `float16` | `[out_features, in_features / group_size]` | MLX affine 容器所需 bias |

这张表的依据有三层：MLX 官方文档说明 `quantized_matmul` 的量化矩阵 `w` 是“packed in unsigned 32-bit integer”；`dequantize` 的公式是 `w_i = s * \hat{w_i} + β`；MLX 模型代码里又给出了从 packed tensor shape 反推 `bits` 和 `group_size` 的公式。把三者合起来，就能无歧义地恢复 shape 关系。citeturn25view0turn25view1turn9search14

如果进一步按本仓库配置代入 `bits=2, group_size=128`，那么每个 128 权重组的存储可写成：

- **原生 ternary g128**：`128 × ternary code + 1 × FP16 scale`  
- **当前部署 2-bit slot**：`128 × 2-bit slot + 1 × FP16 scale = 32 B + 2 B = 34 B/group`  
- **当前 MLX 落盘**：`128 × 2-bit slot + 1 × FP16 scale + 1 × FP16 bias = 32 B + 2 B + 2 B = 36 B/group`  

所以才会出现：

- 原生理想位宽 `≈ 1.71 bpw`
- 原生部署位宽 `2.125 bpw`
- MLX safetensors 落盘位宽 `2.25 bpw`  

这个计算与模型卡里的三组数字是完全吻合的。citeturn14view0turn13view1

### 关于视觉塔的布局

这里有一个必须诚实说明的未解点。模型卡声称“vision tower 为 HQQ 4-bit”；但索引文件里，`vision_tower.blocks.*.attn.qkv.weight`、`vision_tower.blocks.*.mlp.linear_fc1.weight` 等名字并**没有**像语言模型线性层那样成对出现 `.scales/.biases`，至少从可见索引文本里看不到。再考虑到本仓库没有独立 `mmproj` 文件而是把视觉部分并入 `model.safetensors`，所以更稳妥的结论是：**视觉塔在这个单文件包里确实存在，但它究竟是 HQQ 的某种单 tensor packed 表示、还是普通高精度 tensor、还是其他内部格式，仅靠当前公开页面无法定论；需要本地读取 safetensors header 里的 `dtype` 与 `shape` 才能下最终结论。**citeturn13view1turn4view1turn31view0turn31view3turn44view0

## 关键伪代码与反量化示例

### 本仓库最贴近事实的反量化路径

对这个仓库而言，**最可靠的反量化方式不是自己猜 2-bit codebook，而是直接使用 MLX 的 `dequantize`**，因为仓库本身就是按 MLX grouped affine quantization 容器打包的，`mlx_lm` 也会根据 `config.json` 中的 `quantization` 先创建量化模块，再把 safetensors 中的 packed arrays 更新进去。MLX 示例代码明确显示：加载模型时，如果配置含 `quantization`，就会调用 `nn.quantize(model, **quantization)`，随后再 `model.update(weights)`。citeturn8search9turn8search6turn33view0

下面这段代码不是“官方 Bonsai 训练脚本”，但它是**官方容器兼容的本地反量化方式**：

```python
# 需要 Apple Silicon + MLX
import mlx.core as mx

arrays, metadata = mx.load("model.safetensors", return_metadata=True)

name = "language_model.model.layers.7.self_attn.q_proj"
wq = arrays[f"{name}.weight"]
scales = arrays[f"{name}.scales"]
biases = arrays[f"{name}.biases"]

# 该仓库 config 已给出
BITS = 2
GROUP_SIZE = 128

# 直接由 MLX 反量化回浮点矩阵
w = mx.dequantize(
    wq,
    scales=scales,
    biases=biases,
    bits=BITS,
    group_size=GROUP_SIZE,
    mode="affine",
)

print(w.shape, w.dtype)
```

这段路径的价值在于：它完全尊重仓库当前已发布的 runtime 语义，不需要你先猜测 2-bit codeword 到 ternary 值的内部映射。citeturn25view1turn33view0

### 用本地头信息恢复 packed shape 与原始矩阵维度

如果你想完全脱离 MLX，仅从 safetensors 头信息恢复每个量化张量的几何关系，可以用下面的纯 Python 脚本。它依据 Hugging Face safetensors 头规范和 MLX 源码里公开的 shape 推断公式。citeturn30view0turn9search14

```python
import json
import struct
from pathlib import Path

def load_safetensors_header(path: str):
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
    return header_len, header

def infer_quant_layout(weight_entry, scales_entry, group_size=128, bits=2):
    # packed last dim * 32 / bits = original in_features
    packed_last = weight_entry["shape"][-1]
    in_features = packed_last * 32 // bits

    # scales last dim = number of groups along in_features
    groups = scales_entry["shape"][-1]
    assert in_features == groups * group_size, (
        f"in_features={in_features}, groups={groups}, group_size={group_size}"
    )

    out_features = weight_entry["shape"][0]
    return {
        "out_features": out_features,
        "in_features": in_features,
        "groups_per_row": groups,
        "packed_u32_per_row": packed_last,
    }

header_len, header = load_safetensors_header("model.safetensors")

base = "language_model.model.layers.7.self_attn.q_proj"
info = infer_quant_layout(
    header[f"{base}.weight"],
    header[f"{base}.scales"],
    group_size=128,
    bits=2,
)
print("header_len =", header_len)
print(info)
```

### 一个“格式一致”的 ternary 量化近似重现器

下面这段代码不是官方 Bonsai 求解器，而是**与仓库格式一致、可重复实验、适合本地验证的 surrogate quantizer**。它做的是：沿最后一维按 128 分组，把实值权重近似成 `{−1,0,+1}`，并用最小二乘更新每组 scale。这样的“阈值分配 + 组级 scale 回归”非常适合做 reverse-engineering baseline。若你要对照 unpacked 仓库评估 MSE / cosine / logit KL，它是一个合理起点。citeturn14view0turn43search0

```python
import numpy as np

def ternary_groupwise_quantize(W: np.ndarray, group_size: int = 128,
                               iters: int = 8, thr_ratio: float = 0.5):
    """
    W: [..., in_features], 最后一维做分组
    返回:
      q: 与 W 同 shape, 元素属于 {-1,0,+1}
      s: [..., n_groups, 1], 每组一个 scale
    """
    assert W.shape[-1] % group_size == 0
    G = W.shape[-1] // group_size

    X = W.reshape(*W.shape[:-1], G, group_size).astype(np.float32)

    # scale 初始化
    s = np.maximum(np.mean(np.abs(X), axis=-1, keepdims=True), 1e-8)

    for _ in range(iters):
        thr = thr_ratio * s
        q = np.where(X > thr, 1.0, np.where(X < -thr, -1.0, 0.0))
        denom = np.sum(q * q, axis=-1, keepdims=True)
        denom = np.maximum(denom, 1.0)
        s = np.sum(X * q, axis=-1, keepdims=True) / denom
        s = np.maximum(np.abs(s), 1e-8)

    q = q.astype(np.int8).reshape(W.shape)
    s = s.reshape(*W.shape[:-1], G)
    return q, s

def dequantize_ternary_native(q: np.ndarray, s: np.ndarray, group_size: int = 128):
    G = q.shape[-1] // group_size
    qg = q.reshape(*q.shape[:-1], G, group_size).astype(np.float32)
    sg = s.reshape(*s.shape[:-1], G, 1).astype(np.float32)
    return (qg * sg).reshape(q.shape).astype(np.float32)
```

### 原生 ternary 到 MLX affine 容器的打包思路

根据模型卡，Bonsai 原生表示是“每组只有 scale，没有 bias”；落到 MLX 时，为了适配 grouped affine quantization，会增加一个 bias。公开材料给出了转换关系 `s_mlx = 2·s_g`、`bias = -s_g`，但**没有公开 MLX 内部 2-bit codebook 与 ternary 值的一一映射**。因此，最严谨的做法是：

> **⚠️ 修正注记（2026-07-29）**：`s_mlx = 2·s_g` 对于三元是错误的（见上文 §量化流程还原的修正注记）。正确映射为 `scale = s_g, bias = -s_g`，codes {0,1,2} → {-s_g, 0, +s_g}。此结论已通过穷举验证并在 `affine_2bit.py` 中实现，因此下面的 brute-force codebook 搜索**不再必要**——默认映射 `ternary_012` 已是唯一解。**真实数据复核（2026-07-29）**：已下载真实 packed + unpacked 仓库并执行该 brute-force（`experiment/analyze_bonsai_vs_unpacked.py`），6 个小层的最优映射全部为 `[-1,0,1,0]`（MSE=0），与默认映射 `ternary_012` 完全一致，code 3 确实未被使用。

- 若只想跑推理或还原浮点权重：用 `mx.dequantize`。  
- 若想恢复“原生 ternary codebook”：本地对照 unpacked 权重做映射搜索。  

下面给出一个**验证用**的 brute-force 思路，它可以在本地确定 2-bit codeword 到 `{−1,0,+1,unused}` 的最佳映射，而不依赖猜测。citeturn14view0turn25view1turn32view0

```python
import itertools
import numpy as np

VALUES = [-1.0, 0.0, 1.0, None]  # None 表示 unused / 合并到其他值另行处理

def unpack_u32_to_2bit_codes(packed_u32: np.ndarray, n_values: int):
    flat = packed_u32.astype(np.uint32).reshape(-1)
    out = np.empty(flat.size * 16, dtype=np.uint8)
    for i, x in enumerate(flat):
        for j in range(16):
            out[i * 16 + j] = (x >> (2 * j)) & 0b11
    return out[:n_values]

def search_codebook(codes, dense_group):
    best = None
    for perm in itertools.permutations([0, 1, 2, 3], 4):
        mapped = np.empty_like(codes, dtype=np.float32)
        valid = True
        lut = {}
        values = [-1.0, 0.0, 1.0, np.nan]  # 第四个码先当无效
        for c, v in zip(perm, values):
            lut[c] = v
        mapped[:] = np.vectorize(lut.get)(codes)
        if np.isnan(mapped).any():
            # 简化处理：把 unused code 合并到 0
            mapped = np.nan_to_num(mapped, nan=0.0)
        denom = np.sum(mapped * mapped)
        s = 0.0 if denom == 0 else np.sum(dense_group * mapped) / denom
        recon = s * mapped
        mse = np.mean((dense_group - recon) ** 2)
        cand = (mse, lut, s)
        if best is None or cand[0] < best[0]:
            best = cand
    return best
```

这段搜索代码的意义不在于“官方就是这样做的”，而在于：**它能把当前仓库中一个尚未公开的实现细节转化为可证伪、可本地确认的问题。**

## 验证与复现计划

### 建议的验证路径

如果要把这次 reverse engineering 从“高可信报告”推进到“本地可复核结果”，建议按下面三阶段做。第一阶段只验证文件格式；第二阶段验证数学一致性；第三阶段验证模型行为一致性。这样做能把“格式”“算子”“模型质量”三个层次分开。citeturn30view0turn33view1turn32view0

#### 环境

对 Apple Silicon 上的 MLX 真实反量化与推理，建议采用 Prism 官方 demo 给出的可复现实验组合：Python 3.11、`mlx-lm==0.31.2`。Demo 仓库同时强调：**2-bit ternary pack 跑在 stock MLX 上即可**，而 1-bit pack 才需要 PrismML 的 MLX fork；不过官方验证过的 27B 运行环境里也给出了 `PrismML-Eng/mlx` 的 `prism` 分支与提交 `88c9c20`。如果你只做 header 解析和对照 unpacked 权重，则任意平台只需 Python + `safetensors` / `numpy` 即可。citeturn33view0turn33view1turn24view0

#### 下载命令

```bash
pip install -U "huggingface_hub[cli]" safetensors numpy

# MLX 包
hf download prism-ml/Ternary-Bonsai-27B-mlx-2bit --local-dir ./Ternary-Bonsai-27B-mlx-2bit

# 对照用 unpacked 包
hf download prism-ml/Ternary-Bonsai-27B-unpacked --local-dir ./Ternary-Bonsai-27B-unpacked
```

这些仓库都是公开仓库，无需认证即可下载；而 unpacked 仓库的存在，正好给了我们一个逐 tensor 比对 dequantized 权重的基准。citeturn23view0turn32view0

### 建议测试用例

下表给出我建议的最小验证矩阵。它兼顾本地成本与结论强度。citeturn25view1turn32view0turn13view1

| 测试 | 目标 | 关键指标 | 预期 |
|---|---|---|---|
| Header 解析 | 验证 safetensors 容器与 offset 连续性 | header 长度、tensor 数、offset 连续性 | 头长约 278,376 B，payload 总和对齐 `total_size` |
| 量化层 shape 恢复 | 验证 `bits=2, group_size=128` 与 packed shape | `in_features`、`groups_per_row`、`packed_u32_per_row` | 与 config、index 一致 |
| 单层反量化 | 验证 `mx.dequantize` 能恢复浮点权重 | MSE、cosine、max abs error | 与 unpacked 同名层高度接近 |
| codebook 搜索 | 恢复 native ternary code 映射 | 全组 MSE 最小的 codeword 映射 | 找到稳定映射与可疑 unused code |
| logits 对比 | 验证 packed vs unpacked 网络行为 | 单步 logits KL、top-k overlap | 小且稳定 |
| 短文本生成 | 验证推理兼容性 | 首 token、一致性、温度 0 结果 | 语义接近，贪心时更可比 |
| 长上下文 / KV cache | 验证 runtime 兼容性 | 峰值内存、tok/s、长上下文可用性 | 与模型卡量级接近 |

### 一个可操作的本地验证脚本骨架

```python
# compare_one_tensor.py
import json, struct
import numpy as np
from safetensors import safe_open

PACKED = "./Ternary-Bonsai-27B-mlx-2bit/model.safetensors"
UNPACKED = "./Ternary-Bonsai-27B-unpacked/model.safetensors"
NAME = "language_model.model.layers.7.self_attn.q_proj"

def load_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    return n, h

n, h = load_header(PACKED)
print("header_len:", n)

with safe_open(PACKED, framework="np") as f:
    wq = f.get_tensor(f"{NAME}.weight")
    s  = f.get_tensor(f"{NAME}.scales")
    b  = f.get_tensor(f"{NAME}.biases")

# 如果在 Apple Silicon 上，可切到 mlx.dequantize；
# 若非 Apple 平台，可先只做 shape/offset/codebook 恢复，再与 unpacked 对照。
print("packed weight shape:", wq.shape, wq.dtype)
print("scales shape:", s.shape, s.dtype)
print("biases shape:", b.shape, b.dtype)

with safe_open(UNPACKED, framework="np") as f:
    w_ref = f.get_tensor(f"{NAME}.weight")
print("unpacked shape:", w_ref.shape, w_ref.dtype)
```

### 评估指标建议

如果你的目标是“真正还原量化过程”，我建议至少保留以下四个指标组，而不是只盯着单一的 MSE：

- **权重误差**：MSE、MAE、max abs、cosine。  
- **激活 / 隐状态误差**：逐层 hidden-state cosine、L2。  
- **输出误差**：step-wise logits KL、交叉熵、top-k rank overlap。  
- **行为误差**：困惑度、代码通过率、工具调用结构化成功率、长上下文稳定性。  

这样做的原因在于：超低 bit 量化往往不是“所有指标一起均匀下滑”，而是先在推理、工具调用或长程 reasoning 上出现**性质级崩塌**；模型卡本身也反复强调 conventional sub-4-bit builds 的问题不是单纯精度下降，而是 reasoning / tool use / agentic coherence 的塌缩。citeturn13view1turn42view0

## 风险、限制与改进建议

这次还原最核心的限制有三条。第一，**仓库公开的是结果格式，不是求解脚本**；因此我们能高置信地还原“格式、容器、运行时、张量组织”，却不能把“训练时到底用了哪一种优化器、蒸馏损失、校准集、再训练轮数”写成确定事实。第二，**视觉塔的 HQQ 4-bit 具体落盘形式仍未在线证实**；因为文件树与模型卡都说明视觉塔存在，但仅凭当前索引页看不到语言模型那样的 `scales/biases` 三元组。第三，**MLX 2-bit affine 容器内部的 2-bit codeword 到 ternary 值的精确映射**没有在仓库页面公开。citeturn13view1turn31view0turn31view3turn14view0

基于这些限制，我给出如下判断等级：

| 结论 | 置信度 | 说明 |
|---|---|---|
| 语言模型主表示是 ternary g128 | 很高 | 模型卡直接声明 |
| `group_size=128, bits=2` | 很高 | `config.json` 直接声明 |
| 语言主干以 `weight/scales/biases` 形式存储 | 很高 | `weight_map` 直接可见 |
| “2bit”是 slot width，不是真实信息位宽 | 很高 | 模型卡给出 1.71 / 2.125 / 2.25 三种位宽 |
| 这不是公开标准 GPTQ/AWQ/HQQ 语言权重格式 | 高 | 文件格式与模型卡都不匹配这些标准导出 |
| 视觉塔如何在单文件中表示 HQQ 4-bit | 中低 | 需本地 header 读取确认 |
| 官方求解器是 PTQ、QAT 还是蒸馏重构 | 中低 | 公开材料未披露 |

改进建议也分成两类。一类是**对反向工程者**：优先下载 unpacked 仓库，把每个量化层的 dequantized 结果与 unpacked 同名层逐层比对；并对一个小层做 codeword brute-force 搜索，这能一次性解开“MLX 2-bit slot 到 ternary codebook”的具体映射。 **【✅ 已于 2026-07-29 完成】**该建议已完整执行：下载 `Ternary-Bonsai-27B-mlx-2bit`（packed）+ `Ternary-Bonsai-27B-unpacked`（FP16），498 层逐层比对 MSE 中位数 = 0、cosine = 1.0；6 个小层 brute-force 最优映射全部为 `[-1,0,1,0]`（MSE=0），codebook 映射已一次性解开并坐实。详见 `experiment/analysis_bonsai_vs_unpacked/REPORT.md`。另一类是**如果 PrismML 未来要增强可审计性**：建议公开至少三项内容——一份最小化量化导出脚本、一份 safetensors 元数据样例、以及一个说明“视觉塔在 MLX 单文件中的 HQQ 打包方式”的技术说明。这样就能把当前很多“高可信推断”升级为“直接证实”。citeturn32view0turn30view0turn13view1

综合而言，本报告的最终结论可以压缩成一句话：**`prism-ml/Ternary-Bonsai-27B-mlx-2bit` 的“2bit”不是普通 int2，而是 Bonsai 的 ternary g128 语言权重通过 MLX grouped affine quantization 容器落盘后的结果；主干线性层采用 128 分组、组级 scale、MLX 容器级 bias、packed uint32 权重；runtime 直接消费 packed weights，不先展开回 FP16；训练/量化求解器未公开，但公开证据足以重建其文件格式、反量化路径与主要工程语义。**citeturn13view1turn14view0turn24view0turn25view0turn25view1turn33view0