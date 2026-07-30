# Bonsai 式 1-bit 量化实现报告

## 1. 结论

本项目现已具备一条可测试、可导出的 Bonsai 式二值 1-bit 权重量化路径：

- PyTorch 参考量化与反量化；
- Linear、Embedding 和 tied output head 的整模替换；
- MLX 数组打包；
- 符合真实 Bonsai 1-bit checkpoint 已核验 tensor/layout 契约的 U32/F16 safetensors 导出；
- Gemma 4 Torch-only 转换入口，以及一次 Gemma 4 12B 的真实完整转换；
- 独立标量参考、位序、dtype、manifest、缓存和导出契约测试。

这里的“Bonsai 式”有严格边界：实现复现了公开 checkpoint 的二值表示、分组布局和 MLX 容器契约；权值求解采用基于权重 FP32 视图的确定性 `sign × mean(abs(weight))` 最小二乘代理，再舍入到容器 dtype。Bonsai 官方的训练或迁移求解器并未公开，因此本实现不声称复现其训练算法或最终模型质量。

## 2. 依据的 1-bit 格式契约

实现前对本地缓存的 `prism-ml/Bonsai-1.7B-mlx-1bit` checkpoint（revision `fac480d016cf265838ad087f9f3f2abac4a2c1a5`）进行了逐张量核验。观察到的契约为：

| 项目 | 真实 checkpoint | 本实现 |
|---|---:|---:|
| bits | 1 | 1 |
| group size | 128 | 默认 128 |
| packed dtype | U32 | MLX 导出 U32；PyTorch 容器 I32 |
| packed shape | `[out, in / 32]` | 相同 |
| scale/bias dtype | F16 | MLX 导出强制 F16 |
| scale/bias shape | `[out, in / 128]` | 相同 |
| bit order | 每个 U32 内 LSB-first | 相同 |
| code 0 | `-d` | `-d` |
| code 1 | `+d` | `+d` |
| affine 参数 | `scale = 2d, bias = -d` | 相同 |
| config | `{"bits": 1, "group_size": 128}` | 相同，不写 `mode` |

该 checkpoint 共有 704 个张量：197 个 U32 packed 权重和 507 个 F16 张量。其中 197 个完整量化三元组由 1 个 embedding 加 28 层 × 7 个 attention/MLP projection 组成；剩余 113 个 F16 张量是各层 norm 与 final norm。全部 13,436,752 个 scale/bias 元素均按位满足 `bias == -scale / 2`，scale 全为正且非零。Qwen 的输出打分复用 tied embedding，因此没有独立的 `lm_head.weight`。

原生 Q1_0 每 128 个权重需要 128 bit code 和 16 bit magnitude，共 144 bit，即 1.125 bit/weight。在 MLX 容器中再存一个 16-bit bias，共 160 bit，即 1.25 bit/weight。这里的额外 0.125 bit/weight 是 grouped-affine 容器的冗余 bias 成本。

## 3. 量化算法

### 3.1 分组与二值中心

对二维权重 `W[out, in]` 沿最后一维按 `G` 个元素分组。每组计算：

```text
d = mean(abs(w))
q(w) = +d, if w >= 0
       -d, otherwise
```

对固定的符号分配，`mean(abs(w))` 是最小化组内未加权权重平方误差的闭式解。符号和统计从输入的 FP32 视图计算，最终 scale/bias 再舍入到 FP16 或 BF16，因此这个闭式结论描述的是舍入前的局部权重目标。它不涉及网络 loss、activation-aware 目标、校准或蒸馏。零值采用非负约定，编码为正中心；因此全零组的 bit 全为 1，但 `d = 0` 且不引入 epsilon，仍精确重建为零。极小的非零组可能因最终 FP16 存储下溢为零。

`refine_iters` 为兼容现有接口而保留；对这一对称二值闭式解，符号分配和中心更新已经处在最小二乘不动点，因此额外迭代不会改变结果，默认值为 0。

### 3.2 位打包

每 32 个二值 code 打包到一个 32-bit word：

```text
packed = sum(bit[i] << i), i = 0..31
```

即第一个权重写入最低有效位。PyTorch 没有通用 U32 tensor 运算路径，因此内存和 PyTorch checkpoint 使用位模式相同的 I32；导出 safetensors 时按位重解释为 U32，不进行数值转换。

### 3.3 MLX affine 映射

MLX grouped-affine 反量化形式为：

```text
dequantized = code * scale + bias
```

当 `code ∈ {0, 1}` 时，写入：

```text
scale = 2d
bias  = -scale / 2
```

因此 code 0、1 分别精确映射到 `-d`、`+d`。bias 从最终存储 dtype 的 scale 推导，保证 F16/BF16 舍入后仍满足 checkpoint 中的精确关系。若 scale 在目标 dtype 中溢出或出现非有限值，量化器会直接报错。

## 4. PyTorch 实现

核心实现在 `src/shadow_peft/quantization/affine_1bit.py`。

### 4.1 公共 API

```python
from shadow_peft.quantization import quantize_1bit_bonsai

packed, scales, biases = quantize_1bit_bonsai(
    weight,
    group_size=128,
    refine_iters=0,
)
```

输入要求：

- 权重必须为二维；
- Torch reference 的 `group_size >= 32` 且为 32 的倍数；
- 输入维度必须同时能被 group size 和 32 整除；
- `refine_iters >= 0`。

组统计和 sign bit 从输入的 FP32 视图计算；FP16/BF16 输入保留相应 scale dtype，其他输入将 scale/bias 存为 FP16。`quantize_1bit_sign` 保留为兼容别名，调用相同的规范实现。

### 4.2 整模转换

```python
from shadow_peft.quantization import quantize_model_1bit

manifest = quantize_model_1bit(
    model,
    group_size=128,
    method="bonsai",
)
```

显式选择 `method="bonsai"` 后：

- Linear 替换为 `QuantizedLinear1BitSign`；
- Embedding 替换为 `QuantizedEmbedding1BitSign`；
- tied output head 默认也单独量化，并记录到 `manifest["quantized_tied"]`；manifest 的 `tied_to` 同时记录共享来源。这是为普通 Hugging Face Torch forward 保留完整量化语义的独立副本，不再共享 storage；
- 模块原有 train/eval 状态被保留；
- `padding_idx`、bias、dtype、scaled-embedding 的输出缩放和可选 vocabulary trim 被保留或验证；
- 前向反量化权重会匹配 activation dtype，支持 FP32 输入；
- eval 反量化缓存会在 `load_state_dict`、设备/dtype 迁移和重新进入训练态时失效。

为避免破坏既有调用：

- 未传 `method` 和 `variant` 时仍走旧的 affine 1-bit 路径；
- `variant="sign"` 仍可用，并保持旧的 tied-head 默认行为；
- 新功能的规范入口是 `method="bonsai"`。

Embedding 前向只反量化当前 input 中实际出现的 unique rows，而不是展开完整词表。对 Gemma 4 的 `Gemma4UnifiedTextScaledWordEmbedding`，量化模块还保留 `sqrt(hidden_size)` 输出缩放；这是模型语义的一部分，不能只复制权重而忽略。

### 4.3 Torch checkpoint 重载

转换目录不是 Transformers 原生量化格式，不能直接交给 `AutoModelForCausalLM.from_pretrained()`。项目提供显式加载器：

```python
from shadow_peft import load_quantized_checkpoint_1bit

model = load_quantized_checkpoint_1bit("./gemma-4-12b-it-torch-1bit")
```

加载器读取 `config.json`、`quantization.json` 和 `model.safetensors`，在 meta device 创建原始 architecture skeleton，再按 manifest 直接安装空的量化 Linear/Embedding，最后以 `strict=True, assign=True` 装入 packed state。该过程不会重新创建或量化 12B dense 权重，并保留 BF16 scale/bias；独立保存的 embedding 与 `lm_head` 不会被重新 tie。

Gemma 4 的 embedding scale 与四个非持久 RoPE frequency buffer 不在 state dict 中。加载器分别从 skeleton 的标量元数据和 Gemma 自身的 RoPE 初始化函数重建它们；完成后若仍有 meta buffer，会明确拒绝返回半初始化模型。

## 5. MLX 实现与 checkpoint 导出

### 5.1 原生数组打包

`src/shadow_peft_mlx/quantization.py` 新增：

```python
from shadow_peft_mlx import pack_1bit_bonsai

packed, scales, biases = pack_1bit_bonsai(weight, group_size=128)
```

该路径直接产生 U32 packed 权重及 F16 scale/bias。`quantize_model_1bit(..., method="bonsai")` 会在 MLX 模型中采用同一映射，并在开始时探测 runtime 是否真的支持 `bits=1`。该后端目前没有 PyTorch 路径同等的非有限值显式报错，因此不承诺极限 FP32 输入在两个后端逐 bit 一致。

### 5.2 safetensors 导出

```python
from shadow_peft_mlx import save_1bit_mlx_checkpoint

info = save_1bit_mlx_checkpoint(
    model,
    config,
    manifest,
    "./model-mlx-1bit",
    group_size=128,
)
```

导出器会：

- 将 I32 packed 位模式写成 safetensors U32；
- 将所有浮点张量写成 F16；
- 写入顶层 `quantization = {"bits": 1, "group_size": 128}`；
- 若配置含 `text_config`，作为嵌套模型工程扩展同步写入 quantization；
- 生成单 shard 或 Hugging Face 风格多 shard index；
- 交叉校验 manifest 的 bits、group size 与导出参数；
- 要求 manifest 非空且完整覆盖 state_dict 中全部量化模块，拒绝漏项、重复项、不完整或 shape 错误的三元组；
- 拒绝非 sign/Bonsai manifest，以及不满足 F16 `bias == -scale / 2` 的非对称状态；
- 根据 `tied_to` 和模块语义优先保留 token embedding、移除 output head，不依赖注册顺序；无法判断 canonical 一侧时拒绝自动去重，可显式设置 `deduplicate_tied_weights=False`；
- 仅在 manifest 记录了实际裁剪且 state_dict 行数一致时，同步顶层或嵌套 `vocab_size`；
- 保留源 config 对象不变。

1-bit 并非上游标准 MLX runtime 的通用加载能力。导出的文件格式与观测到的 Bonsai checkpoint 一致，但实际推理仍需要支持 `bits=1` 的 MLX/MLX-LM 实现，例如 Bonsai checkpoint 所使用的相应 fork。Torch reference 可实验任意 32 的倍数组大小；可执行 MLX affine 容器只允许 runtime 支持的 32、64、128，而 Bonsai 发布合同仅为 128。

## 6. Gemma 4 Torch-only 转换与实跑

新增脚本。当前只生成 Torch 产物的调用方式为：

```bash
.venv/bin/python experiment/convert_gemma4_to_1bit.py \
  --torch-only \
  --model /Users/devel0per/.cache/modelscope/models/google--gemma-4-12B-it/snapshots/master \
  --out-torch ./gemma-4-12b-it-torch-1bit
```

`--torch-only` 会在导入 MLX 导出器之前关闭该分支，不创建 MLX 目录。Torch 路径执行以下步骤：

1. 先用 `AutoConfig` 验证 `gemma4_unified` 架构是否已注册；
2. 以 BF16/CPU 从 safetensors 加载模型；
3. 跳过视觉、音频和多模态组件；
4. 以 `method="bonsai"`、group size 128 量化语言权重；
5. 保存 I32 packed weight 与 BF16 scale/bias 的 PyTorch 容器；
6. 把量化配置写入 `config.json` 的 `shadow_peft_quantization`，并复制本地 tokenizer/processor 辅助文件；
7. 读取 safetensors header，确认首个 packed weight 的 dtype 为 I32。

项目依赖现已限定为 `transformers>=5.10.1,<6`；本次解析锁定并实际使用 5.12.1。该版本能够识别 snapshot 中的 `gemma4_unified`，普通 CPU 加载不需要安装 Accelerate。

### 6.1 真实转换结果

2026-07-30 在本地完整执行了上述命令，进程退出码为 0：

| 项目 | 实测结果 |
|---|---:|
| 输入权重 | 677 个 BF16 tensor，23,919,549,408 bytes（22.277 GiB） |
| 量化模块 | 330 |
| 独立量化 tied head | 1（`lm_head`） |
| 跳过模块 | 3（视觉/音频 embedding projection 与 vision patch dense） |
| `model.safetensors` | 2,124,161,616 bytes（1.978 GiB） |
| 相对源权重文件 | 11.26× 更小，字节数减少 91.12% |
| 辅助文件 | 6 个 tokenizer/processor/config 文件 |
| wall time | 49.26 秒 |
| maximum resident set size | 37,150,785,536 bytes（34.60 GiB） |

Torch 输出位于 `gemma-4-12b-it-torch-1bit/`；本次没有创建 `gemma-4-12b-it-mlx-1bit/`。模型文件 SHA-256 为：

```text
f6a88b985e0ed97d357201007f08506606d6ab2e68dae78dcba5140e01480e32
```

### 6.2 产物级验证

真实产物另做了独立于转换脚本首张量自检的全量检查：

- 1,338 个 tensor，严格为 330 个 I32 packed weight 和 1,008 个 BF16 tensor；
- 每个量化模块都恰有 `.weight/.scales/.biases`，shape 分别满足 `[out,in/32]` 与 `[out,in/128]`；
- 全部 100,884,480 个 scale/bias 元素有限、scale 非负，并精确满足 `bias == -scale/2`；
- 对 330 个模块各抽取原始权重首尾行，共 660 行重新量化，packed bits、scale 和 bias 全部逐 tensor 相等；
- 从产物装载第 0 层 `q_proj` 到 `QuantizedLinear1BitSign` 后，`[1,2,3840]` FP32 输入得到 `[1,2,4096]` FP32 有限输出，重复前向与 eval cache 一致。

### 6.3 真实重载与整模 forward

随后直接从生成目录调用 `load_quantized_checkpoint_1bit`，实际重建出 `Gemma4UnifiedForConditionalGeneration`：

- 329 个 `QuantizedLinear1BitSign` 和 1 个 `QuantizedEmbedding1BitSign`；
- meta parameter 和 meta buffer 均为 0；
- token embedding 输出为 BF16，保留 `sqrt(3840) ≈ 61.9677` 的 Gemma 4 缩放；
- 四个 sliding/full-attention RoPE buffer 均为 CPU FP32；
- embedding 与 `lm_head` 的 packed storage 相互独立；
- 本地 `AutoConfig` 和 `AutoTokenizer` 可仅从输出目录读取。

用 tokenizer 对 `"Hello"` 取单个 token（id 9259），执行完整文本 forward：

```python
with torch.inference_mode():
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        logits_to_keep=1,
        return_dict=True,
    )
```

实测 logits shape 为 `[1,1,262144]`、dtype 为 BF16、全部有限，且 `past_key_values is None`。forward 用时 9.104 秒，整个进程 wall time 13.63 秒，maximum resident set size 为 32,379,240,448 bytes（30.16 GiB），无 swap。

这些检查证明了转换产物的结构完整性、与源 BF16 权重的抽样逐位一致性，以及从磁盘重建后完整 Torch 文本 forward 的可执行性；有限 logits 或一次 top-k 结果不等同于生成质量证明。

## 7. 验证结果

执行：

```bash
.venv/bin/python -m pytest -q tests
```

结果：

```text
153 passed, 2 skipped
```

覆盖重点包括：

- 与独立标量参考逐 bit、逐 scale/bias 对比；
- LSB-first 固定位型；
- FP16、BF16、FP32 输入与全零组；
- 非法 group size、维度和 refine 参数；
- scale 存储溢出；
- Linear/Embedding/tied-head 整模替换；
- meta skeleton checkpoint 重建、strict/assign 加载与错误路径；
- Gemma scaled embedding 语义、unique-row 解码与 RoPE buffer 恢复；
- legacy 模型调用与既有 manifest 字段兼容；
- train/eval 与 state-dict 缓存失效；
- vocabulary trim 与 `padding_idx`；
- safetensors U32/F16 dtype、shape、bit parity 和 config 契约；
- manifest 与导出参数不一致时拒绝写出。

同时通过 Ruff、Python bytecode 编译和 `git diff --check`。当前环境没有安装 `mlx`、`mlx_lm` 或 `mlx_vlm`，因此 `tests/mlx` 中新增的原生 MLX 测试已经编写，但未在本机执行。

## 8. 已知边界

1. 本实现是格式兼容、数学确定的二值量化代理，不是未公开的 Bonsai 官方求解器。
2. PyTorch Linear 前向目前通过参考反量化再调用 dense 算子，eval 会缓存反量化后的 dense 权重；Embedding 已只解码输入中的 unique rows，但它仍不是 packed 1-bit GEMM 性能内核，运行期内存和速度不能按 1.25 bit/weight 推断。
3. MLX 1-bit 推理依赖支持该位宽的 runtime fork；标准环境不能仅凭 checkpoint 文件获得加载能力。
4. Gemma 4 12B 的 Torch-only 转换已经真实完成，但仍未进行整模文本生成质量、困惑度或任务指标评估。
5. 导出器不会清理输出目录中的历史 shard；生产转换应使用新的空目录。
6. 直接传远程 model id 且不先下载为本地 snapshot 时，转换器只能对缺失的 tokenizer/processor 文件发出警告；完整交付应使用本地 snapshot 路径或另行保存这些辅助文件。

这些边界不影响当前实现作为格式生成器、研究参考路径和后续 packed kernel/质量实验的基础。
