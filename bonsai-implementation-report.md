# Bonsai 式三值 2-bit 量化实现报告

> 实现日期：2026-07-30
> 实现状态：核心量化、模型替换、双格式导出、非 MLX 回归测试与合成导出 smoke test 已完成
> 研究依据：[deep-research-report.md](deep-research-report.md)

## 1. 摘要

本次工作在 shadow-peft 中新增了一条 Bonsai 式三值 2-bit 量化路径。实现覆盖以下能力：

- 将二维浮点权重沿最后一维按组切分，投影到三值集合 {-1, 0, +1}；
- 使用组内绝对值均值初始化 scale，并以阈值分配和最小二乘更新构造确定性代理求解器；
- 按 2-bit slot 将三值码打包为 int32/uint32；
- 以 scale = s、bias = -s 的 grouped affine 形式生成 MLX 兼容张量；
- 原地替换 PyTorch 模型中的 Linear 与 Embedding；
- 导出 PyTorch safetensors 和 MLX safetensors；
- 覆盖数值边界、打包布局和模型兼容性的回归测试，并完成导出格式的合成 smoke test。

实现复现的是研究报告确认的 Bonsai ternary g128 表示、二进制布局和一个公开可复现的代理量化方法。Prism ML 没有公开其迁移、训练、蒸馏或误差优化求解器，因此本实现不声称复现官方 Bonsai solver。

## 2. 实现范围

### 2.1 已完成

| 范围 | 实现结果 |
| --- | --- |
| 权重表示 | 三值 {-1, 0, +1}，默认 group size 128 |
| 代理求解 | 绝对值均值初始化、严格阈值分配、组级最小二乘 scale 更新 |
| 二进制打包 | 每个 32-bit word 打包 16 个 2-bit code，LSB-first |
| MLX 映射 | code 0/1/2 对应 -s/0/+s，scale = s，bias = -s |
| PyTorch 模块 | QuantizedLinear2Bit、QuantizedEmbedding2Bit |
| 模型转换 | Linear/Embedding 原地替换、跳过规则、词表裁剪、共享权重处理 |
| 检查点 | PyTorch I32 容器、MLX U32 容器、FP16 scales/biases、分片索引 |
| Gemma 4 转换 | 双格式转换脚本和 MLX 键名适配 |
| 验证 | 单元测试、回归测试、静态检查和导出 smoke test |

### 2.2 不在本次范围内

- Prism ML 未公开的正式 Bonsai 训练或迁移算法；
- 基于校准集、Hessian、激活统计或蒸馏的精度恢复；
- PyTorch 侧直接消费 packed weight 的融合矩阵乘内核；
- Gemma 4 视觉塔的 HQQ 4-bit 量化；
- 对完整大模型重新执行困惑度、生成质量或下游任务评测；
- 重新生成工作区中已有的大型 Gemma 4 量化产物。

## 3. 从研究结论到工程约束

研究报告能够直接确认的是结果表示，而不是权重求解过程。本实现将信息分为“格式硬约束”和“代理算法选择”两层。

### 3.1 格式硬约束

| 项目 | 工程约束 |
| --- | --- |
| 量化集合 | {-1, 0, +1} |
| 默认分组 | 沿权重最后一维每 128 个元素一组 |
| slot 宽度 | 每个元素占用一个 2-bit slot |
| group scale | 每组一个 scale，MLX 落盘为 FP16 |
| affine bias | 每组一个 bias，固定为 -scale |
| code 映射 | 0 → -1，1 → 0，2 → +1 |
| code 3 | 编码器永不产生；若外部数据包含 code 3，通用 affine 公式会解码为 +2s |
| packed dtype | PyTorch 保存为 I32 bit pattern；MLX 保存为同一 bit pattern 的 U32 |
| packed 顺序 | 每 16 个 code 组成一个 word，第 0 个 code 位于最低 2 bit |

研究报告中的真实 packed/unpacked 对照覆盖 498 个量化三元组，支持上述容器布局、code 0/1/2 映射和 scale/bias 关系。该证据只证明结果表示兼容，不证明本实现的 dense-to-ternary 代理求解器等同于官方求解器。

对于原始权重 W 的形状 [out_features, in_features]，三个量化张量的形状为：

| 张量 | 形状 | PyTorch 容器 | MLX 容器 |
| --- | --- | --- | --- |
| weight | [out_features, in_features / 16] | int32 | uint32 |
| scales | [out_features, in_features / group_size] | FP16 或 BF16 | FP16 |
| biases | [out_features, in_features / group_size] | FP16 或 BF16 | FP16 |

以 group size 128 计算，一个 MLX 量化组包含 32 字节 code、2 字节 scale 和 2 字节 bias，共 36 字节，即量化三元组本身约 2.25 bit/weight。仅含 scale 的 2-bit-slot 部署表示为 34 字节，即 2.125 bit/weight；这不同于采用理想 ternary 编码时约 1.71 bit/weight 的理论表示。MLX 中额外的 bias 是 grouped affine 容器兼容成本。

### 3.2 代理求解器选择

由于官方求解器未公开，实现采用研究报告给出的可复现 baseline：

1. 使用组内 mean(abs(w)) 初始化 scale；
2. 在正负 thr_ratio × scale 处执行严格阈值分配；
3. 固定三值 assignment，以最小二乘闭式解更新 scale；
4. 默认重复 8 次 assignment → scale update。

这一路径是确定性的，不依赖校准数据、随机采样或训练过程。

## 4. 核心算法

实现入口为 [quantize_2bit_bonsai](src/shadow_peft/quantization/affine_2bit.py)。

### 4.1 输入约束

- 输入必须是二维权重 [out, in]；
- group_size 必须大于等于 16 且为 16 的倍数；
- in 必须能被 group_size 整除；
- refine_iters 必须大于等于 0；
- thr_ratio 必须位于 (0, 1]；
- FP16/BF16 输入保留其 scale 存储 dtype，其他输入先转换为 FP16；
- 分组计算和最小二乘更新在 FP32 中完成。

允许非 128 的 group size 是为了小模型测试和实验，但只有默认值 128 对应报告中的 Bonsai g128 格式。

当前实现没有显式拒绝包含 NaN 或 Inf 的输入权重，因此调用方应保证源权重为有限值。

### 4.2 分组与 scale 初始化

将权重重排为：

    X.shape = [out_features, in_features / group_size, group_size]

每组初始 scale 为：

    s_0 = max(mean(abs(X)), epsilon)

epsilon 用于避免内部求解阶段出现零尺度。

### 4.3 严格阈值分配

给定当前 scale s，阈值为 t = thr_ratio × s。每个元素的三值 assignment 为：

    q_i = -1,  当 x_i < -t
    q_i =  0,  当 -t <= x_i <= t
    q_i = +1,  当 x_i > +t

实现刻意使用严格小于和严格大于。恰好落在正负阈值上的值归入 0，与研究报告的伪代码一致。

### 4.4 最小二乘 scale 更新

固定 q 后，每组 scale 使用下式更新：

    s = abs(sum(x_i × q_i) / max(sum(q_i²), 1))

默认 refine_iters = 8。N 次迭代的语义是恰好执行 N 次“assignment → least-squares update”。最后返回的 scale 与最后一次使用的 assignment 对应，不会在最后一次 scale 更新后再产生一个未拟合的新 assignment。

当 refine_iters = 0 时，仍执行一次基于初始 scale 的三值分配，但不执行最小二乘重估，便于构造无 refinement 的对照基线。

### 4.5 code 映射与打包

三值通过 c = q + 1 映射到 2-bit code：

| q | code |
| ---: | ---: |
| -1 | 0 |
| 0 | 1 |
| +1 | 2 |

第 j 个 32-bit word 的打包公式为：

    packed_j = sum(code_(16j+k) << (2k)), k = 0..15

因此 code 以 LSB-first 顺序存储。PyTorch 中将最终 32 位 bit pattern 包装为 int32；导出 MLX 时只做零拷贝 dtype reinterpret，不改变任何 bit。

### 4.6 MLX affine 映射

MLX grouped affine 反量化使用：

    w_hat = scale × code + bias

令 scale = s、bias = -s，可得：

| code | 反量化值 |
| ---: | ---: |
| 0 | -s |
| 1 | 0 |
| 2 | +s |

这正好表达 Bonsai 三值集合。code 3 不由本量化器产生；通用公式下它对应 +2s。

反量化器是内部参考路径，不会额外校验 shape、dtype、group size 或 code 范围，调用方必须提供彼此一致的 packed/scales/biases 三元组。

### 4.7 零组和极小 scale

全零组会得到全 code 1。由于 FP16 中内部 epsilon 可能舍入为 0，实现只替换“转换到目标 dtype 后确实为 0”的 scale，替换值为该 dtype 的最小 normal 正数。bias 同时设为其负值，因此：

    scale × 1 + bias = 0

可继续精确重建全零组。

对于仍能由 FP16 表示的 subnormal scale，实现保留原值，不会无条件抬升到最小 normal。

## 5. 软件结构

### 5.1 数据流

~~~mermaid
flowchart LR
    A[浮点 Linear / Embedding] --> B[最后一维分组]
    B --> C[三值分配与 LS scale 更新]
    C --> D[2-bit LSB-first 打包]
    D --> E[QuantizedLinear2Bit / QuantizedEmbedding2Bit]
    E --> F[PyTorch I32 checkpoint]
    E --> G[MLX U32 + FP16 scale/bias checkpoint]
~~~

### 5.2 文件职责

| 文件 | 职责 |
| --- | --- |
| [src/shadow_peft/quantization/affine_2bit.py](src/shadow_peft/quantization/affine_2bit.py) | Bonsai 量化器、反量化器、量化模块和模型级转换 |
| [src/shadow_peft/quantization/__init__.py](src/shadow_peft/quantization/__init__.py) | 量化 API 导出 |
| [src/shadow_peft/__init__.py](src/shadow_peft/__init__.py) | 顶层公开 API 导出 |
| [src/shadow_peft_mlx/export_2bit.py](src/shadow_peft_mlx/export_2bit.py) | MLX dtype 转换、键名适配、分片与配置写入 |
| [experiment/convert_gemma4_to_2bit.py](experiment/convert_gemma4_to_2bit.py) | Gemma 4 双格式端到端转换 |
| [tests/test_bonsai_quantization.py](tests/test_bonsai_quantization.py) | 算法、边界条件和模型替换测试 |
| [tests/mlx/test_export_2bit_mlx.py](tests/mlx/test_export_2bit_mlx.py) | MLX 容器格式测试定义；当前环境未执行 |

## 6. PyTorch 模型级集成

模型入口为 [quantize_model_2bit](src/shadow_peft/quantization/affine_2bit.py)，它会原地扫描并替换 Linear 与 Embedding。

### 6.1 调用方式

~~~python
from shadow_peft.quantization import quantize_model_2bit

manifest = quantize_model_2bit(
    model,
    method="bonsai",
)
~~~

Bonsai 模式默认解析为：

| 参数 | 默认值 |
| --- | ---: |
| group_size | 128 |
| refine_iters | 8 |
| thr_ratio | 0.5 |
| quantize_tied_weights | True |

为了保持兼容，quantize_model_2bit 的全局默认 method 仍是 affine；只有显式传入 method="bonsai" 才启用本次实现。旧 affine 路径继续使用 refine_iters=0 和默认跳过 tied head 的行为。

QuantizedLinear2Bit.from_linear 和 QuantizedEmbedding2Bit.from_embedding 若被单独调用且未提供 quant_fn，也仍走旧量化器，其默认 thr_ratio 为 0.7；它们本身不是 Bonsai 的推荐入口。

### 6.2 替换规则

- skip_patterns 以大小写敏感的模块全名 substring 匹配；命中模块保持原样，并在 tied/dimension 判断前记录到 manifest.skipped；
- 最后一维不能被 group_size 整除的层保持原样，并记录跳过原因；
- trim_vocab_to 可在打包前裁掉 Embedding 和 lm_head 的无效尾部词表行；
- 当模型使用对象式 config 且已有 vocab_size 大于裁剪值时，config.vocab_size 会同步更新；
- 替换后的模块保持原模块的 train/eval 状态；
- Bonsai 模式默认把 tied lm_head 单独量化，并记录在 manifest.quantized_tied；
- 也可通过 quantize_tied_weights 显式覆盖 tied weight 策略。

模型级参数会在扫描层之前完成校验，因此即使模型中没有可量化层，非法参数也会立即报错。

tied head 被量化后会拥有独立的 packed/scales/biases buffer。由于量化器是确定性的，它可与 embedding 得到相同张量值，但不再保持源模型中的对象级 Parameter 共享关系。

### 6.3 QuantizedLinear2Bit

QuantizedLinear2Bit 将 packed weight、scales 和 biases 注册为无梯度 buffer，并将可选输出 bias 保存为冻结参数。

当前 PyTorch forward 是参考实现：

1. 将整层 packed weight 反量化；
2. 将计算视图转换为输入 activation dtype；
3. 调用常规 F.linear。

eval 模式会缓存反量化权重；以下操作会使缓存失效：

- 切换回 train 模式；
- 调用 to/cuda 等触发 module._apply 的操作；
- 通过父模型或模块自身执行 load_state_dict。

因此当前实现验证了格式和模型行为，但没有获得 packed kernel 应有的运行时内存与吞吐收益。

### 6.4 QuantizedEmbedding2Bit

QuantizedEmbedding2Bit 同样保存 packed weight、scales 和 biases。当前 forward 会先反量化整个 embedding table，再执行 gather。这是正确性优先的参考路径；后续可通过“先按 token 定位组，再局部解码”或原生 packed embedding kernel 降低峰值内存。

普通替换路径会保留 padding_idx，但 max_norm、norm_type、scale_grad_by_freq 和 sparse 等特殊 Embedding 行为没有完整复刻；词表裁剪时新建的 Embedding 也不会继承 padding_idx。因此它目前面向常规语言模型 embedding，不是覆盖 nn.Embedding 全部配置的通用替换。

### 6.5 manifest

模型转换返回 JSON 可序列化 manifest，记录：

- mode、bits、ternary、method；
- group_size、refine_iters、thr_ratio；
- quantize_tied_weights、trim_vocab_to；
- quantized、skipped、tied、quantized_tied、trimmed 模块集合。

manifest 同时作为导出器识别量化张量的依据，并写入 quantization.json。

## 7. 检查点导出

### 7.1 PyTorch 容器

PyTorch 导出通过 save_quantized_checkpoint 完成：

- packed weight 保持 int32；
- scale/bias 保持模型中的浮点 dtype；
- 输出 model.safetensors；
- 输出 quantization.json。

该容器服务于本仓库自己的反量化和分析路径，不宣称可由通用 Transformers 自动识别为标准 int2 模型。

### 7.2 MLX 容器

[save_2bit_mlx_checkpoint](src/shadow_peft_mlx/export_2bit.py) 执行以下转换：

- 将量化 weight 从 int32 按位 reinterpret 为 uint32；
- 将量化层 scales/biases 统一导出为 FP16，即使源模型是 BF16；
- 保留未量化张量的原始 dtype；
- 根据目标大小写入一个或多个 safetensors shard；
- 多分片时生成 model.safetensors.index.json；
- 在 config.json 顶层，以及 text_config 存在时，在其中写入相同的量化配置；
- 写入 quantization.json；
- 支持 exclude_patterns 排除目标运行时不存在的张量；
- 对 Gemma 4 HF 与 mlx-vlm 的模块层级差异执行键名重映射。

默认分片上限为 5 GiB，采用按 tensor 的贪心分片；只有输出包含多个 shard 时才生成 index 文件。导出器会交叉校验 manifest 中的 bits、group_size 与导出参数，发现不一致时直接拒绝写出。输出目录不会清理历史 shard，因此生产转换应使用空目录或预先确认目录内容。使用 exclude_patterns 后，quantized_count 表示实际导出的 packed tensor 数，可能小于原 manifest.quantized 的长度，而 quantization.json 仍保留原 manifest。

写入 config.json 的配置为：

~~~json
{
  "quantization": {
    "bits": 2,
    "group_size": 128,
    "mode": "affine"
  }
}
~~~

这里的 mode="affine" 描述的是 MLX 容器和运行时接口；权重内容仍是 Bonsai 式三值 code。

### 7.3 tied lm_head 的 MLX 处理

Gemma 4 的目标 MLX 实现通过 embed_tokens.as_linear 完成输出打分，不需要独立 lm_head 参数。端到端转换脚本因此在 MLX 导出时使用：

~~~python
exclude_patterns=("lm_head.",)
~~~

PyTorch 容器仍可保留单独量化的 tied head，二者的差异由目标运行时结构决定。

## 8. Gemma 4 转换流程

转换入口：

~~~bash
python experiment/convert_gemma4_to_2bit.py \
  --model /path/to/gemma-4-12B-it \
  --out-mlx ./gemma-4-12b-it-mlx-2bit \
  --out-torch ./gemma-4-12b-it-torch-2bit
~~~

脚本默认 method=bonsai、group size 128、refine iterations 8，并执行：

1. 以 BF16 在 CPU 加载源模型；
2. 跳过视觉、音频、多模态投影等非语言主干模块；
3. 量化语言主干的 Linear 与 Embedding；
4. 写出 PyTorch I32 容器；
5. 写出 MLX U32 容器；
6. 复制 tokenizer、processor 和 generation 配置；
7. 读取 safetensors header，自检首个量化层的 I32/U32 dtype。

当前工作区中的 gemma-4-12b-it-mlx-2bit 和 gemma-4-12b-it-torch-2bit 大型目录没有在本次实现完成后重新生成。现有产物的 manifest 缺少新字段，MLX text_config.quantization 缺失，部分 scale/bias 仍为 BF16，且 lm_head 仍被保留，可确认它们来自修正前的转换路径，不能作为本实现结果的交付物；需要在干净输出目录中重新运行上述脚本。

## 9. 验证

### 9.1 算法测试

[tests/test_bonsai_quantization.py](tests/test_bonsai_quantization.py) 使用独立、标量化的报告参考实现进行对照，覆盖：

- 默认参数下 packed weight、scale 和 bias 逐元素完全一致；
- LSB-first 打包和 code 范围；
- 正负阈值相等边界归零；
- 全零组 scale 有限、为正且精确重建零；
- 可表示的 FP16 subnormal scale 不被错误抬升；
- 默认 thr_ratio=0.5 的固定随机回归样本中，0 到 8 次最小二乘 refinement 的 MSE 不增加；
- Bonsai 模式默认参数解析；
- 旧 affine 默认行为保持不变；
- FP32 模型和 FP32 activation 可正常前向；
- load_state_dict 后 eval cache 正确失效；
- tied embedding/lm_head 量化和 eval 状态保持；
- 无可量化层时仍执行入口参数校验；
- 非法 solver 参数正确拒绝。

### 9.2 导出测试

[tests/mlx/test_export_2bit_mlx.py](tests/mlx/test_export_2bit_mlx.py) 覆盖：

该测试文件已经编写，但由于当前环境缺少 MLX 相关可选依赖，本次没有收集或执行；以下项目通过代码审查和独立的合成 exporter smoke test 进行了补充验证。

- config.json 顶层和 text_config 量化配置；
- MLX packed weight 为 U32；
- PyTorch packed weight 为 I32；
- BF16 源模型导出的 scales/biases 为 F16；
- PyTorch 与 MLX packed weight bit pattern 完全一致；
- 未量化层完整保留；
- 量化层数量与 manifest 一致；
- tied lm_head 可按目标运行时要求排除。

### 9.3 本地验证结果

执行结果：

~~~text
.venv/bin/python -m pytest -q tests --ignore=tests/mlx
121 passed, 1 skipped

.venv/bin/ruff check src tests --exclude tests/mlx
All checks passed!

.venv/bin/ruff check tests/mlx/test_export_2bit_mlx.py experiment/convert_gemma4_to_2bit.py
All checks passed!

git diff --check
通过
~~~

另外执行了两个手工 smoke test：

- FP32 小型 tied-weight Llama 在 Bonsai 量化后完成有限值 FP32 前向；
- BF16 合成模型完成 MLX 导出，确认 U32 packed、F16 scale/bias、双层 config、lm_head 排除以及源 config 不被修改。

当前环境未安装 mlx 和 mlx_lm，因此没有运行原生 MLX runtime 加载测试。这是尚未覆盖的环境验证项，不是已发现的格式错误。

## 10. 兼容性与设计决策

### 10.1 保留旧默认行为

模型级 API 没有把默认 method 从 affine 改成 bonsai，避免现有调用在升级后静默改变数值结果。新功能通过显式 method="bonsai" 启用；Gemma 4 新转换脚本则默认选择 Bonsai。

### 10.2 区分容器模式与求解方法

manifest 同时记录：

- mode="affine"：描述 scale × code + bias 的容器反量化形式；
- method="bonsai"：描述本次选择的三值代理量化方法。

这样可以避免把 MLX affine 容器误解成普通的四电平 int2 量化。

### 10.3 dtype 策略

- 量化计算使用 FP32；
- FP16/BF16 输入的 scale 在 PyTorch 模块中跟随源权重 dtype；
- FP32 及其他输入落入 FP16 量化容器；
- PyTorch Linear forward 将反量化权重转回 activation dtype；
- MLX 落盘始终将量化 scale/bias 规范化为 FP16。

## 11. 已知限制与后续建议

### 11.1 已知限制

1. **不是官方 solver**：无法复现未公开的迁移、再训练、蒸馏与能力恢复过程。
2. **缺少融合 kernel**：PyTorch Linear 和 Embedding 当前都会反量化完整权重，运行时内存收益有限。
3. **缺少端到端质量评测**：尚未报告完整模型的 perplexity、logit KL、生成质量和任务准确率。
4. **MLX 原生加载待验证**：当前环境缺少 mlx、mlx_lm 和加载脚本所需的 mlx_vlm，只有容器级 smoke test。
5. **大模型产物待重建**：现有 Gemma 4 输出目录不是新实现的可信产物。
6. **范围限于语言主干**：Gemma 4 转换脚本保留视觉、音频和多模态投影为原精度。
7. **模型转换为原地操作**：量化后源模块会被替换；如需回退或对照，应保留原检查点或重新加载模型。
8. **输出目录需要干净**：MLX 导出器会自动核验 manifest 的 bits 与 group size，但不会清理旧 shard。
9. **Embedding 兼容面有限**：非常规 Embedding 选项和裁剪后的 padding_idx 尚未完整保留。

### 11.2 建议的下一阶段

建议按以下顺序推进：

1. 安装匹配版本的 mlx/mlx_lm/mlx_vlm，完成真实 MLX load 和首 token 生成；
2. 重新转换 Gemma 4，并对输出 shard、config、tensor shape 和首层反量化进行验收；
3. 在代表性层上报告 FP32/BF16 原权重与代理量化权重的 MSE、cosine 和最大误差；
4. 在完整模型上比较 perplexity、logit KL、短生成和多模态任务；
5. 实现直接消费 packed U32/I32 的矩阵乘与 embedding kernel；
6. 若需要逼近官方 Bonsai 能力，再引入校准、蒸馏或可学习 scale/weight 重构，而不是把当前 baseline 误标为官方算法。

## 12. 结论

本次实现已经完成从研究结论到可运行工程路径的闭环：

- 已复现 Bonsai ternary g128 的核心张量表示；
- 已实现确定性的 dense-to-ternary 代理量化器；
- 已实现与真实容器一致的 code 映射、scale/bias 关系和 packed 布局；
- 已接入 PyTorch 模型替换与双格式检查点导出；
- 已通过非 MLX 回归测试、静态检查和容器级验证。

因此，该实现适合作为 Bonsai 格式研究、权重量化实验和导出验证的可靠基线。它当前不应被描述为 Prism ML 官方求解器的复现，也不应在缺少端到端模型评测的情况下被描述为达到官方模型质量。
