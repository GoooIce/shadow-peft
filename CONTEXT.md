# CONTEXT.md — ShadowPEFT 领域术语表（中文）

> 本文件是领域词汇的权威定义。架构评审、代码评审、新模块命名时以此为准；引入新概念时同步更新本表。

## 核心概念

- **影子模型（Shadow Model）** — 小型、可训练、可独立预训练和拆卸的网络，与冻结的基座模型并行运行，向每个解码层注入学习到的修正。
- **基座模型（Base Model）** — 冻结的大模型主干，永远不被训练。
- **影子注入（ShadowInjection）** — 把「基座隐藏态与影子隐藏态之差」经瓶颈投影回基座的模块；由 `injection_hidden_size`、`alpha` 控制。
- **影子更新（ShadowUpdate）** — 门控更新模块，随基座逐层前向演化影子隐藏态。
- **影子适配器（Shadow Adapter）** — 检查点中独立保存的可训练部分：影子模型 + 注入 + 更新（`shadow_adapter.safetensors`）；不含基座权重。
- **隐藏投影（Hidden Projection）** — 影子隐藏尺寸 → 基座隐藏尺寸的投影层。跨尺寸搭配时由 `ProjectedCausalLM` / `AutoModelForCausalLMWithHiddenProjection` 承载，可用伪逆初始化。
- **推理模式（Inference Mode）** — `base_shadow`（基座+影子双路）与 `shadow_only`（仅影子路径，轻量推理）。
- **显式 / 隐式影子模型（explicit / implicit shadow model）** — 显式：外部传入的预训练影子模型；隐式：按基座架构自动构建的小影子。

## 训练与量化

- **蒸馏（Distillation）** — 教师模型（基座，全精度）向学生（影子路径，量化基座之上）的知识迁移：KL（logits）+ 隐藏态 MSE。
- **教师数据（Teacher Data）** — 自蒸馏协议生成的 JSONL 前缀/续写数据（LLM-QAT 风格，贪心前缀 + EOS 截断）。
- **量化方案（Quantization Scheme）** — 一组打包/编码规则：位数、层级数、每整数字打包字数、分组大小约束、编码/解码。现有：1-bit 仿射、2-bit 三值。
- **打包权重（Packed Weights）** — 按方案压缩进整数字的权重张量；训练中必须保持位级不变（由 STE/缓存反量化保证）。
- **Lloyd 精炼（Lloyd Refinement）** — 打包后在码本上迭代 Lloyd–Max，降低量化 MSE。
- **量化清单（Quantization Manifest）** — `quantize_model_*` 返回的字典：`quantized` / `skipped` / `tied` / `trimmed` / `quantization`。
- **词表裁剪（Vocab Trimming）** — 把 `lm_head` 与嵌入裁剪到分词器实际长度（`trim_vocab_to`），需整词表处理绑定权重。

## 推理挽具（harness）

- **专家（Specialist）** — 一份已训练的影子适配器 + 其元数据；在同一基座上热切换。
- **路由（Routing）** — 四级优先级决定选用哪个专家：元数据护栏 → 会话粘滞 → 关键字（优先级破平）→ 默认回退。
- **会话粘滞（Sticky Session）** — 一旦某会话命中某专家，后续轮次继续用它；会话同时持有各自的聊天历史。
- **注册表（Registry）** — 预载所有专家的适配器状态，`ensure()` 经 `swap_adapter` 热切换：原地换权重并原子重置影子运行态（隐藏态 + 影子 KV 缓存）。时序约束活在模块内部，不在调用方。

## 检查点格式

- 适配器目录：`shadow_config.json` + `shadow_adapter.safetensors`（+ 可选 `shadow_modules.safetensors`），torch 与 MLX 同构，可互转（`convert.py`）。
- 量化基座：`model.safetensors` + `quantization.json`。

## 后端

- **torch 包（`shadow_peft`）** 与 **MLX 包（`shadow_peft_mlx`）** — 同一概念的两个后端实现；共享检查点格式这一数据接缝。
- **核心共享包（shadow_core）** — 纯 Python 包（`src/shadow_core/`，零第三方依赖）：影子配置（`ShadowConfig`）与挽具路由/会话/配置的后端无关家。torch 与 MLX 包重导出同一批类（导入路径不变）；张量数学不进 core，留在各后端。量化方案表（QuantScheme 概念）目前在 torch 侧 `quantization/scheme.py`，将来可随拆分迁入 core。

## 本次架构评审敲定的术语（2026-08-24，已全部落地）

- **实验工作台（experiment lab）** — `experiment/lab/` 下的研究代码深模块群：`trainer.train_shadow(cfg) -> TrainReport`（位宽是配置字段）、`eval.py`（评测资产轻量家）、`trainers.py`（唯一的 MMLU/GSM8K/SQuAD 评测循环，GSM8K 训练器带 `answer_extractor` 接缝）、`atoms.py`（设备、分块困惑度、wikitext parquet 加载）、`data.py`（数据集构建）。CLI 脚本只是薄壳。
- **量化方案（Quantization Scheme）与方案表（scheme table）** — `quantization/scheme.py` 的一等概念：bits/levels/words_per_int/group_multiple/ternary/编解码对。1-bit 与 2-bit 是两个表项；模块类按方案参数化，旧类名为钉死方案的子类（`isinstance` 不变）；加 3-bit = 一行表项。
- **编解码器（codec）** — `unpack_codes(packed, bits)`：移位掩码解包习语的全仓唯一实现家（torch 侧）；模块、测试、分析脚本一律消费它。
- **适配器热切换（adapter swap）** — `ShadowPeftModel.swap_adapter(state_dict)` 公开单方法（torch/MLX 同名同语义），原子完成换权重 + 重置影子运行态；注册表只经它切换专家。
- **加载器（loader）** — `load_quantized_checkpoint(model, dir)`：`save_quantized_checkpoint` 的配对读取方；按清单重建打包结构再覆盖状态字典，save→load 往返位级等价。
- **2-bit 三值（MLX）** — `shadow_peft_mlx.quantize_model_2bit` / `pack_2bit_ternary`：torch 2-bit 方案的 MLX 孪生（原生 bits=2，无需 PR 构建）。
