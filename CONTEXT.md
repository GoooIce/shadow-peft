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
- **非对称量化（Asymmetric Quantization）** — backbone 与影子适配器按不同精度部署的形态：backbone 低精度（PTQ），shadow 保持高精度承载任务专属修正。
- **压缩引擎封装（`shadow_lab.slim`）** — AngelSlim 五压缩器（PTQ+nvfp4 / Distill / QAD / Sparsity / Speculative）的深模块封装层；backbone PTQ 的第二个压缩后端（trainer `backbone_backend="angelslim"`）。取代分两步：第一步并存（torchao tiers 与自家投机解码原型保留），第二步凭 QAD/AngelSpec 对照数据删旧。
- **校准式量化（calibration-based quantization）** — GPTQ/AWQ：裸基座上跑校准前向、按 Hessian/激活统计求解权重格点，区别于 RTN。GPTQ 族的 `dequant_to_bf16` 把 fake-quant 权重直接留在原 Linear（训练数值，挂影子即可蒸馏）；weight-only 档无联合校准问题，静态激活档（fp8_static/w4a8）在裸基座校准、manifest 标 `joint_calibration=False`。
- **通用 Llama 同构包装（GenericLlamaLike）** — 未被 AngelSlim 上游注册的 Llama 同构家族（Gemma/LFM2，投影层名与 Llama 一致）经 model_type 前缀解析后由桥接层惰性注册的通用包装。
- **挽回率（Recovery Rate）** — shadow 对量化损失的挽回比例：(ppl_quant − ppl_quant+shadow) / (ppl_quant − ppl_bf16)。
- **联合校准（Joint Calibration）** — 激活量化 scale 的校准前向挂上 shadow 联跑；若在无注入分布上校准，注入修正会落在校准范围外被截断。
- **原生影子量化（Native Shadow Quantization）** — 影子权重在训练期间即处于部署精度格式（nvfp8 / nvfp4 两级缩放格点）的形态：权重全程在格点上，存训一致（部署即所见），无主权重与 STE。
- **QUASAR 重建（QUASAR Reconstruction）** — 无 shadow 的 NVFP4 量化学生训练：保留 latent 高精度权重，以 AdamW 二阶矩作在线显著性，在每次前向搜索 E2M1 码分配并加权拟合 E4M3 块 scale，再用 STE 更新 latent 权重；它属于 QAD 对照臂，不改变“基座永远冻结”的 shadow 训练语义。
- **裁剪基因组（clip genome）** — 把参与 NVFP4 重建的各层、各分组裁剪阈值按稳定布局展平后的搜索向量；布局负责在向量切片与具名权重分组之间往返，使候选可重放、可哈希比较。
- **演化策略搜索中心（ES search center）** — 演化策略当前搜索分布的中心点；在无约束 latent 空间中接受成对噪声估计的更新，再映射为有界裁剪基因组。配置、中心和 generation 游标共同构成可续跑的搜索状态。
- **奖励校准（reward calibration）** — 在冻结提示词池和评分器上预先确定各质量轴的方向、中心与尺度，把异量纲分数变成可聚合、跨 episode 可比的标准化奖励规则。
- **单轮聚合奖励（episode reward）** — 同一冻结 prompt episode 内，各样本在校准后质量轴上的聚合标量；它是 ES 候选排序和中心非退化判定的唯一优化目标，失败样本会让该轮结果显式失效。
- **跨 prompt 梯度泛化门禁（cross-prompt gradient generalization gate）** — 在多个互不重叠的训练 prompt 批上分别估计搜索方向，以方向一致性和连续未见批次的中心非退化作为放行条件的实验门禁；通过只表示更新有跨批泛化证据，不表示最终质量改善。
- **置信度软门（Confidence Soft Gate）** — 蒸馏中按教师预测熵对 token 软降权的门：自信（低熵）token 权重高，不可靠 token 只降权不丢弃（源自 GRACE）。
- **门控退化判据（Gate Degradation Criterion）** — 软门在自蒸馏设定下的退化监控：批内变异系数 <0.05 或门均值 >0.95 持续 500 步 → verdict 标 `gate_degraded=true`；只标注不回退（负结果时区分「配方无效」与「门控失效」）。
- **解耦知识蒸馏（DKD）** — 把 KL 蒸馏分解为目标类与非目标类两项、分别加权的蒸馏形式。
- **span-CKA** — 关系 CKA 的文本化：仅在续写 span 的 token 子集上逐样本计算 centered CKA 的特征结构传递损失，维度无关无需投影。
- **IB 控制器（Information Bottleneck Controller）** — 以拉格朗日对偶上升动态调节蒸馏损失权重的机制，权衡贴教师（fidelity）与量化预算（capacity）。
- **KLD 轴（KLD Axis）** — 评测轴之一：KLD(基座 bf16 ‖ 量化+影子) 的每 token 平均 KL（nats），与 ppl 轴同窗口同行走（`eval_kld`）；恢复臂改善该轴 ⇔ 相对 quant-only 基线向 0 收敛。
- **留出判定轴（axis2）** — 留出自蒸馏池上的评测轴（同 prompt 源、不交叠 seed、全行流）：RL/instruct 类模型的默认判定轴——wikitext 类外部语料对后训练模型深度 OOD，会扭曲恢复率原点、掩蔽真实信号（exp1-grace rl 遍实测）。
- **残差匹配噪声注入（residual-matched noise injection）** — 机制归因臂的噪声协议：每步对影子权重计算量化格点的舍入残差并在块内随机重排，前向/反向在加噪权重上进行、优化器步作用在干净主权重上——统计上匹配原生量化训练的隐式噪声，但不落格点（`shadow_lab.shadow_quant`）。
- **影子独立生成（shadow-only generation）** — 推理模式 `shadow_only`：仅影子干 + 冻结的基座头副本出词，基座不参与。训练中的影子路 CE（`shadow_loss_weight`，默认 0.05）是其唯一监督来源；该权重即「恢复角色 vs 独立生成角色」的双角色权衡旋钮。

## 分层恢复调度（exp6）

- **贪心逐组层调度（greedy layer-group schedule）** — `train_shadow` 的 `layer_schedule`：`end2end`（现行端到端，逐位不变）之外的新训练形态——按层序把基座适配层切成组，前缀 no_grad 推进、组内带梯度重放，最后以端到端 polish 阶段（`polish_token_frac` 预算）收尾；贪心阶段只用隐态 MSE，KL/CE 只活在 polish。
- **捕获-重放（capture-and-replay）** — `shadow_lab.layerwise` 的层驱动接缝：一次 no_grad 前向用 hooks 记录每个包装层的调用帧（注意力掩码/位置编码原样）与层边界状态 `(基座隐态 h, 影子态 s)`；组训练只重放组内的层。按模型族的掩码数学零重实现，重放与模型自身前向逐位一致（`replay_parity` 预检门，GB10 实测 max_abs_diff=0）。
- **影子链重算（shadow-chain recompute）** — 利用 `ShadowUpdate` 只吃基座隐态的代数（`s' = s + g(h)·(T(h)−s)`），从缓存 h 轨迹带梯度重建前缀影子链，让梯度回传到前缀 update 模块。
- **局部梯度盲区（local gradient blind spot）** — 对齐第 k 层教师隐态的局部损失给不了 `update[k]` 梯度（它在第 k 层之后执行、只喂第 k+1 层的注入）；组末层的 update 永远留给 polish 阶段，`tests/lab/test_layerwise.py` 把该事实钉成受保护断言。
- **注入零初始化（injection ZeroInit）** — `injection_ups` 初始为全零：挂载即恒等映射；首个 optimizer step 之前，注入输出上游的一切梯度（downs、影子态、重算链）精确为零，第一步只训 ups。
- **逐层归因（per-layer attribution）** — 新测量轴：训练前后各拍一次每层归一化隐态 MSE（学生 vs 教师），回答「恢复发生在哪些层」；落在 `TrainReport.layer_attribution` / verdict。

## 推理挽具（harness）

- **专家（Specialist）** — 一份已训练的影子适配器 + 其元数据；在同一基座上热切换。
- **路由（Routing）** — 四级优先级决定选用哪个专家：元数据护栏 → 会话粘滞 → 关键字（优先级破平）→ 默认回退。
- **会话粘滞（Sticky Session）** — 一旦某会话命中某专家，后续轮次继续用它；会话同时持有各自的聊天历史。
- **草稿-验证（Draft-Verify）** — 投机解码的两段循环：小草稿模型先行起草若 token，目标模型批量验证并按拒绝采样保留前缀（lossless 要求两者同词表、同温度）。
- **平均接受长度（τ）** — 每个 draft-verify 周期平均产出的 token 数；硬件无关的可比指标，不含草稿自身开销。
- **再循环（Recirculation）** — 推理时把深层激活按输入步回注浅层的零训练增强（凸混合/纯加性 + L2 范数匹配）；收益强依赖 Gemma 系 Peri-LN 归一化；代价在 prefill 逐 token 串行。
- **注册表（Registry）** — 预载所有专家的适配器状态，`ensure()` 经 `swap_adapter` 热切换：原地换权重并原子重置影子运行态（隐藏态 + 影子 KV 缓存）。时序约束活在模块内部，不在调用方。

## 检查点格式

- 适配器目录：`shadow_config.json` + `shadow_adapter.safetensors`（+ 可选 `shadow_modules.safetensors`），torch 与 MLX 同构，可互转（`convert.py`）。
- 量化基座：`model.safetensors` + `quantization.json`。

## 后端

- **torch 包（`shadow_peft`）** 与 **MLX 包（`shadow_peft_mlx`）** — 同一概念的两个后端实现；共享检查点格式这一数据接缝。
- **核心共享包（shadow_core）** — 纯 Python 包（`src/shadow_core/`，零第三方依赖）：影子配置（`ShadowConfig`）与挽具路由/会话/配置的后端无关家。torch 与 MLX 包重导出同一批类（导入路径不变）；张量数学不进 core，留在各后端。量化方案表（QuantScheme 概念）目前在 torch 侧 `quantization/scheme.py`，将来可随拆分迁入 core。

## 本次架构评审敲定的术语（2026-08-24，已全部落地）

- **实验工作台（experiment lab，包名 `shadow_lab`）** — `src/shadow_lab/` 下的研究代码深模块群：`trainer.train_shadow(cfg) -> TrainReport`（位宽是配置字段）、`eval.py`（评测资产轻量家）、`trainers.py`（唯一的 MMLU/GSM8K/SQuAD 评测循环，GSM8K 训练器带 `answer_extractor` 接缝）、`atoms.py`（设备、分块困惑度、wikitext parquet 加载）、`data.py`（数据集构建）、`tiers.py`（backbone PTQ 档位，蒸馏训练依赖）。实验 campaign 与 CLI 已清理，新实验直接复用这套机器。
- **量化方案（Quantization Scheme）与方案表（scheme table）** — `quantization/scheme.py` 的一等概念：bits/levels/words_per_int/group_multiple/ternary/编解码对。1-bit 与 2-bit 是两个表项；模块类按方案参数化，旧类名为钉死方案的子类（`isinstance` 不变）；加 3-bit = 一行表项。
- **编解码器（codec）** — `unpack_codes(packed, bits)`：移位掩码解包习语的全仓唯一实现家（torch 侧）；模块、测试、分析脚本一律消费它。
- **适配器热切换（adapter swap）** — `ShadowPeftModel.swap_adapter(state_dict)` 公开单方法（torch/MLX 同名同语义），原子完成换权重 + 重置影子运行态；注册表只经它切换专家。
- **加载器（loader）** — `load_quantized_checkpoint(model, dir)`：`save_quantized_checkpoint` 的配对读取方；按清单重建打包结构再覆盖状态字典，save→load 往返位级等价。
- **2-bit 三值（MLX）** — `shadow_peft_mlx.quantize_model_2bit` / `pack_2bit_ternary`：torch 2-bit 方案的 MLX 孪生（原生 bits=2，无需 PR 构建）。
