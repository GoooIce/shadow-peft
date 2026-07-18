# 1-bit 逐位/逐层对比分析：朴素 PTQ（ours） vs Bonsai-1.7B

数据来源：`experiment/analyze_1bit_layers.py`，明细见 `per_layer_metrics.csv`（197 个量化模块 × 19 项指标）、`bit_position_density.csv`、`overview.png`。

对比对象：

- **ours**：Qwen/Qwen3-1.7B fp16 经 `mx.quantize(group_size=128, bits=1)` 朴素 min/max 打包
- **bonsai**：prism-ml/Bonsai-1.7B-mlx-1bit（Qwen3-1.7B 架构上训练出的 1-bit 模型）
- 参照：Qwen3-1.7B fp16 原始权重

## 0. 结构层面：完全相同

| 项 | ours | bonsai |
|---|---|---|
| 量化模块数 | 197（embed + 28 层 × 7 proj） | 197（同） |
| 格式 | uint32 LSB-first，gs=128，fp16 scale/bias | 同 |
| norms / tied lm_head | fp16 / tied | 同 |
| vocab | 151936 | **151669**（裁掉 267 行） |

结论：差异不在结构，全部在 bit/scale/bias 的**取值**上。

## 1. 逐位（bitwise）：统计上不可区分

| 指标 | ours | bonsai |
|---|---|---|
| bit 密度（1 的比例） | 0.4996–0.5021 | 0.4978–0.5002 |
| 逐 bit 位密度（0..31 位） | 平坦 0.498–0.503 | 平坦 0.498–0.503 |
| 全 0 / 全 1 退化 group | 0 | 0 |
| 零 scale group | 0 | 0 |

两个模型的 bit 流在统计上都无法与随机 50% 区分——**Bonsai 的优势不在 bit 模式里**（没有稀疏性、没有位面结构、没有退化 group 压缩技巧）。信息差全在每组 (scale, bias) 的放置和权重值本身。

## 2. 逐层（layerwise）：三个本质差异

### 2.1 Bonsai 的量化网格窄得多（最大差异）

每组 scale（两档电平的间距）：

| 模块 | ours scale_mean | bonsai scale_mean | 比值 |
|---|---:|---:|---:|
| embed_tokens | 0.1767 | 0.0544 | **3.25×** |
| q_proj | 0.1812 | 0.0825 | 2.20× |
| k_proj | 0.1650 | 0.0847 | 1.95× |
| o_proj | 0.1614 | 0.1100 | 1.47× |
| gate_proj | 0.1943 | 0.1471 | 1.32× |
| down_proj | 0.1842 | 0.1110 | 1.66× |
| （全体中位数） | | | **1.49×** |

min/max 打包让 scale 覆盖每组 128 个权重的完整极值范围，离群值把网格撑宽；Bonsai 学到的网格贴着分布主体，**主动裁掉离群值**。bonsai 的 scale_std 也更小（0.010–0.027 vs 0.033–0.069），网格跨 group 更均匀。

### 2.2 我们的反量化权重方差被系统性放大 ~2.7×

| 模块 | fp16 std | ours deq std | bonsai deq std |
|---|---:|---:|---:|
| embed | 0.0345 | 0.0881（2.6×） | 0.0276（0.8×） |
| q_proj | 0.0355 | 0.0938（2.6×） | 0.0427（1.2×） |
| gate_proj | 0.0370 | 0.0968（2.6×） | 0.0747（2.0×） |

min/max 二值化把每个权重推到组内极值，权重分布被人为拉宽——这是 PPL 爆炸的直接机制之一。Bonsai 的 deq 权重方差贴着 fp16 水平。

### 2.3 与 fp16 的距离：我们是噪声，Bonsai 是训练漂移

| 指标 | ours | bonsai |
|---|---|---|
| rel_l2 to fp16 | 1.88–2.05（各层平坦） | 0.60–2.30（逐层分化） |
| cos to fp16 | 0.75–0.78 | 0.44–0.80 |

- ours：误差范数 ≈ 2× 信号范数（SNR ≈ −6 dB），但方向仍与 fp16 相关（cos≈0.76）——纯噪声破坏。
- bonsai：q_proj 最保守（rel_l2 1.03–1.07），gate/up_proj 漂得最远（2.29–2.30，比我们噪声还大），down_proj 随深度上升（1.10→1.38）。说明 Bonsai 是**重训过的**（尤其 MLP），不是"量化得更好的 Qwen"。
- embedding 对比最说明问题：我们 rel_l2 1.88，Bonsai 只有 0.60（cos 0.80）——训练刻意保住了 embedding，印证了"1-bit embedding 是重灾区"。

## 3. 对 shadow 恢复方案的直接指导

1. **P4（交替最小二乘优化 scale/bias）优先级最高**：最大可量化差距在 scale 网格（宽 1.5–3.25×）和方差膨胀（2.6×）。逐 group 对 (s,b) 做最小二乘裁剪拟合即可在训练前白捡一大块精度，格式不变、推理零成本。
2. **embedding 是第一修复对象**：它是我们误差最大、Bonsai 最保守的层。混合精度 manifest 里 embedding 保 4-bit 应该是第一个对照实验。
3. **shadow 的角色定位**：Bonsai 证明终点不是靠更好的量化几何达到的，而是靠训练移动权重。shadow 在 activation 层面做同类事情——评估恢复效果时应以"PPL 从 10¹⁵ 回到两位数"为第一阶段目标，以 bonsai 的 28.7 为长期参照而非承诺。
4. **bit 层面无需对标**：密度、位面、退化 group 两边无差异，不必在这些维度上做文章。

## 5. 深挖：Bonsai 的底层结构是符号量化（±d），不是 affine

线索来自 PrismML-Eng/llama.cpp 的 `prism` 分支（Q1_0 已上游合入 ggml-org/llama.cpp master）：

```c
#define QK1_0 128
typedef struct { ggml_half d; uint8_t qs[QK1_0 / 8]; } block_q1_0;  // 只有 delta，没有 bias
// dequant: y = bit ? d : -d
// quantize ref: d = mean(|x|), bit = sign(x)
```

**GGUF Q1_0 是 BitNet 式符号量化**：每 128 权重一个 mean-abs 缩放，1 bit 只存符号，1.125 bpw（与 README 宣称一致）。

**MLX 版是同一结构的 affine 编码**：对 Bonsai MLX checkpoint 抽查，`bias == -scale/2` 在 fp16 精度内精确成立（残差为 0）——即两个电平关于 0 对称（±d），affine 格式只是 ±d 的一种编码（bias=−d, scale=2d）。这解释了第 2 节观察到的"Bonsai scale 更小、deq 方差贴 fp16"：它的电平天然以 0 为中心，而 min/max affine 被离群值拉偏。

**含义**：

1. Bonsai 不是"更好的 affine PTQ"，而是**训练进符号结构的 QAT 模型**（STE 符号量化，BitNet 路线），PTX 层面的几何差异（对称 vs min/max）是训练方法的指纹。
2. MLX affine 格式是 ±d 的超集，我们的打包器可以直接产出符号结构（令 `bias=-scale/2`）以与 GGUF Q1_0 完全对齐——如果需要和 llama.cpp 生态互通的话。
3. 纯 PTQ 角度，符号 ±mean|w| 与 Lloyd-affine 的 MSE 相当（对离群值都稳健）；Bonsai 的优势来自训练而非格式。

**另一个线索：KV-cache mean-centering**（commit `afc74b75`/`a5527fc8`，对应 Bonsai-demo 的 `KV-CACHE.md` 与 `scripts/make_kv_bias.sh`）：推理时用自建校准语料对 Q4_0 K-cache 做逐通道均值置中，可与 K rotation 组合——这是他们在推理侧的"免费精度"手段，思路可借鉴到我们后续的 1-bit 模型 serving。

## 6. 三份白皮书的确认与缺口（2026-07-18 精读）

**直接确认我们逆向结论的原文**（1-bit 白皮书 p.7）：MLX 版用 `s_mlx = 2·s_g, b_mlx = −s_g` 编码对称 ±d，bias 不带新信息，故 MLX 1.25 bpw vs GGUF 1.125 bpw；g128、端到端量化含 embedding/LM head、仅 norms+scale 元数据高精度——与我们测量完全一致。

**训练方法被刻意隐藏**：三份都只说"proprietary Caltech intellectual property"，无 STE/损失/数据/成本任何细节。仅确认：从现成预训练 Qwen3 出发做"表示变换"（post-training，非从头预训练），架构不改。

**评测缺口**：全文无 PPL 数字（理由：sub-4-bit 崩溃是"质变非渐变"，短基准会掩盖）；无 BitNet 等 1-bit 基线对比。

**关键数字**：

| 模型 | 配置 | 平均 | fp16 | 保持率 |
|---|---|---:|---:|---:|
| 8B（10 基准） | 1-bit | 59.86 | 71.02 | 84% |
| 4B | 1-bit | 55.39 | 68.31 | 81% |
| 1.7B | 1-bit | 40.88 | 58.24 | 70%（**低于 Qwen3-0.6B fp16 的 43.34**） |
| 27B（15 基准） | 1-bit / ternary | 76.11 / 80.49 | 85.07 | 89.5% / 94.6% |
| 27B 参照 | IQ2_XXS（2.8 bpw） | 72.73 | 85.07 | 85.5% |

规律：规模越小 1-bit 损失越大；Bonsai 1-bit（1.125 bpw）在 27B 上反超 IQ2_XXS（2.8 bpw）4 个百分点；低比特模型对 4-bit KV cache 的容忍度比 fp16 高 12–95×（前向 KL）。

ternary 家族：{−1,0,+1} g128，1.71 bpw，部署走 2-bit kernel；8B 保 95%+。

## 7. 文献调研（2026-07-18，arXiv）

**Bonsai 方法溯源**：源自 Caltech 电机工程教授 **Babak Hassibi**（PrismML CEO/创始人，1993 年 Optimal Brain Surgeon 发明人，二阶压缩三十年谱系）。公开可查的仅两篇理论论文：arXiv:2402.10474（ℓ∞ 正则化在强正则极限下直接产生 one-bit 权重解）、arXiv:2510.16250（Random Features 模型 1-bit 渐近零泛化损失）。LLM 级工程配方无公开出处；The Register 2026-04-04 报道直接点名 Hassibi。

**1-bit 训练配方全景**：

| 方法 | arXiv | 路线 | 关键数字 |
|---|---|---|---|
| BitNet b1.58 2B4T | 2504.12285 | from-scratch 三值，4T token | 对齐同尺寸 fp16 |
| FBI-LLM | 2407.07093 | from-scratch 纯二值 + 自回归蒸馏，108B token | 7B Wiki2 9.1（fp16 5.5） |
| OneBit | 2402.11295 | sign + rank-1 值向量，SVID 初始化（NMF>SVD），CE+逐层 hidden MSE 蒸馏 | 13B W1 Wiki2 9.18（fp16 5.09） |
| BinaryMoS | 2406.12311 | **冻结 sign 矩阵**，只训 scaling experts+router | 超过全部 W2A16 PTQ 基线 |
| BiSCo-LLM | 2607.08643（2026-07，极新） | 二值 codec + **LoRA 旁路** + recovery 蒸馏 | 与我们设定几乎同构，待复现 |
| EfficientQAT | 2407.11062 | Block-AP + 端到端只训量化参数 | 2-bit 70B 单卡 41h |

**共性**：logits 蒸馏是压倒性主流（FBI-LLM 消融：软标签一致优于 one-hot CE）；post-hoc 恢复档数据量仅需 ~1-3 亿 token；embedding/LM head/LayerNorm 必须保高精度（与我们的实验发现冲突——我们在 tied 1.7B 上观察到 fp16 head 反而更差，这是个值得深挖的矛盾点）；ParetoQ（2502.02631）发现 ≤2-bit 表示空间会剧变——恢复必须允许表示重构，而非"还原"fp16 权重。

**量化+适配器先验（任务C）**：LoftQ/ApiQ/QERA 等权重级低秩补偿最低只覆盖 2-bit，且 2-bit 下 GSM8K 仍腰斩（LoftQ 25.4 vs fp16 43.1）；SketchTune（2410.06364）证明量化误差**不是低秩的**（非低秩适配器 GSM8K 比 LoftQ 高 14.5%）；ApiQ（2402.05147）以"逐层保持激活精度、阻断误差传播"为核心在 2-bit 全面领先——**直接支持 shadow 的逐层注入设计**。LLM-QAT 发现 4-bit 时 hidden 蒸馏有害，OneBit 发现 1-bit 时有益——位宽依赖性，值得做消融。

**我们的定位**：严格意义上"纯 1-bit 冻结基座 + 小容量 activation 级旁路 + 蒸馏"在文献中是空白；最近的是 BinaryMoS（权重级 scaling）和 BiSCo-LLM（LoRA 旁路）。差异化成立，但无现成强度基准。

**可直接借鉴的配方**：OneBit 损失 `CE(teacher logits) + α·逐层 hidden MSE（L2 归一化后）`；BitDistiller（2402.10631）"初始化决定天花板"（我们 refine_iters 已做）+ CAKLD 置信度加权双向 KL + 同架构自蒸馏优于更大 teacher；LLM-QAT 自生成数据协议（首 3-5 token 贪心 + 后续采样，~100k 条）；小 lr（8e-6~2e-5）、无 weight decay、Adam β2=0.98。

## 4. P4 实施结果（Lloyd 精炼打包，已实现）

`pack_1bit_refined`（MLX）/ `quantize_1bit_affine(refine_iters=)`（torch）：从 min/max 初始出发，逐 group 交替执行"两电平最小二乘拟合 → 中点重分配"，MSE 单调下降，输出格式不变。

**权重空间（高斯权重）**：iters=0 与 `mx.quantize` 逐位一致；iters=1 即 −89.8% MSE，iters≥5 收敛于 −90.1%。

**功能空间（Qwen3-1.7B，wikitext-2，20480 tokens）**：

| 配置 | PPL |
|---|---:|
| fp16 | 19.64 |
| naive PTQ（iters=0） | 2.9×10¹⁵ |
| refined（iters=10，gs=128） | 3.4×10⁵ |
| refined gs=64 | 4.4×10⁵ |
| refined gs=32 | 1.3×10⁵ |
| embed 保 fp16（naive / refined） | 3.4×10⁶ / 3.7×10⁶ |
| Bonsai（训练态） | 28.71 |

**结论（修正第 3 节的预测）**：

1. 精炼打包在权重空间严格更优、应作为默认（`refine_iters=10`），但功能层面所有 PTQ 配置都仍是天文数字 PPL——**免费精度的天花板就到这里**，剩余差距只能靠训练（shadow + 蒸馏）弥补。这正好给出干净的实验归因：基线彻底坏掉，任何恢复都来自 shadow。
2. **第 3 节的"embedding 保高精度"预测被实验推翻**：在 tied lm_head 下，embedding 保 fp16 反而更差（3.4×10⁶ vs 全量化的 3.4×10⁵）——精确的输出头会在已被破坏的 hidden state 上给出更自信的错误 logits。正确配方是**全部量化 + 精炼**。
3. group_size 在 32/64/128 间的差异（1.3×10⁵ ~ 4.4×10⁵）在这个量级上属于噪声，维持 gs=128（与 Bonsai 格式兼容、额外参数最少）。
