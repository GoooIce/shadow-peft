# EXP009 候选清单（未立项，仅为挂起项备忘）

来源：EXP008（mage-qad-exp008-20260905）正式训练期间的监控与离线分析讨论。EXP008 按冻结协议跑完，本轮对运行中的 campaign 不做任何改动；以下全部推迟到下一个 campaign 立项时评估。注意：任何涉及 `scripts/run_mage_flow_healing.py` 或 `src/shadow_lab/*.py` 的改动都会变更代码身份哈希，只能在新 campaign 中生效。

## 已确认记入（2026-09-07 用户拍板）

### 0. 主线：路线 A —— 从 healed master 重做 ES clip refinement（2026-09-08 用户拍板）

EXP008 的 QAD 半程成立（`best-673` 过 `checkpoint_val` 与 `human_dev` 29/30），ES sigma 阶段在
candidate-00 烧门。EXP009 主线定为在 `best-673` healed master 上以新冻结协议重做 ES；LLaDA-Image
缝合（第 6 项）不在本 campaign 内，其缝合诊断可离流并行。

归因（`exp008-es-sigma-attribution-20260908.md`）给出三项必须进入新协议的变更：

- **门**：探索候选只保留灾难底线，教师轨迹改批均值容差带；非退化判定只施加在更新后的中心上，
  形态沿用 EXP-007 的"未见批平均增量 > 0 且 ≥ 6/8 批不退化"。逐样本零容差与批均值零容差
  对镜像候选在结构上不可通过，与人口大小无关。
- **σ=0.003**：EXP-005 同维度校准值、EXP-007 测到信号时的值；σ=0.03 是 `ESConfig` 默认值，禁用。
- **人口按 EXP-007 量级**：N=80 一批、多批互不重叠 prompt、共同 antithetic noise；以批梯度两两
  cosine 与符号一致率为预注册信号指标（旧 master 基准：cosine 均值 0.31、96% 为正）。维护者
  判断：EXP-006/007 的"负"是欠功效而非无梯度，扩散模型的损失结构不应套用自回归经验；数据
  支持这一判断——N=8 单批测不到、N=80×24 批测到了。healed master 靠近 BF16，余量小、梯度可能
  更弱，需预设 k 批后的停止阈值。

立项时的决策项：N（80 或更大）、批数 k 与停止阈值、候选是否保留教师轨迹损失评估、是否设
低维重参数化对照臂、候选级精确重放是否降为每批一例（成本以周计 vs 以天计）。人工视觉门不可省。
第 1 项检查点双写消除随本 campaign 一并生效。

### 1. best/latest 检查点双写消除

`src/shadow_lab/mage_qad_campaign.py` 保存块（EXP008 时约 336-353 行）：当保存事件是"刷新最佳"时，代码先写 `best-{step}.pt`，随后把内容完全相同的 `latest.pt` 再整写一遍。实测（2026-09-07）：best-561 01:03 → latest 01:07，各约 5 分钟，43.6 GB × 2。

问题：
- 保存窗口翻倍（约 10 分钟），而保存期正是历史上 kernel OOM 的高风险窗口（序列化时内存冲 61.8 GiB 峰值）；
- 两次写之间 latest 与 best 短暂内容一致，纯磁盘冗余。

改法方向：两者内容一致时用 `os.link` 硬链接替代第二次写入（注意 `atomic_save` 的 tempfile+rename 原子语义要保持：先写临时文件、再 link/rename 到两个名字）；或仅当内容与最近一次 best 不同才写 latest。约束：不得破坏"latest 永远是最新完整可恢复状态"的恢复语义，不得放宽身份校验。

## 讨论中候选（未拍板，仅备忘）

### 2. 扩大 QAD 样本集

EXP008 为 1024 条 qad_train / 1313 总步数 ≈ 1.28 epoch，末段约 289 步重复见数据。若 EXP009 扩数据，需走全新冻结协议与数据构建（分层、排除、评估池隔离）；身份模型下不存在"中途换数据续训"。

### 3. （低优先）LR 温和退火

2026-09-07 跨机 AUS 实测（gooo 流式计算检查点对 1↔11、11↔545）：中位 AUS/步约 2.3e-6 rad，比 LLM 预训练 ELR 区间低 4-5 个数量级，权重范数稳态（中位比值 1.000000）。"恒定 lr=1e-5 偏热"缺乏证据，退火预期收益很小；仅当 EXP009 出现不同的验证曲线形态再考虑。运动集中在 block11.to_add_out / proj_out 等少数热层，若有动作优先考虑按层的机制而非全局 LR。

### 4. AUS 定期测量的供给方式

EXP008 期间验证的零成本方案：外挂硬链接快照（盯 campaign 目录，新检查点落盘即 hardlink，保留最近 N 份）+ 另一台 GB10 流式计算（float64 点积、逐层 mmap，峰值内存 <1.5 GB）。若 EXP009 想要更密的 AUS 采样点，优先走"保留更多检查点"（如 ring buffer 硬链接）而不是缩短保存间隔——每次保存本身就是 OOM 风险窗口。

### 5. 双机大文件传输走光纤

双机手册已更新（docs/gb10-machines.md）：大流量互联优先 SFP+ 光纤直连；EXP008 期间实测 87 GB×2 走 2.5 GbE 铜口约 280 MB/s。

### 6. 统一理解+生成：LLaDA-Image 模式缝合 Mage-VL × Mage-Flow（2026-09-07 建票；缝合诊断通过后再拍板）

背景（当晚离流实验依据）：2026-09-07 三臂 A/B/C 条件编码器实验证明**直接替换是死路**——Qwen3-VL 基线 7/10、纯文本 Qwen3-4B 2/10（尺度统计几乎重合但方向语义崩）、VL 训练的 Mage-VL 0/10（空白纹理场，尺度也漂移 ~18%）。EXP008 的 MMDiT 绑定 Qwen3-VL 文本塔的特异表示；任何编码器替换只剩"带适配器的桥接"这一种形态，本票即该路线。

**推荐路线：LLaDA-Image 模式**（[[entities/2609.03796-llada-image]]，2026-09-06 入库）——这就是「理解模型 + 生成模型 → 统一模型」已被验证的配方：

- **冻结 Mage-VL 当理解端**：它本来 Stage 5 就是主干全冻结、只训门控，冻结友好；
- **RQA（残差查询适配器，一组可学习的查询 token，从冻结模型里"问"出信息）+ 浅层 Transformer connector**，把 Mage-VL 的隐状态投影到 Mage-Flow MMDiT 的条件空间——LLaDA-Image 用这套在冻结 dLLM 和从零 DiT 之间架桥，明文解决了「冻结理解模型不直接暴露生成所需细粒度视觉先验」的问题；
- **注入通道**：Mage-Flow 是双流 MMDiT（文本流 + 图像流），Mage-VL 的条件注入走文本流即可；编辑场景下 Mage-ViT 的 codec 稀疏特征可以当参考图通路直注 MMDiT——恰好对应 LLaDA-Image「参考图绕过 VLM 直注 DiT」的设计；
- **训练配方参照**：LLaDA-Image 的 T2I:I2I = 1:1 混合 + Mage-Flow 自己的 4:1 编辑:生成混合流，防理解 / 生成互相遗忘。

**动手前的前置门——缝合诊断（先测再投钱）**：拿线性缝合层测 Mage-VL 各层隐状态 → Mage-Flow 条件空间的「缝合惩罚」（两个表征空间翻译过来会损失多少信息，[[entities/2106.07682-model-stitching]]），惩罚最小的层就是桥该架的位置。[[concepts/platonic-representation-hypothesis]]（好模型的表征趋同假说）给这条路理论许可，但 Mage-ViT 是 cluster-discrimination（聚类判别）目标训出来的，表征偏判别式，缝合惩罚可能比 SigLIP 系大——**先测再投钱**。

实操资产注记：`scripts/ab_conditioning_gb10.py` 的 `StaticTextShim` 就是现成的条件注入点，缝合诊断的"各层隐状态 → DiT 条件空间"扫描可直接复用该管线（预计算脚本同理）；Mage-VL / Qwen3-4B 快照已在 dc4a，影子 transformers 5.7 在 `~/Code/Mage/tf57-shadow`。本票的三个 `[[...]]` 引用已核实存在于知识库 **gooo:~/wiki/**（`entities/2609.03796-llada-image.md`、`entities/2106.07682-model-stitching.md`、`concepts/platonic-representation-hypothesis.md`）；**本机 `~/wiki` 是旧镜像，查库一律以 gooo 为准**。注意本票涉及的一切训练仍需新冻结协议，且会改动 `src/shadow_lab/*.py` 代码身份，只能在新 campaign 生效。
