---
id: EXP-008
type: engineering
category: enhancement
status: done（QAD 双门通过；ES sigma 门失败，不可部署）
priority: P0
depends_on: [ENG-003, ENG-004]
blocks: [EXP-001]
hardware: GB10
phase: 2d
---

# Mage NVFP4 两阶段修复

## Agent Brief

**Category:** enhancement

**Summary:** 为 Mage-Flow-Turbo 建立先 Mage QUASAR QAD、后 ES clip refinement 的
两阶段 NVFP4 修复流程，并用自动质量门和人工视觉门阻止 ES 从不可用的量化起点继续搜索。

**Current behavior:**

当前 Mage ES 使用静态 QUASAR-grid 初始化。该初始化以均匀显著性在 15 个 clip factor 中
最小化权重重建误差，没有训练 latent 权重，也没有利用 Mage 的 prompt、去噪轨迹或图像
质量。ES 随后从原始 BF16 master 的静态量化邻域搜索 clip。

现有完整 QUASAR QAD 只面向因果语言模型，以 token teacher KL 训练，不能处理 Mage 的
MMDiT 去噪轨迹。已停止的 16,000 图 ES 扩展在相对 Reward 上找到正增量，但人工查看仍是
明显崩坏图；该结果证明相对改进不能替代绝对质量门。

**Desired behavior:**

第一阶段执行 Mage QUASAR QAD。BF16 教师与 NVFP4 学生在四个去噪位置接收相同的学生当前
latent、prompt conditioning 和 timestep，学生以教师 MMDiT 输出为监督训练全部 174 个
MMDiT Linear 的 latent 权重。权重和 activation 在训练前向中保持 NVFP4 W4A4 fake-quant
语义；Qwen3-VL 与 VAE 保持 BF16。

2026-09-05 协议修订（维护者已明确批准“全部训练 go”）：保留全部 174 层训练，增加末层
文本输出教师 MSE。原图像损失无法监督最终被丢弃的三条文本 Linear 分支；独立 8 步对照
中，辅助组图像验证误差改善 5.33%、8/8 条胜出，维护者盲投为辅助组 7 票、持平 1 票。
正式训练从原模型重新开始；在正式 `qad_train` 首条样本上将文本项初始贡献校准为图像项
的 10%，随后固定系数，并保存进检查点以保证续训不重新校准。检查点选择和阶段门仍仅
依据图像教师误差。小实验全部 16 条提示词已从正式数据中排除。采用与小实验一致的
AdamW `foreach=False`。其余数据、阶段预算、自动门、人工门和 ES 约束保持原协议。

QAD 使用新建且相互隔离的 1,024 条 `qad_train` 与 128 条 `qad_val`，不占用现有 500 条
评估池。运行必须逐级通过接口 smoke、32-step 全层 preflight、256-step pilot，才可进入
最多 1,024 steps 的正式训练；`qad_val` 连续 128 steps 无改善时提前停止。先只根据
`qad_val` 教师轨迹损失和训练健康状态选择 checkpoint，而不是无条件采用最后一步；选定后
才允许读取 `checkpoint_val` 执行一次阶段质量门。

选定 checkpoint 后，必须在现有 `checkpoint_val` 60 条上一次性执行自动门：paired Reward
至少为 -1.0，HPS-v2.1 与 CLIP H/14 均至少保留对应 BF16 均值的 80%，零生成失败，并通过
确定性和原生 GB10 NVFP4 kernel 审计。随后在 `human_dev` 30 条上对 BF16、RTN、静态
QUASAR-grid 和 Mage QUASAR QAD 做隐藏身份、随机顺序的四路盲测；QAD 必须零明显崩坏图，
并至少在 24/30 条上优于静态 QUASAR-grid。必须获得维护者明确的人工放行，不能自动启动
ES。

通过两道门后，QAD latent 权重冻结为 healed master。后续 ES 只能从该 master 派生
saliency-aware clip genome，且只能更新 clip；不得回退到原始 BF16 master，也不得继续
更新 QAD latent 权重。旧 sigma、alpha 和 16,000 图预算全部失效：新起点先做 N=8 sigma
preflight，再做 N=30 步长确认，只有稳定通过才另行冻结扩大协议。

ES Reward 保持现有的 paired、等权标准化 HPS-v2.1 与 CLIP H/14，以便和历史结果比较；
但候选必须同时满足 QAD 绝对质量底线、教师轨迹一致性硬约束和逐批图片落盘。一个轴的改善
不能补偿另一个轴的退化。

ES 最终候选只解封一次 `auto_test` 60 条：平均 paired Reward delta 必须大于零，至少
45/60 条不退化，HPS 与 CLIP 均值都不退化，继续满足绝对质量底线，且零失败、确定性审计
通过。随后在 `human_test` 30 条上做最终盲测；只有维护者明确批准后才能标记成功和允许
部署。

**Key interfaces:**

- `QuasarNVFP4Controller` — 应支持 Mage MMDiT 的在线教师轨迹蒸馏，同时保持显著性来自
  AdamW 二阶矩、权重/activation NVFP4 fake-quant 和非量化模块冻结的合同。
- Mage QAD campaign configuration/report — 应表达数据身份、四阶段预算、四步轨迹、验证与
  early-stop 状态，并能原子续跑。
- Mage QAD checkpoint — 同时包含可续训 latent master、optimizer/二阶矩、可部署 packed
  NVFP4、saliency-aware 初始 clip genome，以及模型、数据、代码、配置和产物哈希。
- `FrozenLinearWeights` / `ClipGenomeLayout` — ES handoff 必须明确绑定 healed master 与对应
  layout；所有候选重建和身份验证都以该 master 为源。
- `RewardCalibration` 与阶段门报告 — 保持 paired HPS/CLIP Reward，同时表达不能被综合分
  抵消的绝对轴门、教师轨迹门、原生 kernel 门和人工放行状态。

**Acceptance criteria:**

- [x] 独立的 Mage QAD 单元测试证明教师/学生接收相同 latent、conditioning、timestep，且
      四个去噪位置均被覆盖。
- [x] 只有 174 个 MMDiT Linear latent 权重可训练；Qwen3-VL、VAE、norm、bias 和其他参数
      保持冻结，训练前向使用 NVFP4 W4A4 权重与 activation fake quant。
- [x] `qad_train` 1,024 条与 `qad_val` 128 条唯一且互不重叠，并与现有 500 条评估池无
      prompt/sample/seed 泄漏；数据来源、分层与哈希写入 manifest。
- [x] 接口 smoke、32-step preflight、256-step pilot 和最多 1,024-step 正式训练具有独立
      输出与停止门；任一级失败都不能启动下一级。
- [x] checkpoint 可精确续训，且完整记录 latent、optimizer/二阶矩、packed NVFP4、初始
      clip genome、配置和实现身份；重复物化得到相同权重与图片哈希。
- [x] 选定 checkpoint 是训练状态有限、零失败且 `qad_val` 轨迹损失最低的 checkpoint，
      且没有读取 `checkpoint_val`、`auto_test`、`human_dev` 或 `human_test` 来选择训练步数。
- [x] `checkpoint_val` 自动门强制 Reward ≥ -1.0、HPS/CLIP 各 ≥ 80% paired BF16、零失败、
      精确重放与 GB10 原生 NVFP4 kernel；失败时流程停止且不生成 ES 启动许可。
- [x] `human_dev` 四路盲测隐藏候选身份并随机化顺序；只有零明显崩坏且 QAD 至少 24/30
      优于静态 QUASAR-grid，并收到维护者明确批准，才生成 ES 启动许可。
- [x] ES 使用 healed master 和 saliency-aware clip genome；测试能检测并拒绝原始 BF16
      master、旧 ES center、QAD 后继续更新 latent 权重及旧 sigma/alpha 的复用。
- [ ] 新起点依次完成 N=8 sigma preflight 与 N=30 步长确认；扩大预算必须由后续冻结票据
      明确授权，不能由本票自动启动。**未达成：sigma 阶段在 candidate-00 失败，一次性门已消费。**
- [ ] 最终 `auto_test` 与 `human_test` 只解封一次；自动门、人工门全部通过前不得产出
      deployable/success 判定。**未触及：ES 链不可达，`deployable=false`、`success=false`。**

**Out of scope:**

- Qwen3-VL、VAE、INT4、INT2/INT3 或其他量化格式。
- 把 HPS、CLIP 或人工评分作为 QAD 的可微训练损失。
- 联合执行 QAD latent 更新与 ES clip 更新。
- 自动复用旧实验的 center、sigma、alpha、prompt 统计或 16,000 图预算。
- 在人工视觉门之前启动长期 ES，或仅凭相对 Reward 提升宣称图片可用。

## Triage evidence

- 2026-09-05：确认现有 ES 的初始化记录为 `quasar-grid`，且静态重建调用使用均匀显著性；
  没有 Mage 训练阶段。
- 2026-09-05：确认现有完整 QAD campaign 仅支持因果语言模型 teacher-token/KL 接缝；仓库
  尚无 Mage MMDiT 在线教师轨迹蒸馏或 healed-master handoff。
- 2026-09-05：`.out-of-scope/` 不存在，没有相似的历史拒绝；该需求不是已实现功能。
- 2026-09-05：维护者完成三轮决策并全部接受推荐答案；本票进入 `ready-for-agent`，但创建
  票据不代表已领取、实现或启动实验。

## 实现进度与证据

- 2026-09-05：领取并实现 `scripts/run_mage_flow_healing.py`；数据隔离、四阶段 QAD、原子
  检查点、自动/人工门、显著性 healed master、受限 N=8/N=30 和最终门已接通。
- 操作流程与产物合同见 `docs/mage-two-stage-healing.md`；配置见
  `configs/mage-qad/exp008.json`。
- 本地验证覆盖四步同输入、174 层训练范围、W4A4、精确续训、最小验证损失选择、128-step
  early stop、packed 重放、数据泄漏、非补偿质量门、盲测和显式人工放行。
- 状态保持 `in-progress`：真实新数据池、GB10 分级训练/原生 kernel/图片质量证据及维护者
  两次人工放行尚未完成。没有启动旧 ES 或生成真实部署许可。
- 最终验证：`.venv/bin/python -m pytest -q` → **416 passed, 18 skipped**；新增/修改文件
  Ruff 检查通过，Mage QAD/healing 模块及 CLI 的 `uvx ty check ... --python .venv/bin/python`
  通过；CLI `--help` / `train --help` 通过。
- Code review 起点：`00413c270e89d9966520a326994ab52b77b3396c`。
  **Standards**：硬规范问题 0，低优先级维护性建议 1（ES 状态读写可进一步集中到工作流对象）。
  **Spec**：阻断性发现 0；后续复核建议的排除 manifest 完整性校验已补充并经失败→通过测试验证。

## 终局（2026-09-07 收官，2026-09-08 归因）

- 正式训练在累计第 801 步触发 128 步早停；选定 `best-673`（`qad_val` 图像教师 MSE 0.21871），
  选点未读取任何留出集。检查点 `best-673.pt` 与 `latest.pt`（801）在 dc4a 落盘。
- `checkpoint_val` 自动门通过：CLIP 0.3768 / HPS 0.2622 对 BF16 0.3757 / 0.2794，paired Reward
  −0.216，non_regressions 13，零失败，原生 NVFP4 kernel 与精确重放审计通过。
- `human_dev` 四路盲测通过：维护者 goooice 投票 wins 29/30 ≥ 24，visibly_broken 0，`approved`；
  ES 启动许可 `approval_sha256=6a3593a4…` 生成。
- ES sigma 阶段（σ=0.03、α=0.0025456、seed=42，32 条 `es_train`）在 candidate-00 失败：
  `clip_h14 batch mean regressed`、`hps_v2_1 batch mean regressed`、`teacher trajectory regression`
  三硬门全破（CLIP 0.3574 / HPS 0.2561 对 BF16 0.3800 / 0.2787，reward −0.489，non_regr 6/32）。
  一次性门已消费，ES 链不可达；ledger `deployable=false`、`success=false`。
- 存活产物：41 GB 训练态检查点、`qad-lean-best-673.pt`（25.27 GiB，sha256 a4f4ff23…）、
  `qad-nvfp4-best-673.pt`（2.276 GiB，sha256 437f1c79…，174 层 packed + 32.1M clip 基因组），
  后者已在 GB10 原生 kernel 与 Mac MPS（bf16）双端生成验证。
- 归因见 `docs/experiments/exp008-es-sigma-attribution-20260908.md`：失败由硬门结构性不可通过
  （对镜像候选施加逐样本与批均值零容差）、σ=0.03 为代码默认值而非测量值（EXP-005 同维度校准为
  0.003）、N=8 人口比 EXP-007 实测出跨 prompt 梯度信号的量级（N=80×24 批）低两个数量级三者叠加
  造成，单纯调小 σ 不会改变结局。
- 本票结论：Mage QUASAR QAD → healed master 这一半成立并有完整证据；从 healed master 出发的
  ES clip refinement 在本票协议下失败。后继 ES 由 EXP009（路线 A，2026-09-08 拍板）以新冻结
  协议另行立项，本票不再重开任何门。
