---
id: EXP-009
type: experiment
status: in-progress
priority: P0
depends_on: [EXP-008]
blocks: [EXP-001]
hardware: GB10
phase: 2e
---

# Phase 2e：healed master 上的多批梯度 ES clip refinement

## 背景

EXP-008 的 QAD 半程成立：`best-673` 通过 `checkpoint_val` 自动门与 `human_dev` 盲测
（29/30，零崩坏，维护者批准），产出 healed master。其 ES sigma 阶段在 candidate-00 三硬门
全破、一次性门消费。归因（`docs/experiments/exp008-es-sigma-attribution-20260908.md`）：
门对镜像候选施加逐样本与批均值零容差、在结构上不可通过；σ=0.03 是代码默认值而非测量值；
N=8 比 EXP-007 实测出跨 prompt 梯度信号的 N=80×24 批低两个数量级。维护者 2026-09-08 拍板
路线 A，并判断 EXP-005/006 的"负"是欠功效而非无梯度："按推荐来，多跑跑，多记录些数据。"

本票把 EXP-007 已验证的批结构（N=80、P=8、共同 antithetic noise、批梯度一致性、未见批选择
与确认）搬到 healed master 上，把 EXP-008 的绝对底线与教师轨迹约束改成对中心施加的容差带，
并把"是否测到信号"变成预注册的可停止判据。

## 冻结协议

### 起点与身份

- master：EXP-008 `best-673`，`selected_sha256=1c98af13f8f79a750216f3adf246ff6ee300269a3ff8edefc8a20489c76ed986`；
  初始基因组为其 saliency-aware clips（`initial_clips_sha256=2c541f56…`）。latent 权重冻结，只搜索 clip。
- 基因组：全维 32,145,408（每 128 元素重建分组一个 clip），latent↔clip 映射沿用 `es.py`
  默认（clip∈[0.30,1.00]，T=0.10）。不做低维重参数化（记为可选后续对照，不在本票内）。
- 新 campaign 目录与新 ledger；以哈希核验的方式继承 EXP-008 ledger 中 `checkpoint_val` 门与
  `human_dev` 许可，EXP-008 ledger 不再写入。模型、`mage_flow` 源码与依赖版本身份必须与
  EXP-008 QAD 训练一致；本票代码身份在 `init` 时冻结，之后每个命令、每台机器都必须相同。

### 搜索参数

- σ=0.003（EXP-005 同维度校准值；EXP-007 出信号时的值）。禁止 0.03。
- α∈{3e-5, 1e-4}（EXP-007 预注册对），z-score shaping，`clip_step_rms ≤ 0.02`。
- N=80（40 对镜像），prompt 批 P=8，`schedule_seed=20260904`（与 EXP-007 相同的 es_train 分批，
  episode 与 prompt 集逐一对应），`es_seed=20260908`。同一代内所有 episode 共享 antithetic noise。
- 梯度 episode 0–23（24 批，192 条 prompt）；选择 episode 24–31；确认 episode 32–39。

### 候选评估与灾难底线（探索候选）

- 每个候选单次物化 + 生成 8 张，记录逐样本教师轨迹损失、HPS、CLIP、原生 kernel 证据。
- 灾难底线：零生成失败；每轴批均值 ≥ 80% 配对 BF16 批均值；教师轨迹批均值 ≤ 1.05 × 当前
  中心在同批的教师批均值。不通过的候选按 ES 失败惩罚（最小奖励 − 标准差）计入，不中断 episode。
- 同一 episode 失败候选 > 16/80 → episode 无效，campaign 停止。
- 每 episode 对 `population_index=0` 的候选做完整双重精确重放审计；packed 权重哈希、图片哈希、
  分数任一不一致 → episode 无效，campaign 停止。
- 每 episode 先评估当前中心（单次，8 张）作为该批教师基线与奖励参照。

### 批梯度一致性与停止规则（预注册）

- 每完成一个 episode 重算所有已完成 episode 梯度的两两 cosine 与逐维符号一致率（N=80），并
  记录 N=40、N=20 子人口重算的同一指标。
- 检查点 k=8 与 k=16：继续 ⇔ cosine 均值 ≥ 0.10 且正对比例 ≥ 0.70；否则 campaign 以
  `no-gradient-signal` 停止，不进入选择集。无信号零假设下单对 cosine ≈ N(0, 1/40)（sd 0.158）；
  EXP-007 旧 master 基准 0.315 / 0.957。
- 24 个 episode 全部完成后逐维算术平均，对两个 α 物化中心。

### 选择与确认（对中心施加的门）

- 选择：每个 episode 24–31 配对 BF16，完整评估（含双重重放）当前中心与两个 α 中心；候选中心
  通过 ⇔ 8 批平均配对 Reward 增量 > 0，且 ≥ 6/8 批不退化，且每批每轴 ≥ 80% BF16，且每批教师
  批均值 ≤ 1.05 × 当前中心。两者通过取较大 α；仅一个通过取该 α；均未通过 → `selection-failed`。
- 确认：episode 32–39，只评估入选中心与当前中心，同一判据；通过 → 写入 ledger `es.confirmation`
  （clips、`candidate_sha256`、`clips_sha256`）；否则 `confirmation-failed`。
- 一个轴的改善不补偿另一个轴：逐轴 80% 地板与教师容差带独立判定。

### 续代与最终门

- 一次确认通过 = 一代。维护者可显式授权下一代（最多 3 代）：以确认中心为起点，重用 episode
  0–39，noise 随代索引刷新，门完全相同。协调器不自动续代。
- `auto_test`（60 条）与 `human_test`（30 条）只解封一次，且只在维护者显式声明"本代为末代"后
  执行。最终自动门：每轴 ≥ 80% BF16；教师批均值 ≤ 1.05 × healed master；相对 healed master
  配对增量均值 > 0 且 ≥ 45/60 不退化；零失败、精确重放、原生 kernel。随后 `human_test` 盲测，
  维护者明确批准才可 `deployable`。

### 记录（全部落盘）

每候选：原始奖励、逐轴分数、逐样本教师损失、kernel 证据、图片；每 episode：`gradient.pt`
（float32 全维）、shaped utilities、noise seeds、中心评估、重放审计；每次一致性重算的完整矩阵
与子人口变体；α 中心张量与哈希；选择/确认的全部配对评估与图片。

### 执行

- dc4a 为协调器与偶数 episode；gooo 为奇数 episode。gooo 需持有 `best-673.pt`、EXP-008
  `campaign.json`、评估池、校准、HPS/CLIP 权重，身份哈希与 dc4a 一致。大文件互传优先光纤直连。
- 预算：梯度 24×(80+1+1)×8 = 15,744 张；选择 8×56 = 448；确认 8×40 = 320；每代约 16.5k 张。

## 验收标准

- [ ] 协议校验拒绝：σ≠正有限值或等于 0.03、非 EXP-008 master、初始 clips 不匹配、旧 center/
      sigma/alpha 导入、未继承或身份不符的 EXP-008 许可。
- [ ] 同一代内各 episode 候选使用相同 noise seeds（共同 antithetic noise），不同代 noise 不同。
- [ ] 探索候选门只含灾难底线；失败候选按惩罚计入且不中断；>16/80 失败或重放审计不一致使
      episode 无效并停止 campaign。
- [ ] 一致性报告含 N=80/40/20 三档两两 cosine 与符号一致率；k=8、k=16 停止规则按预注册阈值执行。
- [ ] 步长物化拒绝 `clip_step_rms > 0.02`；选择与确认按"均值 > 0 且 ≥ 6/8 不退化 + 逐批地板 +
      教师容差带"判定，一个轴不补偿另一个轴。
- [ ] 确认通过后 `es.confirmation` 记录可被最终门直接消费；续代需显式授权且不超过 3 代；
      `auto_test`/`human_test` 只解封一次且只在末代声明后。
- [ ] 最终门教师约束为批均值容差带，不再是逐样本零容差。
- [ ] QAD 检查点"刷新最佳"时 `latest.pt` 以硬链接指向 `best-*.pt`，不再整写第二遍；恢复语义
      与内容哈希不变（EXP009 候选清单第 1 项，随本票代码身份生效）。
- [ ] 全部落盘记录项存在且可离线重算一致性与选择判定。

## 非目标

- 不重训 QAD、不扩 QAD 数据、不调 LR。
- 不做低维重参数化（记为后续可选对照）。
- 不把 HPS/CLIP/人工评分作为可微损失；不联合更新 latent 权重。
- 不自动续代、不自动解封最终门、不用相对 Reward 提升替代人工视觉门。

## 证据

- 2026-09-08：维护者拍板路线 A 并批准推荐协议；票据创建。实现与运行证据见下。
