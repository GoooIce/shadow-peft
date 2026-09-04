---
id: EXP-006
type: experiment
status: done
priority: P0
depends_on: [EXP-005]
blocks: [EXP-001]
hardware: GB10
phase: 2b
---

# Phase 2b：ES 跨 prompt 梯度泛化门禁

## 背景

EXP-005 证明 sigma=0.003 在 32M 维 NVFP4 clip 空间有稳定可测的 antithetic reward 信号，
但 N=30 单批梯度更新中心在下一未见 prompt batch 上退化 0.03241。继续 100 代会把采样噪声
累积为高成本错误，因此需另立协议，不能在已完成票据内事后调参。

## 冻结实验协议

- 冻结 sigma=0.003、N=30、`es_seed=20260906`、`schedule_seed=20260904`、相同
  QUASAR 初始化和共同 antithetic noise。
- 用互不重叠的 `es_train` episode 4/5/6（各 P=8）分别估计梯度，记录两两 cosine、
  两两及三者逐维符号一致率；不解封其他 split。
- 离线比较单批梯度与三批平均梯度；只对预注册步长 `{0.00003, 0.0001}` 物化中心。
- 在 episode 7 上评估初始中心（含一次 exact replay）与两个平均梯度候选；非退化定义为
  聚合奖励不低于同批初始中心。若两步长均通过，固定选较大的 0.0001；若仅一个通过则选它；
  若均失败则立即停止。
- 入选中心在 episode 8 上再次与初始中心配对，并重放初始中心；第二次仍不退化才解锁
  EXP-001。任一门禁失败即停止，不继续缩步长或更换 prompt。
- 票据草案曾要求记录三条单批梯度“各自的 next-batch center reward”，但这会使中心评估
  超过同一草案预注册的 32 张预算。领取评审时将其收敛为只评估两条三批平均梯度候选；
  单批梯度只用于一致性诊断，不参与中心选择。

## 预算

- 梯度观测：30 candidates × 3 batches × 8 images = 720 张，另有每批首个 candidate
  的 exact replay 共 3 × 8 = 24 张，合计 744 张，4 steps。
- 中心评估：episode 7 的 initial、2 个候选及 initial replay 共最多 32 张；episode 8 的
  initial、入选中心及 initial replay 再最多 24 张。含全部审计的冻结总上限为 800 张。
- 预计 gooo 单 GB10 约 45–60 分钟，峰值内存沿用 EXP-005 的约 31.2GB。

## 验收标准

- 运行前冻结 prompt episode、noise seed、聚合公式、步长、停止规则和总生成预算。
- 三批观测各自零失败、reward std≥0.02、replay exact，且输出梯度一致性报告。
- 平均梯度中心 clip step RMSE≤0.02；两个连续未见 batch 均不劣于初始中心才可解锁主实验。
- 远端/本地 artifact hash 一致；负结果同样完成本票。

## 非目标

- 不继续缩小 alpha 直到偶然过门，不把 center evaluation prompt 纳入梯度。
- 不启动 100 代搜索，不比较网格/随机，不量化 Qwen3-VL/VAE。

## 证据

- 2026-09-04：用户指示 `$implement 继续`，据此完成领取前评审并领取本票。运行前已冻结
  episode 4/5/6 梯度观测、episode 7 首门禁、episode 8 复核、共同 noise seed、两档步长、
  非退化选择规则与停止规则；同时消除了草案单批中心评估要求和 32 张预算之间的冲突。
- 2026-09-04：episode 4/5/6 的 90/90 candidates 成功，0 failures，三臂 reward std 为
  0.07541/0.05158/0.06521，均 exact replay。梯度两两 cosine 为
  0.23982/0.34277/0.17732，逐维符号一致率为 57.62%/61.02%/55.62%，三者同号 37.13%。
- 2026-09-04：平均梯度的 alpha=0.00003/0.0001 clip step RMSE 为
  0.000454/0.001516。episode 7 上两者相对 initial reward Δ 为 +0.018483/−0.002200，
  按规则唯一入选 0.00003；该中心在 episode 8 相对 initial Δ=−0.022817，最终门禁失败。
- 2026-09-04：`result.json` 冻结为 `status=stopped-confirmation`、`passes=false`、
  `frozen_config=null`，不启动 EXP-001。远端/本地 38 个文件 SHA-256 全部一致；完整报告见
  `docs/experiments/mage-flow-es-cross-prompt-gradient-gate-gb10-20260904.md`。
- 2026-09-05：提交后规格评审发现预算段只列了 720 张主候选、漏列启动脚本从运行前即固定
  启用的三次首候选 exact replay（24 张）。实际执行路径固定为 744 张梯度观测加 56 张中心
  评测，最大总数 800 张；这是预算披露偏差，不是事后增加生成或选择自由度。门禁终结器现从
  record reward 重算差值与非退化结论，并拒绝与中间 comparison 字段不一致的报告；原始产物
  重算后仍为 `stopped-confirmation`。
