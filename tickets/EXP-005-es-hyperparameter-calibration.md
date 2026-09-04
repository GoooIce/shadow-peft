---
id: EXP-005
type: experiment
status: done
priority: P0
depends_on: [ENG-003, ENG-004]
blocks: [EXP-001]
hardware: GB10
phase: 2
---

# Phase 2：ES sigma / alpha 校准

## 目标

在不解封 checkpoint/auto/human test 集的前提下，确定 31M 维 NVFP4 clip latent 的可辨识
扰动尺度和稳定更新尺度，冻结主实验配置。

## 工作范围

- 从同一 QUASAR 有效初始化独立扫描 `sigma ∈ {0.003, 0.01, 0.03, 0.1}`；sigma 臂之间
  共享初始化、prompt batch 和总生成预算，不累计跨臂更新。
- 每个 sigma 记录 reward std、antithetic pair 差值、clip perturbation RMSE、重构误差和
  失败率；先做 N=8/P=8 preflight，再只对候选值用 N=30 复核。
- 以同一批 observation 离线重放更新，扫描 `alpha ∈ {0.0003, 0.001, 0.003}`；记录
  latent/clip center step RMS，不为每个 alpha 重复消费生成预算。
- 对通过步长门禁的 alpha，在下一批未见 `es_train` prompts 上各做一次中心配置评估。
- 运行时、实现源码、ModelScope 资产、prompt/calibration 和 checkpoint 身份全部冻结。

## 预注册选择规则

- 排除任何 candidate failure、奖励全同或精确重放失败的 sigma 臂。
- Preflight 数值门禁冻结为：reward std `>=0.02`、antithetic pair 的平均绝对 reward 差
  `>=0.02`、平均重构误差不超过原始 QUASAR grid 的 `1.05x`、candidate clip perturbation
  RMS `<=0.03`、0 failure 且 replay exact。从全部通过的臂中选择最小 sigma；不能用单个
  幸运 best-of-N 作为依据。
- alpha 的首要门禁是 center clip step RMSE 不超过 0.02 且下一批中心奖励不恶化；若没有
  alpha 通过，则缩小步长，不进入主实验。
- 若首轮 alpha 无一通过，fallback 固定为几何缩小后的 `{0.00003, 0.0001}`，仍用同一批
  observation 离线复算；所有通过步长门禁的 fallback 在新的 episode 2 上评估，并显式排除
  已用于首轮判断的 episode 1。若两者均不恶化，冻结较大的 0.0001；若仅一个不恶化则冻结
  该值；若仍无一通过，报告校准失败。
- N=30 复核冻结为 P=8、1 generation、独立 `es_seed=20260905`，仍使用相同
  `schedule_seed=20260904`、QUASAR 初始化和 generation-0 prompt batch；避免把 N=8 的
  前八个扰动重复计作独立证据。复核沿用 sigma 数值门禁，并要求选定 alpha 的 center clip
  step RMSE 仍不超过 0.02。N=30 实际更新出的中心必须再在 episode 3 与初始中心配对评估，
  显式排除先前 episode 1/2，奖励不恶化方可完成校准。
- smoke 的 `N=4, sigma=0.03, alpha=0.015` 只作反例，不参与选择。

## 验收标准

- 四个 sigma preflight 使用同 prompt/seed 和同有效初始化，生成预算相等。
- sigma 报告包含每个 antithetic pair 的 paired reward 差，且所有指标可由 seed 重放。
- alpha 扫描由同一 observation 离线复算，输出每个候选的 center hash 与 step RMS。
- N=30 复核后冻结且只冻结一组 sigma/alpha；没有通过则按停止规则报告失败。
- GB10 日志、manifest、checkpoint 和独立验证报告同步回 `runs-local/`。

## 非目标

- 不比较 ES 与随机/网格搜索，不宣称最终质量改善。
- 不使用 checkpoint_val、auto_test、human_dev 或 human_test 选择超参数。
- 不量化 Qwen3-VL/VAE，不引入 INT4 或 2-bit。

## 证据

- 2026-09-04：ENG-004 smoke 显示 `alpha=0.015` 下中心 clip 单步 RMSE
  0.148/0.190/0.183，平均重构误差约 38k→72k→120k，因此本票成为主实验硬门禁。
- 2026-09-04：依赖全部完成，领取本票；先实现共享初始化、候选扰动/更新步长诊断和离线
  alpha 重放，再启动 GB10 sigma preflight。
- 2026-09-04：N=8/P=8 preflight 中 sigma 0.003、0.01、0.03 通过；sigma 0.1 的
  perturbation RMSE=0.04752、重构误差比=1.10009，未通过。依预注册规则选最小的
  sigma=0.003 进入复核。原定 alpha 扫描仅 0.0003 通过步长门禁（clip step
  RMSE=0.01743；0.001/0.003 分别为 0.08085/0.23630）。alpha=0.0003 在 episode 1
  的中心奖励相对初始中心下降 0.03642，因此按停止规则进入预注册 fallback，不启动 N=30。
- 2026-09-04：fallback 的 0.00003/0.0001 clip step RMSE 分别为 0.00160/0.00538；在
  全新 episode 2 上相对初始中心奖励分别 +0.18655/+0.05416，均不恶化。按预注册规则冻结
  较大的 alpha=0.0001 进入独立 N=30 复核，不以该批次上的 reward 最大值反选步长。
- 2026-09-04：独立 N=30 复核通过 sigma 数值门禁（reward std=0.07070、mean |pair
  Δ|=0.08181、perturb max=0.001061、recon ratio=1.001001、30/30 成功、重放 exact），
  alpha=0.0001 的 center step RMSE=0.002290 且中心哈希与 checkpoint 重放一致；但实际中心
  在全新 episode 3 相对初始中心 reward Δ=−0.032409，最终泛化门禁失败。
- 2026-09-04：实验按停止规则结束，不冻结配置；`calibration.json` 为 `passes=false`、
  `frozen_config=null`。远端/本地 calibration、validation、checkpoint 哈希一致；完整 2.4G
  产物已同步到 `runs-local/mage-flow/es-sigma-preflight-n8-p8-20260904/`，报告见
  `docs/experiments/mage-flow-es-hyperparameter-calibration-gb10-20260904.md`。
