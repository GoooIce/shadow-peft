---
id: EXP-007
type: experiment
status: in-progress
priority: P0
depends_on: [EXP-006]
blocks: [EXP-001]
hardware: GB10
phase: 2c
---

# Phase 2c：ES 跨 prompt 梯度门禁 20× 扩展

## 背景

EXP-006 的 800 张预算只足以快速否决三批平均梯度，无法充分区分梯度估计噪声和真实的
跨 prompt 退化。用户要求把总生成预算扩大到原协议的 20 倍；本票保留 EXP-006 负结果，
另用尚未参与中心选择的冻结 prompt 批完成一次高功效复核。

## 冻结实验协议

- 总生成上限精确冻结为 16,000 张，512×512、4 steps、CFG=1.0；不解封其他 split。
- 使用冻结 `es_train` 的完整 320 条 prompt，并按 `schedule_seed=20260904` 排成 P=8 的
  episode 0–39；同一协议内每条 prompt 恰好使用在一个 episode，候选间保持 paired。
- 梯度观测使用 episode 0–23，共 24 个互不重叠批；每批 N=80、`sigma=0.003`、
  `es_seed=20260907`、z-score shaping、相同 QUASAR 初始化和共同 antithetic noise。
- 对 24 条梯度做逐维算术平均，记录全部两两 cosine/符号一致率；只物化预注册步长
  `alpha ∈ {0.00003, 0.0001}`，要求 center clip step RMSE≤0.02。
- 选择集使用 episode 24–31。每批同批评估 initial、两个候选及 initial exact replay。
  候选须满足 8 批平均 reward delta>0 且至少 6/8 批不退化才通过；两者均通过时选较大
  alpha，仅一个通过时选该值。
- 若两候选均未通过，仍按预注册诊断规则选择平均 delta 较大者（同值选较大 alpha）跑完
  episode 32–39，但不得解锁。确认集每批评估 initial、诊断/入选中心及 initial replay；
  仍须平均 delta>0 且至少 6/8 批不退化。只有选择与确认两门均通过才解锁 EXP-001。
- 不根据中途结果换 noise、prompt、alpha、阈值或缩步长；确认集始终运行，避免数据依赖预算。

## 精确预算

- 梯度主候选：24 × 80 × 8 = 15,360 张。
- 梯度首候选 exact replay：24 × 8 = 192 张。
- 选择集：8 × (initial + replay + 2 candidates) × 8 = 256 张。
- 确认集：8 × (initial + replay + selected/diagnostic) × 8 = 192 张。
- 总计：15,360 + 192 + 256 + 192 = **16,000 张**。

## 验收标准

- 双 GB10 分摊 24 个梯度 episode，所有 source run 完整、零失败、reward std≥0.02、exact replay。
- source identity 在预期 episode offset 外一致，24 批 prompt 无重叠，初始化和 noise 逐字段一致。
- 选择/确认 16 批与梯度批及彼此均无 prompt 重叠，所有 initial replay exact。
- 门禁只从 record reward 重算 paired delta；产出批级通过率、平均 delta、最终判定及冻结配置。
- 远端/本地 artifact SHA-256 一致；负结果同样完成本票。

## 非目标

- 不启动长期 ES 主实验，不解封 checkpoint/auto/human split，不增加第三个 alpha。
- 不把此前看过的 episode 4–8 作为本票的选择或确认集；它们只可进入梯度观测区。

## 证据

- 2026-09-05：用户要求将 EXP-006 的 800 张预算放大到 20 倍；领取本票并在运行前冻结为
  上述恰好 16,000 张、40 个互不重叠 episode 的协议。
