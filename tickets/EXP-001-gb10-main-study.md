---
id: EXP-001
type: experiment
status: blocked
priority: P0
depends_on: [ENG-005, EXP-005, EXP-006, EXP-007, EXP-008]
hardware: GB10
phase: 3
---

# GB10 主实验与人工验收

## 目标

在 GB10 上比较 BF16、NVFP4 RTN、NVFP4 QUASAR 网格、随机搜索和 ES×QUASAR，检验 ES 的演化累积是否优于等预算对照。

## 工作范围

- 先 profile 单 episode，依据 GB10 实测而非 H100/B200 估计冻结预算。
- 在验证集完成 sigma 校准后锁定主实验配置。
- 等生成次数比较 ES 与随机搜索；保存每代 seed、奖励、checkpoint 和运行环境。
- 定期自动评测，但最终人工 test set 只在模型选择结束后解封一次。
- 人工 A/B 采用同 prompt/seed、盲化身份、平衡左右位置和预注册统计。

## 验收标准

- 至少报告多个独立 ES seed 或说明统计功效不足。
- 自动指标报告 paired 差值及置信区间，不只报告最优样本。
- 人工验收报告 ES 对 RTN、QUASAR 和 BF16 的胜/平/负及 95% CI。
- profiler 证明正式性能数字对应 GB10 原生 NVFP4 路径；模拟结果单列。
- 原始 manifest、日志、checkpoint 和标注导出可追溯。

## 停止规则

- 若 NVFP4 RTN 对 BF16 没有可测质量差距，暂停主实验并评审问题价值。
- 若所有 sigma 的个体奖励不可区分，先评审奖励噪声和 clip 粒度，不自动切换位宽。

## 证据

- 2026-09-04：EXP-006 跨 prompt 梯度门禁在第二个连续未见 batch 失败，未冻结 ES 配置；
  按停止规则本票保持 blocked，不启动 100 代搜索或主实验。
- 2026-09-05：用户要求以 20× 预算运行 EXP-007；本票新增该依赖，在高功效复核通过前继续
  blocked。
