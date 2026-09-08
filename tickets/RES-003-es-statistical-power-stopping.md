---
id: RES-003
type: research
status: done
priority: P0
epic: EPIC-002
depends_on: [EXP-008]
blocks: []
hardware: [GB10, RTX-PRO-6000-96GB]
preferred_hardware: GB10-CPU
review_findings: [F02]
---

# ES 统计功效、停止规则与最小有用增益设计

## 来源与授权范围

来源：[EXP-009 独立 review](../docs/exp009-review-20260908.html)，2026-09-08 用户要求分别建票，并在各票估算 GB10 与 Pro 6000 算力以安排下一轮并行实验。本票当前只完成立项；创建不代表已执行、已冻结新实验参数或已经授权租用/启动云实例。现有 EXP-009、原说明、review、代码与旧票据保持原样。入口：[EPIC-002 新票索引](EPIC-002-exp009-followups-gb10-pro6000.md)。

## 问题与预先定义

cosine pair 共享 episode；6/8 不退步也不是显著性保证。下一轮需要在看正式结果前指定主效应、独立重复单位、功效和停止规则。

- 主比较：最终候选 E 对 raw healed H 的未见 prompt 配对效应；ES 对 effective center P 与 P 对 H 为机制分解。HPS/CLIP、教师损失、人工偏好单列。
- 最小有用增益 δ 和分轴非劣容差在正式数据出现前确定；开发 pilot 可用于方差估算，不作为确认结果。
- 使用 episode×方向系数/Gram 矩阵处理共享噪声结构，避免把 32M 个坐标或 276 个 cosine 对当作独立样本；零假设模拟明确符号翻转/置换成立条件，保留 shaping 与失败惩罚。
- 对 k=8/16 联合流程模拟误报/漏报及合理效应场景的功效；至少 10,000 次预定重采样/模拟，固定 seed，记录数值稳定性。
- 噪声组、prompt、生成 seed、评审员分别定义重复层级。跨噪声组优先看固定候选的外样本效应，不强求高 cosine。
- 预定两个 α、多个噪声组/低维臂的选择与多重比较处理；最终候选仅按 selection 固定，不能从 confirmation 挑赢家。

## 预算冻结方式

EXP-010 的三噪声组上限和 EXP-011/013 的样本数是待功效核查的设计上限。若预算下不能支持 δ，正式运行前记录“只能给区间/证据不足”的结论范围，或重新立项；不得看 test 后加样本。

## GB10 / Pro 6000 算力与耗时估算

| 配置 | GPU 卡时 | CPU / 人工排程 |
|---|---:|---|
| 单或双 GB10 | **0 GPUh** | 离线 CPU 工作约 2–8 h；不占生成 worker |
| 单或双 RTX PRO 6000 | **0 GPUh** | 同样约 2–8 h；不因 GPU 峰值算力而加速 |

主机内存规划 16–32 GiB；上述 CPU 时长是工作量占位，尚未跑计时基准。不要仅为本票启动计费 GPU 实例；可在已有主机空闲 CPU 上执行。算法统计工作遵守项目实验环境要求，在 GB10/获准的 Pro 主机上运行，Mac 仅做文档及代码可行性检查。人工研究/实现时间另列，不能当 GPU 小时。

硬件规格与估算依据见[共享笔记](../docs/research/exp010-gb10-pro6000-hardware-estimates-20260908.md)；GPU 任务的统一换算见[本轮索引](EPIC-002-exp009-followups-gb10-pro6000.md#统一估算口径)。

## 依赖、并行与工程工期

- `depends_on` 均完成后才领取；未完成前可准备本票规范，不能把准备误报为正式实验。现有 EXP-009 占用的 GB10 需待释放后排程，本票不停止它。
- 建议执行位置：**GB10-CPU**。硬件桥接未通过前，不混合两种 GPU 的奖励/梯度；同一个候选与其参考必须在同硬件上配对。
- 人工/实现工作量占位：约 1–2 人日；不包含在 GPU 墙钟表内。

## 验收标准

- [x] 完整规则的零假设与效应场景结果齐备，假设和低功效场景明确。
- [x] 主终点、δ、区间算法、分析次数、α/噪声组选择和停止预算均在新数据前冻结。
- [x] 生成 seed 整簇附着 prompt；重复盲测或精确重放不增加独立 prompt 数。
- [x] 对每类结果给出“可继续 / 证据不足 / 已排除实用增益”的可支持措辞，未把不显著说成无梯度。

## 证据

- 2026-09-08：依据 review 与用户双硬件拆分要求建票。耗时为估算，未运行本票；无新模型结果、无实例启动、无远端变更。
- 2026-09-08：完成。冻结统计协议
  [docs/research/res003-es-statistical-protocol-20260908.md](../docs/research/res003-es-statistical-protocol-20260908.md)；
  冻结计划 `configs/mage-healed-es/res003-statistical-plan.json`
  （SHA-256 `dbefe8f5ef892d91911f43f345d1df1dc9043c115788e54175d144b273f96c76`，δ_primary=0.25 reward-z、
  分轴非劣 −0.10z、双 α=0.05/0.01、prompt 簇 bootstrap seed 20260904×10,000 次、episode 符号翻转零假设、
  梯度预算 24 批 + 顺序双检查点校准、固定顺序族/Holm 多重比较、三措辞判定映射）。
- 2026-09-08：功效/零假设校准实现 `src/shadow_lab/es_inference.py` +
  `scripts/res003_power_analysis.py`，契约测试 `tests/lab/test_es_inference.py`（20 项，Mac 与 GB10 双侧通过，
  `ruff` 通过）。正式 10,000 次/格全网格在 GB10 `gooo` 空闲 CPU 运行（源码基线 d647c95），
  墙钟 105 s、峰值内存 651 MiB、**GPU 卡时 0**、未启动/租用实例；制品
  `runs-local/mage-flow/res003-power-analysis-full.json`
  （SHA-256 `17b50f0fb39ee379028877a1322ec72a02b67a7f2f36a65b75a1c117043d834e`）与日志已回传。
- 2026-09-08：关键校准结论：k=8 联合门零假设通过率 0.146（胜场门≠显著性）；整套顺序双检查点规则在真实
  ρ=0.10 时 72.6% 概率停掉 campaign、ρ=0.05 时 99.4%（停止≠无梯度，实际是 ≈0.15 的方向效应量门）；
  δ 预算表给出 k=8 在 σ_ep≤1.0δ 时可支持「已排除实用增益」、1.25–1.5δ 仅区间、≥2δ 需重新立项；
  EXP-011 30 prompt 场景 q=0.7 时支持率 ≥0.99。EXP-009 停止门的可支撑措辞映射见协议 §10
  （既有 EXP-009 记录未改动）。首版计划/制品（0ab04ccb…/ec3de863…）在出现任何结果数据前被补校准版本取代，
  版本关系记录于协议 §12。
- 2026-09-08：远端卫生：本次向 `gooo:~/Code/shadow-peft` rsync 时曾误用尾斜杠源把少量新文件散落到其仓库
  根目录，已全部移入 `gooo:~/quarantine-res003-spill/` 隔离（未删除）；根目录另有 2026-08-24/25 的
  旧 rsync 溢出物（根级 `shadow_core/`、`mage-qad/`、散落 *.py 等），非本票造成，保持原样待维护者处置。
