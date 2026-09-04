# Mage-Flow NVFP4 ES 超参数校准（GB10，2026-09-04）

- 票据：`EXP-005`
- 硬件：gooo GB10，CC 12.1
- 量化：仅 MMDiT native NVFP4 W4A4，reconstruction group 128
- 结论：**校准未通过；没有冻结 sigma/alpha，主实验不得启动**
- 完整本地产物：`runs-local/mage-flow/es-sigma-preflight-n8-p8-20260904/`

## N=8 sigma preflight

四臂使用同一 QUASAR 有效初始化、同 8 prompts/seeds、同 antithetic noise specs。选择规则在
候选结果出现前写入票据：reward std ≥0.02、平均绝对 pair 差 ≥0.02、重构误差 ≤1.05×、
perturbation RMSE ≤0.03、零失败且重放完全一致，取最小通过 sigma。

| sigma | reward std | mean |pair Δ| | perturb RMSE max | recon / QUASAR | 结果 |
|---:|---:|---:|---:|---:|---|
| 0.003 | 0.08479 | 0.12566 | 0.001060 | 1.00100× | 通过、入选 |
| 0.01 | 0.06776 | 0.11047 | 0.003550 | 1.00267× | 通过 |
| 0.03 | 0.08200 | 0.11637 | 0.011047 | 1.01010× | 通过 |
| 0.1 | 0.11498 | 0.18262 | 0.047549 | 1.10009× | 不通过 |

这证明 32,145,408 维 clip genome 在 sigma=0.003 已有可测的真实生成奖励信号；无需用更大扰动
换取辨识度。sigma=0.1 虽有奖励方差，但已明显破坏重构，不能因信号更大而选它。

## alpha 校准

同一 N=8 observation 离线重放更新，不为 alpha 扫描重复生成：

| alpha | center clip step RMSE | 下一未见 episode | reward Δ vs initial | 结果 |
|---:|---:|---:|---:|---|
| 0.003 | 0.23630 | — | — | 步长超限 |
| 0.001 | 0.08085 | — | — | 步长超限 |
| 0.0003 | 0.01743 | episode 1 | −0.03642 | 泛化门禁失败 |
| 0.0001 | 0.00538 | episode 2 | +0.05416 | fallback 通过、按规则入选 |
| 0.00003 | 0.00160 | episode 2 | +0.18655 | fallback 通过，但按预注册规则不追分 |

episode 2 显示更小 alpha 的观测分数更高，但规则已指定两者均不恶化时选较大的 0.0001，故没有
按该批次 reward 反向调参。

## 独立 N=30 复核

使用独立 `es_seed=20260905`、同 generation-0 prompt batch、P=8、sigma=0.003、alpha=0.0001：

- 30/30 candidate 成功，failure=0；首 candidate 图像、量化与奖励精确重放。
- reward std=0.070698；15 个 pair 的平均绝对差=0.081812。
- perturbation RMSE max=0.00106061；平均重构误差=1.00100074× QUASAR。
- 更新中心 clip step RMSE=0.00229028；离线重放中心哈希与 checkpoint 相同。
- checkpoint SHA-256：`4f5d1bba737f580632c3254b526a22435b08b6407b55bece286264363cecd5aa`。

上述 sigma/步长门禁全部通过。但 N=30 实际中心在新的 episode 3 上从 −4.830776 降至
−4.863185，配对 Δ=−0.032409；HPS 0.087342→0.085366，CLIP 0.093430→0.092665，二轴都
略降。因此最终泛化门禁失败。

## 判定与后续

`calibration.json` 明确记录 `passes=false`、`frozen_config=null`。这不是“ES 不可行”的证明；
它支持更窄的判断：当前用 8 prompts 估计的一代梯度即使在 N=30 下数值稳定，也没有证明对下一
prompt batch 泛化。不能按本协议直接进入 100 代主实验。

后续若重新开票，应在看新结果前选择一种新协议，例如增大 P、跨 prompt mini-batch 聚合梯度、
或以多批下置信界而非单批点估计作更新门禁；不得在本票内继续事后缩 alpha。

关键产物哈希：

- `calibration.json`：`bd6c0880f7e1eda87b2fa8d6aa94994aa0a1713743c258c63634d45c6aa95950`
- `n30-validation.json`：`7d44cd4617e70dd2cda1e774232dca7be5a365b4442e4e54717f065d71d5df17`
- 远端与本地三项哈希逐一一致；本地镜像 2.4G，共 72 文件。
