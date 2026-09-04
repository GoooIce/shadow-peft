# Mage-Flow ES 跨 prompt 梯度泛化门禁（GB10，2026-09-04）

- 票据：`EXP-006`
- 硬件：gooo / goooinext GB10，CC 12.1
- 量化：仅 MMDiT native NVFP4 W4A4，reconstruction group 128
- 结论：**第二个未见 prompt 门禁失败；没有冻结 sigma/alpha，EXP-001 不得启动**
- 完整本地产物：`runs-local/mage-flow/es-cross-prompt-gate-n30-p8-20260904/`

## 冻结协议

使用相同 QUASAR 有效初始化、`sigma=0.003`、N=30、P=8、
`es_seed=20260906`、`schedule_seed=20260904` 和共同 antithetic noise，在互不重叠的
`es_train` episode 4/5/6 分别估计梯度。只对三批梯度的逐维算术平均物化
`alpha ∈ {0.00003, 0.0001}` 两个中心。

episode 7 同批比较 initial 与两个候选；若两者均不退化选较大的 alpha，若仅一个不退化则选
该值，若均退化则停止。入选中心必须再在 episode 8 相对 initial 不退化，才可冻结配置。
所有 initial 评估均做一次逐字段 exact replay；不得在失败后继续缩步长或更换 prompt。

## 三批梯度观测

三个臂共 90/90 candidates 成功，均包含一次 exact replay，模型权重保持冻结。

| episode | reward std | gradient norm | gradient RMS | failures | 峰值 RSS |
|---:|---:|---:|---:|---:|---:|
| 4 | 0.075408 | 324,087.50 | 57.4883 | 0 | 31.21GB |
| 5 | 0.051581 | 270,889.28 | 48.0331 | 0 | 29.63GB |
| 6 | 0.065208 | 417,058.25 | 73.9961 | 0 | 31.18GB |

episode 4/6 在 gooo 运行，episode 5 在 goooinext 运行。两机的 `experiment.identity` 在去掉
预期不同的 episode offset 后逐字段一致；episode 5 经双机直连 rsync 回 gooo，并做全文件
SHA-256 对账。

## 梯度一致性与平均中心

| 梯度对 | cosine | 逐维符号一致率 |
|---|---:|---:|
| episode 4 / 5 | 0.239818 | 57.6237% |
| episode 4 / 6 | 0.342767 | 61.0202% |
| episode 5 / 6 | 0.177317 | 55.6169% |

三条梯度逐维同号比例为 37.1304%，全零比例为 0。平均梯度 norm/RMS 为
241,707.88 / 42.8616，SHA-256 为
`189e7dc707969f16f8e835dacdd327b709428a5deb7e2e28652f1f37e300be7b`。

| alpha | latent step RMS | center clip step RMSE | 步长门禁 |
|---:|---:|---:|---|
| 0.00003 | 0.00128585 | 0.00045444 | 通过 |
| 0.0001 | 0.00428616 | 0.00151604 | 通过 |

## 连续未见批次门禁

episode 7 的 initial exact replay 完全一致：

| center | reward | Δ vs initial | HPS | CLIP | 结果 |
|---|---:|---:|---:|---:|---|
| initial | −4.784312 | 0 | 0.074888 | 0.081914 | 基线 |
| alpha=0.00003 | −4.765829 | +0.018483 | 0.074501 | 0.084678 | 通过、唯一入选 |
| alpha=0.0001 | −4.786512 | −0.002200 | 0.073161 | 0.084312 | 失败 |

episode 8 再次对 initial 做 exact replay；入选中心未保持非退化：

| center | reward | Δ vs initial | HPS | CLIP | 结果 |
|---|---:|---:|---:|---:|---|
| initial | −4.758259 | 0 | 0.090031 | 0.100983 | 基线 |
| alpha=0.00003 | −4.781076 | −0.022817 | 0.090230 | 0.097999 | 失败 |

最终 `result.json` 为 `status=stopped-confirmation`、`passes=false`、
`frozen_config=null`。这支持一个更窄的结论：三批平均梯度比 EXP-005 的单批梯度更可能在首个
未见批次上保持，但仍没有通过连续两批泛化门禁；当前证据不支持启动长期 ES 主实验。

## 产物与完整性

远端结果同步后，对目录内 38 个文件逐一比较 SHA-256，远端与本地完全一致；本地目录约
1.1GB。关键哈希：

- `gradient-report.json`：`60a39f3491386204f2d32b3a2a09a134cb75d60a384e4673783ae25b4ac094de`
- `center-gate-episode7.json`：`ccf40b2087902014b3f49f7c7f81fd7b6b42a4e5c3e73f4b9e614da6e6bda268`
- `center-confirmation-episode8.json`：`5e359e58a784d425ab46aa9b49018b40aa35507c180ae56efed21811135954a7`
- `result.json`：`f9a7fd682ae33465273177c483cba38c7d64f3d944dbd90eccca971587433b60`
