# 分析实验总报告：补做 deep-research-report.md 的未完成验证

> 日期：2026-07-29　基座：Qwen3-1.7B（本地 modelscope 缓存）　硬件：单卡 RTX 5090
>
> 本报告汇总本次新增的三项分析，**补做** `deep-research-report.md`
> 「验证与复现计划」中标注为未完成、而既有实验（`analysis_1bit/`、
> `analysis_2bit/`）只覆盖了权重层 rel_l2/cosine 与 PPL 的部分。

## 0. 范围与方法

研究报告把验证分成「格式 / 算子 / 模型质量」三层（§验证与复现计划）。
此前已做：算法实现（`affine_1bit.py` / `affine_2bit.py`，77→90 测试全绿）、
PPL 与 shadow 恢复（`analysis_2bit/REPORT.md`）。**未做**的是格式容器层
与三层误差评估——这正是本次补做的内容。本机无 27B，故用量化的
Qwen3-1.7B 经我们自己的打包器落盘后做**自洽验证**，全部本地可复现。

| 分析 | 对应报告章节 | 产出 |
|---|---|---|
| `analyze_container_format.py` | §数据结构/二进制布局 + §验证表「Header 解析 / shape 恢复 / codebook」 | `analysis_container/REPORT.md` |
| `analyze_error_layers.py` | §评估指标建议 + §验证表「单层反量化 / logits 对比」 | `analysis_error_layers/REPORT.md` |
| `analyze_short_gen.py` | §验证表最后一行「短文本生成 / 首 token 一致性」 | `analysis_short_gen/REPORT.md` |

**新增实现**：在 1-bit 量化里加了**严格符号变体** `quantize_1bit_sign`
（`scale=2d, bias=−d`，codes {0,1}→{−d,+d}），与 Bonsai GGUF Q1_0 /
MLX ±d 完全对齐（既有 affine 是 ±d 的超集）。`quantize_model_1bit`
新增 `variant="affine"|"sign"`（默认 affine，不破坏既有行为与 manifest），
配套 `QuantizedLinear1BitSign` / `QuantizedEmbedding1BitSign` 与
13 条新测试（`TestQuantize1BitSign`）。全量 `pytest` 90 passed。

一键复现：
```bash
MODEL=/root/.cache/modelscope/models/Qwen--Qwen3-1.7B/snapshots/master
bash experiment/run_analysis_experiments.sh   # 或设 MODEL=... 覆盖
```

## 1. 容器格式层（analysis_container/）—— 报告声明全部坐实

把 affine-1bit / sign-1bit / ternary-2bit 量化后的模型用
`save_quantized_checkpoint` 落盘成 `model.safetensors`，再按报告伪代码
`load_safetensors_header` / `infer_quant_layout` 解析：

| 检查 | 结果 |
|---|---|
| 文件级布局 `overhead == 8 + header_len` | ✅ 三变体全部成立 |
| `data_offsets` 连续递增 | ✅ |
| 每量化层 = `(weight, scales, biases)` 三元组 | ✅ 197/197 完整 |
| `infer_quant_layout` 恢复原层几何 | ✅ 197/197 零失败 |

**实测 bits-per-weight**（逐层加权平均，对照报告声明值）：

| 变体 | native | slot | MLX-on-disk（实测） | 报告声明 |
|---|---:|---:|---:|---:|
| affine1bit | 1.125 | 1.125 | **1.250** | 1.25 ✅ |
| sign1bit | 1.125 | 1.125 | **1.250** | 1.25 ✅ |
| ternary2bit | 1.71 | 2.125 | **2.250** | 2.25 ✅ |

→ 报告的「MLX 比 GGUF 多 0.125 bpw」正是容器多记的那个 bias 组。
**Codebook 经真实落盘数据确认**：ternary codes {0,1,2}→{−s_g,0,+s_g}
（`scale=s_g, bias=−s_g`），印证报告修正注记（`s_mlx=2·s_g` 对三元不可能）；
sign codes {0,1}→{−d,+d}（`scale=2d, bias=−d`），与 GGUF Q1_0 完全对齐。

## 2. 三层误差层（analysis_error_layers/）—— 量化「免费精度天花板」

补齐报告点名缺失的指标，三档打包横向对比：

| 变体 | 权重 cos | hidden cos 均值 | KL(teacher‖student) | top-10 overlap |
|---|---:|---:|---:|---:|
| affine1bit | 0.796 | 0.137 | 12.96 | 0.022 |
| sign1bit | 0.792 | 0.095 | 11.24 | 0.027 |
| **ternary2bit** | **0.897** | **0.342** | **10.26** | **0.089** |

- **权重层新增 MAE / max-abs**（既有只测 rel_l2/cos）。
- **激活层**：逐层 hidden-state cosine 全程远低于权重 cosine（噪声层间累积），
  是「性质级崩塌」最早的指标——坐实报告「sub-4-bit 是质变非渐变」。
- **输出层**：logits KL + top-k overlap + teacher-greedy-CE 三项此前仓库内均无。
- **三层单调一致**：权重几何越好 → hidden cosine↑ → KL↓ / top-k↑。
- **三元 2-bit 全面最优**（3 电平 > 任一 2 电平）。

> 非平凡发现：sign-1bit 的权重 cosine（0.792）略逊 affine（0.796），
> 但其 KL（11.24）与 top-k overlap（0.027）反而更好——对称 ±d 的输出
> 分布略优于 min/max affine，即便权重几何略差。

## 3. 短生成层（analysis_short_gen/）—— 贪心解码崩塌的直观证据

同 prompt 贪心续写 24 token，比 fp16 teacher：

| 变体 | 首 token 一致（与 teacher） |
|---|---:|
| affine1bit | 0/8 (0%) |
| sign1bit | 0/8 (0%) |
| ternary2bit | 3/8 (38%) |

文本对照（Prompt 1「…Eiffel Tower.」）：
- teacher：`The Eiffel Tower is a wrought iron lattice structure…`
- affine1bit：`[ [ [ [ [ [ [ [ [ [ [ [ [ [`
- sign1bit：`"uring  " "`
- ternary2bit：`HH    H       /`

→ 1-bit 贪心续写彻底退化为乱码/单 token 循环；三元勉强保留一点一致性。
这给「PTQ 免费精度到此为止」最直观的证据，与既有 `analysis_2bit/REPORT.md`
的天文数字 PPL（447 万）相互印证：**剩余恢复只能靠训练（shadow + 蒸馏）**。

## 4. 结论与对既有工作的衔接

1. **报告的格式层声明全部本地坐实**：容器布局、三元组、shape 恢复、
   三档 bpw（1.125 / 1.25 / 1.71 / 2.125 / 2.25）逐位吻合，codebook 映射
   经真实数据确认——修正注记（`s_mlx=2·s_g` 错误）成立。
2. **三层误差量化了「免费精度天花板」**：ternary 2-bit > 任一 1-bit；
   1-bit 在输出层（KL≈12、top-k≈0.02、首 token 0%）已质变崩塌。
3. **与既有 E2 训练实验衔接**：本分析的 KL/hidden-cos/top-k 为 shadow
   恢复提供了**干净的逐层误差基线**——恢复后这些指标应被拉回，可作为
   `analysis_2bit/REPORT.md` 中 PPL 之外的第二组训练目标/监控指标。
4. **新 sign 变体的定位**：与 Bonsai GGUF Q1_0 / MLX ±d 完全对齐（容器兼容），
   便于后续若需与 llama.cpp 生态互通；当前在 1.7B 上输出分布略优于 affine。

## 5. 产物清单

```
src/shadow_peft/quantization/affine_1bit.py   # + quantize_1bit_sign, *1BitSign, variant
src/shadow_peft/quantization/__init__.py      # + exports
src/shadow_peft/__init__.py                   # + exports
tests/test_quantization.py                    # + TestQuantize1BitSign (13 tests)
experiment/analyze_container_format.py        # 新
experiment/analyze_error_layers.py            # 新
experiment/analyze_short_gen.py               # 新
experiment/run_analysis_experiments.sh        # 新（一键跑全部）
experiment/analysis_container/REPORT.md       # + header_layout.json
experiment/analysis_error_layers/REPORT.md    # + weight/hidden/logits CSV
experiment/analysis_short_gen/REPORT.md       # + gen_table.md
```
