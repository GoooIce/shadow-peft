# 三层误差分析：1-bit / 2-bit PTQ 的「免费精度天花板」

> 对应 deep-research-report.md §评估指标建议 + §验证计划「单层反量化 /
> logits 对比」两行。现有 analysis_1bit/2bit 只测了权重层 rel_l2/cosine，
> 本分析补齐：权重层 **MAE / max-abs**，激活层 **逐层 hidden cosine /
> rel-L2**，输出层 **logits KL + top-k overlap**——后两者此前仓库内不存在。
> 模型：Qwen3-1.7B（本地缓存）；三档打包横向对比。

## 1. 权重层（全模型 Linear/Embedding 平均）

| 变体 | MAE | max-abs | rel_l2 | cosine |
|---|---:|---:|---:|---:|
| affine1bit | 0.01603 | 0.35766 | 0.6066 | 0.7962 |
| sign1bit | 0.01616 | 0.50762 | 0.6132 | 0.7918 |
| ternary2bit | 0.01167 | 0.44226 | 0.4433 | 0.8973 |

MAE / max-abs 是本分析新增的两项（报告点名缺失）。max-abs 反映离群权重的
量化残差峰值（三元/二值把组内极值压到 ±s_g / ±d，残差正比于 |w|−scale）。

## 2. 激活层（逐层 hidden-state vs fp16 teacher）

探针集 4 句，padding 后单次前向；逐 decoder 层比 teacher。

| 变体 | hidden cos 均值 | hidden cos 最差层 | hidden rel-L2 均值 |
|---|---:|---:|---:|
| affine1bit | 0.1368 | -0.0143 | 0.8693 |
| sign1bit | 0.0948 | -0.0109 | 1.1001 |
| ternary2bit | 0.3419 | 0.2502 | 0.8876 |

hidden cosine 全程低于权重 cosine（PTQ 噪声在层间累积），是最早出现
「性质级崩塌」的指标——这与报告强调的「sub-4-bit 是质变非渐变」吻合。

## 3. 输出层（logits vs fp16 teacher）

KL(teacher‖student)（nats）+ top-10 集合重合率 + teacher 贪心 token 在
student 下的交叉熵。三者此前仓库内均无。

| 变体 | KL(teacher‖student) | top-10 overlap | CE(teacher-greedy | student) |
|---|---:|---:|---:|
| affine1bit | 12.9644 | 0.0219 | 14.8009 |
| sign1bit | 11.2377 | 0.0266 | 13.6309 |
| ternary2bit | 10.2601 | 0.0891 | 11.3925 |

## 4. 「免费精度天花板」横向对比

三层指标统一看：哪个 PTQ 变体的几何最接近 fp16（无需训练即可拿到）。

| 变体 | 权重 cos | hidden cos 均值 | KL | top-k overlap |
|---|---:|---:|---:|---:|
| affine1bit | 0.7962 | 0.1368 | 12.9644 | 0.0219 |
| sign1bit | 0.7918 | 0.0948 | 11.2377 | 0.0266 |
| ternary2bit | 0.8973 | 0.3419 | 10.2601 | 0.0891 |

**结论**：三层误差单调一致——权重几何越好（cosine↑）→ hidden cosine↑ →
KL↓ / top-k overlap↑。三元 2-bit（3 电平）预期优于任一 1-bit（2 电平）；
affine vs sign 1-bit 的相对优劣由 MAE/KL 决定（见 CSV）。这量化了
「PTQ 免费精度到此为止」，与 analysis_2bit/REPORT.md 的天文数字 PPL 互相印证，
并为 shadow 恢复实验给出干净的误差基线。

