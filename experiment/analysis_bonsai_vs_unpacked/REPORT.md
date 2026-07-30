# 真实 Bonsai packed vs unpacked 逐层比对报告

> 数据来源：`prism-ml/Ternary-Bonsai-27B-mlx-2bit`（packed）+ `prism-ml/Ternary-Bonsai-27B-unpacked`（unpacked FP16）
> 量化层三元组总数：498，匹配到 unpacked：498

## 1. 全局统计（逐层指标汇总）

| 指标 | 均值 | 中位数 | 最大 |
|---|---:|---:|---:|
| cosine | 1.000000 | 1.000000 | 1.000000 |
| MSE | 5.783938e-23 | 0.000000e+00 | 2.880401e-20 |
| rel-L2 | 0.000000 | 0.000000 | 0.000000 |

## 2. 小层 codeword brute-force 结果

对每个小层穷举全部 24 种 2-bit code → ternary 映射，取 MSE 最小者：

| 层 | 最优映射 (code0,1,2,3 → ternary) | 最优 MSE | 与理论一致 |
|---|---|---:|:---:|
| embed_tokens | [-1, 0, 1, 0] | 1.430700e-20 | ✅ |
| layers.0.linear_attn.in_proj_a | [-1, 0, 1, 0] | 0.000000e+00 | ✅ |
| layers.0.linear_attn.in_proj_b | [-1, 0, 1, 0] | 0.000000e+00 | ✅ |
| lm_head | [-1, 0, 1, 0] | 0.000000e+00 | ✅ |
| layers.11.self_attn.k_proj | [-1, 0, 1, 0] | 0.000000e+00 | ✅ |
| layers.0.linear_attn.out_proj | [-1, 0, 1, 0] | 0.000000e+00 | ✅ |

> 理论映射：code {0,1,2} → ternary {-1,0,+1}，code 3 未使用。 若所有小层最优映射一致即证实该 codebook 是唯一解。

## 3. 结论

- 若逐层 cosine ≈ 1.0 且 MSE ≈ 0，则「MLX 容器 codebook = {0,1,2}→{-s_g,0,+s_g}, scale=s_g, bias=-s_g」在**真实 Bonsai 数据**上得到直接证实。
- 若 brute-force 最优映射在所有小层上一致且为 {-1,0,+1,unused}，则该 codebook 是唯一最优映射，报告 §改进建议的这一步验证完成。
