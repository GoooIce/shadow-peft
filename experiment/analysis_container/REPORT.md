# 容器格式分析：1-bit / 2-bit 打包器的 safetensors 布局验证

> 对应 deep-research-report.md §数据结构/二进制布局 + §验证与复现计划
> 的「Header 解析 / shape 恢复 / codebook」三行。本机无 27B，故用量化的
> Qwen3-1.7B（本地 modelscope 缓存）经我们自己的打包器落盘后做自洽验证。

## 1. Header 解析与文件级布局

读前 8 字节 little-endian 长度前缀 + JSON header（报告伪代码
`load_safetensors_header`），验证 payload 总和、offset 连续性、
overhead = 8 + header_len。

| 变体 | 量化层 | header_len (B) | file (B) | overhead==8+hdr | offset 连续 |
|---|---:|---:|---:|:---:|:---:|
| affine1bit | 197 | 78,104 | 891,476,256 | ✅ | ✅ |
| sign1bit | 197 | 78,104 | 891,476,256 | ✅ | ✅ |
| ternary2bit | 197 | 79,400 | 1,106,533,936 | ✅ | ✅ |

## 2. 张量三元组与 shape 恢复

报告伪代码 `infer_quant_layout`：`in_features = packed*32//bits`，
`group_size = in_features//scales.shape[-1]`。验证每个量化层都是
`(weight, scales, biases)` 三元组，且恢复出的几何 = fp16 原层。

| 变体 | bits | 三元组完整 | shape 恢复 ok / fail |
|---|---:|---:|---|
| affine1bit | 1 | 197 / 0 部分 | 197 / 0 |
| sign1bit | 1 | 197 / 0 部分 | 197 / 0 |
| ternary2bit | 2 | 197 / 0 部分 | 197 / 0 |

## 3. 实测 bits-per-weight（三档）vs 报告声明值

按真实落盘字节（weight uint32 + scales/biases fp16）逐层加权平均，
对照报告声明值：1-bit 1.125(native) / 1.25(MLX-on-disk)；
2-bit ternary 1.71(native) / 2.125(slot) / 2.25(MLX-on-disk)。

| 变体 | native (code+1×scale) | slot (code+1×scale) | MLX-on-disk (实测) |
|---|---:|---:|---:|
| affine1bit | 1.1250 | 1.1250 | **1.2500** |
| sign1bit | 1.1250 | 1.1250 | **1.2500** |
| ternary2bit | 1.7100 | 2.1250 | **2.2500** |

说明：`native` 是三元信息位宽 `log2(3)+16/128≈1.71`（2-bit）/ `1+16/128=1.125`（1-bit）；
`slot` 是 2-bit 容器位宽 `2+16/128=2.125`；`MLX-on-disk` 多记一个 bias 组 → `2+32/128=2.25`。
1-bit affine/sign 的 slot == native（2 电平，1 bit 足够），MLX-on-disk = `1+32/128=1.25`。

### 按模块类型（MLX-on-disk bpw）

| 模块 | affine1bit | sign1bit | ternary2bit |
|---|---:|---:|---:|
| embed_tokens | 1.2500 | 1.2500 | 2.2500 |
| mlp.down_proj | 1.2500 | 1.2500 | 2.2500 |
| mlp.gate_proj | 1.2500 | 1.2500 | 2.2500 |
| mlp.up_proj | 1.2500 | 1.2500 | 2.2500 |
| self_attn.k_proj | 1.2500 | 1.2500 | 2.2500 |
| self_attn.o_proj | 1.2500 | 1.2500 | 2.2500 |
| self_attn.q_proj | 1.2500 | 1.2500 | 2.2500 |
| self_attn.v_proj | 1.2500 | 1.2500 | 2.2500 |

## 4. Codebook 确认（读一个真实落盘 group）

### 2-bit ternary（验证报告修正注记）

- 层：`model.layers.0.self_attn.q_proj`；出现的 code：`[0, 1, 2]`（code 3 出现：否 ✅）
- 首组 scale=0.0134048, bias=-0.0134048；bias == −scale：✅
- 反量化电平：`[-0.0134, 0.0, 0.0134]`，全部 ∈ {−s_g, 0, +s_g}：✅
→ 确认唯一有效映射 `scale=s_g, bias=−s_g, codes {{0,1,2}} → {{−s_g, 0, +s_g}}`，
报告 §量化流程的修正注记（`s_mlx=2·s_g` 对三元不可能）成立。

### 1-bit sign（BitNet/Bonsai Q1_0：sign·mean|w|）

- 层：`model.layers.0.self_attn.q_proj`；codes：`[0, 1]`
- 首组 scale=0.019455 → d=scale/2=0.00972748, bias=-0.00972748；bias == −d：✅
- 反量化电平：`[-0.00973, 0.00973]`，全部 ∈ {−d, +d}：✅
→ 确认 `scale=2d, bias=−d, codes {{0,1}} → {{−d, +d}}`，与 GGUF Q1_0 / MLX ±d 完全对齐。

## 5. 结论

- 我们 1-bit/2-bit 打包器落盘的 safetensors 与报告描述的 grouped-affine 容器
  **完全自洽**：三元组齐全、offset 连续、overhead 恰为 8 + header_len。
- `infer_quant_layout` 从 packed+scales 形状无歧义恢复原层几何，零失败。
- 实测 MLX-on-disk bpw（1.25 / 1.25 / 2.25）与报告声明值逐位吻合——容器多记的
  bias 组正是 Bonsai 报告的「MLX 比 GGUF 多 0.125 bpw」来源。
- 2-bit ternary 与 1-bit sign 的 codebook 均经真实落盘数据确认，印证报告修正注记。

