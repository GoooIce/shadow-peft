# 短生成一致性分析：贪心解码首 token 与文本对照

> 对应 deep-research-report.md §验证计划最后一行（短文本生成 / 首 token
> 一致性）。fp16 teacher vs affine-1bit / sign-1bit / ternary-2-bit，
> 同 prompt 贪心解码 24 token。

## 首 token 一致率（与 teacher 贪心首 token 相同的比例）

| 变体 | 首 token 一致 | 非空输出 |
|---|---:|---:|
| affine1bit | 0/8 (0%) | 8/8 |
| sign1bit | 0/8 (0%) | 8/8 |
| ternary2bit | 3/8 (38%) | 7/8 |

## 文本对照（节选，完整见 gen_table.md）

每个变体输出均经 `<think>`/特殊 token 清洗。量化（尤其 1-bit）下贪心
续写通常退化为重复/截断，首 token 一致率刻画「最便宜的语义保持度」。

**Prompt 1**: The capital of France is Paris, a city known for the Eiffel Tower.

- teacher: 'The Eiffel Tower is a wrought iron lattice structure, and the Eiffel Tower is the tallest structure in the'
- affine1bit: '[ [ [ [ [ [ [ [ [ [ [ [ [ [ [ [ [ [ [ [ [ [ [ ['
- sign1bit: '”uring\xa0\xa0”\xa0”'
- ternary2bit: 'HH    H       /'

**Prompt 5**: In machine learning, gradient descent minimizes a loss function.

- teacher: 'The gradient of the loss function is computed, and the parameters are updated in the direction of the negative gradient. This is'
- affine1bit: 'sssssss'
- sign1bit: 'uringuringinge\xa0\xa0ounturingingeuringuringuringuringinge,ounturinguringingeuringuringinge\xa0\xa0,'
- ternary2bit: '-------------------- --'

## 结论

- 首 token 一致率随量化位宽下降：ternary-2bit > 1-bit；与三层误差分析
  （KL / top-k overlap）单调一致。
- 1-bit 下贪心续写大概率崩溃（重复 / 单 token 循环），印证「sub-4-bit 是
  质变」——免费精度到此为止，恢复需 shadow + 蒸馏（见 analysis_2bit/REPORT.md）。

