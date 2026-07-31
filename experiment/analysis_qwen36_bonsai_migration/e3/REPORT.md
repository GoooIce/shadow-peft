# E3 — Bonsai/PTQ main × auxiliary four-way MLX ablation

A is the all-official reference. KL is `KL(A || case)`.

## Aggregate

| Case | Mean KL | Mean logit cosine | top-10 overlap |
|---|---:|---:|---:|
| A_official_main_official_aux | 0.00000000 | 1.00000000 | 40/40 |
| B_official_main_qwen_aux | 0.25831230 | 0.95445148 | 30/40 |
| C_ptq_main_official_aux | 11.43976887 | 0.51399402 | 1/40 |
| D_ptq_main_qwen_aux | 11.84127439 | 0.50549457 | 1/40 |

## Per-prompt result

| Case | Prompt | KL | logit cosine | top-10 overlap | Generation |
|---|---|---:|---:|---:|---|
| A_official_main_official_aux | fact | 0.00000000 | 1.00000000 | 10/10 | Paris |
| A_official_main_official_aux | math | 0.00000000 | 1.00000000 | 10/10 | 391 |
| A_official_main_official_aux | zh_math | 0.00000000 | 1.00000000 | 10/10 | 54 |
| A_official_main_official_aux | code | 0.00000000 | 1.00000000 | 10/10 | ```python\ndef add(a, b):\n    return a + b\n``` |
| B_official_main_qwen_aux | fact | 0.01367416 | 0.96781663 | 8/10 | Paris |
| B_official_main_qwen_aux | math | 0.77440158 | 0.96708855 | 7/10 | 399 |
| B_official_main_qwen_aux | zh_math | 0.07037585 | 0.93772077 | 8/10 | 51 |
| B_official_main_qwen_aux | code | 0.17479760 | 0.94517995 | 7/10 | ```python\ndef add(a, b):\n    return a + b\n``` |
| C_ptq_main_official_aux | fact | 13.02118862 | 0.47871699 | 1/10 |  |
| C_ptq_main_official_aux | math | 11.09818683 | 0.53856555 | 0/10 | ........................ |
| C_ptq_main_official_aux | zh_math | 11.29382705 | 0.50329549 | 0/10 | . |
| C_ptq_main_official_aux | code | 10.34587298 | 0.53539805 | 0/10 | . |
| D_ptq_main_qwen_aux | fact | 13.71659335 | 0.48017474 | 1/10 | fkk fkk fkk fkk fkk fkk fkk fkk fkk fkk fkk fkk fkk fkk fkk fkk fkk fkk fkk fkk fkk fkk fkk fkk |
| D_ptq_main_qwen_aux | math | 10.89667818 | 0.53878613 | 0/10 | fkk fkk fkk fkk fkk fkk fkk fkk fkk fkk fkk fkk fkk fkk fkk fkk fkk fkk fkk fkk fkk fkk fkk fkk |
| D_ptq_main_qwen_aux | zh_math | 11.25611121 | 0.48591560 | 0/10 | fkk fkk fkk fkk fkk fkk fkk fkk fkk fkk fkk fkk fkk fkk fkk fkk fkk fkk fkk fkk fkk fkk fkk fkk |
| D_ptq_main_qwen_aux | code | 11.49571482 | 0.51710182 | 0/10 | fkk fkk fkk fkk fkk fkk fkk fkk fkk fkk fkk fkk fkk fkk fkk fkk fkk fkk fkk fkk fkk fkk fkk fkk |

## Interpretation

- The one-shot main matrices still fail all four semantic probes (case D aggregate top-10 overlap 1/40). Better offline code agreement is therefore insufficient for recovery.
- Official auxiliary parameters do not rescue the one-shot main weights (case C aggregate top-10 overlap 1/40).
- This rejects a pure threshold/scale PTQ reproduction as the full Bonsai mechanism. Training-aware weight migration or distillation is required for the remaining main-matrix gap.

## Protocol

- Quantized main matrices: 498
- Main parameter arrays: 1494
- Auxiliary parameter arrays: 353
- Peak MLX memory: 14.440 GiB
- Greedy generation, thinking disabled, max 24 tokens.
