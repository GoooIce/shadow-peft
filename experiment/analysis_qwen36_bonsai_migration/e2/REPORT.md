# E2 — Bonsai/PTQ main × auxiliary four-way MLX ablation

A is the all-official reference. KL is `KL(A || case)`.

## Aggregate

| Case | Mean KL | Mean logit cosine | top-10 overlap |
|---|---:|---:|---:|
| A_official_main_official_aux | 0.00000000 | 1.00000000 | 40/40 |
| B_official_main_qwen_aux | 0.25831230 | 0.95445148 | 30/40 |
| C_ptq_main_official_aux | 9.93583477 | 0.32852332 | 0/40 |
| D_ptq_main_qwen_aux | 12.17343537 | 0.46378523 | 1/40 |

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
| C_ptq_main_official_aux | fact | 11.51704783 | 0.21166464 | 0/10 | what does what does done about what is done about done what is done about done what is done about done what is done |
| C_ptq_main_official_aux | math | 9.51636367 | 0.32935025 | 0/10 | give only the result about given effect given by given effect given by given effect given by given effect given by given effect given |
| C_ptq_main_official_aux | zh_math | 9.60336935 | 0.35294890 | 0/10 | <unk newcom<unk> newcom<unk> newcom<unk> newcom<unk> newcom<unk> newcom<unk> newcom<unk> newcom<unk |
| C_ptq_main_official_aux | code | 9.10655822 | 0.42012949 | 0/10 | add a description about their own role add a description about their own role add a description about their own role add a description |
| D_ptq_main_qwen_aux | fact | 14.76320265 | 0.41642115 | 0/10 | what's what's what's what's what's what's what's what's what's what's what's what's |
| D_ptq_main_qwen_aux | math | 10.20120929 | 0.49545306 | 0/10 | 's wants what's wants what's wants what's where's what's where's where's where's where's where's |
| D_ptq_main_qwen_aux | zh_math | 12.05100398 | 0.52186009 | 1/10 | 话论话论论论论论论论论论论论论论论论论论论论论论 |
| D_ptq_main_qwen_aux | code | 11.67832556 | 0.42140663 | 0/10 | 's's what's what's what's what's what's what's what's what's what's what's what's |

## Interpretation

- The quantized main matrices are the decisive source of the semantic gap: case C keeps every official auxiliary parameter but has only 0/40 aggregate top-10 overlap.
- Official main matrices plus Qwen auxiliary parameters (case B) retain factual and code behavior, but arithmetic regresses. Auxiliary migration is therefore secondary, not irrelevant.
- Official auxiliary parameters reduce the broken-PTQ mean KL, but cannot rescue generation. The next reproduction experiment should change the main-matrix code/scale rule first.

## Protocol

- Quantized main matrices: 498
- Main parameter arrays: 1494
- Auxiliary parameter arrays: 353
- Peak MLX memory: 14.440 GiB
- Greedy generation, thinking disabled, max 24 tokens.
