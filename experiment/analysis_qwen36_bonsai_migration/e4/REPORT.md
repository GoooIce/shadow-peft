# E4 — Qwen3.6 Bonsai matrix-family and layer-block localization

All intervention cases use Qwen/PTQ auxiliary parameters. Positive score means that the family either recovers the PTQ baseline when injected, or damages the official-main upper bound when left as PTQ.

## Anchors

| Case | Mean KL | Mean cosine | Top-10 overlap |
|---|---:|---:|---:|
| reference | 0.000000 | 1.000000 | 40/40 |
| ptq_baseline | 11.841274 | 0.505495 | 1/40 |
| official_main_qwen_aux | 0.258312 | 0.954451 | 30/40 |

## Phase 1 — family interventions

| Direction | Family | Matrices | Weight share | Mean KL | Score | Top-10 |
|---|---|---:|---:|---:|---:|---:|
| leaveout | broad.mlp | 192 | 63.63% | 12.127395 | +11.869083 | 8/40 |
| leaveout | atomic.mlp | 192 | 63.63% | 12.127395 | +11.869083 | 8/40 |
| leaveout | atomic.linear_attn.in_proj_z | 48 | 5.61% | 11.938423 | +11.680111 | 1/40 |
| leaveout | broad.linear_attn | 240 | 20.67% | 11.066676 | +10.808364 | 8/40 |
| leaveout | atomic.linear_attn.out_proj | 48 | 5.61% | 7.424497 | +7.166185 | 9/40 |
| inject | atomic.linear_attn.out_proj | 48 | 5.61% | 8.145343 | +3.695932 | 13/40 |
| inject | broad.self_attn | 64 | 6.24% | 8.923088 | +2.918186 | 7/40 |
| inject | atomic.self_attn | 64 | 6.24% | 8.923088 | +2.918186 | 7/40 |
| inject | broad.mlp | 192 | 63.63% | 9.279893 | +2.561382 | 1/40 |
| inject | atomic.mlp | 192 | 63.63% | 9.279893 | +2.561382 | 1/40 |
| inject | broad.linear_attn | 240 | 20.67% | 10.513006 | +1.328268 | 2/40 |
| leaveout | broad.self_attn | 64 | 6.24% | 1.336945 | +1.078633 | 21/40 |
| leaveout | atomic.self_attn | 64 | 6.24% | 1.336945 | +1.078633 | 21/40 |
| inject | atomic.linear_attn.in_proj_z | 48 | 5.61% | 11.104957 | +0.736317 | 1/40 |
| inject | atomic.lm_head | 1 | 4.73% | 11.383321 | +0.457953 | 1/40 |
| inject | broad.boundary | 2 | 9.46% | 11.397292 | +0.443982 | 1/40 |
| leaveout | atomic.linear_attn.in_proj_b | 48 | 0.04% | 0.288576 | +0.030263 | 29/40 |
| leaveout | atomic.lm_head | 1 | 4.73% | 0.284370 | +0.026057 | 28/40 |
| leaveout | broad.boundary | 2 | 9.46% | 0.281772 | +0.023460 | 29/40 |
| inject | atomic.linear_attn.in_proj_b | 48 | 0.04% | 11.822660 | +0.018615 | 1/40 |
| leaveout | atomic.linear_attn.in_proj_a | 48 | 0.04% | 0.269612 | +0.011300 | 30/40 |
| inject | atomic.linear_attn.in_proj_a | 48 | 0.04% | 11.832213 | +0.009062 | 1/40 |
| leaveout | atomic.embed_tokens | 1 | 4.73% | 0.257716 | -0.000596 | 31/40 |
| inject | atomic.embed_tokens | 1 | 4.73% | 11.850266 | -0.008991 | 1/40 |
| leaveout | atomic.linear_attn.in_proj_qkv | 48 | 9.36% | 0.155634 | -0.102679 | 21/40 |
| inject | atomic.linear_attn.in_proj_qkv | 48 | 9.36% | 12.640220 | -0.798945 | 1/40 |

## Phase 2 — `mlp` by 8-layer block

| Direction | Layers | Matrices | Weight share | Mean KL | Score | Top-10 |
|---|---:|---:|---:|---:|---:|---:|
| leaveout | 0-7 | 24 | 7.95% | 11.972483 | +11.714171 | 9/40 |
| inject | 56-63 | 24 | 7.95% | 9.359522 | +2.481753 | 4/40 |
| leaveout | 56-63 | 24 | 7.95% | 0.742836 | +0.484524 | 29/40 |
| inject | 48-55 | 24 | 7.95% | 11.364223 | +0.477051 | 1/40 |
| inject | 32-39 | 24 | 7.95% | 11.531670 | +0.309604 | 1/40 |
| leaveout | 32-39 | 24 | 7.95% | 0.538804 | +0.280491 | 24/40 |
| leaveout | 48-55 | 24 | 7.95% | 0.460740 | +0.202427 | 29/40 |
| inject | 24-31 | 24 | 7.95% | 11.675144 | +0.166131 | 1/40 |
| leaveout | 24-31 | 24 | 7.95% | 0.405092 | +0.146780 | 29/40 |
| inject | 8-15 | 24 | 7.95% | 11.722391 | +0.118883 | 1/40 |
| leaveout | 40-47 | 24 | 7.95% | 0.377117 | +0.118805 | 28/40 |
| leaveout | 16-23 | 24 | 7.95% | 0.357328 | +0.099016 | 31/40 |
| inject | 40-47 | 24 | 7.95% | 11.778954 | +0.062320 | 1/40 |
| inject | 16-23 | 24 | 7.95% | 11.804422 | +0.036852 | 1/40 |
| leaveout | 8-15 | 24 | 7.95% | 0.274537 | +0.016225 | 29/40 |
| inject | 0-7 | 24 | 7.95% | 11.861338 | -0.020063 | 1/40 |

## Phase 3 — `mlp` individual layers

| Direction | Layer | Matrices | Weight share | Mean KL | Score | Top-10 |
|---|---:|---:|---:|---:|---:|---:|
| inject | 63 | 3 | 0.99% | 9.847582 | +1.993693 | 3/40 |
| inject | 62 | 3 | 0.99% | 10.417590 | +1.423684 | 1/40 |
| inject | 58 | 3 | 0.99% | 10.992686 | +0.848588 | 1/40 |
| inject | 59 | 3 | 0.99% | 11.073553 | +0.767721 | 0/40 |
| leaveout | 0 | 3 | 0.99% | 0.880779 | +0.622466 | 31/40 |
| inject | 61 | 3 | 0.99% | 11.330761 | +0.510513 | 1/40 |
| leaveout | 1 | 3 | 0.99% | 0.604176 | +0.345864 | 27/40 |
| inject | 57 | 3 | 0.99% | 11.642075 | +0.199199 | 1/40 |
| leaveout | 2 | 3 | 0.99% | 0.366267 | +0.107954 | 30/40 |
| leaveout | 6 | 3 | 0.99% | 0.292471 | +0.034159 | 28/40 |
| leaveout | 3 | 3 | 0.99% | 0.292391 | +0.034079 | 29/40 |
| leaveout | 7 | 3 | 0.99% | 0.288125 | +0.029812 | 30/40 |
| leaveout | 5 | 3 | 0.99% | 0.261001 | +0.002688 | 29/40 |
| leaveout | 4 | 3 | 0.99% | 0.260993 | +0.002681 | 30/40 |
| inject | 56 | 3 | 0.99% | 12.129165 | -0.287890 | 1/40 |
| inject | 60 | 3 | 0.99% | 12.277139 | -0.435864 | 1/40 |

## Selected greedy generations

### `reference`

- fact: `'Paris'`
- math: `'391'`
- zh_math: `'54'`
- code: `'```python\ndef add(a, b):\n    return a + b\n'`

### `ptq_baseline`

- fact: `'fkk fkk fkk fkk fkk fkk fkk fkk fkk fkk fkk fkk fkk fkk fkk fkk'`
- math: `'fkk fkk fkk fkk fkk fkk fkk fkk fkk fkk fkk fkk fkk fkk fkk fkk'`
- zh_math: `'fkk fkk fkk fkk fkk fkk fkk fkk fkk fkk fkk fkk fkk fkk fkk fkk'`
- code: `'fkk fkk fkk fkk fkk fkk fkk fkk fkk fkk fkk fkk fkk fkk fkk fkk'`

### `official_main_qwen_aux`

- fact: `'Paris'`
- math: `'399'`
- zh_math: `'51'`
- code: `'```python\ndef add(a, b):\n    return a + b\n'`

### `inject.atomic.linear_attn.out_proj`

- fact: `''`
- math: `''`
- zh_math: `''`
- code: `''`

### `leaveout.broad.mlp`

- fact: `''`
- math: `''`
- zh_math: `''`
- code: `''`

### `inject.block_56_63.mlp`

- fact: `'........,,, the....'`
- math: `'................'`
- zh_math: `'.          ,    '`
- code: `'.......,,,.,....'`

### `leaveout.block_00_07.mlp`

- fact: `''`
- math: `''`
- zh_math: `''`
- code: `'<think>'`

### `inject.layer_63.mlp`

- fact: `',\n\n,,,,,,,,,,,,,,'`
- math: `',,,,,,,,,,,,,,,,'`
- zh_math: `'\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n'`
- code: `',,,,,,,,,,,,,,,,'`

### `leaveout.layer_00.mlp`

- fact: `'Paris'`
- math: `'17 * 17 = 17'`
- zh_math: `'54'`
- code: `'```python\n def add(a, b ) -> int:\n ```'`

## Result

- Strongest recovery intervention: `inject.atomic.linear_attn.out_proj` (score +3.695932).
- Strongest necessity intervention: `leaveout.broad.mlp` (score +11.869083).
- Selected layered family: `mlp`.
- Peak MLX memory: 14.419 GiB.

## Interpretation

- No single family or layer restores semantic generation. The failure is an interacting multi-family effect, not one poisoned matrix.
- `linear_attn.out_proj` is the strongest efficient recovery target: replacing 5.61% of original weights gives the largest family injection improvement.
- `linear_attn.in_proj_z` is strongly necessary but weak in isolation: leaving its 5.61% as PTQ nearly destroys the official main path, while injecting it alone barely repairs the PTQ path.
- MLP damage is depth-asymmetric. Layers 0-7 are the critical necessity block, while layers 56-63 provide nearly all MLP-side recovery from the PTQ baseline.
- The strongest individual recovery layer is `inject.layer_63.mlp`; the strongest individual necessity layer is `leaveout.layer_00.mlp`. Their effects remain much smaller than the corresponding 8-layer blocks, showing cross-layer interaction.
- Embedding and lm_head replacements have negligible effect relative to their size, so they are poor first QAT targets.
