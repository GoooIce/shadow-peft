# E1 — Qwen3.6 → official Bonsai g128 rule inversion

## Global result

- Matrices: 498
- Groups: 210,104,320; sampled: 509,952
- Official zero rate: 29.701407%
- Official nonzero sign preservation: 98.707235%
- Report surrogate code agreement: 81.434530%
- One-shot absmean/threshold code agreement: 87.809749%
- One-shot zero rate: 31.146488%
- Best global magnitude threshold: 0.5000 (accuracy 87.809749%)
- Best per-layer threshold accuracy: 87.845809%
- Best per-group threshold code upper bound: 89.504180%
- Best per-group zero-mask upper bound: 89.630933%
- Same-sparsity magnitude-rank mask accuracy: 87.868093%
- Official vs Qwen cosine: 0.862650
- Official vs Qwen NRMSE: 0.531909
- Fixed official codes + Qwen LS scale NRMSE: 0.503744
- Official scale / base absmean (group mean): 1.051980
- Official scale vs base absmean correlation: 0.993437
- Official scale / Qwen LS scale (group mean): 0.818422

## Interpretation

- The current 8-iteration LS surrogate over-prunes: its zero rate is 45.96%, versus 29.70% in the official checkpoint.
- A single absmean/0.5 pass is a substantially better reproduction (87.81% code agreement), and the histogram-optimal global threshold is exactly 0.5.
- Even an oracle threshold chosen independently for every g128 group tops out at 89.50%. The remaining gap therefore cannot be explained by a different magnitude threshold alone.
- Official scales track the original Qwen group absmean very closely; re-fitting scales by LS against the official codes moves away from the official scale distribution.

## Worst report-surrogate layer agreement

| Layer | Agreement | Official zero | Sign preservation | Group-threshold upper bound |
|---|---:|---:|---:|---:|
| `language_model.model.layers.0.linear_attn.in_proj_b` | 57.89% | 29.55% | 80.92% | 62.83% |
| `language_model.model.layers.62.linear_attn.in_proj_b` | 64.10% | 26.40% | 88.43% | 72.37% |
| `language_model.model.layers.54.linear_attn.in_proj_b` | 65.37% | 28.12% | 88.28% | 71.03% |
| `language_model.model.layers.52.linear_attn.in_proj_b` | 65.62% | 28.43% | 88.92% | 71.14% |
| `language_model.model.layers.60.linear_attn.in_proj_b` | 66.57% | 28.82% | 90.65% | 74.04% |
| `language_model.model.layers.41.linear_attn.in_proj_b` | 66.60% | 30.73% | 90.22% | 72.14% |
| `language_model.model.layers.42.linear_attn.in_proj_b` | 66.65% | 30.74% | 89.88% | 71.94% |
| `language_model.model.layers.48.linear_attn.in_proj_b` | 66.74% | 31.06% | 90.37% | 71.81% |
| `language_model.model.layers.32.linear_attn.in_proj_b` | 67.07% | 33.78% | 91.38% | 72.51% |
| `language_model.model.layers.36.linear_attn.in_proj_b` | 67.25% | 30.95% | 90.54% | 72.57% |
| `language_model.model.layers.44.linear_attn.in_proj_b` | 67.28% | 31.57% | 90.74% | 72.60% |
| `language_model.model.layers.40.linear_attn.in_proj_b` | 67.40% | 30.50% | 91.06% | 73.37% |
| `language_model.model.layers.61.linear_attn.in_proj_b` | 67.49% | 26.83% | 90.68% | 75.55% |
| `language_model.model.layers.50.linear_attn.in_proj_b` | 67.50% | 30.11% | 90.62% | 72.62% |
| `language_model.model.layers.28.linear_attn.in_proj_b` | 67.52% | 31.94% | 91.42% | 73.54% |
