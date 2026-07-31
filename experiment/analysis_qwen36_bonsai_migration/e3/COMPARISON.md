# E3 — v2 8-LS versus v3 one-shot

## Offline reproduction

| Candidate | Code agreement | Zero rate |
|---|---:|---:|
| Official | 100.00% | 29.70% |
| v2 8-LS | 81.43% | 45.96% |
| v3 one-shot | 87.81% | 31.15% |

## End-to-end first-token comparison

| Aux parameters | Candidate | Mean KL | Mean cosine | Top-10 overlap |
|---|---|---:|---:|---:|
| Official | v2 8-LS | 9.935835 | 0.328523 | 0/40 |
| Official | v3 one-shot | 11.439769 | 0.513994 | 1/40 |
| Qwen | v2 8-LS | 12.173435 | 0.463785 | 1/40 |
| Qwen | v3 one-shot | 11.841274 | 0.505495 | 1/40 |

## Verdict

One-shot `absmean + 0.5` is a better container-level reproduction, but it does not restore semantic generation. With Qwen auxiliary parameters it changes mean KL from 12.173 to 11.841 and cosine from 0.464 to 0.505; all four generations remain broken.

The next experiment must target training-aware main-matrix migration rather than another global threshold sweep.
