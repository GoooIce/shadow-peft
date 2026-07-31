# E5 — one-stage Shadow recovery training

## Aggregate

| Metric | Before | After | Change |
|---|---:|---:|---:|
| Train teacher-forced CE | 9.772161 | 4.349142 | -5.423019 |
| Held-out teacher-forced CE | 10.051911 | 4.923509 | -5.128402 |
| Mean KL to official | 11.842596 | 5.938354 | -5.904242 |
| Mean logit cosine | 0.505046 | 0.781558 | +0.276512 |
| Top-10 overlap | 1/40 | 17/40 |  |

## Held-out generations

| Prompt | Before | After | Official teacher |
|---|---|---|---|
| fact |  fkk fkk fkk fkk fkk fkk fkk fkk fkk fkk fkk fkk |  | Paris |
| math |  fkk fkk fkk fkk fkk fkk fkk fkk fkk fkk fkk fkk | 0 | 391 |
| zh_math |  fkk fkk fkk fkk fkk fkk fkk fkk fkk fkk fkk fkk | 0 | 54 |
| code |  fkk fkk fkk fkk fkk fkk fkk fkk fkk fkk fkk fkk |  | ```python\ndef add(a, b):\n    return a + b\n |

## Protocol

- Steps: 36; learning rate: 1.00e-04.
- Trainable Shadow parameters: 131,507,424.
- Frozen packed base unchanged: True.
- Selected checkpoint step: 24.
- Training time: 89.1s.
- Peak MLX memory: 25.027 GiB.

## Verdict

- Shadow recovery is meaningful: the held-out loss and first-token distribution both improve substantially while every packed base weight remains frozen.
- This short sequence-level distillation stage does not yet recover autoregressive semantics. Generation changes from the original `fkk` loop to early EOS or `0`, so the run is a positive gradient signal rather than a successful model checkpoint.
- Step 24 beats step 36 on held-out CE, confirming that the 12-example teacher set is already overfitting. The next run needs a larger, balanced teacher corpus and token-level KL, not simply more epochs on these examples.
