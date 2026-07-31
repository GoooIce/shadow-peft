# E5-R1 — Tulu token-level Shadow recovery

## Aggregate

| Metric | Before | After | Change |
|---|---:|---:|---:|
| Train top-k+tail KL | 7.364436 | 4.087543 | -3.276893 |
| Held-out top-k+tail KL | 7.211718 | 3.750994 | -3.460724 |
| Probe full-vocab KL | 11.842596 | 8.870928 | -2.971668 |
| Probe logit cosine | 0.505046 | 0.732729 | +0.227683 |
| Probe top-10 overlap | 1/40 | 3/40 | |

## Protocol

- Tulu samples: 64 train / 16 held out.
- Cached teacher tokens: 7431 train / 1807 held out.
- Windowed KL coverage: 1984 train / 496 held out; optimization consumed 496 token positions.
- Teacher approximation: exact top-32 tokens plus one aggregated tail bucket at every position.
- Steps: 16; learning rate: 5.00e-05; selected checkpoint: 16.
- Training/evaluation token window: 32.
- Trainable Shadow parameters: 131,507,424.
- Frozen packed base unchanged: True.
- Training time: 90.1s; peak MLX memory: 20.555 GiB.

## Held-out probe generations

| Prompt | Before | After |
|---|---|---|
| fact |  fkk fkk fkk fkk fkk fkk fkk fkk fkk fkk fkk fkk |  |
| math |  fkk fkk fkk fkk fkk fkk fkk fkk fkk fkk fkk fkk |  |
| zh_math |  fkk fkk fkk fkk fkk fkk fkk fkk fkk fkk fkk fkk |  |
| code |  fkk fkk fkk fkk fkk fkk fkk fkk fkk fkk fkk fkk |  |
