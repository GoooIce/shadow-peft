# QUASAR NVFP4 QAD

首期范围只有 NVFP4 W4A4，不包含论文的 INT2/INT3/INT4 路径，也不复用仓库已有的
1-bit/2-bit codec。

## 训练语义

`shadow_lab.quasar` 在每个目标 `Linear` 上保留 BF16/FP32 latent 权重。前向时，它从
AdamW 的 `exp_avg_sq` 直接读取显著性（优化器首步尚无状态时使用均匀显著性），在论文的
`0.30, 0.35, ..., 1.00` clipping ranges 中搜索 E2M1 code assignment，并对每个 assignment
求解显著性加权最小二乘 scale。拟合结果量化到 E4M3，配合每个 tensor 的 FP32 absmax
scale，构成 block-16 的 NVFP4 两级缩放。权重和 activation fake quant 都使用 STE。

QAD 与 shadow 训练是两个独立策略：QAD 冻结 embedding、norm、bias、`lm_head` 等非量化
模块，只训练 transformer Linear 的 latent 权重；`train_shadow` 的冻结基座约束不变。

## 运行

输入沿用实验框架的 teacher-token JSONL，每行一个 `{"ids": [...]}`：

```bash
.venv/bin/python scripts/quasar_qad_arm.py \
  --model Qwen/Qwen3-8B \
  --data experiment/data/teacher_qwen3_8b.jsonl \
  --out runs/quasar/qwen3_8b_nvfp4 \
  --steps 4096 --seq-len 4096 --batch-seq 1 --grad-accum 32 \
  --lr 1e-6 --gradient-checkpointing
```

训练目标是 `KL(P_teacher || P_student)`。结束时会写出：

- compressed-tensors `nvfp4-pack-quantized` checkpoint；
- `quasar_manifest.json`（参数、层清单和格式）；
- `train_log.jsonl`（forward KL、token 数、吞吐和 saliency 覆盖）。

Python 训练循环也可只使用接缝：

```python
import torch

from shadow_lab import QuasarNVFP4Config, prepare_quasar_nvfp4

controller = prepare_quasar_nvfp4(model, QuasarNVFP4Config())
optimizer = torch.optim.AdamW(controller.parameters(), lr=1e-6)
controller.bind_optimizer(optimizer)
```

绑定后每次 `optimizer.step()` 都会让后续前向读取最新的 AdamW 二阶矩；不复制额外的
weight-sized saliency buffer。
