"""QAD (quantization-aware distillation) arm — the no-shadow recovery baseline.

AngelSlim's QAD is an HF-Trainer pipeline that trains a quantized student
against a full-precision teacher. It has no concept of shadow adapters, so
in the shadow-peft recovery coordinate system it is the **comparison arm**:
same backbone tier, recovery without a shadow. Two ways to line the grids
up:

* ``quant="w4a8_fp8"`` (the only officially-supported QAD student format):
  LSQ-style learnable scales, its own grid.
* ``from_ptq_ckpt=<dir>``: start from a checkpoint produced by
  ``slim.ptq.convert_backbone(..., fake_quant=False, save_dir=...)`` — the
  student then carries the exact GPTQ grid the shadow arms train against,
  which is the directly comparable protocol.

Recovering a backbone this way never touches the shadow training protocol
(``trainer.train_shadow``); the two curves land in the same recovery-rate
plot.
"""

from __future__ import annotations

from . import engine as _engine
from .ptq import _tier_builder

# QAD students AngelSlim officially supports (its plugin/QAT machinery);
# the from_ptq_ckpt path is format-agnostic because the grid comes from the
# checkpoint's own quantization_config.
QAD_QUANT_TIERS = ("w4a8_fp8",)

_DEFAULT_HF_ARGS = {
    "bf16": True,
    "logging_steps": 1,
    "per_device_train_batch_size": 1,
    "learning_rate": 1e-5,
    "lr_scheduler_type": "cosine",
    "num_train_epochs": 1,
    "save_strategy": "no",
}


def run_qad(
    model_path: str,
    teacher_path: str,
    data_path: str,
    out_dir: str,
    *,
    quant: str = "w4a8_fp8",
    from_ptq_ckpt: str | None = None,
    model_name: str | None = None,
    hf_args: dict | None = None,
    qad_overrides: dict | None = None,
    max_seq_length: int = 2048,
    num_samples: int = 128,
    batch_size: int = 1,
) -> dict:
    """Train a quantized student from ``teacher_path``; returns the run manifest.

    ``hf_args`` are HF TrainingArguments kwargs (lr, epochs, deepspeed...);
    ``qad_overrides`` set any other ``QADTrainingConfig`` field (loss_type,
    plugin_config, trainable_parameters...).
    """
    if from_ptq_ckpt is None and quant not in QAD_QUANT_TIERS:
        raise ValueError(
            f"unsupported QAD student tier {quant!r} (expected one of "
            f"{QAD_QUANT_TIERS}, or pass from_ptq_ckpt= to inherit a GPTQ grid)"
        )
    from angelslim.utils.config_parser import CompressionConfig, QADTrainingConfig

    quantization = None
    if from_ptq_ckpt is None:
        quantization = _tier_builder(quant)()["compress_config"].quantization

    qad_cfg = QADTrainingConfig(
        teacher_model_path=teacher_path,
        student_type="quantized",
        trainable_parameters="quant",
        from_ptq_ckpt=from_ptq_ckpt,
        hf_args=hf_args or dict(_DEFAULT_HF_ARGS),
    )
    for key, value in (qad_overrides or {}).items():
        setattr(qad_cfg, key, value)

    compress_config = CompressionConfig(name="QAD", quantization=quantization, QAD=qad_cfg)
    manifest = _engine.drive_campaign(
        "QAD",
        model_path,
        data_path,
        out_dir,
        compress_config,
        model_name=model_name,
        max_seq_length=max_seq_length,
        num_samples=num_samples,
        batch_size=batch_size,
    )
    manifest["tier"] = quant if from_ptq_ckpt is None else f"ptq_ckpt:{from_ptq_ckpt}"
    return manifest
