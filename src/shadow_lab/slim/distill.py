"""Full-precision distillation arm (AngelSlim ``Distill``).

fp students only (AngelSlim keeps the quantized-student path in QAD). In the
shadow-peft lab this is the generic KD reference: an fp student distilled
from the same teacher, for upper-bound rows next to the recovery arms.
"""

from __future__ import annotations

from . import engine as _engine

_DEFAULT_HF_ARGS = {
    "bf16": True,
    "logging_steps": 1,
    "per_device_train_batch_size": 1,
    "learning_rate": 1e-5,
    "lr_scheduler_type": "cosine",
    "num_train_epochs": 1,
    "save_strategy": "no",
}


def run_distill(
    model_path: str,
    teacher_path: str,
    data_path: str,
    out_dir: str,
    *,
    model_name: str | None = None,
    hf_args: dict | None = None,
    distill_overrides: dict | None = None,
    max_seq_length: int = 2048,
    num_samples: int = 128,
    batch_size: int = 1,
) -> dict:
    """Train an fp student from ``teacher_path``; returns the run manifest."""
    from angelslim.utils.config_parser import CompressionConfig, DistillTrainingConfig

    cfg = DistillTrainingConfig(
        teacher_model_path=teacher_path,
        student_type="fp",
        trainable_parameters="all",
        hf_args=hf_args or dict(_DEFAULT_HF_ARGS),
    )
    for key, value in (distill_overrides or {}).items():
        setattr(cfg, key, value)

    compress_config = CompressionConfig(name="Distill", Distill=cfg)
    manifest = _engine.drive_campaign(
        "Distill",
        model_path,
        data_path,
        out_dir,
        compress_config,
        model_name=model_name,
        max_seq_length=max_seq_length,
        num_samples=num_samples,
        batch_size=batch_size,
    )
    manifest["tier"] = "fp"
    return manifest
