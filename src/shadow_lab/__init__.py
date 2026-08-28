"""Shadow-PEFT experiment lab: deep modules behind thin CLI scripts."""

from .atoms import (
    QWEN3_ENDOFTEXT_ID,
    eval_ppl,
    load_teacher_data,
    load_wikitext2_test_text,
    make_batch,
    pick_device,
)
from .trainer import LOSS_MODES, TrainConfig, TrainReport, train_shadow

__all__ = [
    "LOSS_MODES",
    "QWEN3_ENDOFTEXT_ID",
    "TrainConfig",
    "TrainReport",
    "eval_ppl",
    "load_teacher_data",
    "load_wikitext2_test_text",
    "make_batch",
    "pick_device",
    "train_shadow",
]
