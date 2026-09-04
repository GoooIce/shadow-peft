"""Speculative-decoding seam over AngelSpec.

AngelSpec trains its own drafters (Eagle3 / DFlash, online or offline) and
benchmarks them (``accept_length`` per verify cycle — the same τ the lab's
own specdec engine measures). Interface facts:

* The trainers own their fit loop (Eagle3Trainer-style ``.train()``); this
  seam only resolves and constructs them. Trainer kwargs are AngelSpec's
  contract, not ours — they are passed through untouched and may drift with
  the angelslim version (the reason this wrapper stays minimal).
* τ benchmarking uses the **pytorch backend** — no vLLM install needed.
  The vLLM benchmark backend and sglang serving stay deployment-side: a
  trained drafter checkpoint is a standard HF-style draft model either way.
"""

from __future__ import annotations

SPECDEC_METHODS = ("eagle3", "dflash")
SPECDEC_MODES = ("online", "offline")


def build_drafter_trainer(method: str, mode: str, **trainer_kwargs):
    """Resolve and construct an AngelSpec drafter trainer for (method, mode).

    ``trainer_kwargs`` are the chosen trainer's own constructor arguments —
    target model path, draft model path, data, training hyperparameters —
    and are forwarded untouched.
    """
    if method not in SPECDEC_METHODS:
        raise ValueError(
            f"unsupported drafter method {method!r} (expected one of {SPECDEC_METHODS})"
        )
    if mode not in SPECDEC_MODES:
        raise ValueError(f"unsupported drafter mode {mode!r} (expected one of {SPECDEC_MODES})")
    if method == "eagle3":
        if mode == "online":
            from angelslim.compressor.speculative.train.trainer.trainer_factory import (
                Eagle3TrainerFactory,
            )

            return Eagle3TrainerFactory.create(
                training_mode="online", modal_type="LLM", **trainer_kwargs
            )
        from angelslim.compressor.speculative.train.trainer.offline_eagle3_trainer import (
            OfflineEagle3Trainer,
        )

        return OfflineEagle3Trainer(**trainer_kwargs)

    from angelslim.compressor.speculative.train.trainer import (
        offline_dflash_trainer,
        online_dflash_trainer,
    )

    trainer_cls = (
        online_dflash_trainer.OnlineDFlashTrainer
        if mode == "online"
        else offline_dflash_trainer.OfflineDFlashTrainer
    )
    return trainer_cls(**trainer_kwargs)


def benchmark_engine():
    """Return AngelSpec's pytorch benchmark companion classes.

    Returns ``(BenchmarkConfig, BenchmarkEngine)`` — construct the config,
    then the engine, and read ``accept_length`` from its results for the τ
    axis. Pytorch backend only: the vLLM benchmark path is a deployment-side
    concern.
    """
    from angelslim.compressor.speculative.benchmark.pytorch.benchmark_engine import (
        BenchmarkConfig,
        BenchmarkEngine,
    )

    return BenchmarkConfig, BenchmarkEngine
