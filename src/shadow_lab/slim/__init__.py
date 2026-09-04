"""AngelSlim compression-engine bridge — deep wrappers over the five compressors.

Public surface (one entry per AngelSlim capability; everything
Engine/SlimModelFactory/config-shaped stays inside):

* ``convert_backbone`` — in-memory backbone PTQ (the trainer's compression
  backend; calibration-based GPTQ/AWQ/nvfp4 tiers).
* ``run_qad`` / ``run_distill`` — path-based training campaigns (recovery
  baselines; QAD is the no-shadow comparison arm).
* ``apply_sparse`` — sparse-attention patching.
* ``build_drafter_trainer`` / ``benchmark_engine`` — AngelSpec drafter
  training and pytorch τ benchmarking.

Replacement plan (step 1, coexistence): the torchao backend
(``shadow_lab.tiers``) and the lab's own speculative-decoding prototypes
stay untouched while the comparison arms run; step 2 deletes them.
"""

from .distill import run_distill
from .ptq import ANGELSLIM_TIERS, FAKE_QUANT_TIERS, convert_backbone
from .qad import QAD_QUANT_TIERS, run_qad
from .sparsity import SPARSE_ALGORITHMS, apply_sparse
from .specdec import (
    SPECDEC_METHODS,
    SPECDEC_MODES,
    benchmark_engine,
    build_drafter_trainer,
)

__all__ = [
    "ANGELSLIM_TIERS",
    "FAKE_QUANT_TIERS",
    "QAD_QUANT_TIERS",
    "SPARSE_ALGORITHMS",
    "SPECDEC_METHODS",
    "SPECDEC_MODES",
    "apply_sparse",
    "benchmark_engine",
    "build_drafter_trainer",
    "convert_backbone",
    "run_distill",
    "run_qad",
]
