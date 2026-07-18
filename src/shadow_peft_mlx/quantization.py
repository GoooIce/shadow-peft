"""1-bit affine quantization entry points for the MLX backend.

``bits=1`` support is not yet in a PyPI release of MLX; it lives in
ml-explore/mlx PR #3161 ("Add 1-bit affine quantization support (Metal)").
Build MLX from that PR (or any branch containing it) before calling
:func:`quantize_model_1bit`::

    git clone https://github.com/ml-explore/mlx.git
    cd mlx && git fetch origin refs/pull/3161/head && git checkout FETCH_HEAD
    pip install -e .

Layout produced here is the standard MLX quantized format: packed ``uint32``
weights (32 values per word, LSB-first), one ``scale``/``bias`` pair per
``group_size`` block along the last dim, ``w_hat = scale * q + bias``.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

_1BIT_PR_URL = "https://github.com/ml-explore/mlx/pull/3161"


def ensure_1bit_support() -> None:
    """Raise a RuntimeError unless the installed MLX supports ``bits=1``."""
    probe = mx.zeros((1, 128), dtype=mx.float16)
    try:
        mx.quantize(probe, group_size=128, bits=1)
    except ValueError as exc:
        raise RuntimeError(
            "The installed MLX does not support bits=1 affine quantization. "
            f"Build MLX from {_1BIT_PR_URL} (or any branch containing it) and "
            "re-install it into this environment."
        ) from exc


def quantize_model_1bit(
    model: nn.Module,
    group_size: int = 128,
    skip_patterns: tuple[str, ...] = (),
) -> dict:
    """
    Quantize every eligible ``nn.Linear`` / ``nn.Embedding`` of ``model`` to
    1-bit affine, in place.

    Args:
        model: The model to quantize.
        group_size: Weights per affine group along the last dim. Layers whose
            last weight dim is not divisible by it are left untouched.
        skip_patterns: Fully-qualified module names containing any of these
            substrings are left untouched.

    Returns:
        A manifest describing what was quantized::

            {
              "quantization": {"mode": "affine", "bits": 1, "group_size": ...},
              "quantized": [module paths ...],
              "skipped": {module path: reason},
              "tied": [module paths sharing an already-quantized weight],
            }
    """
    ensure_1bit_support()

    manifest: dict = {
        "quantization": {"mode": "affine", "bits": 1, "group_size": group_size},
        "quantized": [],
        "skipped": {},
        "tied": [],
    }
    seen_weight_ids: set[int] = set()

    def predicate(path: str, module: nn.Module) -> bool:
        if not isinstance(module, (nn.Linear, nn.Embedding)):
            return False
        if any(pattern in path for pattern in skip_patterns):
            manifest["skipped"][path] = "skip_patterns"
            return False
        if id(module.weight) in seen_weight_ids:
            # e.g. an lm_head sharing its weight with the token embedding.
            manifest["tied"].append(path)
            return False
        if module.weight.shape[-1] % group_size != 0:
            manifest["skipped"][path] = (
                f"last dim {module.weight.shape[-1]} not divisible by {group_size}"
            )
            return False
        seen_weight_ids.add(id(module.weight))
        manifest["quantized"].append(path)
        return True

    nn.quantize(model, group_size=group_size, bits=1, class_predicate=predicate)
    return manifest
