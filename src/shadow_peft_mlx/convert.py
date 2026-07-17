"""
Convert ShadowPEFT adapter checkpoints between the torch and MLX formats.

Both implementations store the same artifacts (`shadow_config.json` +
`shadow_adapter.safetensors`, optionally `shadow_modules.safetensors`) with identical
tensor shapes; the only differences are:

- the update MLPs: torch `nn.Sequential` keys look like
  `shadow_update_model.update_gates.0.0.weight`, while mlx `nn.Sequential` stores its
  children under `layers`: `shadow_update_model.update_gates.0.layers.0.weight`
  (the numeric indices line up — dropout/activation modules occupy indices in both).
- everything else (shadow backbone, injection tensors, hidden_norm, projection, heads)
  uses identical keys.

This enables "train on a CUDA box with torch, deploy the adapter on Apple Silicon with
MLX" (or the reverse) without retraining. Note the base model weights are NOT part of
the adapter checkpoint — you need the same base model on both sides.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Literal

import mlx.core as mx
import numpy as np

_ADAPTER_FILE = "shadow_adapter.safetensors"
_MODULES_FILE = "shadow_modules.safetensors"
_CONFIG_FILE = "shadow_config.json"

# shadow_update_model.{update_gates|update_transforms}.<layer>.<seq_idx>.<param>
_TORCH_UPDATE_RE = re.compile(
    r"^(shadow_update_model\.(?:update_gates|update_transforms)\.\d+)\.(\d+\..+)$"
)
_MLX_UPDATE_RE = re.compile(
    r"^(shadow_update_model\.(?:update_gates|update_transforms)\.\d+)\.layers\.(\d+\..+)$"
)

Direction = Literal["torch_to_mlx", "mlx_to_torch"]


def torch_key_to_mlx(key: str) -> str:
    m = _TORCH_UPDATE_RE.match(key)
    return f"{m.group(1)}.layers.{m.group(2)}" if m else key


def mlx_key_to_torch(key: str) -> str:
    m = _MLX_UPDATE_RE.match(key)
    return f"{m.group(1)}.{m.group(2)}" if m else key


def _to_float32_numpy(t) -> np.ndarray:
    # numpy has no bf16; upcast half-precision dtypes for a lossless-enough transfer.
    if str(t.dtype) in ("torch.bfloat16", "bfloat16"):
        return np.asarray(t.to("float32") if hasattr(t, "to") else t.astype(mx.float32))
    return np.asarray(t)


def _convert_file(src: Path, dst: Path, direction: Direction) -> None:
    if direction == "torch_to_mlx":
        from safetensors.torch import load_file  # lazy: only needed on this path

        state = load_file(str(src))
        mx_state = {torch_key_to_mlx(k): mx.array(_to_float32_numpy(v)) for k, v in state.items()}
        mx.save_safetensors(str(dst), mx_state)
    else:
        import torch  # lazy: only needed on this path
        from safetensors.torch import save_file

        state = mx.load(str(src))
        torch_state = {
            mlx_key_to_torch(k): torch.from_numpy(np.asarray(v.astype(mx.float32)))
            for k, v in state.items()
        }
        save_file(torch_state, str(dst))


def convert_checkpoint(
    src_dir: str | Path,
    dst_dir: str | Path,
    *,
    direction: Direction,
) -> None:
    """
    Convert a ShadowPEFT checkpoint directory between frameworks.

    Copies `shadow_config.json` verbatim (the schema is shared) and converts
    `shadow_adapter.safetensors` (plus `shadow_modules.safetensors` when present).
    """
    src, dst = Path(src_dir), Path(dst_dir)
    if direction not in ("torch_to_mlx", "mlx_to_torch"):
        raise ValueError(f"Unknown direction: {direction}")
    if not (src / _CONFIG_FILE).exists():
        raise FileNotFoundError(f"Missing {_CONFIG_FILE} in {src}")
    if not (src / _ADAPTER_FILE).exists():
        raise FileNotFoundError(f"Missing {_ADAPTER_FILE} in {src}")

    dst.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src / _CONFIG_FILE, dst / _CONFIG_FILE)
    _convert_file(src / _ADAPTER_FILE, dst / _ADAPTER_FILE, direction)
    if (src / _MODULES_FILE).exists():
        _convert_file(src / _MODULES_FILE, dst / _MODULES_FILE, direction)
