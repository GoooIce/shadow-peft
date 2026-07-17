"""CLI: convert a ShadowPEFT adapter checkpoint between torch and MLX formats.

用法:
    .venv/bin/python experiment/convert_adapter.py TORCH_CKPT MLX_CKPT --direction torch_to_mlx
    .venv/bin/python experiment/convert_adapter.py MLX_CKPT TORCH_CKPT --direction mlx_to_torch

注意: adapter checkpoint 不含 base 模型权重——目标侧需要准备同一个 base 模型。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shadow_peft_mlx.convert import convert_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("src", help="源 checkpoint 目录（含 shadow_config.json）")
    parser.add_argument("dst", help="目标 checkpoint 目录")
    parser.add_argument(
        "--direction",
        choices=["torch_to_mlx", "mlx_to_torch"],
        required=True,
        help="转换方向",
    )
    args = parser.parse_args()
    convert_checkpoint(args.src, args.dst, direction=args.direction)
    print(f"Converted {args.src} -> {args.dst} ({args.direction})")


if __name__ == "__main__":
    main()
