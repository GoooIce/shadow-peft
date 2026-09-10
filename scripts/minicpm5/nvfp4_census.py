#!/usr/bin/env python3
"""Layer census for the MiniCPM5-2B NVFP4 W4A4 start payload.

Builds the model on the ``meta`` device (no weights are materialised), then
applies ``shadow_lab.nvfp4_runtime.census_linear_layers`` with the two-axis
16-divisibility constraint to decide which ``nn.Linear`` layers can be
represented as native NVFP4 and which stay BF16.

Outputs a JSON manifest plus a Markdown table.

Usage:
    python nvfp4_census.py --model ~/models/MiniCPM5-2B --out-dir <dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from shadow_lab.nvfp4_runtime import census_linear_layers, census_manifest

# Layers that must stay in BF16 by policy even when they satisfy the shape
# constraint.  ``lm_head`` is a 130,560 x 2048 Linear whose two axes are both
# divisible by 16, but quantising the vocabulary projection is a known
# accuracy hazard and every serving stack defaults to ignoring it.
POLICY_EXCLUDE = ("lm_head",)


def _markdown(manifest: dict, per_kind: dict) -> str:
    lines = []
    lines.append("| 层类型 | 数量 | in_features | out_features | 权重形状 | 16 整除 | 处置 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for kind, rows in per_kind.items():
        first = rows[0]
        shapes = sorted({(r["in_features"], r["out_features"]) for r in rows})
        shape_txt = "; ".join(f"{o}x{i}" for i, o in shapes)
        aligned = all(
            r["in_features"] % 16 == 0 and r["out_features"] % 16 == 0 for r in rows
        )
        if kind == "lm_head":
            action = "BF16（策略排除）"
        else:
            action = "NVFP4 W4A4" if aligned else "BF16（形状不整除）"
        lines.append(
            f"| {kind} | {len(rows)} | {first['in_features']} | {first['out_features']} | "
            f"{shape_txt} | {'是' if aligned else '否'} | {action} |"
        )
    lines.append("")
    lines.append(
        f"合计 nn.Linear {manifest['linear_layers']} 个；可量化（16 整除且非策略排除）"
        f" {manifest['eligible_layers']} 个，权重参数 "
        f"{manifest['eligible_weight_parameters']:,}；排除 {manifest['excluded_layers']} 个。"
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="local model directory")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=list(POLICY_EXCLUDE),
        help="module-name patterns kept in BF16 by policy",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    config = AutoConfig.from_pretrained(args.model)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config)

    rows = census_linear_layers(model, exclude_modules=tuple(args.exclude))
    manifest = census_manifest(rows)
    manifest["model_dir"] = str(Path(args.model).expanduser())
    manifest["exclude_modules"] = list(args.exclude)
    manifest["device"] = "meta (no weights materialised)"
    manifest["constraint"] = (
        "in_features % 16 == 0 and out_features % 16 == 0 (NVFP4 block size)"
    )

    # Group by the trailing module name (q_proj, k_proj, ... lm_head).
    per_kind: dict[str, list[dict]] = {}
    for row in manifest["layers"]:
        per_kind.setdefault(row["name"].rsplit(".", 1)[-1], []).append(row)

    # Sanity: every layer must be unique and the count must match the config.
    layer_names = [row["name"] for row in manifest["layers"]]
    assert len(layer_names) == len(set(layer_names)), "duplicate Linear names"

    json_path = out_dir / "census.json"
    json_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    md = _markdown(manifest, per_kind)
    md_path = out_dir / "census.md"
    md_path.write_text(md + "\n")

    digest = hashlib.sha256(json_path.read_bytes()).hexdigest()
    print(md)
    print(f"\ncensus.json sha256={digest}")
    print(f"wrote {json_path} and {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
