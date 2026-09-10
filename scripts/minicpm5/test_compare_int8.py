"""INT8 calibre for compare_kernel_values (hard gate, W8A8 line).  TDD first.

The frozen gate reads ``pass_value``; on the uniform int8 grid the value
calibre collapses to exact signed-code equality (no 0<->8 channel, no
canonical folding), so::

    raw_code_diffs == value_diffs == canonical_code_diffs == byte diffs
    zero8_diffs == 0 by definition

Usage:
  cd $(git rev-parse --show-toplevel)/scripts/minicpm5 && \
      python test_compare_int8.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from safetensors.torch import save_file

HERE = os.path.dirname(os.path.abspath(__file__))


def check(name, cond, detail=""):
    if not cond:
        raise AssertionError(f"{name}: {detail}")
    print(f"  ok  {name}")


def run_compare(left: Path, right: Path, out: Path) -> dict:
    script = str(Path(HERE) / "compare_kernel_values.py")
    subprocess.run([sys.executable, script, "--left", str(left),
                    "--right", str(right), "--out", str(out)],
                   check=True, capture_output=True, text=True)
    return json.loads(out.read_text())


def test_int8_weight_calibre():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        g = torch.Generator().manual_seed(20260910)
        w = torch.randint(-127, 128, (64, 96), generator=g).to(torch.int8)
        scale = torch.rand(96, 1, generator=g).float()
        save_file({"m.weight": w.clone(), "m.weight_scale": scale.clone()},
                  str(tmp / "left.safetensors"))
        # perturb exactly 3 codes on the right side
        def bump(tensor, index, delta):
            flat = tensor.reshape(-1)
            value = int(flat[index])
            assert -127 <= value + delta <= 127, "pick a non-clamping coordinate"
            flat[index] = value + delta

        w2 = w.clone()
        bump(w2, 7, 3)
        bump(w2, 1000, -5)
        bump(w2, 6143, 1)
        save_file({"m.weight": w2, "m.weight_scale": scale.clone()},
                  str(tmp / "right.safetensors"))
        report = run_compare(tmp / "left.safetensors", tmp / "right.safetensors",
                             tmp / "cmp.json")
        row = next(r for r in report["rows"] if r["key"] == "m.weight")
        check("int8: raw == value == canonical == 3",
              row["raw_code_diffs"] == 3 and row["value_diffs"] == 3
              and row["canonical_code_diffs"] == 3, row)
        check("int8: zero8 channel is empty by construction",
              row["zero8_diffs"] == 0 and report["zero8_diffs"] == 0, row)
        check("int8: gate fails on value diffs", report["pass_value"] is False)
        check("int8: diff examples carry signed codes",
              len(row.get("diff_examples") or []) == 3)

        # identical files pass every calibre
        save_file({"m.weight": w.clone(), "m.weight_scale": scale.clone()},
                  str(tmp / "right2.safetensors"))
        report = run_compare(tmp / "left.safetensors", tmp / "right2.safetensors",
                             tmp / "cmp2.json")
        check("int8: identical dumps pass the frozen gate",
              report["pass_value"] is True and report["value_diffs"] == 0)


def test_int8_scale_still_bitwise():
    """fp32 weight_scale keeps the float-side bitwise calibre."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        w = torch.zeros(16, 32, dtype=torch.int8)
        s1 = torch.full((32, 1), 0.5)
        s2 = s1.clone(); s2[5, 0] = 0.5000001
        save_file({"m.weight": w, "m.weight_scale": s1}, str(tmp / "l.st"))
        save_file({"m.weight": w, "m.weight_scale": s2}, str(tmp / "r.st"))
        report = run_compare(tmp / "l.st", tmp / "r.st", tmp / "c.json")
        check("scale diff is a value diff on the float side",
              report["value_diffs"] >= 1 and report["pass_value"] is False)


def test_nvfp4_packed_path_unchanged():
    """Regression: the packed-uint8 nibble path still reports 0<->8 apart."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        base = torch.zeros(4, 8, dtype=torch.uint8)
        edited = base.clone()
        edited.reshape(-1)[0] = 0x08       # low nibble code 8 vs code 0
        edited.reshape(-1)[1] = 0x20       # high nibble 2 vs 0 -> value diff
        save_file({"m.weight": base}, str(tmp / "l.st"))
        save_file({"m.weight": edited}, str(tmp / "r.st"))
        report = run_compare(tmp / "l.st", tmp / "r.st", tmp / "c.json")
        row = next(r for r in report["rows"] if r["key"] == "m.weight")
        check("packed uint8: zero8 channel still accounted",
              row["zero8_diffs"] == 1 and row["value_diffs"] == 1, row)


def test_int8_adversarial_codes():
    """The decisive cases the nibble path gets WRONG on int8 tensors.

    byte 0x00 vs 0x08 is code 0 vs code +8 -- a REAL value difference on the
    int8 grid, but the packed-nibble path folds 8 <-> 0 onto the same E2M1
    zero and would report zero value diffs (a hard-gate BYPASS).
    byte 0xFF vs 0x01 is code -1 vs +1 -- one code diff, but nibble
    arithmetic reports two diffs with nonsense codes.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        base = torch.zeros(4, 8, dtype=torch.int8)
        save_file({"m.weight": base}, str(tmp / "l.st"))
        edited = base.clone()
        edited.reshape(-1)[3] = 8            # 0 -> +8
        save_file({"m.weight": edited}, str(tmp / "r1.st"))
        report = run_compare(tmp / "l.st", tmp / "r1.st", tmp / "c1.json")
        row = next(r for r in report["rows"] if r["key"] == "m.weight")
        check("int8: code 0 vs +8 is a value diff (gate must NOT fold it)",
              row["value_diffs"] == 1 and report["pass_value"] is False, row)
        check("int8: zero8 channel stays empty for 0 vs +8",
              row["zero8_diffs"] == 0, row)

        edited2 = base.clone()
        edited2.reshape(-1)[3] = 1           # 0 -> +1
        edited2.reshape(-1)[4] = -1          # 0 -> -1 (byte 0xFF)
        save_file({"m.weight": edited2}, str(tmp / "r2.st"))
        report = run_compare(tmp / "l.st", tmp / "r2.st", tmp / "c2.json")
        row = next(r for r in report["rows"] if r["key"] == "m.weight")
        check("int8: -1 vs 0 and +1 vs 0 count as exactly two code diffs",
              row["value_diffs"] == 2 and row["raw_code_diffs"] == 2, row)


def main():
    test_int8_weight_calibre()
    test_int8_adversarial_codes()
    test_int8_scale_still_bitwise()
    test_nvfp4_packed_path_unchanged()
    print("compare_kernel_values int8: all checks passed")


if __name__ == "__main__":
    main()
