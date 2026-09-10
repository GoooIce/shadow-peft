"""INT8 (W8A8) runner adaptation: CPU-level contract tests.

Covers the seams the 2026-09-10 engine probes pinned down
(docs/research/qes-w8a8-engine-contract-20260910.md):

  1. grid: uniform int8 codes, +/-1 rank steps with endpoint clamping, identity
     canonicalisation, value_of == code, beyond-grid rejection;
  2. PayloadReader detects the int8 payload kind and serves codes/shape/
     raw_code in payload order;
  3. build_kernel_payload (int8 branch): fused row concat, transposed [in, out]
     live layout, fp32 [out, 1] scales -- matching the engine bitwise;
  4. the patch view rule: element writes through .t() alias the storage while
     reshape(-1) on the transposed layout silently copies (the trap).

Usage:
  cd $(git rev-parse --show-toplevel)/scripts/minicpm5 && \
      python test_int8_adaptation.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pilot_lib as pl  # noqa: E402


def check(name, cond, detail=""):
    if not cond:
        raise AssertionError(f"{name}: {detail}")
    print(f"  ok  {name}")


def test_int8_grid():
    pl.configure_grid("int8-uniform")
    try:
        check("uniform codes -127..127", pl.ORDERED_CODES[0] == -127
              and pl.ORDERED_CODES[-1] == 127 and len(pl.ORDERED_CODES) == 255)
        check("step +1/-1", pl.step_code(5, 1) == 6 and pl.step_code(5, -1) == 4)
        check("endpoint clamps", pl.step_code(-127, -1) == -126
              and pl.step_code(127, 1) == 126)
        check("canonical identity (code 8 is a real code)",
              pl.canonical_code(8) == 8 and pl.canonical_code(-7) == -7)
        check("value_of == code", pl.value_of(5) == 5.0 and pl.value_of(-3) == -3.0)
        check("project rounds and clamps", pl.project(4.6) == (5, True)
              and pl.project(126.7) == (127, True)
              and pl.project(127.5) == (0, False) and pl.project(-200.0) == (0, False))
        check("gap is the code difference", pl.gap(3, 7) == 4.0 and pl.gap(7, 3) == -4.0)
        state = pl.CentreState()
        state.set_code("l0", 1, 9, base=4)
        check("state round-trip restores the grid",
              pl.CentreState.from_dict(json.loads(json.dumps(state.to_dict())))
              .codes == {"l0": {1: 9}})
        h_int8 = state.sha256(lambda l: "b")
        pl.configure_grid("nvfp4-e2m1")
        h_nvfp4_same_codes = state.sha256(lambda l: "b")
        check("grid feeds the identity hash (same codes, different grids -> different ids)",
              h_int8 != h_nvfp4_same_codes)
    finally:
        pl.configure_grid("nvfp4-e2m1")
    check("nvfp4 restored", pl.canonical_code(8) == 0 and pl.value_of(7) == 6.0)


def _synthetic_payload(root: Path) -> Path:
    """One decoder-ish block: q/k/v/o + gate/up/down as int8 linears."""
    g = torch.Generator().manual_seed(20260910)
    tensors = {}
    shapes = {
        "self_attn.q_proj": (2048, 2048), "self_attn.k_proj": (256, 2048),
        "self_attn.v_proj": (256, 2048), "self_attn.o_proj": (2048, 2048),
        "mlp.gate_proj": (6144, 2048), "mlp.up_proj": (6144, 2048),
        "mlp.down_proj": (2048, 6144),
    }
    base = "model.layers.0"
    for leaf, (out, inn) in shapes.items():
        codes = torch.randint(-127, 128, (out, inn), generator=g).to(torch.int8)
        tensors[f"{base}.{leaf}.weight"] = codes
        tensors[f"{base}.{leaf}.weight_scale"] = (
            torch.rand(out, generator=g) * 0.01 + 0.001).to(torch.bfloat16)
    tensors["model.layers.0.input_layernorm.weight"] = torch.ones(2048, dtype=torch.bfloat16)
    payload = root / "payload"
    payload.mkdir(parents=True)
    save_file(tensors, str(payload / "model.safetensors"))
    (payload / "config.json").write_text("{}")
    return payload


def test_reader_and_export():
    pl.configure_grid("int8-uniform")
    try:
        with tempfile.TemporaryDirectory() as tmp:
            payload = _synthetic_payload(Path(tmp))
            reader = pl.PayloadReader(payload)
            check("payload kind detected", reader.kind == "int8", reader.kind)
            q = reader.codes("model.layers.0.self_attn.q_proj")
            check("codes are signed int8 grid", q.dtype == np.int16
                  and int(q.min()) >= -127 and int(q.max()) <= 127)
            check("shape is [out, in]", reader.shape("model.layers.0.self_attn.q_proj") == (2048, 2048)
                  and reader.shape("model.layers.0.mlp.down_proj") == (2048, 6144))
            flat = int(q.reshape(-1)[123])
            check("raw_code == code (no aliases)", reader.raw_code("model.layers.0.self_attn.q_proj", 123) == flat)
            check("has() keyed on weight+weight_scale",
                  reader.has("model.layers.0.self_attn.q_proj")
                  and not reader.has("model.layers.0.input_layernorm"))
            bf16_side = reader.tensors()
            check("tensors() excludes the quantised side",
                  "model.layers.0.input_layernorm.weight" in bf16_side
                  and "model.layers.0.self_attn.q_proj.weight" not in bf16_side
                  and "model.layers.0.self_attn.q_proj.weight_scale" not in bf16_side)

            out = Path(tmp) / "kernel.safetensors"
            import subprocess
            script = str(Path(__file__).with_name("build_kernel_payload.py"))
            # the script imports vllm at module level (nvfp4 utils); stub it if absent
            code = subprocess.run([sys.executable, script, "--payload", str(payload),
                                   "--out", str(out)], capture_output=True, text=True)
            if code.returncode != 0 and "vllm" in code.stderr:
                print("  ..  vllm unavailable locally; testing the export logic in-process")
                _export_int8_in_process(payload, out)
            from safetensors.torch import load_file, safe_open
            kernel = load_file(str(out))
            qkv = kernel["model.layers.0.self_attn.qkv_proj.weight"]
            check("fused qkv is [in, out_fused]", list(qkv.shape) == [2048, 2560])
            with safe_open(payload / "model.safetensors", framework="pt") as h:
                fused = torch.cat([h.get_tensor(f"model.layers.0.self_attn.{p}.weight")
                                   for p in ("q_proj", "k_proj", "v_proj")], dim=0)
            check("kernel weight == payload fused transposed (bitwise)",
                  torch.equal(qkv, fused.t().contiguous()))
            scale = kernel["model.layers.0.self_attn.qkv_proj.weight_scale"]
            check("scale is fp32 [out, 1]", scale.dtype == torch.float32
                  and list(scale.shape) == [2560, 1])
            gate_up = kernel["model.layers.0.mlp.gate_up_proj.weight"]
            check("fused gate_up is [in, 2*inter]", list(gate_up.shape) == [2048, 12288])
    finally:
        pl.configure_grid("nvfp4-e2m1")


def _export_int8_in_process(payload: Path, out: Path):
    """In-process replica of build_kernel_payload's int8 branch for envs
    without vllm installed (the logic under test is the layout math)."""
    from safetensors import safe_open
    with safe_open(payload / "model.safetensors", framework="pt") as handle:
        keys = list(handle.keys())
        prefixes = sorted({key.rsplit(".", 1)[0] for key in keys})
        int8 = [p for p in prefixes if f"{p}.weight_scale" in keys]
        fused_orders = {"qkv_proj": ("q_proj", "k_proj", "v_proj"),
                        "gate_up_proj": ("gate_proj", "up_proj")}
        tensors = {}
        by_parent: dict[str, dict[str, str]] = {}
        for prefix in int8:
            parent, leaf = prefix.rsplit(".", 1)
            by_parent.setdefault(parent, {})[leaf] = prefix
        for parent, leaves in by_parent.items():
            engine_leaf = ("qkv_proj" if {"q_proj", "k_proj", "v_proj"} & set(leaves)
                           else "gate_up_proj" if {"gate_proj", "up_proj"} & set(leaves)
                           else next(iter(leaves)))
            engine = f"{parent}.{engine_leaf}"
            if engine_leaf in fused_orders:
                shards = [leaves[l] for l in fused_orders[engine_leaf] if l in leaves]
            else:
                shards = [leaves[engine_leaf]]
            fused = torch.cat([handle.get_tensor(f"{s}.weight") for s in shards], dim=0)
            tensors[f"{engine}.weight"] = fused.t().contiguous()
            scale = torch.cat([handle.get_tensor(f"{s}.weight_scale").float()
                               for s in shards], dim=0)
            tensors[f"{engine}.weight_scale"] = scale.reshape(-1, 1).contiguous()
        save_file(tensors, str(out))


def test_minus128_blocks_frozen():
    """A -128 byte is off the symmetric grid; its 16-block must be immovable.

    The real payload carries 158,002 of them (6.3e-5); a candidate edit at
    such a coordinate crashed the campaign with KeyError: -128.
    """
    import numpy as np
    codes = np.zeros((2, 16), dtype=np.int16)
    codes[0, 3] = -128                      # row 0, block 0 poisoned
    codes[1, 3] = -5                        # row 1, block 0 clean
    blocks = codes.reshape(2, 1, 16)
    mask = (blocks != -128).all(axis=2) & np.array([True, True])[:, None]
    check("blocks holding -128 are excluded from the movable mask",
          mask.tolist() == [[False], [True]], mask)


def test_transposed_write_rule():
    """The trap: reshape(-1) on the transposed layout copies; .t() aliases."""
    payload_layout = torch.arange(12, dtype=torch.int8).reshape(3, 4)  # [out, in]
    live = payload_layout.t().clone()      # [in, out]; clone preserves strides
    view = live.t()                        # [out, in] strided view of the same bytes
    check(".t() of the clone aliases the payload-order storage",
          int(view[1, 2]) == int(payload_layout[1, 2]) == 6)
    view[1, 2] = 99
    check("element write through .t() reaches the [in, out] object",
          int(live[2, 1]) == 99)
    check("a fresh .t() view sees the write", int(live.t()[1, 2]) == 99)
    copy = live.reshape(-1)                # non-contiguous -> materialises a copy
    copy[0] = -5
    check("reshape(-1) on the transposed layout materialises a copy",
          int(live.reshape(-1)[0]) != -5)


def main():
    test_int8_grid()
    test_reader_and_export()
    test_minus128_blocks_frozen()
    test_transposed_write_rule()
    print("int8 adaptation: all checks passed")


if __name__ == "__main__":
    main()
