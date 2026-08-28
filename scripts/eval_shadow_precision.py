"""Shadow-precision PTQ probe on trained w4a16 shadows (1.2B by default).

Question: can the recovery shadow itself ship at lower precision
(fp16 / fp8-e4m3 / nvfp4) without losing the recovered ppl?

Methodology mirrors the backbone fake-quant convention: shadow Linear
weights are round-tripped through the target format and computed back in
bf16; the w4a16 fake-quant backbone is untouched. The bf16 cell re-evals
the untouched shadow and must reproduce the training verdict exactly
(determinism gate for the whole probe).

Formats:
  fp16   — dtype round-trip (storage error only)
  fp8    — weight-only e4m3, one fp32 scale per output channel
  nvfp4  — e2m1 codebook, block=16 along in_features, per-block scale
           stored in e4m3 (two-level nvfp4 scheme)

    ssh goooinext 'cd ~/Code/shadow-peft && nohup .venv/bin/python \
      scripts/eval_shadow_precision.py > runs/logs/0827-shadow-precision.log 2>&1 < /dev/null &'
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from shadow_lab import tiers
from shadow_lab.atoms import despace_wikitext, eval_ppl, load_wikitext2_test_text
from shadow_peft import ShadowPeftModel

E2M1_MAGS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
AXIS_CTX = 512
AXIS_CHUNKS = 20


def _fp8_e4m3_rt(t: torch.Tensor) -> torch.Tensor:
    return t.to(torch.float32).to(torch.float8_e4m3fn).to(torch.float32)


def fake_fp16(w: torch.Tensor) -> torch.Tensor:
    return w.detach().to(torch.float16).to(w.dtype)


def fake_fp8_rowwise(w: torch.Tensor) -> torch.Tensor:
    wf = w.detach().float()
    scale = wf.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / 448.0
    q = _fp8_e4m3_rt((wf / scale).clamp(-448.0, 448.0))
    return (q * scale).to(w.dtype)


def fake_fp8_blockwise(w: torch.Tensor, block: int = 128) -> torch.Tensor:
    # e4m3 elements with fine-grained per-block fp32 scales (DeepSeek-style):
    # the fair "granularity, not bit-width" competitor to nvfp4.
    def fn(blocks):
        scale = blocks.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / 448.0
        q = _fp8_e4m3_rt((blocks / scale).clamp(-448.0, 448.0))
        return q * scale
    return _blockwise(w, block, fn)


def _two_level_fake_quant(blocks: torch.Tensor, *, e2m1: bool) -> torch.Tensor:
    # NVFP4/NVFP8 two-level scaling: per-block fp32 scale s_i = max_i/elem_max,
    # global fp32 scale g calibrated so s_i/g lands inside e4m3's range,
    # stored block scale = e4m3(s_i/g), effective scale = g * that.
    elem_max = 6.0 if e2m1 else 448.0
    s32 = blocks.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / elem_max
    g = s32.max().clamp_min(1e-30) / 448.0
    s8 = _fp8_e4m3_rt(s32 / g)
    scale = g * s8
    n = (blocks / scale).clamp(-elem_max, elem_max)
    if e2m1:
        cb = torch.tensor(E2M1_MAGS, device=blocks.device, dtype=torch.float32)
        idx = (n.abs().unsqueeze(-1) - cb).abs().argmin(dim=-1)
        q = torch.sign(n) * cb[idx]
    else:
        q = _fp8_e4m3_rt(n)
    return q * scale


def _blockwise(w: torch.Tensor, block: int, fn) -> torch.Tensor:
    wf = w.detach().float()
    out_f, in_f = wf.shape
    pad = (-in_f) % block
    if pad:
        wf = torch.nn.functional.pad(wf, (0, pad))
    out = fn(wf.view(out_f, -1, block)).view(out_f, -1)
    if pad:
        out = out[:, :in_f]
    return out.to(w.dtype)


def fake_nvfp4(w: torch.Tensor, block: int = 16) -> torch.Tensor:
    # e2m1 codebook + block-16 e4m3 scales under a global fp32 scale (4.5 b/w)
    return _blockwise(w, block, lambda b: _two_level_fake_quant(b, e2m1=True))


def fake_nvfp8(w: torch.Tensor, block: int = 16) -> torch.Tensor:
    # same two-level scheme with e4m3 elements (8.5 b/w)
    return _blockwise(w, block, lambda b: _two_level_fake_quant(b, e2m1=False))


TRANSFORMS = {
    "bf16": None,
    "fp16": fake_fp16,
    "fp8": fake_fp8_rowwise,
    "fp8b128": fake_fp8_blockwise,
    "nvfp8": fake_nvfp8,
    "nvfp4": fake_nvfp4,
}


def quantized_base(model_path: str):
    base = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16)
    tiers.apply_training_conversion(base, "w4a16", torch.device("cuda"), fake_quant=True)
    return base.to("cuda").eval()


def apply_shadow_ptq(peft: ShadowPeftModel, fn) -> int:
    swapped = 0
    for mod in peft.shadow_model.modules():
        if isinstance(mod, torch.nn.Linear):
            with torch.no_grad():
                mod.weight = torch.nn.Parameter(fn(mod.weight), requires_grad=False)
            swapped += 1
    return swapped


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(Path.home() / "models" / "LFM2.5-1.2B-Base"))
    ap.add_argument("--ckpt", action="append", required=True,
                    help="shadow checkpoint dir (repeatable), e.g. runs/exp1/floor2_w4a16_1p2b_L1dw/train/final")
    ap.add_argument("--modes", default="bf16,fp16,fp8,nvfp4")
    ap.add_argument("--out", default="runs/exp1/shadow_precision_1p2b/verdict.json")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    axis_text = despace_wikitext(load_wikitext2_test_text())
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    rows = []
    for ckpt in args.ckpt:
        for mode in modes:
            base = quantized_base(args.model)
            peft = ShadowPeftModel.from_pretrained(base, ckpt).eval()
            n_lin = apply_shadow_ptq(peft, TRANSFORMS[mode]) if TRANSFORMS[mode] else 0
            ppl = eval_ppl(peft, tok, axis_text, chunks=AXIS_CHUNKS, device="cuda",
                           context_len=AXIS_CTX)
            row = {"ckpt": ckpt, "mode": mode, "ppl": round(ppl, 4), "linears_swapped": n_lin}
            rows.append(row)
            print(json.dumps(row), flush=True)
            del peft, base
            torch.cuda.empty_cache()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"written {out}", flush=True)


if __name__ == "__main__":
    main()
