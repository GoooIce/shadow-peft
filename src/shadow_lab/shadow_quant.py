"""Native nvfp8/nvfp4 shadow-weight training (projected-gradient form).

The shadow's Linear weights live ON the format grid for the whole run:
the random init is projected onto the dequantization grid before the
optimizer is created, and after every optimizer step the weights are
projected back. Gradients are therefore always evaluated exactly at the
weights that ship: ``save_pretrained`` stores the on-grid tensor, and the
deployed model is bit-identical to the trained one — no bf16 master
weight, no straight-through estimator.

Formats (two-level scaling, mirrors the eval probe in
scripts/eval_shadow_precision.py):
  nvfp8 — e4m3 elements, block-16 scales stored in e4m3 under a global
          fp32 scale (8.5 bits/weight)
  nvfp4 — e2m1 elements, same two-level scheme (4.5 bits/weight)
"""

from __future__ import annotations

import torch

E2M1_MAGS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _fp8_e4m3_rt(t: torch.Tensor) -> torch.Tensor:
    return t.to(torch.float32).to(torch.float8_e4m3fn).to(torch.float32)


def _two_level_grid(blocks: torch.Tensor, *, e2m1: bool) -> torch.Tensor:
    elem_max = 6.0 if e2m1 else 448.0
    s32 = blocks.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / elem_max
    g = s32.max().clamp_min(1e-30) / 448.0
    scale = g * _fp8_e4m3_rt(s32 / g)
    n = (blocks / scale).clamp(-elem_max, elem_max)
    if e2m1:
        cb = torch.tensor(E2M1_MAGS, device=blocks.device, dtype=torch.float32)
        idx = (n.abs().unsqueeze(-1) - cb).abs().argmin(dim=-1)
        q = torch.sign(n) * cb[idx]
    else:
        q = _fp8_e4m3_rt(n)
    return q * scale


def nvfp8_weight(w: torch.Tensor, block: int = 16) -> torch.Tensor:
    out_f, in_f = w.shape
    wf = w.detach().float()
    pad = (-in_f) % block
    if pad:
        wf = torch.nn.functional.pad(wf, (0, pad))
    out = _two_level_grid(wf.view(out_f, -1, block), e2m1=False).view(out_f, -1)
    return (out[:, :in_f] if pad else out).to(w.dtype)


def nvfp4_weight(w: torch.Tensor, block: int = 16) -> torch.Tensor:
    out_f, in_f = w.shape
    wf = w.detach().float()
    pad = (-in_f) % block
    if pad:
        wf = torch.nn.functional.pad(wf, (0, pad))
    out = _two_level_grid(wf.view(out_f, -1, block), e2m1=True).view(out_f, -1)
    return (out[:, :in_f] if pad else out).to(w.dtype)


FORMATS = {"nvfp8": nvfp8_weight, "nvfp4": nvfp4_weight}


def collect_shadow_linears(peft_model) -> list[torch.nn.Linear]:
    """Every Linear inside the mounted shadow stack (the trainable surface)."""
    return [m for m in peft_model.shadow_model.modules() if isinstance(m, torch.nn.Linear)]


def project_linears(linears: list[torch.nn.Linear], fmt: str) -> int:
    """Project each Linear weight onto the format grid, in place."""
    fn = FORMATS.get(fmt)
    if fn is None:
        raise ValueError(f"unknown shadow weight format {fmt!r} (expected one of {sorted(FORMATS)})")
    with torch.no_grad():
        for mod in linears:
            mod.weight.copy_(fn(mod.weight))
    return len(linears)


def residual_shuffle_noise(w: torch.Tensor, fmt: str = "nvfp4", mult: float = 1.0,
                           block: int = 16) -> torch.Tensor:
    """Residual-matched noise for the mechanism-attribution arms (exp1-native4b).

    Returns w + mult * shuffle_within_block(w - Q_fmt(w)): the format's actual
    rounding residual with the deterministic element-wise coupling to w broken
    by an i.i.d. permutation inside each block — per-block scale and the
    residual marginal are preserved exactly, the value at each position is not
    a rounding of w's value there. A statistical stand-in for the implicit
    noise of projected-gradient native training, with no grid constraint.
    """
    fn = FORMATS.get(fmt)
    if fn is None:
        raise ValueError(f"unknown shadow weight format {fmt!r} (expected one of {sorted(FORMATS)})")
    out_f, in_f = w.shape
    wf = w.detach().float()
    pad = (-in_f) % block
    if pad:
        wf = torch.nn.functional.pad(wf, (0, pad))
    blocks = wf.view(out_f, -1, block)
    residual = blocks - fn(wf).view(out_f, -1, block)
    perm = torch.argsort(torch.rand_like(residual), dim=-1)
    noisy = blocks + mult * torch.gather(residual, -1, perm)
    out = noisy.view(out_f, -1)
    return (out[:, :in_f] if pad else out).to(w.dtype)
