"""exp1-floor-2 C2: DLM-style masked refinement head (ppl-axis diagnostic).

LLaDA-convention mask-predict refinement of the final state: positions where
the base is UNCONFIDENT (top-1 prob < tau) get re-predicted by a BIDIRECTIONAL
denoising block conditioned on the confident context; the mask is recomputed
between rounds. ppl/teacher-forced axis only — bidirectional attention breaks
the AR KV-cache contract for deployment generation (recorded in the verdict).

    ssh <host> 'cd ~/Code/shadow-peft && .venv/bin/python scripts/exp1_floor2_mask.py \
      --out-dir runs/exp1/floor2_C2 --max-tokens 1e7'
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

import shadow_lab.trainer as trainer_mod
from shadow_lab.trainer import TrainConfig
from shadow_peft import ShadowForCausalLM
from shadow_peft.task_models import (
    ShadowCausalLMOutputWithPast,
    _shifted_ce_loss,
    _shifted_kd_loss,
)

AXIS_CTX = 512
HELD: list = []


class MaskRefineBlock(nn.Module):
    """One bidirectional denoising block: SDPA attention + gated MLP."""

    def __init__(self, hidden: int = 1024, heads: int = 4, kv: int = 2, intermediate: int = 1664):
        super().__init__()
        self.h = heads
        self.hd = hidden // heads
        self.q = nn.Linear(hidden, heads * self.hd, bias=False)
        self.k = nn.Linear(hidden, kv * self.hd, bias=False)
        self.v = nn.Linear(hidden, kv * self.hd, bias=False)
        self.o = nn.Linear(heads * self.hd, hidden, bias=False)
        self.gate = nn.Linear(hidden, intermediate, bias=False)
        self.up = nn.Linear(hidden, intermediate, bias=False)
        self.down = nn.Linear(intermediate, hidden, bias=False)
        self.norm = nn.RMSNorm(hidden)
        # Zero the output projections: delta == 0 at start, so the arm begins
        # exactly at the A4 (L2hw, no head) initialization (budget-fair).
        torch.nn.init.zeros_(self.o.weight)
        torch.nn.init.zeros_(self.down.weight)

    def forward(self, x: torch.Tensor, pad: torch.Tensor | None) -> torch.Tensor:
        b, L, _ = x.shape
        q = self.q(self.norm(x)).view(b, L, self.h, self.hd).transpose(1, 2)
        k = self.k(x).view(b, L, -1, self.hd).transpose(1, 2)
        v = self.v(x).view(b, L, -1, self.hd).transpose(1, 2)
        # Bidirectional: no causal mask; padding only.
        attn_mask = None
        if pad is not None:
            attn_mask = (pad[:, None, None, :] == 0).to(x.dtype) * torch.finfo(x.dtype).min
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, enable_gqa=True)
        a = a.transpose(1, 2).reshape(b, L, -1)
        h = x + self.o(a)
        return h + self.down(F.silu(self.gate(h)) * self.up(h))


class MaskRefineCausalLM(ShadowForCausalLM):
    """Base logits score the mask-refined final state (r mask-predict rounds)."""

    def __init__(self, peft_model, *, tau: float = 0.5, rounds: int = 2, **kw):
        super().__init__(peft_model, **kw)
        self.tau = tau
        self.mask_rounds = rounds
        self.shadow_mask_refiner = MaskRefineBlock()
        self.shadow_mask_refiner.to(next(peft_model.base_model.parameters()).dtype)
        HELD.append(self)

    def forward(self, *args, **kw):
        kw.setdefault("output_hidden_states", True)
        outputs, shadow_hidden = self.peft_model.forward_with_shadow(
            *args, use_cache=False, past_key_values=None, **kw)
        h = outputs.hidden_states[-1]
        pad = kw.get("attention_mask")
        base_logits = self.lm_head(h)
        for _ in range(self.mask_rounds):
            probs = base_logits.float().softmax(dim=-1)
            m = (probs.max(dim=-1).values < self.tau)  # [b, L] uncertain positions
            if pad is not None:
                m = m & (pad > 0)
            if not bool(m.any()):
                break
            delta = self.shadow_mask_refiner(h, pad) - h  # correction proposal
            h = h + delta * m.unsqueeze(-1).to(h.dtype)  # apply only where uncertain
            base_logits = self.lm_head(h)

        labels = kw.get("labels")
        shadow_logits = self.shadow_lm_head(shadow_hidden)
        loss = None
        if labels is not None:
            loss = _shifted_ce_loss(base_logits, labels)
            if self.shadow_loss_weight > 0:
                loss = loss + self.shadow_loss_weight * _shifted_ce_loss(shadow_logits, labels)
            teacher_logits = kw.get("teacher_logits")
            if teacher_logits is not None and self.distill_weight > 0:
                loss = loss + self.distill_weight * _shifted_kd_loss(
                    base_logits, teacher_logits, labels, self.distill_temperature)
        return ShadowCausalLMOutputWithPast(
            loss=loss, logits=base_logits, shadow_logits=shadow_logits,
            kd_loss=None, hidden_loss=None, past_key_values=None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-tokens", type=float, default=1e7)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--main-layers", type=int, default=2)
    ap.add_argument("--main-intermediate", type=int, default=3328)
    ap.add_argument("--main-heads", type=int, default=8)
    ap.add_argument("--main-kv-heads", type=int, default=4)
    ap.add_argument("--model", default=str(Path.home() / "models" / "LFM2.5-350M-Base"))
    ap.add_argument("--data", default="runs/exp1/teacher_floor_pool.jsonl")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    class Task(MaskRefineCausalLM):
        def __init__(self, peft, **kw):
            super().__init__(peft, tau=args.tau, rounds=args.rounds, **kw)

    orig_cls = trainer_mod.ShadowForCausalLM
    trainer_mod.ShadowForCausalLM = Task
    try:
        tok = AutoTokenizer.from_pretrained(args.model)
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        cfg = TrainConfig(
            bits=2, loss_mode="ce_kl", model=args.model, data=args.data,
            max_tokens=args.max_tokens, seq_len=512, batch_seq=8, lr=1e-5,
            shadow_layers=args.main_layers,
            shadow_intermediate_size=args.main_intermediate,
            shadow_num_attention_heads=args.main_heads,
            shadow_num_key_value_heads=args.main_kv_heads,
            thr_ratio=0.5, ppl_context_len=AXIS_CTX, pad_id=pad_id,
            save_dir=str(Path(args.out_dir) / "train"),
        )
        report = trainer_mod.train_shadow(cfg)
    finally:
        trainer_mod.ShadowForCausalLM = orig_cls

    out_dir = Path(args.out_dir)
    if HELD:
        torch.save(HELD[-1].shadow_mask_refiner.state_dict(), out_dir / "train" / "mask_refiner.pt")
    verdict = {
        "arm": f"C2_mask_tau{args.tau}_rounds{args.rounds}",
        "main": {"layers": args.main_layers, "intermediate": args.main_intermediate,
                 "heads": args.main_heads, "kv": args.main_kv_heads},
        "mask_head_params": sum(p.numel() for p in HELD[-1].shadow_mask_refiner.parameters()),
        "tau": args.tau, "rounds": args.rounds,
        "trainable_params": report.trainable_params,
        "tokens": report.tokens,
        "ppl_shadow": round(report.final_ppl_20chunks, 4),
        "axis": "despaced wikitext, ctx=512, chunks=20",
        "caveats": ["bidirectional head: ppl/teacher-forced axis only, NOT AR-deployable",
                    "refiner outside spike-guard rollback and core checkpoint contract"],
    }
    (out_dir / "verdict.json").write_text(json.dumps(verdict, indent=2) + "\n")
    print(json.dumps(verdict, indent=2), flush=True)


if __name__ == "__main__":
    main()
