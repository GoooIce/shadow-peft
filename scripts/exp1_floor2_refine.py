"""exp1-floor-2 C1: iterative-refinement (diffusion-style) shadow head.

Weight-shared r-round residual refinement of the FINAL base hidden state
before lm_head: h ← h + g(h), g = one causal attention layer (same family as
the shadow trunk). Params fixed, compute ×r — the compute-vs-params axis.
Deployable (causal); screening-only scaffolding: the refiner is saved beside
the adapter (winner gets promoted into the core contract, not now).

    ssh <host> 'cd ~/Code/shadow-peft && .venv/bin/python scripts/exp1_floor2_refine.py \
      --rounds 2 --out-dir runs/exp1/floor2_C1r2 --max-tokens 1e7'
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import shadow_lab.trainer as trainer_mod
from shadow_lab.atoms import despace_wikitext, load_wikitext2_test_text
from shadow_lab.trainer import TrainConfig
from shadow_peft import ShadowForCausalLM
from shadow_peft.model_utils import build_implicit_shadow_model
from shadow_peft.task_models import (
    ShadowCausalLMOutputWithPast,
    _shifted_ce_loss,
    _shifted_kd_loss,
)

AXIS_CTX = 512
HELD: list = []


class RefineCausalLM(ShadowForCausalLM):
    """ShadowForCausalLM + r weight-shared refinement rounds on the final state.

    The spike-guard's load_state_dict covers adapter weights only (the parent's
    state_dict delegates to peft_model); the refiner is outside rollback —
    acceptable for screening, recorded as a caveat.
    """

    def __init__(self, peft_model, *, refine_rounds: int = 1, refine_width: dict | None = None,
                 **kw):
        super().__init__(peft_model, **kw)
        self.refine_rounds = refine_rounds
        self.shadow_refiner = build_implicit_shadow_model(
            peft_model.base_model, num_shadow_layers=1, **(refine_width or {}))
        self.shadow_refiner.to(next(peft_model.base_model.parameters()).dtype)
        # The refiner only ever runs via inputs_embeds; its freshly-built
        # embed_tokens (67M params for a 65k vocab) would silently join the
        # trainable set and inflate the budget — freeze it out.
        self.shadow_refiner.get_input_embeddings().weight.requires_grad_(False)
        # Zero-init the output projections so g(h) == 0 at start: C1 begins
        # exactly at the A4 (L2hw, r=0) initialization instead of a random
        # scramble hole, making the r-axis readout budget-fair.
        for proj in ("self_attn.out_proj", "feed_forward.w2"):
            for n, m in self.shadow_refiner.named_modules():
                if n.endswith(proj):
                    torch.nn.init.zeros_(m.weight)
                    break
        HELD.append(self)

    def forward(self, *args, **kw):
        # Parent's base_shadow path, except base_logits score the refined state.
        kw.setdefault("output_hidden_states", True)
        outputs, shadow_hidden = self.peft_model.forward_with_shadow(
            *args, use_cache=False, past_key_values=None, **kw)
        h = outputs.hidden_states[-1]
        am = kw.get("attention_mask")
        for _ in range(self.refine_rounds):
            # The layer is internally residual (h + attn + mlp); with the
            # zero-init projections this is exactly identity at start.
            h = self.shadow_refiner(inputs_embeds=h, attention_mask=am).last_hidden_state
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
    ap.add_argument("--rounds", type=int, required=True)
    ap.add_argument("--max-tokens", type=float, default=1e7)
    ap.add_argument("--main-layers", type=int, default=2)
    ap.add_argument("--main-intermediate", type=int, default=3328)
    ap.add_argument("--main-heads", type=int, default=8)
    ap.add_argument("--main-kv-heads", type=int, default=4)
    ap.add_argument("--refine-intermediate", type=int, default=1664)
    ap.add_argument("--refine-heads", type=int, default=4)
    ap.add_argument("--refine-kv-heads", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--lr-schedule", choices=["constant", "cosine"], default="constant")
    ap.add_argument("--model", default=str(Path.home() / "models" / "LFM2.5-350M-Base"))
    ap.add_argument("--data", default="runs/exp1/teacher_floor_pool.jsonl")
    ap.add_argument("--tier", choices=["bits2", "w4a16"], default="bits2")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    refine_width = dict(shadow_intermediate_size=args.refine_intermediate,
                        shadow_num_attention_heads=args.refine_heads,
                        shadow_num_key_value_heads=args.refine_kv_heads)

    class Task(RefineCausalLM):
        def __init__(self, peft, **kw):
            super().__init__(peft, refine_rounds=args.rounds, refine_width=refine_width, **kw)

    orig_cls = trainer_mod.ShadowForCausalLM
    trainer_mod.ShadowForCausalLM = Task
    try:
        tok = AutoTokenizer.from_pretrained(args.model)
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        tier_kw = (dict(bits=0, backbone_tier="w4a16", tier_fake_quant=True)
                   if args.tier == "w4a16" else dict(bits=2, thr_ratio=0.5))
        cfg = TrainConfig(
            loss_mode="ce_kl", model=args.model, data=args.data,
            max_tokens=args.max_tokens, seq_len=512, batch_seq=8, lr=args.lr,
            lr_schedule=args.lr_schedule,
            shadow_layers=args.main_layers,
            shadow_intermediate_size=args.main_intermediate,
            shadow_num_attention_heads=args.main_heads,
            shadow_num_key_value_heads=args.main_kv_heads,
            ppl_context_len=AXIS_CTX, pad_id=pad_id,
            save_dir=str(Path(args.out_dir) / "train"),
            **tier_kw,
        )
        report = trainer_mod.train_shadow(cfg)
    finally:
        trainer_mod.ShadowForCausalLM = orig_cls

    out_dir = Path(args.out_dir)
    if HELD:
        torch.save(HELD[-1].shadow_refiner.state_dict(), out_dir / "train" / "refiner.pt")
    verdict = {
        "arm": f"C1_rounds{args.rounds}_{args.tier}",
        "main": {"layers": args.main_layers, "intermediate": args.main_intermediate,
                 "heads": args.main_heads, "kv": args.main_kv_heads},
        "refiner": {"intermediate": args.refine_intermediate, "heads": args.refine_heads,
                    "kv": args.refine_kv_heads, "rounds": args.rounds},
        "trainable_params": report.trainable_params,
        "tokens": report.tokens,
        "ppl_shadow": round(report.final_ppl_20chunks, 4),
        "axis": "despaced wikitext, ctx=512, chunks=20",
        "caveats": ["refiner outside spike-guard rollback and core checkpoint contract"],
    }
    (out_dir / "verdict.json").write_text(json.dumps(verdict, indent=2) + "\n")
    print(json.dumps(verdict, indent=2), flush=True)


if __name__ == "__main__":
    main()
