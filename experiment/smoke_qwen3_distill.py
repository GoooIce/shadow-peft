"""Smoke test: packed quantized Qwen3 + Shadow adapter + teacher distillation (torch).

Verifies the full recovery path end-to-end, for either bit-width:
  1. Qwen3 quantizes to 1-bit (binary) or 2-bit (ternary) in place and
     still forwards (garbage is OK, NaN is not).
  2. The Shadow adapter wraps the quantized base and trains.
  3. KL distillation from the full-precision teacher flows gradients into
     shadow params only, leaving packed int32 weights bit-identical.

Run: python experiment/smoke_qwen3_distill.py --bits 1
     python experiment/smoke_qwen3_distill.py --bits 2 --model /path/to/local
"""

from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from shadow_peft import ShadowConfig, ShadowForCausalLM, get_shadow_model
from shadow_peft.quantization import quantize_model_1bit, quantize_model_2bit

GROUP_SIZE = 128
STEPS = 10

# fp16 training diverges (loss -> nan) at lr=3e-4 under 2-bit ternary.
_DTYPES = {1: torch.float16, 2: torch.bfloat16}
_REFINE_ITERS = {1: 0, 2: 8}

SENTENCES = [
    "The capital of France is Paris, a city known for the Eiffel Tower.",
    "Water is made of two hydrogen atoms and one oxygen atom.",
    "The stock market fell sharply after the announcement of new tariffs.",
    "She opened the old book and began to read the first chapter aloud.",
    "In machine learning, gradient descent minimizes a loss function.",
    "The recipe calls for two cups of flour and a teaspoon of salt.",
    "Their team won the championship after a dramatic final match.",
    "The committee will review the proposal at next week's meeting.",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, choices=[1, 2], default=1)
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B", help="HF model id or local path")
    args = ap.parse_args()

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    dtype = _DTYPES[args.bits]
    label = "1-bit" if args.bits == 1 else "ternary 2-bit"

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # --- 1. quantize the base ---
    base = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device)
    if args.bits == 1:
        manifest = quantize_model_1bit(base, group_size=GROUP_SIZE)
    else:
        manifest = quantize_model_2bit(base, group_size=GROUP_SIZE, refine_iters=_REFINE_ITERS[args.bits])
    print(
        f"quantized={len(manifest['quantized'])} "
        f"skipped={len(manifest['skipped'])} tied={manifest['tied']}"
    )
    base.eval()

    batch = tokenizer(SENTENCES, return_tensors="pt", padding=True).to(device)
    labels = batch.input_ids.clone()
    labels[batch.attention_mask == 0] = -100

    with torch.no_grad():
        ref = base(**batch)
    assert torch.isfinite(ref.logits).all(), f"{label} forward produced NaN/Inf"
    print(f"{label} forward OK: logits {tuple(ref.logits.shape)}")

    # --- 2. teacher (full precision, frozen, same device) ---
    teacher = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    # --- 3. shadow wrap + distillation ---
    cfg = ShadowConfig(num_shadow_layers=2, dropout=0.0)
    task = ShadowForCausalLM(
        get_shadow_model(base, cfg),
        distill_weight=1.0,
        distill_temperature=1.0,
    )
    task.to(device)
    task.train()

    packed_before = base.model.layers[0].self_attn.q_proj.weight.clone()
    trainable = [p for p in task.parameters() if p.requires_grad]
    print(f"trainable params: {sum(p.numel() for p in trainable):,}")
    opt = torch.optim.AdamW(trainable, lr=3e-4)

    for step in range(STEPS):
        with torch.no_grad():
            teacher_logits = teacher(**batch).logits
        out = task(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            labels=labels,
            teacher_logits=teacher_logits,
        )
        opt.zero_grad()
        out.loss.backward()
        opt.step()
        assert out.kd_loss is not None and torch.isfinite(out.loss)
        print(
            f"step {step}: loss={out.loss.item():.4f} kd={out.kd_loss.item():.4f}"
        )

    packed_after = base.model.layers[0].self_attn.q_proj.weight
    assert torch.equal(packed_before, packed_after), f"packed {label} weights changed!"
    print(f"packed {label} weights untouched by training: OK")
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
