"""
Smoke test: 1-bit packed Qwen3-1.7B + Shadow adapter + teacher distillation (torch).

Verifies the full recovery path end-to-end:
  1. Qwen3-1.7B quantizes to 1-bit in place and still forwards (garbage is OK,
     NaN is not).
  2. The Shadow adapter wraps the quantized base and trains.
  3. KL distillation from the fp16 teacher flows gradients into shadow params
     only, leaving packed int32 weights bit-identical.

Run: .venv/bin/python experiment/smoke_qwen3_1bit_distill.py
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from shadow_peft import ShadowConfig, ShadowForCausalLM, get_shadow_model
from shadow_peft.quantization import quantize_model_1bit

MODEL = "Qwen/Qwen3-1.7B"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
DTYPE = torch.float16
GROUP_SIZE = 128
STEPS = 10

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
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    # --- 1. quantize the base to 1-bit -------------------------------------
    base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=DTYPE).to(DEVICE)
    manifest = quantize_model_1bit(base, group_size=GROUP_SIZE)
    print(
        f"quantized={len(manifest['quantized'])} "
        f"skipped={len(manifest['skipped'])} tied={manifest['tied']}"
    )
    base.eval()

    batch = tokenizer(SENTENCES, return_tensors="pt", padding=True).to(DEVICE)
    labels = batch.input_ids.clone()
    labels[batch.attention_mask == 0] = -100

    with torch.no_grad():
        ref = base(**batch)
    assert torch.isfinite(ref.logits).all(), "1-bit forward produced NaN/Inf"
    print(f"1-bit forward OK: logits {tuple(ref.logits.shape)}")

    # --- 2. teacher (fp16, frozen, same device) -----------------------------
    teacher = AutoModelForCausalLM.from_pretrained(MODEL, dtype=DTYPE).to(DEVICE)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    # --- 3. shadow wrap + distillation --------------------------------------
    cfg = ShadowConfig(num_shadow_layers=2, dropout=0.0)
    task = ShadowForCausalLM(
        get_shadow_model(base, cfg),
        distill_weight=1.0,
        distill_temperature=1.0,
    )
    task.to(DEVICE)
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
    assert torch.equal(packed_before, packed_after), "packed 1-bit weights changed!"
    print("packed 1-bit weights untouched by training: OK")
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
