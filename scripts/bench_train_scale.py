"""Single-GB10 training-throughput scaling probe: tok/s vs batch_seq.

Loads the real E1 distillation stack (2-bit base + shadow + online bf16
teacher) once, then times teacher-fwd + student-fwd/bwd + opt.step at several
batch_seq values. Decides whether the loop is compute-bound (linear scaling ->
2-node DDP ~1.9x) or launch-bound (sublinear -> prefer larger local batch).

    ssh goooinext 'cd ~/Code/shadow-peft && .venv/bin/python scripts/bench_train_scale.py'
"""

from __future__ import annotations

import random
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from shadow_lab.atoms import load_teacher_data, make_batch
from shadow_lab.trainer import TrainConfig, _quantize_base
from shadow_peft import ShadowConfig, ShadowForCausalLM, get_shadow_model

MODEL = "/home/goooice/models/LFM2.5-350M-Base"
DATA = "runs/exp1/teacher_lfm25_350m_base.jsonl"
SEQ = 512
BATCHES = [2, 4, 8, 16, 32]
WARMUP = 6
TIMED = 24


def main() -> None:
    device = "cuda"
    cfg = TrainConfig(bits=2, loss_mode="ce_kl", model=MODEL, data=DATA, shadow_layers=2)
    tok = AutoTokenizer.from_pretrained(MODEL)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(device)
    manifest = _quantize_base(base, cfg, len(tok))
    v = base.get_output_embeddings().weight.shape[0]
    base.eval()
    shadow_cfg = ShadowConfig(num_shadow_layers=2, alpha=0.1, dropout=0.0)
    task = ShadowForCausalLM(get_shadow_model(base, shadow_cfg)).to(device)
    task.train()
    teacher = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    trainable = [p for p in task.parameters() if p.requires_grad]
    print(f"trainable={sum(p.numel() for p in trainable):,} quantized={len(manifest['quantized'])}", flush=True)
    opt = torch.optim.AdamW(trainable, lr=1e-5)

    seqs = load_teacher_data(DATA)
    rng = random.Random(0)
    print(f"{'batch':>6} {'ms/step':>9} {'tok/s':>9} {'vs bs=2':>8}", flush=True)
    base_rate = None
    for bs in BATCHES:
        batches = [make_batch(seqs, rng, bs, SEQ, device, pad_id=pad_id) for _ in range(WARMUP + TIMED)]
        for i in range(WARMUP):
            b = batches[i]
            with torch.no_grad():
                t_out = teacher(input_ids=b["input_ids"], attention_mask=b["attention_mask"])
            fwd = dict(input_ids=b["input_ids"], attention_mask=b["attention_mask"],
                       labels=b["labels"], teacher_logits=t_out.logits[..., :v])
            out = task(**fwd)
            opt.zero_grad()
            out.loss.backward()
            opt.step()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        ntok = 0
        for i in range(WARMUP, WARMUP + TIMED):
            b = batches[i]
            ntok += int(b["attention_mask"].sum().item())
            with torch.no_grad():
                t_out = teacher(input_ids=b["input_ids"], attention_mask=b["attention_mask"])
            fwd = dict(input_ids=b["input_ids"], attention_mask=b["attention_mask"],
                       labels=b["labels"], teacher_logits=t_out.logits[..., :v])
            out = task(**fwd)
            opt.zero_grad()
            out.loss.backward()
            opt.step()
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / TIMED
        rate = ntok / (dt * TIMED)
        if base_rate is None:
            base_rate = rate
        print(f"{bs:>6} {dt * 1e3:>9.1f} {rate:>9.0f} {rate / base_rate:>7.2f}x", flush=True)
    print("SCALE-OK", flush=True)


if __name__ == "__main__":
    main()
