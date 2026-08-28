"""Phase profiler for the real train_shadow loop (make_batch..opt.step).

Times each phase with cuda events over N steps on the real 2-bit stack to
find the ~100ms/step gap between the raw-loop bench (257ms @bs4) and the
smoke run (358ms @bs4).

    ssh goooinext 'cd ~/Code/shadow-peft && .venv/bin/python scripts/bench_trainer_phases.py'
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
DATA = "runs/exp1/teacher_floor_pool.jsonl"
BS = 4
SEQ = 512
STEPS = 60


def main() -> None:
    device = "cuda"
    cfg = TrainConfig(bits=2, loss_mode="ce_kl", model=MODEL, data=DATA, shadow_layers=2)
    tok = AutoTokenizer.from_pretrained(MODEL)
    pad_id = tok.pad_token_id or tok.eos_token_id
    base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(device)
    _quantize_base(base, cfg, len(tok))
    v = base.get_output_embeddings().weight.shape[0]
    base.eval()
    task = ShadowForCausalLM(get_shadow_model(base, ShadowConfig(num_shadow_layers=2, alpha=0.1))).to(device)
    task.train()
    teacher = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(device).eval()
    trainable = [p for p in task.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=1e-5)
    seqs = load_teacher_data(DATA)
    rng = random.Random(0)

    phases = {"make_batch": 0.0, "teacher": 0.0, "student_fwd": 0.0, "item": 0.0,
              "bwd": 0.0, "clip": 0.0, "step": 0.0, "zero": 0.0}
    wall0 = time.perf_counter()
    for _ in range(STEPS):
        ev = [torch.cuda.Event(enable_timing=True) for _ in range(9)]

        ev[0].record()
        batch = make_batch(seqs, rng, BS, SEQ, device, pad_id=pad_id)
        ev[1].record()
        with torch.no_grad():
            t_out = teacher(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        fwd = dict(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                   labels=batch["labels"], teacher_logits=t_out.logits[..., :v])
        ev[2].record()
        out = task(**fwd)
        ev[3].record()
        loss_val = out.loss.item()
        ev[4].record()
        opt.zero_grad()
        out.loss.backward()
        ev[5].record()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        ev[6].record()
        opt.step()
        ev[7].record()
        ev[8].record()
        torch.cuda.synchronize()
        names = list(phases)
        for i, n in enumerate(names):
            phases[n] += ev[i].elapsed_time(ev[i + 1])

    wall = (time.perf_counter() - wall0) / STEPS * 1000
    print(f"wall/step {wall:.1f} ms   (sum of phases below, ms/step)")
    for n, t in phases.items():
        print(f"  {n:>12}: {t / STEPS:7.2f}")


if __name__ == "__main__":
    main()
