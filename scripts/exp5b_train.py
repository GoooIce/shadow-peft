"""Exp5 decisive arm: pretrained 350M draft + surgical KV injection (spec exp5).

Identical recipe to exp2 v1 (CE + KL, lr 1e-5, 5M tokens) — the ONLY
difference is the per-layer injection seams fed by free teacher hidden-state
captures. tau(arm) vs tau(v1)=3.4701 isolates the value of conditioning.

Run on GB10:
    ssh goooinext 'cd ~/Code/shadow-peft && .venv/bin/python scripts/exp5b_train.py'
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from shadow_lab.atoms import make_batch
from shadow_lab.dflash import InjectedLFMDrafter
from shadow_lab.specdec import greedy_speculative_generate, measure_tau

TAU_PROMPTS = [
    "Explain recursion.",
    "Write a short paragraph about the silk road.",
    "List three key facts about photosynthesis.",
    "What is the most common mistake people make with pricing?",
    "Summarize jet lag in two sentences.",
    "Give a concrete example that illustrates hash tables.",
    "Why does sleep hygiene matter in practice?",
    "Describe sourdough bread as if to a complete beginner.",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft-model", default=str(Path.home() / "models" / "LFM2.5-350M"))
    ap.add_argument("--teacher", default=str(Path.home() / "models" / "LFM2.5-1.2B-Instruct"))
    ap.add_argument("--data", default="runs/exp2/shared/teacher_chat_1p2b.jsonl")
    ap.add_argument("--out", default="runs/exp5b")
    ap.add_argument("--max-tokens", type=float, default=5e6)
    ap.add_argument("--batch-seq", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--k", type=int, default=4)
    args = ap.parse_args()
    dev = "cuda"
    out_dir = Path(args.out)
    (out_dir / "eval").mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)
    torch.manual_seed(0)

    tok = AutoTokenizer.from_pretrained(args.teacher)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    with open(args.data) as f:
        seqs = [json.loads(line)["ids"] for line in f if len(json.loads(line)["ids"]) >= 33]

    tau_ids = []
    for p in TAU_PROMPTS:
        enc = tok.apply_chat_template([{"role": "user", "content": p}],
                                      add_generation_prompt=True, return_tensors="pt")
        tau_ids.append((enc["input_ids"] if hasattr(enc, "keys") else enc).to(dev))

    teacher = AutoModelForCausalLM.from_pretrained(args.teacher, dtype=torch.bfloat16).to(dev).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    lm = AutoModelForCausalLM.from_pretrained(args.draft_model, dtype=torch.bfloat16).to(dev)
    drafter = InjectedLFMDrafter(lm, target_hidden=teacher.config.hidden_size)
    n_inj = sum(p.numel() for m in drafter.inject_params() for p in m.parameters())
    print(f"injection seams: {len(drafter.seams)} layers, {n_inj:,} injected params | "
          f"sources={drafter.sources[:4]}...", flush=True)
    lm.train()
    opt = torch.optim.AdamW(lm.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=0.0)

    seq_len, tokens, step = 512, 0, 0
    t0 = time.time()
    traj_path = out_dir / "eval" / "trajectory.jsonl"
    while tokens < args.max_tokens:
        fwd = make_batch(seqs, rng, args.batch_seq, seq_len, dev, pad_id=pad_id)
        ids, labels, attn = fwd["input_ids"], fwd["labels"], fwd["attention_mask"]
        with torch.no_grad():
            tout = teacher(input_ids=ids, attention_mask=attn, output_hidden_states=True)
            drafter.set_feats(tout.hidden_states)  # full-coverage injection
        s_logits = lm(input_ids=ids, attention_mask=attn).logits
        drafter.clear_feats()

        ce = F.cross_entropy(s_logits[:, :-1].reshape(-1, s_logits.shape[-1]).float(),
                             labels[:, 1:].reshape(-1), ignore_index=-100)
        mask = labels[:, 1:] != -100
        s_lp = F.log_softmax(s_logits[:, :-1].float(), dim=-1)
        t_p = F.softmax(tout.logits[:, :-1].float(), dim=-1)
        kd = (F.kl_div(s_lp, t_p, reduction="none").sum(-1) * mask).sum() / mask.sum()
        loss = ce + kd

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(lm.parameters(), 1.0)
        opt.step()

        tokens += int(attn.sum())
        step += 1
        if step % 25 == 0:
            rate = tokens / (time.time() - t0)
            print(f"{{'step': {step}, 'tokens': {tokens}, 'ce': {float(ce):.4f}, "
                  f"'kd': {float(kd):.4f}, 'tok_per_s': {rate:.0f}}}", flush=True)
        if step % 250 == 0 or tokens >= args.max_tokens:
            lm.eval()
            tau = measure_tau(teacher, drafter, tau_ids, k=args.k, max_new_tokens=32)
            rec = {"step": step, "tokens": tokens, **tau}
            with open(traj_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"[tau] {json.dumps(rec)}", flush=True)
            lm.train()

    lm.eval()
    torch.save(lm.state_dict(), out_dir / "draft_injected_final.pt")
    taus = []
    for tids in tau_ids:
        _, st = greedy_speculative_generate(teacher, drafter, tids, k=args.k, max_new_tokens=64)
        taus.append(st["tau"])
    verdict = {
        "tokens": tokens, "steps": step, "injected_params": n_inj, "k": args.k,
        "tau_final": round(sum(taus) / len(taus), 4),
        "references": {"detached_v1": 3.4701, "oracle": args.k + 1, "success_tau": 4.0},
        "wall_s": round(time.time() - t0, 1),
    }
    (out_dir / "verdict.json").write_text(json.dumps(verdict, indent=2) + "\n")
    print(json.dumps(verdict, indent=2), flush=True)


if __name__ == "__main__":
    main()
