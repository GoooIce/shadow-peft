"""Exp5: train the DFlash2-structured shadow drafter (spec exp5-dflash2-shadow).

Loss = CE (causal next-token) + KL (live teacher, masked positions) +
block-CE (k parallel heads predict the next k target tokens at every
position). Injection features come from the teacher forward's free
hidden-state captures. tau trajectory measured with the block engine (r=2).

Run on GB10 (teacher data reused from exp2 shared):
    ssh goooinext 'cd ~/Code/shadow-peft && .venv/bin/python scripts/exp5_train.py'
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
from shadow_lab.dflash import KVInjectedDrafter
from shadow_lab.specdec import block_speculative_generate_fullctx

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
    ap.add_argument("--teacher", default=str(Path.home() / "models" / "LFM2.5-1.2B-Instruct"))
    ap.add_argument("--data", default="runs/exp2/shared/teacher_chat_1p2b.jsonl")
    ap.add_argument("--out", default="runs/exp5")
    ap.add_argument("--max-tokens", type=float, default=5e6)
    ap.add_argument("--batch-seq", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--r", type=int, default=2)
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
    print(f"teacher data: {len(seqs)} seqs", flush=True)

    tau_ids = []
    for p in TAU_PROMPTS:
        enc = tok.apply_chat_template([{"role": "user", "content": p}],
                                      add_generation_prompt=True, return_tensors="pt")
        tau_ids.append((enc["input_ids"] if hasattr(enc, "keys") else enc).to(dev))

    teacher = AutoModelForCausalLM.from_pretrained(args.teacher, dtype=torch.bfloat16).to(dev).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    tcfg = teacher.config

    drafter = KVInjectedDrafter(
        vocab=tcfg.vocab_size, hidden=448, n_layers=4, n_heads=8, n_kv_heads=4,
        target_hidden=tcfg.hidden_size, target_n_layers=tcfg.num_hidden_layers, k=args.k,
    ).to(dev, dtype=torch.bfloat16)
    n_params = sum(p.numel() for p in drafter.parameters())
    print(f"drafter params: {n_params:,} | inject sources: {drafter.sources}", flush=True)
    drafter.train()
    opt = torch.optim.AdamW(drafter.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=0.0)

    seq_len, tokens, step = 512, 0, 0
    t0 = time.time()
    traj_path = out_dir / "eval" / "trajectory.jsonl"
    while tokens < args.max_tokens:
        fwd = make_batch(seqs, rng, args.batch_seq, seq_len, dev, pad_id=pad_id)
        ids, labels, attn = fwd["input_ids"], fwd["labels"], fwd["attention_mask"]
        with torch.no_grad():
            tout = teacher(input_ids=ids, attention_mask=attn, output_hidden_states=True)
            feats = drafter.inject_features(tout.hidden_states)

        h = drafter._trunk(ids, feats, n_ctx=ids.shape[1])
        s_logits = drafter.lm_head(h)

        ce = F.cross_entropy(s_logits[:, :-1].reshape(-1, s_logits.shape[-1]).float(),
                             labels[:, 1:].reshape(-1), ignore_index=-100)
        mask = labels[:, 1:] != -100
        s_lp = F.log_softmax(s_logits[:, :-1].float(), dim=-1)
        t_p = F.softmax(tout.logits[:, :-1].float(), dim=-1)
        kd = (F.kl_div(s_lp, t_p, reduction="none").sum(-1) * mask).sum() / mask.sum()

        # block-CE: head j at position t predicts token t+1+j
        block_losses = []
        b, s = ids.shape
        for j in range(args.k):
            if j + 1 >= s:
                break
            m_j = mask[:, : s - 1 - j]
            head_logits = drafter.lm_head(drafter.block_tf[j](h[:, : s - 1 - j]))
            bl = F.cross_entropy(head_logits.reshape(-1, head_logits.shape[-1]).float(),
                                 ids[:, j + 1:].reshape(-1), reduction="none").view(b, -1)
            block_losses.append((bl * m_j).sum() / m_j.sum().clamp(min=1))
        block_ce = torch.stack(block_losses).mean()

        loss = ce + kd + block_ce
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(drafter.parameters(), 1.0)
        opt.step()

        tokens += int(attn.sum())
        step += 1
        if step % 25 == 0:
            rate = tokens / (time.time() - t0)
            print(f"{{'step': {step}, 'tokens': {tokens}, 'ce': {float(ce):.4f}, "
                  f"'kd': {float(kd):.4f}, 'block_ce': {float(block_ce):.4f}, "
                  f"'tok_per_s': {rate:.0f}}}", flush=True)
        if step % 250 == 0 or tokens >= args.max_tokens:
            drafter.eval()
            taus, produced = [], 0
            agree_c, agree_b, pairs = 0, 0, []
            for tids in tau_ids:
                _, st = block_speculative_generate_fullctx(teacher, drafter, tids, k=args.k,
                                                   r=args.r, max_new_tokens=32)
                taus.append(st["tau"])
                produced += st["produced"]
                with torch.no_grad():
                    tout = teacher(input_ids=tids, output_hidden_states=True)
                    feats = drafter.inject_features(tout.hidden_states)
                    t_next = tout.logits[:, -1].argmax(-1)
                    h = drafter._trunk(tids, feats, n_ctx=tids.shape[1])
                    causal = drafter.lm_head(h[:, -1]).argmax(-1)
                    blk0 = drafter.forward_block(tids, feats)[:, 0].argmax(-1)
                    agree_c += int((causal == t_next).sum())
                    agree_b += int((blk0 == t_next).sum())
                    if len(pairs) < 5:
                        pairs.append([int(causal[0]), int(blk0[0]), int(t_next[0])])
            rec = {"step": step, "tokens": tokens, "tau_mean": round(sum(taus) / len(taus), 4),
                   "k": args.k, "r": args.r, "n_prompts": len(taus),
                   "agree_causal": round(agree_c / (8 * len(tau_ids) / 8) / len(tau_ids), 3),
                   "agree_block0": round(agree_b / len(tau_ids), 3),
                   "pairs_c_b_t": pairs}
            with open(traj_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"[tau] {json.dumps(rec)}", flush=True)
            drafter.train()

    drafter.eval()
    torch.save(drafter.state_dict(), out_dir / "drafter_final.pt")
    taus = []
    for tids in tau_ids:
        _, st = block_speculative_generate_fullctx(teacher, drafter, tids, k=args.k, r=args.r,
                                           max_new_tokens=64)
        taus.append(st["tau"])
    verdict = {
        "tokens": tokens, "steps": step, "params": n_params, "k": args.k, "r": args.r,
        "tau_final": round(sum(taus) / len(taus), 4),
        "references": {"detached_v1": 3.4701, "oracle": args.k + 1, "success_tau": 4.0},
        "wall_s": round(time.time() - t0, 1),
    }
    (out_dir / "verdict.json").write_text(json.dumps(verdict, indent=2) + "\n")
    print(json.dumps(verdict, indent=2), flush=True)


if __name__ == "__main__":
    main()
