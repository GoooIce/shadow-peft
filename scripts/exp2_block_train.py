"""Exp2 v2-block: train single-forward block drafters (DSpark vs DFlash2 structure).

Same trunk (pretrained LFM2.5-350M), same data (teacher_chat_1p2b.jsonl, 5M
tokens — the v1 chain's exact source), same trunk recipe (causal CE + KL, all
positions — which is also what DFlash2 refinement consumes). Structure knob:

- dspark:  k parallel low-rank heads + confidence head (per-position
           draft==teacher argmax BCE); inference = one forward, r=0.
- dflash2: k parallel low-rank heads only; inference = heads + r causal
           refinement rounds (trunk's own next-token head re-scores).

Run on GB10 (sequential arms):
    ssh goooinext 'cd ~/Code/shadow-peft && .venv/bin/python scripts/exp2_block_train.py --structure dspark'
    ssh goooinext 'cd ~/Code/shadow-peft && .venv/bin/python scripts/exp2_block_train.py --structure dflash2'
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
from shadow_lab.block_drafter import CachedBlockDrafter
from shadow_lab.specdec import block_speculative_generate

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
R_FOR = {"dspark": 0, "dflash2": 2}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--structure", required=True, choices=["dspark", "dflash2"])
    ap.add_argument("--trunk", default=str(Path.home() / "models" / "LFM2.5-350M"))
    ap.add_argument("--teacher", default=str(Path.home() / "models" / "LFM2.5-1.2B-Instruct"))
    ap.add_argument("--data", default="runs/exp2/shared/teacher_chat_1p2b.jsonl")
    ap.add_argument("--out-root", default="runs/exp2")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--max-tokens", type=float, default=5e6)
    ap.add_argument("--batch-seq", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-5)
    args = ap.parse_args()
    dev = "cuda"
    out_dir = Path(args.out_root) / f"v2block_{args.structure}"
    (out_dir / "eval").mkdir(parents=True, exist_ok=True)
    r_eval = R_FOR[args.structure]

    rng = random.Random(0)
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(args.teacher)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    with open(args.data) as f:
        seqs = [r for r in (json.loads(line)["ids"] for line in f) if len(r) >= 33]
    print(f"[{args.structure}] teacher data: {len(seqs)} seqs", flush=True)

    tau_ids = []
    for p in TAU_PROMPTS:
        enc = tok.apply_chat_template([{"role": "user", "content": p}],
                                      add_generation_prompt=True, return_tensors="pt")
        tau_ids.append((enc["input_ids"] if hasattr(enc, "keys") else enc).to(dev))

    teacher = AutoModelForCausalLM.from_pretrained(args.teacher, dtype=torch.bfloat16).to(dev).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    trunk = AutoModelForCausalLM.from_pretrained(args.trunk, dtype=torch.bfloat16)
    drafter = CachedBlockDrafter(trunk, k=args.k,
                                 with_confidence=args.structure == "dspark").to(dev)
    drafter.train()
    opt = torch.optim.AdamW(drafter.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=0.0)
    lm_head = trunk.get_output_embeddings()

    tokens, step = 0, 0
    t0 = time.time()
    traj_path = out_dir / "eval" / "trajectory.jsonl"

    def tau_probe() -> dict:
        drafter.eval()
        taus = []
        with torch.no_grad():
            for ids in tau_ids:
                _, st = block_speculative_generate(teacher, drafter, ids, k=args.k, r=r_eval,
                                                   max_new_tokens=32)
                taus.append(st["tau"])
        drafter.train()
        return {"tau_mean": round(sum(taus) / len(taus), 4), "k": args.k, "r": r_eval}

    while tokens < args.max_tokens:
        fwd = make_batch(seqs, rng, args.batch_seq, args.seq_len, dev, pad_id=pad_id)
        ids, labels, attn = fwd["input_ids"], fwd["labels"], fwd["attention_mask"]
        b, s = ids.shape
        with torch.no_grad():
            t_logits = teacher(input_ids=ids, attention_mask=attn).logits
        out = trunk(input_ids=ids, attention_mask=attn, output_hidden_states=True)
        s_logits, h = out.logits, out.hidden_states[-1]

        # trunk causal loss (v1 recipe)
        ce = F.cross_entropy(s_logits[:, :-1].reshape(-1, s_logits.shape[-1]).float(),
                             labels[:, 1:].reshape(-1), ignore_index=-100)
        mask = labels[:, 1:] != -100
        s_lp = F.log_softmax(s_logits[:, :-1].float(), dim=-1)
        t_p = F.softmax(t_logits[:, :-1].float(), dim=-1)
        kd = (F.kl_div(s_lp, t_p, reduction="none").sum(-1) * mask).sum() / mask.sum()

        # k block heads: head j at context i predicts the token at i+1+j
        head_ce = head_kd = conf_bce = None
        ces, kds, bfs = [], [], []
        for j in range(args.k):
            sl = slice(0, s - 1 - j)
            h_j = drafter.heads.tf[j](h[:, sl])                       # [b, s-1-j, hid]
            lg_j = lm_head(h_j)                                       # [b, s-1-j, V]
            y_j = labels[:, 1 + j:]                                   # [b, s-1-j]
            m_j = y_j != -100
            ces.append(F.cross_entropy(lg_j.reshape(-1, lg_j.shape[-1]).float(),
                                       y_j.reshape(-1), ignore_index=-100))
            lp = F.log_softmax(lg_j.float(), dim=-1)
            tp = F.softmax(t_logits[:, j : s - 1].float(), dim=-1)
            kds.append((F.kl_div(lp, tp, reduction="none").sum(-1) * m_j).sum() / m_j.sum())
            if drafter.heads.conf is not None:
                agree = (lg_j.argmax(-1) == y_j).float()
                cf = drafter.heads.conf[j](h_j).squeeze(-1)
                msk = m_j.float()
                denom = msk.sum().clamp(min=1.0)
                bfs.append((F.binary_cross_entropy_with_logits(cf, agree, reduction="none") * msk
                            ).sum() / denom)
        head_ce = torch.stack(ces).mean()
        head_kd = torch.stack(kds).mean()
        loss = (ce + kd) + (head_ce + head_kd)
        if bfs:
            conf_bce = torch.stack(bfs).mean()
            loss = loss + conf_bce

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(drafter.parameters(), 1.0)
        opt.step()

        tokens += int(attn.sum())
        step += 1
        if step % 25 == 0:
            msg = {"step": step, "tokens": tokens, "ce": round(float(ce), 4),
                   "kd": round(float(kd), 4), "head_ce": round(float(head_ce), 4),
                   "head_kd": round(float(head_kd), 4), "tok_per_s": round(tokens / (time.time() - t0))}
            if conf_bce is not None:
                msg["conf_bce"] = round(float(conf_bce), 4)
            print(json.dumps(msg), flush=True)
        if step % 250 == 0 or tokens >= args.max_tokens:
            rec = {"step": step, "tokens": tokens, **tau_probe()}
            with open(traj_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"[tau] {json.dumps(rec)}", flush=True)

    drafter.eval()
    taus = []
    with torch.no_grad():
        for ids in tau_ids:
            _, st = block_speculative_generate(teacher, drafter, ids, k=args.k, r=r_eval,
                                               max_new_tokens=64)
            taus.append(st["tau"])
    tau_final = round(sum(taus) / len(taus), 4)
    torch.save({"lm": trunk.state_dict(), "heads": drafter.heads.state_dict()},
               out_dir / "drafter_final.pt")
    verdict = {"structure": args.structure, "k": args.k, "r_eval": r_eval, "tokens": tokens,
               "steps": step, "tau_final": tau_final, "wall_s": round(time.time() - t0, 1)}
    (out_dir / "verdict_train.json").write_text(json.dumps(verdict, indent=2) + "\n")
    print(json.dumps(verdict, indent=2), flush=True)


if __name__ == "__main__":
    main()
