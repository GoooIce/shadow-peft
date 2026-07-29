"""
Generate teacher self-distillation data with torch (LLM-QAT protocol).

Torch port of ``gen_teacher_data.py`` (which uses MLX and so only runs on
Apple Silicon). Useful on CUDA/Linux where MLX is unavailable.

Protocol (arXiv:2305.17888): each sequence starts from one random token;
the first ``--greedy-prefix`` continuations are greedy (top-1), the rest are
sampled. Sequences are truncated at EOS. Output is JSONL: {"ids": [...]}.

Run: .venv/bin/python experiment/gen_teacher_data_torch.py
     .venv/bin/python experiment/gen_teacher_data_torch.py --model /path/to/local/model
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen3-1.7B"
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
DTYPE = torch.float16
VOCAB = 151669  # Qwen3 full tokenizer length (incl. special tokens)
EOS = 151643  # <|endoftext|>


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL, help="HF model id or local path")
    ap.add_argument("--out", default="experiment/data/teacher_qwen3_1p7b.jsonl")
    ap.add_argument("--num-seq", type=int, default=10000)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--greedy-prefix", type=int, default=4)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=0, help="0 => full vocab sampling")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=DTYPE).to(DEVICE)
    model.eval()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # id -> token string lookup so EOS-style special tokens match MLX behavior.
    written = 0
    n_tokens = 0
    t0 = time.time()
    with open(out_path, "w") as f:
        while written < args.num_seq:
            bs = min(args.batch, args.num_seq - written)
            starts = torch.randint(0, VOCAB, (bs, 1), device=DEVICE)
            seqs = starts  # [bs, cur_len]
            # Manual KV-free generation: recompute is cheap enough at 1.7B and
            # keeps this port dependency-free (no cache plumbing to wire up).
            past = None
            cur = starts
            for pos in range(args.seq_len - 1):
                with torch.no_grad():
                    if past is None:
                        out = model(input_ids=cur, use_cache=True)
                    else:
                        out = model(input_ids=cur, past_key_values=past, use_cache=True)
                    past = out.past_key_values
                    logits = out.logits[:, -1].float()  # [bs, vocab]
                if pos < args.greedy_prefix:
                    nxt = logits.argmax(dim=-1, keepdim=True)  # [bs, 1]
                else:
                    if args.top_k and args.top_k > 0:
                        k = min(args.top_k, logits.size(-1))
                        vals, idx = logits.topk(k, dim=-1)
                        probs = torch.softmax(vals / args.temp, dim=-1)
                        sampled = torch.multinomial(probs, num_samples=1)  # [bs, 1]
                        nxt = idx.gather(-1, sampled)
                    else:
                        probs = torch.softmax(logits / args.temp, dim=-1)
                        nxt = torch.multinomial(probs, num_samples=1)  # [bs, 1]
                seqs = torch.cat([seqs, nxt], dim=1)
                cur = nxt

            for row in seqs.tolist():
                if EOS in row:
                    row = row[: row.index(EOS)]
                if len(row) >= 8:
                    f.write(json.dumps({"ids": row}) + "\n")
                    n_tokens += len(row)
            written += bs
            rate = n_tokens / (time.time() - t0)
            print(
                f"{written}/{args.num_seq} seqs, {n_tokens:,} tokens "
                f"({rate:.0f} tok/s)",
                flush=True,
            )

    print(f"done: {written} seqs, {n_tokens:,} tokens -> {out_path}")


if __name__ == "__main__":
    main()
