"""
Generate teacher self-distillation data with MLX (LLM-QAT protocol).

Protocol (arXiv:2305.17888): each sequence starts from one random token;
the first `--greedy-prefix` continuations are greedy (top-1), the rest are
sampled. Sequences are truncated at EOS. Output is JSONL: {"ids": [...]}.

Run: .venv/bin/python experiment/gen_teacher_data.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import make_sampler

VOCAB = 151669  # Qwen3 full tokenizer length (incl. special tokens)
EOS = 151643  # <|endoftext|>


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--out", default="experiment/data/teacher_qwen3_1p7b.jsonl")
    ap.add_argument("--num-seq", type=int, default=10000)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--greedy-prefix", type=int, default=4)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    model, _ = load(args.model)
    model.eval()
    mx.random.seed(args.seed)
    sampler = make_sampler(temp=args.temp)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    n_tokens = 0
    t0 = time.time()
    with open(out_path, "w") as f:
        while written < args.num_seq:
            bs = min(args.batch, args.num_seq - written)
            seqs = mx.random.randint(0, VOCAB, (bs, 1))
            caches = make_prompt_cache(model)
            cur = seqs
            for pos in range(args.seq_len - 1):
                logits = model(cur, cache=caches)[:, -1].astype(mx.float32)
                if pos < args.greedy_prefix:
                    nxt = mx.argmax(logits, axis=-1)
                else:
                    nxt = sampler(logits)
                nxt = nxt[:, None]
                seqs = mx.concatenate([seqs, nxt], axis=1)
                cur = nxt
            mx.eval(seqs)

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
