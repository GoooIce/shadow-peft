"""Teacher self-distillation data generation (LLM-QAT protocol, torch).

Each sequence starts from ``<bos>`` + one random token; the first
``--greedy-prefix`` continuations are greedy (top-1), the rest sampled.
Sequences truncate at EOS. Output JSONL: ``{"ids": [...]}`` — consumed by
``shadow_lab.trainer.train_shadow``.

Run: python -m shadow_lab.teacher_data --model ~/models/LFM2.5-350M-Base
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .atoms import pick_device


def _chat_buckets(tok, prompts_file: str, max_prompt_len: int = 128) -> dict[int, list[list[int]]]:
    """Tokenize the chat-template prompts and bucket by EXACT token length.

    Conv hybrids cannot take padded batches (left/right pads both corrupt the
    conv left-context), so only exactly-equal prompt lengths share a batch.
    """
    buckets: dict[int, list[list[int]]] = {}
    with open(prompts_file) as f:
        for line in f:
            prompt = json.loads(line)["prompt"]
            enc = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                          add_generation_prompt=True)
            if hasattr(enc, "keys") and "input_ids" in enc:  # BatchEncoding
                enc = enc["input_ids"]
            for _ in range(2):  # unwrap [[...]] / ["..."] single-element wrappers
                if isinstance(enc, list) and len(enc) == 1:
                    enc = enc[0]
            if isinstance(enc, str):  # template rendered as text (no return_tensors)
                ids = tok(enc, add_special_tokens=False).input_ids
            elif hasattr(enc, "ids"):  # tokenizers.Encoding
                ids = list(enc.ids)
            elif isinstance(enc, torch.Tensor):
                ids = enc.tolist()
            else:
                ids = list(enc)
            assert ids and all(isinstance(i, int) for i in ids), type(enc)
            if len(ids) <= max_prompt_len:
                buckets.setdefault(len(ids), []).append(ids)
    return buckets


@torch.no_grad()
def _sample_chat_batch(model, prompts: list[list[int]], *, seq_len: int, device: str,
                       greedy_prefix: int, temp: float, eos: int, vocab: int, top_k: int = 0,
                       gen: torch.Generator | None = None) -> list[list[int]]:
    """Sample one response per prompt (chat-seeded, EOS-truncated)."""
    bs = len(prompts)
    cur = torch.tensor(prompts, device=device)
    full = cur.clone()
    finished = torch.zeros(bs, dtype=torch.bool, device=device)
    n_new = seq_len - cur.shape[1]
    past = None
    for pos in range(n_new):
        out = model(input_ids=cur, past_key_values=past, use_cache=True)
        past = out.past_key_values
        logits = out.logits[:, -1].float()
        if pos < greedy_prefix:
            nxt = logits.argmax(dim=-1, keepdim=True)
        elif top_k and top_k > 0:
            k = min(top_k, logits.size(-1))
            vals, idx = logits.topk(k, dim=-1)
            probs = torch.softmax(vals / temp, dim=-1)
            nxt = idx.gather(-1, torch.multinomial(probs, num_samples=1, generator=gen))
        else:
            nxt = torch.multinomial(torch.softmax(logits / temp, dim=-1), num_samples=1, generator=gen)
        nxt = torch.where(finished.unsqueeze(1), torch.full_like(nxt, eos), nxt)
        full = torch.cat([full, nxt], dim=1)
        finished |= nxt.squeeze(1) == eos
        cur = nxt
        if bool(finished.all()):
            break
    rows = []
    for i in range(bs):
        row = full[i].tolist()
        if eos in row[len(prompts[i]):]:
            row = row[: row.index(eos, len(prompts[i]))] + [eos]
        rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(Path.home() / "models" / "LFM2.5-350M-Base"))
    ap.add_argument("--out", default="runs/exp1/teacher_lfm25_350m_base.jsonl")
    ap.add_argument("--num-seq", type=int, default=10000)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--bucket-repeat", type=int, default=1,
                    help="chat mode: replicate each equal-length prompt bucket this many "
                         "times — bigger batches without padding (conv hybrids require "
                         "exactly-equal prompt lengths; sampling temp varies the "
                         "continuations across replicas)")
    ap.add_argument("--greedy-prefix", type=int, default=4)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=0, help="0 => full vocab sampling")
    ap.add_argument("--vocab", type=int, default=None,
                    help="sampling vocab size; default = len(tokenizer)")
    ap.add_argument("--eos", type=int, default=None,
                    help="EOS id to truncate at; default = tokenizer.eos_token_id")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chat", action="store_true",
                    help="prompt-seeded chat-template sampling (instruct teachers)")
    ap.add_argument("--prompts-file", default="runs/exp2/shared/prompts.jsonl")
    args = ap.parse_args()

    device = pick_device()
    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(device)
    model.eval()

    vocab = args.vocab if args.vocab is not None else len(tokenizer)
    eos = args.eos if args.eos is not None else tokenizer.eos_token_id
    # Ground every sampled sequence at <bos>: a BOS-less start un-anchors the
    # model and yields off-distribution teacher data (Gemma4 lesson).
    bos = tokenizer.bos_token_id

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    buckets = _chat_buckets(tokenizer, args.prompts_file) if args.chat else None
    if buckets and args.bucket_repeat > 1:
        for L in buckets:
            buckets[L] = buckets[L] * args.bucket_repeat
        print(f"bucket-repeat x{args.bucket_repeat}: top bucket now "
              f"{max(len(b) for b in buckets.values())} prompts", flush=True)
    lengths = sorted(buckets, key=lambda L: -len(buckets[L])) if buckets else []
    if buckets:
        print(f"chat buckets: {len(buckets)} lengths, top sizes "
              f"{[len(buckets[L]) for L in lengths[:6]]}", flush=True)

    written = 0
    n_tokens = 0
    t0 = time.time()
    gen = torch.Generator(device=device).manual_seed(args.seed)
    with open(out_path, "w") as f:
        while written < args.num_seq:
            if buckets:
                # cycle buckets largest-first; batch = whole bucket (equal length)
                L = lengths[written % len(lengths)]
                prompts = buckets[L]
                rows = _sample_chat_batch(
                    model, prompts, seq_len=args.seq_len, device=device,
                    greedy_prefix=args.greedy_prefix, temp=args.temp, eos=eos if eos is not None else -1,
                    vocab=vocab, gen=gen)
                for row in rows:
                    if len(row) >= 16:
                        f.write(json.dumps({"ids": row}) + "\n")
                        n_tokens += len(row)
                written += len(prompts)
                rate = n_tokens / (time.time() - t0)
                print(f"{written} seqs (chat), {n_tokens:,} tokens ({rate:.0f} tok/s)", flush=True)
                continue
            bs = min(args.batch, args.num_seq - written)
            starts = torch.randint(0, vocab, (bs, 1), device=device)
            if bos is not None:
                bos_col = torch.full((bs, 1), bos, dtype=starts.dtype, device=device)
                starts = torch.cat([bos_col, starts], dim=1)
            seqs = starts
            past = None
            cur = starts
            for pos in range(args.seq_len - 2):
                with torch.no_grad():
                    out = model(input_ids=cur, past_key_values=past, use_cache=True)
                    past = out.past_key_values
                    logits = out.logits[:, -1].float()
                if pos < args.greedy_prefix:
                    nxt = logits.argmax(dim=-1, keepdim=True)
                elif args.top_k and args.top_k > 0:
                    k = min(args.top_k, logits.size(-1))
                    vals, idx = logits.topk(k, dim=-1)
                    probs = torch.softmax(vals / args.temp, dim=-1)
                    nxt = idx.gather(-1, torch.multinomial(probs, num_samples=1))
                else:
                    probs = torch.softmax(logits / args.temp, dim=-1)
                    nxt = torch.multinomial(probs, num_samples=1)

                seqs = torch.cat([seqs, nxt], dim=1)
                cur = nxt

            for row in seqs.tolist():
                if eos in row:
                    row = row[: row.index(eos)]
                if len(row) >= 8:
                    f.write(json.dumps({"ids": row}) + "\n")
                    n_tokens += len(row)
            written += bs
            rate = n_tokens / (time.time() - t0)
            print(f"{written}/{args.num_seq} seqs, {n_tokens:,} tokens ({rate:.0f} tok/s)", flush=True)

    print(f"done: {written} seqs, {n_tokens:,} tokens -> {out_path}")


if __name__ == "__main__":
    main()
