"""
PPL benchmark for ternary 2-bit PTQ (torch).

Compares wikitext-2 perplexity across configurations:
  fp16 / ternary 2-bit PTQ (refine_iters = 0, 8) / (optionally) Bonsai ternary.

The 1-bit analysis showed naive PTQ PPL is astronomical (~3.4×10^5 refined).
Ternary (3 levels vs 2) should be strictly better but still non-functional,
confirming that the "free accuracy ceiling" is reached — recovery must come
from training (shadow + distillation).

Run: .venv/bin/python experiment/benchmark_2bit_ptq.py
     .venv/bin/python experiment/benchmark_2bit_ptq.py --model /path/to/local/model
"""

from __future__ import annotations

import argparse
import gc
import math

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from shadow_peft.quantization import quantize_model_2bit

MODEL = "Qwen/Qwen3-1.7B"
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
DTYPE = torch.float16
CHUNK = 512
MAX_CHUNKS = 40  # ~20k tokens


@torch.no_grad()
def perplexity(model, tokenizer, text: str) -> float:
    ids = tokenizer.encode(text)
    model.eval()
    total_nll, total_tok = 0.0, 0
    limit = min(len(ids), CHUNK * MAX_CHUNKS) - CHUNK - 1
    for start in range(0, limit, CHUNK):
        x = torch.tensor(ids[start : start + CHUNK + 1], device=DEVICE)[None]
        logits = model(input_ids=x).logits.float()
        logp = logits[:, :-1] - torch.logsumexp(logits[:, :-1], -1, keepdim=True)
        tgt = x[:, 1:]
        total_nll += -logp.gather(-1, tgt.unsqueeze(-1)).sum().item()
        total_tok += tgt.numel()
    return math.exp(total_nll / total_tok)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL, help="HF model id or local path")
    ap.add_argument(
        "--refine-iters", type=int, nargs="+", default=[0, 8],
        help="refine_iters to benchmark (default: 0 8)",
    )
    ap.add_argument("--group-size", type=int, default=128)
    args = ap.parse_args()

    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n".join(line for line in ds["text"] if line.strip())[:200_000]

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    print("\n| model | PPL |", flush=True)
    print("|---|---:|", flush=True)

    # fp16 baseline
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=DTYPE).to(DEVICE)
    ppl = perplexity(model, tokenizer, text)
    print(f"| {args.model} fp16 | {ppl:.2f} |", flush=True)
    del model
    gc.collect()
    if DEVICE == "mps":
        torch.mps.empty_cache()

    # Ternary 2-bit PTQ at various refine_iters
    for iters in args.refine_iters:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=DTYPE).to(DEVICE)
        manifest = quantize_model_2bit(
            model, group_size=args.group_size, refine_iters=iters
        )
        n_quant = len(manifest["quantized"])
        model.eval()
        ppl = perplexity(model, tokenizer, text)
        print(f"| ternary 2-bit PTQ refine={iters} ({n_quant} modules) | {ppl:.2f} |", flush=True)
        del model
        gc.collect()
        if DEVICE == "mps":
            torch.mps.empty_cache()


if __name__ == "__main__":
    main()
