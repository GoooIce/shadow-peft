"""
PPL benchmark for Lloyd-refined 1-bit packing (P4).

Same wikitext-2 slice and protocol as compare_bonsai_1bit.py:
  fp16 / naive 1-bit PTQ / refined PTQ (iters = 1, 5, 10) / Bonsai trained 1-bit.

Run: .venv/bin/python experiment/benchmark_refine_1bit.py
"""

from __future__ import annotations

import gc
import sys
from pathlib import Path

import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler

sys.path.insert(0, str(Path(__file__).parent))
from compare_bonsai_1bit import PROMPTS, perplexity  # noqa: E402

from shadow_peft_mlx import quantize_model_1bit  # noqa: E402

REFINE_ITERS = (0, 1, 5, 10)


def sample_generation(model, tokenizer, prompt: str) -> str:
    sampler = make_sampler(temp=0.0)
    return "".join(
        r.text
        for r in stream_generate(
            model, tokenizer, prompt=prompt, max_tokens=30, sampler=sampler
        )
    )


def main() -> None:
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n".join(line for line in ds["text"] if line.strip())[:200_000]

    print("\n| model | PPL |", flush=True)
    print("|---|---:|")

    model, tokenizer = load("Qwen/Qwen3-1.7B")
    ppl = perplexity(model, tokenizer, text)
    print(f"| Qwen3-1.7B fp16 | {ppl:.2f} |", flush=True)
    del model
    gc.collect()
    mx.clear_cache()

    for iters in REFINE_ITERS:
        model, tokenizer = load("Qwen/Qwen3-1.7B")
        quantize_model_1bit(model, group_size=128, refine_iters=iters)
        ppl = perplexity(model, tokenizer, text)
        gen = sample_generation(model, tokenizer, PROMPTS[0])
        print(f"| 1-bit PTQ refine_iters={iters} | {ppl:.2f} |", flush=True)
        if iters == REFINE_ITERS[-1]:
            print(f"\ngeneration @iters={iters}: {gen!r}", flush=True)
        del model
        gc.collect()
        mx.clear_cache()

    bonsai, bonsai_tok = load("prism-ml/Bonsai-1.7B-mlx-1bit")
    ppl = perplexity(bonsai, bonsai_tok, text)
    print(f"| Bonsai-1.7B (trained) | {ppl:.2f} |", flush=True)


if __name__ == "__main__":
    main()
