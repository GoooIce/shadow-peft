"""
Compare 1-bit MLX models on Qwen3-1.7B scale:

  1. Qwen/Qwen3-1.7B (fp16)                     — upper bound
  2. Qwen/Qwen3-1.7B + shadow_peft_mlx 1-bit PTQ — naive packing lower bound
  3. prism-ml/Bonsai-1.7B-mlx-1bit              — trained 1-bit reference

Metrics: wikitext-2 test perplexity + greedy generation on fixed prompts.
"""

from __future__ import annotations

import gc
import math

import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler

from shadow_peft_mlx import quantize_model_1bit

CHUNK = 512
MAX_CHUNKS = 40  # ~20k tokens, enough for a stable comparison

PROMPTS = [
    "The capital of France is",
    "In a shocking finding, scientists discovered a herd of unicorns living in",
    "def fibonacci(n):",
]


def perplexity(model, tokenizer, text: str) -> float:
    ids = tokenizer.encode(text)
    total_nll, total_tok = 0.0, 0
    limit = min(len(ids), CHUNK * MAX_CHUNKS) - CHUNK - 1
    for start in range(0, limit, CHUNK):
        x = mx.array(ids[start : start + CHUNK + 1])[None]
        logits = model(x).astype(mx.float32)
        logp = logits[:, :-1] - mx.logsumexp(logits[:, :-1], axis=-1, keepdims=True)
        tgt = x[:, 1:]
        nll = -mx.take_along_axis(logp, tgt[..., None], axis=-1).sum()
        total_nll += nll.item()
        total_tok += tgt.size
    return math.exp(total_nll / total_tok)


def sample_generations(model, tokenizer) -> list[str]:
    sampler = make_sampler(temp=0.0)
    outs = []
    for prompt in PROMPTS:
        chunks = [
            r.text
            for r in stream_generate(
                model, tokenizer, prompt=prompt, max_tokens=40, sampler=sampler
            )
        ]
        outs.append("".join(chunks))
    return outs


def evaluate(name: str, model, tokenizer, text: str) -> None:
    ppl = perplexity(model, tokenizer, text)
    gens = sample_generations(model, tokenizer)
    print(f"\n=== {name} ===", flush=True)
    print(f"  PPL(wikitext-2, {CHUNK * MAX_CHUNKS} tokens): {ppl:.2f}")
    for prompt, out in zip(PROMPTS, gens, strict=True):
        print(f"  > {prompt!r}\n    {out!r}")


def unload(model) -> None:
    del model
    gc.collect()
    mx.metal.clear_cache()


def main() -> None:
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n".join(line for line in ds["text"] if line.strip())[:200_000]

    # 1. fp16 upper bound
    model, tokenizer = load("Qwen/Qwen3-1.7B")
    evaluate("Qwen3-1.7B fp16 (upper bound)", model, tokenizer, text)

    # 2. naive 1-bit PTQ lower bound (same arch, quantized in place)
    manifest = quantize_model_1bit(model, group_size=128)
    print(f"\n[quantized {len(manifest['quantized'])} modules to 1-bit]", flush=True)
    evaluate("Qwen3-1.7B + 1-bit PTQ (ours, lower bound)", model, tokenizer, text)
    unload(model)

    # 3. Bonsai trained 1-bit
    bonsai, bonsai_tok = load("prism-ml/Bonsai-1.7B-mlx-1bit")
    evaluate("Bonsai-1.7B-mlx-1bit (trained 1-bit)", bonsai, bonsai_tok, text)
    unload(bonsai)


if __name__ == "__main__":
    main()
