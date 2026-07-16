"""
ShadowPEFT KV-Cache Benchmark.

比较 use_cache=True vs use_cache=False 在生成不同数量 token 时的延迟和加速比。

用法:
    .venv/bin/python experiment/benchmark_kv_cache.py
    .venv/bin/python experiment/benchmark_kv_cache.py --prompt-lens 8 32 --new-tokens 16 64 --models llama qwen2 qwen3_5
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import time
from pathlib import Path

import torch
from transformers import PreTrainedModel

from shadow_peft import ShadowConfig, ShadowForCausalLM, get_shadow_model

# ---------------------------------------------------------------------------
# Tiny model factories
# ---------------------------------------------------------------------------


def _tiny_llama(num_layers: int = 8, hidden_size: int = 64) -> PreTrainedModel:
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(
        vocab_size=256,
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 2,
        num_hidden_layers=num_layers,
        num_attention_heads=8,
        num_key_value_heads=4,
        max_position_embeddings=1024,
    )
    return LlamaForCausalLM(cfg)


def _tiny_qwen2(num_layers: int = 8, hidden_size: int = 64) -> PreTrainedModel:
    from transformers import Qwen2Config, Qwen2ForCausalLM

    cfg = Qwen2Config(
        vocab_size=256,
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 2,
        num_hidden_layers=num_layers,
        num_attention_heads=8,
        num_key_value_heads=4,
        max_position_embeddings=1024,
    )
    return Qwen2ForCausalLM(cfg)


def _tiny_qwen3_5(num_layers: int = 8, hidden_size: int = 64) -> PreTrainedModel:
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    head_dim = hidden_size // 8
    layer_types = [
        "linear_attention" if i % 4 != 3 else "full_attention" for i in range(num_layers)
    ]
    cfg = Qwen3_5TextConfig(
        vocab_size=256,
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 2,
        num_hidden_layers=num_layers,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=head_dim,
        linear_num_value_heads=4,
        linear_value_head_dim=head_dim,
        linear_num_key_heads=4,
        linear_key_head_dim=head_dim,
        linear_conv_kernel_dim=4,
        layer_types=layer_types,
        max_position_embeddings=1024,
    )
    return Qwen3_5ForCausalLM(cfg)


MODEL_BUILDERS = {
    "llama": ("Llama-8L", _tiny_llama),
    "qwen2": ("Qwen2-8L", _tiny_qwen2),
    "qwen3_5": ("Qwen3.5-8L (GDN)", _tiny_qwen3_5),
}


# ---------------------------------------------------------------------------
# Benchmark core
# ---------------------------------------------------------------------------


def _build_shadow(base: PreTrainedModel) -> ShadowForCausalLM:
    torch.manual_seed(42)
    cfg = ShadowConfig(
        num_shadow_layers=2,
        injection_hidden_size=16,
        gate_hidden_size=20,
        alpha=0.1,
        dropout=0.0,
    )
    peft = get_shadow_model(base, cfg)
    return ShadowForCausalLM(peft, inference_mode="base_shadow").eval()


def _timed_generate(
    model: ShadowForCausalLM,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    use_cache: bool,
) -> float:
    """Run generate() and return wall-clock time in milliseconds."""
    model.peft_model._shadow_cache = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start = time.perf_counter()
    with torch.no_grad():
        model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            use_cache=use_cache,
            do_sample=False,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) * 1000
    return elapsed


def benchmark_single(
    model_key: str,
    prompt_len: int,
    max_new_tokens: int,
    *,
    warmup: int = 2,
    runs: int = 5,
) -> dict:
    """Benchmark one (model, prompt_len, new_tokens) configuration."""
    label, builder = MODEL_BUILDERS[model_key]
    torch.manual_seed(42)
    base = builder()
    model = _build_shadow(base)

    input_ids = torch.randint(0, base.config.vocab_size, (1, prompt_len))

    # Warmup (discard timings).
    for _ in range(warmup):
        _timed_generate(model, input_ids, max_new_tokens, use_cache=False)
        _timed_generate(model, input_ids, max_new_tokens, use_cache=True)

    # Timed runs.
    nocache_times = [_timed_generate(model, input_ids, max_new_tokens, use_cache=False) for _ in range(runs)]
    cache_times = [_timed_generate(model, input_ids, max_new_tokens, use_cache=True) for _ in range(runs)]

    nocache_med = statistics.median(nocache_times)
    cache_med = statistics.median(cache_times)
    speedup = nocache_med / cache_med if cache_med > 0 else float("inf")

    return {
        "model": label,
        "prompt_len": prompt_len,
        "new_tokens": max_new_tokens,
        "nocache_ms": round(nocache_med, 1),
        "cache_ms": round(cache_med, 1),
        "speedup": round(speedup, 2),
    }


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------


def print_results_table(results: list[dict]) -> None:
    """Print a formatted results table."""
    header = (
        f"{'Model':<20} {'Prompt':>7} {'NewTok':>7} "
        f"{'NoCache(ms)':>12} {'Cache(ms)':>12} {'Speedup':>8}"
    )
    print(f"\n{'=' * 72}")
    print("ShadowPEFT KV-Cache Benchmark Results")
    print(f"{'=' * 72}")
    print(header)
    print("-" * 72)
    for r in results:
        print(
            f"{r['model']:<20} {r['prompt_len']:>7} {r['new_tokens']:>7} "
            f"{r['nocache_ms']:>12.1f} {r['cache_ms']:>12.1f} {r['speedup']:>7.2f}x"
        )
    print("=" * 72)


def print_speedup_chart(results: list[dict]) -> None:
    """Print a simple ASCII bar chart of speedups."""
    print("\nSpeedup by configuration (cache vs no-cache):")
    print("-" * 60)
    max_speedup = max(r["speedup"] for r in results) if results else 1
    for r in results:
        bar_len = int((r["speedup"] / max_speedup) * 35) if max_speedup > 0 else 0
        bar = "#" * bar_len
        tag = f"{r['model']} p{r['prompt_len']} t{r['new_tokens']}"
        print(f"  {tag:<30} {r['speedup']:>5.2f}x {bar}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark ShadowPEFT KV-cache speedup")
    parser.add_argument(
        "--models",
        nargs="*",
        default=["llama", "qwen2", "qwen3_5"],
        choices=list(MODEL_BUILDERS.keys()),
    )
    parser.add_argument("--prompt-lens", nargs="*", type=int, default=[4, 16, 32])
    parser.add_argument("--new-tokens", nargs="*", type=int, default=[16, 64])
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--output-json", type=str, default=None, help="Save results as JSON")
    args = parser.parse_args()

    device = "GPU" if torch.cuda.is_available() else "CPU"
    print(f"Device: {device}")
    print(f"Warmup: {args.warmup} runs, Timed: {args.runs} runs (median)")

    all_results: list[dict] = []
    for model_key in args.models:
        for prompt_len in args.prompt_lens:
            for new_tokens in args.new_tokens:
                label = MODEL_BUILDERS[model_key][0]
                print(f"  Benchmarking {label} prompt={prompt_len} new_tokens={new_tokens}...", flush=True)
                r = benchmark_single(
                    model_key, prompt_len, new_tokens, warmup=args.warmup, runs=args.runs
                )
                all_results.append(r)

    print_results_table(all_results)
    print_speedup_chart(all_results)

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
        print(f"Results saved to {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
