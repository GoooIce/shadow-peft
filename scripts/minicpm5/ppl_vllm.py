#!/usr/bin/env python3
"""Perplexity of a vLLM-served checkpoint on a fixed tokenised corpus.

Uses ``prompt_logprobs`` so both the BF16 baseline and the NVFP4 payload are
measured through the *same* serving runtime, tokenizer and token ids.

Usage:
    python ppl_vllm.py --model <path> --eval-jsonl <file> --label bf16 \
        --out <dir> [--quantization compressed-tensors] [--limit 64]
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from vllm import LLM, SamplingParams


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--eval-jsonl", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.30)
    parser.add_argument("--quantization", default=None)
    # Added for the W8A8 ticket: INT8 W8A8 has no working CUTLASS kernel on
    # SM121, so the serving arm must name `triton` explicitly.  Default None
    # keeps the ticket-10 NVFP4/BF16 invocations byte-identical.
    parser.add_argument("--linear-backend", default=None)
    parser.add_argument("--max-num-seqs", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0, help="0 = all sequences")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    sequences = []
    with Path(args.eval_jsonl).expanduser().open() as fh:
        for line in fh:
            if line.strip():
                sequences.append(json.loads(line)["input_ids"])
    if args.limit:
        sequences = sequences[: args.limit]
    print(f"[ppl] {args.label}: {len(sequences)} sequences, "
          f"{sum(len(s) for s in sequences)} tokens", flush=True)

    started = time.time()
    llm_kwargs = {}
    if args.linear_backend:
        llm_kwargs["linear_backend"] = args.linear_backend
    llm = LLM(
        model=args.model,
        quantization=args.quantization,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        enforce_eager=args.enforce_eager,
        seed=args.seed,
        disable_log_stats=True,
        **llm_kwargs,
    )
    load_seconds = time.time() - started

    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        prompt_logprobs=0,
        detokenize=False,
    )

    total_nll = 0.0
    total_tokens = 0
    per_sequence = []
    infer_started = time.time()
    outputs = llm.generate(
        [{"prompt_token_ids": ids} for ids in sequences], sampling
    )
    infer_seconds = time.time() - infer_started

    for index, output in enumerate(outputs):
        ids = output.prompt_token_ids
        prompt_logprobs = output.prompt_logprobs
        seq_nll = 0.0
        seq_tokens = 0
        for position in range(1, len(ids)):
            entry = prompt_logprobs[position]
            if not entry:
                continue
            token_id = ids[position]
            logprob = entry.get(token_id)
            if logprob is None:
                # prompt_logprobs may only carry the sampled token; skip if absent.
                continue
            seq_nll += -float(logprob.logprob)
            seq_tokens += 1
        total_nll += seq_nll
        total_tokens += seq_tokens
        per_sequence.append(
            {
                "index": index,
                "tokens": seq_tokens,
                "mean_nll": seq_nll / seq_tokens if seq_tokens else None,
                "ppl": math.exp(seq_nll / seq_tokens) if seq_tokens else None,
            }
        )

    result = {
        "label": args.label,
        "model": args.model,
        "quantization": args.quantization,
        "linear_backend": args.linear_backend,
        "eval_jsonl": str(Path(args.eval_jsonl).expanduser()),
        "sequences": len(sequences),
        "scored_tokens": total_tokens,
        "mean_nll": total_nll / total_tokens if total_tokens else None,
        "ppl": math.exp(total_nll / total_tokens) if total_tokens else None,
        "engine_load_seconds": round(load_seconds, 2),
        "inference_seconds": round(infer_seconds, 2),
        "prefill_tokens_per_second": (
            round(total_tokens / infer_seconds, 1) if infer_seconds else None
        ),
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enforce_eager": args.enforce_eager,
        "per_sequence": per_sequence,
    }
    path = out_dir / f"ppl_{args.label}.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "per_sequence"},
                     indent=2, sort_keys=True))
    print(f"[ppl] wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
