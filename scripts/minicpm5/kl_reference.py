#!/usr/bin/env python3
"""BF16 reference logprobs for the ticket-14 pilot KL monitor (ticket 05).

Re-runs ticket 11's probe protocol on the **BF16** engine (no quantization):
greedy-decode 8 frozen 64-token prompts for 32 tokens, then teacher-force
``prompt + continuation`` and record the per-token logprob of every continuation
token.  The pilot then monitors ``Generation KL Error = mean exp(|logp_NVFP4 -
logp_BF16|)`` on exactly those continuations, and the q0 value must reproduce
ticket 11's registered 2.112.

Usage (gooo / goooinext):
    python kl_reference.py --model <bf16 dir> --probes <probes.json> \
        --continuations <continuations_bf16.json> --out <bf16_kl.json>
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--probes", required=True)
    parser.add_argument("--continuations", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--linear-backend", default="cutlass")
    parser.add_argument("--max-model-len", type=int, default=2049)
    parser.add_argument("--gpu-mem", type=float, default=0.30)
    parser.add_argument("--skip-decode-check", action="store_true")
    args = parser.parse_args()

    from vllm import LLM, SamplingParams

    probes = json.loads(Path(args.probes).expanduser().read_text())
    continuations = json.loads(Path(args.continuations).expanduser().read_text())

    started = time.perf_counter()
    kwargs = dict(
        model=args.model,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem,
        enable_prefix_caching=False,
        trust_remote_code=True,
        disable_log_stats=True,
    )
    if args.linear_backend:
        kwargs["linear_backend"] = args.linear_backend
    llm = LLM(**kwargs)
    load_seconds = round(time.perf_counter() - started, 2)
    print(f"[kl-ref] engine loaded in {load_seconds}s", flush=True)

    report = {"model": args.model, "engine_load_seconds": load_seconds}

    if not args.skip_decode_check:
        sampling = SamplingParams(temperature=0.0, max_tokens=probes["decode_tokens"],
                                  logprobs=1, detokenize=False, ignore_eos=True)
        outputs = llm.generate(
            [{"prompt_token_ids": list(ids)} for ids in probes["decode_prompts"]], sampling,
            use_tqdm=False,
        )
        mismatches = []
        for index, output in enumerate(outputs):
            mine = list(output.outputs[0].token_ids)
            theirs = list(continuations[str(index)])
            if mine != theirs:
                mismatches.append({"index": index, "first": next(
                    (position for position, (a, b) in enumerate(zip(mine, theirs)) if a != b), None)})
        report["decode_check"] = {"prompts": len(outputs), "mismatches": mismatches}
        print(f"[kl-ref] decode check: {len(mismatches)} mismatching prompts", flush=True)

    requests = []
    spans = []
    for index, prompt_ids in enumerate(probes["decode_prompts"]):
        sequence = list(prompt_ids) + list(continuations[str(index)])
        requests.append({"prompt_token_ids": sequence})
        spans.append((len(prompt_ids), len(sequence)))
    sampling = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0, detokenize=False)
    started = time.perf_counter()
    outputs = llm.generate(requests, sampling, use_tqdm=False)
    logprobs = {}
    for index, output in enumerate(outputs):
        start, end = spans[index]
        row = []
        for position in range(start, end):
            entry = output.prompt_logprobs[position]
            token_id = output.prompt_token_ids[position]
            row.append(float(entry[token_id].logprob) if entry and token_id in entry else None)
        logprobs[str(index)] = row
    report["logprobs"] = logprobs
    report["tokens"] = sum(len(row) for row in logprobs.values())
    report["seconds"] = round(time.perf_counter() - started, 3)
    report["wall_seconds"] = round(time.perf_counter() - started, 3)
    report["continuations"] = continuations
    report["prompts"] = probes["decode_prompts"]
    Path(args.out).expanduser().write_text(json.dumps(report) + "\n")
    print(f"[kl-ref] wrote {args.out}: {report['tokens']} continuation tokens", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
