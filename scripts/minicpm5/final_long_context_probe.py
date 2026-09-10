#!/usr/bin/env python3
"""Ticket 15: long-context (131k) boundary probe for the deployed NVFP4 centre.

Ticket 15's question asks for the long-context / KV-cache boundary of the
delivered checkpoint.  This probe sends a single prompt of the checkpoint's
advertised maximum length (131,072 tokens by construction, built from a
repeated filler sentence) with ``max_tokens=1`` and records:

* whether the engine accepted it at all (a length rejection is itself the
  boundary answer at this ``--gpu-memory-utilization``);
* the prompt token count actually tokenised, the wall time, and therefore the
  long-context prefill rate;
* the engine's own KV-cache accounting lines from the serve log (KV cache
  tokens, concurrency for a full-length request, activation peaks).

Usage:
    python final_long_context_probe.py --port 8133 --out <json> --target-tokens 131072
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import requests

SENTENCE = "The quick brown fox jumps over the lazy dog. "


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--serve-log", required=True)
    parser.add_argument("--target-tokens", type=int, default=131072)
    parser.add_argument("--max-tokens", type=int, default=1)
    args = parser.parse_args()

    base = f"http://127.0.0.1:{args.port}"
    report: dict = {"port": args.port, "target_tokens": args.target_tokens}
    report["models"] = requests.get(f"{base}/v1/models", timeout=60).json()

    # Build the filler by tokenisation feedback: the probe measures with the
    # server's own tokeniser via /tokenize, so the request really carries
    # ~target_tokens prompt tokens (tokenizer.json is 9.9 MB; chunked calls).
    def count(text: str) -> int:
        response = requests.post(f"{base}/tokenize", json={"model": "minicpm5", "prompt": text},
                                 timeout=300)
        response.raise_for_status()
        return response.json()["count"]

    suffix = "\nRepeat the last sentence back verbatim."

    def build(repeats: int) -> str:
        return SENTENCE * repeats + suffix

    low, high = 1, 40000           # repeats of SENTENCE (~11 tokens each)
    while low < high:
        middle = (low + high + 1) // 2
        if count(build(middle)) <= args.target_tokens:
            low = middle
        else:
            high = middle - 1
    repeats = low
    report["filler_sentence"] = SENTENCE
    report["filler_repeats"] = repeats
    report["suffix"] = suffix
    prompt = build(repeats)
    report["prompt_tokens_measured"] = count(prompt)
    report["fits_max_model_len"] = report["prompt_tokens_measured"] <= args.target_tokens

    started = time.perf_counter()
    try:
        response = requests.post(f"{base}/v1/completions", json={
            "model": "minicpm5", "prompt": prompt, "max_tokens": args.max_tokens,
            "temperature": 0.0,
        }, timeout=1800)
        elapsed = time.perf_counter() - started
        response.raise_for_status()
        body = response.json()
        report["accepted"] = True
        report["usage"] = body.get("usage")
        report["text"] = body["choices"][0].get("text")
        report["finish_reason"] = body["choices"][0].get("finish_reason")
        report["wall_seconds"] = round(elapsed, 3)
        report["prefill_tokens_per_second"] = round(
            (body.get("usage") or {}).get("prompt_tokens", 0) / elapsed, 2
        ) if elapsed else None
    except Exception as error:
        report["accepted"] = False
        report["error"] = repr(error)
        report["wall_seconds"] = round(time.perf_counter() - started, 3)

    report["free_after_request"] = subprocess.run(
        ["free", "-b"], capture_output=True, text=True, check=True
    ).stdout.splitlines()
    log = Path(args.serve_log)
    if log.exists():
        keys = ("KV cache", "GPU KV cache", "concurrency", "Activation", "activation",
                "Available KV", "peak", "NVFP4 GEMM")
        report["serve_log_interesting"] = [
            line for line in log.read_text(errors="replace").splitlines()
            if any(key.lower() in line.lower() for key in keys)
        ][-40:]

    destination = Path(args.out).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=1, sort_keys=True, ensure_ascii=False) + "\n")
    print(json.dumps({key: report.get(key) for key in
                      ("accepted", "prompt_tokens_measured", "wall_seconds",
                       "prefill_tokens_per_second", "usage", "finish_reason", "error")},
                     indent=1, ensure_ascii=False))
    print(f"[long-context] wrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
