#!/usr/bin/env python3
"""Ticket 15: end-to-end serving probe for the deployed NVFP4 W4A4 centre.

Runs against an already-serving vLLM instance (the one ``kernel_evidence.py``
leaves behind with ``--keep-server``) and records, in one JSON:

* served-model identity (``/v1/models``) and the checkpoint directory;
* a small end-to-end generation probe (chat, ``enable_thinking=false``) with the
  generated text and token accounting -- the "does it actually answer" check;
* decode throughput from a streaming request (TTFT, inter-token interval);
* prefill rate from a long-prompt request;
* aggregate throughput over concurrent requests;
* memory accounting from ``free -b`` snapshots, ``/proc/<pid>/status`` RSS of the
  serve + EngineCore processes, and the vLLM startup lines that report the
  checkpoint size and the KV-cache allocation (GB10 note: ``nvidia-smi`` memory
  columns are ``[N/A]`` on this platform, so unified-memory accounting has to
  come from ``free -b`` / procfs / the engine log).

Usage:
    python final_serve_probe.py --port 8131 --out <json> --label centre-cutlass
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

import requests  # vLLM's own dependency; present in the serving venv


def post(port: int, path: str, payload: dict, timeout: float = 300.0) -> dict:
    response = requests.post(f"http://127.0.0.1:{port}{path}", json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


def get(port: int, path: str, timeout: float = 30.0) -> dict:
    response = requests.get(f"http://127.0.0.1:{port}{path}", timeout=timeout)
    response.raise_for_status()
    return response.json()


def free_bytes() -> dict:
    out = subprocess.run(["free", "-b"], capture_output=True, text=True, check=True).stdout
    lines = [line.split() for line in out.splitlines() if line.strip()]
    values = lines[1]
    # the body line is localised in its first label ("内存：" etc.), so keep the
    # numeric fields only: total used free shared buff/cache available
    numbers = [int(token) for token in values if token.isdigit()]
    keys = ["total", "used", "free", "shared", "buff_cache", "available"]
    return {key: numbers[index] for index, key in enumerate(keys) if index < len(numbers)}


def proc_rss() -> dict:
    result = {}
    listing = subprocess.run(["ps", "-eo", "pid,comm,rss"], capture_output=True, text=True).stdout
    for line in listing.splitlines()[1:]:
        parts = line.split(None, 2)
        if len(parts) != 3:
            continue
        pid, comm, rss_kb = parts
        if "vllm" in comm.lower() or "EngineCore" in comm or "VLLM" in comm:
            result[pid] = {"comm": comm, "rss_kb": int(rss_kb)}
    return result


def streaming_decode(port: int, prompt: str, max_tokens: int) -> dict:
    started = time.perf_counter()
    first = None
    tokens = 0
    response = requests.post(
        f"http://127.0.0.1:{port}/v1/completions",
        json={"model": "minicpm5", "prompt": prompt, "max_tokens": max_tokens,
              "temperature": 0.0, "stream": True},
        stream=True, timeout=300.0,
    )
    response.raise_for_status()
    for raw in response.iter_lines():
        if not raw:
            continue
        line = raw.decode() if isinstance(raw, bytes) else raw
        if not line.startswith("data: "):
            continue
        body = line[6:].strip()
        if body == "[DONE]":
            break
        chunk = json.loads(body)
        if chunk.get("choices") and chunk["choices"][0].get("text") is not None:
            if first is None:
                first = time.perf_counter()
            tokens += 1
    ended = time.perf_counter()
    return {
        "output_tokens": tokens,
        "ttft_seconds": None if first is None else round(first - started, 4),
        "total_seconds": round(ended - started, 4),
        "decode_tokens_per_second": (
            None if first is None or tokens < 2 else round((tokens - 1) / (ended - first), 2)
        ),
    }


def log_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line.rstrip("\n") for line in path.read_text(errors="replace").splitlines()]


INTERESTING = re.compile(
    r"KV cache|kv_cache|checkpoint size|Memory profiling|memory profiling|"
    r"init engine|CUDA graph|graph capture|torch.compile|Using .* for NVFP4 GEMM|"
    r"Available KV|concurrency|GiB|Warming up|compile_or_warm_up",
    re.IGNORECASE,
)


SUITE = [
    "What is 2 + 5? Answer with the number only.",
    "What is 2 + 5 - 10? Answer with the number only.",
    "What is 7 * 8? Answer with the number only.",
    "What is 29 modulo 5? Answer with the number only.",
    "What is the capital of France? Answer with the city name only.",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--serve-log", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()

    report: dict = {"label": args.label, "port": args.port, "serve_log": args.serve_log}
    report["free_before"] = free_bytes()
    report["models"] = get(args.port, "/v1/models")
    report["model_dir"] = args.model_dir

    # --- end-to-end generation probe (short answer, thinking off) -------------
    probe = post(args.port, "/v1/chat/completions", {
        "model": "minicpm5",
        "messages": [{"role": "user", "content": "What is the value of 2 + 5 - 10? Answer with the number only."}],
        "max_tokens": 48,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    })
    choice = probe["choices"][0]
    report["generation_probe"] = {
        "request": {"prompt": "What is the value of 2 + 5 - 10? Answer with the number only.",
                    "max_tokens": 48, "temperature": 0.0, "enable_thinking": False},
        "text": choice["message"]["content"],
        "finish_reason": choice["finish_reason"],
        "usage": probe.get("usage"),
    }

    # --- a small generation suite (thinking off) ------------------------------
    # The frozen task pool is deliberately balanced 50/50 on items q0 gets right
    # and wrong, so a single prompt cannot distinguish "wrong answer" from
    # "broken deployment"; the suite gives the deployment a fair reading.
    suite = []
    for text in SUITE:
        started = time.perf_counter()
        try:
            body = post(args.port, "/v1/chat/completions", {
                "model": "minicpm5",
                "messages": [{"role": "user", "content": text}],
                "max_tokens": 48, "temperature": 0.0,
                "chat_template_kwargs": {"enable_thinking": False},
            })
            row = {"prompt": text, "text": body["choices"][0]["message"]["content"],
                   "finish_reason": body["choices"][0]["finish_reason"],
                   "usage": body.get("usage"), "seconds": round(time.perf_counter() - started, 3)}
        except Exception as error:
            row = {"prompt": text, "error": repr(error)}
        suite.append(row)
    report["generation_suite"] = suite

    # --- decode throughput ----------------------------------------------------
    report["decode"] = streaming_decode(
        args.port,
        "Write a short paragraph explaining why 7 * 8 = 56.",
        args.max_tokens,
    )

    # --- prefill rate (long prompt, one output token) -------------------------
    # kept inside max_model_len=2048: ~120 repeats is ~1.3k tokens
    filler = ("The quick brown fox jumps over the lazy dog. ") * 120
    prompt = filler + "\nSummarise the sentence above in one word."
    try:
        started = time.perf_counter()
        counted = post(args.port, "/v1/completions", {
            "model": "minicpm5", "prompt": prompt, "max_tokens": 1, "temperature": 0.0,
        })
        elapsed = time.perf_counter() - started
        report["prefill"] = {
            "prompt_tokens": counted["usage"]["prompt_tokens"],
            "max_tokens_1_request_seconds": round(elapsed, 4),
            "prefill_tokens_per_second": round(counted["usage"]["prompt_tokens"] / elapsed, 2),
            "usage": counted.get("usage"),
            "text": counted["choices"][0].get("text"),
        }
    except Exception as error:  # keep the report even if the probe leg fails
        report["prefill"] = {"error": repr(error)}

    # --- aggregate throughput under concurrency -------------------------------
    results: list[dict] = []
    lock = threading.Lock()

    def worker(index: int) -> None:
        started = time.perf_counter()
        body = post(args.port, "/v1/completions", {
            "model": "minicpm5",
            "prompt": f"Count from 1 to 20 in words (request {index}).",
            "max_tokens": 64, "temperature": 0.0,
        }, timeout=600.0)
        elapsed = time.perf_counter() - started
        with lock:
            results.append({"seconds": elapsed, "usage": body.get("usage")})

    try:
        threads = [threading.Thread(target=worker, args=(index,))
                   for index in range(args.concurrency)]
        started = time.perf_counter()
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        wall = time.perf_counter() - started
        completion = sum((row["usage"] or {}).get("completion_tokens", 0) for row in results)
        report["concurrency"] = {
            "requests": args.concurrency,
            "wall_seconds": round(wall, 3),
            "total_completion_tokens": completion,
            "aggregate_completion_tokens_per_second": round(completion / wall, 2),
            "per_request_seconds": [
                round(row["seconds"], 3) for row in sorted(results, key=lambda r: r["seconds"])
            ],
            "mean_request_seconds": round(statistics.fmean(row["seconds"] for row in results), 3),
        }
    except Exception as error:  # keep the report even if the probe leg fails
        report["concurrency"] = {"error": repr(error)}

    report["free_during"] = free_bytes()
    report["process_rss"] = proc_rss()
    lines = log_lines(Path(args.serve_log))
    report["serve_log_interesting"] = [line for line in lines if INTERESTING.search(line)][-80:]
    report["serve_log_bytes"] = Path(args.serve_log).stat().st_size if Path(args.serve_log).exists() else 0

    destination = Path(args.out).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=1, sort_keys=True, ensure_ascii=False) + "\n")
    print(json.dumps({
        "generation_probe": report["generation_probe"]["text"],
        "finish_reason": report["generation_probe"]["finish_reason"],
        "decode": report["decode"],
        "prefill": report["prefill"],
        "generation_suite": report["generation_suite"],
        "concurrency": report["concurrency"]["aggregate_completion_tokens_per_second"],
        "free_during": report["free_during"],
    }, indent=1, ensure_ascii=False))
    print(f"[probe] wrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
