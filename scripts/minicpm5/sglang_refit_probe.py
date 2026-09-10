#!/usr/bin/env python3
"""SGLang 0.5.19 refit equivalence probe (ticket-11 A6).

Talks to a running ``sglang.launch_server`` over HTTP:

1. greedy decode with logprobs (baseline);
2. ``POST /update_weights_from_disk`` with ``recapture_cuda_graph=False``
   (timed) and, for contrast, ``recapture_cuda_graph=True``;
3. the same greedy decode after each reload, compared token by token with
   SGLang's own NVFP4 CI tolerance (atol=1e-4).

Usage:
    python sglang_refit_probe.py --port 8137 --model <payload> --out <json> \
        --prompts <probe json>
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path


def post(url: str, payload: dict, timeout: float = 900.0):
    data = json.dumps(payload).encode()
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read()
    return json.loads(body) if body else None


def get(url: str, timeout: float = 10.0):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read()


def wait_health(port: int, deadline: float) -> bool:
    while time.time() < deadline:
        try:
            get(f"http://127.0.0.1:{port}/health")
            return True
        except Exception:  # noqa: BLE001 - keep polling
            time.sleep(3)
    return False


def decode(port: int, prompt_ids: list[int], max_new_tokens: int) -> dict:
    started = time.perf_counter()
    result = post(
        f"http://127.0.0.1:{port}/generate",
        {
            "input_ids": prompt_ids,
            "sampling_params": {
                "temperature": 0.0,
                "max_new_tokens": max_new_tokens,
                "ignore_eos": True,
            },
            "return_logprob": True,
        },
    )
    elapsed = time.perf_counter() - started
    meta = result.get("meta_info", {})
    token_logprobs = meta.get("output_token_logprobs") or []
    return {
        "token_ids": [item[1] for item in token_logprobs],
        "logprobs": [float(item[0]) for item in token_logprobs],
        "seconds": elapsed,
        "text_head": result.get("text", "")[:80],
    }


def compare(left: dict, right: dict, atol: float) -> dict:
    if left["token_ids"] != right["token_ids"]:
        return {"tokens_identical": False, "atol": atol}
    diffs = [abs(a - b) for a, b in zip(left["logprobs"], right["logprobs"])]
    return {
        "tokens_identical": True,
        "max_abs_logprob_diff": max(diffs) if diffs else 0.0,
        "within_atol": all(value <= atol for value in diffs),
        "atol": atol,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--prompt-index", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--atol", type=float, default=1e-4)
    args = parser.parse_args()

    probe = json.loads(Path(args.prompts).read_text())
    prompt_ids = probe["decode_prompts"][args.prompt_index]

    if not wait_health(args.port, time.time() + 900):
        raise SystemExit("server did not become healthy")
    report = {"port": args.port, "model": args.model}

    baseline = decode(args.port, prompt_ids, args.max_new_tokens)
    report["baseline"] = baseline
    print("baseline tokens:", baseline["token_ids"][:8], flush=True)

    for label, recapture in (("no_recapture", False), ("recapture", True)):
        started = time.perf_counter()
        try:
            response = post(
                f"http://127.0.0.1:{args.port}/update_weights_from_disk",
                {
                    "model_path": args.model,
                    "recapture_cuda_graph": recapture,
                },
            )
            error = None
        except Exception as exc:  # noqa: BLE001 - the failure itself is the datum
            response = None
            error = f"{type(exc).__name__}: {exc}"
        seconds = time.perf_counter() - started
        entry = {
            "response": response,
            "error": error,
            "seconds": seconds,
        }
        try:
            after = decode(args.port, prompt_ids, args.max_new_tokens)
            entry["decode"] = after
            entry["compare_vs_baseline"] = compare(baseline, after, args.atol)
        except Exception as exc:  # noqa: BLE001
            entry["decode_error"] = f"{type(exc).__name__}: {exc}"
        report[label] = entry
        print(label, seconds, entry.get("compare_vs_baseline"), error, flush=True)

    Path(args.out).expanduser().write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(f"[sglang] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
