#!/usr/bin/env python3
"""OpenAI-compatible client probe for a running `vllm serve` (map ticket 06).

Measures both thinking modes, records finish_reason/usage/latency and a
steady-state decode throughput number. Stdlib only (no openai package needed).
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request

PROMPT = "用一句话解释什么是 NVFP4 量化，并说明它和 FP8 的区别。"


def post(url: str, payload: dict, timeout: float = 600) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def get(url: str, timeout: float = 60) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8123")
    ap.add_argument("--model", default="minicpm5-2b")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=192)
    ap.add_argument("--tool-test", action="store_true",
                    help="send a function-calling request (needs --enable-auto-tool-choice)")
    args = ap.parse_args()

    report: dict = {"base": args.base, "model": args.model, "prompt": PROMPT}
    report["models"] = get(f"{args.base}/v1/models")

    for name, thinking in (("thinking_true", True), ("thinking_false", False)):
        payload = {
            "model": args.model,
            "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": args.max_tokens,
            "temperature": 0,
            "top_p": 1,
            "chat_template_kwargs": {"enable_thinking": thinking},
        }
        t0 = time.perf_counter()
        resp = post(f"{args.base}/v1/chat/completions", payload)
        wall = time.perf_counter() - t0
        choice = resp["choices"][0]
        usage = resp.get("usage", {})
        n_out = usage.get("completion_tokens")
        msg = choice["message"]
        reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
        report[name] = {
            "wall_seconds": round(wall, 3),
            "finish_reason": choice.get("finish_reason"),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": n_out,
            "tokens_per_second_wall": round(n_out / wall, 2) if n_out and wall else None,
            "message_keys": sorted(msg.keys()),
            "content_head": (msg.get("content") or "")[:400],
            "reasoning_head": reasoning[:400],
            "content_tail": (msg.get("content") or "")[-200:],
            "reasoning_field_used": (
                "reasoning_content" if "reasoning_content" in msg
                else "reasoning" if "reasoning" in msg else None
            ),
            "content_is_none": msg.get("content") is None,
        }
        print(f"[{name}] finish={choice.get('finish_reason')} "
              f"out={n_out} {report[name]['tokens_per_second_wall']} tok/s wall "
              f"reasoning_field={report[name]['reasoning_field_used']}")

    # steady-state decode throughput: long output, single request
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": "写一段 800 字左右的短文，主题是分布式训练的梯度同步。"}],
        "max_tokens": 768,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    t0 = time.perf_counter()
    resp = post(f"{args.base}/v1/chat/completions", payload)
    wall = time.perf_counter() - t0
    usage = resp.get("usage", {})
    n_out = usage.get("completion_tokens")
    report["long_decode"] = {
        "wall_seconds": round(wall, 3),
        "completion_tokens": n_out,
        "prompt_tokens": usage.get("prompt_tokens"),
        "tokens_per_second_wall": round(n_out / wall, 2) if n_out and wall else None,
        "finish_reason": resp["choices"][0].get("finish_reason"),
    }
    print(f"[long] out={n_out} {report['long_decode']['tokens_per_second_wall']} tok/s wall")

    if args.tool_test:
        # exercises --enable-auto-tool-choice --tool-call-parser minicpm5
        tools = [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "查询指定城市的当前天气",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "城市名"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["city"],
                },
            },
        }]
        payload = {
            "model": args.model,
            "messages": [{"role": "user", "content": "北京现在天气怎么样？用工具查一下。"}],
            "tools": tools,
            "max_tokens": 256,
            "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        try:
            resp = post(f"{args.base}/v1/chat/completions", payload)
            choice = resp["choices"][0]
            msg = choice["message"]
            report["tool_call"] = {
                "finish_reason": choice.get("finish_reason"),
                "message_keys": sorted(msg.keys()),
                "tool_calls": msg.get("tool_calls"),
                "content": (msg.get("content") or "")[:400],
            }
            print(f"[tool] finish={choice.get('finish_reason')} "
                  f"tool_calls={json.dumps(msg.get('tool_calls'), ensure_ascii=False)[:200]}")
        except Exception as exc:
            report["tool_call"] = {"ok": False, "error": repr(exc)}
            print(f"[tool] FAILED: {exc!r}")

    with open(args.out, "w") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
