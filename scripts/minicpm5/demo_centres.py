#!/usr/bin/env python3
"""Qualitative demo: do the campaign centres still answer questions?

One engine on the INT8 q0 payload; deploy each centre via the frozen patch
path; ask a fixed set of FRESH questions (none from any pool) in the frozen
prompt form the search optimized, plus one plain chat turn each.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

PILOT_DIR = Path(os.environ.get("MINICPM5_PILOT_DIR",
                                str(Path.home() / "minicpm5-w8a8qes/scripts/minicpm5"))).expanduser()
sys.path.insert(0, str(PILOT_DIR))

import pilot_lib as pl  # noqa: E402
import pilot_search as ps  # noqa: E402

QUESTIONS = [
    "What is the value of 37 + 45?",
    "What is the value of 128 divided by 8?",
    "Anna has 3 apples and buys 5 more. How many apples does she have?",
    "If today is Wednesday, what day will it be in 5 days?",
    "What is 20% of 150?",
    "Is 17 a prime number?",
]


def frozen_prompt(q: str) -> str:
    return (f"<s><|im_start|>user\n{q}\n\n"
            "Reply with only the final answer, with no explanation.<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\nAnswer:")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--states", nargs="+", required=True,
                        help="label=path/to/state.json pairs")
    parser.add_argument("--kernel-base", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--linear-backend", default="triton")
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--gpu-mem", type=float, default=0.30)
    args = parser.parse_args()
    args.confirm = None
    args.feedback_slice = None
    args.terminal_swap = False
    args.skip_gate = False
    args.gate_halt = False
    args.taskset = None
    args.limit = 0
    args.quantization = "compressed-tensors"
    args.edits = 32
    args.edits_plan = None
    args.kernel_base_script = str(PILOT_DIR / "build_kernel_payload.py")
    args.rebuild_kernel = False

    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    runner = ps.Runner(args)
    runner.kernel_base = runner.ensure_kernel_base()
    runner.verify_layout()
    runner.reader.release_codes()
    runner.base_codes.clear()
    runner.start_engine()
    from vllm import SamplingParams
    sp_frozen = SamplingParams(temperature=0.0, max_tokens=16, stop=["<|im_end|>"])
    sp_chat = SamplingParams(temperature=0.0, max_tokens=64, stop=["<|im_end|>"])

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)

    def ask(arm: str, state) -> dict:
        live: dict = {}
        if state is not None:
            runner.patch(live, ps.center_delta(state))
        frozen_prompts = [frozen_prompt(q) for q in QUESTIONS]
        outs = runner.llm.generate(frozen_prompts, sp_frozen)
        frozen_answers = [o.outputs[0].text.strip() for o in outs]
        chat_prompts = [tok.apply_chat_template(
            [{"role": "user", "content": q + " Answer in one short sentence."}],
            tokenize=False, add_generation_prompt=True) for q in QUESTIONS[:3]]
        outs = runner.llm.generate(chat_prompts, sp_chat)
        chat_answers = [o.outputs[0].text.strip() for o in outs]
        return {"arm": arm,
                "sha": (state.sha256(runner._base_hash)[:12] if state is not None else "q0"),
                "frozen": dict(zip(QUESTIONS, frozen_answers)),
                "chat": dict(zip(QUESTIONS[:3], chat_answers))}

    report = {"protocol": "minicpm5-w8a8-centre-demo/v1"}
    report["q0"] = ask("q0", None)
    for pair in args.states:
        label, path = pair.split("=", 1)
        state = pl.CentreState.from_dict(json.loads(Path(path).expanduser().read_text()))
        report[label] = ask(label, state)

    (out / "centre_demo.json").write_text(
        json.dumps(report, indent=1, ensure_ascii=False) + "\n")
    for arm in ("q0",) + tuple(p.split("=", 1)[0] for p in args.states):
        r = report[arm]
        print(f"\n===== {arm} (sha {r['sha']}) =====")
        for q, a in r["frozen"].items():
            print(f"  {q}  ->  {a!r}")
        for q, a in r["chat"].items():
            print(f"  [chat] {q}  ->  {a!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
