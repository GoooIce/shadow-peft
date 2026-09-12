#!/usr/bin/env python3
"""Evaluate a centre state on arbitrary tasksets (working sets only, no sealed pools).

Deploys the state via the frozen patch path and scores each --taskset entry
as {label=path}; writes one summary per taskset.
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


def log(*parts) -> None:
    print(f"[{time.strftime('%H:%M:%S')}]", *parts, flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--taskset", required=True, help="dev-shaped taskset for requests")
    parser.add_argument("--eval", action="append", required=True,
                        help="label=path tasksets to score on the deployed centre")
    parser.add_argument("--state", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--kernel-base", required=True)
    parser.add_argument("--linear-backend", default="triton")
    parser.add_argument("--quantization", default="compressed-tensors")
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--gpu-mem", type=float, default=0.30)
    parser.add_argument("--chunk", type=int, default=256)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--edits", type=int, default=32)
    parser.add_argument("--edits-plan")
    parser.add_argument("--probe-prompts")
    parser.add_argument("--bf16-logprobs")
    args = parser.parse_args()
    args.confirm = None
    args.feedback_slice = None
    args.terminal_swap = False
    args.skip_gate = False
    args.gate_halt = False
    args.kernel_base_script = str(PILOT_DIR / "build_kernel_payload.py")
    args.rebuild_kernel = False

    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    state = pl.CentreState.from_dict(json.loads(Path(args.state).expanduser().read_text()))
    runner = ps.Runner(args)
    runner.kernel_base = runner.ensure_kernel_base()
    runner.verify_layout()
    runner.reader.release_codes()
    runner.base_codes.clear()
    runner.start_engine()
    runner.prepare_requests()

    live: dict = {}
    runner.patch(live, ps.center_delta(state))
    report = {"protocol": "minicpm5-prime-eval/v1",
              "state_generation": state.generation,
              "centre_sha256": state.sha256(runner._base_hash)}
    for entry in args.eval:
        label, path = entry.split("=", 1)
        from pilot_search import load_items
        items = load_items(Path(path).expanduser(), 0)
        reqs = runner.build_reqs(items, "frozen", strict=True)
        rows, timing = runner.score(reqs, f"eval-{label}")
        summary = pl.fitness_summary(rows)
        report[label] = {"n": len(rows), "acc": summary["acc"],
                         "gold_prob": summary.get("gold_prob"),
                         "no_answer": runner.no_answer(rows)["overall"]["rate"],
                         "timing": timing}
        log(f"{label}: acc={summary['acc']:.4f} n={len(rows)}")
    (out / "eval_report.json").write_text(json.dumps(report, indent=1) + "\n")
    log(f"report -> {out / 'eval_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
