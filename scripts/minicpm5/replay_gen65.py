#!/usr/bin/env python3
"""gen65 desync discriminator: score state dd25f0c8 in a FRESH engine.

The frozen run (goooinext, seed 20260913) measured centre acc 0.525 on v2 dev
for centre dd25f0c8 (gens 65-68, four identical readings) while every candidate
built as dd25f0c8 + 32 random code edits scored ~4 points below it -- and the
collapse survived a process restart.  Two readings survive:

  (a) fresh-engine dd25f0c8 == 0.525  -> no desync; the state really scores
      0.525 and the landscape around it is genuinely sharp -> the freeze is
      the per-side gate starving (the pivot's ungated aggregation fixes it);
  (b) fresh-engine dd25f0c8 << 0.525  -> the in-run centre score was reading
      something other than the recorded state -> engine/state desync on the
      centre-scoring path; long runs blocked until fixed.

One process, one engine: score q0 as loaded (anchor), deploy the state with
the same patch path the search used, score again, KL-probe both arms.

Usage (gooo, ~/venvs/minicpm5-vllm):
    python replay_gen65.py \
        --model ~/minicpm5-payload/out/MiniCPM5-2B-NVFP4-W4A4 \
        --taskset ~/minicpm5-search/artifacts/taskset_dev.jsonl \
        --state ~/state_dd25f0c8.json \
        --probe-prompts ~/minicpm5-payload/out/refit/probes.json \
        --bf16-logprobs ~/minicpm5-search/out/bf16_kl.json \
        --kernel-base ~/minicpm5-run300/out/seed20260912/kernel_base.safetensors \
        --out ~/minicpm5-gen65replay/out
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

PILOT_DIR = Path(
    os.environ.get("MINICPM5_PILOT_DIR",
                   str(Path.home() / "minicpm5-run300/scripts/minicpm5"))
).expanduser()
sys.path.insert(0, str(PILOT_DIR))

import pilot_lib as pl  # noqa: E402
import pilot_search as ps  # noqa: E402


def log(*parts) -> None:
    print(f"[{time.strftime('%H:%M:%S')}]", *parts, flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="q0 NVFP4 payload directory")
    parser.add_argument("--taskset", required=True, help="v2 dev taskset jsonl")
    parser.add_argument("--state", required=True, help="state.json of centre dd25f0c8")
    parser.add_argument("--out", required=True)
    parser.add_argument("--kernel-base")
    parser.add_argument("--kernel-base-script")
    parser.add_argument("--rebuild-kernel", action="store_true")
    parser.add_argument("--probe-prompts")
    parser.add_argument("--bf16-logprobs")
    parser.add_argument("--linear-backend", default="cutlass")
    parser.add_argument("--quantization", default="compressed-tensors")
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--gpu-mem", type=float, default=0.30)
    parser.add_argument("--chunk", type=int, default=256)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--edits", type=int, default=32)
    parser.add_argument("--edits-plan")
    args = parser.parse_args()
    # fields the frozen Runner reads on paths this replay never exercises
    args.confirm = None
    args.feedback_slice = None
    args.terminal_swap = False
    args.skip_gate = False
    args.gate_halt = False
    if not args.kernel_base:
        args.kernel_base = str(Path(args.out) / "kernel_base.safetensors")
    if not args.kernel_base_script:
        args.kernel_base_script = str(PILOT_DIR / "build_kernel_payload.py")

    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    state = pl.CentreState.from_dict(json.loads(Path(args.state).expanduser().read_text()))

    runner = ps.Runner(args)
    report: dict = {"protocol": "minicpm5-gen65-replay/v1",
                    "state_generation": state.generation,
                    "delta_coords": sum(len(b) for b in state.codes.values()),
                    "residual_coords": sum(len(b) for b in state.residual.values())}
    runner.kernel_base = runner.ensure_kernel_base()
    layout = runner.verify_layout()
    report["layout_ok"] = not layout.get("errors")
    runner.reader.release_codes()
    runner.base_codes.clear()
    runner.start_engine()
    runner.prepare_requests()
    report["engine_load_seconds"] = runner.engine_load_seconds

    # arm A: q0 exactly as loaded (anchor -- in-run gen0 read 0.5000)
    rows, timing = runner.score(runner.dev_reqs, "q0_fresh")
    report["q0_fresh"] = pl.fitness_summary(rows)
    report["q0_fresh"]["timing"] = timing
    report["q0_kl"] = runner.kl_probe()
    log(f"q0 fresh acc={report['q0_fresh']['acc']:.4f} "
        f"kl={report['q0_kl']['generation_kl_error'] if report['q0_kl'] else None}")

    # arm B: deploy the shipped centre and score it (in-run read 0.525, KL 2.0596)
    live: dict = {}
    report["patch"] = runner.patch(live, ps.center_delta(state))
    report["centre_sha256_in_process"] = state.sha256(runner._base_hash)
    rows, timing = runner.score(runner.dev_reqs, "centre_fresh")
    report["centre_fresh"] = pl.fitness_summary(rows)
    report["centre_fresh"]["timing"] = timing
    report["centre_kl"] = runner.kl_probe()
    log(f"centre fresh acc={report['centre_fresh']['acc']:.4f} "
        f"kl={report['centre_kl']['generation_kl_error'] if report['centre_kl'] else None} "
        f"sha={report['centre_sha256_in_process'][:8]}")

    report["in_run_reference"] = {
        "centre_acc_dd25f0c8_gens65_68": 0.525,
        "centre_kl": 2.0595957755521828,
        "q0_acc_gen0": 0.5,
        "candidate_delta_mean_gen65": -0.0401,
    }
    (out / "replay_report.json").write_text(
        json.dumps(report, indent=1, sort_keys=True, ensure_ascii=False) + "\n")
    log(f"replay report written to {out / 'replay_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
