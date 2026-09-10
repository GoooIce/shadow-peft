#!/usr/bin/env python3
"""Ticket 15: delivery-time train/inference consistency check for the shipped centre.

Runs on the GB10 host that holds the payload, in **one process**, and reuses the
frozen ticket-18 machinery (``pilot_search.Runner``) without modifying it:

1. static layout self-check (the ticket-18 fused-offset guard);
2. engine start + deployment of the shipped sparse centre (``state.json``) with
   the same patch path the search used (readback-verified, prefix cache flushed);
3. ``verify_live``: every live coordinate read back from both the worker cache
   and the live engine parameter, compared by value and by raw code;
4. engine dump vs the **delivered payload directory** re-exported by the
   reference exporter, compared with the frozen three-calibre value gate
   (``raw`` / ``value`` / ``canonical``, ``0 <-> 8`` accounted apart) --
   this is the contract-05 weight-path evidence for the *ship* artefact, not
   for the in-memory centre the search already gated every generation;
5. ``Generation KL Error`` probe of the deployed centre against the frozen BF16
   reference logprobs (contract 05's monitored activation-side statistic).

Nothing here opens any taskset pool: the script never reads ``--confirm``.

Usage (gooo, ``~/venvs/minicpm5-vllm``):
    python final_consistency_check.py \
        --model ~/minicpm5-payload/out/MiniCPM5-2B-NVFP4-W4A4 \
        --taskset ~/minicpm5-search/artifacts/taskset_dev.jsonl \
        --state ~/minicpm5-search/out/formal/state.json \
        --payload-dir ~/minicpm5-search/out/formal/payload_gen17 \
        --out ~/minicpm5-search/final/out/consistency \
        --kernel-base ~/minicpm5-search/out/kernel_base.safetensors \
        --probe-prompts ~/minicpm5-payload/out/refit/probes.json \
        --bf16-logprobs ~/minicpm5-search/out/bf16_kl.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

PILOT_DIR = Path(
    os.environ.get("MINICPM5_PILOT_DIR", str(Path.home() / "minicpm5-search/scripts/minicpm5"))
).expanduser()
sys.path.insert(0, str(PILOT_DIR))

import pilot_lib as pl  # noqa: E402
import pilot_search as ps  # noqa: E402


def log(*parts) -> None:
    print(f"[{time.strftime('%H:%M:%S')}]", *parts, flush=True)


def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="q0 NVFP4 payload directory")
    parser.add_argument("--taskset", required=True, help="dev taskset jsonl (tokenizer only)")
    parser.add_argument("--state", required=True, help="shipped sparse centre (state.json)")
    parser.add_argument("--payload-dir", required=True, help="delivered payload directory")
    parser.add_argument("--out", required=True)
    parser.add_argument("--kernel-base", required=True)
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
    parser.add_argument("--kernel-base-script")
    parser.add_argument("--rebuild-kernel", action="store_true")
    parser.add_argument("--keep-dumps", action="store_true",
                        help="keep the 1.1 GB engine dump and reference export")
    args = parser.parse_args()
    # The frozen Runner reads these two in paths this script never exercises.
    args.confirm = None       # the delivery check never opens a pool
    if not args.kernel_base_script:
        args.kernel_base_script = str(PILOT_DIR / "build_kernel_payload.py")
    args.skip_gate = False
    args.gate_halt = False

    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    report: dict = {"protocol": "minicpm5-final-consistency/v1"}

    state_path = Path(args.state).expanduser()
    state = pl.CentreState.from_dict(json.loads(state_path.read_text()))
    payload_dir = Path(args.payload_dir).expanduser()
    report["centre"] = {
        "state_path": str(state_path),
        "state_sha256": sha256_file(state_path),
        "generation": state.generation,
        "delta_coords": sum(len(bucket) for bucket in state.codes.values()),
        "residual_coords": sum(len(bucket) for bucket in state.residual.values()),
    }
    report["payload"] = {
        "dir": str(payload_dir),
        "model_sha256": sha256_file(payload_dir / "model.safetensors"),
        "model_bytes": (payload_dir / "model.safetensors").stat().st_size,
    }

    runner = ps.Runner(args)
    report["census"] = runner.census
    runner.kernel_base = runner.ensure_kernel_base()
    layout = runner.verify_layout()
    report["layout_check"] = layout
    runner.reader.release_codes()
    runner.base_codes.clear()
    runner.start_engine()
    runner.prepare_requests()
    report["engine_load_seconds"] = runner.engine_load_seconds
    report["quantization"] = args.quantization
    report["linear_backend"] = args.linear_backend

    live: dict = {}
    report["patch"] = runner.patch(live, ps.center_delta(state))
    report["centre"]["centre_sha256_in_process"] = state.sha256(runner._base_hash)
    report["verify_live"] = runner.verify_live(live, "delivery")

    dump_path = out / "engine_dump.safetensors"
    report["engine_dump"] = runner.llm.collective_rpc(
        ps.pw.search_dump, kwargs={"out_path": str(dump_path)}
    )[0]

    reference = out / "payload_reference.safetensors"
    subprocess.run([
        sys.executable, str(PILOT_DIR / "build_kernel_payload.py"),
        "--payload", str(payload_dir), "--out", str(reference),
    ], check=True, stdout=subprocess.DEVNULL)
    compare_path = out / "compare_files.json"
    values_path = out / "compare_values.json"
    subprocess.run([
        sys.executable, str(PILOT_DIR / "compare_kernel_files.py"),
        "--left", str(dump_path), "--right", str(reference), "--out", str(compare_path),
    ], check=True, stdout=subprocess.DEVNULL)
    subprocess.run([
        sys.executable, str(PILOT_DIR / "compare_kernel_values.py"),
        "--left", str(dump_path), "--right", str(reference), "--out", str(values_path),
    ], check=True, stdout=subprocess.DEVNULL)
    files = json.loads(compare_path.read_text())
    values = json.loads(values_path.read_text())
    report["engine_vs_payload"] = {
        "tensors": values["tensors"],
        "structural_mismatches": values["structural_mismatches"],
        "raw_code_diffs": values["raw_code_diffs"],
        "value_diffs": values["value_diffs"],
        "canonical_code_diffs": values["canonical_code_diffs"],
        "zero8_diffs": values["zero8_diffs"],
        "tensors_with_value_diffs": values["tensors_with_value_diffs"],
        "pass_value": values["pass_value"],
        "file_mismatches": files.get("mismatches"),
        "notes": values["notes"][:10],
    }
    report["engine_dump_sha256"] = sha256_file(dump_path)
    report["payload_reference_sha256"] = sha256_file(reference)
    report["kl_probe"] = runner.kl_probe()
    report["seconds"] = round(time.time() - started, 1)

    if not args.keep_dumps:
        dump_path.unlink(missing_ok=True)
        reference.unlink(missing_ok=True)
    (out / "consistency_report.json").write_text(
        json.dumps(report, indent=1, sort_keys=True, ensure_ascii=False) + "\n"
    )
    log(json.dumps(report["engine_vs_payload"], ensure_ascii=False))
    log(f"consistency report written to {out / 'consistency_report.json'} "
        f"({report['seconds']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
