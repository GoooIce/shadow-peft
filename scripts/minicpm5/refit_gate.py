#!/usr/bin/env python3
"""Ticket-11 refit feasibility gate: drive one vLLM engine through a reload
chain and record decode logprobs, scores, pointer/graph censuses and timings.

Modes
-----
``prepare-probes``
    Build the frozen probe set (decode prompts + scoring sequences) from a
    tokenised jsonl file; deterministic and reused by every engine run.
``run``
    Load one engine on ``--model`` and execute ``--chain`` (a JSON list of
    reload steps).  Every step is followed by the same probe battery, so the
    resulting JSON is a per-step snapshot of the engine.
``compare``
    Offline comparison of two ``run`` JSONs (token-by-token ids/logprobs).

Usage (one engine):
    python refit_gate.py --mode run --model <payload> --label q0 \
        --out out/refit/q0.json --probe-set out/refit/probes.json \
        --chain '[{"kind":"cp","path":"<candidate>"}]'
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from vllm import LLM, SamplingParams

import refit_worker as rw

PROBE_PROMPTS = 8
PROBE_PROMPT_LEN = 64
PROBE_DECODE_TOKENS = 32
SCORE_SEQS = 16
SCORE_LEN = 2048


# --------------------------------------------------------------------------- #
# probe set
# --------------------------------------------------------------------------- #


def build_probe_set(eval_jsonl: Path, path: Path) -> dict:
    sequences = []
    with eval_jsonl.open() as handle:
        for line in handle:
            if line.strip():
                sequences.append(json.loads(line)["input_ids"])
    if len(sequences) < SCORE_SEQS:
        raise SystemExit(f"need >= {SCORE_SEQS} sequences, got {len(sequences)}")
    probe = {
        "source": str(eval_jsonl),
        "decode_prompts": [ids[:PROBE_PROMPT_LEN] for ids in sequences[:PROBE_PROMPTS]],
        "score_sequences": [ids[:SCORE_LEN] for ids in sequences[:SCORE_SEQS]],
        "decode_tokens": PROBE_DECODE_TOKENS,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(probe) + "\n")
    print(f"[probes] wrote {path}: {len(probe['decode_prompts'])} prompts, "
          f"{len(probe['score_sequences'])} score sequences")
    return probe


# --------------------------------------------------------------------------- #
# probes
# --------------------------------------------------------------------------- #


def decode_probe(llm: LLM, probe: dict) -> dict:
    """Greedy decode per prompt in isolation; keep token ids + logprobs."""
    results = []
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=probe.get("decode_tokens", PROBE_DECODE_TOKENS),
        logprobs=1,
        detokenize=False,
        ignore_eos=True,
    )
    for index, prompt_ids in enumerate(probe["decode_prompts"]):
        started = time.perf_counter()
        output = llm.generate([{"prompt_token_ids": list(prompt_ids)}], sampling)
        elapsed = time.perf_counter() - started
        completion = output[0].outputs[0]
        token_ids = list(completion.token_ids)
        logprobs = []
        for position, token_id in enumerate(token_ids):
            entry = completion.logprobs[position]
            logprobs.append(float(entry[token_id].logprob) if entry else None)
        results.append(
            {
                "index": index,
                "prompt_sha256": rw._sha256(
                    torch.tensor(list(prompt_ids), dtype=torch.long)
                )[:16],
                "token_ids": token_ids,
                "logprobs": logprobs,
                "seconds": round(elapsed, 4),
            }
        )
    return {"prompts": results}


def score_probe(llm: LLM, probe: dict) -> dict:
    """Teacher-forced mean NLL / ppl on the frozen scoring sequences."""
    sampling = SamplingParams(
        temperature=0.0, max_tokens=1, prompt_logprobs=0, detokenize=False
    )
    started = time.perf_counter()
    outputs = llm.generate(
        [{"prompt_token_ids": list(ids)} for ids in probe["score_sequences"]], sampling
    )
    elapsed = time.perf_counter() - started
    total_nll = 0.0
    total_tokens = 0
    per_sequence = []
    for output in outputs:
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
                continue
            seq_nll += -float(logprob.logprob)
            seq_tokens += 1
        total_nll += seq_nll
        total_tokens += seq_tokens
        per_sequence.append(
            {
                "tokens": seq_tokens,
                "mean_nll": seq_nll / seq_tokens if seq_tokens else None,
            }
        )
    mean_nll = total_nll / total_tokens if total_tokens else None
    return {
        "sequences": len(outputs),
        "scored_tokens": total_tokens,
        "mean_nll": mean_nll,
        "ppl": math.exp(mean_nll) if mean_nll is not None else None,
        "seconds": round(elapsed, 3),
        "per_sequence": per_sequence,
    }


def kl_probe(llm: LLM, probe: dict, continuations: dict) -> dict:
    """Per-token logprob of a fixed continuation under the current engine."""
    sampling = SamplingParams(
        temperature=0.0, max_tokens=1, prompt_logprobs=0, detokenize=False
    )
    prompts = []
    spans = []
    for index, prompt_ids in enumerate(probe["decode_prompts"]):
        continuation = continuations[str(index)]
        sequence = list(prompt_ids) + list(continuation)
        prompts.append({"prompt_token_ids": sequence})
        spans.append((len(prompt_ids), len(sequence)))
    started = time.perf_counter()
    outputs = llm.generate(prompts, sampling)
    elapsed = time.perf_counter() - started
    results = []
    for index, output in enumerate(outputs):
        start, end = spans[index]
        logprobs = []
        for position in range(start, end):
            entry = output.prompt_logprobs[position]
            token_id = output.prompt_token_ids[position]
            logprobs.append(
                float(entry[token_id].logprob) if entry and token_id in entry else None
            )
        results.append({"index": index, "logprobs": logprobs})
    return {"continuation_logprobs": results, "seconds": round(elapsed, 3)}


def kernel_probe(llm: LLM, probe: dict) -> dict:
    """torch-profiler kernel census over a prefill-heavy generate call."""
    from torch.profiler import ProfilerActivity, profile

    from kernel_evidence import _classify

    prompts = [
        {"prompt_token_ids": list(ids)}
        for ids in probe["score_sequences"][:8]
    ]
    sampling = SamplingParams(temperature=0.0, max_tokens=1, detokenize=False)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        llm.generate(prompts, sampling)
        torch.cuda.synchronize()
    events = [
        {"name": event.name, "dur": float(event.device_time_total)}
        for event in prof.key_averages()
        if event.device_time_total > 0
    ]
    summary = _classify(events)
    return {
        bucket: {
            "calls": entry["calls"],
            "dur_share": entry["dur_share"],
            "top": list(entry["kernels"])[:5],
        }
        for bucket, entry in summary.items()
    }


# --------------------------------------------------------------------------- #
# engine snapshot
# --------------------------------------------------------------------------- #


def snapshot(llm: LLM, probe: dict, tag: str, args, continuations=None) -> dict:
    entry = {"tag": tag, "wall": time.strftime("%Y-%m-%dT%H:%M:%S")}
    entry["kernel_census"] = llm.collective_rpc(rw.kernel_census)[0]
    entry["graph_census"] = llm.collective_rpc(rw.graph_census)[0]
    # vLLM fuses q/k/v and gate/up, so hash a small, fixed set of real layers.
    all_layers = sorted(entry["kernel_census"]["per_layer"])
    hash_layers = [name for name in all_layers if name.endswith("qkv_proj")][:1] + [
        name for name in all_layers if name.endswith("down_proj")
    ][:1]
    entry["census"] = llm.collective_rpc(
        rw.census, kwargs={"hash_params": ("weight", "weight_scale"), "names": hash_layers}
    )[0]
    entry["decode"] = decode_probe(llm, probe)
    entry["score"] = score_probe(llm, probe)
    if continuations is not None:
        entry["kl"] = kl_probe(llm, probe, continuations)
    if args.profile and tag == "baseline":
        entry["kernels"] = kernel_probe(llm, probe)
    return entry


def run_chain_step(llm: LLM, step: dict, state: dict) -> dict:
    kind = step["kind"]
    if kind == "cp":
        return llm.collective_rpc(
            rw.reload_checkpoint, kwargs={"weights_path": step["path"]}
        )[0]
    if kind == "kf_dump":
        path = step["path"]
        result = llm.collective_rpc(
            rw.dump_kernel_params, kwargs={"out_path": path}
        )[0]
        state["kf_path"] = path
        return result
    if kind == "kf_prepare":
        path = step.get("path", state.get("kf_path"))
        return llm.collective_rpc(
            rw.kf_prepare,
            kwargs={
                "path": path,
                "cache_key": step["cache_key"],
                "device": step.get("device", "cpu"),
            },
        )[0]
    if kind == "kf":
        return llm.collective_rpc(
            rw.kf_reload,
            kwargs={
                "cache_key": step.get("cache_key"),
                "path": step.get("path", state.get("kf_path")),
                "mutate": step.get("mutate"),
            },
        )[0]
    if kind == "kf_materialize":
        return llm.collective_rpc(
            rw.kf_materialize, kwargs={"cache_key": step["cache_key"]}
        )[0]
    if kind == "kf_reload_repeat":
        return llm.collective_rpc(
            rw.kf_reload_repeat,
            kwargs={"cache_key": step["cache_key"], "repeats": step.get("repeats", 5)},
        )[0]
    if kind == "kf_copy_floor":
        return llm.collective_rpc(
            rw.kf_copy_floor, kwargs={"cache_key": step["cache_key"]}
        )[0]
    if kind == "readback":
        return llm.collective_rpc(
            rw.readback_payload,
            kwargs={
                "payload_path": step["payload"],
                "out_path": step["out"],
                "names": step.get("names"),
            },
        )[0]
    if kind == "capture_register":
        return llm.collective_rpc(rw.capture_register)[0]
    if kind == "capture_collect":
        return llm.collective_rpc(
            rw.capture_collect, kwargs={"out_path": step["out"]}
        )[0]
    if kind == "profile_run":
        # vLLM runs the model in a separate EngineCore process, so a profiler in
        # this driver process sees nothing; use the engine's own torch profiler
        # and classify the trace it writes from the worker.
        from kernel_evidence import _classify, _load_trace

        probe_data = json.loads(Path(state["probe_set"]).read_text())
        prompts = [
            {"prompt_token_ids": list(ids)}
            for ids in probe_data["score_sequences"][:8]
        ]
        sampling = SamplingParams(temperature=0.0, max_tokens=1, detokenize=False)
        llm.start_profile()
        llm.generate(prompts, sampling)
        llm.stop_profile()
        time.sleep(2.0)
        profile_dir = Path(step["dir"]).expanduser()
        traces = sorted(
            profile_dir.rglob("*"), key=lambda path: path.stat().st_mtime, reverse=True
        )
        traces = [path for path in traces if path.suffix in (".json", ".gz")]
        if not traces:
            return {"error": "no trace file found", "dir": str(profile_dir)}
        events = _load_trace(traces[0])
        summary = _classify(events)
        return {
            "trace": str(traces[0]),
            "events": len(events),
            "classes": {
                bucket: {
                    "calls": entry["calls"],
                    "dur_share": entry["dur_share"],
                    "top": list(entry["kernels"])[:4],
                }
                for bucket, entry in summary.items()
            },
        }
    if kind == "capture_run":
        # Register hooks on the worker, push one deterministic prefill through
        # the real engine, then persist the captured layer outputs.
        probe_data = json.loads(Path(state["probe_set"]).read_text())
        prompt = probe_data["decode_prompts"][step.get("prompt_index", 0)]
        llm.collective_rpc(rw.capture_register)
        llm.generate(
            [{"prompt_token_ids": list(prompt)}],
            SamplingParams(temperature=0.0, max_tokens=1, detokenize=False),
        )
        return llm.collective_rpc(
            rw.capture_collect, kwargs={"out_path": step["out"]}
        )[0]
    raise SystemExit(f"unknown chain step kind: {kind}")


# --------------------------------------------------------------------------- #
# modes
# --------------------------------------------------------------------------- #


def mode_run(args) -> int:
    probe_path = Path(args.probe_set).expanduser()
    if probe_path.exists():
        probe = json.loads(probe_path.read_text())
    else:
        probe = build_probe_set(Path(args.eval_jsonl).expanduser(), probe_path)

    continuations = None
    if args.continuations_in:
        continuations = json.loads(Path(args.continuations_in).read_text())

    chain = json.loads(args.chain) if args.chain else []
    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    quantization = args.quantization
    if quantization in (None, "", "none", "None"):
        quantization = None
    args.quantization = quantization

    profiler_config = None
    if args.profile_dir:
        from vllm.config import ProfilerConfig

        profiler_config = ProfilerConfig(
            profiler="torch", torch_profiler_dir=args.profile_dir
        )

    started = time.perf_counter()
    llm = LLM(
        model=args.model,
        quantization=quantization,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        linear_backend=args.linear_backend,
        enforce_eager=args.enforce_eager,
        enable_prefix_caching=not args.no_prefix_caching,
        seed=args.seed,
        disable_log_stats=True,
        profiler_config=profiler_config,
    )
    load_seconds = time.perf_counter() - started
    print(f"[run] {args.label}: engine loaded in {load_seconds:.1f}s", flush=True)

    report = {
        "label": args.label,
        "model": args.model,
        "quantization": args.quantization,
        "linear_backend": args.linear_backend,
        "enforce_eager": args.enforce_eager,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "engine_load_seconds": round(load_seconds, 2),
        "probe_set": str(probe_path),
        "steps": [],
    }

    state: dict = {"probe_set": str(probe_path)}
    report["steps"].append(snapshot(llm, probe, "baseline", args, continuations))
    if args.continuations_out:
        # The baseline greedy decode *is* the frozen continuation set that other
        # engines teacher-force to compute the Generation KL Error.
        continuations_out = {
            str(item["index"]): item["token_ids"]
            for item in report["steps"][0]["decode"]["prompts"]
        }
        Path(args.continuations_out).expanduser().write_text(
            json.dumps(continuations_out) + "\n"
        )
        print(f"[run] wrote continuations {args.continuations_out}")
    for index, step in enumerate(chain):
        tag = step.get("label", f"step{index}:{step['kind']}")
        print(f"[run] {args.label}: chain step {tag} ...", flush=True)
        timing = run_chain_step(llm, step, state)
        report["steps"].append(
            {
                "tag": tag,
                "step": step,
                "timing": timing,
                "snapshot": snapshot(llm, probe, tag, args, continuations),
            }
        )
        print(f"[run] {args.label}: {tag} done in {timing}", flush=True)

    report["total_seconds"] = round(time.perf_counter() - started, 2)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"[run] wrote {out_path}")
    return 0


def _tokens_equal(left, right):
    return list(left) == list(right)


def mode_compare(args) -> int:
    left = json.loads(Path(args.left).read_text())
    right = json.loads(Path(args.right).read_text())
    left_steps = {step["tag"]: step for step in left["steps"]}
    right_steps = {step["tag"]: step for step in right["steps"]}

    def pick(step):
        return step["snapshot"] if "snapshot" in step else step

    report = {"left": args.left, "right": args.right, "tags": {}, "pairs": []}
    for left_tag in args.pair:
        right_tag = args.pair[right_tag] if False else args.pair[left_tag]
        if left_tag not in left_steps or right_tag not in right_steps:
            report["pairs"].append({"left": left_tag, "right": right_tag, "error": "missing"})
            continue
        lstep = pick(left_steps[left_tag])
        rstep = pick(right_steps[right_tag])
        pair = {"left": left_tag, "right": right_tag}
        # decode token/logprob comparison
        per_prompt = []
        max_abs = 0.0
        for lprompt, rprompt in zip(lstep["decode"]["prompts"], rstep["decode"]["prompts"]):
            identical = _tokens_equal(lprompt["token_ids"], rprompt["token_ids"])
            diffs = [
                abs((la if la is not None else float("nan"))
                    - (ra if ra is not None else float("nan")))
                for la, ra in zip(lprompt["logprobs"], rprompt["logprobs"])
            ]
            finite = [value for value in diffs if value == value]
            max_abs = max([max_abs] + finite) if finite else max_abs
            per_prompt.append(
                {
                    "tokens_identical": identical,
                    "first_divergence": next(
                        (
                            index
                            for index, (la, ra) in enumerate(
                                zip(lprompt["token_ids"], rprompt["token_ids"])
                            )
                            if la != ra
                        ),
                        None,
                    ),
                    "max_abs_logprob_diff": max(finite) if finite else None,
                }
            )
        pair["decode"] = {
            "all_tokens_identical": all(item["tokens_identical"] for item in per_prompt),
            "max_abs_logprob_diff": max_abs,
            "per_prompt": per_prompt,
        }
        # score comparison
        pair["score"] = {
            "left_ppl": lstep["score"]["ppl"],
            "right_ppl": rstep["score"]["ppl"],
            "ppl_delta": (
                lstep["score"]["ppl"] - rstep["score"]["ppl"]
                if lstep["score"]["ppl"] is not None and rstep["score"]["ppl"] is not None
                else None
            ),
            "mean_nll_delta": (
                lstep["score"]["mean_nll"] - rstep["score"]["mean_nll"]
                if lstep["score"]["mean_nll"] is not None
                and rstep["score"]["mean_nll"] is not None
                else None
            ),
        }
        # KL comparison when both scored the same continuations
        if "kl" in lstep and "kl" in rstep:
            deltas = []
            for lseq, rseq in zip(
                lstep["kl"]["continuation_logprobs"],
                rstep["kl"]["continuation_logprobs"],
            ):
                for lv, rv in zip(lseq["logprobs"], rseq["logprobs"]):
                    if lv is None or rv is None:
                        continue
                    deltas.append(abs(lv - rv))
            pair["kl"] = {
                "tokens": len(deltas),
                "mean_abs_delta": sum(deltas) / len(deltas) if deltas else None,
                "max_abs_delta": max(deltas) if deltas else None,
                "generation_kl_error": (
                    sum(math.exp(value) for value in deltas) / len(deltas)
                    if deltas
                    else None
                ),
            }
        report["pairs"].append(pair)
    Path(args.out).expanduser().write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report["pairs"], indent=2)[:4000])
    print(f"[compare] wrote {args.out}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=("prepare-probes", "run", "compare"))
    parser.add_argument("--model")
    parser.add_argument("--label")
    parser.add_argument("--out")
    parser.add_argument("--eval-jsonl")
    parser.add_argument("--probe-set")
    parser.add_argument("--chain", default="")
    parser.add_argument("--continuations-in")
    parser.add_argument("--continuations-out")
    parser.add_argument("--quantization", default=None)
    parser.add_argument("--linear-backend", default="cutlass")
    parser.add_argument("--max-model-len", type=int, default=2049)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.30)
    parser.add_argument("--max-num-seqs", type=int, default=16)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--no-prefix-caching", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-dir")
    parser.add_argument("--seed", type=int, default=0)
    # compare mode
    parser.add_argument("--left")
    parser.add_argument("--right")
    parser.add_argument("--pair", nargs="*", default=[])
    args = parser.parse_args()

    if args.mode == "prepare-probes":
        build_probe_set(
            Path(args.eval_jsonl).expanduser(), Path(args.probe_set).expanduser()
        )
        return 0
    if args.mode == "run":
        return mode_run(args)
    if args.mode == "compare":
        # --pair is a list of "left_tag=right_tag"
        pairs = {}
        for item in args.pair:
            left_tag, _, right_tag = item.partition("=")
            pairs[left_tag] = right_tag or left_tag
        args.pair = pairs
        return mode_compare(args)
    raise SystemExit("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
