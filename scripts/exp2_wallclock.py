"""Exp2 wall-clock judgment: crop snapshot + deferred bonus + k sweep + RTT matrix.

Success lines (spec, frozen): bs=1 wall-clock >= 1.4x over local vanilla greedy
AND >= 1.1x at RTT=50ms — the edge/cloud reading: draft on the edge box,
verify on the cloud paying one round trip per round, still faster than running
the vanilla target locally. The RTT matrix also reports the stateless-cloud
comparison (one round trip per *token*), where the KB-scale candidate uplink
beats MB-scale KV streaming.

Run on GB10:
    ssh goooinext 'cd ~/Code/shadow-peft && .venv/bin/python scripts/exp2_wallclock.py'
"""

from __future__ import annotations

import argparse
import json
import socket
import statistics
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from shadow_lab.specdec import greedy_reference, greedy_speculative_generate

TAU_PROMPTS = [
    "Explain recursion.",
    "Write a short paragraph about the silk road.",
    "List three key facts about photosynthesis.",
    "What is the most common mistake people make with pricing?",
    "Summarize jet lag in two sentences.",
    "Give a concrete example that illustrates hash tables.",
    "Why does sleep hygiene matter in practice?",
    "Describe sourdough bread as if to a complete beginner.",
]
RTT_TARGETS = [("hf.co", 443), ("www.baidu.com", 443), ("api.deepseek.com", 443)]


def timed(fn) -> tuple[float, object]:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0, out


def measure_rtt_samples() -> dict:
    """TCP connect time (~1 RTT) to a few public cloud fronts, from GB10."""
    res = {}
    for host, port in RTT_TARGETS:
        ts = []
        for _ in range(10):
            t0 = time.perf_counter()
            try:
                s = socket.create_connection((host, port), timeout=3)
                s.close()
                ts.append((time.perf_counter() - t0) * 1000.0)
            except OSError:
                pass
        if ts:
            res[host] = {"min_ms": round(min(ts), 1),
                         "median_ms": round(statistics.median(ts), 1), "n": len(ts)}
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft", default=str(Path.home() / "models" / "LFM2.5-350M"))
    ap.add_argument("--draft-state", default="runs/exp2/v1/draft_final.pt")
    ap.add_argument("--target", default=str(Path.home() / "models" / "LFM2.5-1.2B-Instruct"))
    ap.add_argument("--out", default="runs/exp2/wallclock")
    ap.add_argument("--new-tokens", type=int, default=64)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--k-list", default="2,3,4,6,8",
                    help="DFlash2-class drafters accept up to 8 tokens — sweep the full range")
    ap.add_argument("--long-tokens", type=int, default=128,
                    help="long-generation stability check at the best k (single repeat)")
    args = ap.parse_args()
    k_list = [int(x) for x in args.k_list.split(",")]
    dev = "cuda"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.target)
    prompt_ids = []
    for p in TAU_PROMPTS:
        enc = tok.apply_chat_template([{"role": "user", "content": p}],
                                      add_generation_prompt=True, return_tensors="pt")
        prompt_ids.append((enc["input_ids"] if hasattr(enc, "keys") else enc).to(dev))

    target = AutoModelForCausalLM.from_pretrained(args.target, dtype=torch.bfloat16).to(dev).eval()
    draft = AutoModelForCausalLM.from_pretrained(args.draft, dtype=torch.bfloat16)
    draft.load_state_dict(torch.load(args.draft_state, map_location="cpu"))
    draft = draft.to(dev).eval()

    n_tok = args.new_tokens * len(prompt_ids)

    # warmup both paths (CUDA context, kernel selection)
    timed(lambda: greedy_reference(target, prompt_ids[0], 16))
    timed(lambda: greedy_speculative_generate(target, draft, prompt_ids[0], k=4, max_new_tokens=16))

    def run_vanilla(n_tokens: int | None = None) -> float:
        n = n_tokens or args.new_tokens
        total = 0.0
        for ids in prompt_ids:
            ms, _ = timed(lambda: greedy_reference(target, ids, n))
            total += ms
        return total

    def run_engine(k: int, snapshot: str = "crop", defer: bool = True,
                   n_tokens: int | None = None):
        n = n_tokens or args.new_tokens
        total_ms, rounds, mismatch = 0.0, 0, 0
        accept_hist = None
        for ids, ref in zip(prompt_ids, _refs_for(n), strict=True):
            ms, (o, st) = timed(lambda: greedy_speculative_generate(
                target, draft, ids, k=k, max_new_tokens=n,
                snapshot=snapshot, defer_bonus=defer))
            total_ms += ms
            rounds += st["rounds"]
            accept_hist = st["accept_hist"] if accept_hist is None else \
                [a + b for a, b in zip(accept_hist, st["accept_hist"])]
            mismatch += int((o[0, -n:] != ref[0, -n:]).sum())
        return total_ms, rounds, accept_hist, mismatch

    def _refs_for(n: int) -> list[torch.Tensor]:
        if n not in _refs_cache:
            _refs_cache[n] = [greedy_reference(target, ids, n) for ids in prompt_ids]
        return _refs_cache[n]

    _refs_cache: dict[int, list[torch.Tensor]] = {}
    refs = _refs_for(args.new_tokens)
    vanilla_ms = statistics.median([run_vanilla() for _ in range(args.repeats)])
    report = {
        "n_prompts": len(prompt_ids), "tokens": n_tok,
        "vanilla_ms": round(vanilla_ms, 1),
        "vanilla_ms_per_tok": round(vanilla_ms / n_tok, 2),
        "engine": {},
    }

    best = None
    for k in k_list:
        ms_rows, rounds, accept_hist, mismatch = [], 0, None, 0
        for _ in range(args.repeats):
            ms, r, ah, mm = run_engine(k)
            ms_rows.append(ms)
            rounds, accept_hist, mismatch = r, ah, mm
        ms = statistics.median(ms_rows)
        rec = {
            "ms": round(ms, 1),
            "ms_per_tok": round(ms / n_tok, 2),
            "speedup_local": round(vanilla_ms / ms, 3),
            "tau_effective": round(n_tok / rounds, 3),
            "rounds": rounds,
            "mismatch_tokens_vs_ref": mismatch,
            "accept_hist": accept_hist,
        }
        report["engine"][f"k{k}_crop_defer"] = rec
        if best is None or rec["speedup_local"] > best[1]:
            best = (k, rec["speedup_local"])
        print(f"[k={k}] {json.dumps(rec)}", flush=True)

    k_best = best[0]
    for label, kw in [("deep_snapshot", {"snapshot": "deep"}),
                      ("no_defer", {"defer": False})]:
        ms_rows, rounds = [], 0
        for _ in range(args.repeats):
            ms, r, _, _ = run_engine(k_best, **kw)
            ms_rows.append(ms)
            rounds = r
        ms = statistics.median(ms_rows)
        report["engine"][f"k{k_best}_{label}"] = {
            "ms": round(ms, 1), "ms_per_tok": round(ms / n_tok, 2),
            "speedup_local": round(vanilla_ms / ms, 3),
            "tau_effective": round(n_tok / rounds, 3),
        }
        print(f"[{label}] {json.dumps(report['engine'][f'k{k_best}_{label}'])}", flush=True)

    # phase breakdown at best k (synchronized timing, single pass)
    t_tot: dict[str, float] = {}
    for ids in prompt_ids:
        _, st = greedy_speculative_generate(
            target, draft, ids, k=k_best, max_new_tokens=args.new_tokens,
            collect_timing=True)
        for kk, v in st["t_ms"].items():
            t_tot[kk] = t_tot.get(kk, 0.0) + v
    measured_ms = sum(t_tot.values())
    report["phase_ms"] = {kk: round(v, 1) for kk, v in sorted(t_tot.items())}
    report["phase_share"] = {kk: round(v / measured_ms, 3) for kk, v in sorted(t_tot.items())}
    report["t_draft_share"] = report["phase_share"].get("draft", 0.0)
    print(f"[phases] {json.dumps(report['phase_ms'])}", flush=True)

    # RTT matrix at best k: engine pays one round trip per round; the
    # stateless-cloud vanilla pays one per token (cloud compute assumed ~GB10)
    eng = report["engine"][f"k{k_best}_crop_defer"]
    rtt_matrix = {}
    for rtt in (0, 10, 50, 200):
        eng_eff = eng["ms"] + eng["rounds"] * rtt
        cloud_tok = vanilla_ms + n_tok * rtt
        rtt_matrix[str(rtt)] = {
            "engine_eff_ms": round(eng_eff, 1),
            "engine_ms_per_tok": round(eng_eff / n_tok, 2),
            "speedup_vs_local_vanilla": round(vanilla_ms / eng_eff, 3),
            "speedup_vs_cloud_stateless_per_token": round(cloud_tok / eng_eff, 3),
        }
    report["rtt_matrix"] = rtt_matrix
    report["measured_rtt"] = measure_rtt_samples()

    ok_local = best[1] >= 1.4
    ok_rtt50 = rtt_matrix["50"]["speedup_vs_local_vanilla"] >= 1.1
    report["verdict"] = {
        "success_line_local": ">=1.4x", "success_line_rtt50": ">=1.1x vs local vanilla",
        "best_k": k_best, "speedup_local_best": best[1],
        "pass_local_1p4x": ok_local, "pass_rtt50_1p1x": ok_rtt50,
        "overall": "PASS" if (ok_local and ok_rtt50) else ("PARTIAL" if best[1] >= 1.0 else "FAIL"),
    }
    # long-generation stability at best k: tau and speedup must not decay with
    # generation length (prefill amortization + deep-context agreement)
    if args.long_tokens:
        lt = args.long_tokens
        vanilla_long = run_vanilla(lt)
        eng_ms_l, rounds_l, _, mm_l = run_engine(k_best, n_tokens=lt)
        report["long_gen"] = {
            "new_tokens": lt,
            "vanilla_ms": round(vanilla_long, 1),
            "engine_ms": round(eng_ms_l, 1),
            "speedup_local": round(vanilla_long / eng_ms_l, 3),
            "tau_effective": round(len(prompt_ids) * lt / rounds_l, 3),
            "mismatch_tokens_vs_ref": mm_l,
        }
        print(f"[long-{lt}] {json.dumps(report['long_gen'])}", flush=True)

    (out_dir / "verdict.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["verdict"], indent=2), flush=True)


if __name__ == "__main__":
    main()
