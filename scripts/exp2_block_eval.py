"""Exp2 v2-block eval: tau + wall-clock matrix for the trained block drafters.

Conditions per arm checkpoint: k∈{4,8}; dflash2 adds r∈{0,1,2} (r=0 is the
same-structure control vs dspark); dspark adds the confidence-threshold
dynamic-k ablation at k=8. Same wall-clock protocol as exp2_wallclock.py
(8 prompts x 64 tokens, 3-repeat median, bs=1, defer+crop), plus phase
breakdown, fp32 lossless spot-check and a 128-token long-generation check.

Run on GB10 (after both trainings):
    ssh goooinext 'cd ~/Code/shadow-peft && .venv/bin/python scripts/exp2_block_eval.py'
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from shadow_lab.block_drafter import CachedBlockDrafter
from shadow_lab.specdec import block_speculative_generate, greedy_reference

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


def timed(fn):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0, out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default="runs/exp2")
    ap.add_argument("--trunk", default=str(Path.home() / "models" / "LFM2.5-350M"))
    ap.add_argument("--teacher", default=str(Path.home() / "models" / "LFM2.5-1.2B-Instruct"))
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--new-tokens", type=int, default=64)
    args = ap.parse_args()
    dev = "cuda"
    out_dir = Path(args.out_root) / "v2block_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.teacher)
    prompt_ids = []
    for p in TAU_PROMPTS:
        enc = tok.apply_chat_template([{"role": "user", "content": p}],
                                      add_generation_prompt=True, return_tensors="pt")
        prompt_ids.append((enc["input_ids"] if hasattr(enc, "keys") else enc).to(dev))
    n_tok = args.new_tokens * len(prompt_ids)

    target = AutoModelForCausalLM.from_pretrained(args.teacher, dtype=torch.bfloat16).to(dev).eval()

    def build_drafter(structure: str) -> CachedBlockDrafter:
        ckpt = torch.load(Path(args.out_root) / f"v2block_{structure}" / "drafter_final.pt",
                          map_location="cpu")
        trunk = AutoModelForCausalLM.from_pretrained(args.trunk, dtype=torch.bfloat16)
        trunk.load_state_dict(ckpt["lm"])
        d = CachedBlockDrafter(trunk, k=8, with_confidence=structure == "dspark")
        d.heads.load_state_dict(ckpt["heads"])
        return d.to(dev).eval()

    def run_vanilla(n_tokens: int) -> float:
        total = 0.0
        for ids in prompt_ids:
            ms, _ = timed(lambda: greedy_reference(target, ids, n_tokens))
            total += ms
        return total

    def run_engine(drafter, k, r, n_tokens, refs, conf_threshold=None):
        total_ms, rounds, mismatch = 0.0, 0, 0
        accept_hist = None
        for ids, ref in zip(prompt_ids, refs, strict=True):
            ms, (o, st) = timed(lambda: block_speculative_generate(
                target, drafter, ids, k=k, r=r, max_new_tokens=n_tokens,
                conf_threshold=conf_threshold))
            total_ms += ms
            rounds += st["rounds"]
            accept_hist = st["accept_hist"] if accept_hist is None else \
                [a + b for a, b in zip(accept_hist, st["accept_hist"])]
            mismatch += int((o[0, -n_tokens:] != ref[0, -n_tokens:]).sum())
        return total_ms, rounds, accept_hist, mismatch

    report = {"n_prompts": len(prompt_ids), "tokens": n_tok, "repeats": args.repeats,
              "arms": {}}

    # warmup + fp32 spot-check pool
    dspark = build_drafter("dspark")
    timed(lambda: greedy_reference(target, prompt_ids[0], 16))
    timed(lambda: block_speculative_generate(target, dspark, prompt_ids[0], k=4, r=0,
                                             max_new_tokens=16))
    del dspark
    torch.cuda.empty_cache()

    for structure in ("dspark", "dflash2"):
        drafter = build_drafter(structure)
        refs = [greedy_reference(target, ids, args.new_tokens) for ids in prompt_ids]
        conds = []
        for k in (4, 8):
            rs = (0,) if structure == "dspark" else (0, 1, 2)
            for r in rs:
                ms_rows, rounds, accept_hist, mismatch = [], 0, None, 0
                for _ in range(args.repeats):
                    ms, rr, ah, mm = run_engine(drafter, k, r, args.new_tokens, refs)
                    ms_rows.append(ms)
                    rounds, accept_hist, mismatch = rr, ah, mm
                ms = statistics.median(ms_rows)
                rec = {"k": k, "r": r, "ms": round(ms, 1),
                       "ms_per_tok": round(ms / n_tok, 2),
                       "tau_effective": round(n_tok / rounds, 3),
                       "rounds": rounds, "accept_hist": accept_hist,
                       "mismatch_tokens_vs_ref": mismatch}
                conds.append(rec)
                print(f"[{structure}] {json.dumps(rec)}", flush=True)
        arm = {"conditions": conds}

        # dspark confidence-threshold ablation at k=8
        if structure == "dspark":
            ab = []
            for thr in (0.5, 0.7, 0.9):
                ms_rows, rounds, mismatch = [], 0, 0
                for _ in range(args.repeats):
                    ms, rr, _, mm = run_engine(drafter, 8, 0, args.new_tokens, refs,
                                               conf_threshold=thr)
                    ms_rows.append(ms)
                    rounds, mismatch = rr, mm
                ms = statistics.median(ms_rows)
                ab.append({"threshold": thr, "ms": round(ms, 1),
                           "ms_per_tok": round(ms / n_tok, 2),
                           "tau_effective": round(n_tok / rounds, 3)})
                print(f"[dspark-thr{thr}] {json.dumps(ab[-1])}", flush=True)
            arm["conf_threshold_ablation"] = ab
        report["arms"][structure] = arm
        del drafter
        torch.cuda.empty_cache()

    # shared vanilla baseline + phase breakdown + fp32 spot + long-128 at the
    # globally best condition
    vanilla_ms = statistics.median([run_vanilla(args.new_tokens) for _ in range(args.repeats)])
    report["vanilla_ms"] = round(vanilla_ms, 1)
    report["vanilla_ms_per_tok"] = round(vanilla_ms / n_tok, 2)

    best = None  # (structure, k, r, speedup)
    for structure, arm in report["arms"].items():
        for rec in arm["conditions"]:
            sp = vanilla_ms / rec["ms"]
            rec["speedup_local"] = round(sp, 3)
            if best is None or sp > best[3]:
                best = (structure, rec["k"], rec["r"], sp)
    b_struct, b_k, b_r, b_sp = best
    report["best"] = {"structure": b_struct, "k": b_k, "r": b_r,
                      "speedup_local": round(b_sp, 3)}

    drafter = build_drafter(b_struct)
    refs = [greedy_reference(target, ids, args.new_tokens) for ids in prompt_ids]

    t_tot: dict[str, float] = {}
    for ids in prompt_ids:
        _, st = block_speculative_generate(target, drafter, ids, k=b_k, r=b_r,
                                           max_new_tokens=args.new_tokens,
                                           collect_timing=True)
        for kk, v in st["t_ms"].items():
            t_tot[kk] = t_tot.get(kk, 0.0) + v
    measured = sum(t_tot.values())
    report["phase_ms"] = {kk: round(v, 1) for kk, v in sorted(t_tot.items())}
    report["phase_share"] = {kk: round(v / measured, 3) for kk, v in sorted(t_tot.items())}
    report["t_draft_share"] = report["phase_share"].get("draft", 0.0)

    # fp32 lossless spot-check (3 prompts)
    mm32 = 0
    target32 = AutoModelForCausalLM.from_pretrained(args.teacher, dtype=torch.float32).to(dev).eval()
    trunk32 = AutoModelForCausalLM.from_pretrained(args.trunk, dtype=torch.float32)
    ck = torch.load(Path(args.out_root) / f"v2block_{b_struct}" / "drafter_final.pt",
                    map_location="cpu")
    trunk32.load_state_dict(ck["lm"])
    d32 = CachedBlockDrafter(trunk32, k=8, with_confidence=b_struct == "dspark")
    d32.heads.load_state_dict(ck["heads"])
    d32 = d32.to(dev).eval()
    for ids in prompt_ids[:3]:
        ref = greedy_reference(target32, ids, 64)
        o, _ = block_speculative_generate(target32, d32, ids, k=b_k, r=b_r, max_new_tokens=64)
        mm32 += int((o[0, -64:] != ref[0, -64:]).sum())
    del target32, trunk32, d32
    torch.cuda.empty_cache()
    report["fp32_mismatch_3prompts"] = mm32

    # long-generation stability at best condition
    refs128 = [greedy_reference(target, ids, 128) for ids in prompt_ids]
    ms_l, rounds_l, _, mm_l = run_engine(drafter, b_k, b_r, 128, refs128)
    report["long_gen"] = {"new_tokens": 128, "engine_ms": round(ms_l, 1),
                          "speedup_local": round(
                              run_vanilla(128) / ms_l, 3),
                          "tau_effective": round(len(prompt_ids) * 128 / rounds_l, 3),
                          "mismatch_tokens_vs_ref": mm_l}

    report["verdict"] = {
        "success_line": ">=1.4x vs local vanilla (bs=1)",
        "best": report["best"],
        "pass": b_sp >= 1.4,
        "anchors": {"v1_chain_tau": {"k4": 3.483, "k8": 4.655},
                    "v1_chain_wallclock_best": 1.021},
    }
    (out_dir / "verdict.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["verdict"], indent=2), flush=True)


if __name__ == "__main__":
    main()
