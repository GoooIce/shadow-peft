"""Exp2 P0 on real models (spec exp2 前置 2/3 + τ 校准基线).

1. Download-less smoke: engine lossless check — speculative output vs plain
   greedy target, token-identical, on real 350M-draft / 1.2B-Instruct-target.
2. Oracle: draft == target -> tau == k+1 and identity holds.
3. Baseline tau with the UNTRAINED draft (the pre-distillation number).
4. Rough wall-clock: cached target greedy vs speculative (info only).

Run on GB10:
    ssh goooinext 'cd ~/Code/shadow-peft && .venv/bin/python scripts/exp2_p0.py'
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from shadow_lab.specdec import (
    _cached_kwargs,
    greedy_reference,
    greedy_speculative_generate,
    measure_tau,
)

P0_PROMPTS = [
    "Explain recursion.",
    "Write a short paragraph about the silk road.",
    "List three key facts about photosynthesis.",
    "What is the most common mistake people make with pricing?",
    "Summarize jet lag in two sentences.",
    "Give a concrete example that illustrates hash tables.",
    "Why does sleep hygiene matter in practice?",
    "Describe sourdough bread as if to a complete beginner.",
]


def timed_vanilla(target, ids: torch.Tensor, n: int, dev) -> tuple[torch.Tensor, float]:
    t0 = time.perf_counter()
    out = target(input_ids=ids, use_cache=True)
    cache = out.past_key_values
    gen = [out.logits[:, -1].argmax(dim=-1, keepdim=True)]
    for _ in range(n - 1):
        o = target(input_ids=gen[-1], past_key_values=cache, **_cached_kwargs(cache, 1, dev))
        cache = o.past_key_values
        gen.append(o.logits[:, -1].argmax(dim=-1, keepdim=True))
    dt = time.perf_counter() - t0
    return torch.cat([ids] + gen, dim=1), dt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default=str(Path.home() / "models" / "LFM2.5-1.2B-Instruct"))
    ap.add_argument("--draft", default=str(Path.home() / "models" / "LFM2.5-350M"))
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--out", default="runs/exp2/p0/verdict.json")
    args = ap.parse_args()
    dev = "cuda"

    tok = AutoTokenizer.from_pretrained(args.target)
    target = AutoModelForCausalLM.from_pretrained(args.target, dtype=torch.bfloat16).to(dev).eval()
    draft = AutoModelForCausalLM.from_pretrained(args.draft, dtype=torch.bfloat16).to(dev).eval()

    prompts_ids = []
    for p in P0_PROMPTS:
        enc = tok.apply_chat_template([{"role": "user", "content": p}],
                                      add_generation_prompt=True, return_tensors="pt")
        ids = enc["input_ids"] if hasattr(enc, "keys") else enc
        prompts_ids.append(ids.to(dev))

    report: dict = {"k": args.k, "max_new_tokens": args.max_new, "n_prompts": len(prompts_ids)}

    # 1. lossless on real pair — exact identity where numerics allow, plus
    # agreement stats (bf16 chunk-k verify vs chunk-1 reference flips near-ties)
    n_ident, agrees, first_diffs = 0, [], []
    for ids in prompts_ids:
        out, _ = greedy_speculative_generate(target, draft, ids, k=args.k, max_new_tokens=args.max_new)
        ref = greedy_reference(target, ids, args.max_new)
        same = torch.equal(out, ref)
        n_ident += int(same)
        g_o, g_r = out[0, ids.shape[1]:], ref[0, ids.shape[1]:]
        agrees.append(float((g_o == g_r).float().mean()))
        if not same:
            first_diffs.append(int((g_o != g_r).nonzero()[0]))
    report["lossless_identical"] = f"{n_ident}/{len(prompts_ids)}"
    report["agreement"] = {"mean": round(sum(agrees) / len(agrees), 4),
                           "min": round(min(agrees), 4),
                           "first_diff_positions": first_diffs}
    print(f"[1] lossless: {n_ident}/{len(prompts_ids)} exact; "
          f"agreement mean={report['agreement']['mean']} min={report['agreement']['min']} "
          f"first_diffs={first_diffs}", flush=True)

    # 2. oracle: draft == target
    n_oracle = 0
    for ids in prompts_ids[:3]:
        out, s = greedy_speculative_generate(target, target, ids, k=args.k, max_new_tokens=16)
        ref = greedy_reference(target, ids, 16)
        n_oracle += int(torch.equal(out, ref) and abs(s["tau"] - (args.k + 1)) < 1e-6)
    report["oracle_identical_tau_k1"] = f"{n_oracle}/3"
    print(f"[2] oracle identity + tau==k+1: {n_oracle}/3", flush=True)

    # 3. baseline tau with untrained draft
    tau0 = measure_tau(target, draft, prompts_ids, k=args.k, max_new_tokens=args.max_new)
    report["baseline_tau_untrained_draft"] = tau0
    print(f"[3] baseline tau (untrained draft): {tau0}", flush=True)

    # 4. rough wall-clock (info only)
    ids = prompts_ids[0]
    _, dt_v = timed_vanilla(target, ids, args.max_new, dev)
    _, s = greedy_speculative_generate(target, draft, ids, k=args.k, max_new_tokens=args.max_new)
    t0 = time.perf_counter()
    greedy_speculative_generate(target, draft, ids, k=args.k, max_new_tokens=args.max_new)
    dt_s = time.perf_counter() - t0
    report["rough_wallclock"] = {
        "vanilla_s": round(dt_v, 3), "speculative_s": round(dt_s, 3),
        "speedup": round(dt_v / dt_s, 3), "tau_first_prompt": s["tau"],
    }
    print(f"[4] wall-clock: vanilla {dt_v:.2f}s vs spec {dt_s:.2f}s "
          f"({dt_v / dt_s:.2f}x, tau={s['tau']})", flush=True)

    # calibration proposal per spec: success line = ~60% oracle tau, abandon = ~40%
    report["calibration_proposal"] = {
        "oracle_tau": args.k + 1,
        "success_line": round(0.6 * (args.k + 1), 2),
        "abandon_line": round(0.4 * (args.k + 1), 2),
        "note": "回写 spec 后冻结;基线 tau 是未蒸馏 draft 的参照",
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["calibration_proposal"], indent=2), flush=True)


if __name__ == "__main__":
    main()
