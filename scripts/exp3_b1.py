"""Exp3 B1: recirculation boundary on the conv-hybrid LFM2.5 (training-free).

Paper recipe adapted to LFM2.5-350M-Base (RMSNorm + dual-residual, no
Peri-LN): deep->shallow activation recirculation — the source layer's output
at token t-1 (L2-norm-matched) is mixed into the destination layer's input at
token t: ``x_d = beta * x_d + alpha * sum_{i=1..r} s_hat(t-i)``. beta = 1-alpha
is the paper's convex mix, beta = 1 the pure-additive variant. s > d always
(the deeper source is not yet computed when d runs), which forces serial
per-token prefill — the eval walks one continuous token stream with a live KV
cache, scoring everything after the first context_len tokens.

Grid (spec-frozen central config, 48 cells): s in {13 attn, 14 conv} x
d in {3 conv, 4 conv, 5 attn} x alpha in {0.07, 0.15} x beta in {1-alpha, 1}
x r in {1, 2}. Judgment: any config ppl improvement >= 2% vs the same-machinery
alpha=0 baseline.

Run on the second GB10:
    ssh gooo 'cd ~/Code/shadow-peft && .venv/bin/python scripts/exp3_b1.py'
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from collections import deque
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from shadow_lab.atoms import despace_wikitext, load_wikitext2_test_text
from shadow_lab.specdec import _cached_kwargs


class Recirculator:
    """Deep->shallow recirculation via layer hooks (serial 1-token forwards).

    ``hist`` keeps the last r source-layer outputs (last position only);
    the destination pre-hook mixes norm-matched copies into its input.
    """

    def __init__(self, layers, s: int, d: int, alpha: float, beta: float, r: int):
        self.alpha, self.beta, self.r = alpha, beta, r
        self.hist: deque[torch.Tensor] = deque(maxlen=r)
        self._src = layers[s].register_forward_hook(self._capture)
        self._dst = layers[d].register_forward_pre_hook(self._pre)

    def _capture(self, _module, _args, output):
        h = output[0] if isinstance(output, tuple) else output
        self.hist.append(h[:, -1:].detach())

    def _pre(self, _module, args):
        (x,) = args
        if not self.hist:
            return args
        new = self.beta * x
        xnorm = x.norm(dim=-1, keepdim=True)
        for h in self.hist:
            hnorm = h.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            new = new + self.alpha * h * (xnorm / hnorm)
        return (new,)

    def close(self):
        self._src.remove()
        self._dst.remove()


@torch.no_grad()
def serial_ppl(model, ids: list[int], *, context_len: int, device: str,
               limit: int) -> float:
    """Continuous-stream ppl: 1-token cached forwards, score from context_len on."""
    cache = None
    total_nll, total_tok = 0.0, 0
    n = min(len(ids), limit)
    for t in range(n):
        tok = torch.tensor([[ids[t]]], device=device)
        if t == 0:
            out = model(input_ids=tok, use_cache=True,
                        attention_mask=torch.ones(1, 1, dtype=torch.long, device=device))
        else:
            out = model(input_ids=tok, past_key_values=cache,
                        **_cached_kwargs(cache, 1, device))
        cache = out.past_key_values
        if t >= context_len and t + 1 < n:
            logp = out.logits[0, -1].float()
            logp = logp - torch.logsumexp(logp, -1)
            total_nll -= logp[ids[t + 1]].item()
            total_tok += 1
    return math.exp(total_nll / max(total_tok, 1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(Path.home() / "models" / "LFM2.5-350M-Base"))
    ap.add_argument("--out", default="runs/exp3/b1")
    ap.add_argument("--context-len", type=int, default=512)
    ap.add_argument("--limit", type=int, default=16000, help="scored stream length per config")
    ap.add_argument("--recheck-limit", type=int, default=48000)
    ap.add_argument("--top", type=int, default=5)
    args = ap.parse_args()
    device = "cuda"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    text = despace_wikitext(load_wikitext2_test_text())
    ids = tok.encode(text)
    b = getattr(tok, "bos_token_id", None)
    if b is not None and ids and ids[0] != b:
        ids = [b] + ids
    print(f"stream: {len(ids)} tokens (despaced wikitext + BOS)", flush=True)

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(device).eval()
    layers = model.model.layers

    def run(alpha, beta, r, s, d, limit):
        recirc = Recirculator(layers, s, d, alpha, beta, r) if alpha > 0 else None
        t0 = time.time()
        ppl = serial_ppl(model, ids, context_len=args.context_len, device=device,
                         limit=limit)
        if recirc:
            recirc.close()
        return ppl, round(time.time() - t0, 1)

    base_ppl, base_s = run(0.0, 1.0, 1, 13, 4, args.limit)
    base_rec = {"config": "baseline_alpha0", "ppl": round(base_ppl, 4), "s": base_s}
    print(json.dumps(base_rec), flush=True)
    with open(out_dir / "sweep.jsonl", "a") as f:
        f.write(json.dumps(base_rec) + "\n")

    rows = []
    for s, d, alpha, r in itertools.product((13, 14), (3, 4, 5), (0.07, 0.15), (1, 2)):
        beta = 1.0 - alpha
        for beta_v in (beta, 1.0):
            ppl, sec = run(alpha, beta_v, r, s, d, args.limit)
            rec = {"s": s, "d": d, "alpha": alpha, "beta": round(beta_v, 4), "r": r,
                   "ppl": round(ppl, 4), "delta_pct": round((ppl / base_ppl - 1) * 100, 3),
                   "s_sec": sec}
            rows.append(rec)
            print(json.dumps(rec), flush=True)
            with open(out_dir / "sweep.jsonl", "a") as f:
                f.write(json.dumps(rec) + "\n")

    rows.sort(key=lambda x: x["ppl"])
    top = rows[: args.top]
    print(f"top-{args.top} @ {args.limit}: " + json.dumps(top), flush=True)

    # full-length recheck of the top cells + baseline
    base_ppl_l, _ = run(0.0, 1.0, 1, 13, 4, args.recheck_limit)
    recheck = []
    for rec in top:
        ppl, sec = run(rec["alpha"], rec["beta"], rec["r"], rec["s"], rec["d"],
                       args.recheck_limit)
        recheck.append({**rec, "ppl_long": round(ppl, 4), "base_long": round(base_ppl_l, 4),
                        "delta_pct_long": round((ppl / base_ppl_l - 1) * 100, 3)})
        print(json.dumps(recheck[-1]), flush=True)

    best = min(recheck, key=lambda x: x["delta_pct_long"])
    verdict = {
        "axis": "despaced wikitext + BOS, continuous-stream serial ppl (unbounded carry), bf16",
        "baseline_ppl": {"16k": round(base_ppl, 4), "48k": round(base_ppl_l, 4)},
        "grid": "s{13,14} x d{3,4,5} x a{0.07,0.15} x b{1-a,1} x r{1,2} = 48",
        "best_config": best,
        "judgment": "PASS_mix_injectable" if best["delta_pct_long"] <= -2.0 else
                    ("NEGATIVE_boundary" if best["delta_pct_long"] >= 0 else "INCONCLUSIVE_small"),
        "success_line": "any config ppl improvement >= 2% at recheck length",
    }
    (out_dir / "verdict.json").write_text(json.dumps(verdict, indent=2) + "\n")
    print(json.dumps(verdict, indent=2), flush=True)


if __name__ == "__main__":
    main()
