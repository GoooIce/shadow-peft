"""Stage-0 eval-axis freeze for LFM2.5 (spec: p0-common 阶段 0).

Measures the ppl axis sensitivity matrix on the real checkpoints:

    {raw, despaced} wikitext x {BOS, no-BOS} x {ctx=0, ctx=512} x 3 repeats

plus tokenizer facts. Freezes the axis every later experiment must use.
Run on the GB10:

    ssh goooinext 'cd ~/Code/shadow-peft && .venv/bin/python scripts/stage0_eval_axis.py'
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from shadow_lab.atoms import despace_wikitext, load_wikitext2_test_text


@torch.no_grad()
def _ppl(model, tok, text: str, *, bos: bool, chunks: int = 8, chunk_len: int = 512,
         device: str, context_len: int = 0) -> float:
    """Mirror of shadow_lab.atoms.eval_ppl with an explicit BOS switch."""
    ids = tok.encode(text)
    if bos:
        b = getattr(tok, "bos_token_id", None)
        if b is not None and (not ids or ids[0] != b):
            ids = [b] + ids
    model.eval()
    total_nll, total_tok = 0.0, 0
    limit = min(len(ids), chunks * chunk_len) - chunk_len - 1
    if limit <= 0 and len(ids) >= chunk_len + 1:
        limit = 1
    for start in range(0, limit, chunk_len):
        ctx = max(0, start - context_len)
        x = torch.tensor(ids[ctx : start + chunk_len + 1], device=device)[None]
        logits = model(input_ids=x).logits.float()
        logp = logits[:, :-1] - torch.logsumexp(logits[:, :-1], -1, keepdim=True)
        tgt = x[:, 1:]
        if context_len > 0 and start > 0:
            keep_from = start - ctx - 1
            logp = logp[:, keep_from:]
            tgt = tgt[:, keep_from:]
        total_nll += -logp.gather(-1, tgt.unsqueeze(-1)).sum().item()
        total_tok += tgt.numel()
    return math.exp(total_nll / total_tok)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=[
        str(Path.home() / "models" / "LFM2.5-350M-Base"),
        str(Path.home() / "models" / "LFM2.5-350M"),
    ], help="paths: quant axis (Base) first, then the it model")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", default="runs/stage0_eval_axis/report.json")
    args = ap.parse_args()

    raw_text = load_wikitext2_test_text()
    de_text = despace_wikitext(raw_text)
    report: dict = {"wikitext_chars": len(raw_text), "despaced_chars": len(de_text), "models": {}}

    for mp in args.models:
        name = Path(mp).name
        tok = AutoTokenizer.from_pretrained(mp)
        model = AutoModelForCausalLM.from_pretrained(mp, dtype=torch.bfloat16).to(args.device).eval()
        facts = {
            "bos_token_id": getattr(tok, "bos_token_id", None),
            "adds_bos_by_default": bool(getattr(tok, "add_bos_token", False)),
            "vocab_size": len(tok),
        }
        arms: dict = {}
        for text_name, text in [("raw", raw_text), ("despaced", de_text)]:
            for bos in [True, False]:
                for ctx in [0, 512]:
                    key = f"{text_name}|bos={int(bos)}|ctx={ctx}"
                    vals = [_ppl(model, tok, text, bos=bos, context_len=ctx, device=args.device)
                            for _ in range(args.repeats)]
                    mean = sum(vals) / len(vals)
                    rel_std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5 / mean
                    arms[key] = {"ppl": round(mean, 4), "rel_std": round(rel_std, 6),
                                 "vals": [round(v, 4) for v in vals]}
                    print(f"[{name}] {key}: ppl={mean:.4f} rel_std={rel_std:.5f}", flush=True)
        report["models"][name] = {"facts": facts, "arms": arms}
        del model
        torch.cuda.empty_cache()

    base = report["models"][Path(args.models[0]).name]["arms"]
    de_bos0 = base["despaced|bos=1|ctx=0"]["ppl"]
    summary = {
        "axis_choice": "despaced|bos=1|ctx=512 (frozen, pending variance check)",
        "tokenizer_ood_raw_over_despaced": round(base["raw|bos=1|ctx=0"]["ppl"] / de_bos0, 2),
        "bos_sensitivity_noBOS_over_BOS": round(base["despaced|bos=0|ctx=0"]["ppl"] / de_bos0, 2),
        "ctx_sensitivity_cold_over_carried": round(de_bos0 / base["despaced|bos=1|ctx=512"]["ppl"], 2),
        "max_rel_std": max(a["rel_std"] for m in report["models"].values() for a in m["arms"].values()),
        "variance_gate_lt_1pct": None,
    }
    summary["variance_gate_lt_1pct"] = summary["max_rel_std"] < 0.01
    report["summary"] = summary

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print("\nsummary:", json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
