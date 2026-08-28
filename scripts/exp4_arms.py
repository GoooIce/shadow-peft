"""Exp4 arms: shadow-architecture heterogeneity ablation (spec: exp4).

Arms A1 (conv prefix), A3 (mixed) use the shadow_layer_types knob; arm B uses
an explicit cross-family Llama shadow parameter-matched to the implicit one.
All arms share exp1 rung B's teacher data, 2-bit damaged base, and eval axis.
A2 (= exp1 rung B) is read from its verdict, not retrained.

Run on the GB10:

    ssh goooinext 'cd ~/Code/shadow-peft && .venv/bin/python scripts/exp4_arms.py'
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig, LlamaModel

from shadow_lab.atoms import despace_wikitext, eval_ppl, load_wikitext2_test_text
from shadow_lab.trainer import TrainConfig, _quantize_base, train_shadow

AXIS_CTX = 512
AXIS_CHUNKS = 20
BASE_MODEL = str(Path.home() / "models" / "LFM2.5-350M-Base")
TEACHER_DATA = "runs/exp1/teacher_lfm25_350m_base.jsonl"

ARMS: dict[str, dict] = {
    "A1_conv": {"layer_types": ["conv", "conv"]},
    "A3_mixed": {"layer_types": ["conv", "full_attention"]},
    "B_llama": {"cross_family": True},
}


def build_llama_shadow(target_params: int) -> LlamaModel:
    """Parameter-match a 2-layer Llama shadow to the implicit LFM2 shadow."""
    hidden = 1024  # match base hidden -> no projection confound
    kv_heads = 8
    heads = 16
    # params/layer = attn(q 1024*1024 + k,v 2*1024*512 + o 1024*1024) + 3 * hidden * inter
    attn_per_layer = 2 * hidden * hidden + 2 * hidden * (hidden // 2)
    inter = max(256, round((target_params / 2 - attn_per_layer) / (3 * hidden) / 256) * 256)
    cfg = LlamaConfig(
        hidden_size=hidden, intermediate_size=inter, num_hidden_layers=2,
        num_attention_heads=heads, num_key_value_heads=kv_heads, vocab_size=65536,
    )
    m = LlamaModel(cfg)
    n = sum(p.numel() for p in m.parameters())
    print(f"llama shadow: inter={inter} params={n:,} (target≈{target_params:,})", flush=True)
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--out-dir", default="runs/exp4")
    ap.add_argument("--max-tokens", type=float, default=5e6)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    axis_text = despace_wikitext(load_wikitext2_test_text())

    def ppl(model) -> float:
        return eval_ppl(model, tok, axis_text, chunks=AXIS_CHUNKS, device="cuda", context_len=AXIS_CTX)

    # Shared reference points (identical across arms; computed once).
    ref_path = out_dir / "reference.json"
    if ref_path.exists():
        ref = json.loads(ref_path.read_text())
    else:
        base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.bfloat16).to("cuda").eval()
        ppl_bf16 = ppl(base)
        del base
        q = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.bfloat16).to("cuda").eval()
        probe_cfg = TrainConfig(bits=2, model=BASE_MODEL, data=TEACHER_DATA)
        _quantize_base(q, probe_cfg, None)
        ppl_quant = ppl(q)
        del q
        torch.cuda.empty_cache()
        ref = {"ppl_bf16": round(ppl_bf16, 4), "ppl_quant": round(ppl_quant, 4)}
        ref_path.write_text(json.dumps(ref, indent=2) + "\n")
    print("reference:", ref, flush=True)

    # Implicit-shadow param count for the parameter-matching rule (arm B).
    from shadow_peft import ShadowConfig, get_shadow_model

    probe = get_shadow_model(
        AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.bfloat16),
        ShadowConfig(num_shadow_layers=2),
    )
    implicit_params = sum(p.numel() for p in probe.shadow_model.parameters())
    del probe

    results = {}
    for arm in args.arms:
        spec = ARMS[arm]
        cfg = TrainConfig(
            bits=2, loss_mode="ce_kl", model=BASE_MODEL, data=TEACHER_DATA,
            max_tokens=args.max_tokens, seq_len=512, batch_seq=8, lr=1e-5,
            shadow_layers=2, thr_ratio=0.5, ppl_context_len=AXIS_CTX,
            pad_id=pad_id, shadow_layer_types=spec.get("layer_types"),
            save_dir=str(out_dir / arm / "train"),
        )
        shadow_model = build_llama_shadow(implicit_params) if spec.get("cross_family") else None
        report = train_shadow(cfg, shadow_model=shadow_model)
        rec = (ref["ppl_quant"] - report.final_ppl_20chunks) / (ref["ppl_quant"] - ref["ppl_bf16"])
        results[arm] = {
            "ppl_shadow": round(report.final_ppl_20chunks, 4),
            "recovery": round(rec, 4),
            "trainable_params": report.trainable_params,
            "tokens": report.tokens,
        }
        print(f"[{arm}] {json.dumps(results[arm])}", flush=True)

    a2 = json.loads(Path("runs/exp1/rungB/verdict.json").read_text())
    results["A2_attention(default)"] = {
        "ppl_shadow": a2["ppl_shadow"], "recovery": a2["recovery"],
        "trainable_params": a2["trainable_params"], "tokens": a2["tokens"],
    }
    (out_dir / "verdict.json").write_text(json.dumps(
        {"reference": ref, "arms": results}, indent=2) + "\n")
    print(json.dumps({"reference": ref, "arms": results}, indent=2), flush=True)


if __name__ == "__main__":
    main()
