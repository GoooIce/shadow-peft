"""exp1-solo arm runner: shadow-only generation arms (spec: exp1-solo).

Two modes on the frozen L1_5x structure (2.6B x w4a16 testbed):

  train      --solo-weight 0.3|1.0  -> arm S3/S10: ce_kl distillation with the
             shadow-path CE weight raised (TrainConfig.shadow_loss_weight —
             the existing knob; 0.05 is the historical default). The verdict
             carries BOTH roles: recovery (base_shadow ppl/KLD, floor-series
             isomorphic) and solo (shadow_only ppl/KLD + a greedy probe
             sample, judgment-free eyeball evidence).

  eval-only  --ckpt <dir> [--src-verdict <verdict.json>]  -> solo-role
             evaluation of an EXISTING checkpoint (refs at lambda=0.05);
             recovery fields copied from the source verdict when given.

Solo axis: base pass = despaced wikitext (same as the primary axis); rl pass
= held-out self-distillation pool full-row stream (the rl line's primary
axis — wiki is distorted on that model).

    ssh <host> 'cd ~/Code/shadow-peft && nohup .venv/bin/python \
      scripts/exp1_solo_arm.py --solo-weight 0.3 --data runs/exp1/teacher_2p6b_base.jsonl \
      > runs/logs/<date>-solo-S3.log 2>&1 < /dev/null &'
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from shadow_lab import tiers
from shadow_lab.atoms import (
    despace_wikitext,
    eval_kld,
    eval_kld_ids,
    eval_ppl,
    eval_ppl_ids,
    load_teacher_data,
    load_wikitext2_test_text,
)
from shadow_lab.trainer import TrainConfig, train_shadow
from shadow_peft import ShadowForCausalLM, ShadowPeftModel

AXIS_CTX = 512
AXIS_CHUNKS = 20
SAMPLE_TOKENS = 40
PROBE_BASE = "The capital of France is"
PROBE_RL = "用一句话解释什么是模型量化"
WEIGHT_ARMS = {0.3: "S3", 1.0: "S10"}


def load_shadow_instance(shadow_path: str, base_model_path: str):
    """Explicit shadow instance with the embedding table resized to the BASE
    vocab — the small LFM2.5 checkpoints ship 65,536 rows while the 2.6B
    token space is 128,000 (same family tokenizer, cropped tables); feeding
    base ids into the cropped table device-asserts."""
    from transformers import AutoConfig

    inst = AutoModelForCausalLM.from_pretrained(shadow_path, dtype=torch.bfloat16)
    target = AutoConfig.from_pretrained(base_model_path).vocab_size
    current = inst.get_input_embeddings().weight.shape[0]
    if current < target:
        print(f"resizing shadow embeddings {current} -> {target} (base vocab)", flush=True)
        inst.resize_token_embeddings(target)
    return inst


def build_solo(model_path: str, ckpt: str, shadow_path: str | None = None,
               device: str = "cuda") -> ShadowForCausalLM:
    base_q = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16)
    tiers.apply_training_conversion(base_q, "w4a16", torch.device(device), fake_quant=True)
    shadow_inst = load_shadow_instance(shadow_path, model_path) if shadow_path else None
    peft = ShadowPeftModel.from_pretrained(base_q.to(device), ckpt,
                                           shadow_model=shadow_inst).eval()
    return ShadowForCausalLM(peft, inference_mode="shadow_only").eval()


def greedy_sample(model: ShadowForCausalLM, tok, prompt: str, chat: bool, n: int) -> str:
    if chat:
        text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                       add_generation_prompt=True, tokenize=False)
        ids = tok(text, add_special_tokens=False).input_ids
    else:
        ids = tok(prompt, add_special_tokens=False).input_ids
        if tok.bos_token_id is not None:
            ids = [tok.bos_token_id] + ids
    ids = torch.tensor([ids], device="cuda")
    produced = []
    with torch.no_grad():
        for _ in range(n):
            lg = model(input_ids=ids).logits[0, -1].float()
            nxt = int(lg.argmax())
            if tok.eos_token_id is not None and nxt == tok.eos_token_id:
                break
            ids = torch.cat([ids, torch.tensor([[nxt]], device=ids.device)], dim=1)
            produced.append(nxt)
    return tok.decode(produced)


def main() -> None:
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--solo-weight", type=float, default=None,
                      help=f"train mode: shadow-path CE weight (one of {sorted(WEIGHT_ARMS)})")
    mode.add_argument("--ckpt", default=None,
                      help="eval-only mode: existing shadow checkpoint dir")
    ap.add_argument("--shadow-model", default=None,
                    help="explicit pretrained shadow model path (e.g. ~/models/LFM2.5-350M-Base); "
                         "train mode mounts it, eval paths reload checkpoints onto it")
    ap.add_argument("--src-verdict", default=None,
                    help="eval-only: copy recovery fields from this verdict")
    ap.add_argument("--model", default=str(Path.home() / "models" / "LFM2.5-2.6B-Base"))
    ap.add_argument("--data", default=None, help="teacher JSONL pool (train mode)")
    ap.add_argument("--heldout", default=None,
                    help="held-out pool for the rl-pass SOLO axis (ids stream)")
    ap.add_argument("--tag", default="", help="'' = base -> solo_{arm}; 'rl' -> solo-rl_{arm}")
    ap.add_argument("--suffix", default="",
                    help="train mode: free-form arm name (e.g. E35 for the explicit-shadow arms)")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--max-tokens", type=float, default=1e7)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--baselines-from", default=None)
    args = ap.parse_args()

    is_rl = args.tag == "rl"
    prefix = "solo-rl" if is_rl else "solo"
    chat = is_rl
    if is_rl and not args.heldout:
        ap.error("the rl pass needs --heldout (solo axis = held-out pool stream)")
    if args.solo_weight is not None and args.solo_weight not in WEIGHT_ARMS:
        ap.error(f"--solo-weight must be one of {sorted(WEIGHT_ARMS)}")
    if args.solo_weight is not None and not args.data:
        ap.error("train mode needs --data")

    if args.ckpt:
        src = Path(args.ckpt)
        arm = f"ref_{src.name}"
        out_dir = Path(args.out_dir or f"runs/exp1/{prefix}_{arm}")
    else:
        arm = args.suffix or WEIGHT_ARMS[args.solo_weight]
        out_dir = Path(args.out_dir or f"runs/exp1/{prefix}_{arm}")
    out_dir.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    # --- solo axis: wiki text (base) or held-out ids stream (rl) ---
    teacher = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to("cuda").eval()
    solo_stream: list[int] | None = None
    axis_text = None
    if is_rl:
        solo_stream = []
        for row in load_teacher_data(args.heldout, min_len=16):
            solo_stream.extend(row)
        print(f"solo axis: held-out stream, {len(solo_stream):,} tokens", flush=True)
    else:
        axis_text = despace_wikitext(load_wikitext2_test_text())

    def solo_eval(ckpt: str) -> dict:
        print(f"=== shadow_only eval ({ckpt}) ===", flush=True)
        solo = build_solo(args.model, ckpt, shadow_path=args.shadow_model)
        if is_rl:
            ppl = eval_ppl_ids(solo, solo_stream, chunks=AXIS_CHUNKS, device="cuda",
                               context_len=AXIS_CTX)
            kld = eval_kld_ids(teacher, solo, solo_stream, chunks=AXIS_CHUNKS, device="cuda",
                               context_len=AXIS_CTX)
        else:
            ppl = eval_ppl(solo, tok, axis_text, chunks=AXIS_CHUNKS, device="cuda",
                           context_len=AXIS_CTX)
            kld = eval_kld(teacher, solo, tok, axis_text, chunks=AXIS_CHUNKS, device="cuda",
                           context_len=AXIS_CTX)
        sample = greedy_sample(solo, tok, PROBE_RL if chat else PROBE_BASE, chat, SAMPLE_TOKENS)
        out = {"solo_ppl": round(ppl, 4), "solo_kld": round(kld, 4), "solo_sample": sample}
        print(json.dumps(out, ensure_ascii=False, indent=2), flush=True)
        del solo
        torch.cuda.empty_cache()
        return out

    if args.ckpt:
        solo = solo_eval(args.ckpt + "/train/final")
        src_v = json.loads(Path(args.src_verdict).read_text()) if args.src_verdict else {}
        verdict = {
            "arm": arm,
            "campaign": args.tag or "base",
            "mode": "eval_only",
            "src_ckpt": args.ckpt,
            "solo_weight": src_v.get("shadow_loss_weight", src_v.get("solo_weight")),
            "tier": "w4a16",
            "model": args.model,
            "shadow_layers": src_v.get("shadow_layers", 1),
            "shadow_intermediate": src_v.get("shadow_intermediate", 53760),
            "trainable_params": src_v.get("trainable_params"),
            **{k: src_v.get(k) for k in
               ("ppl_bf16", "ppl_quant", "ppl_shadow", "recovery", "kld_shadow", "kld_quant",
                "tokens", "loss_mode")},
            **solo,
            "axis": "held-out self-distilled pool (solo)" if is_rl
                    else "despaced wikitext, ctx=512, chunks=20 (solo)",
        }
        (out_dir / "verdict.json").write_text(json.dumps(verdict, indent=2) + "\n")
        print(json.dumps(verdict, ensure_ascii=False, indent=2), flush=True)
        return

    # --- train mode ---
    if args.baselines_from:
        prior = json.loads(Path(args.baselines_from).read_text())
        ppl_bf16, ppl_quant, kld_quant = (prior[k] for k in ("ppl_bf16", "ppl_quant", "kld_quant"))
        print(f"reusing baselines from {args.baselines_from}: "
              f"bf16={ppl_bf16} quant={ppl_quant} kld_quant={kld_quant}", flush=True)
    else:
        print("=== 1/4 bf16 baseline ===", flush=True)
        base = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to("cuda").eval()

        def _ppl(m):
            return (eval_ppl_ids(m, solo_stream, chunks=AXIS_CHUNKS, device="cuda", context_len=AXIS_CTX)
                    if is_rl else
                    eval_ppl(m, tok, axis_text, chunks=AXIS_CHUNKS, device="cuda", context_len=AXIS_CTX))

        def _kld(ref, m):
            return (eval_kld_ids(ref, m, solo_stream, chunks=AXIS_CHUNKS, device="cuda", context_len=AXIS_CTX)
                    if is_rl else
                    eval_kld(ref, m, tok, axis_text, chunks=AXIS_CHUNKS, device="cuda", context_len=AXIS_CTX))

        ppl_bf16 = round(_ppl(base), 4)
        quant_copy = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
        tiers.apply_training_conversion(quant_copy, "w4a16", torch.device("cuda"), fake_quant=True)
        quant_copy = quant_copy.to("cuda").eval()
        ppl_quant = round(_ppl(quant_copy), 4)
        kld_quant = round(_kld(base, quant_copy), 4)
        print(f"ppl_bf16={ppl_bf16} ppl_quant={ppl_quant} kld_quant={kld_quant}", flush=True)
        del base, quant_copy
        torch.cuda.empty_cache()

    print(f"=== 2/4 arm {arm} (ce_kl + shadow-CE x{args.solo_weight}"
          f"{', explicit shadow ' + args.shadow_model if args.shadow_model else ''}) ===", flush=True)
    shadow_inst = load_shadow_instance(args.shadow_model, args.model) if args.shadow_model else None
    cfg = TrainConfig(
        bits=0,
        backbone_tier="w4a16",
        tier_fake_quant=True,
        loss_mode="ce_kl",
        model=args.model,
        data=args.data,
        max_tokens=args.max_tokens,
        seq_len=512,
        batch_seq=8,
        lr=args.lr,
        shadow_layers=1,
        shadow_intermediate_size=53760,
        shadow_loss_weight=args.solo_weight,
        ppl_context_len=AXIS_CTX,
        eval_every=250,
        pad_id=pad_id,
        save_dir=str(out_dir / "train"),
    )
    report = train_shadow(cfg, shadow_model=shadow_inst)
    ppl_shadow = report.final_ppl_20chunks  # internal axis; rl recovery judged on axis2
    del shadow_inst

    print("=== 3/4 KLD(bf16 || quant+shadow) ===", flush=True)
    base = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to("cuda").eval()
    eval_base = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    tiers.apply_training_conversion(eval_base, "w4a16", torch.device("cuda"), fake_quant=True)
    reload_shadow = load_shadow_instance(args.shadow_model, args.model) if args.shadow_model else None
    student = ShadowPeftModel.from_pretrained(eval_base.to("cuda"), str(out_dir / "train" / "final"),
                                              shadow_model=reload_shadow).eval()
    if is_rl:
        # wiki axis for the process record only; recovery judgment lands in the axis2 re-eval
        axis_text_wiki = despace_wikitext(load_wikitext2_test_text())
        kld_shadow = round(eval_kld(base, student, tok, axis_text_wiki, chunks=AXIS_CHUNKS,
                                    device="cuda", context_len=AXIS_CTX), 4)
    else:
        kld_shadow = round(eval_kld(base, student, tok, axis_text, chunks=AXIS_CHUNKS,
                                    device="cuda", context_len=AXIS_CTX), 4)
    print(f"kld_shadow(wiki) = {kld_shadow}", flush=True)
    del base, eval_base, student
    torch.cuda.empty_cache()

    solo = solo_eval(str(out_dir / "train" / "final"))
    recovery = (ppl_quant - ppl_shadow) / (ppl_quant - ppl_bf16)
    verdict = {
        "arm": arm,
        "campaign": args.tag or "base",
        "mode": "train",
        "tier": "w4a16",
        "loss_mode": "ce_kl",
        "solo_weight": args.solo_weight,
        "shadow_model": args.shadow_model,
        "model": args.model,
        "shadow_layers": 1,
        "shadow_intermediate": 53760,
        "shadow_quant": None,
        "trainable_params": report.trainable_params,
        "ppl_bf16": ppl_bf16,
        "ppl_quant": ppl_quant,
        "ppl_shadow": round(ppl_shadow, 4),
        "recovery": round(recovery, 4),
        "kld_quant": kld_quant,
        "kld_shadow": kld_shadow,
        **solo,
        "axis": "despaced wikitext, ctx=512, chunks=20",
        "solo_axis": "held-out self-distilled pool" if is_rl else "despaced wikitext",
        "tokens": report.tokens,
        "train_seconds": round(report.wall_time_s, 1),
        "data": args.data,
        "lr": args.lr,
        "baselines_from": args.baselines_from,
    }
    (out_dir / "verdict.json").write_text(json.dumps(verdict, indent=2) + "\n")
    print(json.dumps(verdict, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
