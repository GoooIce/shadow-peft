"""P0 mount smoke for LFM2.5 on the GB10 (spec: .scratch/experiment-redesign/specs/p0-common.md).

Six checks against the real 350M checkpoint — any failure means fix the library
before experiments start. Run on the GB10:

    ssh goooinext 'cd ~/Code/shadow-peft && .venv/bin/python scripts/p0_smoke_lfm2.py'
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import tempfile
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from shadow_lab.atoms import bos_ids, eval_ppl
from shadow_peft import ShadowConfig, get_shadow_model
from shadow_peft.model_utils import _get_decoder_layers
from shadow_peft.quantization import (
    load_quantized_checkpoint,
    quantize_model_1bit,
    quantize_model_2bit,
    quantize_model_8bit,
    save_quantized_checkpoint,
)

SMOKE_TEXT = (
    "The shadow network tracks the residual stream of a frozen backbone. "
    "Each decoder layer injects a small correction learned by distillation, "
    "and the correction evolves layer by layer as the sequence grows. "
    "Quantizing the backbone to low bit widths damages the residual stream; "
    "the question is how much of that damage a high-precision shadow recovers."
) * 8

results: dict[str, dict] = {}


def check(name: str, ok: bool, **facts) -> None:
    results[name] = {"ok": bool(ok), **{k: v for k, v in facts.items()}}
    mark = "PASS" if ok else "FAIL"
    extras = " ".join(f"{k}={v}" for k, v in facts.items())
    print(f"[{mark}] {name}  {extras}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(Path.home() / "models" / "LFM2.5-350M"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="runs/p0_smoke_lfm2/report.json")
    args = ap.parse_args()
    dev = args.device

    tok = AutoTokenizer.from_pretrained(args.model)
    base = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(dev).eval()
    ids = torch.tensor([bos_ids(tok, SMOKE_TEXT)], device=dev)

    # ---- 1. mount: decoder layers + layer-type census + shadow forward ----
    try:
        holder, layers, path = _get_decoder_layers(base)
        n = len(layers)
        types = getattr(base.config, "layer_types", None)
        census = {"conv": types.count("conv"), "attn": types.count("full_attention")} if types else {}
        cfg = ShadowConfig(num_shadow_layers=2)
        # Mount on a copy: mounting wraps layers in place, and later checks need
        # the plain backbone untouched.
        peft = get_shadow_model(copy.deepcopy(base), cfg).eval()
        with torch.no_grad():
            logits = peft(input_ids=ids).logits
        vocab = logits.shape[-1]
        check("1_mount", n == 16 and logits.shape[1] == ids.shape[1], layers_path=path,
              n_layers=n, census=census, vocab=vocab, hidden=base.config.hidden_size)
    except Exception as exc:
        check("1_mount", False, error=repr(exc))
        return

    # ---- 2. conv cache: prefill + incremental step + generation path ----
    try:
        with torch.no_grad():
            o1 = peft(input_ids=ids[:, :8], use_cache=True)
            o2 = peft(input_ids=ids[:, 8:16], past_key_values=o1.past_key_values, use_cache=True)
        # Generation goes through the task wrapper (its forward initializes shadow state).
        from shadow_peft.task_models import ShadowForCausalLM

        lm = ShadowForCausalLM(peft).eval()
        with torch.no_grad():
            gen = lm.generate(ids[:, :4], max_new_tokens=8, do_sample=False)
        check("2_conv_cache_forward", o2.logits is not None and gen.shape[1] == 12,
              seq_after_prefill=o1.past_key_values.get_seq_length(), gen_tokens=int(gen.shape[1]))
    except Exception as exc:
        check("2_conv_cache_forward", False, error=repr(exc))

    # ---- cache structure with vs without shadow (clean copy: mounting mutates in place) ----
    def cache_keys(model) -> list[str]:
        with torch.no_grad():
            o = model(input_ids=ids[:, :4], use_cache=True)
        c = o.past_key_values
        return sorted({k for layer in c.layers for k in vars(layer)}) if hasattr(c, "layers") else sorted(vars(c))

    try:
        k_shadow, k_plain = cache_keys(peft), cache_keys(base)
        check("4_cache_parity", k_shadow == k_plain, keys=k_shadow[:6])
    except Exception as exc:
        check("4_cache_parity", False, error=repr(exc))

    # ---- 3. swap_adapter: bit-identical after round trip ----
    try:
        with torch.no_grad():
            ref = peft(input_ids=ids).logits
        sd = peft.adapter_state_dict()
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "a.safetensors"
            from safetensors.torch import save_file

            save_file(sd, str(p))
            peft.swap_adapter(load_file(str(p)))
            with torch.no_grad():
                after = peft(input_ids=ids).logits
        check("3_swap_adapter", torch.equal(ref, after), n_tensors=len(sd),
              n_params=sum(v.numel() for v in sd.values()))
    except Exception as exc:
        check("3_swap_adapter", False, error=repr(exc))

    # ---- 5. tied embedding + one training step ----
    try:
        emb = base.get_input_embeddings().weight
        head = base.get_output_embeddings()
        tied = head is not None and head.weight is emb
        peft.train()
        loss = peft(input_ids=ids, labels=ids, use_cache=False).loss
        loss.backward()
        grads = [n for n, p in peft.named_parameters() if p.requires_grad and p.grad is not None]
        peft.eval()
        check("5_tied_embedding_train", loss.isfinite() and len(grads) > 0,
              tied=tied, loss=float(loss), trainable_tensors=len(grads))
    except Exception as exc:
        check("5_tied_embedding_train", False, error=repr(exc))

    # ---- 6. quant schemes: PTQ + save/load roundtrip + damage ----
    try:
        ppl_fp16 = eval_ppl(base, tok, SMOKE_TEXT, chunks=1, device=dev)
        schemes = {
            "8bit": lambda m: quantize_model_8bit(m, group_size=128),
            "2bit": lambda m: quantize_model_2bit(m, group_size=128, thr_ratio=0.5),
            "1bit": lambda m: quantize_model_1bit(m, group_size=128),
        }
        rows = {}
        ok_all = True
        for name, fn in schemes.items():
            m1 = copy.deepcopy(base)
            manifest = fn(m1)
            with tempfile.TemporaryDirectory() as td:
                save_quantized_checkpoint(m1, manifest, td)
                m2 = copy.deepcopy(base)
                load_quantized_checkpoint(m2, td)
                with torch.no_grad():
                    l1 = m1(input_ids=ids).logits
                    l2 = m2(input_ids=ids).logits
                bit_equal = torch.equal(l1, l2)
            ppl_q = eval_ppl(m1, tok, SMOKE_TEXT, chunks=1, device=dev)
            rows[name] = {"roundtrip_bit_equal": bool(bit_equal), "ppl": round(ppl_q, 3)}
            ok_all = ok_all and bit_equal
        check("6_quant_schemes", ok_all and rows["8bit"]["ppl"] < rows["2bit"]["ppl"],
              fp16_ppl=round(ppl_fp16, 3), **rows)
    except Exception as exc:
        check("6_quant_schemes", False, error=repr(exc))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, default=str) + "\n")
    n_ok = sum(1 for r in results.values() if r["ok"])
    print(f"\n{n_ok}/{len(results)} checks passed -> {out}", flush=True)
    sys.exit(0 if n_ok == len(results) else 1)


if __name__ == "__main__":
    main()
