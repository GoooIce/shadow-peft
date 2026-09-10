#!/usr/bin/env python3
"""MiniCPM5-2B BF16 bring-up smoke test (map ticket 06).

Records: env versions, chat-template rendering for the three enable_thinking
states, dual-EOS stop behaviour, load time, unified/GPU memory and tokens/s.

Run:
    python bf16_smoke.py --model ~/models/MiniCPM5-2B --out ~/runs/minicpm5/bf16-smoke.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import subprocess
import sys
import time
from datetime import datetime, timezone

import torch


PROMPT = "用一句话解释什么是 NVFP4 量化，并说明它和 FP8 的区别。"
MAX_NEW_TOKENS = 192


def mem_snapshot() -> dict:
    snap = {
        "cuda_allocated_bytes": torch.cuda.memory_allocated(),
        "cuda_reserved_bytes": torch.cuda.memory_reserved(),
        "cuda_max_allocated_bytes": torch.cuda.max_memory_allocated(),
        "cuda_max_reserved_bytes": torch.cuda.max_memory_reserved(),
        "process_maxrss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    }
    try:
        out = subprocess.run(
            ["free", "-b"], capture_output=True, text=True, timeout=10
        ).stdout.splitlines()
        for line in out:
            if line.startswith("Mem:"):
                parts = line.split()
                snap["system_mem_total_bytes"] = int(parts[1])
                snap["system_mem_available_bytes"] = int(parts[-1])
    except Exception as exc:  # pragma: no cover - best effort
        snap["free_error"] = repr(exc)
    return snap


def eos_report(tokenizer, eos_ids: list[int]) -> dict:
    return {
        "eos_token_id_config": eos_ids,
        "eos_token_id_type": type(eos_ids[0]).__name__ if eos_ids else None,
        "eos_tokens": [tokenizer.convert_ids_to_tokens(i) for i in eos_ids],
        "pad_token_id": tokenizer.pad_token_id,
        "pad_token": tokenizer.pad_token,
        "bos_token_id": tokenizer.bos_token_id,
        "bos_token": tokenizer.bos_token,
    }


def render_states(tokenizer, messages: list[dict]) -> dict:
    states = {}
    for name, kwargs in (
        ("enable_thinking_true", {"enable_thinking": True}),
        ("enable_thinking_false", {"enable_thinking": False}),
        ("unset", {}),
    ):
        try:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, **kwargs
            )
            ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            states[name] = {
                "ok": True,
                "rendered_repr": repr(text),
                "n_tokens": len(ids),
                "first_ids": ids[:12],
                "first_tokens": tokenizer.convert_ids_to_tokens(ids[:12]),
                "last_tokens": tokenizer.convert_ids_to_tokens(ids[-12:]),
                "tail_repr": repr(text[-160:]),
            }
        except Exception as exc:
            states[name] = {"ok": False, "error": repr(exc)}
    return states


def run_one(model, tokenizer, messages, enable_thinking, max_new_tokens, eos_ids,
            eos_override=None):
    kwargs = {} if enable_thinking is None else {"enable_thinking": enable_thinking}
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, **kwargs
    )
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to("cuda")
    prompt_tokens = int(inputs["input_ids"].shape[1])

    gen_kwargs = {}
    if eos_override is not None:
        gen_kwargs["eos_token_id"] = eos_override

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy: deterministic, isolates stop behaviour
            pad_token_id=tokenizer.pad_token_id,
            **gen_kwargs,
        )
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0

    gen_ids = out[0][prompt_tokens:].tolist()
    text_out = tokenizer.decode(gen_ids, skip_special_tokens=False)

    stopped_by = None
    for pos, tid in enumerate(gen_ids):
        if tid in eos_ids:
            stopped_by = {
                "position": pos,
                "token_id": tid,
                "token": tokenizer.convert_ids_to_tokens(tid),
                "is_last": pos == len(gen_ids) - 1,
                "matched_eos_list": eos_ids,
            }
            break
    n_new = len(gen_ids)
    return {
        "enable_thinking": enable_thinking,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": n_new,
        "wall_seconds": round(wall, 3),
        "tokens_per_second": round(n_new / wall, 2) if wall > 0 else None,
        "stop": stopped_by,
        "reached_max_new_tokens": n_new >= max_new_tokens
        and (stopped_by is None or not stopped_by["is_last"]),
        "output_text": text_out,
        "output_token_ids_head": gen_ids[:16],
        "output_token_ids_tail": gen_ids[-8:],
        "output_contains": {
            "id_1": 1 in gen_ids,
            "id_130073": 130073 in gen_ids,
            "id_130073_positions": [i for i, t in enumerate(gen_ids) if t == 130073][:5],
        },
        "mem_after": mem_snapshot(),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.expanduser("~/models/MiniCPM5-2B"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    ap.add_argument("--repeat", action="store_true", help="repeat one mode for determinism check")
    ap.add_argument("--eos-ablation", action="store_true",
                    help="force eos_token_id=[1] and show <|im_end|> no longer stops")
    args = ap.parse_args()

    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    report: dict = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "device": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "model_path": args.model,
        "prompt": PROMPT,
        "max_new_tokens": args.max_new_tokens,
    }

    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    report["tokenizer_load_seconds"] = round(time.perf_counter() - t0, 2)
    report["tokenizer_class"] = type(tokenizer).__name__
    eos_ids = tokenizer.eos_token_id
    eos_ids = eos_ids if isinstance(eos_ids, list) else [eos_ids]
    report["tokenizer"] = eos_report(tokenizer, eos_ids)
    report["chat_template_len"] = len(tokenizer.chat_template or "")
    report["chat_template_from_jinja_file"] = os.path.exists(
        os.path.join(args.model, "chat_template.jinja")
    )
    report["special_tokens_map"] = tokenizer.special_tokens_map

    messages = [{"role": "user", "content": PROMPT}]
    report["chat_template_render"] = render_states(tokenizer, messages)

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()
    torch.cuda.synchronize()
    report["model_load_seconds"] = round(time.perf_counter() - t0, 2)
    cfg = model.config
    cfg_dict = cfg.to_dict()
    # transformers >=5.16 nests rope_theta under rope_parameters; older versions
    # expose cfg.rope_theta directly. Record both so the note is version-proof.
    report["config"] = {
        "architectures": getattr(cfg, "architectures", None),
        "num_hidden_layers": getattr(cfg, "num_hidden_layers", None),
        "hidden_size": getattr(cfg, "hidden_size", None),
        "intermediate_size": getattr(cfg, "intermediate_size", None),
        "num_attention_heads": getattr(cfg, "num_attention_heads", None),
        "num_key_value_heads": getattr(cfg, "num_key_value_heads", None),
        "vocab_size": getattr(cfg, "vocab_size", None),
        "max_position_embeddings": getattr(cfg, "max_position_embeddings", None),
        "rope_parameters": cfg_dict.get("rope_parameters"),
        "rope_theta_getattr": getattr(cfg, "rope_theta", None),
        "tie_word_embeddings": getattr(cfg, "tie_word_embeddings", None),
        "dtype": str(getattr(cfg, "dtype", None)),
        "transformers_version_in_config": cfg_dict.get("transformers_version"),
        "bos_token_id": getattr(cfg, "bos_token_id", None),
        "eos_token_id": getattr(cfg, "eos_token_id", None),
        "pad_token_id": getattr(cfg, "pad_token_id", None),
    }
    report["param_count"] = sum(p.numel() for p in model.parameters())
    report["param_count_non_embedding"] = report["param_count"] - sum(
        p.numel() for n, p in model.named_parameters() if "embed_tokens" in n or "lm_head" in n
    )

    # Dual-EOS bookkeeping: tokenizer exposes only </s> (id 1); the <|im_end|>
    # stop (id 130073) lives in generation_config. Record both and use the
    # union for stop detection, otherwise a <|im_end|> stop looks like a miss.
    gc = model.generation_config
    gc_eos = gc.eos_token_id
    gc_eos = gc_eos if isinstance(gc_eos, list) else [gc_eos]
    stop_ids = sorted(set(gc_eos) | set(eos_ids))
    report["generation_config"] = {
        "eos_token_id": gc_eos,
        "eos_tokens": [tokenizer.convert_ids_to_tokens(i) for i in gc_eos],
        "pad_token_id": gc.pad_token_id,
        "bos_token_id": gc.bos_token_id,
        "do_sample": gc.do_sample,
        "temperature": gc.temperature,
        "top_p": gc.top_p,
    }
    report["stop_token_ids_used"] = stop_ids
    report["stop_token_union_tokens"] = [
        tokenizer.convert_ids_to_tokens(i) for i in stop_ids
    ]
    report["mem_after_load"] = mem_snapshot()

    results = {}
    for name, flag in (
        ("thinking_true", True),
        ("thinking_false", False),
        ("unset", None),
    ):
        try:
            results[name] = run_one(
                model, tokenizer, messages, flag, args.max_new_tokens, stop_ids
            )
            print(f"[{name}] {results[name]['tokens_per_second']} tok/s "
                  f"gen={results[name]['generated_tokens']} stop={results[name]['stop']}")
        except Exception as exc:
            results[name] = {"ok": False, "error": repr(exc)}
            print(f"[{name}] FAILED: {exc!r}")
    report["generations"] = results

    if args.repeat and results.get("thinking_false", {}).get("generated_tokens"):
        first = results["thinking_false"]
        try:
            again = run_one(model, tokenizer, messages, False, args.max_new_tokens, stop_ids)
            report["determinism"] = {
                "same_output_ids": first["output_token_ids_head"] == again["output_token_ids_head"]
                and first["output_token_ids_tail"] == again["output_token_ids_tail"],
                "same_generated_tokens": first["generated_tokens"] == again["generated_tokens"],
                "first": {"generated_tokens": first["generated_tokens"],
                          "tokens_per_second": first["tokens_per_second"]},
                "second": {"generated_tokens": again["generated_tokens"],
                           "tokens_per_second": again["tokens_per_second"]},
            }
        except Exception as exc:
            report["determinism"] = {"ok": False, "error": repr(exc)}

    if args.eos_ablation:
        # Behavioural proof of the dual-EOS contract: with eos_token_id forced to
        # [1] only, a <|im_end|> (130073) emission must NOT stop generation.
        try:
            base = results.get("thinking_false", {})
            single = run_one(
                model, tokenizer, messages, False, args.max_new_tokens,
                eos_ids=[1], eos_override=[1],
            )
            report["eos_ablation"] = {
                "note": "eos_token_id=[1] forced; 130073 is not a stop token in this run",
                "with_generation_config_eos": {
                    "generated_tokens": base.get("generated_tokens"),
                    "stop": base.get("stop"),
                },
                "with_single_eos_1": {
                    "generated_tokens": single["generated_tokens"],
                    "reached_max_new_tokens": single["reached_max_new_tokens"],
                    "output_contains": single["output_contains"],
                    "tail_ids": single["output_token_ids_tail"],
                    "stop": single["stop"],
                },
            }
            print("eos_ablation:", json.dumps(report["eos_ablation"], ensure_ascii=False)[:400])
        except Exception as exc:
            report["eos_ablation"] = {"ok": False, "error": repr(exc)}

    report["mem_final"] = mem_snapshot()
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    print(json.dumps({
        "out": args.out,
        "model_load_seconds": report["model_load_seconds"],
        "peak_reserved_gib": round(report["mem_final"]["cuda_max_reserved_bytes"] / 2**30, 2),
        "peak_allocated_gib": round(report["mem_final"]["cuda_max_allocated_bytes"] / 2**30, 2),
        "maxrss_gib": round(report["mem_final"]["process_maxrss_bytes"] / 2**30, 2),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
