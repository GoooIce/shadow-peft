"""Bounded GPU probe for the BF16 early-termination anomaly (gooo, GB10).

Identical scoring protocol to scripts/minicpm5/score_taskset.py:
  apply_chat_template(..., add_generation_prompt=True, enable_thinking=False) + "Answer:",
  gold text = " " + answer; TF = SamplingParams(prompt_logprobs=1, max_tokens=1, temperature=0);
  GEN = greedy max_tokens=24, stop_token_ids=[1, 130073]; prefix cache off.
The only thing this script adds is (a) the raw output token ids and (b) prompt variants that
are built into the probe item file (filler / no-"Answer:"-prefix).

Usage (gooo):
  VLLM_WORKER_MULTIPROC_METHOD=spawn ~/venvs/minicpm5-vllm/bin/python anomaly_probe.py \
      --model ~/models/MiniCPM5-2B --quantization= --items probe_items.jsonl --out probe_bf16.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from taskset_lib import ANSWER_PREFIX, GOLD_PREFIX, render_prompt  # noqa: E402
from score_taskset import parse_prompt_logprobs  # noqa: E402

IM_END = 130073


def build(tokenizer, items, max_prompt_tokens=0):
    reqs, n_unstable = [], 0
    for it in items:
        prompt = render_prompt(tokenizer, it, enable_thinking=False)
        if (it.get("_opt") or {}).get("strip_answer_prefix"):
            assert prompt.endswith(ANSWER_PREFIX)
            prompt = prompt[: -len(ANSWER_PREFIX)]
        p_ids = tokenizer.encode(prompt, add_special_tokens=False)
        full = tokenizer.encode(prompt + GOLD_PREFIX + it["answer"], add_special_tokens=False)
        k = 0
        while k < min(len(p_ids), len(full)) and p_ids[k] == full[k]:
            k += 1
        stable = (k == len(p_ids))
        n_unstable += (not stable)
        reqs.append(dict(item=it, prompt=prompt, p_ids=p_ids, g_ids=full[k:], stable=stable))
    return reqs, n_unstable


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--items", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--quantization", default="compressed-tensors")
    ap.add_argument("--linear-backend", default="cutlass")
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--gpu-mem", type=float, default=0.30)
    ap.add_argument("--max-model-len", type=int, default=1024)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    def log(*x):
        print(f"[{time.strftime('%H:%M:%S')}]", *x, flush=True)

    items = [json.loads(l) for l in open(a.items)]
    log(f"items={len(items)} model={a.model} quant={a.quantization!r}")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(a.model)
    reqs, n_unstable = build(tokenizer, items)
    log(f"unstable_prefix={n_unstable} max_prompt_tokens={max(len(r['p_ids']) for r in reqs)}")

    from vllm import LLM, SamplingParams
    kw = dict(model=a.model, dtype="bfloat16", max_model_len=a.max_model_len,
              gpu_memory_utilization=a.gpu_mem, enable_prefix_caching=False,
              trust_remote_code=True, disable_log_stats=True)
    if a.quantization:
        kw["quantization"] = a.quantization
    if a.linear_backend:
        kw["linear_backend"] = a.linear_backend
    t0 = time.time()
    try:
        llm = LLM(**kw)
    except TypeError as e:
        if "linear_backend" in str(e):
            kw.pop("linear_backend")
            llm = LLM(**kw)
        else:
            raise
    log(f"engine loaded in {time.time()-t0:.1f}s")

    rows = {}
    for s in range(0, len(reqs), a.chunk):
        batch = reqs[s:s + a.chunk]
        sp = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=1)
        res = llm.generate([r["p_ids"] + r["g_ids"] for r in batch], sp, use_tqdm=False)
        for r, o in zip(batch, res):
            n, ng = len(r["p_ids"]), len(r["g_ids"])
            plp = o.prompt_logprobs or []
            _, top1 = parse_prompt_logprobs(plp)
            gold_lps, hits = [], []
            for k in range(ng):
                pos = n + k
                d = plp[pos] if pos < len(plp) else None
                gid = r["g_ids"][k]
                lp = d[gid].logprob if (d and gid in d) else None
                if lp is not None:
                    gold_lps.append(lp)
                hits.append(int(top1.get(pos) == gid))
            nxt = list(o.outputs[0].token_ids)
            rows[r["item"]["id"] + "|" + r["item"]["_group"]] = dict(
                id=r["item"]["id"], group=r["item"]["_group"], stratum=r["item"]["stratum"],
                gold=r["item"]["answer"], n_prompt_tokens=n, tf_tokens=ng, stable=r["stable"],
                tf_correct=bool(hits) and all(hits), tf_token_hits=sum(hits),
                tf_first_is_gold=bool(hits) and hits[0] == 1,
                tf_first_argmax=top1.get(n),
                tf_gold_logprob=(sum(gold_lps) / len(gold_lps)) if gold_lps else None,
                tf_next_token=(nxt[0] if nxt else None),
                tf_next_is_im_end=(bool(nxt) and nxt[0] == IM_END))
    t1 = time.time()
    log(f"tf done {t1-t0:.1f}s")

    gen_rows = {}
    for s in range(0, len(reqs), a.chunk):
        batch = reqs[s:s + a.chunk]
        sp = SamplingParams(max_tokens=24, temperature=0.0, stop_token_ids=[1, IM_END])
        res = llm.generate([r["p_ids"] for r in batch], sp, use_tqdm=False)
        for r, o in zip(batch, res):
            toks = list(o.outputs[0].token_ids)
            gen_rows[r["item"]["id"] + "|" + r["item"]["_group"]] = dict(
                gen_text=o.outputs[0].text.strip(), gen_token_ids=toks[:8],
                gen_n_tokens=len(toks), gen_finish=o.outputs[0].finish_reason,
                gen_first_token=(toks[0] if toks else None))
    t2 = time.time()
    log(f"gen done {t2-t1:.1f}s")

    with open(a.out, "w") as f:
        for iid, row in rows.items():
            row.update(gen_rows.get(iid, {}))
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    log(f"wrote {a.out}; tf={t1-t0:.1f}s gen={t2-t1:.1f}s")


if __name__ == "__main__":
    main()
