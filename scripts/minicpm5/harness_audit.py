#!/usr/bin/env python3
"""Standing harness audit: degenerate-output scan across every family (§5).

Two of the three RSI papers put a *harness defect* at the centre of their
failures: HarnessDev attributes 77.8% of its observed failures to harness defects
(including a fixed executor that collapsed a system 69.3 -> 33.0), and Aspire
watches for numeric-label collapse -- a model that stops emitting numbers while
still producing well-formed short answers.  We found our own instance of the
class **by accident** (the frozen ``Answer:`` suffix inducing an "end the turn"
mode that silently zeroed hard logic/ordering items, ticket 19).  This script
turns that accident into a routine check with a reported rate per family.

It measures, on a set of items and under the *active* prompt form:

* ``no_answer``  -- the argmax at the first gold position is ``<|im_end|>``
  (130073), or the greedy generation is a single stop token;
* ``truncated``  -- the greedy generation hit the token cap;
* ``empty_output`` -- the greedy text is empty;
* ``non_numeric`` -- the verifier needs an integer and the output has none
  (Aspire's numeric-label collapse, measured rather than assumed);
* ``non_name``   -- ordering items with no name in the output;
* ``first_token_not_numeric`` -- for numeric items, the first predicted token is
  not one that decodes to something starting with a digit or ``-``;
* ``multi_line`` / ``verbose_first_line`` / ``think_leak`` -- format drift
  against the frozen instruction ("Reply with only the final answer, with no
  explanation.");
* a mechanical **failure attribution**: of the items the teacher-forced axis
  marks wrong, how many failed through a degenerate mode rather than through a
  wrong-but-well-formed answer.

Pure-CPU core (:func:`classify_rows` + :func:`summarise`) so the classification
is unit-testable without an engine; the GPU part is a thin driver that only
collects rows.

Usage (machine with the payload; the audit is one TF pass + one greedy pass):
    python harness_audit.py --model ~/minicpm5-payload/out/MiniCPM5-2B-NVFP4-W4A4 \
        --items artifacts/taskset_dev.jsonl --out harness_audit_dev.json \
        --variant frozen --arm both
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

IM_END = 130073

#: verifier kinds that require a number in the output
NUMERIC_KINDS = (
    "arith_int", "maxexpr", "logic_count", "modulo_rem", "modulo_last_digit",
    "modulo_gcd", "pct_of", "frac_of", "pct_original", "pct_net", "countdown_expr",
)
NAME_KINDS = ("ordering",)

MODE_NOTES = {
    "no_answer": "argmax at the first gold position == 130073 (<|im_end|>) / greedy output "
                 "is a single stop token: the model ends the turn without answering",
    "truncated": "greedy generation hit the token cap",
    "empty_output": "greedy text is empty",
    "non_numeric": "verifier requires an integer, the output contains none",
    "non_name": "ordering item with no name in the output",
    "multi_line": "output has more than one non-empty line",
    "verbose_first_line": "output's first non-empty line has more than 12 whitespace tokens",
    "think_leak": "output mentions thinking although enable_thinking=False",
}


def required_answer_kind(item: dict) -> str | None:
    """``"integer"`` / ``"name"`` / None, from the item's own verifier spec."""
    kind = (item.get("verify") or {}).get("kind")
    if kind in NUMERIC_KINDS:
        return "integer"
    if kind in NAME_KINDS:
        return "name"
    return None


def is_no_answer(row: dict, stop_token: int = IM_END) -> bool:
    """Ticket 19's definition, kept identical (TF field first, GEN marker as fallback)."""
    if row.get("tf_first_argmax") is not None:
        return int(row["tf_first_argmax"]) == stop_token
    return row.get("gen_n_tokens") == 1


def _flatten(text: str) -> list[str]:
    return [line.strip() for line in (text or "").splitlines() if line.strip()]


def classify_row(row: dict, numeric_token=None) -> list[str]:
    """Every degenerate mode a single item's row exhibits (pure function)."""
    modes: list[str] = []
    if is_no_answer(row):
        modes.append("no_answer")
    text = row.get("gen_text")
    if text is not None:
        if text == "":
            modes.append("empty_output")
        if row.get("gen_finish") == "length":
            modes.append("truncated")
        if len(_flatten(text)) > 1:
            modes.append("multi_line")
        first_line = _flatten(text)[0] if _flatten(text) else ""
        if len(first_line.split()) > 12:
            modes.append("verbose_first_line")
        if "think" in text.lower():
            modes.append("think_leak")
        required = row.get("required")
        if required == "integer" and row.get("gen_parsed") is None:
            # a parse of None on an item that must carry an integer = the numeric mode is gone
            modes.append("non_numeric")
        if required == "name" and row.get("gen_parsed") is None:
            modes.append("non_name")
    if (row.get("required") == "integer" and numeric_token is not None
            and row.get("tf_first_argmax") is not None
            and not numeric_token(int(row["tf_first_argmax"]))):
        modes.append("first_token_not_numeric")
    return modes


def classify_rows(rows: list[dict], numeric_token=None) -> list[dict]:
    """Attach ``modes`` + a per-item failure attribution to every row (in place copy)."""
    out = []
    for row in rows:
        entry = dict(row)
        modes = classify_row(row, numeric_token)
        entry["modes"] = modes
        if not row.get("tf_correct"):
            if "no_answer" in modes:
                entry["failure_mode"] = "no_answer"
            elif "empty_output" in modes:
                entry["failure_mode"] = "empty_output"
            elif "truncated" in modes:
                entry["failure_mode"] = "truncated"
            elif "non_numeric" in modes or "non_name" in modes:
                entry["failure_mode"] = "unparsable_output"
            else:
                entry["failure_mode"] = "wrong_answer"
        else:
            entry["failure_mode"] = None
        out.append(entry)
    return out


def _rate(k: int, n: int) -> float | None:
    return (k / n) if n else None


def _bucket(rows: list[dict]) -> dict:
    n = len(rows)
    modes = {mode: sum(1 for row in rows if mode in row["modes"]) for mode in MODE_NOTES}
    failures = [row for row in rows if not row.get("tf_correct")]
    attribution: dict[str, int] = {}
    for row in failures:
        key = row.get("failure_mode") or "wrong_answer"
        attribution[key] = attribution.get(key, 0) + 1
    degenerate = sum(1 for row in rows if row["modes"])
    return {
        "n": n,
        "tf_acc": _rate(sum(1 for row in rows if row.get("tf_correct")), n),
        "gen_acc": _rate(sum(1 for row in rows if row.get("gen_correct")), n)
        if any("gen_correct" in row for row in rows) else None,
        "modes": {mode: {"n": count, "rate": _rate(count, n)} for mode, count in modes.items()},
        "degenerate_rate": _rate(degenerate, n),
        "failures": len(failures),
        "failure_attribution": {key: {"n": count, "share_of_failures": _rate(count, len(failures))}
                                for key, count in sorted(attribution.items())},
        "harness_defect_share": _rate(attribution.get("no_answer", 0)
                                      + attribution.get("empty_output", 0)
                                      + attribution.get("truncated", 0)
                                      + attribution.get("unparsable_output", 0), len(failures)),
    }


def summarise(rows: list[dict], margin: float = 0.05) -> dict:
    families: dict[str, list[dict]] = {}
    strata: dict[str, list[dict]] = {}
    for row in rows:
        families.setdefault(str(row.get("family")), []).append(row)
        strata.setdefault(str(row.get("stratum")), []).append(row)
    overall = _bucket(rows)
    per_family = {family: _bucket(members) for family, members in sorted(families.items())}
    per_stratum = {key: _bucket(members) for key, members in sorted(strata.items())}
    flags = []
    for family, block in per_family.items():
        for mode in ("no_answer", "non_numeric", "non_name", "truncated"):
            rate = block["modes"][mode]["rate"]
            if rate is not None and rate > margin:
                flags.append({"scope": "family", "family": family, "mode": mode, "rate": rate,
                              "margin": margin})
    first_tokens: dict[str, int] = {}
    for row in rows:
        token = row.get("tf_first_argmax")
        if token is not None:
            first_tokens[str(token)] = first_tokens.get(str(token), 0) + 1
    top = sorted(first_tokens.items(), key=lambda kv: (-kv[1], kv[0]))[:15]
    return {
        "n": len(rows),
        "margin": margin,
        "modes_metric": MODE_NOTES,
        "overall": overall,
        "per_family": per_family,
        "per_stratum": per_stratum,
        "flags": flags,
        "flagged": bool(flags),
        "first_token_top": [{"token": token, "n": count, "rate": _rate(count, len(rows))}
                            for token, count in top],
        "distinct_first_tokens": len(first_tokens),
    }


# --------------------------------------------------------------------------- #
# GPU side: collect the rows the classifier needs
# --------------------------------------------------------------------------- #


def score_tf_probe(llm, SamplingParams, tokenizer, reqs, chunk=256, tag="", log=print):
    """Teacher-forced pass that also records the first-gold-position argmax detail.

    Deliberately a local implementation rather than a call into
    ``score_taskset.score_tf``: the audit has to keep working whether or not the
    ticket-19 instrumentation (``tf_first_argmax``) is present in the scorer, and
    it must not change the scorer's frozen semantics.  The TF decision rule is
    identical (all gold tokens must have the gold id as argmax).
    """
    out = []
    t0 = time.time()
    for start in range(0, len(reqs), chunk):
        batch = reqs[start:start + chunk]
        prompts = [r["p_ids"] + r["g_ids"] for r in batch]
        sp = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=1)
        results = llm.generate(prompts, sp, use_tqdm=False)
        for req, output in zip(batch, results):
            n = len(req["p_ids"])
            ng = len(req["g_ids"])
            plp = output.prompt_logprobs or []
            top1: dict[int, int] = {}
            for pos, entry in enumerate(plp):
                if not entry:
                    continue
                best_id, best_lp = None, None
                for token_id, logprob in entry.items():
                    if best_lp is None or logprob.logprob > best_lp:
                        best_id, best_lp = token_id, logprob.logprob
                top1[pos] = best_id
            gold_lps, hits = [], []
            for k in range(ng):
                pos = n + k
                entry = plp[pos] if pos < len(plp) else None
                gold_id = req["g_ids"][k]
                lp = entry[gold_id].logprob if (entry and gold_id in entry) else None
                if lp is not None:
                    gold_lps.append(lp)
                hits.append(int(top1.get(pos) == gold_id))
            first_pos = n
            entry = plp[first_pos] if first_pos < len(plp) else None
            alternatives = sorted(
                ((token_id, logprob.logprob) for token_id, logprob in (entry or {}).items()),
                key=lambda item: -item[1])[:5]
            nxt = list(output.outputs[0].token_ids)
            item = req["item"]
            out.append(dict(
                id=item["id"], family=item.get("family"), tier=item.get("tier"),
                stratum=item.get("stratum"),
                required=required_answer_kind(item),
                tf_correct=bool(hits) and all(hits),
                tf_token_hits=sum(hits), tf_tokens=ng,
                tf_gold_logprob=(sum(gold_lps) / len(gold_lps)) if gold_lps else None,
                tf_next_token=(nxt[0] if nxt else None),
                tf_next_is_im_end=(bool(nxt) and nxt[0] == IM_END),
                tf_first_argmax=(top1.get(first_pos) if ng else None),
                tf_first_top5=[[int(token), float(lp)] for token, lp in alternatives],
                n_prompt_tokens=n,
                prompt_variant=req.get("variant"),
                prompt_stable=req.get("stable"),
            ))
        if (start // chunk) % 4 == 0:
            log(f"  [{tag}] tf-probe {start + len(batch)}/{len(reqs)} t={time.time() - t0:.0f}s")
    dt = time.time() - t0
    return out, {"tf_seconds": dt}


def numeric_token_predicate(tokenizer):
    """Token id -> is it something that decodes to a digit-initial string?"""
    cache: dict[int, bool] = {}

    def predicate(token_id: int) -> bool:
        if token_id not in cache:
            try:
                text = tokenizer.decode([int(token_id)]).strip()
            except Exception:  # pragma: no cover - tokenizer-side guard
                text = ""
            cache[token_id] = bool(text) and (text[0].isdigit() or text[0] == "-")
        return cache[token_id]

    return predicate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--items", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--variant", default="frozen")
    parser.add_argument("--arm", default="both", choices=["tf", "gen", "both"])
    parser.add_argument("--gen-max-tokens", type=int, default=24)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--chunk", type=int, default=256)
    parser.add_argument("--tag", default="harness-audit")
    parser.add_argument("--margin", type=float, default=0.05,
                        help="a per-family degenerate-mode rate above this is flagged")
    parser.add_argument("--quantization", default="compressed-tensors")
    parser.add_argument("--linear-backend", default="cutlass")
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--gpu-mem", type=float, default=0.30)
    args = parser.parse_args()

    def log(*parts):
        print(f"[{time.strftime('%H:%M:%S')}]", *parts, flush=True)

    items = [json.loads(line) for line in Path(args.items).expanduser().open() if line.strip()]
    if args.limit:
        items = items[:args.limit]
    from taskset_lib import render_span
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    reqs = []
    unstable = 0
    for item in items:
        prompt, p_ids, g_ids, stable = render_span(tokenizer, item, variant=args.variant)
        unstable += int(not stable)
        reqs.append({"item": item, "prompt": prompt, "p_ids": p_ids, "g_ids": g_ids,
                     "stable": stable, "variant": args.variant})
    log(f"items={len(reqs)} variant={args.variant} unstable_prefix={unstable}")

    from vllm import LLM, SamplingParams

    kwargs = dict(model=args.model, quantization=args.quantization, dtype="bfloat16",
                  max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_mem,
                  enable_prefix_caching=False, trust_remote_code=True, disable_log_stats=True)
    if args.linear_backend:
        kwargs["linear_backend"] = args.linear_backend
    llm = LLM(**kwargs)
    log("engine loaded")

    rows_by_arm = {}
    if args.arm in ("tf", "both"):
        rows, timing = score_tf_probe(llm, SamplingParams, tokenizer, reqs, args.chunk, args.tag, log)
        rows_by_arm["tf"] = rows
        log(f"TF probe done in {timing['tf_seconds']:.1f}s")
    if args.arm in ("gen", "both"):
        from score_taskset import score_gen

        gen_reqs = [{"item": r["item"], "p_ids": r["p_ids"]} for r in reqs]
        rows, timing = score_gen(llm, SamplingParams, tokenizer, gen_reqs, args.chunk,
                                 args.tag, log, max_tokens=args.gen_max_tokens)
        rows_by_arm["gen"] = rows
        log(f"GEN probe done in {timing['gen_seconds']:.1f}s")

    merged: dict[str, dict] = {}
    for arm, rows in rows_by_arm.items():
        for row in rows:
            merged.setdefault(row["id"], {}).update(row)
    table = [merged[item["id"]] for item in items if item["id"] in merged]
    classified = classify_rows(table, numeric_token_predicate(tokenizer))
    report = summarise(classified, margin=args.margin)
    report["variant"] = args.variant
    report["arm"] = args.arm
    report["items_file"] = str(args.items)
    report["model"] = args.model
    report["unstable_prefix"] = unstable
    report["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    report["examples"] = {
        mode: [{"id": row["id"], "family": row.get("family"), "stratum": row.get("stratum"),
                "gen_text": (row.get("gen_text") or "")[:120]}
               for row in classified if mode in row["modes"]][:5]
        for mode in MODE_NOTES
    }
    Path(args.out).expanduser().write_text(
        json.dumps(report, ensure_ascii=False, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"variant": args.variant, "n": report["n"],
                      "overall": report["overall"]["modes"],
                      "degenerate_rate": report["overall"]["degenerate_rate"],
                      "harness_defect_share": report["overall"]["harness_defect_share"],
                      "flags": report["flags"]}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
