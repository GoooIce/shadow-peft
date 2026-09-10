"""适应度题集评分器（16 票交付给 14 票使用）。

两个口径，同一份 prompt（显式 enable_thinking=False，助手侧固定 'Answer: ' 前缀）：
  * TF（默认，主轴）——teacher-forced：把 gold 答案拼在 prompt 后一次前向，逐位置比较
    模型 argmax 与 gold token；item 正确 = 全部 gold token 的 argmax 命中。同时记录
    gold 平均 logprob（连续分数，供 13 票估 σ / 排序参考）与 gold 之后的续写 token
    （格式硬门：应为 <|im_end|> = 130073）。
  * GEN（辅助）——贪心生成（停止条件 = generation_config 的双 eos [1, 130073]），再用
    同一个验证器（taskset_lib.verify）判定，用于与 TF 主轴对照 / 真实任务正确率。

11 票的 prefix cache 坑：本脚本默认 `enable_prefix_caching=False`——题集内每个 prompt 只出现
一次，关缓存不损失吞吐，却彻底消除「refit 后命中旧权重 KV」这一类错误。

用法（gooo）：
  python3 score_taskset.py --model ~/minicpm5-payload/out/MiniCPM5-2B-NVFP4-W4A4 \
      --items artifacts/pool.jsonl --out artifacts/q0_tf.jsonl --summary artifacts/q0_tf_summary.json \
      --arm tf [--gen-also] [--limit N] [--tag q0]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from taskset_lib import GOLD_PREFIX, answer_span, render_prompt, verify  # noqa: E402

IM_END = 130073


def build_requests(tokenizer, items, enable_thinking=False):
    reqs, n_unstable = [], 0
    for it in items:
        prompt = render_prompt(tokenizer, it, enable_thinking=enable_thinking)
        p_ids, g_ids, stable = answer_span(tokenizer, prompt, GOLD_PREFIX + it["answer"])
        n_unstable += (not stable)
        reqs.append(dict(item=it, prompt=prompt, p_ids=p_ids, g_ids=g_ids, stable=stable))
    return reqs, n_unstable


def parse_prompt_logprobs(plp):
    """vLLM 返回 list[dict[int, Logprob] | None]（index 0 为 None）。
    返回 (actual_map, top1_map)：pos -> (gold_logprob when present) / (top1 token id)。"""
    actual, top1 = {}, {}
    for pos, d in enumerate(plp):
        if not d:
            continue
        best_id, best_lp = None, None
        for tid, lp in d.items():
            if best_lp is None or lp.logprob > best_lp:
                best_id, best_lp = tid, lp.logprob
        top1[pos] = best_id
    return actual, top1


def score_tf(llm, SamplingParams, tokenizer, reqs, chunk=256, tag="", log=print):
    """teacher-forced 评分。返回 (per-item 结果, timing)。"""
    out = []
    t0 = time.time()
    n_tok = 0
    for s in range(0, len(reqs), chunk):
        batch = reqs[s:s + chunk]
        prompts = [r["p_ids"] + r["g_ids"] for r in batch]
        sp = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=1)
        res = llm.generate(prompts, sp, use_tqdm=False)
        for r, o in zip(batch, res):
            n = len(r["p_ids"])
            ng = len(r["g_ids"])
            plp = o.prompt_logprobs or []
            _, top1 = parse_prompt_logprobs(plp)
            gold_lps = []
            hits = []
            for k in range(ng):
                pos = n + k
                d = plp[pos] if pos < len(plp) else None
                gid = r["g_ids"][k]
                lp = d[gid].logprob if (d and gid in d) else None
                if lp is not None:
                    gold_lps.append(lp)
                hits.append(int(top1.get(pos) == gid))
            nxt = list(o.outputs[0].token_ids)
            out.append(dict(
                id=r["item"]["id"], family=r["item"]["family"], tier=r["item"]["tier"],
                stratum=r["item"]["stratum"],
                tf_correct=bool(hits) and all(hits),
                tf_token_hits=sum(hits), tf_tokens=ng,
                tf_gold_logprob=(sum(gold_lps) / len(gold_lps)) if gold_lps else None,
                tf_next_token=(nxt[0] if nxt else None),
                tf_next_is_im_end=(bool(nxt) and nxt[0] == IM_END),
                # 第一个 gold 位的 argmax（位点 == len(p_ids)）。"未作答" = 该位 argmax 为
                # 130073 (<|im_end|>)：模型在 Answer: 之后直接结束回合而不给答案。此字段
                # 只做记录与健康度统计，**不参与 tf_correct**（适应度函数逐字不变）。
                tf_first_argmax=(top1.get(n) if ng else None),
                n_prompt_tokens=n,
            ))
        n_tok += sum(len(r["p_ids"]) + len(r["g_ids"]) for r in batch)
        if (s // chunk) % 4 == 0:
            log(f"  [{tag}] tf {s+len(batch)}/{len(reqs)} t={time.time()-t0:.0f}s")
    dt = time.time() - t0
    return out, dict(tf_seconds=dt, tf_prompt_tokens=n_tok,
                     tf_tokens_per_s=(n_tok / dt if dt else None))


def score_gen(llm, SamplingParams, tokenizer, reqs, chunk=256, tag="", log=print):
    out = []
    t0 = time.time()
    n_out = 0
    for s in range(0, len(reqs), chunk):
        batch = reqs[s:s + chunk]
        prompts = [r["p_ids"] for r in batch]
        sp = SamplingParams(max_tokens=24, temperature=0.0,
                            stop_token_ids=[1, IM_END], ignore_eos=False)
        res = llm.generate(prompts, sp, use_tqdm=False)
        for r, o in zip(batch, res):
            txt = o.outputs[0].text
            ok, parsed, reason = verify(r["item"], txt)
            n_out += len(o.outputs[0].token_ids)
            out.append(dict(
                id=r["item"]["id"], family=r["item"]["family"], tier=r["item"]["tier"],
                stratum=r["item"]["stratum"],
                gen_correct=bool(ok), gen_text=txt.strip(), gen_parsed=parsed,
                gen_reason=reason, gen_finish=o.outputs[0].finish_reason,
                gen_n_tokens=len(o.outputs[0].token_ids),
            ))
        if (s // chunk) % 4 == 0:
            log(f"  [{tag}] gen {s+len(batch)}/{len(reqs)} t={time.time()-t0:.0f}s")
    dt = time.time() - t0
    return out, dict(gen_seconds=dt, gen_output_tokens=n_out,
                     gen_tokens_per_s=(n_out / dt if dt else None))


# ---------------------------------------------------------------- 未作答健康度（19 票）
# 背景（19 票）：v3 生成池上 BF16 在 logic/hard / ordering/hard 的负 gap 全部来自一种
# "提前结束回合"的行为——模型在 Answer: 之后第一个 token 就是 <|im_end|>（130073），
# 一个答案都没给；而 tf_correct 把"没作答"与"答错"记成同一件事。该行为可精确复现、
# 与题面长度无关、去掉 "Answer:" 后缀即消失（见 minicpm5-bf16-anomaly-check-20260910.md）。
# 这里只增加**记录与聚合**，不改变任何判据：tf_correct 的定义、阈值、主轴全部不变。

NO_ANSWER_TOKEN = IM_END          # 观测恒为 <|im_end|>（129/129、88/88、8/8 条探针）
NO_ANSWER_METRIC = ("no_answer = argmax at the first gold-token position == 130073 "
                    "(<|im_end|>); GEN-arm marker gen_n_tokens == 1 is recorded alongside")


def is_no_answer(row):
    """单题"未作答"判定（逐字按 19 票定义）。

    首选 `tf_first_argmax`（第一个 gold 位的 argmax）；该字段缺失时退回 GEN 臂的
    `gen_n_tokens == 1`（第一个生成 token 即 stop token，19 票实测与 <|im_end|> 一一对应）。
    两个标记都没有时返回 False（例如只有旧版 TF 行），并把这种情况在聚合里计数上报。
    """
    if row.get("tf_first_argmax") is not None:
        return int(row["tf_first_argmax"]) == NO_ANSWER_TOKEN
    return row.get("gen_n_tokens") == 1


def _na_block(rows):
    n = len(rows)
    k = sum(1 for r in rows if is_no_answer(r))
    return dict(n=n, n_no_answer=k, rate=(k / n if n else None))


def no_answer_summary(rows):
    """未作答率的总体 / 逐 family / 逐 stratum / 逐 tier 聚合（零额外 GPU 成本）。"""
    per_fam, per_str, per_tier = {}, {}, {}
    for r in rows:
        per_fam.setdefault(r.get("family"), []).append(r)
        per_str.setdefault(r.get("stratum"), []).append(r)
        per_tier.setdefault(r.get("tier"), []).append(r)
    return dict(
        overall=_na_block(rows),
        per_family={k: _na_block(v) for k, v in sorted(per_fam.items(), key=lambda kv: str(kv[0]))},
        per_stratum={k: _na_block(v) for k, v in sorted(per_str.items(), key=lambda kv: str(kv[0]))},
        per_tier={k: _na_block(v) for k, v in sorted(per_tier.items(), key=lambda kv: str(kv[0]))},
        rows_without_tf_first_argmax=sum(1 for r in rows if r.get("tf_first_argmax") is None),
        rows_without_any_marker=sum(1 for r in rows
                                    if r.get("tf_first_argmax") is None and r.get("gen_n_tokens") is None),
        metric=NO_ANSWER_METRIC,
    )


def joint_decomposition(cand_rows, ref_rows, key="tf_correct"):
    """候选 × 参考的 2x2 分解 + 未作答归因（用于"增益是否被伪影吃掉"的定量说明）。

    设 a=(候选对,参考对)、b=(候选对,参考错)、c=(候选错,参考对)、d=(双错)：
        gap = (b - c) / n
    再把 c（候选错而参考对的题）按原因拆开：
        c_no_answer  —— 候选在这些题上**根本没作答**（每个都白丢 1 分）
        c_wrong      —— 候选作答了但答错（真实能力差）
    """
    ref = {r["id"]: r for r in ref_rows}
    pairs = [(c, ref[c["id"]]) for c in cand_rows if c["id"] in ref]
    n = len(pairs)
    if not n:
        return dict(n_common=0)
    a = sum(1 for c, q in pairs if c.get(key) and q.get(key))
    b = sum(1 for c, q in pairs if c.get(key) and not q.get(key))
    c_ = sum(1 for c, q in pairs if not c.get(key) and q.get(key))
    c_na = sum(1 for c, q in pairs if not c.get(key) and q.get(key) and is_no_answer(c))
    q_na = sum(1 for c, q in pairs if not c.get(key) and q.get(key) and is_no_answer(q))
    return dict(
        n_common=n, n_cand_right_ref_right=a, n_cand_right_ref_wrong=b,
        n_cand_wrong_ref_right=c_, n_cand_wrong_ref_wrong=n - a - b - c_,
        gap=(b - c_) / n,
        cand_acc=(a + b) / n, ref_acc=(a + c_) / n,
        cand_loss_from_no_answer=c_na / n, cand_loss_from_wrong=(c_ - c_na) / n,
        ref_loss_from_no_answer=q_na / n, ref_bail_here=q_na,
        both_answered=(a + b + (c_ - c_na)) / n,
        metric=NO_ANSWER_METRIC,
    )


def attribution_verdict(cand_rows, ref_rows, margin=0.05, key="tf_correct"):
    """机械归因规则（19 票写进 run300 台账）：

    1. 声称的增益 `gain_all` = 全共同题上的准确率差；
    2. 归因增益 `gain_both` = **两臂都作答**子集上的准确率差；
    3. `no_answer_excess` = 候选未作答率 − 参考未作答率。

    判定：
      * `gain_all > 0` 而 `gain_both <= 0`  → `artifact_dominated`（增益不成立，是作答模式差异）；
      * `no_answer_excess > margin`         → `flagged_no_answer_excess`（无论增益正负都要单独报）；
      * `gain_all > 0` 且 `gain_both > 0`   → `gain_survives_both_answered`；
      * 其余                                 → `no_gain` / `no_common_items`。
    """
    ref = {r["id"]: r for r in ref_rows}
    pairs = [(c, ref[c["id"]]) for c in cand_rows if c["id"] in ref]
    n = len(pairs)
    if not n:
        return dict(n_common=0, verdict="no_common_items", flagged=False, margin=margin,
                    metric=NO_ANSWER_METRIC)
    both = [(c, q) for c, q in pairs if not is_no_answer(c) and not is_no_answer(q)]

    def acc(rs, idx):
        return (sum(1 for p in rs if p[idx].get(key)) / len(rs)) if rs else None

    acc_c_all, acc_r_all = acc(pairs, 0), acc(pairs, 1)
    acc_c_both, acc_r_both = acc(both, 0), acc(both, 1)
    gain_all = (acc_c_all - acc_r_all) if (acc_c_all is not None and acc_r_all is not None) else None
    gain_both = (acc_c_both - acc_r_both) if (acc_c_both is not None and acc_r_both is not None) else None
    ca = sum(1 for c, _ in pairs if is_no_answer(c)) / n
    ra = sum(1 for _, q in pairs if is_no_answer(q)) / n
    excess = ca - ra
    if gain_all is None:
        verdict = "no_common_items"
    elif gain_all > 0 and (gain_both is None or gain_both <= 0):
        verdict = "artifact_dominated"
    elif excess > margin:
        verdict = "flagged_no_answer_excess"
    elif gain_all > 0:
        verdict = "gain_survives_both_answered"
    else:
        verdict = "no_gain"
    return dict(
        metric=NO_ANSWER_METRIC, margin=margin,
        n_common=n, n_both_answered=len(both),
        cand_no_answer_rate=ca, ref_no_answer_rate=ra, no_answer_excess=excess,
        acc_cand_all=acc_c_all, acc_ref_all=acc_r_all, gain_all=gain_all,
        acc_cand_both=acc_c_both, acc_ref_both=acc_r_both, gain_both=gain_both,
        verdict=verdict,
        flagged=bool(excess > margin or verdict == "artifact_dominated"),
    )


def summarize(rows, key):
    n = len(rows)
    acc = sum(1 for r in rows if r[key]) / n if n else None
    per = {}
    for r in rows:
        d = per.setdefault(r["stratum"], [0, 0])
        d[0] += 1
        d[1] += int(bool(r[key]))
    per = {k: dict(n=v[0], acc=v[1] / v[0]) for k, v in sorted(per.items())}
    fam = {}
    for r in rows:
        d = fam.setdefault(r["family"], [0, 0])
        d[0] += 1
        d[1] += int(bool(r[key]))
    fam = {k: dict(n=v[0], acc=v[1] / v[0]) for k, v in sorted(fam.items())}
    return dict(n=n, acc=acc, per_stratum=per, per_family=fam)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--items", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--arm", default="tf", choices=["tf", "gen", "both"])
    ap.add_argument("--gen-also", action="store_true", help="跑 TF 后再跑 GEN 臂")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--tag", default="")
    ap.add_argument("--linear-backend", default="cutlass")
    ap.add_argument("--quantization", default="compressed-tensors")
    ap.add_argument("--max-model-len", type=int, default=1024)
    ap.add_argument("--gpu-mem", type=float, default=0.30)
    ap.add_argument("--enable-prefix-caching", action="store_true",
                    help="默认关闭（11 票：refit 后旧 KV 会污染分数）")
    ap.add_argument("--enable-thinking", action="store_true", help="默认 False（06 口径）")
    ap.add_argument("--no-selftest", action="store_true")
    a = ap.parse_args()

    def log(*x):
        print(f"[{time.strftime('%H:%M:%S')}]", *x, flush=True)

    items = [json.loads(l) for l in open(a.items)]
    if a.limit:
        items = items[:a.limit]
    log(f"items={len(items)} model={a.model}")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(a.model)
    reqs, n_unstable = build_requests(tokenizer, items, enable_thinking=a.enable_thinking)
    qlens = [len(r["p_ids"]) for r in reqs]
    log(f"prompt tokens: min={min(qlens)} median={sorted(qlens)[len(qlens)//2]} max={max(qlens)}; "
        f"gold tokens: {sum(len(r['g_ids']) for r in reqs)}; unstable_prefix: {n_unstable}")

    from vllm import LLM, SamplingParams
    kw = dict(model=a.model, dtype="bfloat16", max_model_len=a.max_model_len,
              gpu_memory_utilization=a.gpu_mem, enable_prefix_caching=a.enable_prefix_caching,
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
            log(f"LLM() 不接受 linear_backend（{e}），退回默认内核选择")
            kw.pop("linear_backend")
            llm = LLM(**kw)
        else:
            raise
    load_s = time.time() - t0
    log(f"engine loaded in {load_s:.1f}s")

    summary = dict(model=a.model, arm=a.arm, tag=a.tag, n_items=len(items),
                   enable_thinking=a.enable_thinking,
                   enable_prefix_caching=a.enable_prefix_caching,
                   linear_backend=kw.get("linear_backend", "default"),
                   engine_load_s=load_s, prompt_tokens=dict(min=min(qlens),
                                                            median=sorted(qlens)[len(qlens)//2],
                                                            max=max(qlens), total=sum(qlens)),
                   generated_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
                   n_unstable_prefix=n_unstable)

    rows_by_arm = {}
    if not a.no_selftest:
        # 自检：prompt_logprobs 语义（TF 的最后一位 top1 应等于贪心首 token）+ 模板口径
        sub = reqs[:min(8, len(reqs))]
        spt = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=1)
        rtf = llm.generate([r["p_ids"] + r["g_ids"] for r in sub], spt, use_tqdm=False)
        spg = SamplingParams(max_tokens=1, temperature=0.0, stop_token_ids=[1, IM_END])
        rgn = llm.generate([r["p_ids"] for r in sub], spg, use_tqdm=False)
        agree, template_ok, gold_lp_ok = 0, 0, 0
        for r, otf, ogn in zip(sub, rtf, rgn):
            plp = otf.prompt_logprobs or []
            _, top1 = parse_prompt_logprobs(plp)
            # 语义校验：pos=len(p_ids)（第一个 gold 位）的 argmax 应等于贪心生成的首 token
            first_gold_pos = len(r["p_ids"])
            if top1.get(first_gold_pos) == list(ogn.outputs[0].token_ids)[0]:
                agree += 1
            d0 = plp[first_gold_pos] if first_gold_pos < len(plp) else None
            if d0 and r["g_ids"] and r["g_ids"][0] in d0:
                gold_lp_ok += 1
            if "</think>" in r["prompt"]:
                template_ok += 1
        summary["selftest"] = dict(n=len(sub), tf_firstgold_top1_eq_greedy=agree,
                                   gold_token_logprob_present=gold_lp_ok,
                                   prompt_has_closed_think=template_ok,
                                   sample_prompt_ids=sub[0]["p_ids"][:40],
                                   sample_prompt_text=sub[0]["prompt"],
                                   sample_gold_ids=sub[0]["g_ids"])
        log(f"selftest: {summary['selftest']}")

    if a.arm in ("tf", "both"):
        rows, timing = score_tf(llm, SamplingParams, tokenizer, reqs, a.chunk, a.tag, log)
        summary["tf_timing"] = timing
        summary["tf"] = summarize(rows, "tf_correct")
        summary["tf_format_im_end_rate"] = (sum(1 for r in rows if r["tf_next_is_im_end"]) / len(rows))
        summary["tf_gold_logprob_mean"] = (
            sum(r["tf_gold_logprob"] for r in rows if r["tf_gold_logprob"] is not None)
            / max(1, sum(1 for r in rows if r["tf_gold_logprob"] is not None)))
        rows_by_arm["tf"] = rows
        # 健康度（不计入判据）：未作答率的总体 / 逐 family / 逐 stratum
        summary["no_answer"] = no_answer_summary(rows)
        _na = summary["no_answer"]
        log(f"TF acc = {summary['tf']['acc']:.4f} in {timing['tf_seconds']:.1f}s "
            f"({timing['tf_tokens_per_s']:.0f} tok/s)")
        log("no-answer rate overall={:.4f} ({}/{}) per-family={}".format(
            _na["overall"]["rate"] or 0.0, _na["overall"]["n_no_answer"], _na["overall"]["n"],
            {k: round(v["rate"], 4) for k, v in _na["per_family"].items()}))

    if a.arm in ("gen", "both") or a.gen_also:
        rows, timing = score_gen(llm, SamplingParams, tokenizer, reqs, a.chunk, a.tag, log)
        summary["gen_timing"] = timing
        summary["gen"] = summarize(rows, "gen_correct")
        stop = {}
        for r in rows:
            stop[r.get("gen_finish", "?")] = stop.get(r.get("gen_finish", "?"), 0) + 1
        summary["gen"]["stop_reason"] = stop
        summary["gen"]["length_cap_hit_rate"] = stop.get("length", 0) / max(1, len(rows))
        summary["gen"]["max_tokens"] = 24
        rows_by_arm["gen"] = rows
        log(f"GEN acc = {summary['gen']['acc']:.4f} in {timing['gen_seconds']:.1f}s")

    # 合并输出（按 id）
    ids = [r["item"]["id"] for r in reqs]
    merged = {i: {} for i in ids}
    for arm, rows in rows_by_arm.items():
        for r in rows:
            merged[r["id"]].update(r)
    with open(a.out, "w") as f:
        for i in ids:
            f.write(json.dumps(merged[i], ensure_ascii=False, sort_keys=True) + "\n")
    with open(a.summary, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, sort_keys=True)
    log(json.dumps({k: summary[k] for k in summary if k in
                    ("tf", "gen", "tf_timing", "gen_timing", "engine_load_s")},
                   ensure_ascii=False)[:1500])


if __name__ == "__main__":
    main()
