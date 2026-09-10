"""BF16 anomaly check: per-item join, failure classification, compliance signature, sub-modes.

Input: full.jsonl produced by anomaly_join.py (85,815 items, both arms, TF+GEN).

Classification of a BF16-failure / q0-success item (hypothesis 2 vs 1):
  (a) wrong final answer, format kept
  (b) correct final answer, format/continuation deviation
  (c) other (unparseable, empty, reasoning leak, refusal)
The GEN arm (free greedy generation scored by the same verifier) is the arbiter of
"does the model know the answer": TF-fail + GEN-gold == the forced continuation from the
gold prefix deviates although the model's own generation parses to gold.
"""
from __future__ import annotations

import argparse
import collections
import json
import re


def load(p):
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def tokens_of(r, arm):
    return (r["tf_next_token"] if arm == "q0" else r["bf16_tf_next_token"]), \
           (r["tf_next_is_im_end"] if arm == "q0" else r["bf16_tf_next_is_im_end"]), \
           (r["tf_token_hits"] if arm == "q0" else r["bf16_tf_token_hits"]), \
           (r["tf_tokens"] if arm == "q0" else r["bf16_tf_tokens"])


def gen_of(r, arm):
    if arm == "q0":
        return r["gen_correct"], r["gen_text"], r["gen_parsed"], r["gen_reason"], r["gen_finish"]
    return (r["bf16_gen_correct"], r["bf16_gen_text"], r["bf16_gen_parsed"],
            r["bf16_gen_reason"], r["bf16_gen_finish"])


# ---- sub-mode derivation (meta has no explicit 'sub' for logic/ordering in v3) -------------
def sub_mode(r):
    m = r.get("meta") or {}
    v = r.get("verify") or {}
    fam = r["family"]
    if m.get("sub") or m.get("kind"):
        return str(m.get("sub") or m.get("kind"))
    if fam == "logic":
        return f"nform={m.get('n_formulas')}"
    if fam == "ordering":
        return f"nrank={m.get('rank')}"
    if fam == "arith":
        return "arith_expr"
    if fam == "modulo":
        return v.get("kind", "?")
    if fam == "pct":
        return v.get("kind", "?")
    return "-"


def head_style(text):
    """Classify the shape of a GEN output: 'terse' (answer then stop/punct) vs prose vs empty."""
    if text is None:
        return "none"
    t = text.strip()
    if not t:
        return "empty"
    if len(t) <= 6 and re.fullmatch(r"[-+]?\d[\d,]*[.,!?]?", t):
        return "terse_num"
    if re.fullmatch(r"[A-Za-z]+[.,!?]?", t):
        return "terse_name"
    if re.search(r"\d", t):
        return "prose_with_num"
    return "prose_no_num"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", default="/tmp/minicpm5-sanity-data/full.jsonl")
    ap.add_argument("--strata", default="logic/hard,ordering/hard,pct/easy")
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples", required=True, type=int, default=25)
    a = ap.parse_args()

    strata = a.strata.split(",")
    rows = [r for r in load(a.full) if r["stratum"] in strata]
    rep = dict(strata={}, samples={})

    for s in strata:
        sub = [r for r in rows if r["stratum"] == s]
        n = len(sub)
        c = collections.Counter()
        for r in sub:
            c[(bool(r["tf_correct"]), bool(r["bf16_tf_correct"]))] += 1
        q0_acc = sum(1 for r in sub if r["tf_correct"]) / n
        b_acc = sum(1 for r in sub if r["bf16_tf_correct"]) / n
        rep["strata"][s] = dict(
            n=n, q0_acc=q0_acc, bf16_acc=b_acc, gap=b_acc - q0_acc,
            q0_ok_bf16_ok=c[(True, True)], q0_ok_bf16_bad=c[(True, False)],
            q0_bad_bf16_ok=c[(False, True)], q0_bad_bf16_bad=c[(False, False)],
            q0_gen_acc=sum(1 for r in sub if r["gen_correct"]) / n,
            bf16_gen_acc=sum(1 for r in sub if r["bf16_gen_correct"]) / n,
            q0_im_end=sum(1 for r in sub if r["tf_next_is_im_end"]) / n,
            bf16_im_end=sum(1 for r in sub if r["bf16_tf_next_is_im_end"]) / n,
            gold_tokens=collections.Counter(r["tf_tokens"] for r in sub),
        )

        # ---------- classification of BF16-fail / q0-success ----------
        cls = collections.Counter()
        detail = collections.Counter()
        for r in sub:
            if not (r["tf_correct"] and not r["bf16_tf_correct"]):
                continue
            b_gen_ok, b_txt, b_parsed, b_reason, b_fin = gen_of(r, "bf16")
            hits = r["bf16_tf_token_hits"]
            tot = r["bf16_tf_tokens"]
            if b_gen_ok:
                cls["b_correct_answer_fmt_dev"] += 1
                detail[f"b|hits={hits}/{tot}|{head_style(b_txt)}"] += 1
            elif b_parsed is not None:
                cls["a_wrong_answer"] += 1
                detail[f"a|hits={hits}/{tot}|{b_reason}"] += 1
            else:
                cls["c_other"] += 1
                detail[f"c|hits={hits}/{tot}|{b_reason}|{head_style(b_txt)}"] += 1
        rep["strata"][s]["bf16_fail_q0_ok_cls"] = dict(cls)
        rep["strata"][s]["bf16_fail_q0_ok_detail"] = dict(detail)

        # mirror image: q0-fail / bf16-success (is q0 the one deviating?)
        cls2 = collections.Counter()
        for r in sub:
            if not (r["bf16_tf_correct"] and not r["tf_correct"]):
                continue
            g_ok, g_txt, g_parsed, g_reason, g_fin = gen_of(r, "q0")
            hits, tot = r["tf_token_hits"], r["tf_tokens"]
            if g_ok:
                cls2["b_correct_answer_fmt_dev"] += 1
            elif g_parsed is not None:
                cls2["a_wrong_answer"] += 1
            else:
                cls2["c_other"] += 1
        rep["strata"][s]["q0_fail_bf16_ok_cls"] = dict(cls2)

        # ---------- compliance signature: first gold token hit, later missed ----------
        sig = {}
        for arm in ("q0", "bf16"):
            tot_all = 0
            first_hit_later_miss = 0
            zero_hit = 0
            fails = 0
            fails_first_hit = 0
            for r in sub:
                _, _, hits, ntok = tokens_of(r, arm)
                ok = r["tf_correct"] if arm == "q0" else r["bf16_tf_correct"]
                if ntok and ntok > 1:
                    tot_all += 1
                    if hits and hits < ntok:
                        first_hit_later_miss += 1
                    if hits == 0:
                        zero_hit += 1
                if not ok:
                    fails += 1
                    if hits and hits < ntok:
                        fails_first_hit += 1
            sig[arm] = dict(multi_token_items=tot_all, first_hit_later_miss=first_hit_later_miss,
                            rate_all=(first_hit_later_miss / tot_all if tot_all else None),
                            zero_hit=zero_hit, fails=fails,
                            fails_first_hit_later_miss=fails_first_hit,
                            rate_of_fails=(fails_first_hit / fails if fails else None))
        rep["strata"][s]["compliance_signature"] = sig

        # among the BF16-fail/q0-ok set only
        set_rows = [r for r in sub if r["tf_correct"] and not r["bf16_tf_correct"]]
        sig2 = dict(n=len(set_rows))
        for arm in ("q0", "bf16"):
            hits_gt0 = sum(1 for r in set_rows
                           if (r["tf_token_hits"] if arm == "q0" else r["bf16_tf_token_hits"]) > 0)
            hits_partial = sum(1 for r in set_rows
                               if 0 < (r["tf_token_hits"] if arm == "q0" else r["bf16_tf_token_hits"])
                               < (r["tf_tokens"] if arm == "q0" else r["bf16_tf_tokens"]))
            imend = sum(1 for r in set_rows
                        if (r["tf_next_is_im_end"] if arm == "q0" else r["bf16_tf_next_is_im_end"]))
            sig2[arm] = dict(hits_gt0=hits_gt0, partial_hits=hits_partial, im_end=imend)
        rep["strata"][s]["bf16fail_set_signature"] = sig2

        # ---------- GEN text shape by arm (willingness to elaborate) ----------
        shape = {}
        for arm in ("q0", "bf16"):
            cnt = collections.Counter()
            for r in sub:
                cnt[head_style(gen_of(r, arm)[1])] += 1
            shape[arm] = dict(cnt)
        rep["strata"][s]["gen_shape"] = shape

        # ---------- sub-mode breakdown ----------
        sm = collections.defaultdict(lambda: collections.Counter())
        for r in sub:
            k = sub_mode(r)
            sm[k]["n"] += 1
            sm[k]["q0_ok"] += int(bool(r["tf_correct"]))
            sm[k]["bf16_ok"] += int(bool(r["bf16_tf_correct"]))
            sm[k]["bf16_fail_q0_ok"] += int(bool(r["tf_correct"] and not r["bf16_tf_correct"]))
            sm[k]["q0_fail_bf16_ok"] += int(bool(r["bf16_tf_correct"] and not r["tf_correct"]))
        rep["strata"][s]["sub_mode"] = {k: dict(v) for k, v in sorted(sm.items())}

        # ---------- samples: BF16-fail/q0-success, side by side ----------
        samp = []
        for r in set_rows[: a.samples]:
            samp.append(dict(
                id=r["id"], question=r["question"], gold=r["answer"],
                q0_hits=f"{r['tf_token_hits']}/{r['tf_tokens']}",
                bf16_hits=f"{r['bf16_tf_token_hits']}/{r['bf16_tf_tokens']}",
                q0_next=r["tf_next_token"], bf16_next=r["bf16_tf_next_token"],
                bf16_gen=r["bf16_gen_text"], bf16_gen_parse=r["bf16_gen_parsed"],
                bf16_gen_reason=r["bf16_gen_reason"], bf16_gen_finish=r["bf16_gen_finish"],
                q0_gen=r["gen_text"], q0_gen_correct=r["gen_correct"],
                bf16_gen_correct=r["bf16_gen_correct"],
                cls=("b" if r["bf16_gen_correct"] else
                     ("a" if r["bf16_gen_parsed"] is not None else "c"))))
        rep["samples"][s] = samp

        # ---------- paired episode view ----------
        # episodes are not in the generation pool; use 8-item pseudo-clusters by index
        d = [int(bool(r["bf16_tf_correct"])) - int(bool(r["tf_correct"])) for r in sub]
        rep["strata"][s]["item_level_means"] = dict(mean_gap=sum(d) / len(d))

    json.dump(rep, open(a.out, "w"), indent=2, ensure_ascii=False, sort_keys=True)
    print(json.dumps({s: {k: v for k, v in rep["strata"][s].items()
                          if k in ("n", "q0_acc", "bf16_acc", "gap", "q0_ok_bf16_bad",
                                   "q0_bad_bf16_ok", "bf16_fail_q0_ok_cls",
                                   "q0_fail_bf16_ok_cls", "bf16_im_end", "q0_im_end")}
                      for s in strata}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
