"""Gold / verifier integrity check for the strata under investigation.

Two independent questions:
  1. Does the QUESTION TEXT agree with the verify spec (the thing the checker actually uses)?
     If the text and the spec disagree, the gold can be wrong for a careful reader.
  2. Is the gold UNIQUELY determined by the question text on its own terms?
     For `ordering`: relations are sampled with replacement, so the same sentence can appear
     twice; the gold is only determined if the relation set admits exactly one topological
     order. Contradictory relations would make the gold a convention rather than a fact.
The logic count is recomputed here with an independent evaluator (text -> python eval) rather
than taskset_lib.eval_bool_expr, so a shared-implementation bug cannot hide the answer.
"""
from __future__ import annotations

import argparse
import collections
import itertools
import json
import re


def load(p):
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


# ---- independent boolean evaluator (recursive descent, not taskset_lib) --------------------
def _tokens(f):
    return re.findall(r"\(|\)|\bAND\b|\bOR\b|\bNOT\b|\b[A-Za-z_]\w*\b", f)


def _parse(toks, i, env):
    """returns (value, next_index); grammar: OR := AND ('OR' AND)* , AND := UNARY ('AND' UNARY)*,
    UNARY := 'NOT' UNARY | '(' OR ')' | VAR"""
    def parse_unary(i):
        t = toks[i]
        if t == "NOT":
            v, i = parse_unary(i + 1)
            return (not v), i
        if t == "(":
            v, i = parse_or(i + 1)
            assert toks[i] == ")", f"bad parens at {i}"
            return v, i + 1
        return bool(env[t]), i + 1

    def parse_and(i):
        v, i = parse_unary(i)
        while i < len(toks) and toks[i] == "AND":
            w, i = parse_unary(i + 1)
            v = v and w
        return v, i

    def parse_or(i):
        v, i = parse_and(i)
        while i < len(toks) and toks[i] == "OR":
            w, i = parse_and(i + 1)
            v = v or w
        return v, i

    v, j = parse_or(i)
    assert j == len(toks), f"trailing tokens in {toks}"
    return v


def eval_formula(f: str, env: dict) -> bool:
    toks = _tokens(f)
    assert toks, f"empty formula {f!r}"
    return _parse(toks, 0, env)


def text_formulas(q: str):
    """numbered lines '1) ...' of the question text."""
    return [m.group(1).strip() for m in re.finditer(r"^\s*\d+\)\s*(.+?)\s*$", q, re.M)]


def parse_relations(q: str):
    """'X is taller than Y.' / 'X is shorter than Y.' lines -> (taller, shorter) pairs."""
    out = []
    for line in q.splitlines():
        m = re.match(r"^(\w+) is taller than (\w+)\.\s*$", line.strip())
        if m:
            out.append((m.group(1), m.group(2)))
            continue
        m = re.match(r"^(\w+) is shorter than (\w+)\.\s*$", line.strip())
        if m:
            out.append((m.group(2), m.group(1)))
    return out


def n_linear_orders(pairs, names):
    """count topological orders of the strict partial order induced by pairs."""
    n = 0
    orders = []
    for perm in itertools.permutations(names):
        pos = {nm: i for i, nm in enumerate(perm)}
        if all(pos[a] < pos[b] for a, b in pairs):
            n += 1
            orders.append(perm)
            if n > 2:
                break
    return n, orders


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", default="/tmp/minicpm5-sanity-data/full.jsonl")
    ap.add_argument("--strata", default="logic/hard,ordering/hard,pct/easy")
    ap.add_argument("--limit", type=int, default=0, help="0 = all items")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    strata = a.strata.split(",")

    rep = {}
    for s in strata:
        fam = s.split("/")[0]
        n = 0
        text_spec_mismatch = []
        gold_mismatch = []
        ambiguous = 0
        contradictory = 0
        dup_lines = 0
        for r in load(a.full):
            if r["stratum"] != s:
                continue
            n += 1
            if a.limit and n > a.limit:
                break
            q, v = r["question"], r["verify"]
            if fam == "logic":
                fs_text = text_formulas(q)
                fs_spec = v["formulas"]
                same = (len(fs_text) == len(fs_spec) and
                        all(re.sub(r"\s+", "", x) == re.sub(r"\s+", "", y)
                            for x, y in zip(fs_text, fs_spec)))
                if not same:
                    text_spec_mismatch.append(r["id"])
                cnt = sum(1 for f in fs_spec if eval_formula(f, v["assignment"]))
                if int(r["answer"]) != cnt:
                    gold_mismatch.append((r["id"], r["answer"], cnt))
            elif fam == "ordering":
                rels_text = parse_relations(q)
                rels_spec = [tuple(x) for x in (r["meta"] or {}).get("relations", [])]
                if [tuple(x) for x in rels_text] != rels_spec:
                    text_spec_mismatch.append(r["id"])
                if len({tuple(x) for x in rels_text}) < len(rels_text):
                    dup_lines += 1
                names = v["names"]
                cnt, orders = n_linear_orders(rels_text, names)
                if cnt == 0:
                    contradictory += 1
                elif cnt > 1:
                    ambiguous += 1
                else:
                    top = orders[0]
                    rank = (r["meta"] or {}).get("rank", 1)
                    gold_by_order = top[rank - 1]
                    if gold_by_order != r["answer"]:
                        gold_mismatch.append((r["id"], r["answer"], gold_by_order))
        rep[s] = dict(n=n, text_spec_mismatch=len(text_spec_mismatch),
                      text_spec_mismatch_examples=text_spec_mismatch[:5],
                      gold_mismatch=len(gold_mismatch), gold_mismatch_examples=gold_mismatch[:5],
                      ambiguous_multi_order=ambiguous, contradictory=contradictory,
                      items_with_duplicate_relation_lines=dup_lines)
    json.dump(rep, open(a.out, "w"), indent=2, ensure_ascii=False, sort_keys=True)
    print(json.dumps(rep, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
