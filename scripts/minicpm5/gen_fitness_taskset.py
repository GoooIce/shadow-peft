"""程序化合成 MiniCPM5-2B NVFP4 W4A4 适应度题集（16 票）。

用法（Mac / GB10 均可，纯标准库）：
  python3 gen_fitness_taskset.py gen --out docs/experiments/artifacts/.../pool.jsonl --per-stratum 270
  python3 gen_fitness_taskset.py selfcheck --pool <pool.jsonl>

可复算：所有随机数由 BASE_SEED + family + tier + index 的 SHA256 派生；
输出 JSONL 按 id 排序，逐字节可复现（pool_manifest.json 记 sha256）。
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from taskset_lib import (  # noqa: E402
    BASE_SEED, FAMILIES, NAME_BANK, TIERS, eval_bool_expr, eval_int_expr,
    max_expr_value, self_check, sha12, stable_seed, verify,
)


# ---------------------------------------------------------------- 各家族生成器

def _int_ok(v, cap=999999):
    return abs(int(v)) <= cap


def _expr_max_abs(node_str, cap=10 ** 7):
    import ast
    node = ast.parse(node_str, mode="eval").body
    worst = [0]

    def rec(n):
        if isinstance(n, ast.Constant):
            return abs(int(n.value))
        if isinstance(n, ast.UnaryOp):
            return rec(n.operand)
        a, b = rec(n.left), rec(n.right)
        if isinstance(n.op, ast.Mult):
            v = a * b
        elif isinstance(n.op, ast.Add):
            v = a + b
        elif isinstance(n.op, ast.Sub):
            v = a + b
        else:
            v = a + b
        worst[0] = max(worst[0], v)
        return v

    rec(node)
    return worst[0]


def gen_arith(rng, tier, idx):
    # 档位经一次 q0 标定后收紧：原 medium/hard（49/99 量级、4-5 项）q0 准确率仅
    # 5.9% / 0.7%，落在判别带外，故降到 12/29 量级、2-4 项。
    cfg = {
        "easy":   dict(n=(2, 3), lo=2, hi=19, ops="+-", depth=0),
        "medium": dict(n=(2, 3), lo=2, hi=12, ops="+-*", depth=1),
        "hard":   dict(n=(3, 4), lo=3, hi=29, ops="+-*", depth=2),
    }[tier]

    def build(n):
        if n == 1:
            return str(rng.randint(cfg["lo"], cfg["hi"]))
        k = rng.randint(1, n - 1)
        left, right = build(k), build(n - k)
        op = rng.choice(cfg["ops"])
        s = f"{left} {op} {right}"
        if cfg["depth"] > 0 and rng.random() < 0.45:
            s = f"({s})"
        return s

    for _ in range(200):
        expr = build(rng.randint(*cfg["n"]))
        if cfg["ops"] != "+-" and "*" not in expr:
            continue
        try:
            val = eval_int_expr(expr)
        except ValueError:
            continue
        if not _int_ok(val):
            continue
        if _expr_max_abs(expr) > 10 ** 6:
            continue
        # 排除退化：表达式值等于某个操作数、或结果为 0/1
        if abs(val) in (0, 1):
            continue
        tmpl = rng.choice([
            "Compute {e}.",
            "What is the value of {e}?",
            "Evaluate {e}.",
            "Find {e}.",
            "Calculate {e}.",
        ])
        q = tmpl.format(e=expr)
        return dict(question=q, answer=str(int(val)),
                    verify=dict(kind="arith_int", expr=expr),
                    meta=dict(expr=expr))
    raise RuntimeError("arith generation failed")


_MAXEXPR_TEMPLATES = (
    "Using each of the numbers {l} exactly once, with the operations +, -, and * "
    "(parentheses allowed), what is the largest value you can make?",
    "You are given the numbers {l}. Combine all of them, each exactly once, using only "
    "addition, subtraction and multiplication, and arrange parentheses freely. "
    "What is the greatest result you can obtain?",
    "With the numbers {l}, each used exactly once and only the operators +, - and *, "
    "what is the maximum possible value?",
    "Take the numbers {l}. Insert +, - or * between them (in any grouping with "
    "parentheses), using every number exactly once, to maximise the value. "
    "What is that maximum?",
)


def gen_maxexpr(rng, tier, idx):
    # 数字池必须含 1：所有数都 >=2 时最大值恒等于全乘积、题目退化（max >= prod 恒成立）。
    # 用有放回采样扩大题面空间（保证每个 stratum 都能取到足够多的互异题面）。
    k, hi = {"easy": (3, 20), "medium": (4, 15), "hard": (5, 15)}[tier]
    for _ in range(400):
        nums = sorted(rng.choices(range(1, hi + 1), k=k))
        best = max_expr_value(nums)
        prod = 1
        for x in nums:
            prod *= x
        if best <= 0 or best > 10 ** 6:
            continue
        if best == prod:
            continue  # 平凡：最大值就是乘积（无搜索量）
        lst = ", ".join(str(x) for x in nums)
        q = rng.choice(_MAXEXPR_TEMPLATES).format(l=lst)
        return dict(question=q, answer=str(best),
                    verify=dict(kind="maxexpr", numbers=nums),
                    meta=dict(product=prod, best=best))
    raise RuntimeError("maxexpr generation failed")


_PCT_TEMPLATES = ("What is {x} of {n}?", "Compute {x} of {n}.", "Find {x} of {n}.",
                  "Calculate {x} of {n}.")


def gen_pct(rng, tier, idx):
    """百分比 / 分数 / 反推原值 / 复合增减 —— 全部整数答案。"""
    for _ in range(200):
        sub = {"easy": ["pct"], "medium": ["pct", "frac"],
               "hard": ["reverse", "net"]}[tier][0]
        pools = {"easy": ["pct"], "medium": ["pct", "frac"], "hard": ["reverse", "net"]}[tier]
        sub = rng.choice(pools)
        if sub == "pct":
            if tier == "easy":
                p, n = rng.choice([10, 20, 25, 50, 75]), rng.randrange(4, 41) * 4
            else:
                p, n = rng.choice([5, 15, 35, 45, 60, 85]), rng.randrange(4, 41) * 20
            if (n * p) % 100:
                continue
            ans = n * p // 100
            x = f"{p}%"
            q = rng.choice(_PCT_TEMPLATES).format(x=x, n=n)
            return dict(question=q, answer=str(ans), verify=dict(kind="pct_of", n=n, p=p),
                        meta=dict(sub=sub))
        if sub == "frac":
            b = rng.choice([4, 5, 8, 10])
            a = rng.randrange(1, b)
            n = rng.randrange(3, 60) * b
            if (n * a) % b:
                continue
            ans = n * a // b
            q = rng.choice(_PCT_TEMPLATES).format(x=f"{a}/{b}", n=n)
            return dict(question=q, answer=str(ans),
                        verify=dict(kind="frac_of", n=n, a=a, b=b), meta=dict(sub=sub))
        if sub == "reverse":
            p = rng.choice([20, 25, 40, 50, 60, 75])
            o = rng.randrange(4, 60) * (100 // __import__("math").gcd(100, p))
            v = o * (100 + p) // 100
            if o * (100 + p) % 100 or v > 99999:
                continue
            q = rng.choice([
                f"After an increase of {p}%, a value becomes {v}. What was the original value?",
                f"A number grows by {p}% and the result is {v}. What was the number?",
                f"After a {p}% raise, a price is {v}. What was the price before the raise?",
            ])
            return dict(question=q, answer=str(o),
                        verify=dict(kind="pct_original", v=v, p=p), meta=dict(sub=sub))
        # net
        p = rng.choice([10, 20, 25, 30, 40])
        qd = rng.choice([10, 20, 25, 50])
        o = rng.randrange(4, 60) * 20
        num = o * (100 + p) * (100 - qd)
        if num % 10000 or num // 10000 > 99999:
            continue
        ans = num // 10000
        q = rng.choice([
            f"A quantity starts at {o}. It rises by {p}% and then falls by {qd}%. "
            f"What is the final value?",
            f"Start with {o}. Increase it by {p}%, then decrease the result by {qd}%. "
            f"What number do you end with?",
        ])
        return dict(question=q, answer=str(ans),
                    verify=dict(kind="pct_net", o=o, p=p, q=qd), meta=dict(sub=sub))
    raise RuntimeError("pct generation failed")


def _gen_bool_formula(rng, variables, depth):
    if depth <= 0 or rng.random() < 0.28:
        v = rng.choice(variables)
        return f"(NOT {v})" if rng.random() < 0.25 else v
    op = "AND" if rng.random() < 0.5 else "OR"
    a = _gen_bool_formula(rng, variables, depth - 1)
    b = _gen_bool_formula(rng, variables, depth - 1)
    s = f"({a} {op} {b})"
    if rng.random() < 0.15:
        s = f"(NOT {s})"
    return s


def gen_logic(rng, tier, idx):
    nv, nf, depth = {"easy": (3, 3, 1), "medium": (4, 4, 2), "hard": (4, 5, 3)}[tier]
    variables = list("abcd"[:nv])
    assignment = {v: rng.random() < 0.5 for v in variables}
    formulas = [_gen_bool_formula(rng, variables, depth) for _ in range(nf)]
    cnt = sum(1 for f in formulas if eval_bool_expr(f, assignment))
    if cnt == 0 or cnt == nf:  # 全真/全假太容易靠猜
        return gen_logic(rng, tier, idx + 100000)
    assigns = ", ".join(f"{v} = {assignment[v]}" for v in variables)
    lines = "\n".join(f"{i+1}) {f}" for i, f in enumerate(formulas))
    q = (f"Given {assigns}.\nConsider the following statements:\n{lines}\n"
         f"How many of the statements are true?")
    return dict(question=q, answer=str(cnt),
                verify=dict(kind="logic_count", assignment=assignment, formulas=formulas),
                meta=dict(n_true=cnt, n_formulas=nf))


def _forced_below(relations, name):
    """relation = (a, b) 表示 a 比 b 高。返回传递闭包下 name 的强制下方集合。"""
    below = set()
    changed = True
    while changed:
        changed = False
        for a, b in relations:
            if (a == name or a in below) and b not in below:
                below.add(b)
                changed = True
    return below


def gen_ordering(rng, tier, idx):
    n, qtype = {"easy": (3, "tallest"), "medium": (4, "tallest"), "hard": (5, "rank")}[tier]
    names = rng.sample(NAME_BANK, n)
    order = names[:]  # order[0] 最高
    relations = []
    # 先给出链（保证全序可推），再掺入冗余/乱序
    for i in range(n - 1):
        relations.append((order[i], order[i + 1]))
    extra = rng.randint(1, 3)
    for _ in range(extra):
        i, j = sorted(rng.sample(range(n), 2))
        relations.append((order[i], order[j]))
    rng.shuffle(relations)
    stmts = []
    for a, b in relations:
        if rng.random() < 0.5:
            stmts.append(f"{a} is taller than {b}.")
        else:
            stmts.append(f"{b} is shorter than {a}.")
    if qtype == "tallest":
        k = 1
        qline = "Who is the tallest?"
    else:
        k = rng.choice([2, 3])
        suffix = "nd" if k == 2 else "rd"
        qline = f"Who is the {k}{suffix} tallest?"
    gold = order[k - 1]
    q = "\n".join(stmts) + "\n" + qline
    return dict(question=q, answer=gold,
                verify=dict(kind="ordering", names=names, answer_name=gold),
                meta=dict(order=order, relations=relations, rank=k))


def gen_modulo(rng, tier, idx):
    kind = rng.choice({"easy": ["rem"], "medium": ["rem", "last_digit", "gcd"],
                       "hard": ["rem", "last_digit", "gcd"]}[tier])
    if kind == "rem":
        if tier == "easy":
            a, b = rng.randint(10, 99), rng.randint(2, 9)
        elif tier == "medium":
            a, b = rng.randint(1000, 9999), rng.randint(7, 13)
        else:
            a, b = rng.randint(100000, 999999), rng.randint(17, 31)
        if a % b == 0:
            return gen_modulo(rng, tier, idx + 100000)
        q = f"What is the remainder when {a} is divided by {b}?"
        return dict(question=q, answer=str(a % b),
                    verify=dict(kind="modulo_rem", a=a, b=b), meta=dict(kind="rem"))
    if kind == "last_digit":
        if tier == "medium":
            a, e = rng.randint(2, 9), rng.randint(2, 99)
        else:
            a, e = rng.randint(12, 99), rng.randint(100, 9999)
        q = f"What is the last digit of {a}^{e}?"
        return dict(question=q, answer=str(pow(a, e, 10)),
                    verify=dict(kind="modulo_last_digit", a=a, e=e), meta=dict(kind="last_digit"))
    # gcd
    if tier == "medium":
        a, b = rng.randint(10, 99), rng.randint(10, 99)
    else:
        a, b = rng.randint(1000, 9999), rng.randint(1000, 9999)
    g = __import__("math").gcd(a, b)
    if g <= 1 or g == min(a, b):
        return gen_modulo(rng, tier, idx + 100000)
    q = f"What is the greatest common divisor of {a} and {b}?"
    return dict(question=q, answer=str(g),
                verify=dict(kind="modulo_gcd", a=a, b=b), meta=dict(kind="gcd"))


GENERATORS = {
    "arith": gen_arith,
    "pct": gen_pct,
    "maxexpr": gen_maxexpr,   # 未随题集交付（q0 <=1.5%，见 taskset_lib.FAMILIES 注释）
    "logic": gen_logic,
    "ordering": gen_ordering,
    "modulo": gen_modulo,
}


# ---------------------------------------------------------------- 组装

def build_pool(per_stratum: int):
    items, seen = [], set()
    for fam in FAMILIES:
        for tier in TIERS:
            got = 0
            attempt = 0
            while got < per_stratum:
                if attempt > 400 * per_stratum:  # 防死循环：题面空间不足时显式失败
                    raise RuntimeError(
                        f"stratum {fam}/{tier}: only {got}/{per_stratum} distinct items "
                        f"after {attempt} attempts")
                seed = stable_seed(BASE_SEED, fam, tier, attempt)
                rng = random.Random(seed)
                try:
                    it = GENERATORS[fam](rng, tier, attempt)
                except RuntimeError:
                    attempt += 1
                    continue
                attempt += 1
                if it["question"] in seen:
                    continue
                ok, _, _ = verify(it, it["answer"])
                if not ok:
                    continue
                seen.add(it["question"])
                it.update(family=fam, tier=tier, stratum=f"{fam}/{tier}", seed=seed,
                          id=f"mini16-{fam}-{tier}-{got:04d}-{sha12(it['question'] + '|' + it['answer'])}")
                items.append(it)
                got += 1
    items.sort(key=lambda x: x["id"])
    return items


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--out", required=True)
    g.add_argument("--per-stratum", type=int, default=270)
    s = sub.add_parser("selfcheck")
    s.add_argument("--pool", required=True)
    a = ap.parse_args()

    if a.cmd == "gen":
        items = build_pool(a.per_stratum)
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False, sort_keys=True) + "\n")
        chk = self_check(items)
        import hashlib
        h = hashlib.sha256(open(a.out, "rb").read()).hexdigest()
        manifest = dict(schema="mini16-taskset-v1", base_seed=BASE_SEED,
                        per_stratum=a.per_stratum, families=list(FAMILIES), tiers=list(TIERS),
                        n_items=len(items), sha256=h, self_check=chk, path=os.path.basename(a.out))
        with open(a.out.replace(".jsonl", "_manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False, sort_keys=True)
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
    else:
        items = [json.loads(l) for l in open(a.pool)]
        print(json.dumps(self_check(items), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
