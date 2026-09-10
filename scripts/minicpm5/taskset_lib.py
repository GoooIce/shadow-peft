"""MiniCPM5-2B NVFP4 W4A4 适应度题集：题目生成原语 / 验证器 / prompt 渲染（16 票）。

设计约束（07 冻结协议）：
  * 短答案、可程序化验证、单一主轴（可验证正确率）+ 硬门；
  * 每题含 prompt、标准答案、验证器（验证器 = 本模块，生成与评分共用同一实现）；
  * 难度分档（family x tier），episode 级统计（episode = 同 stratum 的固定大小题簇）；
  * 纯标准库，Mac（代码检查）与 GB10（评分）行为一致。

题集条目 JSONL 字段：
  id, family, tier, stratum, question, answer(规范答案串), verify(验证器规格),
  solution(可选规范解), seed, meta
"""
from __future__ import annotations

import ast
import hashlib
import random
import re
from fractions import Fraction

BASE_SEED = 20260910
# maxexpr（用 +-* 凑最大值）在有 q0 标定后被移出交付题集：q0 准确率 <= 1.5%（搜索型任务，
# 2B 量化模型几乎全错），无判别力。生成器保留在 gen_fitness_taskset.py 里作硬尾探针。
FAMILIES = ("arith", "pct", "logic", "ordering", "modulo")
TIERS = ("easy", "medium", "hard")

INSTRUCTION = "Reply with only the final answer, with no explanation."
ANSWER_PREFIX = "Answer:"          # 不带尾随空格（保证 BPE 前缀稳定）
GOLD_PREFIX = " "                  # gold 文本 = GOLD_PREFIX + answer

NAME_BANK = (
    "Anna", "Ben", "Cara", "Dan", "Ella", "Finn", "Gina", "Hugo", "Iris", "Jack",
    "Kira", "Leo", "Mona", "Noah", "Olga", "Pete", "Quinn", "Rosa", "Sam", "Tara",
    "Umar", "Vera", "Wes", "Xena", "Yuri", "Zoe", "Adam", "Bella", "Cole", "Dora",
    "Evan", "Faye", "Glen", "Hana", "Ivan", "June", "Karl", "Lena", "Milo", "Nina",
)


# ---------------------------------------------------------------- 基础工具

def stable_seed(*parts) -> int:
    """跨进程稳定的种子（不用内置 hash，它对 str 随机化）。"""
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(h[:8], "big")


def sha12(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:12]


# ---------------------------------------------------------------- 安全求值

_ALLOWED_BINOPS = {ast.Add: lambda a, b: a + b,
                   ast.Sub: lambda a, b: a - b,
                   ast.Mult: lambda a, b: a * b,
                   ast.Div: lambda a, b: a / b}


def eval_int_expr(expr: str, allow_div: bool = False):
    """只允许整数常量与 + - * (/)，返回精确值（Fraction 或 int）。非法输入抛 ValueError。"""
    try:
        node = ast.parse(expr, mode="eval").body
    except SyntaxError as e:
        raise ValueError(f"syntax: {e}") from e

    def rec(n):
        if isinstance(n, ast.Constant):
            if isinstance(n.value, bool) or not isinstance(n.value, (int, float)):
                raise ValueError("bad constant")
            if isinstance(n.value, float):
                if n.value != int(n.value):
                    raise ValueError("float constant")
                return int(n.value)
            return n.value
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
            return -rec(n.operand)
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.UAdd):
            return rec(n.operand)
        if isinstance(n, ast.BinOp):
            op = n.op
            if isinstance(op, ast.Div):
                if not allow_div:
                    raise ValueError("division not allowed")
                a, b = rec(n.left), rec(n.right)
                if b == 0:
                    raise ValueError("division by zero")
                return Fraction(a) / Fraction(b)
            if isinstance(op, ast.Mult):
                return rec(n.left) * rec(n.right)
            if isinstance(op, ast.Add):
                return rec(n.left) + rec(n.right)
            if isinstance(op, ast.Sub):
                return rec(n.left) - rec(n.right)
            raise ValueError("op not allowed")
        raise ValueError("node not allowed")

    return rec(node)


def eval_bool_expr(expr: str, env: dict) -> bool:
    """命题逻辑公式求值；只允许 AND / OR / NOT / 括号 / env 里的变量名。

    题面用大写算子（对模型更可读），求值前把 AND/OR/NOT 翻成 Python 算子再走 ast。
    """
    py = re.sub(r"\bAND\b", "and", expr)
    py = re.sub(r"\bOR\b", "or", py)
    py = re.sub(r"\bNOT\b", "not", py)
    node = ast.parse(py, mode="eval").body

    def rec(n):
        if isinstance(n, ast.Name):
            if n.id not in env:
                raise ValueError(f"unknown var {n.id}")
            return bool(env[n.id])
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.Not):
            return not rec(n.operand)
        if isinstance(n, ast.BoolOp):
            if isinstance(n.op, ast.And):
                return all(rec(v) for v in n.values)
            if isinstance(n.op, ast.Or):
                return any(rec(v) for v in n.values)
        raise ValueError("node not allowed")

    return bool(rec(node))


def normalize_expr_text(text: str) -> str:
    """把模型输出的表达式规范化成可解析形式。"""
    t = text.strip()
    t = t.splitlines()[0] if t else t
    t = t.replace("×", "*").replace("·", "*").replace("÷", "/").replace("−", "-")
    t = t.replace("**", "^")  # 不支持幂运算，交给验证器拒绝
    t = re.sub(r"[^0-9+\-*/().\s]", " ", t)
    t = t.strip().rstrip(".")
    return t


# ---------------------------------------------------------------- 答案解析

def parse_int_answer(text: str):
    """从模型输出里抽第一个整数（允许千分位逗号、前导 'Answer:'）。"""
    t = text.strip()
    if not t:
        return None
    t = t.splitlines()[0]
    if ":" in t:
        t = t.rsplit(":", 1)[1]
    m = re.search(r"-?\d[\d,]*", t)
    if not m:
        return None
    return int(m.group(0).replace(",", ""))


def parse_bool_answer(text: str):
    t = text.strip().lower()
    if re.search(r"\btrue\b|\byes\b", t):
        return True
    if re.search(r"\bfalse\b|\bno\b", t):
        return False
    return None


def parse_name_answer(text: str, names):
    t = text.strip()
    for line in [t] + t.splitlines():
        for n in names:
            if re.search(rf"\b{re.escape(n)}\b", line, re.IGNORECASE):
                return n
    return None


# ---------------------------------------------------------------- 验证器

def verify(item: dict, model_text: str):
    """返回 (ok, parsed, reason)。验证器规格完全由 item['verify'] 决定。"""
    spec = item["verify"]
    kind = spec["kind"]
    try:
        if kind == "arith_int":
            got = parse_int_answer(model_text)
            if got is None:
                return False, None, "no_integer"
            gold = eval_int_expr(spec["expr"])
            return int(got) == int(gold), got, "ok" if int(got) == int(gold) else "mismatch"

        if kind == "maxexpr":
            nums = list(spec["numbers"])
            got = parse_int_answer(model_text)
            if got is None:
                return False, None, "no_integer"
            gold = max_expr_value(nums)
            return int(got) == int(gold), got, "ok" if int(got) == int(gold) else "mismatch"

        if kind == "logic_count":
            got = parse_int_answer(model_text)
            if got is None:
                return False, None, "no_integer"
            env = spec["assignment"]
            cnt = sum(1 for f in spec["formulas"] if eval_bool_expr(f, env))
            return int(got) == cnt, got, "ok" if int(got) == cnt else "mismatch"

        if kind == "ordering":
            names = spec["names"]
            got = parse_name_answer(model_text, names)
            if got is None:
                return False, None, "no_name"
            gold = spec["answer_name"]
            return got == gold, got, "ok" if got == gold else "mismatch"

        if kind == "modulo_rem":
            got = parse_int_answer(model_text)
            if got is None:
                return False, None, "no_integer"
            gold = spec["a"] % spec["b"]
            return int(got) == gold, got, "ok" if int(got) == gold else "mismatch"

        if kind == "modulo_last_digit":
            got = parse_int_answer(model_text)
            if got is None:
                return False, None, "no_integer"
            gold = pow(spec["a"], spec["e"], 10)
            return int(got) == gold, got, "ok" if int(got) == gold else "mismatch"

        if kind == "modulo_gcd":
            got = parse_int_answer(model_text)
            if got is None:
                return False, None, "no_integer"
            gold = _gcd(spec["a"], spec["b"])
            return int(got) == gold, got, "ok" if int(got) == gold else "mismatch"

        if kind == "pct_of":
            got = parse_int_answer(model_text)
            if got is None:
                return False, None, "no_integer"
            gold = Fraction(spec["n"] * spec["p"], 100)
            if gold.denominator != 1:
                raise ValueError("non-integer gold")
            return int(got) == int(gold), got, "ok" if int(got) == int(gold) else "mismatch"

        if kind == "frac_of":
            got = parse_int_answer(model_text)
            if got is None:
                return False, None, "no_integer"
            gold = Fraction(spec["n"] * spec["a"], spec["b"])
            if gold.denominator != 1:
                raise ValueError("non-integer gold")
            return int(got) == int(gold), got, "ok" if int(got) == int(gold) else "mismatch"

        if kind == "pct_original":
            got = parse_int_answer(model_text)
            if got is None:
                return False, None, "no_integer"
            gold = Fraction(spec["v"] * 100, 100 + spec["p"])
            if gold.denominator != 1:
                raise ValueError("non-integer gold")
            return int(got) == int(gold), got, "ok" if int(got) == int(gold) else "mismatch"

        if kind == "pct_net":
            got = parse_int_answer(model_text)
            if got is None:
                return False, None, "no_integer"
            gold = Fraction(spec["o"] * (100 + spec["p"]) * (100 - spec["q"]), 100 * 100)
            if gold.denominator != 1:
                raise ValueError("non-integer gold")
            return int(got) == int(gold), got, "ok" if int(got) == int(gold) else "mismatch"

        if kind == "countdown_expr":
            # 经典 Countdown：任一合法表达式都算对（本票作辅助臂，不进 teacher-forced 主轴）
            expr = normalize_expr_text(model_text)
            if not expr:
                return False, None, "empty"
            try:
                val = eval_int_expr(expr, allow_div=True)
            except ValueError as e:
                return False, None, f"bad_expr:{e}"
            used = sorted(int(x) for x in re.findall(r"\d+", expr))
            want = sorted(spec["numbers"])
            if used != want:
                return False, None, "numbers_mismatch"
            if Fraction(val) != Fraction(spec["target"]):
                return False, None, "value_mismatch"
            return True, str(val), "ok"

        raise ValueError(f"unknown verifier kind {kind}")
    except Exception as e:  # 验证器自身异常按不通过处理，但要显式标注
        return False, None, f"verifier_error:{type(e).__name__}:{e}"


def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return abs(a)


# ---------------------------------------------------------------- maxexpr 精确解

def _minmax_rec(nums, mask, memo):
    """子集上所有表达式（+ - *，每个数一次）的 (min, max)。因为 + 与 * 在盒上是双线性的、
    - 的极值取 (max_a - min_b)，(min,max) 对足以精确给出全局极值。"""
    if mask in memo:
        return memo[mask]
    bits = [i for i in range(len(nums)) if (mask >> i) & 1]
    if len(bits) == 1:
        v = nums[bits[0]]
        memo[mask] = (v, v)
        return memo[mask]
    lo = hi = None
    low = bits[0]
    sub = mask
    while True:
        sub = (sub - 1) & mask
        if sub == 0:
            break
        if not (sub & (1 << low)):
            continue
        a_lo, a_hi = _minmax_rec(nums, sub, memo)
        b_lo, b_hi = _minmax_rec(nums, mask ^ sub, memo)
        cands = (a_lo + b_lo, a_hi + b_hi, a_lo - b_hi, a_hi - b_lo,
                 a_lo * b_lo, a_lo * b_hi, a_hi * b_lo, a_hi * b_hi)
        mn, mx = min(cands), max(cands)
        lo = mn if lo is None else min(lo, mn)
        hi = mx if hi is None else max(hi, mx)
    memo[mask] = (lo, hi)
    return memo[mask]


def max_expr_value(nums) -> int:
    """用 + - * 与括号、每个数恰好一次能取得的最大值（整数，精确解）。"""
    nums = [int(x) for x in nums]
    return _minmax_rec(nums, (1 << len(nums)) - 1, {})[1]


def max_expr_value_bruteforce(nums) -> int:
    """穷举校验用（小规模）：枚举全部可达值集合。"""
    nums = [int(x) for x in nums]
    n = len(nums)
    memo: dict[int, set] = {}

    def rec(mask: int):
        if mask in memo:
            return memo[mask]
        bits = [i for i in range(n) if (mask >> i) & 1]
        if len(bits) == 1:
            memo[mask] = {nums[bits[0]]}
            return memo[mask]
        out = set()
        low = bits[0]
        sub = mask
        while True:
            sub = (sub - 1) & mask
            if sub == 0:
                break
            if not (sub & (1 << low)):
                continue
            for x in rec(sub):
                for y in rec(mask ^ sub):
                    out.update((x + y, x - y, y - x, x * y))
        memo[mask] = out
        return out

    return max(rec((1 << n) - 1))


# ---------------------------------------------------------------- prompt 渲染

def question_text(item: dict) -> str:
    return item["question"]


def render_prompt(tokenizer, item: dict, enable_thinking: bool = False) -> str:
    """06 口径：显式传 enable_thinking；末尾固定 'Answer: ' 前缀，答案 span 从这里开始。"""
    msgs = [{"role": "user", "content": item["question"] + "\n\n" + INSTRUCTION}]
    text = tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
    )
    return text + ANSWER_PREFIX


def answer_span(tokenizer, prompt: str, answer: str):
    """返回 (prompt_ids, gold_ids, stable)。

    prompt 以 'Answer:' 结尾（不带尾随空格），gold 文本 = ' ' + answer —— 这样 BPE 前缀
    稳定（否则 'Answer: ' 末尾的空格 token 会与答案首 token 合并，prompt 截断位置漂移）。
    仍做公共前缀校验并返回 stable 标志，不稳定的条目在评分器里计数上报。
    """
    p_ids = tokenizer.encode(prompt, add_special_tokens=False)
    full = tokenizer.encode(prompt + answer, add_special_tokens=False)
    k = 0
    while k < min(len(p_ids), len(full)) and p_ids[k] == full[k]:
        k += 1
    if k == len(p_ids):
        return p_ids, full[k:], True
    return p_ids[:k], full[k:], False


# ------------------------------------------------- prompt 变体（harness swap，第 2 轮）

PROMPT_VARIANTS = ("frozen", "plain")
DEFAULT_PROMPT_VARIANT = "frozen"
#: `frozen` = 冻结口径（chat template + 固定 'Answer:' 后缀）；`plain` = 自然 chat template，
#: **不带**后缀。第 2 轮的 harness swap 臂（HarnessDev 的执行器替换）用后者：`Answer:` 后缀
#: 已被证明会诱发 logic/hard、ordering/hard 上的"直接结束回合"模式，只在带后缀的臂上量到的
#: 增益可能不迁移到部署形态。
PROMPT_VARIANT_NOTES = {
    "frozen": "chat template + frozen 'Answer:' suffix (the frozen primary form)",
    "plain": "natural chat template, no 'Answer:' suffix (harness-swap form)",
}


def render_prompt_variant(tokenizer, item: dict, variant: str = DEFAULT_PROMPT_VARIANT,
                          enable_thinking: bool = False) -> str:
    """按变体渲染 prompt。**`frozen` 与 :func:`render_prompt` 逐字节相同。**

    这是第 2 轮加入的唯一渲染入口改动，纯追加：既有调用方（`score_taskset.py`、
    `pilot_search.py` 的冻结路径）继续走 :func:`render_prompt`，其输出不变。
    """
    variant = variant or DEFAULT_PROMPT_VARIANT
    if variant not in PROMPT_VARIANTS:
        raise ValueError(f"unknown prompt variant {variant!r}; want one of {PROMPT_VARIANTS}")
    if variant == "frozen":
        return render_prompt(tokenizer, item, enable_thinking=enable_thinking)
    msgs = [{"role": "user", "content": item["question"] + "\n\n" + INSTRUCTION}]
    return tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
    )


def render_span(tokenizer, item: dict, variant: str = DEFAULT_PROMPT_VARIANT,
                enable_thinking: bool = False):
    """渲染 + 切出 gold span：返回 ``(prompt, p_ids, g_ids, stable)``。

    `frozen` 与 `score_taskset.build_requests` 的路径逐字一致（含 ' ' + answer 的 gold
    文本）。`plain` 形态下 prompt 结尾不是 'Answer:'，`' ' + answer` 的首 token 可能与
    prompt 末尾的换行 token 合并 —— 此时 `answer_span` 返回 ``stable=False`` 与截断到公共
    前缀的 prompt。**调用方必须自己决定是否接受不稳定的条目**：冻结主轴仍按原样拒绝
    （`pilot_search` 里 `raise`），swap 臂按"公共前缀 + 其余 gold"计分并在记录里报数。
    """
    prompt = render_prompt_variant(tokenizer, item, variant, enable_thinking)
    p_ids, g_ids, stable = answer_span(tokenizer, prompt, GOLD_PREFIX + item["answer"])
    return prompt, p_ids, g_ids, stable


# ---------------------------------------------------------------- 题集自检

def self_check(items) -> dict:
    """题集内部一致性自检：答案可被自己的验证器通过、无重复题干。"""
    seen = {}
    dup = 0
    bad = 0
    per_stratum = {}
    for it in items:
        q = it["question"]
        if q in seen:
            dup += 1
        seen[q] = it["id"]
        ok, parsed, reason = verify(it, it["answer"])
        if not ok:
            bad += 1
        per_stratum[it["stratum"]] = per_stratum.get(it["stratum"], 0) + 1
    return {"n": len(items), "dup_questions": dup, "verifier_rejects_gold": bad,
            "per_stratum": dict(sorted(per_stratum.items()))}
