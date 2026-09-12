#!/usr/bin/env python3
"""Prime-judgement taskset generator (fitness + held-out test).

Question form: "Is N a prime number?" -> "Yes"/"No"  (first-token scorable).
Strata: small (<100), medium (100-999), large (1000-9999).  Within each
stratum, half true primes and half *tricky* composites (semiprimes of two
mid-size primes, squares of primes, small-prime multiples with non-obvious
digits, Grothendieck-style traps) so the prior gives nothing away.

Usage:
  gen:    python gen_prime_pool.py gen --out candidates.jsonl [--n 2400]
  select: python gen_prime_pool.py select --candidates candidates.jsonl \
              --scores scores_q0.jsonl --fitness taskset_prime_fitness.jsonl \
              --test taskset_prime_test.jsonl [--fitness-n 480] [--test-n 240]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


def primes_between(lo: int, hi: int) -> list[int]:
    return [n for n in range(lo, hi) if is_prime(n)]


TRAPS = [57, 87, 91, 119, 121, 123, 143, 169, 187, 209, 221, 247, 289,
         299, 323, 361, 391, 437, 529, 667, 841, 899, 1147, 1273, 1369,
         1681, 1849, 2209, 3481, 4489, 5041, 6241, 7921, 9409]


def gen(out: Path, n: int) -> None:
    rng = random.Random(20260912)
    strata = {"small": (11, 100), "medium": (101, 1000), "large": (1001, 10000)}
    items = []
    for stratum, (lo, hi) in strata.items():
        primes = primes_between(lo, hi)
        rng.shuffle(primes)
        # tricky composites: semiprimes of nearby primes, prime squares,
        # small-prime multiples; exclude perfect trivial cases (even, ÷5)
        composites: list[int] = [t for t in TRAPS if lo <= t < hi]
        plist = [p for p in primes_between(7, 313)]
        while len(composites) < n:
            p = rng.choice(plist)
            q = rng.choice(plist)
            m = p * q
            if lo <= m < hi and m % 2 and m % 5:
                composites.append(m)
        rng.shuffle(composites)
        half = n // (2 * len(strata))
        for n_val, truth in [(v, True) for v in primes[:half]] + \
                            [(v, False) for v in composites[:half]]:
            q = f"Is {n_val} a prime number?"
            digest = hashlib.sha256(f"prime-{n_val}".encode()).hexdigest()[:12]
            items.append({
                "id": f"prime-{stratum}-{n_val}-{digest}",
                "question": q,
                "answer": "Yes" if truth else "No",
                "family": "prime",
                "stratum": f"prime/{stratum}",
                "tier": stratum,
                "episode": "",            # assigned at selection
                "meta": {"n": n_val, "is_prime": truth},
                "verify": {"kind": "prime", "n": n_val},
                "seed": rng.getrandbits(63),
                "source": "prime-gen-20260912",
                "pool": "candidate",
            })
    rng.shuffle(items)
    with out.open("w") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"candidates: {len(items)} -> {out}")


def select(candidates: Path, scores: Path, fitness: Path, test: Path,
           fitness_n: int, test_n: int) -> None:
    """Fitness: episodes of 8 with exactly 4 q0-correct / 4 q0-wrong per
    stratum (the pool the search optimises; balanced at q0).  Test: FRESH
    questions not used in fitness, q0-unbalanced is fine (paired before/
    after is the statistic)."""
    rows = {json.loads(r)["id"]: json.loads(r)
            for r in scores.open() if r.strip()}
    cand = [json.loads(l) for l in candidates.open()]
    scored = []
    for item in cand:
        row = rows.get(item["id"])
        if row is None:
            continue
        item = dict(item)
        item["tf_correct"] = bool(row.get("tf_correct"))
        scored.append(item)
    by_stratum: dict[str, list[dict]] = {}
    for item in scored:
        by_stratum.setdefault(item["tier"], []).append(item)
    rng = random.Random(20260913)

    # fitness: 4/4 episodes per stratum, 60 episodes total
    per_stratum_eps = max(1, round(60 / len(by_stratum)))
    fitness_items = []
    ep_counter = {s: 0 for s in by_stratum}
    for stratum, items in sorted(by_stratum.items()):
        correct = [i for i in items if i["tf_correct"]]
        wrong = [i for i in items if not i["tf_correct"]]
        rng.shuffle(correct)
        rng.shuffle(wrong)
        n_eps = min(per_stratum_eps, len(correct) // 4, len(wrong) // 4)
        for ep in range(n_eps):
            episode = f"prime-{stratum}-ep{ep:02d}"
            for item in correct[ep * 4:(ep + 1) * 4] + wrong[ep * 4:(ep + 1) * 4]:
                item = dict(item)
                item["episode"] = episode
                item["pool"] = "fitness"
                fitness_items.append(item)
            ep_counter[stratum] += 1
    # top up to fitness_n with whole episodes if short
    with fitness.open("w") as handle:
        for item in fitness_items[:fitness_n]:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    # test: fresh items not in fitness ids
    chosen_ids = {i["id"] for i in fitness_items}
    rest = [i for i in scored if i["id"] not in chosen_ids]
    rng.shuffle(rest)
    with test.open("w") as handle:
        for item in rest[:test_n]:
            item = dict(item)
            item["episode"] = f"prime-test-{item['tier']}"
            item["pool"] = "test"
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    n_ep = len({i["episode"] for i in fitness_items})
    print(f"fitness: {min(len(fitness_items), fitness_n)} items / {n_ep} episodes; "
          f"test: {min(len(rest), test_n)}")


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--out", required=True)
    g.add_argument("--n", type=int, default=2400)
    s = sub.add_parser("select")
    s.add_argument("--candidates", required=True)
    s.add_argument("--scores", required=True)
    s.add_argument("--fitness", required=True)
    s.add_argument("--test", required=True)
    s.add_argument("--fitness-n", type=int, default=480)
    s.add_argument("--test-n", type=int, default=240)
    args = parser.parse_args()
    if args.cmd == "gen":
        gen(Path(args.out).expanduser(), args.n)
    else:
        select(Path(args.candidates).expanduser(), Path(args.scores).expanduser(),
               Path(args.fitness).expanduser(), Path(args.test).expanduser(),
               args.fitness_n, args.test_n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
