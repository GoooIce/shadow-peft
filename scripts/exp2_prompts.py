"""Exp2 seed prompt pool (~640, combinatorial, deterministic).

Domains x topics x task patterns — the pool drives teacher-data generation
(chat-seeded sampling) and τ evaluation prompts. No randomness: same pool
everywhere, spec'd in run-conventions.md (shared asset).

Run: python scripts/exp2_prompts.py  ->  runs/exp2/shared/prompts.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DOMAINS = {
    "cooking": ["sourdough bread", "knife skills", "Italian pasta", "food safety",
                "grilling vegetables", "making stock", "spice pairing", "meal prep"],
    "travel": ["budget airlines", "jet lag", "packing light", "train travel in Europe",
               "travel insurance", "hostels vs hotels", "visa rules", "mountain hiking"],
    "programming": ["recursion", "version control", "unit testing", "SQL joins",
                    "hash tables", "REST APIs", "code review", "concurrency"],
    "math": ["prime numbers", "matrix multiplication", "probability", "derivatives",
             "modulo arithmetic", "averages", "geometry proofs", "binary numbers"],
    "history": ["the Roman Republic", "the industrial revolution", "the silk road",
                "ancient Egypt", "the space race", "the printing press",
                "the Cold War", "medieval trade routes"],
    "health": ["sleep hygiene", "hydration", "stretching", "balanced diet",
               "walking", "stress management", "posture", "sunscreen"],
    "business": ["pricing", "customer feedback", "cash flow", "marketing basics",
                 "hiring", "product-market fit", "negotiation", "supply chains"],
    "science": ["photosynthesis", "plate tectonics", "vaccines", "the water cycle",
                "electric circuits", "DNA", "states of matter", "the scientific method"],
    "writing": ["clear sentences", "story structure", "editing", "outlining",
                "metaphors", "pacing", "dialogue", "research notes"],
    "daily-life": ["apartment cleaning", "personal budgeting", "bicycle maintenance",
                   "laundry care", "time management", "houseplants", "gift choosing",
                   "commuting options"],
}

PATTERNS = [
    "Explain {t}.",
    "Write a short paragraph about {t}.",
    "List three key facts about {t}.",
    "What is the most common mistake people make with {t}?",
    "Summarize {t} in two sentences.",
    "Give a concrete example that illustrates {t}.",
    "Why does {t} matter in practice?",
    "Describe {t} as if to a complete beginner.",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/exp2/shared/prompts.jsonl")
    args = ap.parse_args()

    prompts = []
    for domain, topics in DOMAINS.items():
        for topic in topics:
            for pat in PATTERNS:
                prompts.append({"domain": domain, "prompt": pat.format(t=topic)})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for p in prompts:
            f.write(json.dumps(p) + "\n")
    print(f"{len(prompts)} prompts -> {out}")


if __name__ == "__main__":
    main()
