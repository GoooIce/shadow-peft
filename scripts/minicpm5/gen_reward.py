#!/usr/bin/env python3
"""Generation-based rollout scoring (the paper's fitness form).

Candidate fitness = mean reward over a batch of problems, where each reward
judges the FINAL ANSWER of a freely generated response (paper:
``reward_function(response, ...)`` over up-to-1024-token generations).
No teacher forcing, no first-token probing, no frozen ``Answer:`` suffix.

Reward for yes/no tasks (prime line):
    parse the generated text for the final yes/no decision;
    reward = 1.0 if the decision matches the item's truth, else 0.0;
    an unparsable response earns 0 (it failed to answer).

The parser is deliberately conservative: the LAST yes/no token in the
response wins (models that reason first answer last); thinking blocks are
ignored the same way.
"""
from __future__ import annotations

import re

YES_NO = re.compile(r"\b(yes|no)\b", re.IGNORECASE)
THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


def parse_yes_no(text: str) -> str | None:
    """Last yes/no mention outside <think> blocks, lower-cased; None if none."""
    cleaned = THINK.sub(" ", text)
    found = YES_NO.findall(cleaned)
    if not found:
        return None
    return found[-1].lower()


def reward_yes_no(text: str, truth: str) -> float:
    """1.0 iff the generated decision matches ``truth`` ('Yes'/'No')."""
    decision = parse_yes_no(text)
    if decision is None:
        return 0.0
    return 1.0 if decision == truth.strip().lower() else 0.0


def question_prompt(tokenizer, question: str) -> str:
    """Plain chat prompt: the bare question, no Answer suffix, no think forced."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False,
        add_generation_prompt=True,
    )


def score_rows_gen(outputs, items) -> list[dict]:
    """Attach rewards to rows: one output per item, order-aligned."""
    rows = []
    for output, item in zip(outputs, items):
        text = output.outputs[0].text
        rows.append({
            "item": item,
            "gen_text": text,
            "gen_n_tokens": len(output.outputs[0].token_ids),
            "gen_decision": parse_yes_no(text),
            "gen_reward": reward_yes_no(text, item["answer"]),
            "gen_correct": reward_yes_no(text, item["answer"]) > 0,
        })
    return rows
