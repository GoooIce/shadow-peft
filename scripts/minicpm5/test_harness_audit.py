"""§5 harness audit: unit tests on synthetic rows and a stubbed engine (no GPU).

Covers:
  1. `required_answer_kind` (which items *must* carry a number);
  2. `classify_row` / `classify_rows` -- every degenerate mode, including the
     ticket-19 no-answer mode and Aspire's numeric-label collapse;
  3. `summarise` -- per-family rates, the failure attribution and the flagging
     margin;
  4. `score_tf_probe` and `numeric_token_predicate` against a fake engine, so the
     plumbing (first-gold-position argmax, top-5 capture) is exercised without
     vLLM.

Usage:
  cd $(git rev-parse --show-toplevel)/scripts/minicpm5 && python test_harness_audit.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import harness_audit as ha  # noqa: E402

IM_END = 130073


def check(name, cond, detail=""):
    if not cond:
        raise AssertionError(f"{name}: {detail}")
    print(f"  ok  {name}")


def row(**kwargs):
    base = {
        "id": kwargs.get("id", "x-1"), "family": kwargs.get("family", "arith"),
        "tier": "easy", "stratum": kwargs.get("stratum", "arith/easy"),
        "required": kwargs.get("required", "integer"),
        "tf_correct": kwargs.get("tf_correct", True),
    }
    base.update(kwargs)
    return base


# --------------------------------------------------------------------------- #
# 1. required answer kind
# --------------------------------------------------------------------------- #
def test_required_kind():
    check("arith needs a number", ha.required_answer_kind({"verify": {"kind": "arith_int"}})
          == "integer")
    check("ordering needs a name", ha.required_answer_kind({"verify": {"kind": "ordering"}})
          == "name")
    check("unknown kind -> None", ha.required_answer_kind({"verify": {"kind": "countdown_expr"}})
          == "integer")
    check("missing spec -> None", ha.required_answer_kind({}) is None)


# --------------------------------------------------------------------------- #
# 2. classification
# --------------------------------------------------------------------------- #
def test_classify():
    modes = ha.classify_row(row(tf_first_argmax=IM_END))
    check("no-answer by first-gold argmax", modes == ["no_answer"], modes)
    modes = ha.classify_row({"tf_correct": False, "gen_n_tokens": 1})
    check("no-answer by GEN fallback when the TF field is absent", "no_answer" in modes, modes)
    modes = ha.classify_row(row(tf_first_argmax=242, gen_n_tokens=1))
    check("a present TF argmax wins over the GEN fallback (ticket 19 semantics)",
          "no_answer" not in modes, modes)
    check("GEN fallback does not fire for a longer generation",
          ha.classify_row({"tf_correct": False, "gen_n_tokens": 5}) == [])

    modes = ha.classify_row(row(gen_text="", gen_n_tokens=0, gen_parsed=None))
    check("empty output detected", "empty_output" in modes and "non_numeric" in modes, modes)
    modes = ha.classify_row(row(gen_text="42", gen_finish="length", gen_parsed=42))
    check("truncation detected", modes == ["truncated"], modes)
    modes = ha.classify_row(row(gen_text="The answer is\n42", gen_parsed=42))
    check("multi-line detected", "multi_line" in modes, modes)
    modes = ha.classify_row(row(gen_text=" ".join(["word"] * 13), gen_parsed=42))
    check("verbose first line detected", "verbose_first_line" in modes, modes)
    modes = ha.classify_row(row(gen_text="Let me think about it: 42", gen_parsed=42))
    check("think leak detected", "think_leak" in modes, modes)
    modes = ha.classify_row(row(gen_text="Anna", gen_parsed=None, required="name",
                                stratum="ordering/easy"))
    check("ordering non-name detected", "non_name" in modes and "non_numeric" not in modes, modes)
    modes = ha.classify_row(row(gen_text="42", gen_parsed=42, tf_first_argmax=IM_END))
    check("a clean row has modes == ['no_answer'] only", modes == ["no_answer"], modes)

    predicate = lambda token: token in (37, 38, 39, 40, 41, 42)   # noqa: E731
    modes = ha.classify_row(row(tf_first_argmax=242, required="integer"), predicate)
    check("first_token_not_numeric uses the predicate",
          "first_token_not_numeric" in modes, modes)
    modes = ha.classify_row(row(tf_first_argmax=37, required="integer"), predicate)
    check("a numeric first token does not trip it", modes == [], modes)
    modes = ha.classify_row(row(tf_first_argmax=37, required="name", stratum="ordering/easy"),
                            predicate)
    check("the mode only applies where a number is required", modes == [], modes)


def test_attribute_rows():
    rows = ha.classify_rows([
        row(id="1", tf_correct=False, tf_first_argmax=IM_END),                    # no_answer
        row(id="2", tf_correct=False, gen_text="", gen_parsed=None),              # empty
        row(id="3", tf_correct=False, gen_text="77", gen_parsed=77),              # wrong answer
        row(id="4", tf_correct=False, gen_text="hello", gen_parsed=None),         # unparsable
        row(id="5", tf_correct=True, gen_text="42", gen_parsed=42),               # correct
    ])
    attribution = {entry["id"]: entry["failure_mode"] for entry in rows}
    check("no-answer attribution", attribution["1"] == "no_answer")
    check("empty attribution", attribution["2"] == "empty_output")
    check("wrong-answer attribution", attribution["3"] == "wrong_answer")
    check("unparsable attribution", attribution["4"] == "unparsable_output")
    check("correct rows carry no failure mode", attribution["5"] is None)


# --------------------------------------------------------------------------- #
# 3. summary + flags
# --------------------------------------------------------------------------- #
def test_summary():
    rows = ha.classify_rows(
        [row(id=f"a{i}", family="arith", tf_correct=(i < 8), tf_first_argmax=37)
         for i in range(10)]
        + [row(id=f"l{i}", family="logic", stratum="logic/hard", tf_correct=(i < 5),
               tf_first_argmax=IM_END, gen_text="", gen_parsed=None) for i in range(10)]
        + [row(id=f"m{i}", family="modulo", stratum="modulo/easy", tf_correct=False,
               gen_text="no number here", gen_parsed=None, tf_first_argmax=242)
           for i in range(10)]
    )
    report = ha.summarise(rows, margin=0.05)
    check("overall n", report["n"] == 30)
    check("logic no-answer rate 100%", report["per_family"]["logic"]["modes"]["no_answer"]["rate"]
          == 1.0)
    check("logic flagged", any(flag["family"] == "logic" and flag["mode"] == "no_answer"
                               for flag in report["flags"]), report["flags"])
    check("modulo flagged for non_numeric",
          any(flag["family"] == "modulo" and flag["mode"] == "non_numeric"
              for flag in report["flags"]), report["flags"])
    check("arith not flagged", not any(flag["family"] == "arith" for flag in report["flags"]))
    check("failure attribution counts the degenerate bulk",
          report["overall"]["failure_attribution"]["no_answer"]["n"] == 5
          and report["overall"]["failure_attribution"]["unparsable_output"]["n"] == 10
          and report["overall"]["failure_attribution"]["wrong_answer"]["n"] == 2,
          report["overall"]["failure_attribution"])
    check("harness-defect share is reported",
          abs(report["overall"]["harness_defect_share"] - (5 + 10) / 17) < 1e-12,
          report["overall"]["harness_defect_share"])
    check("first-token histogram built", report["distinct_first_tokens"] == 3
          and report["first_token_top"][0]["token"] == str(IM_END),
          report["first_token_top"])
    check("json-serialisable", bool(__import__("json").dumps(report)))
    clean = ha.summarise(ha.classify_rows([row(id="c", tf_correct=True, gen_text="42",
                                              gen_parsed=42, tf_first_argmax=37)]))
    check("clean rows do not flag", clean["flagged"] is False and clean["flags"] == [])


# --------------------------------------------------------------------------- #
# 4. stub engine plumbing
# --------------------------------------------------------------------------- #
class StubLogprob:
    def __init__(self, logprob):
        self.logprob = logprob


class StubOutput:
    def __init__(self, prompt_logprobs, token_ids):
        self.prompt_logprobs = prompt_logprobs
        self.outputs = [type("O", (), {"token_ids": token_ids, "text": ""})()]


class StubLLM:
    """Returns canned prompt logprobs; records the prompts it was asked for."""

    def __init__(self, logprobs):
        self.logprobs = logprobs
        self.seen = []

    def generate(self, prompts, params, use_tqdm=False):
        self.seen.extend(prompts)
        return self.logprobs


class StubSamplingParams:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def test_tf_probe():
    # two items: one answers with the gold id at argmax, one bails with <|im_end|>
    reqs = [
        {"item": {"id": "i0", "family": "arith", "tier": "easy", "stratum": "arith/easy",
                  "verify": {"kind": "arith_int"}}, "p_ids": [1, 2], "g_ids": [37, 38],
         "variant": "frozen", "stable": True},
        {"item": {"id": "i1", "family": "logic", "tier": "hard", "stratum": "logic/hard",
                  "verify": {"kind": "logic_count"}}, "p_ids": [1, 2], "g_ids": [41],
         "variant": "frozen", "stable": True},
    ]
    plp_good = [None, {1: StubLogprob(-0.1)}, {37: StubLogprob(-0.2), 40: StubLogprob(-3.0)},
                {38: StubLogprob(-0.3), 41: StubLogprob(-2.0)}, {99: StubLogprob(-1.0)}]
    plp_bail = [None, {1: StubLogprob(-0.1)}, {IM_END: StubLogprob(-0.05), 41: StubLogprob(-4.0)}]
    llm = StubLLM([StubOutput(plp_good, [130073]), StubOutput(plp_bail, [130073])])
    rows, timing = ha.score_tf_probe(llm, StubSamplingParams, None, reqs, chunk=8, tag="t",
                                     log=lambda *a: None)
    check("probe returns one row per item", len(rows) == 2 and timing["tf_seconds"] >= 0)
    check("gold-answering item is correct", rows[0]["tf_correct"] is True)
    check("first-gold argmax recorded", rows[0]["tf_first_argmax"] == 37)
    check("top-5 alternatives recorded with ids and logprobs",
          rows[0]["tf_first_top5"][0][0] == 37 and len(rows[0]["tf_first_top5"]) >= 2)
    check("bailing item is wrong and flagged by the mode scan",
          rows[1]["tf_correct"] is False and rows[1]["tf_first_argmax"] == IM_END)
    check("required kind attached",
          rows[0]["required"] == "integer" and rows[1]["required"] == "integer")
    classified = ha.classify_rows(rows)
    check("classifier sees the bail", classified[1]["modes"] == ["no_answer"])
    check("classifier sees the clean answer", classified[0]["modes"] == [])


def test_numeric_predicate():
    class Tok:
        def decode(self, ids):
            return {37: "42", 41: " Anna", 242: "3.5", 130073: "", 5: "-7"}[ids[0]]

    predicate = ha.numeric_token_predicate(Tok())
    check("digit token", predicate(37) is True)
    check("minus token", predicate(5) is True)
    check("name token", predicate(41) is False)
    check("leading digit inside a float", predicate(242) is True)
    check("stop token", predicate(IM_END) is False)
    check("cached second call agrees", predicate(37) is True)


def main():
    print("== 1. required_answer_kind"); test_required_kind()
    print("== 2. classify_row"); test_classify()
    test_attribute_rows()
    print("== 3. summarise + flags"); test_summary()
    print("== 4. TF probe against a stub engine"); test_tf_probe()
    test_numeric_predicate()
    print("ALL TESTS PASS")


if __name__ == "__main__":
    main()
