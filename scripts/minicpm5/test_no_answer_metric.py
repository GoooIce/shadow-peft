"""19 票：未作答健康度 + 归因规则的单元测试（合成行，纯 CPU，不起任何引擎）。

覆盖：
  1. `score_taskset.is_no_answer` 的判定（含 GEN 臂回退、`</s>` 不算未作答）；
  2. `no_answer_summary` 的总体 / 逐 family / 逐 stratum / 逐 tier 聚合；
  3. `joint_decomposition` 的 2x2 + 未作答归因算术；
  4. `attribution_verdict` 的四种判定（artifact_dominated / flagged / survives / no_gain）；
  5. `pilot_search.Runner.q0_reference_measure`（桩引擎：patch/score 被替换）——写盘、
     恢复 live、以及 `no_answer_gate` 的打 flag 行为与 JSON 可序列化。

用法（gooo，CPU 即可）：
  cd ~/minicpm5-run300/scripts/minicpm5 && ~/venvs/minicpm5-vllm/bin/python test_no_answer_metric.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import score_taskset as st  # noqa: E402
import pilot_lib as pl  # noqa: E402
import pilot_search as ps  # noqa: E402


def row(i, fam, stratum, correct, first_argmax=None, gen_n=None, tier="easy"):
    r = {"id": f"{fam}-{stratum}-{i}", "family": fam, "stratum": stratum, "tier": tier,
         # `episode` 是 pl.fitness_summary 需要的字段（真实行由 Runner.score 里的
         # attach_episodes 挂上；这里按同 stratum 每题一个 episode 合成）
         "episode": f"{fam}-{tier}-ep{i:02d}",
         "tf_correct": correct, "tf_next_is_im_end": True, "tf_gold_logprob": -0.5}
    if first_argmax is not None:
        r["tf_first_argmax"] = first_argmax
    if gen_n is not None:
        r["gen_n_tokens"] = gen_n
    return r


def check(name, cond, detail=""):
    if not cond:
        raise AssertionError(f"{name}: {detail}")
    print(f"  ok  {name}")


# ---------------------------------------------------------------- 1. is_no_answer
def test_is_no_answer():
    check("bail token is <|im_end|>", st.NO_ANSWER_TOKEN == 130073, st.NO_ANSWER_TOKEN)
    check("argmax==130073 -> no answer", st.is_no_answer({"tf_first_argmax": 130073}) is True)
    check("argmax==digit -> answered", st.is_no_answer({"tf_first_argmax": 37}) is False)
    check("argmax==</s>(1) is NOT the bail token", st.is_no_answer({"tf_first_argmax": 1}) is False)
    check("GEN fallback (gen_n_tokens==1)", st.is_no_answer({"gen_n_tokens": 1}) is True)
    check("GEN fallback negative (3 tokens)", st.is_no_answer({"gen_n_tokens": 3}) is False)
    check("TF field wins over GEN", st.is_no_answer({"tf_first_argmax": 130073, "gen_n_tokens": 3}) is True)
    check("no marker -> False", st.is_no_answer({}) is False)


# ---------------------------------------------------------------- 2. no_answer_summary
def test_summary():
    rows = [
        row(0, "logic", "logic/hard", False, 130073, gen_n=1),
        row(1, "logic", "logic/hard", False, 130073, gen_n=1),
        row(2, "logic", "logic/hard", True, 242, gen_n=3),
        row(3, "logic", "logic/hard", False, 242, gen_n=3),
        row(4, "ordering", "ordering/hard", False, 130073, gen_n=1),
        row(5, "ordering", "ordering/hard", True, 41, gen_n=2),
        row(6, "arith", "arith/easy", True, 37, gen_n=3),
    ]
    na = st.no_answer_summary(rows)
    check("overall", na["overall"] == {"n": 7, "n_no_answer": 3, "rate": 3 / 7}, na["overall"])
    check("per_family logic", na["per_family"]["logic"] == {"n": 4, "n_no_answer": 2, "rate": 0.5})
    check("per_family ordering", na["per_family"]["ordering"]["rate"] == 0.5)
    check("per_family arith", na["per_family"]["arith"]["rate"] == 0.0)
    check("per_stratum", na["per_stratum"]["logic/hard"]["n_no_answer"] == 2)
    check("per_tier easy", na["per_tier"]["easy"] == {"n": 7, "n_no_answer": 3, "rate": 3 / 7})
    check("marker-less rows counted",
          st.no_answer_summary([{"id": "y", "family": "f", "stratum": "s", "tier": "t",
                                 "tf_correct": True}])["rows_without_any_marker"] == 1)
    check("json-serialisable", bool(json.dumps(na)))


# ---------------------------------------------------------------- 3. joint_decomposition
def test_joint():
    ref = [row(0, "logic", "logic/hard", True), row(1, "logic", "logic/hard", True),
           row(2, "logic", "logic/hard", False), row(3, "logic", "logic/hard", True)]
    cand = [row(0, "logic", "logic/hard", True, 242),      # a
            row(1, "logic", "logic/hard", False, 130073),  # c, by bail
            row(2, "logic", "logic/hard", True, 242),      # b
            row(3, "logic", "logic/hard", False, 242)]     # c, by wrong answer
    d = st.joint_decomposition(cand, ref)
    check("2x2 counts", (d["n_cand_right_ref_right"], d["n_cand_right_ref_wrong"],
                         d["n_cand_wrong_ref_right"]) == (1, 1, 2), d)
    check("gap = (b-c)/n", abs(d["gap"] - (1 - 2) / 4) < 1e-12)
    check("loss split: bail half", abs(d["cand_loss_from_no_answer"] - 0.25) < 1e-12, d)
    check("loss split: wrong half", abs(d["cand_loss_from_wrong"] - 0.25) < 1e-12, d)
    check("loss split adds up",
          abs(d["cand_loss_from_no_answer"] + d["cand_loss_from_wrong"] - 0.5) < 1e-12)
    check("no overlap -> n_common 0", st.joint_decomposition([row(99, "logic", "logic/hard", True)], ref)
          ["n_common"] == 0)


# ---------------------------------------------------------------- 4. attribution_verdict
def test_attribution():
    ref = [row(i, "logic", "logic/hard", True) for i in range(4)]
    # (i) same accuracy story, but the candidate bails -> excess flag, no gain
    cand = [row(0, "logic", "logic/hard", True, 242), row(1, "logic", "logic/hard", True, 242),
            row(2, "logic", "logic/hard", True, 242), row(3, "logic", "logic/hard", False, 130073)]
    v = st.attribution_verdict(cand, ref, margin=0.05)
    check("bail excess flagged", v["verdict"] == "flagged_no_answer_excess", v)
    check("excess value", abs(v["no_answer_excess"] - 0.25) < 1e-12)
    # (ii) apparent gain produced purely by the reference bailing
    ref3 = [row(i, "logic", "logic/hard", False, 130073) for i in range(4)]
    cand3 = [row(i, "logic", "logic/hard", True, 242) for i in range(4)]
    v = st.attribution_verdict(cand3, ref3)
    check("candidate-only bail -> artifact_dominated", v["verdict"] == "artifact_dominated", v)
    check("no both-answered items", v["n_both_answered"] == 0 and v["gain_both"] is None)
    check("flagged", bool(v["flagged"]))
    # (iii) honest gain survives on the both-answered subset
    ref4 = [row(0, "logic", "logic/hard", True, 242), row(1, "logic", "logic/hard", True, 242),
            row(2, "logic", "logic/hard", False, 242), row(3, "logic", "logic/hard", False, 242)]
    cand4 = [row(0, "logic", "logic/hard", True, 242), row(1, "logic", "logic/hard", True, 242),
             row(2, "logic", "logic/hard", True, 242), row(3, "logic", "logic/hard", False, 242)]
    v = st.attribution_verdict(cand4, ref4)
    check("honest gain survives", v["verdict"] == "gain_survives_both_answered", v)
    check("gain_all == gain_both == 0.25",
          abs(v["gain_all"] - 0.25) < 1e-12 and abs(v["gain_both"] - 0.25) < 1e-12)
    # (iv) honest loss, no bail movement
    cand5 = [row(i, "logic", "logic/hard", False, 242) for i in range(4)]
    v = st.attribution_verdict(cand5, ref4)
    check("loss -> no_gain", v["verdict"] == "no_gain", v)
    check("nothing flagged on a pure loss", v["flagged"] is False)
    # (v) empty overlap
    check("empty overlap",
          st.attribution_verdict([row(99, "logic", "logic/hard", True)], ref4)["verdict"] == "no_common_items")


# ---------------------------------------------------------------- 5. Runner glue (stubbed engine)
class StubRunner(ps.Runner):
    """绕开 __init__（不起引擎），只保留被测试的那几个方法。"""

    def __init__(self, rows_by_call, out_dir, margin=0.05, reference=None):
        self.args = type("A", (), {"no_answer_margin": margin, "taskset": "dev.jsonl",
                                   "chunk": 256})()
        self.out = out_dir
        (self.out / "records").mkdir(parents=True, exist_ok=True)
        self.dev_reqs = [{"item": {"id": r["id"]}} for r in rows_by_call[0]]
        self._rows = rows_by_call
        self._calls = 0
        self.q0_reference = reference
        self.patch_log = []

    def patch(self, live, target):                 # 桩：只改 live
        self.patch_log.append(dict(target))
        live.clear()
        live.update(target)
        return {"modules": 1, "refit_seconds": 0.0}

    def score(self, reqs, tag):                    # 桩：返回预置行
        rows = self._rows[self._calls]
        self._calls += 1
        return rows, {"wall_seconds": 0.0}


def test_runner_glue():
    with tempfile.TemporaryDirectory() as tmp:
        out = __import__("pathlib").Path(tmp)
        q0_rows = [row(0, "logic", "logic/hard", True, 242), row(1, "logic", "logic/hard", False, 242),
                   row(2, "logic", "logic/hard", True, 242), row(3, "logic", "logic/hard", True, 242)]
        centre_rows = [row(0, "logic", "logic/hard", True, 242), row(1, "logic", "logic/hard", False, 130073),
                       row(2, "logic", "logic/hard", True, 242), row(3, "logic", "logic/hard", True, 242)]
        r = StubRunner([q0_rows], out, margin=0.05)

        # (a) reference measurement: live must be restored, file written, rates right
        live = {("model.layers.0.mlp.down_proj", 5): 3}
        before = dict(live)
        payload = r.q0_reference_measure(live)
        check("live restored after the q0 measurement", live == before, live)
        check("q0 reference json written", (out / "records" / "q0_reference.json").exists())
        check("q0 reference rate", payload["no_answer"]["overall"]["rate"] == 0.0)
        check("two patches issued (to q0, back to centre)", len(r.patch_log) == 2, r.patch_log)

        # (b) attribution against the stored reference
        attr = r.attribution(centre_rows)
        check("attribution computed", attr is not None and attr["n_common"] == 4, attr)
        check("attribution sees a 25% bail excess", abs(attr["no_answer_excess"] - 0.25) < 1e-12, attr)
        check("verdict = flagged_no_answer_excess", attr["verdict"] == "flagged_no_answer_excess", attr)

        # (c) gate: centre flagged, a clean candidate not flagged
        clean = [row(i, "logic", "logic/hard", True, 242) for i in range(4)]
        cand_records = [{"index": 0, "fitness": {"acc": 1.0}, "no_answer": r.no_answer(clean),
                         "attribution": r.attribution(clean)}]
        gate = r.no_answer_gate(r.no_answer(centre_rows), attr, cand_records)
        check("gate json-serialisable", bool(json.dumps(gate)))
        check("centre flagged for excess", "centre" in gate["flagged_arms"], gate["flagged"])
        check("clean candidate not flagged", "best_candidate" not in gate["flagged_arms"])
        check("q0 reference embedded", gate["q0_reference"]["overall"]["rate"] == 0.0)
        check("candidate spread reported", gate["candidates_no_answer"]["max"] == 0.0)

        # (d) with no reference, the gate degrades gracefully (no attribution, no crash)
        r2 = StubRunner([q0_rows], out / "nor", reference=None)
        (out / "nor").mkdir(parents=True, exist_ok=True)
        gate2 = r2.no_answer_gate(r2.no_answer(centre_rows), r2.attribution(centre_rows), [])
        check("no reference -> no flags, no crash", gate2["flagged"] == []
              and gate2["q0_reference"] is None and gate2["centre"]["attribution"] is None)


def main():
    print("== 1. is_no_answer");            test_is_no_answer()
    print("== 2. no_answer_summary");       test_summary()
    print("== 3. joint_decomposition");     test_joint()
    print("== 4. attribution_verdict");     test_attribution()
    print("== 5. Runner glue (stub engine)"); test_runner_glue()
    print("ALL TESTS PASS")


if __name__ == "__main__":
    main()
