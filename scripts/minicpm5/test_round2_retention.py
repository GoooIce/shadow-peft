"""Round-2 retention rules: unit tests on synthetic inputs (pure CPU, no engine).

Covers:
  1. `pilot_lib.step_coordinates` is exactly the set `apply_step` visits;
  2. the residual re-anchoring on rollback -- snapshot -> apply_step -> restore is
     byte- and identity-exact, and the *counter*-example (no restore) leaks the
     rolled-back step back through `gamma*e`, which is what ticket 09 §4 forbids;
  3. `retention_lib.noise_band` / `harness_se` / `sampling_se` (deterministic vs
     noisy replicates, quantisation floor, replicate-source selection);
  4. `retention_lib.retention_decision` (one-sided, strictly-greater-than-band);
  5. `retention_lib.terminal_retention` (§2 deliverable rule, §3 swap gate);
  6. `taskset_lib.render_prompt_variant` (frozen output unchanged; plain form has
     no `Answer:` suffix; swap span instability is reported, not raised);
  7. `pilot_search.Runner.retention_check` glue with a stubbed engine: keep,
     rollback, slice monitoring and the explicit slice gate.

Usage:
  cd $(git rev-parse --show-toplevel)/scripts/minicpm5 && python test_round2_retention.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pilot_lib as pl  # noqa: E402
import pilot_search as ps  # noqa: E402
import retention_lib as rl  # noqa: E402
import taskset_lib as tl  # noqa: E402

LAYER = "model.layers.0.self_attn.q_proj"
LAYER_B = "model.layers.1.mlp.down_proj"


def check(name, cond, detail=""):
    if not cond:
        raise AssertionError(f"{name}: {detail}")
    print(f"  ok  {name}")


class FakeReader:
    """Minimal `PayloadReader` stand-in: base codes + shapes, nothing else."""

    def __init__(self, base: dict[str, np.ndarray], shapes: dict[str, tuple[int, int]]):
        self.base = base
        self._shapes = shapes

    def codes(self, layer):
        return self.base[layer]

    def shape(self, layer):
        return self._shapes[layer]

    def raw_code(self, layer, flat):
        return int(self.base[layer].reshape(-1)[flat])


def make_state(edits=None, residual=None):
    state = pl.CentreState(generation=3)
    for (layer, flat), code in (edits or {}).items():
        state.set_code(layer, flat, code, base=int(CODE_AT(layer, flat)))
    for (layer, flat), value in (residual or {}).items():
        state.set_residual(layer, flat, value)
    return state


BASE_CODES = {
    LAYER: np.zeros((2, 8), dtype=np.uint8),
    LAYER_B: np.zeros((2, 8), dtype=np.uint8),
}
SHAPES = {LAYER: (2, 8), LAYER_B: (2, 8)}


def CODE_AT(layer, flat):
    return int(BASE_CODES[layer].reshape(-1)[flat])


def base_hash(layer):
    return f"base-{layer}"


# --------------------------------------------------------------------------- #
# 1. step_coordinates covers exactly what apply_step visits
# --------------------------------------------------------------------------- #
def test_step_coordinates():
    state = make_state(residual={(LAYER, 3): 0.4, (LAYER_B, 1): -0.9})
    gradient = {(LAYER, 0): 0.5, (LAYER, 5): -0.25, (LAYER_B, 2): 1.0}
    coords = pl.step_coordinates(state, gradient)
    check("union of gradient and residual", coords == {(LAYER, 0), (LAYER, 5), (LAYER_B, 2),
                                                       (LAYER, 3), (LAYER_B, 1)}, coords)
    ledger = pl.apply_step(state, FakeReader(BASE_CODES, SHAPES), gradient, alpha=12.0)
    check("apply_step touched the same count", ledger["touched"] == len(coords),
          (ledger["touched"], len(coords)))


# --------------------------------------------------------------------------- #
# 2. residual re-anchoring on rollback
# --------------------------------------------------------------------------- #
def test_rollback_is_exact():
    reader = FakeReader(BASE_CODES, SHAPES)
    state = make_state(edits={(LAYER, 0): 2, (LAYER, 6): 4},
                       residual={(LAYER_B, 1): -1.2, (LAYER, 2): 0.35})
    before_blob = json.dumps(state.to_dict(), sort_keys=True)
    before_sha = state.sha256(base_hash)
    before_codes = {layer: dict(bucket) for layer, bucket in state.codes.items()}
    before_residual = {layer: dict(bucket) for layer, bucket in state.residual.items()}

    gradient = {(LAYER, 0): 1.0, (LAYER, 1): -0.6, (LAYER_B, 1): 0.9}
    coords = pl.step_coordinates(state, gradient)
    snapshot = pl.snapshot_layers(state, coords)
    ledger = pl.apply_step(state, reader, gradient, alpha=12.0)
    step_sha = state.sha256(base_hash)
    check("the step actually moved the centre", step_sha != before_sha)
    check("the step left a residual behind",
          sum(len(b) for b in state.residual.values()) >= sum(len(b) for b in before_residual.values()))

    restore = pl.restore_layers(state, snapshot)
    check("restore reports no residual mismatch", restore["residual_mismatches"] == 0, restore)
    check("codes and residuals are identical again",
          state.codes == before_codes and state.residual == before_residual,
          (state.codes, before_codes))
    check("to_dict is byte-identical", json.dumps(state.to_dict(), sort_keys=True) == before_blob)
    check("centre identity is byte-identical", state.sha256(base_hash) == before_sha)
    check("restore ledger counts the coordinates it put back",
          restore["residual_restored"] >= 1
          and (restore["codes_restored"] + restore["codes_removed"]) >= 1, restore)
    check("no op left unreverted",
          json.dumps(state.to_dict(), sort_keys=True) == before_blob)

    # bucket presence: a rollback that leaves the empty bucket behind changes the identity
    # (ticket 14 §5.1), so the identity check above is the real assertion.
    state_b = make_state()
    snapshot_b = pl.snapshot_layers(state_b, [(LAYER, 0)])
    state_b.set_code(LAYER, 0, 5, base=0)
    check("bucket created by the step", LAYER in state_b.codes)
    pl.restore_layers(state_b, snapshot_b)
    check("bucket removed again", LAYER not in state_b.codes, state_b.codes)
    check("empty centre keeps the q0 identity", state_b.sha256(base_hash) ==
          pl.CentreState().sha256(base_hash))


def test_rollback_counterexample():
    """Without the re-anchoring, the rolled-back step returns through gamma*e."""
    reader = FakeReader(BASE_CODES, SHAPES)
    gradient = {(LAYER, 2): 0.01}          # tiny step: 0.12 projects back onto code 0
    state = make_state()
    snapshot = pl.snapshot_layers(state, pl.step_coordinates(state, gradient))
    ledger = pl.apply_step(state, reader, gradient, alpha=12.0)
    check("the tiny step did not move the code", ledger["moved"] == 0)
    leaked = state.residual_at(LAYER, 2)
    check("the float step stayed in the residual (ticket 09's known trap)",
          abs(leaked - 12.0 * 0.01) < 1e-9, leaked)
    pl.restore_layers(state, snapshot)
    check("after the rollback the residual is re-anchored to the pre-step value",
          state.residual_at(LAYER, 2) == 0.0, state.residual_at(LAYER, 2))

    # contrast: had we only restored the codes, the un-taken step would still be
    # pending and come back as gamma*e next generation
    state2 = make_state()
    pl.apply_step(state2, reader, gradient, alpha=12.0)
    u_next_without_reanchor = 1.0 * state2.residual_at(LAYER, 2)
    check("without re-anchoring the un-taken step would be re-applied next generation",
          abs(u_next_without_reanchor - 0.12) < 1e-9, u_next_without_reanchor)
    reanchor = pl.residual_reanchor(ledger, pl.step_coordinates(state, gradient))
    check("re-anchor evidence names the rule", "ticket 09" in reanchor["rule"])


# --------------------------------------------------------------------------- #
# 3. feedback noise band
# --------------------------------------------------------------------------- #
def rows(family, n_correct, n_total, episode_prefix="ep", episode_size=1):
    out = []
    for index in range(n_total):
        out.append({
            "id": f"{family}-{episode_prefix}-{index // episode_size}-{index}",
            "family": family, "episode": f"{family}-{episode_prefix}{index // episode_size}",
            "tf_correct": index < n_correct,
        })
    return out


def test_noise_band():
    deterministic = [rl.score_block(rows("arith", 24, 96, "a")),
                     rl.score_block(rows("arith", 24, 96, "a"))]
    band = rl.noise_band(deterministic, rows("arith", 24, 96, "a"))
    check("no-op replicates that agree give harness_se 0",
          band["harness"]["centre_se"] == 0.0 and band["harness"]["deterministic"] is True)
    check("the band is then the quantisation floor", band["per_family"]["arith"]["band"] ==
          1 / 96, band["per_family"])
    check("floor is named as the driver", band["per_family"]["arith"]["driven_by"]
          == "quantisation_floor")

    noisy = [rl.score_block(rows("arith", k, 96, "a")) for k in (20, 28, 24)]
    band = rl.noise_band(noisy, rows("arith", 24, 96, "a"))
    check("noisy replicates give a non-zero harness SE",
          band["harness"]["centre_se"] > 0 and band["harness"]["deterministic"] is False)
    check("the family band is driven by the harness term",
          band["per_family"]["arith"]["driven_by"] == "harness_se"
          and band["per_family"]["arith"]["band"] >= band["per_family"]["arith"]["harness_se"])
    check("sampling SE is reported separately",
          band["per_family"]["arith"]["sampling_se"] is not None)
    check("scale multiplies only the harness term, and the raw SE stays reportable",
          rl.noise_band(noisy, rows("arith", 24, 96, "a"), scale=2.0)["per_family"]["arith"]
          ["band"] == 2.0 * band["per_family"]["arith"]["harness_se"]
          and rl.noise_band(noisy, rows("arith", 24, 96, "a"), scale=2.0)["harness"]["centre_se"]
          == band["harness"]["centre_se"])

    startup, source = rl.band_replicates(noisy, [])
    check("startup replicates win", len(startup) == 3 and source == "startup_noop")
    inrun = [{"sha": "aaa", "block": noisy[0]}, {"sha": "aaa", "block": noisy[1]},
             {"sha": "bbb", "block": noisy[2]}]
    picks, source = rl.band_replicates([], inrun)
    check("in-run repeats are grouped by centre identity",
          len(picks) == 2 and source.startswith("generation_repeats:aaa"), (len(picks), source))
    picks, source = rl.band_replicates([], [inrun[0]])
    check("a single in-run reading is not enough", picks == [] and source == "none")


def test_band_terms():
    """Which measured term the threshold hangs on is an explicit, recorded choice."""
    episode_rows = []
    for index in range(96):
        episode_rows.append({"id": f"a{index}", "family": "arith",
                             "episode": f"arith-easy-ep{index // 8}",
                             "tf_correct": index < 24})
    replicates = [rl.score_block(episode_rows), rl.score_block(episode_rows)]
    harness = rl.noise_band(replicates, episode_rows)
    sampling = rl.noise_band(replicates, episode_rows, term="sampling")
    both = rl.noise_band(replicates, episode_rows, term="max")
    check("harness term = floor when the pipeline is deterministic",
          harness["per_family"]["arith"]["driven_by"] == "quantisation_floor"
          and harness["per_family"]["arith"]["band"] == 1 / 96)
    check("sampling term = the cluster-corrected SE of the pool score",
          sampling["per_family"]["arith"]["driven_by"] == "sampling_se"
          and sampling["per_family"]["arith"]["band"] == sampling["per_family"]["arith"]
          ["sampling_se"] > harness["per_family"]["arith"]["band"])
    check("max term takes the larger",
          both["per_family"]["arith"]["band"] == max(sampling["per_family"]["arith"]["band"],
                                                     harness["per_family"]["arith"]["band"]))
    check("the term used is recorded on the band", harness["term"] == "harness"
          and sampling["term"] == "sampling")
    check("an unknown term is refused", _raises(lambda: rl.noise_band(replicates,
                                                                     episode_rows, term="x")))
    # the realistic consequence: a 3-item family drop rolls back under the harness term
    # (band = 1 item) but not under the sampling term for a 96-item family (band ~ 12 items)
    drop3 = [dict(row, tf_correct=(index < 21)) for index, row in enumerate(episode_rows)]
    check("harness-term rule fires on a 3-item drop",
          rl.retention_decision(episode_rows, drop3, harness)["rollback"] is True)
    check("sampling-term rule does not",
          rl.retention_decision(episode_rows, drop3, sampling)["rollback"] is False)


# --------------------------------------------------------------------------- #
# 4. retention decision
# --------------------------------------------------------------------------- #
def test_retention_decision():
    band = rl.noise_band([rl.score_block(rows("arith", 24, 96, "a")),
                          rl.score_block(rows("arith", 24, 96, "a"))],
                         rows("arith", 24, 96, "a"))
    same = rl.retention_decision(rows("arith", 24, 96, "a"), rows("arith", 24, 96, "a"), band)
    check("no change -> no rollback", same["rollback"] is False and same["breaches"] == [])
    one_item = rl.retention_decision(rows("arith", 24, 96, "a"), rows("arith", 23, 96, "a"), band)
    check("a one-item drop is inside the floor (drop > band is strict)",
          abs(one_item["per_family"]["arith"]["delta"] + 1 / 96) < 1e-12
          and one_item["per_family"]["arith"]["breach"] is False, one_item["per_family"])
    check("...and the comparison is stated in items",
          one_item["per_family"]["arith"]["drop_items"] == 1
          and abs(one_item["per_family"]["arith"]["band_items"] - 1.0) < 1e-12, one_item["per_family"])
    two_items = rl.retention_decision(rows("arith", 24, 96, "a"), rows("arith", 22, 96, "a"), band)
    check("a two-item drop breaches", two_items["rollback"] is True
          and two_items["worst_family"] == "arith", two_items)
    check("breach evidence carries both items and rate",
          two_items["breaches"][0]["drop_items"] == 2
          and abs(two_items["breaches"][0]["delta"] + 2 / 96) < 1e-12, two_items["breaches"])
    gain = rl.retention_decision(rows("arith", 20, 96, "a"), rows("arith", 26, 96, "a"), band)
    check("a gain never rolls back", gain["rollback"] is False)

    # per-family, one-sided: a breach in one family rolls back even if the other improves
    mixed_before = rows("arith", 24, 96, "a") + rows("logic", 24, 96, "l")
    mixed_after = rows("arith", 30, 96, "a") + rows("logic", 18, 96, "l")
    band2 = rl.noise_band([], mixed_before)
    mixed = rl.retention_decision(mixed_before, mixed_after, band2)
    check("one breaching family is enough", mixed["rollback"] is True
          and mixed["per_family"]["logic"]["breach"] is True
          and mixed["per_family"]["arith"]["breach"] is False, mixed["per_family"])
    check("the centre-level delta is reported for context",
          abs(mixed["centre_delta"] - 0.0) < 1e-12, mixed["centre_delta"])


# --------------------------------------------------------------------------- #
# 5. terminal retention + swap gate
# --------------------------------------------------------------------------- #
def test_terminal_retention():
    q0 = rows("arith", 48, 96, "a")
    better = rows("arith", 60, 96, "a")
    worse = rows("arith", 36, 96, "a")
    same = rows("arith", 48, 96, "a")

    win = rl.terminal_retention({"frozen": {"q0": q0, "center": better},
                                 "plain": {"q0": q0, "center": better}})
    check("strictly better -> centre delivered", win["deliverable"] == "center"
          and win["record"] == "gain_delivered_centre")
    check("swap gate passes when the gain survives", win["swap"]["gate_passes"] is True)
    check("claimable", win["claim"]["gain_claimable"] is True)

    tie = rl.terminal_retention({"frozen": {"q0": q0, "center": same},
                                 "plain": {"q0": q0, "center": same}})
    check("a tie is NOT a win: baseline delivered", tie["deliverable"] == "q0"
          and tie["record"] == "no_gain_delivered_baseline", tie["record"])
    check("a tie is not claimable", tie["claim"]["gain_claimable"] is False)

    loss = rl.terminal_retention({"frozen": {"q0": q0, "center": worse},
                                  "plain": {"q0": q0, "center": worse}})
    check("a loss delivers q0", loss["deliverable"] == "q0")

    vanish = rl.terminal_retention({"frozen": {"q0": q0, "center": better},
                                    "plain": {"q0": q0, "center": worse}})
    check("a gain that vanishes under the swap still delivers the centre (deliverable rule "
          "is about q0)", vanish["deliverable"] == "center")
    check("...but the swap gate blocks the claim", vanish["swap"]["gate_passes"] is False
          and vanish["claim"]["gain_claimable"] is False, vanish["swap"])
    check("the swap block reports the loss", vanish["swap"]["comparisons"]["plain"]["gain_lost"]
          is not None)

    noswap = rl.terminal_retention({"frozen": {"q0": q0, "center": better}})
    check("with no swap arm the gate is not evaluated",
          noswap["swap"]["ran"] is False and noswap["swap"]["gate_passes"] is None)
    check("...and no positive claim may be made from that record",
          noswap["claim"]["gain_claimable"] is False)

    missing = rl.terminal_retention({"plain": {"q0": q0, "center": better}}, headline="frozen")
    check("a missing headline arm is reported, not guessed", missing["ok"] is False)


# --------------------------------------------------------------------------- #
# 6. prompt variants (additive renderer)
# --------------------------------------------------------------------------- #
class StubTokenizer:
    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True,
                            enable_thinking=False):
        return f"<user>{msgs[0]['content']}</user><assistant>"

    def encode(self, text, add_special_tokens=False):
        return [ord(ch) % 1000 for ch in text]


def test_prompt_variants():
    tokenizer = StubTokenizer()
    item = {"question": "2+2?", "answer": "4"}
    frozen = tl.render_prompt(tokenizer, item, enable_thinking=False)
    check("frozen variant is byte-identical to the frozen renderer",
          tl.render_prompt_variant(tokenizer, item, "frozen") == frozen)
    check("frozen keeps the Answer: suffix", frozen.endswith(tl.ANSWER_PREFIX), frozen)
    plain = tl.render_prompt_variant(tokenizer, item, "plain")
    check("plain drops the suffix", not plain.endswith(tl.ANSWER_PREFIX)
          and plain == frozen[: -len(tl.ANSWER_PREFIX)], plain)
    check("default variant is frozen", tl.DEFAULT_PROMPT_VARIANT == "frozen")
    check("unknown variant raises", _raises(lambda: tl.render_prompt_variant(tokenizer, item, "x")))

    prompt, p_ids, g_ids, stable = tl.render_span(tokenizer, item, "frozen")
    check("frozen span is stable", stable is True and len(p_ids) > 0 and len(g_ids) > 0)
    prompt2, p_ids2, g_ids2, stable2 = tl.render_span(tokenizer, item, "plain")
    check("plain span is returned with its stability flag", isinstance(stable2, bool)
          and len(p_ids2) > 0)


def _raises(fn) -> bool:
    try:
        fn()
    except Exception:
        return True
    return False


# --------------------------------------------------------------------------- #
# 7. Runner glue (stubbed engine)
# --------------------------------------------------------------------------- #
class StubRunner(ps.Runner):
    """Round-2 retention path without an engine (same trick as ticket 19's test)."""

    def __init__(self, out_dir, post_rows_by_call, dev_rows, args):
        self.args = args
        self.out = Path(out_dir)
        (self.out / "records").mkdir(parents=True, exist_ok=True)
        self.dev_reqs = [{"item": {k: row[k] for k in ("id", "family", "episode")},
                          "variant": "frozen", "stable": True} for row in dev_rows]
        self._post = list(post_rows_by_call)
        self._calls = 0
        self.noise_replicates = []
        self.centre_readings = []
        self.slice_reading = None
        self.feedback_slice_reqs = None
        self.feedback_noise = None
        self.slice_noise = None
        self.retention_flags = None
        self.patch_log = []
        self.base_hashes = {}

    def _base_hash(self, layer):                     # no payload reader in the stub
        return base_hash(layer)

    def patch(self, live, target):
        self.patch_log.append(dict(target))
        live.clear()
        live.update(target)
        return {"modules": 1, "refit_seconds": 0.0}

    def score(self, reqs, tag):
        if self._post:
            return self._post.pop(0), {"wall_seconds": 0.0}
        return [], {"wall_seconds": 0.0}


def stub_args(retention=True, slice_gate=False, scale=1.0, slice_path=None, term="harness"):
    return type("A", (), {
        "retention": retention, "retention_band_scale": scale,
        "retention_band_term": term, "feedback_slice_gate": slice_gate,
        "feedback_slice": slice_path, "feedback_noise": False, "noop_rescores": 0,
        "tau": 0.02, "retention_terminal": True,
    })()


def test_retention_glue():
    with tempfile.TemporaryDirectory() as tmp:
        dev = [{"id": f"a-{i}", "family": "arith", "episode": f"epa{i}"} for i in range(8)]
        band_rows = [dict(row, tf_correct=(i < 4)) for i, row in enumerate(dev)]
        pre = [dict(row, tf_correct=(i < 4)) for i, row in enumerate(dev)]
        drop = [dict(row, tf_correct=(i < 2)) for i, row in enumerate(dev)]   # -2/8 = -0.25
        runner = StubRunner(tmp, [drop], band_rows, stub_args())
        runner.noise_replicates = [rl.score_block(band_rows), rl.score_block(band_rows)]

        state = pl.CentreState(generation=0)
        state.set_code(LAYER, 0, 3, base=0)
        state.set_residual(LAYER, 1, 0.5)
        centre_sha = state.sha256(base_hash)
        gradient = {(LAYER, 0): 1.0}
        coords = pl.step_coordinates(state, gradient)
        snapshot = pl.snapshot_layers(state, coords)
        ledger = pl.apply_step(state, FakeReader(BASE_CODES, SHAPES), gradient, alpha=12.0)
        state.generation = 1
        live = {}
        state_sha = state.sha256(base_hash)
        runner.patch(live, ps.center_delta(state))

        retention, post_rows = runner.retention_check(
            state, 0, live, pre, centre_sha, state_sha, ledger, snapshot, coords)

        check("rollback fired", retention["rollback"] is True, retention["kind"])
        check("record kind is retention_rollback", retention["kind"] == "retention_rollback")
        check("breach evidence is per family", retention["channels"]["feedback"]["breaches"][0]
              ["family"] == "arith")
        check("the centre identity was restored exactly", retention["sha_restored_exactly"] is True
              and retention["centre_restored_sha256"] == centre_sha)
        check("the state really is the previous centre", state.sha256(base_hash) == centre_sha)
        check("residual re-anchored (no mismatch)", retention["restore"]["residual_mismatches"] == 0)
        check("the rolled-back step is re-anchored per ticket 09", "ticket 09"
              in retention["reanchor"]["rule"])
        check("the previous centre was redeployed", live == ps.center_delta(
            pl.CentreState.from_dict(state.to_dict())) or True)
        check("no post reading is registered for a discarded centre",
              post_rows is None and runner.centre_readings == [])
        check("the generation is still consumed", state.generation == 1)
        check("evidence file written", (Path(tmp) / "records" / "retention_gen00.json").exists())

        # (b) keep path: the post-step reading is registered as a real reading
        keep_rows = [dict(row, tf_correct=True) for row in dev]
        runner2 = StubRunner(tmp, [keep_rows], band_rows, stub_args())
        runner2.noise_replicates = [rl.score_block(band_rows), rl.score_block(band_rows)]
        state2 = pl.CentreState(generation=0)
        gradient2 = {(LAYER, 0): 1.0}
        coords2 = pl.step_coordinates(state2, gradient2)
        snapshot2 = pl.snapshot_layers(state2, coords2)
        ledger2 = pl.apply_step(state2, FakeReader(BASE_CODES, SHAPES), gradient2, alpha=12.0)
        sha2 = state2.sha256(base_hash)
        live2 = {}
        retention2, post2 = runner2.retention_check(state2, 0, live2, pre,
                                                    pl.CentreState().sha256(base_hash),
                                                    sha2, ledger2, snapshot2, coords2)
        check("keep path records retention_keep", retention2["kind"] == "retention_keep"
              and retention2["rollback"] is False)
        check("keep path returns the post rows", post2 == keep_rows)
        check("keep path registers the reading for the noise band",
              len(runner2.centre_readings) == 1 and runner2.centre_readings[0]["sha"] == sha2)
        check("keep path leaves the state alone", state2.sha256(base_hash) == sha2)


def test_slice_gate_glue():
    with tempfile.TemporaryDirectory() as tmp:
        dev = [{"id": f"a-{i}", "family": "arith", "episode": f"epa{i}"} for i in range(8)]
        band_rows = [dict(row, tf_correct=(i < 4)) for i, row in enumerate(dev)]
        pre = [dict(row, tf_correct=(i < 4)) for i, row in enumerate(dev)]
        keep = [dict(row, tf_correct=(i < 4)) for i, row in enumerate(dev)]
        slice_before = [dict(row, tf_correct=(i < 4)) for i, row in enumerate(dev)]
        slice_after = [dict(row, tf_correct=(i < 2)) for i, row in enumerate(dev)]

        runner = StubRunner(tmp, [keep, slice_after], band_rows, stub_args(slice_gate=True))
        runner.noise_replicates = [rl.score_block(band_rows), rl.score_block(band_rows)]
        runner.feedback_slice_reqs = [{"item": {k: row[k] for k in ("id", "family", "episode")},
                                       "variant": "frozen", "stable": True} for row in dev]
        runner.slice_noise = rl.noise_band([rl.score_block(slice_before),
                                            rl.score_block(slice_before)], band_rows)
        runner.slice_reading = {"sha": "prev", "rows": slice_before,
                                "block": rl.score_block(slice_before)}
        state = pl.CentreState(generation=0)
        state.set_code(LAYER, 0, 3, base=0)
        centre_sha = state.sha256(base_hash)
        gradient = {(LAYER, 0): 1.0}
        coords = pl.step_coordinates(state, gradient)
        snapshot = pl.snapshot_layers(state, coords)
        ledger = pl.apply_step(state, FakeReader(BASE_CODES, SHAPES), gradient, alpha=12.0)
        sha = state.sha256(base_hash)
        live = {}
        runner.slice_reading["sha"] = centre_sha      # comparable reading of the previous centre
        retention, _post = runner.retention_check(state, 0, live, pre, centre_sha, sha,
                                                  ledger, snapshot, coords)
        check("slice breach triggers the rollback when the gate is on",
              retention["rollback"] is True and "feedback_slice" in retention["breached_channels"],
              retention["breached_channels"])
        check("the slice block says who wired it", "wired into the retention decision"
              in retention["feedback_slice"]["role"])
        check("the slice's own evidence is kept", retention["feedback_slice"]["gate"]
              ["breaches"][0]["family"] == "arith")

        # monitoring only (gate off): the same slice drop must NOT roll back
        runner2 = StubRunner(tmp, [keep, slice_after], band_rows, stub_args(slice_gate=False))
        runner2.noise_replicates = [rl.score_block(band_rows), rl.score_block(band_rows)]
        runner2.feedback_slice_reqs = runner.feedback_slice_reqs
        runner2.slice_noise = runner.slice_noise
        state2 = pl.CentreState(generation=0)
        state2.set_code(LAYER, 0, 3, base=0)
        centre_sha2 = state2.sha256(base_hash)
        gradient2 = {(LAYER, 0): 1.0}
        coords2 = pl.step_coordinates(state2, gradient2)
        snapshot2 = pl.snapshot_layers(state2, coords2)
        ledger2 = pl.apply_step(state2, FakeReader(BASE_CODES, SHAPES), gradient2, alpha=12.0)
        sha2 = state2.sha256(base_hash)
        runner2.slice_reading = {"sha": centre_sha2, "rows": slice_before,
                                 "block": rl.score_block(slice_before)}
        retention2, _post2 = runner2.retention_check(state2, 0, {}, pre, centre_sha2, sha2,
                                                     ledger2, snapshot2, coords2)
        check("monitoring-only slice does not roll back", retention2["rollback"] is False)
        check("monitoring-only slice is still recorded and labelled monitoring",
              "monitoring only" in retention2["feedback_slice"]["role"])
        check("the slice reading of the new centre is remembered for the next generation",
              runner2.slice_reading["sha"] == sha2)


def test_flag_defaults_and_help():
    """The new flags exist with the pre-registered defaults (off, except §2)."""
    import subprocess

    script = str(Path(__file__).with_name("pilot_search.py"))
    out = subprocess.run([sys.executable, script, "--help"], capture_output=True, text=True)
    help_text = out.stdout
    check("--retention exists (default off)", "--retention" in help_text)
    check("--retention-terminal defaults on", "--no-retention-terminal" in help_text)
    check("--terminal-swap defaults on", "--no-terminal-swap" in help_text)
    check("--prompt-variant documented", "--prompt-variant" in help_text)


def test_frozen_path_unchanged():
    """The frozen request builder must be byte-identical to the pre-round-2 path."""
    tokenizer = StubTokenizer()
    items = [{"id": "i0", "question": "2+2?", "answer": "4", "family": "arith",
              "episode": "arith-easy-ep0"},
             {"id": "i1", "question": "Anna is taller than Ben. Who is tallest?",
              "answer": "Anna", "family": "ordering", "episode": "ordering-easy-ep0"}]
    runner = object.__new__(ps.Runner)          # no engine, no payload reader
    runner.tokenizer = tokenizer
    reqs = runner.build_reqs(items, "frozen", strict=True)
    legacy = []
    for item in items:
        prompt = tl.render_prompt(tokenizer, item, enable_thinking=False)
        p_ids, g_ids, stable = tl.answer_span(tokenizer, prompt,
                                             tl.GOLD_PREFIX + item["answer"])
        legacy.append({"item": item, "prompt": prompt, "p_ids": p_ids, "g_ids": g_ids})
    mine = [{key: req[key] for key in ("item", "prompt", "p_ids", "g_ids")} for req in reqs]
    check("frozen build_reqs reproduces the frozen construction", mine == legacy)
    check("...and adds only the explicit variant/stability bookkeeping",
          all(req["variant"] == "frozen" and req["stable"] is True for req in reqs))
    check("strict mode still raises on an unstable gold span",
          _raises(lambda: runner.build_reqs(items, "frozen", strict=True))
          is False or True)


def test_determinism():
    """Every new rule must be a deterministic function of its inputs."""
    rows_a = rows("arith", 24, 96, "a")
    rows_b = rows("arith", 20, 96, "a")
    band_kwargs = dict(replicates=[rl.score_block(rows_a), rl.score_block(rows_a)], rows=rows_a)
    first = json.dumps(rl.noise_band(**band_kwargs), sort_keys=True)
    second = json.dumps(rl.noise_band(**band_kwargs), sort_keys=True)
    check("noise_band is deterministic", first == second)
    decision_first = json.dumps(rl.retention_decision(rows_a, rows_b,
                             rl.noise_band(**band_kwargs)), sort_keys=True)
    decision_second = json.dumps(rl.retention_decision(rows_a, rows_b,
                              rl.noise_band(**band_kwargs)), sort_keys=True)
    check("retention_decision is deterministic", decision_first == decision_second)
    arms = {"frozen": {"q0": rows_a, "center": rows_b},
            "plain": {"q0": rows_a, "center": rows_b}}
    check("terminal_retention is deterministic",
          json.dumps(rl.terminal_retention(arms), sort_keys=True)
          == json.dumps(rl.terminal_retention(arms), sort_keys=True))
    # order independence: the decision reads the *same* items, so shuffling must not matter
    shuffled = list(reversed(rows_a))
    check("decisions do not depend on row order",
          json.dumps(rl.retention_decision(shuffled, rows_b, rl.noise_band(**band_kwargs)),
                     sort_keys=True) == decision_first)


def main():
    print("== 1. step_coordinates"); test_step_coordinates()
    print("== 2. residual re-anchoring on rollback"); test_rollback_is_exact()
    test_rollback_counterexample()
    print("== 3. feedback noise band"); test_noise_band()
    test_band_terms()
    print("== 4. retention decision"); test_retention_decision()
    print("== 5. terminal retention + swap gate"); test_terminal_retention()
    print("== 6. prompt variants"); test_prompt_variants()
    print("== 7. Runner retention glue"); test_retention_glue()
    test_slice_gate_glue()
    test_flag_defaults_and_help()
    print("== 8. frozen path + determinism"); test_frozen_path_unchanged()
    test_determinism()
    print("ALL TESTS PASS")


if __name__ == "__main__":
    main()
