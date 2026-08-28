"""Trainer-level end-to-end smoke for the GRACE loss modes.

Runs ``train_shadow`` on a tiny locally-built Llama (random init, saved to a
tmp dir) over random teacher data: a few steps must complete with finite
losses, the train log must carry the new GRACE fields (gated-DKD components,
gate stats records, beta trajectory), and summary.json must carry the
gate/IB verdict block. ``dkd_gate_ib_cka`` exercises the full stack
(gate + IB + span-CKA); ``dkd_gate`` pins the no-IB branch (beta fixed 1).
"""

from __future__ import annotations

import json
import math

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("tokenizers")

from tokenizers import Tokenizer  # noqa: E402
from tokenizers import models as tk_models  # noqa: E402
from tokenizers import pre_tokenizers as tk_pre  # noqa: E402
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast  # noqa: E402

from shadow_lab.trainer import TrainConfig, train_shadow  # noqa: E402

VOCAB = 96


@pytest.fixture(scope="module")
def tiny_model_dir(tmp_path_factory):
    """Random-init tiny Llama + a matching WordLevel tokenizer, on disk."""
    d = tmp_path_factory.mktemp("tiny-model")
    cfg = LlamaConfig(
        vocab_size=VOCAB,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=128,
    )
    torch.manual_seed(0)
    LlamaForCausalLM(cfg).save_pretrained(d)
    tk = Tokenizer(tk_models.WordLevel(vocab={f"w{i}": i for i in range(VOCAB)}, unk_token="w0"))
    tk.pre_tokenizer = tk_pre.Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=tk, pad_token="w0", eos_token="w1", bos_token="w2"
    ).save_pretrained(d)
    return d


@pytest.fixture(scope="module")
def teacher_data(tmp_path_factory):
    """One short row (exercises pad masking) among full-length rows."""
    torch.manual_seed(7)
    p = tmp_path_factory.mktemp("data") / "teacher.jsonl"
    rows = [torch.randint(2, VOCAB, (64,)).tolist() for _ in range(6)]
    rows.append(torch.randint(2, VOCAB, (40,)).tolist())  # shorter than seq_len
    p.write_text("".join(json.dumps({"ids": r}) + "\n" for r in rows))
    return str(p)


def _run(loss_mode: str, tmp_path, tiny_model_dir, teacher_data, **overrides) -> tuple:
    cfg = TrainConfig(
        bits=0,
        backbone_tier="w4a16",
        tier_fake_quant=True,  # the exp1-grace arm configuration class
        loss_mode=loss_mode,
        model=str(tiny_model_dir),
        data=teacher_data,
        max_tokens=6 * 2 * 64,  # 6 steps at batch_seq=2, seq_len=64
        seq_len=64,
        batch_seq=2,
        lr=1e-4,
        shadow_layers=1,
        log_every=1,
        eval_every=3,
        gate_log_every=2,
        save_dir=str(tmp_path / "train"),
        device="cpu",
        pad_id=0,
        **overrides,
    )
    report = train_shadow(cfg)
    lines = [
        json.loads(line)
        for line in (tmp_path / "train" / "train_log.jsonl").read_text().splitlines()
    ]
    summary = json.loads((tmp_path / "train" / "summary.json").read_text())
    return report, lines, summary


@pytest.mark.filterwarnings("ignore")
def test_dkd_gate_ib_cka_smoke(tiny_model_dir, teacher_data, tmp_path):
    report, lines, summary = _run("dkd_gate_ib_cka", tmp_path, tiny_model_dir, teacher_data)

    assert report.loss_mode == "dkd_gate_ib_cka"
    assert report.steps >= 5
    assert math.isfinite(report.final_ppl_20chunks) and report.final_ppl_20chunks > 0
    assert (tmp_path / "train" / "final").exists()

    step_lines = [ln for ln in lines if "loss" in ln]
    assert len(step_lines) >= 5
    for ln in step_lines:
        assert math.isfinite(ln["loss"])
        # wrapper KD is off in GRACE modes; the GRACE components replace it
        assert ln["kd"] is None
        for key in ("gdkd", "tckd", "nckd", "gate_mean", "beta", "cka"):
            assert key in ln and ln[key] is not None and math.isfinite(ln[key])

    gate_lines = [ln for ln in lines if "gate_hist" in ln]
    assert [ln["step"] for ln in gate_lines] == [2, 4, 6]  # gate_log_every=2
    for gl in gate_lines:
        assert sum(gl["gate_hist"]) > 0
        assert math.isfinite(gl["gate_cv"])
        assert "beta" in gl and "gate_degraded" in gl

    grace = summary["grace"]
    assert grace["gate_degraded"] is False
    assert grace["beta_min"] >= 0.1 and grace["beta_max"] <= 5.0
    assert math.isfinite(grace["gate_final_mean"])
    # args dump carries the GRACE knobs (verdict reproducibility)
    assert summary["args"]["dkd_temperature"] == 2.0


@pytest.mark.filterwarnings("ignore")
def test_dkd_gate_smoke_beta_fixed(tiny_model_dir, teacher_data, tmp_path):
    report, lines, summary = _run("dkd_gate", tmp_path, tiny_model_dir, teacher_data)

    assert report.loss_mode == "dkd_gate"
    step_lines = [ln for ln in lines if "loss" in ln]
    assert all(math.isfinite(ln["loss"]) for ln in step_lines)
    # No IB controller in this mode: beta stays at the fixed identity value.
    assert all(ln["beta"] == 1.0 for ln in step_lines)
    assert all("cka" not in ln for ln in step_lines)
    assert summary["grace"]["beta_final"] == 1.0
    assert summary["grace"]["gate_degraded"] is False


def _step_gdkd(lines) -> list[float]:
    return [ln["gdkd"] for ln in lines if "loss" in ln]


@pytest.mark.filterwarnings("ignore")
def test_ib_disabled_identity(tiny_model_dir, teacher_data, tmp_path):
    """Ticket 03 identity: with the dual ascent frozen (eta=0, beta=1),
    dkd_gate_ib is bit-identical to dkd_gate (same seed -> same data order)."""
    _, base_lines, base_sum = _run("dkd_gate", tmp_path / "a", tiny_model_dir, teacher_data)
    _, ib_lines, ib_sum = _run(
        "dkd_gate_ib", tmp_path / "b", tiny_model_dir, teacher_data, ib_eta=0.0
    )
    assert _step_gdkd(ib_lines) == pytest.approx(_step_gdkd(base_lines), abs=1e-7)
    assert [ln["loss"] for ln in ib_lines if "loss" in ln] == pytest.approx(
        [ln["loss"] for ln in base_lines if "loss" in ln], abs=1e-7
    )
    assert ib_sum["grace"]["beta_final"] == 1.0


@pytest.mark.filterwarnings("ignore")
def test_cka_zero_weight_identity(tiny_model_dir, teacher_data, tmp_path):
    """Ticket 04 identity: cka_weight=0 reduces dkd_gate_ib_cka to dkd_gate_ib."""
    _, ib_lines, _ = _run(
        "dkd_gate_ib", tmp_path / "a", tiny_model_dir, teacher_data, ib_eta=0.0
    )
    _, cka_lines, _ = _run(
        "dkd_gate_ib_cka",
        tmp_path / "b",
        tiny_model_dir,
        teacher_data,
        ib_eta=0.0,
        cka_weight=0.0,
    )
    assert _step_gdkd(cka_lines) == pytest.approx(_step_gdkd(ib_lines), abs=1e-7)
    assert [ln["loss"] for ln in cka_lines if "loss" in ln] == pytest.approx(
        [ln["loss"] for ln in ib_lines if "loss" in ln], abs=1e-7
    )
