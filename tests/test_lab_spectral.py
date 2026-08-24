"""Spectral diagnostics suite tests (ticket 08): collector -> metrics -> report."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

_EXPERIMENT = Path(__file__).resolve().parents[1] / "experiment"
sys.path.insert(0, str(_EXPERIMENT))

from lab import spectral  # noqa: E402

from shadow_peft import ShadowConfig, get_shadow_model  # noqa: E402

VOCAB = 128


def _tiny_llama(num_layers: int = 4):
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(
        vocab_size=VOCAB,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=num_layers,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
    )
    return LlamaForCausalLM(cfg)


def _shadow_cfg() -> ShadowConfig:
    return ShadowConfig(
        num_shadow_layers=1,
        injection_hidden_size=8,
        gate_hidden_size=10,
        alpha=0.1,
        dropout=0.0,
    )


def _batches(n: int = 2):
    torch.manual_seed(0)
    return [
        {
            "input_ids": torch.randint(1, VOCAB, (1, 24)),
            "attention_mask": torch.ones(1, 24, dtype=torch.long),
        }
        for _ in range(n)
    ]


def test_collector_captures_per_layer_states():
    torch.manual_seed(0)
    base = _tiny_llama()
    peft = get_shadow_model(base, _shadow_cfg())
    peft.eval()

    collector = spectral.ShadowStateCollector(peft)
    collector.run(peft, _batches(2))

    # One entry per layer (state entering each wrapped layer), two batches each.
    assert sorted(collector.captures) == [0, 1, 2, 3]
    for idx, caps in collector.captures.items():
        assert len(caps) == 2
        assert caps[0].shape == (1, 24, 128)
        assert caps[0].dtype == torch.float32


def test_collector_save_load_roundtrip(tmp_path: Path):
    torch.manual_seed(0)
    base = _tiny_llama()
    peft = get_shadow_model(base, _shadow_cfg())
    peft.eval()

    collector = spectral.ShadowStateCollector(peft)
    collector.run(peft, _batches(1))
    collector.save(tmp_path / "capture.pt")

    loaded = spectral.ShadowStateCollector.load(tmp_path / "capture.pt")
    assert sorted(loaded) == [0, 1, 2, 3]
    assert torch.equal(loaded[0], torch.stack(collector.captures[0]))


def test_effective_rank_and_lar_sanity():
    torch.manual_seed(0)
    # Rank-1 data: erank -> 1, LAR -> 1.
    rank1 = torch.randn(64, 1) @ torch.randn(1, 128)
    assert spectral.effective_rank(rank1) == pytest.approx(1.0, abs=1e-3)
    assert spectral.lar(rank1) == pytest.approx(1.0, abs=1e-6)
    # Isotropic-ish random data: erank approaches min-dim, LAR ~ 1/rank.
    iso = torch.randn(512, 128)
    assert spectral.effective_rank(iso) > 100
    assert spectral.lar(iso) < 0.05
    # Scale invariance of both metrics.
    assert spectral.effective_rank(iso * 7.0) == pytest.approx(spectral.effective_rank(iso))
    assert spectral.lar(iso * 7.0) == pytest.approx(spectral.lar(iso))


def test_spectral_drift_identical_and_scaled():
    torch.manual_seed(0)
    x = torch.randn(128, 64)
    d0 = spectral.spectral_drift(x, x.clone())
    assert d0["spectrum_cosine"] == pytest.approx(1.0, abs=1e-6)
    # The k-th principal angle amplifies float noise in acos; ~0.05° is noise.
    assert d0["subspace_angle_deg"] < 0.1
    assert d0["energy_ratio"] == pytest.approx(1.0, abs=1e-6)

    d_scaled = spectral.spectral_drift(x, x * 2.0)
    assert d_scaled["spectrum_cosine"] == pytest.approx(1.0, abs=1e-6)  # shape unchanged
    assert d_scaled["energy_ratio"] == pytest.approx(4.0, rel=1e-4)  # Σλ scales quadratically

    # Two independent Gaussians share the Marchenko–Pastur spectrum shape, so
    # the cosine stays high — a *structured* difference (column-variance skew)
    # is what visibly bends the spectrum.
    y = torch.randn(128, 64)
    y[:, 32:] *= 0.05
    d_diff = spectral.spectral_drift(x, y)
    assert d_diff["spectrum_cosine"] < 0.99
    assert d_diff["subspace_angle_deg"] > 1.0


def test_layer_scalar_normalization():
    class FakeLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("layer_scalar", torch.tensor(2.0))

    state = torch.randn(2, 8, 16)
    normalized = spectral._divide_layer_scalar(state, FakeLayer())
    assert torch.allclose(normalized, state / 2.0)
    # Without the buffer, the state passes through untouched.
    assert spectral._divide_layer_scalar(state, torch.nn.Identity()) is state


def test_tiny_end_to_end_demo(tmp_path: Path):
    """采集 -> 指标 -> 谱漂移 -> 报告 全链路."""
    torch.manual_seed(0)
    batches = _batches(2)

    base_a = _tiny_llama()
    peft_a = get_shadow_model(base_a, _shadow_cfg())
    peft_a.eval()

    # "post" model: same architecture, different weights (stands in for a
    # quantization-shifted state distribution).
    torch.manual_seed(3)
    base_b = _tiny_llama()
    peft_b = get_shadow_model(base_b, _shadow_cfg())
    peft_b.eval()

    col_a = spectral.ShadowStateCollector(peft_a)
    col_a.run(peft_a, batches)
    col_b = spectral.ShadowStateCollector(peft_b)
    col_b.run(peft_b, batches)

    pre_metrics = spectral.metrics_from_capture(col_a.captures)
    post_metrics = spectral.metrics_from_capture(col_b.captures)
    drift = spectral.drift_report(col_a.captures, col_b.captures)
    report = spectral.render_report(pre_metrics, post_metrics, drift, title="tiny demo")

    assert set(pre_metrics) == {"0", "1", "2", "3"}
    for m in pre_metrics.values():
        assert m["erank"] > 1.0
        assert 0.0 < m["lar"] <= 1.0
        assert m["energy"] > 0.0
    assert drift["summary"]["mean_spectrum_cosine"] < 1.0
    assert "not part of the Pareto axes" in report
    assert "| pre/0 |" in report and "| post/0 |" in report
    assert "| 0 |" in report  # drift table rows

    # Everything JSON-serializable for the independent report artifact.
    json.dumps({"pre": pre_metrics, "post": post_metrics, "drift": drift})
