"""Residual-matched noise seam (exp1-native4b ticket 01).

Math properties of ``residual_shuffle_noise``: the within-block shuffle
preserves each block's residual multiset exactly (per-block RMS and the
global marginal are unchanged), the deterministic element-wise coupling with
w is broken, the multiplier scales block RMS linearly, and padded columns
never leak. Trainer-level: the noise arm runs end-to-end on a tiny model
with finite losses and a loadable checkpoint, the perturbation is real (same
seed differs from the no-noise run) and deterministic (same seed repeats),
and ``shadow_quant``/``shadow_noise`` mutual exclusion is enforced.
"""

from __future__ import annotations

import json
import math

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("tokenizers")

from shadow_lab.shadow_quant import residual_shuffle_noise  # noqa: E402
from shadow_lab.trainer import TrainConfig, train_shadow  # noqa: E402

BLOCK = 16


def _residual(w: torch.Tensor) -> torch.Tensor:
    from shadow_lab.shadow_quant import nvfp4_weight

    return w.float() - nvfp4_weight(w).float()


def _corr(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().float()
    b = b.flatten().float()
    a = a - a.mean()
    b = b - b.mean()
    return (a * b).sum().item() / (a.norm().item() * b.norm().item())


def test_shuffle_preserves_block_rms_and_marginal():
    torch.manual_seed(3)
    w = torch.randn(64, 96) * 0.05  # in_f divisible by BLOCK: no padding
    noisy = residual_shuffle_noise(w, "nvfp4")
    noise = noisy - w

    residual = _residual(w)
    # within-block permutation: the global multiset is identical up to the
    # (w + e) - w float round-trip in this test's subtraction
    assert torch.allclose(
        noise.flatten().sort().values, residual.flatten().sort().values, atol=1e-7, rtol=1e-6
    )
    # and so is every block's RMS (scale matching is per block, not global)
    nb = noise.view(64, -1, BLOCK).norm(dim=-1)
    rb = residual.view(64, -1, BLOCK).norm(dim=-1)
    assert torch.allclose(nb, rb, rtol=1e-5)


def test_shuffle_breaks_elementwise_coupling():
    torch.manual_seed(4)
    w = torch.randn(64, 96) * 0.05
    noisy = residual_shuffle_noise(w, "nvfp4")
    noise = noisy - w
    residual = _residual(w)
    # a rounding of w is correlated with w's residual by construction; the
    # shuffled noise must not be (i.i.d. position swap breaks the pairing)
    assert abs(_corr(noise, residual)) < 0.2
    assert not torch.equal(noise, residual)


def test_mult_scales_block_rms_linearly():
    torch.manual_seed(5)
    w = torch.randn(48, 128) * 0.05
    n1 = residual_shuffle_noise(w, "nvfp4", mult=1.0) - w
    n2 = residual_shuffle_noise(w, "nvfp4", mult=2.0) - w
    r1 = n1.view(48, -1, BLOCK).norm(dim=-1)
    r2 = n2.view(48, -1, BLOCK).norm(dim=-1)
    # block multisets are permutation-invariant, so the ratio is mult-exact
    # even though the two draws use different permutations
    assert torch.allclose(r2, 2.0 * r1, rtol=1e-5)


def test_padded_width_keeps_shape_and_advances_rng():
    torch.manual_seed(6)
    w = torch.randn(32, 50) * 0.05  # pads to 64 internally
    a = residual_shuffle_noise(w, "nvfp4")
    b = residual_shuffle_noise(w, "nvfp4")
    assert a.shape == w.shape and b.shape == w.shape
    assert torch.isfinite(a).all()
    assert not torch.equal(a, b)  # fresh noise each call


def test_unknown_format_rejected():
    w = torch.randn(8, 16)
    with pytest.raises(ValueError, match="unknown shadow weight format"):
        residual_shuffle_noise(w, "fp8")


def test_config_noise_validation():
    kw = dict(bits=0, backbone_tier="w4a16", loss_mode="ce_kl", model="m", data="d")
    TrainConfig(shadow_noise="nvfp4", **kw)  # valid on its own
    with pytest.raises(ValueError, match="mutually exclusive"):
        TrainConfig(shadow_noise="nvfp4", shadow_quant="nvfp4", **kw)
    with pytest.raises(ValueError, match="unsupported shadow_noise"):
        TrainConfig(shadow_noise="int8", **kw)


# --- trainer-level smoke: tiny locally-built Llama (self-contained fixtures,
# mirroring tests/lab/test_trainer_grace.py) ---

VOCAB = 96


@pytest.fixture(scope="module")
def tiny_model_dir(tmp_path_factory):
    from tokenizers import Tokenizer
    from tokenizers import models as tk_models
    from tokenizers import pre_tokenizers as tk_pre
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

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
    torch.manual_seed(7)
    p = tmp_path_factory.mktemp("data") / "teacher.jsonl"
    rows = [torch.randint(2, VOCAB, (64,)).tolist() for _ in range(6)]
    rows.append(torch.randint(2, VOCAB, (40,)).tolist())
    p.write_text("".join(json.dumps({"ids": r}) + "\n" for r in rows))
    return str(p)


def _run(tiny_model_dir, teacher_data, tmp_path, **overrides):
    cfg = TrainConfig(
        bits=0,
        backbone_tier="w4a16",
        tier_fake_quant=True,
        loss_mode="ce_kl",
        model=str(tiny_model_dir),
        data=teacher_data,
        max_tokens=6 * 2 * 64,
        seq_len=64,
        batch_seq=2,
        lr=1e-4,
        shadow_layers=1,
        log_every=1,
        eval_every=3,
        save_dir=str(tmp_path / "train"),
        device="cpu",
        pad_id=0,
        **overrides,
    )
    report = train_shadow(cfg)
    losses = [
        json.loads(line)["loss"]
        for line in (tmp_path / "train" / "train_log.jsonl").read_text().splitlines()
        if '"loss"' in line
    ]
    summary = json.loads((tmp_path / "train" / "summary.json").read_text())
    return report, losses, summary


@pytest.mark.filterwarnings("ignore")
def test_noise_arm_smoke(tiny_model_dir, teacher_data, tmp_path):
    report, losses, summary = _run(
        tiny_model_dir, teacher_data, tmp_path, shadow_noise="nvfp4", shadow_noise_mult=1.5
    )
    assert report.steps >= 5
    assert all(math.isfinite(v) for v in losses)
    assert math.isfinite(report.final_ppl_20chunks) and report.final_ppl_20chunks > 0
    assert (tmp_path / "train" / "final").exists()
    assert summary["args"]["shadow_noise"] == "nvfp4"
    assert summary["args"]["shadow_noise_mult"] == 1.5


@pytest.mark.filterwarnings("ignore")
def test_noise_is_real_and_deterministic(tiny_model_dir, teacher_data, tmp_path):
    """Same seed: the noisy run differs from the clean run (the perturbation
    reaches the training signal) and repeats itself exactly."""
    _, clean, _ = _run(tiny_model_dir, teacher_data, tmp_path / "clean")
    _, noisy_a, _ = _run(tiny_model_dir, teacher_data, tmp_path / "a", shadow_noise="nvfp4")
    _, noisy_b, _ = _run(tiny_model_dir, teacher_data, tmp_path / "b", shadow_noise="nvfp4")
    assert noisy_a != pytest.approx(clean, abs=1e-9)
    assert noisy_a == pytest.approx(noisy_b, abs=1e-9)
