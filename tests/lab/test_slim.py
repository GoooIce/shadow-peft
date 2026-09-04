"""AngelSlim compression-engine bridge tests (replacement step 1: coexistence).

Pure-logic tests run everywhere — no angelslim install needed (the bridge
imports angelslim lazily): family resolution, tier validation, the
CalibLoader contract, and the Engine orchestration order against a fake
engine, mirroring how test_lab_tiers pins the torchao convert-before-mount
order. The real-conversion test needs angelslim; importorskip keeps Mac
(no triton) green while GB10 runs it for real.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from shadow_lab.slim import engine as slim_engine  # noqa: E402
from shadow_lab.slim import ptq as slim_ptq  # noqa: E402
from shadow_lab.slim.ptq import (  # noqa: E402
    ANGELSLIM_TIERS,
    FAKE_QUANT_TIERS,
    convert_backbone,
)
from shadow_lab.trainer import TrainConfig  # noqa: E402


def _tiny_llama(num_layers: int = 2):
    from transformers import LlamaConfig, LlamaForCausalLM

    # hidden/intermediate sizes must clear the int4 group size (128) so the
    # GPTQ observer set is non-empty on every projection.
    cfg = LlamaConfig(
        vocab_size=128,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=num_layers,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
        torch_dtype=torch.float32,
    )
    return LlamaForCausalLM(cfg)


def _calib_batches(n: int = 2, seq: int = 32):
    return [
        {
            "input_ids": torch.randint(0, 128, (1, seq)),
            "attention_mask": torch.ones(1, seq, dtype=torch.long),
        }
        for _ in range(n)
    ]


# --- family resolution (pure table; no angelslim involved) ---


def test_family_resolution_table():
    assert slim_engine.resolve_family_by_type("qwen2") == "Qwen"
    assert slim_engine.resolve_family_by_type("qwen3") == "Qwen"
    assert slim_engine.resolve_family_by_type("llama") == "Llama"
    # longer prefixes win: glm5 → GLM5_1, not GLM
    assert slim_engine.resolve_family_by_type("glm5_1") == "GLM5_1"
    assert slim_engine.resolve_family_by_type("glm4") == "GLM"
    assert slim_engine.resolve_family_by_type("deepseek_v3") == "DeepSeek"
    # Llama-isomorphic but unregistered upstream → generic wrapper
    assert slim_engine.resolve_family_by_type("gemma3") == slim_engine.GENERIC_LLAMA_LIKE
    assert slim_engine.resolve_family_by_type("lfm2") == slim_engine.GENERIC_LLAMA_LIKE
    with pytest.raises(ValueError, match="model_name="):
        slim_engine.resolve_family_by_type("mistral")


def test_resolve_family_from_model():
    assert slim_engine.resolve_family(_tiny_llama()) == "Llama"


def test_generic_observer_suffixes_match_llama_layout():
    # The generic wrapper must observe exactly the Llama projection layout —
    # Gemma/LFM2 share it, which is what makes the fallback sound.
    assert slim_engine._LLM_OBSERVER_SUFFIXES == (
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.up_proj",
        "mlp.gate_proj",
        "mlp.down_proj",
    )


# --- CalibLoader contract ---


def test_calib_loader_contract():
    batches = _calib_batches(n=3)
    loader = slim_engine.CalibLoader(batches)
    assert len(loader) == 3
    seen = list(loader)
    assert len(seen) == 3
    first = seen[0]
    assert set(first.keys()) == {"input_ids", "labels", "attention_mask"}
    assert torch.equal(first["labels"], first["input_ids"])
    assert first["attention_mask"].shape == first["input_ids"].shape


# --- tier validation (no angelslim needed: checks run before any import) ---


def test_tier_validation():
    model = _tiny_llama()
    with pytest.raises(ValueError, match="unsupported AngelSlim tier"):
        convert_backbone(model, object(), "w4a16", _calib_batches())
    # awq always packs WQLinearGEMM — no fake-quant path
    with pytest.raises(ValueError, match="fake_quant=True requires"):
        convert_backbone(model, object(), "int4_awq", _calib_batches())
    with pytest.raises(ValueError, match="calibration-based"):
        convert_backbone(model, object(), "int4_gptq", [])
    # bf16 needs no conversion and touches nothing
    manifest = convert_backbone(model, object(), "bf16", [])
    assert manifest == {
        "tier": "bf16",
        "backend": "angelslim",
        "format": "bf16",
        "sequence": ["noop"],
    }


def test_tier_tables():
    assert "nvfp4_gptq" in ANGELSLIM_TIERS
    assert set(FAKE_QUANT_TIERS) <= set(ANGELSLIM_TIERS)


# --- orchestration order against a fake engine (the seam test) ---


class _FakeEngine:
    def __init__(self):
        self.calls: list[tuple] = []

    def prepare_model(self, **kwargs):
        self.calls.append(("prepare_model", kwargs.get("model_name")))

    def prepare_data(self, custom_dataloader=None, **kwargs):
        self.calls.append(("prepare_data", custom_dataloader))

    def prepare_compressor(self, compress_name, global_config=None, compress_config=None, **kw):
        self.calls.append(("prepare_compressor", compress_name))

    def run(self):
        self.calls.append(("run",))

    def convert(self):
        self.calls.append(("convert",))

    def save(self, save_path):
        self.calls.append(("save", save_path))


def test_convert_backbone_orchestration(monkeypatch):
    """Calibrate → convert → (save): the AngelSlim drive order, pinned."""
    fake = _FakeEngine()

    def fake_build_engine(base, tok, model_name=None, deploy_backend="huggingface"):
        # stand-in for the real build_engine: prepare_model is its first
        # (and only) engine-side effect on the drive order we pin here.
        fake.prepare_model(model_name=model_name or slim_engine.resolve_family(base))
        return fake

    monkeypatch.setattr(slim_engine, "build_engine", fake_build_engine)
    monkeypatch.setattr(slim_ptq, "_assemble", lambda *a, **k: (object(), object()))

    model = _tiny_llama()
    batches = _calib_batches()

    manifest = convert_backbone(model, object(), "int4_gptq", batches, save_dir="/tmp/slim-out")
    assert [c[0] for c in fake.calls] == [
        "prepare_model",
        "prepare_data",
        "prepare_compressor",
        "run",
        "convert",
        "save",
    ]
    # the calib batches cross the seam as the AngelSlim-contract loader
    assert isinstance(fake.calls[1][1], slim_engine.CalibLoader)
    assert fake.calls[2][1] == "PTQ"
    assert manifest["format"] == "int4_gptq_fake_bf16"
    assert manifest["calib_forwards"] == len(batches)
    assert manifest["model_name"] == "Llama"
    assert manifest["joint_calibration"] is None  # weight-only: rule not applicable

    # static-activation tier records the joint-calibration caveat
    fake.calls.clear()
    manifest = convert_backbone(model, object(), "fp8_static", batches, fake_quant=False)
    assert [c[0] for c in fake.calls] == [
        "prepare_model",
        "prepare_data",
        "prepare_compressor",
        "run",
        "convert",
    ]
    assert manifest["format"] == "fp8_static_packed"
    assert manifest["joint_calibration"] is False


# --- trainer wiring (validation only; the drive is covered above) ---


def test_train_config_backbone_backend():
    TrainConfig(
        bits=0, backbone_tier="int4_gptq", backbone_backend="angelslim", tier_fake_quant=True
    )
    with pytest.raises(ValueError, match="backbone_tier='w4a16'.*backend 'angelslim'"):
        TrainConfig(bits=0, backbone_tier="w4a16", backbone_backend="angelslim")
    with pytest.raises(ValueError, match="tier_fake_quant requires"):
        TrainConfig(
            bits=0, backbone_tier="fp8_static", backbone_backend="angelslim", tier_fake_quant=True
        )
    with pytest.raises(ValueError, match="backbone_backend"):
        TrainConfig(bits=0, backbone_tier="bf16", backbone_backend="mxfp4")
    # torchao path is untouched (replacement step 2 deletes it)
    TrainConfig(bits=0, backbone_tier="w4a16", tier_fake_quant=True)
    with pytest.raises(ValueError, match="for backend 'torchao'"):
        TrainConfig(bits=0, backbone_tier="int4_gptq")


# --- real conversion (needs the angelslim install; GB10 runs it) ---


def test_real_gptq_fake_quant_conversion():
    pytest.importorskip("angelslim")
    torch.manual_seed(0)
    model = _tiny_llama()
    q_proj = model.model.layers[0].self_attn.q_proj
    weight_before = q_proj.weight.detach().clone()

    manifest = convert_backbone(model, object(), "int4_gptq", _calib_batches(n=2))

    # dequant_to_bf16: weights move onto the GPTQ grid but stay plain
    # Parameters in the original Linear modules — mountable and trainable.
    assert not torch.equal(weight_before, q_proj.weight.detach())
    assert isinstance(q_proj, torch.nn.Linear)
    assert isinstance(q_proj.weight, torch.nn.Parameter)
    assert q_proj.weight.dtype in (torch.float32, torch.bfloat16)
    assert manifest["format"] == "int4_gptq_fake_bf16"
    assert manifest["sequence"] == ["calibrate", "convert"]
