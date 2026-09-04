"""Pure-Torch contract tests for QUASAR's first-phase NVFP4 implementation."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from shadow_lab import quasar_qad  # noqa: E402
from shadow_lab.quasar import (  # noqa: E402
    E2M1_VALUES,
    PackedNVFP4Linear,
    QuasarNVFP4Config,
    QuasarNVFP4Linear,
    e2m1_decode,
    e2m1_encode,
    materialize_quasar_nvfp4,
    nvfp4_clip_shape,
    pack_e2m1,
    prepare_quasar_nvfp4,
    reconstruct_nvfp4,
    reconstruct_nvfp4_with_clip,
    unpack_e2m1,
)
from shadow_lab.quasar_qad import (  # noqa: E402
    QuasarQADConfig,
    forward_kl_loss,
)


def test_e2m1_round_half_to_even_and_pack_layout():
    midpoints = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
    codes = e2m1_encode(midpoints)
    assert codes.tolist() == [0, 2, 2, 4, 4, 6, 6]
    assert e2m1_decode(codes).tolist() == [0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0]

    all_codes = torch.arange(16, dtype=torch.uint8).reshape(2, 8)
    packed = pack_e2m1(all_codes)
    assert packed[0, 0].item() == 0x10  # even index = low nibble
    assert torch.equal(unpack_e2m1(packed), all_codes)
    assert e2m1_decode(torch.arange(16, dtype=torch.uint8)).tolist() == list(E2M1_VALUES)


def test_search_never_exceeds_full_range_weighted_error():
    torch.manual_seed(4)
    weight = torch.randn(3, 37)
    saliency = torch.rand_like(weight).square()
    full_range = reconstruct_nvfp4(
        weight, saliency, QuasarNVFP4Config(clip_factors=(1.0,), quantize_activations=False)
    )
    searched = reconstruct_nvfp4(
        weight, saliency, QuasarNVFP4Config(quantize_activations=False)
    )
    assert searched.weighted_error <= full_range.weighted_error + 1e-7
    assert searched.dequantize().shape == weight.shape
    assert searched.weight_scale.dtype == torch.float8_e4m3fn
    assert searched.weight_packed.dtype == torch.uint8


def test_reconstruction_group_shares_one_grid_choice_across_eight_nvfp4_blocks():
    torch.manual_seed(41)
    weight = torch.randn(2, 256)
    config = QuasarNVFP4Config(
        reconstruction_group_size=128,
        quantize_activations=False,
    )

    searched = reconstruct_nvfp4(weight, config=config)
    full_range = reconstruct_nvfp4(
        weight,
        config=QuasarNVFP4Config(
            clip_factors=(1.0,),
            reconstruction_group_size=128,
            quantize_activations=False,
        ),
    )

    assert nvfp4_clip_shape(weight.shape, 128) == (2, 2)
    assert searched.clip_factor_index.shape == (2, 2)
    assert searched.clip_factor_value.shape == (2, 2)
    assert searched.weight_scale.shape == (2, 16)
    assert searched.weighted_error <= full_range.weighted_error + 1e-7


def test_external_clip_vector_is_deterministic_and_one_matches_rtn():
    torch.manual_seed(42)
    weight = torch.randn(2, 256)
    factors = torch.tensor([[0.30, 0.65], [0.85, 1.00]])
    config = QuasarNVFP4Config(
        reconstruction_group_size=128,
        quantize_activations=False,
    )

    first = reconstruct_nvfp4_with_clip(weight, factors, config=config)
    second = reconstruct_nvfp4_with_clip(weight, factors.flatten(), config=config)
    rtn = reconstruct_nvfp4_with_clip(weight, 1.0, config=config)
    grid_rtn = reconstruct_nvfp4(
        weight,
        config=QuasarNVFP4Config(
            clip_factors=(1.0,),
            reconstruction_group_size=128,
            quantize_activations=False,
        ),
    )

    assert torch.equal(first.codes, second.codes)
    assert torch.equal(first.weight_scale, second.weight_scale)
    assert torch.equal(first.clip_factor_value, factors)
    assert torch.equal(first.clip_factor_index, torch.full((2, 2), -1, dtype=torch.int16))
    assert torch.equal(rtn.codes, grid_rtn.codes)
    assert torch.equal(rtn.weight_scale, grid_rtn.weight_scale)


def test_external_clip_vector_rejects_values_outside_es_bounds():
    weight = torch.randn(1, 128)
    config = QuasarNVFP4Config(
        reconstruction_group_size=128,
        quantize_activations=False,
    )
    with pytest.raises(ValueError, match=r"\[0.30, 1.00\]"):
        reconstruct_nvfp4_with_clip(weight, torch.tensor([0.29]), config=config)


def test_adam_saliency_changes_the_loss_aware_solution():
    torch.manual_seed(0)
    weight = torch.randn(1, 16)
    uniform = torch.ones_like(weight)
    salient_first = uniform.clone()
    salient_first[0, 0] = 100_000.0

    plain = reconstruct_nvfp4(weight, uniform)
    loss_aware = reconstruct_nvfp4(weight, salient_first)

    plain_first_error = (plain.dequantize()[0, 0] - weight[0, 0]).abs()
    aware_first_error = (loss_aware.dequantize()[0, 0] - weight[0, 0]).abs()
    assert aware_first_error < plain_first_error
    assert not torch.equal(plain.clip_factor_index, loss_aware.clip_factor_index)


def test_training_wrapper_uses_ste_for_latent_weight():
    torch.manual_seed(1)
    linear = torch.nn.Linear(17, 3, bias=False)
    raw_weight = linear.weight.detach().clone()
    wrapped = QuasarNVFP4Linear(
        linear,
        QuasarNVFP4Config(clip_factors=(1.0,), quantize_activations=False),
    )
    value = torch.randn(2, 17)

    wrapped(value).sum().backward()
    expected_gradient = value.sum(dim=0).expand_as(raw_weight)

    assert torch.allclose(wrapped.weight.grad, expected_gradient)
    assert torch.equal(wrapped.weight.detach(), raw_weight)  # latent master did not snap to grid


class _ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(32, 16)
        self.block = torch.nn.Linear(16, 16)
        self.lm_head = torch.nn.Linear(16, 32, bias=False)
        self.config = SimpleNamespace()

    def forward(self, value):
        return self.lm_head(torch.relu(self.block(value)))


def test_prepare_binds_live_adam_second_moment_and_freezes_non_quantized():
    model = _ToyModel()
    controller = prepare_quasar_nvfp4(model)
    assert list(controller.layers) == ["block"]
    assert isinstance(model.block, QuasarNVFP4Linear)
    assert model.block.weight.requires_grad
    assert not model.block.bias.requires_grad
    assert not model.embed.weight.requires_grad
    assert not model.lm_head.weight.requires_grad

    optimizer = torch.optim.AdamW(controller.parameters(), lr=1e-3)
    controller.bind_optimizer(optimizer)
    assert controller.saliency_layer_count == 0  # uniform first-step fallback

    loss = model(torch.randn(2, 16)).square().mean()
    loss.backward()
    optimizer.step()

    exp_avg_sq = optimizer.state[model.block.weight]["exp_avg_sq"]
    assert controller.saliency_layer_count == 1
    assert model.block._saliency is exp_avg_sq  # reused, not copied into another weight-sized buffer


def test_materialized_module_matches_fake_quant_forward_and_state_contract():
    torch.manual_seed(2)
    model = _ToyModel().eval()
    controller = prepare_quasar_nvfp4(model)
    value = torch.randn(2, 16)
    expected = model(value)

    manifest = materialize_quasar_nvfp4(controller)
    actual = model(value)

    assert isinstance(model.block, PackedNVFP4Linear)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)
    assert manifest["algorithm"] == "quasar"
    assert manifest["format"] == "nvfp4-pack-quantized"
    # One block per output row in the one wrapped 16x16 Linear.
    assert sum(manifest["clip_factor_counts"]) == model.block.out_features
    assert model.config.quantization_config["format"] == "nvfp4-pack-quantized"
    state = model.state_dict()
    assert "block.weight" not in state
    assert state["block.weight_packed"].dtype == torch.uint8
    assert state["block.weight_scale"].dtype == torch.float8_e4m3fn
    assert state["block.weight_global_scale"].dtype == torch.float32
    assert state["block.input_global_scale"].dtype == torch.float32


def test_forward_kl_is_zero_for_identical_logits_and_masks_padding():
    torch.manual_seed(3)
    teacher = torch.randn(2, 4, 7)
    student = teacher.clone()
    student[0, 2] += 100.0  # predicts the masked fourth token; must not contribute
    attention_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]])
    assert forward_kl_loss(student, teacher, attention_mask).abs() < 1e-7


def test_first_phase_qad_rejects_non_nvfp4_activation_path():
    with pytest.raises(ValueError, match="native NVFP4 W4A4"):
        QuasarQADConfig(
            model="unused",
            data="unused",
            save_dir="unused",
            nvfp4=QuasarNVFP4Config(quantize_activations=False),
        )


class _FusedToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = torch.nn.ModuleDict(
            {
                "q_proj": torch.nn.Linear(16, 16, bias=False),
                "k_proj": torch.nn.Linear(16, 16, bias=False),
                "v_proj": torch.nn.Linear(16, 16, bias=False),
            }
        )

    def forward(self, value):
        return sum(layer(value) for layer in self.attn.values())


def test_materialization_harmonizes_fused_projection_global_scales():
    model = _FusedToyModel().eval()
    with torch.no_grad():
        model.attn["q_proj"].weight.mul_(0.2)
        model.attn["v_proj"].weight.mul_(4.0)
    controller = prepare_quasar_nvfp4(model)
    model(torch.randn(2, 16))  # initialize the shared-input activation scales

    manifest = materialize_quasar_nvfp4(controller)

    globals_ = [model.attn[name].weight_global_scale for name in ("q_proj", "k_proj", "v_proj")]
    assert torch.equal(globals_[0], globals_[1])
    assert torch.equal(globals_[1], globals_[2])
    fusion = manifest["fusion_scale_harmonization"]
    assert fusion["fused_group_count"] == 1
    assert fusion["fused_layer_count"] == 3


class _TinyCausalLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(32, 16)
        self.block = torch.nn.Linear(16, 16, bias=False)
        self.lm_head = torch.nn.Linear(16, 32, bias=False)
        self.config = SimpleNamespace(use_cache=True)

    def forward(self, input_ids, attention_mask=None):
        del attention_mask
        return SimpleNamespace(logits=self.lm_head(torch.tanh(self.block(self.embed(input_ids)))))

    def save_pretrained(self, output_dir, max_shard_size=None):
        del max_shard_size
        torch.save(self.state_dict(), output_dir / "model.pt")


class _TinyModelLoader:
    @staticmethod
    def from_pretrained(_path, dtype=None):
        torch.manual_seed(19)
        return _TinyCausalLM().to(dtype=dtype)


class _TinyTokenizer:
    def save_pretrained(self, output_dir):
        (output_dir / "tokenizer.stub").write_text("stub")


class _TinyTokenizerLoader:
    @staticmethod
    def from_pretrained(_path):
        return _TinyTokenizer()


def test_one_step_qad_campaign_reaches_packed_checkpoint(monkeypatch, tmp_path):
    monkeypatch.setattr(quasar_qad, "AutoModelForCausalLM", _TinyModelLoader)
    monkeypatch.setattr(quasar_qad, "AutoTokenizer", _TinyTokenizerLoader)
    monkeypatch.setattr(
        quasar_qad,
        "load_teacher_data",
        lambda _path: [[1, 2, 3, 4], [4, 3, 2, 1]],
    )
    output_dir = tmp_path / "quasar"

    report = quasar_qad.run_quasar_qad(
        QuasarQADConfig(
            model="tiny",
            data="tokens.jsonl",
            save_dir=str(output_dir),
            steps=1,
            seq_len=4,
            batch_seq=1,
            gradient_accumulation_steps=1,
            dtype=torch.float32,
            device="cpu",
            log_every=1,
        )
    )

    assert report.quantized_layers == 1
    assert report.saliency_layers == 1
    assert report.format == "nvfp4-pack-quantized"
    assert (output_dir / "model.pt").is_file()
    manifest = json.loads((output_dir / "quasar_manifest.json").read_text())
    assert manifest["quantization"]["algorithm"] == "quasar"
