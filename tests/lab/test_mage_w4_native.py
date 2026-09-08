"""Public contracts for the independent packed-weight runtime; CPU fixtures only."""
# ruff: noqa: E402 -- optional torch must be checked before torch-dependent imports.

from __future__ import annotations

import hashlib

import pytest

torch = pytest.importorskip("torch")

from shadow_lab.mage_qad_checkpoint import content_hash
from shadow_lab.mage_w4_search.native import NativeContractError, load_quantized_assets


def _write(path, value):
    torch.save({**value, "sha256": content_hash(value)}, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _files(tmp_path, *, change=None):
    layer = {
        "weight_packed": torch.full((16, 8), 0x21, dtype=torch.uint8),
        "weight_scale": torch.ones((16, 1), dtype=torch.float8_e4m3fn),
        "tensor_scale": torch.tensor(0.25, dtype=torch.float32),
        "input_tensor_scale": torch.tensor(0.5),
    }
    if change:
        change(layer)
    packed, remainder = tmp_path / "packed.pt", tmp_path / "remainder.pt"
    allowed = {
        str(packed): _write(packed, {"schema_version": 1, "packed": {"linear": layer}}),
        str(remainder): _write(
            remainder,
            {
                "linear.bias": torch.arange(16, dtype=torch.bfloat16),
                "linear.input_tensor_scale": torch.tensor(0.5),
            },
        ),
    }
    return packed, remainder, allowed


def test_verified_handoff_retains_packed_bytes_and_scale_identity(tmp_path):
    packed, remainder, allowed = _files(tmp_path)
    assets = load_quantized_assets(
        packed,
        remainder,
        allowed_assets=allowed,
        expected_layer_count=1,
        expected_shapes={"linear": (16, 16)},
    )
    assert assets.layers["linear"].shape == (16, 16)
    assert assets.layers["linear"].qdata.tolist() == [[0x21] * 8] * 16
    assert assets.layers["linear"].tensor_scale.item() == 0.25
    assert assets.source_files == allowed
    assert len(assets.sha256) == len(assets.fixed_sha256) == 64


@pytest.mark.parametrize(
    "field,value",
    [
        ("weight_packed", torch.zeros((16, 8), dtype=torch.int8)),
        ("weight_packed", torch.zeros((16, 7), dtype=torch.uint8)),
        ("weight_scale", torch.ones((16, 1), dtype=torch.float32)),
        ("weight_scale", torch.ones((16, 2), dtype=torch.float8_e4m3fn)),
        ("weight_scale", torch.full((16, 1), float("nan")).to(torch.float8_e4m3fn)),
        ("tensor_scale", torch.tensor(0.0)),
        ("tensor_scale", torch.tensor(float("inf"))),
    ],
)
def test_invalid_layout_or_scale_is_rejected_before_installation(tmp_path, field, value):
    packed, rest, allowed = _files(tmp_path, change=lambda layer: layer.update({field: value}))
    with pytest.raises(NativeContractError):
        load_quantized_assets(packed, rest, allowed_assets=allowed, expected_layer_count=1)


def test_handoff_requires_exact_layer_census_and_shapes(tmp_path):
    packed, rest, allowed = _files(tmp_path)
    with pytest.raises(NativeContractError, match="layer-count"):
        load_quantized_assets(packed, rest, allowed_assets=allowed)
    with pytest.raises(NativeContractError, match="shape"):
        load_quantized_assets(
            packed,
            rest,
            allowed_assets=allowed,
            expected_layer_count=1,
            expected_shapes={"linear": (32, 16)},
        )


def test_file_authorization_precedes_deserialization(tmp_path):
    packed, rest, allowed = _files(tmp_path)
    packed.write_bytes(b"not a checkpoint")
    with pytest.raises(NativeContractError, match="file-hash"):
        load_quantized_assets(packed, rest, allowed_assets=allowed, expected_layer_count=1)


def _model():
    model = torch.nn.Module()
    model.linear = torch.nn.Linear(16, 16, dtype=torch.bfloat16)
    return model


def _fixture_tensor(layer, *, device, dtype):
    # External tensor backend stand-in: never represents a native GPU test.
    return torch.full(layer.shape, 3, device=device, dtype=dtype)


def test_install_replaces_every_weight_and_restores_bias_without_activation_buffer(tmp_path):
    from shadow_lab.mage_w4_search.native import install_quantized_assets

    packed, rest, allowed = _files(tmp_path)
    assets = load_quantized_assets(packed, rest, allowed_assets=allowed, expected_layer_count=1)
    model = _model()
    report = install_quantized_assets(model, assets, device="cpu", tensor_factory=_fixture_tensor)
    assert model.linear.weight.unique().tolist() == [3]
    assert torch.equal(model.linear.bias, assets.remainder["linear.bias"])
    assert not hasattr(model.linear, "input_tensor_scale")
    assert not model.linear.weight.requires_grad
    assert report["native_compute"] is False
    assert report["activation_quantization"] == "dynamic_per_tensor_and_block16"
    assert report["ignored_calibration_buffers"] == ["linear.input_tensor_scale"]


def test_coverage_failure_leaves_structure_untouched(tmp_path):
    from shadow_lab.mage_w4_search.native import install_quantized_assets

    packed, rest, allowed = _files(tmp_path)
    assets = load_quantized_assets(packed, rest, allowed_assets=allowed, expected_layer_count=1)
    model = _model()
    model.register_buffer("uncovered", torch.ones(1))
    before = model.linear.weight.detach().clone()
    with pytest.raises(NativeContractError, match="remainder-coverage"):
        install_quantized_assets(model, assets, device="cpu", tensor_factory=_fixture_tensor)
    assert torch.equal(model.linear.weight, before)


def test_mutated_payload_cannot_reuse_old_identity(tmp_path):
    from shadow_lab.mage_w4_search.native import install_quantized_assets

    packed, rest, allowed = _files(tmp_path)
    assets = load_quantized_assets(packed, rest, allowed_assets=allowed, expected_layer_count=1)
    assets.layers["linear"].qdata[0, 0] = 1
    with pytest.raises(NativeContractError, match="center-hash"):
        install_quantized_assets(_model(), assets, device="cpu", tensor_factory=_fixture_tensor)


def test_center_save_reload_preserves_packed_and_fixed_content(tmp_path):
    from shadow_lab.mage_w4_search.native import save_quantized_assets

    packed, rest, allowed = _files(tmp_path)
    original = load_quantized_assets(packed, rest, allowed_assets=allowed, expected_layer_count=1)
    saved = save_quantized_assets(original, tmp_path / "center")
    loaded = load_quantized_assets(
        saved["packed_path"],
        saved["remainder_path"],
        allowed_assets=saved["file_sha256"],
        expected_layer_count=1,
    )
    assert loaded.sha256 == original.sha256
    assert loaded.fixed_sha256 == original.fixed_sha256
    assert torch.equal(loaded.layers["linear"].qdata, original.layers["linear"].qdata)


def test_generation_configuration_freezes_comparator_before_use():
    from shadow_lab.mage_w4_search.native import GenerationConfig

    config = GenerationConfig(watermark_key=123)
    assert (config.height, config.width, config.steps, config.cfg) == (512, 512, 4, 1.0)
    assert config.replay_max_abs_pixel_error == 0
    assert "watermark_key" not in config.identity()
    with pytest.raises(NativeContractError):
        GenerationConfig(watermark_key=123, replay_max_abs_pixel_error=-1)


class _FixturePipeline:
    def __init__(self, *, fail=False):
        self.model = torch.nn.Module()
        self.model.transformer = _model()
        self.fail = fail

    def generate(self, prompts, **kwargs):
        from PIL import Image

        if self.fail:
            raise RuntimeError("fixture generation failed")
        return [Image.new("RGB", (512, 512), color=(kwargs["seeds"][0], 2, 3))]


def _runtime(tmp_path, *, fail=False):
    from shadow_lab.mage_w4_search.native import (
        GenerationConfig,
        NativePipeline,
        install_quantized_assets,
    )

    packed, rest, allowed = _files(tmp_path)
    assets = load_quantized_assets(packed, rest, allowed_assets=allowed, expected_layer_count=1)
    pipeline = _FixturePipeline(fail=fail)
    receipt = install_quantized_assets(
        pipeline.model.transformer, assets, device="cpu", tensor_factory=_fixture_tensor
    )
    return NativePipeline(
        pipeline,
        assets,
        GenerationConfig(watermark_key=123),
        receipt,
        {"fixture": True},
        device="cpu",
        tensor_factory=_fixture_tensor,
    )


def test_save_reload_replay_compares_images_but_fixture_cannot_pass_native_gate(tmp_path):
    runtime = _runtime(tmp_path)
    report = runtime.replay(
        [{"sample_id": "a", "prompt": "a chair", "seed": 4}], tmp_path / "replay"
    )
    assert report["packed_reload_equal"] is True
    assert report["images_match"] is True
    assert report["comparisons"][0]["max_abs_pixel_error"] == 0
    assert report["native_compute"] is False
    assert report["qualified"] is False


def test_generation_failure_has_a_record_and_no_placeholder_success(tmp_path):
    runtime = _runtime(tmp_path, fail=True)
    report = runtime.generate(
        [{"sample_id": "a", "prompt": "a chair", "seed": 4}], tmp_path / "failed"
    )
    assert report["status"] == "failed"
    assert report["rows"][0]["status"] == "failed"
    assert "image_path" not in report["rows"][0]
    assert (tmp_path / "failed/report.json").is_file()


def test_ocr_generation_uses_only_its_five_complete_selection_prompts():
    from types import SimpleNamespace

    from scripts.mage_w4_native_probe import select_probe_records

    rows = [
        SimpleNamespace(
            prompt_id=str(i),
            development_uses=("ocr_selection",),
            target_text=("sign",),
            complete_target_inventory=True,
        )
        for i in range(5)
    ]
    rows.append(
        SimpleNamespace(
            prompt_id="development",
            development_uses=("candidate_development",),
            target_text=("sign",),
            complete_target_inventory=True,
        )
    )
    assert [
        r.prompt_id for r in select_probe_records(SimpleNamespace(records=rows), "generate")
    ] == list("01234")
    with pytest.raises(ValueError, match="five"):
        select_probe_records(SimpleNamespace(records=rows[:-2]), "generate")
