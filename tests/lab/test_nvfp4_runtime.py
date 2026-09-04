"""Contracts for the truthful TorchAO NVFP4 runtime adapter."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from shadow_lab.nvfp4_runtime import (  # noqa: E402
    census_linear_layers,
    census_manifest,
    install_reconstructed_nvfp4_linear,
    materialize_native_nvfp4,
    native_kernel_evidence,
)
from shadow_lab.quasar import (  # noqa: E402
    QuasarNVFP4Config,
    reconstruct_nvfp4_with_clip,
)


class _CensusToy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.good = torch.nn.Linear(32, 48)
        self.bad_in = torch.nn.Linear(31, 48)
        self.bad_out = torch.nn.Linear(32, 15)
        self.skip = torch.nn.Linear(32, 32)


def test_census_records_explicit_eligibility_and_exclusions():
    rows = census_linear_layers(_CensusToy(), exclude_modules=("skip",))
    by_name = {row.name: row for row in rows}

    assert by_name["good"].eligible
    assert by_name["good"].reason is None
    assert not by_name["bad_in"].eligible
    assert by_name["bad_in"].reason == "weight dimensions are not both divisible by 16"
    assert not by_name["bad_out"].eligible
    assert by_name["skip"].reason == "excluded by module rule"

    manifest = census_manifest(rows)
    assert manifest["eligible_weight_parameters"] == 48 * 32
    assert manifest["eligible_bias_parameters"] == 48
    assert manifest["es_clip_dimensions_group_128"] == 48


def test_kernel_evidence_requires_both_block_scale_and_e2m1_operands():
    valid = [
        "cutlass3x_sm120_gemm_block_scaled_ue4m3xe2m1_ue4m3xe2m1_f32_bf16"
    ]
    assert native_kernel_evidence(valid)["native_compute"] is True
    assert native_kernel_evidence(["aten::_scaled_mm"])["native_compute"] is False
    assert native_kernel_evidence(["cutlass_tensorop_bf16"])["native_compute"] is False


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (12, 0),
    reason="native NVFP4 requires Blackwell CUDA",
)
def test_reconstructed_weight_bridges_to_torchao_without_qdq_drift():
    pytest.importorskip("torchao")
    torch.manual_seed(9)
    linear = torch.nn.Linear(128, 128, bias=False, device="cuda", dtype=torch.bfloat16)
    reconstruction = reconstruct_nvfp4_with_clip(
        linear.weight,
        torch.full((128, 1), 0.85, device="cuda"),
        config=QuasarNVFP4Config(
            reconstruction_group_size=128,
            quantize_activations=True,
        ),
    )
    expected = reconstruction.dequantize(torch.bfloat16)

    install_reconstructed_nvfp4_linear(linear, reconstruction)

    assert type(linear.weight).__name__ == "NVFP4Tensor"
    assert torch.equal(linear.weight.dequantize(torch.bfloat16), expected)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (12, 0),
    reason="native NVFP4 requires Blackwell CUDA",
)
def test_external_materialization_freezes_entire_root_and_reports_exact_clip_space():
    pytest.importorskip("torchao")

    class NativeToy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(
                128, 128, bias=True, device="cuda", dtype=torch.bfloat16
            )
            self.other = torch.nn.Parameter(
                torch.ones(3, device="cuda", dtype=torch.bfloat16)
            )

    root = NativeToy()
    report = materialize_native_nvfp4(
        root,
        scheme="external-clip",
        clip_by_layer={"linear": torch.full((128, 1), 0.85, device="cuda")},
    )

    assert type(root.linear.weight).__name__ == "NVFP4Tensor"
    assert report["trainable_parameters_before"] == 128 * 128 + 128 + 3
    assert report["trainable_parameters_after"] == 0
    assert report["census"]["es_clip_dimensions_group_128"] == 128
    assert report["clip_group_count"] == 128
    assert all(not parameter.requires_grad for parameter in root.parameters())
