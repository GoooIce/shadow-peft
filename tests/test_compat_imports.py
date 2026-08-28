"""Import-path compatibility tests for the shadow_core refactor.

The refactor (extraction of ShadowConfig + harness into shadow_core, and the
quantization/ split into scheme/modules/walker) turned the old modules into
re-export shims. These tests pin every previously-public name to its OLD
import path so the claimed "public import surface unchanged" stays true.

They are intentionally lightweight: imports and identity assertions only,
no model construction.
"""

from __future__ import annotations

from typing import get_args

import pytest

import shadow_core.config
import shadow_core.harness.config
import shadow_core.harness.router
import shadow_core.harness.session

# --------------------------------------------------------------------------
# shadow_peft.config
# --------------------------------------------------------------------------

def test_shadow_config_importable_from_old_path():
    from shadow_peft.config import ShadowConfig

    assert ShadowConfig is shadow_core.config.ShadowConfig


# --------------------------------------------------------------------------
# shadow_peft.harness.{config,router,session}
# --------------------------------------------------------------------------

def test_harness_config_names_via_old_path():
    from shadow_peft.harness.config import (
        InferenceMode,
        RoutingRule,
        ShadowHarnessConfig,
        SpecialistConfig,
    )

    core = shadow_core.harness.config
    assert InferenceMode is core.InferenceMode
    assert RoutingRule is core.RoutingRule
    assert ShadowHarnessConfig is core.ShadowHarnessConfig
    assert SpecialistConfig is core.SpecialistConfig
    assert get_args(InferenceMode) == ("base_shadow", "shadow_only")


def test_harness_router_names_via_old_path():
    from shadow_peft.harness.router import (
        RouteDecision,
        Router,
        Session,
        ShadowHarnessConfig,
    )

    assert RouteDecision is shadow_core.harness.router.RouteDecision
    assert Router is shadow_core.harness.router.Router
    assert Session is shadow_core.harness.session.Session
    assert ShadowHarnessConfig is shadow_core.harness.config.ShadowHarnessConfig


def test_harness_session_name_via_old_path():
    from shadow_peft.harness.session import Session

    assert Session is shadow_core.harness.session.Session


# --------------------------------------------------------------------------
# shadow_peft.quantization.affine_{1,2}bit
# --------------------------------------------------------------------------

def test_affine_1bit_names_via_old_path():
    from shadow_peft.quantization import affine_1bit
    from shadow_peft.quantization.affine_1bit import (
        QuantizedEmbedding1Bit,
        QuantizedLinear1Bit,
        dequantize_1bit_affine,
        quantize_1bit_affine,
        quantize_model_1bit,
        save_quantized_checkpoint,
    )
    from shadow_peft.quantization.walker import save_quantized_checkpoint as canonical

    assert callable(quantize_1bit_affine)
    assert callable(dequantize_1bit_affine)
    assert callable(quantize_model_1bit)
    assert isinstance(QuantizedLinear1Bit, type)
    assert isinstance(QuantizedEmbedding1Bit, type)
    # save_quantized_checkpoint moved to walker; the old path must re-export
    # the very same function object.
    assert save_quantized_checkpoint is canonical
    assert affine_1bit.save_quantized_checkpoint is canonical


def test_affine_2bit_names_via_old_path():
    from shadow_peft.quantization.affine_2bit import (
        QuantizedEmbedding2Bit,
        QuantizedLinear2Bit,
        dequantize_2bit_ternary,
        quantize_2bit_ternary,
        quantize_model_2bit,
    )

    assert callable(quantize_2bit_ternary)
    assert callable(dequantize_2bit_ternary)
    assert callable(quantize_model_2bit)
    assert isinstance(QuantizedLinear2Bit, type)
    assert isinstance(QuantizedEmbedding2Bit, type)


def test_quantization_package_old_all_names():
    # Every name that was in quantization/__init__'s public surface pre-split.
    from shadow_peft.quantization import (
        QuantizedEmbedding1Bit,
        QuantizedLinear1Bit,
        affine_1bit,
    )

    assert QuantizedLinear1Bit is affine_1bit.QuantizedLinear1Bit
    assert QuantizedEmbedding1Bit is affine_1bit.QuantizedEmbedding1Bit


# --------------------------------------------------------------------------
# shadow_peft_mlx shims (MLX-gated, mirroring tests/mlx/conftest.py)
# --------------------------------------------------------------------------

def test_mlx_shadow_config_via_old_path():
    pytest.importorskip("mlx")
    from shadow_peft.config import ShadowConfig as TorchShadowConfig
    from shadow_peft_mlx.config import ShadowConfig as MlxShadowConfig

    assert MlxShadowConfig is TorchShadowConfig
    assert MlxShadowConfig is shadow_core.config.ShadowConfig


def test_mlx_harness_config_names_via_old_path():
    pytest.importorskip("mlx")
    from shadow_peft_mlx.harness.config import (
        InferenceMode,
        RoutingRule,
        ShadowHarnessConfig,
        SpecialistConfig,
    )

    core = shadow_core.harness.config
    assert InferenceMode is core.InferenceMode
    assert RoutingRule is core.RoutingRule
    assert ShadowHarnessConfig is core.ShadowHarnessConfig
    assert SpecialistConfig is core.SpecialistConfig


def test_mlx_harness_router_names_via_old_path():
    pytest.importorskip("mlx")
    from shadow_peft_mlx.harness.router import (
        RouteDecision,
        Router,
        Session,
        ShadowHarnessConfig,
    )

    assert RouteDecision is shadow_core.harness.router.RouteDecision
    assert Router is shadow_core.harness.router.Router
    assert Session is shadow_core.harness.session.Session
    assert ShadowHarnessConfig is shadow_core.harness.config.ShadowHarnessConfig


def test_mlx_harness_session_name_via_old_path():
    pytest.importorskip("mlx")
    from shadow_peft_mlx.harness.session import Session

    assert Session is shadow_core.harness.session.Session
