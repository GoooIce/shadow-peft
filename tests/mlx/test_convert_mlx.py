"""
Cross-framework adapter conversion tests (torch <-> MLX).

Uses identical tiny llama architectures on both sides and transfers base weights
directly (safetensors is framework-neutral and HF/mlx-lm share key names), so a
converted adapter must produce matching logits on the other framework.
"""

from __future__ import annotations

import pytest

pytest.importorskip("mlx")
pytest.importorskip("mlx_lm")
torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

import mlx.core as mx  # noqa: E402
from mlx.utils import tree_flatten  # noqa: E402
from mlx_lm.models import llama  # noqa: E402
from safetensors.torch import load_file as st_load_file  # noqa: E402
from safetensors.torch import save_file as st_save_file  # noqa: E402

from shadow_peft import ShadowConfig as TorchShadowConfig  # noqa: E402
from shadow_peft import ShadowPeftModel as TorchShadowPeftModel  # noqa: E402
from shadow_peft import get_shadow_model as torch_get_shadow_model  # noqa: E402
from shadow_peft_mlx import ShadowConfig as MlxShadowConfig  # noqa: E402
from shadow_peft_mlx import ShadowPeftModel as MlxShadowPeftModel  # noqa: E402
from shadow_peft_mlx import get_shadow_model as mlx_get_shadow_model  # noqa: E402
from shadow_peft_mlx.convert import (  # noqa: E402
    convert_checkpoint,
    mlx_key_to_torch,
    torch_key_to_mlx,
)

_IDS = [[1, 5, 7, 9, 3]]
_CFG = {
    "num_shadow_layers": 1,
    "injection_hidden_size": 8,
    "gate_hidden_size": 10,
    "alpha": 0.1,
    "dropout": 0.0,
}


def _torch_tiny_llama(seed: int):
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(seed)
    cfg = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        tie_word_embeddings=False,
    )
    return LlamaForCausalLM(cfg).eval()


def _mlx_tiny_llama(seed: int):
    mx.random.seed(seed)
    args = llama.ModelArgs(
        model_type="llama",
        hidden_size=32,
        num_hidden_layers=3,
        intermediate_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        rms_norm_eps=1e-5,
        vocab_size=64,
        tie_word_embeddings=False,
    )
    model = llama.Model(args)
    model.eval()
    return model


def test_key_mapping_roundtrip():
    identical = [
        "shadow_model.layers.0.self_attn.q_proj.weight",
        "shadow_model.layers.1.mlp.gate_proj.weight",
        "shadow_model.norm.weight",
        "shadow_injection_model.injection_downs",
        "shadow_injection_model.injection_ups",
        "shadow_update_model.hidden_norm.weight",
        "shadow_update_model.hidden_norm.bias",
        "shadow_hidden_projection.weight",
        "lm_head.weight",
        "shadow_lm_head.weight",
    ]
    for key in identical:
        assert torch_key_to_mlx(key) == key
        assert mlx_key_to_torch(key) == key

    mapped = {
        "shadow_update_model.update_gates.0.0.weight": (
            "shadow_update_model.update_gates.0.layers.0.weight"
        ),
        "shadow_update_model.update_gates.1.2.weight": (
            "shadow_update_model.update_gates.1.layers.2.weight"
        ),
        "shadow_update_model.update_transforms.2.3.weight": (
            "shadow_update_model.update_transforms.2.layers.3.weight"
        ),
    }
    for torch_key, mlx_key in mapped.items():
        assert torch_key_to_mlx(torch_key) == mlx_key
        assert mlx_key_to_torch(mlx_key) == torch_key


def test_convert_torch_to_mlx_outputs_match(tmp_path):
    # torch side: capture base weights BEFORE wrapping (in-place wrapping changes
    # state_dict keys to "...layers.N.layer...."), then wrap, forward, save adapter.
    base_t = _torch_tiny_llama(seed=0)
    base_weights = {k: v.detach().clone() for k, v in base_t.state_dict().items()}
    peft_t = torch_get_shadow_model(base_t, TorchShadowConfig(**_CFG))
    peft_t.eval()
    with torch.no_grad():
        logits_t = peft_t(input_ids=torch.tensor(_IDS)).logits[0]

    torch_dir = tmp_path / "torch_ckpt"
    peft_t.save_pretrained(torch_dir)
    st_save_file(
        {k: v.contiguous() for k, v in base_weights.items()},
        str(tmp_path / "base.safetensors"),
    )

    convert_checkpoint(torch_dir, tmp_path / "mlx_ckpt", direction="torch_to_mlx")

    # mlx side: same-arch base with the torch base weights, then the converted adapter.
    base_m = _mlx_tiny_llama(seed=999)  # init values are overwritten below
    base_m.load_weights(list(mx.load(str(tmp_path / "base.safetensors")).items()), strict=False)
    base_m.eval()
    peft_m = MlxShadowPeftModel.from_pretrained(base_m, tmp_path / "mlx_ckpt")
    peft_m.eval()

    logits_m = peft_m(mx.array(_IDS))[0]
    maxdiff = mx.abs(logits_m - mx.array(logits_t.numpy())).max().item()
    assert maxdiff < 5e-3, f"torch->mlx converted adapter logits maxdiff {maxdiff:.2e}"


def test_convert_mlx_to_torch_outputs_match(tmp_path):
    # mlx side: capture base weights BEFORE wrapping (in-place wrapping changes
    # parameter keys to "...layers.N.layer...."), then wrap, forward, save adapter.
    base_m = _mlx_tiny_llama(seed=0)
    base_weights = dict(tree_flatten(base_m.parameters()))
    peft_m = mlx_get_shadow_model(base_m, MlxShadowConfig(**_CFG))
    peft_m.eval()
    logits_m = peft_m(mx.array(_IDS))[0]

    mlx_dir = tmp_path / "mlx_ckpt"
    peft_m.save_pretrained(mlx_dir)
    mx.save_safetensors(str(tmp_path / "base.safetensors"), base_weights)

    convert_checkpoint(mlx_dir, tmp_path / "torch_ckpt", direction="mlx_to_torch")

    # torch side: same-arch base with the mlx base weights, then the converted adapter.
    base_t = _torch_tiny_llama(seed=999)  # init values are overwritten below
    base_t.load_state_dict(st_load_file(str(tmp_path / "base.safetensors")), strict=False)
    base_t.eval()
    peft_t = TorchShadowPeftModel.from_pretrained(base_t, tmp_path / "torch_ckpt")
    peft_t.eval()

    with torch.no_grad():
        logits_t = peft_t(input_ids=torch.tensor(_IDS)).logits[0]
    maxdiff = mx.abs(mx.array(logits_t.numpy()) - logits_m).max().item()
    assert maxdiff < 5e-3, f"mlx->torch converted adapter logits maxdiff {maxdiff:.2e}"
