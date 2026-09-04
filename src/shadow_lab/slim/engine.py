"""Shared AngelSlim engine base for the shadow_lab compression bridge.

Internal seam of ``shadow_lab.slim`` — not exported on the package surface.
It owns every fact about driving ``angelslim.engine.Engine`` so the
capability leaves (ptq / qad / distill / sparsity / specdec) stay thin:

* **In-memory models.** Engine's path-based conveniences (GlobalConfig auto
  fill from ``model_path``) are bypassed: the caller's loaded model is
  injected via ``prepare_model(model=..., tokenizer=...)`` and the global
  config is populated from the model itself (``hidden_size``) and the
  calibration batches (``max_seq_length``). ``GlobalConfig.update`` is a
  no-op when both args are None, so nothing Engine does later can clobber
  the fields we filled (in-memory models never set ``Engine.model_path``).
* **Family resolution.** ``config.model_type`` → SlimModelFactory registry
  name. Llama-isomorphic families AngelSlim does not register upstream
  (Gemma, LFM2 — same projection layout as Llama) get a generic
  ``GenericLlamaLike`` wrapper registered lazily on first use.
* **Calibration adaptation.** shadow-peft's ``calib_batches``
  (``{input_ids, attention_mask}`` dicts) are wrapped into the loader
  contract AngelSlim's calibration loops consume: ``len()`` plus dicts of
  ``{input_ids, labels, attention_mask}``. Labels are synthesized from
  input_ids — the loss in the calibration forward is logging only, the
  GPTQ Hessian is captured by forward hooks.

Everything angelslim-touching imports lazily: trainer and the pure-logic
tests must not require the dependency.
"""

from __future__ import annotations

import torch

# model_type prefix → SlimModelFactory registry name. Order matters: longer
# prefixes first (glm5 before glm).
_FAMILY_PREFIXES: tuple[tuple[str, str], ...] = (
    ("qwen", "Qwen"),
    ("llama", "Llama"),
    ("glm5", "GLM5_1"),
    ("glm", "GLM"),
    ("deepseek", "DeepSeek"),
    ("kimi", "KimiK2"),
    ("seed_oss", "SeedOss"),
)

# Llama-isomorphic families AngelSlim does not register upstream: same
# block name ("model.layers") and the same attention/MLP projection layout
# as Llama, so a generic wrapper suffices for weight-only PTQ walking.
_LLAMA_LIKE_PREFIXES = ("gemma", "lfm2")

GENERIC_LLAMA_LIKE = "GenericLlamaLike"

_LLM_OBSERVER_SUFFIXES = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.up_proj",
    "mlp.gate_proj",
    "mlp.down_proj",
)

_generic_llama_like_registered = False


def resolve_family_by_type(model_type: str) -> str:
    """Map a ``config.model_type`` string to its SlimModelFactory registry name.

    Pure table lookup (no angelslim import): registered families resolve to
    their upstream wrapper; Llama-isomorphic but unregistered families
    (Gemma, LFM2) resolve to ``GenericLlamaLike``. Anything else raises —
    pass ``model_name=`` explicitly at the call site.
    """
    model_type = str(model_type or "").lower()
    for prefix, name in _FAMILY_PREFIXES:
        if model_type.startswith(prefix):
            return name
    for prefix in _LLAMA_LIKE_PREFIXES:
        if model_type.startswith(prefix):
            return GENERIC_LLAMA_LIKE
    known = [p for p, _ in _FAMILY_PREFIXES] + list(_LLAMA_LIKE_PREFIXES)
    raise ValueError(
        f"cannot resolve AngelSlim family for model_type={model_type!r} "
        f"(known prefixes: {known}) — pass model_name= explicitly"
    )


def resolve_family(base) -> str:
    """``resolve_family_by_type`` on an in-memory model's config."""
    return resolve_family_by_type(getattr(base.config, "model_type", ""))


class CalibLoader:
    """Adapt shadow-peft calib batches to AngelSlim's dataloader contract.

    AngelSlim's calibration loops iterate the dataloader with ``len()`` and
    read ``batch["input_ids"] / ["labels"] / ["attention_mask"]``. The
    synthesized labels only feed the calibration-forward loss printout; the
    quantization statistics travel through forward hooks, so labels=input_ids
    is sound.
    """

    def __init__(self, batches: list[dict[str, torch.Tensor]]):
        self._batches = list(batches)

    def __len__(self) -> int:
        return len(self._batches)

    def __iter__(self):
        for batch in self._batches:
            input_ids = batch["input_ids"]
            yield {
                "input_ids": input_ids,
                "labels": input_ids.clone(),
                "attention_mask": batch.get("attention_mask", torch.ones_like(input_ids)),
            }


def _register_generic_llama_like() -> None:
    """Register the generic Llama-isomorphic wrapper once (lazy, idempotent)."""
    global _generic_llama_like_registered
    if _generic_llama_like_registered:
        return

    from angelslim.models.base_model import BaseLLMModel
    from angelslim.models.model_factory import SlimModelFactory
    from angelslim.utils.utils import find_layers

    class GenericLlamaLike(BaseLLMModel):
        """Llama projection layout for families upstream does not register.

        ``get_observer_layers`` mirrors AngelSlim's own Llama wrapper: the
        decoder-block prefix plus the seven projection suffixes. Layer-name
        overrides ride the standard ``custom_observe_layers_names`` config.
        """

        def __init__(self, model=None, deploy_backend="vllm"):
            super().__init__(model=model, deploy_backend=deploy_backend)
            self.block_name = "model.layers"

        def get_observer_layers(self):
            layers = find_layers(self.model, layers=self.observer_layer_classes)
            layers = {
                k: v
                for k, v in layers.items()
                if k.startswith(self.block_name)
                and k.split(".")[-2] + "." + k.split(".")[-1] in _LLM_OBSERVER_SUFFIXES
            }
            if self.quant_config.custom_observe_layers_names != "default":
                for custom_name in self.quant_config.custom_observe_layers_names:
                    for default_name in list(layers.keys()):
                        if custom_name not in default_name:
                            layers.pop(default_name)
            return layers

        def get_save_func(self):
            from angelslim.compressor.quant.core import PTQSaveVllmHF

            return PTQSaveVllmHF

    SlimModelFactory.register(GenericLlamaLike)
    _generic_llama_like_registered = True


def build_engine(base, tokenizer, *, model_name: str | None = None, deploy_backend="huggingface"):
    """Return an Engine with the in-memory ``base`` already prepared.

    ``deploy_backend`` defaults to ``"huggingface"``: this bridge serves the
    training/eval seam, and the HF-format save keeps checkpoints loadable by
    plain transformers (vllm still accepts them; the difference only names
    the save function).
    """
    from angelslim.engine import Engine

    name = model_name or resolve_family(base)
    if name == GENERIC_LLAMA_LIKE:
        _register_generic_llama_like()

    engine = Engine()
    engine.prepare_model(
        model_name=name,
        model=base,
        tokenizer=tokenizer,
        deploy_backend=deploy_backend,
    )
    return engine


def build_global_config(
    base,
    calib_batches: list[dict[str, torch.Tensor]] | None = None,
    *,
    save_path: str = "./output",
    deploy_backend: str = "huggingface",
):
    """Populate GlobalConfig for an in-memory model.

    The path-based ``GlobalConfig.update`` is bypassed entirely: hidden size
    comes from the model config, max_seq_length from the longest calib batch
    (GPTQ Hessian buffers are sized by it). ``absolute_model_path`` stays
    None — it is only read to re-materialize meta-device weights, which an
    in-memory model never has.
    """
    from angelslim.utils.config_parser import GlobalConfig

    seq = 2048
    if calib_batches:
        seq = max(int(b["input_ids"].shape[-1]) for b in calib_batches)
    architectures = getattr(base.config, "architectures", None)
    return GlobalConfig(
        save_path=save_path,
        max_seq_length=seq,
        hidden_size=int(getattr(base.config, "hidden_size", 4096)),
        model_arch_type=architectures[0] if architectures else None,
        deploy_backend=deploy_backend,
    )


def drive_campaign(
    compress_name: str,
    model_path: str,
    data_path: str,
    out_dir: str,
    compress_config,
    *,
    model_name: str | None = None,
    deploy_backend: str = "huggingface",
    max_seq_length: int = 2048,
    num_samples: int = 128,
    batch_size: int = 1,
) -> dict:
    """Drive a path-based training campaign (QAD / Distill): load → train → convert → save.

    Path-based flows let Engine fill GlobalConfig itself — ``prepare_model``
    records the model path and ``prepare_data`` the max length, so the
    ``GlobalConfig.update`` call inside ``prepare_compressor`` does the fill.
    The caller supplies the save path and the compression config; nothing
    else about the campaign is this bridge's business.
    """
    from angelslim.engine import Engine
    from angelslim.utils.config_parser import GlobalConfig

    if model_name is None:
        from transformers import AutoConfig

        model_name = resolve_family_by_type(AutoConfig.from_pretrained(model_path).model_type)
    if model_name == GENERIC_LLAMA_LIKE:
        _register_generic_llama_like()

    engine = Engine()
    engine.prepare_model(
        model_name=model_name, model_path=model_path, deploy_backend=deploy_backend
    )
    engine.prepare_data(
        data_path=data_path,
        data_type="TextDataset",
        max_length=max_seq_length,
        batch_size=batch_size,
        num_samples=num_samples,
    )
    engine.prepare_compressor(
        compress_name,
        global_config=GlobalConfig(save_path=out_dir, deploy_backend=deploy_backend),
        compress_config=compress_config,
    )
    engine.run()
    engine.convert()
    engine.save(out_dir)
    return {
        "backend": "angelslim",
        "compress": compress_name,
        "model_name": model_name,
        "save_dir": out_dir,
        "sequence": ["prepare", "train", "convert", "save"],
    }
