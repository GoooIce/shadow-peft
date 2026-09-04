"""Shadow-PEFT experiment lab with lazy top-level exports.

Keeping the package initializer dependency-free lets the local annotation app use
``shadow_lab.annotation`` without importing PyTorch or any model runtime.
"""

from importlib import import_module

_EXPORTS = {
    "LOSS_MODES": (".trainer", "LOSS_MODES"),
    "QWEN3_ENDOFTEXT_ID": (".atoms", "QWEN3_ENDOFTEXT_ID"),
    "QuasarNVFP4Config": (".quasar", "QuasarNVFP4Config"),
    "QuasarQADConfig": (".quasar_qad", "QuasarQADConfig"),
    "QuasarQADReport": (".quasar_qad", "QuasarQADReport"),
    "TrainConfig": (".trainer", "TrainConfig"),
    "TrainReport": (".trainer", "TrainReport"),
    "eval_ppl": (".atoms", "eval_ppl"),
    "load_teacher_data": (".atoms", "load_teacher_data"),
    "load_wikitext2_test_text": (".atoms", "load_wikitext2_test_text"),
    "make_batch": (".atoms", "make_batch"),
    "pick_device": (".atoms", "pick_device"),
    "prepare_quasar_nvfp4": (".quasar", "prepare_quasar_nvfp4"),
    "run_quasar_qad": (".quasar_qad", "run_quasar_qad"),
    "train_shadow": (".trainer", "train_shadow"),
}

__all__ = [
    "LOSS_MODES",
    "QWEN3_ENDOFTEXT_ID",
    "QuasarNVFP4Config",
    "QuasarQADConfig",
    "QuasarQADReport",
    "TrainConfig",
    "TrainReport",
    "eval_ppl",
    "load_teacher_data",
    "load_wikitext2_test_text",
    "make_batch",
    "pick_device",
    "prepare_quasar_nvfp4",
    "run_quasar_qad",
    "train_shadow",
]


def __getattr__(name: str):
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from error
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value
