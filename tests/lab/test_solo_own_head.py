"""Own-head solo seam (exp1-solo2b).

With ``solo_own_head=True`` and an explicit shadow model, the trainer scores
the shadow-path CE through the SHADOW's own pretrained lm_head (not the
base-head copy); the head stays frozen (checkpoint contract unchanged).
Same-hidden shadows only — a shape mismatch must raise a clear error.
"""

from __future__ import annotations

import json
import math

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("tokenizers")

from shadow_lab.trainer import TrainConfig, train_shadow  # noqa: E402

VOCAB = 96


def _tiny_model(dirpath, hidden=32, seed=0):
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(
        vocab_size=VOCAB, hidden_size=hidden, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=128,
    )
    torch.manual_seed(seed)
    LlamaForCausalLM(cfg).save_pretrained(dirpath)
    return dirpath


@pytest.fixture(scope="module")
def tiny_base(tmp_path_factory):
    from tokenizers import Tokenizer
    from tokenizers import models as tk_models
    from tokenizers import pre_tokenizers as tk_pre
    from transformers import PreTrainedTokenizerFast

    d = _tiny_model(tmp_path_factory.mktemp("base"), seed=0)
    tk = Tokenizer(tk_models.WordLevel(vocab={f"w{i}": i for i in range(VOCAB)}, unk_token="w0"))
    tk.pre_tokenizer = tk_pre.Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=tk, pad_token="w0", eos_token="w1", bos_token="w2"
    ).save_pretrained(d)
    return d


@pytest.fixture(scope="module")
def tiny_shadow(tmp_path_factory):
    """Explicit shadow with a DIFFERENT seed (distinct head weights) but the
    same hidden size — the shape-clean own-head case."""
    return _tiny_model(tmp_path_factory.mktemp("shadow"), seed=7)


@pytest.fixture(scope="module")
def teacher_data(tmp_path_factory):
    torch.manual_seed(3)
    p = tmp_path_factory.mktemp("data") / "teacher.jsonl"
    rows = [torch.randint(2, VOCAB, (64,)).tolist() for _ in range(6)]
    p.write_text("".join(json.dumps({"ids": r}) + "\n" for r in rows))
    return str(p)


def _load_head(model_dir):
    from transformers import LlamaForCausalLM

    m = LlamaForCausalLM.from_pretrained(model_dir)
    return m.lm_head.state_dict()


def _run(base, shadow, data, tmp, **overrides):
    import transformers

    shadow_inst = transformers.AutoModelForCausalLM.from_pretrained(shadow, dtype=torch.bfloat16)
    cfg = TrainConfig(
        bits=0, backbone_tier="w4a16", tier_fake_quant=True, loss_mode="ce_kl",
        model=str(base), data=data, max_tokens=6 * 2 * 64, seq_len=64, batch_seq=2,
        lr=1e-4, shadow_layers=1, log_every=1, eval_every=3,
        save_dir=str(tmp / "train"), device="cpu", pad_id=0, **overrides,
    )
    return train_shadow(cfg, shadow_model=shadow_inst)


@pytest.mark.filterwarnings("ignore")
def test_own_head_training_and_checkpoint_contract(tiny_base, tiny_shadow, teacher_data, tmp_path):
    report = _run(tiny_base, tiny_shadow, teacher_data, tmp_path,
                  solo_own_head=True, shadow_loss_weight=1.0)
    assert report.steps >= 5
    assert math.isfinite(report.final_ppl_20chunks)

    # frozen heads never enter the checkpoint contract: the saved adapter must
    # carry shadow/injection/update weights only (the own head is re-applied
    # by the consumer at reload, like the runner's build_solo does)
    import safetensors.torch as st

    sd = st.load_file(str(tmp_path / "train" / "final" / "shadow_adapter.safetensors"))
    assert sd, "empty adapter checkpoint"
    assert not [k for k in sd if "lm_head" in k], [k for k in sd if "lm_head" in k]


@pytest.mark.filterwarnings("ignore")
def test_own_head_shape_mismatch_raises(tiny_base, teacher_data, tmp_path, tmp_path_factory):
    cross = _tiny_model(tmp_path_factory.mktemp("cross"), hidden=16, seed=1)
    with pytest.raises(ValueError, match="shape mismatch"):
        _run(tiny_base, cross, teacher_data, tmp_path, solo_own_head=True)
