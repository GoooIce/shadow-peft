"""exp6 greedy layer-group schedule: capture/replay contract tests.

Pins the four load-bearing facts from docs/exp6-layer-group-recovery.md on a
tiny random-init Llama:

- replay parity: driving the wrapped layers from the captured frames
  reproduces the model's own forward bit-for-bit (the frames carry the
  family's mask/rotary math verbatim — nothing is reimplemented here);
- shadow-chain algebra: recomputing the prefix chain from the update modules
  equals the boundary states the wrapper evolved in the real forward;
- gradient flow: the local blind spot (``update[k]`` runs after layer k, so a
  local layer-k loss starves it) and its chain-recompute mitigation, plus the
  no-grad prefix invariant (out-of-group parameters see no gradient);
- the trainer wiring: ``layer_schedule='groups'`` runs greedy phases then the
  polish phase inside the same token budget and reports the schedule axes.
"""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("tokenizers")

from tokenizers import Tokenizer  # noqa: E402
from tokenizers import models as tk_models  # noqa: E402
from tokenizers import pre_tokenizers as tk_pre  # noqa: E402
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast  # noqa: E402

from shadow_lab.layerwise import (  # noqa: E402
    capture_frames,
    group_forward_loss,
    recompute_shadow_chain,
    replay_parity,
)
from shadow_lab.trainer import TrainConfig, train_shadow  # noqa: E402
from shadow_peft import ShadowConfig, get_shadow_model  # noqa: E402

VOCAB = 96
N_LAYERS = 4  # adapted base layers 1..3 -> multiple groups under group_size=2


def _tiny_cfg() -> LlamaConfig:
    return LlamaConfig(
        vocab_size=VOCAB,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=N_LAYERS,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=128,
    )


@pytest.fixture(scope="module")
def tiny_model_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("tiny-model")
    torch.manual_seed(0)
    LlamaForCausalLM(_tiny_cfg()).save_pretrained(d)
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
    p.write_text("".join(json.dumps({"ids": r}) + "\n" for r in rows))
    return str(p)


@pytest.fixture()
def mounted():
    torch.manual_seed(1)
    peft = get_shadow_model(
        LlamaForCausalLM(_tiny_cfg()),
        ShadowConfig(
            num_shadow_layers=1,
            injection_hidden_size=8,
            gate_hidden_size=4,
            dropout=0.0,
            alpha=0.1,
        ),
    )
    peft.eval()
    return peft


def _batch():
    torch.manual_seed(3)
    input_ids = torch.randint(2, VOCAB, (2, 16))
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    labels[:, -4:] = -100
    return input_ids, attention_mask, labels


def test_replay_parity(mounted):
    input_ids, attention_mask, _ = _batch()
    report = replay_parity(mounted, input_ids=input_ids, attention_mask=attention_mask)
    assert report["max_abs_diff"] < 1e-5, report


def test_shadow_chain_recompute_equals_forward(mounted):
    input_ids, attention_mask, _ = _batch()
    cap = capture_frames(mounted, input_ids=input_ids, attention_mask=attention_mask)
    for upto in (1, 2, 3):
        s = recompute_shadow_chain(mounted, cap, upto=upto)
        ref = cap.frames[upto - 1].shadow_out
        assert torch.allclose(s.detach(), ref, atol=1e-6), f"chain mismatch at layer {upto}"


def _grad_flow(recompute: bool):
    torch.manual_seed(1)
    peft = get_shadow_model(
        LlamaForCausalLM(_tiny_cfg()),
        ShadowConfig(
            num_shadow_layers=1,
            injection_hidden_size=8,
            gate_hidden_size=4,
            dropout=0.0,
            alpha=0.1,
        ),
    )
    peft.train()
    input_ids, attention_mask, labels = _batch()

    def step_once():
        cap = capture_frames(peft, input_ids=input_ids, attention_mask=attention_mask)
        targets = tuple(torch.randn_like(h) for h in cap.model_hidden_states)
        loss = group_forward_loss(
            peft, cap, group=(3, 3), teacher_hidden_states=targets,
            labels=labels, recompute_chain=recompute,
        )
        loss.backward()
        return cap, loss

    cap, _ = step_once()
    # ZeroInit fact: W_up starts at zero, so before the first optimizer step
    # every gradient upstream of the injection output (downs, shadow state,
    # recomputed chain) is exactly zero — only ups trains on step one.
    downs0 = peft.shadow_injection_model.injection_downs.grad
    ups0 = peft.shadow_injection_model.injection_ups.grad
    assert downs0[2].abs().sum() == 0 and ups0[2].abs().sum() > 0

    opt = torch.optim.SGD([p for p in peft.parameters() if p.requires_grad], lr=0.1)
    opt.step()
    opt.zero_grad()

    cap, loss = step_once()
    # Prefix ran under no_grad: boundary tensors must be detached constants.
    assert not cap.frames[2].hidden_out.requires_grad
    assert not cap.frames[2].shadow_out.requires_grad
    return peft


def test_group_grad_flow_recompute_on():
    peft = _grad_flow(recompute=True)
    for inj in (peft.shadow_injection_model.injection_downs.grad,
                peft.shadow_injection_model.injection_ups.grad):
        assert inj is not None
        assert inj[2].abs().sum() > 0, "group's own injection must train"
        assert inj[0].abs().sum() == 0 and inj[1].abs().sum() == 0, (
            "out-of-group injections must see no gradient"
        )
    for idx in (0, 1):  # prefix updates fed via the recomputed chain
        g = peft.shadow_update_model.update_transforms[idx][0].weight.grad
        assert g is not None and g.abs().sum() > 0, f"prefix update[{idx}] starved"
    # The blind spot, pinned: update of the group's LAST layer runs after the
    # loss target and receives no local gradient (polish-phase territory).
    g = peft.shadow_update_model.update_transforms[2][0].weight.grad
    assert g is None or g.abs().sum() == 0


def test_group_grad_flow_recompute_off():
    peft = _grad_flow(recompute=False)
    ups = peft.shadow_injection_model.injection_ups.grad
    assert ups is not None and ups[2].abs().sum() > 0
    assert ups[0].abs().sum() == 0 and ups[1].abs().sum() == 0
    for idx in range(3):
        g = peft.shadow_update_model.update_transforms[idx][0].weight.grad
        assert g is None or g.abs().sum() == 0, (
            f"without chain recompute, update[{idx}] must be left to polish"
        )


def test_train_shadow_layerwise(tiny_model_dir, teacher_data, tmp_path):
    cfg = TrainConfig(
        bits=0,
        backbone_tier="w4a16",
        tier_fake_quant=True,
        loss_mode="ce_kl",
        layer_schedule="groups",
        group_size=2,
        polish_token_frac=0.3,
        model=str(tiny_model_dir),
        data=teacher_data,
        max_tokens=6 * 2 * 64,  # ~10 steps total across greedy + polish
        seq_len=64,
        batch_seq=2,
        lr=1e-4,
        shadow_layers=1,
        log_every=1,
        eval_every=3,
        save_dir=str(tmp_path / "run"),
    )
    report = train_shadow(cfg)

    assert report.phase_seconds is not None
    assert set(report.phase_seconds) == {"greedy", "polish"}
    assert report.layer_attribution is not None
    assert len(report.layer_attribution["init"]) == N_LAYERS
    assert len(report.layer_attribution["final"]) == N_LAYERS
    # The blind-spot mitigation must actually move the update modules during
    # the greedy phase (default recompute=True), not leave them at init.
    assert report.layer_attribution["final"] != report.layer_attribution["init"]

    log_lines = [json.loads(x) for x in (tmp_path / "run" / "train_log.jsonl").read_text().splitlines()]
    phases = {entry.get("phase") for entry in log_lines}
    assert "greedy1-2" in phases and "greedy3-3" in phases and "polish" in phases
    assert report.tokens >= cfg.max_tokens

    summary = json.loads((tmp_path / "run" / "summary.json").read_text())
    assert summary["layer_attribution"] == report.layer_attribution
    assert "layer_schedule" in summary["args"] and summary["args"]["layer_schedule"] == "groups"
