"""6-cell eval-grid harness tests (ticket 05).

The tiny end-to-end run validates metric plumbing and JSON shape on CPU
(peak-memory stats are CUDA-only and recorded as null with a device note);
the 5090 run fills real peaks. The stub tokenizer satisfies the harness's
only tokenizer contract (``encode(text) -> list[int]``).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
torchao = pytest.importorskip("torchao")  # noqa: F841

_EXPERIMENT = Path(__file__).resolve().parents[1] / "experiment"
sys.path.insert(0, str(_EXPERIMENT))

from lab import grid, tiers  # noqa: E402

from shadow_peft import ShadowConfig, get_shadow_model  # noqa: E402
from shadow_peft.quantization import QuantizedLinear8Bit  # noqa: E402

VOCAB = 128


class StubTokenizer:
    """encode maps chars to ids in [1, VOCAB); decode inverts."""

    def encode(self, text: str) -> list[int]:
        return [ord(c) % (VOCAB - 1) + 1 for c in text]


def _decode(ids: list[int]) -> str:
    return "".join(chr((i - 1) % 90 + 32) for i in ids if 0 < i < VOCAB)


def _tiny_llama(num_layers: int = 4):
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(
        vocab_size=VOCAB,
        hidden_size=128,  # clears torchao Int4 groupsize + 8-bit walker group
        intermediate_size=256,
        num_hidden_layers=num_layers,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
    )
    # bf16 end-to-end: the packed shadow modules and torchao tiers both
    # follow the model dtype, and fp32/fp16 mixing is what the real runs
    # never do.
    return LlamaForCausalLM(cfg).to(torch.bfloat16)


def _shadow_cfg() -> ShadowConfig:
    return ShadowConfig(
        num_shadow_layers=1,
        injection_hidden_size=8,
        gate_hidden_size=10,
        alpha=0.1,
        dropout=0.0,
    )


@pytest.fixture(scope="module")
def master():
    torch.manual_seed(0)
    m = _tiny_llama()
    return {k: v.detach().clone() for k, v in m.state_dict().items()}


@pytest.fixture(scope="module")
def adapter_state(master):
    base = _tiny_llama()
    base.load_state_dict(master)
    peft = get_shadow_model(base, _shadow_cfg())
    return peft.adapter_state_dict()


EVAL_TEXT = "the quick brown fox jumps over the lazy dog. " * 40


def _kld_batches():
    return [
        {
            "input_ids": torch.randint(1, VOCAB, (1, 24)),
            "attention_mask": torch.ones(1, 24, dtype=torch.long),
        }
        for _ in range(2)
    ]


def _gen_examples():
    return [{"prompt_ids": [5, 6, 7, 8, 9], "answer": "A"} for _ in range(2)]


def test_evaluate_cell_all_metrics_present(master, adapter_state):
    def factory():
        m = _tiny_llama()
        m.load_state_dict(master)
        return m

    cell = grid.evaluate_cell(
        base=factory(),
        shadow_config=_shadow_cfg(),
        adapter_state=adapter_state,
        backbone_tier="w4a16",
        shadow_precision="int8",
        kernel_path="fake_quant",
        tokenizer=StubTokenizer(),
        eval_text=EVAL_TEXT,
        device=torch.device("cpu"),
        ref_model=None,
        kld_batches=_kld_batches(),
        mmlu_examples=_gen_examples(),
        gsm8k_examples=_gen_examples(),
        decode=_decode,
    )
    assert cell["cell"] == {
        "backbone_tier": "w4a16",
        "shadow_precision": "int8",
        "kernel_path": "fake_quant",
    }
    assert isinstance(cell["backbone_format"], str)
    assert cell["fake_quant_swaps"] > 0  # torchao weights were dequantized
    assert cell["shadow_ptq_modules"] > 0  # shadow walker replaced Linears
    assert cell["ppl"] > 0 and torch.isfinite(torch.tensor(cell["ppl"]))
    assert cell["mmlu"]["n"] == 2 and cell["gsm8k"]["n"] == 2
    assert cell["kld_vs_baseline"] is None  # no ref model passed
    mem = cell["memory"]
    assert mem["weight_residency_bytes"] > 0
    assert mem["total_peak_allocated_bytes"] is None  # CPU: CUDA-only stat
    assert "CUDA-only" in mem["note"]
    assert cell["vocab"] == {"embedding_rows": VOCAB, "trimmed": False}
    json.dumps(cell)  # JSON-serializable


def test_shadow_int8_ptq_replaces_shadow_linears(master):
    base = _tiny_llama()
    base.load_state_dict(master)
    peft = get_shadow_model(base, _shadow_cfg())
    manifest = grid.ptq_shadow_int8(peft)
    assert len(manifest["quantized"]) > 0
    assert any(
        isinstance(m, QuantizedLinear8Bit) for _, m in peft.shadow_model.named_modules()
    )


def test_kld_zero_for_identical_models(master, adapter_state):
    def make():
        b = _tiny_llama()
        b.load_state_dict(master)
        p = get_shadow_model(b, _shadow_cfg())
        p.load_adapter_state_dict(adapter_state)
        return p

    a, b = make(), make()
    assert grid.eval_kld(a, b, _kld_batches()) == pytest.approx(0.0, abs=1e-9)
    # A different-weight model must diverge.
    torch.manual_seed(1)
    other = _tiny_llama()
    assert grid.eval_kld(a, other, _kld_batches()) > 0.0


def test_run_grid_tiny_end_to_end(master, adapter_state):
    def factory():
        m = _tiny_llama()
        m.load_state_dict(master)
        return m

    result = grid.run_grid(
        base_factory=factory,
        shadow_config=_shadow_cfg(),
        adapter_state=adapter_state,
        tokenizer=StubTokenizer(),
        eval_text=EVAL_TEXT,
        device=torch.device("cpu"),
        kld_batches=_kld_batches(),
        mmlu_examples=_gen_examples(),
        gsm8k_examples=_gen_examples(),
        decode=_decode,
        ppl_chunks=2,
        max_new_tokens=4,
    )

    cells = result["cells"]
    # 6 grid cells: bf16 evaluated once (no torchao conversion to split),
    # w8a8/w4a16 in both kernel paths.
    combos = {(c["cell"]["backbone_tier"], c["cell"]["shadow_precision"]) for c in cells}
    assert len(combos) == 6
    # bf16 tier evaluates once per precision (no kernel split); the two
    # quantized tiers split across both kernel paths: 2 + 2*2 + 2*2 = 10.
    assert len(cells) == 10
    assert all(c["ppl"] > 0 for c in cells)
    assert all(c["kld_vs_baseline"] is not None for c in cells if c["cell"]["backbone_tier"] != "bf16")

    # Both kernel paths recorded and labeled for quantized tiers.
    for tier in ("w8a8_fp8", "w4a16"):
        paths = {
            c["cell"]["kernel_path"]
            for c in cells
            if c["cell"]["backbone_tier"] == tier and c["cell"]["shadow_precision"] == "bf16"
        }
        assert paths == set(grid.KERNEL_PATHS)

    # Anchor + bare baselines for 挽回率 raw ppls.
    assert result["anchor"]["cell"]["backbone_tier"] == "bf16"
    assert {b["backbone_tier"] for b in result["bare_bases"]} == set(tiers.TIERS)
    assert len(result["bare_bases"]) == 1 + 2 * len(grid.KERNEL_PATHS)

    # Pareto table: every quantized row labeled with its kernel path.
    table = result["pareto"]
    assert "kernel" in table and "fake_quant" in table
    assert table.count("|") >= 3 * len(cells)

    # Recovery fields on benchmark scores.
    for c in cells:
        for bench in ("mmlu", "gsm8k"):
            assert "recovery_vs_bf16" in c[bench]

    json.dumps(result)  # whole result JSON-serializable


def test_pareto_table_sorted_by_x():
    cells = [
        {"cell": {"backbone_tier": "w4a16", "shadow_precision": "bf16", "kernel_path": "kernel"},
         "memory": {"total_peak_allocated_bytes": 300, "weight_residency_bytes": 100},
         "ppl": 10.0},
        {"cell": {"backbone_tier": "w8a8_fp8", "shadow_precision": "int8", "kernel_path": "fake_quant"},
         "memory": {"total_peak_allocated_bytes": None, "weight_residency_bytes": 200},
         "ppl": 20.0},
        {"cell": {"backbone_tier": "bf16", "shadow_precision": "bf16", "kernel_path": "n/a"},
         "memory": {"total_peak_allocated_bytes": 100, "weight_residency_bytes": 50},
         "ppl": 30.0},
    ]
    table = grid.pareto_table(cells)
    lines = table.strip().splitlines()
    rows = lines[2:]
    # columns: | cell | kernel_path | x | ppl |
    # sorted purely by x: 100 < 200 (proxy) < 300
    assert [r.split("|")[3].strip() for r in rows] == [
        "100", "200 (weight-residency proxy)", "300"
    ]
