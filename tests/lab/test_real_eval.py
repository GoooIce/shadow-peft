"""Real-corpus evaluation harness (``shadow_lab.real_eval``) — behavior tests.

The suites are exercised with tiny fake models/backends so the tests never
load Qwen/GSM8K weights. Network-backed loaders are monkeypatched at their
``datasets`` seam. Anchors:

- JSONL loading keeps ``id``/``prompt``/``task``/``expected``/``metadata`` and
  rejects duplicate ids instead of silently overwriting per-example rows.
- GSM8K loading is deterministic for a fixed seed/limit and keeps the numeric
  final answer for ``grade_task``.
- Every suite returns a uniform ``SuiteReport``; missing weights, memory,
  data or an n-best backend produce ``status="data_missing"`` rather than a
  fabricated score.
- ``build_report``/``write_report`` emit report.json, summary.csv and
  per_example.jsonl with a dataset fingerprint and conservative claims.
"""

from __future__ import annotations

import json
import math
import sys
import types
from pathlib import Path
from typing import Any

import pytest
import torch

from shadow_lab.real_eval import (
    EVIDENCE_INSUFFICIENT,
    PER_EXAMPLE_FILENAME,
    REPORT_FILENAME,
    SCHEMA_VERSION,
    SUITES,
    SUMMARY_FILENAME,
    EvalExample,
    build_report,
    load_gsm8k_examples,
    load_jsonl_examples,
    run_gkd_suite,
    run_jitrl_suite,
    run_restrain_suite,
    run_s2t_suite,
    run_suites,
    write_report,
)

# ---------------------------------------------------------------------------
# Fixtures: tiny deterministic model, tokenizer and backends
# ---------------------------------------------------------------------------


class CharTokenizer:
    """Character-level tokenizer: no network, no model files."""

    eos_token_id = 2
    pad_token_id = 0

    def __init__(self) -> None:
        self.vocab = {chr(code): index + 3 for index, code in enumerate(range(32, 127))}
        self.vocab["<eos>"] = self.eos_token_id
        self.vocab["<pad>"] = self.pad_token_id
        self.inverse = {index: char for char, index in self.vocab.items()}

    def __call__(self, text: str, **kwargs: Any) -> Any:
        ids = [self.vocab.get(char, 3) for char in str(text)]
        if ids and ids[-1] == self.eos_token_id:
            ids = ids[:-1]
        if kwargs.get("return_tensors") == "pt":
            return {"input_ids": torch.tensor([ids], dtype=torch.long)}
        return ids

    def decode(self, ids: Any, skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        chars = [self.inverse.get(int(token), "?") for token in ids]
        return "".join(chars)


class FakeShadowModel:
    """Forward-only ``base_shadow`` model with deterministic logits."""

    def __init__(self, vocab_size: int = 128, shadow_agrees: bool = True) -> None:
        self.vocab_size = vocab_size
        self.shadow_agrees = shadow_agrees
        self.device = torch.device("cpu")
        self.inference_mode = "base_shadow"
        self.training = False
        self.modes: list[str] = []

    def set_inference_mode(self, mode: str) -> None:
        self.inference_mode = mode
        self.modes.append(mode)

    def eval(self) -> FakeShadowModel:
        self.training = False
        return self

    def train(self, flag: bool = True) -> None:
        self.training = flag

    def parameters(self):
        yield torch.zeros(1)

    def __call__(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        use_cache: bool = False,
        past_key_values: Any = None,
    ) -> Any:
        del attention_mask, use_cache, past_key_values
        batch, length = input_ids.shape
        logits = torch.full((batch, length, self.vocab_size), -10.0)
        for row in range(batch):
            last = int(input_ids[row, -1])
            base_top = 3 + (last % (self.vocab_size - 3))
            logits[row, :, base_top] = 10.0
            logits[row, :, (base_top + 1) % self.vocab_size] = 9.0
        shadow = logits.clone()
        if not self.shadow_agrees:
            for row in range(batch):
                last = int(input_ids[row, -1])
                base_top = 3 + (last % (self.vocab_size - 3))
                shadow[row, :, (base_top + 2) % self.vocab_size] = 11.0
        return types.SimpleNamespace(logits=logits, shadow_logits=shadow)


class FakeNBestBackend:
    """Two-candidate backend: wrong top-1, correct memory-supported second."""

    model_id = "fake-nbest"

    def __init__(self, answers: dict[str, list[tuple[str, float]]]) -> None:
        self.answers = answers
        self.calls = 0

    def generate_nbest(self, request: Any, n: int = 4) -> list[Any]:
        from edge_cloud.backends import Generation

        self.calls += 1
        rows = self.answers[request.prompt][:n]
        return [
            Generation(
                text=text,
                token_logprobs=[logprob] * 4,
                model_id=self.model_id,
                finish_reason="stop",
                output_tokens=4,
            )
            for text, logprob in rows
        ]


def _example(
    example_id: str = "e1",
    prompt: str = "判定：来料是否合格？",
    task: str = "quality_verdict",
    expected: Any = "rework",
    metadata: dict[str, Any] | None = None,
) -> EvalExample:
    return EvalExample(
        id=example_id,
        prompt=prompt,
        task=task,
        expected=expected,
        metadata=metadata or {},
    )


# ---------------------------------------------------------------------------
# Loader / schema
# ---------------------------------------------------------------------------


def test_load_jsonl_examples_keeps_schema_and_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "examples.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "gsm-1",
                        "prompt": "2+2=?",
                        "task": "gsm8k",
                        "expected": "4",
                        "metadata": {"split": "test"},
                    }
                ),
                "",
                json.dumps({"prompt": "second", "task": "standard_qa"}),
            ]
        ),
        encoding="utf-8",
    )

    examples = load_jsonl_examples(path)

    assert [example.id for example in examples] == ["gsm-1", "ex2"]
    assert examples[0].metadata == {"split": "test"}
    assert examples[1].expected is None
    assert examples[1].task == "standard_qa"


def test_load_jsonl_examples_rejects_duplicate_ids(tmp_path: Path) -> None:
    path = tmp_path / "dupes.jsonl"
    rows = [{"id": "same", "prompt": f"p{index}"} for index in range(2)]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate example id"):
        load_jsonl_examples(path)


def test_load_jsonl_examples_rejects_missing_prompt(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"id": "x", "task": "gsm8k"}), encoding="utf-8")

    with pytest.raises(ValueError, match="line 1"):
        load_jsonl_examples(path)


def test_load_gsm8k_examples_is_deterministic_and_grades_final_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [
        {
            "question": f"Question {index}?",
            "answer": f"reasoning...\n#### {index * 10}" + (",000" if index == 2 else ""),
        }
        for index in range(1, 7)
    ]
    calls: list[Any] = []

    def fake_load_dataset(*args: Any, **kwargs: Any) -> list[dict[str, str]]:
        calls.append((args, kwargs))
        return rows

    fake_module = types.ModuleType("datasets")
    fake_module.load_dataset = fake_load_dataset
    monkeypatch.setitem(sys.modules, "datasets", fake_module)

    first = load_gsm8k_examples(limit=3, seed=7)
    second = load_gsm8k_examples(limit=3, seed=7)

    assert [example.id for example in first] == [example.id for example in second]
    assert [example.expected for example in first] == [
        example.expected for example in second
    ]
    assert len(first) == 3
    assert calls[0][0] == ("openai/gsm8k", "main")
    assert calls[0][1]["split"] == "test"
    assert all(example.task == "gsm8k" for example in first)
    assert all("####" in example.metadata["reference"] for example in first)
    assert all(example.prompt.endswith("Answer:") for example in first)
    assert all(example.expected.isdigit() for example in first)
    all_rows = load_gsm8k_examples(seed=7)
    assert "20000" in {example.expected for example in all_rows}


def test_load_gsm8k_examples_requires_datasets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "datasets", None)

    with pytest.raises(RuntimeError, match="datasets"):
        load_gsm8k_examples(limit=1)


# ---------------------------------------------------------------------------
# Suite dispatch and status
# ---------------------------------------------------------------------------


def test_run_suites_dispatch_and_data_missing_paths() -> None:
    examples = [_example()]
    model = FakeShadowModel()
    tokenizer = CharTokenizer()

    reports = run_suites(
        ["gkd", "s2t", "restrain", "jitrl"],
        examples=examples,
        model=model,
        tokenizer=tokenizer,
        gkd_options={"max_new_tokens": 2, "max_windows": 1, "seq_len": 8},
        s2t_options={"max_new_tokens": 2, "k": 8},
        restrain_options={"n": 2, "max_new_tokens": 2},
        backend=FakeNBestBackend({}),
    )

    assert tuple(reports) == SUITES
    assert reports["gkd"].status == "ok"
    assert reports["s2t"].status == "ok"
    assert reports["restrain"].status == "ok"
    assert reports["jitrl"].status == "data_missing"
    assert "memory" in reports["jitrl"].notes[0].lower()


def test_run_suites_missing_model_is_data_missing_not_crash() -> None:
    reports = run_suites(["gkd"], examples=[_example()])

    assert reports["gkd"].status == "data_missing"
    assert "model" in reports["gkd"].notes[0].lower()


def test_run_suites_rejects_unknown_suite() -> None:
    with pytest.raises(ValueError, match="unknown suite"):
        run_suites(["nope"], examples=[_example()])


# ---------------------------------------------------------------------------
# Suite metric aggregation
# ---------------------------------------------------------------------------


class FixedAnswerTokenizer(CharTokenizer):
    """Tokenizer whose decode always returns a scripted answer."""

    def __init__(self, answer: str) -> None:
        super().__init__()
        self.answer = answer

    def decode(self, ids: Any, skip_special_tokens: bool = True) -> str:
        del ids, skip_special_tokens
        return self.answer


def test_run_gkd_suite_reports_ppl_and_task_accuracy() -> None:
    examples = [_example(task="process_calc", expected="4", prompt="2+2=?")]
    reports = run_suites(
        ["gkd"],
        examples=examples,
        model=FakeShadowModel(),
        tokenizer=FixedAnswerTokenizer("4"),
        gkd_options={"max_new_tokens": 2, "seq_len": 8, "max_windows": 2, "seed": 3},
    )
    report = reports["gkd"]

    assert report.status == "ok"
    assert report.metrics["joint_ppl"] > 0
    assert report.metrics["shadow_ppl"] > 0
    assert report.metrics["ppl_tokens"] > 0
    assert report.metrics["task_accuracy"] == 1.0
    assert report.metrics["n_graded"] == 1
    assert report.per_example[0]["correct"] is True
    assert report.claims_gain is False
    assert report.evidence == EVIDENCE_INSUFFICIENT


def test_run_s2t_suite_reports_hit_and_agree_fractions() -> None:
    examples = [_example(prompt="a", task="quality_verdict")]
    agreeing = run_s2t_suite(
        FakeShadowModel(shadow_agrees=True),
        CharTokenizer(),
        examples,
        max_new_tokens=2,
        k=8,
        n_boot=10,
    )
    disagreeing = run_s2t_suite(
        FakeShadowModel(shadow_agrees=False),
        CharTokenizer(),
        examples,
        max_new_tokens=2,
        k=8,
        n_boot=10,
    )

    assert agreeing.metrics["hit_at_8"] == 1.0
    assert agreeing.metrics["agree_at_1"] == 1.0
    assert disagreeing.metrics["hit_at_8"] == 1.0
    assert disagreeing.metrics["agree_at_1"] == 0.0
    assert agreeing.metrics["selector_gain"] is None
    assert agreeing.per_example[0]["hit_at_k"] == 1.0


def test_run_restrain_suite_aggregates_consistency_and_accuracy() -> None:
    examples = [_example(expected="rework")]
    report = run_restrain_suite(
        FakeShadowModel(),
        FixedAnswerTokenizer("判定：返修。"),
        examples,
        n=2,
        max_new_tokens=2,
        temperature=0.0,
        kappa=2,
        n_boot=10,
    )

    assert report.status == "ok"
    assert report.metrics["accuracy"] == 1.0
    assert report.metrics["mean_consistency"] == 1.0
    assert report.metrics["low_consistency_rate"] == 0.0
    assert report.metrics["skip_rate"] == 0.0
    assert report.metrics["n_rollouts"] == 2


def test_run_restrain_suite_counts_unparseable_answers_as_skips() -> None:
    report = run_restrain_suite(
        FakeShadowModel(),
        FixedAnswerTokenizer("no idea"),
        [_example(expected="rework")],
        n=2,
        max_new_tokens=2,
        temperature=0.0,
        kappa=1,
        n_boot=10,
    )

    assert report.metrics["skip_rate"] == 1.0
    assert report.metrics["accuracy"] == 0.0
    assert report.per_example[0]["skipped"] is True


def _memory_entry(
    *,
    prompt: str = "p1",
    task: str = "quality_verdict",
    answer: str = "判定：返修。",
    tenant: str = "default",
    return_value: float = 1.0,
) -> dict[str, Any]:
    return {
        "prompt": prompt,
        "task": task,
        "candidates": [answer],
        "return_value": return_value,
        "accepted": True,
        "corrected": return_value > 0,
        "adapter_version": "base",
        "tenant": tenant,
        "created_at": 0.0,
        "ttl_s": 10**12,
    }


def _memory(*entries: dict[str, Any]) -> Any:
    from edge_cloud.memory import ExperienceMemory, MemoryEntry

    return ExperienceMemory([MemoryEntry.from_dict(entry) for entry in entries])


def test_run_jitrl_suite_reranks_with_memory_and_counts_false_bias() -> None:
    examples = [_example(prompt="p1", expected="rework")]
    backend = FakeNBestBackend(
        {
            "p1": [
                ("判定：合格。", math.log(0.64)),
                ("判定：返修。", math.log(0.62)),
            ]
        }
    )
    report = run_jitrl_suite(
        examples,
        memory=_memory(_memory_entry(), _memory_entry()),
        backend=backend,
        threshold=0.0,
        min_support=1,
        n_boot=10,
    )

    assert report.status == "ok"
    assert report.metrics["no_memory_accuracy"] == 0.0
    assert report.metrics["rerank_accuracy"] == 1.0
    assert report.metrics["accuracy_delta"] == 1.0
    assert report.metrics["false_bias_count"] == 0
    assert report.metrics["false_bias_denominator"] == 0
    assert report.metrics["false_bias_rate"] is None
    assert report.metrics["memory_hits"] == 1
    assert report.metrics["reranked_cases"] == 1
    assert report.metrics["cloud_calls_no_memory"] is None
    assert report.per_example[0]["reranked"] is True


def test_run_jitrl_suite_records_false_bias_when_memory_hurts() -> None:
    examples = [_example(prompt="p1", expected="rework")]
    backend = FakeNBestBackend(
        {
            "p1": [
                ("判定：返修。", math.log(0.64)),
                ("判定：合格。", math.log(0.62)),
            ]
        }
    )
    report = run_jitrl_suite(
        examples,
        memory=_memory(_memory_entry(answer="判定：合格。")),
        backend=backend,
        threshold=0.0,
        min_support=1,
    )

    assert report.metrics["no_memory_accuracy"] == 1.0
    assert report.metrics["rerank_accuracy"] == 0.0
    assert report.metrics["false_bias_count"] == 1
    assert report.metrics["false_bias_rate"] == 1.0


def test_run_jitrl_suite_counts_cloud_calls_when_a_client_is_supplied() -> None:
    class FakeCloud:
        def __init__(self) -> None:
            self.calls = 0

        def enhance(self, request: Any) -> Any:
            self.calls += 1
            return types.SimpleNamespace(text="判定：返修。")

    cloud = FakeCloud()
    examples = [_example(prompt="p1", expected="rework")]
    backend = FakeNBestBackend(
        {"p1": [("判定：合格。", math.log(0.64)), ("判定：返修。", math.log(0.62))]}
    )
    report = run_jitrl_suite(
        examples,
        memory=_memory(_memory_entry()),
        backend=backend,
        cloud=cloud,
        threshold=0.0,
        min_support=1,
        escalation_threshold=1.0,
    )

    assert report.metrics["cloud_calls_no_memory"] == 1
    assert report.metrics["cloud_calls_rerank"] == 1
    assert report.metrics["cloud_calls_delta"] == 0
    assert report.per_example[0]["cloud_called_no_memory"] is True
    assert report.per_example[0]["cloud_called_rerank"] is True
    assert cloud.calls == 2


def test_run_jitrl_suite_data_missing_for_missing_memory_file(tmp_path: Path) -> None:
    report = run_jitrl_suite(
        [_example()],
        memory=tmp_path / "missing.jsonl",
        backend=FakeNBestBackend({}),
    )

    assert report.status == "data_missing"
    assert "memory" in report.notes[0].lower()


def test_run_jitrl_suite_data_missing_without_nbest_backend() -> None:
    report = run_jitrl_suite([_example()], memory=_memory(_memory_entry()), backend=None)

    assert report.status == "data_missing"
    assert "n-best" in report.notes[0].lower()


# ---------------------------------------------------------------------------
# Unified report
# ---------------------------------------------------------------------------


def test_build_report_and_write_report_schema(tmp_path: Path) -> None:
    examples = [_example(), _example(example_id="e2", prompt="p2", expected="pass")]
    reports = run_suites(
        ["gkd", "s2t", "jitrl"],
        examples=examples,
        model=FakeShadowModel(),
        tokenizer=FixedAnswerTokenizer("rework"),
        backend=FakeNBestBackend({}),
        gkd_options={"max_new_tokens": 2, "seq_len": 8, "max_windows": 1},
        s2t_options={"max_new_tokens": 2, "n_boot": 10},
        jitrl_options={"threshold": 0.0},
    )
    report = build_report(
        reports,
        examples=examples,
        dataset={"name": "unit-test", "split": "test"},
        config={"model": "tiny", "seed": 0},
    )
    paths = write_report(report, tmp_path / "out")

    assert report["schema_version"] == SCHEMA_VERSION
    assert report["dataset"]["name"] == "unit-test"
    assert report["dataset"]["n_examples"] == 2
    assert report["dataset"]["fingerprint"].startswith("sha256:")
    assert set(report["suites"]) == {"gkd", "s2t", "jitrl"}
    assert report["claims_gain"] is False
    assert report["evidence"] == EVIDENCE_INSUFFICIENT
    assert all(claim["claims_gain"] is False for claim in report["claims"])
    assert report["summary"]["jitrl"]["status"] == "data_missing"

    on_disk = json.loads(Path(paths["report"]).read_text(encoding="utf-8"))
    assert on_disk["schema_version"] == SCHEMA_VERSION
    csv_text = Path(paths["summary"]).read_text(encoding="utf-8")
    header = csv_text.splitlines()[0].split(",")
    assert header[:4] == ["suite", "status", "evidence", "claims_gain"]
    example_rows = [
        json.loads(line)
        for line in Path(paths["per_example"]).read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert example_rows
    assert all("suite" in row for row in example_rows)
    assert {row["suite"] for row in example_rows} == {"gkd", "s2t"}


def test_build_report_requires_a_suite() -> None:
    with pytest.raises(ValueError, match="at least one"):
        build_report([])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _load_cli() -> Any:
    import importlib.util

    path = Path(__file__).resolve().parents[2] / "scripts" / "exp17_real_eval.py"
    spec = importlib.util.spec_from_file_location("exp17_real_eval", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_cli_parser_accepts_issue_scope_flags() -> None:
    cli = _load_cli()
    args = cli.build_parser().parse_args(
        [
            "--suite",
            "gkd",
            "--dataset",
            "gsm8k",
            "--model",
            "m",
            "--shadow-ckpt",
            "s",
            "--shadow-model",
            "sm",
            "--memory",
            "mem.jsonl",
            "--limit",
            "5",
            "--seed",
            "2",
            "--device",
            "cpu",
            "--dtype",
            "float32",
            "--out",
            "o",
            "--smoke",
        ]
    )
    cli._resolve_smoke(args)

    assert args.suite == "gkd"
    assert args.limit == 5
    assert args.max_new_tokens == 16
    assert args.n == 2


def test_cli_without_model_writes_data_missing_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cli = _load_cli()
    examples = tmp_path / "examples.jsonl"
    examples.write_text(
        json.dumps({"id": "x", "prompt": "p", "task": "quality_verdict", "expected": "rework"}),
        encoding="utf-8",
    )
    out = tmp_path / "out"

    exit_code = cli.main(
        ["--suite", "gkd", "--dataset", str(examples), "--out", str(out), "--smoke"]
    )
    printed = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert printed["summary"]["gkd"]["status"] == "data_missing"
    assert printed["evidence"] == EVIDENCE_INSUFFICIENT
    assert printed["claims_gain"] is False
    report = json.loads((out / REPORT_FILENAME).read_text(encoding="utf-8"))
    assert report["suites"]["gkd"]["status"] == "data_missing"
    assert "model" in report["suites"]["gkd"]["notes"][0].lower()
    assert (out / SUMMARY_FILENAME).exists()
    assert (out / PER_EXAMPLE_FILENAME).exists()


def test_cli_all_without_model_never_loads_gsm8k(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cli = _load_cli()
    out = tmp_path / "out"
    monkeypatch.setitem(sys.modules, "datasets", None)

    exit_code = cli.main(["--suite", "all", "--out", str(out), "--smoke"])
    capsys.readouterr()

    assert exit_code == 0
    report = json.loads((out / REPORT_FILENAME).read_text(encoding="utf-8"))
    assert set(report["suites"]) == set(SUITES)
    assert all(
        payload["status"] == "data_missing" for payload in report["suites"].values()
    )


def test_tiny_shadow_model_runs_real_wrapper_paths() -> None:
    """Exercise evaluate_ppl/generate/diagnose on a random tiny ShadowPEFT model."""

    from transformers import LlamaConfig, LlamaForCausalLM

    from shadow_peft import ShadowConfig, ShadowForCausalLM, get_shadow_model

    torch.manual_seed(0)
    base = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=128,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=128,
        )
    )
    model = ShadowForCausalLM(
        get_shadow_model(
            base,
            ShadowConfig(
                num_shadow_layers=1,
                injection_hidden_size=8,
                gate_hidden_size=10,
                alpha=0.1,
                dropout=0.0,
            ),
        ),
        inference_mode="base_shadow",
    ).eval()
    tokenizer = FixedAnswerTokenizer("4")
    examples = [
        _example(example_id="m1", task="process_calc", expected="4", prompt="2+2?"),
        _example(example_id="m2", task="process_calc", expected="6", prompt="3+3?"),
    ]

    gkd = run_gkd_suite(
        model,
        tokenizer,
        examples,
        max_new_tokens=2,
        seq_len=16,
        max_windows=2,
        n_boot=10,
    )
    s2t = run_s2t_suite(model, tokenizer, examples, max_new_tokens=2, k=8, n_boot=10)

    assert gkd.status == "ok"
    assert gkd.metrics["joint_ppl"] > 0
    assert gkd.metrics["shadow_ppl"] > 0
    assert gkd.metrics["task_accuracy"] == 0.5
    assert s2t.status == "ok"
    assert 0.0 <= s2t.metrics["hit_at_8"] <= 1.0
    assert 0.0 <= s2t.metrics["agree_at_1"] <= 1.0
