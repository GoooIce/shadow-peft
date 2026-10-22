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
    EVIDENCE_COLLECTED,
    EVIDENCE_INSUFFICIENT,
    EVIDENCE_MIN_EXAMPLES_BY_SUITE,
    PER_EXAMPLE_FILENAME,
    REPORT_FILENAME,
    SCHEMA_VERSION,
    SUITES,
    SUMMARY_FILENAME,
    EvalExample,
    build_report,
    evaluate_standalone_ppl,
    evidence_for,
    evidence_min_examples,
    load_gsm8k_examples,
    load_jsonl_examples,
    load_standalone_shadow,
    run_gkd_suite,
    run_jitrl_suite,
    run_restrain_suite,
    run_s2t_suite,
    run_suite,
    run_suites,
    standalone_s2t_diagnose,
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


def test_evidence_bars_are_per_suite() -> None:
    assert set(EVIDENCE_MIN_EXAMPLES_BY_SUITE) == set(SUITES)
    assert evidence_min_examples("gkd") == 200
    assert evidence_min_examples("s2t") == 200
    assert evidence_min_examples("jitrl") == 200
    assert evidence_min_examples("restrain") == 100
    assert evidence_for("gkd", 199) == EVIDENCE_INSUFFICIENT
    assert evidence_for("gkd", 200) == EVIDENCE_COLLECTED
    assert evidence_for("restrain", 99) == EVIDENCE_INSUFFICIENT
    assert evidence_for("restrain", 100) == EVIDENCE_COLLECTED
    with pytest.raises(ValueError, match="unknown suite"):
        evidence_min_examples("nope")


def test_run_suite_shares_options_and_defaults_with_run_suites() -> None:
    reports = run_suite(
        "gkd",
        [_example(task="process_calc", expected="4", prompt="2+2=?")],
        model=FakeShadowModel(),
        tokenizer=FixedAnswerTokenizer("4"),
        options={"max_new_tokens": 2, "seq_len": 8, "max_windows": 1},
    )

    assert reports.status == "ok"
    assert reports.metrics["task_accuracy"] == 1.0
    with pytest.raises(ValueError, match="unknown suite"):
        run_suite("nope", [_example()])


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


def test_run_gkd_suite_token_budget_caps_ppl_windows() -> None:
    examples = [
        _example(
            example_id=f"e{index}",
            task="process_calc",
            expected="4",
            prompt="x" * 40,
        )
        for index in range(3)
    ]

    report = run_gkd_suite(
        FakeShadowModel(),
        FixedAnswerTokenizer("4"),
        examples,
        max_new_tokens=2,
        seq_len=8,
        token_budget=16,
        n_boot=10,
    )

    assert report.metrics["ppl_windows"] == 2
    assert report.metrics["ppl_tokens"] == 16
    assert report.config["token_budget"] == 16
    with pytest.raises(ValueError, match="token_budget"):
        run_gkd_suite(
            FakeShadowModel(),
            FixedAnswerTokenizer("4"),
            examples,
            max_new_tokens=2,
            seq_len=8,
            token_budget=0,
        )


def test_run_s2t_suite_reports_hit_at_k_and_agree_fractions() -> None:
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

    assert agreeing.metrics["k"] == 8
    assert agreeing.metrics["hit_at_k"] == 1.0
    assert agreeing.metrics["hit_at_8"] == agreeing.metrics["hit_at_k"]
    assert agreeing.metrics["agree_at_1"] == 1.0
    assert disagreeing.metrics["hit_at_k"] == 1.0
    assert disagreeing.metrics["agree_at_1"] == 0.0
    assert agreeing.metrics["selector_gain"] is None
    assert agreeing.per_example[0]["hit_at_k"] == 1.0


def test_run_s2t_suite_k_below_8_has_no_hit_at_8_alias() -> None:
    report = run_s2t_suite(
        FakeShadowModel(),
        CharTokenizer(),
        [_example(prompt="a")],
        max_new_tokens=2,
        k=4,
        n_boot=10,
    )

    assert report.metrics["k"] == 4
    assert report.metrics["hit_at_k"] == 1.0
    assert "hit_at_8" not in report.metrics


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
    assert report.metrics["memory_latency_ms_n"] == 1
    assert report.metrics["memory_latency_ms_p95"] is not None
    assert (
        report.metrics["memory_latency_ms_max"]
        >= report.metrics["memory_latency_ms_p95"]
        >= 0.0
    )
    assert report.per_example[0]["memory_latency_ms"] >= 0.0
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
    assert report.metrics["memory_latency_ms_p95"] is None
    assert report.metrics["memory_latency_ms_max"] is None


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
    assert any("Evidence bars" in note for note in on_disk["notes"])
    csv_text = Path(paths["summary"]).read_text(encoding="utf-8")
    header = csv_text.splitlines()[0].split(",")
    assert header[:4] == ["suite", "status", "evidence", "claims_gain"]
    for field in ("k", "hit_at_k", "ppl_tokens", "ppl_windows", "memory_latency_ms_p95"):
        assert field in header
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
            "--token-budget",
            "4096",
            "--selector",
            "selector-dir",
        ]
    )
    cli._resolve_smoke(args)

    assert args.suite == "gkd"
    assert args.limit == 5
    assert args.max_new_tokens == 16
    assert args.n == 2
    assert args.token_budget == 4096
    assert args.selector == "selector-dir"


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


def _tiny_shadow_model() -> Any:
    """Random tiny ShadowPEFT model that exercises the real wrapper paths."""

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
    return ShadowForCausalLM(
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


def test_tiny_shadow_model_runs_real_wrapper_paths() -> None:
    """Exercise evaluate_ppl/generate/diagnose on a random tiny ShadowPEFT model."""

    model = _tiny_shadow_model()
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
    assert 0.0 <= s2t.metrics["hit_at_k"] <= 1.0
    assert s2t.metrics["hit_at_8"] == s2t.metrics["hit_at_k"]
    assert 0.0 <= s2t.metrics["agree_at_1"] <= 1.0


def test_run_s2t_suite_selector_gain_from_saved_selector(tmp_path: Path) -> None:
    from shadow_lab.s2t_selector import SelectorHead, save_selector

    model = _tiny_shadow_model()
    torch.manual_seed(0)
    head = SelectorHead(hidden_size=32, mlp_hidden=8)
    save_selector(head, tmp_path / "selector", k=8)
    examples = [
        _example(example_id="m1", task="process_calc", expected="4", prompt="2+2?"),
        _example(example_id="m2", task="process_calc", expected="6", prompt="3+3?"),
    ]

    report = run_s2t_suite(
        model,
        FixedAnswerTokenizer("4"),
        examples,
        max_new_tokens=2,
        k=8,
        selector=str(tmp_path / "selector"),
        n_boot=10,
    )

    assert report.status == "ok"
    assert report.config["selector"].endswith("selector")
    assert report.metrics["n_selector_tokens"] > 0
    assert report.metrics["selector_gain"] == pytest.approx(
        report.metrics["selector_agree_at_1"] - report.metrics["greedy_agree_at_1"]
    )
    with pytest.raises(FileNotFoundError, match="selector"):
        run_s2t_suite(
            model,
            FixedAnswerTokenizer("4"),
            examples,
            max_new_tokens=2,
            k=8,
            selector=str(tmp_path / "missing"),
        )


# ---------------------------------------------------------------------------
# Standalone recovered shadow (exp16 exports)
# ---------------------------------------------------------------------------


def _tiny_causal_lm(seed: int = 0, *, vocab_size: int = 128) -> Any:
    """Random tiny standard HF causal LM (never the 0.6B/4B checkpoints)."""

    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(seed)
    return LlamaForCausalLM(
        LlamaConfig(
            vocab_size=vocab_size,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=256,
        )
    ).eval()


def _save_wordlevel_tokenizer(directory: Path, *, vocab_size: int = 128) -> None:
    """Write a minimal WordLevel tokenizer into an export directory."""

    from tokenizers import Tokenizer
    from tokenizers import models as tk_models
    from tokenizers import pre_tokenizers as tk_pre
    from transformers import PreTrainedTokenizerFast

    tokenizer = Tokenizer(
        tk_models.WordLevel(
            vocab={f"w{index}": index for index in range(vocab_size)},
            unk_token="w0",
        )
    )
    tokenizer.pre_tokenizer = tk_pre.Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="w0",
        eos_token="w1",
        bos_token="w2",
    ).save_pretrained(directory)


def _save_tiny_export(
    directory: Path, *, seed: int = 0, vocab_size: int = 128
) -> Path:
    """Save a tiny HF model + WordLevel tokenizer as an export directory."""

    directory.mkdir(parents=True, exist_ok=True)
    _tiny_causal_lm(seed, vocab_size=vocab_size).save_pretrained(directory)
    _save_wordlevel_tokenizer(directory, vocab_size=vocab_size)
    return directory


def _projected_shadow_model(seed: int = 0, *, vocab_size: int = 128) -> Any:
    """Tiny exp16-style export: shadow Llama backbone + projection + base head."""

    from transformers import LlamaConfig

    from shadow_peft import (
        AutoModelForCausalLMWithHiddenProjection,
        AutoModelForCausalLMWithHiddenProjectionConfig,
    )

    shadow_config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=256,
    )
    config = AutoModelForCausalLMWithHiddenProjectionConfig(
        shadow_model_class="transformers.models.llama.modeling_llama:LlamaModel",
        shadow_model_config_class="transformers.LlamaConfig",
        shadow_model_config=shadow_config.to_dict(),
        base_hidden_size=32,
        vocab_size=vocab_size,
    )
    torch.manual_seed(seed)
    return AutoModelForCausalLMWithHiddenProjection(config).eval()


class ScriptedLogitsModel:
    """Deterministic logits double for standalone S2T (forward only)."""

    def __init__(self, ranking: list[int], vocab_size: int = 128) -> None:
        self.ranking = [int(token) for token in ranking]
        self.vocab_size = vocab_size
        self.device = torch.device("cpu")
        self.training = False

    def eval(self) -> ScriptedLogitsModel:
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
        **kwargs: Any,
    ) -> Any:
        del attention_mask, kwargs
        batch, length = input_ids.shape
        logits = torch.full((batch, length, self.vocab_size), -10.0)
        for index, token in enumerate(self.ranking):
            logits[:, :, token] = 10.0 - index
        return types.SimpleNamespace(logits=logits)


def _scripted_generate_model(seed: int, generated_ids: list[int]) -> Any:
    """Tiny HF model whose ``generate`` always appends scripted token ids."""

    model = _tiny_causal_lm(seed)

    def generate(self: Any, *, input_ids: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        del kwargs
        suffix = torch.tensor(
            [generated_ids], dtype=input_ids.dtype, device=input_ids.device
        )
        return torch.cat([input_ids, suffix], dim=1)

    model.generate = types.MethodType(generate, model)
    return model


def test_load_standalone_shadow_loads_export_and_tokenizer(tmp_path: Path) -> None:
    directory = _save_tiny_export(tmp_path / "shadow", seed=1)
    expected = _tiny_causal_lm(1)
    ids = torch.tensor([[4, 5, 6]])

    model, tokenizer = load_standalone_shadow(
        directory, device="cpu", dtype=torch.float32
    )

    assert model.training is False
    assert next(model.parameters()).dtype == torch.float32
    assert tokenizer.pad_token_id is not None
    with torch.no_grad():
        reference = expected(input_ids=ids).logits
        loaded = model(input_ids=ids).logits
    assert torch.allclose(reference, loaded, atol=1e-5)


def test_load_standalone_shadow_loads_projected_exp16_export(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "projected"
    directory.mkdir()
    expected = _projected_shadow_model(0)
    expected.save_pretrained(directory)
    _save_wordlevel_tokenizer(directory)

    model, tokenizer = load_standalone_shadow(
        directory, device="cpu", dtype=torch.float32
    )

    assert type(model).__name__ == "AutoModelForCausalLMWithHiddenProjection"
    ids = torch.tensor([[3, 4, 5]])
    with torch.no_grad():
        reference = expected(input_ids=ids).logits
        loaded = model(input_ids=ids).logits
    assert torch.allclose(reference, loaded, atol=1e-5)
    assert tokenizer.pad_token_id is not None
    generated = model.generate(
        input_ids=ids,
        max_new_tokens=2,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    assert generated.shape[-1] > ids.shape[-1]


def test_load_standalone_shadow_rejects_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="standalone shadow"):
        load_standalone_shadow(
            tmp_path / "missing", device="cpu", dtype=torch.float32
        )


def test_evaluate_standalone_ppl_and_s2t_accept_projected_export() -> None:
    base = _tiny_causal_lm(0)
    shadow = _projected_shadow_model(1)

    result = evaluate_standalone_ppl(
        base, shadow, CharTokenizer(), ["abcdefgh"], seq_len=4, max_windows=1
    )
    records = standalone_s2t_diagnose(
        base, shadow, CharTokenizer(), ["abcd"], k=4, max_new_tokens=2
    )

    assert result["base_ppl"] > 0
    assert result["shadow_ppl"] > 0
    assert result["joint_ppl"] == pytest.approx(
        result["base_ppl"] * result["shadow_ppl"] ** 0.05, rel=1e-6
    )
    assert len(records) == 2
    assert 0 <= records[0]["base_top1"] < 128


def test_evaluate_standalone_ppl_combines_base_and_shadow_losses() -> None:
    base = _tiny_causal_lm(0)
    shadow = _tiny_causal_lm(1)

    result = evaluate_standalone_ppl(
        base,
        shadow,
        CharTokenizer(),
        ["abcdefghij", "klmnopqrst"],
        seq_len=4,
        max_windows=3,
    )

    assert result["tokens"] == sum(result["window_tokens"]) > 0
    assert (
        len(result["base_losses"])
        == len(result["shadow_losses"])
        == len(result["joint_losses"])
        == len(result["window_tokens"])
    )
    for base_loss, shadow_loss, joint_loss in zip(
        result["base_losses"],
        result["shadow_losses"],
        result["joint_losses"],
        strict=True,
    ):
        assert joint_loss == pytest.approx(base_loss + 0.05 * shadow_loss)
    assert result["joint_ppl"] == pytest.approx(
        result["base_ppl"] * result["shadow_ppl"] ** 0.05, rel=1e-6
    )
    assert result["base_ppl"] != pytest.approx(result["shadow_ppl"])


def test_evaluate_standalone_ppl_zero_weight_and_identical_models() -> None:
    base = _tiny_causal_lm(0)
    same = _tiny_causal_lm(0)

    identical = evaluate_standalone_ppl(
        base,
        same,
        CharTokenizer(),
        ["abcdefghij"],
        seq_len=4,
        max_windows=2,
    )
    weighted_out = evaluate_standalone_ppl(
        base,
        same,
        CharTokenizer(),
        ["abcdefghij"],
        seq_len=4,
        max_windows=2,
        shadow_loss_weight=0.0,
    )

    assert identical["base_ppl"] == pytest.approx(identical["shadow_ppl"], rel=1e-6)
    assert weighted_out["joint_ppl"] == pytest.approx(weighted_out["base_ppl"])
    with pytest.raises(ValueError, match="shadow_loss_weight"):
        evaluate_standalone_ppl(
            base, same, CharTokenizer(), ["abcdefghij"], shadow_loss_weight=-0.1
        )
    with pytest.raises(ValueError, match="2\\+ tokens"):
        evaluate_standalone_ppl(base, same, CharTokenizer(), ["a"], seq_len=4)


def test_standalone_s2t_diagnose_records_base_top1_and_shadow_rank() -> None:
    base = ScriptedLogitsModel([7, 8, 9])
    shadow = ScriptedLogitsModel([11, 7, 12, 13, 14, 15, 16, 17])

    records = standalone_s2t_diagnose(
        base,
        shadow,
        CharTokenizer(),
        [{"prompt_id": "p1", "prompt": "ab", "task": "toy"}],
        k=4,
        max_new_tokens=3,
    )

    assert len(records) == 3
    record = records[0]
    assert record["prompt_id"] == "p1"
    assert record["task"] == "toy"
    assert record["base_top1"] == 7
    assert record["shadow_topk_ids"][:2] == [11, 7]
    assert record["base_top1_rank_in_shadow"] == 2
    assert record["shadow_hidden"] == "standalone"


def test_run_s2t_suite_standalone_reports_hit_and_agree() -> None:
    examples = [_example(prompt="a", task="quality_verdict")]
    base = ScriptedLogitsModel([7, 8, 9])

    report = run_s2t_suite(
        None,
        CharTokenizer(),
        examples,
        base_model=base,
        standalone_shadow=ScriptedLogitsModel([11, 7, 12, 13, 14, 15, 16, 17]),
        max_new_tokens=2,
        k=8,
        n_boot=5,
    )

    assert report.status == "ok"
    assert report.metrics["mode"] == "standalone"
    assert report.metrics["hit_at_k"] == 1.0
    assert report.metrics["agree_at_1"] == 0.0
    assert report.config["mode"] == "standalone"
    assert report.per_example[0]["hit_at_k"] == 1.0

    agreeing = run_s2t_suite(
        None,
        CharTokenizer(),
        examples,
        base_model=base,
        standalone_shadow=ScriptedLogitsModel([7, 11, 12]),
        max_new_tokens=2,
        k=8,
        n_boot=5,
    )
    assert agreeing.metrics["agree_at_1"] == 1.0

    missing_base = run_s2t_suite(
        None,
        CharTokenizer(),
        examples,
        standalone_shadow=ScriptedLogitsModel([7, 11, 12]),
        max_new_tokens=2,
    )
    assert missing_base.status == "data_missing"
    assert "base" in missing_base.notes[0].lower()

    with pytest.raises(ValueError, match="selector"):
        run_s2t_suite(
            None,
            CharTokenizer(),
            examples,
            base_model=base,
            standalone_shadow=ScriptedLogitsModel([7, 11, 12]),
            selector="selector-dir",
            max_new_tokens=1,
        )


def test_run_suites_passes_standalone_shadow_to_model_suites() -> None:
    base = ScriptedLogitsModel([7, 8, 9])
    reports = run_suites(
        ["s2t"],
        examples=[_example(prompt="a")],
        model=base,
        tokenizer=CharTokenizer(),
        base_model=base,
        standalone_shadow=ScriptedLogitsModel([7, 11, 12]),
        s2t_options={"max_new_tokens": 2, "k": 8, "n_boot": 5},
    )

    assert reports["s2t"].metrics["mode"] == "standalone"
    assert reports["s2t"].metrics["agree_at_1"] == 1.0


def test_run_gkd_suite_standalone_scores_base_and_generates_with_shadow() -> None:
    tokenizer = CharTokenizer()
    base = _scripted_generate_model(0, [tokenizer.vocab["9"]])
    shadow = _scripted_generate_model(1, [tokenizer.vocab["4"]])
    examples = [_example(task="process_calc", expected="4", prompt="2+2=?")]

    report = run_gkd_suite(
        base,
        tokenizer,
        examples,
        base_model=base,
        standalone_shadow=shadow,
        max_new_tokens=2,
        seq_len=4,
        max_windows=2,
        n_boot=5,
    )

    assert report.status == "ok"
    assert report.metrics["mode"] == "standalone"
    assert report.config["mode"] == "standalone"
    assert report.metrics["base_ppl"] > 0
    assert report.metrics["joint_ppl"] == pytest.approx(
        report.metrics["base_ppl"] * report.metrics["shadow_ppl"] ** 0.05, rel=1e-6
    )
    # Generation runs on the recovered shadow (``4``), not the scripted base.
    assert report.metrics["task_accuracy"] == 1.0

    missing_base = run_gkd_suite(
        None,
        tokenizer,
        examples,
        standalone_shadow=shadow,
        max_new_tokens=2,
    )
    assert missing_base.status == "data_missing"
    assert "base" in missing_base.notes[0].lower()


def test_run_restrain_suite_accepts_standalone_shadow_as_generator() -> None:
    report = run_restrain_suite(
        _tiny_causal_lm(1),
        FixedAnswerTokenizer("判定：返修。"),
        [_example(expected="rework")],
        n=2,
        max_new_tokens=2,
        temperature=0.0,
        kappa=2,
        n_boot=5,
    )

    assert report.status == "ok"
    assert report.metrics["accuracy"] == 1.0
    assert report.config["adapter"] is None


def test_cli_standalone_flags_are_mutually_exclusive() -> None:
    cli = _load_cli()
    parser = cli.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--model",
                "m",
                "--shadow-ckpt",
                "s",
                "--shadow-model-dir",
                "d",
            ]
        )


def test_cli_standalone_rejects_adapter_only_flags(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cli = _load_cli()
    out = tmp_path / "out"

    with pytest.raises(SystemExit):
        cli.main(
            [
                "--suite",
                "gkd",
                "--shadow-model-dir",
                str(tmp_path / "shadow"),
                "--shadow-model",
                "sm",
                "--out",
                str(out),
            ]
        )
    with pytest.raises(SystemExit):
        cli.main(
            [
                "--suite",
                "s2t",
                "--shadow-model-dir",
                str(tmp_path / "shadow"),
                "--selector",
                "selector-dir",
                "--out",
                str(out),
            ]
        )
    capsys.readouterr()


def test_cli_standalone_missing_dir_is_data_missing_and_records_config(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cli = _load_cli()
    examples = tmp_path / "examples.jsonl"
    examples.write_text(
        json.dumps({"id": "x", "prompt": "p", "task": "quality_verdict", "expected": "rework"}),
        encoding="utf-8",
    )
    missing = tmp_path / "missing-shadow"
    out = tmp_path / "out"

    exit_code = cli.main(
        [
            "--suite",
            "gkd",
            "--dataset",
            str(examples),
            "--shadow-model-dir",
            str(missing),
            "--out",
            str(out),
            "--smoke",
        ]
    )
    capsys.readouterr()

    assert exit_code == 0
    report = json.loads((out / REPORT_FILENAME).read_text(encoding="utf-8"))
    assert report["config"]["mode"] == "standalone"
    assert report["config"]["shadow_model_dir"] == str(missing)
    assert report["config"]["shadow_ckpt"] is None
    suite = report["suites"]["gkd"]
    assert suite["status"] == "data_missing"
    assert "standalone" in suite["notes"][0].lower()


def test_cli_standalone_gkd_uses_recovered_shadow_and_records_mode(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cli = _load_cli()
    base_dir = _save_tiny_export(tmp_path / "base", seed=0)
    shadow_dir = _save_tiny_export(tmp_path / "shadow", seed=1)
    examples = tmp_path / "examples.jsonl"
    examples.write_text(
        json.dumps(
            {
                "id": "x",
                "prompt": "w3 w4 w5 w6 w7 w8",
                "task": "process_calc",
                "expected": "4",
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "out"

    exit_code = cli.main(
        [
            "--suite",
            "gkd",
            "--dataset",
            str(examples),
            "--model",
            str(base_dir),
            "--shadow-model-dir",
            str(shadow_dir),
            "--limit",
            "1",
            "--smoke",
            "--device",
            "cpu",
            "--dtype",
            "float32",
            "--out",
            str(out),
        ]
    )
    capsys.readouterr()

    assert exit_code == 0
    report = json.loads((out / REPORT_FILENAME).read_text(encoding="utf-8"))
    assert report["config"]["mode"] == "standalone"
    assert report["config"]["shadow_model_dir"] == str(shadow_dir)
    assert report["config"]["shadow_ckpt"] is None
    suite = report["suites"]["gkd"]
    assert suite["status"] == "ok"
    assert suite["metrics"]["mode"] == "standalone"
    assert suite["config"]["mode"] == "standalone"
    assert suite["metrics"]["base_ppl"] > 0
    assert suite["metrics"]["joint_ppl"] == pytest.approx(
        suite["metrics"]["base_ppl"] * suite["metrics"]["shadow_ppl"] ** 0.05,
        rel=1e-6,
    )


def test_cli_standalone_restrain_needs_only_shadow_model_dir(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cli = _load_cli()
    shadow_dir = _save_tiny_export(tmp_path / "shadow", seed=1)
    heldout = tmp_path / "heldout.jsonl"
    heldout.write_text(
        json.dumps(
            {
                "id": "h1",
                "prompt": "w3 w4 w5 w6",
                "task": "quality_verdict",
                "expected": "rework",
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "out"

    exit_code = cli.main(
        [
            "--suite",
            "restrain",
            "--dataset",
            str(heldout),
            "--shadow-model-dir",
            str(shadow_dir),
            "--limit",
            "1",
            "--smoke",
            "--n",
            "1",
            "--temperature",
            "0.0",
            "--device",
            "cpu",
            "--dtype",
            "float32",
            "--out",
            str(out),
        ]
    )
    capsys.readouterr()

    assert exit_code == 0
    report = json.loads((out / REPORT_FILENAME).read_text(encoding="utf-8"))
    assert report["config"]["mode"] == "standalone"
    suite = report["suites"]["restrain"]
    assert suite["status"] == "ok"
    assert suite["config"]["adapter"] is None
