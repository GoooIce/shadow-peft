from __future__ import annotations

import importlib.util
import json
from dataclasses import asdict
from pathlib import Path

import pytest

from shadow_lab.es import CandidateSpec
from shadow_lab.es_quant import ClipGenomeLayout, LayerClipSlice

SCRIPT_PATH = Path(__file__).parents[2] / "scripts" / "run_mage_flow_es.py"
SPEC = importlib.util.spec_from_file_location("run_mage_flow_es", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def _spec(index: int = 0) -> CandidateSpec:
    return CandidateSpec(
        generation=0,
        population_index=index,
        pair_index=index // 2,
        sign=1 if index % 2 == 0 else -1,
        noise_seed=123 + index // 2,
    )


def test_observation_resume_accepts_only_the_replayed_prefix(tmp_path: Path) -> None:
    expected = tuple(_spec(index) for index in range(4))
    path = tmp_path / "candidates.jsonl"
    rows = [
        {"spec": asdict(expected[0]), "reward": 1.25, "failure_reason": None},
        {"spec": asdict(expected[1]), "reward": None, "failure_reason": "OOM"},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    loaded_rows, observations = RUNNER.load_observation_prefix(path, expected)

    assert loaded_rows == rows
    assert [item.spec for item in observations] == list(expected[:2])
    assert observations[0].reward == 1.25
    assert observations[1].failure_reason == "OOM"

    rows[1]["spec"]["noise_seed"] = 999
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="replayed candidate spec"):
        RUNNER.load_observation_prefix(path, expected)


def test_atomic_append_preserves_a_valid_jsonl_prefix(tmp_path: Path) -> None:
    path = tmp_path / "observations.jsonl"

    RUNNER.append_jsonl(path, {"candidate": 0})
    RUNNER.append_jsonl(path, {"candidate": 1})

    assert [json.loads(line) for line in path.read_text().splitlines()] == [
        {"candidate": 0},
        {"candidate": 1},
    ]
    assert not (tmp_path / ".observations.jsonl.tmp").exists()


def test_checkpoint_cursor_requires_all_generation_commit_files(tmp_path: Path) -> None:
    summary = {"generation": 0}
    RUNNER.atomic_write_json(RUNNER.generation_summary_path(tmp_path, 0), summary)
    candidates = tmp_path / "generation-0000-candidates.jsonl"
    candidates.write_text("{}\n{}\n", encoding="utf-8")

    RUNNER.validate_committed_generations(tmp_path, completed=1, population_size=2)

    candidates.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="has 1 candidate rows"):
        RUNNER.validate_committed_generations(tmp_path, completed=1, population_size=2)


def test_replay_audit_requires_every_scientific_output_to_match() -> None:
    metrics = {
        "clip_sha256": "clip",
        "clip_statistics": {
            "min": 0.3,
            "max": 1.0,
            "mean": 0.9,
            "std": 0.1,
            "perturbation_rms": 0.01,
        },
        "weighted_reconstruction_error": 1.0,
        "quantized_layers": 2,
        "clip_group_count": 8,
        "trainable_parameters_after": 0,
        "axis_means": {"hps_v2_1": 0.2, "clip_h14": 0.3},
        "image_sha256": ["image"],
    }

    assert RUNNER.replay_audit(0.5, metrics, 0.5, dict(metrics))["exact"]
    changed = {**metrics, "image_sha256": ["different"]}
    assert not RUNNER.replay_audit(0.5, metrics, 0.5, changed)["exact"]


def test_source_identity_excludes_virtual_environment(tmp_path: Path) -> None:
    (tmp_path / "runtime.py").write_text("VALUE = 1\n", encoding="utf-8")
    ignored = tmp_path / ".venv" / "lib" / "ignored.py"
    ignored.parent.mkdir(parents=True)
    ignored.write_text("VALUE = 2\n", encoding="utf-8")

    first = RUNNER.source_tree_identity(tmp_path)
    ignored.write_text("VALUE = 3\n", encoding="utf-8")
    second = RUNNER.source_tree_identity(tmp_path)

    assert first["file_count"] == 1
    assert first == second


def test_prompt_episode_offset_does_not_change_es_generation() -> None:
    assert RUNNER.prompt_episode_index(0, offset=4) == 4
    assert RUNNER.prompt_episode_index(2, offset=4) == 6
    with pytest.raises(ValueError, match="non-negative"):
        RUNNER.prompt_episode_index(0, offset=-1)


def test_layout_record_survives_json_roundtrip_for_resume_comparison() -> None:
    layout = ClipGenomeLayout(
        layers=(LayerClipSlice("proj", (8, 2), 0, 16),),
        dimension=16,
    )
    record = RUNNER.layout_record(layout)

    assert json.loads(json.dumps(record)) == record
