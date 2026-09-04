from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from shadow_lab.image_eval import SourcePrompt, canonical_json

SCRIPT_PATH = Path(__file__).parents[2] / "scripts" / "build_mage_flow_eval_pool.py"
SPEC = importlib.util.spec_from_file_location("build_mage_flow_eval_pool", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)
apply_content_gate = BUILDER.apply_content_gate
file_sha256 = BUILDER.file_sha256
validate_content_gate_run = BUILDER.validate_content_gate_run


def _write_gate_run(tmp_path: Path, *, screen_all: bool = True) -> tuple[Path, Path]:
    candidates_path = tmp_path / "candidates.jsonl"
    candidates_path.write_text('{"source_id":"1"}\n{"source_id":"2"}\n', encoding="utf-8")
    accepted_path = tmp_path / "accepted.jsonl"
    accepted_path.write_text(
        '{"source":"geneval","source_id":"1","content_gate":{"passed":true}}\n',
        encoding="utf-8",
    )
    audit_path = tmp_path / "audit.jsonl"
    audit_path.write_text('{"candidate_index":0}\n{"candidate_index":1}\n', encoding="utf-8")
    run = {
        "screen_all": screen_all,
        "screened": 2,
        "accepted": 1,
        "rejected": 1,
        "candidate_source": {"rows": 2, "sha256": file_sha256(candidates_path)},
        "accepted_sha256": file_sha256(accepted_path),
        "audit_sha256": file_sha256(audit_path),
    }
    (tmp_path / "run.json").write_text(json.dumps(run), encoding="utf-8")
    return candidates_path, accepted_path


def test_validate_content_gate_run_requires_full_verified_screening(tmp_path: Path) -> None:
    candidates_path, accepted_path = _write_gate_run(tmp_path)

    run_path, run = validate_content_gate_run(
        candidates_path=candidates_path,
        candidate_count=2,
        accepted_path=accepted_path,
    )

    assert run_path == tmp_path / "run.json"
    assert run["accepted"] == 1


def test_validate_content_gate_run_rejects_prefix_screen(tmp_path: Path) -> None:
    candidates_path, accepted_path = _write_gate_run(tmp_path, screen_all=False)

    with pytest.raises(ValueError, match="screen_all"):
        validate_content_gate_run(
            candidates_path=candidates_path,
            candidate_count=2,
            accepted_path=accepted_path,
        )


def test_apply_content_gate_rejects_duplicate_accepted_identity(tmp_path: Path) -> None:
    accepted_path = tmp_path / "accepted.jsonl"
    row = {"source": "geneval", "source_id": "1", "content_gate": {"passed": True}}
    accepted_path.write_text(
        json.dumps(row) + "\n" + json.dumps(row) + "\n",
        encoding="utf-8",
    )
    candidates = [
        SourcePrompt(
            source="geneval",
            revision="revision",
            source_id="1",
            prompt="a prompt",
            stratum="object",
            metadata_json=canonical_json({}),
        )
    ]

    with pytest.raises(ValueError, match="duplicate accepted prompt identity"):
        apply_content_gate(candidates, accepted_path)
