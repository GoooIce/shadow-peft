from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).parents[2] / "scripts" / "serve_mage_flow_es_monitor.py"
SPEC = importlib.util.spec_from_file_location("serve_mage_flow_es_monitor", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MONITOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MONITOR)


def test_status_summarizes_live_candidate_rows_and_gallery(tmp_path: Path) -> None:
    episode = tmp_path / "gradient-episode-0"
    episode.mkdir()
    rows = [
        {
            "spec": {"population_index": 0},
            "reward": 1.25,
            "failure_reason": None,
            "axis_means": {"hps_v2_1": 0.2, "clip_h14": 0.3},
            "replay_audit": {"exact": True, "matches": {"reward": True}},
        },
        {
            "spec": {"population_index": 1},
            "reward": None,
            "failure_reason": "failed",
        },
    ]
    (episode / "generation-0000-candidates.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    gallery = tmp_path / "gallery" / "selection-episode-24" / "small"
    gallery.mkdir(parents=True)
    (gallery / "000-sample.png").write_bytes(b"png")

    status = MONITOR.build_status(tmp_path)

    assert status["gradient"]["candidate_rows"] == 2
    assert status["gradient"]["failures"] == 1
    assert status["gradient"]["generated_images"] == 24
    assert status["gradient"]["episodes"][0]["state"] == "running"
    assert status["latest_candidates"][0]["reward"] == 1.25
    assert status["gallery"]["image_count"] == 1
    assert status["gallery"]["images"][0]["url"].startswith("/gallery/")


def test_gallery_path_cannot_escape_the_controlled_root(tmp_path: Path) -> None:
    gallery = tmp_path / "gallery"
    gallery.mkdir()

    with pytest.raises(ValueError, match="outside"):
        MONITOR.resolve_gallery_path(gallery, "../secret.png")


def test_empty_precreated_episode_directory_is_still_pending(tmp_path: Path) -> None:
    (tmp_path / "gradient-episode-3").mkdir()

    status = MONITOR.build_status(tmp_path)

    assert status["gradient"]["episodes"][3]["state"] == "pending"
