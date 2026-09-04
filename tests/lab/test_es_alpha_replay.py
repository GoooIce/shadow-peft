from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

from shadow_lab.es import CandidateObservation, ESConfig, candidate_specs, initialize_es

SCRIPT_PATH = Path(__file__).parents[2] / "scripts" / "replay_mage_flow_es_alpha.py"
SPEC = importlib.util.spec_from_file_location("replay_mage_flow_es_alpha", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
REPLAY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPLAY)


def test_alpha_replay_reuses_one_gradient_and_scales_latent_step() -> None:
    clips = torch.linspace(0.4, 0.9, 16)
    config = ESConfig(dimension=16, population_size=4, sigma=0.03)
    state = initialize_es(clips, config)
    observations = [
        CandidateObservation(spec, float(spec.sign)) for spec in candidate_specs(state)
    ]

    records = REPLAY.alpha_records(clips, config, observations, [0.001, 0.003])

    assert records[0]["gradient_norm"] == records[1]["gradient_norm"]
    assert records[1]["latent_step_rms"] == pytest.approx(
        3 * records[0]["latent_step_rms"]
    )
    assert records[1]["clip_step_rms"] > records[0]["clip_step_rms"]
    assert records[0]["center_clip_sha256"] != records[1]["center_clip_sha256"]


def test_atomic_center_tensor_round_trips_with_metadata(tmp_path: Path) -> None:
    clips = torch.linspace(0.3, 1.0, 8)
    path = tmp_path / "alpha-0p001-clips.pt"

    REPLAY.atomic_save_tensor(path, clips, {"alpha": 0.001})

    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert torch.equal(payload["clips"], clips)
    assert payload["metadata"] == {"alpha": 0.001}
    assert REPLAY.sha256_tensor(payload["clips"]) == REPLAY.sha256_tensor(clips)
    assert not (tmp_path / ".alpha-0p001-clips.pt.tmp").exists()


def test_alpha_slug_is_filename_stable() -> None:
    assert REPLAY.alpha_slug(0.0001) == "0p0001"
    assert REPLAY.alpha_slug(0.003) == "0p003"
