from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from shadow_lab import mage_es_eval


class _Image:
    def save(self, buffer, *, format: str) -> None:
        assert format == "PNG"
        buffer.write(b"png")


class _Pipeline:
    def generate(self, prompts, **kwargs):
        assert prompts == ["prompt"]
        assert kwargs == {
            "seeds": [7],
            "heights": [512],
            "widths": [512],
            "steps": 4,
            "cfg": 1.0,
        }
        return [_Image()]


def test_evaluate_clips_owns_the_shared_materialize_generate_score_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        mage_es_eval,
        "materialize_es_candidate",
        lambda *args, **kwargs: {
            "weighted_reconstruction_error": 0.25,
            "quantized_layers": 2,
            "clip_group_count": 4,
            "trainable_parameters_after": 0,
        },
    )
    monkeypatch.setattr(
        mage_es_eval,
        "score_reward_batch",
        lambda inputs, scorers: [
            {"scores": {"hps_v2_1": 0.2, "clip_h14": 0.4}}
            for _ in inputs
        ],
    )
    monkeypatch.setattr(
        mage_es_eval,
        "aggregate_episode_reward",
        lambda scores, calibration: SimpleNamespace(
            valid=True,
            value=1.25,
            failure_reason=None,
        ),
    )
    synchronizations = []
    torch_runtime = SimpleNamespace(
        cuda=SimpleNamespace(synchronize=lambda: synchronizations.append(True))
    )
    clips = torch.tensor([0.4, 0.6])

    result = mage_es_eval.evaluate_clips(
        torch_runtime=torch_runtime,
        clips=clips,
        transformer=object(),
        master=object(),
        layout=object(),
        pipeline=_Pipeline(),
        prompt_batch=[SimpleNamespace(prompt="prompt", seed=7)],
        scorers=object(),
        calibration=object(),
        height=512,
        width=512,
        steps=4,
        cfg=1.0,
    )

    assert result["reward"] == 1.25
    assert result["weighted_reconstruction_error"] == 0.25
    assert result["axis_means"] == {"hps_v2_1": 0.2, "clip_h14": 0.4}
    assert len(result["image_sha256"]) == 1
    assert len(synchronizations) == 3
