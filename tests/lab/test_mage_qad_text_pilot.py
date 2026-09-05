import copy

import pytest
import torch
from torch import nn

from shadow_lab.mage_qad import MageTrajectory, prepare_mage_qad, trajectory_loss
from shadow_lab.mage_qad_text_pilot import compare_pilot_reports, text_pilot_loss


class DualBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.image = nn.Linear(16, 16)
        self.text = nn.Linear(16, 16)

    def forward(self, txt, img):
        return self.text(txt + img), self.image(img + txt)


class DualModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_blocks = nn.ModuleList([DualBlock(), DualBlock()])

    def forward(self, img, txt, timesteps):
        for block in self.transformer_blocks:
            txt, img = block(txt, img)
        return img


def test_auxiliary_reaches_unused_text_branch_without_changing_zero_weight_objective():
    torch.manual_seed(3)
    teacher = DualModel()
    a, b, reference = (copy.deepcopy(teacher) for _ in range(3))
    for model in (a, b, reference):
        prepare_mage_qad(model, expected_layers=4)
    trajectory = MageTrajectory(
        torch.randn(1, 2, 16), {"txt": torch.randn(1, 2, 16)}, (1.0, 0.9, 0.75, 0.5, 0.0)
    )
    original = trajectory_loss(reference, teacher, trajectory, backward=True)
    image = text_pilot_loss(a, teacher, trajectory, backward=True)
    auxiliary = text_pilot_loss(b, teacher, trajectory, coefficient=0.1, backward=True)
    assert image["image_mse"] == image["objective"] == original
    assert image["image_mse"] == auxiliary["image_mse"]
    assert a.transformer_blocks[-1].text.weight.grad is None
    assert auxiliary["objective"] > image["objective"]
    for p, expected in zip(a.parameters(), reference.parameters(), strict=True):
        assert (p.grad is None and expected.grad is None) or torch.equal(p.grad, expected.grad)
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
        for p in b.parameters()
        if p.requires_grad
    )
    assert not torch.equal(
        a.transformer_blocks[0].image.weight.grad, b.transformer_blocks[0].image.weight.grad
    )
    assert all(p.grad is None for p in teacher.parameters())
    assert not a.transformer_blocks[-1]._forward_hooks
    assert not teacher.transformer_blocks[-1]._forward_hooks


def test_pilot_requires_paired_identity_and_image_benefit():
    report = dict(
        status="completed",
        data_sha256="data",
        initial_state_sha256="weights",
        code_sha256="code",
        protocol={},
        baseline=[dict(image_mse=2.0)] * 8,
        validation=[dict(image_mse=1.0)] * 8,
    )
    auxiliary = copy.deepcopy(report)
    auxiliary["validation"] = [dict(image_mse=0.95)] * 8
    assert compare_pilot_reports(report, auxiliary)["recommendation"].startswith("auxiliary")
    auxiliary["validation"] = [dict(image_mse=1.001, text_mse=0.0)] * 8
    assert compare_pilot_reports(report, auxiliary)["recommendation"].startswith("retain")
    auxiliary["data_sha256"] = "other"
    with pytest.raises(ValueError, match="data_sha256"):
        compare_pilot_reports(report, auxiliary)


def test_formal_text_coefficient_is_frozen_across_resume_and_selection_uses_image_loss(
    tmp_path, monkeypatch
):
    from shadow_lab import mage_qad_campaign as campaign
    from shadow_lab.mage_qad_checkpoint import read_checkpoint
    from shadow_lab.mage_qad_data import build_qad_pool

    torch.manual_seed(9)
    base = DualModel()
    sources = [
        dict(
            prompt=f"formal {i}",
            source="test",
            source_revision="v1",
            source_id=str(i),
            stratum="test",
        )
        for i in range(1152)
    ]
    excluded = [dict(prompt=f"excluded {i}", sample_id=f"e{i}", seed=i) for i in range(500)]
    rows, manifest = build_qad_pool(sources, excluded)
    seen = []

    def source(row):
        seen.append(row["sample_id"])
        generator = torch.Generator().manual_seed(row["seed"])
        return MageTrajectory(
            torch.randn(1, 2, 16, generator=generator),
            {"txt": torch.randn(1, 2, 16, generator=generator)},
            (1.0, 0.9, 0.75, 0.5, 0.0),
        )

    # A controlled image validation surface distinguishes the selection metric
    # from the combined image/text objective used for backward.
    monkeypatch.setattr(campaign, "trajectory_loss", lambda *args, **kwargs: 0.75)
    cfg = campaign.MageQADConfig(text_initial_fraction=0.1, adamw_foreach=False)
    identity = dict(model="tiny-real-dual-stream", code="test")
    continuous = prepare_mage_qad(copy.deepcopy(base), expected_layers=4)
    report = campaign.run_mage_qad(
        continuous,
        copy.deepcopy(base),
        source,
        rows,
        manifest,
        identity,
        cfg,
        tmp_path / "continuous",
        stop_after=2,
    )
    assert report["status"] == "paused", report.get("error")
    assert report["progress"]["best_loss"] == 0.75
    assert report["progress"]["history"][0]["training_loss"] != 0.75
    train_id = next(r["sample_id"] for r in rows if r["split"] == "qad_train")
    assert seen.count(train_id) == 2  # calibration once, then training once
    seen.clear()
    resumed = prepare_mage_qad(copy.deepcopy(base), expected_layers=4)
    first = campaign.run_mage_qad(
        resumed,
        copy.deepcopy(base),
        source,
        rows,
        manifest,
        identity,
        cfg,
        tmp_path / "resumed",
        stop_after=1,
    )
    coefficient = first["text_supervision"]["coefficient"]
    second = campaign.run_mage_qad(
        resumed,
        copy.deepcopy(base),
        source,
        rows,
        manifest,
        identity,
        cfg,
        tmp_path / "resumed",
        resume=True,
        stop_after=1,
    )
    assert second["status"] == "paused", second.get("error")
    assert second["text_supervision"]["coefficient"] == coefficient
    assert seen.count(train_id) == 2  # resume must not recalibrate
    a = read_checkpoint(tmp_path / "continuous/latest.pt")
    b = read_checkpoint(tmp_path / "resumed/latest.pt")
    assert a["sha256"] == b["sha256"]


@pytest.mark.parametrize("fraction", [-0.1, float("nan"), float("inf"), 1.1])
def test_formal_rejects_invalid_auxiliary_fraction(fraction):
    from shadow_lab.mage_qad_campaign import MageQADConfig

    with pytest.raises(ValueError, match="fraction"):
        MageQADConfig(text_initial_fraction=fraction)
