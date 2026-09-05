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
