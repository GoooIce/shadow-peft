import copy

import pytest
import torch
from torch import nn

from shadow_lab.mage_qad import MageTrajectory, prepare_mage_qad, trajectory_loss


class Denoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(16, 16)
        self.norm = nn.LayerNorm(16)
        self.seen = []

    def forward(self, img, txt, timesteps):
        self.seen.append((img.detach().clone(), txt.detach().clone(), timesteps.clone()))
        return self.proj(self.norm(img)) + txt * timesteps.reshape(-1, 1, 1)


def test_online_teacher_uses_student_state_at_all_four_positions():
    teacher = Denoiser()
    student = copy.deepcopy(teacher)
    controller = prepare_mage_qad(student, expected_layers=1)
    trajectory = MageTrajectory(
        torch.randn(1, 2, 16), {"txt": torch.ones(1, 2, 16)}, (1.0, 0.9, 0.75, 0.5, 0.0)
    )
    loss = trajectory_loss(student, teacher, trajectory, backward=True)
    assert loss > 0
    assert len(teacher.seen) == len(student.seen) == 4
    for actual, expected in zip(student.seen, teacher.seen, strict=True):
        assert all(torch.equal(a, b) for a, b in zip(actual, expected, strict=True))
    assert not torch.equal(student.seen[0][0], student.seen[1][0])
    assert [n for n, p in student.named_parameters() if p.requires_grad] == ["proj.weight"]
    assert controller.parameters()[0].grad is not None
    assert all(p.grad is None for p in teacher.parameters())


def test_rejects_wrong_layer_census_before_mutating_model():
    model = Denoiser()
    with pytest.raises(ValueError, match="174"):
        prepare_mage_qad(model)
    assert isinstance(model.proj, nn.Linear)


def test_qad_pool_is_stratified_isolated_and_manifest_rejects_leaks():
    from shadow_lab.mage_qad_data import build_qad_pool, validate_qad_pool

    sources = [
        dict(
            prompt=f"new prompt {i}",
            source="new",
            source_revision="r1",
            source_id=str(i),
            stratum=str(i % 2),
        )
        for i in range(1200)
    ]
    evaluation = [dict(prompt=f"eval {i}", sample_id=f"e{i}", seed=i) for i in range(500)]
    rows, manifest = build_qad_pool(sources, evaluation)
    assert sum(r["split"] == "qad_train" for r in rows) == 1024
    assert sum(r["split"] == "qad_val" for r in rows) == 128
    assert len({r["seed"] for r in rows}) == 1152
    assert {r["seed"] for r in rows}.isdisjoint(range(500))
    assert manifest["strata"]["qad_val"] == {"0": 64, "1": 64}
    from shadow_lab.mage_qad_data import validate_qad_manifest

    validate_qad_manifest(rows, manifest)
    for field in ("seed", "sample_id"):
        incomplete = copy.deepcopy(manifest)
        incomplete["exclusion_hashes"][field].pop()
        with pytest.raises(ValueError, match="exclusion"):
            validate_qad_manifest(rows, incomplete)
    with pytest.raises(ValueError, match="evaluation"):
        validate_qad_manifest(rows, dict(manifest, evaluation_sha256=""))
    corrupted = copy.deepcopy(rows)
    corrupted[0]["prompt"] = "  EVAL 0 "
    with pytest.raises(ValueError, match="leak"):
        validate_qad_pool(corrupted, evaluation)


def test_checkpoint_resumes_optimizer_saliency_rng_and_packed_weights(tmp_path):
    from shadow_lab.mage_qad_checkpoint import load_checkpoint, save_checkpoint

    model = Denoiser()
    controller = prepare_mage_qad(model, expected_layers=1)
    optimizer = torch.optim.AdamW(controller.parameters(), lr=0.001)
    controller.bind_optimizer(optimizer)
    teacher = Denoiser().requires_grad_(False)
    batch = MageTrajectory(
        torch.randn(1, 2, 16), {"txt": torch.ones(1, 2, 16)}, (1.0, 0.9, 0.75, 0.5, 0.0)
    )
    trajectory_loss(model, teacher, batch, backward=True)
    optimizer.step()
    identity = {"model": "test", "data": "test", "code": "test", "config": "test"}
    path = tmp_path / "state.pt"
    saved = save_checkpoint(path, controller, optimizer, identity, {"step": 1})
    noise = torch.rand(3)
    optimizer.zero_grad()
    trajectory_loss(model, teacher, batch, backward=True)
    optimizer.step()
    expected = model.proj.weight.detach().clone()
    load_checkpoint(path, controller, optimizer, identity)
    assert torch.equal(torch.rand(3), noise)
    assert controller.saliency_layer_count == 1
    optimizer.zero_grad()
    trajectory_loss(model, teacher, batch, backward=True)
    optimizer.step()
    assert torch.equal(model.proj.weight, expected)
    assert saved["packed"]["proj"]["weight_packed"].dtype == torch.uint8
    with pytest.raises(ValueError, match="identity"):
        load_checkpoint(path, controller, optimizer, dict(identity, model="wrong"))


def test_campaign_failure_is_terminal_and_never_reads_held_out_splits(tmp_path):
    from shadow_lab.mage_qad_campaign import MageQADConfig, run_mage_qad
    from shadow_lab.mage_qad_data import build_qad_pool

    sources = [
        dict(prompt=f"p{i}", source="test", source_revision="v1", source_id=str(i), stratum="test")
        for i in range(1152)
    ]
    excluded = [dict(prompt=f"e{i}", sample_id=f"e{i}", seed=i) for i in range(500)]
    rows, manifest = build_qad_pool(sources, excluded)
    student, teacher = Denoiser(), Denoiser()
    controller = prepare_mage_qad(student, expected_layers=1)
    seen = []

    def source(row):
        seen.append(row["split"])
        raise RuntimeError("generation failed")

    cfg = MageQADConfig(formal_steps=1)
    report = run_mage_qad(
        controller,
        teacher,
        source,
        rows,
        manifest,
        {"model": "test", "code": "test"},
        cfg,
        tmp_path,
    )
    assert report["status"] == "failed"
    assert report["stage"] == "smoke"
    assert seen == ["qad_train"]
    with pytest.raises(RuntimeError, match="terminal"):
        run_mage_qad(
            controller,
            teacher,
            source,
            rows,
            manifest,
            {"model": "test", "code": "test"},
            cfg,
            tmp_path,
            resume=True,
        )


def test_healed_handoff_binds_master_saliency_layout_and_rejects_mutation(tmp_path):
    from shadow_lab.es_quant import ClipGenomeLayout
    from shadow_lab.mage_healed_master import HealedMaster
    from shadow_lab.mage_qad_checkpoint import save_checkpoint

    model = Denoiser()
    controller = prepare_mage_qad(model, expected_layers=1)
    optimizer = torch.optim.AdamW(controller.parameters())
    controller.bind_optimizer(optimizer)
    batch = MageTrajectory(
        torch.randn(1, 2, 16), {"txt": torch.ones(1, 2, 16)}, (1.0, 0.9, 0.75, 0.5, 0.0)
    )
    trajectory_loss(model, Denoiser(), batch, backward=True)
    optimizer.step()
    saved = save_checkpoint(tmp_path / "qad.pt", controller, optimizer, {}, {})
    original = Denoiser().requires_grad_(False)
    layout = ClipGenomeLayout.from_model(original)
    master = HealedMaster(saved, selected_sha256=saved["sha256"])
    master.validate_layout(layout)
    master.restore(original, "cpu")
    assert torch.equal(original.proj.weight, saved["latent"]["proj"])
    assert master.saliency_by_layer["proj"].sum() > 0
    assert not any(p.requires_grad for p in original.parameters())
    with pytest.raises(ValueError, match="selected"):
        HealedMaster(saved, selected_sha256="original-bf16")
    original.proj.weight.requires_grad_(True)
    with pytest.raises(ValueError, match="trainable"):
        master.restore(original, "cpu")
    original.requires_grad_(False)
    master._weights[0].value.add_(1)  # Fault injection: corrupted frozen storage.
    with pytest.raises(ValueError, match="mutated"):
        master.restore(original, "cpu")


def test_fresh_es_protocol_rejects_legacy_state_and_binds_new_sigma(tmp_path):
    from shadow_lab.mage_healed_es import validate_fresh_protocol
    from shadow_lab.mage_healed_master import HealedMaster
    from shadow_lab.mage_qad_checkpoint import content_hash, save_checkpoint

    model = Denoiser()
    controller = prepare_mage_qad(model, expected_layers=1)
    optimizer = torch.optim.AdamW(controller.parameters())
    controller.bind_optimizer(optimizer)
    batch = MageTrajectory(
        torch.randn(1, 2, 16), {"txt": torch.ones(1, 2, 16)}, (1.0, 0.9, 0.75, 0.5, 0.0)
    )
    trajectory_loss(model, Denoiser(), batch, backward=True)
    optimizer.step()
    artifact = save_checkpoint(tmp_path / "qad.pt", controller, optimizer, {}, {})
    master = HealedMaster(artifact, selected_sha256=artifact["sha256"])
    protocol = dict(
        selected_sha256=master.selected_sha256,
        initial_clips_sha256=content_hash(master.initial_clips),
        phase="sigma",
        sigma=0.001,
        alpha=0.002,
        seed=9008,
    )
    assert validate_fresh_protocol(protocol, master, "sigma").population_size == 8
    assert (
        validate_fresh_protocol(
            dict(protocol, phase="confirmation"), master, "confirmation"
        ).population_size
        == 30
    )
    with pytest.raises(ValueError, match="old"):
        validate_fresh_protocol(dict(protocol, center="legacy"), master, "sigma")
    with pytest.raises(ValueError, match="bound"):
        validate_fresh_protocol(dict(protocol, selected_sha256="bf16"), master, "sigma")
    with pytest.raises(ValueError):
        validate_fresh_protocol(dict(protocol, phase="expanded"), master, "expanded")
    # A selected genome must reconstruct the same packed weights with saved saliency.
    from shadow_lab.quasar import reconstruct_nvfp4_with_clip

    rebuilt = reconstruct_nvfp4_with_clip(
        artifact["latent"]["proj"],
        master.initial_clips,
        artifact["saliency"]["proj"],
        controller.config,
    )
    assert torch.equal(rebuilt.weight_packed, artifact["packed"]["proj"]["weight_packed"])


def test_all_174_linears_use_w4a4_and_nonquantized_modules_stay_frozen():
    from shadow_lab.quasar import QuasarNVFP4Linear, quantize_nvfp4_activation

    class Transformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([nn.Linear(16, 16) for _ in range(174)])
            self.norm = nn.LayerNorm(16)

        def forward(self, img, txt, timesteps):
            return sum(layer(self.norm(img)) for layer in self.layers) / 174

    pipeline = nn.ModuleDict(
        dict(transformer=Transformer(), qwen=nn.Linear(16, 16), vae=nn.Linear(16, 16))
    )
    pipeline.requires_grad_(False)
    controller = prepare_mage_qad(pipeline["transformer"])
    assert len(controller.parameters()) == 174
    assert sum(p.requires_grad for p in pipeline.parameters()) == 174
    assert all(
        isinstance(layer, QuasarNVFP4Linear)
        and layer.config.quantize_activations
        and layer.config.dynamic_activation_scale
        for layer in controller.layers.values()
    )
    layer = pipeline["transformer"].layers[0]
    x = torch.linspace(-3, 3, 16).reshape(1, 16)
    xq, _ = quantize_nvfp4_activation(x)
    expected = torch.nn.functional.linear(
        xq, layer.reconstruction().dequantize(layer.weight.dtype), layer.bias
    )
    assert torch.allclose(layer(x), expected, atol=1e-6)
    assert not torch.equal(xq, x)
    assert all(not p.requires_grad for p in pipeline["qwen"].parameters())
    assert all(not p.requires_grad for p in pipeline["vae"].parameters())


def test_staged_campaign_resume_matches_uninterrupted_training(tmp_path):
    from shadow_lab.mage_qad_campaign import MageQADConfig, run_mage_qad
    from shadow_lab.mage_qad_checkpoint import read_checkpoint
    from shadow_lab.mage_qad_data import build_qad_pool

    sources = [
        dict(prompt=f"p{i}", source="test", source_revision="v1", source_id=str(i), stratum="test")
        for i in range(1152)
    ]
    excluded = [dict(prompt=f"e{i}", sample_id=f"e{i}", seed=i) for i in range(500)]
    rows, manifest = build_qad_pool(sources, excluded)
    base, teacher = Denoiser(), Denoiser()

    def source(row):
        generator = torch.Generator().manual_seed(row["seed"])
        return MageTrajectory(
            torch.randn(1, 2, 16, generator=generator),
            {"txt": torch.ones(1, 2, 16)},
            (1.0, 0.9, 0.75, 0.5, 0.0),
        )

    config = MageQADConfig(formal_steps=1)
    identity = {"model": "test", "code": "test"}
    first = prepare_mage_qad(copy.deepcopy(base), expected_layers=1)
    report = run_mage_qad(
        first,
        teacher,
        source,
        rows,
        manifest,
        identity,
        config,
        tmp_path / "continuous",
        stop_after=2,
    )
    assert report["progress"]["stages"][0]["stage"] == "smoke"
    assert report["progress"]["stages"][0]["passed"]
    split = prepare_mage_qad(copy.deepcopy(base), expected_layers=1)
    run_mage_qad(
        split, teacher, source, rows, manifest, identity, config, tmp_path / "resumed", stop_after=1
    )
    run_mage_qad(
        split,
        teacher,
        source,
        rows,
        manifest,
        identity,
        config,
        tmp_path / "resumed",
        resume=True,
        stop_after=1,
    )
    a = read_checkpoint(tmp_path / "continuous" / "latest.pt")
    b = read_checkpoint(tmp_path / "resumed" / "latest.pt")
    assert a["sha256"] == b["sha256"]


def test_campaign_selects_earliest_best_and_early_stops_after_128_formal_steps(tmp_path):
    from shadow_lab.mage_qad_campaign import MageQADConfig, run_mage_qad
    from shadow_lab.mage_qad_data import build_qad_pool

    class FlatDenoiser(nn.Module):
        def __init__(self, weight):
            super().__init__()
            self.proj = nn.Linear(16, 16, bias=False)
            nn.init.constant_(self.proj.weight, weight)

        def forward(self, img, txt, timesteps):
            return self.proj(img)

    sources = [
        dict(prompt=f"p{i}", source="test", source_revision="v1", source_id=str(i), stratum="test")
        for i in range(1152)
    ]
    excluded = [dict(prompt=f"e{i}", sample_id=f"e{i}", seed=i) for i in range(500)]
    rows, manifest = build_qad_pool(sources, excluded)
    controller = prepare_mage_qad(FlatDenoiser(0.125), expected_layers=1)
    teacher = FlatDenoiser(0.25)

    def source(row):
        assert row["split"] in {"qad_train", "qad_val"}
        return MageTrajectory(
            torch.ones(1, 2, 16), {"txt": torch.ones(1, 2, 16)}, (1.0, 0.9, 0.75, 0.5, 0.0)
        )

    config = MageQADConfig(learning_rate=1e-30, weight_decay=0.0)
    report = run_mage_qad(
        controller,
        teacher,
        source,
        rows,
        manifest,
        {"model": "test", "code": "test"},
        config,
        tmp_path,
    )
    assert report["status"] == "selected", report.get("error")
    assert report["progress"]["best_step"] == 1
    assert [r["steps"] for r in report["progress"]["stages"]] == [1, 32, 256, 128]
    assert report["progress"]["stages"][-1]["early_stop"]
    assert report["selected_checkpoint"] == "best-1.pt"
    assert not report["es_authorized"]


def test_rare_qad_stratum_is_represented_in_both_train_and_validation():
    from shadow_lab.mage_qad_data import build_qad_pool

    sources = [
        dict(
            prompt=f"isolated {i}",
            source="test",
            source_revision="v1",
            source_id=str(i),
            stratum="rare" if i < 90 else "common",
        )
        for i in range(1200)
    ]
    excluded = [dict(prompt=f"eval {i}", sample_id=f"e{i}", seed=i) for i in range(500)]
    rows, manifest = build_qad_pool(sources, excluded)
    assert manifest["strata"]["qad_val"].get("rare", 0) == 10
    assert manifest["strata"]["qad_train"]["rare"] == 80
    assert len(rows) == 1152
