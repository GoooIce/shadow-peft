import hashlib
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from shadow_lab.es import clips_to_latent, latent_to_clips
from shadow_lab.image_eval import RewardCalibration
from shadow_lab.mage_healed_es_campaign import (
    CampaignStopped,
    ESCampaign,
    ESCampaignProtocol,
    exploration_gate,
    gradient_pairwise_stats,
)
from shadow_lab.mage_healed_master import HealedMaster
from shadow_lab.mage_qad import prepare_mage_qad
from shadow_lab.mage_qad_checkpoint import content_hash, save_checkpoint

NATIVE_KERNEL = "sm120_block_scaled_ue4m3xe2m1_ue4m3xe2m1"
AXES = ("hps_v2_1", "clip_h14")


def make_master(tmp_path, seed=20260908):
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(16, 16))
    controller = prepare_mage_qad(model, expected_layers=1)
    optimizer = torch.optim.AdamW(controller.parameters())
    controller.bind_optimizer(optimizer)
    model(torch.ones(1, 16)).sum().backward()
    optimizer.step()
    artifact = save_checkpoint(tmp_path / "qad.pt", controller, optimizer, {}, {})
    return HealedMaster(artifact, selected_sha256=artifact["sha256"])


def source_ledger(selected_sha256):
    return {
        "identity": {"selected_sha256": selected_sha256, "evaluation_sha256": "pool"},
        "gates": {
            "checkpoint_val": {"passed": True, "candidate_sha256": selected_sha256},
            "human_dev": {
                "passed": True,
                "status": "approved",
                "candidate_sha256": selected_sha256,
            },
        },
        "approvals": {
            "human_dev": {
                "passed": True,
                "approved": True,
                "candidate_sha256": selected_sha256,
                "maintainer": "tester",
                "wins": 29,
            }
        },
        "es": {"sigma": {"passed": False, "status": "failed"}},
        "success": False,
        "deployable": False,
    }


def small_protocol(master, **overrides):
    fields = dict(
        selected_sha256=master.selected_sha256,
        initial_clips_sha256=content_hash(master.initial_clips),
        dimension=master.layout.dimension,
        sigma=0.003,
        alphas=(1e-06, 1e-05),
        population_size=4,
        prompt_batch_size=2,
        gradient_episodes=(0, 1, 2, 3),
        selection_episodes=(4, 5),
        confirmation_episodes=(6, 7),
        consistency_checkpoints=((2, 0.10, 0.70), (4, 0.10, 0.70)),
        subset_populations=(2,),
        min_non_degrading_batches=2,
        max_failed_candidates=1,
        clip_step_rms_max=0.5,
        max_generations=2,
    )
    fields.update(overrides)
    return ESCampaignProtocol(**fields)


def samples(count=16):
    return [
        SimpleNamespace(sample_id=f"s{i}", prompt=f"p{i}", seed=i, split="es_train")
        for i in range(count)
    ]


def effective_initial(master):
    config = small_protocol(master).es_config()
    return latent_to_clips(clips_to_latent(master.initial_clips, config), config)


def calibration_for(batch):
    return RewardCalibration(
        {axis: 10.0 for axis in AXES},
        {axis: 2.0 for axis in AXES},
        {axis: 0.5 for axis in AXES},
        {sample.sample_id: {axis: 10.0 for axis in AXES} for sample in batch},
    )


class FakeRuntime:
    """Deterministic stand-in for MageHealingRuntime.evaluate.

    Reward is linear in the clip displacement along ``direction``; ``twist`` flips
    that direction per episode to manufacture inconsistent batch gradients.
    """

    def __init__(self, initial_clips, direction, *, gain=40.0):
        self.initial = initial_clips.clone()
        self.direction = direction
        self.gain = gain
        self.twist = None
        self.episode = 0
        self.tamper_replay = False
        self.fail_indices = set()
        self.teacher_for_alpha = {}
        self.calls = []

    def score(self, clips, directory):
        direction = self.direction
        if self.twist is not None and self.twist(self.episode):
            direction = -direction
        signal = float(torch.dot(clips - self.initial, direction))
        return 10.0 + self.gain * signal

    def __call__(self, clips, batch, directory, *, candidate_sha256, replay=True, arm="qad"):
        directory.mkdir(parents=True, exist_ok=False)
        self.calls.append((directory.name, replay))
        value = self.score(clips, directory)
        name = directory.name
        if any(name == f"candidate-{index:02d}" for index in self.fail_indices):
            value = 5.0
        teacher = self.teacher_for_alpha.get(name, 1.0)
        rows = []
        for index, sample in enumerate(batch):
            payload = f"{candidate_sha256}-{sample.sample_id}".encode()
            path = directory / f"{index:03d}.png"
            path.write_bytes(payload)
            sha = hashlib.sha256(payload).hexdigest()
            scores = {axis: value for axis in AXES}
            row = dict(
                sample_id=sample.sample_id,
                scores=scores,
                image_path=str(path),
                image_sha256=sha,
                teacher_loss=teacher,
            )
            if replay:
                replay_scores = dict(scores)
                if self.tamper_replay and name == "candidate-00":
                    replay_scores[AXES[0]] += 1.0
                row.update(replay_sha256=sha, replay_scores=replay_scores)
            rows.append(row)
        report = dict(
            candidate_sha256=candidate_sha256,
            rows=rows,
            failures=0,
            compute_capability=[12, 1],
            kernel_names=[NATIVE_KERNEL],
            sample_ids=[sample.sample_id for sample in batch],
            weight_sha256=f"packed-{candidate_sha256}",
            replay=replay,
        )
        if replay:
            report["replay_weight_sha256"] = report["weight_sha256"]
        return report


def new_campaign(tmp_path, master, protocol=None):
    protocol = protocol or small_protocol(master)
    ledger = source_ledger(master.selected_sha256)
    return ESCampaign.create(
        tmp_path / "exp009",
        protocol=protocol,
        master=master,
        identity={"evaluation_sha256": "pool", "code": "test", "model": "test"},
        source_ledger=ledger,
        source_ledger_sha256=content_hash(ledger),
    )


def batch_for(pool, episode, size=2):
    return pool[episode * size : (episode + 1) * size]


def run_gradient(campaign, master, runtime, pool, episodes, **kwargs):
    for episode in episodes:
        runtime.episode = episode
        batch = batch_for(pool, episode)
        campaign.run_gradient_episode(
            master, episode, batch, calibration_for(batch), runtime, **kwargs
        )


def test_protocol_validation_bans_default_sigma_and_binds_master(tmp_path):
    master = make_master(tmp_path)
    protocol = small_protocol(master)
    assert protocol.alphas == (1e-06, 1e-05)
    for bad in (
        dict(sigma=0.03),
        dict(sigma=0.02),
        dict(alphas=(1e-05, 1e-06)),
        dict(selection_episodes=(0, 5)),
        dict(consistency_checkpoints=((5, 0.1, 0.7),)),
        dict(subset_populations=(4,)),
        dict(population_size=5),
        dict(min_non_degrading_batches=3),
        dict(replay_audit_index=4),
        dict(initial_clips_sha256="0" * 64),
    ):
        with pytest.raises(ValueError):
            candidate = small_protocol(master, **bad)
            candidate.bind(master)
    # JSON round trip keeps the frozen record identical.
    restored = ESCampaignProtocol(**json.loads(json.dumps(protocol.__dict__)))
    assert restored == protocol
    other = make_master(tmp_path / "other", seed=1)
    with pytest.raises(ValueError, match="bound"):
        protocol.bind(other)


def test_exploration_gate_uses_floors_and_tolerance_without_replay_evidence(tmp_path):
    master = make_master(tmp_path)
    runtime = FakeRuntime(effective_initial(master), torch.zeros(16))
    batch = samples(2)
    calibration = calibration_for(batch)
    report = runtime(
        master.initial_clips, batch, tmp_path / "c", candidate_sha256="c", replay=False
    )
    ids = [s.sample_id for s in batch]
    gate = exploration_gate(report, calibration, ids, candidate_sha256="c", bf16_floor=0.8)
    assert gate["passed"] and gate["reward"] == 0.0 and gate["teacher_mean"] == 1.0
    assert not exploration_gate(
        report,
        calibration,
        ids,
        candidate_sha256="c",
        bf16_floor=0.8,
        trajectory_tolerance=(0.9, 1.05),
    )["passed"]
    for row in report["rows"]:
        row["scores"] = {axis: 7.9 for axis in AXES}
    floors = exploration_gate(report, calibration, ids, candidate_sha256="c", bf16_floor=0.8)
    assert floors["reasons"] == [f"{axis} below 80% BF16 floor" for axis in sorted(AXES)]
    assert not exploration_gate(
        dict(report, kernel_names=[]), calibration, ids, candidate_sha256="c", bf16_floor=0.8
    )["passed"]


def test_full_generation_flow_with_common_noise_selection_confirmation_and_continuation(tmp_path):
    master = make_master(tmp_path)
    campaign = new_campaign(tmp_path, master)
    assert campaign.workflow.read()["inherited_from"]["healing_sha256"] == content_hash(
        source_ledger(master.selected_sha256)
    )
    direction = torch.ones(16)
    runtime = FakeRuntime(effective_initial(master), direction)
    pool = samples()
    run_gradient(campaign, master, runtime, pool, (0, 1, 2, 3))
    state = campaign.state
    generation = state["generations"][0]
    episodes = generation["gradient"]["episodes"]
    assert sorted(episodes) == ["0", "1", "2", "3"]
    # Common antithetic noise: identical seeds across the generation's episodes.
    seeds = {
        key: [spec["noise_seed"] for spec in record["specs"]] for key, record in episodes.items()
    }
    assert len({tuple(value) for value in seeds.values()}) == 1
    assert all(
        record["replay_audit"] == {"candidate": 0, "exact": True} for record in episodes.values()
    )
    assert [call for call in runtime.calls if call[0] == "candidate-00"] == [
        ("candidate-00", True)
    ] * 4
    assert all(call[1] is False for call in runtime.calls if call[0] in {"center", "candidate-01"})
    consistency = generation["consistency"]
    assert consistency["episodes"] == [0, 1, 2, 3]
    assert set(consistency["populations"]) == {"4", "2"}
    assert consistency["populations"]["4"]["mean_cosine"] > 0.9
    assert [check["checkpoint"] for check in consistency["checks"]] == [2, 4]
    assert all(check["passed"] for check in consistency["checks"])
    written = json.loads((tmp_path / "exp009" / "generation-00" / "consistency.json").read_text())
    assert len(written["populations"]["4"]["pairwise"]) == 6

    with pytest.raises(RuntimeError, match="gradient"):
        campaign.run_selection_episode(
            master, 4, batch_for(pool, 4), calibration_for(pool), runtime
        )
    step = campaign.step(master)
    assert state["status"] == "selection"
    assert [c["alpha"] for c in step["candidates"]] == [1e-06, 1e-05]
    assert all(c["valid"] for c in step["candidates"])
    assert step["candidates"][1]["clip_step_rms"] > step["candidates"][0]["clip_step_rms"]

    for episode in (4, 5):
        batch = batch_for(pool, episode)
        record = campaign.run_selection_episode(
            master, episode, batch, calibration_for(batch), runtime
        )
        assert all(c["gate_passed"] for c in record["candidates"].values())
        assert (
            record["candidates"]["1e-05"]["reward_delta"]
            > record["candidates"]["1e-06"]["reward_delta"]
            > 0
        )
    decision = campaign.decide_selection()
    assert decision["selected_alpha"] == 1e-05 and state["status"] == "confirmation"

    for episode in (6, 7):
        batch = batch_for(pool, episode)
        record = campaign.run_confirmation_episode(
            master, episode, batch, calibration_for(batch), runtime
        )
        assert list(record["candidates"]) == ["1e-05"]
    report = campaign.decide_confirmation()
    assert report["passed"] and state["status"] == "awaiting_maintainer"
    ledger = campaign.workflow.read()
    assert ledger["es"]["confirmation"]["candidate_sha256"] == report["candidate_sha256"]
    artifact = torch.load(report["clips_path"], weights_only=True)
    assert content_hash(artifact) == report["clips_sha256"]
    assert artifact["selected_sha256"] == master.selected_sha256
    with pytest.raises(RuntimeError, match="confirmation"):
        campaign.decide_confirmation()

    # Final holdouts stay sealed until the maintainer declares the last generation;
    # an authorized continuation starts from the confirmed center with fresh noise.
    generation_two = campaign.authorize_generation("tester")
    assert generation_two["index"] == 1 and state["status"] == "gradient"
    assert generation_two["start_clips_sha256"] == content_hash(artifact["clips"])
    assert torch.equal(campaign.start_clips(master), artifact["clips"])
    assert campaign.es_state(master).generation == 1
    runtime.episode = 0
    batch = batch_for(pool, 0)
    record = campaign.run_gradient_episode(master, 0, batch, calibration_for(batch), runtime)
    assert [s["noise_seed"] for s in record["specs"]] != seeds["0"]
    with pytest.raises(RuntimeError, match="awaiting_maintainer"):
        campaign.declare_final_generation("tester")
    with pytest.raises(RuntimeError, match="awaiting_maintainer"):
        campaign.authorize_generation("tester")


def test_replay_audit_mismatch_and_failed_candidate_budget_stop_the_campaign(tmp_path):
    master = make_master(tmp_path)
    pool = samples()
    runtime = FakeRuntime(effective_initial(master), torch.ones(16))
    campaign = new_campaign(tmp_path / "audit", master)
    runtime.tamper_replay = True
    with pytest.raises(CampaignStopped, match="replay audit"):
        run_gradient(campaign, master, runtime, pool, (0,))
    assert campaign.state["status"] == "stopped"
    assert campaign.state["verdict"] == "episode-invalid"
    with pytest.raises(RuntimeError, match="stopped"):
        run_gradient(campaign, master, runtime, pool, (1,))

    runtime = FakeRuntime(effective_initial(master), torch.ones(16))
    runtime.fail_indices = {1, 2}
    tolerant = new_campaign(
        tmp_path / "tolerant", master, small_protocol(master, max_failed_candidates=2)
    )
    run_gradient(tolerant, master, runtime, pool, (0,))
    record = tolerant.state["generations"][0]["gradient"]["episodes"]["0"]
    assert record["failure_count"] == 2
    assert [c["gate"]["passed"] for c in record["candidates"]] == [True, False, False, True]
    assert record["candidates"][1]["gate"]["reasons"] == [
        f"{axis} below 80% BF16 floor" for axis in sorted(AXES)
    ]

    strict = new_campaign(
        tmp_path / "strict", master, small_protocol(master, max_failed_candidates=1)
    )
    with pytest.raises(CampaignStopped, match="failed catastrophe floors"):
        run_gradient(strict, master, runtime, pool, (0,))
    assert strict.state["verdict"] == "episode-invalid"


def test_consistency_stop_rule_records_a_negative_result(tmp_path):
    master = make_master(tmp_path)
    campaign = new_campaign(tmp_path, master)
    runtime = FakeRuntime(effective_initial(master), torch.ones(16))
    runtime.twist = lambda episode: episode % 2 == 1
    pool = samples()
    run_gradient(campaign, master, runtime, pool, (0,))
    assert campaign.state["generations"][0]["consistency"]["checks"] == []
    with pytest.raises(CampaignStopped, match="no-gradient-signal"):
        run_gradient(campaign, master, runtime, pool, (1,))
    check = campaign.state["generations"][0]["consistency"]["checks"][0]
    assert check["checkpoint"] == 2 and not check["passed"]
    assert check["mean_cosine"] < -0.9
    assert campaign.state["status"] == "stopped"
    assert campaign.state["verdict"] == "no-gradient-signal"
    with pytest.raises(RuntimeError, match="stopped"):
        campaign.step(master)


def test_selection_and_confirmation_enforce_mean_delta_batches_and_teacher_band(tmp_path):
    master = make_master(tmp_path)
    pool = samples()
    runtime = FakeRuntime(effective_initial(master), torch.ones(16))
    campaign = new_campaign(tmp_path / "select", master)
    run_gradient(campaign, master, runtime, pool, (0, 1, 2, 3))
    campaign.step(master)
    # The larger step regresses the teacher trajectory beyond the 5% band: the
    # smaller alpha is the only one that may be selected.
    runtime.teacher_for_alpha = {"alpha-0p000010": 1.2}
    for episode in (4, 5):
        batch = batch_for(pool, episode)
        record = campaign.run_selection_episode(
            master, episode, batch, calibration_for(batch), runtime
        )
        assert not record["candidates"]["1e-05"]["gate_passed"]
        assert record["candidates"]["1e-05"]["reasons"] == [
            "teacher trajectory batch mean exceeds tolerance"
        ]
    decision = campaign.decide_selection()
    assert decision["selected_alpha"] == 1e-06
    summaries = {s["alpha"]: s for s in decision["summaries"]}
    assert not summaries[1e-05]["passes"] and summaries[1e-06]["passes"]

    runtime.teacher_for_alpha = {}
    runtime.direction = -torch.ones(16)  # the confirmed step now regresses reward
    for episode in (6, 7):
        batch = batch_for(pool, episode)
        campaign.run_confirmation_episode(master, episode, batch, calibration_for(batch), runtime)
    with pytest.raises(CampaignStopped, match="confirmation-failed"):
        campaign.decide_confirmation()
    assert "confirmation" not in campaign.workflow.read()["es"]

    failing = new_campaign(tmp_path / "none", master)
    runtime = FakeRuntime(effective_initial(master), torch.ones(16))
    run_gradient(failing, master, runtime, pool, (0, 1, 2, 3))
    failing.step(master)
    runtime.direction = -torch.ones(16)
    for episode in (4, 5):
        batch = batch_for(pool, episode)
        failing.run_selection_episode(master, episode, batch, calibration_for(batch), runtime)
    with pytest.raises(CampaignStopped, match="selection-failed"):
        failing.decide_selection()


def test_remote_episode_registration_verifies_artifacts_and_schedule(tmp_path):
    master = make_master(tmp_path)
    campaign = new_campaign(tmp_path, master)
    runtime = FakeRuntime(effective_initial(master), torch.ones(16))
    pool = samples()
    run_gradient(campaign, master, runtime, pool, (0,), register=False)
    assert campaign.state["generations"][0]["gradient"]["episodes"] == {}
    directory = tmp_path / "exp009" / "generation-00" / "gradient-episode-00"
    assert (directory / "episode.json").exists() and (directory / "gradient.pt").exists()
    with pytest.raises(FileExistsError):
        run_gradient(campaign, master, runtime, pool, (0,), register=False)
    payload = torch.load(directory / "gradient.pt", weights_only=True)
    payload["gradient"] = payload["gradient"] + 1.0
    torch.save(payload, directory / "gradient.pt")
    with pytest.raises(ValueError, match="identity"):
        campaign.register_episode(0)
    # A forced rerun reopens only an unfinished episode; this one is finished.
    with pytest.raises(FileExistsError):
        run_gradient(campaign, master, runtime, pool, (0,), register=False, force=True)
    (directory / "episode.json").unlink()
    run_gradient(campaign, master, runtime, pool, (0,), register=False, force=True)
    campaign.register_episode(0)
    assert campaign.state["generations"][0]["gradient"]["completed"] == [0]
    with pytest.raises(FileExistsError):
        campaign.register_episode(0)


def test_gradient_pairwise_stats_reports_cosine_sign_and_positive_fraction():
    a = torch.tensor([1.0, 0.0, 0.0, 1.0])
    b = torch.tensor([1.0, 0.0, 0.0, -1.0])
    stats = gradient_pairwise_stats([a, a, b])
    assert stats["pair_count"] == 3
    assert stats["positive_fraction"] == pytest.approx(1 / 3)
    assert stats["pairwise"][0]["cosine"] == pytest.approx(1.0)
    assert stats["pairwise"][1]["cosine"] == pytest.approx(0.0)
    with pytest.raises(ValueError):
        gradient_pairwise_stats([a])
    with pytest.raises(ValueError):
        gradient_pairwise_stats([a, torch.zeros(4)])
