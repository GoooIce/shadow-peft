import copy
import hashlib
from pathlib import Path

import pytest

from shadow_lab.image_eval import RewardCalibration
from shadow_lab.mage_healing import quality_gate


def evidence(tmp_path, n=60):
    rows = []
    for i in range(n):
        path = tmp_path / f"{i}.png"
        path.write_bytes(f"image{i}".encode())
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append(
            dict(
                sample_id=f"s{i}",
                scores={"hps_v2_1": 10.0, "clip_h14": 10.0},
                image_path=str(path),
                image_sha256=sha,
                replay_sha256=sha,
                replay_scores={"hps_v2_1": 10.0, "clip_h14": 10.0},
                teacher_loss=1.0,
            )
        )
    calibration = RewardCalibration(
        dict(hps_v2_1=10.0, clip_h14=10.0),
        dict(hps_v2_1=2.0, clip_h14=2.0),
        dict(hps_v2_1=0.5, clip_h14=0.5),
        {r["sample_id"]: r["scores"].copy() for r in rows},
    )
    return dict(
        rows=rows,
        failures=0,
        candidate_sha256="qad",
        weight_sha256="packed",
        replay_weight_sha256="packed",
        compute_capability=[12, 1],
        kernel_names=["sm120_block_scaled_ue4m3xe2m1_ue4m3xe2m1"],
    ), calibration


def test_quality_gate_requires_absolute_axes_native_replay_and_complete_pairs(tmp_path):
    result, calibration = evidence(tmp_path)
    ids = [r["sample_id"] for r in result["rows"]]
    assert quality_gate(result, calibration, ids, candidate_sha256="qad")["passed"]
    bad = copy.deepcopy(result)
    for row in bad["rows"]:
        row["scores"] = dict(hps_v2_1=7.9, clip_h14=30.0)
        row["replay_scores"] = row["scores"].copy()
    assert not quality_gate(bad, calibration, ids, candidate_sha256="qad")["passed"]
    for field, value in (("kernel_names", []), ("failures", 1), ("candidate_sha256", "old")):
        assert not quality_gate(
            dict(result, **{field: value}), calibration, ids, candidate_sha256="qad"
        )["passed"]
    result["rows"][0]["replay_sha256"] = "different"
    assert not quality_gate(result, calibration, ids, candidate_sha256="qad")["passed"]


def test_final_gate_rejects_compensating_axes_and_trajectory_regressions(tmp_path):
    result, calibration = evidence(tmp_path)
    ids = [r["sample_id"] for r in result["rows"]]
    for row in result["rows"]:
        row["scores"] = dict(hps_v2_1=9.0, clip_h14=12.0)
        row["replay_scores"] = row["scores"].copy()
    assert not quality_gate(result, calibration, ids, candidate_sha256="qad", final=True)["passed"]
    for row in result["rows"]:
        row["scores"] = dict(hps_v2_1=11.0, clip_h14=11.0)
        row["replay_scores"] = row["scores"].copy()
    assert quality_gate(result, calibration, ids, candidate_sha256="qad", final=True)["passed"]
    assert not quality_gate(
        result,
        calibration,
        ids,
        candidate_sha256="qad",
        trajectory_limits={sid: 0.9 for sid in ids},
    )["passed"]


def test_holdout_is_consumed_before_callback_even_on_crash(tmp_path):
    from shadow_lab.mage_healing import HealingWorkflow

    workflow = HealingWorkflow(tmp_path, selected_sha256="qad", evaluation_sha256="pool")
    calls = []

    def crash():
        calls.append(True)
        raise RuntimeError("crash")

    assert not workflow.once("checkpoint_val", "qad", crash)["passed"]
    with pytest.raises(FileExistsError):
        workflow.once("checkpoint_val", "qad", crash)
    assert len(calls) == 1
    with pytest.raises(RuntimeError, match="approval"):
        workflow.es_permit()
    with pytest.raises(RuntimeError, match="confirmation"):
        workflow.once("auto_test", "final", crash)


def test_trajectory_tolerance_is_a_batch_mean_band_not_a_per_sample_bound(tmp_path):
    result, calibration = evidence(tmp_path, 8)
    ids = [r["sample_id"] for r in result["rows"]]
    # Half the samples regress by 8%, half improve by 4%: per-sample limits fail,
    # the batch-mean band (baseline 1.0, factor 1.05) passes at mean 1.02.
    for index, row in enumerate(result["rows"]):
        row["teacher_loss"] = 1.08 if index % 2 else 0.96
    assert quality_gate(
        result, calibration, ids, candidate_sha256="qad", trajectory_tolerance=(1.0, 1.05)
    )["passed"]
    assert not quality_gate(
        result,
        calibration,
        ids,
        candidate_sha256="qad",
        trajectory_limits={sid: 1.0 for sid in ids},
    )["passed"]
    tight = quality_gate(
        result, calibration, ids, candidate_sha256="qad", trajectory_tolerance=(1.0, 1.01)
    )
    assert tight["reasons"] == ["teacher trajectory batch mean exceeds tolerance"]
    result["rows"][0]["teacher_loss"] = float("nan")
    assert not quality_gate(
        result, calibration, ids, candidate_sha256="qad", trajectory_tolerance=(1.0, 1.05)
    )["passed"]


def test_inherited_ledger_carries_only_the_master_permits_and_verifies_source(tmp_path):
    import copy as _copy

    from shadow_lab.mage_healing import HealingWorkflow
    from shadow_lab.mage_qad_checkpoint import content_hash

    source = HealingWorkflow(tmp_path / "exp008", selected_sha256="qad", evaluation_sha256="pool")
    source.once("checkpoint_val", "qad", lambda: dict(passed=True, candidate_sha256="qad"))
    state = source.read()
    with pytest.raises(ValueError, match="human_dev"):
        HealingWorkflow.inherit(
            tmp_path / "early",
            state,
            selected_sha256="qad",
            evaluation_sha256="pool",
            source_sha256=content_hash(state),
        )
    state["gates"]["human_dev"] = dict(passed=True, status="approved", candidate_sha256="qad")
    state["approvals"]["human_dev"] = dict(
        passed=True, approved=True, candidate_sha256="qad", maintainer="tester", wins=29
    )
    state["es"]["sigma"] = dict(passed=False, status="failed")
    with pytest.raises(ValueError, match="hash"):
        HealingWorkflow.inherit(
            tmp_path / "bad-hash",
            state,
            selected_sha256="qad",
            evaluation_sha256="pool",
            source_sha256="0" * 64,
        )
    with pytest.raises(ValueError, match="another master"):
        HealingWorkflow.inherit(
            tmp_path / "bad-master",
            state,
            selected_sha256="other",
            evaluation_sha256="pool",
            source_sha256=content_hash(state),
        )
    inherited = HealingWorkflow.inherit(
        tmp_path / "exp009",
        _copy.deepcopy(state),
        selected_sha256="qad",
        evaluation_sha256="pool",
        source_sha256=content_hash(state),
    )
    ledger = inherited.read()
    assert ledger["es"] == {} and not ledger["deployable"] and not ledger["success"]
    assert ledger["inherited_from"] == {"healing_sha256": content_hash(state)}
    assert inherited.es_permit()["selected_sha256"] == "qad"
    with pytest.raises(RuntimeError, match="confirmation"):
        inherited.once("auto_test", "final", lambda: {})
    with pytest.raises(FileExistsError):
        HealingWorkflow.inherit(
            tmp_path / "exp009",
            state,
            selected_sha256="qad",
            evaluation_sha256="pool",
            source_sha256=content_hash(state),
        )


def test_blind_gate_hides_arms_and_requires_24_wins_and_explicit_human(tmp_path):
    Image = pytest.importorskip("PIL.Image")
    from shadow_lab.mage_healing import HealingWorkflow
    from shadow_lab.mage_healing_blind import build_blind_packet, judge_ballots

    image = tmp_path / "source.png"
    Image.new("RGB", (2, 2)).save(image)
    arms = {
        arm: {f"s{i}": image for i in range(30)} for arm in ("bf16", "rtn", "quasar-grid", "qad")
    }
    packet = build_blind_packet(tmp_path / "blind", arms, candidate_sha256="qad")
    public = (tmp_path / "blind" / "public" / "packet.json").read_text()
    assert all(arm not in public for arm in arms)
    import json

    private = json.loads(Path(packet["private_key"]).read_text())
    ballots = []
    for trial in private["trials"]:
        labels = trial["labels"]
        ranking = sorted(labels, key=lambda label: 0 if labels[label] == "qad" else 1)
        ballots.append(dict(trial_id=trial["trial_id"], ranking=ranking, broken=[]))
    assert judge_ballots(Path(packet["private_key"]), ballots)["wins"] == 30
    workflow = HealingWorkflow(
        tmp_path / "workflow", selected_sha256="qad", evaluation_sha256="pool"
    )
    workflow.once("checkpoint_val", "qad", lambda: dict(passed=True, candidate_sha256="qad"))
    workflow.once("human_dev", "qad", lambda: packet)
    with pytest.raises(RuntimeError):
        workflow.es_permit()
    workflow.approve_human("human_dev", ballots, maintainer="maintainer", approved=True)
    assert workflow.es_permit()["selected_sha256"] == "qad"
    assert not workflow.read()["deployable"]
    ballots[0]["broken"] = [
        next(label for label, arm in private["trials"][0]["labels"].items() if arm == "qad")
    ]
    assert not judge_ballots(Path(packet["private_key"]), ballots)["passed"]


def test_fresh_es_phases_and_final_gates_require_order_and_explicit_decision(tmp_path):
    import json
    from types import SimpleNamespace

    import torch
    from torch import nn

    from shadow_lab.mage_healed_es import run_healed_es
    from shadow_lab.mage_healed_master import HealedMaster
    from shadow_lab.mage_healing import HealingWorkflow
    from shadow_lab.mage_healing_blind import build_blind_packet
    from shadow_lab.mage_qad import prepare_mage_qad
    from shadow_lab.mage_qad_checkpoint import content_hash, save_checkpoint

    Image = pytest.importorskip("PIL.Image")
    model = nn.Sequential(nn.Linear(16, 16))
    controller = prepare_mage_qad(model, expected_layers=1)
    opt = torch.optim.AdamW(controller.parameters())
    controller.bind_optimizer(opt)
    model(torch.ones(1, 16)).sum().backward()
    opt.step()
    artifact = save_checkpoint(tmp_path / "qad.pt", controller, opt, {}, {})
    master = HealedMaster(artifact, selected_sha256=artifact["sha256"])
    workflow = HealingWorkflow(
        tmp_path / "workflow", selected_sha256=master.selected_sha256, evaluation_sha256="pool"
    )
    image = tmp_path / "source.png"
    Image.new("RGB", (2, 2)).save(image)

    def blind(split, sha):
        arms = {
            arm: {f"s{i}": image for i in range(30)}
            for arm in ("bf16", "rtn", "quasar-grid", "qad")
        }
        packet = build_blind_packet(tmp_path / split, arms, candidate_sha256=sha)
        key = json.loads(Path(packet["private_key"]).read_text())
        ballots = [
            dict(
                trial_id=t["trial_id"],
                broken=[],
                ranking=sorted(
                    t["labels"], key=lambda label: 0 if t["labels"][label] == "qad" else 1
                ),
            )
            for t in key["trials"]
        ]
        return packet, ballots

    workflow.once(
        "checkpoint_val",
        master.selected_sha256,
        lambda: dict(passed=True, candidate_sha256=master.selected_sha256),
    )
    packet, ballots = blind("human_dev", master.selected_sha256)
    workflow.once("human_dev", master.selected_sha256, lambda: packet)
    workflow.approve_human("human_dev", ballots, maintainer="tester", approved=True)
    protocol = dict(
        selected_sha256=master.selected_sha256,
        initial_clips_sha256=content_hash(master.initial_clips),
        phase="sigma",
        sigma=0.001,
        alpha=0.001,
        seed=9008,
    )
    samples = [SimpleNamespace(sample_id=f"s{i}", split="es_train") for i in range(64)]
    axes = dict(hps_v2_1=10.0, clip_h14=10.0)
    calibration = RewardCalibration(
        axes,
        dict(hps_v2_1=2.0, clip_h14=2.0),
        dict(hps_v2_1=0.5, clip_h14=0.5),
        {s.sample_id: axes for s in samples},
    )
    evaluations = []

    def evaluate(clips, batch, directory, *, candidate_sha256):
        directory.mkdir(parents=True)
        result, _ = evidence(directory, len(batch))
        gain = 0.0 if directory.name == "healed-baseline" else 0.2
        if directory.name.startswith("candidate-"):
            gain += int(directory.name.split("-")[-1]) * 0.01
        for row, sample in zip(result["rows"], batch, strict=True):
            row["sample_id"] = sample.sample_id
            row["scores"] = {axis: 10 + gain for axis in axes}
            row["replay_scores"] = row["scores"].copy()
        result["candidate_sha256"] = candidate_sha256
        evaluations.append(directory.name)
        return result

    with pytest.raises(RuntimeError, match="N=8"):
        run_healed_es(
            workflow,
            master,
            dict(protocol, phase="confirmation"),
            samples[32:],
            calibration,
            evaluate,
            tmp_path / "premature",
        )
    assert not evaluations
    first = run_healed_es(
        workflow, master, protocol, samples[:32], calibration, evaluate, tmp_path / "sigma"
    )
    assert first["passed"], first
    assert len(evaluations) == 10  # baseline + N=8 + updated center
    second = run_healed_es(
        workflow,
        master,
        dict(protocol, phase="confirmation"),
        samples[32:],
        calibration,
        evaluate,
        tmp_path / "confirmation",
    )
    assert second["passed"], second
    assert len(evaluations) == 42  # plus baseline + N=30 + updated center
    assert not workflow.read()["success"]
    sha = second["candidate_sha256"]
    final_eval, final_calibration = evidence(tmp_path, 60)
    final_eval["candidate_sha256"] = sha
    for row in final_eval["rows"]:
        row["scores"] = dict(hps_v2_1=11.0, clip_h14=11.0)
        row["replay_scores"] = row["scores"].copy()
    gate = workflow.once(
        "auto_test",
        sha,
        lambda: quality_gate(
            final_eval,
            final_calibration,
            [r["sample_id"] for r in final_eval["rows"]],
            candidate_sha256=sha,
            final=True,
        ),
    )
    assert gate["passed"]
    with pytest.raises(FileExistsError):
        workflow.once("auto_test", sha, lambda: gate)
    assert not workflow.read()["deployable"]
    packet, ballots = blind("human_test", sha)
    workflow.once("human_test", sha, lambda: packet)
    assert not workflow.read()["deployable"]
    workflow.approve_human("human_test", ballots, maintainer="tester", approved=True)
    assert workflow.read()["success"] and workflow.read()["deployable"]
