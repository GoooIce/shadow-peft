#!/usr/bin/env python3
"""EXP-009: consistency-gated multi-batch ES on the EXP-008 healed master.

Every GB10 command re-derives the runtime identity and refuses to run unless the
model, Mage source, dependency versions and this campaign's frozen code identity all
match. CPU-only ledger commands (consistency, decisions, continuation) never touch
the model.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import sys
from dataclasses import replace
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_mage_flow_healing import (  # noqa: E402
    file_hash,
    read_jsonl,
    runtime_identity,
    selected_campaign,
)

from shadow_lab.image_eval import (  # noqa: E402
    EVAL_SPLIT_COUNTS,
    RewardCalibration,
    episode_prompt_batch,
    load_frozen_eval_samples,
    validate_frozen_eval_pool,
)
from shadow_lab.mage_healed_es_campaign import ESCampaign, ESCampaignProtocol  # noqa: E402
from shadow_lab.mage_healed_master import HealedMaster  # noqa: E402
from shadow_lab.mage_healing import quality_gate  # noqa: E402
from shadow_lab.mage_healing_blind import build_blind_packet  # noqa: E402
from shadow_lab.mage_healing_runtime import MageHealingRuntime, require_gb10  # noqa: E402
from shadow_lab.mage_qad_campaign import MageQADConfig, atomic_json  # noqa: E402
from shadow_lab.mage_qad_checkpoint import content_hash  # noqa: E402
from shadow_lab.mage_qad_data import data_hash  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
IDENTITY_KEYS = ("model", "mage_source", "versions")


def code_identity() -> str:
    paths = [
        Path(__file__).resolve(),
        REPO / "scripts" / "run_mage_flow_healing.py",
        *sorted((REPO / "src" / "shadow_lab").glob("*.py")),
    ]
    return content_hash({str(p.relative_to(REPO)): file_hash(p) for p in paths})


def load_master(source_campaign: Path) -> tuple[dict, HealedMaster]:
    """Build the healed master and drop the 40 GiB checkpoint before any generation."""
    report, checkpoint = selected_campaign(source_campaign)
    master = HealedMaster(checkpoint, selected_sha256=checkpoint["sha256"])
    del checkpoint
    gc.collect()
    return report, master


def source_ledger(source_campaign: Path) -> tuple[dict, str]:
    ledger = json.loads((source_campaign / "healing" / "healing.json").read_text())
    return ledger, content_hash(ledger)


def scoring_identity(args) -> dict:
    return {
        "calibration": file_hash(args.calibration),
        "hps": file_hash(args.hps_checkpoint),
        "clip": file_hash(args.clip_checkpoint),
    }


def frozen_samples(args, identity: dict):
    if data_hash(read_jsonl(args.eval_pool)) != identity["evaluation_sha256"]:
        raise ValueError("evaluation pool identity differs from the QAD exclusions")
    samples = load_frozen_eval_samples(args.eval_pool)
    validate_frozen_eval_pool(samples)
    return samples


def episode_batch(campaign: ESCampaign, samples, episode: int):
    protocol = campaign.protocol
    return episode_prompt_batch(
        [s for s in samples if s.split == "es_train"],
        episode,
        batch_size=protocol.prompt_batch_size,
        schedule_seed=protocol.schedule_seed,
    )


def verify_runtime(campaign: ESCampaign, args, report: dict) -> None:
    identity = runtime_identity(args.model_path)
    frozen = campaign.state["identity"]
    for key in IDENTITY_KEYS:
        if identity[key] != frozen[key] or identity[key] != report["identity"][key]:
            raise ValueError(f"runtime {key} identity differs from the frozen campaign")
    if code_identity() != frozen["code"]:
        raise ValueError("code identity differs from the frozen EXP-009 campaign")
    if scoring_identity(args) != frozen["scorers"]:
        raise ValueError("scorer or calibration identity changed")


class Runtime:
    """GB10 generation runtime plus per-episode paired BF16 references."""

    def __init__(self, args, report: dict, master: HealedMaster):
        from mage_flow import MageFlowPipeline  # ty: ignore[unresolved-import]

        from shadow_lab.image_reward_models import HPSv21Scorer, OpenCLIPH14Scorer

        pipeline = MageFlowPipeline.from_pretrained(str(args.model_path), "cuda")
        teacher = copy.deepcopy(pipeline.model.transformer).requires_grad_(False).eval()
        config = MageQADConfig(**report["identity"]["config"])
        self.master = master
        self.runtime = MageHealingRuntime(
            pipeline,
            teacher,
            master,
            [HPSv21Scorer(args.hps_checkpoint), OpenCLIPH14Scorer(args.clip_checkpoint)],
            height=config.height,
            width=config.width,
        )
        self.calibration = RewardCalibration.from_record(json.loads(args.calibration.read_text()))

    @property
    def evaluate(self):
        return self.runtime.evaluate

    def paired(self, batch, directory: Path) -> RewardCalibration:
        """BF16 references are deterministic; reuse a completed episode's on disk."""
        path = directory / "evaluation.json"
        if path.exists():
            bf16 = json.loads(path.read_text())
            if bf16.get("sample_ids") != [s.sample_id for s in batch]:
                raise ValueError("cached BF16 reference belongs to another prompt batch")
        else:
            bf16 = self.runtime.evaluate(
                self.master.initial_clips, batch, directory, candidate_sha256="bf16", arm="bf16"
            )
        if bf16["failures"] or len(bf16["rows"]) != len(batch):
            raise RuntimeError("BF16 paired reference generation failed")
        for row in bf16["rows"]:
            if row["image_sha256"] != row["replay_sha256"] or row["scores"] != row["replay_scores"]:
                raise RuntimeError("BF16 reference replay differs")
        return replace(
            self.calibration,
            sample_baselines={r["sample_id"]: r["scores"] for r in bf16["rows"]},
        )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def gb10(name):
        action = sub.add_parser(name)
        action.add_argument("--campaign", required=True, type=Path)
        action.add_argument("--source-campaign", required=True, type=Path)
        action.add_argument("--model-path", required=True, type=Path)
        action.add_argument("--eval-pool", required=True, type=Path)
        action.add_argument("--calibration", required=True, type=Path)
        action.add_argument("--hps-checkpoint", required=True, type=Path)
        action.add_argument("--clip-checkpoint", required=True, type=Path)
        return action

    init = gb10("init")
    init.add_argument("--protocol", required=True, type=Path)
    for name in ("gradient", "select", "confirm"):
        action = gb10(name)
        action.add_argument("--episode", required=True, type=int)
        action.add_argument("--force", action="store_true", help="reopen an unfinished episode")
        if name == "gradient":
            action.add_argument(
                "--no-register",
                action="store_true",
                help="remote worker mode: write artifacts, leave the ledger to the coordinator",
            )
    gb10("final-gate")
    gb10("human-test")
    gb10("step")
    for name in ("register", "consistency", "select-decide", "confirm-decide", "status"):
        action = sub.add_parser(name)
        action.add_argument("--campaign", required=True, type=Path)
        if name == "register":
            action.add_argument("--episode", required=True, type=int)
            action.add_argument("--eval-pool", required=True, type=Path)
    for name in ("authorize-generation", "declare-final"):
        action = sub.add_parser(name)
        action.add_argument("--campaign", required=True, type=Path)
        action.add_argument("--maintainer", required=True)
    human = sub.add_parser("approve-human")
    human.add_argument("--campaign", required=True, type=Path)
    human.add_argument("--ballots", required=True, type=Path)
    human.add_argument("--maintainer", required=True)
    human.add_argument("--decision", choices=("approve", "reject"), required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    command = args.command

    if command == "status":
        campaign = ESCampaign(args.campaign)
        summary = {
            "status": campaign.state["status"],
            "verdict": campaign.state["verdict"],
            "final_generation_declared": campaign.state["final_generation_declared"],
            "generations": [
                {
                    "index": g["index"],
                    "status": g["status"],
                    "gradient_completed": sorted(int(k) for k in g["gradient"]["episodes"]),
                    "consistency": (g["consistency"] or {}).get("checks"),
                    "populations": (g["consistency"] or {}).get("populations"),
                    "step": g["step"]
                    and [
                        {k: c[k] for k in ("alpha", "clip_step_rms", "valid")}
                        for c in g["step"]["candidates"]
                    ],
                    "selection_completed": sorted(int(k) for k in g["selection"]["episodes"]),
                    "selection_decision": g["selection_decision"]
                    and {
                        "selected_alpha": g["selection_decision"]["selected_alpha"],
                        "summaries": g["selection_decision"]["summaries"],
                    },
                    "confirmation_completed": sorted(int(k) for k in g["confirmation"]["episodes"]),
                    "confirmation_decision": g["confirmation_decision"]
                    and g["confirmation_decision"]["selection_summary"],
                }
                for g in campaign.state["generations"]
            ],
            "healing": campaign.workflow.read(),
        }
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    if command == "init":
        require_gb10()
        report, master = load_master(args.source_campaign)
        identity = runtime_identity(args.model_path)
        for key in IDENTITY_KEYS:
            if identity[key] != report["identity"][key]:
                raise ValueError(f"runtime {key} identity differs from EXP-008 QAD training")
        evaluation_sha256 = report["identity"]["data"]["evaluation_sha256"]
        frozen_samples(args, {"evaluation_sha256": evaluation_sha256})
        protocol = ESCampaignProtocol(**json.loads(args.protocol.read_text()))
        ledger, ledger_sha256 = source_ledger(args.source_campaign)
        campaign = ESCampaign.create(
            args.campaign,
            protocol=protocol,
            master=master,
            identity={
                "evaluation_sha256": evaluation_sha256,
                "code": code_identity(),
                "qad_code": report["identity"]["code"],
                **{key: identity[key] for key in IDENTITY_KEYS},
                "model_files": identity["model_files"],
                "scorers": scoring_identity(args),
                "source_campaign": str(args.source_campaign.resolve()),
                "source_campaign_sha256": file_hash(args.source_campaign / "campaign.json"),
                "protocol_file_sha256": file_hash(args.protocol),
            },
            source_ledger=ledger,
            source_ledger_sha256=ledger_sha256,
        )
        print(
            json.dumps(
                {
                    "status": campaign.state["status"],
                    "identity": {
                        k: v for k, v in campaign.state["identity"].items() if k != "model_files"
                    },
                },
                indent=2,
            )
        )
        return

    campaign = ESCampaign(args.campaign)
    # Every ledger mutation, GPU or not, must come from the frozen code.
    if code_identity() != campaign.state["identity"]["code"]:
        raise ValueError("code identity differs from the frozen EXP-009 campaign")

    if command == "register":
        samples = load_frozen_eval_samples(args.eval_pool)
        expected = [s.sample_id for s in episode_batch(campaign, samples, args.episode)]
        record = campaign.register_episode(args.episode, expected_sample_ids=expected)
        print(
            json.dumps(
                {
                    "episode": args.episode,
                    "gradient_norm": record["gradient_norm"],
                    "consistency": campaign.generation()["consistency"],
                },
                indent=2,
            )
        )
        return
    if command == "consistency":
        print(json.dumps(campaign.refresh_consistency(), indent=2))
        return
    if command == "select-decide":
        print(json.dumps(campaign.decide_selection(), indent=2))
        return
    if command == "confirm-decide":
        print(json.dumps(campaign.decide_confirmation(), indent=2))
        return
    if command == "authorize-generation":
        print(json.dumps(campaign.authorize_generation(args.maintainer), indent=2))
        return
    if command == "declare-final":
        print(json.dumps(campaign.declare_final_generation(args.maintainer), indent=2))
        return
    if command == "approve-human":
        result = campaign.workflow.approve_human(
            "human_test",
            json.loads(args.ballots.read_text()),
            maintainer=args.maintainer,
            approved=args.decision == "approve",
        )
        print(json.dumps(result, indent=2))
        return

    # Everything below touches the model: GB10 only, identity re-verified.
    report, master = load_master(args.source_campaign)
    if command == "step":
        # Stepping is CPU arithmetic on the ledger's gradients; it still needs the
        # master for the start center and identity binding, not the GPU.
        print(json.dumps(campaign.step(master), indent=2))
        return
    require_gb10()
    verify_runtime(campaign, args, report)
    samples = frozen_samples(args, campaign.state["identity"])
    runtime = Runtime(args, report, master)

    if command in {"gradient", "select", "confirm"}:
        batch = episode_batch(campaign, samples, args.episode)
        calibration = runtime.paired(batch, args.campaign / "bf16" / f"episode-{args.episode:02d}")
        if command == "gradient":
            result = campaign.run_gradient_episode(
                master,
                args.episode,
                batch,
                calibration,
                runtime.evaluate,
                register=not args.no_register,
                force=args.force,
            )
        elif command == "select":
            result = campaign.run_selection_episode(
                master, args.episode, batch, calibration, runtime.evaluate, force=args.force
            )
        else:
            result = campaign.run_confirmation_episode(
                master, args.episode, batch, calibration, runtime.evaluate, force=args.force
            )
        print(json.dumps({k: v for k, v in result.items() if k != "candidates"}, indent=2))
        return

    # Final holdouts: sealed until the maintainer declares the last generation.
    if not campaign.state["final_generation_declared"]:
        raise RuntimeError("final gates require an explicit final-generation declaration")
    confirmation = campaign.workflow.read()["es"].get("confirmation", {})
    if not confirmation.get("passed"):
        raise RuntimeError("no confirmed ES center")
    artifact = torch.load(confirmation["clips_path"], map_location="cpu", weights_only=True)
    if (
        content_hash(artifact) != confirmation["clips_sha256"]
        or artifact["selected_sha256"] != master.selected_sha256
        or artifact["candidate_sha256"] != confirmation["candidate_sha256"]
    ):
        raise ValueError("confirmed clips/master identity mismatch")
    clips, sha = artifact["clips"], confirmation["candidate_sha256"]
    split = "auto_test" if command == "final-gate" else "human_test"
    batch = [s for s in samples if s.split == split]
    if len(batch) != EVAL_SPLIT_COUNTS[split]:
        raise ValueError("incorrect frozen split size")
    directory = args.campaign / "healing" / split
    ids = [s.sample_id for s in batch]

    def evaluate():
        if split == "human_test":
            arms = {}
            for arm in ("bf16", "rtn", "quasar-grid", "qad"):
                result = runtime.evaluate(
                    clips, batch, directory / arm, candidate_sha256=sha, arm=arm
                )
                if result["failures"] or len(result["rows"]) != 30:
                    raise RuntimeError("blind image generation failed")
                arms[arm] = {row["sample_id"]: Path(row["image_path"]) for row in result["rows"]}
            return build_blind_packet(
                directory / "blind",
                arms,
                candidate_sha256=sha,
                prompts={s.sample_id: s.prompt for s in batch},
            )
        paired = runtime.paired(batch, directory / "bf16")
        # The deployment comparison is against the raw EXP-008 payload, so the
        # latent-box projection is part of the candidate's total change here.
        healed = runtime.evaluate(
            master.initial_clips,
            batch,
            directory / "healed",
            candidate_sha256=master.selected_sha256,
        )
        healed_gate = quality_gate(healed, paired, ids, candidate_sha256=master.selected_sha256)
        atomic_json(directory / "healed-gate.json", healed_gate)
        if not healed_gate["passed"]:
            return dict(healed_gate, candidate_sha256=sha)
        healed_teacher = sum(r["teacher_loss"] for r in healed["rows"]) / len(healed["rows"])
        evaluation = runtime.evaluate(clips, batch, directory / "candidate", candidate_sha256=sha)
        return quality_gate(
            evaluation,
            paired,
            ids,
            candidate_sha256=sha,
            final=True,
            trajectory_tolerance=(healed_teacher, campaign.protocol.center_teacher_tolerance),
            comparison_scores={r["sample_id"]: r["scores"] for r in healed["rows"]},
        )

    result = campaign.workflow.once(split, sha, evaluate)
    print(json.dumps(result, indent=2))
    if not result.get("passed") and result.get("status") != "awaiting_human":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
