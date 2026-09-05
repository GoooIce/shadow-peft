"""Non-compensating image gates and durable one-shot split access for EXP-008."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path

from .image_eval import PRIMARY_REWARD_AXES, RewardCalibration, aggregate_primary_reward
from .mage_qad_campaign import atomic_json
from .mage_qad_checkpoint import content_hash
from .nvfp4_runtime import native_kernel_evidence


def quality_gate(
    evaluation: dict,
    calibration: RewardCalibration,
    sample_ids: list[str],
    *,
    candidate_sha256: str,
    final: bool = False,
    trajectory_limits: dict[str, float] | None = None,
    axis_floors: dict[str, float] | None = None,
    comparison_scores: dict[str, dict[str, float]] | None = None,
) -> dict:
    """Evaluate paired Reward plus independent axes; missing evidence fails closed."""
    reasons = []
    try:
        if calibration.weights != {axis: 0.5 for axis in PRIMARY_REWARD_AXES}:
            raise ValueError("requires equal-weight paired HPS-v2.1 / CLIP H/14 calibration")
        if not sample_ids or len(set(sample_ids)) != len(sample_ids):
            raise ValueError("expected sample IDs must be nonempty and unique")
        rows = evaluation["rows"]
        if len(rows) != len(sample_ids) or {r["sample_id"] for r in rows} != set(sample_ids):
            raise ValueError("incomplete or duplicate paired samples")
        if evaluation["candidate_sha256"] != candidate_sha256:
            raise ValueError("candidate identity mismatch")
        if evaluation["failures"] != 0:
            reasons.append("generation failures")
        if (
            evaluation["compute_capability"] != [12, 1]
            or not native_kernel_evidence(evaluation["kernel_names"])["native_compute"]
        ):
            reasons.append("missing GB10 native NVFP4 kernel evidence")
        if (
            not evaluation["weight_sha256"]
            or evaluation["weight_sha256"] != evaluation["replay_weight_sha256"]
        ):
            reasons.append("packed weight rematerialization differs")
        rewards = []
        deltas = []
        for row in rows:
            sid = row["sample_id"]
            if sid not in calibration.sample_baselines:
                raise ValueError("missing paired BF16 baseline")
            reward = aggregate_primary_reward(row["scores"], calibration, sample_id=sid)
            if not reward.valid or reward.value is None:
                raise ValueError("non-finite/missing primary score")
            rewards.append(reward.value)
            reference_reward = 0.0
            if comparison_scores is not None:
                reference = aggregate_primary_reward(
                    comparison_scores[sid], calibration, sample_id=sid
                )
                if not reference.valid or reference.value is None:
                    raise ValueError("invalid paired QAD comparison")
                reference_reward = reference.value
            deltas.append(reward.value - reference_reward)
            if (
                hashlib.sha256(Path(row["image_path"]).read_bytes()).hexdigest()
                != row["image_sha256"]
                or row["image_sha256"] != row["replay_sha256"]
                or row["scores"] != row["replay_scores"]
            ):
                reasons.append("image persistence or exact replay failed")
            if trajectory_limits is not None:
                loss, limit = row["teacher_loss"], trajectory_limits[sid]
                if (
                    not math.isfinite(loss)
                    or not math.isfinite(limit)
                    or loss < 0
                    or limit < 0
                    or loss > limit
                ):
                    reasons.append("teacher trajectory regression")
        axes = {
            axis: statistics.fmean(row["scores"][axis] for row in rows)
            for axis in PRIMARY_REWARD_AXES
        }
        bf16 = {
            axis: statistics.fmean(calibration.sample_baselines[sid][axis] for sid in sample_ids)
            for axis in PRIMARY_REWARD_AXES
        }
        reference_axes = (
            bf16
            if comparison_scores is None
            else {
                axis: statistics.fmean(comparison_scores[sid][axis] for sid in sample_ids)
                for axis in PRIMARY_REWARD_AXES
            }
        )
        for axis in PRIMARY_REWARD_AXES:
            if bf16[axis] <= 0 or axes[axis] < 0.8 * bf16[axis]:
                reasons.append(f"{axis} below 80% BF16 floor")
            if final and axes[axis] < reference_axes[axis]:
                reasons.append(f"{axis} mean regressed")
            if axis_floors is not None and axes[axis] < axis_floors[axis]:
                reasons.append(f"{axis} batch mean regressed")
        mean_reward = statistics.fmean(rewards)
        if mean_reward < -1:
            reasons.append("paired Reward below -1")
        if final and (
            len(rows) != 60 or statistics.fmean(deltas) <= 0 or sum(r >= 0 for r in deltas) < 45
        ):
            reasons.append("final requires positive mean delta and at least 45/60 non-regressions")
        return {
            "passed": not reasons,
            "reasons": sorted(set(reasons)),
            "reward": mean_reward,
            "axis_means": axes,
            "bf16_means": bf16,
            "reward_delta": statistics.fmean(deltas),
            "non_regressions": sum(r >= 0 for r in deltas),
            "candidate_sha256": candidate_sha256,
            "evaluation_sha256": content_hash(evaluation),
        }
    except (KeyError, TypeError, ValueError, OSError) as error:
        return {
            "passed": False,
            "reasons": [*reasons, str(error)],
            "candidate_sha256": candidate_sha256,
        }


class HealingWorkflow:
    """One campaign ledger. Claims precede callbacks so crashes cannot reopen holdouts.

    This is a local provenance boundary, not a security service: maintainers own
    the files. Explicit approvals record a named human and the exact blind ballot.
    """

    def __init__(self, directory: Path, *, selected_sha256: str, evaluation_sha256: str):
        self.directory = directory
        self.identity = {"selected_sha256": selected_sha256, "evaluation_sha256": evaluation_sha256}
        self.path = directory / "healing.json"
        if self.path.exists():
            if self.read()["identity"] != self.identity:
                raise ValueError("healing campaign identity mismatch")
        else:
            atomic_json(
                self.path,
                {
                    "identity": self.identity,
                    "gates": {},
                    "approvals": {},
                    "es": {},
                    "success": False,
                    "deployable": False,
                },
            )

    def read(self) -> dict:
        return json.loads(self.path.read_text())

    def once(self, split: str, candidate_sha256: str, evaluate) -> dict:
        prerequisites = {
            "checkpoint_val": None,
            "human_dev": "checkpoint_val",
            "auto_test": "es_confirmation",
            "human_test": "auto_test",
        }
        if split not in prerequisites:
            raise ValueError("unknown healing gate")
        state = self.read()
        prior = prerequisites[split]
        if prior == "es_confirmation":
            if not state["es"].get("confirmation", {}).get("passed"):
                raise RuntimeError("N=30 step confirmation must pass before final evaluation")
            if candidate_sha256 != state["es"]["confirmation"]["candidate_sha256"]:
                raise ValueError("final candidate differs from confirmed ES candidate")
        elif prior and not state["gates"].get(prior, {}).get("passed"):
            raise RuntimeError(f"{prior} must pass first")
        if (
            split in {"checkpoint_val", "human_dev"}
            and candidate_sha256 != self.identity["selected_sha256"]
        ):
            raise ValueError("stage gate must use selected QAD checkpoint")
        if (
            split == "human_test"
            and candidate_sha256 != state["gates"]["auto_test"]["candidate_sha256"]
        ):
            raise ValueError("human_test candidate mismatch")
        # Exclusive creation also serializes competing processes at the split boundary.
        claim = self.directory / f"{split}.claimed"
        with claim.open("x") as stream:
            stream.write(candidate_sha256)
        state["gates"][split] = {
            "passed": False,
            "status": "claimed",
            "candidate_sha256": candidate_sha256,
        }
        atomic_json(self.path, state)
        try:
            report = evaluate()
            if report.get("candidate_sha256") != candidate_sha256:
                raise ValueError("gate result candidate mismatch")
            # Blind packets are pending; only explicit ballots may resolve them.
            if split.startswith("human_"):
                report.update(passed=False, status="awaiting_human")
        except Exception as error:
            report = {
                "passed": False,
                "status": "failed",
                "error": str(error),
                "candidate_sha256": candidate_sha256,
            }
        state["gates"][split] = report
        atomic_json(self.path, state)
        return report

    def approve_human(
        self, split: str, ballots: list[dict], *, maintainer: str, approved: bool
    ) -> dict:
        from .mage_healing_blind import judge_ballots

        if split not in {"human_dev", "human_test"}:
            raise ValueError("unknown human gate")
        state = self.read()
        gate = state["gates"].get(split, {})
        if gate.get("status") != "awaiting_human" or split in state["approvals"]:
            raise RuntimeError("human gate is not awaiting its first decision")
        if not maintainer.strip() or type(approved) is not bool:
            raise ValueError("explicit named maintainer decision required")
        key = Path(gate["private_key"])
        if content_hash(json.loads(key.read_text())) != gate["private_sha256"]:
            raise ValueError("blind packet key identity changed")
        result = judge_ballots(key, ballots, final=split == "human_test")
        if result["candidate_sha256"] != gate["candidate_sha256"]:
            raise ValueError("blind candidate identity changed")
        passed = result["passed"] and approved
        state["approvals"][split] = dict(
            result,
            passed=passed,
            maintainer=maintainer,
            approved=approved,
            ballots_sha256=content_hash(ballots),
            candidate_sha256=gate["candidate_sha256"],
        )
        gate.update(passed=passed, status="approved" if passed else "rejected")
        if split == "human_test":
            state.update(success=passed, deployable=passed)
        atomic_json(self.path, state)
        return state["approvals"][split]

    def es_permit(self) -> dict:
        state = self.read()
        if not state["gates"].get("checkpoint_val", {}).get("passed") or not state["approvals"].get(
            "human_dev", {}
        ).get("passed"):
            raise RuntimeError("automatic and explicit human_dev approval required before ES")
        return dict(self.identity, approval_sha256=content_hash(state["approvals"]["human_dev"]))
