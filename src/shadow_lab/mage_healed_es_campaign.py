"""EXP-009: consistency-gated multi-batch antithetic ES on a healed master.

EXP-008 proved the healed master; its ES stage died on gates that no antithetic
candidate can pass and on a population two orders of magnitude below the signal
regime EXP-007 measured. This module runs the EXP-007 batch structure (N per
batch, disjoint prompt episodes, common antithetic noise) against the healed
master, gates exploration candidates on catastrophe floors only, and moves every
non-regression decision to the updated center with tolerance bands.
"""

from __future__ import annotations

import itertools
import json
import math
import shutil
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from .es import (
    CandidateObservation,
    ESConfig,
    ESState,
    candidate_clips,
    candidate_specs,
    clips_to_latent,
    estimate_es_gradient,
    initialization_projection_report,
    latent_to_clips,
    replay_noise,
    shape_rewards,
)
from .image_eval import PRIMARY_REWARD_AXES, RewardCalibration, aggregate_primary_reward
from .mage_healed_master import HealedMaster
from .mage_healing import HealingWorkflow, quality_gate
from .mage_qad_campaign import atomic_json
from .mage_qad_checkpoint import atomic_save, content_hash
from .nvfp4_runtime import native_kernel_evidence


def alpha_slug(alpha: float) -> str:
    return f"{alpha:.6f}".replace(".", "p").replace("-", "m")


@dataclass(frozen=True)
class ESCampaignProtocol:
    """Frozen EXP-009 search protocol; every field is part of the campaign identity."""

    selected_sha256: str
    initial_clips_sha256: str
    dimension: int
    sigma: float = 0.003
    alphas: tuple[float, ...] = (3e-05, 1e-04)
    population_size: int = 80
    prompt_batch_size: int = 8
    schedule_seed: int = 20260904
    es_seed: int = 20260908
    gradient_episodes: tuple[int, ...] = tuple(range(24))
    selection_episodes: tuple[int, ...] = tuple(range(24, 32))
    confirmation_episodes: tuple[int, ...] = tuple(range(32, 40))
    shaping: str = "zscore"
    clip_step_rms_max: float = 0.02
    teacher_tolerance: float = 1.05
    bf16_floor: float = 0.8
    max_failed_candidates: int = 16
    consistency_checkpoints: tuple[tuple[int, float, float], ...] = (
        (8, 0.10, 0.70),
        (16, 0.10, 0.70),
    )
    subset_populations: tuple[int, ...] = (40, 20)
    min_non_degrading_batches: int = 6
    max_generations: int = 3
    replay_audit_index: int = 0

    def __post_init__(self) -> None:
        # JSON round-trips turn tuples into lists; normalize so the frozen record
        # hashes identically whether it was built in Python or read back from disk.
        for name in ("alphas", "gradient_episodes", "selection_episodes", "confirmation_episodes"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        object.__setattr__(
            self,
            "subset_populations",
            tuple(int(value) for value in self.subset_populations),
        )
        object.__setattr__(
            self,
            "consistency_checkpoints",
            tuple((int(k), float(c), float(p)) for k, c, p in self.consistency_checkpoints),
        )
        for name in ("selected_sha256", "initial_clips_sha256"):
            value = getattr(self, name)
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(f"{name} must be a sha256 hex digest")
        if self.dimension < 1:
            raise ValueError("genome dimension must be positive")
        # sigma must come from a measurement: the unmeasured ESConfig default 0.03
        # burned the EXP-008 ES stage and is far outside the calibrated region.
        if not (0 < self.sigma <= 0.01):
            raise ValueError("sigma must be a calibrated value in (0, 0.01]; 0.03 is banned")
        if not self.alphas or not all(math.isfinite(a) and a > 0 for a in self.alphas):
            raise ValueError("alphas must be positive finite and non-empty")
        if list(self.alphas) != sorted(set(self.alphas)):
            raise ValueError("alphas must be strictly increasing")
        if self.population_size < 2 or self.population_size % 2:
            raise ValueError("antithetic population size must be even and at least 2")
        if self.prompt_batch_size < 1:
            raise ValueError("prompt batch size must be positive")
        for name in ("gradient_episodes", "selection_episodes", "confirmation_episodes"):
            episodes = getattr(self, name)
            if (
                not episodes
                or len(set(episodes)) != len(episodes)
                or any(e < 0 or not isinstance(e, int) for e in episodes)
            ):
                raise ValueError(f"{name} must be non-empty unique non-negative integers")
        if (
            set(self.gradient_episodes) & set(self.selection_episodes)
            | set(self.gradient_episodes) & set(self.confirmation_episodes)
            | set(self.selection_episodes) & set(self.confirmation_episodes)
        ):
            raise ValueError("gradient/selection/confirmation episodes must be disjoint")
        if not self.consistency_checkpoints or any(
            not (2 <= k <= len(self.gradient_episodes)) or not (-1 < c < 1) or not (0 <= p <= 1)
            for k, c, p in self.consistency_checkpoints
        ):
            raise ValueError("consistency checkpoints need k in range and thresholds in scale")
        if [k for k, _, _ in self.consistency_checkpoints] != sorted(
            {k for k, _, _ in self.consistency_checkpoints}
        ):
            raise ValueError("consistency checkpoint episodes must be strictly increasing")
        if any(m % 2 or m < 2 or m >= self.population_size for m in self.subset_populations):
            raise ValueError("subset populations must be even, at least 2, below the full size")
        if (
            not 1
            <= self.min_non_degrading_batches
            <= min(len(self.selection_episodes), len(self.confirmation_episodes))
        ):
            raise ValueError("non-degrading batch rule does not fit the episode budgets")
        if self.max_generations < 1 or self.teacher_tolerance < 1:
            raise ValueError("max generations must be positive and tolerance at least 1")
        if not 0 < self.bf16_floor <= 1:
            raise ValueError("BF16 floor must be in (0, 1]")
        if not 0 <= self.max_failed_candidates < self.population_size:
            raise ValueError("failed-candidate budget must fit inside the population")
        if not 0 <= self.replay_audit_index < self.population_size:
            raise ValueError("replay audit index must address a candidate")
        if self.clip_step_rms_max <= 0:
            raise ValueError("clip step RMS bound must be positive")
        if self.shaping not in {"zscore", "rank"} or self.schedule_seed < 0 or self.es_seed < 0:
            raise ValueError("invalid shaping or negative seed")

    def bind(self, master: HealedMaster) -> None:
        if (
            master.selected_sha256 != self.selected_sha256
            or content_hash(master.initial_clips) != self.initial_clips_sha256
            or master.layout.dimension != self.dimension
        ):
            raise ValueError("protocol is not bound to this healed master/genome")

    def es_config(self) -> ESConfig:
        return ESConfig(
            dimension=self.dimension,
            population_size=self.population_size,
            sigma=self.sigma,
            # Alpha stepping happens against the pre-registered alphas outside the
            # config; the estimate never consumes the learning rate.
            learning_rate=min(self.alphas),
            seed=self.es_seed,
            shaping=self.shaping,
        )


class CampaignStopped(RuntimeError):
    """A pre-registered stop rule fired; the campaign ledger records the verdict."""


def exploration_gate(
    evaluation: dict,
    calibration: RewardCalibration,
    sample_ids: list[str],
    *,
    candidate_sha256: str,
    bf16_floor: float,
    trajectory_tolerance: tuple[float, float] | None = None,
) -> dict:
    """Catastrophe floors for exploration candidates; replay evidence is not required."""
    reasons: list[str] = []
    try:
        if evaluation["candidate_sha256"] != candidate_sha256:
            raise ValueError("candidate identity mismatch")
        rows = evaluation["rows"]
        if len(rows) != len(sample_ids) or {r["sample_id"] for r in rows} != set(sample_ids):
            raise ValueError("incomplete or duplicate paired samples")
        if evaluation["failures"] != 0:
            reasons.append("generation failures")
        if (
            evaluation["compute_capability"] != [12, 1]
            or not native_kernel_evidence(evaluation["kernel_names"])["native_compute"]
        ):
            reasons.append("missing GB10 native NVFP4 kernel evidence")
        rewards, losses = [], []
        for row in rows:
            reward = aggregate_primary_reward(
                row["scores"], calibration, sample_id=row["sample_id"]
            )
            if not reward.valid or reward.value is None:
                raise ValueError("non-finite/missing primary score")
            rewards.append(reward.value)
            loss = row["teacher_loss"]
            if not math.isfinite(loss) or loss < 0:
                raise ValueError("invalid teacher trajectory loss")
            losses.append(loss)
        axes = {
            axis: statistics.fmean(row["scores"][axis] for row in rows)
            for axis in PRIMARY_REWARD_AXES
        }
        bf16 = {
            axis: statistics.fmean(calibration.sample_baselines[sid][axis] for sid in sample_ids)
            for axis in PRIMARY_REWARD_AXES
        }
        for axis in PRIMARY_REWARD_AXES:
            if bf16[axis] <= 0 or axes[axis] < bf16_floor * bf16[axis]:
                reasons.append(f"{axis} below {bf16_floor:.0%} BF16 floor")
        teacher_mean = statistics.fmean(losses)
        if trajectory_tolerance is not None:
            baseline, factor = trajectory_tolerance
            if teacher_mean > baseline * factor:
                reasons.append("teacher trajectory batch mean exceeds tolerance")
        return {
            "passed": not reasons,
            "reasons": sorted(set(reasons)),
            "reward": statistics.fmean(rewards),
            "axis_means": axes,
            "bf16_means": bf16,
            "teacher_mean": teacher_mean,
            "candidate_sha256": candidate_sha256,
        }
    except (KeyError, TypeError, ValueError) as error:
        return {"passed": False, "reasons": [str(error)], "candidate_sha256": candidate_sha256}


def replay_matches(evaluation: dict) -> bool:
    if not evaluation.get("weight_sha256") or evaluation.get("weight_sha256") != evaluation.get(
        "replay_weight_sha256"
    ):
        return False
    rows = evaluation.get("rows") or []
    return bool(rows) and all(
        row.get("image_sha256") == row.get("replay_sha256")
        and row.get("scores") == row.get("replay_scores")
        for row in rows
    )


def gradient_pairwise_stats(gradients: list[torch.Tensor]) -> dict:
    if len(gradients) < 2:
        raise ValueError("gradient consistency needs at least two gradients")
    for gradient in gradients:
        if (
            gradient.device.type != "cpu"
            or gradient.dtype != torch.float32
            or gradient.ndim != 1
            or not torch.isfinite(gradient).all()
        ):
            raise ValueError("gradients must be finite CPU float32 vectors")
    signs = [torch.sign(gradient) for gradient in gradients]
    pairwise = []
    for left, right in itertools.combinations(range(len(gradients)), 2):
        denominator = torch.linalg.vector_norm(gradients[left]) * torch.linalg.vector_norm(
            gradients[right]
        )
        if denominator == 0:
            raise ValueError("gradient consistency is undefined for a zero vector")
        pairwise.append(
            {
                "left": left,
                "right": right,
                "cosine": float(torch.dot(gradients[left], gradients[right]) / denominator),
                "sign_agreement_fraction": float(torch.mean((signs[left] == signs[right]).float())),
            }
        )
    cosines = [pair["cosine"] for pair in pairwise]
    return {
        "pair_count": len(pairwise),
        "pairwise": pairwise,
        "mean_cosine": statistics.fmean(cosines),
        "positive_fraction": sum(c > 0 for c in cosines) / len(cosines),
        "mean_sign_agreement": statistics.fmean(
            pair["sign_agreement_fraction"] for pair in pairwise
        ),
    }


def subset_gradient(config: ESConfig, episode_record: dict, pairs: int) -> torch.Tensor:
    """Replay the episode gradient for a hypothetical population of ``2 * pairs``."""
    raw = episode_record["raw_rewards"]
    specs = episode_record["specs"]
    if 2 * pairs > len(raw) or len(raw) != len(specs):
        raise ValueError("subset population does not fit the recorded episode")
    shaped = shape_rewards(list(raw[: 2 * pairs]), config.shaping)
    vector = torch.zeros(config.dimension)
    for utility, spec in zip(shaped, specs[: 2 * pairs], strict=True):
        vector.add_(replay_noise(config, spec["noise_seed"]), alpha=utility * spec["sign"])
    vector.div_(2 * pairs * config.sigma)
    return vector


def _new_generation(
    index: int,
    start_clips_sha256: str,
    start_clips_path: str | None,
    start_artifact_sha256: str | None,
) -> dict:
    return {
        "index": index,
        "start_clips_sha256": start_clips_sha256,
        "start_clips_path": start_clips_path,
        "start_artifact_sha256": start_artifact_sha256,
        "status": "gradient",
        "gradient": {"episodes": {}},
        "consistency": None,
        "step": None,
        "selection": {"episodes": {}},
        "selection_decision": None,
        "confirmation": {"episodes": {}},
        "confirmation_decision": None,
    }


class ESCampaign:
    """The EXP-009 ledger; claims precede evaluation so crashes cannot reopen episodes."""

    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.path = self.directory / "campaign.json"
        self.state = json.loads(self.path.read_text())
        identity = self.state["identity"]
        self.workflow = HealingWorkflow(
            self.directory / "healing",
            selected_sha256=identity["selected_sha256"],
            evaluation_sha256=identity["evaluation_sha256"],
        )
        permit = self.workflow.es_permit()
        if permit != identity["es_permit"]:
            raise ValueError("inherited human_dev permit differs from the frozen campaign one")
        self.protocol = ESCampaignProtocol(**self.state["protocol"])

    @classmethod
    def create(
        cls,
        directory: Path,
        *,
        protocol: ESCampaignProtocol,
        master: HealedMaster,
        identity: dict,
        source_ledger: dict,
        source_ledger_sha256: str,
    ) -> ESCampaign:
        protocol.bind(master)
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        workflow = HealingWorkflow.inherit(
            directory / "healing",
            source_ledger,
            selected_sha256=master.selected_sha256,
            evaluation_sha256=identity["evaluation_sha256"],
            source_sha256=source_ledger_sha256,
        )
        state = {
            "schema_version": 1,
            "identity": {
                **identity,
                "selected_sha256": master.selected_sha256,
                "es_permit": workflow.es_permit(),
            },
            "protocol": asdict(protocol),
            "status": "gradient",
            "verdict": None,
            "final_generation_declared": False,
            "continuations": [],
            "generations": [
                {
                    **_new_generation(0, content_hash(master.initial_clips), None, None),
                    # The latent box moves saturated clips (1.0 -> 0.999); the search
                    # compares against this effective center so that projection is
                    # never booked as an ES step (EXP-007 convention).
                    "start_projection": initialization_projection_report(
                        master.initial_clips, protocol.es_config()
                    ),
                }
            ],
        }
        atomic_json(directory / "campaign.json", state)
        return cls(directory)

    # -- ledger helpers ------------------------------------------------

    def _write(self) -> None:
        atomic_json(self.path, self.state)

    def generation(self) -> dict:
        return self.state["generations"][-1]

    def _generation_dir(self) -> Path:
        return self.directory / f"generation-{self.generation()['index']:02d}"

    def _require_status(self, status: str) -> None:
        if self.state["status"] != status:
            raise RuntimeError(f"campaign status is {self.state['status']!r}, not {status!r}")

    def _stop(self, verdict: str, detail) -> None:
        self.generation()["status"] = "stopped"
        self.state.update(status="stopped", verdict=verdict, stop_detail=detail)
        self._write()
        raise CampaignStopped(f"EXP-009 stopped: {verdict}: {detail}")

    def _claim(self, directory: Path, *, force: bool = False) -> None:
        """Exclusive claim per episode; ``force`` reopens only an unfinished episode."""
        claim = directory.with_name(f"{directory.name}.claimed")
        claim.parent.mkdir(parents=True, exist_ok=True)
        if force and claim.exists() and not (directory / "episode.json").exists():
            if directory.exists():
                shutil.rmtree(directory)
            claim.unlink()
        with claim.open("x") as stream:
            stream.write(content_hash(self.state["protocol"]))

    def start_clips(self, master: HealedMaster) -> torch.Tensor:
        self.protocol.bind(master)
        generation = self.generation()
        if generation["index"] == 0:
            clips = master.initial_clips
        else:
            artifact = torch.load(
                generation["start_clips_path"], map_location="cpu", weights_only=True
            )
            if (
                content_hash(artifact) != generation["start_artifact_sha256"]
                or artifact["selected_sha256"] != master.selected_sha256
            ):
                raise ValueError("generation start artifact identity mismatch")
            clips = artifact["clips"]
        if content_hash(clips) != generation["start_clips_sha256"]:
            raise ValueError("generation start clips identity mismatch")
        return clips

    def effective_start_clips(self, master: HealedMaster) -> torch.Tensor:
        """The start center as the search sees it: raw clips through the latent box."""
        config = self.protocol.es_config()
        return latent_to_clips(clips_to_latent(self.start_clips(master), config), config)

    def es_state(self, master: HealedMaster) -> ESState:
        config = self.protocol.es_config()
        return ESState(
            config=config,
            center=clips_to_latent(self.start_clips(master), config),
            generation=self.generation()["index"],
        )

    # -- gradient phase ------------------------------------------------

    def run_gradient_episode(
        self,
        master: HealedMaster,
        episode: int,
        batch,
        calibration: RewardCalibration,
        evaluate,
        *,
        register: bool = True,
        force: bool = False,
    ) -> dict:
        """Evaluate one prompt episode; ``register=False`` leaves the ledger alone.

        Remote workers run unregistered episodes in their own campaign copy; the
        coordinator syncs the artifacts back and registers them centrally.
        """
        self._require_status("gradient")
        protocol = self.protocol
        if episode not in protocol.gradient_episodes:
            raise ValueError("episode outside the frozen gradient range")
        generation = self.generation()
        directory = self._generation_dir() / f"gradient-episode-{episode:02d}"
        self._claim(directory, force=force)
        ids = [sample.sample_id for sample in batch]
        state = self.es_state(master)
        center_clips = latent_to_clips(state.center, state.config)
        center_sha = content_hash({"master": master.selected_sha256, "clips": center_clips})
        center_eval = evaluate(
            center_clips, batch, directory / "center", candidate_sha256=center_sha, replay=False
        )
        center_gate = exploration_gate(
            center_eval,
            calibration,
            ids,
            candidate_sha256=center_sha,
            bf16_floor=protocol.bf16_floor,
        )
        atomic_json(directory / "center-gate.json", center_gate)
        if not center_gate["passed"]:
            self._stop("episode-invalid", f"center failed floors: {center_gate['reasons']}")
        observations, candidates = [], []
        audit_exact = None
        for spec in candidate_specs(state):
            clips = candidate_clips(state, spec)
            sha = content_hash({"master": master.selected_sha256, "clips": clips})
            audit = spec.population_index == protocol.replay_audit_index
            evaluation = evaluate(
                clips,
                batch,
                directory / f"candidate-{spec.population_index:02d}",
                candidate_sha256=sha,
                replay=audit,
            )
            gate = exploration_gate(
                evaluation,
                calibration,
                ids,
                candidate_sha256=sha,
                bf16_floor=protocol.bf16_floor,
                trajectory_tolerance=(center_gate["teacher_mean"], protocol.teacher_tolerance),
            )
            if audit:
                audit_exact = replay_matches(evaluation)
                atomic_json(
                    directory / "replay-audit.json",
                    {"candidate": spec.population_index, "exact": audit_exact},
                )
                if not audit_exact:
                    self._stop(
                        "episode-invalid",
                        f"replay audit mismatch on candidate {spec.population_index}",
                    )
            candidates.append(
                {
                    "population_index": spec.population_index,
                    "pair_index": spec.pair_index,
                    "sign": spec.sign,
                    "noise_seed": spec.noise_seed,
                    "candidate_sha256": sha,
                    "gate": gate,
                }
            )
            observations.append(
                CandidateObservation(
                    spec,
                    gate["reward"] if gate["passed"] else None,
                    None if gate["passed"] else "; ".join(gate["reasons"]),
                )
            )
        failed = sum(1 for item in observations if item.reward is None)
        if failed > protocol.max_failed_candidates:
            self._stop(
                "episode-invalid",
                f"{failed} of {len(observations)} candidates failed catastrophe floors",
            )
        estimate = estimate_es_gradient(state, observations)
        payload = {
            "gradient": estimate.vector,
            "master": master.selected_sha256,
            "generation": generation["index"],
            "episode": episode,
        }
        gradient_sha256 = content_hash(payload)
        atomic_save(payload, directory / "gradient.pt")
        record = {
            "episode": episode,
            "generation": generation["index"],
            "sample_ids": ids,
            "population_size": protocol.population_size,
            "sigma": protocol.sigma,
            "center": {"candidate_sha256": center_sha, "gate": center_gate},
            "specs": [
                {
                    key: candidate[key]
                    for key in ("population_index", "pair_index", "sign", "noise_seed")
                }
                for candidate in candidates
            ],
            "candidates": candidates,
            "raw_rewards": list(estimate.raw_rewards),
            "shaped_rewards": list(estimate.shaped_rewards),
            "failure_count": estimate.failure_count,
            "reward_mean": estimate.reward_mean,
            "reward_std": estimate.reward_std,
            "gradient_norm": estimate.norm,
            "gradient_rms": estimate.rms,
            "gradient_sha256": gradient_sha256,
            "replay_audit": {"candidate": protocol.replay_audit_index, "exact": audit_exact},
            "status": "completed",
        }
        atomic_json(directory / "episode.json", record)
        if register:
            self.register_episode(episode)
        return record

    def register_episode(
        self, episode: int, *, expected_sample_ids: list[str] | None = None
    ) -> dict:
        """Adopt a completed remote episode into this ledger after hash checks."""
        self._require_status("gradient")
        generation = self.generation()
        episodes = generation["gradient"]["episodes"]
        if str(episode) in episodes:
            raise FileExistsError(f"gradient episode {episode} is already registered")
        directory = self._generation_dir() / f"gradient-episode-{episode:02d}"
        record = json.loads((directory / "episode.json").read_text())
        if (
            record.get("status") != "completed"
            or record.get("generation") != generation["index"]
            or record.get("episode") != episode
            or not record.get("replay_audit", {}).get("exact")
            or record.get("failure_count", 0) > self.protocol.max_failed_candidates
        ):
            raise ValueError("episode record is not a completed audited run of this generation")
        if expected_sample_ids is not None and record.get("sample_ids") != list(
            expected_sample_ids
        ):
            raise ValueError("episode prompts differ from the frozen schedule")
        payload = torch.load(directory / "gradient.pt", map_location="cpu", weights_only=True)
        if content_hash(payload) != record["gradient_sha256"]:
            raise ValueError("gradient artifact identity mismatch")
        episodes[str(episode)] = record
        generation["gradient"]["completed"] = sorted(int(key) for key in episodes)
        self._write()
        self.refresh_consistency()
        return record

    def refresh_consistency(self) -> dict:
        """Recompute batch-gradient consistency and enforce the pre-registered stops."""
        self._require_status("gradient")
        protocol = self.protocol
        generation = self.generation()
        episodes = generation["gradient"]["episodes"]
        completed = sorted(int(key) for key in episodes)
        config = protocol.es_config()
        gradients = {}
        for episode in completed:
            directory = self._generation_dir() / f"gradient-episode-{episode:02d}"
            payload = torch.load(directory / "gradient.pt", map_location="cpu", weights_only=True)
            if content_hash(payload) != episodes[str(episode)]["gradient_sha256"]:
                raise ValueError(f"gradient artifact identity mismatch for episode {episode}")
            gradients[episode] = payload["gradient"]
        populations: dict[str, dict] = {}
        if len(completed) >= 2:
            stats = gradient_pairwise_stats([gradients[e] for e in completed])
            stats["pairwise"] = [
                {**pair, "left": completed[pair["left"]], "right": completed[pair["right"]]}
                for pair in stats["pairwise"]
            ]
            populations[str(protocol.population_size)] = stats
            for size in protocol.subset_populations:
                subset = gradient_pairwise_stats(
                    [subset_gradient(config, episodes[str(e)], size // 2) for e in completed]
                )
                subset["pairwise"] = [
                    {**pair, "left": completed[pair["left"]], "right": completed[pair["right"]]}
                    for pair in subset["pairwise"]
                ]
                populations[str(size)] = subset
        checks = list((generation["consistency"] or {}).get("checks", []))
        recorded = {check["checkpoint"] for check in checks}
        full = populations.get(str(protocol.population_size))
        for k, min_cosine, min_positive in protocol.consistency_checkpoints:
            if len(completed) < k or k in recorded or full is None:
                continue
            checks.append(
                {
                    "checkpoint": k,
                    "episodes_used": len(completed),
                    "population": protocol.population_size,
                    "mean_cosine": full["mean_cosine"],
                    "positive_fraction": full["positive_fraction"],
                    "min_mean_cosine": min_cosine,
                    "min_positive_fraction": min_positive,
                    "passed": full["mean_cosine"] >= min_cosine
                    and full["positive_fraction"] >= min_positive,
                }
            )
        summary = {
            "generation": generation["index"],
            "episodes": completed,
            "populations": {
                name: {key: value for key, value in stats.items() if key != "pairwise"}
                for name, stats in populations.items()
            },
            "checks": checks,
        }
        generation["consistency"] = summary
        atomic_json(
            self._generation_dir() / "consistency.json", {**summary, "populations": populations}
        )
        self._write()
        failed = next((check for check in checks if not check["passed"]), None)
        if failed:
            self._stop("no-gradient-signal", failed)
        return summary

    # -- step, selection, confirmation ---------------------------------

    def step(self, master: HealedMaster) -> dict:
        self._require_status("gradient")
        protocol = self.protocol
        generation = self.generation()
        completed = sorted(int(key) for key in generation["gradient"]["episodes"])
        if completed != sorted(protocol.gradient_episodes):
            raise RuntimeError("every frozen gradient episode must complete before stepping")
        config = protocol.es_config()
        start = self.effective_start_clips(master)
        mean = torch.zeros(config.dimension)
        for episode in completed:
            payload = torch.load(
                self._generation_dir() / f"gradient-episode-{episode:02d}" / "gradient.pt",
                map_location="cpu",
                weights_only=True,
            )
            mean += payload["gradient"]
        mean /= len(completed)
        mean_payload = {
            "mean_gradient": mean,
            "master": master.selected_sha256,
            "generation": generation["index"],
        }
        atomic_save(mean_payload, self._generation_dir() / "mean-gradient.pt")
        candidates = []
        for alpha in protocol.alphas:
            clips = latent_to_clips(clips_to_latent(start, config) + alpha * mean, config)
            step_rms = float(torch.sqrt(torch.mean((clips - start).square())))
            sha = content_hash({"master": master.selected_sha256, "clips": clips, "alpha": alpha})
            path = self._generation_dir() / f"alpha-{alpha_slug(alpha)}-clips.pt"
            valid = math.isfinite(step_rms) and step_rms <= protocol.clip_step_rms_max
            artifact = {
                "clips": clips,
                "selected_sha256": master.selected_sha256,
                "candidate_sha256": sha,
                "alpha": alpha,
                "generation": generation["index"],
            }
            if valid:
                atomic_save(artifact, path)
            candidates.append(
                {
                    "alpha": alpha,
                    "clip_step_rms": step_rms,
                    "valid": valid,
                    "candidate_sha256": sha,
                    "artifact_sha256": content_hash(artifact),
                    "clips_path": str(path.resolve()) if valid else None,
                }
            )
        if not any(candidate["valid"] for candidate in candidates):
            self._stop(
                "step-rejected",
                f"every alpha exceeds clip_step_rms_max={protocol.clip_step_rms_max}",
            )
        record = {
            "mean_gradient_sha256": content_hash(mean_payload),
            "candidates": candidates,
            "status": "stepped",
        }
        generation["step"] = record
        generation["status"] = "selection"
        self.state["status"] = "selection"
        self._write()
        return record

    def _run_centered_episode(
        self, master, episode, batch, calibration, evaluate, *, phase: str, force: bool = False
    ) -> dict:
        self._require_status(phase)
        protocol = self.protocol
        episodes_for_phase = getattr(protocol, f"{phase}_episodes")
        if episode not in episodes_for_phase:
            raise ValueError(f"episode outside the frozen {phase} range")
        generation = self.generation()
        episodes = generation[phase]["episodes"]
        if str(episode) in episodes:
            raise FileExistsError(f"{phase} episode {episode} is already recorded")
        directory = self._generation_dir() / f"{phase}-episode-{episode:02d}"
        self._claim(directory, force=force)
        ids = [sample.sample_id for sample in batch]
        center_clips = self.effective_start_clips(master)
        center_sha = content_hash({"master": master.selected_sha256, "clips": center_clips})
        center_eval = evaluate(
            center_clips, batch, directory / "center", candidate_sha256=center_sha
        )
        center_gate = quality_gate(center_eval, calibration, ids, candidate_sha256=center_sha)
        atomic_json(directory / "center-gate.json", center_gate)
        if not center_gate["passed"]:
            self._stop(
                "episode-invalid", f"{phase} center failed full gate: {center_gate['reasons']}"
            )
        center_scores = {row["sample_id"]: row["scores"] for row in center_eval["rows"]}
        center_teacher = statistics.fmean(row["teacher_loss"] for row in center_eval["rows"])
        if phase == "selection":
            alpha_records = [c for c in generation["step"]["candidates"] if c["valid"]]
        else:
            selected = generation["selection_decision"]["selected_alpha"]
            alpha_records = [c for c in generation["step"]["candidates"] if c["alpha"] == selected]
        candidates = {}
        for record in alpha_records:
            artifact = torch.load(record["clips_path"], map_location="cpu", weights_only=True)
            if (
                content_hash(artifact) != record["artifact_sha256"]
                or artifact["candidate_sha256"] != record["candidate_sha256"]
            ):
                raise ValueError("alpha center artifact identity mismatch")
            sha = artifact["candidate_sha256"]
            evaluation = evaluate(
                artifact["clips"],
                batch,
                directory / f"alpha-{alpha_slug(record['alpha'])}",
                candidate_sha256=sha,
            )
            gate = quality_gate(
                evaluation,
                calibration,
                ids,
                candidate_sha256=sha,
                trajectory_tolerance=(center_teacher, protocol.teacher_tolerance),
                comparison_scores=center_scores,
            )
            atomic_json(directory / f"alpha-{alpha_slug(record['alpha'])}-gate.json", gate)
            candidates[str(record["alpha"])] = {
                "alpha": record["alpha"],
                "candidate_sha256": sha,
                "gate_passed": gate["passed"],
                "reasons": gate.get("reasons", []),
                "reward_delta": gate.get("reward_delta"),
                "non_regressions": gate.get("non_regressions"),
                "axis_means": gate.get("axis_means"),
            }
        episode_record = {
            "episode": episode,
            "generation": generation["index"],
            "sample_ids": ids,
            "center": {"candidate_sha256": center_sha, "gate": center_gate},
            "candidates": candidates,
            "status": "completed",
        }
        atomic_json(directory / "episode.json", episode_record)
        episodes[str(episode)] = episode_record
        self._write()
        return episode_record

    def run_selection_episode(
        self, master, episode, batch, calibration, evaluate, *, force: bool = False
    ) -> dict:
        return self._run_centered_episode(
            master, episode, batch, calibration, evaluate, phase="selection", force=force
        )

    def run_confirmation_episode(
        self, master, episode, batch, calibration, evaluate, *, force: bool = False
    ) -> dict:
        return self._run_centered_episode(
            master, episode, batch, calibration, evaluate, phase="confirmation", force=force
        )

    def _decide(self, phase: str) -> list[dict]:
        self._require_status(phase)
        protocol = self.protocol
        generation = self.generation()
        episodes = generation[phase]["episodes"]
        if sorted(int(key) for key in episodes) != sorted(getattr(protocol, f"{phase}_episodes")):
            raise RuntimeError(f"every frozen {phase} episode must complete first")
        if phase == "selection":
            alpha_records = [c for c in generation["step"]["candidates"] if c["valid"]]
        else:
            selected = generation["selection_decision"]["selected_alpha"]
            alpha_records = [c for c in generation["step"]["candidates"] if c["alpha"] == selected]
        summaries = []
        for record in alpha_records:
            key = str(record["alpha"])
            deltas = [ep["candidates"][key]["reward_delta"] for ep in episodes.values()]
            gates_ok = all(ep["candidates"][key]["gate_passed"] for ep in episodes.values())
            non_degrading = sum(delta >= 0 for delta in deltas)
            summaries.append(
                {
                    "alpha": record["alpha"],
                    "candidate_sha256": record["candidate_sha256"],
                    "clips_path": record["clips_path"],
                    "episodes": len(deltas),
                    "mean_reward_delta": statistics.fmean(deltas),
                    "non_degrading_batches": non_degrading,
                    "all_batch_gates_passed": gates_ok,
                    "passes": gates_ok
                    and statistics.fmean(deltas) > 0
                    and non_degrading >= protocol.min_non_degrading_batches,
                }
            )
        return summaries

    def decide_selection(self) -> dict:
        summaries = self._decide("selection")
        passing = [summary for summary in summaries if summary["passes"]]
        if not passing:
            self._stop("selection-failed", summaries)
        selected = max(passing, key=lambda summary: summary["alpha"])
        decision = {
            "summaries": summaries,
            "selected_alpha": selected["alpha"],
            "candidate_sha256": selected["candidate_sha256"],
            "clips_path": selected["clips_path"],
            "status": "selected",
        }
        generation = self.generation()
        generation["selection_decision"] = decision
        generation["status"] = "confirmation"
        self.state["status"] = "confirmation"
        self._write()
        return decision

    def decide_confirmation(self) -> dict:
        summaries = self._decide("confirmation")
        summary = summaries[0]
        if not summary["passes"]:
            self._stop("confirmation-failed", summary)
        generation = self.generation()
        artifact = torch.load(summary["clips_path"], map_location="cpu", weights_only=True)
        clips_sha256 = content_hash(artifact)
        report = {
            "passed": True,
            "status": "passed",
            "phase": "confirmation",
            "generation": generation["index"],
            "alpha": summary["alpha"],
            "candidate_sha256": summary["candidate_sha256"],
            "clips_path": summary["clips_path"],
            "clips_sha256": clips_sha256,
            "selection_summary": summary,
            "population_size": self.protocol.population_size,
            "expanded_budget_authorized": False,
        }
        self.workflow.record_es("confirmation", report)
        generation["confirmation_decision"] = report
        generation["status"] = "confirmed"
        self.state["status"] = "awaiting_maintainer"
        self._write()
        return report

    # -- continuation and final declaration ----------------------------

    def authorize_generation(self, maintainer: str) -> dict:
        self._require_status("awaiting_maintainer")
        if self.state["final_generation_declared"]:
            raise RuntimeError("final generation already declared")
        if len(self.state["generations"]) >= self.protocol.max_generations:
            raise RuntimeError("max generations reached")
        confirmed = self.generation()["confirmation_decision"]
        artifact = torch.load(confirmed["clips_path"], map_location="cpu", weights_only=True)
        if content_hash(artifact) != confirmed["clips_sha256"]:
            raise ValueError("confirmed center artifact identity mismatch")
        generation = _new_generation(
            len(self.state["generations"]),
            content_hash(artifact["clips"]),
            confirmed["clips_path"],
            confirmed["clips_sha256"],
        )
        self.state["generations"].append(generation)
        self.state["continuations"].append(
            {
                "generation": generation["index"],
                "maintainer": maintainer,
                "confirmed_candidate_sha256": confirmed["candidate_sha256"],
            }
        )
        self.state["status"] = "gradient"
        self._write()
        return generation

    def declare_final_generation(self, maintainer: str) -> dict:
        self._require_status("awaiting_maintainer")
        generation = self.generation()
        if generation.get("confirmation_decision") is None:
            raise RuntimeError("no confirmed center to declare final")
        self.state.update(
            final_generation_declared=True,
            final_declared_by=maintainer,
            final_generation=generation["index"],
            status="final",
        )
        self._write()
        return {
            "final_generation": generation["index"],
            "candidate_sha256": generation["confirmation_decision"]["candidate_sha256"],
        }
