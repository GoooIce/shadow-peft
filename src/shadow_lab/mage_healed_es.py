"""Bounded fresh N=8 / N=30 clip searches; no expanded-budget continuation."""

from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path

from .es import (
    CandidateObservation,
    ESConfig,
    candidate_clips,
    candidate_specs,
    initialize_es,
    latent_to_clips,
    update_es,
)
from .mage_healed_master import HealedMaster
from .mage_healing import quality_gate
from .mage_qad_campaign import atomic_json
from .mage_qad_checkpoint import atomic_save, content_hash


def validate_fresh_protocol(protocol: dict, master: HealedMaster, phase: str) -> ESConfig:
    required = {"selected_sha256", "initial_clips_sha256", "phase", "sigma", "alpha", "seed"}
    if set(protocol) != required:
        raise ValueError(
            "fresh ES protocol required; old center/sigma/alpha/budget cannot be imported"
        )
    if protocol["selected_sha256"] != master.selected_sha256 or protocol[
        "initial_clips_sha256"
    ] != content_hash(master.initial_clips):
        raise ValueError("ES protocol is not bound to the healed master and saliency-aware genome")
    if protocol["phase"] != phase or phase not in {"sigma", "confirmation"}:
        raise ValueError("only fresh N=8 sigma and N=30 confirmation are authorized")
    if any(not math.isfinite(protocol[k]) or protocol[k] <= 0 for k in ("sigma", "alpha")):
        raise ValueError("sigma and alpha must be fresh positive finite measurements")
    return ESConfig(
        master.layout.dimension,
        population_size=8 if phase == "sigma" else 30,
        sigma=protocol["sigma"],
        learning_rate=protocol["alpha"],
        seed=protocol["seed"],
    )


def run_healed_es(
    workflow, master: HealedMaster, protocol: dict, samples, calibration, evaluate, out: Path
) -> dict:
    """A single preflight/confirmation generation, checked against the healed baseline.

    evaluate(clips, samples, directory, candidate_sha256=...) must use the GB10
    runtime. It saves every candidate batch and independently replays it.
    """
    permit = workflow.es_permit()
    if permit["selected_sha256"] != master.selected_sha256:
        raise ValueError("ES permit refers to another master")
    phase = protocol["phase"]
    config = validate_fresh_protocol(protocol, master, phase)
    ledger = workflow.read()
    if phase in ledger["es"]:
        raise RuntimeError("ES phase already consumed")
    if not samples or any(s.split != "es_train" for s in samples):
        raise ValueError("ES may read only es_train prompts")
    if len({s.sample_id for s in samples}) != len(samples):
        raise ValueError("ES prompt batch contains duplicates")
    initial = master.initial_clips
    if phase == "confirmation":
        previous = ledger["es"].get("sigma", {})
        if not previous.get("passed"):
            raise RuntimeError("N=8 sigma preflight must pass before N=30")
        if protocol["sigma"] != previous["protocol"]["sigma"]:
            raise ValueError("confirmation must use the newly measured sigma")
        # Confirmation starts at the healed genome too: it confirms the chosen
        # step size on an independent larger population, not an inherited center.
        if set(previous["sample_ids"]) & {s.sample_id for s in samples}:
            raise ValueError("confirmation requires an unseen es_train batch")
    out.mkdir(parents=True, exist_ok=False)
    with (workflow.directory / f"es-{phase}.claimed").open("x") as stream:
        stream.write(content_hash(protocol))
    ledger["es"][phase] = {"passed": False, "status": "running", "protocol": protocol}
    atomic_json(workflow.path, ledger)
    ids = [s.sample_id for s in samples]
    report = {
        "passed": False,
        "phase": phase,
        "protocol": protocol,
        "sample_ids": ids,
        "permit": permit,
        "population_size": config.population_size,
        "expanded_budget_authorized": False,
    }
    try:
        baseline = evaluate(
            initial, samples, out / "healed-baseline", candidate_sha256=master.selected_sha256
        )
        baseline_gate = quality_gate(
            baseline, calibration, ids, candidate_sha256=master.selected_sha256
        )
        if not baseline_gate["passed"]:
            raise RuntimeError(f"healed baseline failed: {baseline_gate['reasons']}")
        limits = {r["sample_id"]: r["teacher_loss"] for r in baseline["rows"]}
        state = initialize_es(initial, config)
        observations = []
        for spec in candidate_specs(state):
            clips = candidate_clips(state, spec)
            sha = content_hash({"master": master.selected_sha256, "clips": clips})
            evaluation = evaluate(
                clips, samples, out / f"candidate-{spec.population_index:02d}", candidate_sha256=sha
            )
            gate = quality_gate(
                evaluation,
                calibration,
                ids,
                candidate_sha256=sha,
                trajectory_limits=limits,
                axis_floors=baseline_gate["axis_means"],
            )
            atomic_json(out / f"candidate-{spec.population_index:02d}" / "gate.json", gate)
            if not gate["passed"]:
                raise RuntimeError(f"candidate failed hard constraints: {gate['reasons']}")
            observations.append(CandidateObservation(spec, gate["reward"]))
        update = update_es(state, observations)
        clips = latent_to_clips(update.state.center, config)
        sha = content_hash({"master": master.selected_sha256, "clips": clips})
        result = evaluate(clips, samples, out / "updated-center", candidate_sha256=sha)
        gate = quality_gate(
            result,
            calibration,
            ids,
            candidate_sha256=sha,
            trajectory_limits=limits,
            axis_floors=baseline_gate["axis_means"],
        )
        if not gate["passed"] or gate["reward"] < baseline_gate["reward"]:
            raise RuntimeError(
                "updated center failed absolute/trajectory/axis non-regression constraints"
            )
        artifact = {
            "clips": clips,
            "selected_sha256": master.selected_sha256,
            "candidate_sha256": sha,
            "protocol": protocol,
            "config": asdict(config),
        }
        atomic_save(artifact, out / "confirmed-clips.pt")
        report.update(
            passed=True,
            status="passed",
            candidate_sha256=sha,
            clips_path=str((out / "confirmed-clips.pt").resolve()),
            clips_sha256=content_hash(artifact),
            gate=gate,
        )
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
    ledger["es"][phase] = report
    atomic_json(workflow.path, ledger)
    atomic_json(out / "report.json", report)
    return report
