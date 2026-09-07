#!/usr/bin/env python3
"""EXP-009 coordinator: two GB10 workers for gradient episodes, one ledger.

Both workers run ``gradient --no-register``; only this process registers episodes,
recomputes consistency and applies the pre-registered stop rules, so the campaign
ledger has a single writer. Selection and confirmation run locally in sequence.
Restarting the coordinator resumes from whatever the ledger and the episode
directories already prove.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from shadow_lab.mage_qad_campaign import atomic_json  # noqa: E402

SCRIPT = "scripts/run_mage_healed_es_campaign.py"


class CampaignHalted(RuntimeError):
    pass


def read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


class Coordinator:
    def __init__(self, plan: dict):
        self.plan = plan
        self.root = Path(plan["monitor_directory"])
        self.root.mkdir(parents=True, exist_ok=True)
        self.campaign = Path(plan["campaign"])
        self.remote = plan["remote"]
        self.status_path = self.root / "status.json"
        self.status: dict = read_json(self.status_path) or {"events": []}
        self.log_path = self.root / "coordinator.log"

    # -- logging / status ---------------------------------------------

    def log(self, message: str) -> None:
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"{stamp} {message}"
        with self.log_path.open("a") as stream:
            stream.write(line + "\n")
        print(line, flush=True)
        self.status.setdefault("events", []).append(line)
        self.status["events"] = self.status["events"][-200:]

    def publish(self, **updates) -> None:
        self.status.update(updates)
        self.status["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        ledger = read_json(self.campaign / "campaign.json")
        if ledger:
            generation = ledger["generations"][-1]
            self.status["campaign"] = {
                "status": ledger["status"],
                "verdict": ledger.get("verdict"),
                "generation": generation["index"],
                "registered": sorted(int(k) for k in generation["gradient"]["episodes"]),
                "consistency": (generation.get("consistency") or {}).get("checks"),
                "populations": (generation.get("consistency") or {}).get("populations"),
                "selection_completed": sorted(int(k) for k in generation["selection"]["episodes"]),
                "confirmation_completed": sorted(
                    int(k) for k in generation["confirmation"]["episodes"]
                ),
                "selection_decision": generation.get("selection_decision")
                and generation["selection_decision"]["selected_alpha"],
            }
        atomic_json(self.status_path, self.status)

    # -- command plumbing ---------------------------------------------

    def common_args(self) -> list[str]:
        plan = self.plan
        return [
            "--source-campaign",
            plan["source_campaign"],
            "--model-path",
            plan["model"],
            "--eval-pool",
            plan["evaluation_pool"],
            "--calibration",
            plan["calibration"],
            "--hps-checkpoint",
            plan["hps"],
            "--clip-checkpoint",
            plan["clip"],
        ]

    def local(self, *args: str, log_name: str, check: bool = True) -> subprocess.CompletedProcess:
        command = [self.plan["python"], "-u", SCRIPT, *args]
        logs = self.campaign / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        with (logs / log_name).open("a") as log:
            log.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} {shlex.join(command)}\n")
            log.flush()
            result = subprocess.run(
                command, cwd=self.plan["repository"], stdout=log, stderr=subprocess.STDOUT
            )
        if check and result.returncode:
            raise CampaignHalted(f"{args[0]} exited {result.returncode}; see {logs / log_name}")
        return result

    def ssh(self, command: str, *, timeout: int = 600) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["ssh", "-o", "BatchMode=yes", self.remote["host"], command],
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    def rsync(self, source: str, target: str) -> None:
        result = subprocess.run(
            ["rsync", "-a", "--partial", source, target], capture_output=True, text=True
        )
        if result.returncode:
            raise CampaignHalted(f"rsync {source} -> {target} failed: {result.stderr.strip()}")

    def ledger(self) -> dict:
        ledger = read_json(self.campaign / "campaign.json")
        if ledger is None:
            raise CampaignHalted("campaign ledger missing")
        return ledger

    def generation_dir(self, ledger: dict) -> Path:
        return self.campaign / f"generation-{ledger['generations'][-1]['index']:02d}"

    # -- remote worker -------------------------------------------------

    def sync_remote_campaign(self, ledger: dict) -> None:
        host = self.remote["host"]
        remote_campaign = self.remote["campaign"]
        result = self.ssh(
            f"mkdir -p {shlex.quote(remote_campaign)}/logs {shlex.quote(remote_campaign)}/healing"
        )
        if result.returncode:
            raise CampaignHalted(f"remote mkdir failed: {result.stderr.strip()}")
        self.rsync(str(self.campaign / "campaign.json"), f"{host}:{remote_campaign}/campaign.json")
        self.rsync(str(self.campaign / "healing") + "/", f"{host}:{remote_campaign}/healing/")
        generation = ledger["generations"][-1]
        if generation["index"] > 0:
            previous = self.campaign / f"generation-{generation['index'] - 1:02d}"
            self.ssh(f"mkdir -p {shlex.quote(str(remote_campaign))}/{previous.name}")
            for artifact in previous.glob("alpha-*-clips.pt"):
                self.rsync(
                    str(artifact), f"{host}:{remote_campaign}/{previous.name}/{artifact.name}"
                )

    def remote_episode_dir(self, ledger: dict, episode: int) -> str:
        index = ledger["generations"][-1]["index"]
        return f"{self.remote['campaign']}/generation-{index:02d}/gradient-episode-{episode:02d}"

    def remote_signature(self, episode: int) -> str:
        # Matches the exact worker command line (trailing flag disambiguates 1 vs 10).
        return f"{SCRIPT} gradient --campaign {self.remote['campaign']} --episode {episode} --no-register"

    def dispatch_remote(self, ledger: dict, episode: int) -> int:
        remote = self.remote
        episode_dir = self.remote_episode_dir(ledger, episode)
        probe = self.ssh(
            f"if [ -f {shlex.quote(episode_dir)}/episode.json ]; then echo done; "
            f"elif pgrep -f {shlex.quote(self.remote_signature(episode))} >/dev/null; then echo running; "
            f"elif [ -f {shlex.quote(episode_dir)}.claimed ]; then echo stale; else echo fresh; fi"
        )
        state = probe.stdout.strip()
        if state in {"done", "running"}:
            return 0
        force = ["--force"] if state == "stale" else []
        args = [
            remote["python"],
            "-u",
            SCRIPT,
            "gradient",
            "--campaign",
            remote["campaign"],
            "--episode",
            str(episode),
            "--no-register",
            *force,
            *self.common_args(),
        ]
        log = f"{remote['campaign']}/logs/gradient-episode-{episode:02d}.log"
        command = (
            f"cd {shlex.quote(remote['repository'])} && "
            f"setsid nohup {shlex.join(args)} >> {shlex.quote(log)} 2>&1 < /dev/null & echo $!"
        )
        result = self.ssh(command)
        if result.returncode or not result.stdout.strip().isdigit():
            raise CampaignHalted(
                f"remote dispatch failed: {result.stderr.strip()} {result.stdout.strip()}"
            )
        return int(result.stdout.strip())

    def poll_remote(self, ledger: dict, episode: int) -> str:
        episode_dir = self.remote_episode_dir(ledger, episode)
        result = self.ssh(
            f"if [ -f {shlex.quote(episode_dir)}/episode.json ]; then echo done; "
            f"elif pgrep -f {shlex.quote(self.remote_signature(episode))} >/dev/null; then echo running; "
            f"else echo dead; fi"
        )
        if result.returncode:
            return "unreachable"
        return result.stdout.strip() or "unreachable"

    def fetch_remote(self, ledger: dict, episode: int) -> None:
        host = self.remote["host"]
        remote_dir = self.remote_episode_dir(ledger, episode)
        local_dir = self.generation_dir(ledger) / f"gradient-episode-{episode:02d}"
        local_dir.parent.mkdir(parents=True, exist_ok=True)
        self.rsync(f"{host}:{remote_dir}/", str(local_dir) + "/")
        self.rsync(f"{host}:{remote_dir}.claimed", str(local_dir) + ".claimed")
        self.rsync(
            f"{host}:{self.remote['campaign']}/logs/gradient-episode-{episode:02d}.log",
            str(self.campaign / "logs" / f"remote-gradient-episode-{episode:02d}.log"),
        )
        remote_manifest = self.ssh(
            f"cd {shlex.quote(remote_dir)} && find . -type f -print0 | sort -z | xargs -0 sha256sum"
        ).stdout
        local_manifest = subprocess.run(
            "find . -type f -print0 | sort -z | xargs -0 sha256sum",
            shell=True,
            cwd=local_dir,
            capture_output=True,
            text=True,
        ).stdout
        if remote_manifest != local_manifest:
            raise CampaignHalted(f"episode {episode} artifact hashes differ after peer sync")

    def kill_remote(self, episode: int | None) -> None:
        if episode is not None:
            self.ssh(f"pkill -f {shlex.quote(self.remote_signature(episode))} || true", timeout=60)

    # -- phases ---------------------------------------------------------

    def register(self, episode: int) -> None:
        self.local(
            "register",
            "--campaign",
            str(self.campaign),
            "--episode",
            str(episode),
            "--eval-pool",
            self.plan["evaluation_pool"],
            log_name="register.log",
            check=False,
        )
        ledger = self.ledger()
        registered = ledger["generations"][-1]["gradient"]["episodes"]
        if str(episode) not in registered and ledger["status"] != "stopped":
            raise CampaignHalted(f"registering episode {episode} failed; see logs/register.log")
        self.log(f"registered gradient episode {episode}")
        self.publish()

    def gradient_phase(self) -> None:
        ledger = self.ledger()
        protocol = ledger["protocol"]
        generation = ledger["generations"][-1]
        registered = {int(k) for k in generation["gradient"]["episodes"]}
        local_queue = [e for e in self.plan["local_episodes"] if e not in registered]
        remote_queue = [e for e in self.plan["remote_episodes"] if e not in registered]
        if set(self.plan["local_episodes"]) | set(self.plan["remote_episodes"]) != set(
            protocol["gradient_episodes"]
        ):
            raise CampaignHalted("episode assignment does not cover the frozen gradient episodes")
        if remote_queue:
            self.sync_remote_campaign(ledger)
        local_proc: subprocess.Popen | None = None
        local_episode: int | None = None
        remote_pid: int | None = None
        remote_episode: int | None = None
        generation_dir = self.generation_dir(ledger)
        logs = self.campaign / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        try:
            while local_queue or remote_queue or local_proc or remote_episode is not None:
                # Local slot.
                if local_proc is None and local_queue:
                    local_episode = local_queue.pop(0)
                    episode_dir = generation_dir / f"gradient-episode-{local_episode:02d}"
                    if (episode_dir / "episode.json").exists():
                        self.register(local_episode)
                        local_episode = None
                    else:
                        force = (
                            ["--force"]
                            if episode_dir.with_name(f"{episode_dir.name}.claimed").exists()
                            else []
                        )
                        command = [
                            self.plan["python"],
                            "-u",
                            SCRIPT,
                            "gradient",
                            "--campaign",
                            str(self.campaign),
                            "--episode",
                            str(local_episode),
                            "--no-register",
                            *force,
                            *self.common_args(),
                        ]
                        log = (logs / f"gradient-episode-{local_episode:02d}.log").open("a")
                        local_proc = subprocess.Popen(
                            command,
                            cwd=self.plan["repository"],
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            start_new_session=True,
                        )
                        self.log(
                            f"local gradient episode {local_episode} started pid {local_proc.pid}"
                        )
                        self.publish(
                            local={
                                "episode": local_episode,
                                "pid": local_proc.pid,
                                "state": "running",
                            }
                        )
                elif local_proc is not None and local_proc.poll() is not None:
                    code = local_proc.returncode
                    local_proc = None
                    if code:
                        raise CampaignHalted(
                            f"local gradient episode {local_episode} exited {code}"
                        )
                    assert local_episode is not None
                    self.register(local_episode)
                    self.publish(local={"episode": local_episode, "state": "registered"})
                    local_episode = None
                # Remote slot.
                if remote_episode is None and remote_queue:
                    remote_episode = remote_queue.pop(0)
                    remote_pid = self.dispatch_remote(ledger, remote_episode)
                    self.log(
                        f"remote gradient episode {remote_episode} dispatched pid {remote_pid}"
                    )
                    self.publish(
                        remote={"episode": remote_episode, "pid": remote_pid, "state": "running"}
                    )
                elif remote_episode is not None:
                    state = self.poll_remote(ledger, remote_episode)
                    if state == "done":
                        self.fetch_remote(ledger, remote_episode)
                        self.register(remote_episode)
                        self.publish(remote={"episode": remote_episode, "state": "registered"})
                        remote_episode = None
                        remote_pid = None
                    elif state == "dead":
                        raise CampaignHalted(
                            f"remote gradient episode {remote_episode} died; see remote log"
                        )
                    elif state == "unreachable":
                        self.log("remote unreachable; retrying next poll")
                ledger = self.ledger()
                if ledger["status"] == "stopped":
                    raise CampaignHalted(
                        f"campaign stopped: {ledger['verdict']}: {ledger.get('stop_detail')}"
                    )
                self.publish(phase="gradient")
                time.sleep(self.plan.get("poll_seconds", 60))
        except BaseException:
            if local_proc is not None and local_proc.poll() is None:
                os.killpg(local_proc.pid, signal.SIGTERM)
            self.kill_remote(remote_episode)
            raise

    def centered_phase(self, phase: str) -> None:
        ledger = self.ledger()
        generation = ledger["generations"][-1]
        done = {int(k) for k in generation[phase]["episodes"]}
        for episode in ledger["protocol"][f"{phase}_episodes"]:
            if episode in done:
                continue
            command = "select" if phase == "selection" else "confirm"
            episode_dir = self.generation_dir(ledger) / f"{phase}-episode-{episode:02d}"
            force = (
                ["--force"] if episode_dir.with_name(f"{episode_dir.name}.claimed").exists() else []
            )
            self.log(f"{phase} episode {episode} started")
            self.publish(phase=phase, local={"episode": episode, "state": "running"})
            self.local(
                command,
                "--campaign",
                str(self.campaign),
                "--episode",
                str(episode),
                *force,
                *self.common_args(),
                log_name=f"{phase}-episode-{episode:02d}.log",
            )
            self.log(f"{phase} episode {episode} completed")
            self.publish()
        self.local(
            "select-decide" if phase == "selection" else "confirm-decide",
            "--campaign",
            str(self.campaign),
            log_name=f"{phase}-decide.log",
            check=False,
        )
        ledger = self.ledger()
        if ledger["status"] == "stopped":
            raise CampaignHalted(
                f"campaign stopped: {ledger['verdict']}: {ledger.get('stop_detail')}"
            )
        self.log(f"{phase} decided; campaign status {ledger['status']}")
        self.publish()

    def run(self) -> None:
        plan = self.plan
        if not (self.campaign / "campaign.json").exists():
            self.log("initializing campaign")
            self.publish(phase="init")
            self.local(
                "init",
                "--campaign",
                str(self.campaign),
                "--protocol",
                plan["protocol"],
                *self.common_args(),
                log_name="init.log",
            )
        while True:
            status = self.ledger()["status"]
            self.publish(phase=status)
            if status == "gradient":
                self.gradient_phase()
                ledger = self.ledger()
                if ledger["status"] != "gradient":
                    continue
                self.log("all gradient episodes registered; stepping")
                self.local(
                    "step",
                    "--campaign",
                    str(self.campaign),
                    *self.common_args(),
                    log_name="step.log",
                )
            elif status == "selection":
                self.centered_phase("selection")
            elif status == "confirmation":
                self.centered_phase("confirmation")
            elif status == "awaiting_maintainer":
                self.log(
                    "generation confirmed; awaiting maintainer: authorize-generation or declare-final"
                )
                self.publish(phase="awaiting_maintainer")
                return
            elif status == "stopped":
                ledger = self.ledger()
                raise CampaignHalted(
                    f"campaign stopped: {ledger['verdict']}: {ledger.get('stop_detail')}"
                )
            else:
                self.log(f"campaign status {status}; coordinator has nothing to run")
                self.publish(phase=status)
                return


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    coordinator = Coordinator(plan)
    try:
        coordinator.run()
    except CampaignHalted as error:
        coordinator.log(f"HALTED: {error}")
        coordinator.publish(phase="halted", error=str(error))
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
