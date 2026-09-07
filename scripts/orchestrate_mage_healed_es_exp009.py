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
import shlex
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

    def signature(self, campaign: str, episode: int) -> str:
        # pgrep regex for the exact worker command line. The bracketed first character
        # keeps the probing shell, whose own command line contains this text, from
        # matching itself; the trailing flag disambiguates episode 1 from 10.
        return (
            f"[s]cripts/run_mage_healed_es_campaign.py gradient --campaign {campaign} "
            f"--episode {episode} --no-register"
        )

    def probe_state(self, *, remote: bool, ledger: dict, episode: int) -> str:
        """done | running | stale | fresh | unreachable for one worker slot."""
        if remote:
            episode_dir = self.remote_episode_dir(ledger, episode)
            signature = self.signature(self.remote["campaign"], episode)
            result = self.ssh(
                f"if [ -f {shlex.quote(episode_dir)}/episode.json ]; then echo done; "
                f"elif pgrep -f {shlex.quote(signature)} >/dev/null; then echo running; "
                f"elif [ -f {shlex.quote(episode_dir)}.claimed ]; then echo stale; else echo fresh; fi"
            )
            if result.returncode:
                return "unreachable"
            return result.stdout.strip() or "unreachable"
        episode_dir = self.generation_dir(ledger) / f"gradient-episode-{episode:02d}"
        if (episode_dir / "episode.json").exists():
            return "done"
        probe = subprocess.run(
            ["pgrep", "-f", self.signature(str(self.campaign), episode)], capture_output=True
        )
        if probe.returncode == 0:
            return "running"
        if episode_dir.with_name(f"{episode_dir.name}.claimed").exists():
            return "stale"
        return "fresh"

    def dispatch(self, *, remote: bool, ledger: dict, episode: int) -> str:
        """Start a worker unless one already finished or is running; return the state.

        A running worker is adopted rather than restarted, so a coordinator restart
        never clobbers an episode that is still being written.
        """
        state = self.probe_state(remote=remote, ledger=ledger, episode=episode)
        if state in {"done", "running", "unreachable"}:
            return state
        force = ["--force"] if state == "stale" else []
        target = (
            self.remote
            if remote
            else {
                "python": self.plan["python"],
                "campaign": str(self.campaign),
                "repository": self.plan["repository"],
            }
        )
        args = [
            target["python"],
            "-u",
            SCRIPT,
            "gradient",
            "--campaign",
            target["campaign"],
            "--episode",
            str(episode),
            "--no-register",
            *force,
            *self.common_args(),
        ]
        log = f"{target['campaign']}/logs/gradient-episode-{episode:02d}.log"
        if remote:
            command = (
                f"cd {shlex.quote(target['repository'])} && "
                f"mkdir -p {shlex.quote(target['campaign'])}/logs && "
                f"setsid nohup {shlex.join(args)} >> {shlex.quote(log)} 2>&1 < /dev/null & echo $!"
            )
            result = self.ssh(command)
            if result.returncode or not result.stdout.strip().isdigit():
                raise CampaignHalted(
                    f"remote dispatch failed: {result.stderr.strip()} {result.stdout.strip()}"
                )
        else:
            Path(log).parent.mkdir(parents=True, exist_ok=True)
            with open(log, "a") as stream:
                subprocess.Popen(
                    args,
                    cwd=target["repository"],
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    start_new_session=True,
                )
        return "running"

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

    def kill_worker(self, *, remote: bool, episode: int | None) -> None:
        if episode is None:
            return
        if remote:
            signature = self.signature(self.remote["campaign"], episode)
            self.ssh(f"pkill -f {shlex.quote(signature)} || true", timeout=60)
        else:
            subprocess.run(
                ["pkill", "-f", self.signature(str(self.campaign), episode)], capture_output=True
            )

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
        queues = {
            "local": [e for e in self.plan["local_episodes"] if e not in registered],
            "remote": [e for e in self.plan["remote_episodes"] if e not in registered],
        }
        if set(self.plan["local_episodes"]) | set(self.plan["remote_episodes"]) != set(
            protocol["gradient_episodes"]
        ):
            raise CampaignHalted("episode assignment does not cover the frozen gradient episodes")
        if queues["remote"]:
            self.sync_remote_campaign(ledger)
        active: dict[str, int | None] = {"local": None, "remote": None}
        try:
            while any(queues.values()) or any(e is not None for e in active.values()):
                for slot in ("local", "remote"):
                    remote = slot == "remote"
                    if active[slot] is None and queues[slot]:
                        episode = queues[slot][0]
                        state = self.dispatch(remote=remote, ledger=ledger, episode=episode)
                        if state == "unreachable":
                            self.log(f"{slot} unreachable while dispatching episode {episode}")
                            continue
                        queues[slot].pop(0)
                        active[slot] = episode
                        self.log(f"{slot} gradient episode {episode} {state}")
                        self.publish(**{slot: {"episode": episode, "state": state}})
                    elif active[slot] is not None:
                        episode = active[slot]
                        assert episode is not None
                        state = self.probe_state(remote=remote, ledger=ledger, episode=episode)
                        if state == "done":
                            if remote:
                                self.fetch_remote(ledger, episode)
                            self.register(episode)
                            self.publish(**{slot: {"episode": episode, "state": "registered"}})
                            active[slot] = None
                        elif state in {"stale", "fresh"}:
                            raise CampaignHalted(
                                f"{slot} gradient episode {episode} exited without episode.json; "
                                f"see logs/gradient-episode-{episode:02d}.log"
                            )
                        elif state == "unreachable":
                            self.log(f"{slot} unreachable; retrying next poll")
                ledger = self.ledger()
                if ledger["status"] == "stopped":
                    raise CampaignHalted(
                        f"campaign stopped: {ledger['verdict']}: {ledger.get('stop_detail')}"
                    )
                self.publish(phase="gradient")
                time.sleep(self.plan.get("poll_seconds", 60))
        except CampaignHalted:
            # A pre-registered stop or a dead worker ends the campaign: stop the peer
            # too. Any other failure leaves workers running for the next coordinator.
            for slot, episode in active.items():
                self.kill_worker(remote=slot == "remote", episode=episode)
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
