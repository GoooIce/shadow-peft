"""SSH-stdio bridge: GB10 owns search state and arithmetic; Mac owns the only budget.

No listener, remote release, settings mutation, or ledger initialization exists
on this transport. A lost reply never cancels an already recorded reservation.
"""

from __future__ import annotations

import json
import os
import platform
import resource
import selectors
import socket
import subprocess
import sys
import time
import uuid
from dataclasses import asdict
from pathlib import Path

from .budget import BudgetError, DeviceLease
from .contracts import CandidateRequest, DataManifest, digest_json


class FrameChannel:
    def __init__(self, reader, writer, *, timeout=45):
        self.reader, self.writer, self.timeout = reader, writer, timeout
        self.pending = b""

    def send(self, value):
        self.writer.write(json.dumps(value, sort_keys=True, allow_nan=False).encode() + b"\n")
        self.writer.flush()

    def receive(self):
        deadline = time.monotonic() + self.timeout
        with selectors.DefaultSelector() as selected:
            selected.register(self.reader, selectors.EVENT_READ)
            while b"\n" not in self.pending:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not selected.select(remaining):
                    raise TimeoutError("coordinator transport timeout; any lease remains occupied")
                chunk = os.read(self.reader.fileno(), 65536)
                if not chunk:
                    raise EOFError("coordinator transport disconnected; any lease remains occupied")
                self.pending += chunk
                if len(self.pending) > 8 * 1024 * 1024:
                    raise ValueError("coordinator frame exceeds its bounded size")
        line, self.pending = self.pending.split(b"\n", 1)
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError("coordinator protocol requires JSON object frames")
        return value


def budget_authority(ledger):
    value = {
        "schema": "mage-w4-budget-authority/v1",
        "path": str(ledger.path.resolve()),
        "settings": ledger.settings,
    }
    return {**value, "sha256": digest_json(value)}


class BudgetProxy:
    """The minimum existing BudgetLedger protocol, forwarded to one SSH-authenticated authority."""

    def __init__(self, channel, authority):
        expected = {key: authority[key] for key in ("schema", "path", "settings")}
        if (
            set(authority) != {"schema", "path", "settings", "sha256"}
            or authority["schema"] != "mage-w4-budget-authority/v1"
            or not Path(authority["path"]).is_absolute()
            or authority["sha256"] != digest_json(expected)
        ):
            raise ValueError("invalid unique budget authority identity")
        self.channel, self.authority = channel, authority
        self.path, self.settings = Path(authority["path"]), authority["settings"]
        self.sequence = 0

    def _call(self, method, args):
        self.sequence += 1
        self.channel.send(
            {
                "type": "budget-call",
                "sequence": self.sequence,
                "authority_sha256": self.authority["sha256"],
                "method": method,
                "args": args,
            }
        )
        reply = self.channel.receive()
        if (
            reply.get("type") != "budget-result"
            or reply.get("sequence") != self.sequence
            or reply.get("authority_sha256") != self.authority["sha256"]
        ):
            raise ValueError("budget response authority/sequence mismatch")
        if not reply["ok"]:
            raise BudgetError(reply["error"])
        return reply["result"]

    def snapshot(self):
        return self._call("snapshot", {})

    def reserve(self, request_id, ticket_id, device_id, seconds, *, run_id=None):
        return DeviceLease(
            **self._call(
                "reserve",
                dict(
                    request_id=request_id,
                    ticket_id=ticket_id,
                    device_id=device_id,
                    seconds=seconds,
                    run_id=run_id,
                ),
            )
        )

    def assert_active(self, lease_id):
        return DeviceLease(**self._call("assert_active", {"lease_id": lease_id}))

    def bind_request(self, lease_id, request_id, request_digest):
        return self._call(
            "bind_request",
            dict(lease_id=lease_id, request_id=request_id, request_digest=request_digest),
        )


class RemoteCoordinator:
    """Authenticated subprocess controller; production command is the configured LAN SSH host."""

    def __init__(
        self,
        command,
        config,
        budget,
        *,
        initialize=False,
        execution_kind="gb10_measured",
        timeout=45,
    ):
        self.command, self.config, self.budget = command, config, budget
        self.initialize, self.execution_kind, self.timeout = initialize, execution_kind, timeout
        self.authority = budget_authority(budget)
        self.budget_sequence, self.operation_sequence = 0, 0
        self.process = None
        self.initialization_timing = None
        self.last_operation_timing = None
        self.timing_artifact_path = None

    def __enter__(self):
        self.process = subprocess.Popen(
            self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None, bufsize=0
        )
        self.channel = FrameChannel(self.process.stdout, self.process.stdin, timeout=self.timeout)
        self.channel.send(
            {
                "type": "initialize",
                "authority": self.authority,
                "config": self.config,
                "initialize": self.initialize,
                "execution_kind": self.execution_kind,
            }
        )
        try:
            ready = self._receive()
            self.initialization_timing = ready.get("timing")
            self.timing_artifact_path = ready.get("timing_artifact_path")
            if (
                ready.get("type") != "ready"
                or ready.get("authority_sha256") != self.authority["sha256"]
            ):
                raise ValueError(f"coordinator initialization failed: {ready}")
        except BaseException:
            self.close()
            raise
        return self

    def _receive(self):
        while True:
            value = self.channel.receive()
            if value.get("type") != "budget-call":
                return value
            self.budget_sequence += 1
            if (
                set(value) != {"type", "sequence", "authority_sha256", "method", "args"}
                or value["sequence"] != self.budget_sequence
                or value["authority_sha256"] != self.authority["sha256"]
                or budget_authority(self.budget) != self.authority
            ):
                raise ValueError("budget call authority/sequence mismatch")
            reply = {
                "type": "budget-result",
                "sequence": self.budget_sequence,
                "authority_sha256": self.authority["sha256"],
                "ok": False,
            }
            try:
                method, args = value["method"], value["args"]
                required = {
                    "snapshot": set(),
                    "reserve": {"request_id", "ticket_id", "device_id", "seconds", "run_id"},
                    "assert_active": {"lease_id"},
                    "bind_request": {"lease_id", "request_id", "request_digest"},
                }
                if (
                    method not in required
                    or not isinstance(args, dict)
                    or set(args) != required[method]
                ):
                    raise BudgetError("unsupported remote budget method or arguments")
                result = getattr(self.budget, method)(**args)
                if isinstance(result, DeviceLease):
                    result = asdict(result)
                reply.update(ok=True, result=result)
            except (BudgetError, TypeError, ValueError) as error:
                reply["error"] = str(error)
            self.channel.send(reply)

    def call(self, operation, args):
        self.operation_sequence += 1
        self.last_operation_timing = None
        self.channel.send(
            {
                "type": "operation",
                "sequence": self.operation_sequence,
                "operation": operation,
                "args": args,
            }
        )
        result = self._receive()
        if result.get("type") != "result" or result.get("sequence") != self.operation_sequence:
            raise ValueError("coordinator operation response mismatch")
        self.last_operation_timing = result.get("timing")
        if not result["ok"]:
            raise ValueError(result["error"])
        return result["result"]

    def close(self):
        if self.process is None:
            return
        if self.process.poll() is None:
            try:
                self.channel.send({"type": "close"})
                self.process.wait(timeout=2)
            except (OSError, subprocess.TimeoutExpired):
                self.process.kill()
                self.process.wait(timeout=5)
        for stream in (self.process.stdin, self.process.stdout):
            if stream is not None:
                stream.close()

    def __exit__(self, *exc):
        self.close()


def _execute(coordinator, operation, args, kind):
    from .center_execution import assignment_from_prepared, decode_full_center
    from .coordinator import DevelopmentConfirmation
    from .development import verify_evaluation_job

    def read(path):
        return json.loads(Path(path).read_text())

    if operation == "snapshot":
        return coordinator.snapshot()
    if operation == "next_mutation_seeds":
        return coordinator.next_mutation_seeds(**args)
    if operation == "claim":
        return asdict(coordinator.claim(**args))
    if operation == "work_request":
        return asdict(coordinator.work_request(**args))
    if operation == "reconcile_claim":
        return asdict(coordinator.reconcile_claim(**args))
    if operation == "register":
        manifest = DataManifest.from_dict(read(args["manifest_path"]))
        assignments = tuple(
            assignment_from_prepared(
                read(row["prepared_path"]), manifest, decode_full_center(read(row["center_path"]))
            )
            for row in args["assignments"]
        )
        return coordinator.begin_round(args["round_id"], assignments)
    if operation == "submit":
        manifest = DataManifest.from_dict(read(args["manifest_path"]))
        snapshot = coordinator.budget.snapshot()
        batches, proofs = [], []
        expected = coordinator.work_request(args["round_id"], args["candidate_id"])
        for role in ("candidate", "current", "q0"):
            directory = Path(args[f"{role}_job_directory"])
            request = CandidateRequest.from_dict(read(directory / "prepared.json")["request"])
            if role == "candidate" and request.request_digest != expected.request_digest:
                raise ValueError("submitted job differs from its registered candidate")
            batch, proof = verify_evaluation_job(
                directory, request, manifest, snapshot, execution_kind=kind
            )
            batches.append(batch)
            proofs.append(proof)
        return {"submission": coordinator.submit(args["round_id"], *batches), "evidence": proofs}
    if operation == "screen":
        return coordinator.screen(**args)
    if operation == "finish":
        confirmation = (
            DevelopmentConfirmation(**args["confirmation"]) if args.get("confirmation") else None
        )
        return coordinator.finish_round(args["round_id"], confirmation)
    if operation == "abort":
        return coordinator.abort_round(**args)
    raise ValueError("unsupported coordinator operation")


def _timing(started, operation, status, *, error=None):
    return {
        "operation": operation,
        "status": status,
        "started_at": started[0],
        "finished_at": time.time(),
        "wall_seconds": time.monotonic() - started[1],
        "host_cpu_seconds": time.process_time() - started[2],
        "hostname": socket.gethostname(),
        "max_rss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "max_rss_unit": "bytes" if platform.system() == "Darwin" else "KiB",
        "cpu_scope": "coordinator process; peak RSS is cumulative for this process",
        "accounting": "remote CPU work and RPC wait; not additional device occupancy",
        "error": error,
    }


def serve_coordinator():
    from .center_execution import open_coordinator_config

    channel = FrameChannel(sys.stdin.buffer, sys.stdout.buffer)
    hello = channel.receive()
    if (
        set(hello) != {"type", "authority", "config", "initialize", "execution_kind"}
        or hello["type"] != "initialize"
    ):
        raise ValueError("explicit coordinator handshake required")
    kind = hello["execution_kind"]
    if kind not in {"fixture", "gb10_measured"} or type(hello["initialize"]) is not bool:
        raise ValueError("explicit coordinator execution kind required")
    if kind == "gb10_measured" and (
        platform.system() != "Linux" or platform.machine() != "aarch64"
    ):
        raise ValueError("real coordinator arithmetic requires GB10 Linux ARM")
    proxy = BudgetProxy(channel, hello["authority"])
    timing_directory = Path(hello["config"]["state_path"]).parent / "coordinator-timings"
    timing_directory.mkdir(parents=True, exist_ok=True)
    timing_path = timing_directory / (str(uuid.uuid4()) + ".jsonl")

    def record(value):
        value = {**value, "authority_sha256": proxy.authority["sha256"], "execution_kind": kind}
        with timing_path.open("a") as stream:
            stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        return value

    started = (time.time(), time.monotonic(), time.process_time())
    try:
        coordinator = open_coordinator_config(
            hello["config"], budget=proxy, require_existing=not hello["initialize"]
        )
    except BaseException as error:
        timing = record(
            _timing(started, "initialize", "failed", error=f"{type(error).__name__}: {error}")
        )
        channel.send(
            {
                "type": "initialization-failed",
                "timing": timing,
                "timing_artifact_path": str(timing_path),
            }
        )
        raise
    timing = record(_timing(started, "initialize", "ok"))
    channel.send(
        {
            "type": "ready",
            "authority_sha256": proxy.authority["sha256"],
            "timing": timing,
            "timing_artifact_path": str(timing_path),
        }
    )
    sequence = 0
    while True:
        frame = channel.receive()
        if frame == {"type": "close"}:
            return
        sequence += 1
        if (
            set(frame) != {"type", "sequence", "operation", "args"}
            or frame["type"] != "operation"
            or frame["sequence"] != sequence
        ):
            raise ValueError("coordinator operation sequence mismatch")
        reply = {"type": "result", "sequence": sequence, "ok": False}
        started = (time.time(), time.monotonic(), time.process_time())
        try:
            reply.update(
                ok=True, result=_execute(coordinator, frame["operation"], frame["args"], kind)
            )
        except Exception as error:
            reply["error"] = f"{type(error).__name__}: {error}"
        finally:
            reply["timing"] = record(
                _timing(
                    started,
                    frame["operation"],
                    "ok" if reply["ok"] else "failed",
                    error=reply.get("error"),
                )
            )
        channel.send(reply)


if __name__ == "__main__":
    serve_coordinator()
