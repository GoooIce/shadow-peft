#!/usr/bin/env python3
"""Collect a two-host pilot and serve a blind ballot on the LAN."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import threading
import time
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from shadow_lab.mage_pilot_ballot import PilotBallots
from shadow_lab.mage_qad_campaign import atomic_json
from shadow_lab.mage_qad_text_pilot import compare_pilot_reports


def bind_experiment(root, identity):
    path = root / "experiment.json"
    if path.is_file():
        if json.loads(path.read_text()) != identity:
            raise ValueError("ballot root belongs to a different experiment")
    else:
        if (root / "verdict.json").exists():
            raise ValueError("cannot adopt an unbound completed ballot")
        atomic_json(path, identity)
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def handler_class(ballots, html, readiness=None):
    def ready():
        if readiness is not None:
            return readiness.is_set()
        path = ballots.root / "verdict.json"
        return path.is_file() and json.loads(path.read_text())["status"] == "completed"

    class Handler(BaseHTTPRequestHandler):
        def respond(self, value, status=200, kind="application/json; charset=utf-8", token=None):
            payload = value if isinstance(value, bytes) else json.dumps(value).encode()
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            if token:
                self.send_header(
                    "Set-Cookie",
                    f"pilot={token}; HttpOnly; SameSite=Strict; Path=/; Max-Age=2592000",
                )
            self.end_headers()
            self.wfile.write(payload)

        def session(self):
            cookies = SimpleCookie(self.headers.get("Cookie", ""))
            return ballots.session(cookies["pilot"].value if "pilot" in cookies else None)

        def do_GET(self):
            with ballots.lock:
                try:
                    if self.path == "/":
                        return self.respond(html.read_bytes(), kind="text/html; charset=utf-8")
                    if self.path == "/api/state":
                        session = self.session()
                        progress_path = ballots.root / "progress.json"
                        progress = (
                            json.loads(progress_path.read_text())
                            if progress_path.exists()
                            else {"phase": "connecting"}
                        )
                        state = dict(ready=ready(), progress=progress)
                        if state["ready"]:
                            state["ballot"] = ballots.public(session)
                        return self.respond(state, token=session["token"])
                    if self.path.startswith("/asset/") and ready():
                        path = ballots.image_path(self.session(), self.path.removeprefix("/asset/"))
                        return self.respond(path.read_bytes(), kind="image/png")
                    return self.respond({"error": "未找到"}, 404)
                except (ValueError, OSError) as error:
                    return self.respond({"error": str(error)}, 400)

        def do_POST(self):
            with ballots.lock:
                try:
                    origin = self.headers.get("Origin")
                    if origin and origin != "http://" + self.headers.get("Host", ""):
                        raise ValueError("请求来源不匹配")
                    if not ready():
                        raise ValueError("实验尚未完成")
                    length = int(self.headers.get("Content-Length", "0"))
                    if not 0 < length <= 4096:
                        raise ValueError("请求大小不正确")
                    body = json.loads(self.rfile.read(length))
                    session = self.session()
                    if self.path == "/api/vote":
                        ballots.vote(session, body.get("id"), body.get("choice"))
                    elif self.path == "/api/submit":
                        ballots.submit(session)
                    elif self.path == "/api/reveal":
                        result = ballots.reveal(session)
                        result["metrics"] = json.loads((ballots.root / "verdict.json").read_text())
                        return self.respond(result, token=session["token"])
                    else:
                        return self.respond({"error": "未找到"}, 404)
                    return self.respond(ballots.public(session), token=session["token"])
                except (ValueError, OSError) as error:
                    return self.respond({"error": str(error)}, 400)

    return Handler


def collect(
    root,
    image_directory,
    auxiliary_host,
    auxiliary_directory,
    identity,
    experiment_hash,
    readiness,
    image_pid,
    auxiliary_pid,
):
    committed = root / "verdict.json"
    if committed.is_file():
        try:
            verdict = json.loads(committed.read_text())
            if verdict.get("experiment_sha256") != experiment_hash:
                raise ValueError("completed ballot identity mismatch")
            for relative, digest in json.loads((root / "artifact-sha256.json").read_text()).items():
                if hashlib.sha256((root / relative).read_bytes()).hexdigest() != digest:
                    raise ValueError("completed ballot artifact changed")
            if verdict["status"] == "completed":
                readiness.set()
        except (ValueError, OSError, KeyError) as error:
            atomic_json(root / "progress.json", dict(phase="invalid", updated_at=time.time()))
            atomic_json(root / "collector-error.json", dict(error=str(error)))
        return
    while True:
        try:
            reports = {}
            stat_path = Path(f"/proc/{image_pid}/stat")
            image_alive = (
                stat_path.is_file() and stat_path.read_text().split(") ", 1)[1].split()[0] != "Z"
            )
            local = image_directory / "report.json"
            if local.is_file():
                reports["image"] = json.loads(local.read_text())
            elif not image_alive:
                atomic_json(root / "progress.json", dict(phase="failed", updated_at=time.time()))
                return
            remote_code = (
                "import json; from pathlib import Path; "
                f"p=Path('/proc/{auxiliary_pid}/stat'); "
                "alive=p.is_file() and p.read_text().split(') ',1)[1].split()[0]!='Z'; "
                f"r=Path({str(auxiliary_directory / 'report.json')!r}); "
                "print(json.dumps(dict(alive=alive,report=json.loads(r.read_text()))))"
            )
            remote = subprocess.run(
                [
                    "ssh",
                    "-o",
                    "ConnectTimeout=10",
                    auxiliary_host,
                    "python3 -c " + shlex.quote(remote_code),
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=20,
            )
            remote_state = json.loads(remote.stdout)
            reports["auxiliary"] = remote_state["report"]
            for arm, report in reports.items():
                if report["data_sha256"] != identity["data_sha256"]:
                    raise ValueError("displayed prompts differ from experiment data")
                alive = image_alive if arm == "image" else remote_state["alive"]
                if not alive and report["status"] not in {"completed", "failed"}:
                    report.update(status="failed", phase="failed")
            progress = dict(
                updated_at=time.time(),
                phase="running",
                tasks=[
                    dict(
                        label=f"任务 {index + 1}",
                        phase=r.get("phase"),
                        completed_steps=len(r.get("train", [])),
                        total_steps=8,
                        baseline_done=len(r.get("baseline", [])),
                        validation_done=len(r.get("validation", [])),
                        status=r["status"],
                    )
                    for index, r in enumerate(reports.values())
                ],
            )
            if any(r["status"] == "failed" for r in reports.values()):
                progress["phase"] = "failed"
                atomic_json(root / "progress.json", progress)
                return
            if len(reports) == 2 and all(r["status"] == "completed" for r in reports.values()):
                for arm in reports:
                    (root / arm / "images").mkdir(parents=True, exist_ok=True)
                    atomic_json(root / arm / "report.json", reports[arm])
                subprocess.run(
                    [
                        "rsync",
                        "-a",
                        str(image_directory / "images") + "/",
                        str(root / "image/images") + "/",
                    ],
                    check=True,
                    timeout=60,
                )
                subprocess.run(
                    [
                        "rsync",
                        "-a",
                        f"{auxiliary_host}:{auxiliary_directory}/images/",
                        str(root / "auxiliary/images") + "/",
                    ],
                    check=True,
                    timeout=60,
                )
                checksum_code = (
                    "import hashlib,json; from pathlib import Path; "
                    f"p=Path({str(auxiliary_directory / 'images')!r}); "
                    "print(json.dumps({str(f.relative_to(p)):hashlib.md5(f.read_bytes()).hexdigest() "
                    "for f in p.rglob('*.png')}))"
                )
                checksums = subprocess.run(
                    [
                        "ssh",
                        "-o",
                        "ConnectTimeout=10",
                        auxiliary_host,
                        "python3 -c " + shlex.quote(checksum_code),
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                expected_checksums = json.loads(checksums.stdout)
                actual_checksums = {
                    str(f.relative_to(root / "auxiliary/images")): hashlib.md5(
                        f.read_bytes()
                    ).hexdigest()
                    for f in (root / "auxiliary/images").rglob("*.png")
                }
                if actual_checksums != expected_checksums:
                    raise ValueError("remote image transfer checksum mismatch")
                atomic_json(root / "image-transfer-md5.json", actual_checksums)
                for arm in reports:
                    for index in range(8):
                        if not (root / arm / "images/validation" / f"{index:02d}.png").is_file():
                            raise ValueError("missing final image")
                verdict = compare_pilot_reports(reports["image"], reports["auxiliary"])
                verdict["experiment_sha256"] = experiment_hash
                progress["phase"] = "ready" if verdict["status"] == "completed" else "invalid"
                artifact_paths = [
                    f for arm in reports for f in (root / arm).rglob("*") if f.is_file()
                ]
                atomic_json(
                    root / "artifact-sha256.json",
                    {
                        str(f.relative_to(root)): hashlib.sha256(f.read_bytes()).hexdigest()
                        for f in artifact_paths
                    },
                )
                atomic_json(root / "verdict.json", verdict)
                atomic_json(root / "progress.json", progress)
                if verdict["status"] == "completed":
                    readiness.set()
                return
            atomic_json(root / "progress.json", progress)
        except ValueError as error:
            atomic_json(root / "collector-error.json", dict(error=str(error)))
            atomic_json(
                root / "progress.json",
                dict(phase="invalid", updated_at=time.time(), error_type=type(error).__name__),
            )
            return
        except Exception as error:
            atomic_json(
                root / "progress.json",
                dict(phase="reconnecting", updated_at=time.time(), error_type=type(error).__name__),
            )
        time.sleep(30)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--image-directory", type=Path, required=True)
    parser.add_argument("--auxiliary-host", default="gooo")
    parser.add_argument("--auxiliary-directory", type=Path, required=True)
    parser.add_argument("--image-pid", type=int, required=True)
    parser.add_argument("--auxiliary-pid", type=int, required=True)
    parser.add_argument("--port", type=int, default=8778)
    args = parser.parse_args()
    identity = dict(
        data_sha256=hashlib.sha256(args.data.read_bytes()).hexdigest(),
        image_directory=str(args.image_directory),
        auxiliary_host=args.auxiliary_host,
        auxiliary_directory=str(args.auxiliary_directory),
    )
    experiment_hash = bind_experiment(args.root, identity)
    ballots = PilotBallots(args.root, json.loads(args.data.read_text()))
    readiness = threading.Event()
    html = Path(__file__).resolve().parents[1] / "assets/web/mage-text-pilot.html"
    threading.Thread(
        target=collect,
        args=(
            args.root,
            args.image_directory,
            args.auxiliary_host,
            args.auxiliary_directory,
            identity,
            experiment_hash,
            readiness,
            args.image_pid,
            args.auxiliary_pid,
        ),
        daemon=True,
    ).start()
    ThreadingHTTPServer(
        ("0.0.0.0", args.port), handler_class(ballots, html, readiness)
    ).serve_forever()


if __name__ == "__main__":
    main()
