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


def handler_class(ballots, html):
    def ready():
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


def collect(root, image_directory, auxiliary_host, auxiliary_directory):
    while True:
        try:
            reports = {}
            local = image_directory / "report.json"
            if local.is_file():
                reports["image"] = json.loads(local.read_text())
            remote = subprocess.run(
                [
                    "ssh",
                    "-o",
                    "ConnectTimeout=10",
                    auxiliary_host,
                    "cat",
                    str(auxiliary_directory / "report.json"),
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=20,
            )
            reports["auxiliary"] = json.loads(remote.stdout)
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
                progress["phase"] = "ready" if verdict["status"] == "completed" else "invalid"
                atomic_json(root / "verdict.json", verdict)
                atomic_json(root / "progress.json", progress)
                return
            atomic_json(root / "progress.json", progress)
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
    parser.add_argument("--port", type=int, default=8778)
    args = parser.parse_args()
    ballots = PilotBallots(args.root, json.loads(args.data.read_text()))
    html = Path(__file__).resolve().parents[1] / "assets/web/mage-text-pilot.html"
    threading.Thread(
        target=collect,
        args=(args.root, args.image_directory, args.auxiliary_host, args.auxiliary_directory),
        daemon=True,
    ).start()
    ThreadingHTTPServer(("0.0.0.0", args.port), handler_class(ballots, html)).serve_forever()


if __name__ == "__main__":
    main()
