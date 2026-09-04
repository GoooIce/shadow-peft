"""Small localhost HTTP adapter for the AnnotationStore interface."""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .errors import AnnotationError, AuthorizationError, ConflictError, NotFoundError
from .models import AnswerCommand, ReviewCommand
from .store import AnnotationStore

STATIC_ROOT = Path(__file__).with_name("static")


def _error_status(error: AnnotationError) -> HTTPStatus:
    if isinstance(error, AuthorizationError):
        return HTTPStatus.UNAUTHORIZED
    if isinstance(error, NotFoundError):
        return HTTPStatus.NOT_FOUND
    if isinstance(error, ConflictError):
        return HTTPStatus.CONFLICT
    return HTTPStatus.BAD_REQUEST


def make_handler(store: AnnotationStore, experiment_id: str):
    class Handler(BaseHTTPRequestHandler):
        server_version = "ShadowAnnotation/0.1"

        def log_message(self, format: str, *args: Any) -> None:
            super().log_message(format, *args)

        def _headers(self, status: HTTPStatus, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; img-src 'self' blob:; style-src 'self'; "
                "script-src 'self'; connect-src 'self'; frame-ancestors 'none'",
            )

        def _json(self, status: HTTPStatus, value: object) -> None:
            payload = json.dumps(value, ensure_ascii=False, sort_keys=True).encode()
            self._headers(status, "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _static(self, name: str, content_type: str) -> None:
            path = STATIC_ROOT / name
            payload = path.read_bytes()
            self._headers(HTTPStatus.OK, content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _principal(self):
            authorization = self.headers.get("Authorization", "")
            if not authorization.startswith("Bearer "):
                raise AuthorizationError("bearer token required")
            return store.authenticate(authorization.removeprefix("Bearer "))

        def _body(self) -> dict:
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError as error:
                raise ConflictError("invalid content length") from error
            if length <= 0 or length > 64 * 1024:
                raise ConflictError("JSON body must be between 1 and 65536 bytes")
            try:
                body = json.loads(self.rfile.read(length))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ConflictError("invalid JSON body") from error
            if not isinstance(body, dict):
                raise ConflictError("JSON body must be an object")
            return body

        def do_GET(self) -> None:
            try:
                self._do_get()
            except AnnotationError as error:
                self._json(_error_status(error), {"error": str(error)})

        def _do_get(self) -> None:
            path = self.path.split("?", 1)[0]
            if path == "/":
                self._static("index.html", "text/html; charset=utf-8")
                return
            if path == "/app.js":
                self._static("app.js", "text/javascript; charset=utf-8")
                return
            if path == "/styles.css":
                self._static("styles.css", "text/css; charset=utf-8")
                return
            if path == "/api/health":
                self._json(HTTPStatus.OK, {"ok": True})
                return
            if path == "/favicon.ico":
                self.send_response(HTTPStatus.NO_CONTENT)
                self.end_headers()
                return
            principal = self._principal()
            if path == "/api/me":
                self._json(HTTPStatus.OK, asdict(principal))
                return
            if path == "/api/next":
                if principal.role == "reviewer":
                    assignment = store.claim_review(principal, experiment_id)
                else:
                    assignment = store.claim_next(principal, experiment_id)
                self._json(
                    HTTPStatus.OK,
                    {"assignment": asdict(assignment) if assignment is not None else None},
                )
                return
            if path == "/api/progress":
                progress = (
                    store.review_progress(principal, experiment_id)
                    if principal.role == "reviewer"
                    else store.progress(principal, experiment_id)
                )
                self._json(
                    HTTPStatus.OK,
                    asdict(progress),
                )
                return
            match = re.fullmatch(r"/api/assignments/([0-9a-f-]{36})/media/(left|right)", path)
            if match:
                lease_token = self.headers.get("X-Lease-Token", "")
                payload, mime_type = store.read_media(
                    principal,
                    match.group(1),
                    match.group(2),
                    lease_token,
                )
                self._headers(HTTPStatus.OK, mime_type)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            match = re.fullmatch(r"/api/reviews/([0-9a-f-]{36})/media/(left|right)", path)
            if match:
                lease_token = self.headers.get("X-Lease-Token", "")
                payload, mime_type = store.read_review_media(
                    principal,
                    match.group(1),
                    match.group(2),
                    lease_token,
                )
                self._headers(HTTPStatus.OK, mime_type)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            raise NotFoundError("route not found")

        def do_POST(self) -> None:
            try:
                self._do_post()
            except AnnotationError as error:
                self._json(_error_status(error), {"error": str(error)})

        def _do_post(self) -> None:
            principal = self._principal()
            body = self._body()
            if self.path == "/api/answers":
                command = AnswerCommand(
                    assignment_id=body.get("assignment_id", ""),
                    lease_token=body.get("lease_token", ""),
                    idempotency_key=body.get("idempotency_key", ""),
                    decision=body.get("decision", ""),
                    reason_codes=tuple(body.get("reason_codes", ())),
                    reason_target=body.get("reason_target", "none"),
                    comment=body.get("comment"),
                    flagged=body.get("flagged", False),
                    client_version=body.get("client_version", "web-0.1"),
                )
                receipt = store.submit_answer(
                    principal,
                    command,
                    session_id=body.get("session_id", ""),
                )
                self._json(HTTPStatus.OK, asdict(receipt))
                return
            if self.path == "/api/reviews":
                command = ReviewCommand(
                    review_assignment_id=body.get("review_assignment_id", ""),
                    lease_token=body.get("lease_token", ""),
                    idempotency_key=body.get("idempotency_key", ""),
                    decision=body.get("decision", ""),
                    comment=body.get("comment"),
                    client_version=body.get("client_version", "web-0.2"),
                )
                receipt = store.submit_review(
                    principal,
                    command,
                    session_id=body.get("session_id", ""),
                )
                self._json(HTTPStatus.OK, asdict(receipt))
                return
            match = re.fullmatch(r"/api/answers/([0-9a-f-]{36})/undo", self.path)
            if match:
                receipt = store.undo_answer(
                    principal,
                    match.group(1),
                    body.get("idempotency_key", ""),
                    session_id=body.get("session_id", ""),
                )
                self._json(HTTPStatus.OK, asdict(receipt))
                return
            match = re.fullmatch(r"/api/reviews/([0-9a-f-]{36})/undo", self.path)
            if match:
                receipt = store.undo_review(
                    principal,
                    match.group(1),
                    body.get("idempotency_key", ""),
                    session_id=body.get("session_id", ""),
                )
                self._json(HTTPStatus.OK, asdict(receipt))
                return
            raise NotFoundError("route not found")

    return Handler


def serve(
    store: AnnotationStore,
    experiment_id: str,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
) -> None:
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("non-localhost binding requires a reviewed TLS/auth deployment adapter")
    server = ThreadingHTTPServer((host, port), make_handler(store, experiment_id))
    print(f"Shadow annotation UI: http://{host}:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
