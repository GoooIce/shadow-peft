"""Minimal thread-based JSON HTTP service helpers (stdlib only).

The reference deployment keeps the dependency surface small so the edge box can
run the agent with nothing but a Python 3.10+ interpreter. Production packaging
may put the same handlers behind uvicorn/nginx without changing the protocol.
"""

from __future__ import annotations

import contextlib
import json
import os
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlsplit

from .protocol import ProtocolError, dumps


@dataclass(slots=True)
class RawResponse:
    text: str
    content_type: str
    status: int = 200


class HttpError(RuntimeError):
    def __init__(self, status: int, message: str, *, detail: Any = None) -> None:
        super().__init__(message)
        self.status = status
        self.message = message
        self.detail = detail


class BaseJsonHandler(BaseHTTPRequestHandler):
    server_version = "EdgeCloud/0.1"
    protocol_version = "HTTP/1.1"
    max_body_bytes = 1_000_000

    def do_GET(self) -> None:  # noqa: N802 - stdlib naming
        self._dispatch("GET")

    def do_POST(self) -> None:  # noqa: N802 - stdlib naming
        self._dispatch("POST")

    def _dispatch(self, method: str) -> None:
        path = urlsplit(self.path).path
        try:
            if method == "GET":
                result = self.route_get(path)
            else:
                raw = self._read_body()
                payload = json.loads(raw.decode("utf-8")) if raw else {}
                if not isinstance(payload, dict):
                    raise HttpError(400, "请求体必须是 JSON 对象")
                result = self.route_post(path, payload)
        except HttpError as exc:
            self._send_json(
                {"error": exc.message, "detail": exc.detail}, status=exc.status
            )
            return
        except ProtocolError as exc:
            self._send_json({"error": str(exc)}, status=400)
            return
        except json.JSONDecodeError as exc:
            self._send_json({"error": f"JSON 解析失败：{exc.msg}"}, status=400)
            return
        except Exception as exc:  # noqa: BLE001 - never leak a traceback to the line
            self._send_json({"error": "内部错误", "detail": str(exc)}, status=500)
            return

        if isinstance(result, RawResponse):
            self._send_bytes(
                result.text.encode("utf-8"), content_type=result.content_type, status=result.status
            )
        else:
            self._send_json(result)

    def route_get(self, path: str) -> Any:
        raise HttpError(404, f"未知路径: {path}")

    def route_post(self, path: str, payload: dict[str, Any]) -> Any:
        raise HttpError(404, f"未知路径: {path}")

    def _read_body(self) -> bytes:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise HttpError(400, "Content-Length 不合法") from exc
        if length < 0 or length > self.max_body_bytes:
            raise HttpError(413, "请求体过大")
        return self.rfile.read(length)

    def _send_json(self, payload: Any, *, status: int = 200) -> None:
        body = dumps(payload).encode("utf-8")
        self._send_bytes(body, content_type="application/json; charset=utf-8", status=status)

    def _send_bytes(self, body: bytes, *, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        with contextlib.suppress(BrokenPipeError):
            self.wfile.write(body)

    def log_message(self, fmt: str, *args: Any) -> None:
        if os.environ.get("EDGE_CLOUD_HTTP_LOG") == "1":
            super().log_message(fmt, *args)


class ServiceServer:
    """A ThreadingHTTPServer wrapper that can bind to port 0 and be stopped."""

    def __init__(self, handler_cls: type[BaseHTTPRequestHandler], host: str, port: int) -> None:
        self.httpd = ThreadingHTTPServer((host, port), handler_cls)
        self.httpd.daemon_threads = True
        self.httpd.allow_reuse_address = True
        self._thread: threading.Thread | None = None

    @property
    def address(self) -> tuple[str, int]:
        host, port = self.httpd.server_address[:2]
        return str(host), int(port)

    @property
    def url(self) -> str:
        host, port = self.address
        display_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
        return f"http://{display_host}:{port}"

    def start(self) -> ServiceServer:
        self._thread = threading.Thread(
            target=self.httpd.serve_forever, name="edge-cloud-http", daemon=True
        )
        self._thread.start()
        return self

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)
