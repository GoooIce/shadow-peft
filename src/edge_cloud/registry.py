"""Adapter registry with canary rollout and rollback (适配器灰度下发).

The cloud trains a new edge adapter from flywheel data, publishes it here, and
edge devices resolve which version they should run. Canary traffic is assigned
by a stable device hash so a device does not flap between versions.
"""

from __future__ import annotations

import json
import os
import threading
import time
import zlib
from pathlib import Path
from typing import Any


class RegistryError(ValueError):
    pass


class AdapterRegistry:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        if not self.path.exists():
            self._save_locked(self._empty())

    @staticmethod
    def _empty() -> dict[str, Any]:
        return {
            "active": None,
            "canary": None,
            "previous_active": None,
            "versions": {},
            "history": [],
        }

    def _load_locked(self) -> dict[str, Any]:
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                state = json.load(handle)
        except (json.JSONDecodeError, FileNotFoundError):
            state = self._empty()
        state.setdefault("active", None)
        state.setdefault("canary", None)
        state.setdefault("previous_active", None)
        state.setdefault("versions", {})
        state.setdefault("history", [])
        return state

    def _save_locked(self, state: dict[str, Any]) -> None:
        temp = self.path.with_suffix(self.path.suffix + ".tmp")
        with temp.open("w", encoding="utf-8") as handle:
            json.dump(state, handle, ensure_ascii=False, indent=2)
        os.replace(temp, self.path)

    def publish(
        self,
        version: str,
        *,
        artifact: str,
        notes: str = "",
        metrics: dict[str, Any] | None = None,
        canary_ratio: float = 0.1,
    ) -> dict[str, Any]:
        version = version.strip()
        if not version:
            raise RegistryError("version 不能为空")
        if not 0.0 <= canary_ratio <= 1.0:
            raise RegistryError("canary_ratio 必须在 [0, 1] 内")
        with self._lock:
            state = self._load_locked()
            state["versions"][version] = {
                "version": version,
                "artifact": artifact,
                "notes": notes,
                "metrics": metrics or {},
                "created_at": time.time(),
            }
            if state["active"] is None:
                state["active"] = version
                state["canary"] = None
            else:
                state["canary"] = {"version": version, "ratio": canary_ratio}
            state["history"].append(
                {"action": "publish", "version": version, "canary_ratio": canary_ratio, "at": time.time()}
            )
            self._save_locked(state)
            return state

    def promote(self, version: str) -> dict[str, Any]:
        with self._lock:
            state = self._load_locked()
            if version not in state["versions"]:
                raise RegistryError(f"版本不存在: {version}")
            if state["active"] == version:
                state["canary"] = None
            else:
                state["previous_active"] = state["active"]
                state["active"] = version
                state["canary"] = None
            state["history"].append({"action": "promote", "version": version, "at": time.time()})
            self._save_locked(state)
            return self._public(state)

    def rollback(self) -> dict[str, Any]:
        with self._lock:
            state = self._load_locked()
            previous = state.get("previous_active")
            if not previous:
                raise RegistryError("没有可回滚的历史版本")
            state["active"] = previous
            state["previous_active"] = None
            state["canary"] = None
            state["history"].append({"action": "rollback", "version": previous, "at": time.time()})
            self._save_locked(state)
            return self._public(state)

    def resolve(self, device_id: str) -> dict[str, Any]:
        with self._lock:
            state = self._load_locked()
        active = state.get("active")
        canary = state.get("canary")
        if canary and active:
            bucket = zlib.crc32(f"{device_id}:{canary['version']}".encode()) % 10_000
            if bucket < int(float(canary.get("ratio", 0.0)) * 10_000):
                return {
                    "device_id": device_id,
                    "version": canary["version"],
                    "channel": "canary",
                    "active": active,
                    "canary_ratio": canary.get("ratio", 0.0),
                }
        return {
            "device_id": device_id,
            "version": active,
            "channel": "stable",
            "active": active,
            "canary_ratio": (canary or {}).get("ratio", 0.0),
        }

    def state(self) -> dict[str, Any]:
        with self._lock:
            return self._public(self._load_locked())

    def _public(self, state: dict[str, Any]) -> dict[str, Any]:
        return {
            "active": state.get("active"),
            "canary": state.get("canary"),
            "versions": state.get("versions", {}),
            "history": state.get("history", [])[-50:],
        }
