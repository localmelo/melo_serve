"""A client of the control socket, for the command line and for scripts.

The standard library is enough for a few JSON calls, and it keeps
``melo-serve status`` quick to start.
"""

from __future__ import annotations

import http.client
import json
import socket
from pathlib import Path
from typing import Any

from melo_serve import paths


class ControlError(RuntimeError):
    """The daemon refused a call; ``kind`` and ``data`` are its error object."""

    def __init__(
        self, message: str, kind: str = "", data: dict[str, Any] | None = None
    ) -> None:
        super().__init__(message)
        self.kind, self.data = kind, data or {}


class NotRunning(ControlError):
    pass


class _Connection(http.client.HTTPConnection):
    """HTTP over the control socket."""

    def __init__(self, socket_path: Path, timeout: float | None) -> None:
        super().__init__("melo-serve", timeout=timeout)
        self.socket_path = socket_path

    def connect(self) -> None:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(self.timeout)
        try:
            sock.connect(str(self.socket_path))
        except OSError:
            sock.close()
            raise
        self.sock = sock


class Client:
    def __init__(self, socket_path: Path | None = None) -> None:
        self.socket_path = socket_path or paths.socket_path()

    def call(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
        timeout: float | None = 10.0,
    ) -> dict[str, Any]:
        connection = _Connection(self.socket_path, timeout)
        payload = None if body is None else json.dumps(body).encode()
        headers = {} if payload is None else {"Content-Type": "application/json"}
        try:
            connection.request(method, path, body=payload, headers=headers)
            response = connection.getresponse()
            raw = response.read()
        except (FileNotFoundError, ConnectionRefusedError) as exc:
            raise NotRunning("melo_serve is not running") from exc
        except (OSError, http.client.HTTPException) as exc:
            raise ControlError(f"the control socket failed: {exc}") from exc
        finally:
            connection.close()
        try:
            data = json.loads(raw) if raw else {}
        except ValueError:
            data = {}
        if response.status >= 400:
            problem = data.get("error", {}) if isinstance(data, dict) else {}
            raise ControlError(
                problem.get("message") or f"HTTP {response.status}",
                problem.get("type", ""),
                problem,
            )
        return data if isinstance(data, dict) else {}

    def running(self) -> bool:
        try:
            self.call("GET", "/state", timeout=2.0)
        except ControlError:
            return False
        return True

    def state(self) -> dict[str, Any]:
        return self.call("GET", "/state")

    def picks(self) -> dict[str, Any]:
        return self.call("GET", "/picks", timeout=30.0)

    def switch(self, pick: str, **options: Any) -> dict[str, Any]:
        return self.call("POST", "/switch", {"pick": pick, **options}, timeout=None)

    def up(self, role: str = "chat") -> dict[str, Any]:
        return self.call("POST", "/up", {"role": role}, timeout=None)

    def stop(self) -> dict[str, Any]:
        return self.call("POST", "/stop")
