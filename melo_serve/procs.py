"""Processes: backends melo_serve starts and stops, and the hooks it runs.

A backend runs in a session of its own, so stopping it reaches every process it
started. It gets its own stop signal first (Splash releases its engine on
SIGINT), then TERM, then KILL.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import socket
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Child:
    process: asyncio.subprocess.Process
    log_path: Path
    stop_signal: int = signal.SIGINT
    started_at: float = field(default_factory=time.time)

    @property
    def pid(self) -> int:
        return self.process.pid

    @property
    def alive(self) -> bool:
        return self.process.returncode is None

    async def stop(self, grace_s: float = 60.0) -> None:
        for sig, wait_s in ((self.stop_signal, grace_s), (signal.SIGTERM, 10.0)):
            if not self.alive:
                return
            signal_group(self.pid, sig)
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self.process.wait(), wait_s)
        if self.alive:
            signal_group(self.pid, signal.SIGKILL)
            await self.process.wait()


def signal_group(pid: int, sig: int) -> None:
    try:
        os.killpg(pid, sig)
    except ProcessLookupError:
        pass
    except PermissionError:
        with contextlib.suppress(ProcessLookupError):
            os.kill(pid, sig)


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


async def spawn(
    argv: Sequence[str], env: Mapping[str, str], log_path: Path, stop_signal: int
) -> Child:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "ab") as log:
        process = await asyncio.create_subprocess_exec(
            *argv,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            env={**os.environ, **env},
            start_new_session=True,
        )
    return Child(process, log_path, stop_signal)


@dataclass(frozen=True)
class ShellResult:
    ok: bool
    output: str


async def run_shell(
    command: str, env: Mapping[str, str], timeout_s: float
) -> ShellResult:
    """Run a hook through the shell; its output is the reason when it refuses."""
    process = await asyncio.create_subprocess_shell(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env={**os.environ, **env},
        start_new_session=True,
    )
    try:
        out, _ = await asyncio.wait_for(process.communicate(), timeout_s)
    except TimeoutError:
        signal_group(process.pid, signal.SIGKILL)
        await process.wait()
        return ShellResult(False, f"the hook did not finish within {timeout_s:.0f} s")
    return ShellResult(process.returncode == 0, out.decode(errors="replace").strip())


def free_port() -> int:
    """A free loopback port, picked by the system (never a fixed one)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


async def wait_closed(port: int, timeout_s: float = 30.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            _, writer = await asyncio.open_connection("127.0.0.1", port)
        except OSError:
            return True
        writer.close()
        with contextlib.suppress(OSError):
            await writer.wait_closed()
        await asyncio.sleep(0.2)
    return False


def tail(path: Path, lines: int = 20) -> str:
    try:
        with open(path, "rb") as handle:
            handle.seek(0, os.SEEK_END)
            handle.seek(max(0, handle.tell() - 16384))
            data = handle.read()
    except OSError:
        return ""
    return "\n".join(data.decode(errors="replace").splitlines()[-lines:])


def start_time(pid: int) -> float | None:
    """When *pid* started, to tell it from a later process that reuses the pid."""
    try:
        out = subprocess.run(
            ["ps", "-o", "lstart=", "-p", str(pid)],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
            env={**os.environ, "LC_ALL": "C"},
        ).stdout.strip()
        return time.mktime(time.strptime(out, "%a %b %d %H:%M:%S %Y")) if out else None
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
