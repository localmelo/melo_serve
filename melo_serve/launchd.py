"""The login service: a per-user LaunchAgent that keeps melo_serve running.

It starts at login and after a crash, not after a clean stop
(``KeepAlive.SuccessfulExit = false``), so ``melo-serve stop`` stops it until the
next login or ``melo-serve up``.
"""

from __future__ import annotations

import os
import plistlib
import re
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from melo_serve import paths

LABEL = "ai.localmelo.melo_serve"
ENV = "MELO_SERVE_LAUNCHD"  # set for the daemon that launchd starts
AGENTS_ENV = "MELO_SERVE_LAUNCH_AGENTS"  # moves the plist folder (tests)
PATH = "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"


def plist_path() -> Path:
    folder = os.environ.get(AGENTS_ENV) or "~/Library/LaunchAgents"
    return Path(folder).expanduser() / f"{LABEL}.plist"


def program() -> list[str]:
    exe = shutil.which("melo-serve")
    return (
        [exe, "up", "-f"] if exe else [sys.executable, "-m", "melo_serve", "up", "-f"]
    )


def render(args: Sequence[str]) -> bytes:
    env = {"PATH": f"{PATH}:{Path.home() / '.local' / 'bin'}", ENV: "1"}
    for name in ("HF_HOME", paths.HOME_ENV, paths.SOCKET_ENV):
        if os.environ.get(name):
            env[name] = os.environ[name]
    log = str(paths.logs_dir() / "daemon.log")
    plist: dict[str, Any] = {
        "Label": LABEL,
        "ProgramArguments": list(args),
        "RunAtLoad": True,
        "KeepAlive": {"SuccessfulExit": False},
        "ThrottleInterval": 10,
        "ProcessType": "Standard",
        "EnvironmentVariables": env,
        "StandardOutPath": log,
        "StandardErrorPath": log,
    }
    return plistlib.dumps(plist)


def _launchctl(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["launchctl", *args], capture_output=True, text=True, check=False
    )


def _target() -> str:
    return f"gui/{os.getuid()}/{LABEL}"


def install() -> str:
    """Make the service match what ``render`` writes now.

    Returns ``"unchanged"`` (nothing touched: the daemon keeps its models),
    ``"installed"`` (new, or only needed loading) or ``"updated"`` (the plist
    differed, so the daemon was restarted).
    """
    path = plist_path()
    wanted = render(program())
    current = path.read_bytes() if path.exists() else None
    if current == wanted and status()["loaded"]:
        return "unchanged"
    paths.ensure_home()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(wanted)
    _launchctl("bootout", _target())
    done = _launchctl("bootstrap", f"gui/{os.getuid()}", str(path))
    if done.returncode != 0:
        raise RuntimeError(done.stderr.strip() or "launchctl bootstrap failed")
    return "updated" if current not in (None, wanted) else "installed"


def uninstall() -> bool:
    _launchctl("bootout", _target())
    path = plist_path()
    existed = path.exists()
    path.unlink(missing_ok=True)
    return existed


def status() -> dict[str, Any]:
    installed = plist_path().exists()
    out = _launchctl("print", _target()) if installed else None
    running = bool(out and out.returncode == 0 and "state = running" in out.stdout)
    found = re.search(r"\bpid = (\d+)", out.stdout) if out else None
    return {
        "installed": installed,
        "loaded": bool(out and out.returncode == 0),
        "running": running,
        "pid": int(found.group(1)) if found else None,
    }


def kickstart() -> None:
    _launchctl("kickstart", _target())
