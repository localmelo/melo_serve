"""Where melo_serve keeps its files: one folder, ``~/.cache/melo_serve`` by default.

``MELO_SERVE_HOME`` moves the folder (tests use it) and ``MELO_SERVE_SOCKET``
moves the control socket alone, since macOS limits a socket path to 104 bytes.
"""

from __future__ import annotations

import os
from pathlib import Path

HOME_ENV = "MELO_SERVE_HOME"
SOCKET_ENV = "MELO_SERVE_SOCKET"
SOCKET_PATH_MAX = 100


def home() -> Path:
    value = os.environ.get(HOME_ENV)
    return Path(value).expanduser() if value else Path.home() / ".cache" / "melo_serve"


def ensure_home() -> Path:
    """Create the folder (only the user may enter it) and its ``logs`` folder."""
    root = home()
    root.mkdir(parents=True, exist_ok=True)
    os.chmod(root, 0o700)
    (root / "logs").mkdir(exist_ok=True)
    return root


def config_path() -> Path:
    return home() / "config.toml"


def state_path() -> Path:
    return home() / "state.json"


def local_picks_path() -> Path:
    return home() / "picks.toml"


def logs_dir() -> Path:
    return home() / "logs"


def downloads_dir() -> Path:
    return home() / "downloads"


def lock_path() -> Path:
    return home() / "daemon.lock"


def get_lock_path() -> Path:
    return home() / "get.lock"


def backend_dir(name: str) -> Path:
    """A folder of melo_serve's own for one backend instance (an oMLX base path)."""
    return home() / "backends" / name


def socket_path() -> Path:
    value = os.environ.get(SOCKET_ENV)
    path = Path(value).expanduser() if value else home() / "control.sock"
    if len(str(path)) > SOCKET_PATH_MAX:
        raise ValueError(
            f"the control socket path is too long for macOS ({len(str(path))} > "
            f"{SOCKET_PATH_MAX} bytes): {path}; set {SOCKET_ENV} to a shorter one"
        )
    return path


def tilde(path: str | Path) -> str:
    """*path* with the home folder written as ``~``, for showing to a person."""
    text, home_dir = str(path), str(Path.home())
    if text == home_dir or text.startswith(home_dir + "/"):
        return "~" + text[len(home_dir) :]
    return text
