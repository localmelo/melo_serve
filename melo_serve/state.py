"""``state.json``: what melo_serve serves, for clients that read a file.

It is replaced whole on every change (written beside it, then renamed), so a
reader never sees half a file. It never holds a key. ``schema/state.v1.json``
in the repository describes it.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import json
import os
import tempfile
from pathlib import Path
from typing import Any

SCHEMA = "melo_serve.state/1"


def now_iso() -> str:
    return dt.datetime.now().astimezone().replace(microsecond=0).isoformat()


def write(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".state-", suffix=".json", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=1, sort_keys=False)
            handle.write("\n")
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def read(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or data.get("schema") != SCHEMA:
        return None
    return data
