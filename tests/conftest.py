"""Shared test setup: melo_serve's folder in a temp dir, and the fake backends.

No test loads a model or binds a fixed port: the fakes and the front door take
ports the system picks.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import shutil
import sys
import tempfile
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from melo_serve import daemon, fit, settings
from melo_serve.catalog import Catalog, pick_from_dict
from melo_serve.client import Client
from melo_serve.engine import Engine

FAKE = Path(__file__).parent / "fakes" / "fake_backend.py"
MEMORY = fit.Memory(total_bytes=64 * fit.GIB, free_pct=80.0)


@pytest.fixture(autouse=True)
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    monkeypatch.setenv("MELO_SERVE_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("MELO_SERVE_LAUNCH_AGENTS", str(tmp_path / "agents"))
    short = tempfile.mkdtemp(prefix="ms-")  # macOS caps a socket path at 104 bytes
    monkeypatch.setenv("MELO_SERVE_SOCKET", f"{short}/c.sock")
    monkeypatch.delenv("MELO_SERVE_LAUNCHD", raising=False)
    yield tmp_path
    shutil.rmtree(short, ignore_errors=True)


def fake_settings(tmp_path: Path, **extra: Any) -> settings.Settings:
    fake = [sys.executable, str(FAKE), "--kind"]
    return settings.from_dict(
        {
            "port": 0,
            "idle_release": "off",
            "drain": "10s",
            "start_timeout": "20s",
            "picks": {"chat": "chat-a", "embedding": "embed-a"},
            "backends": {
                "splash": {
                    "command": [*fake, "splash"],
                    "models_dir": str(tmp_path / "splash"),
                },
                "omlx": {"command": [*fake, "omlx"]},
            },
            **extra,
        }
    )


def fake_catalog(tmp_path: Path, record: Path) -> Catalog:
    """Two Splash chat picks, one that cannot start, one too big, an oMLX embedder."""
    env = {"FAKE_RECORD": str(record)}
    entries = [
        {
            "id": "chat-a",
            "aliases": ["a"],
            "env": {**env, "FAKE_CHUNKS": "8", "FAKE_CHUNK_S": "0.15"},
        },
        {"id": "chat-b", "aliases": ["b"], "env": env},
        {"id": "broken", "env": {**env, "FAKE_FAIL": "start"}},
        {"id": "huge", "mem_gb": 500},
    ]
    picks = []
    for entry in entries:
        source = f"org/{entry['id']}:Q4"
        (tmp_path / "splash" / source).mkdir(parents=True)
        picks.append(
            pick_from_dict(
                {"role": "chat", "backend": "splash", "source": source, "mem_gb": 4,
                 "measured_ram_gb": 64, **entry}
            )
        )  # fmt: skip
    model = tmp_path / "models" / "embed-a"
    model.mkdir(parents=True)
    (model / "config.json").write_text("{}")
    (model / "model.safetensors").write_bytes(b"")
    picks.append(
        pick_from_dict(
            {"id": "embed-a", "role": "embedding", "backend": "omlx", "source": str(model),
             "mem_gb": 1, "measured_ram_gb": 64, "env": env}
        )
    )  # fmt: skip
    return Catalog(picks)


def records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()]


@contextlib.asynccontextmanager
async def running(
    tmp_path: Path, **extra: Any
) -> AsyncIterator[tuple[Engine, httpx.AsyncClient]]:
    """A daemon on a free port with the fakes, and an HTTP client for its front door."""
    started: asyncio.Future[Engine] = asyncio.get_running_loop().create_future()
    task = asyncio.create_task(
        daemon.run(
            fake_settings(tmp_path, **extra),
            catalog=fake_catalog(tmp_path, tmp_path / "record.jsonl"),
            memory=lambda: MEMORY,
            ready=started.set_result,
        )
    )
    engine = await asyncio.wait_for(started, 10)
    base = f"http://127.0.0.1:{engine.port}"
    try:
        async with httpx.AsyncClient(
            base_url=base, timeout=30, trust_env=False
        ) as client:
            yield engine, client
    finally:
        if not task.done():
            await asyncio.to_thread(Client().stop)
        assert await asyncio.wait_for(task, 60) == 0
