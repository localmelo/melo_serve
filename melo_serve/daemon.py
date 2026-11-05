"""The daemon: the front door and the control socket, in one process.

One daemon runs per user (a lock file says so). SIGTERM or SIGINT, or ``POST
/stop`` on the control socket, stop every backend and then the daemon, which
exits 0, so launchd restarts it only after a crash.
"""

from __future__ import annotations

import asyncio
import fcntl
import logging
import os
import signal
import socket
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from aiohttp import web

from melo_serve import catalog as catalog_module
from melo_serve import control, fit, launchd, paths, router, state
from melo_serve.adapters import Adapter, default_adapters
from melo_serve.catalog import Catalog
from melo_serve.engine import Engine, EngineError
from melo_serve.settings import Settings

log = logging.getLogger("melo_serve")


def _lock(path: Path) -> int | None:
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(fd)
        return None
    os.ftruncate(fd, 0)
    os.write(fd, f"{os.getpid()}\n".encode())
    return fd


async def _front_socket(settings: Settings) -> socket.socket:
    """The front door's socket; under launchd a busy port is retried, not fatal."""
    family = socket.AF_INET6 if ":" in settings.host else socket.AF_INET
    while True:
        sock = socket.socket(family, socket.SOCK_STREAM)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((settings.host, settings.port))
            sock.listen(256)
            sock.setblocking(False)
            return sock
        except OSError:
            sock.close()
            if not os.environ.get(launchd.ENV):
                raise
            log.warning("port %d is in use; trying again in 30 s", settings.port)
            await asyncio.sleep(30)


def _control_socket(path: Path) -> socket.socket:
    path.unlink(missing_ok=True)  # a stale file: the lock says no daemon owns it
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.bind(str(path))
    os.chmod(path, 0o600)
    sock.listen(64)
    sock.setblocking(False)
    return sock


async def _serve(
    app: web.Application, sock: socket.socket, **options: Any
) -> web.AppRunner:
    """Serves *app* on *sock*, bound already; signals stay with the daemon."""
    runner = web.AppRunner(app, access_log=None, shutdown_timeout=5.0, **options)
    await runner.setup()
    await web.SockSite(runner, sock).start()
    return runner


async def run(
    settings: Settings,
    *,
    pick: str | None = None,
    catalog: Catalog | None = None,
    adapters: Mapping[str, Adapter] | None = None,
    memory: Callable[[], fit.Memory | None] = fit.probe,
    ready: Callable[[Engine], None] | None = None,
) -> int:
    """Serve until stopped; *pick* is switched to once the daemon is up."""
    if not logging.getLogger().handlers:
        logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s")
    log.setLevel(logging.INFO)  # melo_serve's own lines, not one per proxied request
    paths.ensure_home()
    lock = _lock(paths.lock_path())
    if lock is None:
        print("melo_serve is already running", file=sys.stderr)
        return 2
    engine = Engine(
        settings,
        catalog or catalog_module.load(),
        adapters or default_adapters(),
        memory=memory,
    )
    try:
        front = await _front_socket(settings)
    except OSError as exc:
        print(
            f"melo_serve: cannot listen on {settings.host}:{settings.port}: {exc.strerror}",
            file=sys.stderr,
        )
        os.close(lock)
        await engine.aclose()
        return 1
    engine.port = front.getsockname()[1]
    engine.login_service = bool(os.environ.get(launchd.ENV))
    saved = state.read(paths.state_path())
    await engine.clean_orphans(saved)
    engine.restore(saved)
    control_path = paths.socket_path()
    stopping = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stopping.set)
    runners = [
        # A client that leaves cancels its request, so the backend stops writing it.
        await _serve(router.build(engine), front, handler_cancellation=True),
        # A switch goes on when the command that asked for it quits.
        await _serve(
            control.build(engine, stopping.set), _control_socket(control_path)
        ),
    ]
    idle = asyncio.create_task(engine.idle_loop())
    engine.changed()
    log.info(
        "melo_serve on http://%s:%d/v1, control %s",
        settings.host,
        engine.port,
        control_path,
    )
    if ready is not None:
        ready(engine)
    first = asyncio.create_task(_serve_first(engine, pick)) if pick else None
    await stopping.wait()
    log.info("melo_serve is stopping")
    await engine.stop_all()
    for runner in runners:
        await runner.cleanup()
    idle.cancel()
    if first is not None:
        first.cancel()
    await engine.aclose()
    engine.changed()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.remove_signal_handler(sig)
    control_path.unlink(missing_ok=True)
    os.close(lock)
    return 0


async def _serve_first(engine: Engine, pick: str) -> None:
    try:
        await engine.switch(pick, initiator="script", reason="melo-serve up")
    except EngineError as exc:
        log.error("could not serve %s: %s", pick, exc)
