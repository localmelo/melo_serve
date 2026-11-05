"""``melo-serve get``: download the models of some picks into the Hugging Face cache.

Model files go to ``<hf_home>/hub`` (no ``local_dir``), where the backends look
for them; Xet, the hub's transfer layer, keeps a chunk cache in ``<hf_home>/xet``.
A pick that is already complete is skipped, so every run is safe to repeat. Each
outcome is one line from a closed set, and ``--json`` prints the same lines as JSON
events (``item``, ``state``, ``done_bytes``, ``total_bytes``, ``line``) for a client
such as localmelo. A refused or failed run downloads nothing, but melo_serve's own
folder and ``get.lock`` are made first.

Only oMLX picks are downloaded here; a Splash model comes in a later version.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import fcntl
import json
import os
import signal
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from types import FrameType
from typing import Any

from melo_serve import catalog as catalog_module
from melo_serve import fit, paths
from melo_serve import settings as settings_module
from melo_serve.adapters import default_adapters
from melo_serve.catalog import Pick
from melo_serve.settings import Settings

OK, FAILED, CANNOT_START, BUSY, STOPPED = 0, 1, 2, 75, 130
DOWNLOADS = ("omlx",)  # backends whose models this command downloads
PROGRESS_EVERY = (
    0.5  # seconds between two JSON progress events; text waits 10 times as long
)
NOTHING = "nothing was downloaded"
KEPT = "finished files are kept; run the same command again to download the rest"


class _Cancelled(BaseException):
    """Raised inside a download thread; a BaseException, so no retry loop eats it."""


def _shown(text: str) -> str:
    """*text* for a person: long, and with the home folder written as ``~``."""
    text = " ".join(text.replace(str(Path.home()), "~").split())
    return text if len(text) <= 200 else text[:197] + "..."


class _Out:
    """The lines of one run: plain text for a person, or JSON events."""

    def __init__(self, as_json: bool) -> None:
        self.as_json = as_json
        self.gone = False  # the reader of stdout has left

    def event(
        self,
        item: str,
        state: str,
        line: str,
        done: int | None = None,
        total: int | None = None,
        text: str = "",
    ) -> None:
        if self.as_json:
            body: dict[str, Any] = {"item": item, "state": state}
            if total is not None:
                body.update(done_bytes=done or 0, total_bytes=total)
            shown = json.dumps({**body, "line": line})
        else:
            shown = text or line
        try:
            print(shown, flush=True)
        except BrokenPipeError:  # the caller closed the pipe: stop, say nothing
            self.gone = True

    def running(self, item: str, name: str, done: int, total: int) -> None:
        line = f"downloading {name}"
        text = f"{line}  {fit.size_text(done)} / {fit.size_text(total)}"
        self.event(item, "running", line, done, total, text)


class _Stop:
    """SIGINT and SIGTERM stop a download; the files it finished stay."""

    def __init__(self, out: _Out) -> None:
        self.out = out
        self.event = threading.Event()
        self.downloading = False
        self.line: Callable[[], str] = lambda: (
            "Stopped before the download began. Run the same command again to start it."
        )

    def _handle(self, signum: int, frame: FrameType | None) -> None:
        if self.event.is_set():  # the second one: leave at once
            self.out.event("get", "failed", self.line())
            os._exit(STOPPED)
        self.event.set()
        if not self.downloading:  # nothing to wind down: leave at once
            raise _Cancelled
        # hf_xet's own threads swallow what a progress bar raises: ask the hub to
        # abort its Xet session, as its own downloader does on Ctrl-C.
        with contextlib.suppress(Exception):
            from huggingface_hub.utils._xet import abort_xet_session

            abort_xet_session()

    @contextlib.contextmanager
    def watching(self) -> Iterator[None]:
        old = {
            s: signal.signal(s, self._handle) for s in (signal.SIGINT, signal.SIGTERM)
        }
        try:
            yield
        finally:
            for sig, handler in old.items():
                signal.signal(sig, handler)


def _lock_busy(fd: int) -> str | None:
    """Take the lock on *fd*; the other process's pid text if it is held."""
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return os.pread(fd, 32, 0).decode(errors="replace").strip() or "unknown"
    os.ftruncate(fd, 0)
    os.pwrite(fd, str(os.getpid()).encode(), 0)
    return None


def _progress_class(tick: Callable[[], None]) -> type:
    """A tqdm that draws nothing; each update ticks (progress, and the stop check)."""
    from huggingface_hub.utils import tqdm

    class Ticker(tqdm):  # type: ignore[misc]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **{**kwargs, "disable": True})

        def update(self, n: float | None = 1) -> bool | None:
            tick()
            return super().update(n)

    return Ticker


def _hub_bytes(repo: str) -> int:
    """Exact bytes of every file of *repo* on the Hub."""
    from huggingface_hub import HfApi

    info = HfApi().model_info(repo, files_metadata=True)
    return sum(f.size or 0 for f in info.siblings or [])


def _save_hf_home(out: _Out, saved: Settings, explicit: str) -> None:
    """Remember the cache folder in config.toml, as the daemon has no shell."""
    if saved.hf_home and (not explicit or saved.resolved_hf_home() == explicit):
        return
    home = explicit or saved.resolved_hf_home()
    settings_module.save_hf_home(home)
    out.event(
        "config:hf_home", "done",
        f"saved hf_home = {paths.tilde(home)} in {paths.tilde(paths.config_path())}",
    )  # fmt: skip


def _fail(out: _Out, pick: Pick, why: str, kept: str = NOTHING) -> None:
    line = f"failed: {_shown(why)}; {kept}"
    out.event(f"model:{pick.id}", "failed", line, text=f"{pick.name}: {line}")


def run(args: argparse.Namespace, settings: Settings) -> int:
    out = _Out(args.json)
    stop = _Stop(out)
    with stop.watching():  # first, so a stop at any moment gets its line
        try:
            return _run(out, stop, args, settings)
        except _Cancelled:
            out.event("get", "failed", stop.line())
            return STOPPED


def _run(out: _Out, stop: _Stop, args: argparse.Namespace, settings: Settings) -> int:
    try:
        catalog = catalog_module.load()
        resolved = {pick.id: pick for pick in map(catalog.resolve, args.picks)}
    except (catalog_module.UnknownPick, catalog_module.CatalogError) as exc:
        item = f"model:{getattr(exc, 'name', 'catalog')}"
        out.event(item, "failed", f"failed: {_shown(str(exc).rstrip('.'))}; {NOTHING}")
        return CANNOT_START
    try:
        explicit = settings_module.absolute_path(args.hf_home) if args.hf_home else ""
        cache = Path(explicit or settings.resolved_hf_home())
    except settings_module.SettingsError as exc:
        out.event("cache:hf_home", "failed", f"failed: {_shown(str(exc))}; {NOTHING}")
        return CANNOT_START
    wanted = os.environ.get("HF_HUB_CACHE")
    if wanted and os.path.realpath(os.path.expanduser(wanted)) != str(
        (cache / "hub").resolve()
    ):
        out.event(
            "cache:hf_home", "failed",
            f"failed: HF_HUB_CACHE is {paths.tilde(wanted)}, not {paths.tilde(cache / 'hub')}; "
            f"unset it, or set HF_HOME to match; {NOTHING}",
        )  # fmt: skip
        return CANNOT_START
    paths.ensure_home()
    fd = os.open(paths.get_lock_path(), os.O_RDWR | os.O_CREAT, 0o600)
    try:
        busy = _lock_busy(fd)
        if busy is not None:
            out.event("get", "failed", f"another download is running (pid {busy})")
            return BUSY
        used = dataclasses.replace(settings, hf_home=str(cache))
        # The hub reads these when it is imported. HF_HOME also moves its Xet cache;
        # without the telemetry switch it names the caller's agent in a request header
        # and writes a dotfile into the cache.
        os.environ["HF_HOME"] = used.hf_home
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        save = lambda: _save_hf_home(out, settings, explicit)  # noqa: E731
        return _get(out, stop, list(resolved.values()), used, save)
    finally:
        os.close(fd)  # drops the lock


def _get(
    out: _Out,
    stop: _Stop,
    picks: list[Pick],
    settings: Settings,
    save_config: Callable[[], None],
) -> int:
    hub = Path(settings.hf_home) / "hub"
    adapters = default_adapters()
    skipped: list[Pick] = []
    jobs: list[tuple[Pick, Any]] = []
    for pick in picks:
        adapter = adapters.get(pick.backend)
        if adapter is None:
            _fail(out, pick, f"there is no backend called {pick.backend}")
        elif not adapter.installed(pick, settings):
            skipped.append(pick)
        elif pick.backend == "splash":
            _fail(out, pick, "Splash models come in a later version of melo-serve")
        elif pick.backend not in DOWNLOADS or pick.is_local:
            _fail(out, pick, f"{pick.source} is not a Hugging Face repo to download")
        else:
            jobs.append((pick, adapter))
    if len(skipped) + len(jobs) < len(picks):
        return CANNOT_START
    sizes: list[int] = []
    for pick, _ in jobs:
        try:
            sizes.append(_hub_bytes(pick.source))
        except Exception as exc:
            _fail(out, pick, f"could not read the size of {pick.source}: {exc}")
            return FAILED
    need = sum(
        max(0, size - adapter.cached_bytes(pick, settings))
        for (pick, adapter), size in zip(jobs, sizes, strict=True)
    )
    free = fit.free_bytes(hub)
    if jobs and free < need + fit.MARGIN_BYTES:
        out.event(
            "disk", "failed",
            f"failed: not enough space: needs {fit.size_text(need)} and 5 GB to spare, "
            f"{fit.size_text(free)} free in {paths.tilde(hub)}; "
            f"free some space, or use --hf-home PATH; {NOTHING}",
        )  # fmt: skip
        return CANNOT_START
    try:
        save_config()
    except (
        OSError,
        ValueError,
    ) as exc:  # a file that cannot be written, or edited safely
        line = f"failed: could not save hf_home: {_shown(str(exc))}; {NOTHING}"
        out.event("config:hf_home", "failed", line)
        return FAILED if isinstance(exc, OSError) else CANNOT_START
    for pick in skipped:
        line = "already downloaded, skipped"
        out.event(f"model:{pick.id}", "skipped", line, text=f"{pick.name}: {line}")
    for (pick, adapter), size in zip(jobs, sizes, strict=True):
        code = _download(out, stop, pick, adapter, settings, size)
        if code:
            return code
        stop.line = lambda: f"Stopped between models. The {KEPT}."
    return OK


def _download(
    out: _Out, stop: _Stop, pick: Pick, adapter: Any, settings: Settings, total: int
) -> int:
    item = f"model:{pick.id}"

    def finished() -> int:
        return min(total, adapter.cached_bytes(pick, settings))

    def on_disk() -> int:  # finished files and the one being written
        return min(total, adapter.cached_bytes(pick, settings, include_partial=True))

    def stopped() -> str:
        return f"Stopped at {fit.size_text(finished())} / {fit.size_text(total)}. The {KEPT}."

    start, lock = finished(), threading.Lock()
    seen = {"at": 0.0, "bytes": start}
    every = PROGRESS_EVERY if out.as_json else 10 * PROGRESS_EVERY

    def tick() -> None:  # runs in the hub's download threads
        if stop.event.is_set() or out.gone:
            raise _Cancelled
        now = time.monotonic()
        with lock:
            if now - seen["at"] < every:
                return
            seen["at"], done = now, on_disk()
            if done != seen["bytes"]:  # say nothing when nothing moved
                seen["bytes"] = done
                out.running(item, pick.name, done, total)

    from huggingface_hub import snapshot_download

    out.running(item, pick.name, start, total)
    stop.line = stopped
    stop.downloading = True
    try:
        snapshot_download(
            repo_id=pick.source,
            cache_dir=str(Path(settings.hf_home) / "hub"),
            tqdm_class=_progress_class(tick),
        )
    except _Cancelled:
        out.event(item, "failed", stopped())
        return STOPPED
    except Exception as exc:
        _fail(out, pick, str(exc), f"the {KEPT}" if finished() > start else NOTHING)
        return FAILED
    finally:
        stop.downloading = False
    problem = adapter.installed(pick, settings)
    if problem:
        _fail(
            out, pick, f"the download ended but {problem}", "run the same command again"
        )
        return FAILED
    line = f"downloaded {pick.name} into {paths.tilde(settings.hf_home)}"
    out.event(item, "done", line, total, total)
    return OK
