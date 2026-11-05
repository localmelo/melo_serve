"""``melo-serve get``: download the models of some picks into the Hugging Face cache.

Files go to ``<hf_home>/hub`` and nowhere else (no ``local_dir``), where the
backends look for them. A pick that is already complete is skipped, so every run
is safe to repeat. Each outcome is one line from a closed set, and ``--json``
prints the same lines as JSON events (``item``, ``state``, ``done_bytes``,
``total_bytes``, ``line``) for a client such as localmelo.

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
import sys
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
PROGRESS_EVERY = 0.5  # seconds between two progress lines


class _Cancelled(BaseException):
    """Raised inside a download thread; a BaseException, so no retry loop eats it."""


def _gb(done: int, total: int) -> str:
    return f"{done / fit.GIB:.1f} / {total / fit.GIB:.1f} GB"


class _Out:
    """The lines of one run: plain text for a person, or JSON events."""

    def __init__(self, as_json: bool) -> None:
        self.as_json = as_json
        self.tty = not as_json and sys.stdout.isatty()
        self.open_line = False  # a progress line on a terminal, not yet ended

    def _print(self, text: str, end: str = "\n") -> None:
        if self.open_line:
            text, self.open_line = "\r" + text, False
        print(text, end=end, flush=True)

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
            print(json.dumps({**body, "line": line}), flush=True)
        else:
            self._print(text or line)

    def running(
        self, item: str, name: str, done: int, total: int, first: bool = False
    ) -> None:
        line = f"downloading {name}"
        if self.as_json:
            self.event(item, "running", line, done, total)
        elif self.tty:  # one line, rewritten in place
            self._print(f"{line}  {_gb(done, total)}   ", end="")
            self.open_line = True
        elif first:  # no terminal to rewrite: say it once
            self._print(f"{line} ({fit.size_text(total)})")


class _Stop:
    """SIGINT and SIGTERM stop a download; the files it finished stay."""

    def __init__(self) -> None:
        self.event = threading.Event()
        self.downloading = False

    def _handle(self, signum: int, frame: FrameType | None) -> None:
        self.event.set()
        if not self.downloading:  # nothing to wind down: leave at once
            raise _Cancelled

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


def _reason(exc: BaseException) -> str:
    text = " ".join(str(exc).split()) or type(exc).__name__
    return text if len(text) <= 200 else text[:197] + "..."


def _save_hf_home(out: _Out, saved: Settings, explicit: str) -> None:
    """Remember the cache folder in config.toml, as the daemon has no shell."""
    home = explicit or saved.resolved_hf_home()
    if saved.hf_home == home or (saved.hf_home and not explicit):
        return
    settings_module.save_hf_home(home)
    out.event(
        "config:hf_home", "done",
        f"saved hf_home = {paths.tilde(home)} in {paths.tilde(paths.config_path())}",
    )  # fmt: skip


def _fail(out: _Out, pick: Pick, why: str, kept: str = "nothing was changed") -> None:
    line = f"failed: {why}; {kept}"
    out.event(f"model:{pick.id}", "failed", line, text=f"{pick.name}: {line}")


def run(args: argparse.Namespace, settings: Settings) -> int:
    out = _Out(args.json)
    try:
        catalog = catalog_module.load()
        resolved = {pick.id: pick for pick in map(catalog.resolve, args.picks)}
        picks = list(resolved.values())
    except catalog_module.UnknownPick as exc:
        line = f"failed: {str(exc).rstrip('.')}; nothing was changed"
        out.event(f"model:{exc.name}", "failed", line)
        return CANNOT_START
    explicit = os.path.abspath(os.path.expanduser(args.hf_home)) if args.hf_home else ""
    cache = Path(explicit or settings.resolved_hf_home())
    wanted = os.environ.get("HF_HUB_CACHE")
    if wanted and Path(wanted).expanduser().resolve() != (cache / "hub").resolve():
        out.event(
            "cache:hf_home", "failed",
            f"failed: HF_HUB_CACHE is {wanted}, not {paths.tilde(cache / 'hub')}; "
            "unset it, or set HF_HOME to match; nothing was changed",
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
        # The hub reads HF_HOME when it is imported (its xet cache goes there too).
        os.environ["HF_HOME"] = used.hf_home
        return _get(out, picks, used, lambda: _save_hf_home(out, settings, explicit))
    finally:
        os.close(fd)  # drops the lock


def _get(
    out: _Out, picks: list[Pick], settings: Settings, save_config: Callable[[], None]
) -> int:
    stop = _Stop()
    hub = Path(settings.hf_home) / "hub"
    adapters = default_adapters()
    skipped: list[Pick] = []
    jobs: list[tuple[Pick, Any]] = []
    with stop.watching():
        try:
            for pick in picks:
                adapter = adapters[pick.backend]
                if not adapter.installed(pick, settings):
                    skipped.append(pick)
                elif pick.backend == "splash":
                    _fail(
                        out, pick, "Splash models come in a later version of melo-serve"
                    )
                elif pick.backend not in DOWNLOADS or pick.source.startswith(
                    ("/", "~", ".")
                ):
                    _fail(
                        out,
                        pick,
                        f"{pick.source} is not a Hugging Face repo to download",
                    )
                else:
                    jobs.append((pick, adapter))
            if len(skipped) + len(jobs) < len(picks):
                return CANNOT_START
            sizes: list[int] = []
            for pick, _ in jobs:
                try:
                    sizes.append(_hub_bytes(pick.source))
                except Exception as exc:
                    _fail(
                        out,
                        pick,
                        f"could not read the size of {pick.source}: {_reason(exc)}",
                    )
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
                    "free some space, or use --hf-home PATH; nothing was changed",
                )  # fmt: skip
                return CANNOT_START
            save_config()
            for pick in skipped:
                line = "already downloaded, skipped"
                out.event(
                    f"model:{pick.id}", "skipped", line, text=f"{pick.name}: {line}"
                )
            for (pick, adapter), size in zip(jobs, sizes, strict=True):
                code = _download(out, stop, pick, adapter, settings, size)
                if code:
                    return code
            return OK
        except _Cancelled:  # stopped before the first byte
            out.event(
                "get",
                "failed",
                "Stopped before the download began. Run the same command again to continue.",
            )
            return STOPPED


def _download(
    out: _Out, stop: _Stop, pick: Pick, adapter: Any, settings: Settings, total: int
) -> int:
    item = f"model:{pick.id}"

    def on_disk() -> int:  # files and partial files in the cache, resume included
        return min(total, adapter.cached_bytes(pick, settings))

    start, lock = on_disk(), threading.Lock()
    seen = {"at": 0.0, "bytes": start}

    def tick() -> None:  # runs in the hub's download threads
        if stop.event.is_set():
            raise _Cancelled
        now = time.monotonic()
        with lock:
            if now - seen["at"] < PROGRESS_EVERY:
                return
            seen["at"], done = now, on_disk()
            if done != seen["bytes"]:  # say nothing when nothing moved
                seen["bytes"] = done
                out.running(item, pick.name, done, total)

    from huggingface_hub import snapshot_download

    out.running(item, pick.name, start, total, first=True)
    stop.downloading = True
    try:
        snapshot_download(
            repo_id=pick.source,
            cache_dir=str(Path(settings.hf_home) / "hub"),
            tqdm_class=_progress_class(tick),
        )
    except _Cancelled:
        out.event(item, "failed", _stopped(on_disk(), total))
        return STOPPED
    except Exception as exc:
        kept = (
            "the files finished so far are kept; run the same command again to continue"
            if on_disk() > start
            else "nothing was changed"
        )
        _fail(out, pick, _reason(exc), kept)
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


def _stopped(done: int, total: int) -> str:
    return f"Stopped at {_gb(done, total)}. Run the same command again to continue."
