"""``melo-serve get``: set up some picks: install their backends, download their models.

A backend that is not on this Mac is installed first (``melo_serve.installer``: the
pinned oMLX DMG into ``~/Applications``, Splash through Homebrew); one that is there,
at any version, is left alone. Model files go to ``<hf_home>/hub`` (no ``local_dir``), where the backends look
for them; Xet, the hub's transfer layer, keeps a chunk cache in ``<hf_home>/xet``.
A pick that is already complete is skipped, so every run is safe to repeat. Each
outcome is one line from a closed set, and ``--json`` prints the same lines as JSON
events (``item``, ``state``, ``done_bytes``, ``total_bytes``, ``line``) for a client
such as localmelo. A refused run installs and downloads nothing, but melo_serve's own
folder and ``get.lock`` are made first. ``--skip ITEM`` leaves an item out.

Only oMLX picks are downloaded here; a Splash model comes in a later version.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import fcntl
import json
import os
import platform
import signal
import sys
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from types import FrameType
from typing import Any

from melo_serve import catalog as catalog_module
from melo_serve import fit, installer, paths
from melo_serve import plan as plan_module
from melo_serve import settings as settings_module
from melo_serve.adapters import Adapter, default_adapters
from melo_serve.adapters.base import Install, InstallPlan
from melo_serve.catalog import Pick
from melo_serve.settings import Settings

OK, FAILED, CANNOT_START, BUSY, STOPPED = 0, 1, 2, 75, 130
DOWNLOADS = ("omlx",)  # backends whose models this command downloads
PROGRESS_EVERY = (
    0.5  # seconds between two JSON progress events; text waits 10 times as long
)
NOTHING = "nothing was downloaded"
CHOSEN = "skipped (you chose to skip it)"
BETWEEN = "Stopped between steps. Run the same command again to continue."
KEPT = "finished files are kept; run the same command again to download the rest"


class _Cancelled(BaseException):
    """Stops the run; a BaseException, so no retry loop or ``except Exception`` eats it."""


def _shown(text: str) -> str:
    """*text* for a person: long, and with the home folder written as ``~``."""
    text = " ".join(text.replace(str(Path.home()), "~").split())
    return text if len(text) <= 200 else text[:197] + "..."


class _Out:
    """The lines of one run: plain text for a person, or JSON events."""

    def __init__(self, as_json: bool) -> None:
        self.as_json = as_json
        self.gone = False  # the reader of stdout has left
        self.every = PROGRESS_EVERY if as_json else 10 * PROGRESS_EVERY

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
            os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())

    def running(self, item: str, name: str, done: int, total: int) -> None:
        line = f"downloading {name}"
        text = f"{line}  {fit.size_text(done)} / {fit.size_text(total)}"
        self.event(item, "running", line, done, total, text)


class _Stop:
    """SIGINT, SIGTERM and SIGHUP stop the run; what it finished stays."""

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
            s: signal.signal(s, self._handle)
            for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
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


def _note(out: _Out, item: str, label: str, state: str, line: str) -> None:
    """One closing line; a terminal gets it after the name of what it is about."""
    out.event(item, state, line, text=f"{label}: {line}")


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
        except Exception as exc:  # a folder that cannot be made, a lock file, ...
            out.event("get", "failed", f"failed: {_shown(str(exc))}")
            return FAILED


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
    known = {f"model:{p.id}" for p in resolved.values()} | {
        f"backend:{p.backend}" for p in resolved.values()
    }
    unknown = [item for item in args.skip if item not in known]
    if unknown:
        line = f"failed: {_shown(unknown[0])} is not an item of this run; {NOTHING}"
        out.event("get", "failed", line)
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
        picks = list(resolved.values())
        return _get(out, stop, picks, used, save, frozenset(args.skip))
    finally:
        os.close(fd)  # drops the lock


def _get(
    out: _Out,
    stop: _Stop,
    picks: list[Pick],
    settings: Settings,
    save_config: Callable[[], None],
    skip: frozenset[str],
) -> int:
    hub = Path(settings.hf_home) / "hub"
    adapters = default_adapters()
    skipped: list[tuple[Pick, str]] = []
    jobs: list[tuple[Pick, Any]] = []
    for pick in picks:
        adapter = adapters.get(pick.backend)
        if adapter is None:
            _fail(out, pick, f"there is no backend called {pick.backend}")
        elif f"model:{pick.id}" in skip:
            skipped.append((pick, CHOSEN))
        elif not adapter.installed(pick, settings):
            skipped.append((pick, "already downloaded, skipped"))
        elif pick.backend == "splash":
            _fail(out, pick, "Splash models come in a later version of melo-serve")
        elif pick.backend not in DOWNLOADS or pick.is_local:
            _fail(out, pick, f"{pick.source} is not a Hugging Face repo to download")
        else:
            jobs.append((pick, adapter))
    if len(skipped) + len(jobs) < len(picks):
        return CANNOT_START
    backends = _backends_to_install(out, adapters, picks, settings, skip)
    if backends is None:
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
    wants = [(hub, need)] if jobs else []
    wants += [
        (paths.downloads_dir(), installer.disk_need(how))
        for _, how in backends.values()
        if how and how.via == "dmg"
    ]
    short = _short_of_space(wants)
    if short:
        folder, need, free = short
        out.event(
            "disk", "failed",
            f"failed: not enough space: needs {fit.size_text(need)} and 5 GB to spare, "
            f"{fit.size_text(free)} free in {paths.tilde(folder)}; "
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
    for key, (found, how) in backends.items():
        code = _backend(out, stop, adapters[key], found, how, settings)
        if code:
            return code
        stop.line = lambda: BETWEEN
    for pick, line in skipped:
        _note(out, f"model:{pick.id}", pick.name, "skipped", line)
    for (pick, adapter), size in zip(jobs, sizes, strict=True):
        code = _download(out, stop, pick, adapter, settings, size)
        if code:
            return code
        stop.line = lambda: f"Stopped between models. The {KEPT}."
    return OK


def _short_of_space(wants: list[tuple[Path, int]]) -> tuple[Path, int, int] | None:
    """The first folder whose volume lacks room for what it is to hold, plus 5 GiB.

    Folders on one volume are added up: the cache and the DMG may share it.
    """
    volumes: dict[int, tuple[Path, int]] = {}
    for folder, size in wants:
        device = os.stat(fit.existing_parent(folder)).st_dev
        first, total = volumes.get(device, (folder, 0))
        volumes[device] = (first, total + size)
    for folder, total in volumes.values():
        free = fit.free_bytes(folder)
        if free < total + fit.MARGIN_BYTES:
            return folder, total, free
    return None


def _backends_to_install(
    out: _Out,
    adapters: dict[str, Adapter],
    picks: list[Pick],
    settings: Settings,
    skip: frozenset[str],
) -> dict[str, tuple[Install | None, InstallPlan | None]] | None:
    """The picks' backends in order, each with what was found and, when it is missing,
    how to install it; the found part is None for one the person chose to skip.

    Nothing is installed yet: None comes back when one cannot be, so a run that would
    stop halfway is refused before it starts.
    """
    macos = plan_module.version_tuple(plan_module.macos_version())
    backends: dict[str, tuple[Install | None, InstallPlan | None]] = {}
    refused = False
    for key in dict.fromkeys(p.backend for p in picks):
        adapter = adapters[key]
        found = None if f"backend:{key}" in skip else adapter.detect(settings)
        if found is None or found.found:
            backends[key] = (found, None)
            continue
        how = adapter.install_plan(macos)
        command = settings.backend(key).command
        if command:
            why = f"the command in config.toml ({_shown(command[0])}) was not found; fix or remove it"
        elif platform.machine() != "arm64":
            why = f"{adapter.name} needs Apple silicon"
        elif how is None:
            why = f"{adapter.name} does not run on macOS {plan_module.macos_version() or '?'}"
        else:
            why = installer.problem(how, adapter.name)
        if why:
            line = f"failed: {why}; {installer.NOT_INSTALLED}"
            _note(out, f"backend:{key}", adapter.name, "failed", line)
            refused = True
        backends[key] = (found, how)
    return None if refused else backends


def _backend(
    out: _Out,
    stop: _Stop,
    adapter: Adapter,
    found: Install | None,
    how: InstallPlan | None,
    settings: Settings,
) -> int:
    """Install one backend if *how* says to, and say what happened."""
    item = f"backend:{adapter.key}"

    def say(
        state: str, line: str, done: int | None = None, total: int | None = None
    ) -> None:
        out.event(item, state, line, done, total, text=f"{adapter.name}: {line}")

    if found is None:
        say("skipped", CHOSEN)
        return OK
    if how is None:  # it was there before the run began
        if not found.version:
            say("skipped", "already installed (it did not say its version), skipped")
        elif found.tested:
            say("skipped", f"already installed ({found.version}), skipped")
        else:
            say(
                "skipped", f"installed ({found.version}, not a tested version), skipped"
            )
        return OK
    name = f"{adapter.name} {how.version}"
    stop.line = lambda: (
        f"Stopped while installing {adapter.name}. Run the same command again to continue."
    )
    last = [0.0]

    def progress(done: int, total: int) -> None:
        if out.gone:
            raise _Cancelled
        now = time.monotonic()
        if done >= total or now - last[0] >= out.every:
            last[0] = now
            out.running(item, name, done, total)

    try:
        note = lambda text: say("running", _shown(text))  # noqa: E731
        if how.via == "dmg":
            installer.install_app(how, progress, note)
        else:
            installer.install_brew(how, note)
        found = adapter.detect(settings)
        if not found.version:
            raise installer.InstallError(
                f"{adapter.name} was installed but does not run: "
                f"{found.problem or 'it did not say its version'}",
                "run melo-serve doctor",
            )
        if not found.tested and how.via == "dmg":
            raise installer.InstallError(
                f"{adapter.name} says it is version {found.version}, not {how.version}",
                f"it was left in {how.dest}",
            )
    except (installer.InstallError, OSError) as exc:  # OSError: a folder, a full disk
        after = getattr(exc, "after", installer.NOT_INSTALLED)
        say("failed", f"failed: {_shown(str(exc))}; {after}")
        return FAILED
    untested = "" if found.tested else " (not a tested version)"
    say("done", f"installed {found.version}{untested} into {how.dest}")
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

    def tick() -> None:  # runs in the hub's download threads
        if stop.event.is_set() or out.gone:
            raise _Cancelled
        now = time.monotonic()
        with lock:
            if now - seen["at"] < out.every:
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
