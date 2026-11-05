"""What a backend adapter tells melo_serve: how to start, check and free one model.

An adapter is data: its command, how to install it, the path that says it is
ready and the path that frees a model's memory; the engine makes the calls.
``launch()`` returns the command line and environment without running anything,
so tests check it as is.
"""

from __future__ import annotations

import abc
import contextlib
import os
import re
import shutil
import signal
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from melo_serve.catalog import HubRepo, Pick
from melo_serve.settings import Settings

_VERSION_RE = re.compile(r"(\d+\.\d+(?:\.\d+)?[\w.+-]*)")


def hf_store(repo: str, hf_home: str) -> Path:
    return Path(hf_home) / "hub" / ("models--" + repo.replace("/", "--"))


def hf_snapshot(repo: str, hf_home: str) -> Path | None:
    """The local snapshot of a Hugging Face repo, found without the network."""
    store = hf_store(repo, hf_home)
    try:
        revision = (store / "refs" / "main").read_text(encoding="utf-8").strip()
    except OSError:
        revision = ""
    if revision and (store / "snapshots" / revision).is_dir():
        return store / "snapshots" / revision
    snapshots = sorted((store / "snapshots").glob("*")) if store.is_dir() else []
    return snapshots[-1] if snapshots else None


def snapshot_files(repo: HubRepo, hf_home: str) -> dict[str, Path] | None:
    """The files of *repo*'s local snapshot that its pick needs, by name in the repo
    (``IQ3_XXS/x.gguf``); None without a snapshot. A link to a missing blob is listed."""
    snapshot = hf_snapshot(repo.repo, hf_home)
    if snapshot is None:
        return None
    found = {p.relative_to(snapshot).as_posix(): p for p in snapshot.rglob("*")}
    return {n: p for n, p in found.items() if repo.matches(n) and not p.is_dir()}


def _repo_bytes(repo: HubRepo, hf_home: str, include_partial: bool) -> int:
    total = 0
    if repo.files:  # the repo may hold other files that are not this pick's
        for path in (snapshot_files(repo, hf_home) or {}).values():
            with contextlib.suppress(OSError):  # a link to a missing blob
                total += path.stat().st_size
    try:
        for blob in (hf_store(repo.repo, hf_home) / "blobs").iterdir():
            partial = blob.suffix == ".incomplete"
            if blob.is_file() and (include_partial if partial else not repo.files):
                total += blob.stat().st_size
    except OSError:
        pass
    return total


@dataclass(frozen=True)
class Launch:
    argv: tuple[str, ...]
    env: Mapping[str, str] = field(default_factory=dict)
    stop_signal: int = signal.SIGINT


@dataclass(frozen=True)
class Install:
    backend: str
    name: str
    found: bool
    version: str = ""
    path: str = ""
    problem: str = ""
    hint: str = ""
    tested: bool = True


@dataclass(frozen=True)
class InstallPlan:
    """How to put a backend on a Mac, as data; ``melo-serve plan`` shows it."""

    via: str  # "homebrew" or "dmg"
    version: str
    size_bytes: int
    dest: str
    needs_homebrew: bool = False
    url: str = ""  # a DMG's download address
    file: str = ""  # a DMG's name
    sha256: str = ""  # a DMG's pinned checksum
    formula: str = ""  # a Homebrew formula


class Adapter(abc.ABC):
    key = ""
    name = ""
    default_command: tuple[str, ...] = ()
    known_paths: tuple[str, ...] = ()  # where it lives when PATH does not reach it
    install_hint = ""
    tested_versions: tuple[str, ...] = ()
    ready_path = "/health"  # answers 200 once requests are served

    def command(self, settings: Settings) -> tuple[str, ...]:
        configured = settings.backend(self.key).command
        if configured:
            return configured
        if shutil.which(self.default_command[0]) is None:
            for known in map(os.path.expanduser, self.known_paths):
                if os.access(known, os.X_OK):
                    return (known,)
        return self.default_command

    def release_path(self, pick: Pick) -> str:
        """The POST that frees *pick*'s memory and keeps the process; "" for none."""
        return ""

    def detect(self, settings: Settings) -> Install:
        """Whether the backend is installed, and its version; no network."""
        command = self.command(settings)
        exe = shutil.which(command[0]) if command else None
        if exe is None:
            missing = command[0] if command else self.key
            return Install(
                self.key,
                self.name,
                False,
                problem=f"{missing} is not installed",
                hint=self.install_hint,
            )
        try:
            out = subprocess.run(
                [*command, "--version"],
                capture_output=True,
                text=True,
                errors="replace",
                timeout=30,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return Install(self.key, self.name, True, path=exe, problem=str(exc))
        match = _VERSION_RE.search(out.stdout or out.stderr)
        version = match.group(1) if match else ""
        tested = not self.tested_versions or version.startswith(self.tested_versions)
        return Install(self.key, self.name, True, version, exe, tested=tested)

    @abc.abstractmethod
    def install_plan(self, macos: tuple[int, ...]) -> InstallPlan | None:
        """How to install the backend on this macOS; None when it cannot run on it.

        Both backends need Apple silicon, which ``melo-serve plan`` checks once.
        """

    def hub_files(self, pick: Pick) -> tuple[HubRepo, ...]:
        """What ``melo-serve get`` downloads for *pick*: none for a model it cannot fetch."""
        return pick.hub

    def cached_bytes(
        self, pick: Pick, settings: Settings, include_partial: bool = False
    ) -> int:
        """Bytes of *pick*'s files already in the cache; 0 if unknown.

        Of a repo with a file list, only the files that are in it; else every finished
        blob. Finished files only, unless *include_partial*: a file still downloading
        (the hub never reuses a cut-short ``.incomplete``)."""
        home = settings.resolved_hf_home()
        return sum(_repo_bytes(r, home, include_partial) for r in self.hub_files(pick))

    def auth(self, key: str) -> dict[str, str]:
        return {"Authorization": f"Bearer {key}"} if key else {}

    @abc.abstractmethod
    def installed(self, pick: Pick, settings: Settings) -> str:
        """``""`` when *pick*'s model is on this Mac, else why it cannot load."""

    @abc.abstractmethod
    def launch(
        self, pick: Pick, *, port: int, key: str, workdir: Path, settings: Settings
    ) -> Launch:
        """The process that serves *pick* on 127.0.0.1:*port* and requires *key*."""
