"""What a backend adapter tells melo_serve: how to start, check and free one model.

An adapter is data: its command, how to install it, the path that says it is
ready and the path that frees a model's memory; the engine makes the calls.
``launch()`` returns the command line and environment without running anything,
so tests check it as is.
"""

from __future__ import annotations

import abc
import re
import shutil
import signal
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from melo_serve.catalog import Pick
from melo_serve.settings import Settings

_VERSION_RE = re.compile(r"(\d+\.\d+(?:\.\d+)?[\w.+-]*)")


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


class Adapter(abc.ABC):
    key = ""
    name = ""
    default_command: tuple[str, ...] = ()
    install_hint = ""
    tested_versions: tuple[str, ...] = ()
    ready_path = "/health"  # answers 200 once requests are served

    def command(self, settings: Settings) -> tuple[str, ...]:
        return settings.backend(self.key).command or self.default_command

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

    def cached_bytes(
        self, pick: Pick, settings: Settings, include_partial: bool = False
    ) -> int:
        """Bytes of *pick*'s model already downloaded; 0 if unknown.

        Finished files only, unless *include_partial*: a file still downloading."""
        return 0

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
