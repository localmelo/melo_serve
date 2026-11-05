"""melo_serve's own settings: ``config.toml`` in its folder, then ``-X key=value``.

Every key is optional::

    port = 8321                 # the front door, on 127.0.0.1
    hf_home = "~/models/hf"     # else $HF_HOME, else ~/.cache/huggingface
    idle_release = "10m"        # free a model's memory after this long unused; "off"
    floor_pct = 30              # free memory that other heavy work needs
    min_pct = 5                 # never load below this much free memory
    drain = "2m"                # how long a switch waits for requests in flight
    start_timeout = "20m"       # how long a model may take to load

    [picks]                     # what each role serves, by pick id or alias
    chat = "27b"
    embedding = "embed"

    [backends.omlx]             # command, env, models_dir (Splash)
    command = "/Applications/oMLX.app/Contents/MacOS/omlx-cli"

    [hooks]                     # shell commands; see melo_serve.hooks
    before_load = "~/bin/book-gpu"
    after_stop = "~/bin/unbook-gpu"

launchd starts the service without a shell environment, so a path the service
needs (``hf_home``, a backend's ``command``) belongs here.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from melo_serve import paths

DEFAULT_PORT = 8321
LOOPBACK = ("127.0.0.1", "localhost", "::1")


class SettingsError(ValueError):
    """A setting has a value melo_serve cannot use; the message says which."""


@dataclass(frozen=True)
class BackendSettings:
    command: tuple[str, ...] = ()
    env: Mapping[str, str] = field(default_factory=dict)
    models_dir: str = ""


@dataclass(frozen=True)
class Hooks:
    before_load: str = ""
    after_stop: str = ""
    timeout_s: float = 1800.0


@dataclass(frozen=True)
class Settings:
    host: str = "127.0.0.1"
    port: int = DEFAULT_PORT
    hf_home: str = ""
    idle_release_s: float = 600.0
    floor_pct: float = 30.0
    min_pct: float = 5.0
    drain_s: float = 120.0
    start_timeout_s: float = 1200.0
    chat: str = ""
    embedding: str = ""
    backends: Mapping[str, BackendSettings] = field(default_factory=dict)
    hooks: Hooks = field(default_factory=Hooks)

    def backend(self, key: str) -> BackendSettings:
        return self.backends.get(key, BackendSettings())

    def hf_home_source(self) -> str:
        """Where the cache folder comes from: ``config``, ``env`` or ``default``."""
        if self.hf_home:
            return "config"
        return "env" if os.environ.get("HF_HOME") else "default"

    def resolved_hf_home(self) -> str:
        value = self.hf_home or os.environ.get("HF_HOME") or "~/.cache/huggingface"
        return str(Path(value).expanduser())


def parse_duration(value: Any, key: str = "duration") -> float:
    """Seconds from ``90``, ``"90s"``, ``"10m"``, ``"2h"``; ``"off"`` and ``0`` are 0."""
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        raise SettingsError(f"{key}: not a duration: {value!r}")
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("off", "never"):
            return 0.0
        scale = {"s": 1.0, "m": 60.0, "h": 3600.0}.get(text[-1:])
        try:
            value = float(text[:-1] if scale else text) * (scale or 1.0)
        except ValueError:
            raise SettingsError(
                f"{key}: not a duration: {value!r} (use 90, 90s, 10m, 2h or off)"
            ) from None
    if value < 0:
        raise SettingsError(f"{key}: a duration cannot be negative")
    return float(value)


def _command(value: Any, key: str) -> tuple[str, ...]:
    if isinstance(value, str):
        parts = shlex.split(value)
    elif isinstance(value, list) and all(isinstance(part, str) for part in value):
        parts = list(value)
    else:
        raise SettingsError(f"{key}: a command is a string or a list of strings")
    return (str(Path(parts[0]).expanduser()), *parts[1:]) if parts else ()


def _percent(data: Mapping[str, Any], key: str, default: float) -> float:
    value = data.get(key, default)
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not 0 <= value <= 90
    ):
        raise SettingsError(f"{key}: a number from 0 to 90")
    return float(value)


def _override(data: dict[str, Any], item: str) -> None:
    key, sep, raw = item.partition("=")
    key = key.strip()
    if not sep or not key:
        raise SettingsError(f"-X {item}: use -X key=value")
    try:
        value = tomllib.loads(f"v = {raw}")["v"]
    except tomllib.TOMLDecodeError:
        value = raw
    *parents, last = key.split(".")
    target = data
    for part in parents:
        child = target.setdefault(part, {})
        if not isinstance(child, dict):
            raise SettingsError(f"-X {item}: {part} is not a table")
        target = child
    target[last] = value


_KNOWN = {
    "host", "port", "hf_home", "idle_release", "floor_pct", "min_pct", "drain",
    "start_timeout", "picks", "backends", "hooks",
}  # fmt: skip


def from_dict(data: Mapping[str, Any]) -> Settings:
    unknown = sorted(set(data) - _KNOWN)
    if unknown:
        raise SettingsError(f"unknown setting: {', '.join(unknown)}")
    host = str(data.get("host", "127.0.0.1"))
    if host not in LOOPBACK:
        raise SettingsError(f"host: {host} is not loopback; only 127.0.0.1 is served")
    port = data.get("port", DEFAULT_PORT)
    if isinstance(port, bool) or not isinstance(port, int) or not 0 <= port < 65536:
        raise SettingsError(f"port: not a port: {port!r}")
    floor, minimum = _percent(data, "floor_pct", 30.0), _percent(data, "min_pct", 5.0)
    if minimum > floor:
        raise SettingsError("min_pct: cannot be above floor_pct")
    backends: dict[str, BackendSettings] = {}
    for name, table in dict(data.get("backends", {})).items():
        env = table.get("env", {}) if isinstance(table, dict) else None
        if not isinstance(env, dict):
            raise SettingsError(
                f"backends.{name}: a table, with env a table of strings"
            )
        backends[name] = BackendSettings(
            command=_command(table.get("command", []), f"backends.{name}.command"),
            env={str(k): str(v) for k, v in env.items()},
            models_dir=str(table.get("models_dir", "")),
        )
    picks, hooks = data.get("picks", {}), data.get("hooks", {})
    return Settings(
        host=host,
        port=port,
        hf_home=str(data.get("hf_home", "")),
        idle_release_s=parse_duration(data.get("idle_release", "10m"), "idle_release"),
        floor_pct=floor,
        min_pct=minimum,
        drain_s=parse_duration(data.get("drain", "2m"), "drain"),
        start_timeout_s=parse_duration(
            data.get("start_timeout", "20m"), "start_timeout"
        )
        or 1200.0,
        chat=str(picks.get("chat", "")),
        embedding=str(picks.get("embedding", "")),
        backends=backends,
        hooks=Hooks(
            before_load=str(hooks.get("before_load", "")),
            after_stop=str(hooks.get("after_stop", "")),
            timeout_s=parse_duration(hooks.get("timeout", "30m"), "hooks.timeout")
            or 1800.0,
        ),
    )


def load(path: Path | None = None, overrides: Sequence[str] = ()) -> Settings:
    """The settings of ``config.toml`` (none when it is missing) with *overrides*."""
    path = paths.config_path() if path is None else path
    data: dict[str, Any] = {}
    if path.is_file():
        try:
            data = tomllib.loads(path.read_text(encoding="utf-8"))
        except tomllib.TOMLDecodeError as exc:
            raise SettingsError(f"{path}: {exc}") from None
    for item in overrides:
        _override(data, item)
    return from_dict(data)


def save_hf_home(value: str, path: Path | None = None) -> None:
    """Set ``hf_home`` in ``config.toml``, keeping every other line as it is.

    The key goes before the first table, since a key after ``[picks]`` would belong
    to it. The new text must parse to the old settings plus ``hf_home``; else the
    file is left alone.
    """
    path = paths.config_path() if path is None else path
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    lines = text.splitlines()
    top = next(
        (i for i, line in enumerate(lines) if line.lstrip().startswith("[")), len(lines)
    )
    entry = f"hf_home = {json.dumps(value)}"
    for i in range(top):
        if re.match(r"\s*hf_home\s*=", lines[i]):
            lines[i] = entry
            break
    else:
        lines[top:top] = [entry, ""] if top < len(lines) else [entry]
    new = "\n".join(lines) + "\n"
    try:
        ok = tomllib.loads(new) == {**tomllib.loads(text), "hf_home": value}
    except tomllib.TOMLDecodeError:
        ok = False
    if not ok:
        raise SettingsError(f"{path}: could not set hf_home safely; set it by hand")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".new")
    temp.write_text(new, encoding="utf-8")
    if path.exists():
        os.chmod(temp, path.stat().st_mode & 0o777)
    os.replace(temp, path)
