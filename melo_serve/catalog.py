"""Picks: a model on a backend, with an id short enough to type.

The built-in catalog is ``melo_serve/picks.toml``. A ``picks.toml`` in melo_serve's
folder adds picks of this computer, and replaces a built-in pick with the same id.
"""

from __future__ import annotations

import tomllib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from difflib import get_close_matches
from importlib import resources
from pathlib import Path
from typing import Any

from melo_serve import paths

ROLES = ("chat", "embedding")


class CatalogError(ValueError):
    """A catalog file has an entry melo_serve cannot use."""


class UnknownPick(LookupError):
    """No pick has that id, alias or served name; ``suggestions`` has near ones."""

    def __init__(self, name: str, suggestions: Iterable[str] = ()) -> None:
        self.name = name
        self.suggestions = tuple(suggestions)
        hint = (
            f" Did you mean {', '.join(self.suggestions)}?" if self.suggestions else ""
        )
        super().__init__(f"no pick is called {name!r}.{hint}")


@dataclass(frozen=True)
class Pick:
    id: str
    name: str
    role: str
    backend: str
    source: str
    served_name: str
    aliases: tuple[str, ...] = ()
    summary: str = ""
    disk_gb: float | None = None
    mem_gb: float | None = None
    measured_ram_gb: int | None = None
    load_s: float | None = None
    decode_tps: float | None = None
    caps: tuple[str, ...] = ()
    min_ram_gb: int | None = None
    args: tuple[str, ...] = ()
    env: Mapping[str, str] = field(default_factory=dict)

    @property
    def short(self) -> str:
        """The name to type: the first alias, else the id."""
        return self.aliases[0] if self.aliases else self.id

    def names(self) -> tuple[str, ...]:
        return (self.id, *self.aliases, self.served_name)

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "short": self.short,
            "name": self.name,
            "role": self.role,
            "backend": self.backend,
            "model": self.served_name,
            "summary": self.summary,
            "disk_gb": self.disk_gb,
            "mem_gb": self.mem_gb,
            "load_s": self.load_s,
            "decode_tps": self.decode_tps,
            "caps": list(self.caps),
        }


def _optional_number(entry: Mapping[str, Any], key: str, where: str) -> float | None:
    value = entry.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise CatalogError(f"{where}: {key} is not a number")
    return float(value)


def pick_from_dict(entry: Mapping[str, Any], where: str = "pick") -> Pick:
    for key in ("id", "role", "backend", "source"):
        if not isinstance(entry.get(key), str) or not entry[key].strip():
            raise CatalogError(f"{where}: {key} is missing")
    pick_id = entry["id"].strip()
    where = f"{where} {pick_id}"
    if entry["role"] not in ROLES:
        raise CatalogError(f"{where}: role is {entry['role']!r}, not one of {ROLES}")
    ram = entry.get("measured_ram_gb")
    minimum = entry.get("min_ram_gb")
    env = entry.get("env", {})
    if not isinstance(env, dict):
        raise CatalogError(f"{where}: env is not a table")
    return Pick(
        id=pick_id,
        name=str(entry.get("name") or pick_id),
        role=entry["role"],
        backend=entry["backend"].strip(),
        source=entry["source"].strip(),
        served_name=str(entry.get("served_name") or pick_id).strip(),
        aliases=tuple(
            str(a).strip() for a in entry.get("aliases", ()) if str(a).strip()
        ),
        summary=str(entry.get("summary", "")),
        disk_gb=_optional_number(entry, "disk_gb", where),
        mem_gb=_optional_number(entry, "mem_gb", where),
        measured_ram_gb=int(ram) if isinstance(ram, int | float) else None,
        load_s=_optional_number(entry, "load_s", where),
        decode_tps=_optional_number(entry, "decode_tps", where),
        caps=tuple(str(c) for c in entry.get("caps", ())),
        min_ram_gb=int(minimum) if isinstance(minimum, int | float) else None,
        args=tuple(str(a) for a in entry.get("args", ())),
        env={str(k): str(v) for k, v in env.items()},
    )


class Catalog:
    """Picks by id; every id, alias and served name names one pick."""

    def __init__(self, picks: Iterable[Pick]) -> None:
        self._picks: dict[str, Pick] = {}
        for pick in picks:
            self._picks[pick.id] = pick
        self._names: dict[str, Pick] = {}
        for pick in self._picks.values():
            for name in pick.names():
                key = name.lower()
                other = self._names.get(key)
                if other is not None and other.id != pick.id:
                    raise CatalogError(
                        f"the name {name!r} belongs to both {other.id} and {pick.id}"
                    )
                self._names[key] = pick

    def all(self) -> list[Pick]:
        return list(self._picks.values())

    def for_role(self, role: str) -> list[Pick]:
        return [pick for pick in self._picks.values() if pick.role == role]

    def get(self, pick_id: str) -> Pick | None:
        return self._picks.get(pick_id)

    def resolve(self, name: str) -> Pick:
        """The pick with this id, alias or served name (case and spaces ignored)."""
        key = name.strip().lower()
        found = self._names.get(key)
        if found is None:
            raise UnknownPick(name.strip(), get_close_matches(key, self._names, n=3))
        return found


def _entries(text: str, where: str) -> list[Mapping[str, Any]]:
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise CatalogError(f"{where}: {exc}") from None
    entries = data.get("pick", [])
    if not isinstance(entries, list):
        raise CatalogError(f"{where}: [[pick]] entries expected")
    return entries


def builtin_text() -> str:
    return resources.files("melo_serve").joinpath("picks.toml").read_text("utf-8")


def load(local: Path | None = None, *, builtin: bool = True) -> Catalog:
    """The built-in picks, then this computer's ``picks.toml`` over them."""
    picks: dict[str, Pick] = {}
    if builtin:
        for entry in _entries(builtin_text(), "picks.toml (built in)"):
            pick = pick_from_dict(entry, "picks.toml (built in)")
            picks[pick.id] = pick
    local = paths.local_picks_path() if local is None else local
    if local.is_file():
        for entry in _entries(local.read_text(encoding="utf-8"), str(local)):
            pick = pick_from_dict(entry, str(local))
            picks[pick.id] = pick
    return Catalog(picks.values())
