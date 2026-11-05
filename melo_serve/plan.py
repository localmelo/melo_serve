"""``melo-serve plan``: what a first run would set up on this Mac, and what it costs.

It only reads: no network, no writes, nothing started. The plan is one JSON object
(``schema`` ``melo_serve.plan/1``) so a client such as localmelo can show the same
list before it asks the person for one OK; the terminal prints a short summary.
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from melo_serve import fit, paths
from melo_serve.adapters.base import Adapter, Install, InstallPlan
from melo_serve.catalog import Catalog, Pick
from melo_serve.fit import GIB, MARGIN_BYTES, size_text
from melo_serve.settings import Settings

SCHEMA = "melo_serve.plan/1"
BREW = "/opt/homebrew/bin/brew"


@dataclass(frozen=True)
class Machine:
    chip: str
    arch: str
    ram_gb: int | None
    macos: str
    macos_build: str
    free_gb: float
    hf_home: str  # shown with ~
    hf_home_from: str  # config, env or default
    homebrew: bool


def _run(*argv: str) -> str:
    try:
        out = subprocess.run(
            argv, capture_output=True, text=True, timeout=10, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.stdout.strip()


def find_brew() -> str | None:
    """Homebrew's ``brew``: on PATH, else at its Apple silicon place; None without it."""
    found = shutil.which("brew")
    if found:
        return found
    return BREW if os.access(BREW, os.X_OK) else None


def macos_version() -> str:
    return _run("sw_vers", "-productVersion")


def probe_machine(settings: Settings) -> Machine:
    ram = fit.total_ram_bytes()
    hf_home = settings.resolved_hf_home()
    return Machine(
        chip=_run("sysctl", "-n", "machdep.cpu.brand_string") or "unknown",
        arch=platform.machine(),
        ram_gb=round(ram / GIB) if ram else None,
        macos=macos_version(),
        macos_build=_run("sw_vers", "-buildVersion"),
        free_gb=round(fit.free_bytes(Path(hf_home) / "hub") / GIB, 1),
        hf_home=paths.tilde(hf_home),
        hf_home_from=settings.hf_home_source(),
        homebrew=find_brew() is not None,
    )


def version_tuple(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in re.findall(r"\d+", text)) or (0,)


def _item(
    kind: str, key: str, name: str, via: str, size: int, dest: str, state: str
) -> dict[str, Any]:
    return {
        "id": f"{kind}:{key}", "kind": kind, "name": name, "via": via,
        "size_bytes": size, "dest": dest, "state": state,
    }  # fmt: skip


def _found_at(path: str) -> str:
    """Where a backend is: its app when the command lives in one, else the command."""
    app, sep, _ = path.partition(".app/")
    return paths.tilde(app + ".app" if sep else path)


def _backend_item(
    adapter: Adapter, install: InstallPlan, found: Install
) -> dict[str, Any]:
    version = found.version if found.found and found.version else install.version
    item = _item(
        "backend", adapter.key, f"{adapter.name} {version}", install.via,
        install.size_bytes, _found_at(found.path) if found.found else install.dest,
        "done" if found.found else "todo",
    )  # fmt: skip
    if install.sha256:
        item.update(sha256=install.sha256, file=install.file)
    return item


def _model_item(
    pick: Pick, adapter: Adapter, settings: Settings, machine: Machine
) -> dict[str, Any]:
    size = round((pick.disk_gb or 0) * GIB)
    missing = adapter.installed(pick, settings)
    cached = adapter.cached_bytes(pick, settings) if missing else 0
    state = "partial" if cached else "todo" if missing else "done"
    item = _item(
        "model", pick.id, pick.name, "huggingface", size, machine.hf_home, state
    )
    item.update(role=pick.role, backend=pick.backend)
    if state == "partial":
        item["cached_bytes"] = cached
    return item


def _refusal(
    machine: Machine, catalog: Catalog, adapters: Mapping[str, Adapter]
) -> str:
    """Why this Mac cannot run melo_serve at all; "" when it can."""
    macos = version_tuple(machine.macos)
    if machine.arch != "arm64":
        return f"This Mac has an Intel chip ({machine.arch}); melo_serve needs Apple silicon."
    if not any(a.install_plan(macos) for a in adapters.values()):
        return f"macOS {machine.macos or 'unknown'} is too old; no backend runs on it."
    smallest = min(
        (p.min_ram_gb for p in catalog.for_role("chat") if p.min_ram_gb), default=0
    )
    if machine.ram_gb is None:
        return "This Mac's memory could not be read."
    if machine.ram_gb < smallest:
        return f"This Mac has {machine.ram_gb} GB of memory; the smallest model needs {smallest} GB."
    return ""


def build(
    machine: Machine,
    memory: fit.Memory | None,
    catalog: Catalog,
    adapters: Mapping[str, Adapter],
    settings: Settings,
    chosen: str = "",
) -> dict[str, Any]:
    """The plan for *machine*, around the pick *chosen* or the recommended one.

    Every probe is an argument, so tests pass fakes.
    """
    plan: dict[str, Any] = {
        "schema": SCHEMA, "machine": asdict(machine), "supported": False, "reason": "",
        "reason_hint": "", "pick": None, "items": [], "download_bytes": 0,
        "enough_disk": True, "alternatives": [],
    }  # fmt: skip
    plan["reason"] = _refusal(machine, catalog, adapters)
    if plan["reason"]:
        return plan
    macos = version_tuple(machine.macos)
    installs = {key: adapter.detect(settings) for key, adapter in adapters.items()}
    plans = {k: p for k, a in adapters.items() if (p := a.install_plan(macos))}
    limits = {"floor_pct": settings.floor_pct, "min_pct": settings.min_pct}
    no_brew: list[Pick] = []  # fit this Mac's memory, and lack only Homebrew

    def gettable(pick: Pick) -> bool:
        """``get`` can finish it: it has files to download, or the model is here."""
        adapter = adapters[pick.backend]
        return bool(adapter.hub_files(pick)) or not adapter.installed(pick, settings)

    def runs_here(pick: Pick) -> bool:
        install = plans.get(pick.backend)
        if install is None or (pick.min_ram_gb or 0) > (machine.ram_gb or 0):
            return False
        if (
            install.needs_homebrew
            and not machine.homebrew
            and not installs[pick.backend].found
        ):
            no_brew.append(pick)
            return False
        return True

    chats = [p for p in catalog.for_role("chat") if gettable(p) and runs_here(p)]
    embedders = [
        p for p in catalog.for_role("embedding") if gettable(p) and runs_here(p)
    ]
    if no_brew:
        names = " and ".join(p.name for p in no_brew if p.role == "chat")
        if names:
            plan["reason_hint"] = f"Installing Homebrew would allow {names}."
    if not chats:
        plan["reason"] = "No model fits this Mac."
        return plan
    chat = next((p for p in chats if p.id == chosen), None) if chosen else chats[0]
    if chat is None:
        plan["reason"] = f"{chosen} cannot be set up on this Mac."
        return plan
    verdict = fit.check(chat, memory, **limits)  # this moment's, for information
    picks = [chat, *embedders[:1]]
    keys = list(dict.fromkeys(p.backend for p in picks))
    items = [_backend_item(adapters[k], plans[k], installs[k]) for k in keys]
    items += [_model_item(p, adapters[p.backend], settings, machine) for p in picks]
    todo = [i for i in items if i["state"] != "done"]
    download = sum(max(0, i["size_bytes"] - i.get("cached_bytes", 0)) for i in todo)
    plan.update(
        supported=True,
        pick={
            "id": chat.id,
            "name": chat.name,
            "backend": chat.backend,
            "fit": verdict.verdict,
        },
        items=items,
        download_bytes=download,
        enough_disk=machine.free_gb * GIB >= download + MARGIN_BYTES,
        alternatives=[
            {"id": p.id, "name": p.name, "backend": p.backend}
            for p in chats
            if p is not chat
        ],
    )
    return plan


FIT_WORDS = {
    fit.FITS: "fits",
    fit.ESTIMATE: "likely fits",
    fit.ALONE: "fits alone; close other models before the first load",
    fit.TOO_BIG: "too big while other models run; it fits this Mac",
    fit.UNKNOWN: "fit unknown",
}


def render(plan: Mapping[str, Any], backend_names: Mapping[str, str]) -> str:
    """The short summary for a terminal."""
    machine = plan["machine"]
    ram = f"{machine['ram_gb']} GB" if machine["ram_gb"] else "memory unknown"
    lines = [
        f"This Mac: {machine['chip']} · {ram} · macOS {machine['macos']} · "
        f"{machine['free_gb']:.0f} GB free"
    ]
    if plan["pick"]:
        pick = plan["pick"]
        backend = backend_names.get(pick["backend"], pick["backend"])
        verdict = FIT_WORDS.get(pick["fit"], pick["fit"])
        lines.append(f"Recommended: {pick['name']} on {backend} ({verdict})")
    else:
        lines.append(f"melo_serve cannot serve a model here: {plan['reason']}")
    if plan["reason_hint"]:
        lines.append(plan["reason_hint"])
    marks = {"done": "●", "partial": "◐", "todo": "○"}
    width = max((len(i["name"]) for i in plan["items"]), default=0)
    for item in plan["items"]:
        cached = item.get("cached_bytes", 0)
        if item["state"] == "done":
            note = "already there"
        elif cached:
            note = f"{size_text(item['size_bytes'])}, {size_text(cached)} already downloaded"
        else:
            note = size_text(item["size_bytes"])
        via = {"homebrew": "Homebrew", "dmg": "DMG", "huggingface": "Hugging Face"}[
            item["via"]
        ]
        lines.append(
            f"  {marks[item['state']]} {item['name']:<{width}}  {note} · {via} → {item['dest']}"
        )
    if plan["pick"]:
        disk = (
            "" if plan["enough_disk"] else " That is more than fits with 5 GB to spare."
        )
        lines.append(
            f"Download about {size_text(plan['download_bytes'])}; {machine['free_gb']:.0f} GB free "
            f"in {machine['hf_home']} (from {machine['hf_home_from']}).{disk}"
        )
    if plan["alternatives"]:
        names = ", ".join(a["name"] for a in plan["alternatives"])
        lines.append(f"Other picks that run here: {names}")
    return "\n".join(lines)
