"""``melo-serve plan`` on fake Macs: every probe is a fake and no file of this Mac is read."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from melo_serve import catalog, fit, plan, settings
from melo_serve.adapters import default_adapters
from melo_serve.fit import GIB

FIXTURE = Path(__file__).parent / "fixtures" / "plan.v1.json"
EMBED = "model:qwen3-embedding-0.6b-omlx"
EMBED_BYTES = round(1.2 * GIB)
EMBED_REPO = "models--Qwen--Qwen3-Embedding-0.6B"
FOUR_B = "model:qwen3-4b-omlx"
FOUR_B_REPO = "models--mlx-community--Qwen3-4B-Instruct-2507-4bit"


def mac(**changes: Any) -> plan.Machine:
    values: dict[str, Any] = {
        "chip": "Apple M5 Max", "arch": "arm64", "ram_gb": 128, "macos": "27.2",
        "macos_build": "TEST1", "free_gb": 80.0, "hf_home": "~/hf",
        "hf_home_from": "env", "homebrew": True,
    }  # fmt: skip
    return plan.Machine(**{**values, **changes})


def make_plan(
    tmp: Path, machine: plan.Machine, prep: Callable[[Path], None] | None
) -> dict[str, Any]:
    """The plan for *machine*; the backends and the cache live in *tmp*."""
    hf = tmp / "hf"
    (tmp / "bin").mkdir(parents=True)
    if prep:
        prep(tmp)
    config = settings.from_dict(
        {
            "hf_home": str(hf),
            "backends": {
                "splash": {"command": [str(tmp / "bin/splash")], "models_dir": str(tmp / "splash")},
                "omlx": {"command": [str(tmp / "bin/omlx")]},
            },
        }
    )  # fmt: skip
    memory = fit.Memory(total_bytes=(machine.ram_gb or 0) * GIB, free_pct=80.0)
    picks = catalog.load(tmp / "none.toml")
    return plan.build(machine, memory, picks, default_adapters(), config)


def omlx_installed_4b_done_embed_cut_short(tmp: Path) -> None:
    omlx = tmp / "bin/omlx"
    omlx.write_text("#!/bin/sh\necho omlx 0.7.3\n")
    omlx.chmod(0o755)
    snapshot = tmp / "hf/hub" / FOUR_B_REPO / "snapshots/rev"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}")
    (snapshot / "model.safetensors").write_bytes(b"")
    (snapshot.parents[1] / "refs").mkdir()
    (snapshot.parents[1] / "refs/main").write_text("rev")
    blobs = tmp / "hf/hub" / EMBED_REPO / "blobs"
    blobs.mkdir(parents=True)
    (blobs / "abc.incomplete").write_bytes(b"x" * 1000)


CASES: list[dict[str, Any]] = [
    {"id": "16 GB: the small model", "mac": {"ram_gb": 16}, "pick": "qwen3-4b-omlx",
     "states": {"backend:omlx": "todo", FOUR_B: "todo", EMBED: "todo"}},
    {"id": "24 GB: the light model", "mac": {"ram_gb": 24}, "pick": "qwen3-8b-omlx"},
    {"id": "48 GB: Splash and oMLX", "mac": {"ram_gb": 48}, "pick": "qwen3.8-27b-splash",
     "states": {"backend:splash": "todo", "backend:omlx": "todo",
                "model:qwen3.8-27b-splash": "todo", EMBED: "todo"}},
    {"id": "128 GB: the 105 GB model is too big for the free memory", "mac": {},
     "pick": "qwen3.8-27b-splash", "alternatives": ["qwen3-8b-omlx", "qwen3-4b-omlx"]},
    {"id": "macOS 15: no Splash, the 15 DMG", "mac": {"ram_gb": 48, "macos": "15.6"},
     "pick": "qwen3-8b-omlx", "file": "oMLX-0.7.3-macos15-sequoia.dmg", "hint": ""},
    {"id": "macOS 26.3 is too old for Splash", "mac": {"macos": "26.3"},
     "pick": "qwen3-8b-omlx", "hint": ""},
    {"id": "no Homebrew: no Splash, and a hint", "mac": {"homebrew": False},
     "pick": "qwen3-8b-omlx", "hint": "Installing Homebrew would allow Qwen3.8 27B"},
    {"id": "low disk", "mac": {"ram_gb": 48, "free_gb": 20.0}, "pick": "qwen3.8-27b-splash",
     "enough_disk": False},
    {"id": "a backend, a model and part of a model are there", "mac": {"ram_gb": 16},
     "prep": omlx_installed_4b_done_embed_cut_short, "pick": "qwen3-4b-omlx",
     "states": {"backend:omlx": "done", FOUR_B: "done", EMBED: "partial"},
     "download": EMBED_BYTES - 1000},
    {"id": "Intel", "mac": {"arch": "x86_64"}, "pick": None, "reason": "Intel"},
    {"id": "8 GB", "mac": {"ram_gb": 8}, "pick": None, "reason": "needs 16 GB"},
    {"id": "macOS 14", "mac": {"macos": "14.7"}, "pick": None, "reason": "too old"},
]  # fmt: skip


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_plan_for_a_fake_mac(tmp_path: Path, case: dict[str, Any]) -> None:
    result = make_plan(tmp_path, mac(**case["mac"]), case.get("prep"))
    assert (result["pick"] or {}).get("id") == case["pick"]
    assert result["supported"] is (case["pick"] is not None)
    assert case.get("reason", "") in result["reason"]
    assert bool(result["reason"]) is (case["pick"] is None)
    if "hint" in case:
        assert result["reason_hint"].startswith(case["hint"])
        assert bool(result["reason_hint"]) is bool(case["hint"])
    states = {i["id"]: i["state"] for i in result["items"]}
    if "states" in case:
        assert states == case["states"]
    if case["pick"] is None:
        assert result["items"] == [] and result["alternatives"] == []
    else:
        assert f"model:{case['pick']}" in states
        todo = [i for i in result["items"] if i["state"] != "done"]
        assert result["download_bytes"] == case.get(
            "download", sum(i["size_bytes"] for i in todo)
        )
        assert result["enough_disk"] is case.get("enough_disk", True)
    if "alternatives" in case:
        assert [a["id"] for a in result["alternatives"]] == case["alternatives"]
    if "file" in case:
        omlx = next(i for i in result["items"] if i["id"] == "backend:omlx")
        assert omlx["file"] == case["file"] and len(omlx["sha256"]) == 64
    assert "This Mac:" in plan.render(result, {"splash": "Splash", "omlx": "oMLX"})


def types(value: dict[str, Any]) -> dict[str, str]:
    return {key: type(item).__name__ for key, item in value.items()}


def test_plan_keeps_to_the_published_example(tmp_path: Path) -> None:
    """localmelo's contract test loads the same file: its keys and types must not drift."""
    example = json.loads(FIXTURE.read_text())
    real = make_plan(tmp_path, mac(), omlx_installed_4b_done_embed_cut_short)
    assert real["schema"] == example["schema"] == plan.SCHEMA
    assert types(real) == types(example)
    for part in ("machine", "pick"):
        assert types(real[part]) == types(example[part])
    allowed: dict[str, str] = {}
    for item in example["items"]:
        allowed |= types(item)
    assert any("cached_bytes" in i for i in real["items"])
    for item in real["items"]:
        assert types(item).items() <= allowed.items()
        assert {
            "id",
            "kind",
            "name",
            "via",
            "size_bytes",
            "dest",
            "state",
        } <= item.keys()
    assert types(real["alternatives"][0]) == types(example["alternatives"][0])
