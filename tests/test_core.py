"""Settings, the catalog, the fit check, launch lines and the login plist."""

from __future__ import annotations

import plistlib
from pathlib import Path

import pytest

from melo_serve import catalog, fit, launchd, settings
from melo_serve.adapters import OmlxAdapter, SplashAdapter
from melo_serve.catalog import pick_from_dict


def test_settings_file_then_overrides(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text('port = 8330\nidle_release = "10m"\n[picks]\nchat = "27b"\n')
    loaded = settings.load(
        path, ["port=8322", "drain=5s", "backends.splash.command=~/bin/splash -v"]
    )
    assert (loaded.port, loaded.idle_release_s, loaded.drain_s, loaded.chat) == (
        8322,
        600,
        5,
        "27b",
    )
    command = loaded.backend("splash").command
    assert command[1:] == ("-v",) and not command[0].startswith("~")
    for bad in (["idle_release=soon"], ["host=0.0.0.0"], ["min_pct=40"], ["nope=1"]):
        with pytest.raises(settings.SettingsError):
            settings.load(path, bad)


def test_catalog_names_and_local_picks(tmp_path: Path) -> None:
    local = tmp_path / "picks.toml"
    local.write_text(
        '[[pick]]\nid = "qwen3.8-27b-splash"\nrole = "chat"\nbackend = "splash"\n'
        'source = "me/x:Q8"\naliases = ["27b"]\n\n'
        '[[pick]]\nid = "mine"\nrole = "embedding"\nbackend = "omlx"\nsource = "~/m"\n'
    )
    picks = catalog.load(local)
    # This Mac's entry replaces the built-in one.
    assert picks.resolve(" 27B ").source == "me/x:Q8"
    assert picks.resolve("Qwen3.8-Flash-Next-IQ3XXS-splash").short == "flash-next"
    assert picks.resolve("mine").role == "embedding"
    with pytest.raises(catalog.UnknownPick) as unknown:
        picks.resolve("flash-nxt")
    assert "flash-next" in unknown.value.suggestions


@pytest.mark.parametrize(
    ("free_pct", "freeing_gb", "verdict"),
    [(80, 0, fit.FITS), (40, 0, fit.ALONE), (12, 0, fit.TOO_BIG), (12, 20, fit.ALONE)],
)
def test_fit_verdicts(free_pct: float, freeing_gb: float, verdict: str) -> None:
    pick = pick_from_dict(
        {
            "id": "p",
            "role": "chat",
            "backend": "splash",
            "source": "a/b:c",
            "mem_gb": 16,
            "measured_ram_gb": 64,
        }
    )
    assert (
        fit.check(
            pick, fit.Memory(64 * fit.GIB, free_pct), freeing_gb=freeing_gb
        ).verdict
        == verdict
    )


def test_fit_estimates_what_was_not_measured_here() -> None:
    pick = pick_from_dict(
        {"id": "p", "role": "chat", "backend": "omlx", "source": "a/b", "disk_gb": 4}
    )
    result = fit.check(pick, fit.Memory(64 * fit.GIB, 80))
    assert result.verdict == fit.ESTIMATE and result.need_gb == pytest.approx(6.1)
    # An estimate does not lift the floor: below it, other heavy work waits.
    assert fit.check(pick, fit.Memory(64 * fit.GIB, 20)).verdict == fit.ALONE


def test_launch_lines(tmp_path: Path) -> None:
    hf = tmp_path / "hf"
    snapshot = hf / "hub/models--org--emb/snapshots/abc"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}")
    index = '{"weight_map": {"a": "model-1.safetensors", "b": "model-1.safetensors"}}'
    (snapshot / "model.safetensors.index.json").write_text(index)
    (hf / "hub/models--org--emb/refs").mkdir()
    (hf / "hub/models--org--emb/refs/main").write_text("abc")
    config = settings.from_dict({"hf_home": str(hf), "idle_release": "10m"})

    chat = pick_from_dict(
        {
            "id": "c",
            "role": "chat",
            "backend": "splash",
            "source": "org/m:Q4",
            "served_name": "M",
        }
    )
    line = SplashAdapter().launch(
        chat, port=5000, key="k", workdir=tmp_path, settings=config
    )
    assert line.argv[:4] == ("splash", "serve", "--model", "org/m:Q4")
    assert {"--offline", "--announce-served-name", "--no-webui", "600s"} <= set(
        line.argv
    )
    assert line.argv[line.argv.index("--served-model-name") + 1] == "M"
    assert line.env["SPLASH_API_KEY"] == "k" and "k" not in line.argv
    assert "not downloaded" in SplashAdapter().installed(chat, config)

    embed = pick_from_dict(
        {
            "id": "e",
            "role": "embedding",
            "backend": "omlx",
            "source": "org/emb",
            "served_name": "E",
        }
    )
    # A download cut short: the index is there, its shard is not.
    assert "1 of 1 weight files" in OmlxAdapter().installed(embed, config)
    (snapshot / "model-1.safetensors").write_bytes(b"")
    assert OmlxAdapter().installed(embed, config) == ""
    line = OmlxAdapter().launch(
        embed, port=5001, key="k", workdir=tmp_path / "w", settings=config
    )
    assert (
        line.env["OMLX_PORT"],
        line.env["OMLX_API_KEY"],
        line.env["OMLX_HF_CACHE_ENABLED"],
    ) == ("5001", "k", "0")
    assert (tmp_path / "w/models/E").resolve() == snapshot.resolve()


def test_login_service_plist() -> None:
    plist = plistlib.loads(launchd.render(["/x/melo-serve", "up", "-f"]))
    assert plist["Label"] == "ai.localmelo.melo_serve"
    assert (
        plist["KeepAlive"] == {"SuccessfulExit": False} and plist["RunAtLoad"] is True
    )
    assert plist["ProgramArguments"] == ["/x/melo-serve", "up", "-f"]
    assert plist["EnvironmentVariables"][launchd.ENV] == "1"
