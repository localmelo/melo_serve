"""``melo-serve get`` with a fake Hugging Face hub: no network, no download, temp homes."""

from __future__ import annotations

import fcntl
import fnmatch
import json
import os
import signal
import stat
import threading
import time
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from melo_serve import catalog, cli, fit, get, paths, settings
from melo_serve.adapters import omlx
from melo_serve.adapters.splash import SplashAdapter

PICK = "4b"  # qwen3-4b-omlx
NAME = "Qwen3 4B Instruct"
REPO = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
TOTAL = 6 * fit.GIB  # what the fake hub says the repo weighs
GGUF, DRAFT = "unsloth/Qwen3.8-27B-GGUF", "incoai/Qwen3.8-27B-DFlash2"
WANTED = ["Qwen3.8-27B-UD-Q4_K_M.gguf", "mmproj-BF16.gguf"]  # the 27b's files of GGUF
SPLASH_REPOS = {  # the repo holds another quantization too
    GGUF: {
        WANTED[0]: 20 * fit.GIB,
        "Qwen3.8-27B-Q8_0.gguf": 28 * fit.GIB,
        WANTED[1]: fit.GIB,
    },
    DRAFT: {"config.json": 1, "model.safetensors": 2 * fit.GIB},
}


class FakeHub:
    """Stands in for ``huggingface_hub``: it writes a tiny cache like the real layout."""

    def __init__(self, hf: Path, behaviour: Callable[[Any], None] | None = None):
        self.hf, self.calls, self.behaviour = hf, [], behaviour
        half = TOTAL // 2
        self.repos: dict[str, dict[str, int]] = {  # repo -> file -> size on the Hub
            REPO: {"config.json": half, "model.safetensors": half},
            **SPLASH_REPOS,
        }

    def model_info(self, repo: str, files_metadata: bool = False) -> Any:
        assert files_metadata
        return SimpleNamespace(
            siblings=[SimpleNamespace(rfilename=n, size=s) for n, s in self.repos[repo].items()]
        )  # fmt: skip

    def write_snapshot(self, repo: str = REPO, only: list[str] | None = None) -> None:
        """The files of *repo* (or those named) as the Hub lays them out: blobs, and a
        snapshot of links to them."""
        store = self.hf / "hub" / ("models--" + repo.replace("/", "--"))
        snapshot = store / "snapshots" / "rev1"
        (store / "blobs").mkdir(parents=True, exist_ok=True)
        (store / "refs").mkdir(exist_ok=True)
        (store / "refs" / "main").write_text("rev1")
        for name in only or self.repos[repo]:
            blob = store / "blobs" / name.replace("/", "_")
            blob.write_bytes(b"x" * 1000)
            link = snapshot / name
            link.parent.mkdir(parents=True, exist_ok=True)
            link.unlink(missing_ok=True)
            link.symlink_to(os.path.relpath(blob, link.parent))

    def snapshot_download(
        self,
        repo_id: str,
        cache_dir: str,
        tqdm_class: Any,
        allow_patterns: list[str] | None = None,
    ) -> str:
        assert cache_dir == str(self.hf / "hub")  # no local_dir
        self.calls.append((repo_id, allow_patterns))
        bar = tqdm_class(name="huggingface_hub.snapshot_download", total=0, unit="B")
        bar.update(10)
        if self.behaviour:
            self.behaviour(bar)
        self.write_snapshot(
            repo_id, [n for n in self.repos[repo_id] if matches(n, allow_patterns)]
        )
        bar.update(1000)
        return cache_dir


def matches(name: str, patterns: list[str] | None) -> bool:
    return not patterns or any(fnmatch.fnmatchcase(name, p) for p in patterns)


@pytest.fixture
def hub(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[FakeHub, Path]:
    import huggingface_hub

    hf = tmp_path / "hf"
    fake = FakeHub(hf)
    backend = tmp_path / "bin" / "omlx"  # a tested oMLX is already installed
    backend.parent.mkdir()
    backend.write_text(f"#!/bin/sh\necho omlx {omlx.VERSION}\n")
    backend.chmod(0o755)
    monkeypatch.setenv("PATH", f"{backend.parent}:{os.environ['PATH']}")
    monkeypatch.setenv("HF_HOME", str(hf))
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.setenv("HF_HUB_DISABLE_TELEMETRY", "0")  # get must switch it off
    monkeypatch.setattr(get, "PROGRESS_EVERY", 0)
    monkeypatch.setattr(fit, "free_bytes", lambda path: 100 * fit.GIB)
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: fake)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake.snapshot_download)
    return fake, hf


def run_get(
    capsys: pytest.CaptureFixture[str], *args: str
) -> tuple[int, list[dict[str, Any]], str]:
    code = cli.main(["get", PICK, *args])
    out = capsys.readouterr().out
    events = [json.loads(line) for line in out.splitlines()] if "--json" in args else []
    return code, events, out


def test_a_download_reports_progress_and_remembers_the_cache(
    hub: tuple[FakeHub, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    fake, hf = hub
    paths.config_path().parent.mkdir(parents=True)
    paths.config_path().write_text('port = 8999\n\n[picks]\nchat = "27b"\n')
    code, events, _ = run_get(capsys, "--json")
    assert code == 0 and fake.calls == [(REPO, None)]
    states = [e["state"] for e in events]
    assert states[0] == "done" and events[0]["item"] == "config:hf_home"
    assert (events[1]["item"], states[1]) == ("backend:omlx", "skipped")
    assert states[2:] == ["running", "running", "done"]
    assert all(e["item"] == "model:qwen3-4b-omlx" for e in events[2:])
    assert all(e["total_bytes"] == TOTAL for e in events[2:])
    assert events[-1] == {
        "item": "model:qwen3-4b-omlx", "state": "done", "done_bytes": TOTAL,
        "total_bytes": TOTAL, "line": f"downloaded {NAME} into {hf}",
    }  # fmt: skip
    config = settings.load()
    assert (config.hf_home, config.port, config.chat) == (str(hf), 8999, "27b")
    assert os.environ["HF_HUB_DISABLE_TELEMETRY"] == "1"  # no agent name, no dotfile
    code, events, _ = run_get(capsys, "--json")  # again: the model is there
    assert code == 0 and fake.calls == [(REPO, None)]
    assert [(e["state"], e["line"]) for e in events] == [
        ("skipped", f"already installed ({omlx.VERSION}), skipped"),
        ("skipped", "already downloaded, skipped"),
    ]


def test_a_complete_model_is_skipped_and_a_new_cache_folder_is_remembered(
    hub: tuple[FakeHub, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake, _ = hub
    elsewhere = tmp_path / "elsewhere"
    fake.hf = elsewhere
    fake.write_snapshot()
    paths.config_path().parent.mkdir(parents=True)
    paths.config_path().write_text('hf_home = "/old"\n[picks]\nchat = "27b"\n')
    code, _, out = run_get(capsys, "--hf-home", str(elsewhere))
    assert code == 0 and fake.calls == []
    assert out.splitlines() == [
        f"saved hf_home = {elsewhere} in {paths.config_path()}",
        f"oMLX: already installed ({omlx.VERSION}), skipped",
        f"{NAME}: already downloaded, skipped",
    ]
    assert settings.load().hf_home == str(elsewhere) and settings.load().chat == "27b"
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "other"))  # not <hf_home>/hub
    code, _, out = run_get(capsys, "--hf-home", str(elsewhere))
    assert code == 2 and out.startswith("failed: HF_HUB_CACHE is ")


def test_too_little_disk_is_refused_before_anything_is_downloaded(
    hub: tuple[FakeHub, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake, hf = hub
    monkeypatch.setattr(fit, "free_bytes", lambda path: TOTAL + fit.MARGIN_BYTES - 1)
    code, events, _ = run_get(capsys, "--json")
    assert code == 2 and fake.calls == []
    assert events[0]["line"] == (
        "failed: not enough space: needs 6.0 GB and 5 GB to spare, 11.0 GB free in "
        f"{hf / 'hub'}; free some space, or use --hf-home PATH; nothing was downloaded"
    )
    assert not hf.exists() and not paths.config_path().exists()


def test_a_second_download_waits_its_turn(
    hub: tuple[FakeHub, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    paths.ensure_home()
    fd = os.open(paths.get_lock_path(), os.O_RDWR | os.O_CREAT)
    os.write(fd, b"4242")
    fcntl.flock(fd, fcntl.LOCK_EX)
    try:
        code, _, out = run_get(capsys)
    finally:
        os.close(fd)
    assert code == 75 and out == "another download is running (pid 4242)\n"
    assert hub[0].calls == []


def test_stopping_aborts_the_xet_session_and_says_how_to_continue(
    hub: tuple[FakeHub, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake, _ = hub
    aborted: list[int] = []
    monkeypatch.setattr(
        "huggingface_hub.utils._xet.abort_xet_session", lambda: aborted.append(1)
    )

    def signal_while_a_thread_downloads(bar: Any) -> None:
        errors: list[BaseException] = []
        started = threading.Event()

        def worker() -> None:  # like the hub's download threads
            try:
                for _ in range(5000):
                    bar.update(1)
                    started.set()
                    time.sleep(0.001)
            except BaseException as exc:
                errors.append(exc)

        thread = threading.Thread(target=worker)
        thread.start()
        started.wait(5)
        os.kill(os.getpid(), signal.SIGTERM)
        thread.join(10)
        raise errors[0]  # as the hub does with a failed file

    fake.behaviour = signal_while_a_thread_downloads
    code, events, _ = run_get(capsys, "--json")
    assert code == 130 and aborted == [1]
    assert events[-1]["line"] == (
        "Stopped at 0 MB / 6.0 GB. The finished files are kept; "
        "run the same command again to download the rest."
    )


def test_a_symlinked_config_is_written_through(tmp_path: Path) -> None:
    real, link = tmp_path / "dotfiles" / "config.toml", tmp_path / "config.toml"
    real.parent.mkdir()
    real.write_bytes(b"port = 8999\r\n# the picks\r\n[picks]\r\nchat = '27b'\r\n")
    real.chmod(0o600)
    link.symlink_to(real)
    settings.save_hf_home("/tmp/models \U0001f600", link)
    assert link.is_symlink() and stat.S_IMODE(real.stat().st_mode) == 0o600
    assert real.read_bytes().startswith(
        b'port = 8999\r\n# the picks\r\nhf_home = "/tmp/models'
    )
    config = settings.load(link)
    assert (config.hf_home, config.chat) == ("/tmp/models \U0001f600", "27b")


def test_the_cache_folder_is_made_absolute_and_a_bad_home_is_refused(
    hub: tuple[FakeHub, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake, _ = hub
    fake.hf = tmp_path / "rel" / "x"
    fake.write_snapshot()
    monkeypatch.chdir(tmp_path)
    code, _, _ = run_get(capsys, "--hf-home", "rel/x")
    assert code == 0 and settings.load().hf_home == str(tmp_path / "rel" / "x")
    code, _, out = run_get(capsys, "--hf-home", "~nosuchuser-zz/x")
    assert code == 2 and out == (
        "failed: ~nosuchuser-zz/x: there is no such home folder; nothing was downloaded\n"
    )


def splash_mac(tmp_path: Path) -> Path:
    """A Splash that is installed, with an empty models folder; returns that folder."""
    command, models = tmp_path / "bin" / "splash", tmp_path / "splash"
    command.write_text("#!/bin/sh\necho splash 1.5.0\n")
    command.chmod(0o755)
    paths.config_path().parent.mkdir(parents=True, exist_ok=True)
    paths.config_path().write_text(
        f'[backends.splash]\ncommand = ["{command}"]\nmodels_dir = "{models}"\n'
    )
    return models


def test_a_splash_model_is_downloaded_repo_by_repo_and_only_then_is_there(
    hub: tuple[FakeHub, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fake, hf = hub
    splash_mac(tmp_path)
    pick, config = catalog.load().resolve("27b"), settings.load()
    adapter = SplashAdapter()
    assert "not downloaded" in adapter.installed(pick, config)
    code = cli.main(["get", "27b", "--json"])
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert code == 0 and fake.calls == [(GGUF, WANTED), (DRAFT, None)]
    size = sum(SPLASH_REPOS[GGUF][n] for n in WANTED) + sum(
        SPLASH_REPOS[DRAFT].values()
    )
    assert size < sum(sum(r.values()) for r in SPLASH_REPOS.values())  # not the Q8
    done = events[-1]
    assert (done["state"], done["total_bytes"]) == ("done", size)
    assert adapter.installed(pick, config) == ""
    assert not (
        hf
        / "hub"
        / "models--unsloth--Qwen3.8-27B-GGUF"
        / "snapshots"
        / "rev1"
        / "Qwen3.8-27B-Q8_0.gguf"
    ).exists()
    code = cli.main(["get", "27b", "--json"])  # again: it is there
    assert code == 0 and len(fake.calls) == 2
    assert json.loads(capsys.readouterr().out.splitlines()[-1])["state"] == "skipped"


def test_a_part_of_a_splash_model_is_not_there_and_get_goes_on_with_it(
    hub: tuple[FakeHub, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fake, hf = hub
    splash_mac(tmp_path)
    pick, config = catalog.load().resolve("27b"), settings.load()
    adapter = SplashAdapter()
    fake.write_snapshot(DRAFT)
    fake.write_snapshot(GGUF, [WANTED[0], "Qwen3.8-27B-Q8_0.gguf"])  # no mmproj
    assert "not downloaded" in adapter.installed(pick, config)
    assert adapter.cached_bytes(pick, config) == 3000  # the Q8 does not count
    link = (
        hf
        / "hub"
        / "models--unsloth--Qwen3.8-27B-GGUF"
        / "snapshots"
        / "rev1"
        / WANTED[1]
    )
    link.symlink_to("../../blobs/gone")  # a link to a blob that is not there
    assert "not downloaded" in adapter.installed(pick, config)
    link.unlink()
    code = cli.main(["get", "27b", "--json"])
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    first = next(e for e in events if e["state"] == "running")
    assert code == 0 and first["done_bytes"] == 3000  # it starts from what is there
    assert adapter.installed(pick, config) == ""
    assert adapter.cached_bytes(pick, config) == 4000


def test_a_splash_pick_without_a_file_list_is_refused_unless_its_model_is_there(
    hub: tuple[FakeHub, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    models = splash_mac(tmp_path)
    (paths.config_path().parent / "picks.toml").write_text(
        '[[pick]]\nid = "mine"\nrole = "chat"\nbackend = "splash"\nsource = "me/x:Q8"\n'
    )
    code = cli.main(["get", "mine"])
    assert code == 2 and capsys.readouterr().out == (
        "mine: failed: me/x:Q8 is not a Hugging Face repo to download; nothing was downloaded\n"
    )
    (models / "me/x:Q8").mkdir(parents=True)
    assert cli.main(["get", "mine"]) == 0 and hub[0].calls == []


def test_a_stop_between_two_repos_stops_the_pick(
    hub: tuple[FakeHub, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fake, _ = hub
    splash_mac(tmp_path)
    fake.behaviour = lambda bar: os.kill(os.getpid(), signal.SIGTERM)
    code = cli.main(["get", "27b", "--json"])
    last = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert code == 130 and fake.calls == [(GGUF, WANTED)]
    assert (
        last["line"].startswith("Stopped at ")
        and "finished files are kept" in last["line"]
    )


def test_the_cache_is_read_as_the_hub_lays_it_out(tmp_path: Path) -> None:
    fake = FakeHub(tmp_path / "hf")
    flash, draft = (
        "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF",
        "incoai/Qwen3.8-Flash-Next-DFlash2",
    )
    mmproj = "mmproj-Qwen3.8-Flash-Next-BF16.gguf"
    fake.repos = {
        flash: {
            "IQ3_XXS/m-00001-of-00002.gguf": 1,
            "IQ3_XXS/m-00002-of-00002.gguf": 1,
            "Q8_0/a.gguf": 1,
            mmproj: 1,
        },
        draft: {"config.json": 1},
    }
    pick = catalog.load().resolve("fn")
    config = settings.from_dict(
        {
            "hf_home": str(fake.hf),
            "backends": {"splash": {"models_dir": str(tmp_path / "m")}},
        }
    )
    adapter = SplashAdapter()
    fake.write_snapshot(
        flash, ["IQ3_XXS/m-00001-of-00002.gguf", "Q8_0/a.gguf", mmproj]
    )  # the second part is missing
    fake.write_snapshot(draft)
    assert adapter.cached_bytes(pick, config) == 3000
    assert "not downloaded" in adapter.installed(pick, config)
    fake.write_snapshot(flash, ["IQ3_XXS/m-00002-of-00002.gguf"])
    assert adapter.installed(pick, config) == ""
    blobs = fake.hf / "hub" / ("models--" + draft.replace("/", "--")) / "blobs"
    (blobs / "abc.incomplete").write_bytes(
        b"x" * 500
    )  # a file of the draft still downloading
    assert adapter.cached_bytes(pick, config) == 4000
    assert adapter.cached_bytes(pick, config, include_partial=True) == 4500
    assert "not downloaded" in adapter.installed(pick, config)


def test_a_pick_lists_its_hub_files() -> None:
    entry = {"id": "p", "role": "chat", "backend": "splash", "source": "a/b:c"}
    pick = catalog.pick_from_dict(
        {**entry, "hub": [{"repo": "a/b", "files": ["x/*"]}, {"repo": "a/c"}]}
    )
    assert pick.hub == (catalog.HubRepo("a/b", ("x/*",)), catalog.HubRepo("a/c"))
    assert pick.hub[0].matches("x/1.gguf") and not pick.hub[0].matches("y/1.gguf")
    assert pick.hub[1].matches("anything")
    for bad in ({"repo": "nowhere"}, {"repo": "a/b", "files": "x"}, "a/b"):
        with pytest.raises(catalog.CatalogError):
            catalog.pick_from_dict({**entry, "hub": [bad]})
