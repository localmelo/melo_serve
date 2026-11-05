"""``melo-serve get`` with a fake Hugging Face hub: no network, no download, temp homes."""

from __future__ import annotations

import fcntl
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

from melo_serve import cli, fit, get, paths, settings
from melo_serve.adapters import omlx

PICK = "4b"  # qwen3-4b-omlx
NAME = "Qwen3 4B Instruct"
REPO = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
TOTAL = 6 * fit.GIB  # what the fake hub says the repo weighs


class FakeHub:
    """Stands in for ``huggingface_hub``: it writes a tiny cache like the real layout."""

    def __init__(self, hf: Path, behaviour: Callable[[Any], None] | None = None):
        self.hf, self.calls, self.behaviour = hf, [], behaviour

    def model_info(self, repo: str, files_metadata: bool = False) -> Any:
        assert repo == REPO and files_metadata
        return SimpleNamespace(
            siblings=[SimpleNamespace(rfilename="a", size=TOTAL // 2)] * 2
        )

    def write_snapshot(self) -> None:
        store = self.hf / "hub" / ("models--" + REPO.replace("/", "--"))
        snapshot = store / "snapshots" / "rev1"
        (store / "blobs").mkdir(parents=True, exist_ok=True)
        snapshot.mkdir(parents=True, exist_ok=True)
        (store / "refs").mkdir(exist_ok=True)
        (store / "refs" / "main").write_text("rev1")
        (store / "blobs" / "w").write_bytes(b"x" * 1000)
        (snapshot / "config.json").write_text("{}")
        (snapshot / "model.safetensors").write_bytes(b"x" * 1000)

    def snapshot_download(self, repo_id: str, cache_dir: str, tqdm_class: Any) -> str:
        assert repo_id == REPO and cache_dir == str(self.hf / "hub")  # no local_dir
        self.calls.append(repo_id)
        bar = tqdm_class(name="huggingface_hub.snapshot_download", total=0, unit="B")
        bar.update(10)
        if self.behaviour:
            self.behaviour(bar)
        self.write_snapshot()
        bar.update(1000)
        return cache_dir


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
    assert code == 0 and fake.calls == [REPO]
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
    assert code == 0 and fake.calls == [REPO]
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


def test_a_splash_model_that_is_not_there_is_refused(
    hub: tuple[FakeHub, Path],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    paths.config_path().parent.mkdir(parents=True)
    paths.config_path().write_text(
        f'[backends.splash]\nmodels_dir = "{tmp_path}/splash"\n'
    )
    code = cli.main(["get", "27b"])
    assert code == 2 and capsys.readouterr().out == (
        "Qwen3.8 27B: failed: Splash models come in a later version of melo-serve; "
        "nothing was downloaded\n"
    )
    assert hub[0].calls == []
