"""``melo-serve get`` installing a backend, with a fake Mac.

Nothing real runs: hdiutil, ditto, codesign, spctl and brew are small scripts that
record their arguments, the DMG comes from a server on a free local port, and HOME,
PATH and melo_serve's folder are in a temp dir.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import signal
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from melo_serve import cli, fit, get, installer, paths, plan
from melo_serve.adapters import omlx
from melo_serve.adapters.splash import SplashAdapter

PAYLOAD = bytes(range(256)) * 2  # the "DMG"
FILE = "oMLX-test.dmg"
OMLX = ("4b", "--skip", "model:qwen3-4b-omlx")  # a model is not what is tested here
SPLASH = ("27b", "--skip", "model:qwen3.8-27b-splash")


@dataclass
class Mac:
    """The fake Mac: its fake programs and what they were called with."""

    root: Path
    serves: bytes = PAYLOAD  # what the DMG server sends
    honour_range: bool = True
    requests: list[str | None] | None = None  # the Range header of each request

    @property
    def apps(self) -> Path:
        return self.root / "user" / "Applications"

    @property
    def log(self) -> Path:
        return self.root / "calls.log"

    @property
    def splash(self) -> Path:
        return self.root / "known" / "splash"

    @property
    def downloads(self) -> Path:
        return installer.downloads_dir()

    def add(self, name: str, body: str = "", folder: Path | None = None) -> Path:
        """A fake program that logs its arguments, then runs *body*."""
        path = (folder or self.root / "bin") / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f'#!/bin/sh\necho "{name} $*" >> "{self.log}"\n{body}\n')
        path.chmod(0o755)
        return path

    def calls(self) -> list[str]:
        return self.log.read_text().splitlines() if self.log.exists() else []

    def part(self, content: bytes) -> Path:
        self.downloads.mkdir(parents=True, exist_ok=True)
        part = self.downloads / (FILE + ".part")
        part.write_bytes(content)
        return part


@contextlib.contextmanager
def serve(mac: Mac) -> Iterator[str]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            body, wanted = mac.serves, self.headers.get("Range")
            assert mac.requests is not None
            mac.requests.append(wanted)
            start = int(wanted[6:-1]) if wanted and mac.honour_range else 0
            self.send_response(206 if start else 200)
            if start:
                self.send_header(
                    "Content-Range", f"bytes {start}-{len(body) - 1}/{len(body)}"
                )
            self.send_header("Content-Length", str(len(body) - start))
            self.end_headers()
            self.wfile.write(body[start:])

        def log_message(self, *args: Any) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}/"
    server.shutdown()
    server.server_close()


@pytest.fixture
def mac(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Mac]:
    fake = Mac(tmp_path / "fake", requests=[])
    user = fake.apps.parent
    fake.apps.mkdir(parents=True)  # the fake user's ~/Applications
    monkeypatch.setenv("HOME", str(user))
    monkeypatch.setenv("PATH", f"{fake.root / 'bin'}:/usr/bin:/bin")
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    monkeypatch.setenv("HF_HUB_DISABLE_TELEMETRY", "0")
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.setattr(get, "PROGRESS_EVERY", 0)
    monkeypatch.setattr(fit, "free_bytes", lambda path: 100 * fit.GIB)
    monkeypatch.setattr(plan, "macos_version", lambda: "27.2")
    monkeypatch.setattr(plan, "BREW", str(fake.root / "no-brew"))
    monkeypatch.setattr(omlx.OmlxAdapter, "known_paths", (omlx.USER_APP_CLI,))
    monkeypatch.setattr(SplashAdapter, "known_paths", (str(fake.splash),))
    with serve(fake) as base:  # the pinned DMG is replaced by PAYLOAD
        sha = hashlib.sha256(PAYLOAD).hexdigest()
        monkeypatch.setattr(omlx, "DMG_BASE", base)
        monkeypatch.setattr(omlx, "DMGS", ((15, FILE, len(PAYLOAD), sha),))
        make = f"printf '#!/bin/sh\\necho omlx {omlx.VERSION}\\n'"
        cli_path = '"$5/oMLX.app/Contents/MacOS/omlx-cli"'
        fake.add(
            "hdiutil",
            'last=""; for a; do last="$a"; done\n'
            'case "$1" in\n'
            'attach) mkdir -p "$5/oMLX.app/Contents/MacOS"\n'
            f"  {make} > {cli_path}; chmod +x {cli_path};;\n"
            'detach) rm -rf "$last"/*;;\n'
            "esac",
        )
        fake.add("ditto", 'cp -R "$1" "$2"')
        fake.add("codesign")
        fake.add("spctl")
        yield fake


def run_get(
    capsys: pytest.CaptureFixture[str], *args: str
) -> tuple[int, list[dict[str, Any]]]:
    code = cli.main(["get", "--json", *args])
    out = capsys.readouterr().out
    return code, [json.loads(line) for line in out.splitlines()]


def lines(events: list[dict[str, Any]], item: str) -> list[str]:
    return [e["line"] for e in events if e["item"] == item]


def test_the_dmg_is_installed_once_and_a_second_run_leaves_it_alone(
    mac: Mac, capsys: pytest.CaptureFixture[str]
) -> None:
    code, events = run_get(capsys, *OMLX)
    version = omlx.VERSION
    assert code == 0 and mac.requests == [None]
    last = lines(events, "backend:omlx")[-1]
    assert last == f"installed {version} into ~/Applications/oMLX.app"
    assert (mac.apps / "oMLX.app/Contents/MacOS/omlx-cli").is_file()
    assert not (mac.apps / "oMLX.app.new").exists()
    assert list(mac.downloads.iterdir()) == []  # neither the part nor the DMG
    detached = [c for c in mac.calls() if c.startswith("hdiutil detach")]
    assert len(detached) == 1 and mac.calls()[-1].startswith("spctl --assess")
    code, events = run_get(capsys, *OMLX)
    assert code == 0 and mac.requests == [None]  # nothing was fetched again
    assert lines(events, "backend:omlx") == [f"already installed ({version}), skipped"]


@pytest.mark.parametrize(
    ("honour_range", "part", "requests"),
    [
        (True, PAYLOAD[:200], ["bytes=200-"]),
        (False, b"J" * 200, ["bytes=200-"]),  # the server sends it all: start over
        (True, PAYLOAD, []),  # a stop just after the last byte
    ],
    ids=["continues", "server ignores Range", "already complete"],
)
def test_a_stopped_download_continues_where_it_stopped(
    mac: Mac,
    capsys: pytest.CaptureFixture[str],
    honour_range: bool,
    part: bytes,
    requests: list[str],
) -> None:
    mac.honour_range = honour_range
    mac.part(part)
    code, events = run_get(capsys, *OMLX)
    assert code == 0 and mac.requests == requests
    assert lines(events, "backend:omlx")[-1].startswith("installed ")
    assert (mac.apps / "oMLX.app").is_dir() and list(mac.downloads.iterdir()) == []


def test_a_download_that_does_not_match_is_deleted_and_nothing_is_installed(
    mac: Mac, capsys: pytest.CaptureFixture[str]
) -> None:
    mac.serves = PAYLOAD[::-1]
    code, events = run_get(capsys, *OMLX)
    assert code == 1
    assert lines(events, "backend:omlx")[-1] == (
        f"failed: {FILE} does not match the pinned size and sha256; "
        "it was deleted; nothing was installed"
    )
    assert list(mac.downloads.iterdir()) == [] and not list(mac.apps.iterdir())
    assert not any(c.startswith("hdiutil") for c in mac.calls())


def test_a_copy_that_fails_codesign_is_removed(
    mac: Mac, capsys: pytest.CaptureFixture[str]
) -> None:
    mac.add("codesign", "echo 'invalid signature' >&2; exit 1")
    code, events = run_get(capsys, *OMLX)
    assert code == 1
    line = lines(events, "backend:omlx")[-1]
    assert "did not pass the codesign check" in line and "invalid signature" in line
    assert line.endswith("it was removed; nothing was installed")
    assert not list(mac.apps.iterdir())  # no app, and no .new
    assert any(c.startswith("hdiutil detach") for c in mac.calls())


def test_a_stop_during_the_download_keeps_the_part(
    mac: Mac, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(installer, "CHUNK", 16)
    running = get._Out.running

    def stop_after_64_bytes(
        self: Any, item: str, name: str, done: int, total: int
    ) -> None:
        running(self, item, name, done, total)
        if done >= 64:
            os.kill(os.getpid(), signal.SIGTERM)

    monkeypatch.setattr(get._Out, "running", stop_after_64_bytes)
    code, events = run_get(capsys, *OMLX)
    assert code == 130
    assert events[-1]["line"] == (
        "Stopped while installing oMLX. Run the same command again to continue."
    )
    assert (mac.downloads / (FILE + ".part")).stat().st_size == 64
    assert not list(mac.apps.iterdir())


def test_splash_is_installed_with_homebrew_and_found_where_homebrew_puts_it(
    mac: Mac, capsys: pytest.CaptureFixture[str]
) -> None:
    code, events = run_get(capsys, *SPLASH)  # no Homebrew, no Splash
    assert code == 2 and mac.calls() == []
    assert lines(events, "backend:splash") == [
        "failed: Splash needs Homebrew (brew.sh); nothing was installed"
    ]
    mac.add(
        "brew",
        f'echo "HOMEBREW_NO_AUTO_UPDATE=$HOMEBREW_NO_AUTO_UPDATE" >> "{mac.log}"\n'
        f'mkdir -p "{mac.splash.parent}"\n'
        f'printf "#!/bin/sh\\necho splash 1.5.0\\n" > "{mac.splash}"\n'
        f'chmod +x "{mac.splash}"',
    )
    code, events = run_get(capsys, *SPLASH)
    assert code == 0
    assert mac.calls() == [
        "brew install incoai/tap/splash",
        "HOMEBREW_NO_AUTO_UPDATE=1",
    ]
    assert lines(events, "backend:splash")[-1] == "installed 1.5.0 into /opt/homebrew"
    assert (paths.logs_dir() / "brew.log").is_file()
    code, events = run_get(capsys, *SPLASH)  # found only through known_paths
    assert code == 0 and len(mac.calls()) == 2
    assert lines(events, "backend:splash") == ["already installed (1.5.0), skipped"]


def test_a_backend_the_person_skips_is_left_alone(
    mac: Mac, capsys: pytest.CaptureFixture[str]
) -> None:
    code, events = run_get(capsys, *OMLX, "--skip", "backend:omlx")
    assert code == 0 and mac.calls() == [] and mac.requests == []
    assert lines(events, "backend:omlx") == ["skipped (you chose to skip it)"]
    assert not list(mac.apps.iterdir()) and not mac.downloads.exists()
    code, events = run_get(capsys, *OMLX, "--skip", "something-else")
    assert code == 2
    assert lines(events, "get") == [
        "failed: something-else is not an item of this run; nothing was downloaded"
    ]


def test_too_little_room_for_the_dmg_is_refused_before_the_download(
    mac: Mac, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    room = 2 * len(PAYLOAD) + fit.MARGIN_BYTES - 1  # the DMG and the app, less a byte
    monkeypatch.setattr(fit, "free_bytes", lambda path: room)
    code, events = run_get(capsys, *OMLX)
    assert code == 2 and mac.requests == [] and mac.calls() == []
    assert lines(events, "disk")[0].startswith("failed: not enough space: needs ")
    assert not mac.downloads.exists()
