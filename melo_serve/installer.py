"""Put a backend on this Mac: the pinned oMLX DMG, or Splash through Homebrew.

Nothing here asks for a password or runs ``sudo``, and every command is an argument
list with a time limit, never a shell. The DMG's size and sha256 are pinned in the
oMLX adapter; they are the trust. The app is copied to a ``.new`` folder, must pass
``codesign --verify``, and only then takes its place, so a stop never leaves half
an app. A client such as localmelo asks the person first; this module just does it.
"""

from __future__ import annotations

import contextlib
import hashlib
import http.client
import os
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

from melo_serve import __version__, paths
from melo_serve.adapters.base import InstallPlan
from melo_serve.plan import find_brew

CHUNK = 1 << 20
NOT_INSTALLED = "nothing was installed"
HOMEBREW_ENV = {
    "HOMEBREW_NO_AUTO_UPDATE": "1",  # do not update all of Homebrew first
    "HOMEBREW_NO_ANALYTICS": "1",
    "NONINTERACTIVE": "1",  # never wait for input
}


class InstallError(Exception):
    """A step failed; *reason* says why and *after* what is left, in a few words."""

    def __init__(self, reason: str, after: str = NOT_INSTALLED) -> None:
        super().__init__(reason)
        self.after = after


def _last_line(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1][:100] if lines else ""


def _run(
    argv: list[str],
    timeout: int,
    *,
    env: dict[str, str] | None = None,
    log: Path | None = None,
    check: bool = True,
) -> tuple[int, str]:
    """Run *argv* (no shell); its exit code and last line of output.

    With *log* the whole output goes to that file. A failure raises when *check*.
    """
    what = Path(argv[0]).name
    try:
        if log is None:
            done = subprocess.run(
                argv, capture_output=True, text=True, timeout=timeout, env=env,
                stdin=subprocess.DEVNULL, check=False,
            )  # fmt: skip
            text = (done.stderr or "") + (done.stdout or "")
        else:
            with log.open("wb") as sink:
                done = subprocess.run(
                    argv, text=True, stdout=sink, stderr=subprocess.STDOUT, timeout=timeout,
                    env=env, stdin=subprocess.DEVNULL, check=False,
                )  # fmt: skip
            text = log.read_bytes()[-4000:].decode(errors="replace")
    except subprocess.TimeoutExpired:
        raise InstallError(f"{what} did not finish in {timeout} s") from None
    except (OSError, subprocess.SubprocessError) as exc:
        raise InstallError(f"{what} could not run: {exc}") from None
    last = _last_line(text)
    if check and done.returncode:
        raise InstallError(f"{what} failed: {last or f'exit code {done.returncode}'}")
    return done.returncode, last


def app_path(plan: InstallPlan) -> Path:
    return Path(plan.dest).expanduser()


def downloads_dir() -> Path:
    return paths.home() / "downloads"


def problem(plan: InstallPlan, name: str) -> str:
    """Why *plan* cannot be run now ("" when it can); checked before anything starts."""
    if plan.via == "homebrew":
        return "" if find_brew() else f"{name} needs Homebrew (brew.sh)"
    if os.path.lexists(app_path(plan)):
        return (
            f"{paths.tilde(app_path(plan))} exists but is not a working {name}; "
            "move it away first"
        )
    return ""


def disk_need(plan: InstallPlan) -> int:
    """Bytes the home volume needs: what is left to download, and the app."""
    part = downloads_dir() / (plan.file + ".part")
    have = part.stat().st_size if part.exists() else 0
    return max(0, plan.size_bytes - have) + plan.size_bytes


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(CHUNK):
            digest.update(block)
    return digest.hexdigest()


def _fetch(
    url: str, part: Path, total: int, progress: Callable[[int, int], None]
) -> None:
    """Download *url* into *part*, continuing a cut-short file with a Range request."""
    have = part.stat().st_size if part.exists() else 0
    if have > total:
        part.unlink()
        have = 0
    headers = {"User-Agent": f"melo-serve/{__version__}"}
    if have:
        headers["Range"] = f"bytes={have}-"
    try:
        with urllib.request.urlopen(
            urllib.request.Request(url, headers=headers), timeout=60
        ) as reply:
            resumed = reply.status == 206 and (
                (reply.headers.get("Content-Range") or "").startswith(f"bytes {have}-")
            )
            done = have if resumed else 0  # a server that sends it all starts over
            progress(done, total)
            with part.open("ab" if resumed else "wb") as sink:
                while done <= total and (
                    block := reply.read(min(CHUNK, total - done + 1))
                ):
                    sink.write(block)
                    done += len(block)
                    progress(min(done, total), total)
    except (OSError, http.client.HTTPException) as exc:  # URLError is an OSError
        raise InstallError(
            f"could not download {part.name[: -len('.part')]}: {exc}",
            "the part that came is kept; run the same command again to continue",
        ) from None
    size = part.stat().st_size
    if size > total:
        part.unlink()
        raise InstallError(f"the download is larger than the pinned {total} bytes")
    if size < total:
        raise InstallError(
            f"the download stopped at {size} of {total} bytes",
            "the part that came is kept; run the same command again to continue",
        )


def download(plan: InstallPlan, progress: Callable[[int, int], None]) -> Path:
    """The DMG in ``<melo_serve home>/downloads``, with its size and sha256 checked.

    A file that does not match the pinned values is deleted.
    """
    final = downloads_dir() / plan.file
    if not final.exists():
        downloads_dir().mkdir(parents=True, exist_ok=True)
        part = final.with_name(plan.file + ".part")
        _fetch(plan.url, part, plan.size_bytes, progress)
        part.replace(final)
    if final.stat().st_size != plan.size_bytes or _sha256(final) != plan.sha256:
        final.unlink()
        raise InstallError(
            f"{plan.file} does not match the pinned size and sha256; it was deleted"
        )
    return final


def _gatekeeper(app: Path) -> str:
    """What Gatekeeper says about *app*, for information only."""
    try:
        code, last = _run(
            ["spctl", "--assess", "--type", "execute", str(app)], 60, check=False
        )
    except InstallError:
        return "Gatekeeper check did not run"
    return "Gatekeeper accepts it" if code == 0 else f"Gatekeeper says: {last}"


def install_app(
    plan: InstallPlan,
    progress: Callable[[int, int], None],
    note: Callable[[str], None],
) -> None:
    """Download the DMG, check it, and copy its app to ``plan.dest``."""
    dest = app_path(plan)
    new = dest.with_name(dest.name + ".new")
    dmg = download(plan, progress)
    note("checked the size and sha256 of the download")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(new, ignore_errors=True)  # left by a stop
    mount = Path(tempfile.mkdtemp(prefix="melo-serve-dmg-"))
    copied = False
    try:
        try:
            _run(
                ["hdiutil", "attach", "-nobrowse", "-readonly", "-mountpoint",
                 str(mount), str(dmg)], 300,
            )  # fmt: skip
            _run(["ditto", str(mount / dest.name), str(new)], 900)
        finally:
            for extra in ([], ["-force"]):  # the image is never left mounted
                try:
                    _run(["hdiutil", "detach", *extra, str(mount)], 120)
                    break
                except InstallError:
                    continue
            with contextlib.suppress(OSError):
                mount.rmdir()
        try:
            _run(["codesign", "--verify", "--deep", "--strict", str(new)], 300)
        except InstallError as exc:
            raise InstallError(
                f"the copy of {dest.name} did not pass the codesign check ({exc}); "
                "it was removed"
            ) from None
        copied = True
    finally:
        if not copied:
            shutil.rmtree(new, ignore_errors=True)
    new.rename(dest)
    dmg.unlink(missing_ok=True)
    note(_gatekeeper(dest))


def install_brew(plan: InstallPlan) -> Path:
    """``brew install`` the formula; its output goes to ``logs/brew.log``. Returns the log."""
    brew = find_brew()
    if brew is None:
        raise InstallError("Homebrew (brew.sh) was not found")
    log = paths.logs_dir() / "brew.log"
    try:
        _run(
            [brew, "install", plan.formula], 3600,
            env={**os.environ, **HOMEBREW_ENV}, log=log,
        )  # fmt: skip
    except InstallError as exc:
        raise InstallError(
            str(exc), f"the whole output is in {paths.tilde(log)}"
        ) from None
    return log
