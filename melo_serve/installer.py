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
import signal
import subprocess
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

from melo_serve import __version__, paths
from melo_serve.adapters.base import InstallPlan
from melo_serve.plan import find_brew

CHUNK = 1 << 20
GRACE_S = 30  # how long a stopped command gets to clean up
NOT_INSTALLED = "nothing was installed"
PART_KEPT = "what came is kept; run the same command again to continue"
HOMEBREW_ENV = {
    "HOMEBREW_NO_AUTO_UPDATE": "1",  # do not update all of Homebrew first
    "HOMEBREW_NO_ANALYTICS": "1",
    "HOMEBREW_NO_INSTALL_CLEANUP": "1",  # touch this formula only
}


class InstallError(Exception):
    """A step failed; *reason* says why and *after* what is left, in a few words."""

    def __init__(self, reason: str, after: str = NOT_INSTALLED) -> None:
        super().__init__(reason)
        self.after = after


def _last_line(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1].replace(str(Path.home()), "~")[:100] if lines else ""


def _wait(proc: subprocess.Popen[str], timeout: int) -> str:
    """What *proc* printed. A stop or the time limit ends its whole group with SIGTERM.

    SIGKILL only after a grace period: ``brew`` cleans up a half-installed formula
    when it is told to stop, and leaves it looking installed when it is killed.
    """
    try:
        return proc.communicate(timeout=timeout)[0] or ""
    except BaseException:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(proc.pid, signal.SIGTERM)
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(GRACE_S)
        proc.kill()
        raise


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
        with contextlib.ExitStack() as stack:
            sink = stack.enter_context(log.open("wb")) if log else subprocess.PIPE
            proc = stack.enter_context(
                subprocess.Popen(
                    argv,
                    stdin=subprocess.DEVNULL,
                    stdout=sink,
                    stderr=subprocess.STDOUT,
                    env=env,
                    text=True,
                    errors="replace",
                    start_new_session=True,
                )  # fmt: skip
            )
            text = _wait(proc, timeout)
    except subprocess.TimeoutExpired:
        raise InstallError(f"{what} did not finish in {timeout} s") from None
    except (OSError, subprocess.SubprocessError) as exc:
        raise InstallError(f"{what} could not run: {exc}") from None
    if log:
        text = log.read_bytes()[-4000:].decode(errors="replace")
    last = _last_line(text)
    if check and proc.returncode:
        raise InstallError(f"{what} failed: {last or f'exit code {proc.returncode}'}")
    return proc.returncode, last


def app_path(plan: InstallPlan) -> Path:
    return Path(plan.dest).expanduser()


def problem(plan: InstallPlan, name: str) -> str:
    """Why *plan* cannot be run now ("" when it can); checked before anything starts."""
    if plan.via == "homebrew":
        return "" if find_brew() else f"{name} needs Homebrew (brew.sh)"
    app = app_path(plan)
    if os.path.lexists(app):
        return (
            f"{paths.tilde(app)} exists but is not a working {name}; move it away first"
        )
    folder = app.parent
    if os.path.lexists(folder) and not (folder.is_dir() and os.access(folder, os.W_OK)):
        return f"{paths.tilde(folder)} is not a folder melo-serve can write to"
    return ""


def _part(plan: InstallPlan) -> Path:
    return paths.downloads_dir() / (plan.file + ".part")


def disk_need(plan: InstallPlan) -> int:
    """Bytes the home volume needs: what is left to download, and the app.

    The app unpacks to about twice the DMG (oMLX 0.7.3: 881 MB to 1.7 GB).
    """
    part = _part(plan)
    have = part.stat().st_size if part.exists() else 0
    return max(0, plan.size_bytes - have) + 2 * plan.size_bytes


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
    if have == total:  # a stop came just after the last byte
        return
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
            length = reply.headers.get("Content-Length")
            if (reply.status == 206 and not resumed) or (
                reply.status == 200 and length not in (None, str(total))
            ):  # a captive portal, say: what is on disk stays
                raise InstallError("the server did not send the file", PART_KEPT)
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
        if isinstance(exc, urllib.error.HTTPError) and exc.code == 416:
            part.unlink()  # not a start of this file: the next run begins again
        raise InstallError(
            f"could not download {part.name[: -len('.part')]}: {exc}",
            PART_KEPT if part.exists() else "run the same command again",
        ) from None
    size = part.stat().st_size
    if size > total:
        part.unlink()
        raise InstallError(f"the download is larger than the pinned {total} bytes")
    if size < total:
        raise InstallError(
            f"the download stopped at {size} of {total} bytes", PART_KEPT
        )


def download(plan: InstallPlan, progress: Callable[[int, int], None]) -> Path:
    """The DMG in ``<melo_serve home>/downloads``, with its size and sha256 checked.

    A file that does not match the pinned values is deleted.
    """
    final = paths.downloads_dir() / plan.file
    if not final.exists():
        final.parent.mkdir(parents=True, exist_ok=True)
        part = _part(plan)
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


def _detach(mount: Path) -> bool:
    """Detach the image at *mount*, forcing it if it is busy; whether that worked.

    A stop that comes meanwhile waits for this, or the image would stay attached.
    """
    stop: BaseException | None = None
    detached = False
    for extra in ([], ["-force"]):
        try:
            _run(["hdiutil", "detach", *extra, str(mount)], 120)
            detached = True
            break
        except InstallError:
            continue
        except BaseException as exc:
            stop = exc
    if stop:
        raise stop
    return detached


def _remove(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def install_app(
    plan: InstallPlan,
    progress: Callable[[int, int], None],
    note: Callable[[str], None],
) -> None:
    """Download the DMG, check it, and copy its app to ``plan.dest``.

    The mount folder is always the same, so an image that a killed run left attached
    is found and detached here; ``get.lock`` keeps two runs from sharing it.
    """
    dest = app_path(plan)
    new = dest.with_name(dest.name + ".new")
    dmg = download(plan, progress)
    note("checked the size and sha256 of the download")
    dest.parent.mkdir(parents=True, exist_ok=True)
    mount = paths.home() / "dmg"
    if os.path.ismount(mount):  # first: this also ends a copy that outlived its run
        _detach(mount)
    _remove(new)  # left by a stop
    mount.mkdir(exist_ok=True)
    copied = False
    try:
        detached = False
        try:
            _run(
                ["hdiutil", "attach", "-nobrowse", "-readonly", "-mountpoint",
                 str(mount), str(dmg)], 300,
            )  # fmt: skip
            _run(["ditto", str(mount / dest.name), str(new)], 900)
        finally:
            detached = _detach(mount)
            with contextlib.suppress(OSError):
                mount.rmdir()
        if not detached:
            note("the disk image stayed attached; eject it in Finder")
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
            with contextlib.suppress(OSError):
                _remove(new)
    new.rename(dest)
    with contextlib.suppress(OSError):
        dmg.unlink()
    note(_gatekeeper(dest))


def install_brew(plan: InstallPlan, note: Callable[[str], None]) -> None:
    """``brew install`` the formula; its output goes to ``logs/brew.log``."""
    brew = find_brew()
    if brew is None:
        raise InstallError("Homebrew (brew.sh) was not found")
    log = paths.logs_dir() / "brew.log"
    note(f"installing with Homebrew; its output goes to {paths.tilde(log)}")
    try:
        _run(
            [brew, "install", plan.formula], 3600,
            env={**os.environ, **HOMEBREW_ENV}, log=log,
        )  # fmt: skip
    except InstallError as exc:
        raise InstallError(
            str(exc), f"the whole output is in {paths.tilde(log)}"
        ) from None
