"""``melo-serve``: run and switch local model servers behind one address."""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from typing import Any

from melo_serve import __version__, fit, launchd, paths
from melo_serve import catalog as catalog_module
from melo_serve import plan as plan_module
from melo_serve import settings as settings_module
from melo_serve.adapters import default_adapters
from melo_serve.client import Client, ControlError, NotRunning
from melo_serve.settings import Settings

VERDICTS = {
    fit.FITS: "fits",
    fit.ALONE: "fits alone",
    fit.ESTIMATE: "likely fits",
    fit.TOO_BIG: "too big now",
    fit.UNKNOWN: "fit unknown",
}
PHASES = {
    "checking": "checking memory",
    "draining": "waiting for requests in flight",
    "stopping": "stopping the old model",
    "loading": "loading",
    "rolling back": "it failed; starting the old model again",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="melo-serve",
        description="Run and switch local model servers on this Mac behind one address.",
    )
    parser.add_argument(
        "--version", action="version", version=f"melo-serve {__version__}"
    )
    parser.add_argument(
        "-X", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
        help="change a setting for this command, e.g. -X port=8322",
    )  # fmt: skip
    sub = parser.add_subparsers(dest="command", metavar="<command>")
    sub.add_parser("status", help="what is served (the default)")
    listing = sub.add_parser("list", help="the picks on this Mac, with their fit")
    listing.add_argument(
        "-a", "--all", action="store_true", help="also picks not downloaded"
    )
    use = sub.add_parser("use", help="switch to a pick")
    use.add_argument("pick")
    use.add_argument(
        "-n", "--now", action="store_true", help="do not wait for requests in flight"
    )
    use.add_argument(
        "-y", "--yes", action="store_true", help="load even if it does not fit now"
    )
    up = sub.add_parser("up", help="start melo_serve, and serve a pick")
    up.add_argument("pick", nargs="?")
    up.add_argument(
        "-f",
        "--foreground",
        action="store_true",
        help="run here, not in the background",
    )
    up.add_argument("-p", "--port", type=int, help="the front door's port")
    sub.add_parser("stop", help="stop the models and melo_serve")
    plan = sub.add_parser(
        "plan", help="what a first run would set up here, and its size (reads only)"
    )
    plan.add_argument("--json", action="store_true", help="the plan as JSON")
    sub.add_parser("doctor", help="check the backends, the login item and memory")
    sub.add_parser("install", help="start melo_serve at login")
    sub.add_parser("uninstall", help="remove the login item")
    return parser


def _names() -> dict[str, str]:
    return {key: adapter.name for key, adapter in default_adapters().items()}


def _role_text(entry: dict[str, Any]) -> str:
    status = entry["status"]
    if status == "ready":
        word = "ready" if entry["loaded"] else "ready (memory freed while idle)"
    elif status == "stopped":
        word = "starts at its first request"
    elif status == "starting":
        word = "loading…"
    else:
        word = f"failed: {entry['error']}"
    backend = _names().get(entry["backend"], entry["backend"])
    return f"{entry['name']} ({entry['short']}) · {backend} · {word}"


def _print_state(data: dict[str, Any]) -> None:
    print(
        f"melo_serve {data['version']} · {data['status']} · {data['front_door']['url']}"
    )
    for role, entry in data["roles"].items():
        print(f"  {role:<10} {_role_text(entry) if entry else 'none chosen'}")


def _print_picks(view: dict[str, Any], show_all: bool) -> None:
    memory = view["memory"]
    rows = [r for r in view["picks"] if show_all or r["current"] or not r["problem"]]
    head = (
        f"{memory['total_gb']} GB, {memory['free_pct']:.0f}% free"
        if memory
        else "memory unknown"
    )
    print(f"Picks for this Mac ({head})")
    width = max((len(r["short"]) for r in rows), default=4)
    for row in rows:
        mark = "●" if row["current"] else "○"
        if row["problem"]:
            note = row["problem"]
        else:
            note = f"{VERDICTS[row['fit']['verdict']]}: {row['fit']['reason']}"
        role = "" if row["role"] == "chat" else f" [{row['role']}]"
        print(
            f"  {mark} {row['short']:<{width}}  {row['name']}{role} · {row['backend_name']} · {note}"
        )
    if not show_all and len(rows) < len(view["picks"]):
        print("  melo-serve list -a shows the picks not downloaded yet")


def _wait(check: Callable[[], bool], seconds: float) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if check():
            return True
        time.sleep(0.2)
    return False


def _ensure_running(overrides: list[str]) -> Client:
    client = Client()
    if client.running():
        return client
    if launchd.status()["installed"]:
        launchd.kickstart()
    else:
        paths.ensure_home()
        with open(paths.logs_dir() / "daemon.log", "ab") as log:
            flags = [part for item in overrides for part in ("-X", item)]
            subprocess.Popen(
                [sys.executable, "-m", "melo_serve", *flags, "up", "-f"],
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=log,
                start_new_session=True,
            )
    if not _wait(client.running, 20):
        raise ControlError(
            f"melo_serve did not start; see {paths.logs_dir() / 'daemon.log'}"
        )
    return client


class _Progress(threading.Thread):
    """Prints each phase of a switch while the request for it waits."""

    def __init__(self, client: Client) -> None:
        super().__init__(daemon=True)
        self.client, self.done = client, threading.Event()

    def run(self) -> None:
        shown = ""
        while not self.done.wait(0.5):
            try:
                phase = (self.client.state().get("switch") or {}).get("phase", "")
            except ControlError:
                return
            if phase and phase != shown:
                print(f"  {PHASES.get(phase, phase)}…", flush=True)
                shown = phase


def cmd_status(
    args: argparse.Namespace, settings: Settings, overrides: list[str]
) -> int:
    _print_state(Client().state())
    return 0


def cmd_list(args: argparse.Namespace, settings: Settings, overrides: list[str]) -> int:
    try:
        view = Client().picks()
    except NotRunning:
        from melo_serve.engine import pick_rows  # the daemon's code, so imported late

        view = pick_rows(
            catalog_module.load(), default_adapters(), settings, fit.probe()
        )
    _print_picks(view, args.all)
    return 0


def cmd_use(args: argparse.Namespace, settings: Settings, overrides: list[str]) -> int:
    client = _ensure_running(overrides)
    print(f"Switching to {args.pick}…", flush=True)
    progress = _Progress(client)
    progress.start()
    try:
        options: dict[str, Any] = {"force": args.yes}
        if args.now:
            options["drain_s"] = 0
        result = client.switch(args.pick, **options)
    finally:
        progress.done.set()
        progress.join()
    verb = "Now serving" if result["changed"] else "Already serving"
    print(
        f"{verb} {result['model']} for {result['role']} · generation {result['generation']}"
    )
    return 0


def cmd_up(args: argparse.Namespace, settings: Settings, overrides: list[str]) -> int:
    if args.foreground:
        from melo_serve import daemon

        return asyncio.run(daemon.run(settings, pick=args.pick))
    client = _ensure_running(overrides)
    if args.pick:
        return cmd_use(
            argparse.Namespace(pick=args.pick, now=False, yes=False),
            settings,
            overrides,
        )
    _print_state(client.state())
    return 0


def cmd_stop(args: argparse.Namespace, settings: Settings, overrides: list[str]) -> int:
    client = Client()
    try:
        client.stop()
    except NotRunning:
        print("melo_serve is not running.")
        return 0
    if not _wait(lambda: not client.running(), 180):
        print("melo_serve has not stopped yet; see melo-serve doctor", file=sys.stderr)
        return 1
    login = " It starts again at login." if launchd.status()["installed"] else ""
    print(f"melo_serve stopped.{login}")
    return 0


def cmd_plan(args: argparse.Namespace, settings: Settings, overrides: list[str]) -> int:
    plan = plan_module.build(
        plan_module.probe_machine(settings),
        fit.probe(),
        catalog_module.load(),
        default_adapters(),
        settings,
    )
    print(
        json.dumps(plan, indent=2) if args.json else plan_module.render(plan, _names())
    )
    return 0


def cmd_doctor(
    args: argparse.Namespace, settings: Settings, overrides: list[str]
) -> int:
    print(f"melo-serve {__version__}")
    config = paths.config_path()
    print(f"  folder     {paths.home()}")
    print(f"  config     {config if config.exists() else 'none (defaults)'}")
    try:
        data = Client().state()
        print(
            f"  running    yes · {data['front_door']['url']} · pid {data['front_door']['pid']}"
        )
    except ControlError:
        print("  running    no (melo-serve up)")
    service = launchd.status()
    login = (
        ("installed, running" if service["running"] else "installed")
        if service["installed"]
        else "not installed (melo-serve install)"
    )
    print(f"  login item {login}")
    for adapter in default_adapters().values():
        found = adapter.detect(settings)
        if found.found:
            untested = "" if found.tested else " (not a tested version)"
            print(f"  {found.name:<10} {found.version or '?'} · {found.path}{untested}")
        else:
            print(f"  {found.name:<10} {found.problem} · install: {found.hint}")
    print(f"  HF cache   {settings.resolved_hf_home()}")
    memory = fit.probe()
    print(
        f"  memory     {f'{memory.total_gb:.0f} GB, {memory.free_pct:.0f}% free' if memory else 'unknown'}"
    )
    return 0


def cmd_install(
    args: argparse.Namespace, settings: Settings, overrides: list[str]
) -> int:
    client = Client()
    if client.running() and not launchd.status()["installed"]:
        client.stop()  # the login service takes over from this daemon
        _wait(lambda: not client.running(), 180)
    path = launchd.install()
    if not _wait(client.running, 20):
        print(
            f"The login item is installed ({path}), but melo_serve did not start; see melo-serve doctor"
        )
        return 1
    print(f"melo_serve now starts at login ({path}).")
    return 0


def cmd_uninstall(
    args: argparse.Namespace, settings: Settings, overrides: list[str]
) -> int:
    print(
        "Removed the login item; melo_serve is stopped."
        if launchd.uninstall()
        else "No login item was installed."
    )
    return 0


COMMANDS: dict[str, Callable[[argparse.Namespace, Settings, list[str]], int]] = {
    "status": cmd_status,
    "list": cmd_list,
    "use": cmd_use,
    "up": cmd_up,
    "stop": cmd_stop,
    "plan": cmd_plan,
    "doctor": cmd_doctor,
    "install": cmd_install,
    "uninstall": cmd_uninstall,
}


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    overrides = list(args.overrides)
    if getattr(args, "port", None) is not None:
        overrides.append(f"port={args.port}")
    try:
        settings = settings_module.load(overrides=overrides)
        paths.socket_path()  # a socket path macOS cannot bind: said here, once
    except (ValueError, catalog_module.CatalogError) as exc:
        print(f"melo-serve: {exc}", file=sys.stderr)
        return 2
    try:
        return COMMANDS[args.command or "status"](args, settings, overrides)
    except NotRunning:
        print("melo_serve is not running. Start it with: melo-serve up")
        return 1
    except ControlError as exc:
        print(f"melo-serve: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
