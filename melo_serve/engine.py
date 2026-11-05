"""The engine: what each role serves, starting it on demand, and switching it.

Two roles, ``chat`` and ``embedding``, each serve one pick at a time.
- A role starts its backend at its first request, or on a switch. A load first
  passes the fit check and the ``before_load`` hook.
- After ``idle_release`` without requests the model's memory is freed; the
  process stays, and the next request reloads the weights.
- A switch waits up to ``drain`` for the role's requests in flight, stops the
  old backend, starts the new one and checks it with one request. When that
  fails it starts the old pick again.
- A request that names a model the role no longer serves gets 409
  ``model_switched`` with the current id: no client is answered by a model it
  did not ask for.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import math
import os
import secrets
import signal
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import aiohttp

from melo_serve import __version__, fit, paths, procs, state
from melo_serve.adapters import Adapter
from melo_serve.catalog import ROLES, Catalog, Pick, UnknownPick
from melo_serve.settings import Settings

log = logging.getLogger("melo_serve")

RETRY_AFTER_FAILURE_S = 30.0
CONTROL_TIMEOUT = aiohttp.ClientTimeout(total=30)  # readiness and release calls
SHUTDOWN_DRAIN_S = 3.0


class EngineError(Exception):
    """A request or command that cannot be served now; ``kind`` says why."""

    def __init__(self, status: int, kind: str, message: str, **extra: Any) -> None:
        super().__init__(message)
        self.status, self.kind, self.extra = status, kind, extra

    def to_json(self) -> dict[str, Any]:
        return {"error": {"type": self.kind, "message": str(self), **self.extra}}


@dataclass
class Instance:
    pick: Pick
    port: int
    key: str
    child: procs.Child

    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self.port}"


@dataclass
class Slot:
    role: str
    pick: Pick | None = None
    instance: Instance | None = None
    status: str = "stopped"  # stopped, starting, ready or failed
    loaded: bool = False
    error: str = ""
    since: str = ""
    dims: int | None = None
    inflight: int = 0
    last_used: float = 0.0
    failed_at: float = 0.0
    draining: bool = False
    retired: set[str] = field(default_factory=set)
    starting: asyncio.Task[None] | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)  # held to start or stop
    idle: asyncio.Event = field(default_factory=asyncio.Event)  # set: none in flight

    def ready(self) -> bool:
        return (
            self.status == "ready"
            and self.instance is not None
            and self.instance.child.alive
        )


@dataclass(frozen=True)
class Upstream:
    base: str
    headers: Mapping[str, str]
    model: str


def pick_rows(
    catalog: Catalog,
    adapters: Mapping[str, Adapter],
    settings: Settings,
    memory: fit.Memory | None,
    current: Mapping[str, tuple[Pick, bool]] = {},  # noqa: B006 (read only)
) -> dict[str, Any]:
    """Every pick with its fit now; *current* maps a role to (pick, loaded)."""
    rows = []
    for pick in catalog.all():
        serving, loaded = current.get(pick.role, (None, False))
        freeing = 0.0
        if memory is not None and serving is not None and loaded:
            freeing = fit.need_gb(serving, memory.total_gb)[0] or 0.0
        adapter = adapters.get(pick.backend)
        checked = fit.check(
            pick,
            memory,
            freeing_gb=freeing,
            floor_pct=settings.floor_pct,
            min_pct=settings.min_pct,
        )
        rows.append(
            {
                **pick.to_json(),
                "backend_name": adapter.name if adapter else pick.backend,
                "current": serving is not None and serving.id == pick.id,
                "problem": adapter.installed(pick, settings)
                if adapter
                else f"melo_serve has no backend called {pick.backend}",
                "fit": checked.to_json(),
            }
        )
    return {
        "memory": None
        if memory is None
        else {"total_gb": round(memory.total_gb), "free_pct": memory.free_pct},
        "picks": rows,
    }


class Engine:
    def __init__(
        self,
        settings: Settings,
        catalog: Catalog,
        adapters: Mapping[str, Adapter],
        *,
        memory: Callable[[], fit.Memory | None] = fit.probe,
    ) -> None:
        self.settings, self.catalog, self.adapters = settings, catalog, adapters
        self.memory = memory
        self.port = settings.port
        self.slots = {role: Slot(role) for role in ROLES}
        for slot in self.slots.values():
            slot.idle.set()
        self.generation = 0
        self.switching: dict[str, Any] | None = None
        self.closing = False
        self.login_service = False
        self._switch_lock = asyncio.Lock()
        # total=None: a reply may stream for as long as a model writes it.
        self.http = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=None, sock_connect=30),
            connector=aiohttp.TCPConnector(limit=0),
            cookie_jar=aiohttp.DummyCookieJar(),  # one client's cookies are not another's
            auto_decompress=False,  # replies pass through as the backend wrote them
        )

    async def aclose(self) -> None:
        await self.http.close()

    # ── state ──

    @property
    def status(self) -> str:
        if self.closing:
            return "stopped"
        if self.switching:
            return "switching"
        if any(self._effective(slot)[0] == "failed" for slot in self.slots.values()):
            return "degraded"
        return "ready"

    def _effective(self, slot: Slot) -> tuple[str, str]:
        dead = slot.instance
        if slot.status == "ready" and dead is not None and not dead.child.alive:
            code = dead.child.process.returncode
            return "failed", f"{dead.pick.name} stopped unexpectedly (exit code {code})"
        return slot.status, slot.error

    def served(self) -> list[tuple[Slot, Pick]]:
        return [(slot, slot.pick) for slot in self.slots.values() if slot.pick]

    def state(self) -> dict[str, Any]:
        roles: dict[str, Any] = {}
        for role, slot in self.slots.items():
            pick, inst = slot.pick, slot.instance
            if pick is None:
                roles[role] = None
                continue
            status, error = self._effective(slot)
            roles[role] = {
                "pick": pick.id,
                "short": pick.short,
                "name": pick.name,
                "model": pick.served_name,
                "backend": pick.backend,
                "status": status,
                "loaded": slot.loaded and status == "ready",
                "caps": list(pick.caps),
                "dims": slot.dims,
                "since": slot.since,
                "error": error,
                "retired": sorted(slot.retired),
                "process": None
                if inst is None
                else {
                    "pid": inst.child.pid,
                    "started_at": round(inst.child.started_at, 3),
                },
            }
        host = (
            f"[{self.settings.host}]"
            if ":" in self.settings.host
            else self.settings.host
        )
        return {
            "schema": state.SCHEMA,
            "version": __version__,
            "generation": self.generation,
            "status": self.status,
            "updated_at": state.now_iso(),
            "front_door": {
                "url": f"http://{host}:{self.port}/v1",
                "pid": os.getpid(),
                "auth": "none",
            },
            "service": {"login": self.login_service},
            "roles": roles,
            "switch": self.switching,
        }

    def changed(self) -> None:
        try:
            state.write(paths.state_path(), self.state())
        except OSError as exc:
            log.warning("could not write %s: %s", paths.state_path(), exc)

    def picks(self) -> dict[str, Any]:
        current = {
            role: (slot.pick, slot.loaded and slot.ready())
            for role, slot in self.slots.items()
            if slot.pick is not None
        }
        return pick_rows(
            self.catalog, self.adapters, self.settings, self.memory(), current
        )

    def restore(self, saved: Mapping[str, Any] | None) -> None:
        """Select what was served last (else the configured picks), loading nothing."""
        wanted = {"chat": self.settings.chat, "embedding": self.settings.embedding}
        if saved:
            self.generation = int(saved.get("generation") or 0)
            for role, entry in (saved.get("roles") or {}).items():
                if role in self.slots and isinstance(entry, dict):
                    wanted[role] = entry.get("pick") or wanted[role]
                    self.slots[role].retired = set(entry.get("retired") or ())
        for role, name in wanted.items():
            if not name:
                continue
            try:
                pick = self.catalog.resolve(name)
            except UnknownPick:
                log.warning("the %s pick %r is not in the catalog", role, name)
                continue
            if pick.role != role:
                log.warning("%s is a %s pick, not a %s one", pick.id, pick.role, role)
                continue
            self.slots[role].pick = pick

    async def clean_orphans(self, saved: Mapping[str, Any] | None) -> None:
        """Stop backends that an earlier run left behind when it died."""
        for entry in ((saved or {}).get("roles") or {}).values():
            proc = (entry or {}).get("process") or {}
            pid, began = proc.get("pid"), proc.get("started_at")
            if not isinstance(pid, int) or not isinstance(began, int | float):
                continue
            started = await asyncio.to_thread(procs.start_time, pid)
            if started is None or abs(started - began) > 5:
                continue  # gone, or another process that reuses the pid
            log.warning(
                "stopping %s, left running by an earlier run", entry.get("pick")
            )
            procs.signal_group(pid, signal.SIGINT)
            for _ in range(150):
                if not procs.alive(pid):
                    break
                await asyncio.sleep(0.2)
            else:
                procs.signal_group(pid, signal.SIGKILL)

    # ── the front door ──

    def _check_open(self, slot: Slot) -> Pick:
        if self.closing:
            raise EngineError(503, "stopping", "melo_serve is stopping")
        if slot.draining:
            wait = int((self.switching or {}).get("load_s") or 30)
            raise EngineError(
                503,
                "switching",
                f"the {slot.role} model is being switched; try again shortly",
                retry_after=wait,
            )
        if slot.pick is None:
            raise EngineError(
                503,
                "no_model",
                f"no {slot.role} model is chosen; run melo-serve use <pick>",
            )
        return slot.pick

    def _match(self, slot: Slot, pick: Pick, requested: str | None) -> str:
        name = (requested or "").strip()
        if not name or name.lower() in {n.lower() for n in pick.names()}:
            return pick.served_name
        if name.lower() in {r.lower() for r in slot.retired}:
            raise EngineError(
                409,
                "model_switched",
                f"{name} is no longer served; the {slot.role} model is now "
                f"{pick.served_name}",
                current=pick.served_name,
            )
        served = [p.served_name for _, p in self.served()]
        raise EngineError(
            404,
            "model_not_found",
            f"{name} is not served here; this server serves {', '.join(served)}",
            served=served,
        )

    async def acquire(self, role: str, requested: str | None) -> Upstream:
        """Where to send one request for *role*; starts the backend when needed."""
        slot = self.slots[role]
        model = self._match(slot, self._check_open(slot), requested)
        if not slot.ready():
            await self._ensure(slot, "demand")
            self._check_open(slot)  # no await from here on: nothing can interleave
        inst = slot.instance
        if inst is None or not slot.ready():
            raise EngineError(503, "start_failed", slot.error or "the model stopped")
        slot.inflight += 1
        slot.idle.clear()
        slot.last_used = time.monotonic()
        if not slot.loaded:
            slot.loaded = True
            self.changed()
        return Upstream(
            inst.base, self.adapters[inst.pick.backend].auth(inst.key), model
        )

    def done(self, role: str) -> None:
        slot = self.slots[role]
        slot.inflight = max(0, slot.inflight - 1)
        slot.last_used = time.monotonic()
        if slot.inflight == 0:
            slot.idle.set()

    async def warm(self, role: str = "chat") -> None:
        """Start *role* now, as its first request would."""
        slot = self.slots[role]
        self._check_open(slot)
        if not slot.ready():
            await self._ensure(slot, "user")

    async def _ensure(self, slot: Slot, initiator: str) -> None:
        task = slot.starting
        if task is None or task.done():
            task = slot.starting = asyncio.create_task(
                self._start_on_demand(slot, initiator)
            )
        # A caller that gives up (a client that disconnects) leaves the load running.
        await asyncio.shield(task)

    async def _start_on_demand(self, slot: Slot, initiator: str) -> None:
        async with slot.lock:
            pick = self._check_open(slot)
            if slot.ready():
                return
            dead = slot.instance
            if dead is not None:
                slot.instance = None
                self._fail(
                    slot,
                    f"{dead.pick.name} stopped unexpectedly "
                    f"(exit code {dead.child.process.returncode})",
                )
            elif slot.status == "failed":
                wait = RETRY_AFTER_FAILURE_S - (time.monotonic() - slot.failed_at)
                if wait > 0:
                    raise EngineError(
                        503, "start_failed", slot.error, retry_after=math.ceil(wait)
                    )
            await self._admit(slot, pick, initiator=initiator)
            await self._start(slot, pick, initiator)

    # ── loading and stopping ──

    async def _admit(
        self,
        slot: Slot,
        pick: Pick,
        *,
        initiator: str,
        reason: str = "",
        force: bool = False,
    ) -> fit.Fit:
        """The checks before a load: installed, fit, then the before_load hook."""
        adapter = self.adapters.get(pick.backend)
        if adapter is None:
            raise EngineError(
                409,
                "unknown_backend",
                f"{pick.name} needs {pick.backend}, which melo_serve lacks",
            )
        problem = adapter.installed(pick, self.settings)
        if problem:
            raise EngineError(409, "not_installed", f"{pick.name}: {problem}")
        memory = await asyncio.to_thread(self.memory)
        loaded = slot.pick if slot.loaded and slot.ready() else None
        freeing = fit.need_gb(loaded, memory.total_gb)[0] if loaded and memory else None
        checked = fit.check(
            pick,
            memory,
            freeing_gb=freeing or 0.0,
            floor_pct=self.settings.floor_pct,
            min_pct=self.settings.min_pct,
        )
        if not checked.loadable and not force:
            raise EngineError(
                409,
                "does_not_fit",
                f"{pick.name} does not fit now: {checked.reason}",
                fit=checked.to_json(),
            )
        hook = self.settings.hooks.before_load
        if hook:
            env = {
                "MELO_SERVE_PICK": pick.id,
                "MELO_SERVE_MODEL": pick.served_name,
                "MELO_SERVE_ROLE": pick.role,
                "MELO_SERVE_BACKEND": pick.backend,
                "MELO_SERVE_MEM_GB": f"{checked.need_gb:.0f}"
                if checked.need_gb
                else "",
                "MELO_SERVE_LOAD_MIN": str(
                    max(1, math.ceil((pick.load_s or 300) / 60))
                ),
                "MELO_SERVE_INITIATOR": initiator,
                "MELO_SERVE_REASON": reason,
            }
            result = await procs.run_shell(hook, env, self.settings.hooks.timeout_s)
            if not result.ok:
                raise EngineError(
                    409,
                    "refused",
                    result.output or "the before_load hook refused the load",
                )
        return checked

    async def _start(self, slot: Slot, pick: Pick, initiator: str) -> None:
        adapter = self.adapters[pick.backend]
        port, key = procs.free_port(), secrets.token_urlsafe(24)
        workdir = paths.backend_dir(f"{slot.role}-{pick.backend}")
        workdir.mkdir(parents=True, exist_ok=True)
        log_path = (
            paths.logs_dir()
            / f"{slot.role}-{pick.backend}-{time.strftime('%Y%m%d-%H%M%S')}.log"
        )
        slot.status, slot.error = "starting", ""
        self.changed()
        began = time.monotonic()
        child: procs.Child | None = None
        try:
            launch = adapter.launch(
                pick, port=port, key=key, workdir=workdir, settings=self.settings
            )
            child = await procs.spawn(
                launch.argv, launch.env, log_path, launch.stop_signal
            )
            inst = Instance(pick, port, key, child)
            await self._wait_ready(adapter, inst)
            dims = await self._probe(adapter, inst, slot.role)
        except (
            OSError,
            aiohttp.ClientError,
            EngineError,
            asyncio.CancelledError,
        ) as exc:
            if child is not None:
                await child.stop(grace_s=10.0)
            if isinstance(exc, asyncio.CancelledError):
                slot.status = "stopped"
                raise
            message = f"{pick.name} did not start: {exc}"
            tail = procs.tail(log_path, 12)
            self._fail(slot, message)
            raise EngineError(
                502, "start_failed", message, log=tail, log_path=str(log_path)
            ) from None
        slot.instance, slot.pick, slot.dims = inst, pick, dims
        slot.status, slot.loaded, slot.since = "ready", True, state.now_iso()
        slot.last_used = time.monotonic()
        log.info(
            "%s: %s ready in %.1f s on port %d (%s)",
            slot.role, pick.id, time.monotonic() - began, port, initiator,
        )  # fmt: skip
        self.changed()

    async def _wait_ready(self, adapter: Adapter, inst: Instance) -> None:
        url, headers = inst.base + adapter.ready_path, adapter.auth(inst.key)
        deadline = time.monotonic() + self.settings.start_timeout_s
        while True:
            with contextlib.suppress(aiohttp.ClientError, OSError):
                async with self.http.get(
                    url, headers=headers, timeout=CONTROL_TIMEOUT
                ) as response:
                    if response.status == 200:
                        return
            if not inst.child.alive:
                raise EngineError(
                    502,
                    "start_failed",
                    f"it exited with code {inst.child.process.returncode}",
                )
            if self.closing:
                raise EngineError(503, "stopping", "melo_serve is stopping")
            if time.monotonic() > deadline:
                raise EngineError(
                    502,
                    "start_failed",
                    f"it was not ready after {self.settings.start_timeout_s:.0f} s",
                )
            await asyncio.sleep(0.25)

    async def _probe(self, adapter: Adapter, inst: Instance, role: str) -> int | None:
        """One small request: it proves the route and warms the model."""
        model = inst.pick.served_name
        if role == "chat":
            path = "/v1/chat/completions"
            body: dict[str, Any] = {
                "model": model,
                "messages": [{"role": "user", "content": "Reply with OK."}],
                "max_tokens": 1,
                "stream": False,
            }
        else:
            path, body = "/v1/embeddings", {"model": model, "input": "ping"}
        async with self.http.post(
            inst.base + path,
            json=body,
            headers=adapter.auth(inst.key),
            timeout=aiohttp.ClientTimeout(total=self.settings.start_timeout_s),
        ) as response:
            status, raw = response.status, await response.read()
        if status != 200:
            text = raw[:200].decode(errors="replace")
            raise EngineError(
                502, "start_failed", f"its first request failed: HTTP {status} {text}"
            )
        if role != "embedding":
            return None
        try:
            return len(json.loads(raw)["data"][0]["embedding"])
        except (ValueError, KeyError, IndexError, TypeError):
            raise EngineError(
                502, "start_failed", "its first reply had no vector"
            ) from None

    def _fail(self, slot: Slot, message: str) -> None:
        slot.status, slot.error, slot.loaded = "failed", message, False
        slot.failed_at = time.monotonic()
        log.error("%s: %s", slot.role, message)
        self.changed()

    async def _stop(self, slot: Slot) -> None:
        inst, slot.instance = slot.instance, None
        slot.status, slot.loaded = "stopped", False
        if inst is not None:
            await inst.child.stop()
            await procs.wait_closed(inst.port)
        self.changed()

    async def _drain(self, slot: Slot, timeout_s: float) -> None:
        """Wait for the role's requests in flight; stopping the backend cuts the rest."""
        if slot.inflight and timeout_s > 0:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(slot.idle.wait(), timeout_s)

    def _phase(self, phase: str) -> None:
        if self.switching is not None:
            self.switching["phase"] = phase
        self.changed()

    # ── switching ──

    async def switch(
        self,
        name: str,
        *,
        expected_generation: int | None = None,
        drain_s: float | None = None,
        initiator: str = "user",
        reason: str = "",
        force: bool = False,
    ) -> dict[str, Any]:
        try:
            pick = self.catalog.resolve(name)
        except UnknownPick as exc:
            raise EngineError(
                404, "unknown_pick", str(exc), suggestions=list(exc.suggestions)
            ) from None
        if self._switch_lock.locked():
            raise EngineError(
                409, "busy", "another switch is running", switch=self.switching
            )
        async with self._switch_lock:
            if self.closing:
                raise EngineError(503, "stopping", "melo_serve is stopping")
            if (
                expected_generation is not None
                and expected_generation != self.generation
            ):
                raise EngineError(
                    409,
                    "stale_generation",
                    f"the state changed (generation {self.generation}); read it again",
                    generation=self.generation,
                )
            slot = self.slots[pick.role]
            if slot.pick == pick and slot.ready():
                return self._result(pick, changed=False)
            self.switching = {
                "pick": pick.id,
                "role": pick.role,
                "phase": "checking",
                "load_s": pick.load_s,
                "since": state.now_iso(),
            }
            self.changed()
            try:
                checked = await self._admit(
                    slot, pick, initiator=initiator, reason=reason, force=force
                )
                self._phase("draining")
                slot.draining = True
                await self._drain(
                    slot, self.settings.drain_s if drain_s is None else drain_s
                )
                async with slot.lock:
                    previous = slot.pick
                    running = previous if slot.instance is not None else None
                    if running is not None:
                        self._phase("stopping")
                        await self._stop(slot)
                    self._phase("loading")
                    try:
                        await self._start(slot, pick, initiator)
                    except EngineError:
                        if running is not None:
                            self._phase("rolling back")
                            with contextlib.suppress(EngineError):
                                await self._start(slot, running, "rollback")
                        raise
                    if (
                        previous is not None
                        and previous.served_name != pick.served_name
                    ):
                        slot.retired.add(previous.served_name)
                    slot.retired.discard(pick.served_name)
                    self.generation += 1
            finally:
                slot.draining = False
                self.switching = None
                self.changed()
            log.info("%s: now %s (generation %d)", pick.role, pick.id, self.generation)
            return self._result(pick, changed=True, checked=checked)

    def _result(
        self, pick: Pick, *, changed: bool, checked: fit.Fit | None = None
    ) -> dict[str, Any]:
        return {
            "changed": changed,
            "generation": self.generation,
            "role": pick.role,
            "pick": pick.id,
            "model": pick.served_name,
            "fit": checked.to_json() if checked else None,
        }

    # ── the rest of the lifetime ──

    async def idle_loop(self) -> None:
        idle_s = self.settings.idle_release_s
        if idle_s <= 0:
            return
        tick = min(15.0, max(0.2, idle_s / 4))
        while not self.closing:
            await asyncio.sleep(tick)
            for slot in self.slots.values():
                inst = slot.instance
                quiet = time.monotonic() - slot.last_used >= idle_s
                if inst is None or not (
                    slot.ready() and slot.loaded and not slot.inflight and quiet
                ):
                    continue
                adapter = self.adapters[inst.pick.backend]
                path = adapter.release_path(inst.pick)
                if not path:
                    continue
                try:
                    async with self.http.post(
                        inst.base + path,
                        headers=adapter.auth(inst.key),
                        timeout=CONTROL_TIMEOUT,
                    ) as response:
                        response.raise_for_status()
                except (aiohttp.ClientError, OSError) as exc:
                    log.warning("%s: could not free the memory: %s", slot.role, exc)
                    continue
                slot.loaded = False
                log.info(
                    "%s: memory freed after %.0f s without requests", slot.role, idle_s
                )
                self.changed()

    async def stop_all(self, slots: Iterable[Slot] | None = None) -> None:
        """Stop every backend (melo_serve is stopping); then the after_stop hook."""
        self.closing = True
        chosen = list(self.slots.values() if slots is None else slots)
        for slot in chosen:
            await self._drain(slot, SHUTDOWN_DRAIN_S)
        stopped = False
        for slot in chosen:
            if slot.starting is not None and not slot.starting.done():
                slot.starting.cancel()
            async with slot.lock:
                stopped = stopped or slot.instance is not None
                await self._stop(slot)
        hook = self.settings.hooks.after_stop
        if stopped and hook:
            result = await procs.run_shell(
                hook, {"MELO_SERVE_EVENT": "stop"}, self.settings.hooks.timeout_s
            )
            if not result.ok:
                log.warning("the after_stop hook failed: %s", result.output)
