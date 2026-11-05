"""The control API, on a Unix socket only this user can open.

No token is needed: the socket's folder and the socket itself are the user's
alone, and no web page can reach a Unix socket. localmelo's gateway calls it
for the console and the app; ``melo-serve`` calls it from a terminal.
"""

from __future__ import annotations

import asyncio
import json
import math
from collections.abc import Callable
from typing import Any

from aiohttp import web

from melo_serve.engine import Engine, EngineError
from melo_serve.router import errors


async def _json(request: web.Request) -> dict[str, Any]:
    raw = await request.read()
    try:
        body = json.loads(raw) if raw else {}
    except ValueError:
        body = None
    if not isinstance(body, dict):
        raise EngineError(400, "bad_request", "the body must be a JSON object")
    return body


def _optional(
    body: dict[str, Any], name: str, kind: type | tuple[type, ...], what: str
) -> Any:
    """body[name], or None when it is absent; 400 when it has another type.

    A bool is no number here: ``"drain_s": true`` is refused, not read as 1.0.
    """
    value = body.get(name)
    if value is None:
        return None
    kinds = kind if isinstance(kind, tuple) else (kind,)
    if not isinstance(value, kinds) or (isinstance(value, bool) and bool not in kinds):
        raise EngineError(400, "bad_request", f"{name} must be {what}")
    return value


def build(engine: Engine, stop: Callable[[], None]) -> web.Application:
    async def state(request: web.Request) -> web.StreamResponse:
        return web.json_response(engine.state())

    async def picks(request: web.Request) -> web.StreamResponse:
        return web.json_response(await asyncio.to_thread(engine.picks))

    async def switch(request: web.Request) -> web.StreamResponse:
        body = await _json(request)
        if not isinstance(body.get("pick"), str):
            raise EngineError(400, "bad_request", 'name the pick: {"pick": "27b"}')
        drain = _optional(body, "drain_s", (int, float), "a number of seconds")
        if drain is not None and not (math.isfinite(drain) and drain >= 0):
            raise EngineError(400, "bad_request", "drain_s must be a number of seconds")
        generation = _optional(body, "expected_generation", int, "a whole number")
        force = _optional(body, "force", bool, "true or false")
        initiator = _optional(body, "initiator", str, "text")
        reason = _optional(body, "reason", str, "text")
        result = await engine.switch(
            body["pick"],
            expected_generation=generation,
            drain_s=float(drain) if drain is not None else None,
            initiator=initiator or "user",
            reason=reason or "",
            force=bool(force),
        )
        return web.json_response(result)

    async def up(request: web.Request) -> web.StreamResponse:
        role = (await _json(request)).get("role") or "chat"
        if role not in engine.slots:
            raise EngineError(400, "bad_request", f"no role called {role!r}")
        await engine.warm(role)
        return web.json_response(engine.state())

    async def stop_route(request: web.Request) -> web.StreamResponse:
        stop()
        return web.json_response({"stopping": True})

    app = web.Application(middlewares=[errors])
    app.router.add_get("/state", state)
    app.router.add_get("/picks", picks)
    app.router.add_post("/switch", switch)
    app.router.add_post("/up", up)
    app.router.add_post("/stop", stop_route)
    return app
