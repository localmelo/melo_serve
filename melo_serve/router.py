"""The front door: one OpenAI-compatible address in front of the backends.

``/v1/embeddings`` goes to the embedding role and every other path to the chat
role, as the router it replaces did. Bodies and replies pass through unchanged,
streamed, except that a request's ``model`` is set to the role's served name (an
alias or stray spaces still reach the right model) and the backend's key is
added. ``/v1/models`` lists only what is served.

aiohttp both serves and relays: in front of one fake backend a minimal aiohttp
proxy costs 0.31 ms of CPU per request, as much as a plain threaded proxy, where
Starlette, uvicorn and httpx cost 0.84 ms.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any

import aiohttp
from aiohttp import web
from yarl import URL

from melo_serve import __version__
from melo_serve.engine import Engine, EngineError

Handler = Callable[[web.Request], Awaitable[web.StreamResponse]]

MAX_BODY = 256 * 1024 * 1024
HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te",
    "trailer", "trailers", "transfer-encoding", "upgrade",
}  # fmt: skip
DROP_REQUEST = HOP | {"host", "content-length", "authorization", "x-api-key"}
DROP_RESPONSE = HOP | {"date", "server"}
# Headers the client did not send stay unsent: with an Accept-Encoding added
# here, a backend could compress a reply for a client that never asked for it.
NOT_ADDED = ("User-Agent", "Accept", "Accept-Encoding", "Content-Type")


def error(exc: EngineError) -> web.Response:
    retry = exc.extra.get("retry_after")
    headers = {"Retry-After": str(int(retry))} if retry else None
    return web.json_response(exc.to_json(), status=exc.status, headers=headers)


@web.middleware
async def errors(request: web.Request, handler: Handler) -> web.StreamResponse:
    """An :class:`EngineError` becomes its JSON error reply."""
    try:
        return await handler(request)
    except EngineError as exc:
        return error(exc)


def guard(port: int) -> Any:
    """Refuses a Host that is not this server (DNS rebinding) and other sites' pages."""
    hosts = {f"{name}:{port}" for name in ("127.0.0.1", "localhost", "[::1]")}
    origins = {f"http://{host}" for host in hosts}

    @web.middleware
    async def check(request: web.Request, handler: Handler) -> web.StreamResponse:
        if request.headers.get("Host", "").lower() not in hosts:
            raise EngineError(421, "misdirected_request", "the Host is not this server")
        origin = request.headers.get("Origin")
        if origin is not None and origin.lower() not in origins:
            raise EngineError(
                403, "foreign_origin", "web pages of other sites are refused"
            )
        return await handler(request)

    return check


async def _body(request: web.Request) -> bytes:
    try:
        return await request.read()
    except web.HTTPRequestEntityTooLarge:
        raise EngineError(
            413, "too_large", f"a request may hold at most {MAX_BODY >> 20} MB"
        ) from None


async def proxy(engine: Engine, request: web.Request, role: str) -> web.StreamResponse:
    body = await _body(request)
    payload: Any = None
    if body[:1] == b"{" or "json" in request.content_type:
        try:
            payload = json.loads(body)
        except ValueError:
            payload = None
    named = payload.get("model") if isinstance(payload, dict) else None
    upstream = await engine.acquire(role, named if isinstance(named, str) else None)
    try:
        if isinstance(payload, dict) and payload.get("model") != upstream.model:
            payload["model"] = upstream.model
            body = json.dumps(payload, ensure_ascii=False).encode()
        headers = [
            (k, v) for k, v in request.headers.items() if k.lower() not in DROP_REQUEST
        ]
        headers.extend(upstream.headers.items())
        # The path and query as the client sent them: decoding would turn %2F,
        # %3F or %23 into separators, and the backend would see another URL.
        url = URL(upstream.base + request.raw_path, encoded=True)
        return await _relay(engine.http, request, url, headers, body, role)
    finally:
        engine.done(role)


async def _relay(
    http: aiohttp.ClientSession,
    request: web.Request,
    url: URL,
    headers: list[tuple[str, str]],
    body: bytes,
    role: str,
) -> web.StreamResponse:
    """Streams the backend's reply as it comes.

    A client that disconnects cancels this handler, which closes the upstream
    connection, so the backend stops writing a reply that nobody reads.
    """
    try:
        upstream = await http.request(
            request.method, url, headers=headers, data=body,
            allow_redirects=False, skip_auto_headers=NOT_ADDED,
        )  # fmt: skip
    except (aiohttp.ClientError, OSError) as exc:
        raise EngineError(
            502, "backend_unreachable", f"the {role} backend did not answer: {exc}"
        ) from None
    async with upstream:
        reply = web.StreamResponse(status=upstream.status, reason=upstream.reason)
        for key, value in upstream.headers.items():
            if key.lower() not in DROP_RESPONSE:
                reply.headers.add(key, value)
        await reply.prepare(request)
        try:
            async for chunk in upstream.content.iter_any():
                await reply.write(chunk)
        except (aiohttp.ClientError, OSError):
            # The backend stopped mid-reply, or the client left: end the
            # connection, so the client sees a cut reply, not a complete one.
            if request.transport is not None:
                request.transport.close()
            return reply
        await reply.write_eof()
    return reply


def build(engine: Engine) -> web.Application:
    async def health(request: web.Request) -> web.StreamResponse:
        return web.json_response(
            {
                "status": "ok",
                "service": "melo_serve",
                "version": __version__,
                "state": engine.status,
            }
        )

    async def models(request: web.Request) -> web.StreamResponse:
        data = [
            {
                "id": pick.served_name,
                "object": "model",
                "created": 0,
                "owned_by": "melo_serve",
                "melo": {
                    "role": slot.role,
                    "backend": pick.backend,
                    "generation": engine.generation,
                    "loaded": slot.loaded and slot.ready(),
                },
            }
            for slot, pick in engine.served()
        ]
        return web.json_response({"object": "list", "data": data})

    async def state(request: web.Request) -> web.StreamResponse:
        return web.json_response(engine.state())

    async def managed(request: web.Request) -> web.StreamResponse:
        raise EngineError(404, "not_here", "melo_serve manages models: use melo-serve")

    async def embeddings(request: web.Request) -> web.StreamResponse:
        return await proxy(engine, request, "embedding")

    async def chat(request: web.Request) -> web.StreamResponse:
        return await proxy(engine, request, "chat")

    app = web.Application(
        middlewares=[errors, guard(engine.port)], client_max_size=MAX_BODY
    )
    app.router.add_get("/health", health)
    app.router.add_get("/v1/models", models)
    app.router.add_get("/melo/v1/state", state)
    app.router.add_post("/v1/embeddings", embeddings)
    for prefix in ("/v1/models/", "/v1/memory/", "/melo/"):
        app.router.add_route("*", prefix + "{rest:.*}", managed)
    app.router.add_route("*", "/{rest:.*}", chat)  # registered last: tried last
    return app
