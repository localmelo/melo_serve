"""The daemon end to end, with fake backends started the way real ones would be."""

from __future__ import annotations

import asyncio
import os
import signal
from pathlib import Path

import httpx
import pytest

from melo_serve import cli
from melo_serve.client import Client, ControlError
from tests.conftest import records, running


async def test_starts_on_first_request_and_relays(tmp_path: Path) -> None:
    async with running(tmp_path) as (engine, front):
        listed = (await front.get("/v1/models")).json()["data"]
        assert [m["id"] for m in listed] == ["chat-a", "embed-a"]
        # Nothing starts before its first request.
        assert not listed[0]["melo"]["loaded"]

        reply = await front.post(
            "/v1/chat/completions", json={"model": " a ", "messages": []}
        )
        assert (
            reply.status_code == 200
            and reply.json()["choices"][0]["message"]["content"] == "OK"
        )
        async with front.stream(
            "POST", "/v1/chat/completions", json={"model": "chat-a", "stream": True}
        ) as stream:
            text = "".join([chunk async for chunk in stream.aiter_text()])
        assert text.count("data: ") == 9 and text.endswith("[DONE]\n\n")
        vector = (
            await front.post("/v1/embeddings", json={"model": "embed-a", "input": "x"})
        ).json()
        assert len(vector["data"][0]["embedding"]) == 8
        await front.post("/v1/a%2Fb?q=%3F", json={"model": "chat-a"})

        seen = records(tmp_path / "record.jsonl")
        assert "/v1/a%2Fb?q=%3F" in {r["path"] for r in seen}  # as sent
        keys = {
            f"Bearer {slot.instance.key}"
            for slot in engine.slots.values()
            if slot.instance
        }
        # The router adds each backend's own key and names the served model.
        assert {r["auth"] for r in seen} == keys
        chat = {r["model"] for r in seen if r["path"] == "/v1/chat/completions"}
        assert chat == {"chat-a"}
        state = (await front.get("/melo/v1/state")).json()
        assert (
            state["roles"]["chat"]["status"] == "ready"
            and state["roles"]["embedding"]["dims"] == 8
        )


async def test_switch_retires_the_old_model(tmp_path: Path) -> None:
    async with running(tmp_path) as (engine, front):
        await front.post("/v1/chat/completions", json={"model": "chat-a"})
        result = await asyncio.to_thread(Client().switch, "b")
        assert (result["changed"], result["model"], result["generation"]) == (
            True,
            "chat-b",
            1,
        )

        old = await front.post("/v1/chat/completions", json={"model": "chat-a"})
        assert old.status_code == 409 and old.json()["error"]["current"] == "chat-b"
        assert (
            await front.post("/v1/chat/completions", json={"model": "nope"})
        ).status_code == 404
        assert (
            await front.post("/v1/chat/completions", json={"model": "chat-b"})
        ).status_code == 200


async def test_refused_or_failed_switches_keep_the_model(tmp_path: Path) -> None:
    async with running(tmp_path) as (engine, front):
        await front.post("/v1/chat/completions", json={"model": "chat-a"})
        client = Client()
        with pytest.raises(ControlError) as refused:
            await asyncio.to_thread(client.switch, "huge")
        assert refused.value.kind == "does_not_fit"
        with pytest.raises(ControlError) as failed:
            await asyncio.to_thread(client.switch, "broken")
        assert failed.value.kind == "start_failed" and "exited with code 3" in str(
            failed.value
        )

        chat = (await asyncio.to_thread(client.state))["roles"]["chat"]
        assert (chat["pick"], chat["status"], engine.generation) == (
            "chat-a",
            "ready",
            0,
        )
        assert (
            await front.post("/v1/chat/completions", json={"model": "chat-a"})
        ).status_code == 200


async def test_a_switch_waits_for_replies_and_turns_new_requests_away(
    tmp_path: Path,
) -> None:
    async with running(tmp_path) as (engine, front):

        async def long_reply() -> str:
            async with front.stream(
                "POST", "/v1/chat/completions", json={"model": "a", "stream": True}
            ) as reply:
                return "".join([chunk async for chunk in reply.aiter_text()])

        reply = asyncio.create_task(long_reply())
        slot = engine.slots["chat"]
        while not slot.inflight:
            await asyncio.sleep(0.02)
        switch = asyncio.create_task(asyncio.to_thread(Client().switch, "b"))
        while not slot.draining:
            await asyncio.sleep(0.02)
        turned_away = await front.post("/v1/chat/completions", json={"model": "a"})
        assert turned_away.status_code == 503 and turned_away.headers["retry-after"]
        assert (await reply).endswith("[DONE]\n\n")  # the drain let it finish
        assert (await switch)["model"] == "chat-b"


async def test_guard_refuses_other_hosts_and_sites(tmp_path: Path) -> None:
    async with running(tmp_path) as (engine, front):
        assert (
            await front.get("/v1/models", headers={"Host": "evil.example"})
        ).status_code == 421
        assert (
            await front.get("/v1/models", headers={"Origin": "http://evil.example"})
        ).status_code == 403
        assert (await front.get("/health")).json()["state"] == "ready"


async def test_front_door_refuses_everything_but_the_openai_api(tmp_path: Path) -> None:
    async with running(tmp_path) as (engine, front):
        for path in ("/admin/api/models", "/melo/x", "/v1/models/a/unload", "/props"):
            reply = await front.get(path)
            assert reply.status_code == 404
            assert reply.json()["error"]["type"] == "not_here"


async def test_idle_release_then_restart_after_a_crash(tmp_path: Path) -> None:
    async with running(tmp_path, idle_release="1s") as (engine, front):
        await front.post("/v1/chat/completions", json={"model": "a"})
        slot = engine.slots["chat"]
        for _ in range(60):
            if not slot.loaded:
                break
            await asyncio.sleep(0.1)
        assert not slot.loaded
        assert any(
            r["path"] == "/v1/memory/release"
            for r in records(tmp_path / "record.jsonl")
        )

        # The backend dies mid-reply: the client sees a cut reply, not a whole one.
        assert slot.instance is not None
        body = {"model": "a", "stream": True}
        async with front.stream("POST", "/v1/chat/completions", json=body) as stream:
            lines = stream.aiter_lines()
            await anext(lines)
            os.kill(slot.instance.child.pid, signal.SIGKILL)
            with pytest.raises(httpx.RemoteProtocolError):
                async for _ in lines:
                    pass
        for _ in range(50):
            if (await front.get("/health")).json()["state"] == "degraded":
                break
            await asyncio.sleep(0.05)
        assert (await front.get("/health")).json()["state"] == "degraded"
        assert (
            await front.post("/v1/chat/completions", json={"model": "a"})
        ).status_code == 200


async def test_cli_switches_lists_and_stops(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    async with running(tmp_path) as (engine, front):
        assert await asyncio.to_thread(cli.main, ["use", "b"]) == 0
        assert await asyncio.to_thread(cli.main, ["list"]) == 0
        assert await asyncio.to_thread(cli.main, []) == 0
        out = capsys.readouterr().out
        assert (
            "Now serving chat-b for chat" in out and "● b" in out and "· ready" in out
        )
        assert await asyncio.to_thread(cli.main, ["stop"]) == 0
    assert "melo_serve stopped." in capsys.readouterr().out
