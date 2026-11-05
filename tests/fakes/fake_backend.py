"""A stand-in for Splash or oMLX, started with the command lines the adapters build.

    fake_backend.py --kind splash serve --model M --port P --served-model-name N ...
    fake_backend.py --kind omlx serve          (OMLX_PORT, OMLX_MODEL_DIR, OMLX_API_KEY)

It requires the key like the real ones, lists its model, answers chat (JSON, or
streamed in chunks as the real ones stream) and embeddings, and frees memory on
the backend's own route.
FAKE_RECORD names a file that gets one JSON line per request; FAKE_FAIL=start
exits at once; FAKE_CHUNKS and FAKE_CHUNK_S shape a streamed reply.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def options() -> tuple[str, int, list[str], str]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", required=True)
    parser.add_argument("command")
    parser.add_argument("--port", type=int)
    parser.add_argument("--served-model-name", action="append", default=[])
    args, _ = parser.parse_known_args()
    if args.kind == "splash":
        return (
            args.kind,
            args.port,
            args.served_model_name,
            os.environ.get("SPLASH_API_KEY", ""),
        )
    models = sorted(p.name for p in Path(os.environ["OMLX_MODEL_DIR"]).iterdir())
    return (
        args.kind,
        int(os.environ["OMLX_PORT"]),
        models,
        os.environ.get("OMLX_API_KEY", ""),
    )


KIND, PORT, MODELS, KEY = options()
RECORD = os.environ.get("FAKE_RECORD", "")
LOCK = threading.Lock()


def record(**entry: object) -> None:
    if RECORD:
        with LOCK, open(RECORD, "a", encoding="utf-8") as handle:
            handle.write(json.dumps({"kind": KIND, "pid": os.getpid(), **entry}) + "\n")


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args: object) -> None:
        pass

    def reply(self, status: int, payload: object) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def chunk(self, data: bytes) -> None:
        self.wfile.write(b"%x\r\n%s\r\n" % (len(data), data))
        self.wfile.flush()

    def authorized(self) -> bool:
        if KEY and self.headers.get("Authorization") != f"Bearer {KEY}":
            self.reply(401, {"error": {"message": "API key required"}})
            return False
        return True

    def do_GET(self) -> None:
        if self.path in ("/health", "/ready"):
            return self.reply(200, {"status": "ok"})
        if self.path == "/v1/models" and self.authorized():
            return self.reply(
                200, {"data": [{"id": m, "owned_by": KIND} for m in MODELS]}
            )
        self.reply(404, {"error": {"message": "not found"}})

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length) or b"{}")
        record(
            path=self.path,
            model=body.get("model"),
            auth=self.headers.get("Authorization"),
        )
        if not self.authorized():
            return
        if self.path in ("/v1/memory/release", f"/v1/models/{MODELS[0]}/unload"):
            return self.reply(200, {"released": True})
        if body.get("model") not in MODELS:
            return self.reply(
                404, {"error": {"message": f"unknown model {body.get('model')}"}}
            )
        if self.path == "/v1/embeddings":
            inputs = body.get("input")
            count = len(inputs) if isinstance(inputs, list) else 1
            data = [{"index": i, "embedding": [0.5] * 8} for i in range(count)]
            return self.reply(200, {"data": data, "model": MODELS[0]})
        if self.path != "/v1/chat/completions":
            return self.reply(404, {"error": {"message": "not found"}})
        if not body.get("stream"):
            message = {"role": "assistant", "content": "OK"}
            return self.reply(
                200, {"model": MODELS[0], "choices": [{"index": 0, "message": message}]}
            )
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        for i in range(int(os.environ.get("FAKE_CHUNKS", "3"))):
            delta = {"choices": [{"index": 0, "delta": {"content": f"t{i}"}}]}
            self.chunk(f"data: {json.dumps(delta)}\n\n".encode())
            time.sleep(float(os.environ.get("FAKE_CHUNK_S", "0")))
        self.chunk(b"data: [DONE]\n\n")
        self.chunk(b"")


def main() -> None:
    if os.environ.get("FAKE_FAIL") == "start":
        print("failing on purpose", flush=True)
        sys.exit(3)
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    server.daemon_threads = True

    def stop(*_: object) -> None:
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    print(f"fake {KIND} on {PORT}: {MODELS}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
