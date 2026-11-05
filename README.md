# melo_serve

Run local model servers on a Mac behind one stable OpenAI-compatible address,
and switch models with one command.

**Status:** pre-alpha (v0.1). Backends: [Splash](https://github.com/incoai/splash)
and [oMLX](https://github.com/jundot/omlx).

- **One address.** `http://127.0.0.1:8321/v1` stays the same across switches. Chat goes to the chat model and `/v1/embeddings` to the embedding model, and `/v1/models` lists only what is served.
- **Memory on demand.** A model loads at its first request and frees its memory after 10 minutes without requests.
- **Safe switches.** `melo-serve use <pick>` lets requests in flight finish first. A model that fails to start is replaced by the one before. A model that would leave too little free memory is refused.
- **Always there.** `melo-serve install` starts it at login, and it restarts after a crash.

## Install

```bash
uv tool install git+https://github.com/localmelo/melo_serve
brew install incoai/tap/splash                     # Splash
brew tap jundot/omlx https://github.com/jundot/omlx && brew install jundot/omlx/omlx   # oMLX
melo-serve doctor                                  # what is installed
```

## Use

```bash
melo-serve list          # the picks on this Mac, with their fit
melo-serve use 27b       # serve Qwen3.8 27B for chat
melo-serve               # what is served
melo-serve stop          # stop the models and melo_serve
melo-serve install       # start at login
```

Then point any OpenAI-compatible client at `http://127.0.0.1:8321/v1`.

| Command | Does |
|---|---|
| `melo-serve` | what is served, and its address |
| `list [-a]` | the picks on this Mac with their fit; `-a` adds those not downloaded |
| `use <pick> [-n] [-y]` | switch; `-n` does not wait for requests in flight, `-y` loads even if it does not fit |
| `up [pick] [-f] [-p port]` | start melo_serve; `-f` stays in the foreground |
| `stop` | stop the models and melo_serve |
| `plan [--json]` | what a first run would install here and how big it is; reads only |
| `get <pick…> [--json] [--hf-home PATH] [--skip ITEM]` | install the backend of those picks if it is missing (oMLX, Splash), then download their models into the Hugging Face cache (oMLX models; Splash models come later); a backend or model already there is skipped |
| `doctor` | backends, the login item and memory |
| `install`, `uninstall` | add or remove the login item; `install` again changes nothing (`already installed, skipped`) unless its settings changed, such as `PATH` or `HF_HOME`, and then restarts melo_serve |

`-X key=value` changes a setting for one command, for example `-X port=8322`.

## Picks

A pick is a model on a backend, with a short name: `27b`, `flash-next`, `8b`, `4b`
and `embed` are built in. A `picks.toml` in `~/.cache/melo_serve` adds models of
your own, in the same format as [the built-in list](melo_serve/picks.toml). An entry with a built-in id replaces that pick whole, aliases and sizes included, because a changed source makes the built-in sizes wrong; give such an entry every field you want.

## Settings

`~/.cache/melo_serve/config.toml`; every key is optional:

```toml
port = 8321               # the address's port, on 127.0.0.1
idle_release = "10m"      # free a model's memory after this long without requests
hf_home = "~/models/hf"   # the Hugging Face cache (else $HF_HOME)

[picks]                   # what is served when nothing was served before
chat = "27b"
embedding = "embed"

[hooks]                   # shell commands around loads; non-zero refuses the load
before_load = "~/bin/before-load"
after_stop = "~/bin/after-stop"
```

## For client authors

### Front door

Base URL: `http://127.0.0.1:8321/v1` (the port is the `port` setting). No API key is needed. `Host` must be `127.0.0.1`, `localhost` or `[::1]` with that port.

| Path | Goes to |
|---|---|
| `POST /v1/embeddings` | the embedding model |
| any other `/v1/...` path except those below | the chat model |
| `GET /v1/models`, `GET /v1/models/status`, `GET /health`, `GET /melo/v1/state` | melo_serve itself |
| everything else, including `/v1/models/...` and `/v1/memory/...` | `404` `not_here` |

Bodies and replies pass through unchanged and streamed. A request's `model` may be a pick id, an alias or the served name (or be left out); it is rewritten to the served name.

`GET /v1/models` lists only what is served (a role with no pick is absent). Each entry is an OpenAI model object (`id` is the served name) plus:

| Field | Meaning |
|---|---|
| `vision` | the model reads pictures |
| `melo.role` | `chat` or `embedding` |
| `melo.backend` | `splash` or `omlx` |
| `melo.generation` | the current generation (see State) |
| `melo.loaded` | the model is in memory now; it loads at its first request |

`GET /v1/models/status` answers oMLX's shape, `{"models": [...]}`, one entry per served model: `id`, `model_alias`, `model_type` (`vlm`, `llm` or `embedding`), `loaded`, `engine` (the backend) and `source`.

`GET /health` answers `{"status": "ok", "service": "melo_serve", "version", "state"}`; `state` is `ready`, `switching`, `degraded` (a model failed) or `stopped`.

Errors are `{"error": {"type", "message", ...}}`:

| Status | `error.type` | When | Extra fields |
|---|---|---|---|
| 409 | `model_switched` | `model` names one that was served and is not any more | `current`: the served name now |
| 404 | `model_not_found` | `model` was never served here | `served`: the served names |
| 503 | `switching` | the role is being switched | `retry_after`, and a `Retry-After` header |
| 503 | `start_failed` | the model is down after a failed start; asking again within 30 s gets this again | `retry_after`, and a `Retry-After` header |
| 503 | `no_model`, `stopping` | no pick is chosen for the role; melo_serve is stopping | |
| 502 | `backend_unreachable` | the backend did not answer | |
| 404 | `not_here` | a path melo_serve does not serve | |
| 421 | `misdirected_request` | the `Host` is not this server | |
| 403 | `foreign_origin` | an `Origin` that is not this server (web pages of other sites) | |

A request that finds its model unloaded starts it, and can fail with the load errors listed under `POST /switch` (`not_installed`, `does_not_fit`, ...). Nothing is answered by a model the request did not name: after `model_switched`, retry with `current`.

### State

`~/.cache/melo_serve/state.json` and `GET /melo/v1/state` hold the same JSON. The file is replaced whole on every change, so a reader never sees half of it. It never holds a secret.

```json
{
  "schema": "melo_serve.state/1",
  "version": "0.3.0",
  "generation": 3,
  "status": "ready",
  "updated_at": "2026-11-05T02:30:37-05:00",
  "front_door": {"url": "http://127.0.0.1:8321/v1", "pid": 51898, "auth": "none"},
  "service": {"login": true},
  "roles": {
    "chat": {
      "pick": "qwen3.8-27b-splash", "short": "27b", "name": "Qwen3.8 27B",
      "model": "Qwen3.8-27B-Q4KM-splash", "backend": "splash", "backend_name": "Splash",
      "status": "ready", "loaded": true, "caps": ["tools", "vision", "reasoning"],
      "dims": null, "since": "2026-11-05T02:20:00-05:00", "error": "",
      "retired": ["Qwen3-8B-4bit"],
      "process": {"pid": 51920, "started_at": 1793860800.0}
    },
    "embedding": null
  },
  "switch": null
}
```

The values are illustrative. A role is `null` when no pick is chosen. The top-level `status` is `ready`, `switching`, `degraded` or `stopped`; a role's is `stopped`, `starting`, `ready` or `failed` (then `error` says why). `dims` is the embedding size (else `null`). `backend_name` is the display name of `backend` (`Splash`, `oMLX`). `retired` lists the served names that now get `model_switched`. `switch` is `null`, or `{"pick", "role", "phase", "load_s", "since"}` while a switch runs; `phase` is `checking`, `draining`, `stopping`, `loading` or `rolling back`.

`generation` starts at 0, survives restarts, and goes up by one each time a switch finishes loading a model. A switch to what is already served and ready, a failed switch and a load on demand leave it as it is.

### Plan

`melo-serve plan --json` prints what a first run would set up on this Mac, so a client can show one list and ask for one OK. It only reads: no network, no writes, nothing is started or installed. Without `--json` it prints a short summary.

```json
{
  "schema": "melo_serve.plan/1",
  "machine": {"chip": "Apple M5 Max", "arch": "arm64", "ram_gb": 128, "macos": "27.2", "macos_build": "26B123",
              "free_gb": 83.0, "hf_home": "~/models/huggingface", "hf_home_from": "env", "homebrew": true},
  "supported": true, "reason": "", "reason_hint": "",
  "pick": {"id": "qwen3.8-27b-splash", "name": "Qwen3.8 27B", "backend": "splash", "fit": "estimate"},
  "items": [
    {"id": "backend:omlx", "kind": "backend", "name": "oMLX 0.7.3", "via": "dmg", "size_bytes": 881239063,
     "dest": "~/Applications/oMLX.app", "state": "todo", "sha256": "2acfde41…", "file": "oMLX-0.7.3-macos26-27.dmg"},
    {"id": "model:qwen3.8-27b-splash", "kind": "model", "name": "Qwen3.8 27B", "via": "huggingface", "size_bytes": 22764000000,
     "dest": "~/models/huggingface", "state": "partial", "role": "chat", "backend": "splash", "cached_bytes": 14100000000}
  ],
  "download_bytes": 9545239063, "enough_disk": true,
  "alternatives": [{"id": "qwen3-8b-omlx", "name": "Qwen3 8B", "backend": "omlx"}]
}
```

[`tests/fixtures/plan.v1.json`](tests/fixtures/plan.v1.json) is a complete example, and a test keeps the real output to its keys and types.

- `machine.hf_home_from` says where the cache folder comes from: `config` (the `hf_home` setting), `env` (`HF_HOME`) or `default`. `free_gb` is the free space on the cache's volume. Paths are shown with `~`.
- `supported` is `false` when no pick can run: an Intel Mac, less memory than the smallest model needs, a macOS no backend supports, or no pick whose backend can run here. `reason` then says why, in one line, and `pick` is `null`. `reason_hint` is set when something would help: with no Homebrew, the Splash models drop out and it says installing Homebrew would allow them.
- `pick` is the first chat pick, in `picks.toml` order, that `get` can finish (for now, a Splash pick only when its model is already on this Mac), that fits the Mac's memory (`min_ram_gb`) and whose backend can run on this macOS (Splash needs macOS 26.4 or later and Homebrew; oMLX needs macOS 15, or 26 and later). It goes by the Mac, not by how much memory is free at this moment, so a busy Mac is not sent a smaller model for good. `pick.fit` is this moment's verdict from the fit check (`fits`, `estimate`, `alone`, `too_big` or `unknown`), for information only: the load step checks again. `alternatives` are the other chat picks that pass the same rules.
- `items` are the pick's backend, the embedding model's backend (when it differs), the chat model and the embedding model, each with a `state` of `done`, `partial` (`cached_bytes` is what is already in the cache) or `todo`. `size_bytes` of a model is the catalog's `disk_gb`.
- `download_bytes` adds up the items not `done`, less `cached_bytes`. `enough_disk` is `free_gb` of the cache's volume at least that plus 5 GiB.

### Get

`melo-serve get <pick…> --json` sets up those picks (an id, alias or served name from `melo-serve list -a`) and prints one JSON object per line. Without `--json` it prints a progress line and the same closing lines as text. It installs a pick's backend first. oMLX: the pinned DMG for this macOS is downloaded into melo_serve's `downloads` folder, its size and sha256 are checked against the values pinned in melo_serve (a file that does not match is deleted), the app is copied to `~/Applications/oMLX.app.new`, must pass `codesign --verify`, and only then takes its place as `~/Applications/oMLX.app`; no `sudo`, no password. Splash: `brew install incoai/tap/splash` with Homebrew's auto-update off; Homebrew must already be there (its output goes to `logs/brew.log` in melo_serve's folder). A backend that is already installed, at any version, is left alone. `--skip ITEM` leaves an item out of the run and can be repeated: `backend:<key>` (`backend:omlx`, `backend:splash`) or `model:<pick id>`; anything that is not an item of the run is refused. A Splash pick's model is not downloaded by `get` yet, so a Splash pick whose model is not on this Mac is still refused with exit 2 unless `model:<id>` is skipped. Model files go into `<hf_home>/hub`, the Hugging Face cache, never a project folder; Xet, the Hub's transfer layer, keeps its chunk cache beside it in `<hf_home>/xet`, which the free-space check does not count. `HF_HOME` is honoured, `--hf-home PATH` picks another folder, and `~` and relative paths are made absolute. `get` turns the Hub's telemetry off for itself (`HF_HUB_DISABLE_TELEMETRY=1`). Run it again after any stop or failure: a DMG download that was stopped continues from where it stopped (with a `Range` request; a server that ignores it starts the file again), and finished model files are kept and a model that is complete is skipped, but a model file cut short starts again, because the Hub does not resume inside a file across runs.

```json
{"item": "config:hf_home", "state": "done", "line": "saved hf_home = ~/models/huggingface in ~/.cache/melo_serve/config.toml"}
{"item": "model:qwen3-8b-omlx", "state": "running", "done_bytes": 310000000, "total_bytes": 4600000000, "line": "downloading Qwen3 8B"}
{"item": "model:qwen3-8b-omlx", "state": "done", "done_bytes": 4600000000, "total_bytes": 4600000000, "line": "downloaded Qwen3 8B into ~/models/huggingface"}
```

- `item` is `backend:<key>` or `model:<pick id>`, or `config:hf_home`, `disk`, `cache:hf_home` and `get` for what is not one model. `state` is `running`, `done`, `skipped` or `failed`; `done_bytes` and `total_bytes` come with the `running` and `done` events of a model and of a DMG download (a backend's other `running` events are plain steps, such as `checked the size and sha256 of the download`). `line` is the sentence to show. A terminal gets the same text, except that skipped and failed lines start with the model's or backend's name (`Qwen3 8B: already downloaded, skipped`, `oMLX: already installed (0.7.3), skipped`), since a run can name several.
- The closing lines are a closed set. A model: `downloaded <name> into <path>`, `already downloaded, skipped`, and `failed: <reason>; nothing was downloaded`. A backend: `installed <version> into <path>` (`~/Applications/oMLX.app`, `/opt/homebrew`), `already installed (<version>), skipped`, `skipped (you chose to skip it)`, and `failed: <reason>; nothing was installed`; a failed DMG download whose part is kept says to run the same command again to continue. When a failure leaves finished files in the cache, the end of the line says so and says to run the same command again. A stop ends with `Stopped at 1.0 GB / 4.6 GB. The finished files are kept; run the same command again to download the rest.` (state `failed`, exit code 130); a stop while a backend installs says `Stopped while installing oMLX. Run the same command again to continue.` and keeps the DMG part; a second SIGINT or SIGTERM exits at once with that same line. Paths in the lines are shown with `~`.
- Exit codes: `0` done, `1` failed (a backend install, a download, the size check or saving `hf_home` failed), `2` cannot start (an unknown pick, an unknown `--skip` item, a Splash model not downloaded yet, `HF_HUB_CACHE` set to somewhere other than `<hf_home>/hub`, a backend that cannot be installed here: an Intel Mac, Splash without Homebrew, a macOS too old for it, an `oMLX.app` that is in the way and not a working oMLX, a `~/Applications` that cannot be written, or a `command` in `config.toml` that is gone, or less free space than the download plus 5 GiB on the cache's volume, where the DMG and the app count for the home volume; nothing is installed or downloaded, though melo_serve's own folder and `get.lock` were made first), `75` another `get` is running (`another download is running (pid N)`), `130` stopped by SIGINT, SIGTERM or SIGHUP. A command that is running gets SIGTERM and 30 s to clean up, so `brew` can undo a half-installed formula; an image a killed run left attached is detached by the next run.
- When `config.toml` has no `hf_home`, `get` saves the folder it used there, since the login item has no shell environment; the other lines of the file stay as they are. `--hf-home PATH` saves that folder instead.
- The size check asks Hugging Face for the exact size, before the first byte, so `get` needs the network, and nothing else here does. Only one `get` runs at a time (`get.lock` in melo_serve's folder).

### Control socket

`~/.cache/melo_serve/control.sock` is plain HTTP over a Unix socket that only your user can open. There is no token and no Host check. Errors have the shape above.

```python
import httpx, pathlib

sock = pathlib.Path.home() / ".cache/melo_serve/control.sock"
client = httpx.Client(
    transport=httpx.HTTPTransport(uds=str(sock)), base_url="http://melo_serve"
)
```

| Call | Body | Answers |
|---|---|---|
| `GET /state` | | the state JSON |
| `GET /picks` | | `{"memory": {"total_gb", "free_pct"} or null, "picks": [...]}` |
| `POST /switch` | see below | see below |
| `POST /up` | `{"role": "chat"}` (`chat` or `embedding`; default `chat`) | loads that role's pick if it is not ready, then the state JSON; `400` `bad_request` for another role |
| `POST /stop` | | `{"stopping": true}`; stops the models and melo_serve |

One pick in `GET /picks`:

| Field | Meaning |
|---|---|
| `id`, `short`, `name`, `role`, `backend`, `model` | identity; `model` is the served name, `short` the name to type |
| `summary`, `disk_gb`, `mem_gb`, `load_s`, `decode_tps`, `caps` | description and sizes (a number may be `null`) |
| `backend_name` | the backend's display name |
| `current` | this is the role's pick now |
| `problem` | why it cannot load (not downloaded, ...), or `""` |
| `fit` | `{"verdict", "need_gb", "after_pct", "measured", "reason"}`; `verdict` is `fits`, `alone`, `estimate`, `too_big` or `unknown`, and only `too_big` is refused |

`POST /switch` answers after the new model is ready, which can take minutes, so pass `timeout=None` (httpx defaults to 5 s). Fields:

| Field | Meaning |
|---|---|
| `pick` | required: a pick id, alias or served name |
| `expected_generation` | refuse with `stale_generation` unless it equals the current `generation` |
| `drain_s` | seconds to wait for requests in flight (default: the `drain` setting, 2 minutes); `0` does not wait |
| `force` | load even if it does not fit; the `before_load` hook can still refuse |
| `initiator`, `reason` | text for the `before_load` hook and the log (default `user` and empty) |

It answers `{"changed", "generation", "role", "pick", "model", "fit"}`; `changed` is `false` (and `fit` `null`) when the pick is already served and ready. If the new model fails to start, the previous one is started again.

| Status | `error.type` | Extra fields |
|---|---|---|
| 400 | `bad_request`: no `pick`, a field of the wrong type, or the body is not a JSON object | |
| 404 | `unknown_pick` | `suggestions` |
| 409 | `busy`: another switch is running | `switch`: its progress |
| 409 | `stale_generation` | `generation`: the current one |
| 409 | `does_not_fit` | `fit` |
| 409 | `not_installed`, `unknown_backend`, `refused` (the `before_load` hook said no) | |
| 502 | `start_failed` | `log`, `log_path` |
| 503 | `stopping` | |

## Development

```bash
uv sync
uv run pytest -n auto -q     # the whole suite, in parallel
uv run pytest --testmon -q   # only the tests a change affects
uv run ruff check . && uv run ruff format --check . && uv run mypy melo_serve
```

The tests use fake backends: no model loads and no fixed port is bound.
Changes are listed in [CHANGELOG.md](CHANGELOG.md).

## License

MIT © The localmelo Authors. See [LICENSE](LICENSE).
