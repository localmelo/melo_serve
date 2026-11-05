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
| `doctor` | backends, the login item and memory |
| `install`, `uninstall` | add or remove the login item |

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

- A request that names a model no longer served gets `409` with `{"error": {"type": "model_switched", "current": "<id>"}}`; nothing is answered by a model the client did not ask for.
- During a switch, new requests for that model get `503` with `Retry-After`.
- What is served, and a `generation` that counts the switches, are in `~/.cache/melo_serve/state.json` and at `GET /melo/v1/state`.
- Only `/v1` paths reach the models (`/v1/embeddings` the embedding model, the rest the chat model); any other path that is not melo_serve's own gets `404` `not_here`, so a backend's admin API stays closed.
- Control (switch, start, stop) goes through a Unix socket that only your user can open: `~/.cache/melo_serve/control.sock`.

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
