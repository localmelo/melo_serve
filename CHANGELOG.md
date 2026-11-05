# Changelog

All notable changes to melo_serve are documented in this file. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

melo_serve has no release yet.

## [Unreleased]

### Added

- One address for local models on a Mac. `http://127.0.0.1:8321/v1` serves a chat model and an embedding model on
  Splash or oMLX, and stays the same when `melo-serve use <pick>` switches them. A switch lets requests in flight
  finish; a request for a model no longer served gets `409` with the current one, and one that arrives during a
  switch gets `503` with `Retry-After`. A model loads at its first request and frees its memory after `idle_release`
  (10 minutes) without requests. A pick that would leave too little free memory is refused, and one that fails to
  start is replaced by the one before. `melo-serve install` starts melo_serve at login.

- `GET /v1/models` says which model reads pictures (`"vision": true` or `false`), and `GET /v1/models/status` lists the
  served models and whether each is loaded, in oMLX's shape, so a client such as localmelo can turn pictures on for
  the models that read them. Each role in the state names its backend for display (`backend_name`). (#4)

- `melo-serve plan [--json]` shows what a first run would set up on this Mac: the chip, memory, macOS, free space
  where the Hugging Face cache is, the recommended chat model with its backend, the embedding model, what is already
  there, the size of the download and whether the disk has room. It only reads. The JSON (`melo_serve.plan/1`) is
  described in the README for client authors. (#10)

- `melo-serve get <pick…> [--json] [--hf-home PATH]` downloads the models of some picks into the Hugging Face cache
  (`<hf_home>/hub`, never a project folder; `HF_HOME` is respected). A model already there is skipped, so every run is safe to repeat. It checks the
  exact size with Hugging Face and refuses, downloading nothing, when the cache's volume would have less than 5 GiB left.
  It saves `hf_home` in `config.toml` when the file has none (or sets it from `--hf-home`), keeping the rest of the
  file. One `get` runs at a time (a second says `another download is running (pid N)` and exits 75), and Ctrl-C or
  SIGTERM stops it with exit 130 (a second one exits at once), keeping the files it had finished; a file cut short starts again, as the Hub does not resume inside a file across runs. `get` turns the Hub's telemetry off for itself, and `~` and relative cache paths are made absolute. `--json` prints the progress as JSON lines for
  clients. melo_serve now depends on `huggingface_hub` (2.4 up to 3, with its transfer layer `hf-xet` and a few more packages), imported only by `get`. The version is 0.2.0, the first
  with `plan` and `get`. (#12)

- `melo-serve get <pick…>` now installs a pick's backend before it downloads the model, and `--skip ITEM`
  (`backend:<key>` or `model:<id>`, repeatable) leaves an item out. oMLX comes from its pinned DMG into
  `~/Applications` (size and sha256 checked, the copy verified with `codesign`, no `sudo`, no password); Splash comes
  from `brew install incoai/tap/splash`, and Homebrew must already be there. A backend already installed, at any
  version, is left alone. A stopped DMG download continues from where it stopped. The free-space check also counts the
  DMG and the app on the home volume, and the JSON has a new `item`, `backend:<key>`. Each model item of the
  plan names its `backend`. (#14)

- `melo-serve plan --pick PICK` plans another chat pick, so a client can show the true sizes and backends of an
  alternative before anyone says yes. (#16)

- `melo-serve get` now downloads the models of Splash picks too, and `plan` no longer holds a Splash pick back until its
  model is there. A pick lists what to download in `picks.toml` (`[[pick.hub]]` tables with a `repo` and, optionally,
  `files` patterns; no `files` means every file), so the other quantizations in a repo are not downloaded. The built-in
  Splash picks list their model, multimodal projector and DFlash2 draft. The first load of a Splash model prepares it
  from the Hugging Face cache with links and no second copy, and the model counts as downloaded when all its listed files
  are in the cache. A Splash pick in your own `picks.toml` without `[[pick.hub]]` is not downloaded by `get`, and is
  refused unless its model is in Splash's models folder. The version is 0.4.0. (#18)

### Changed

- `POST /switch` on the control socket refuses a field of the wrong type with `400` `bad_request` instead of guessing:
  `"force": "false"` used to force a load. (#4)

- Only the OpenAI API reaches the models through the front door. A path outside `/v1` that is not melo_serve's own
  (`/health`, `/melo/v1/state`) gets `404` `not_here`, as `/melo/*`, `/v1/models/*` and `/v1/memory/*` did, so
  a backend's own management API, such as oMLX's `/admin/*`, is no longer open to local clients. A `/v1` path with `.` or `..` segments is refused too, and so is any method but GET on `/v1/models`. (#2)

### Fixed

- `melo-serve install` is safe to run again. It no longer restarts melo_serve (which unloaded the model) when the login item is already installed and unchanged: it says `already installed, skipped`. A new install says `installed ai.localmelo.melo_serve (starts at login)`, and a changed one (a new `PATH` or `HF_HOME`) says the daemon restarted. (#8)

- A client that connects to the control socket and leaves at once no longer leaves a traceback in the daemon log. (#6)
