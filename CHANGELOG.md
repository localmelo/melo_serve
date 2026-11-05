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

- `melo-serve plan [--json]` shows what a first run would set up on this Mac: the chip, memory, macOS, free space where the Hugging Face cache is, the recommended chat model with its backend, the embedding model, what is already there, the size of the download and whether the disk has room. It only reads. The JSON (`melo_serve.plan/1`) is described in the README for client authors. (#10)

### Changed

- `POST /switch` on the control socket refuses a field of the wrong type with `400` `bad_request` instead of guessing:
  `"force": "false"` used to force a load. (#4)

- Only the OpenAI API reaches the models through the front door. A path outside `/v1` that is not melo_serve's own
  (`/health`, `/melo/v1/state`) gets `404` `not_here`, as `/melo/*`, `/v1/models/*` and `/v1/memory/*` did, so
  a backend's own management API, such as oMLX's `/admin/*`, is no longer open to local clients. A `/v1` path with `.` or `..` segments is refused too, and so is any method but GET on `/v1/models`. (#2)

### Fixed

- `melo-serve install` is safe to run again. It no longer restarts melo_serve (which unloaded the model) when the login item is already installed and unchanged: it says `already installed, skipped`. A new install says `installed ai.localmelo.melo_serve (starts at login)`, and a changed one (a new `PATH` or `HF_HOME`) says the daemon restarted. (#8)

- A client that connects to the control socket and leaves at once no longer leaves a traceback in the daemon log. (#6)
