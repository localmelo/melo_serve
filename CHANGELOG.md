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

### Changed

- Only the OpenAI API reaches the models through the front door. A path outside `/v1` that is not melo_serve's own
  (`/health`, `/melo/v1/state`) gets `404` `not_here`, as `/melo/*`, `/v1/models/*` and `/v1/memory/*` did, so
  a backend's own management API, such as oMLX's `/admin/*`, is no longer open to local clients. A `/v1` path with `.` or `..` segments is refused too, and so is any method but GET on `/v1/models`. (#2)
