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
