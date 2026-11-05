"""Splash: one model per process, with its DFlash2 draft; no embeddings.

The process serves the model it was started with, so a switch restarts it. It
answers to the served name (``--announce-served-name``), requires the key on every
route but ``/health`` and ``/ready``, and frees its weights after
``--idle-release`` without requests.
"""

from __future__ import annotations

import re
import signal
from collections import Counter
from collections.abc import Iterable
from fnmatch import fnmatchcase
from pathlib import Path

from melo_serve.adapters.base import (
    Adapter,
    InstallPlan,
    Launch,
    snapshot_files,
)
from melo_serve.catalog import HubRepo, Pick
from melo_serve.settings import Settings

MODELS_DIR = "~/Library/Application Support/Splash/models"


SHARD = re.compile(
    r"^(.+)-\d{5}-of-(\d{5})\.gguf$"
)  # a split GGUF: name-00001-of-00002


def _shards_whole(names: Iterable[str]) -> bool:
    """Whether each split GGUF among *names* has all its parts."""
    parts = Counter(m.groups() for n in names if (m := SHARD.match(n)))
    return all(count == int(total) for (_, total), count in parts.items())


def _in_cache(repo: HubRepo, hf_home: str) -> bool:
    """Whether the cache has all of *repo*'s files that the pick needs.

    With a file list, each pattern must match a file, and no split GGUF may lack a part
    (a pattern such as ``IQ3_XXS/*`` cannot say how many files it should match). Without
    one the Hub's own list is not at hand, so a snapshot with files is taken as whole.
    A link to a missing blob never counts.
    """
    files = snapshot_files(repo, hf_home)
    if not files or not all(p.exists() for p in files.values()):
        return False
    found = all(any(fnmatchcase(n, pat) for n in files) for pat in repo.files)
    return found and _shards_whole(files)


class SplashAdapter(Adapter):
    key = "splash"
    name = "Splash"
    default_command = ("splash",)
    known_paths = ("/opt/homebrew/bin/splash",)  # an app's PATH has no Homebrew
    install_hint = "brew install incoai/tap/splash"
    tested_versions = ("1.5.",)
    ready_path = "/ready"

    def install_plan(self, macos: tuple[int, ...]) -> InstallPlan | None:
        if macos < (26, 4):
            return None
        return InstallPlan(
            "homebrew", "1.5.0", 264_000_000, "/opt/homebrew",
            needs_homebrew=True, formula="incoai/tap/splash",
        )  # fmt: skip

    def release_path(self, pick: Pick) -> str:
        return "/v1/memory/release"

    def installed(self, pick: Pick, settings: Settings) -> str:
        repo, sep, variant = pick.source.partition(":")
        if "/" not in repo or (sep and not variant):
            return f"{pick.source!r} is not OWNER/REPO[:VARIANT]"
        models = Path(settings.backend(self.key).models_dir or MODELS_DIR).expanduser()
        if (models / pick.source).exists():
            return ""
        # Splash prepares its model from a complete cache, with links and no second copy.
        home = settings.resolved_hf_home()
        repos = self.hub_files(pick)
        if repos and all(_in_cache(r, home) for r in repos):
            return ""
        return "not downloaded yet (Splash has not prepared it)"

    def launch(
        self, pick: Pick, *, port: int, key: str, workdir: Path, settings: Settings
    ) -> Launch:
        idle = settings.idle_release_s
        argv = (
            *self.command(settings),
            "serve",
            "--model",
            pick.source,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--no-webui",
            "--served-model-name",
            pick.served_name,
            "--announce-served-name",
            "--offline",
            "--idle-release",
            f"{int(idle)}s" if idle > 0 else "off",
            *pick.args,
        )
        env = {
            "HF_HOME": settings.resolved_hf_home(),
            **settings.backend(self.key).env,
            **pick.env,
        }
        if key:
            env["SPLASH_API_KEY"] = key
        return Launch(argv, env, signal.SIGINT)
