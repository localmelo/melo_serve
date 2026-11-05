"""oMLX: models in folders; chat and embeddings in one process.

melo_serve gives each oMLX instance a base path of its own and a model folder
holding one link, named after the served name, to the pick's model. Settings go
in ``OMLX_*`` variables, since ``serve --port`` and the like are written back into
oMLX's ``settings.json``.
"""

from __future__ import annotations

import json
import shutil
import signal
from pathlib import Path

from melo_serve.adapters.base import Adapter, InstallPlan, Launch
from melo_serve.catalog import Pick
from melo_serve.settings import Settings

APP_CLI = "/Applications/oMLX.app/Contents/MacOS/omlx-cli"
USER_APP_CLI = "~/Applications/oMLX.app/Contents/MacOS/omlx-cli"  # where get puts it
VERSION = "0.7.3"
DMG_BASE = f"https://github.com/jundot/omlx/releases/download/v{VERSION}/"
DMGS = (  # (first macOS major it is for, file, bytes, sha256); one DMG per macOS line
    (
        15,
        f"oMLX-{VERSION}-macos15-sequoia.dmg",
        838158761,
        "cd5e3b454798a002f32ef2f3bf06ae033ea2f4e5b02aa039a2c68445498605f1",
    ),
    (
        26,
        f"oMLX-{VERSION}-macos26-27.dmg",
        881239063,
        "2acfde41aeac0b022dbca803ca109437f3ccaf0c33f3415881526b985bee86f2",
    ),
)


def hf_store(repo: str, hf_home: str) -> Path:
    return Path(hf_home) / "hub" / ("models--" + repo.replace("/", "--"))


def hf_snapshot(repo: str, hf_home: str) -> Path | None:
    """The local snapshot of a Hugging Face repo, found without the network."""
    store = hf_store(repo, hf_home)
    try:
        revision = (store / "refs" / "main").read_text(encoding="utf-8").strip()
    except OSError:
        revision = ""
    if revision and (store / "snapshots" / revision).is_dir():
        return store / "snapshots" / revision
    snapshots = sorted((store / "snapshots").glob("*")) if store.is_dir() else []
    return snapshots[-1] if snapshots else None


def missing_weights(path: Path) -> str:
    """Why the model folder at *path* cannot load yet; "" when its weights are there.

    A download cut short leaves the small files and the shard index, but not
    every shard; a cache link to a missing blob counts as missing.
    """
    index = path / "model.safetensors.index.json"
    if not index.is_file():
        found = any(p.is_file() for p in path.glob("*.safetensors"))
        return "" if found else "it has no weight files"
    try:
        shards = set(
            json.loads(index.read_text(encoding="utf-8"))["weight_map"].values()
        )
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return f"{index.name} cannot be read"
    missing = [shard for shard in shards if not (path / shard).is_file()]
    return (
        f"{len(missing)} of {len(shards)} weight files are missing" if missing else ""
    )


class OmlxAdapter(Adapter):
    key = "omlx"
    name = "oMLX"
    default_command = ("omlx",)
    install_hint = "melo-serve get <pick> installs it"
    tested_versions = ("0.7.",)
    known_paths = (APP_CLI, USER_APP_CLI)

    def install_plan(self, macos: tuple[int, ...]) -> InstallPlan | None:
        """The pinned DMG for this macOS: 15, or 26 and later; none for older."""
        fits = [dmg for dmg in DMGS if dmg[0] <= macos[0]]
        if not fits:
            return None
        _, file, size, sha256 = fits[-1]
        return InstallPlan(
            "dmg", VERSION, size, "~/Applications/oMLX.app",
            url=DMG_BASE + file, file=file, sha256=sha256,
        )  # fmt: skip

    def release_path(self, pick: Pick) -> str:
        return f"/v1/models/{pick.served_name}/unload"

    def cached_bytes(
        self, pick: Pick, settings: Settings, include_partial: bool = False
    ) -> int:
        """Bytes of finished files; the hub never reuses a cut-short ``.incomplete``."""
        if pick.is_local:
            return 0
        blobs = hf_store(pick.source, settings.resolved_hf_home()) / "blobs"
        try:
            return sum(
                f.stat().st_size
                for f in blobs.iterdir()
                if f.is_file() and (include_partial or f.suffix != ".incomplete")
            )
        except OSError:
            return 0

    def model_path(self, pick: Pick, settings: Settings) -> Path | None:
        if pick.is_local:
            path = Path(pick.source).expanduser()
            return path if path.is_dir() else None
        return hf_snapshot(pick.source, settings.resolved_hf_home())

    def installed(self, pick: Pick, settings: Settings) -> str:
        path = self.model_path(pick, settings)
        if path is None:
            if pick.is_local:
                return f"the folder {pick.source} does not exist"
            return "not downloaded yet (not in the Hugging Face cache)"
        if not (path / "config.json").is_file():
            return f"{path} has no config.json"
        problem = missing_weights(path)
        return f"only partly downloaded: {problem}" if problem else ""

    def launch(
        self, pick: Pick, *, port: int, key: str, workdir: Path, settings: Settings
    ) -> Launch:
        model = self.model_path(pick, settings)
        if model is None:
            raise FileNotFoundError(self.installed(pick, settings))
        models = workdir / "models"
        if models.is_dir() and not models.is_symlink():
            shutil.rmtree(models)  # holds links only, and rmtree never follows them
        elif models.exists() or models.is_symlink():
            models.unlink()
        models.mkdir(parents=True)
        (models / pick.served_name).symlink_to(model.resolve())
        env = {
            "OMLX_BASE_PATH": str(workdir),
            "OMLX_MODEL_DIR": str(models),
            "OMLX_HOST": "127.0.0.1",
            "OMLX_PORT": str(port),
            "OMLX_HEADLESS": "1",
            "OMLX_HF_CACHE_ENABLED": "0",
            "HF_HOME": settings.resolved_hf_home(),
            **settings.backend(self.key).env,
            **pick.env,
        }
        if key:
            env["OMLX_API_KEY"] = key
        return Launch(
            (*self.command(settings), "serve", *pick.args), env, signal.SIGINT
        )
