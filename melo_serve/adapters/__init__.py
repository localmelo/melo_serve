"""Backend adapters, by key."""

from melo_serve.adapters.base import Adapter, Install, InstallPlan, Launch
from melo_serve.adapters.omlx import OmlxAdapter
from melo_serve.adapters.splash import SplashAdapter


def default_adapters() -> dict[str, Adapter]:
    return {"splash": SplashAdapter(), "omlx": OmlxAdapter()}


__all__ = [
    "Adapter",
    "Install",
    "InstallPlan",
    "Launch",
    "OmlxAdapter",
    "SplashAdapter",
    "default_adapters",
]
