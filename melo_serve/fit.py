"""The fit check: does a pick load without starving the Mac of memory?

Free memory is the number ``memory_pressure -Q`` prints (the system-wide free
percentage), so the check speaks the same number as the person reading it. A pick's
``mem_gb`` is how much that number drops while the pick answers, measured on a Mac
with ``measured_ram_gb`` of memory; on any other Mac it is an estimate, and a pick
with no measurement at all is estimated from its download size.
"""

from __future__ import annotations

import ctypes
import os
import platform
import re
import subprocess
from dataclasses import dataclass

from melo_serve.catalog import Pick

GIB = 1024**3
SAME_CLASS_GB = 8  # Macs within this much memory of each other count as one class

FITS = "fits"  # leaves at least the floor that other heavy work needs
ALONE = "alone"  # fits, but other heavy work has to wait
ESTIMATE = "estimate"  # not measured on this class of Mac; likely fits
TOO_BIG = "too_big"  # would leave less than the minimum
UNKNOWN = "unknown"  # free memory or the pick's size is unknown


@dataclass(frozen=True)
class Memory:
    total_bytes: int
    free_pct: float

    @property
    def total_gb(self) -> float:
        return self.total_bytes / GIB


@dataclass(frozen=True)
class Fit:
    verdict: str
    need_gb: float | None
    after_pct: float | None
    measured: bool
    reason: str

    @property
    def loadable(self) -> bool:
        return self.verdict != TOO_BIG

    def to_json(self) -> dict[str, object]:
        return {
            "verdict": self.verdict,
            "need_gb": self.need_gb,
            "after_pct": None if self.after_pct is None else round(self.after_pct, 1),
            "measured": self.measured,
            "reason": self.reason,
        }


def total_ram_bytes() -> int | None:
    """Physical memory, read without starting a process."""
    try:
        if platform.system() == "Darwin":
            value, size = ctypes.c_uint64(), ctypes.c_size_t(8)
            libc = ctypes.CDLL(None)
            if (
                libc.sysctlbyname(
                    b"hw.memsize", ctypes.byref(value), ctypes.byref(size), None, 0
                )
                == 0
            ):
                return int(value.value)
            return None
        return int(os.sysconf("SC_PHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, OSError, ValueError):
        return None


_FREE_RE = re.compile(r"free percentage:\s*(\d+(?:\.\d+)?)%")


def free_percent() -> float | None:
    """The system-wide free memory percentage of ``memory_pressure -Q``."""
    try:
        out = subprocess.run(
            ["memory_pressure", "-Q"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    match = _FREE_RE.search(out)
    return float(match.group(1)) if match else None


def probe() -> Memory | None:
    total, free = total_ram_bytes(), free_percent()
    if total is None or free is None:
        return None
    return Memory(total_bytes=total, free_pct=free)


def need_gb(pick: Pick, total_gb: float) -> tuple[float | None, bool]:
    """How much memory *pick* needs here, and whether that was measured here."""
    if pick.mem_gb is not None:
        measured = (
            pick.measured_ram_gb is not None
            and abs(pick.measured_ram_gb - total_gb) <= SAME_CLASS_GB
        )
        return pick.mem_gb, measured
    if pick.disk_gb is not None:
        return round(pick.disk_gb * 1.15 + 1.5, 1), False
    return None, False


def check(
    pick: Pick,
    memory: Memory | None,
    *,
    freeing_gb: float = 0.0,
    floor_pct: float = 30.0,
    min_pct: float = 5.0,
) -> Fit:
    """The verdict for loading *pick* now; *freeing_gb* is what the switch stops."""
    if memory is None:
        return Fit(UNKNOWN, None, None, False, "free memory is unknown")
    gb, measured = need_gb(pick, memory.total_gb)
    if gb is None:
        return Fit(UNKNOWN, None, None, False, "its size is unknown")
    after = min(100.0, memory.free_pct + (freeing_gb - gb) / memory.total_gb * 100)
    about = f"needs about {gb:.0f} GB" + ("" if measured else " (estimate)")
    if after < min_pct:
        return Fit(
            TOO_BIG,
            gb,
            after,
            measured,
            f"{about}; would leave {max(after, 0):.0f}% free",
        )
    if after < floor_pct:  # measured or not: other heavy work has to wait
        return Fit(
            ALONE,
            gb,
            after,
            measured,
            f"{about}; leaves {after:.0f}% free, so other heavy work waits",
        )
    verdict = FITS if measured else ESTIMATE
    return Fit(verdict, gb, after, measured, f"{about}; leaves {after:.0f}% free")
