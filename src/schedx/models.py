"""Shared data models: processes, timeline slices, and schedules.

Time is measured in integer ticks (abstract time units) everywhere, so every
schedule is exact and reproducible with no floating-point drift.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class Process:
    """A CPU-bound job.

    ``burst`` is the total CPU time the process needs. Phase 1 models a
    single CPU burst only; I/O bursts are not simulated (see docs).

    ``priority``: lower number = higher priority.
    """

    pid: str
    arrival: int
    burst: int
    priority: int = 0


@dataclass(frozen=True)
class Slice:
    """A half-open interval ``[start, end)`` on the CPU timeline.

    ``pid is None`` marks an idle interval (no process was ready).
    """

    start: int
    end: int
    pid: str | None

    @property
    def duration(self) -> int:
        return self.end - self.start

    @property
    def idle(self) -> bool:
        return self.pid is None


@dataclass(frozen=True)
class Schedule:
    """Result of one simulation run: the input workload and its timeline.

    The timeline is the single source of truth; all metrics derive from it.
    """

    algorithm: str
    processes: tuple[Process, ...]
    timeline: tuple[Slice, ...]


def _is_int(x: object) -> bool:
    # bool is a subclass of int but is never a meaningful time or priority.
    return isinstance(x, int) and not isinstance(x, bool)


def validate_processes(processes: Iterable[Process]) -> tuple[Process, ...]:
    """Return the workload as a tuple, raising on any invalid process.

    Raises ``TypeError`` for wrong types and ``ValueError`` for bad values
    (negative arrival, non-positive burst, duplicate or empty pid).
    """
    procs = tuple(processes)
    seen: set[str] = set()
    for p in procs:
        if not isinstance(p, Process):
            raise TypeError(f"expected Process, got {type(p).__name__}")
        if not isinstance(p.pid, str) or not p.pid:
            raise ValueError(f"pid must be a non-empty string, got {p.pid!r}")
        if p.pid in seen:
            raise ValueError(f"duplicate pid {p.pid!r}")
        seen.add(p.pid)
        for field in ("arrival", "burst", "priority"):
            if not _is_int(getattr(p, field)):
                raise TypeError(f"{p.pid}: {field} must be an int, got {getattr(p, field)!r}")
        if p.arrival < 0:
            raise ValueError(f"{p.pid}: arrival must be >= 0, got {p.arrival}")
        if p.burst <= 0:
            raise ValueError(f"{p.pid}: burst must be > 0, got {p.burst}")
    return procs
