"""Scheduling metrics, derived only from a ``Schedule``'s timeline.

Definitions (identical for every algorithm; see docs/architecture.md):

* turnaround = completion - arrival
* response   = first dispatch - arrival
* waiting    = turnaround - CPU time received. With CPU-only processes this is
  pure ready-queue waiting; it contains no I/O blocking time.
* Measurement interval = [first arrival, last completion]. Idle time before
  the first arrival is excluded; idle gaps between arrivals are included.
* throughput      = completed processes / interval length
* cpu_utilization = busy time inside the interval / interval length
* context_switches = dispatches of a process different from the previously
  running one. The very first dispatch is not a switch; idle gaps do not reset
  "previously running" (A, idle, A = 0 switches; A, idle, B = 1). Switch
  overhead is not modelled: it adds zero time.

For an empty workload every average/rate is 0.0.
"""

from __future__ import annotations

from dataclasses import dataclass

from .models import Schedule


@dataclass(frozen=True)
class ProcessMetrics:
    pid: str
    arrival: int
    burst: int
    first_run: int
    completion: int
    turnaround: int
    waiting: int
    response: int


@dataclass(frozen=True)
class ScheduleMetrics:
    algorithm: str
    processes: tuple[ProcessMetrics, ...]
    avg_waiting: float
    avg_turnaround: float
    avg_response: float
    interval_start: int
    interval_end: int
    busy_time: int
    throughput: float
    cpu_utilization: float
    context_switches: int


def _mean(values: list[int]) -> float:
    return sum(values) / len(values) if values else 0.0


def compute_metrics(schedule: Schedule) -> ScheduleMetrics:
    """Compute per-process and aggregate metrics for ``schedule``.

    Raises ``ValueError`` if the timeline has empty or overlapping slices,
    runs an unknown pid, does not give every process exactly its burst, or
    runs a process before it arrives.
    """
    known = {p.pid for p in schedule.processes}
    first_run: dict[str, int] = {}
    completion: dict[str, int] = {}
    executed: dict[str, int] = {}
    switches = 0
    last_pid: str | None = None
    busy = 0
    prev_end: int | None = None

    for s in schedule.timeline:
        if s.duration <= 0:
            raise ValueError(f"empty or negative slice {s}")
        if prev_end is not None and s.start < prev_end:
            raise ValueError(f"slice {s} overlaps the previous slice ending at {prev_end}")
        prev_end = s.end
        if s.idle:
            continue
        if s.pid not in known:
            raise ValueError(f"timeline runs unknown pid {s.pid!r}")
        busy += s.duration
        first_run.setdefault(s.pid, s.start)
        completion[s.pid] = s.end
        executed[s.pid] = executed.get(s.pid, 0) + s.duration
        if last_pid is not None and s.pid != last_pid:
            switches += 1
        last_pid = s.pid

    per: list[ProcessMetrics] = []
    for p in schedule.processes:
        if executed.get(p.pid) != p.burst:
            raise ValueError(f"{p.pid}: executed {executed.get(p.pid, 0)} of burst {p.burst}")
        if first_run[p.pid] < p.arrival:
            raise ValueError(f"{p.pid}: dispatched at {first_run[p.pid]} before arrival {p.arrival}")
        turnaround = completion[p.pid] - p.arrival
        per.append(
            ProcessMetrics(
                pid=p.pid,
                arrival=p.arrival,
                burst=p.burst,
                first_run=first_run[p.pid],
                completion=completion[p.pid],
                turnaround=turnaround,
                waiting=turnaround - executed[p.pid],
                response=first_run[p.pid] - p.arrival,
            )
        )

    if per:
        start = min(m.arrival for m in per)
        end = max(m.completion for m in per)
    else:
        start = end = 0
    length = end - start

    return ScheduleMetrics(
        algorithm=schedule.algorithm,
        processes=tuple(per),
        avg_waiting=_mean([m.waiting for m in per]),
        avg_turnaround=_mean([m.turnaround for m in per]),
        avg_response=_mean([m.response for m in per]),
        interval_start=start,
        interval_end=end,
        busy_time=busy,
        throughput=len(per) / length if length else 0.0,
        cpu_utilization=busy / length if length else 0.0,
        context_switches=switches,
    )
