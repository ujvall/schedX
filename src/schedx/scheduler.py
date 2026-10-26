"""Discrete-event CPU scheduling engine and the five Phase 1 algorithms.

All algorithms share one engine (``_simulate``). An algorithm is defined by:

* ``key``        -- ordering of the ready queue; the minimum key is dispatched.
* ``preemptive`` -- whether a new arrival may interrupt the running process.
* ``quantum``    -- optional cap on how long a dispatched process may run.

Determinism rules (applied identically by every algorithm):

* Processes arriving at the same tick are admitted in input order.
* Every key ends with ``(arrival, input index)`` (or an enqueue sequence
  number, which is itself ordered that way), so ties never depend on dict or
  set ordering.
* When a running slice ends at tick ``t``, processes arriving at ``t`` are
  admitted to the ready queue *before* the interrupted process is re-queued.
  For Round Robin this means newcomers run ahead of the preempted process.
* If the same process is dispatched again with no gap, the two slices are
  merged: there was no real context switch.
"""

from __future__ import annotations

from itertools import count
from typing import Callable, Iterable

from .models import Process, Schedule, Slice, _is_int, validate_processes

# key(process, remaining, enqueue_seq, input_index) -> sortable tuple
Key = Callable[[Process, int, int, int], tuple]


def _simulate(
    algorithm: str,
    processes: Iterable[Process],
    key: Key,
    preemptive: bool,
    quantum: int | None = None,
) -> Schedule:
    procs = validate_processes(processes)
    index = {p.pid: i for i, p in enumerate(procs)}
    arrivals = sorted(procs, key=lambda p: (p.arrival, index[p.pid]))
    remaining = {p.pid: p.burst for p in procs}
    by_pid = {p.pid: p for p in procs}

    ready: dict[str, int] = {}  # pid -> enqueue sequence number
    seq = count()
    timeline: list[Slice] = []
    t = 0
    nxt = 0  # index into ``arrivals`` of the next process not yet admitted

    def admit(now: int) -> None:
        # Future hook: an I/O-completion event would re-admit a process here.
        nonlocal nxt
        while nxt < len(arrivals) and arrivals[nxt].arrival <= now:
            ready[arrivals[nxt].pid] = next(seq)
            nxt += 1

    def emit(s: Slice) -> None:
        if timeline and timeline[-1].pid == s.pid and timeline[-1].end == s.start:
            timeline[-1] = Slice(timeline[-1].start, s.end, s.pid)
        else:
            timeline.append(s)

    admit(t)
    while ready or nxt < len(arrivals):
        if not ready:
            next_arrival = arrivals[nxt].arrival
            emit(Slice(t, next_arrival, None))
            t = next_arrival
            admit(t)
            continue

        pid = min(ready, key=lambda q: key(by_pid[q], remaining[q], ready[q], index[q]))
        del ready[pid]

        run = remaining[pid]
        if quantum is not None:
            run = min(run, quantum)
        if preemptive and nxt < len(arrivals):
            # Stop at the next arrival so the scheduler can re-evaluate.
            run = min(run, arrivals[nxt].arrival - t)

        emit(Slice(t, t + run, pid))
        t += run
        remaining[pid] -= run
        admit(t)
        if remaining[pid] > 0:
            ready[pid] = next(seq)

    return Schedule(algorithm, procs, tuple(timeline))


def fcfs(processes: Iterable[Process]) -> Schedule:
    """First Come, First Served (non-preemptive), ordered by arrival."""
    return _simulate("FCFS", processes, lambda p, rem, seq, i: (seq,), preemptive=False)


def sjf(processes: Iterable[Process]) -> Schedule:
    """Shortest Job First (non-preemptive): shortest burst among arrived processes."""
    return _simulate("SJF", processes, lambda p, rem, seq, i: (p.burst, p.arrival, i), preemptive=False)


def srtf(processes: Iterable[Process]) -> Schedule:
    """Shortest Remaining Time First (preemptive SJF).

    A new arrival preempts only if its burst is *strictly* shorter than the
    running process's remaining time; equal remaining time favours the
    earlier arrival, so the running process keeps the CPU.
    """
    return _simulate("SRTF", processes, lambda p, rem, seq, i: (rem, p.arrival, i), preemptive=True)


def round_robin(processes: Iterable[Process], quantum: int) -> Schedule:
    """Round Robin with a fixed time quantum (FIFO ready queue)."""
    if not _is_int(quantum) or quantum <= 0:
        raise ValueError(f"quantum must be a positive int, got {quantum!r}")
    return _simulate(
        f"RR(q={quantum})", processes, lambda p, rem, seq, i: (seq,), preemptive=False, quantum=quantum
    )


def priority(processes: Iterable[Process], preemptive: bool = False) -> Schedule:
    """Priority scheduling. Lower ``priority`` number = higher priority.

    Ties are broken by earlier arrival, then input order. Non-preemptive by
    default; with ``preemptive=True`` a strictly higher-priority arrival
    interrupts the running process.
    """
    name = "Priority(preemptive)" if preemptive else "Priority"
    return _simulate(name, processes, lambda p, rem, seq, i: (p.priority, p.arrival, i), preemptive=preemptive)
