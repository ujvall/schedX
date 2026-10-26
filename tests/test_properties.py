"""Seeded randomized invariant checks across all algorithms (deterministic)."""

import random

import pytest

from schedx import Process, compute_metrics, fcfs, priority, round_robin, sjf, srtf

ALGOS = {
    "fcfs": fcfs,
    "sjf": sjf,
    "srtf": srtf,
    "rr1": lambda w: round_robin(w, 1),
    "rr3": lambda w: round_robin(w, 3),
    "prio": priority,
    "prio_pre": lambda w: priority(w, preemptive=True),
}
NON_PREEMPTIVE = {"fcfs", "sjf", "prio"}


def workload(seed):
    rng = random.Random(seed)
    return [
        Process(f"P{i}", rng.randint(0, 15), rng.randint(1, 8), rng.randint(0, 3))
        for i in range(rng.randint(1, 8))
    ]


SEEDS = range(200)


@pytest.mark.parametrize("name", ALGOS)
def test_timeline_invariants(name):
    for seed in SEEDS:
        w = workload(seed)
        s = ALGOS[name](w)
        tl = s.timeline
        arrival = {p.pid: p.arrival for p in w}
        burst = {p.pid: p.burst for p in w}

        # Contiguous from t=0, non-empty, never overlapping.
        assert tl[0].start == 0
        for a, b in zip(tl, tl[1:]):
            assert a.end == b.start
            assert not (a.pid == b.pid), "adjacent slices of one process must be merged"
        assert all(x.duration > 0 for x in tl)

        # Each process runs exactly its burst, never before arrival; nothing
        # runs after completion (implied by exact burst + end-of-run tracking).
        ran = {}
        for x in tl:
            if not x.idle:
                assert x.start >= arrival[x.pid]
                ran[x.pid] = ran.get(x.pid, 0) + x.duration
        assert ran == burst

        # Work conserving: idle only when no arrived process is unfinished,
        # and idle lasts exactly until the next arrival.
        done_by = {}
        for x in tl:
            if not x.idle:
                done_by[x.pid] = x.end
        for x in tl:
            if x.idle:
                assert all(done_by[p.pid] <= x.start for p in w if p.arrival <= x.start)
                assert x.end == min(p.arrival for p in w if p.arrival > x.start)

        if name in NON_PREEMPTIVE:
            assert sorted(x.pid for x in tl if not x.idle) == sorted(burst)

        # Metrics agree with the timeline.
        m = compute_metrics(s)
        for pm in m.processes:
            in_system = [x for x in tl if x.start < pm.completion and x.end > pm.arrival]
            not_running = sum(
                min(x.end, pm.completion) - max(x.start, pm.arrival) for x in in_system if x.pid != pm.pid
            )
            assert pm.waiting == not_running
            assert pm.turnaround == pm.waiting + pm.burst
            assert 0 <= pm.response <= pm.waiting
        assert m.busy_time == sum(burst.values())
        assert m.interval_end == tl[-1].end
        assert 0 < m.cpu_utilization <= 1


def test_srtf_minimizes_average_turnaround():
    # SRPT is optimal for total completion time on one machine.
    for seed in SEEDS:
        w = workload(seed)
        best = compute_metrics(srtf(w)).avg_turnaround
        for algo in ALGOS.values():
            assert best <= compute_metrics(algo(w)).avg_turnaround + 1e-9


@pytest.mark.parametrize("q", [1, 2, 3])
def test_rr_slices_respect_quantum_when_others_wait(q):
    for seed in SEEDS:
        w = workload(seed)
        tl = round_robin(w, q).timeline
        completion = {x.pid: x.end for x in tl if not x.idle}
        for x in tl:
            if x.idle:
                continue
            # A slice longer than q is only legal if, at every quantum boundary
            # inside it, no other process was waiting (it was re-dispatched alone).
            for boundary in range(x.start + q, x.end, q):
                waiting = [
                    p for p in w if p.pid != x.pid and p.arrival <= boundary and completion[p.pid] > boundary
                ]
                assert not waiting, (seed, x, boundary, waiting)
