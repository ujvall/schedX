"""Hand-verified scheduling scenarios for each algorithm."""

import pytest

from schedx import Process, compute_metrics, fcfs, priority, round_robin, sjf, srtf


def P(pid, arrival, burst, prio=0):
    return Process(pid, arrival, burst, prio)


def segs(schedule):
    return [(s.start, s.end, s.pid) for s in schedule.timeline]


# Classic textbook workload, used by SJF and SRTF.
CLASSIC = [P("P1", 0, 7), P("P2", 2, 4), P("P3", 4, 1), P("P4", 5, 4)]


# --- FCFS -------------------------------------------------------------------


def test_fcfs_runs_in_arrival_order():
    s = fcfs([P("P1", 0, 5), P("P2", 1, 3), P("P3", 2, 1)])
    assert segs(s) == [(0, 5, "P1"), (5, 8, "P2"), (8, 9, "P3")]


def test_fcfs_orders_by_arrival_not_input_position():
    s = fcfs([P("late", 4, 1), P("early", 0, 2)])
    assert segs(s) == [(0, 2, "early"), (2, 4, None), (4, 5, "late")]


def test_fcfs_simultaneous_arrivals_use_input_order():
    s = fcfs([P("B", 0, 2), P("A", 0, 1)])
    assert segs(s) == [(0, 2, "B"), (2, 3, "A")]


def test_fcfs_idle_before_first_arrival_and_between_jobs():
    s = fcfs([P("P1", 2, 3), P("P2", 10, 2)])
    assert segs(s) == [(0, 2, None), (2, 5, "P1"), (5, 10, None), (10, 12, "P2")]


# --- SJF --------------------------------------------------------------------


def test_sjf_classic():
    s = sjf(CLASSIC)
    # P2 and P4 tie on burst 4 at t=8: P2 arrived earlier.
    assert segs(s) == [(0, 7, "P1"), (7, 8, "P3"), (8, 12, "P2"), (12, 16, "P4")]
    m = compute_metrics(s)
    assert [p.waiting for p in m.processes] == [0, 6, 3, 7]
    assert m.avg_waiting == 4.0


def test_sjf_only_considers_arrived_processes():
    # P2 is shorter but has not arrived at t=0.
    s = sjf([P("P1", 0, 10), P("P2", 1, 1)])
    assert segs(s) == [(0, 10, "P1"), (10, 11, "P2")]


def test_sjf_is_non_preemptive():
    s = sjf([P("long", 0, 5), P("short", 1, 1)])
    assert segs(s)[0] == (0, 5, "long")


# --- SRTF -------------------------------------------------------------------


def test_srtf_classic_preemption():
    s = srtf(CLASSIC)
    assert segs(s) == [
        (0, 2, "P1"),
        (2, 4, "P2"),  # P2 (4) < P1 remaining (5): preempt
        (4, 5, "P3"),  # P3 (1) < P2 remaining (2): preempt
        (5, 7, "P2"),
        (7, 11, "P4"),
        (11, 16, "P1"),
    ]
    m = compute_metrics(s)
    assert [p.waiting for p in m.processes] == [9, 1, 0, 2]
    assert [p.response for p in m.processes] == [0, 0, 0, 2]
    assert m.avg_waiting == 3.0
    assert m.context_switches == 5


def test_srtf_equal_remaining_does_not_preempt():
    s = srtf([P("A", 0, 4), P("B", 1, 3)])  # at t=1: A has 3 left, B needs 3
    assert segs(s) == [(0, 4, "A"), (4, 7, "B")]
    assert compute_metrics(s).context_switches == 1


def test_srtf_tracks_remaining_burst_across_preemptions():
    s = srtf([P("A", 0, 10), P("B", 2, 1), P("C", 5, 1)])
    assert segs(s) == [(0, 2, "A"), (2, 3, "B"), (3, 5, "A"), (5, 6, "C"), (6, 12, "A")]


# --- Round Robin --------------------------------------------------------------


def test_rr_queue_order_and_quantum_boundaries():
    s = round_robin([P("P1", 0, 5), P("P2", 1, 3), P("P3", 2, 1)], quantum=2)
    assert segs(s) == [
        (0, 2, "P1"),
        (2, 4, "P2"),
        (4, 5, "P3"),  # finishes early, before the quantum ends
        (5, 7, "P1"),
        (7, 8, "P2"),
        (8, 9, "P1"),
    ]


def test_rr_new_arrival_enqueued_before_preempted_process():
    # B arrives exactly when A's quantum expires: B goes ahead of A.
    s = round_robin([P("A", 0, 4), P("B", 2, 2)], quantum=2)
    assert segs(s) == [(0, 2, "A"), (2, 4, "B"), (4, 6, "A")]


def test_rr_does_not_preempt_mid_quantum():
    s = round_robin([P("A", 0, 6), P("B", 1, 1)], quantum=3)
    assert segs(s) == [(0, 3, "A"), (3, 4, "B"), (4, 7, "A")]


def test_rr_lone_process_is_one_slice_without_switches():
    s = round_robin([P("A", 0, 7)], quantum=2)
    assert segs(s) == [(0, 7, "A")]
    assert compute_metrics(s).context_switches == 0


def test_rr_with_idle_gap():
    s = round_robin([P("A", 0, 3), P("B", 6, 2)], quantum=2)
    assert segs(s) == [(0, 3, "A"), (3, 6, None), (6, 8, "B")]


def test_rr_large_quantum_equals_fcfs():
    w = [P("P1", 0, 5), P("P2", 1, 3), P("P3", 2, 1)]
    assert segs(round_robin(w, quantum=100)) == segs(fcfs(w))


# --- Priority -----------------------------------------------------------------

PRIO = [P("P1", 0, 4, 3), P("P2", 1, 2, 1), P("P3", 2, 3, 2), P("P4", 3, 1, 1)]


def test_priority_non_preemptive():
    s = priority(PRIO)
    # At t=4, P2 and P4 tie on priority 1: P2 arrived earlier.
    assert segs(s) == [(0, 4, "P1"), (4, 6, "P2"), (6, 7, "P4"), (7, 10, "P3")]
    assert s.algorithm == "Priority"


def test_priority_preemptive():
    s = priority(PRIO, preemptive=True)
    assert segs(s) == [(0, 1, "P1"), (1, 3, "P2"), (3, 4, "P4"), (4, 7, "P3"), (7, 10, "P1")]
    assert s.algorithm == "Priority(preemptive)"


def test_priority_lower_number_wins():
    s = priority([P("low", 0, 1, 9), P("high", 0, 1, 0)])
    assert [x.pid for x in s.timeline] == ["high", "low"]


def test_priority_full_tie_uses_input_order():
    s = priority([P("X", 0, 1, 2), P("Y", 0, 1, 2)])
    assert [x.pid for x in s.timeline] == ["X", "Y"]


def test_priority_preemptive_equal_priority_does_not_preempt():
    s = priority([P("A", 0, 3, 1), P("B", 1, 3, 1)], preemptive=True)
    assert segs(s) == [(0, 3, "A"), (3, 6, "B")]


# --- Shared ---------------------------------------------------------------------

ALL = [fcfs, sjf, srtf, lambda w: round_robin(w, 2), priority, lambda w: priority(w, True)]


@pytest.mark.parametrize("algo", ALL)
def test_empty_workload(algo):
    s = algo([])
    assert s.timeline == ()
    m = compute_metrics(s)
    assert (m.avg_waiting, m.throughput, m.cpu_utilization, m.context_switches) == (0.0, 0.0, 0.0, 0)


@pytest.mark.parametrize("algo", ALL)
def test_single_process_late_arrival(algo):
    s = algo([P("A", 3, 2)])
    assert segs(s) == [(0, 3, None), (3, 5, "A")]


# --- Time-varying quantum (Phase 3) ---------------------------------------------------

from schedx import adaptive_round_robin  # noqa: E402


@pytest.mark.parametrize("q", [1, 2, 3, 5])
def test_adaptive_rr_with_constant_quantum_equals_rr(q):
    import random

    for seed in range(50):
        rng = random.Random(seed)
        w = [P(f"P{i}", rng.randint(0, 12), rng.randint(1, 9)) for i in range(rng.randint(1, 7))]
        assert segs(adaptive_round_robin(w, lambda t: q)) == segs(round_robin(w, q))


def test_adaptive_rr_switches_quantum_at_dispatch_time():
    w = [P("A", 0, 6), P("B", 0, 6)]
    # q=1 for dispatches before t=2, then q=3. The dispatch at t=1 still uses q=1.
    s = adaptive_round_robin(w, lambda t: 1 if t < 2 else 3)
    assert segs(s) == [(0, 1, "A"), (1, 2, "B"), (2, 5, "A"), (5, 8, "B"), (8, 10, "A"), (10, 12, "B")]


def test_adaptive_rr_slice_straddling_switch_keeps_old_quantum():
    s = adaptive_round_robin([P("A", 0, 9), P("B", 0, 9)], lambda t: 4 if t < 5 else 1)
    # B is dispatched at t=4 (< 5) with q=4 and runs past the switch to t=8.
    assert segs(s)[:3] == [(0, 4, "A"), (4, 8, "B"), (8, 9, "A")]


@pytest.mark.parametrize("bad", [0, -1, 1.5, None, True])
def test_adaptive_rr_rejects_invalid_quantum(bad):
    with pytest.raises(ValueError):
        adaptive_round_robin([P("A", 0, 3)], lambda t: bad)
    with pytest.raises(TypeError):
        adaptive_round_robin([P("A", 0, 3)], 2)
