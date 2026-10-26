"""Metric definitions and input validation."""

import pytest

from schedx import Process, Schedule, Slice, compute_metrics, fcfs, priority, round_robin, sjf, srtf


def P(pid, arrival, burst, prio=0):
    return Process(pid, arrival, burst, prio)


# --- Metrics --------------------------------------------------------------------


def test_fcfs_per_process_metrics():
    m = compute_metrics(fcfs([P("P1", 0, 5), P("P2", 1, 3), P("P3", 2, 1)]))
    rows = [(p.pid, p.first_run, p.completion, p.turnaround, p.waiting, p.response) for p in m.processes]
    assert rows == [
        ("P1", 0, 5, 5, 0, 0),
        ("P2", 5, 8, 7, 4, 4),
        ("P3", 8, 9, 7, 6, 6),
    ]
    assert m.avg_turnaround == pytest.approx(19 / 3)
    assert m.avg_waiting == pytest.approx(10 / 3)
    assert m.avg_response == pytest.approx(10 / 3)


def test_response_differs_from_waiting_under_preemption():
    m = compute_metrics(round_robin([P("A", 0, 3), P("B", 0, 3)], quantum=1))
    # A B A B A B: A responds at 0 but waits 3 in total.
    a, b = m.processes
    assert (a.response, a.waiting, a.completion) == (0, 2, 5)
    assert (b.response, b.waiting, b.completion) == (1, 3, 6)


def test_interval_excludes_leading_idle_and_includes_gaps():
    m = compute_metrics(fcfs([P("P1", 2, 3), P("P2", 10, 2)]))
    assert (m.interval_start, m.interval_end, m.busy_time) == (2, 12, 5)
    assert m.throughput == pytest.approx(2 / 10)
    assert m.cpu_utilization == pytest.approx(5 / 10)


def test_full_utilization_without_gaps():
    m = compute_metrics(fcfs([P("A", 0, 2), P("B", 1, 2)]))
    assert m.cpu_utilization == 1.0
    assert m.throughput == pytest.approx(2 / 4)


def test_context_switch_counting_rules():
    # First dispatch is not a switch.
    assert compute_metrics(fcfs([P("A", 0, 1)])).context_switches == 0
    # Idle gap followed by the same process: no switch.
    same = Schedule("x", (P("A", 0, 2),), (Slice(0, 1, "A"), Slice(1, 3, None), Slice(3, 4, "A")))
    assert compute_metrics(same).context_switches == 0
    # Idle gap followed by a different process: one switch.
    assert compute_metrics(fcfs([P("A", 0, 1), P("B", 5, 1)])).context_switches == 1
    # Process change and preemptions each count.
    assert compute_metrics(round_robin([P("A", 0, 2), P("B", 0, 2)], quantum=1)).context_switches == 3


def test_metrics_reject_inconsistent_timeline():
    short = Schedule("x", (P("A", 0, 3),), (Slice(0, 2, "A"),))
    with pytest.raises(ValueError, match="executed"):
        compute_metrics(short)
    early = Schedule("x", (P("A", 2, 1),), (Slice(0, 1, "A"),))
    with pytest.raises(ValueError, match="before arrival"):
        compute_metrics(early)


# Regression (Phase 2 audit): these timelines used to be accepted and reported
# CPU utilization of 3.5 and 2.0 respectively.
def test_metrics_reject_unknown_pid():
    s = Schedule("x", (P("A", 0, 2),), (Slice(0, 2, "A"), Slice(2, 7, "GHOST")))
    with pytest.raises(ValueError, match="unknown pid"):
        compute_metrics(s)


def test_metrics_reject_overlapping_slices():
    s = Schedule("x", (P("A", 0, 2), P("B", 0, 2)), (Slice(0, 2, "A"), Slice(0, 2, "B")))
    with pytest.raises(ValueError, match="overlaps"):
        compute_metrics(s)


def test_metrics_reject_empty_slice():
    s = Schedule("x", (P("A", 0, 2),), (Slice(0, 0, None), Slice(0, 2, "A")))
    with pytest.raises(ValueError, match="empty"):
        compute_metrics(s)


# --- Validation -------------------------------------------------------------------

ALGOS = [fcfs, sjf, srtf, lambda w: round_robin(w, 2), priority]


@pytest.mark.parametrize("algo", ALGOS)
@pytest.mark.parametrize(
    "workload, exc",
    [
        ([P("A", 0, 1), P("A", 1, 1)], ValueError),  # duplicate pid
        ([P("A", 0, 0)], ValueError),  # zero burst
        ([P("A", 0, -2)], ValueError),  # negative burst
        ([P("A", -1, 1)], ValueError),  # negative arrival
        ([P("", 0, 1)], ValueError),  # empty pid
        ([P(7, 0, 1)], ValueError),  # non-string pid
        ([P("A", 0, 1.5)], TypeError),  # non-integer burst
        ([P("A", 0.0, 1)], TypeError),  # non-integer arrival
        ([P("A", 0, True)], TypeError),  # bool is not a time
        ([P("A", 0, 1, "high")], TypeError),  # non-integer priority
        ([("A", 0, 1)], TypeError),  # not a Process
    ],
)
def test_invalid_inputs_rejected(algo, workload, exc):
    with pytest.raises(exc):
        algo(workload)


@pytest.mark.parametrize("quantum", [0, -1, 1.5, True, None, "2"])
def test_invalid_quantum_rejected(quantum):
    with pytest.raises(ValueError):
        round_robin([P("A", 0, 1)], quantum)


def test_accepts_any_iterable():
    s = fcfs(p for p in [P("A", 0, 1)])
    assert s.processes == (P("A", 0, 1),)
