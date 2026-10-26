# SchedX Architecture (Phase 1)

Phase 1 delivers a deterministic, CPU-burst-only scheduling simulator and a
shared metrics module. It has no third-party runtime dependencies and no
knowledge of the future ML or UI layers.

## Modules

| Module | Responsibility |
|---|---|
| `schedx/models.py` | `Process`, `Slice`, `Schedule` data classes and `validate_processes`. |
| `schedx/scheduler.py` | The shared simulation engine (`_simulate`) and the five public algorithms. |
| `schedx/metrics.py` | `compute_metrics(schedule)` → per-process and aggregate metrics. |
| `scripts/demo.py` | Runs every algorithm on a sample workload and prints a text Gantt chart. |

Dependency direction: `metrics → models ← scheduler`. Metrics never call the
scheduler, so any timeline (including one produced by a future adaptive or
ML-driven policy) can be measured with the same code.

## Time unit

All times are **integer ticks**, an abstract unit. Integers keep every
schedule exact and reproducible. Inputs that are not `int` (including `bool`
and `float`) are rejected with `TypeError`.

## Process model

```python
Process(pid: str, arrival: int, burst: int, priority: int = 0)
```

* `pid` — unique, non-empty string.
* `arrival` — tick at which the process enters the ready queue (`>= 0`).
* `burst` — total CPU time required (`> 0`).
* `priority` — **lower number = higher priority**. Used only by the priority
  scheduler.

Invalid workloads (duplicate pids, non-positive bursts, negative arrivals,
wrong types) raise before any simulation happens. An empty workload is valid
and produces an empty timeline.

### Extending to I/O bursts (future phase)

Phase 1 does **not** simulate I/O. Each process has exactly one CPU burst and
is never blocked. The engine is structured so I/O can be added without a
rewrite:

* The engine tracks per-process `remaining` CPU time for the current burst.
* A process enters the ready queue only through `admit()`. An I/O-completion
  event would re-admit a process through the same path.
* Selection policies (`key` functions) see only the ready queue, so they are
  unaffected by a blocked state.

A future `Process` would carry a sequence of alternating CPU/I/O bursts;
waiting-time metrics would then need to distinguish ready-queue waiting from
I/O blocking (see Metrics).

## Scheduling events (timeline)

A run produces a `Schedule(algorithm, processes, timeline)`. The timeline is a
tuple of `Slice(start, end, pid)` over half-open intervals `[start, end)`:

* `pid` is a process id for an execution interval, or `None` for an **idle**
  interval (no process was ready).
* Slices are contiguous from tick 0 to the last completion, never overlap, and
  have positive length.
* Consecutive slices of the same process are merged: re-dispatching the
  process that was already running is not a real switch.

The timeline is the single source of truth: metrics and (later) Gantt charts
are derived from it, so they cannot disagree.

## Engine

`_simulate(algorithm, processes, key, preemptive, quantum)` is a
discrete-event loop:

1. Admit every process with `arrival <= t` to the ready queue.
2. If the queue is empty, emit an idle slice up to the next arrival.
3. Otherwise dispatch the ready process with the minimum `key`. It runs for
   its remaining time, capped by `quantum` (if any) and, when `preemptive`,
   by the time until the next arrival.
4. Advance `t`, admit arrivals at the new `t`, then re-queue the process if
   it still has remaining time.

Because preemptive algorithms stop at every arrival and re-evaluate, a
process is preempted exactly when a newly arrived process has a strictly
better key.

### Algorithms

| Function | Ready-queue key | Preemptive | Notes |
|---|---|---|---|
| `fcfs(ws)` | enqueue order | no | Arrival order; same-tick arrivals in input order. |
| `sjf(ws)` | `(burst, arrival, index)` | no | Chooses only among processes that have arrived. |
| `srtf(ws)` | `(remaining, arrival, index)` | yes | Equal remaining time does not preempt (earlier arrival wins). |
| `round_robin(ws, quantum)` | enqueue order | by quantum | `quantum` must be a positive int. |
| `priority(ws, preemptive=False)` | `(priority, arrival, index)` | optional | Non-preemptive by default; equal priority never preempts. |

### Deterministic handling of simultaneous events

* Processes arriving at the same tick are admitted in **input order**.
* All ties are broken by earlier arrival, then input order.
* When a slice ends at tick `t` and a process arrives at `t`, the arrival is
  admitted **before** the interrupted process is re-queued. In Round Robin the
  newcomer therefore runs before the preempted process.
* A process that finishes at `t` frees the CPU for processes arriving at `t`
  with no idle gap.

## Metrics

All metrics use the same definitions for every algorithm.

Per process:

* **Turnaround** = completion − arrival.
* **Response** = first dispatch − arrival.
* **Waiting** = turnaround − CPU time received. In this CPU-only simulator
  that is exactly the time spent in the ready queue. It contains **no I/O
  blocking time**, because none is modelled. When I/O is added, blocked time
  must be reported separately and excluded from ready-queue waiting.

Aggregate:

* **Averages** of waiting, turnaround, and response are arithmetic means over
  all processes.
* **Measurement interval** = `[first arrival, last completion]`. Idle time
  before the first arrival is **excluded** (the system had no work); idle gaps
  between arrivals are **included**. There is never idle time after the last
  completion.
* **Throughput** = number of completed processes / interval length
  (processes per tick).
* **CPU utilization** = busy (executing) ticks / interval length, in `[0, 1]`.
* **Context switches** = number of dispatches of a process *different* from
  the previously running one.
  * The first dispatch of the run (from an initially idle CPU) is not counted.
  * Idle gaps do not reset the previous process: `A, idle, A` = 0 switches;
    `A, idle, B` = 1.
  * A preemption counts only if a different process takes over (merged
    slices are not counted).
  * Switch **overhead is not modelled**: switches take zero time.
* For an empty workload, all averages and rates are reported as `0.0`.

`compute_metrics` also validates the schedule: it raises `ValueError` if any
process receives more or less CPU time than its burst, or is dispatched before
its arrival.

## Testing strategy

* `tests/test_algorithms.py` — small hand-computed timelines for every
  algorithm, covering arrival order, idle gaps, preemption, quantum
  boundaries, queue ordering, and tie-breaking.
* `tests/test_metrics_and_validation.py` — metric formulas, the measurement
  interval, context-switch rules, and rejection of invalid inputs.
* `tests/test_properties.py` — seeded randomized workloads (deterministic)
  checking invariants for every algorithm: contiguity, exact burst execution,
  no execution before arrival, work conservation, waiting time recomputed
  independently from the timeline, Round Robin quantum compliance, and SRTF's
  optimality for average turnaround time.
