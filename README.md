# SchedX

**Machine-Learning-Assisted Adaptive CPU Scheduling**

> **Research question:** Can process-behavior prediction improve
> scheduling-policy selection or time-quantum decisions compared with fixed,
> rule-based CPU scheduling?

SchedX is a reproducible CPU scheduling simulator. The long-term goal is to
generate workload datasets, train ML models that predict process behaviour,
and experimentally compare adaptive scheduling against classic algorithms.

## Status

**Phase 1 — scheduling core: complete.**

* Deterministic discrete-event simulator (integer time ticks).
* FCFS, SJF (non-preemptive), SRTF (preemptive), Round Robin (configurable
  quantum), and Priority scheduling (non-preemptive by default, preemptive
  optional; lower number = higher priority).
* Structured execution/idle timeline suitable for Gantt charts.
* Shared metrics: per-process waiting, turnaround, and response time;
  averages; throughput; CPU utilization; context-switch count.
* Pytest suite with hand-verified cases and seeded randomized invariant tests.

**Not implemented yet:** dataset generation, machine learning, adaptive
scheduling, plots, dashboard, I/O simulation, and any validation against a
real operating system.

## Architecture

```
src/schedx/
  models.py     Process, Slice (timeline interval), Schedule; input validation
  scheduler.py  shared simulation engine + fcfs, sjf, srtf, round_robin, priority
  metrics.py    compute_metrics(schedule) -> ScheduleMetrics
scripts/demo.py runs every algorithm on a sample workload
tests/          pytest suite
docs/           architecture and metric definitions
```

Every algorithm is the same engine configured with a ready-queue ordering,
a preemption flag, and an optional quantum. Metrics are computed only from
the produced timeline, so any scheduler — including future adaptive ones —
is measured identically. See [docs/architecture.md](docs/architecture.md)
for the process model, event handling, tie-breaking, and exact metric
definitions.

## Setup

Requires Python 3.11+. No runtime dependencies; pytest for tests.

```bash
git clone https://github.com/ujvall/schedX.git
cd schedX
python -m venv .venv
```

Activate the environment:

```bash
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate
```

Install:

```bash
pip install -e ".[dev]"
```

## Running tests

```bash
python -m pytest
```

## Example

```python
from schedx import Process, compute_metrics, round_robin

workload = [
    Process("P1", arrival=0, burst=5),
    Process("P2", arrival=1, burst=3),
    Process("P3", arrival=2, burst=1),
]

schedule = round_robin(workload, quantum=2)
for s in schedule.timeline:
    print(s.start, s.end, s.pid or "idle")

m = compute_metrics(schedule)
print(m.avg_waiting, m.avg_turnaround, m.cpu_utilization, m.context_switches)
```

Or run all algorithms side by side:

```bash
python scripts/demo.py
```

## Limitations

* CPU bursts only: no I/O, no blocking, one burst per process. Waiting time
  is pure ready-queue waiting.
* Single CPU.
* Context-switch overhead is counted but takes zero simulated time.
* Integer time only.
* Priority scheduling has no aging, so low-priority processes can starve
  under sustained high-priority load (they still complete in finite workloads).

## Roadmap

1. ~~Scheduling core and metrics~~ (Phase 1)
2. Synthetic workload / dataset generation (NumPy, Pandas)
3. ML classifiers for process-behaviour prediction (scikit-learn)
4. Adaptive scheduling: ML-guided policy selection and time-quantum choice
5. Comparative experiments against fixed algorithms
6. Visualisations (Matplotlib) and an interactive dashboard (Streamlit)
