# SchedX

**Machine-Learning-Assisted Adaptive CPU Scheduling**

> **Research question:** Can process-behavior prediction improve
> scheduling-policy selection or time-quantum decisions compared with fixed,
> rule-based CPU scheduling?

SchedX is a reproducible CPU scheduling simulator. The long-term goal is to
generate workload datasets, train ML models that predict process behaviour,
and experimentally compare adaptive scheduling against classic algorithms.

## Status

**Phase 2 — workload dataset, features, and offline classification: complete.**

* Reproducible generator for CPU-bound, I/O-bound, and mixed workloads
  (alternating CPU/I/O traces; labels derived from the realized traces).
* Every column is tagged by when it is available: **A** pre-execution,
  **B** online (first *k* bursts), **C** retrospective, or never a feature.
* Leakage-safe evaluation: workload-grouped stratified 60/20/20 split,
  preprocessing inside pipelines, tuning on validation only, one test run.
* Logistic Regression, Decision Tree, and Random Forest. On held-out test
  workloads with deployable (A + B) features: accuracy 0.876 / 0.864 /
  0.892, macro-F1 0.874 / 0.864 / 0.892 (details in
  [reports/phase2/summary.md](reports/phase2/summary.md)).
* Offline policy-value experiment on a CPU-only projection: class-conditioned
  quantum choices were not reliably better than a fixed RR2 baseline (all
  95% CIs include 0). This is **not** adaptive scheduling; that is Phase 3.

Design, leakage rules, results, and limitations:
[docs/phase2_ml.md](docs/phase2_ml.md). Column definitions:
[docs/data_dictionary.md](docs/data_dictionary.md).

**Phase 1 — scheduling core: complete.**

* Deterministic discrete-event simulator (integer time ticks).
* FCFS, SJF (non-preemptive), SRTF (preemptive), Round Robin (configurable
  quantum), and Priority scheduling (non-preemptive by default, preemptive
  optional; lower number = higher priority).
* Structured execution/idle timeline suitable for Gantt charts.
* Shared metrics: per-process waiting, turnaround, and response time;
  averages; throughput; CPU utilization; context-switch count.
* Pytest suite with hand-verified cases and seeded randomized invariant tests.

**Not implemented yet:** adaptive scheduling, I/O-blocking simulation, the
dashboard, and any validation against a real operating system.

## Architecture

```
src/schedx/
  models.py       Process, Slice (timeline interval), Schedule; input validation
  scheduler.py    shared simulation engine + fcfs, sjf, srtf, round_robin, priority
  metrics.py      compute_metrics(schedule) -> ScheduleMetrics
  workload.py     seeded CPU/IO workload generator and label rules     (Phase 2)
  features.py     feature catalog (availability A/B/C) and extraction  (Phase 2)
  ml.py           grouped split, pipelines, evaluation, persistence    (Phase 2)
  policy_eval.py  offline policy-value experiment                      (Phase 2)
configs/phase2.json   dataset, split, model, and policy settings
scripts/              demo, generate_dataset, train_evaluate, plot_results, predict
reports/phase2/       generated metrics.json, summary.md, figures (committed)
artifacts/            generated dataset, models, predictions (not committed)
tests/                pytest suite
docs/                 architecture, Phase 2 design/results, data dictionary
```

The scheduling core (`models`, `scheduler`, `metrics`) uses only the
standard library; NumPy, pandas, scikit-learn, and Matplotlib are used by the
Phase 2 modules and scripts.

Every algorithm is the same engine configured with a ready-queue ordering,
a preemption flag, and an optional quantum. Metrics are computed only from
the produced timeline, so any scheduler — including future adaptive ones —
is measured identically. See [docs/architecture.md](docs/architecture.md)
for the process model, event handling, tie-breaking, and exact metric
definitions.

## Setup

Requires Python 3.11+. Installing the package pulls NumPy, pandas,
scikit-learn, and Matplotlib; the `dev` extra adds pytest.

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

## Reproducing Phase 2

Each step is deterministic given `configs/phase2.json` (whole pipeline ≈ 1 minute
on a laptop):

```bash
python scripts/generate_dataset.py      # -> artifacts/dataset/*.csv
python scripts/train_evaluate.py        # -> artifacts/models/, reports/phase2/metrics.json + summary.md
python scripts/plot_results.py          # -> reports/phase2/figures/*.png
python scripts/predict.py --seed 12345  # classify a newly generated workload
```

## Limitations

* The Phase 1 simulator schedules CPU bursts only: no I/O blocking, one
  burst per process. Waiting time is pure ready-queue waiting. Phase 2
  records I/O in its dataset but does not simulate it.
* Phase 2 results are on synthetic workloads from one generator, one
  dataset seed, and one split; see `docs/phase2_ml.md` §8.
* Single CPU.
* Context-switch overhead is counted but takes zero simulated time.
* Integer time only.
* Priority scheduling has no aging, so low-priority processes can starve
  under sustained high-priority load (they still complete in finite workloads).

## Roadmap

1. ~~Scheduling core and metrics~~ (Phase 1)
2. ~~Synthetic workload dataset, leakage-safe features, ML classifiers,
   offline evaluation, initial plots~~ (Phase 2)
3. Adaptive scheduling: ML-guided policy selection and time-quantum choice,
   ideally with an I/O-aware simulator
4. Comparative experiments against fixed algorithms
5. Interactive dashboard (Streamlit)
