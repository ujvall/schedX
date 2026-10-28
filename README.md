# SchedX

**Machine-Learning-Assisted Adaptive CPU Scheduling**

> **Research question:** Can process-behavior prediction improve
> scheduling-policy selection or time-quantum decisions compared with fixed,
> rule-based CPU scheduling?

SchedX is a reproducible CPU scheduling simulator. The long-term goal is to
generate workload datasets, train ML models that predict process behaviour,
and experimentally compare adaptive scheduling against classic algorithms.

## Status

**Phase 4 — improved online workload estimation: complete.**

* Completed bursts seen online are length-biased: short ones finish first.
  Phase 4 adds a Kaplan–Meier estimator that treats unfinished jobs as
  right-censored. Within the identified range it cuts the error in the
  estimated mean burst about fivefold (training workloads).
* A separate controller using these estimates was selected on validation,
  frozen, and evaluated once on the established test split and on 600 fresh
  workloads (new seed). Statistics: sign-flip permutation tests of the mean
  log ratio, paired bootstrap CIs, and Holm over a pre-specified family.
* **Result:** no statistically supported improvement over the frozen
  Phase 3 controller. The primary comparison gave a geometric-mean change in
  turnaround of +0.64% on test (CI +0.02% to +1.28%, Holm p = 0.14) and
  +0.08% on the fresh set (CI −0.35% to +0.51%). The new controller
  switches quanta far more often and worsens response time. Better
  estimates did not yield better decisions.
* A pre-specified secondary result: the Phase 3 controller's small benefit
  over RR1 replicated on the fresh set (−0.51%, CI −0.86% to −0.19%).

Details: [docs/phase4_observation_model.md](docs/phase4_observation_model.md)
and [reports/phase4/summary.md](reports/phase4/summary.md).

**Phase 3 — feature-aware adaptive quantum selection: complete.**

* Round Robin can now change its quantum mid-run (`adaptive_round_robin`).
  A controller picks the quantum from features available at decision time:
  submission-time features, or online observations of the schedule that
  actually ran (only completed bursts are visible).
* Protocol: candidates RR1/2/4/8; best fixed quantum and controller chosen
  on train + validation only; controller frozen and committed before a
  single evaluation on 300 held-out test workloads; paired bootstrap CIs,
  Wilcoxon signed-rank tests, and Holm correction.
* **Result:** the best fixed quantum is RR1. Against it, the frozen
  controller (online decision tree) gave a geometric-mean per-workload change
  in average turnaround of **−0.38%** (95% CI −0.75% to −0.04%; the change in
  mean turnaround is −0.17%). The pre-specified Wilcoxon test gives
  p = 0.50 (Holm p = 1). **This is not a statistically supported
  improvement:** at most weak evidence of a small benefit. A per-workload
  oracle would gain about 3%. Burst variability predicts the right quantum,
  but it is poorly observable online.
* A post-hoc statistical audit
  ([reports/phase3/audit/AUDIT.md](reports/phase3/audit/AUDIT.md)) reproduces
  every committed number and explains why the CI and the Wilcoxon test differ.
  Rerun it with `python scripts/audit_phase3.py`.

Details: [docs/phase3_adaptive_scheduling.md](docs/phase3_adaptive_scheduling.md)
and [reports/phase3/summary.md](reports/phase3/summary.md).

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

**Not implemented yet:** I/O-blocking simulation, context-switch overhead,
the dashboard, and any validation against a real operating system.

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
  observation.py  what an online scheduler has observed by time t      (Phase 3)
  adaptive_eval.py counterfactual candidate runs vs inference path     (Phase 3)
  controller.py   feature -> quantum controller (simulator-independent)(Phase 3)
  stats.py        paired bootstrap CIs, Wilcoxon, effect sizes, Holm   (Phase 3)
  phase3.py       Phase 3 fit/select and test-stage pipeline           (Phase 3)
  estimator.py    Kaplan-Meier estimate of unfinished bursts           (Phase 4)
  phase4.py       Phase 4 fit/select, inference, and analysis          (Phase 4)
configs/              phase2.json, phase3.json, phase4.json
scripts/              demo, generate_dataset, train_evaluate, plot_results, predict,
                      phase3_train, phase3_evaluate, phase3_plots, audit_phase3,
                      phase4_baseline, phase4_train, phase4_evaluate, phase4_plots
reports/phase2/       generated metrics.json, summary.md, figures (committed)
reports/phase3/       frozen controller, validation table, test results, figures, audit/ (committed)
reports/phase4/       baseline check, frozen controller, estimator diagnostic, results, figures (committed)
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

## Reproducing Phase 3

Requires the Phase 2 dataset and models (previous section). Then:

```bash
python scripts/phase3_train.py      # train+validation only -> artifacts/phase3/, reports/phase3/frozen_controller.json
python scripts/phase3_evaluate.py   # frozen controllers, test workloads once -> reports/phase3/results.json, summary.md
python scripts/phase3_plots.py      # -> reports/phase3/figures/*.png
```

`phase3_evaluate.py` refuses to run if `configs/phase3.json` or the dataset
changed since the controllers were frozen.

## Reproducing Phase 4

Requires the Phase 2 dataset and the frozen Phase 3 controllers (previous
sections). Then:

```bash
python scripts/phase4_baseline.py   # read-only check that the frozen Phase 3 controller reproduces
python scripts/phase4_train.py      # train+validation only -> artifacts/phase4/, reports/phase4/frozen_controller.json
python scripts/phase4_evaluate.py   # frozen controllers, test split + fresh set once -> reports/phase4/results.json, summary.md
python scripts/phase4_plots.py      # -> reports/phase4/figures/*.png
```

`phase4_evaluate.py` refuses to run if any configuration or the dataset
changed after freezing.

## Limitations

* The Phase 1 simulator schedules CPU bursts only: no I/O blocking, one
  burst per process. Waiting time is pure ready-queue waiting. Phase 2
  records I/O in its dataset but does not simulate it.
* Phase 2 results are on synthetic workloads from one generator, one
  dataset seed, and one split; see `docs/phase2_ml.md` §8.
* Phase 3 uses an open-loop CPU-only projection with free context switches
  and decides once per workload; see `docs/phase3_adaptive_scheduling.md` §12.
  Phase 4 shares these limits; see `docs/phase4_observation_model.md` §10.
* Single CPU.
* Context-switch overhead is counted but takes zero simulated time.
* Integer time only.
* Priority scheduling has no aging, so low-priority processes can starve
  under sustained high-priority load (they still complete in finite workloads).

## Roadmap

1. ~~Scheduling core and metrics~~ (Phase 1)
2. ~~Synthetic workload dataset, leakage-safe features, ML classifiers,
   offline evaluation, initial plots~~ (Phase 2)
3. ~~Adaptive quantum selection with a leakage-safe, pre-specified
   comparison against fixed quanta~~ (Phase 3; no supported improvement)
4. ~~Censoring-aware online observation with a frozen, pre-specified
   comparison and a fresh confirmation set~~ (Phase 4; no supported
   improvement)
5. Candidate next steps: switching-aware selection criteria, periodic
   re-decision, context-switch overhead, I/O-aware simulation
6. Interactive dashboard (Streamlit)
