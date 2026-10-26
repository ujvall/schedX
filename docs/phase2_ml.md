# Phase 2 — Workload Dataset, Features, and Classification

Phase 2 builds a reproducible workload dataset, defines leakage-safe
features, and evaluates whether a workload's class can be predicted from
information a real scheduler would have. It does **not** implement adaptive
scheduling (Phase 3) and does **not** measure real OS performance.

## 1. Workload model

A *workload* is a batch of processes submitted to one CPU. Each process has:

* `arrival` (int tick `>= 0`) and `priority` (int, 0–4, lower = higher).
* An alternating trace `CPU₁, IO₁, CPU₂, IO₂, …, CPUₙ` of integer bursts
  (`>= 1` tick): `n` CPU bursts and `n − 1` I/O waits. A process ends on a
  CPU burst.

The I/O waits are **recorded, not simulated**: the Phase 1 engine still runs
CPU bursts only. No experiment in Phase 2 claims to model I/O blocking.

### Generation rules (`schedx.workload`)

All randomness comes from `numpy.random.default_rng([master_seed, index])`,
so every workload can be regenerated on its own from the two integers stored
with it. Per workload:

1. `regime` ∈ {`cpu`, `io`, `mixed`} drawn uniformly. This is a generation
   parameter, **not** the label.
2. `n_processes` ~ uniform integer in `[process_count_min, process_count_max]`.
3. Arrivals: gaps ~ Exponential(`mean_interarrival`), with
   `mean_interarrival` ~ U(`interarrival_min`, `interarrival_max`) per
   workload (workload intensity), rounded to integer ticks; the first process
   arrives at 0.
4. Each process is drawn as an *intended* I/O-heavy process with probability
   `p_io` = 0.1 (`cpu`), 0.9 (`io`), or U(0.3, 0.7) (`mixed`); otherwise
   CPU-heavy.
5. Per process: a mean CPU burst, a mean I/O wait, and a burst count, drawn
   from kind-specific ranges (see `GeneratorConfig`). CPU-heavy: long CPU
   bursts, short I/O, few bursts. I/O-heavy: short CPU bursts, long I/O,
   many bursts.
6. Each individual burst ~ Gamma(shape = 1/cv², scale = mean·cv²), rounded
   up to an integer ≥ 1, where `burst_cv` ~ U(`cv_min`, `cv_max`) per
   workload controls burst-length variability.

Because bursts are random, an "intended" I/O-heavy process can end up
CPU-dominated and vice versa. Labels are therefore computed from the
**realized** traces, not from the intent.

### Target labels

Per process, over its whole lifetime:

    io_fraction = total_io / (total_cpu + total_io)
    process_kind = "io_bound" if io_fraction > 0.5 else "cpu_bound"

Per workload, with `share_io` = fraction of processes whose kind is
`io_bound`:

| `workload_class` | rule |
|---|---|
| `cpu_bound` | `share_io <= 0.25` |
| `io_bound`  | `share_io >= 0.75` |
| `mixed`     | otherwise |

The label is a deterministic function of the realized traces; tests
re-derive it from the stored bursts.

## 2. Feature availability

Every column is assigned exactly one category (`schedx.features.CATALOG`):

| Category | Meaning | Used in main model? |
|---|---|---|
| `id` | identifiers (`workload_id`, `pid`) | no |
| `generation` | generator parameters and seeds (`regime`, `master_seed`, `mean_interarrival`, `burst_cv`, …) | no — they encode how the label was produced |
| `target` | `workload_class` | no — it is the answer |
| `label_equivalent` | `share_io`, `process_kind`, per-process `io_fraction` | never — the label is a threshold on them |
| **A** — pre-execution | known when the batch is submitted: process count, arrival times, priorities | yes |
| **B** — online | measured from each process's first *k* CPU bursts and the *k − 1* I/O waits between them | yes |
| **C** — retrospective | lifetime totals: total CPU/I/O, lifetime I/O fraction, burst counts, offered load | only in the clearly labelled retrospective experiment |

### Decision-time assumption

Category A assumes **batch submission**: the scheduler sees the list of
submitted jobs (count, arrival times, priorities) but none of their bursts.

Category B uses a **burst-count horizon** `k ∈ {1, 2, 3}`: the snapshot
contains, for every process, its first `min(k, n)` CPU bursts and the I/O
waits that *completed* before its k-th CPU burst (so at `k = 1` no I/O has
been observed and all I/O features are missing). A process with fewer than
`k` CPU bursts is observed to have terminated. Burst `k + 1` onward is never
read. This is schedule-independent: the *values* observed do not depend on
which policy ran, only *when* they become observable. Phase 3 must compute
the same features from the simulator's own observed history at its decision
instant.

## 3. Dataset layout

`scripts/generate_dataset.py --config configs/phase2.json` writes to
`artifacts/dataset/` (not committed; regenerated deterministically):

* `processes.csv` — one row per process: raw traces and lifetime facts.
* `workloads.csv` — one row per workload: generation parameters, label,
  retrospective summaries.
* `features.csv` — one row per (workload, horizon `k`) snapshot: A, B, C
  features plus `workload_id` and the target.
* `dataset_meta.json` — config, seeds, row counts, class distribution.

Full column-by-column definitions: [`data_dictionary.md`](data_dictionary.md).

## 4. Leakage prevention

* **Grouped split.** Each workload contributes three correlated rows (one per
  horizon). Splitting is done on `workload_id`, stratified by class:
  60% train / 20% validation / 20% test. No workload appears in two splits
  (asserted in code and tests).
* **Feature whitelist.** The model input is built from the catalog by
  category (A + B), never by dropping known-bad columns. Tests assert that
  no `target`, `label_equivalent`, `generation`, `id`, or C column can enter
  the main feature set.
* **No future bursts.** B features read only the first `k` CPU bursts;
  a test mutates later bursts and checks that features are unchanged.
* **Preprocessing inside pipelines.** Imputation and scaling are fitted on
  the training fold only (scikit-learn `Pipeline`).
* **Tuning on validation only.** Hyperparameters are chosen by macro-F1 on
  the validation split. The selected configuration is then refitted on
  train + validation and evaluated **once** on the test split.
* **Model selection for downstream use** (the model whose predictions feed
  the offline policy experiment) is also decided on validation macro-F1.

## 5. Models

| Model | Pipeline | Grid (validation-tuned) |
|---|---|---|
| Logistic Regression (baseline) | median imputer + missing indicators → `StandardScaler` → `LogisticRegression` | `C` ∈ {0.1, 1, 10} |
| Decision Tree | median imputer + missing indicators → `DecisionTreeClassifier` | `max_depth` ∈ {3, 5, 8, None}, `min_samples_leaf` ∈ {1, 5} |
| Random Forest | median imputer + missing indicators → `RandomForestClassifier` (200 trees) | `max_depth` ∈ {None, 8}, `min_samples_leaf` ∈ {1, 3} |

All `random_state`s come from the config. Metrics on the test split:
accuracy, macro precision/recall/F1, per-class precision/recall/F1,
confusion matrix, fit time, and per-row inference time; plus test accuracy
per horizon `k`.

**Retrospective experiment (not deployable).** A Random Forest trained on
A + C features measures how recoverable the label is from lifetime totals.
It is an upper-bound sanity check, not a prediction a scheduler could make.

## 6. Offline policy-value experiment

Classification accuracy does not show that scheduling improves. Phase 2 runs
a deliberately limited offline experiment to ask whether the predicted class
*could* plausibly inform a policy choice.

* **Candidate decisions** (non-clairvoyant only, since real schedulers do not
  know bursts): FCFS and Round Robin with each quantum in the config.
* **CPU-only burst-trace projection.** Each CPU burst becomes an independent
  job released at its contention-free time
  (`arrival + Σ earlier CPU bursts + Σ earlier I/O waits`). This is an
  *open-loop* approximation: real I/O completion would be delayed by CPU
  contention, which is not modelled. It is a CPU-only experiment, not an
  I/O-aware simulation.
* **Score.** Average job turnaround under each candidate, normalized by the
  best candidate on the same workload (`regret = score / best ≥ 1`).
* **Fixed-policy baseline.** The single candidate with the lowest mean
  regret over **train + validation** workloads.
* **Class → policy mapping.** For each class, the candidate with the lowest
  mean regret over train + validation workloads of that (true) class. A
  mapping can also be supplied explicitly in the config.
* **Evaluated on test workloads:** fixed baseline, mapping applied to the
  *predicted* class (per horizon), mapping applied to the *true* class, and
  the per-workload oracle. Predicted class and confidence are recorded per
  row.

This estimates the value of the class signal *under the projection*. It is
not adaptive scheduling and does not show that ML improves scheduling.

## 7. Reproduce

```bash
python scripts/generate_dataset.py --config configs/phase2.json
python scripts/train_evaluate.py --config configs/phase2.json
python scripts/plot_results.py
python scripts/predict.py --seed 12345
```

Results are summarized in [`reports/phase2/summary.md`](../reports/phase2/summary.md).
