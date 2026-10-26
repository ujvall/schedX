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

**Hindsight experiment (not deployable).** A Random Forest trained on
A + B + C features measures what lifetime totals add on top of the
deployable features. It is a research comparison, not a prediction a
scheduler could make. (An early A + C-only run scored *below* the deployable
model: the label counts I/O-bound *processes*, while the C features are
time-weighted workload aggregates that blur per-process composition. C
features are therefore not an upper bound on their own, which is why the
hindsight experiment adds C to A + B rather than replacing B.)

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
  the per-workload oracle. Each rule's per-workload regret is compared with
  the baseline's on the same workloads, with a seeded paired bootstrap 95%
  CI (2,000 resamples). Predicted class, confidence, and class
  probabilities are recorded per row in `artifacts/predictions_test.csv`.

This estimates the value of the class signal *under the projection*. It is
not adaptive scheduling and does not show that ML improves scheduling.

### Information available to the Phase 3 decision

At a decision instant the adaptive scheduler may use only: the submitted job
list (A), and per-process history it has itself observed — completed CPU
bursts, completed I/O waits, and terminations (B). It may not use lifetime
totals, future bursts, the label, or generator parameters. The Phase 2
features are defined so the same functions (`pre_execution_features`,
`online_features`) can be computed from that observed history.

## 7. Results (configs/phase2.json, dataset version `67da9d294be3`)

Full tables: [`reports/phase2/summary.md`](../reports/phase2/summary.md);
figures in `reports/phase2/figures/`. Numbers below are copied from that
generated report.

* **Dataset:** 1,500 workloads, 21,453 processes, 4,500 feature rows.
  Classes: 484 `cpu_bound`, 489 `mixed`, 527 `io_bound`. 12% of workloads
  have a label different from their generator regime; 3% of processes differ
  from their intended kind.
* **Split:** 900 / 300 / 300 workloads (2,700 / 900 / 900 rows), class
  proportions matched by stratification.
* **Test results, deployable features (A + B):**

  | model | val macro-F1 | test acc | macro F1 | acc k=1 / 2 / 3 |
  |---|---|---|---|---|
  | Logistic Regression | 0.869 | 0.876 | 0.874 | 0.800 / 0.913 / 0.913 |
  | Decision Tree | 0.813 | 0.864 | 0.864 | 0.787 / 0.893 / 0.913 |
  | Random Forest | 0.858 | 0.892 | 0.892 | 0.823 / 0.920 / 0.933 |

  Logistic Regression is the **validation-selected** model. Random Forest
  scores higher on test, but switching to it because of that would use the
  test set for model selection, so the selection stands. `mixed` is the
  hardest class (F1 0.80–0.84); `cpu_bound` and `io_bound` are almost never
  confused with each other. Accuracy rises with the observation horizon.
* **Hindsight (A + B + C, not deployable):** Random Forest test accuracy
  0.899 — only +0.007 over deployable features.
* **Policy-value experiment:** fixed baseline RR2 (chosen on
  train + validation); learned map `cpu_bound → FCFS`, `mixed → RR2`,
  `io_bound → RR4`. On test, every class-conditioned rule had slightly lower
  mean regret than RR2 (Δ between −0.006 and −0.014), but **every 95% CI
  includes 0**, including the true-class map. The per-workload oracle is
  ~5.7% better than RR2. Interpretation: under this projection the
  three-class label is a weak proxy for the best quantum; per-workload
  burst statistics, not the class alone, may be needed in Phase 3.
* **Feature importance** (Decision Tree, Random Forest; impurity-based):
  observed CPU-burst mean/median and the observed share of I/O-dominant
  processes dominate. Impurity importance describes how the fitted models
  split, is biased toward continuous and correlated features, and does not
  establish causation.

## 8. Limitations

* **Synthetic data only.** Workloads come from a parametric generator; real
  process traces may differ in distribution and in what separates classes.
  High accuracy here mostly shows that the generator's classes are separable
  from early bursts.
* **Label definition is a choice.** The thresholds (0.5 per process;
  0.25 / 0.75 share per workload) are documented but arbitrary; other
  definitions would change results.
* **Burst-count horizon, not wall-clock.** B features assume the first k
  bursts of *every* process are observed. In a real run, late arrivals are
  observed later, and observation time depends on scheduling.
* **No I/O simulation.** I/O is recorded but never simulated; the policy
  experiment uses an open-loop CPU-only projection with zero switch cost.
* **Small effects, one dataset seed.** The policy-value differences are
  within noise; results are from one master seed and one split seed.
* **Timing numbers** are single-run wall-clock measurements on one laptop
  and vary between runs; `metrics.json` changes slightly on every rerun for
  that reason.
* `load_model` uses joblib (pickle): only load model files you created.

## 9. Reproduce

```bash
python scripts/generate_dataset.py --config configs/phase2.json
python scripts/train_evaluate.py --config configs/phase2.json
python scripts/plot_results.py
python scripts/predict.py --seed 12345
```
