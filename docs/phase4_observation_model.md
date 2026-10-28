# Phase 4 — Improved Online Workload Estimation

Phase 4 asks whether a better estimate of *unfinished* work improves the
online quantum decision beyond the frozen Phase 3 controller. An improvement
is **not** assumed. Everything is an offline simulation on synthetic
workloads with the CPU-only burst-trace projection; no kernel-level scheduling
is performed.

Phases 1–3.1 code, configurations, and committed evidence are unchanged.
Phase 4 adds new modules, scripts, `configs/phase4.json`, and
`reports/phase4/`.

## 1. Baseline reproduction

`scripts/phase4_baseline.py` reloads the frozen Phase 3 controller (config
hash `ae9db57acecd`) and re-runs it on the 300 established test workloads.
Result (`reports/phase4/baseline_reproduction.json`): **0 mismatches** with
the committed `reports/phase3/per_workload_test.csv` across all six metrics,
for both the controller and RR1, including decision sources.

| | Phase 3 controller | RR1 |
|---|---|---|
| Configuration | online `tree_classifier`, depth 3, min leaf 20, decides after 20 completions; A + O features | fixed quantum 1 |
| Quantum chosen (test) | RR1 kept 258, RR1→RR8 25, no decision point 17 | — |
| Avg turnaround | 189.16 | 189.49 |
| Avg waiting | 180.40 | 180.73 |
| Avg response | 25.05 | 20.33 |
| Throughput (jobs/tick) | 0.1344 | 0.1344 |
| CPU utilization | 0.913 | 0.913 |
| Context switches | 552.1 | 575.8 |

## 2. What is observable, and when

Jobs come from Phase 2's burst-trace projection: each CPU burst of a process
becomes a job released at its contention-free time.

| Information | Available | Category |
|---|---|---|
| Process count, process arrival times, priorities | at submission (batch-submission assumption) | A |
| That a job (burst) has been released, and when | from its release tick on | O |
| CPU time a released job has received so far | continuously, from the schedule that actually ran | O |
| Length of a burst | **only once it has completed** | O |
| Busy time, context switches so far | continuously | O |
| Length of an unfinished burst | never before completion; only a lower bound ("longer than the CPU received so far") | censored |
| Jobs not yet released | not at all (not even their existence beyond the A-level process list) | — |
| Lifetime totals, true burst distribution, labels, policy outcomes | only retrospectively | C / never |

## 3. Why completed bursts are biased

The controller runs the default quantum RR1 until the m-th job completes.
Under RR1 every pending job receives CPU at roughly the same rate, so the
jobs that complete first are the **short** ones. The completed bursts are a
length-biased sample from the low end of the distribution, and long jobs are
still running. Statistics of completed bursts therefore underestimate the
mean and misrepresent the variability. The Phase 3 train-only diagnostic
found that the observed CV correlates only 0.42 with the true burst CV.

## 4. Improved estimator (`schedx/estimator.py`)

Each unfinished released job is **right-censored**: its burst is longer
than the CPU it has received. A discrete-time **Kaplan–Meier** estimator
combines completed bursts (events) with these lower bounds. It is the
simplest standard estimator for censored durations, with no tunable
parameters.

Features (category **E**, computed only from the `Observation`):

| Feature | Meaning |
|---|---|
| `km_tau` | largest observed time (completed burst or CPU received); the limit of what is identified |
| `km_censored_frac` | share of released jobs still unfinished |
| `km_surv_tau` | estimated P(burst > `km_tau`): probability mass that is **not** identified |
| `km_rmst` | restricted mean E[min(burst, τ)] = Σ_{k<τ} S(k) |
| `km_q25`, `km_median` | KM quantiles; missing if not reached (not identified) |
| `km_restricted_cv` | CV of the KM distribution with the unidentified tail placed at τ (a lower bound; stated as such) |
| `km_rmst_over_naive_mean` | KM restricted mean divided by the mean of completed bursts (bias indicator) |

**Identification limit.** Beyond τ, the data say nothing about the
distribution's shape. The estimator reports the unidentified mass instead of
extrapolating.

**Insufficient observations.** With fewer than 3 completed bursts, every E
feature is missing and the Phase 4 controller **falls back to the default
quantum** deterministically. With decision points at m ≥ 5 completions this
cannot occur in the experiment, but it is enforced and tested.

**Late arrivals.** Jobs not yet released at the decision tick contribute
nothing, not even a censored observation. Processes that have not arrived
are visible only through the A-level submission list.

### Retrospective check (training workloads only, before any controller was trained)

At the decision point under RR1, against the *true* bursts of the jobs
released by then:

| decision after m completions | mean \|log error\| of restricted mean: completed-only | KM | Spearman with log(T_RR8/T_RR1): completed-burst CV | KM restricted CV |
|---|---|---|---|---|
| 5 | 0.433 | 0.095 | 0.18 | 0.33 |
| 10 | 0.497 | 0.094 | 0.28 | 0.42 |
| 20 | 0.501 | 0.079 | 0.46 | 0.51 |

KM removes most of the bias in the identified range and carries somewhat
more decision-relevant signal. It slightly underestimates the mass beyond τ
(by 0.03–0.05). The training pipeline recomputes and saves these numbers.

## 5. Decision boundary

1. Dispatch with RR1 (the default `q0`, the best fixed quantum on
   train + validation).
2. At the tick `t` when the m-th job completes, build the `Observation` from
   the schedule that actually ran before `t`, plus jobs released at `t`.
3. Compute A, O, and E features from it. If the observation is missing (no
   m-th completion: the workload is too small) or insufficient
   (< 3 completed bursts), use `q0`.
4. The controller decides once. Dispatches at or after `t` use the chosen
   quantum. A slice already running keeps its quantum (engine unchanged:
   `adaptive_round_robin`).

## 6. Pre-specified protocol (fixed before any Phase 4 controller is trained)

**Splits.** The established Phase 2 grouped split (900 / 300 / 300 workloads)
is reused unchanged. The established test split was already used once to
evaluate Phase 3, and its aggregate results are known. Phase 4 therefore
also evaluates on a **fresh confirmation set**: 600 new workloads from the
same generator with master seed 2027. No result, training, or tuning ever
touches them before the final evaluation.

**Candidates.** RR quanta {1, 2, 4, 8}; decision points m ∈ {5, 10, 20};
the same three model families and hyperparameter grids as Phase 3
(`configs/phase4.json`). Seed 42.

**Compared methods.**

| Name | Description | Deployable |
|---|---|---|
| `RR1` | best fixed quantum (re-confirmed on train + validation) | yes |
| `p3_frozen` | frozen Phase 3 controller (A + O; *existing observation*) | yes |
| `p4_selected` | Phase 4 controller with A + O + E (*improved observation*), best online configuration on validation | yes |
| `p4_ablation` | **controlled ablation**: the frozen Phase 3 configuration (family, hyperparameters, m) retrained on train with A + O + E; isolates the effect of the features | yes |
| `submission_ref` | frozen Phase 3 submission-time controller (A only) | yes |
| `oracle_fixed`, `oracle_online` | per-workload best candidate / best switch target | **no** |

Training uses train workloads only. Selection (configuration of
`p4_selected`) uses validation only, by the Phase 3 criterion: mean
log(T_chosen / T_RR1) over validation workloads, ties to the simpler
configuration. The frozen controllers and this protocol are committed
before evaluation.

**Estimand.** For each workload, `log(T_method / T_comparator)`, where T is
the average job turnaround. The estimand is the mean over workloads,
reported as the geometric-mean relative change `exp(mean) − 1`.

**Test aligned with the estimand.** Two-sided paired sign-flip permutation
test of the mean log ratio: exact enumeration when ≤ 26 pairs differ,
otherwise 100,000 Monte Carlo sign flips (seeded, p = (hits + 1)/(n + 1)).
**CI:** paired percentile bootstrap over workloads, 10,000 resamples, seeded.

**Hypothesis family (Holm, primary, established test split):**

* **H1 (primary):** `p4_selected` vs `p3_frozen`.
* H2: `p4_selected` vs `RR1`.
* H3: `p4_ablation` vs `p3_frozen`.

If two hypotheses compare identical decision vectors on every workload
(for example, if `p4_selected` has the same configuration as
`p4_ablation`), they are merged before the Holm adjustment and the merge is
reported.

**Decision rule.** A comparison counts as a statistically supported
improvement only if its Holm-adjusted p < 0.05 **and** the upper bound of
its 95% CI is below 0.

**Secondary (not in the primary family):** the same three comparisons on
the fresh confirmation set (own Holm family, same rule); the Pratt
Wilcoxon test (for continuity with Phase 3); mean differences with
bootstrap CIs for response time, waiting time, throughput, CPU utilization,
and context switches; frequency of quantum changes; `submission_ref` and
oracle comparisons.

**Exploratory (no multiplicity correction):** breakdowns by true class,
`burst_cv` tercile, and process-count tercile (2,000 resamples).
