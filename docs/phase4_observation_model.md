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

## 7. Protocol record

| Step | Commit |
|---|---|
| Baseline reproduction + this protocol (sections 1–6) | `206d494` |
| Estimator + tests | `0fa9a1f` |
| Controller integration, validation selection, **freeze** (train + validation only) | `90dd9e5` |
| Single evaluation on test + fresh set | `b98b5ed` |

No file in `src/`, `configs/`, or `scripts/phase4_train.py` changed
between the freeze and the evaluation. The fresh confirmation set was
generated for the first time inside `phase4_evaluate.py`.

## 8. Results

Numbers are copied from the generated
[`reports/phase4/summary.md`](../reports/phase4/summary.md) (config hash
`f4021560ce9c`). Figures are in `reports/phase4/figures/`.

### Fit stage (train + validation)

* The best fixed quantum is still RR1 (unchanged from Phase 3).
* 57 `online_censored` configurations (3 decision points × 19 grid
  entries). **Frozen `p4_selected`:** logistic classifier (C = 1)
  deciding after **5** completions. Validation: −0.536% vs RR1 (Phase 3
  frozen: −0.467%). **`p4_ablation`** (Phase 3 tree, depth 3, m = 20, with
  E features): −0.287%.
* The KM restricted CV is among the largest standardized logistic
  coefficients, with the sign theory predicts (higher variability → keep
  RR1). The coefficients are in `reports/phase4/frozen_controller.json`.

### Pre-specified family (average turnaround; negative = better)

| | Hypothesis | Established test (n = 300) | Fresh set (n = 600) |
|---|---|---|---|
| **H1 (primary)** | p4_selected vs p3_frozen | **+0.64%** [+0.02%, +1.28%]; sign-flip p = 0.047; Holm p = 0.14 | +0.08% [−0.35%, +0.51%]; p = 0.73; Holm p = 1 |
| H2 | p4_selected vs RR1 | +0.26% [−0.35%, +0.88%]; p = 0.41; Holm p = 0.81 | −0.44% [−0.90%, +0.03%]; p = 0.067; Holm p = 0.20 |
| H3 | p4_ablation vs p3_frozen | +0.01% [−0.39%, +0.42%]; p = 0.96; Holm p = 0.96 | −0.01% [−0.32%, +0.31%]; p = 0.95; Holm p = 1 |

Values are the geometric-mean change with its 95% paired bootstrap CI, the
sign-flip p-value, and the Holm-adjusted p-value. No hypotheses were merged
(all decision vectors differed).

**Answer: the improved observation estimator did not produce a
statistically supported improvement on either evaluation set.** On the
established test split the point estimate even favours the Phase 3
controller (the CI just excludes 0 on the unfavourable side, but Holm
p = 0.14). On the fresh set there is no difference. The controlled
ablation, which changes only the features, shows no effect on either set.

### Secondary outcomes

| | Established test | | Fresh set | |
|---|---|---|---|---|
| | p3_frozen | p4_selected | p3_frozen | p4_selected |
| Mean turnaround | 189.16 | 189.80 | 180.65 | 180.98 |
| Mean response | 25.05 | 41.46 | 25.45 | 38.41 |
| Mean waiting | 180.40 | 181.04 | 171.91 | 172.24 |
| Context switches | 552.1 | 420.2 | 524.1 | 415.1 |
| Quantum changed | 8% | 39% | 9% | 35% |

* **Response time** worsens by +16.4 ticks vs p3_frozen (95% CI
  [+11.6, +21.5]) on test and +13.0 [+10.0, +16.0] on the fresh set.
  Switching to RR8 delays first dispatch.
* Throughput and CPU utilization are identical across methods (work
  conserving), as in Phase 3.
* **Replication of Phase 3 (pre-specified secondary, not in the corrected
  family):** on the fresh set, `p3_frozen` vs RR1 is −0.51%
  [−0.86%, −0.19%], sign-flip p = 0.0024. On the established test split it
  was −0.38% [−0.75%, −0.04%], p = 0.042. The small Phase 3 benefit
  reappears on 600 new workloads.
* `p4_ablation` vs RR1: −0.36% [−0.85%, +0.11%] (test), −0.52%
  [−0.92%, −0.15%] (fresh). This is indistinguishable from `p3_frozen`
  (H3).
* Oracles (not deployable): the per-workload best fixed quantum is −2.92%
  (test) and −3.34% (fresh) vs RR1, so the headroom is unchanged.

### Exploratory breakdowns (p4_selected vs p3_frozen)

On both sets `p4_selected` does worst on `mixed` workloads (+0.96% test,
+0.94% fresh) and in the highest `burst_cv` tercile (+1.13%, +0.80%). On
the fresh set it does best in the lowest `burst_cv` tercile (−0.72%,
CI includes 0). These are uncorrected and descriptive only.

## 9. Interpretation

1. **Better estimates, not better decisions.** The KM estimator removes
   most of the bias in the identified part of the burst distribution
   (error 0.43–0.50 → 0.08–0.10) and raises the correlation with the
   decision target (e.g. 0.18 → 0.33 at m = 5). That did not translate into
   lower turnaround.
2. **The estimator features make controllers switch much more.** Both
   Phase 4 controllers move to RR8 about 4–5 times as often as the Phase 3
   controller (113 and 98 of 300 test workloads, vs 25). The extra
   switches win on some workloads and lose on others about equally
   (22% better / 21% worse for H1 on test). They cost response time
   consistently.
3. **Selection is dominated by noise at this effect size.** The validation
   advantage that selected `p4_selected` (−0.536% vs −0.467%) is about 0.07
   percentage points. That is an order of magnitude smaller than the
   uncertainty of a 300-workload comparison (the H1 test CI is 1.26
   points wide). With 57 candidate configurations, the winner on
   validation is largely a winner's-curse pick. `p4_selected` also decides
   early (m = 5), when the identified range is smallest (median τ = 9
   ticks).
4. **The remaining signal is moderate.** Even the improved CV estimate
   correlates only about 0.5 with the target. The unidentified tail beyond
   τ, which is where long bursts live, is exactly what decides between
   RR1 and RR8.
5. The Phase 3 controller's small benefit over RR1 replicated on fresh
   data (secondary). Its conservative behaviour (switching rarely) appears
   to be an advantage, not a weakness.

## 10. Limitations

* Synthetic workloads, open-loop CPU-only projection, free context
  switches, and one decision per workload, as in Phase 3.
* Kaplan–Meier assumes censoring is independent of burst length given the
  observation time. Under RR this holds only approximately; the estimator
  slightly underestimates the tail mass (−0.03 to −0.05).
* The established test split had been evaluated once before (Phase 3). The
  fresh confirmation set addresses this, and it agrees on the main
  conclusion.
* Only turnaround was optimized. `p4_selected` trades response time for
  nothing measurable in turnaround.
* Model selection used the same validation set as Phase 3 and a
  single split; repeated or nested validation might pick more
  conservative configurations.

**Possible next steps (not done):** a selection criterion that penalizes
switching or response time; a decision rule that switches only when the
predicted gain exceeds a validation-calibrated margin; periodic
re-decisions as more bursts complete; an I/O-aware closed-loop simulator.

## 11. Reproduce

```bash
python scripts/phase4_baseline.py   # verify the frozen Phase 3 controller reproduces (read-only)
python scripts/phase4_train.py      # train + validation only -> frozen controller (artifacts/phase4/, reports/phase4/)
python scripts/phase4_evaluate.py   # frozen controllers, test split + fresh set, once
python scripts/phase4_plots.py      # figures from saved results
```

Requires the Phase 2 dataset (`scripts/generate_dataset.py`) and the frozen
Phase 3 controllers (`scripts/phase3_train.py`). Runtime is about 2 min
for training and 1.5 min for evaluation. `phase4_evaluate.py` refuses to
run if any configuration or the dataset changed after freezing.
