# Phase 3 — Feature-Aware Adaptive Quantum Selection

Phase 3 asks whether workload features can choose a Round Robin quantum
better than a fixed quantum. It does **not** assume they can. All results
are offline simulations on synthetic workloads in a CPU-only simulator; no
OS-level performance is measured.

## 1. Audit of Phase 2 (starting point)

* **Workloads → rows.** `workloads.csv` has one row per workload,
  `processes.csv` one row per process (full CPU/I/O traces), and
  `features.csv` one row per (workload, horizon k ∈ {1, 2, 3}), so three
  correlated rows per workload.
* **Generation.** Per workload: a regime, a process count (4–24), exponential
  arrivals with a workload-specific mean gap, a per-process CPU-heavy or
  I/O-heavy draw, and Gamma-distributed integer bursts (`docs/phase2_ml.md` §1).
  Labels come from the realized traces.
* **Feature availability.** A (submission: count, arrival times,
  priorities) is deployable under a batch-submission assumption. B
  (first k bursts of *every* process) is optimistic: it includes bursts of
  processes that have not arrived yet and ignores when an observation becomes
  available. C (lifetime totals) is retrospective.
* **Simulated jobs.** `policy_eval.burst_jobs` turns every CPU burst into an
  independent job released at its contention-free time (arrival + earlier
  CPU bursts + earlier I/O waits). This open-loop projection is CPU-only;
  real I/O completions would be delayed by contention. Phase 3 keeps it so
  results stay comparable with Phase 2.
* **RR2 baseline.** Chosen in Phase 2 as the lowest mean regret among
  FCFS/RR2/RR4/RR8/RR16 over train + validation workloads.
* **Phase 2 class map.** `cpu_bound → FCFS`, `mixed → RR2`,
  `io_bound → RR4`, learned on train + validation true classes.

No Phase 2 schema, dataset, or result is changed. The Phase 1 engine gained
one backward-compatible feature: Round Robin accepts a time-varying quantum
(`adaptive_round_robin`). With a constant quantum it reproduces
`round_robin` exactly (tested on random workloads).

## 2. Hypothesis

H1: a controller that chooses the RR quantum per workload from deployable
features yields lower average job turnaround than the best fixed quantum
chosen on the same training and validation data.

H0: no difference. Phase 3 reports whichever the data supports.

## 3. Objective

**Primary: average job turnaround** (mean over jobs of completion − release),
compared per workload as a ratio to the baseline.

Why turnaround and not response time: in Round Robin, a smaller quantum
almost always gives an earlier first dispatch, so response time is
essentially minimized by the smallest quantum and leaves little to decide.
Turnaround has a real trade-off: small quanta help short jobs stuck behind
long ones, large quanta avoid stretching out jobs of similar length. On
train + validation workloads the per-workload winner is split mainly between
RR1 and RR8. Turnaround is also the Phase 2 objective.

**Secondary outcomes** (reported, not optimized): average waiting time,
average response time, throughput, CPU utilization, context switches. No
composite score is used. Every candidate is work-conserving on the same jobs,
so throughput and CPU utilization are identical across candidates by
construction; they are still reported.

**Switching cost.** The simulator treats context switches as free. No
overhead value is invented. Context-switch counts are reported per method so
that their direction is visible. Smaller quanta cause more switches, so a
real per-switch cost would penalize small quanta.

## 4. Candidates and baselines

* Controller candidates: RR with quantum ∈ {1, 2, 4, 8} (integer ticks,
  configurable in `configs/phase3.json`).
* **Fixed RR2**: the Phase 2 reference baseline.
* **Best fixed quantum** (`q0`): the candidate with the lowest mean log
  regret, `mean(log(T_q / min_q' T_q'))`, over train + validation workloads.
  Ties go to the smaller quantum. This is the **primary baseline** and also the
  controller's default/fallback quantum.
* **Phase 2 class map (predicted)**: the frozen Phase 2 map applied to the
  class predicted by the frozen Phase 2 Logistic Regression model at
  horizon k = 1. It is labelled as relying on Phase 2's optimistic B features.
* **Phase 2 class map (true)**: same map with the true class. **Diagnostic
  only, not deployable.**
* **Oracles** (not deployable): the best fixed candidate per workload, and the
  best switch target per workload for the online controller. They know every
  candidate's outcome.

## 5. Feature availability and observation model

The controller accepts only two kinds of input, enforced in code
(`controller.ALLOWED_FEATURES`; anything else raises):

* **A — submission:** the Phase 2 pre-execution features (process count,
  arrival span/gaps/rate, priority mean/std). They assume batch submission:
  the job list is known up front, the bursts are not.
* **O — online observation** (`schedx/observation.py`): computed from the
  schedule that *actually ran* up to the decision tick `t`: jobs released by
  `t`, CPU received so far, burst lengths of jobs that have **completed**,
  busy time, context switches, and the fraction of submitted processes that
  have arrived. Unfinished and unreleased bursts are never exposed. Under
  Round Robin (non-clairvoyant), the schedule before `t` does not depend on
  them (tested by making unfinished bursts longer).

Phase 2 B features, C features, labels, and generator parameters are never
controller inputs.

**Two decision regimes:**

* **Submission controller:** decides the quantum once at t = 0 from A
  features.
* **Online controller:** runs `q0` and, when the m-th job completes, builds
  the observation, decides once, and runs the chosen quantum for every later
  dispatch. A slice already running keeps its quantum. Jobs released later
  are scheduled normally; they simply were not part of the observation. If a
  workload has ≤ m jobs, no decision is made and `q0` is used throughout
  (logged as fallback). `m ∈ {5, 10, 20}` is chosen on validation.

## 6. Training targets (counterfactual evaluation)

For each training workload the simulator runs every candidate. This is
training-time counterfactual evaluation and is kept separate from inference:

* Submission regime: outcome of each fixed RR(q) run.
* Online regime: outcome of "q0 until t, then q" for each candidate q.

Targets:

* **Best-candidate label** for classifiers: the q with the lowest average
  turnaround. Exact ties are broken deterministically: prefer `q0`, then the
  smaller quantum.
* **Log-regret vector** for the regressor: `log(T_q / min_q' T_q')` for every
  candidate.

Online-regime workloads with no decision point are excluded from training,
since every candidate gives the same outcome.

## 7. Decision models

Kept simple and interpretable; no reinforcement learning or neural networks.

| family | model | prediction → decision |
|---|---|---|
| `constant` | always `q0` | — |
| `tree_classifier` | `DecisionTreeClassifier` on the best-candidate label | predicted quantum |
| `logistic_classifier` | `LogisticRegression` (standardized) on the label | predicted quantum |
| `regret_regressor` | multi-output `DecisionTreeRegressor` on the log-regret vector | argmin of predicted regret (tie order as above) |

All pipelines impute missing values (median plus missing indicators) fitted
on training rows only. **Fallback:** if a model returns a quantum outside
the candidate set or a non-finite regret vector, the controller uses `q0`
and records the reason. Missing feature columns are a schema error and raise.

## 8. Leakage prevention and protocol

1. Reuse the Phase 2 grouped split (`grouped_split`, seed 7):
   900 / 300 / 300 workloads. Split by workload id, stratified by class.
2. `scripts/phase3_train.py` loads **only** train + validation traces.
   Test traces are never passed to it (asserted).
3. Choose `q0` on train + validation.
4. Fit each (regime, m, family, hyperparameters) configuration on **train**.
   Score it on **validation** by `mean(log(T_chosen / T_q0))` (lower is
   better). Pick the configuration with the lowest score; ties go to the
   simpler configuration (constant < submission < online with smaller m).
   The best configuration within each regime is also frozen.
5. Freeze: save the controllers (`artifacts/phase3/controllers.joblib`) and
   `reports/phase3/frozen_controller.json` (configuration, dataset version,
   config hash, validation table). No refit after selection.
6. `scripts/phase3_evaluate.py` loads the frozen file, refuses to run if the
   config or dataset has changed, and evaluates every method once on the
   300 test workloads under identical conditions.

## 9. Statistical methodology

* **Unit:** the workload. All methods run on the same 300 test workloads,
  so comparisons are paired.
* **Per-workload effect:** `log(T_method / T_baseline)`, which is
  scale-free. Workloads range from tens to thousands of ticks, and raw
  differences would be dominated by the largest workloads. Raw differences in
  ticks are reported too.
* **Reported per comparison:** mean and median of both methods; mean and
  median absolute difference; geometric-mean relative change
  `exp(mean log ratio) − 1`; mean relative change; share of workloads
  better / equal / worse.
* **95% CIs:** paired percentile bootstrap that resamples workloads with
  replacement (10,000 resamples, fixed seed), for the mean log ratio
  (reported as relative change) and the mean absolute difference. This
  assumes workloads are independent draws from the generator, which holds by
  construction (independent seeds).
* **Test:** two-sided Wilcoxon signed-rank test on log ratios (`pratt`
  handling of zero differences, which are common when two methods pick the
  same quantum). It makes no normality assumption.
* **Effect sizes:** matched-pairs rank-biserial correlation, and Cohen's dz
  on log ratios.
* **Multiplicity:** Holm–Bonferroni across the pre-specified deployable
  comparisons. **Primary comparison (pre-specified): selected adaptive
  controller vs best fixed quantum.**
* **Breakdowns:** by true workload class (reporting only), process-count
  tercile, and burst-variability (`burst_cv`) tercile.

> **Audit notes (Phase 3.1, added after the test evaluation; see
> [`reports/phase3/audit/AUDIT.md`](../reports/phase3/audit/AUDIT.md)).**
> The pre-specification above is left as written. The audit found these
> clarifications necessary:
>
> * **Decision rule.** The rule used to call a result "statistically
>   supported", Holm-adjusted p < 0.05 **and** the bootstrap CI's upper bound
>   below 0, is implemented in `scripts/phase3_evaluate.py`. That script was
>   written before the single test run but committed after it (`4ada3b7`).
>   The rule is not stated in this section as frozen in `a13ffba`.
> * **Holm family.** In code, the family is the 9 comparisons of the deployable
>   methods against RR1 and RR2 (frozen in `a13ffba`). Because the selected
>   controller and the best online controller turned out to be the same
>   frozen controller, two of those 9 hypotheses are exact duplicates. That
>   makes the correction slightly conservative; the audit reports the
>   7-hypothesis values as a sensitivity check. No conclusion changes.
> * **Rank-biserial.** It is computed after dropping zero differences
>   (ranks 1..n over the changed workloads). That is a different ranking
>   from the Pratt test it is reported next to.
> * **Resample counts.** Main comparisons use 10,000 bootstrap resamples;
>   the subgroup breakdowns use 2,000.
> * **Estimands.** The bootstrap CI concerns the *mean* log ratio; the Pratt
>   Wilcoxon test is a rank test which, with many tied (zero) pairs, behaves
>   almost like a sign test. They can disagree without either being wrong.

The test set is evaluated once with the frozen strategy. If the result is
inconclusive it is reported as inconclusive; there is no re-tuning against
test.

## 10. Results

Numbers below are copied from the generated
[`reports/phase3/summary.md`](../reports/phase3/summary.md) (config hash
`ae9db57acecd`, dataset `67da9d294be3`). Figures are in
`reports/phase3/figures/`.

### Fit stage (train + validation only; frozen before test)

* **Best fixed quantum: RR1** (mean log regret 0.0312, vs RR2 0.0335,
  RR4 0.0394, RR8 0.0638). The Phase 2 baseline RR2 is therefore not the
  strongest fixed choice once RR1 is a candidate.
* 61 configurations were fitted on train and scored on validation. **Frozen
  controller:** online `tree_classifier` (depth 3, min leaf 20) deciding
  after the 20th job completion. Validation: −0.47% geometric-mean turnaround
  vs RR1, with the quantum changed on 7% of validation workloads. Best
  submission-time controller: `regret_regressor` (depth 2). At m = 20, about
  4% of workloads have no decision point (fallback to RR1).
* The frozen rule (`reports/phase3/frozen_controller.json`) switches RR1 →
  RR8 only when the median completed burst exceeds 1.75 ticks, the
  completed-burst CV is ≤ 0.707, and pending jobs have received ≤ 1.275
  ticks on average. In words: long, regular bursts favour a larger quantum,
  which matches queueing theory (FCFS-like service suits low variability,
  processor-sharing-like service suits high variability).
* The frozen commit (`a13ffba`) predates the test evaluation commit
  (`4ada3b7`) in the Git history.

### Test stage (300 held-out workloads, evaluated once)

"Geo change" is the **geometric mean of per-workload ratios** of average
turnaround, `exp(mean log(T_method / T_comparator)) − 1`. It is not the change
in mean turnaround (for the primary comparison that is −0.17%: 189.16 vs
189.49 ticks). Holm p uses the committed 9-hypothesis family.

| comparison (avg turnaround) | geo change | 95% CI | better / equal / worse | Wilcoxon p | Holm p |
|---|---|---|---|---|---|
| **adaptive (selected) vs best fixed RR1 — primary** | **−0.38%** | [−0.75%, −0.04%] | 5% / 92% / 4% | 0.50 | 1.00 |
| adaptive (submission) vs RR1 | −0.10% | [−0.36%, +0.16%] | 15% / 75% / 9% | 0.049 | 0.29 |
| Phase 2 class map (predicted) vs RR1 | −0.95% | [−2.33%, +0.44%] | 44% / 1% / 55% | 0.87 | 1.00 |
| fixed RR2 vs RR1 | +0.17% | [−0.09%, +0.44%] | 44% / 1% / 55% | 0.076 | 0.38 |
| adaptive (selected) vs fixed RR2 | −0.54% | [−0.92%, −0.19%] | 58% / 1% / 40% | 0.0007 | 0.006 |
| *true-class map vs RR1 (not deployable)* | −0.48% | [−1.94%, +1.04%] | 43% / 0% / 57% | 0.53 | — |
| *oracle, fixed per workload (not deployable)* | −2.92% | [−3.49%, −2.37%] | 54% / 46% / 0% | <1e-30 | — |
| *oracle, best switch at decision point (not deployable)* | −1.78% | [−2.25%, −1.34%] | 52% / 48% / 0% | <1e-30 | — |

Mean absolute difference for the primary comparison: −0.33 ticks, 95% CI
[−0.86, +0.18]. Effect sizes: matched-pairs rank-biserial −0.43 when zero
differences are dropped (ranks 1–25 over the changed workloads; the value in
`summary.md`) and −0.13 with the Pratt ranks that the pre-specified test
uses; Cohen's dz −0.12. Share of the RR1 → oracle gap closed: 12.7%
(selected), 3.4% (submission), 60.7% (online oracle).

Holm sensitivity (audit, post hoc): without the two duplicate hypotheses
(7 distinct), the adjusted p-values are 1.00 (primary), 0.24 (submission vs
RR1), and 0.0048 (selected vs RR2). No conclusion changes.

**Answer to H1: no statistically supported improvement over the best fixed
quantum.** The pre-specified Wilcoxon test does not reject (p = 0.50,
Holm p = 1). The paired bootstrap CI of the geometric-mean change only just
excludes zero. The controller changed the decision on 25 of 300 workloads:
14 improved and 11 got worse, and the gains were larger than the losses. A
*post-hoc* check, labelled as such in the report, gives Wilcoxon p = 0.059 on
those 25 workloads and sign-test p = 0.69. The data are consistent with a
small benefit (well under 1%) concentrated in a few workloads, but they do
not establish it.

The Phase 3.1 audit (post hoc) adds tests of the *mean* log ratio, the
quantity the CI describes: an exact paired sign-flip permutation test gives
p = 0.042 and a t-test p = 0.044. These are unadjusted (Bonferroni over the
7 distinct hypotheses would give 0.29), so they do not change the
pre-specified conclusion. They do show that the result is weak evidence of a
small mean benefit rather than evidence of no effect. The submission
controller's raw Pratt p = 0.049 reflects more workloads improving than
worsening (46 vs 28). Its mean effect is not distinguishable from zero
(CI [−0.36%, +0.16%], permutation p = 0.45).

Against the weaker Phase 2 baseline RR2, the adaptive controller is
significantly better (−0.54%, Holm p = 0.006). Most of that comes from its
RR1 default: RR1 vs RR2 is itself −0.17% (not significant). This is why the
primary comparison is against the best fixed quantum.

### Secondary outcomes (test means)

| method | turnaround | waiting | response | context switches |
|---|---|---|---|---|
| best fixed RR1 | 189.49 | 180.73 | 20.33 | 575.8 |
| adaptive (selected) | 189.16 | 180.40 | 25.05 | 552.1 |
| fixed RR2 | 189.79 | 181.03 | 38.02 | 309.9 |
| Phase 2 class map (predicted) | 184.51 | 175.76 | 124.73 | 173.8 |

Throughput (0.134 jobs/tick) and CPU utilization (91.3%) are identical for
all methods, as expected for work-conserving policies on the same jobs.
Switching to RR8 trades response time (+23% vs RR1) for a small turnaround
gain and fewer context switches. The Phase 2 class map has the lowest mean
turnaround in ticks (its FCFS choice suits large CPU-bound workloads), but
its per-workload effect is not significant and its response time is about 6×
worse. It optimizes a different trade-off than the RR-only controller.

### Where the controller helps (exploratory, no multiplicity correction; CIs from 2,000 bootstrap resamples)

* **By class:** `io_bound` −1.21% [−2.29%, −0.25%]; `cpu_bound` 0.00%
  (never switched); `mixed` +0.15% [0.00%, +0.43%].
* **By burst variability:** lowest `burst_cv` tercile −1.07%
  [−2.07%, −0.26%]; middle −0.15%; highest +0.09%. The oracle's headroom
  shrinks the same way (−5.9% → −0.3%).
* **By process count:** no clear pattern (−0.2% to −0.6%, CIs touching 0).

## 11. Interpretation

1. **There is real headroom.** Choosing the best quantum per workload would
   cut turnaround by about 3% vs RR1 on test; switching only at the online
   decision point (m = 20) could still gain 1.8%. The per-workload winner is
   split mainly between RR1 and RR8.
2. **The signal exists but is poorly observable.** On training workloads
   the true job-burst CV correlates strongly with whether RR8 beats RR1
   (Spearman 0.79). The CV observed online at the decision point correlates
   only 0.46 with that outcome and 0.42 with the true CV. Under the RR1
   default, short bursts finish first, so the completed bursts available at
   decision time are a biased sample. The controller is therefore cautious
   and switches rarely.
3. **Submission-time features are close to useless** for this decision
   (3.4% of the gap closed): arrival patterns and priorities say little about
   burst lengths.
4. **Coarse classes are not the right target.** Even the true-class map does
   not reliably beat RR1, consistent with Phase 2.

## 12. Limitations and future work

* **Synthetic, open-loop, CPU-only.** Bursts become independent jobs
  released at contention-free times; I/O blocking is not simulated and
  contention does not delay later bursts. Results may not transfer to real
  systems or closed-loop I/O.
* **Free context switches.** RR1 performs about 1.9× as many switches as
  RR2 on test workloads. Any real per-switch cost would penalize RR1 and
  could change both the best fixed quantum and the controller's value. No
  overhead value is assumed; a switching-cost sensitivity analysis needs a
  simulator that charges overhead.
* **One decision per workload.** The controller decides once. Periodic
  re-decision, or a decision triggered by elapsed time rather than a
  completion count, is untested.
* **Biased online observation.** Censoring-aware estimates of the burst
  distribution (using the CPU time received by unfinished jobs), or a
  short exploratory phase at a larger quantum, could expose the variability
  signal better. This was identified on training data only *after* the
  configuration was frozen, so it was not applied in this phase.
* **Batch-submission assumption** for A features (arrival times known up
  front).
* **Single dataset seed and split**; 300 test workloads. Subgroup results
  are exploratory.
* The objective is average turnaround only. Response time worsens when the
  controller switches to RR8, so a deployment would need an explicit
  multi-objective choice.

## 13. Reproduce

```bash
python scripts/generate_dataset.py          # Phase 2 dataset (if not present)
python scripts/train_evaluate.py            # Phase 2 models (Phase 2 class-map reference)
python scripts/phase3_train.py              # fit + select on train/validation, freeze
python scripts/phase3_evaluate.py           # evaluate frozen controllers on test, once
python scripts/phase3_plots.py              # figures from saved results
```

Runtime ≈ 1.5 min for training and ≈ 0.5 min for evaluation on a laptop.
`phase3_evaluate.py` refuses to run if `configs/phase3.json` or the dataset
changed after freezing.
