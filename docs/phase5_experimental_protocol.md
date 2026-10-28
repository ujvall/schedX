# Phase 5 — Experimental Protocol: Three Operating Points

**Status:** pre-specified. This document and `configs/phase5.json` were
committed **before** any Phase 5 workload was generated for evaluation or
scheduled. Results are reported in `reports/phase5/summary.md`; the audit of
the experiment is in `docs/phase5_robustness.md`.

Phase 5 validates the existing findings. It trains, tunes, and changes
nothing: both controllers are the frozen Phase 3 and Phase 4 artifacts.
Everything is an offline simulation (CPU-only burst-trace projection,
integer ticks, free context switches). No kernel-level or hardware
scheduling is performed or claimed.

## 1. Stage 1 audit of existing evidence

| Question | Finding |
|---|---|
| Frozen controllers intact? | Yes. Recomputed config hashes equal the committed ones: Phase 3 `ae9db57acecd` and Phase 4 `f4021560ce9c` (artifacts, frozen JSON, and recomputation agree). Default quantum RR1 in both. |
| Do existing results cover three operating points? | **No.** Every existing evaluation set (Phase 2/3 test split, 300 workloads; Phase 4 fresh set, 600) draws the process count of each workload uniformly from 4–24 within one set. Size bands are subsets of a single sample, have unequal and small sizes (57–152 workloads per band), and were analysed only as uncorrected exploratory breakdowns. They are not independent operating points. |
| Seeds in use | Dataset 2026 (train/validation/test), Phase 4 fresh set 2027, `predict.py` default 12345; tests use only small throwaway seeds (3, 5, 11, 12, 21, 33, 44, 99, 999, 4242). |
| Leakage risks | (a) Evaluating on workloads seen in training, validation, or earlier evaluations. Mitigation: new master seeds, plus trace-hash disjointness checks against the full Phase 2 dataset and the Phase 4 fresh set. (b) Re-tuning after seeing results. Mitigation: nothing is fitted in Phase 5, and the protocol is committed first. |
| Metric definitions | Unchanged from Phase 1 (`schedx.metrics`): turnaround, waiting, response, throughput, CPU utilization over [first arrival, last completion], context switches. |
| Can the simulator support the experiment? | Yes, without code changes. The generator's `process_count` range can be fixed to a single value; every other rule stays as in Phase 2. **No correction was needed.** |

## 2. Research questions and hypotheses

**RQ.** Do the frozen ML-assisted adaptive controllers reduce mean job
turnaround compared with the best fixed Round Robin quantum (RR1), at small,
medium, and large process populations?

* **H1 (per operating point):** the frozen Phase 3 controller (`p3_frozen`)
  has lower turnaround than RR1.
* **H2 (per operating point):** the frozen Phase 4 controller
  (`p4_selected`) has lower turnaround than RR1.

Each hypothesis is tested against the null of no difference, two-sided. Phase
3 and Phase 4 findings lead us to *expect* at most a small effect for H1
and none for H2. These expectations do not change the analysis.

**Secondary questions (separate family):** S1 `p4_selected` vs `p3_frozen`;
S2 `p3_frozen` vs RR2; S3 `p4_selected` vs RR2, each per operating point.

## 3. Operating points

| Operating point | Processes per workload | Master seed | Workloads |
|---|---|---|---|
| small | 6 | 5101 | 1,000 |
| medium | 14 | 5102 | 1,000 |
| large | 22 | 5103 | 1,000 |

* **What changes:** only the number of processes per workload, and with it
  the number of jobs (bursts), the total work, and the amount of
  contention.
* **What is held constant:** every other generator rule of Phase 2
  (`configs/phase2.json`): regime mix, I/O-heavy probabilities,
  per-process burst-mean ranges, burst-count ranges, burst variability,
  arrival-intensity range, priorities, and integer rounding. Jobs use the
  same burst-trace projection as Phases 3–4.
* **Why 6 / 14 / 22:** all lie inside the 4–24 process range the frozen
  controllers were trained on, so no out-of-distribution extrapolation is
  involved. They are evenly spaced and span the range with a margin from
  the edges. They correspond to about 33, 76, and 115 jobs per workload
  (measured on throwaway seeds). Larger populations would be outside the
  training support and would test extrapolation rather than the trained
  controllers.
* **Independence:** each workload is generated from `(master_seed, index)`
  with its own RNG stream. Each operating point has its own master seed.
  The audit checks that no trace occurs twice within Phase 5 or overlaps the
  Phase 2 dataset or the Phase 4 fresh set.

**Sample size.** Planning used only already-published Phase 4 results: the
per-workload log-ratio SD is 0.04–0.07, and the standardized effect of
`p3_frozen` vs RR1 is about −0.06 to −0.24 at 12–24 processes and about 0 at
4–8. With 1,000 workloads per operating point, a two-sided test at the most
stringent Holm level (0.05/6) has 80% power for |dz| ≥ 0.11. Runtime is
about 2 minutes on a laptop.

## 4. Methods compared (identical inputs)

For each workload, `burst_jobs(processes)` is computed once and every method
schedules exactly that job list. A hash of the jobs is recorded on every
output row.

| Method | Description | Deployable |
|---|---|---|
| RR1 | fixed quantum 1: best fixed quantum on Phase 3 train + validation; primary baseline | yes |
| RR2 | fixed quantum 2: Phase 2 reference baseline | yes |
| RR4, RR8 | fixed quanta, reported for context and needed for the oracle | yes |
| `p3_frozen` | frozen Phase 3 controller: decides once after the 20th job completion, from submission + online observation features | yes |
| `p4_selected` | frozen Phase 4 controller: decides once after the 5th completion; adds Kaplan–Meier estimates | yes |
| `oracle_fixed` | per-workload best of RR1/2/4/8 (knows every outcome) | **no** |

**Applicability.** Both controllers are applicable at all three operating
points, because all sizes are within their training range. A controller
needs a decision point (its m-th job completion, with jobs left). Small
workloads can end before that. In that case the controller keeps RR1 by its
built-in deterministic fallback. This is part of the controller and is
reported as fallback frequency, not removed.

## 5. Outcomes

* **Primary:** per-workload mean job turnaround `T` (ticks), compared in
  pairs on identical workloads.
* **Secondary:** mean response time, mean waiting time, throughput (jobs per
  tick), CPU utilization, context switches, share of workloads where the
  controller changed the quantum, and fallback frequency.

## 6. Statistics

* **Primary estimand (per operating point):** mean over workloads of
  `log(T_method / T_comparator)`, reported as the **geometric-mean
  per-workload change** `exp(mean) − 1`.
* **Also reported:** the **change in aggregate mean turnaround**,
  `mean(T_method) / mean(T_comparator) − 1`. This weights large workloads
  more and can differ from the geometric-mean change.
* **Test:** two-sided paired sign-flip permutation test of the mean log
  ratio. Exact enumeration when ≤ 26 pairs differ, otherwise 100,000 seeded
  Monte Carlo sign flips.
* **CIs:** paired percentile bootstrap over workloads, 10,000 resamples,
  seeded. Workloads are the resampling unit; they are independent by
  construction.
* **Multiplicity:**
  * Holm over the **primary family** of 6 hypotheses (H1 and H2 × 3
    operating points).
  * Separately, Holm over the **secondary family** of 9 (S1–S3 × 3).
  * Hypotheses that compare identical decision vectors are merged first.
* **Decision rule (each primary hypothesis):**
  * *statistically supported improvement*: Holm p < 0.05 **and** the upper
    bound of the 95% CI is below 0;
  * *practically meaningful*: supported **and** CI upper bound ≤ −1%
    (smallest effect size of interest, SESOI = 1% of turnaround);
  * *practically negligible*: the whole 95% CI lies within (−1%, +1%);
  * otherwise *inconclusive*.
* **Why a 1% SESOI.** The per-workload oracle over fixed quanta improves on
  RR1 by about 3% in earlier sets. A third of that maximum attainable gain
  is the smallest change we would call practically relevant for a quantum
  controller. The SESOI is a pre-declared judgement; full CIs are reported
  so readers can apply another threshold.
* **Pooled summary (descriptive, outside both families):** the
  equal-weight average of the three per-operating-point means, with a
  bootstrap stratified by operating point.
* **Exploratory:** results by true workload class within each operating
  point, uncorrected.

## 7. Procedure

1. `scripts/phase5_run.py` verifies the frozen-controller hashes. It then
   generates the three operating points from their seeds, writes the seed
   manifest and the input traces, evaluates every method once, and writes
   the raw per-workload CSV, `results.json`, the operating-point summary
   CSV, the environment record, and `summary.md`.
2. `scripts/phase5_plots.py` draws all figures from the saved CSV/JSON.
3. `scripts/phase5_audit.py` checks seed and identity uniqueness,
   disjointness from earlier datasets, input pairing, regeneration
   determinism, and an independent recomputation of the statistics.
4. Nothing is re-run to change a result. If a defect forces a rerun, it is
   documented in `docs/phase5_robustness.md`.
