# Phase 5 — Robustness and Reproducibility Audit

Companion to [`phase5_experimental_protocol.md`](phase5_experimental_protocol.md)
(pre-specified, commit `d1b30be`) and the results in
[`reports/phase5/summary.md`](../reports/phase5/summary.md). Machine-readable
checks are in `reports/phase5/robustness_checks.json`, written by
`scripts/phase5_audit.py`.

## 1. Experiment record

| Item | Value |
|---|---|
| Operating points | small: 6 processes per workload; medium: 14; large: 22 |
| Workloads | 1,000 per operating point (3,000 total), master seeds 5101 / 5102 / 5103 |
| Processes / jobs | 6,000 / 32,326 (small); 14,000 / 77,567 (medium); 22,000 / 121,900 (large) |
| Generator | Phase 2 rules (`configs/phase2.json`), only `process_count` fixed per operating point |
| Controllers | frozen Phase 3 (`ae9db57acecd`) and Phase 4 (`f4021560ce9c`), loaded and hash-verified, never refitted |
| Environment | `reports/phase5/environment.json`: Python 3.11.9, NumPy, pandas, scikit-learn, SciPy, Matplotlib, and joblib versions, platform string, generator config, config hash |
| Raw outputs | `workload_manifest.csv` (one row per workload: seed, index, sizes, trace and job hashes); `inputs/processes_<op>.csv.gz` (every process's arrival, priority, CPU and I/O bursts); `raw_workload_results.csv` (one row per workload × method, 21,000 rows); `comparisons.csv`; `operating_point_summary.csv` |

## 2. Checks (`scripts/phase5_audit.py`; all pass)

| Check | Result |
|---|---|
| Master seeds distinct and not used by any earlier dataset (2026, 2027, 12345) | pass |
| Operating-point assignment: seed, process count, and count of 1,000 per operating point | pass |
| Workload ids unique; input-trace hashes unique within Phase 5 | pass |
| No trace overlaps the 1,500 Phase 2 workloads (train, validation, test) or the 600 Phase 4 fresh workloads | pass (no shared trace hash, no shared id) |
| Regenerating from the seeds reproduces the committed manifest and input traces exactly | pass |
| Every workload has all 7 methods exactly once | pass |
| Every method scheduled the identical job list (hash of each schedule's own input = manifest hash) | pass |
| Throughput and CPU utilization identical across methods per workload (work conservation) | pass |
| Oracle equals the minimum of the four fixed quanta | pass |
| Independent recomputation of the primary family from the raw CSV: point estimates (geometric and aggregate) | exact (difference 0) |
| Independent p-values (SciPy `permutation_test`, 20,000 resamples) vs committed sign-flip p | within Monte Carlo error (max \|Δ\| = 0.0039) |
| Independent bootstrap CIs (SciPy, 20,000 resamples) vs committed | within Monte Carlo error (max \|Δ\| = 0.008 percentage points) |
| Holm adjustment recomputed | exact |

Automated tests (`tests/test_phase5.py`, 14 tests) cover:

* operating-point assignment and seed uniqueness/reuse rejection;
* that the generator changes only the process count;
* deterministic manifests and hash sensitivity;
* input pairing across methods;
* the classification rule;
* output schemas and configuration recording;
* byte-identical reruns, and refusal on a mismatched frozen-controller hash;
* the committed evidence itself (counts, pairing, CSV/JSON agreement, audit verdict).

## 3. Defects found and corrected during Phase 5

1. **Summary column labels (rendering only).** The first run's
   `summary.md` printed the secondary-outcome columns in `METRICS` order
   (turnaround, waiting, response) under a header reading turnaround,
   response, waiting. The label order is now explicit. The experiment was
   rerun to regenerate the summary. Every data file (manifest, inputs, raw
   results, `results.json`, comparison and summary CSVs, environment) was
   **byte-identical** to the first run (SHA-256 verified); only
   `summary.md` changed. No number changed. Recorded in commit `db55ae5`.
2. **Audit script skipped a check.** In its first version,
   `phase5_audit.py` compared a NumPy p-value difference, producing a NumPy
   boolean that its `isinstance(..., bool)` filter skipped, so the p-value
   check was neither printed nor counted in `all_pass`. All checks are now
   coerced to Python booleans. The check passes (max \|Δp\| = 0.0039 against
   a tolerance of 0.01).

No controller, configuration, seed, or analysis rule was changed after the
results were seen.

## 4. Earlier evidence preserved

Phase 2, 3, 3.1, and 4 reports, configurations, and frozen controller files
are untouched. Phase 5 writes only to `reports/phase5/`, `docs/phase5_*`,
`configs/phase5.json`, `src/schedx/phase5.py`, `scripts/phase5_*`, and
`tests/test_phase5.py`. Frozen controllers are only read, and their hashes
are checked before use.

## 5. Repository hygiene

* About 9.4 MB tracked in total. The largest file is the raw results CSV
  (≈4 MB). There are no model binaries: `artifacts/`, which holds the
  dataset and the joblib controllers, is git-ignored and rebuilt by the
  documented commands.
* No secret-like patterns are found by a repository scan.
* The only compressed files are the three input-trace CSVs (`.csv.gz`,
  written with a fixed gzip timestamp so they are byte-reproducible).

## 6. Reproduction

From a fresh clone (the frozen controllers are rebuilt deterministically
and hash-checked):

```bash
python scripts/generate_dataset.py
python scripts/phase3_train.py
python scripts/phase4_train.py
python scripts/phase5_run.py
python scripts/phase5_plots.py
python scripts/phase5_audit.py
```

The result of an actual clean-clone reproduction is recorded in §8.

## 7. Professor requirements: status

| Requirement | Status | Evidence / gap |
|---|---|---|
| At least three independent workload sizes / operating points for the principal experiment | **Satisfied** | 6, 14, and 22 processes; separate seeds; 1,000 independently generated workloads each; disjointness and uniqueness verified (§2). |
| Identical configurations when comparing algorithms | **Satisfied** | Every method receives the identical job list (hash-verified per row), the same simulator settings, and the same frozen controllers. |
| Reproducible code and raw CSVs | **Satisfied on this machine** | Scripts, configs, seed manifest, raw input traces, and raw per-workload results are in the repository; reruns are byte-identical. A clean-clone reproduction is recorded in §8. **Not verified:** other operating systems or Python and library versions. |
| Final case-study report, at most eight pages | **Not done** | Only an outline exists (§9); the report has not been written. |
| Real-hardware / kernel validation | **Not applicable / not claimed** | All evidence is simulator-based (CPU-only burst-trace projection, free context switches). |

## 8. Clean-clone reproduction

*Recorded after the Phase 5 commits were pushed; see below.*

## 9. Outline for the eight-page case study (not yet written)

1. **Introduction and research question** (½ page): fixed vs ML-assisted
   Round Robin quantum selection; contributions; simulator-only scope.
2. **System and simulator** (1 page): process model, five classical
   algorithms, metric definitions, burst-trace projection, integer ticks,
   free context switches.
3. **Workloads and data** (¾ page): generator, labels, feature
   availability (submission, online, retrospective), and leakage controls.
4. **Methods** (1½ pages): Phase 2 classifiers (brief); Phase 3 online
   controller and decision boundary; Phase 4 Kaplan–Meier observation
   model; frozen-controller protocol.
5. **Experimental design** (1 page): three operating points, seeds,
   pairing, estimands (geometric vs aggregate mean), sign-flip tests,
   bootstrap CIs, Holm families, 1% SESOI.
6. **Results** (2 pages): primary family table and figure; response-time
   cost; controller behaviour; oracle headroom; replication across Phase 3,
   Phase 4, and Phase 5 evaluation sets.
7. **Threats to validity** (¾ page): synthetic data, open-loop CPU-only
   projection, zero switch cost, single generator family, SESOI choice, test
   reuse before Phase 4.
8. **Conclusion** (½ page): small but consistent turnaround benefit of the
   Phase 3 controller, below the practical threshold and bought with
   response time; no benefit from the improved estimator.

Appendix (outside the eight pages, if permitted): reproduction commands and
the requirements table.
