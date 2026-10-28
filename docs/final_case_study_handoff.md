# Final Case-Study Handoff

**Machine-Learning-Assisted Adaptive CPU Scheduling**
*Predicting workload behavior. Optimizing scheduling decisions.*

This document hands over the frozen research evidence for the eight-page
case study. It is **not** the case study. Every number below is copied from
a committed result file listed in §8. The evidence audit is in
[`phase6_final_audit.md`](phase6_final_audit.md).

* Repository: https://github.com/ujvall/schedX (public)
* Evidence freeze: commit `5c78452` (HEAD when Phase 6 started). The last
  commit that changed code or results is `be5fb9d`; `5c78452` and Phase 6
  add documentation only.

## 1. Research question and objectives

**RQ.** Can process-behaviour prediction improve Round Robin scheduling
decisions (policy or time-quantum choice) compared with fixed, rule-based
scheduling?

Objectives, as carried out:

1. Build a reproducible CPU scheduling simulator with classical algorithms
   and shared metrics (Phase 1).
2. Generate labelled synthetic workloads, and test whether workload class
   can be predicted from information a scheduler actually has (Phase 2).
3. Use workload features to choose the RR quantum, compared fairly with
   fixed quanta (Phase 3), and improve the online observation model
   (Phase 4).
4. Validate the frozen controllers at three independent operating points
   (Phase 5).

## 2. Simulator and algorithms

* **Simulator:** discrete-event, single CPU, integer ticks. Context switches
  are counted but cost zero time. Its tie-breaking rules are documented.
* **Algorithms:** FCFS, non-preemptive SJF, SRTF, Round Robin (fixed
  quantum, plus a time-varying quantum for adaptive control), and Priority
  (non-preemptive and preemptive).
* **Metrics** (per workload):
  * turnaround = completion − release;
  * waiting = turnaround − CPU time;
  * response = first dispatch − release;
  * throughput and CPU utilization over [first arrival, last completion];
  * context switches.
* **Jobs (Phases 3–5):** each CPU burst of a generated process becomes a job
  released at its contention-free time (the "burst-trace projection"). I/O
  is recorded in the traces but **not simulated**.

## 3. Workloads, features, models, controllers

* **Generator:** seeded per workload. Each process alternates CPU and I/O
  bursts drawn from Gamma distributions, from either CPU-heavy or I/O-heavy
  ranges.
* **Labels:** computed from the realized traces, not from generator intent.
  A process is I/O-bound if its I/O share is > 0.5. A workload is cpu-bound if
  ≤ 25% of its processes are I/O-bound, io-bound if ≥ 75%, otherwise mixed.
* **Dataset:** 1,500 workloads, 21,453 processes, master seed 2026.
* **Split:** grouped by workload and stratified by class: 900 train /
  300 validation / 300 test.
* **Feature availability:**
  * A: known at submission;
  * B: first k bursts (Phase 2; optimistic);
  * C: retrospective, never deployable;
  * O: observed online from the schedule actually run (Phase 3);
  * E: Kaplan–Meier estimates of unfinished bursts (Phase 4).
* **Phase 2 classifiers:** Logistic Regression, Decision Tree, and Random
  Forest (scikit-learn) on A + B features. Tuned on validation, evaluated
  once on test.
* **Phase 3 controller (frozen, hash `ae9db57acecd`):** runs RR1 (the best
  fixed quantum on train + validation). After the 20th job completes, a
  depth-3 decision tree on A + O features decides once whether to switch to
  RR8. If no decision point is reached, it keeps RR1.
* **Phase 4 controller (frozen, hash `f4021560ce9c`):** a logistic
  classifier on A + O + E features, deciding after the 5th completion.

## 4. Methodology highlights

* Every comparison is paired on identical job lists; Phase 5 verifies this
  with input hashes on every output row.
* Controllers were selected on validation only and frozen in a commit
  *before* any test evaluation (commit order: Phase 3 `a13ffba` → `4ada3b7`;
  Phase 4 `90dd9e5` → `b98b5ed`; Phase 5 protocol `d1b30be` → run `db55ae5`).
* **Phase 5 operating points:** 6, 14, and 22 processes per workload;
  1,000 workloads each; master seeds 5101–5103. Only the process count
  differs from the Phase 2 generator rules. No workload overlaps any of the
  2,100 earlier workloads.
* **Estimand:** the geometric mean of per-workload turnaround ratios
  (reported as a % change). The change in aggregate mean turnaround is also
  reported; it weights long workloads more.
* **Statistics:** two-sided paired sign-flip permutation test, paired
  bootstrap 95% CIs, Holm correction within pre-specified families, and a
  1% smallest effect size of interest (SESOI).

## 5. Principal findings (with qualifications)

**Phase 5, primary family (Holm over 6), vs RR1.** Values are the
geometric-mean change in per-workload turnaround with its 95% CI; negative
is faster.

| Operating point | Phase 3 controller | Phase 4 controller |
|---|---|---|
| 6 processes | −0.06% [−0.10, −0.02], Holm p = 0.026 | +0.13% [−0.19, +0.45], Holm p = 1 |
| 14 processes | −0.30% [−0.53, −0.08], Holm p = 0.038 | −0.04% [−0.38, +0.30], Holm p = 1 |
| 22 processes | −0.49% [−0.72, −0.26], Holm p = 0.0003 | −0.10% [−0.45, +0.24], Holm p = 1 |

* **Phase 3 controller:** a statistically supported improvement at every
  operating point, but **below the 1% practical threshold**. The change in
  aggregate mean turnaround is smaller (−0.01%, −0.12%, −0.28%). It costs
  response time: +0.0, +2.6, and +10.4 ticks vs RR1.
* **Phase 4 controller:** practically negligible at every operating point.
* **Headroom:** a per-workload oracle over fixed quanta (not deployable)
  would gain 2.4–3.3%.
* **Phase 2:** test accuracy 0.876 (LR, selected on validation), 0.864 (DT),
  0.892 (RF). Mapping predicted classes to policies did not reliably beat a
  fixed quantum.
* **Phase 3** (300 test workloads): −0.38% vs RR1, bootstrap CI
  [−0.75, −0.04], pre-specified Wilcoxon p = 0.50 (inconclusive). A
  post-hoc audit explains the disagreement: the two methods test different
  estimands.
* **Phase 4:** better estimates (error 0.43–0.50 → 0.08–0.10) but no better
  decisions. +0.64% vs Phase 3 on the established test split (Holm
  p = 0.14); +0.08% on 600 fresh workloads.

**Report wording to use:** "small, consistent, statistically supported but
practically negligible turnaround benefit in a synthetic CPU-only
simulation". Avoid "improves CPU scheduling" without these qualifiers.

## 6. Key limitations (state them in the report)

1. Synthetic workloads from one generator family; no real traces.
2. Simulator only: open-loop CPU-only projection, no I/O blocking, zero
   context-switch cost, single CPU. **No kernel-level or hardware
   validation.**
3. Controllers decide once per workload. RR1 is the default, and its many
   context switches would cost time on real hardware.
4. The 1% SESOI is a pre-declared judgement.
5. The Phase 3 test split was reused in Phase 4. Fresh workloads (Phase 4)
   and the independent operating points (Phase 5) address this.
6. Reproduction is verified on one environment only (Windows,
   Python 3.11.9, the library versions in
   `reports/phase5/environment.json`).

## 7. Suggested figures and tables (eight-page budget)

| Item | Source |
|---|---|
| Fig. 1: architecture diagram (simulator → features → controller → metrics) | draw from `docs/architecture.md` (not yet drawn) |
| Table 1: algorithms and metric definitions | `docs/architecture.md` |
| Table 2: feature availability categories A/B/C/O/E | `docs/phase2_ml.md` §2, `docs/phase4_observation_model.md` §2 |
| Table 3: Phase 2 classifier results | `reports/phase2/summary.md` |
| Fig. 2: Phase 5 primary effects with the 1% band and oracle | `reports/phase5/figures/primary_effects.png` |
| Table 4: Phase 5 primary family | `reports/phase5/comparisons.csv` (family = primary) |
| Fig. 3: response-time cost | `reports/phase5/figures/response_cost.png` |
| Fig. 4 (optional): estimator quality | `reports/phase4/figures/estimator_quality.png` |

### Proposed page allocation (8 pages total)

| Section | Pages |
|---|---|
| Title, abstract, problem statement, RQ | 0.75 |
| Background (RR quantum trade-off, SRPT optimality) and related work | 0.75 |
| Methodology (workloads, features, leakage controls, models, statistics) | 1.5 |
| Architecture (Fig. 1, controller decision boundary) | 0.75 |
| Experiments (protocol, three operating points) | 0.75 |
| Results (Tables 3–4, Figs. 2–3) | 1.5 |
| Discussion and limitations | 0.75 |
| Conclusion | 0.25 |
| References, contribution statement, repository link | 1.0 |

## 8. Claim → evidence map

| Claim | Evidence |
|---|---|
| Simulator correctness, metric definitions | `src/schedx/scheduler.py`, `src/schedx/metrics.py`, `tests/test_algorithms.py`, `tests/test_metrics_and_validation.py`, `docs/architecture.md` |
| Dataset size, classes, split | `reports/phase2/metrics.json` (`split`), `docs/data_dictionary.md`, `scripts/generate_dataset.py`, `configs/phase2.json` |
| Phase 2 accuracies | `reports/phase2/summary.md`, `reports/phase2/metrics.json`, `scripts/train_evaluate.py` |
| Phase 3 result and audit | `reports/phase3/summary.md`, `reports/phase3/per_workload_test.csv`, `reports/phase3/audit/AUDIT.md`, `reports/phase3/audit/audit_results.json` |
| Frozen controllers | `reports/phase3/frozen_controller.json`, `reports/phase4/frozen_controller.json` |
| Phase 4 result, estimator quality | `reports/phase4/summary.md`, `reports/phase4/results.json`, `reports/phase4/estimator_diagnostic.json` |
| Phase 5 operating points, seeds, inputs | `configs/phase5.json`, `reports/phase5/workload_manifest.csv`, `reports/phase5/inputs/processes_*.csv.gz` |
| Phase 5 outcomes (per workload, per method) | `reports/phase5/raw_workload_results.csv` |
| Phase 5 statistics | `reports/phase5/comparisons.csv`, `reports/phase5/results.json`, `reports/phase5/summary.md` |
| Phase 5 pairing, disjointness, recomputation | `reports/phase5/robustness_checks.json`, `docs/phase5_robustness.md` |
| Protocols fixed before evaluation | `docs/phase3_adaptive_scheduling.md` §9, `docs/phase4_observation_model.md` §6, `docs/phase5_experimental_protocol.md`, plus the commit order in §4 |

## 9. Reproduction

Python 3.11, from a fresh clone:

```bash
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
.venv\Scripts\python scripts/generate_dataset.py
.venv\Scripts\python scripts/train_evaluate.py
.venv\Scripts\python scripts/phase3_train.py
.venv\Scripts\python scripts/phase3_evaluate.py
.venv\Scripts\python scripts/phase4_train.py
.venv\Scripts\python scripts/phase4_evaluate.py
.venv\Scripts\python scripts/phase5_run.py
.venv\Scripts\python scripts/phase5_plots.py
.venv\Scripts\python scripts/phase5_audit.py
.venv\Scripts\python -m pytest -q
```

Plot and audit scripts for each phase are listed in the README. Expected
outcome and caveats: [`phase6_final_audit.md`](phase6_final_audit.md) §3.

## 10. Submission requirements checklist

Status values: **Verified**, **Partially verified**, **Not yet completed**,
**Not applicable**, **Needs your confirmation**.

| # | Requirement | Status | Evidence / remaining limitation |
|---|---|---|---|
| 1 | ≥ 3 independent workload sizes / operating points | **Verified** | 6 / 14 / 22 processes, 1,000 independently seeded workloads each; uniqueness and disjointness checked (`reports/phase5/robustness_checks.json`). |
| 2 | Identical inputs and configurations within comparisons | **Verified** | Same job list (hash per row), same simulator settings, frozen controllers checked against hashes (`docs/phase5_robustness.md` §2). |
| 3 | Original, verified results; clear limitations | **Partially verified** | Results are reproducible and audited (Phase 6 audit), and limitations are documented (§6). *Originality of authorship* cannot be established from the repository: see item 8 and §11. |
| 4 | Individual contribution statement | **Needs your confirmation** | The repository shows one Git author (`cremebits`). It cannot establish teammates, roles, or the division of work. |
| 5 | Organized code, generators, configs, raw CSVs | **Verified** | `src/schedx/`, `scripts/`, `configs/`, raw CSVs and seed manifest in `reports/phase5/`; per-workload CSVs for Phases 3–4. |
| 6 | Stable repository link | **Partially verified** | Public at https://github.com/ujvall/schedX. There are no release tags, so `main` can move. Recommended: cite a tag or the commit permalink https://github.com/ujvall/schedX/tree/5c78452 (or the final documentation commit). |
| 7 | Eight-page maximum | **Not yet completed** | The case study has not been written. |
| 8 | Plagiarism < 10% and AI-generated content < 10% | **Not yet completed; needs your confirmation** | No score can be predicted or promised. The repository's code and documentation were developed with substantial help from an AI coding assistant, and the Git history does not record this. Check the course's AI-use policy, disclose as it requires, and write the case study yourself from the evidence. |
| 9 | Citations for external material | **Not yet completed** | The repository docs cite no external sources. The methods used need citations in the report (candidates in §11). |

## 11. What you must confirm before finalizing the report

1. **Identity:** that Git author `cremebits <ujvalsaisurya21@gmail.com>` is
   you (GitHub `ujvall`), and the name to use in the report.
2. **Team or individual:** whether others contributed, and the exact
   contribution statement.
3. **AI-assistance disclosure:** how the course requires AI-assisted code
   and documentation to be declared, and that the written case study meets
   the course's AI-content and plagiarism rules.
4. **Stable link:** whether to create a release tag (for example
   `case-study-v1`) or cite a commit permalink.
5. **Citations:** sources for any background material. Candidate
   references for the methods used are listed below; verify each against
   the original before citing:
   * Kaplan & Meier (1958), *JASA* 53(282): 457–481 (Kaplan–Meier estimator).
   * Holm (1979), *Scandinavian Journal of Statistics* 6(2): 65–70 (Holm correction).
   * Efron (1979), *Annals of Statistics* 7(1): 1–26 (bootstrap).
   * Wilcoxon (1945), *Biometrics Bulletin* 1(6): 80–83, and Pratt (1959),
     *JASA* 54(287): 655–667 (signed-rank test, zero handling).
   * Schrage (1968), *Operations Research* 16(3): 687–690 (SRPT optimality).
   * Pedregosa et al. (2011), *JMLR* 12: 2825–2830 (scikit-learn).
   * A standard operating-systems textbook for the scheduling algorithms
     (edition to be confirmed).
6. **Environment statement:** that you accept "tested on one Windows
   environment, Python 3.11.9" as the reproducibility claim.
