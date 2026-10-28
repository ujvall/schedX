# Phase 6 — Final Evidence Audit and Research Freeze

Phase 6 audits the frozen evidence for the case study. It changes no code,
configuration, controller, seed, metric, or result. The only files added
are this audit and [`final_case_study_handoff.md`](final_case_study_handoff.md).

Audited state: `main` = `origin/main` = `5c78452`, clean working tree,
26 commits, all by `cremebits <ujvalsaisurya21@gmail.com>`, with no
co-author trailers or AI attribution in commit messages.

## 1. Claims checked against committed files

| Claim | Verified value | Source |
|---|---|---|
| Phase 5 used 6, 14, 22 processes; 1,000 workloads each | yes: 32,326 / 77,567 / 121,900 jobs | `reports/phase5/results.json`, `workload_manifest.csv` |
| Phase 3 controller: statistically supported, sub-1% gains vs RR1 at all three operating points | yes: −0.06%, −0.30%, −0.49% (Holm p 0.026, 0.038, 0.0003); classified "below the practical threshold" | `reports/phase5/comparisons.csv` |
| Phase 4 estimator: no practically meaningful improvement | yes: Phase 5 negligible at all three operating points; Phase 4 H1 +0.64% (test, Holm p 0.14) and +0.08% (fresh) | `reports/phase5/comparisons.csv`, `reports/phase4/results.json` |
| Synthetic, simulator-based, CPU-only; no kernel or hardware validation | yes: no hardware or kernel experiment exists in the repository | all docs; `reports/phase5/environment.json` (`simulator`) |
| Reproducible from a clean clone | Phase 3–5 were recorded in `docs/phase5_robustness.md` §8, but **Phase 2 was not**. Phase 6 re-ran Phases 2–5: everything is identical except the Phase 2 timing fields (§3) | — |
| Frozen controllers intact | config hashes `ae9db57acecd` / `f4021560ce9c` match the artifacts and frozen JSON | `reports/phase3|4/frozen_controller.json` |
| Phase 5 robustness audit | `all_pass: true` (17 checks) | `reports/phase5/robustness_checks.json` |

No correctness defect was found. No evidence was modified.

## 2. Repository checks

| Check | Result |
|---|---|
| Automated tests | **243 passed** (main checkout, Python 3.11.9) |
| Markdown links (13 files) | no broken relative links; no references to missing files |
| README commands | all 16 referenced scripts exist; every script is documented |
| Repository URL | https://github.com/ujvall/schedX: public (anonymous API status 200); no release tags |
| Personal information | no local paths, email addresses, or credentials in tracked files (the author email appears only in Git metadata) |
| Secrets | none found by pattern scan |
| Size | ≈ 9.8 MB tracked; no model binaries; ignored: `.venv/`, `artifacts/` (rebuilt deterministically), caches, `graphify-out/` |
| Raw data discoverability | per-workload CSVs for Phases 3, 4, and 5; Phase 5 seed manifest and input traces; Phase 2 dataset regenerated from `configs/phase2.json` (not committed; about 3.5 MB) |

**Packaging gap (not an evidence defect):** `scipy` (in `schedx.stats` and
the audit scripts) and `joblib` (model persistence) are imported directly
but not declared in `pyproject.toml`. Installation works because
scikit-learn requires both. No lock file pins the tested versions; they are
recorded in `reports/phase5/environment.json` and the Phase 2–4 metadata.
Suggested follow-up: add `scipy` and `joblib` to `dependencies`, and commit
a `requirements-lock.txt` from the tested environment. Not changed here,
to keep the evidence freeze strict.

## 3. Clean-clone reproduction of Phases 2–5

A fresh clone of `5c78452` was made in a temporary directory, and all 16
documented scripts were run in order with the tested environment (Windows,
Python 3.11.9; library versions as in `reports/phase5/environment.json`):

1. `generate_dataset`, `train_evaluate`, `plot_results`, `predict`, `demo`
2. `phase3_train`, `phase3_evaluate`, `phase3_plots`, `audit_phase3`
3. `phase4_baseline`, `phase4_train`, `phase4_evaluate`, `phase4_plots`
4. `phase5_run`, `phase5_plots`, `phase5_audit`

Every step succeeded. Total runtime was about 11 minutes. The main
checkout was not touched.

**Result: all committed Phase 3, 3.1, 4, and 5 evidence regenerated
identically** (Git reports no differences): frozen controllers, validation
tables, per-workload CSVs, results, summaries, audit files, and all
figures. The Phase 5 robustness audit passed again.

**Phase 2: identical except wall-clock timing.** Only
`reports/phase2/metrics.json` and `summary.md` differ, and only in the
`fit_seconds` and `inference_us_per_row` fields (0 non-timing lines
changed; every summary row identical apart from its two timing columns).
All Phase 2 scores, selections, splits, and figures reproduce. Timing is
measured wall-clock time and cannot be byte-reproducible; report it as
indicative only.

The earlier statement that "Phase 2–5 results are reproducible from a clean
clone" is therefore accurate for every result, with the Phase 2
timing-field exception. Before this audit, Phase 2 had not been part of a
recorded clean-clone run; it now has been.

Not tested: other operating systems, Python versions, or library versions.

## 4. Remaining limitations

See [`final_case_study_handoff.md`](final_case_study_handoff.md) §6
(research limitations) and §10 (submission requirements). In short:

* simulator and synthetic data only;
* verified on one environment;
* the case study, contribution statement, citations, and AI-use disclosure
  are not yet done and need your input.
