"""Paired comparison statistics for Phase 3 (docs/phase3_adaptive_scheduling.md §9).

Resampling unit: the workload. Each method is evaluated on the same
workloads, so every comparison is paired.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata, wilcoxon


def _ci(boot: np.ndarray) -> list[float]:
    return [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]


def rank_biserial(d: np.ndarray) -> float:
    """Matched-pairs rank-biserial correlation of non-zero differences
    (-1: method always lower, +1: always higher)."""
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    r = rankdata(np.abs(d))
    return float((r[d > 0].sum() - r[d < 0].sum()) / r.sum())


def paired_comparison(method, baseline, seed: int = 0, n_boot: int = 10_000) -> dict:
    """Compare a method with a baseline on identical workloads (lower = better).

    Effect per workload: ``log(method / baseline)``. Returns descriptive
    statistics, paired bootstrap 95% CIs (percentile, resampling workloads),
    a two-sided Wilcoxon signed-rank test on the log ratios, and effect sizes.
    """
    m = np.asarray(method, dtype=float)
    b = np.asarray(baseline, dtype=float)
    if m.shape != b.shape or m.ndim != 1 or len(m) == 0:
        raise ValueError("method and baseline must be equal-length non-empty 1-D arrays")
    if np.any(m <= 0) or np.any(b <= 0):
        raise ValueError("values must be positive for log ratios")
    diff = m - b
    logr = np.log(m / b)
    idx = np.random.default_rng(seed).integers(0, len(m), size=(n_boot, len(m)))
    boot_log = logr[idx].mean(axis=1)
    boot_diff = diff[idx].mean(axis=1)
    boot_rel = (m / b - 1)[idx].mean(axis=1)

    nonzero = int(np.count_nonzero(logr))
    if nonzero == 0:
        p = 1.0
    else:
        p = float(wilcoxon(logr, zero_method="pratt", alternative="two-sided").pvalue)
    sd = logr.std(ddof=1) if len(logr) > 1 else 0.0
    return {
        "n": int(len(m)),
        "method_mean": float(m.mean()),
        "method_median": float(np.median(m)),
        "baseline_mean": float(b.mean()),
        "baseline_median": float(np.median(b)),
        "mean_abs_diff": float(diff.mean()),
        "mean_abs_diff_ci95": _ci(boot_diff),
        "median_abs_diff": float(np.median(diff)),
        "geo_rel_change": float(np.expm1(logr.mean())),
        "geo_rel_change_ci95": [float(np.expm1(x)) for x in _ci(boot_log)],
        "mean_rel_change": float((m / b - 1).mean()),
        "mean_rel_change_ci95": _ci(boot_rel),
        "median_rel_change": float(np.median(m / b - 1)),
        "share_better": float((logr < 0).mean()),
        "share_equal": float((logr == 0).mean()),
        "share_worse": float((logr > 0).mean()),
        "n_nonzero": nonzero,
        "wilcoxon_p": p,
        "rank_biserial": rank_biserial(logr),
        "cohens_dz": float(logr.mean() / sd) if sd > 0 else 0.0,
    }


def mean_diff_ci(method, baseline, seed: int = 0, n_boot: int = 10_000) -> dict:
    """Mean paired difference with a bootstrap CI; for metrics that can be 0
    (e.g. response time), where log ratios are undefined."""
    d = np.asarray(method, dtype=float) - np.asarray(baseline, dtype=float)
    idx = np.random.default_rng(seed).integers(0, len(d), size=(n_boot, len(d)))
    return {"mean_diff": float(d.mean()), "mean_diff_ci95": _ci(d[idx].mean(axis=1)), "median_diff": float(np.median(d))}


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm–Bonferroni adjusted p-values (monotone, capped at 1)."""
    names = sorted(pvalues, key=pvalues.get)
    k = len(names)
    out, running = {}, 0.0
    for i, n in enumerate(names):
        running = max(running, min(1.0, (k - i) * pvalues[n]))
        out[n] = running
    return out
