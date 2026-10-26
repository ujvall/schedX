"""Phase 3.1 statistical audit (POST HOC; does not change the Phase 3 experiment).

Independently recomputes the Phase 3 test-set statistics from the saved
per-workload results, without using ``schedx.stats``, and adds estimand-matched
checks. It reads the committed Phase 3 outputs and writes ONLY to the audit
directory; the frozen controller and committed results are never modified.

Usage:  python scripts/audit_phase3.py [--reports reports/phase3] [--artifacts artifacts]
                                       [--out reports/phase3/audit]

Writes  <out>/audit_results.json  (the human-readable report is <out>/AUDIT.md)
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from schedx.ml import grouped_split  # noqa: E402  (only to rebuild the split for integrity checks)

SEED = 2024
EXACT_SIGNFLIP_MAX = 34  # 2^k sign patterns enumerated exactly up to this many non-zero pairs


def holm(p: dict) -> dict:
    names = sorted(p, key=p.get)
    k, running, out = len(names), 0.0, {}
    for i, n in enumerate(names):
        running = max(running, min(1.0, (k - i) * p[n]))
        out[n] = running
    return out


def signflip_p(nz: np.ndarray, rng: np.random.Generator, n_mc: int = 200_000) -> tuple[float, str]:
    """Two-sided paired sign-flip test of the MEAN (zeros contribute nothing).

    Exact (meet-in-the-middle over all 2^k patterns) for k <= EXACT_SIGNFLIP_MAX,
    otherwise Monte Carlo with ``n_mc`` draws (add-one corrected).
    """
    obs = abs(nz.sum())
    if len(nz) == 0:
        return 1.0, "exact"
    if len(nz) <= EXACT_SIGNFLIP_MAX:
        def sums(v):
            if len(v) == 0:
                return np.zeros(1)
            return np.array(np.meshgrid(*[[-x, x] for x in v], indexing="ij")).reshape(len(v), -1).sum(axis=0)

        a = np.abs(nz[: len(nz) // 2])
        c = np.sort(sums(np.abs(nz[len(nz) // 2:])))
        sa = sums(a)
        tol = 1e-12
        ge = len(c) - np.searchsorted(c, obs - sa - tol, side="left")
        le = np.searchsorted(c, -obs - sa + tol, side="right")
        return float((ge + le).sum() / (len(sa) * len(c))), f"exact ({len(sa) * len(c)} patterns)"
    signs = rng.choice([-1.0, 1.0], size=(n_mc, len(nz)))
    hits = (np.abs((signs * np.abs(nz)).sum(axis=1)) >= obs - 1e-12).sum()
    return float((hits + 1) / (n_mc + 1)), f"Monte Carlo ({n_mc} draws)"


def recompute(T: pd.DataFrame, method: str, baseline: str) -> dict:
    m, b = T[method].to_numpy(), T[baseline].to_numpy()
    lr = np.log(m / b)
    nz = lr[lr != 0]
    rng = np.random.default_rng(SEED)
    B = 100_000
    boots = np.concatenate([lr[rng.integers(0, len(lr), size=(10_000, len(lr)))].mean(axis=1) for _ in range(B // 10_000)])
    bca = stats.bootstrap((lr,), np.mean, n_resamples=20_000, method="BCa", random_state=SEED)

    # Pratt Wilcoxon reproduced by hand: rank |d| over ALL pairs, drop zeros,
    # normal approximation with mean/variance of the remaining ranks.
    ranks = stats.rankdata(np.abs(lr))
    keep = lr != 0
    r, sgn = ranks[keep], np.sign(lr[keep])
    t_plus = float(r[sgn > 0].sum())
    z = (t_plus - r.sum() / 2) / np.sqrt((r ** 2).sum() / 4) if len(r) else 0.0
    rz = stats.rankdata(np.abs(nz))
    p_flip, flip_kind = signflip_p(nz, rng)
    return {
        "comparison": f"{method} vs {baseline}",
        "n_workloads": int(len(lr)),
        "n_zero_difference": int((lr == 0).sum()),
        "n_better": int((lr < 0).sum()),
        "n_worse": int((lr > 0).sum()),
        "min_abs_nonzero_log_ratio": float(np.abs(nz).min()) if len(nz) else None,
        "estimates": {
            "geometric_mean_per_workload_change": float(np.expm1(lr.mean())),
            "arithmetic_mean_per_workload_change": float((m / b - 1).mean()),
            "change_in_mean_turnaround": float(m.mean() / b.mean() - 1),
            "mean_difference_ticks": float((m - b).mean()),
        },
        "intervals_for_geometric_mean_change": {
            "percentile_100k": [float(np.expm1(np.percentile(boots, 2.5))), float(np.expm1(np.percentile(boots, 97.5)))],
            "bca_20k": [float(np.expm1(bca.confidence_interval.low)), float(np.expm1(bca.confidence_interval.high))],
            "bootstrap_share_mean_ge_0": float((boots >= 0).mean()),
        },
        "tests": {
            "wilcoxon_pratt_scipy_prespecified": float(stats.wilcoxon(lr, zero_method="pratt").pvalue) if len(nz) else 1.0,
            "wilcoxon_pratt_manual": {"T_plus": t_plus, "z": float(z), "p": float(2 * stats.norm.sf(abs(z))),
                                      "nonzero_rank_range": [float(r.min()), float(r.max())] if len(r) else None},
            "wilcoxon_zeros_dropped": float(stats.wilcoxon(nz).pvalue) if len(nz) else 1.0,
            "sign_test": float(stats.binomtest(int((nz < 0).sum()), len(nz)).pvalue) if len(nz) else 1.0,
            "t_test_log_ratio": float(stats.ttest_1samp(lr, 0).pvalue),
            "signflip_permutation_mean_log_ratio": p_flip,
            "signflip_kind": flip_kind,
        },
        "effect_sizes": {
            "rank_biserial_zeros_dropped": float((rz[nz > 0].sum() - rz[nz < 0].sum()) / rz.sum()) if len(nz) else 0.0,
            "rank_biserial_pratt_ranks": float((r[sgn > 0].sum() - r[sgn < 0].sum()) / r.sum()) if len(r) else 0.0,
            "cohens_dz_log_ratio": float(lr.mean() / lr.std(ddof=1)) if lr.std(ddof=1) > 0 else 0.0,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", default=str(ROOT / "reports" / "phase3"))
    ap.add_argument("--artifacts", default=str(ROOT / "artifacts"))
    ap.add_argument("--out", default=str(ROOT / "reports" / "phase3" / "audit"))
    args = ap.parse_args()
    reports, artifacts, out = Path(args.reports), Path(args.artifacts), Path(args.out)

    R = json.loads((reports / "results.json").read_text(encoding="utf-8"))
    frozen = json.loads((reports / "frozen_controller.json").read_text(encoding="utf-8"))
    long = pd.read_csv(reports / "per_workload_test.csv")
    T = long.pivot(index="workload_id", columns="method", values="avg_turnaround")

    cfg2 = json.loads((ROOT / "configs" / "phase2.json").read_text(encoding="utf-8"))
    feats = pd.read_csv(artifacts / "dataset" / "features.csv")
    s = cfg2["split"]
    split = grouped_split(feats, s["seed"], s["train"], s["validation"], s["test"])
    test_ids = set(split.index[split == "test"])

    integrity = {
        "workloads_per_method": long.groupby("method").workload_id.nunique().to_dict(),
        "duplicate_workload_method_rows": int(long.duplicated(["workload_id", "method"]).sum()),
        "missing_turnaround": int(T.isna().sum().sum()),
        "ids_equal_grouped_split_test_ids": set(T.index) == test_ids,
        "ids_overlapping_train_or_validation": len(set(T.index) & set(split.index[split != "test"])),
        "config_hash_results_equals_frozen": R["config_hash"] == frozen["config_hash"],
        "adaptive_online_identical_to_adaptive_selected": bool((T.adaptive_online == T.adaptive_selected).all()),
    }

    comparisons = {
        "primary": recompute(T, "adaptive_selected", "best_fixed"),
        "submission_vs_best_fixed": recompute(T, "adaptive_submission", "best_fixed"),
        "selected_vs_fixed_RR2": recompute(T, "adaptive_selected", "fixed_RR2"),
    }
    committed = {}
    for key, name in (("primary", "adaptive_selected vs best_fixed"),
                      ("submission_vs_best_fixed", "adaptive_submission vs best_fixed"),
                      ("selected_vs_fixed_RR2", "adaptive_selected vs fixed_RR2")):
        c, mine = R["deployable_comparisons"][name], comparisons[key]
        committed[key] = {
            "geo_rel_change_matches": np.isclose(c["geo_rel_change"], mine["estimates"]["geometric_mean_per_workload_change"], rtol=0, atol=1e-15),
            "wilcoxon_p_matches": np.isclose(c["wilcoxon_p"], mine["tests"]["wilcoxon_pratt_manual"]["p"], rtol=1e-12),
            "rank_biserial_committed": c["rank_biserial"],
            "rank_biserial_committed_is_zeros_dropped_version": bool(np.isclose(
                c["rank_biserial"], mine["effect_sizes"]["rank_biserial_zeros_dropped"])),
            "mean_abs_diff_matches": np.isclose(c["mean_abs_diff"], mine["estimates"]["mean_difference_ticks"]),
            "committed_percentile_ci_10k": c["geo_rel_change_ci95"],
            "committed_holm_p_9": c["holm_p"],
        }
        committed[key] = {k: (bool(v) if isinstance(v, np.bool_) else v) for k, v in committed[key].items()}

    raw = {k: v["wilcoxon_p"] for k, v in R["deployable_comparisons"].items()}
    distinct = {k: v for k, v in raw.items() if not k.startswith("adaptive_online")}
    h9, h7 = holm(raw), holm(distinct)
    holm_table = {k: {"raw_p": raw[k], "holm_committed_9": R["deployable_comparisons"][k]["holm_p"],
                      "holm_recomputed_9": h9[k], "holm_distinct_7": h7.get(k)} for k in raw}

    audit = {
        "label": "Phase 3.1 statistical audit - POST HOC. Does not replace the pre-specified Phase 3 analysis.",
        "inputs": {"results_json_config_hash": R["config_hash"], "dataset_version": R["dataset_version"]},
        "integrity": integrity,
        "comparisons": comparisons,
        "committed_vs_recomputed": committed,
        "holm_family": {"committed_size": len(raw), "distinct_size": len(distinct), "table": holm_table},
        "resampling_counts_used_in_phase3": {
            "main_comparisons": frozen["phase3_config"]["bootstrap"]["n"],
            "breakdowns": 2_000,  # hardcoded in schedx.phase3.analyze (groups)
        },
        "seeds": {"audit": SEED},
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "audit_results.json").write_text(json.dumps(audit, indent=2, default=bool), encoding="utf-8")
    print(f"wrote {out / 'audit_results.json'}")


if __name__ == "__main__":
    main()
