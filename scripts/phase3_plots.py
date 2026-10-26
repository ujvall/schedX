"""Regenerate Phase 3 figures from the saved test results (nothing hardcoded).

Usage:  python scripts/phase3_plots.py [--reports reports/phase3]

Reads <reports>/results.json and per_workload_test.csv; writes <reports>/figures/*.png.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

# Validated categorical palette (fixed order) + neutral gray for non-deployable
# references. Aqua is < 3:1 on the surface, so its marks carry value labels.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE, INK, INK2, GRID, REF = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#8a8984"
COLOR = {"adaptive_selected": BLUE, "adaptive_submission": ORANGE, "phase2_map_predicted": AQUA,
         "fixed_RR2": INK2, "oracle_fixed": REF, "oracle_online": REF, "phase2_map_true": REF}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False,
})


def duplicates(r: dict) -> set[str]:
    """Methods that are the same frozen controller as adaptive_selected."""
    c = r["controllers"]
    alias = {"adaptive_submission": "best_submission", "adaptive_online": "best_online"}
    return {m for m, k in alias.items() if c[k]["label"] == c["selected"]["label"]}


def save(fig, out: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(out / name, dpi=130)
    plt.close(fig)
    print("wrote", out / name)


def ci_rows(ax, rows, key, ci_key, scale, fmt, xlabel):
    """Dot + CI per comparison; rows = [(label, stats, color)]."""
    y = np.arange(len(rows))
    for yi, (label, s, color) in zip(y, rows):
        v, (lo, hi) = s[key] * scale, (s[ci_key][0] * scale, s[ci_key][1] * scale)
        ax.errorbar(v, yi, xerr=[[v - lo], [hi - v]], fmt="o", color=color, ecolor=color, elinewidth=2,
                    capsize=4, markersize=8, markeredgecolor=SURFACE, markeredgewidth=2)
        ax.annotate(fmt.format(v), (v, yi), textcoords="offset points", xytext=(0, 9), ha="center",
                    fontsize=8, color=INK2)
    ax.axvline(0, color=INK2, linewidth=1)
    ax.set_yticks(y, [r[0] for r in rows], fontsize=8.5)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)


def fixed_vs_adaptive(T: pd.DataFrame, r: dict, out: Path) -> None:
    q0 = r["default_quantum_q0"]
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    changed = T.adaptive_selected != T.best_fixed
    ax.scatter(T.best_fixed[~changed], T.adaptive_selected[~changed], s=14, color=REF, alpha=0.6,
               label=f"same decision as RR{q0} ({(~changed).sum()})")
    ax.scatter(T.best_fixed[changed], T.adaptive_selected[changed], s=40, color=BLUE,
               edgecolor=SURFACE, linewidth=1.5, label=f"controller switched ({changed.sum()})")
    lim = [T[["best_fixed", "adaptive_selected"]].min().min() * 0.9, T[["best_fixed", "adaptive_selected"]].max().max() * 1.1]
    ax.plot(lim, lim, color=INK2, linewidth=1, linestyle="--", label="equal turnaround")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel(f"best fixed quantum RR{q0}: average turnaround (ticks)")
    ax.set_ylabel("adaptive (selected): average turnaround (ticks)")
    ax.set_title("Per-workload turnaround, test set\n(below the line = adaptive better)")
    ax.legend(loc="upper left", fontsize=8.5)
    save(fig, out, "fixed_vs_adaptive.png")


def paired_differences(T: pd.DataFrame, r: dict, out: Path) -> None:
    rel = (T.adaptive_selected / T.best_fixed - 1).sort_values().to_numpy() * 100
    fig, ax = plt.subplots(figsize=(9, 3.8))
    x = np.arange(len(rel))
    colors = np.where(rel < 0, BLUE, np.where(rel > 0, ORANGE, REF))
    ax.bar(x, rel, width=1.0, color=colors)
    ax.axhline(0, color=INK2, linewidth=1)
    nb, nw = int((rel < 0).sum()), int((rel > 0).sum())
    ax.set_xlabel(f"test workloads sorted by change ({nb} better, {len(rel) - nb - nw} unchanged, {nw} worse)")
    ax.set_ylabel("change in avg turnaround (%)")
    ax.set_title(f"Paired per-workload difference: adaptive (selected) vs RR{r['default_quantum_q0']}")
    ax.set_xlim(-1, len(rel))
    save(fig, out, "paired_differences.png")


def improvement_cis(r: dict, out: Path) -> None:
    dup = duplicates(r)
    comps = {k: v for k, v in r["deployable_comparisons"].items() if k.split(" vs ")[0] not in dup}
    rows =[(k.replace(" vs ", "\nvs "), v, COLOR.get(k.split(" vs ")[0], INK2)) for k, v in comps.items()]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 6.2), sharey=True)
    ci_rows(a1, rows, "geo_rel_change", "geo_rel_change_ci95", 100, "{:+.2f}%",
            "geometric-mean change in avg turnaround (%), 95% CI")
    ci_rows(a2, rows, "mean_abs_diff", "mean_abs_diff_ci95", 1, "{:+.2f}", "mean difference (ticks), 95% CI")
    a1.set_title("Relative change (negative = better)")
    a2.set_title("Absolute change (negative = better)")
    note = f"\n({', '.join(sorted(dup))}: same controller as adaptive_selected, not repeated)" if dup else ""
    fig.suptitle("Deployable methods, paired over test workloads (paired bootstrap, 10,000 resamples)" + note,
                 fontweight="bold", fontsize=10.5)
    save(fig, out, "improvement_cis.png")


def by_group(r: dict, out: Path) -> None:
    dims = [("workload_class", "true workload class (reporting only)"),
            ("n_processes_tercile", "process count tercile"), ("burst_cv_tercile", "burst-variability (cv) tercile")]
    methods = [("adaptive_selected", BLUE), ("adaptive_submission", ORANGE), ("oracle_fixed", REF)]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), sharey=True)
    for ax, (dim, title) in zip(axes, dims):
        groups = list(r["breakdowns"][dim])
        x = np.arange(len(groups))
        for i, (m, color) in enumerate(methods):
            vals = [r["breakdowns"][dim][g][m] for g in groups]
            v = np.array([s["geo_rel_change"] for s in vals]) * 100
            lo = np.array([s["geo_rel_change_ci95"][0] for s in vals]) * 100
            hi = np.array([s["geo_rel_change_ci95"][1] for s in vals]) * 100
            ax.errorbar(x + (i - 1) * 0.22, v, yerr=[v - lo, hi - v], fmt="o", color=color, ecolor=color,
                        elinewidth=2, capsize=3, markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5,
                        label=m + (" (not deployable)" if m.startswith("oracle") else ""))
        ax.axhline(0, color=INK2, linewidth=1)
        ax.set_xticks(x, [f"{g}\n(n={r['breakdowns'][dim][g]['n']})" for g in groups], fontsize=8)
        ax.set_title(title, fontsize=10)
    axes[0].set_ylabel("geo change vs best fixed (%), 95% CI")
    axes[0].legend(loc="lower left", fontsize=8)
    fig.suptitle("Change in avg turnaround vs best fixed quantum, by workload group", fontweight="bold")
    save(fig, out, "by_group.png")


def selected_quantum(r: dict, out: Path) -> None:
    methods = [m for m in ("adaptive_selected", "adaptive_submission", "oracle_fixed", "oracle_online")
               if m not in duplicates(r)]
    fig, axes = plt.subplots(1, len(methods), figsize=(14, 3.8), sharey=True)
    for ax, m in zip(axes, methods):
        counts = r["decisions"][m]["policy_counts"]
        names = sorted(counts, key=lambda k: (len(k), k))
        bars = ax.bar(names, [counts[n] for n in names], color=COLOR.get(m, REF), width=0.7)
        ax.bar_label(bars, padding=2, fontsize=8, color=INK2)
        ax.set_title(m + ("\n(not deployable)" if m.startswith("oracle") else ""), fontsize=10)
        ax.tick_params(axis="x", labelsize=7.5, rotation=30)
    axes[0].set_ylabel("test workloads")
    fig.suptitle("Chosen policy per test workload. RRa->RRb: ran RRa until the decision point, then RRb.\n"
                 "A plain RRa under an online method: no decision point (fallback).", fontweight="bold", fontsize=10.5)
    save(fig, out, "selected_quantum.png")


def vs_oracle(r: dict, out: Path) -> None:
    gap = r["oracle_gap_closed"]
    names = [n for n in gap if n not in duplicates(r)]
    vals = [100 * (gap[n] or 0) for n in names]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    bars = ax.barh(names, vals, color=[COLOR.get(n, REF) for n in names], height=0.6)
    ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=8, color=INK2)
    ax.axvline(0, color=INK2, linewidth=1)
    ax.invert_yaxis()
    ax.set_xlabel(f"share of the RR{r['default_quantum_q0']} → per-workload oracle gap closed "
                  "(mean log turnaround, %)")
    ax.set_title("How much of the gap to the per-workload oracle each method closes")
    fig.text(0.01, 0.01, "Gray = not deployable (oracles know every candidate's outcome; phase2_map_true uses the "
             "true label). phase2_map_* can also pick FCFS,\nwhich is outside the oracle's RR candidates. "
             "oracle_online = best switch target at the controller's decision point.", fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(out / "adaptive_vs_oracle.png", dpi=130)
    plt.close(fig)
    print("wrote", out / "adaptive_vs_oracle.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", default=str(ROOT / "reports" / "phase3"))
    args = ap.parse_args()
    reports = Path(args.reports)
    r = json.loads((reports / "results.json").read_text(encoding="utf-8"))
    long = pd.read_csv(reports / "per_workload_test.csv")
    T = long.pivot(index="workload_id", columns="method", values="avg_turnaround")
    out = reports / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fixed_vs_adaptive(T, r, out)
    paired_differences(T, r, out)
    improvement_cis(r, out)
    by_group(r, out)
    selected_quantum(r, out)
    vs_oracle(r, out)


if __name__ == "__main__":
    main()
