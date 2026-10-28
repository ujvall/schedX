"""Regenerate Phase 4 figures from saved results (nothing hardcoded).

Usage:  python scripts/phase4_plots.py [--reports reports/phase4]

Reads <reports>/results.json, per_workload_{test,fresh}.csv, estimator_diagnostic.json;
writes <reports>/figures/*.png.
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

# Validated categorical palette (fixed order); gray = baselines / not deployable.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE, INK, INK2, GRID, REF = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#8a8984"
COLOR = {"p4_selected": BLUE, "p4_ablation": ORANGE, "p3_frozen": AQUA}
SETS = (("established_test_split", "test", "established test split"), ("fresh_confirmation", "fresh", "fresh confirmation set"))

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False,
})


def save(fig, out: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(out / name, dpi=130)
    plt.close(fig)
    print("wrote", out / name)


def dots(ax, rows, key="geo_rel_change", ci_key="geo_rel_change_ci95", scale=100, fmt="{:+.2f}%"):
    y = np.arange(len(rows))
    for yi, (label, s, color) in zip(y, rows):
        v, lo, hi = s[key] * scale, s[ci_key][0] * scale, s[ci_key][1] * scale
        ax.errorbar(v, yi, xerr=[[v - lo], [hi - v]], fmt="o", color=color, ecolor=color, elinewidth=2, capsize=4,
                    markersize=8, markeredgecolor=SURFACE, markeredgewidth=2)
        ax.annotate(fmt.format(v), (v, yi), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=8, color=INK2)
    ax.axvline(0, color=INK2, linewidth=1)
    ax.set_yticks(y, [r[0] for r in rows], fontsize=8.5)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.invert_yaxis()


def primary_family(r: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.8), sharex=True)
    for ax, (key, _, title) in zip(axes, SETS):
        fam = r[key]["primary_family"]
        rows = []
        for h, v in fam.items():
            merged = f" (+{','.join(v['merged_with'])})" if v["merged_with"] else ""
            a = v["hypothesis"].split(" vs ")[0]
            rows.append((f"{h}{merged}: {v['hypothesis']}\nsign-flip p={v['signflip_p']:.3g}, Holm p={v['holm_p']:.3g}",
                         v, COLOR.get(a, INK2)))
        dots(ax, rows)
        ax.set_title(f"{title} (n={r[key]['n_workloads']})")
        ax.set_xlabel("geometric-mean change in avg turnaround (%), 95% CI")
    fig.suptitle("Pre-specified hypothesis family (negative = improvement)", fontweight="bold")
    save(fig, out, "primary_family.png")


def methods_vs_rr1(r: dict, out: Path) -> None:
    q0 = r["default_quantum_q0"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), sharex=True)
    for ax, (key, _, title) in zip(axes, SETS):
        s = r[key]
        fam = {v["hypothesis"]: v for v in s["primary_family"].values()}
        comp = {**{k: v for k, v in s["secondary_comparisons"].items() if k.endswith(f"RR{q0}")},
                **{k: v for k, v in fam.items() if k.endswith(f"RR{q0}")}}
        order = ["p4_selected", "p4_ablation", "p3_frozen", "submission_ref", "oracle_online", "oracle_fixed"]
        rows = [(m + (" (not deployable)" if m.startswith("oracle") else ""), comp[f"{m} vs RR{q0}"],
                 COLOR.get(m, REF)) for m in order if f"{m} vs RR{q0}" in comp]
        dots(ax, rows)
        ax.set_title(f"{title} (n={s['n_workloads']})")
        ax.set_xlabel(f"geometric-mean change vs RR{q0} (%), 95% CI")
    fig.suptitle(f"Every method against the best fixed quantum RR{q0}", fontweight="bold")
    save(fig, out, "methods_vs_rr1.png")


def paired_differences(longs: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.8))
    for ax, (_, tag, title) in zip(axes, SETS):
        T = longs[tag].pivot(index="workload_id", columns="method", values="avg_turnaround")
        rel = np.sort((T.p4_selected / T.p3_frozen - 1).to_numpy()) * 100
        colors = np.where(rel < 0, BLUE, np.where(rel > 0, ORANGE, REF))
        ax.bar(np.arange(len(rel)), rel, width=1.0, color=colors)
        ax.axhline(0, color=INK2, linewidth=1)
        nb, nw = int((rel < 0).sum()), int((rel > 0).sum())
        ax.set_xlabel(f"workloads sorted ({nb} better, {len(rel) - nb - nw} equal, {nw} worse)")
        ax.set_ylabel("change in avg turnaround (%)")
        ax.set_title(f"p4_selected vs p3_frozen: {title}")
        ax.set_xlim(-1, len(rel))
    save(fig, out, "paired_differences.png")


def tradeoff(r: dict, out: Path) -> None:
    q0 = r["default_quantum_q0"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for ax, (key, _, title) in zip(axes, SETS):
        M = r[key]["metrics"]
        base_t, base_r = M["avg_turnaround"][f"RR{q0}"]["mean"], M["avg_response"][f"RR{q0}"]["mean"]
        for m in M["avg_turnaround"]:
            if m == f"RR{q0}":
                continue
            x = 100 * (M["avg_turnaround"][m]["mean"] / base_t - 1)
            y = 100 * (M["avg_response"][m]["mean"] / base_r - 1)
            ax.scatter(x, y, s=60, color=COLOR.get(m, REF), edgecolor=SURFACE, linewidth=1.5, zorder=3)
            # Fixed offsets keep the clustered deployable methods' labels apart.
            off = {"p3_frozen": (-8, -14), "p4_ablation": (8, -12), "submission_ref": (8, 10)}.get(m, (6, 4))
            ax.annotate(m, (x, y), textcoords="offset points", xytext=off, fontsize=8, color=INK2,
                        ha="right" if off[0] < 0 else "left")
        ax.axvline(0, color=INK2, linewidth=1)
        ax.axhline(0, color=INK2, linewidth=1)
        ax.set_xlabel(f"change in mean turnaround vs RR{q0} (%)")
        ax.set_ylabel(f"change in mean response vs RR{q0} (%)")
        ax.set_title(title)
    fig.suptitle("Turnaround vs response-time trade-off (lower-left is better on both)", fontweight="bold")
    save(fig, out, "turnaround_response_tradeoff.png")


def quantum_changes(r: dict, out: Path) -> None:
    s = r["established_test_split"]["quantum_changes"]
    methods = ["p3_frozen", "p4_selected", "p4_ablation", "oracle_online"]
    fig, axes = plt.subplots(1, len(methods), figsize=(14, 3.8), sharey=True)
    for ax, m in zip(axes, methods):
        counts = s[m]["policy_counts"]
        names = sorted(counts, key=lambda k: (len(k), k))
        bars = ax.bar(names, [counts[n] for n in names], color=COLOR.get(m, REF), width=0.7)
        ax.bar_label(bars, padding=2, fontsize=8, color=INK2)
        ax.set_title(f"{m}{' (not deployable)' if m.startswith('oracle') else ''}\nchanged: {s[m]['share_quantum_changed']:.0%}",
                     fontsize=10)
        ax.tick_params(axis="x", labelsize=7.5, rotation=30)
    axes[0].set_ylabel("test workloads")
    fig.suptitle("Chosen policy on the established test split (RRa->RRb: switched at the decision point; "
                 "plain RRa: no decision point)", fontweight="bold", fontsize=10.5)
    save(fig, out, "quantum_changes.png")


def estimator_quality(diag: dict, out: Path) -> None:
    by = diag["by_decision_point"]
    ms = list(by)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
    x = np.arange(len(ms))
    for i, (k, label, color) in enumerate((("completed_only", "completed bursts only", AQUA), ("kaplan_meier", "Kaplan-Meier", BLUE))):
        vals = [by[m]["mean_abs_log_error_restricted_mean"][k] for m in ms]
        bars = a1.bar(x + (i - 0.5) * 0.36, vals, 0.34, color=color, label=label)
        a1.bar_label(bars, fmt="%.3f", padding=2, fontsize=8, color=INK2)
    a1.set_xticks(x, [f"m = {m}" for m in ms])
    a1.set_ylabel("mean |log error| of restricted mean burst")
    a1.set_title("Estimation error vs true released bursts")
    a1.set_ylim(0, a1.get_ylim()[1] * 1.3)  # headroom so the legend clears the value labels
    a1.legend(loc="upper center", ncols=2)
    for i, (k, label, color) in enumerate((("completed_cv", "completed-burst CV", AQUA), ("km_restricted_cv", "KM restricted CV", BLUE))):
        vals = [by[m]["spearman_with_target"][k] for m in ms]
        bars = a2.bar(x + (i - 0.5) * 0.36, vals, 0.34, color=color, label=label)
        a2.bar_label(bars, fmt="%.2f", padding=2, fontsize=8, color=INK2)
    a2.set_xticks(x, [f"m = {m}" for m in ms])
    a2.set_ylim(0, 1)
    a2.set_ylabel(f"Spearman with {diag['target']}")
    a2.set_title("Decision-relevant signal")
    a2.legend(loc="upper left")
    fig.suptitle(f"Estimator quality, training workloads only (n={diag['n_train']}); retrospective analysis, "
                 "never a controller input", fontweight="bold", fontsize=10.5)
    save(fig, out, "estimator_quality.png")


def by_group(r: dict, out: Path) -> None:
    dims = [("workload_class", "true class (reporting only)"), ("burst_cv_tercile", "burst_cv tercile"),
            ("n_processes_tercile", "process-count tercile")]
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.4), sharey="row")
    for row, (key, _, title) in enumerate(SETS):
        for ax, (dim, dtitle) in zip(axes[row], dims):
            groups = r[key]["breakdowns_exploratory"][dim]
            names = list(groups)
            x = np.arange(len(names))
            for i, (comp, color) in enumerate((("p4_selected vs p3_frozen", BLUE), ("oracle_fixed vs RR1", REF))):
                comp_key = comp if comp in groups[names[0]] else next(k for k in groups[names[0]] if k.startswith("oracle_fixed"))
                v = np.array([groups[g][comp_key]["geo_rel_change"] for g in names]) * 100
                lo = np.array([groups[g][comp_key]["geo_rel_change_ci95"][0] for g in names]) * 100
                hi = np.array([groups[g][comp_key]["geo_rel_change_ci95"][1] for g in names]) * 100
                ax.errorbar(x + (i - 0.5) * 0.25, v, yerr=[v - lo, hi - v], fmt="o", color=color, ecolor=color,
                            elinewidth=2, capsize=3, markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5,
                            label=comp_key + (" (not deployable)" if comp_key.startswith("oracle") else ""))
            ax.axhline(0, color=INK2, linewidth=1)
            ax.set_xticks(x, [f"{g}\n(n={groups[g]['n']})" for g in names], fontsize=7.5)
            ax.set_title(f"{title}: {dtitle}", fontsize=9.5)
        axes[row][0].set_ylabel("geo change (%), 95% CI")
    axes[0][0].legend(loc="lower left", fontsize=8)
    fig.suptitle("Exploratory breakdowns (no multiplicity correction, 2,000 resamples)", fontweight="bold")
    save(fig, out, "by_group.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", default=str(ROOT / "reports" / "phase4"))
    args = ap.parse_args()
    reports = Path(args.reports)
    r = json.loads((reports / "results.json").read_text(encoding="utf-8"))
    diag = json.loads((reports / "estimator_diagnostic.json").read_text(encoding="utf-8"))
    longs = {t: pd.read_csv(reports / f"per_workload_{t}.csv") for t in ("test", "fresh")}
    out = reports / "figures"
    out.mkdir(parents=True, exist_ok=True)
    primary_family(r, out)
    methods_vs_rr1(r, out)
    paired_differences(longs, out)
    tradeoff(r, out)
    quantum_changes(r, out)
    estimator_quality(diag, out)
    by_group(r, out)


if __name__ == "__main__":
    main()
