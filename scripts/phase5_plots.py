"""Phase 5 figures from saved results (nothing hardcoded).

Usage:  python scripts/phase5_plots.py [--reports reports/phase5]

Reads <reports>/results.json, raw_workload_results.csv, workload_manifest.csv;
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
OPS = ("small", "medium", "large")
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE, INK, INK2, GRID, REF = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#8a8984"
COLOR = {"p3_frozen": BLUE, "p4_selected": ORANGE, "RR2": AQUA, "oracle_fixed": REF}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False,
})


def save(fig, out, name):
    fig.tight_layout()
    fig.savefig(out / name, dpi=130)
    plt.close(fig)
    print("wrote", out / name)


def op_labels(r):
    return [f"{o}\n{r['operating_points'][o]['n_processes_per_workload']} processes\nn = {r['operating_points'][o]['n_workloads']}"
            for o in OPS]


def errdots(ax, x, stats, color, label, key="geo_rel_change", ci="geo_rel_change_ci95", marker="o"):
    v = np.array([s[key] for s in stats]) * 100
    lo = np.array([s[ci][0] for s in stats]) * 100
    hi = np.array([s[ci][1] for s in stats]) * 100
    ax.errorbar(x, v, yerr=[v - lo, hi - v], fmt=marker, color=color, ecolor=color, elinewidth=2, capsize=4,
                markersize=8, markeredgecolor=SURFACE, markeredgewidth=2, label=label)
    for xi, vi in zip(x, v):
        ax.annotate(f"{vi:+.2f}%", (xi, vi), textcoords="offset points", xytext=(10, 0), va="center", fontsize=8, color=INK2)


def primary_effects(r, out):
    sesoi = 100 * r["_cfg"]["statistics"]["sesoi_relative"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [3, 2]})
    x = np.arange(3)
    for ax in (a1,):
        ax.axhspan(-sesoi, sesoi, color=GRID, alpha=0.6, zorder=0, label=f"±{sesoi:.0f}% practical threshold (SESOI)")
        for i, (h, m) in enumerate((("H1", "p3_frozen"), ("H2", "p4_selected"))):
            stats_ = [r["primary_family"][f"{h}_{o}"] for o in OPS]
            errdots(ax, x + (i - 0.5) * 0.25, stats_, COLOR[m], f"{h}: {m} vs RR1")
            for xi, s in zip(x + (i - 0.5) * 0.25, stats_):
                ax.annotate(f"Holm p={s['holm_p']:.2g}", (xi, s["geo_rel_change_ci95"][0] * 100), textcoords="offset points",
                            xytext=(0, -12), ha="center", fontsize=7, color=INK2)
        ax.axhline(0, color=INK2, linewidth=1)
        ax.set_xticks(x, op_labels(r), fontsize=8.5)
        ax.set_ylabel("geometric-mean change in turnaround vs RR1 (%), 95% CI")
        ax.set_title("Primary family (negative = better)")
        ax.legend(loc="lower left", fontsize=8)
    stats_ = [r["operating_points"][o]["oracle_fixed_vs_RR1"] for o in OPS]
    errdots(a2, x, stats_, REF, "per-workload oracle (not deployable)")
    for i, (h, m) in enumerate((("H1", "p3_frozen"), ("H2", "p4_selected"))):
        errdots(a2, x + (i + 1) * 0.2, [r["primary_family"][f"{h}_{o}"] for o in OPS], COLOR[m], m)
    a2.axhline(0, color=INK2, linewidth=1)
    a2.set_xticks(x + 0.2, [o for o in OPS])
    a2.set_ylabel("geometric-mean change vs RR1 (%)")
    a2.set_title("Against the attainable headroom")
    a2.legend(loc="center right", fontsize=8)
    fig.suptitle("Phase 5: frozen controllers vs best fixed quantum at three operating points", fontweight="bold")
    save(fig, out, "primary_effects.png")


def estimands(r, out):
    fig, ax = plt.subplots(figsize=(10, 4.4))
    x = np.arange(3)
    for i, (h, m) in enumerate((("H1", "p3_frozen"), ("H2", "p4_selected"))):
        st = [r["primary_family"][f"{h}_{o}"] for o in OPS]
        errdots(ax, x + (i * 2 - 1.5) * 0.12, st, COLOR[m], f"{m}: geometric mean of per-workload ratios")
        errdots(ax, x + (i * 2 - 0.5) * 0.12, st, COLOR[m], f"{m}: change in aggregate mean turnaround",
                key="change_in_mean_turnaround", ci="change_in_mean_turnaround_ci95", marker="s")
    ax.axhline(0, color=INK2, linewidth=1)
    ax.set_xticks(x, op_labels(r), fontsize=8.5)
    ax.set_ylabel("change vs RR1 (%), 95% CI")
    ax.set_title("Two estimands: per-workload geometric mean (circles) vs aggregate mean (squares)")
    ax.legend(fontsize=7.5, loc="lower left", ncols=2)
    save(fig, out, "estimands.png")


def response_cost(r, out):
    fig, ax = plt.subplots(figsize=(10, 4.4))
    x = np.arange(3)
    for i, m in enumerate(("p3_frozen", "p4_selected", "RR2")):
        st = [r["operating_points"][o]["metrics"]["avg_response"][m] for o in OPS]
        v = np.array([s["vs_RR1_mean_diff"] for s in st])
        lo = np.array([s["vs_RR1_mean_diff_ci95"][0] for s in st])
        hi = np.array([s["vs_RR1_mean_diff_ci95"][1] for s in st])
        bars = ax.bar(x + (i - 1) * 0.26, v, 0.24, color=COLOR[m], label=m, yerr=[v - lo, hi - v], ecolor=INK2, capsize=3)
        ax.bar_label(bars, fmt="%+.1f", padding=3, fontsize=8, color=INK2)
    ax.axhline(0, color=INK2, linewidth=1)
    ax.set_xticks(x, op_labels(r), fontsize=8.5)
    ax.set_ylabel("mean response time minus RR1 (ticks), 95% CI")
    ax.set_title("Secondary outcome: response-time cost relative to RR1 (positive = slower first dispatch)")
    ax.legend(fontsize=8)
    save(fig, out, "response_cost.png")


def behaviour(r, out):
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(3)
    for i, (m, k, lab) in enumerate((("p3_frozen", "share_quantum_changed", "p3_frozen: quantum changed"),
                                     ("p3_frozen", "share_fallback", "p3_frozen: no decision point (fallback)"),
                                     ("p4_selected", "share_quantum_changed", "p4_selected: quantum changed"),
                                     ("p4_selected", "share_fallback", "p4_selected: fallback"))):
        v = [100 * r["operating_points"][o]["controller_behaviour"][m][k] for o in OPS]
        bars = ax.bar(x + (i - 1.5) * 0.2, v, 0.18, color=COLOR[m], alpha=1.0 if "changed" in k else 0.45, label=lab)
        ax.bar_label(bars, fmt="%.0f%%", padding=2, fontsize=7.5, color=INK2)
    ax.set_xticks(x, op_labels(r), fontsize=8.5)
    ax.set_ylabel("share of workloads (%)")
    ax.set_title("Controller behaviour: how often the quantum was changed, and fallbacks")
    ax.legend(fontsize=8, ncols=2, loc="upper left")
    ax.set_ylim(0, 60)
    save(fig, out, "controller_behaviour.png")


def distributions(raw, man, out):
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8), sharey=True)
    T = raw.pivot(index="workload_id", columns="method", values="avg_turnaround")
    for ax, o in zip(axes, OPS):
        ids = man.loc[man.operating_point == o, "workload_id"]
        for m, style in (("p4_selected", {"alpha": 0.45}), ("p3_frozen", {"histtype": "step", "linewidth": 2})):
            d = 100 * (T.loc[ids, m] / T.loc[ids, "RR1"] - 1)
            nz = d[d != 0]
            # Clip into the edge bins so no workload is silently dropped.
            ax.hist(np.clip(nz, -24.99, 24.99), bins=np.linspace(-25, 25, 51), color=COLOR[m],
                    label=f"{m}: {len(nz)} of {len(d)} changed", **style)
        ax.axvline(0, color=INK2, linewidth=1)
        ax.set_title(f"{o} (n = {len(ids)})")
        ax.set_xlabel("per-workload change vs RR1 (%); changed only; |x| > 25% in edge bins", fontsize=8.5)
        ax.legend(fontsize=7.5)
    axes[0].set_ylabel("workloads")
    save(fig, out, "per_workload_changes.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", default=str(ROOT / "reports" / "phase5"))
    ap.add_argument("--config", default=str(ROOT / "configs" / "phase5.json"))
    args = ap.parse_args()
    rep = Path(args.reports)
    r = json.loads((rep / "results.json").read_text(encoding="utf-8"))
    r["_cfg"] = json.loads(Path(args.config).read_text(encoding="utf-8"))
    raw = pd.read_csv(rep / "raw_workload_results.csv")
    man = pd.read_csv(rep / "workload_manifest.csv")
    out = rep / "figures"
    out.mkdir(parents=True, exist_ok=True)
    primary_effects(r, out)
    estimands(r, out)
    response_cost(r, out)
    behaviour(r, out)
    distributions(raw, man, out)


if __name__ == "__main__":
    main()
