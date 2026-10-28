"""Report-sized figures for the case study, drawn from committed Phase 5 results.

Usage (from the repository root):  python docs/case_study_build/report_figures.py

Reads reports/phase5/results.json and configs/phase5.json (committed evidence;
nothing is recomputed or edited) and writes docs/case_study_build/fig_effects.png
and fig_response.png, sized for a 17 cm text width. Exact values are given in the
report's Table 3, so the plots carry no value labels.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
R = json.loads((ROOT / "reports" / "phase5" / "results.json").read_text(encoding="utf-8"))
SESOI = json.loads((ROOT / "configs" / "phase5.json").read_text(encoding="utf-8"))["statistics"]["sesoi_relative"] * 100
OPS = ("small", "medium", "large")
SIZES = [R["operating_points"][o]["n_processes_per_workload"] for o in OPS]
N = {R["operating_points"][o]["n_workloads"] for o in OPS}
XLAB = f"processes per workload (n = {N.pop():,} workloads each)" if len(N) == 1 else "processes per workload"
BLUE, ORANGE, AQUA, GRAY, INK2, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 10.5, "axes.labelsize": 10, "legend.fontsize": 8.5,
    "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "axes.axisbelow": True, "legend.frameon": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
})
x = np.arange(3)


def dots(ax, xs, stats, color, label):
    v = np.array([s["geo_rel_change"] for s in stats]) * 100
    lo = np.array([s["geo_rel_change_ci95"][0] for s in stats]) * 100
    hi = np.array([s["geo_rel_change_ci95"][1] for s in stats]) * 100
    ax.errorbar(xs, v, yerr=[v - lo, hi - v], fmt="o", color=color, ecolor=color, elinewidth=2, capsize=4,
                markersize=7, markeredgecolor="white", markeredgewidth=1.5, label=label)


# Figure 1: (a) primary effects against the practical threshold; (b) oracle headroom.
fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 3.0), gridspec_kw={"width_ratios": [1.2, 1]})
a.axhspan(-SESOI, SESOI, color="#eef2f6", zorder=0)
a.axhline(-SESOI, color=INK2, linestyle="--", linewidth=1, label=f"±{SESOI:.0f}% practical threshold")
a.axhline(SESOI, color=INK2, linestyle="--", linewidth=1)
a.axhline(0, color=INK2, linewidth=0.8)
dots(a, x - 0.12, [R["primary_family"][f"H1_{o}"] for o in OPS], BLUE, "Phase 3 controller")
dots(a, x + 0.12, [R["primary_family"][f"H2_{o}"] for o in OPS], ORANGE, "Phase 4 controller")
a.set_xticks(x, [str(s) for s in SIZES])
a.set_xlim(-0.5, 2.5)
a.set_ylim(-1.3, 1.3)
a.set_xlabel(XLAB, fontsize=9)
a.set_ylabel("change in turnaround vs RR1\n(%, geometric mean, 95% CI)")
a.set_title("(a) Frozen controllers vs RR1")
a.legend(loc="upper left", ncols=1)

b.axhline(0, color=INK2, linewidth=0.8)
dots(b, x - 0.1, [R["operating_points"][o]["oracle_fixed_vs_RR1"] for o in OPS], GRAY, "oracle (not deployable)")
dots(b, x + 0.1, [R["primary_family"][f"H1_{o}"] for o in OPS], BLUE, "Phase 3 controller")
b.set_xticks(x, [str(s) for s in SIZES])
b.set_xlim(-0.5, 2.5)
b.set_xlabel("processes per workload", fontsize=9)
b.set_title("(b) Against the oracle")
b.legend(loc="center right")
fig.tight_layout()
fig.savefig(OUT / "fig_effects.png", dpi=200)
plt.close(fig)

# Figure 2: response-time cost relative to RR1 (ticks).
fig, ax = plt.subplots(figsize=(7.0, 2.5))
for i, (m, color, label) in enumerate((("p3_frozen", BLUE, "Phase 3 controller"),
                                       ("p4_selected", ORANGE, "Phase 4 controller"),
                                       ("RR2", AQUA, "fixed RR2"))):
    st = [R["operating_points"][o]["metrics"]["avg_response"][m] for o in OPS]
    v = np.array([s["vs_RR1_mean_diff"] for s in st])
    lo = np.array([s["vs_RR1_mean_diff_ci95"][0] for s in st])
    hi = np.array([s["vs_RR1_mean_diff_ci95"][1] for s in st])
    bars = ax.bar(x + (i - 1) * 0.26, v, 0.24, color=color, label=label, yerr=[v - lo, hi - v],
                  ecolor=INK2, capsize=3)
    ax.bar_label(bars, fmt="%+.1f", padding=2, fontsize=8, color=INK2)
ax.axhline(0, color=INK2, linewidth=0.8)
ax.set_xticks(x, [str(s) for s in SIZES])
ax.set_xlabel(XLAB, fontsize=9)
ax.set_ylabel("Δ mean response vs RR1\n(ticks, 95% CI)")
ax.set_ylim(0, 40)
ax.legend(loc="upper left")
fig.tight_layout()
fig.savefig(OUT / "fig_response.png", dpi=200)
plt.close(fig)
print("wrote", OUT / "fig_effects.png", OUT / "fig_response.png")
