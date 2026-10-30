"""Column-width figures for the IEEE-format case study, from committed Phase 5 results.

Usage (from the repository root):  python docs/case_study_build/report_figures.py

Reads reports/phase5/results.json and configs/phase5.json (committed evidence;
nothing is recomputed or edited) and writes docs/case_study_build/fig_effects.png
and fig_response.png, sized for one 3.5-inch IEEE column. Exact values are given
in the report's Table III, so the effect plots carry no value labels.
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
XLAB = f"Processes per workload (n = {N.pop():,} workloads each)" if len(N) == 1 else "Processes per workload"
BLUE, ORANGE, AQUA, GRAY, INK2, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
    "axes.linewidth": 0.6, "figure.facecolor": "white", "axes.facecolor": "white",
})
x = np.arange(3)


def dots(ax, xs, stats, color, label):
    v = np.array([s["geo_rel_change"] for s in stats]) * 100
    lo = np.array([s["geo_rel_change_ci95"][0] for s in stats]) * 100
    hi = np.array([s["geo_rel_change_ci95"][1] for s in stats]) * 100
    ax.errorbar(xs, v, yerr=[v - lo, hi - v], fmt="o", color=color, ecolor=color, elinewidth=1.4, capsize=2.5,
                markersize=4.5, markeredgecolor="white", markeredgewidth=0.8, label=label)


# Fig. 1: (a) primary effects against the practical threshold; (b) oracle headroom.
fig, (a, b) = plt.subplots(2, 1, figsize=(3.5, 3.4), sharex=True)
a.axhspan(-SESOI, SESOI, color="#eef2f6", zorder=0)
a.axhline(-SESOI, color=INK2, linestyle="--", linewidth=0.8, label=f"±{SESOI:.0f}% practical threshold")
a.axhline(SESOI, color=INK2, linestyle="--", linewidth=0.8)
a.axhline(0, color=INK2, linewidth=0.6)
dots(a, x - 0.1, [R["primary_family"][f"H1_{o}"] for o in OPS], BLUE, "Phase 3 controller")
dots(a, x + 0.1, [R["primary_family"][f"H2_{o}"] for o in OPS], ORANGE, "Phase 4 controller")
a.set_xlim(-0.5, 2.5)
a.set_ylim(-1.2, 1.75)
a.set_ylabel("Change vs RR1 (%, 95% CI)")
a.set_title("(a) Frozen controllers vs RR1")
a.legend(loc="upper center", ncols=2, columnspacing=1.0, handletextpad=0.4, borderaxespad=0.1)

b.axhline(0, color=INK2, linewidth=0.6)
dots(b, x - 0.1, [R["operating_points"][o]["oracle_fixed_vs_RR1"] for o in OPS], GRAY, "Oracle (not deployable)")
dots(b, x + 0.1, [R["primary_family"][f"H1_{o}"] for o in OPS], BLUE, "Phase 3 controller")
b.set_xticks(x, [str(s) for s in SIZES])
b.set_xlabel(XLAB)
b.set_ylabel("Change vs RR1 (%, 95% CI)")
b.set_title("(b) Phase 3 controller vs per-workload oracle")
b.legend(loc="center left", bbox_to_anchor=(0.0, 0.62), handletextpad=0.4)
fig.tight_layout(pad=0.3, h_pad=0.8)
fig.savefig(OUT / "fig_effects.png", dpi=300)
plt.close(fig)

# Fig. 2: response-time cost relative to RR1 (ticks).
fig, ax = plt.subplots(figsize=(3.5, 2.2))
for i, (m, color, label) in enumerate((("p3_frozen", BLUE, "Phase 3 controller"),
                                       ("p4_selected", ORANGE, "Phase 4 controller"),
                                       ("RR2", AQUA, "Fixed RR2"))):
    st = [R["operating_points"][o]["metrics"]["avg_response"][m] for o in OPS]
    v = np.array([s["vs_RR1_mean_diff"] for s in st])
    lo = np.array([s["vs_RR1_mean_diff_ci95"][0] for s in st])
    hi = np.array([s["vs_RR1_mean_diff_ci95"][1] for s in st])
    bars = ax.bar(x + (i - 1) * 0.28, v, 0.26, color=color, label=label, yerr=[v - lo, hi - v],
                  ecolor=INK2, error_kw={"elinewidth": 0.7, "capsize": 1.5})
    ax.bar_label(bars, fmt="%+.1f", padding=1.5, fontsize=6.5, color=INK2)
ax.axhline(0, color=INK2, linewidth=0.6)
ax.set_xticks(x, [str(s) for s in SIZES])
ax.set_xlabel(XLAB)
ax.set_ylabel("Mean response minus RR1\n(ticks, 95% CI)")
ax.set_ylim(0, 42)
ax.legend(loc="upper left", handletextpad=0.4)
fig.tight_layout(pad=0.3)
fig.savefig(OUT / "fig_response.png", dpi=300)
plt.close(fig)
print("wrote", OUT / "fig_effects.png", OUT / "fig_response.png")
