"""Regenerate Phase 2 figures from the actual dataset and evaluation results.

Usage:  python scripts/plot_results.py [--artifacts artifacts] [--reports reports/phase2]

Reads <artifacts>/dataset/*.csv and <reports>/metrics.json (nothing is
hardcoded); writes PNGs to <reports>/figures/.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from schedx.workload import CLASSES  # noqa: E402

# Validated categorical palette (fixed order); aqua is < 3:1 on the surface,
# so every chart using it carries direct value labels.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
CLASS_COLOR = dict(zip(CLASSES, SERIES))

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


def class_distribution(metrics: dict, out: Path) -> None:
    summary = metrics["split"]["summary"]
    splits = list(summary)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    x = np.arange(len(CLASSES))
    w = 0.26
    for i, s in enumerate(splits):
        counts = [summary[s]["class_counts"][c] for c in CLASSES]
        bars = ax.bar(x + (i - 1) * w, counts, w - 0.02, color=SERIES[i], label=f"{s} ({summary[s]['workloads']})")
        ax.bar_label(bars, padding=2, fontsize=8, color=INK2)
    ax.set_xticks(x, CLASSES)
    ax.set_ylabel("workloads")
    ax.set_title("Workload class distribution by split (grouped, stratified)")
    ax.legend(ncols=3, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    save(fig, out, "class_distribution.png")


def feature_distributions(features: pd.DataFrame, out: Path) -> None:
    cols = [
        ("obs_cpu_mean", "mean observed CPU burst (ticks)"),
        ("obs_proc_cpu_mean_std", "std of per-process mean CPU burst"),
        ("obs_frac_io_dominant", "share of processes I/O-dominant so far"),
        ("obs_io_cpu_ratio", "observed I/O : CPU ratio"),
        ("n_processes", "process count (pre-execution)"),
        ("arrival_rate", "arrival rate (pre-execution)"),
    ]
    k2 = features[features.horizon_k == 2]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6))
    for ax, (col, title) in zip(axes.flat, cols):
        data = [k2.loc[k2.workload_class == c, col].dropna() for c in CLASSES]
        bp = ax.boxplot(data, patch_artist=True, widths=0.55, showfliers=False,
                        medianprops={"color": INK, "linewidth": 1.5})
        for patch, c in zip(bp["boxes"], CLASSES):
            patch.set_facecolor(CLASS_COLOR[c])
            patch.set_alpha(0.85)
            patch.set_edgecolor(SURFACE)
        ax.set_xticks(range(1, 4), CLASSES, fontsize=8)
        ax.set_title(title, fontsize=9.5)
        if col == "obs_io_cpu_ratio":
            ax.set_yscale("log")
    fig.suptitle("Feature distributions by true class at horizon k = 2 (all workloads; outliers hidden)",
                 fontweight="bold")
    save(fig, out, "feature_distributions.png")


def model_comparison(metrics: dict, out: Path) -> None:
    models = metrics["main_experiment"]["models"]
    names = list(models)
    keys = [("accuracy", "accuracy"), ("macro_precision", "macro precision"),
            ("macro_recall", "macro recall"), ("macro_f1", "macro F1")]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [3, 2]})
    x = np.arange(len(keys))
    w = 0.26
    for i, n in enumerate(names):
        vals = [models[n]["test"][k] for k, _ in keys]
        bars = ax.bar(x + (i - 1) * w, vals, w - 0.02, color=SERIES[i], label=n)
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=7.5, color=INK2)
    ax.set_xticks(x, [lbl for _, lbl in keys])
    ax.set_ylim(0, 1.05)  # bars encode length: the baseline must be zero
    ax.set_title("Test-set performance, deployable features (A+B)")
    ax.legend(ncols=3, loc="upper center", bbox_to_anchor=(0.5, -0.08))

    for i, n in enumerate(names):
        hz = models[n]["test_accuracy_by_horizon"]
        ks = [int(k) for k in hz]
        vs = list(hz.values())
        ax2.plot(ks, vs, color=SERIES[i], linewidth=2, marker="o", markersize=7,
                 markeredgecolor=SURFACE, markeredgewidth=2, label=n)
        ax2.annotate(f"{vs[-1]:.3f}", (ks[-1], vs[-1]), textcoords="offset points", xytext=(8, 0),
                     va="center", fontsize=8, color=INK2)
    ax2.set_xticks(ks)
    ax2.set_xlim(min(ks) - 0.3, max(ks) + 0.6)
    ax2.set_xlabel("horizon k (CPU bursts observed per process)")
    ax2.set_ylabel("test accuracy")
    ax2.set_title("Accuracy vs. observation horizon")
    ax2.legend(loc="lower right")
    save(fig, out, "model_comparison.png")


def confusion_matrices(metrics: dict, out: Path) -> None:
    models = metrics["main_experiment"]["models"]
    fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models), 3.9))
    for ax, (n, r) in zip(np.atleast_1d(axes), models.items()):
        cm = np.array(r["test"]["confusion_matrix"]["matrix"])
        labels = r["test"]["confusion_matrix"]["labels"]
        ax.imshow(cm, cmap="Blues", vmin=0)
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, str(v), ha="center", va="center", fontsize=10,
                    color="white" if v > cm.max() * 0.6 else INK)
        ax.set_xticks(range(len(labels)), labels, fontsize=8)
        ax.set_yticks(range(len(labels)), labels, fontsize=8)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.grid(False)
        ax.set_title(f"{n}\nacc {r['test']['accuracy']:.3f}", fontsize=10)
    fig.suptitle("Confusion matrices, test split (rows = true class)", fontweight="bold")
    save(fig, out, "confusion_matrices.png")


def feature_importance(metrics: dict, out: Path, top: int = 12) -> None:
    models = {n: r for n, r in metrics["main_experiment"]["models"].items() if r["feature_importance"]}
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 4.6))
    for ax, (n, r) in zip(np.atleast_1d(axes), models.items()):
        items = list(r["feature_importance"].items())[:top][::-1]
        names = [k.replace("missingindicator_", "missing: ") for k, _ in items]
        vals = [v for _, v in items]
        bars = ax.barh(names, vals, color=SERIES[0], height=0.7)
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=7.5, color=INK2)
        ax.set_title(f"{n}: top {top} impurity importances")
        ax.tick_params(axis="y", labelsize=8)
        ax.set_xlim(0, max(vals) * 1.18)
    fig.text(0.5, 0.01, "Impurity-based importance on train+validation. It reflects how the fitted model splits; "
             "it does not establish causation and is biased toward high-cardinality, correlated features.",
             ha="center", fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out / "feature_importance.png", dpi=130)
    plt.close(fig)
    print("wrote", out / "feature_importance.png")


def policy_regret(metrics: dict, out: Path) -> None:
    p = metrics["policy_experiment"]
    t = p["test"]
    rows = [(f"predicted-class map, k={k}", v) for k, v in t["predicted_class_map_by_horizon"].items()]
    rows.append(("true-class map", t["true_class_map"]))
    fig, ax = plt.subplots(figsize=(8, 3.6))
    y = np.arange(len(rows))
    deltas = [v["delta_vs_fixed"] for _, v in rows]
    err = np.array([[d - v["delta_ci95"][0], v["delta_ci95"][1] - d] for d, (_, v) in zip(deltas, rows)]).T
    ax.errorbar(deltas, y, xerr=err, fmt="o", color=SERIES[0], ecolor=SERIES[0], elinewidth=2,
                capsize=4, markersize=8, markeredgecolor=SURFACE, markeredgewidth=2)
    for yi, d in zip(y, deltas):
        ax.annotate(f"{d:+.4f}", (d, yi), textcoords="offset points", xytext=(0, 9), ha="center",
                    fontsize=8, color=INK2)
    ax.axvline(0, color=INK2, linewidth=1)
    ax.set_yticks(y, [r for r, _ in rows])
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.invert_yaxis()
    ax.set_xlabel(f"mean regret difference vs fixed {p['fixed_baseline']} (negative = better), 95% paired bootstrap CI")
    ax.set_title("Offline policy-value experiment\n(CPU-only burst-trace projection; not adaptive scheduling)")
    save(fig, out, "policy_regret.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default=str(ROOT / "artifacts"))
    ap.add_argument("--reports", default=str(ROOT / "reports" / "phase2"))
    args = ap.parse_args()
    reports = Path(args.reports)
    metrics = json.loads((reports / "metrics.json").read_text(encoding="utf-8"))
    features = pd.read_csv(Path(args.artifacts) / "dataset" / "features.csv")
    out = reports / "figures"
    out.mkdir(parents=True, exist_ok=True)
    class_distribution(metrics, out)
    feature_distributions(features, out)
    model_comparison(metrics, out)
    confusion_matrices(metrics, out)
    feature_importance(metrics, out)
    policy_regret(metrics, out)


if __name__ == "__main__":
    main()
