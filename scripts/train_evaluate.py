"""Train, tune, and evaluate the Phase 2 classifiers; persist models and results.

Usage:  python scripts/train_evaluate.py [--config configs/phase2.json]
                                        [--artifacts artifacts] [--reports reports/phase2]

Reads   <artifacts>/dataset/{features,processes,workloads}.csv, dataset_meta.json
Writes  <artifacts>/models/<model>.joblib + .json   (not committed)
        <artifacts>/predictions_test.csv             (not committed)
        <reports>/metrics.json, summary.md           (committed, small)
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from schedx.features import PRE, ONLINE, RETRO  # noqa: E402
from schedx.ml import GRIDS, grouped_split, library_versions, run_experiment, save_model, split_summary  # noqa: E402
from schedx.policy_eval import choose_policies, evaluate_choices, regret, score_workloads  # noqa: E402
from schedx.workload import CLASSES, dataset_version, processes_from_frame  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "phase2.json"))
    ap.add_argument("--artifacts", default=str(ROOT / "artifacts"))
    ap.add_argument("--reports", default=str(ROOT / "reports" / "phase2"))
    args = ap.parse_args()
    artifacts = Path(args.artifacts)
    DATA, MODELS, REPORTS = artifacts / "dataset", artifacts / "models", Path(args.reports)
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    ds_meta = json.loads((DATA / "dataset_meta.json").read_text(encoding="utf-8"))
    if ds_meta["dataset_version"] != dataset_version(cfg):
        sys.exit("dataset was generated from a different config; rerun scripts/generate_dataset.py")

    features = pd.read_csv(DATA / "features.csv")
    s = cfg["split"]
    split = grouped_split(features, s["seed"], s["train"], s["validation"], s["test"])
    seed = cfg["model_seed"]

    # Main (deployable) experiment: A + B features only.
    main_res, models, preds = run_experiment(features, split, seed, categories=(PRE, ONLINE))
    # Hindsight experiment: adds retrospective C features. NOT deployable.
    retro_res, _, _ = run_experiment(
        features, split, seed, categories=(PRE, ONLINE, RETRO), grids={"random_forest": GRIDS["random_forest"]}
    )

    common = {
        "dataset_version": ds_meta["dataset_version"],
        "dataset_config": ds_meta["config"],
        "split": {**s, "grouped_by": "workload_id", "stratified_by": "workload_class",
                  "summary": split_summary(features, split)},
        "model_seed": seed,
        "trained_on": "train+validation (hyperparameters chosen on validation)",
        "library_versions": library_versions(),
    }
    for name, model in models.items():
        r = main_res["models"][name]
        save_model(model, {"name": name, "model_type": r["estimator"], "params": r["params"],
                           "features": main_res["features"], "classes": list(CLASSES),
                           "test_metrics": r["test"], "fit_seconds": r["fit_seconds"],
                           "inference_us_per_row": r["inference_us_per_row"], **common}, MODELS, name)
    preds.to_csv(artifacts / "predictions_test.csv", index=False)

    # Offline policy-value experiment (CPU-only burst-trace projection).
    traces = processes_from_frame(pd.read_csv(DATA / "processes.csv"))
    labels = pd.read_csv(DATA / "workloads.csv").set_index("workload_id")["workload_class"]
    reg = regret(score_workloads(traces, cfg["policy"]["quanta"]))
    fit_ids = split.index[split != "test"]
    test_ids = split.index[split == "test"]
    fixed, learned_map = choose_policies(reg.loc[fit_ids], labels)
    class_map = cfg["policy"]["class_policy_map"] or learned_map
    selected = main_res["selected_model"]
    policy = {
        "projection": "CPU-only burst-trace projection (open loop, no I/O blocking simulated)",
        "objective": "mean job turnaround, as regret vs best candidate per workload",
        "candidates": list(reg.columns),
        "fixed_baseline": fixed,
        "class_policy_map": class_map,
        "map_source": "config" if cfg["policy"]["class_policy_map"] else "learned on train+validation",
        "prediction_model": selected,
        "test": evaluate_choices(
            reg.loc[test_ids], labels, preds[preds.model == selected], fixed, class_map, seed=seed
        ),
    }

    REPORTS.mkdir(parents=True, exist_ok=True)
    report = {**common, "main_experiment": main_res, "retrospective_experiment": retro_res, "policy_experiment": policy}
    (REPORTS / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (REPORTS / "summary.md").write_text(render_summary(report), encoding="utf-8")
    print(f"wrote {REPORTS}")


def render_summary(r: dict) -> str:
    """Markdown tables generated from the actual results."""
    m = r["main_experiment"]
    lines = [
        "# Phase 2 results (generated by scripts/train_evaluate.py)",
        "",
        f"Dataset version `{r['dataset_version']}`; libraries {r['library_versions']}.",
        "Offline classification on synthetic workloads. No adaptive scheduling and no OS-level measurement.",
        "",
        "## Split (grouped by workload, stratified by class)",
        "",
        "| split | workloads | rows | " + " | ".join(CLASSES) + " |",
        "|---|---|---|" + "---|" * len(CLASSES),
    ]
    for name, sp in r["split"]["summary"].items():
        lines.append(f"| {name} | {sp['workloads']} | {sp['rows']} | "
                     + " | ".join(str(sp["class_counts"][c]) for c in CLASSES) + " |")
    lines += [
        "",
        "## Main experiment — deployable features (A + B), test split",
        "",
        "| model | params | val macro-F1 | test acc | macro P | macro R | macro F1 | acc k=1/2/3 | fit s | infer µs/row |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, x in m["models"].items():
        t = x["test"]
        hz = "/".join(f"{v:.3f}" for v in x["test_accuracy_by_horizon"].values())
        lines.append(
            f"| {name} | `{x['params']}` | {x['validation_macro_f1']:.3f} | {t['accuracy']:.3f} | "
            f"{t['macro_precision']:.3f} | {t['macro_recall']:.3f} | {t['macro_f1']:.3f} | {hz} | "
            f"{x['fit_seconds']:.2f} | {x['inference_us_per_row']:.1f} |"
        )
    lines += ["", f"Selected for downstream use (by validation macro-F1): **{m['selected_model']}**.", "",
              "### Per-class F1 (test)", "", "| model | " + " | ".join(CLASSES) + " |", "|---|" + "---|" * len(CLASSES)]
    for name, x in m["models"].items():
        lines.append(f"| {name} | " + " | ".join(f"{x['test']['per_class'][c]['f1']:.3f}" for c in CLASSES) + " |")
    lines += ["", "### Confusion matrices (rows = true, columns = predicted: " + ", ".join(CLASSES) + ")", ""]
    for name, x in m["models"].items():
        lines.append(f"- {name}: `{x['test']['confusion_matrix']['matrix']}`")
    rr = r["retrospective_experiment"]["models"]["random_forest"]
    lines += [
        "",
        "## Hindsight experiment — A + B + C features (NOT deployable)",
        "",
        f"Random Forest with lifetime totals added: test accuracy {rr['test']['accuracy']:.3f}, "
        f"macro-F1 {rr['test']['macro_f1']:.3f}. Retrospective features are unknown to a real scheduler.",
    ]
    p = r["policy_experiment"]
    t = p["test"]
    lines += [
        "",
        "## Offline policy-value experiment (CPU-only projection; not adaptive scheduling)",
        "",
        f"Candidates {p['candidates']}; fixed baseline **{p['fixed_baseline']}**; "
        f"class map {p['class_policy_map']} ({p['map_source']}); predictions from {p['prediction_model']}.",
        "Mean regret on test workloads (1.000 = per-workload best candidate):",
        "",
        "| rule | mean regret | Δ vs fixed | paired bootstrap 95% CI of Δ | workloads where choice ≠ fixed |",
        "|---|---|---|---|---|",
        f"| fixed baseline ({p['fixed_baseline']}) | {t['fixed_baseline']['mean_regret']:.4f} | — | — | — |",
    ]

    def row(label: str, v: dict) -> str:
        lo, hi = v["delta_ci95"]
        return (f"| {label} | {v['mean_regret']:.4f} | {v['delta_vs_fixed']:+.4f} | "
                f"[{lo:+.4f}, {hi:+.4f}] | {v['differs_from_fixed']:.0%} |")

    for k, v in t["predicted_class_map_by_horizon"].items():
        lines.append(row(f"predicted-class map, k={k}", v))
    lines += [
        row("true-class map", t["true_class_map"]),
        "| per-workload oracle | 1.0000 | — | — | — |",
        "",
        "Δ < 0 means lower regret than the fixed baseline. A CI that includes 0 means no reliable difference.",
        "",
        "These numbers come from an open-loop CPU-only projection of the traces. They indicate whether the "
        "class signal is worth testing in Phase 3; they do not show that ML improves real scheduling.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
