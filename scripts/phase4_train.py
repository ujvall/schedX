"""Phase 4, stage 1: fit and select the improved-observation controller, then freeze.

Uses ONLY training and validation workloads; test traces and the fresh
confirmation set are never loaded.

Usage:  python scripts/phase4_train.py [--config configs/phase4.json] [--phase3-config configs/phase3.json]
                                       [--phase2-config configs/phase2.json] [--artifacts artifacts]
                                       [--phase3-reports reports/phase3] [--reports reports/phase4]

Writes  <artifacts>/phase4/controllers.joblib        (not committed)
        <reports>/frozen_controller.json, validation_table.csv, estimator_diagnostic.json
"""

import argparse
import json
import sys
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from schedx import phase3, phase4  # noqa: E402
from schedx.ml import library_versions  # noqa: E402


def coefficients(ctrl) -> dict:
    """Standardized logistic coefficients per predicted quantum (interpretability only)."""
    clf = ctrl.model[-1]
    names = [str(n) for n in ctrl.model[:-1].get_feature_names_out()]
    if not hasattr(clf, "coef_"):
        return {}
    rows = clf.coef_ if len(clf.classes_) > 2 else clf.coef_[:1]
    labels = [f"RR{c}" for c in clf.classes_] if len(clf.classes_) > 2 else [f"RR{clf.classes_[1]} vs RR{clf.classes_[0]}"]
    return {lab: dict(sorted(zip(names, map(float, r)), key=lambda t: -abs(t[1]))) for lab, r in zip(labels, rows)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "phase4.json"))
    ap.add_argument("--phase3-config", default=str(ROOT / "configs" / "phase3.json"))
    ap.add_argument("--phase2-config", default=str(ROOT / "configs" / "phase2.json"))
    ap.add_argument("--artifacts", default=str(ROOT / "artifacts"))
    ap.add_argument("--phase3-reports", default=str(ROOT / "reports" / "phase3"))
    ap.add_argument("--reports", default=str(ROOT / "reports" / "phase4"))
    args = ap.parse_args()
    load = lambda p: json.loads(Path(p).read_text(encoding="utf-8"))
    cfg4, cfg3, cfg2 = load(args.config), load(args.phase3_config), load(args.phase2_config)
    artifacts, reports = Path(args.artifacts), Path(args.reports)

    _, split, meta = phase3.load_split(artifacts, cfg2)
    p3_frozen = load(Path(args.phase3_reports) / "frozen_controller.json")
    p3_bundle = joblib.load(artifacts / "phase3" / "controllers.joblib")
    p3_hash = phase3.config_hash(cfg3, meta["dataset_version"])
    if not (p3_hash == p3_frozen["config_hash"] == p3_bundle["config_hash"]):
        sys.exit("frozen Phase 3 controller does not match its config/dataset")

    train_ids = list(split.index[split == "train"])
    val_ids = list(split.index[split == "validation"])
    test_ids = set(split.index[split == "test"])
    traces = phase3.load_traces(artifacts, train_ids + val_ids)
    assert not test_ids & set(traces), "test workloads must not be loaded in the fit stage"

    data = phase4.counterfactual_data(traces, cfg4)
    if data["q0"] != p3_bundle["q0"]:
        sys.exit(f"best fixed quantum changed (RR{data['q0']} vs Phase 3 RR{p3_bundle['q0']}); investigate first")
    p3_online = p3_bundle["controllers"]["best_online"]
    ablation = {"family": p3_online.family, "params": p3_online.params, "after_completions": p3_online.after_completions}
    table, ctrls = phase4.search(data, train_ids, val_ids, cfg4, ablation)
    diag = phase4.estimator_diagnostic(traces, data, train_ids, cfg4["decision_after_completions"])
    chash = phase4.config_hash(cfg4, meta["dataset_version"], p3_hash)

    (artifacts / "phase4").mkdir(parents=True, exist_ok=True)
    joblib.dump({"config_hash": chash, "q0": data["q0"],
                 "controllers": {k: ctrls[k] for k in ("p4_selected", "p4_ablation")}},
                artifacts / "phase4" / "controllers.joblib")
    p3_val = p3_frozen["controllers"]["selected"]
    frozen = {
        "frozen": True,
        "config_hash": chash,
        "phase3_config_hash": p3_hash,
        "dataset_version": meta["dataset_version"],
        "phase4_config": cfg4,
        "split": {"train_workloads": len(train_ids), "validation_workloads": len(val_ids),
                  "test_workloads": len(test_ids), "test_workloads_loaded_in_fit_stage": False,
                  "fresh_confirmation_set_generated_in_fit_stage": False},
        "default_quantum_q0": data["q0"],
        "fixed_mean_log_regret_train_val": data["default_table"],
        "validation_criterion": "mean over validation workloads of log(T_chosen / T_q0), lower is better",
        "controllers": {k: phase3.describe(ctrls[k]) for k in ("p4_selected", "p4_ablation")},
        "validation_scores": {
            "p4_selected": float(table.loc[table.selected, "val_mean_log_ratio"].item()),
            "p4_ablation": ctrls["p4_ablation_val_mean_log_ratio"],
            "p3_frozen_from_phase3_validation_table": float(
                __import__("pandas").read_csv(Path(args.phase3_reports) / "validation_table.csv")
                .query("selected")["val_mean_log_ratio"].item()),
        },
        "p3_frozen_controller": p3_val["label"],
        "logistic_coefficients": {k: coefficients(ctrls[k]) for k in ("p4_selected", "p4_ablation")
                                  if ctrls[k].family == "logistic_classifier"},
        "library_versions": library_versions(),
    }
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "frozen_controller.json").write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    (reports / "estimator_diagnostic.json").write_text(json.dumps(diag, indent=2), encoding="utf-8")
    table.to_csv(reports / "validation_table.csv", index=False)

    print(f"q0 = RR{data['q0']}; config hash {chash}")
    print(table.sort_values("val_mean_log_ratio").head(6).to_string(index=False))
    print("validation scores:", json.dumps(frozen["validation_scores"]))
    for k in ("p4_selected", "p4_ablation"):
        print(f"{k}: {ctrls[k].describe()}")


if __name__ == "__main__":
    main()
