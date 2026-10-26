"""Phase 3, stage 1: fit and select quantum controllers, then freeze them.

Uses ONLY training and validation workloads; test traces are never loaded.

Usage:  python scripts/phase3_train.py [--config configs/phase3.json] [--phase2-config configs/phase2.json]
                                       [--artifacts artifacts] [--reports reports/phase3]

Writes  <artifacts>/phase3/controllers.joblib      (not committed)
        <artifacts>/phase3/fit_counterfactuals.csv (not committed)
        <reports>/frozen_controller.json, validation_table.csv (committed)
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from schedx.ml import library_versions  # noqa: E402
from schedx.phase3 import (  # noqa: E402
    config_hash,
    counterfactual_data,
    describe,
    load_split,
    load_traces,
    search,
    variability_diagnostic,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "phase3.json"))
    ap.add_argument("--phase2-config", default=str(ROOT / "configs" / "phase2.json"))
    ap.add_argument("--artifacts", default=str(ROOT / "artifacts"))
    ap.add_argument("--reports", default=str(ROOT / "reports" / "phase3"))
    args = ap.parse_args()
    cfg3 = json.loads(Path(args.config).read_text(encoding="utf-8"))
    cfg2 = json.loads(Path(args.phase2_config).read_text(encoding="utf-8"))
    artifacts, reports = Path(args.artifacts), Path(args.reports)

    _, split, meta = load_split(artifacts, cfg2)
    train_ids = list(split.index[split == "train"])
    val_ids = list(split.index[split == "validation"])
    test_ids = set(split.index[split == "test"])
    traces = load_traces(artifacts, train_ids + val_ids)
    assert not test_ids & set(traces), "test workloads must not be loaded in the fit stage"

    data = counterfactual_data(traces, cfg3)
    table, controllers = search(data, train_ids, val_ids, cfg3)
    chash = config_hash(cfg3, meta["dataset_version"])

    (artifacts / "phase3").mkdir(parents=True, exist_ok=True)
    joblib.dump({"config_hash": chash, "q0": data["q0"], "controllers": controllers},
                artifacts / "phase3" / "controllers.joblib")
    long = [data["fixed_turnaround"].assign(regime="fixed")]
    long += [d["turnaround"].assign(regime=f"online_m{m}") for m, d in data["online"].items()]
    pd.concat(long).rename_axis("workload_id").to_csv(artifacts / "phase3" / "fit_counterfactuals.csv")

    fallback = {int(m): {s: float(d["t_decision"].loc[ids].isna().mean())
                         for s, ids in (("train", train_ids), ("validation", val_ids))}
                for m, d in data["online"].items()}
    frozen = {
        "frozen": True,
        "config_hash": chash,
        "dataset_version": meta["dataset_version"],
        "phase3_config": cfg3,
        "split": {"train_workloads": len(train_ids), "validation_workloads": len(val_ids),
                  "test_workloads": len(test_ids), "test_workloads_loaded_in_fit_stage": False},
        "default_quantum_q0": data["q0"],
        "fixed_mean_log_regret_train_val": data["default_table"],
        "online_no_decision_share": fallback,
        "validation_criterion": "mean over validation workloads of log(T_chosen / T_q0), lower is better",
        "controllers": {k: describe(c) for k, c in controllers.items()},
        "train_only_diagnostic": variability_diagnostic(
            traces, data, train_ids, controllers["best_online"].after_completions
        ),
        "library_versions": library_versions(),
    }
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "frozen_controller.json").write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    table.to_csv(reports / "validation_table.csv", index=False)

    print(f"q0 = RR{data['q0']}  (train+val mean log regret {data['default_table']})")
    print(table.sort_values("val_mean_log_ratio").head(8).to_string(index=False))
    for k, c in controllers.items():
        print(f"{k}: {c.describe()}")


if __name__ == "__main__":
    main()
