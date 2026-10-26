"""Offline policy-value experiment and the end-to-end script pipeline."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from schedx.ml import load_model, predict_workload
from schedx.policy_eval import burst_jobs, candidate_policies, choose_policies, evaluate_choices, regret, score_workloads
from schedx.workload import GeneratedProcess, generate_dataset, generate_workload

ROOT = Path(__file__).resolve().parents[1]


# --- Policy experiment ---------------------------------------------------------------


def test_burst_jobs_release_at_contention_free_times():
    p = GeneratedProcess("P", 2, 1, (3, 4, 1), (5, 6), "cpu_bound")
    jobs = burst_jobs([p])
    assert [(j.pid, j.arrival, j.burst, j.priority) for j in jobs] == [
        ("P.0", 2, 3, 1),
        ("P.1", 10, 4, 1),  # 2 + 3 cpu + 5 io
        ("P.2", 20, 1, 1),  # 10 + 4 cpu + 6 io
    ]


def test_candidates_are_non_clairvoyant():
    assert list(candidate_policies([2, 8])) == ["FCFS", "RR2", "RR8"]


def test_scores_are_valid_schedules_and_regret_is_normalized():
    ws = generate_dataset(8, master_seed=3)
    scores = score_workloads({w.workload_id: w.processes for w in ws}, [2, 4])
    assert list(scores.columns) == ["FCFS", "RR2", "RR4"] and len(scores) == 8
    r = regret(scores)
    assert np.allclose(r.min(axis=1), 1.0) and (r >= 1.0).all().all()


def test_choose_policies_hand_example():
    reg = pd.DataFrame(
        {"A": [1.0, 1.0, 1.5, 1.6], "B": [1.2, 1.3, 1.0, 1.0]}, index=["w1", "w2", "w3", "w4"]
    )
    labels = pd.Series({"w1": "cpu_bound", "w2": "cpu_bound", "w3": "io_bound", "w4": "io_bound", "w9": "mixed"})
    fixed, cmap = choose_policies(reg, labels)
    assert fixed == "B"  # mean A 1.275 > mean B 1.125
    assert cmap == {"cpu_bound": "A", "io_bound": "B"}  # "mixed" absent from fitting rows


def test_evaluate_choices_hand_example():
    reg = pd.DataFrame({"A": [1.0, 1.2], "B": [1.1, 1.0]}, index=["w1", "w2"])
    labels = pd.Series({"w1": "cpu_bound", "w2": "io_bound"})
    cmap = {"cpu_bound": "A", "io_bound": "B"}
    perfect = pd.DataFrame({"workload_id": ["w1", "w2"], "horizon_k": [1, 1], "predicted": ["cpu_bound", "io_bound"]})
    wrong = perfect.assign(horizon_k=2, predicted=["io_bound", "cpu_bound"])
    out = evaluate_choices(reg, labels, pd.concat([perfect, wrong]), fixed="B", class_map=cmap)
    assert out["fixed_baseline"]["mean_regret"] == pytest.approx(1.05)
    assert out["true_class_map"]["mean_regret"] == pytest.approx(1.0)
    assert out["true_class_map"]["delta_vs_fixed"] == pytest.approx(-0.05)
    assert out["true_class_map"]["differs_from_fixed"] == 0.5
    assert out["predicted_class_map_by_horizon"][1] == out["true_class_map"]
    assert out["predicted_class_map_by_horizon"][2]["mean_regret"] == pytest.approx(1.15)
    lo, hi = out["true_class_map"]["delta_ci95"]
    assert lo <= -0.05 <= hi


# --- End-to-end pipeline on a tiny config -----------------------------------------------------


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("e2e")
    cfg = json.loads((ROOT / "configs" / "phase2.json").read_text(encoding="utf-8"))
    cfg["dataset"]["n_workloads"] = 60
    cfg["policy"]["quanta"] = [2, 8]
    (tmp / "cfg.json").write_text(json.dumps(cfg), encoding="utf-8")
    run = lambda *a: subprocess.run([sys.executable, *map(str, a)], check=True, capture_output=True, text=True)
    run(ROOT / "scripts" / "generate_dataset.py", "--config", tmp / "cfg.json", "--out", tmp / "a" / "dataset")
    run(ROOT / "scripts" / "train_evaluate.py", "--config", tmp / "cfg.json", "--artifacts", tmp / "a", "--reports", tmp / "r")
    run(ROOT / "scripts" / "plot_results.py", "--artifacts", tmp / "a", "--reports", tmp / "r")
    return tmp, cfg


def test_pipeline_persists_dataset_models_and_reports(pipeline):
    tmp, cfg = pipeline
    meta = json.loads((tmp / "a" / "dataset" / "dataset_meta.json").read_text(encoding="utf-8"))
    assert meta["rows"]["workloads"] == 60 and meta["rows"]["feature_rows"] == 180
    metrics = json.loads((tmp / "r" / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["dataset_version"] == meta["dataset_version"]
    assert set(metrics["main_experiment"]["models"]) == {"logistic_regression", "decision_tree", "random_forest"}
    assert metrics["main_experiment"]["categories"] == ["A", "B"]
    assert "C" in metrics["retrospective_experiment"]["categories"]
    assert metrics["policy_experiment"]["candidates"] == ["FCFS", "RR2", "RR8"]
    assert metrics["split"]["summary"]["test"]["workloads"] == 12
    summary = (tmp / "r" / "summary.md").read_text(encoding="utf-8")
    assert "NOT deployable" in summary and "not adaptive scheduling" in summary
    figs = {p.name for p in (tmp / "r" / "figures").glob("*.png")}
    assert figs == {"class_distribution.png", "feature_distributions.png", "model_comparison.png",
                    "confusion_matrices.png", "feature_importance.png", "policy_regret.png"}
    preds = pd.read_csv(tmp / "a" / "predictions_test.csv")
    assert len(preds) == 3 * 12 * 3


def test_saved_models_reload_with_metadata_and_predict(pipeline):
    tmp, cfg = pipeline
    for name in ("logistic_regression", "decision_tree", "random_forest"):
        model, meta = load_model(tmp / "a" / "models" / f"{name}.joblib")
        for key in ("model_type", "params", "features", "dataset_version", "dataset_config", "split",
                    "model_seed", "test_metrics", "fit_seconds", "inference_us_per_row", "library_versions"):
            assert key in meta, (name, key)
        out = predict_workload(model, meta, generate_workload(4242, 1).processes, 2)
        assert out["predicted"] in meta["classes"]


def test_training_refuses_a_stale_dataset(pipeline):
    tmp, cfg = pipeline
    cfg = {**cfg, "generator": {**cfg["generator"], "priority_max": 9}}
    (tmp / "stale.json").write_text(json.dumps(cfg), encoding="utf-8")
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "train_evaluate.py"), "--config", str(tmp / "stale.json"),
                        "--artifacts", str(tmp / "a"), "--reports", str(tmp / "r2")], capture_output=True, text=True)
    assert r.returncode != 0 and "different config" in r.stderr
