"""Grouped splits, training/evaluation protocol, persistence, and inference."""

import numpy as np
import pandas as pd
import pytest

from schedx.features import build_feature_table, feature_columns
from schedx.ml import (
    TARGET,
    classification_metrics,
    grouped_split,
    load_model,
    predict_workload,
    run_experiment,
    save_model,
)
from schedx.workload import CLASSES, generate_dataset, generate_workload

SMALL_GRIDS = {
    "logistic_regression": [{"C": 0.1}, {"C": 1.0}],
    "decision_tree": [{"max_depth": 3, "min_samples_leaf": 1}, {"max_depth": 6, "min_samples_leaf": 1}],
    "random_forest": [{"max_depth": None, "min_samples_leaf": 1}],
}


@pytest.fixture(scope="module")
def data():
    feats = build_feature_table(generate_dataset(150, master_seed=5), [1, 2, 3])
    split = grouped_split(feats, seed=3)
    results, models, preds = run_experiment(feats, split, seed=0, grids=SMALL_GRIDS)
    return feats, split, results, models, preds


# --- Splits ----------------------------------------------------------------------


def test_split_is_grouped_disjoint_and_complete(data):
    feats, split, *_ = data
    assert set(split.index) == set(feats.workload_id)
    assert split.index.is_unique  # each workload in exactly one split
    rows_split = feats.workload_id.map(split)
    # All horizon rows of a workload land in the same split.
    assert (feats.assign(s=rows_split).groupby("workload_id")["s"].nunique() == 1).all()
    counts = split.value_counts()
    assert counts["train"] == 90 and counts["validation"] == 30 and counts["test"] == 30


def test_split_is_stratified_and_deterministic(data):
    feats, split, *_ = data
    labels = feats.groupby("workload_id")[TARGET].first()
    overall = labels.value_counts(normalize=True)
    for part in ("train", "validation", "test"):
        dist = labels[split.index[split == part]].value_counts(normalize=True)
        assert (dist.reindex(CLASSES) - overall.reindex(CLASSES)).abs().max() < 0.06
    assert grouped_split(feats, seed=3).equals(split)
    assert not grouped_split(feats, seed=4).equals(split)


def test_split_rejects_bad_input(data):
    feats, *_ = data
    with pytest.raises(ValueError):
        grouped_split(feats, seed=0, train=0.5, validation=0.2, test=0.2)
    broken = feats.copy()
    first = broken.workload_id.iloc[0]
    broken.loc[(broken.workload_id == first) & (broken.horizon_k == 1), TARGET] = "other"
    with pytest.raises(ValueError, match="more than one label"):
        grouped_split(broken, seed=0)


# --- Training protocol -------------------------------------------------------------


def test_experiment_reports_complete_and_consistent_metrics(data):
    feats, split, results, models, preds = data
    n_test_rows = int(feats.workload_id.map(split).eq("test").sum())
    assert set(results["models"]) == set(SMALL_GRIDS) == set(models)
    for name, r in results["models"].items():
        t = r["test"]
        for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1"):
            assert 0.0 <= t[key] <= 1.0
        assert np.array(t["confusion_matrix"]["matrix"]).sum() == n_test_rows == t["n"]
        assert sum(c["support"] for c in t["per_class"].values()) == n_test_rows
        assert set(r["test_accuracy_by_horizon"]) == {1, 2, 3}
        assert r["fit_seconds"] > 0 and r["inference_us_per_row"] > 0
        # Well above the 1/3 chance level on this synthetic task.
        assert t["accuracy"] > 0.6
    assert results["features"] == feature_columns()
    assert len(preds) == n_test_rows * len(SMALL_GRIDS)
    assert set(preds.workload_id) == set(split.index[split == "test"])
    probs = preds[[f"p_{c}" for c in CLASSES]].to_numpy()
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert (preds.confidence >= 1 / 3 - 1e-9).all()


def test_selection_uses_validation_score_only(data):
    *_, results, _, _ = data
    best = max(results["models"], key=lambda n: results["models"][n]["validation_macro_f1"])
    assert results["selected_model"] == best
    for r in results["models"].values():
        assert r["validation_macro_f1"] == max(g["validation_macro_f1"] for g in r["grid"])


def test_test_labels_cannot_influence_tuning(data):
    feats, split, results, *_ = data
    scrambled = feats.copy()
    test_rows = scrambled.workload_id.map(split).eq("test")
    rng = np.random.default_rng(0)
    scrambled.loc[test_rows, TARGET] = rng.permutation(scrambled.loc[test_rows, TARGET].to_numpy())
    again, *_ = run_experiment(scrambled, split, seed=0, grids=SMALL_GRIDS)
    for name in SMALL_GRIDS:
        assert again["models"][name]["grid"] == results["models"][name]["grid"]
        assert again["models"][name]["params"] == results["models"][name]["params"]
    assert again["selected_model"] == results["selected_model"]


def test_preprocessing_is_fitted_on_training_rows_only(data):
    feats, split, results, models, _ = data
    cols = results["features"]
    trva = feats[feats.workload_id.map(split).isin(["train", "validation"])]
    medians = trva[cols].median().fillna(0).to_numpy()
    assert np.allclose(models["random_forest"].named_steps["impute"].statistics_, medians)


def test_experiment_is_deterministic(data):
    feats, split, results, *_ = data
    again, *_ = run_experiment(feats, split, seed=0, grids=SMALL_GRIDS)
    strip = lambda r: {n: {k: v for k, v in m.items() if k not in ("fit_seconds", "inference_us_per_row")}
                       for n, m in r["models"].items()}
    assert strip(again) == strip(results)


def test_classification_metrics_hand_example():
    y = ["cpu_bound", "cpu_bound", "mixed", "io_bound"]
    p = ["cpu_bound", "mixed", "mixed", "io_bound"]
    m = classification_metrics(y, p)
    assert m["accuracy"] == 0.75
    assert m["per_class"]["cpu_bound"] == {"precision": 1.0, "recall": 0.5, "f1": pytest.approx(2 / 3), "support": 2}
    assert m["per_class"]["mixed"]["precision"] == 0.5
    assert m["confusion_matrix"]["matrix"] == [[1, 1, 0], [0, 1, 0], [0, 0, 1]]
    assert m["macro_recall"] == pytest.approx((0.5 + 1 + 1) / 3)


# --- Persistence and inference -----------------------------------------------------------


def test_save_load_round_trip_and_new_workload_prediction(data, tmp_path):
    feats, split, results, models, _ = data
    cols = results["features"]
    meta = {"name": "random_forest", "features": cols}
    path = save_model(models["random_forest"], meta, tmp_path, "random_forest")
    model, loaded_meta = load_model(path)
    assert loaded_meta == meta
    X = feats[cols].head(30)
    assert (model.predict(X) == models["random_forest"].predict(X)).all()

    # A workload from a seed never seen in training, at every horizon.
    fresh = generate_workload(master_seed=999, index=0)
    for k in (1, 2, 3):
        out = predict_workload(model, loaded_meta, fresh.processes, k)
        assert out["predicted"] in CLASSES and out["horizon_k"] == k
        assert sum(out["probabilities"].values()) == pytest.approx(1.0)
        assert out["confidence"] == max(out["probabilities"].values())


def test_schema_mismatch_is_rejected(data, tmp_path):
    *_, results, models, _ = data
    with pytest.raises(ValueError, match="feature order"):
        save_model(models["decision_tree"], {"features": list(reversed(results["features"]))}, tmp_path, "dt")
    path = save_model(models["decision_tree"], {"features": results["features"]}, tmp_path, "dt")
    path.with_suffix(".json").write_text('{"features": ["n_processes"]}')
    with pytest.raises(ValueError, match="do not match"):
        load_model(path)
