"""Leakage-safe training, evaluation, persistence, and inference (Phase 2).

Protocol (docs/phase2_ml.md §4-5):

1. Split *workloads* (not rows) 60/20/20 into train/validation/test,
   stratified by class. Every horizon-k row of a workload stays together.
2. For each model, pick hyperparameters by macro-F1 on validation, training
   on train only. Preprocessing lives inside the Pipeline, so it is fitted on
   the training fold only.
3. Refit the chosen configuration on train + validation; evaluate once on
   test.
"""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .features import DEPLOYABLE, feature_columns, snapshot_features
from .workload import CLASSES, GeneratedProcess

TARGET = "workload_class"

GRIDS: dict[str, list[dict]] = {
    "logistic_regression": [{"C": c} for c in (0.1, 1.0, 10.0)],
    "decision_tree": [{"max_depth": d, "min_samples_leaf": m} for d in (3, 5, 8, None) for m in (1, 5)],
    "random_forest": [{"max_depth": d, "min_samples_leaf": m} for d in (None, 8) for m in (1, 3)],
}


def make_pipeline(name: str, params: dict, seed: int) -> Pipeline:
    # Median imputation + missing-indicator columns; keep_empty_features keeps
    # the output schema fixed even if a column is all-missing in training.
    impute = ("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True))
    if name == "logistic_regression":
        clf = LogisticRegression(max_iter=5000, random_state=seed, **params)
        return Pipeline([impute, ("scale", StandardScaler()), ("clf", clf)])
    if name == "decision_tree":
        return Pipeline([impute, ("clf", DecisionTreeClassifier(random_state=seed, **params))])
    if name == "random_forest":
        clf = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=1, **params)
        return Pipeline([impute, ("clf", clf)])
    raise ValueError(f"unknown model {name!r}")


def grouped_split(
    df: pd.DataFrame, seed: int, train: float = 0.6, validation: float = 0.2, test: float = 0.2
) -> pd.Series:
    """Map each ``workload_id`` to 'train' / 'validation' / 'test'.

    Splits unique workloads, stratified by their class, so correlated rows of
    one workload never cross splits.
    """
    if not np.isclose(train + validation + test, 1.0):
        raise ValueError("split fractions must sum to 1")
    labels = df.groupby("workload_id")[TARGET].agg(lambda s: s.iloc[0])
    if (df.groupby("workload_id")[TARGET].nunique() > 1).any():
        raise ValueError("a workload has more than one label")
    ids = labels.index.to_numpy()
    rest, test_ids = train_test_split(ids, test_size=test, stratify=labels.to_numpy(), random_state=seed)
    train_ids, val_ids = train_test_split(
        rest, test_size=validation / (train + validation), stratify=labels[rest].to_numpy(), random_state=seed
    )
    split = pd.concat(
        [pd.Series("train", index=train_ids), pd.Series("validation", index=val_ids), pd.Series("test", index=test_ids)]
    )
    assert split.index.is_unique and len(split) == len(ids)
    return split.sort_index()


def classification_metrics(y_true, y_pred) -> dict:
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=list(CLASSES), zero_division=0)
    mp, mr, mf, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(CLASSES), average="macro", zero_division=0
    )
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(mp),
        "macro_recall": float(mr),
        "macro_f1": float(mf),
        "per_class": {
            c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
            for i, c in enumerate(CLASSES)
        },
        "confusion_matrix": {
            "labels": list(CLASSES),
            "matrix": confusion_matrix(y_true, y_pred, labels=list(CLASSES)).tolist(),  # rows = true
        },
    }


def _importance(model: Pipeline) -> dict[str, float] | None:
    clf = model.named_steps["clf"]
    if not hasattr(clf, "feature_importances_"):
        return None
    names = model[:-1].get_feature_names_out()
    return {str(n): float(v) for n, v in sorted(zip(names, clf.feature_importances_), key=lambda t: -t[1])}


def run_experiment(
    features: pd.DataFrame,
    split: pd.Series,
    seed: int,
    categories: Iterable[str] = DEPLOYABLE,
    grids: dict[str, list[dict]] = GRIDS,
) -> tuple[dict, dict[str, Pipeline], pd.DataFrame]:
    """Tune on validation, refit on train+validation, evaluate on test.

    Returns ``(results, fitted_models, test_predictions)``.
    """
    cols = feature_columns(categories)
    part = features["workload_id"].map(split)
    if part.isna().any():
        raise ValueError("some workloads are missing from the split")
    tr, va, te = (features[part == s] for s in ("train", "validation", "test"))
    trva = features[part.isin(["train", "validation"])]

    results: dict = {"features": cols, "categories": sorted(set(categories)), "models": {}}
    fitted: dict[str, Pipeline] = {}
    preds = []
    for name, grid in grids.items():
        tried = []
        for params in grid:
            m = make_pipeline(name, params, seed).fit(tr[cols], tr[TARGET])
            tried.append({"params": params, "validation_macro_f1": float(f1_score(va[TARGET], m.predict(va[cols]), average="macro"))})
        best = max(tried, key=lambda t: t["validation_macro_f1"])  # first wins ties

        model = make_pipeline(name, best["params"], seed)
        t0 = time.perf_counter()
        model.fit(trva[cols], trva[TARGET])
        fit_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        y_pred = model.predict(te[cols])
        infer_s = time.perf_counter() - t0
        proba = model.predict_proba(te[cols])

        test = classification_metrics(te[TARGET], y_pred)
        results["models"][name] = {
            "estimator": type(model.named_steps["clf"]).__name__,
            "params": best["params"],
            "grid": tried,
            "validation_macro_f1": best["validation_macro_f1"],
            "fit_seconds": fit_s,
            "inference_us_per_row": 1e6 * infer_s / len(te),
            "test": test,
            "test_accuracy_by_horizon": {
                int(k): float(accuracy_score(te[TARGET][te.horizon_k == k], y_pred[(te.horizon_k == k).to_numpy()]))
                for k in sorted(te.horizon_k.unique())
            },
            "feature_importance": _importance(model),
        }
        fitted[name] = model
        order = list(model.classes_)
        preds.append(
            pd.DataFrame(
                {
                    "model": name,
                    "workload_id": te["workload_id"].to_numpy(),
                    "horizon_k": te["horizon_k"].to_numpy(),
                    "true": te[TARGET].to_numpy(),
                    "predicted": y_pred,
                    "confidence": proba.max(axis=1),
                    **{f"p_{c}": proba[:, order.index(c)] for c in CLASSES},
                }
            )
        )
    # Chosen for downstream use by validation score only (never test).
    results["selected_model"] = max(results["models"], key=lambda n: results["models"][n]["validation_macro_f1"])
    return results, fitted, pd.concat(preds, ignore_index=True)


def split_summary(features: pd.DataFrame, split: pd.Series) -> dict:
    labels = features.groupby("workload_id")[TARGET].first()
    out = {}
    for s in ("train", "validation", "test"):
        ids = split.index[split == s]
        out[s] = {
            "workloads": int(len(ids)),
            "rows": int(features["workload_id"].isin(ids).sum()),
            "class_counts": labels[ids].value_counts().reindex(CLASSES, fill_value=0).astype(int).to_dict(),
        }
    return out


# --- Persistence and inference ------------------------------------------------------


def library_versions() -> dict:
    return {
        "python": platform.python_version(),
        "scikit-learn": sklearn.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }


def save_model(model: Pipeline, metadata: dict, directory: Path, name: str) -> Path:
    if list(model.feature_names_in_) != list(metadata["features"]):
        raise ValueError("metadata feature order does not match the fitted model")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.joblib"
    joblib.dump(model, path)
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2))
    return path


def load_model(path: Path) -> tuple[Pipeline, dict]:
    """Load a pipeline and its metadata. Only load files you created:
    joblib (pickle) can execute code from untrusted files."""
    path = Path(path)
    model = joblib.load(path)
    meta = json.loads(path.with_suffix(".json").read_text())
    if list(model.feature_names_in_) != list(meta["features"]):
        raise ValueError(f"{path}: model features do not match its metadata")
    return model, meta


def predict_workload(model: Pipeline, meta: dict, processes: tuple[GeneratedProcess, ...], k: int) -> dict:
    """Predict the class of one workload from what is observable at horizon k."""
    feats = snapshot_features(processes, k)
    X = pd.DataFrame([feats])[meta["features"]]  # exact training order; KeyError if schema drifted
    proba = model.predict_proba(X)[0]
    probs = {str(c): float(p) for c, p in zip(model.classes_, proba)}
    label = max(probs, key=probs.get)
    return {"predicted": label, "confidence": probs[label], "probabilities": probs, "horizon_k": k}
