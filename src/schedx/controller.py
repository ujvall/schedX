"""Feature-based Round Robin quantum controller (Phase 3).

The controller is a pure decision function: feature dict -> quantum. It does
not import the simulator, so its rules can be tested and replaced on their
own. Inputs are restricted to deployable features (A submission features,
plus O online observations for the online regime); anything else raises.

Families (docs/phase3_adaptive_scheduling.md §7):

* ``constant``            -- always the default quantum ``q0``.
* ``tree_classifier``     -- predicts the best candidate directly.
* ``logistic_classifier`` -- same target, linear model.
* ``regret_regressor``    -- multi-output tree predicting log-regret per
  candidate; picks the argmin.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from .features import feature_columns
from .observation import O_FEATURES

SUBMISSION_FEATURES = feature_columns(["A"])
ALLOWED_FEATURES = {
    "submission": list(SUBMISSION_FEATURES),
    "online": [*SUBMISSION_FEATURES, *O_FEATURES],
}
FAMILIES = ("constant", "tree_classifier", "logistic_classifier", "regret_regressor")


def check_features(features, regime: str) -> None:
    if regime not in ALLOWED_FEATURES:
        raise ValueError(f"unknown regime {regime!r}")
    bad = [f for f in features if f not in ALLOWED_FEATURES[regime]]
    if bad:
        raise ValueError(f"features not available to a {regime} controller: {bad}")


def tie_order(candidates, default: int) -> list[int]:
    """Deterministic preference among equally good candidates: the default
    quantum first, then smaller quanta."""
    return [default, *sorted(q for q in candidates if q != default)]


def best_candidate(values: Mapping[int, float], order: list[int]) -> int:
    best = min(values[q] for q in order)
    # Averages computed the same way are bit-identical when tied; the tiny
    # tolerance only absorbs float summation-order noise.
    return next(q for q in order if math.isclose(values[q], best, rel_tol=1e-12, abs_tol=1e-12))


def targets(outcomes: pd.DataFrame, order: list[int]) -> tuple[pd.Series, pd.DataFrame]:
    """Best-candidate labels and log-regret vectors from counterfactual outcomes.

    ``outcomes``: index = workload id, columns = candidate quanta, values =
    the objective (average turnaround; lower is better).
    """
    labels = outcomes.apply(lambda row: best_candidate(row.to_dict(), order), axis=1).astype(int)
    log_regret = np.log(outcomes[order].div(outcomes[order].min(axis=1), axis=0))
    return labels, log_regret


def _imputer() -> tuple[str, SimpleImputer]:
    return "impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)


def build_model(family: str, params: dict, seed: int, n_classes: int = 2) -> Pipeline:
    if family == "tree_classifier":
        return Pipeline([_imputer(), ("clf", DecisionTreeClassifier(random_state=seed, **params))])
    if family == "logistic_classifier":
        if n_classes < 2:  # LogisticRegression cannot fit a single class
            return Pipeline([_imputer(), ("clf", DummyClassifier(strategy="most_frequent"))])
        return Pipeline([_imputer(), ("scale", StandardScaler()),
                         ("clf", LogisticRegression(max_iter=5000, random_state=seed, **params))])
    if family == "regret_regressor":
        return Pipeline([_imputer(), ("reg", DecisionTreeRegressor(random_state=seed, **params))])
    raise ValueError(f"family {family!r} has no model")


@dataclass
class QuantumController:
    regime: str  # "submission" or "online"
    family: str
    candidates: tuple[int, ...]
    default: int
    features: tuple[str, ...] = ()
    params: dict = field(default_factory=dict)
    after_completions: int | None = None  # online decision point (m)
    model: Pipeline | None = None

    def __post_init__(self) -> None:
        check_features(self.features, self.regime)
        if self.family not in FAMILIES:
            raise ValueError(f"unknown family {self.family!r}")
        if self.default not in self.candidates:
            raise ValueError("default quantum must be a candidate")
        if self.family != "constant" and self.model is None:
            raise ValueError(f"{self.family} controller needs a fitted model")
        if self.regime == "online" and self.family != "constant" and not self.after_completions:
            raise ValueError("online controller needs after_completions")
        self.order = tie_order(self.candidates, self.default)

    def describe(self) -> str:
        m = f", m={self.after_completions}" if self.regime == "online" and self.family != "constant" else ""
        return f"{self.regime}/{self.family}{m} {self.params or ''}".strip()

    def decide_batch(self, X: pd.DataFrame | None) -> list[tuple[int, str]]:
        """Decide for each row of ``X`` (columns must include ``features``)."""
        if X is None or len(X) == 0:
            return []
        if self.family == "constant":
            return [(self.default, "constant")] * len(X)
        missing = [f for f in self.features if f not in X.columns]
        if missing:
            raise KeyError(f"feature schema mismatch, missing {missing}")
        X = X[list(self.features)]
        if self.family == "regret_regressor":
            out = []
            for pred in np.asarray(self.model.predict(X), dtype=float).reshape(len(X), -1):
                if pred.shape[0] != len(self.order) or not np.all(np.isfinite(pred)):
                    out.append((self.default, "fallback:invalid_prediction"))
                else:
                    out.append((best_candidate(dict(zip(self.order, pred)), self.order), "model"))
            return out
        out = []
        for q in self.model.predict(X):
            try:
                qi = int(q)
                valid = qi == q and qi in self.candidates
            except (TypeError, ValueError):
                valid = False
            out.append((qi, "model") if valid else (self.default, "fallback:invalid_prediction"))
        return out

    def decide(self, feats: Mapping[str, float] | None) -> tuple[int, str]:
        """One decision. ``None`` means nothing was observable (no decision point)."""
        if self.family == "constant":
            return self.default, "constant"
        if feats is None:
            return self.default, "fallback:no_observation"
        return self.decide_batch(pd.DataFrame([dict(feats)]))[0]


def fit_controller(
    regime: str,
    family: str,
    params: dict,
    X: pd.DataFrame,
    outcomes: pd.DataFrame,
    candidates: tuple[int, ...],
    default: int,
    seed: int,
    after_completions: int | None = None,
) -> QuantumController:
    """Fit on training workloads: ``X`` rows and ``outcomes`` rows share ids."""
    feats = tuple(ALLOWED_FEATURES[regime])
    if family == "constant":
        return QuantumController(regime, family, tuple(candidates), default, feats, {}, after_completions)
    if not X.index.equals(outcomes.index):
        raise ValueError("features and outcomes must cover the same workloads in the same order")
    order = tie_order(candidates, default)
    labels, log_regret = targets(outcomes, order)
    if family == "regret_regressor":
        model = build_model(family, params, seed).fit(X[list(feats)], log_regret[order].to_numpy())
    else:
        model = build_model(family, params, seed, labels.nunique()).fit(X[list(feats)], labels)
    return QuantumController(regime, family, tuple(candidates), default, feats, dict(params), after_completions, model)


def score(controller: QuantumController, X: pd.DataFrame, outcomes: pd.DataFrame) -> pd.Series:
    """Per-workload ``log(T_chosen / T_default)`` (negative = better than q0).

    ``outcomes`` covers every workload; ``X`` only those with a decision
    point (missing rows mean "nothing observable" and fall back to q0).
    """
    chosen = pd.Series(controller.default, index=outcomes.index)
    rows = X.loc[X.index.intersection(outcomes.index)] if len(X) else X
    if len(rows):
        chosen.loc[rows.index] = [q for q, _ in controller.decide_batch(rows)]
    t = np.array([outcomes.at[w, q] for w, q in chosen.items()])
    return pd.Series(np.log(t / outcomes[controller.default].to_numpy()), index=outcomes.index)
