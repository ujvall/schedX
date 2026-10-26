"""Phase 3 controller (decision rules, schema, fallback) and paired statistics."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from schedx.controller import (
    ALLOWED_FEATURES,
    QuantumController,
    best_candidate,
    check_features,
    fit_controller,
    score,
    targets,
    tie_order,
)
from schedx.stats import holm, paired_comparison, rank_biserial

CANDS = (1, 2, 4, 8)
SUB = ALLOWED_FEATURES["submission"]


def toy_data(n=120, seed=0):
    """Synthetic outcomes where the best quantum depends on one A feature."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, len(SUB))), columns=SUB, index=[f"w{i}" for i in range(n)])
    high = X["arrival_rate"] > 0
    out = pd.DataFrame(10.0, index=X.index, columns=list(CANDS))
    out.loc[high, 8] = 7.0  # big quantum best when arrival_rate > 0
    out.loc[~high, 1] = 6.0  # small quantum best otherwise
    return X, out


# --- Feature whitelist ---------------------------------------------------------------


@pytest.mark.parametrize("bad", ["obs_cpu_mean", "obs_frac_io_dominant", "total_cpu_mean", "lifetime_io_fraction",
                                 "workload_class", "share_io", "regime", "burst_cv", "workload_id"])
def test_forbidden_features_are_rejected(bad):
    with pytest.raises(ValueError, match="not available"):
        check_features([*SUB, bad], "online")


def test_submission_controller_cannot_use_online_observations():
    with pytest.raises(ValueError):
        check_features(["obs_t_decision"], "submission")
    check_features(ALLOWED_FEATURES["online"], "online")  # all allowed online features pass
    assert set(SUB) <= set(ALLOWED_FEATURES["online"])
    with pytest.raises(ValueError):
        QuantumController("online", "constant", CANDS, 2, features=("total_io_mean",))


def test_controller_module_does_not_import_the_simulator():
    src = (Path(__file__).resolve().parents[1] / "src" / "schedx" / "controller.py").read_text(encoding="utf-8")
    for mod in ("scheduler", "adaptive_eval", "policy_eval", "metrics"):
        assert f"from .{mod}" not in src and f"import {mod}" not in src


# --- Targets and tie-breaking ------------------------------------------------------------


def test_tie_order_prefers_default_then_smaller():
    assert tie_order(CANDS, 4) == [4, 1, 2, 8]
    assert best_candidate({1: 5.0, 2: 5.0, 4: 5.0, 8: 9.0}, tie_order(CANDS, 2)) == 2
    assert best_candidate({1: 5.0, 2: 6.0, 4: 5.0, 8: 9.0}, tie_order(CANDS, 2)) == 1
    assert best_candidate({1: 5.0, 2: 6.0, 4: 3.0, 8: 3.0}, tie_order(CANDS, 2)) == 4


def test_targets_hand_example():
    out = pd.DataFrame({1: [4.0, 2.0], 2: [2.0, 2.0], 4: [8.0, 1.0], 8: [2.0, 4.0]}, index=["a", "b"])
    labels, lr = targets(out, tie_order(CANDS, 2))
    assert labels.to_dict() == {"a": 2, "b": 4}  # a: 2 and 8 tie -> default 2
    assert lr.loc["a", 4] == pytest.approx(np.log(4)) and lr.loc["b", 4] == 0.0
    assert list(lr.columns) == [2, 1, 4, 8]


# --- Training and prediction ------------------------------------------------------------------


@pytest.mark.parametrize("family,params", [("tree_classifier", {"max_depth": 2, "min_samples_leaf": 5}),
                                           ("logistic_classifier", {"C": 10.0}),
                                           ("regret_regressor", {"max_depth": 2, "min_samples_leaf": 5})])
def test_controllers_learn_a_simple_rule(family, params):
    X, out = toy_data()
    c = fit_controller("submission", family, params, X, out, CANDS, 2, seed=0)
    Xv, outv = toy_data(seed=1)
    decisions = c.decide_batch(Xv)
    expected = np.where(Xv["arrival_rate"] > 0, 8, 1)
    assert np.mean([q for q, _ in decisions] == expected) > 0.9
    assert {s for _, s in decisions} == {"model"}
    assert score(c, Xv, outv).mean() < -0.2  # clearly better than always q0=2
    one = c.decide(Xv.iloc[0].to_dict())
    assert one == decisions[0]


def test_fit_is_deterministic():
    X, out = toy_data()
    a = fit_controller("submission", "tree_classifier", {"max_depth": 3, "min_samples_leaf": 5}, X, out, CANDS, 2, seed=0)
    b = fit_controller("submission", "tree_classifier", {"max_depth": 3, "min_samples_leaf": 5}, X, out, CANDS, 2, seed=0)
    assert a.decide_batch(X) == b.decide_batch(X)


def test_single_class_training_data_is_handled():
    X, out = toy_data()
    out[:] = 5.0
    out[4] = 1.0  # 4 always best
    c = fit_controller("submission", "logistic_classifier", {"C": 1.0}, X, out, CANDS, 2, seed=0)
    assert {q for q, _ in c.decide_batch(X)} == {4}


def test_constant_controller_and_score():
    X, out = toy_data()
    c = fit_controller("submission", "constant", {}, X, out, CANDS, 2, seed=0)
    assert c.decide(None) == (2, "constant") and c.decide({"anything": 1}) == (2, "constant")
    assert (score(c, X, out) == 0).all()


def test_score_falls_back_for_workloads_without_observation():
    X, out = toy_data()
    c = fit_controller("online", "tree_classifier", {"max_depth": 2, "min_samples_leaf": 5},
                       X.reindex(columns=ALLOWED_FEATURES["online"]), out, CANDS, 2, seed=0, after_completions=5)
    observed = X.reindex(columns=ALLOWED_FEATURES["online"]).iloc[:50]
    s = score(c, observed, out)
    assert len(s) == len(out) and (s.iloc[50:] == 0).all()  # unobserved -> q0 -> log ratio 0


# --- Fallback and schema errors -------------------------------------------------------------------


class _Fixed:
    def __init__(self, value):
        self.value = value

    def predict(self, X):
        return np.array([self.value] * len(X), dtype=object)


@pytest.mark.parametrize("bad", [3, "eight", 2.5, None])
def test_invalid_classifier_prediction_falls_back(bad):
    c = QuantumController("submission", "tree_classifier", CANDS, 2, tuple(SUB), model=_Fixed(bad))
    X, _ = toy_data(5)
    assert c.decide_batch(X) == [(2, "fallback:invalid_prediction")] * 5


def test_invalid_regressor_prediction_falls_back():
    class NaNModel:
        def predict(self, X):
            return np.full((len(X), 4), np.nan)

    class WrongShape:
        def predict(self, X):
            return np.zeros((len(X), 3))

    X, _ = toy_data(3)
    for model in (NaNModel(), WrongShape()):
        c = QuantumController("submission", "regret_regressor", CANDS, 2, tuple(SUB), model=model)
        assert c.decide_batch(X) == [(2, "fallback:invalid_prediction")] * 3


def test_missing_observation_and_schema_errors():
    X, out = toy_data()
    c = fit_controller("submission", "tree_classifier", {"max_depth": 2, "min_samples_leaf": 5}, X, out, CANDS, 2, 0)
    assert c.decide(None) == (2, "fallback:no_observation")
    with pytest.raises(KeyError, match="missing"):
        c.decide({"n_processes": 5})
    with pytest.raises(ValueError):
        QuantumController("submission", "tree_classifier", CANDS, 2, tuple(SUB))  # no model
    with pytest.raises(ValueError):
        QuantumController("submission", "constant", CANDS, 3)  # default not a candidate
    with pytest.raises(ValueError):
        QuantumController("submission", "magic", CANDS, 2)
    with pytest.raises(ValueError):
        fit_controller("submission", "tree_classifier", {}, X.iloc[:10], out, CANDS, 2, 0)


# --- Statistics --------------------------------------------------------------------------------------


def test_paired_comparison_uniform_improvement():
    b = np.array([10.0, 20.0, 40.0, 80.0, 5.0, 7.0, 9.0, 11.0])
    r = paired_comparison(0.9 * b, b, seed=1, n_boot=2000)
    assert r["geo_rel_change"] == pytest.approx(-0.1)
    assert r["geo_rel_change_ci95"] == pytest.approx([-0.1, -0.1])
    assert r["mean_abs_diff"] == pytest.approx(-0.1 * b.mean())
    assert r["share_better"] == 1.0 and r["rank_biserial"] == -1.0
    assert r["wilcoxon_p"] < 0.01


def test_paired_comparison_identical_and_mixed():
    b = np.arange(1.0, 21.0)
    same = paired_comparison(b, b)
    assert same["wilcoxon_p"] == 1.0 and same["share_equal"] == 1.0 and same["geo_rel_change"] == 0.0
    m = b.copy()
    m[:10] *= 1.05
    m[10:] *= 0.95
    r = paired_comparison(m, b, seed=0, n_boot=4000)
    lo, hi = r["geo_rel_change_ci95"]
    assert lo < r["geo_rel_change"] < hi and lo < 0 < hi
    assert r["wilcoxon_p"] > 0.05
    assert paired_comparison(m, b, seed=0, n_boot=4000) == r  # deterministic


def test_paired_comparison_rejects_bad_input():
    with pytest.raises(ValueError):
        paired_comparison([1.0, 2.0], [1.0])
    with pytest.raises(ValueError):
        paired_comparison([0.0], [1.0])
    with pytest.raises(ValueError):
        paired_comparison([], [])


def test_rank_biserial_and_holm():
    assert rank_biserial(np.array([1.0, 2.0, -3.0, 0.0])) == pytest.approx((1 + 2 - 3) / 6)
    adj = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj == pytest.approx({"a": 0.03, "c": 0.06, "b": 0.06})
    assert holm({"x": 0.9, "y": 0.8}) == {"y": 1.0, "x": 1.0}
