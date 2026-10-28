"""Phase 4 controller integration: whitelist, inference, fallback, serialization,
hypothesis merging, and backward compatibility."""

import io

import joblib
import numpy as np
import pandas as pd
import pytest

from schedx import Process, compute_metrics, round_robin
from schedx.adaptive_eval import decision_time, summarize, switched
from schedx.controller import ALLOWED_FEATURES, SUBMISSION_FEATURES, QuantumController, check_features
from schedx.estimator import E_FEATURES
from schedx.observation import O_FEATURES
from schedx.phase4 import REGIME, analyze, counterfactual_data, improved_features, run_p4, search
from schedx.policy_eval import burst_jobs
from schedx.stats import signflip_test
from schedx.workload import GeneratedProcess, generate_dataset

CFG = {"candidates": [1, 2, 4, 8], "decision_after_completions": [5, 10], "seed": 0,
       "grids": {"tree_classifier": [{"max_depth": 2, "min_samples_leaf": 3}],
                 "logistic_classifier": [{"C": 1.0}],
                 "regret_regressor": [{"max_depth": 2, "min_samples_leaf": 3}]}}


@pytest.fixture(scope="module")
def fitted():
    ws = generate_dataset(60, master_seed=44)
    traces = {w.workload_id: w.processes for w in ws}
    ids = list(traces)
    data = counterfactual_data(traces, CFG)
    abl = {"family": "tree_classifier", "params": {"max_depth": 2, "min_samples_leaf": 3}, "after_completions": 10}
    table, ctrls = search(data, ids[:40], ids[40:], CFG, abl)
    return traces, data, table, ctrls


# --- Whitelist and backward compatibility ------------------------------------------------


def test_online_censored_whitelist():
    assert ALLOWED_FEATURES[REGIME] == [*SUBMISSION_FEATURES, *O_FEATURES, *E_FEATURES]
    check_features(ALLOWED_FEATURES[REGIME], REGIME)
    for bad in ("total_cpu_mean", "lifetime_io_fraction", "workload_class", "obs_cpu_mean", "regime", "burst_cv"):
        with pytest.raises(ValueError):
            check_features([bad], REGIME)


def test_phase3_regimes_are_unchanged():
    assert ALLOWED_FEATURES["online"] == [*SUBMISSION_FEATURES, *O_FEATURES]
    with pytest.raises(ValueError):
        check_features(["km_rmst"], "online")  # Phase 3 controllers cannot see E features
    with pytest.raises(ValueError):
        QuantumController(REGIME, "tree_classifier", (1, 2), 1, (), model=object())  # needs after_completions


# --- Fit stage ------------------------------------------------------------------------------


def test_search_outputs(fitted):
    _, data, table, ctrls = fitted
    assert len(table) == 2 * 3 and table.selected.sum() == 1
    assert table.loc[table.selected, "val_mean_log_ratio"].item() == table.val_mean_log_ratio.min()
    sel, abl = ctrls["p4_selected"], ctrls["p4_ablation"]
    assert sel.regime == abl.regime == REGIME
    assert set(sel.features) == set(ALLOWED_FEATURES[REGIME])
    assert (abl.family, abl.after_completions) == ("tree_classifier", 10)
    for m in CFG["decision_after_completions"]:
        X = data["online"][m]["X"]
        assert set(X.columns) == set(ALLOWED_FEATURES[REGIME])


# --- Inference and fallback --------------------------------------------------------------------


def test_inference_reproduces_counterfactual_and_never_sees_outcomes(fitted):
    traces, data, _, ctrls = fitted
    ctrl = ctrls["p4_selected"]
    q0 = data["q0"]
    for w, procs in list(traces.items())[:15]:
        sched, q, source, t = run_p4(procs, ctrl, q0)
        jobs = burst_jobs(procs)
        expected = round_robin(jobs, q0) if t is None or q == q0 else switched(jobs, q0, q, t)
        assert summarize(compute_metrics(sched)) == summarize(compute_metrics(expected))
        if t is not None:
            feats = improved_features(procs, jobs, round_robin(jobs, q0), t)
            assert ctrl.decide(feats) == (q, source)


def test_fallback_when_no_decision_point(fitted):
    *_, ctrls = fitted
    tiny = (GeneratedProcess("P0", 0, 0, (3,), (), "cpu_bound"), GeneratedProcess("P1", 1, 0, (2,), (), "cpu_bound"))
    sched, q, source, t = run_p4(tiny, ctrls["p4_selected"], 1)
    assert (q, source, t) == (1, "fallback:no_observation", None)


def test_fallback_when_observation_insufficient(fitted):
    *_, ctrls = fitted
    ctrl = ctrls["p4_ablation"]
    early = QuantumController(REGIME, ctrl.family, ctrl.candidates, ctrl.default, ctrl.features, ctrl.params,
                              after_completions=2, model=ctrl.model)  # 2 < MIN_COMPLETED
    procs = generate_dataset(1, master_seed=5)[0].processes
    sched, q, source, t = run_p4(procs, early, 1)
    assert (q, source) == (1, "fallback:insufficient_observation") and t is not None
    assert summarize(compute_metrics(sched)) == summarize(compute_metrics(round_robin(burst_jobs(procs), 1)))


def test_controller_serialization_round_trip(fitted):
    traces, data, _, ctrls = fitted
    buf = io.BytesIO()
    joblib.dump(ctrls["p4_selected"], buf)
    buf.seek(0)
    loaded = joblib.load(buf)
    X = data["online"][ctrls["p4_selected"].after_completions]["X"]
    assert loaded.decide_batch(X) == ctrls["p4_selected"].decide_batch(X)


# --- Statistics ----------------------------------------------------------------------------------


def test_signflip_exact_small_cases():
    assert signflip_test([1.0, 1.0, 1.0])["p"] == 0.25  # 2 of 8 patterns as extreme
    assert signflip_test([0.0, 0.0])["p"] == 1.0
    assert signflip_test([2.0, -1.0, 0.0])["p"] == 1.0  # |sum| = 1 reached by all 4 patterns
    r = signflip_test(np.r_[np.full(30, -0.1), np.full(5, 0.1)], seed=1, n_mc=20_000)
    assert r["method"].startswith("monte_carlo") and r["p"] < 0.001


def test_analyze_merges_hypotheses_with_identical_decisions():
    rng = np.random.default_rng(0)
    rows = []
    for i in range(40):
        base = 10 + rng.random()
        same = base * (0.99 if i % 3 == 0 else 1.0)
        for method, val, pol in (("RR1", base, "RR1"), ("p3_frozen", base, "RR1->RR1"),
                                 ("p4_selected", same, "RR1->RR8" if i % 3 == 0 else "RR1->RR1"),
                                 ("p4_ablation", same, "RR1->RR8" if i % 3 == 0 else "RR1->RR1"),
                                 ("submission_ref", base, "RR1"), ("oracle_fixed", same, "RR8"),
                                 ("oracle_online", same, "RR1->RR8")):
            rows.append({"workload_id": f"w{i}", "method": method, "policy": pol, "decision_source": "x",
                         "t_decision": 1, "avg_turnaround": val, "avg_response": 1.0, "avg_waiting": val - 1,
                         "throughput": 0.1, "cpu_utilization": 0.9, "context_switches": 5})
    long = pd.DataFrame(rows)
    info = pd.DataFrame({"workload_class": ["a", "b"] * 20, "n_processes": range(40), "burst_cv": np.linspace(0.3, 1.2, 40)},
                        index=[f"w{i}" for i in range(40)])
    cfg = {"hypotheses": {"H1": ["p4_selected", "p3_frozen"], "H2": ["p4_selected", "RR1"],
                          "H3": ["p4_ablation", "p3_frozen"]}, "primary_hypothesis": "H1",
           "statistics": {"signflip": {"exact_max_nonzero": 26, "monte_carlo_draws": 2000, "seed": 1},
                          "bootstrap": {"n": 500, "seed": 2}, "breakdown_bootstrap_n": 200, "alpha": 0.05}}
    r = analyze(long, info, cfg, q0=1)
    fam = r["primary_family"]
    # H1 and H2 compare the same decision vectors (p3_frozen == RR1 here); H3 equals H1 too.
    assert set(fam) == {"H1"} and set(fam["H1"]["merged_with"]) == {"H2", "H3"}
    assert fam["H1"]["holm_p"] == fam["H1"]["signflip_p"]  # family of one after merging
    assert fam["H1"]["supported_improvement"] in (True, False)
    assert r["quantum_changes"]["p4_selected"]["share_quantum_changed"] == pytest.approx(14 / 40)
