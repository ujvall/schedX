"""Phase 3 pipeline: fit/test separation, reproducibility, saved-result consistency,
and regression protection for Phase 1/2 artifacts."""

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from schedx.adaptive_eval import run_controller, summarize
from schedx.features import build_feature_table
from schedx.metrics import compute_metrics
from schedx.phase3 import choose_default, load_traces
from schedx.policy_eval import burst_jobs
from schedx.scheduler import round_robin
from schedx.workload import dataset_version, generate_dataset, to_frames

ROOT = Path(__file__).resolve().parents[1]
S = ROOT / "scripts"


def run(*args, check=True):
    return subprocess.run([sys.executable, *map(str, args)], check=check, capture_output=True, text=True)


# --- Unit ------------------------------------------------------------------------------------


def test_choose_default_uses_mean_log_regret_and_breaks_ties_small():
    T = pd.DataFrame({1: [10.0, 30.0], 2: [20.0, 15.0], 4: [12.0, 18.0]}, index=["a", "b"])
    q, table = choose_default(T)
    # mean log regret: q1 (0 + log2)/2=.347, q2 (log2 + 0)/2=.347, q4 (log1.2 + log1.2)/2=.182
    assert q == 4 and table[4] == pytest.approx(np.log(1.2))
    tie = pd.DataFrame({1: [1.0, 2.0], 2: [2.0, 1.0]}, index=["a", "b"])
    assert choose_default(tie)[0] == 1


# --- End-to-end on a tiny configuration ----------------------------------------------------------


@pytest.fixture(scope="module")
def e2e(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("p3")
    cfg2 = json.loads((ROOT / "configs" / "phase2.json").read_text(encoding="utf-8"))
    cfg2["dataset"]["n_workloads"] = 60
    cfg3 = json.loads((ROOT / "configs" / "phase3.json").read_text(encoding="utf-8"))
    cfg3["decision_after_completions"] = [3, 6]
    cfg3["grids"] = {"tree_classifier": [{"max_depth": 2, "min_samples_leaf": 3}],
                     "logistic_classifier": [{"C": 1.0}],
                     "regret_regressor": [{"max_depth": 2, "min_samples_leaf": 3}]}
    cfg3["bootstrap"] = {"n": 500, "seed": 1}
    (tmp / "c2.json").write_text(json.dumps(cfg2), encoding="utf-8")
    (tmp / "c3.json").write_text(json.dumps(cfg3), encoding="utf-8")
    a, r2, r3 = tmp / "a", tmp / "r2", tmp / "r3"
    run(S / "generate_dataset.py", "--config", tmp / "c2.json", "--out", a / "dataset")
    run(S / "train_evaluate.py", "--config", tmp / "c2.json", "--artifacts", a, "--reports", r2)
    common = ["--config", tmp / "c3.json", "--phase2-config", tmp / "c2.json", "--artifacts", a]
    run(S / "phase3_train.py", *common, "--reports", r3)
    run(S / "phase3_evaluate.py", *common, "--phase2-reports", r2, "--reports", r3)
    run(S / "phase3_plots.py", "--reports", r3)
    return {"tmp": tmp, "a": a, "r2": r2, "r3": r3, "cfg3": cfg3, "common": common}


def test_pipeline_outputs(e2e):
    r3 = e2e["r3"]
    frozen = json.loads((r3 / "frozen_controller.json").read_text(encoding="utf-8"))
    res = json.loads((r3 / "results.json").read_text(encoding="utf-8"))
    assert frozen["frozen"] and frozen["split"]["test_workloads_loaded_in_fit_stage"] is False
    assert res["config_hash"] == frozen["config_hash"]
    assert res["primary"]["comparison"] == "adaptive_selected vs best_fixed"
    assert res["n_test_workloads"] == frozen["split"]["test_workloads"] == 12
    for c in res["deployable_comparisons"].values():
        assert 0 <= c["holm_p"] <= 1 and c["holm_p"] >= c["wilcoxon_p"] - 1e-12
    table = pd.read_csv(r3 / "validation_table.csv")
    assert table["selected"].sum() == 1
    assert table.loc[table.selected, "val_mean_log_ratio"].item() == table["val_mean_log_ratio"].min()
    assert len({p.name for p in (r3 / "figures").glob("*.png")}) == 6
    assert "statistically supported" in (r3 / "summary.md").read_text(encoding="utf-8")


def test_fit_stage_never_uses_test_workloads(e2e):
    """Deleting every test workload's traces must not change the frozen controller."""
    tmp, a = e2e["tmp"], e2e["a"]
    b = tmp / "a_no_test"
    shutil.copytree(a, b)
    res_split = pd.read_csv(e2e["r3"] / "per_workload_test.csv")
    test_ids = set(res_split.workload_id)
    procs = pd.read_csv(b / "dataset" / "processes.csv")
    procs[~procs.workload_id.isin(test_ids)].to_csv(b / "dataset" / "processes.csv", index=False)
    run(S / "phase3_train.py", "--config", tmp / "c3.json", "--phase2-config", tmp / "c2.json",
        "--artifacts", b, "--reports", tmp / "r3_no_test")
    for f in ("frozen_controller.json", "validation_table.csv"):
        assert (tmp / "r3_no_test" / f).read_bytes() == (e2e["r3"] / f).read_bytes(), f


def test_evaluation_is_reproducible(e2e):
    before = (e2e["r3"] / "results.json").read_bytes()
    before_csv = (e2e["r3"] / "per_workload_test.csv").read_bytes()
    run(S / "phase3_evaluate.py", *e2e["common"], "--phase2-reports", e2e["r2"], "--reports", e2e["r3"])
    assert (e2e["r3"] / "results.json").read_bytes() == before
    assert (e2e["r3"] / "per_workload_test.csv").read_bytes() == before_csv


def test_saved_results_match_recomputed_metrics(e2e):
    long = pd.read_csv(e2e["r3"] / "per_workload_test.csv")
    bundle = joblib.load(e2e["a"] / "phase3" / "controllers.joblib")
    q0, ctrl = bundle["q0"], bundle["controllers"]["selected"]
    ids = sorted(long.workload_id.unique())[:4]
    traces = load_traces(e2e["a"], ids)
    T = long.set_index(["workload_id", "method"])
    for w in ids:
        jobs = burst_jobs(traces[w])
        fixed = {q: summarize(compute_metrics(round_robin(jobs, q))) for q in e2e["cfg3"]["candidates"]}
        assert T.loc[(w, "best_fixed"), "avg_turnaround"] == pytest.approx(fixed[q0]["avg_turnaround"])
        assert T.loc[(w, "oracle_fixed"), "avg_turnaround"] == pytest.approx(min(v["avg_turnaround"] for v in fixed.values()))
        sched, q, source, t = run_controller(traces[w], ctrl.decide, ctrl.regime, q0, ctrl.after_completions)
        m = summarize(compute_metrics(sched))
        for k in ("avg_turnaround", "avg_waiting", "avg_response", "context_switches"):
            assert T.loc[(w, "adaptive_selected"), k] == pytest.approx(m[k])
        assert T.loc[(w, "adaptive_selected"), "decision_source"] == source


def test_evaluation_refuses_changed_config(e2e):
    tmp = e2e["tmp"]
    cfg3 = dict(e2e["cfg3"], seed=e2e["cfg3"]["seed"] + 1)
    (tmp / "c3_changed.json").write_text(json.dumps(cfg3), encoding="utf-8")
    r = run(S / "phase3_evaluate.py", "--config", tmp / "c3_changed.json", "--phase2-config", tmp / "c2.json",
            "--artifacts", e2e["a"], "--phase2-reports", e2e["r2"], "--reports", e2e["r3"], check=False)
    assert r.returncode != 0 and "changed since" in r.stderr


# --- Regression protection for earlier phases -------------------------------------------------------


def test_phase2_dataset_is_unchanged():
    cfg2 = json.loads((ROOT / "configs" / "phase2.json").read_text(encoding="utf-8"))
    assert dataset_version(cfg2) == "67da9d294be3"
    ws = generate_dataset(5, 2026)
    p, _ = to_frames(ws)
    f = build_feature_table(ws, [1, 2, 3])
    assert hashlib.sha256(p.to_csv(index=False).encode()).hexdigest()[:16] == "98f371c8951d4da6"
    assert hashlib.sha256(f.to_csv(index=False).encode()).hexdigest()[:16] == "aa3465a4533be73a"


def test_committed_phase2_report_matches_dataset_and_selection():
    m = json.loads((ROOT / "reports" / "phase2" / "metrics.json").read_text(encoding="utf-8"))
    assert m["dataset_version"] == "67da9d294be3"
    assert m["main_experiment"]["selected_model"] == "logistic_regression"
    assert m["policy_experiment"]["class_policy_map"] == {"cpu_bound": "FCFS", "mixed": "RR2", "io_bound": "RR4"}


def test_committed_audit_matches_committed_phase3_results():
    """The post-hoc audit artifact must agree with the committed evidence it audits."""
    audit = json.loads((ROOT / "reports" / "phase3" / "audit" / "audit_results.json").read_text(encoding="utf-8"))
    res = json.loads((ROOT / "reports" / "phase3" / "results.json").read_text(encoding="utf-8"))
    assert audit["inputs"]["results_json_config_hash"] == res["config_hash"]
    assert all(audit["integrity"][k] for k in ("ids_equal_grouped_split_test_ids", "config_hash_results_equals_frozen"))
    assert audit["integrity"]["ids_overlapping_train_or_validation"] == 0
    for key in ("primary", "submission_vs_best_fixed", "selected_vs_fixed_RR2"):
        c = audit["committed_vs_recomputed"][key]
        assert c["geo_rel_change_matches"] and c["wilcoxon_p_matches"] and c["mean_abs_diff_matches"]
    p = audit["comparisons"]["primary"]
    assert p["tests"]["wilcoxon_pratt_scipy_prespecified"] == res["primary"]["wilcoxon_p"]
    assert p["estimates"]["geometric_mean_per_workload_change"] == res["primary"]["geo_rel_change"]
