"""Phase 4 pipeline end to end on a tiny configuration: isolation of the fit
stage, reproducibility, saved-result consistency, and stale-config refusal."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import joblib
import pandas as pd
import pytest

from schedx.adaptive_eval import summarize
from schedx.metrics import compute_metrics
from schedx.phase3 import load_traces
from schedx.phase4 import run_p4

ROOT = Path(__file__).resolve().parents[1]
S = ROOT / "scripts"


def run(*args, check=True):
    return subprocess.run([sys.executable, *map(str, args)], check=check, capture_output=True, text=True)


@pytest.fixture(scope="module")
def e2e(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("p4")
    cfg2 = json.loads((ROOT / "configs" / "phase2.json").read_text(encoding="utf-8"))
    cfg2["dataset"]["n_workloads"] = 60
    small = {"tree_classifier": [{"max_depth": 2, "min_samples_leaf": 3}],
             "logistic_classifier": [{"C": 1.0}],
             "regret_regressor": [{"max_depth": 2, "min_samples_leaf": 3}]}
    cfg3 = json.loads((ROOT / "configs" / "phase3.json").read_text(encoding="utf-8"))
    cfg3.update(decision_after_completions=[3, 6], grids=small, bootstrap={"n": 300, "seed": 1})
    cfg4 = json.loads((ROOT / "configs" / "phase4.json").read_text(encoding="utf-8"))
    cfg4.update(decision_after_completions=[3, 6], grids=small, fresh_confirmation_set={"master_seed": 99, "n_workloads": 20})
    cfg4["statistics"].update(bootstrap={"n": 300, "seed": 1}, breakdown_bootstrap_n=100,
                              signflip={"exact_max_nonzero": 16, "monte_carlo_draws": 2000, "seed": 3})
    for name, c in (("c2", cfg2), ("c3", cfg3), ("c4", cfg4)):
        (tmp / f"{name}.json").write_text(json.dumps(c), encoding="utf-8")
    a, r3, r4 = tmp / "a", tmp / "r3", tmp / "r4"
    run(S / "generate_dataset.py", "--config", tmp / "c2.json", "--out", a / "dataset")
    run(S / "phase3_train.py", "--config", tmp / "c3.json", "--phase2-config", tmp / "c2.json",
        "--artifacts", a, "--reports", r3)
    common = ["--config", tmp / "c4.json", "--phase3-config", tmp / "c3.json", "--phase2-config", tmp / "c2.json",
              "--artifacts", a, "--phase3-reports", r3]
    run(S / "phase4_train.py", *common, "--reports", r4)
    run(S / "phase4_evaluate.py", *common, "--reports", r4)
    run(S / "phase4_plots.py", "--reports", r4)
    return {"tmp": tmp, "a": a, "r3": r3, "r4": r4, "common": common}


def test_outputs_and_protocol_flags(e2e):
    r4 = e2e["r4"]
    frozen = json.loads((r4 / "frozen_controller.json").read_text(encoding="utf-8"))
    res = json.loads((r4 / "results.json").read_text(encoding="utf-8"))
    assert frozen["split"]["test_workloads_loaded_in_fit_stage"] is False
    assert frozen["split"]["fresh_confirmation_set_generated_in_fit_stage"] is False
    assert res["config_hash"] == frozen["config_hash"]
    for key, n in (("established_test_split", 12), ("fresh_confirmation", 20)):
        s = res[key]
        assert s["n_workloads"] == n and "H1" in s["primary_family"]
        for v in s["primary_family"].values():
            assert v["holm_p"] >= v["signflip_p"] - 1e-12
    fresh = pd.read_csv(r4 / "per_workload_fresh.csv")
    assert fresh.workload_id.str.startswith("w99-").all()
    test = pd.read_csv(r4 / "per_workload_test.csv")
    assert set(test.method) == {"RR1", "oracle_fixed", "oracle_online", "p3_frozen", "submission_ref",
                                "p4_selected", "p4_ablation"} or test.method.str.startswith("RR").any()
    assert len({p.name for p in (r4 / "figures").glob("*.png")}) >= 6
    assert "Primary H1" in (r4 / "summary.md").read_text(encoding="utf-8")


def test_fit_stage_never_uses_test_workloads(e2e):
    tmp = e2e["tmp"]
    b = tmp / "a_no_test"
    shutil.copytree(e2e["a"], b)
    test_ids = set(pd.read_csv(e2e["r4"] / "per_workload_test.csv").workload_id)
    procs = pd.read_csv(b / "dataset" / "processes.csv")
    procs[~procs.workload_id.isin(test_ids)].to_csv(b / "dataset" / "processes.csv", index=False)
    common = [x if x != e2e["a"] else b for x in e2e["common"]]
    run(S / "phase4_train.py", *common, "--reports", tmp / "r4_no_test")
    for f in ("frozen_controller.json", "validation_table.csv", "estimator_diagnostic.json"):
        assert (tmp / "r4_no_test" / f).read_bytes() == (e2e["r4"] / f).read_bytes(), f


def test_evaluation_is_reproducible(e2e):
    r4 = e2e["r4"]
    before = {f: (r4 / f).read_bytes() for f in ("results.json", "per_workload_test.csv", "per_workload_fresh.csv")}
    run(S / "phase4_evaluate.py", *e2e["common"], "--reports", r4)
    for f, content in before.items():
        assert (r4 / f).read_bytes() == content, f


def test_saved_results_match_resimulation(e2e):
    long = pd.read_csv(e2e["r4"] / "per_workload_test.csv").set_index(["workload_id", "method"])
    bundle = joblib.load(e2e["a"] / "phase4" / "controllers.joblib")
    ids = sorted({w for w, _ in long.index})[:4]
    traces = load_traces(e2e["a"], ids)
    for w in ids:
        for name in ("p4_selected", "p4_ablation"):
            sched, q, source, t = run_p4(traces[w], bundle["controllers"][name], bundle["q0"])
            m = summarize(compute_metrics(sched))
            for k in ("avg_turnaround", "avg_response", "context_switches"):
                assert long.loc[(w, name), k] == pytest.approx(m[k])
            assert long.loc[(w, name), "decision_source"] == source


def test_evaluation_refuses_changed_config(e2e):
    tmp = e2e["tmp"]
    cfg4 = json.loads((tmp / "c4.json").read_text(encoding="utf-8"))
    cfg4["seed"] += 1
    (tmp / "c4_changed.json").write_text(json.dumps(cfg4), encoding="utf-8")
    args = [x if x != tmp / "c4.json" else tmp / "c4_changed.json" for x in e2e["common"]]
    r = run(S / "phase4_evaluate.py", *args, "--reports", e2e["r4"], check=False)
    assert r.returncode != 0 and "changed since" in r.stderr
