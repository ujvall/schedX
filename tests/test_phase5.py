"""Phase 5 safeguards: operating-point assignment, seed uniqueness, input pairing,
configuration recording, output schemas, and determinism."""

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from schedx import phase5
from schedx.controller import QuantumController
from schedx.metrics import compute_metrics
from schedx.policy_eval import burst_jobs
from schedx.scheduler import round_robin
from schedx.workload import config_dict, generate_dataset, GeneratorConfig

ROOT = Path(__file__).resolve().parents[1]
CFG5 = json.loads((ROOT / "configs" / "phase5.json").read_text(encoding="utf-8"))
CFG2 = json.loads((ROOT / "configs" / "phase2.json").read_text(encoding="utf-8"))


def tiny_cfg(n=4, seeds=(900101, 900102, 900103)):
    cfg = json.loads(json.dumps(CFG5))
    for (name, op), s in zip(cfg["operating_points"].items(), seeds):
        op.update(n_workloads=n, master_seed=s)
    cfg["statistics"].update(bootstrap={"n": 200, "seed": 1},
                             signflip={"exact_max_nonzero": 12, "monte_carlo_draws": 500, "seed": 2})
    return cfg


# --- Operating points and seeds ---------------------------------------------------------


def test_config_operating_points_are_distinct_and_within_training_support():
    ops = CFG5["operating_points"]
    sizes = [ops[k]["n_processes"] for k in phase5.OP_ORDER]
    assert sizes == sorted(sizes) and len(set(sizes)) == 3
    lo, hi = CFG2["generator"]["process_count"]
    assert all(lo <= n <= hi for n in sizes)  # frozen controllers were trained on this range
    seeds = [ops[k]["master_seed"] for k in phase5.OP_ORDER]
    assert len(set(seeds)) == 3
    assert not set(seeds) & {CFG2["dataset"]["master_seed"], 2027, 12345}
    phase5.check_seeds(CFG5)


def test_check_seeds_rejects_reuse():
    bad = json.loads(json.dumps(CFG5))
    bad["operating_points"]["large"]["master_seed"] = bad["operating_points"]["small"]["master_seed"]
    with pytest.raises(ValueError, match="distinct"):
        phase5.check_seeds(bad)
    bad = json.loads(json.dumps(CFG5))
    bad["operating_points"]["small"]["master_seed"] = 2026
    with pytest.raises(ValueError, match="already used"):
        phase5.check_seeds(bad)


def test_generator_changes_only_the_process_count():
    g = phase5.generator_for(CFG2["generator"], 14)
    base = config_dict(GeneratorConfig.from_dict(CFG2["generator"]))
    new = config_dict(g)
    assert new.pop("process_count") == [14, 14]
    base.pop("process_count")
    assert new == base
    assert {len(w.processes) for w in generate_dataset(10, 900001, g)} == {14}


def test_manifest_assignment_and_identities():
    ops = phase5.generate_operating_points(tiny_cfg(), CFG2["generator"])
    man = phase5.manifest(ops)
    for name in phase5.OP_ORDER:
        sub = man[man.operating_point == name]
        op = tiny_cfg()["operating_points"][name]
        assert len(sub) == op["n_workloads"]
        assert (sub.n_processes == op["n_processes"]).all()
        assert (sub.master_seed == op["master_seed"]).all()
        assert sub.workload_id.str.startswith(f"w{op['master_seed']}-").all()
    assert man.workload_id.is_unique and man.trace_sha256.is_unique
    again = phase5.manifest(phase5.generate_operating_points(tiny_cfg(), CFG2["generator"]))
    pd.testing.assert_frame_equal(man, again)  # deterministic regeneration


def test_hashes_detect_any_input_change():
    w = generate_dataset(1, 900002)[0]
    p0 = w.processes[0]
    changed = (replace(p0, cpu_bursts=(p0.cpu_bursts[0] + 1, *p0.cpu_bursts[1:])), *w.processes[1:])
    assert phase5.trace_hash(changed) != phase5.trace_hash(w.processes)
    j = burst_jobs(w.processes)
    assert phase5.jobs_hash(burst_jobs(changed)) != phase5.jobs_hash(j)
    assert phase5.jobs_hash(round_robin(j, 2).processes) == phase5.jobs_hash(j)


# --- Pairing and evaluation ---------------------------------------------------------------


def test_every_method_schedules_the_identical_job_list():
    c = QuantumController("online", "constant", (1, 2, 4, 8), 1, after_completions=5)
    for w in generate_dataset(6, 900003, phase5.generator_for(CFG2["generator"], 6)):
        rows = phase5.evaluate_workload(w, c, c, 1, [1, 2, 4, 8])
        assert [r["method"] for r in rows] == ["RR1", "RR2", "RR4", "RR8", "oracle_fixed", "p3_frozen", "p4_selected"]
        expected = phase5.jobs_hash(burst_jobs(w.processes))
        assert {r["input_jobs_sha256"] for r in rows} == {expected}
        by = {r["method"]: r for r in rows}
        assert by["RR2"]["avg_turnaround"] == compute_metrics(round_robin(burst_jobs(w.processes), 2)).avg_turnaround
        assert by["oracle_fixed"]["avg_turnaround"] == min(by[f"RR{q}"]["avg_turnaround"] for q in (1, 2, 4, 8))
        assert by["p3_frozen"]["avg_turnaround"] == by["RR1"]["avg_turnaround"]  # constant controller keeps q0


def test_classification_rule():
    c = lambda p, lo, hi: {"holm_p": p, "geo_rel_change_ci95": [lo, hi]}
    assert phase5.classify(c(0.001, -0.03, -0.012), 0.05, 0.01) == "practically meaningful improvement"
    assert phase5.classify(c(0.001, -0.02, -0.002), 0.05, 0.01).startswith("statistically supported improvement below")
    assert phase5.classify(c(0.5, -0.004, 0.003), 0.05, 0.01) == "practically negligible difference"
    assert phase5.classify(c(0.2, -0.02, 0.004), 0.05, 0.01) == "inconclusive"
    assert phase5.classify(c(0.001, 0.002, 0.02), 0.05, 0.01) == "inconclusive"  # significant but worse


# --- End to end with the real frozen controllers on throwaway seeds ------------------------------

needs_artifacts = pytest.mark.skipif(
    not (ROOT / "artifacts" / "phase4" / "controllers.joblib").exists(), reason="frozen controller artifacts not built")


@pytest.fixture(scope="module")
def tiny_run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("p5")
    (tmp / "c5.json").write_text(json.dumps(tiny_cfg()), encoding="utf-8")
    out = []
    for i in (1, 2):
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / "phase5_run.py"), "--config", str(tmp / "c5.json"),
                            "--reports", str(tmp / f"r{i}")], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        out.append(tmp / f"r{i}")
    return tmp, out


@needs_artifacts
def test_output_schemas_and_config_recording(tiny_run):
    _, (r, _) = tiny_run
    raw = pd.read_csv(r / "raw_workload_results.csv")
    man = pd.read_csv(r / "workload_manifest.csv")
    assert list(raw.columns) == ["workload_id", "operating_point", "method", "policy", "decision_source", "t_decision",
                                 "input_jobs_sha256", "avg_turnaround", "avg_waiting", "avg_response", "throughput",
                                 "cpu_utilization", "context_switches"]
    assert len(raw) == len(man) * 7 == 3 * 4 * 7
    # Pairing: one input hash per workload, equal to the manifest's.
    per = raw.groupby("workload_id").input_jobs_sha256.agg(["nunique", "first"])
    assert (per["nunique"] == 1).all()
    assert (per["first"] == man.set_index("workload_id").loc[per.index, "jobs_sha256"]).all()
    env = json.loads((r / "environment.json").read_text(encoding="utf-8"))
    for key in ("python", "library_versions", "generator_base", "phase5_config_sha256", "frozen_controllers", "operating_points"):
        assert key in env
    assert env["frozen_controllers"]["p3_frozen"]["config_hash"] == CFG5["frozen_controllers"]["p3_frozen"]["config_hash"]
    comp = pd.read_csv(r / "comparisons.csv")
    assert set(comp.family) == {"primary", "secondary"}
    for name in phase5.OP_ORDER:
        assert (r / "inputs" / f"processes_{name}.csv.gz").exists()
    assert len(pd.read_csv(r / "inputs" / "processes_large.csv.gz")) == 4 * CFG5["operating_points"]["large"]["n_processes"]


@needs_artifacts
def test_run_is_byte_reproducible(tiny_run):
    _, (r1, r2) = tiny_run
    files = sorted(p.relative_to(r1) for p in r1.rglob("*") if p.is_file())
    assert files
    for f in files:
        assert (r1 / f).read_bytes() == (r2 / f).read_bytes(), f


@needs_artifacts
def test_run_refuses_mismatched_frozen_hash(tiny_run):
    tmp, _ = tiny_run
    cfg = tiny_cfg()
    cfg["frozen_controllers"]["p3_frozen"]["config_hash"] = "000000000000"
    (tmp / "bad.json").write_text(json.dumps(cfg), encoding="utf-8")
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "phase5_run.py"), "--config", str(tmp / "bad.json"),
                        "--reports", str(tmp / "rbad")], capture_output=True, text=True)
    assert r.returncode != 0 and "frozen controllers" in r.stderr
