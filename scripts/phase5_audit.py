"""Phase 5 robustness and reproducibility audit (read-only on committed results).

Checks seed and identity uniqueness, disjointness from every earlier dataset,
regeneration of the committed inputs, input pairing across methods,
work-conservation invariants, and an independent recomputation of the
primary statistics (SciPy permutation_test instead of schedx.stats).

Usage:  python scripts/phase5_audit.py [--reports reports/phase5] [--artifacts artifacts]

Writes  <reports>/robustness_checks.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from schedx import phase5  # noqa: E402
from schedx.workload import GeneratorConfig, generate_dataset, processes_from_frame  # noqa: E402

METHODS = {"RR1", "RR2", "RR4", "RR8", "oracle_fixed", "p3_frozen", "p4_selected"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", default=str(ROOT / "reports" / "phase5"))
    ap.add_argument("--artifacts", default=str(ROOT / "artifacts"))
    ap.add_argument("--config", default=str(ROOT / "configs" / "phase5.json"))
    args = ap.parse_args()
    reports, artifacts = Path(args.reports), Path(args.artifacts)
    cfg5 = json.loads(Path(args.config).read_text(encoding="utf-8"))
    cfg2 = json.loads((ROOT / "configs" / "phase2.json").read_text(encoding="utf-8"))
    man = pd.read_csv(reports / "workload_manifest.csv")
    raw = pd.read_csv(reports / "raw_workload_results.csv")
    res = json.loads((reports / "results.json").read_text(encoding="utf-8"))
    checks = {}

    # 1. Seeds, operating-point assignment, identities.
    ops = cfg5["operating_points"]
    checks["seeds_distinct_and_unreserved"] = (
        len({o["master_seed"] for o in ops.values()}) == 3
        and not {o["master_seed"] for o in ops.values()} & set(cfg5["reserved_seeds_not_to_reuse"]))
    checks["operating_point_assignment"] = all(
        ((man.operating_point == k) == (man.master_seed == o["master_seed"])).all()
        and (man.loc[man.operating_point == k, "n_processes"] == o["n_processes"]).all()
        and (man.operating_point == k).sum() == o["n_workloads"] for k, o in ops.items())
    checks["workload_ids_unique"] = bool(man.workload_id.is_unique)
    checks["trace_hashes_unique_within_phase5"] = bool(man.trace_sha256.is_unique)

    # 2. Disjoint from every earlier dataset (training, validation, test, fresh).
    earlier = processes_from_frame(pd.read_csv(artifacts / "dataset" / "processes.csv"))
    fresh = generate_dataset(600, 2027, GeneratorConfig.from_dict(cfg2["generator"]))
    earlier_hashes = {phase5.trace_hash(p) for p in earlier.values()} | {phase5.trace_hash(w.processes) for w in fresh}
    checks["earlier_datasets_hashed"] = {"phase2_train_validation_test": len(earlier), "phase4_fresh": len(fresh)}
    checks["no_overlap_with_earlier_datasets"] = not set(man.trace_sha256) & earlier_hashes
    checks["no_id_overlap_with_earlier_datasets"] = not set(man.workload_id) & (set(earlier) | {w.workload_id for w in fresh})

    # 3. Regeneration reproduces the committed manifest and input traces.
    regen = phase5.generate_operating_points(cfg5, cfg2["generator"])
    man2 = phase5.manifest(regen)
    checks["manifest_regenerates_exactly"] = bool(
        man2[["workload_id", "trace_sha256", "jobs_sha256"]].equals(man[["workload_id", "trace_sha256", "jobs_sha256"]]))
    inputs_ok = True
    for name in phase5.OP_ORDER:
        committed = pd.read_csv(reports / "inputs" / f"processes_{name}.csv.gz", keep_default_na=False)
        again = phase5.processes_frame(regen[name]).astype(str)
        inputs_ok &= committed.astype(str).reset_index(drop=True).equals(again.reset_index(drop=True))
    checks["input_traces_regenerate_exactly"] = bool(inputs_ok)

    # 4. Pairing and invariants in the raw results.
    per = raw.groupby("workload_id")
    checks["every_workload_has_all_methods_once"] = bool(
        (per.method.nunique() == len(METHODS)).all() and (per.size() == len(METHODS)).all()
        and set(raw.method) == METHODS and set(raw.workload_id) == set(man.workload_id))
    jobs = per.input_jobs_sha256.agg(["nunique", "first"])
    checks["identical_inputs_across_methods"] = bool(
        (jobs["nunique"] == 1).all() and (jobs["first"] == man.set_index("workload_id").loc[jobs.index, "jobs_sha256"]).all())
    checks["throughput_and_utilization_identical_across_methods"] = bool(
        (per.throughput.nunique() == 1).all() and (per.cpu_utilization.nunique() == 1).all())
    T = raw.pivot(index="workload_id", columns="method", values="avg_turnaround")
    checks["oracle_is_minimum_of_fixed_quanta"] = bool(np.allclose(T.oracle_fixed, T[["RR1", "RR2", "RR4", "RR8"]].min(axis=1)))

    # 5. Independent recomputation of the primary family.
    rng_seed = 2024
    recomputed, worst = {}, {"geo": 0.0, "agg": 0.0, "p": 0.0, "ci": 0.0}
    for h, (a, b) in cfg5["primary_hypotheses"].items():
        op = h.rsplit("_", 1)[1]
        ids = man.loc[man.operating_point == op, "workload_id"]
        m, base = T.loc[ids, a].to_numpy(), T.loc[ids, b].to_numpy()
        lr = np.log(m / base)
        geo, agg = float(np.expm1(lr.mean())), float(m.mean() / base.mean() - 1)
        perm = stats.permutation_test((np.log(m), np.log(base)), lambda x, y, axis: np.mean(x - y, axis=axis),
                                      permutation_type="samples", n_resamples=20_000, vectorized=True,
                                      alternative="two-sided", random_state=rng_seed)
        boot = stats.bootstrap((lr,), np.mean, n_resamples=20_000, method="percentile", random_state=rng_seed)
        ci = [float(np.expm1(boot.confidence_interval.low)), float(np.expm1(boot.confidence_interval.high))]
        c = res["primary_family"][h]
        recomputed[h] = {"geo": geo, "aggregate": agg, "scipy_permutation_p": float(perm.pvalue), "scipy_bootstrap_ci": ci,
                         "committed_geo": c["geo_rel_change"], "committed_signflip_p": c["signflip_p"],
                         "committed_ci": c["geo_rel_change_ci95"]}
        worst["geo"] = max(worst["geo"], abs(geo - c["geo_rel_change"]))
        worst["agg"] = max(worst["agg"], abs(agg - c["change_in_mean_turnaround"]))
        worst["p"] = max(worst["p"], abs(perm.pvalue - c["signflip_p"]))
        worst["ci"] = max(worst["ci"], *(abs(x - y) for x, y in zip(ci, c["geo_rel_change_ci95"])))
    raw_p = {h: v["signflip_p"] for h, v in res["primary_family"].items()}
    names = sorted(raw_p, key=raw_p.get)
    holm, run = {}, 0.0
    for i, n in enumerate(names):
        run = max(run, min(1.0, (len(names) - i) * raw_p[n]))
        holm[n] = run
    checks["primary_recomputation"] = recomputed
    checks["primary_max_abs_differences"] = worst
    checks["primary_point_estimates_match"] = worst["geo"] < 1e-12 and worst["agg"] < 1e-12
    # Monte Carlo tolerances: p-values within 0.01; bootstrap CI bounds within 0.1 percentage point.
    checks["primary_pvalues_match_within_monte_carlo_error"] = worst["p"] < 0.01
    checks["primary_cis_match_within_monte_carlo_error"] = worst["ci"] < 0.001
    checks["holm_recomputed_matches"] = all(abs(holm[h] - res["primary_family"][h]["holm_p"]) < 1e-12 for h in holm)
    # Coerce NumPy booleans so no check is silently skipped by the isinstance filter below.
    checks = {k: (bool(v) if isinstance(v, np.bool_) else v) for k, v in checks.items()}
    checks["all_pass"] = all(v for k, v in checks.items() if isinstance(v, bool))

    (reports / "robustness_checks.json").write_text(json.dumps(checks, indent=2, default=bool), encoding="utf-8")
    for k, v in checks.items():
        if isinstance(v, bool):
            print(f"{'PASS' if v else 'FAIL'}  {k}")
    print("max differences:", {k: f"{v:.2e}" for k, v in worst.items()})
    if not checks["all_pass"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
