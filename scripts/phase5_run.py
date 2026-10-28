"""Phase 5 principal experiment: generate the three operating points and evaluate
every method once with the FROZEN Phase 3 and Phase 4 controllers.

Usage:  python scripts/phase5_run.py [--config configs/phase5.json] [--artifacts artifacts]
                                     [--reports reports/phase5]

Writes  <reports>/workload_manifest.csv          seed manifest (one row per workload)
        <reports>/inputs/processes_<op>.csv.gz   raw input traces (one row per process)
        <reports>/raw_workload_results.csv       one row per (workload, method)
        <reports>/operating_point_summary.csv    means per (operating point, method)
        <reports>/comparisons.csv                every hypothesis test
        <reports>/results.json, environment.json, summary.md
"""

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from schedx import phase3, phase4, phase5  # noqa: E402
from schedx.adaptive_eval import METRICS  # noqa: E402
from schedx.ml import library_versions  # noqa: E402

GZ = {"method": "gzip", "mtime": 0}  # fixed header timestamp: byte-reproducible archives


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "phase5.json"))
    ap.add_argument("--phase2-config", default=str(ROOT / "configs" / "phase2.json"))
    ap.add_argument("--phase3-config", default=str(ROOT / "configs" / "phase3.json"))
    ap.add_argument("--phase4-config", default=str(ROOT / "configs" / "phase4.json"))
    ap.add_argument("--artifacts", default=str(ROOT / "artifacts"))
    ap.add_argument("--reports", default=str(ROOT / "reports" / "phase5"))
    args = ap.parse_args()
    load = lambda p: json.loads(Path(p).read_text(encoding="utf-8"))
    cfg5, cfg2, cfg3, cfg4 = (load(p) for p in (args.config, args.phase2_config, args.phase3_config, args.phase4_config))
    artifacts, reports = Path(args.artifacts), Path(args.reports)

    # Frozen controllers must be exactly the committed ones.
    _, _, meta = phase3.load_split(artifacts, cfg2)
    h3 = phase3.config_hash(cfg3, meta["dataset_version"])
    h4 = phase4.config_hash(cfg4, meta["dataset_version"], h3)
    b3 = joblib.load(artifacts / "phase3" / "controllers.joblib")
    b4 = joblib.load(artifacts / "phase4" / "controllers.joblib")
    fc = cfg5["frozen_controllers"]
    if not (h3 == b3["config_hash"] == fc["p3_frozen"]["config_hash"] and h4 == b4["config_hash"] == fc["p4_selected"]["config_hash"]):
        sys.exit("frozen controllers do not match the hashes pre-specified in configs/phase5.json")
    q0 = cfg5["default_quantum"]
    if not (b3["q0"] == b4["q0"] == q0):
        sys.exit("default quantum differs from the frozen controllers")
    p3 = b3["controllers"][fc["p3_frozen"]["key"]]
    p4 = b4["controllers"][fc["p4_selected"]["key"]]

    ops = phase5.generate_operating_points(cfg5, cfg2["generator"])
    man = phase5.manifest(ops)
    if man.workload_id.duplicated().any() or man.trace_sha256.duplicated().any():
        sys.exit("duplicate workload identity within Phase 5")

    rows = []
    for name in phase5.OP_ORDER:
        for w in ops[name]:
            rows += phase5.evaluate_workload(w, p3, p4, q0, cfg5["fixed_quanta"])
    raw = pd.DataFrame(rows)
    raw.insert(1, "operating_point", raw.workload_id.map(man.set_index("workload_id").operating_point))
    results = phase5.analyze(raw, man, cfg5)

    reports.mkdir(parents=True, exist_ok=True)
    (reports / "inputs").mkdir(exist_ok=True)
    man.to_csv(reports / "workload_manifest.csv", index=False)
    for name in phase5.OP_ORDER:
        phase5.processes_frame(ops[name]).to_csv(reports / "inputs" / f"processes_{name}.csv.gz", index=False, compression=GZ)
    raw.to_csv(reports / "raw_workload_results.csv", index=False)
    summary_table(raw, man, results).to_csv(reports / "operating_point_summary.csv", index=False)
    comparisons_table(results).to_csv(reports / "comparisons.csv", index=False)
    env = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "library_versions": {**library_versions(), **{m: __import__(m).__version__ for m in ("scipy", "matplotlib", "joblib")}},
        "dataset_version_phase2": meta["dataset_version"],
        "generator_base": cfg2["generator"],
        "phase5_config_sha256": hashlib.sha256(Path(args.config).read_bytes().replace(b"\r\n", b"\n")).hexdigest()[:12],
        "frozen_controllers": {"p3_frozen": {"config_hash": h3, "label": p3.describe()},
                               "p4_selected": {"config_hash": h4, "label": p4.describe()}},
        "operating_points": cfg5["operating_points"],
        "simulator": "schedx Round Robin engine, integer ticks, CPU-only burst-trace projection, zero-cost context switches",
    }
    (reports / "environment.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
    (reports / "results.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    (reports / "summary.md").write_text(render(results, env, cfg5), encoding="utf-8")
    print(f"wrote {reports}")


def summary_table(raw: pd.DataFrame, man: pd.DataFrame, results: dict) -> pd.DataFrame:
    rows = []
    for name in phase5.OP_ORDER:
        op = results["operating_points"][name]
        sub = raw[raw.operating_point == name]
        for meth, g in sub.groupby("method", sort=False):
            beh = op["controller_behaviour"].get(meth, {})
            rows.append({"operating_point": name, "method": meth, "n_workloads": len(g),
                         "n_processes_per_workload": op["n_processes_per_workload"],
                         "total_processes": op["total_processes"], "total_jobs": op["total_jobs"],
                         **{f"mean_{k}": g[k].mean() for k in METRICS},
                         "share_quantum_changed": beh.get("share_quantum_changed"),
                         "share_fallback": beh.get("share_fallback")})
    return pd.DataFrame(rows)


def comparisons_table(results: dict) -> pd.DataFrame:
    rows = []
    for fam in ("primary_family", "secondary_family"):
        for h, c in results[fam].items():
            a, b = c["hypothesis"].split(" vs ")
            rows.append({"family": fam.split("_")[0], "hypothesis": h, "operating_point": c["operating_point"],
                         "method": a, "comparator": b, "n_workloads": c["n"],
                         "geo_mean_change": c["geo_rel_change"], "geo_ci_low": c["geo_rel_change_ci95"][0],
                         "geo_ci_high": c["geo_rel_change_ci95"][1],
                         "aggregate_mean_change": c["change_in_mean_turnaround"],
                         "aggregate_ci_low": c["change_in_mean_turnaround_ci95"][0],
                         "aggregate_ci_high": c["change_in_mean_turnaround_ci95"][1],
                         "mean_diff_ticks": c["mean_abs_diff"], "share_better": c["share_better"],
                         "share_equal": c["share_equal"], "share_worse": c["share_worse"],
                         "signflip_p": c["signflip_p"], "signflip_method": c["signflip_method"],
                         "holm_p": c["holm_p"], "classification": c["classification"],
                         "merged_with": ";".join(c["merged_with"])})
    return pd.DataFrame(rows)


def pct(x: float) -> str:
    return f"{100 * x:+.2f}%"


def ci(c) -> str:
    return f"[{pct(c[0])}, {pct(c[1])}]"


def render(r: dict, env: dict, cfg5: dict) -> str:
    st = cfg5["statistics"]
    L = [
        "# Phase 5 results: three operating points (generated by scripts/phase5_run.py)",
        "",
        "Offline simulation (CPU-only burst-trace projection, integer ticks, free context switches). "
        "No kernel-level or hardware measurement. Frozen controllers only; nothing was trained or tuned in Phase 5.",
        "",
        f"Python {env['python']}; Phase 3 controller `{env['frozen_controllers']['p3_frozen']['config_hash']}`, "
        f"Phase 4 controller `{env['frozen_controllers']['p4_selected']['config_hash']}`; config `{env['phase5_config_sha256']}`.",
        "",
        "| operating point | processes / workload | workloads | total processes | total jobs | mean jobs / workload |",
        "|---|---|---|---|---|---|",
    ]
    for name in phase5.OP_ORDER:
        o = r["operating_points"][name]
        L.append(f"| {name} | {o['n_processes_per_workload']} | {o['n_workloads']} | {o['total_processes']} | "
                 f"{o['total_jobs']} | {o['mean_jobs_per_workload']:.1f} |")
    L += ["",
          "**Definitions.** *Geo change*: geometric mean over workloads of T_method / T_comparator, minus 1 "
          "(T = mean job turnaround of a workload; the pre-specified estimand). *Aggregate change*: "
          "mean(T_method) / mean(T_comparator) − 1, which weights long-running workloads more. Negative = lower "
          f"turnaround. Test: paired sign-flip permutation; CIs: paired bootstrap ({st['bootstrap']['n']:,}). "
          f"Supported = Holm p < {st['alpha']} and CI upper bound < 0; practically meaningful additionally requires "
          f"CI upper bound ≤ −{100 * st['sesoi_relative']:.0f}%; negligible = CI within ±{100 * st['sesoi_relative']:.0f}%.",
          ""]
    for fam, title in (("primary_family", "Primary family (Holm over 6)"), ("secondary_family", "Secondary family (Holm over 9)")):
        L += [f"## {title}", "",
              "| hypothesis | comparison | geo change | 95% CI | aggregate change | 95% CI | better/equal/worse | sign-flip p | Holm p | classification |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for h, c in r[fam].items():
            merged = f" (+{', '.join(c['merged_with'])})" if c["merged_with"] else ""
            L.append(f"| {h}{merged} | {c['hypothesis']} | {pct(c['geo_rel_change'])} | {ci(c['geo_rel_change_ci95'])} | "
                     f"{pct(c['change_in_mean_turnaround'])} | {ci(c['change_in_mean_turnaround_ci95'])} | "
                     f"{c['share_better']:.0%}/{c['share_equal']:.0%}/{c['share_worse']:.0%} | {c['signflip_p']:.3g} | "
                     f"{c['holm_p']:.3g} | {c['classification']} |")
        L.append("")
    L += ["## Pooled across operating points (descriptive; equal weights; stratified bootstrap)", "",
          "| comparison | geo change | 95% CI |", "|---|---|---|"]
    for k, v in r["pooled_descriptive"].items():
        L.append(f"| {k} | {pct(v['geo_rel_change'])} | {ci(v['geo_rel_change_ci95'])} |")
    L += ["", "## Secondary outcomes (means per operating point)", ""]
    for name in phase5.OP_ORDER:
        o = r["operating_points"][name]
        M = o["metrics"]
        L += [f"**{name}** ({o['n_processes_per_workload']} processes, {o['n_workloads']} workloads)", "",
              "| method | turnaround (ticks) | response (ticks) | waiting (ticks) | throughput (jobs/tick) | CPU util. | context switches | quantum changed | fallback |",
              "|---|---|---|---|---|---|---|---|---|"]
        for meth in M["avg_turnaround"]:
            b = o["controller_behaviour"].get(meth)
            L.append(f"| {meth} | " + " | ".join(f"{M[k][meth]['mean']:.3f}" for k in METRICS[:3])
                     + f" | {M['throughput'][meth]['mean']:.4f} | {M['cpu_utilization'][meth]['mean']:.3f} | "
                     f"{M['context_switches'][meth]['mean']:.1f} | "
                     + (f"{b['share_quantum_changed']:.0%} | {b['share_fallback']:.0%} |" if b else "— | — |"))
        rp = M["avg_response"]
        L += ["", "Response time vs RR1 (mean difference, ticks, 95% CI): " + "; ".join(
            f"{m} {rp[m]['vs_RR1_mean_diff']:+.2f} [{rp[m]['vs_RR1_mean_diff_ci95'][0]:+.2f}, {rp[m]['vs_RR1_mean_diff_ci95'][1]:+.2f}]"
            for m in ("p3_frozen", "p4_selected", "RR2")), ""]
    L += ["Exploratory (uncorrected) geo change vs RR1 by true workload class:", ""]
    for name in phase5.OP_ORDER:
        bc = r["operating_points"][name]["exploratory_by_class_geo_change"]
        L.append(f"- {name}: " + "; ".join(
            f"{c} (n={v['n']}): p3 {pct(v['p3_frozen vs RR1'])}, p4 {pct(v['p4_selected vs RR1'])}, oracle {pct(v['oracle_fixed vs RR1'])}"
            for c, v in bc.items()))
    L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    main()
