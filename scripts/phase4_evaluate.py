"""Phase 4, stage 2: evaluate FROZEN controllers once on the established test split
and on the fresh confirmation set (generated here for the first time).

Refuses to run if any config or the dataset changed since freezing.

Usage:  python scripts/phase4_evaluate.py [--config configs/phase4.json] [--phase3-config configs/phase3.json]
                                          [--phase2-config configs/phase2.json] [--artifacts artifacts]
                                          [--phase3-reports reports/phase3] [--reports reports/phase4]

Writes  <reports>/per_workload_test.csv, per_workload_fresh.csv, results.json, summary.md
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from schedx import phase3, phase4  # noqa: E402
from schedx.workload import GeneratorConfig, generate_dataset  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "phase4.json"))
    ap.add_argument("--phase3-config", default=str(ROOT / "configs" / "phase3.json"))
    ap.add_argument("--phase2-config", default=str(ROOT / "configs" / "phase2.json"))
    ap.add_argument("--artifacts", default=str(ROOT / "artifacts"))
    ap.add_argument("--phase3-reports", default=str(ROOT / "reports" / "phase3"))
    ap.add_argument("--reports", default=str(ROOT / "reports" / "phase4"))
    args = ap.parse_args()
    load = lambda p: json.loads(Path(p).read_text(encoding="utf-8"))
    cfg4, cfg3, cfg2 = load(args.config), load(args.phase3_config), load(args.phase2_config)
    artifacts, reports = Path(args.artifacts), Path(args.reports)

    _, split, meta = phase3.load_split(artifacts, cfg2)
    p3_hash = phase3.config_hash(cfg3, meta["dataset_version"])
    chash = phase4.config_hash(cfg4, meta["dataset_version"], p3_hash)
    frozen = load(reports / "frozen_controller.json")
    bundle = joblib.load(artifacts / "phase4" / "controllers.joblib")
    p3_bundle = joblib.load(artifacts / "phase3" / "controllers.joblib")
    if not (frozen["frozen"] and frozen["config_hash"] == chash == bundle["config_hash"]
            and p3_bundle["config_hash"] == p3_hash):
        sys.exit("config/dataset changed since the Phase 4 controllers were frozen; rerun scripts/phase4_train.py")
    q0 = bundle["q0"]

    # Established test split.
    test_ids = list(split.index[split == "test"])
    test_traces = phase3.load_traces(artifacts, test_ids)
    wl = pd.read_csv(artifacts / "dataset" / "workloads.csv").set_index("workload_id")
    test_info = wl.loc[test_ids, ["workload_class", "n_processes", "burst_cv"]]

    # Fresh confirmation set: same generator, new master seed, never seen before.
    fs = cfg4["fresh_confirmation_set"]
    if fs["master_seed"] == cfg2["dataset"]["master_seed"]:
        sys.exit("fresh set must use a different master seed")
    fresh = generate_dataset(fs["n_workloads"], fs["master_seed"], GeneratorConfig.from_dict(cfg2["generator"]))
    fresh_traces = {w.workload_id: w.processes for w in fresh}
    fresh_info = pd.DataFrame({"workload_class": [w.label for w in fresh], "n_processes": [len(w.processes) for w in fresh],
                               "burst_cv": [w.burst_cv for w in fresh]}, index=list(fresh_traces))

    out = {}
    for name, traces, info in (("test", test_traces, test_info), ("fresh", fresh_traces, fresh_info)):
        long = phase4.evaluate_set(traces, bundle["controllers"], p3_bundle["controllers"], q0, cfg4)
        long.merge(info, left_on="workload_id", right_index=True).to_csv(reports / f"per_workload_{name}.csv", index=False)
        out[name] = phase4.analyze(long, info, cfg4, q0)

    results = {
        "config_hash": chash,
        "phase3_config_hash": p3_hash,
        "dataset_version": meta["dataset_version"],
        "default_quantum_q0": q0,
        "controllers": frozen["controllers"],
        "fresh_confirmation_set": fs,
        "established_test_split": out["test"],
        "fresh_confirmation": out["fresh"],
    }
    (reports / "results.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    (reports / "summary.md").write_text(render(results, frozen), encoding="utf-8")
    print(f"wrote {reports}")


def pct(x: float) -> str:
    return f"{100 * x:+.2f}%"


def ci(c) -> str:
    return f"[{pct(c[0])}, {pct(c[1])}]"


def family_table(fam: dict) -> list[str]:
    lines = ["| hypothesis | comparison | geo change | 95% CI | Δ mean turnaround | better/equal/worse | sign-flip p | Holm p | supported | merged with |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for h, v in fam.items():
        lines.append(f"| {h} | {v['hypothesis']} | {pct(v['geo_rel_change'])} | {ci(v['geo_rel_change_ci95'])} | "
                     f"{pct(v['change_in_mean_turnaround'])} | {v['share_better']:.0%}/{v['share_equal']:.0%}/{v['share_worse']:.0%} | "
                     f"{v['signflip_p']:.3g} | {v['holm_p']:.3g} | {'**yes**' if v['supported_improvement'] else 'no'} | "
                     f"{', '.join(v['merged_with']) or '—'} |")
    return lines


def render(r: dict, frozen: dict) -> str:
    q0 = r["default_quantum_q0"]
    lines = [
        "# Phase 4 results (generated by scripts/phase4_evaluate.py)",
        "",
        f"Config hash `{r['config_hash']}` (Phase 3 `{r['phase3_config_hash']}`, dataset `{r['dataset_version']}`). "
        "Offline simulation, CPU-only burst-trace projection; no kernel-level measurement.",
        "",
        f"- `p4_selected`: **{r['controllers']['p4_selected']['label']}** (A + O + E features)",
        f"- `p4_ablation`: {r['controllers']['p4_ablation']['label']} (the Phase 3 configuration retrained with E features)",
        f"- `p3_frozen`: {frozen['p3_frozen_controller']} (A + O features)",
        f"- Validation (mean log ratio vs RR{q0}): p4_selected {frozen['validation_scores']['p4_selected']:+.5f}, "
        f"p4_ablation {frozen['validation_scores']['p4_ablation']:+.5f}, "
        f"p3_frozen {frozen['validation_scores']['p3_frozen_from_phase3_validation_table']:+.5f}",
        "",
        "Geo change = geometric mean of per-workload ratios of average turnaround (negative = better). "
        "Test: two-sided paired sign-flip permutation test of the mean log ratio; CI: paired percentile bootstrap "
        f"({frozen['phase4_config']['statistics']['bootstrap']['n']:,} resamples). "
        f"Supported improvement = Holm p < {frozen['phase4_config']['statistics']['alpha']} and CI upper bound < 0.",
    ]
    for key, title in (("established_test_split", "Established test split (primary)"),
                       ("fresh_confirmation", "Fresh confirmation set (secondary)")):
        s = r[key]
        prim = s["primary_family"][s["primary_hypothesis"]] if s["primary_hypothesis"] in s["primary_family"] else None
        lines += ["", f"## {title}: {s['n_workloads']} workloads", "", *family_table(s["primary_family"]), ""]
        if prim is not None:
            lines.append(f"**Primary H1 ({prim['hypothesis']}): "
                         f"{'statistically supported improvement' if prim['supported_improvement'] else 'no statistically supported improvement'}.**")
        lines += ["", "Secondary comparisons (not in the Holm family):", "",
                  "| comparison | geo change | 95% CI | sign-flip p | Pratt Wilcoxon p |", "|---|---|---|---|---|"]
        for k, v in s["secondary_comparisons"].items():
            lines.append(f"| {k} | {pct(v['geo_rel_change'])} | {ci(v['geo_rel_change_ci95'])} | {v['signflip_p']:.3g} | "
                         f"{v['pratt_wilcoxon_p']:.3g} |")
        lines += ["", "| method | turnaround | response | waiting | throughput | CPU util. | context switches | quantum changed |",
                  "|---|---|---|---|---|---|---|---|"]
        M = s["metrics"]
        for m in M["avg_turnaround"]:
            ch = s["quantum_changes"].get(m, {}).get("share_quantum_changed")
            lines.append(f"| {m} | " + " | ".join(f"{M[k][m]['mean']:.3f}" for k in
                         ("avg_turnaround", "avg_response", "avg_waiting", "throughput", "cpu_utilization", "context_switches"))
                         + f" | {'—' if ch is None else f'{ch:.0%}'} |")
        rp = M["avg_response"]["p4_selected"]
        lines += ["", f"Response time, p4_selected vs p3_frozen: {rp['vs_p3_frozen_mean_diff']:+.3f} ticks, 95% CI "
                      f"[{rp['vs_p3_frozen_mean_diff_ci95'][0]:+.3f}, {rp['vs_p3_frozen_mean_diff_ci95'][1]:+.3f}]; "
                      f"vs RR{q0}: {rp[f'vs_RR{q0}_mean_diff']:+.3f} ticks, 95% CI "
                      f"[{rp[f'vs_RR{q0}_mean_diff_ci95'][0]:+.3f}, {rp[f'vs_RR{q0}_mean_diff_ci95'][1]:+.3f}].",
                  f"Same effective decision as p3_frozen on {s['share_same_decision_p4_selected_vs_p3_frozen']:.0%} of workloads.",
                  "", "Exploratory breakdown (p4_selected vs p3_frozen, geo change, 2,000 resamples, no multiplicity correction):", ""]
        for dim, groups in s["breakdowns_exploratory"].items():
            cells = [f"{g} (n={v['n']}): {pct(v['p4_selected vs p3_frozen']['geo_rel_change'])} "
                     f"{ci(v['p4_selected vs p3_frozen']['geo_rel_change_ci95'])}" for g, v in groups.items()]
            lines.append(f"- {dim}: " + "; ".join(cells))
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
