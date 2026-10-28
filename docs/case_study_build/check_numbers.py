"""Cross-check every quantitative claim in docs/final_case_study.md against committed evidence.

Usage (from the repository root):  python docs/case_study_build/check_numbers.py

Each check formats a value read from a committed result file the way the report
prints it and asserts that exact text occurs in the Markdown. Exit code 1 if any
claim is missing or different.
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MD = (ROOT / "docs" / "final_case_study.md").read_text(encoding="utf-8")
L = lambda p: json.loads((ROOT / p).read_text(encoding="utf-8"))
p2, p3 = L("reports/phase2/metrics.json"), L("reports/phase3/results.json")
p3a, f3 = L("reports/phase3/audit/audit_results.json"), L("reports/phase3/frozen_controller.json")
p4, d4 = L("reports/phase4/results.json"), L("reports/phase4/estimator_diagnostic.json")
p5, c5 = L("reports/phase5/results.json"), L("configs/phase5.json")
OPS = ("small", "medium", "large")
U = "−"  # the report uses a true minus sign


def pct(x, d=2, sign=True):
    s = f"{abs(x) * 100:.{d}f}"
    return (U if x < 0 else "+" if sign else "") + s


def num(x, d=2):
    return (U if x < 0 else "") + f"{abs(x):.{d}f}"


checks = []


def need(label, text):
    checks.append((label, text, text in MD))


# Phase 2
m = p2["main_experiment"]["models"]
need("LR accuracy", f"{m['logistic_regression']['test']['accuracy'] * 100:.1f}% accuracy")
need("LR macro-F1", f"macro-F1 {m['logistic_regression']['test']['macro_f1']:.3f}")
need("DT accuracy", f"Decision Tree {m['decision_tree']['test']['accuracy'] * 100:.1f}%")
need("RF accuracy", f"Random Forest {m['random_forest']['test']['accuracy'] * 100:.1f}%")
assert p2["main_experiment"]["selected_model"] == "logistic_regression"
need("selected model", "Logistic Regression (chosen by validation score)")
pol = p2["policy_experiment"]["test"]["predicted_class_map_by_horizon"]
assert all(v["delta_ci95"][0] < 0 < v["delta_ci95"][1] for v in pol.values()), "Phase 2 claim: all CIs include zero"
need("dataset size", "1,500 workloads (21,453 processes; 484 CPU-bound, 527 I/O-bound, 489 mixed)")
need("split", "900 training, 300 validation, and 300 test workloads")
# Phase 3
pr = p3["primary"]
need("P3 geo", f"by {pct(pr['geo_rel_change'])}% (95% CI {pct(pr['geo_rel_change_ci95'][0], sign=False)}% to {pct(pr['geo_rel_change_ci95'][1], sign=False)}%)".replace("by +", "by "))
need("P3 wilcoxon", f"p = {pr['wilcoxon_p']:.2f}")
assert p3a["comparisons"]["primary"]["n_zero_difference"] == 275
need("P3 ties", "275 of 300 pairs tied")
# Phase 4
need("P4 KM error", f"from {min(v['mean_abs_log_error_restricted_mean']['completed_only'] for v in d4['by_decision_point'].values()):.2f}"
     f"–{max(v['mean_abs_log_error_restricted_mean']['completed_only'] for v in d4['by_decision_point'].values()):.2f} to "
     f"{min(v['mean_abs_log_error_restricted_mean']['kaplan_meier'] for v in d4['by_decision_point'].values()):.2f}"
     f"–{max(v['mean_abs_log_error_restricted_mean']['kaplan_meier'] for v in d4['by_decision_point'].values()):.2f}")
h = p4["established_test_split"]["primary_family"]["H1"]
need("P4 test H1", f"by {pct(h['geo_rel_change'])}% (95% CI {pct(h['geo_rel_change_ci95'][0])}% to {pct(h['geo_rel_change_ci95'][1])}%, Holm p = {h['holm_p']:.2f})")
h = p4["fresh_confirmation"]["primary_family"]["H1"]
need("P4 fresh H1", f"by {pct(h['geo_rel_change'])}% (95% CI {pct(h['geo_rel_change_ci95'][0])}% to {pct(h['geo_rel_change_ci95'][1])}%) on {p4['fresh_confirmation']['n_workloads']} fresh")
# Phase 5 setup
for o, s in zip(OPS, ("Small", "Medium", "Large")):
    d, cfg = p5["operating_points"][o], c5["operating_points"][o]
    need(f"P5 table2 {o}", f"| {s} | {d['n_processes_per_workload']} | {d['n_workloads']:,} | {cfg['master_seed']} | {d['total_jobs']:,} |")
need("SESOI", f"below {U}{c5['statistics']['sesoi_relative'] * 100:.0f}%")
# Phase 5 primary family (Table 3)
for o in OPS:
    n = p5["operating_points"][o]["n_processes_per_workload"]
    for h, name in (("H1", "Phase 3"), ("H2", "Phase 4")):
        c = p5["primary_family"][f"{h}_{o}"]
        lo, hi = c["geo_rel_change_ci95"]
        holm = f"{c['holm_p']:.2f}" if c["holm_p"] >= 0.1 else (f"{c['holm_p']:.3f}" if c["holm_p"] >= 0.001 else f"{c['holm_p']:.4f}")
        cls = "supported, below 1%" if c["classification"].startswith("statistically supported improvement below") else \
              "negligible" if c["classification"] == "practically negligible difference" else c["classification"]
        need(f"T3 {h} {o}", f"| {n} | {name} | {pct(c['geo_rel_change'])}% [{pct(lo)}, {pct(hi)}] | {pct(c['change_in_mean_turnaround'])}% | {holm} | {cls} |")
# Phase 5 narrative
L5 = p5["operating_points"]["large"]["metrics"]
need("T large", f"from {L5['avg_turnaround']['RR1']['mean']:.2f} to {L5['avg_turnaround']['p3_frozen']['mean']:.2f} ticks")
b = [p5["operating_points"][o]["controller_behaviour"]["p3_frozen"]["share_quantum_changed"] * 100 for o in OPS]
need("P3 change share", f"{b[0]:.1f}%, {b[1]:.1f}%, and {b[2]:.1f}% of workloads")
need("P3 fallback small", f"{p5['operating_points']['small']['controller_behaviour']['p3_frozen']['share_fallback'] * 100:.1f}% of")
orc = [abs(p5["operating_points"][o]["oracle_fixed_vs_RR1"]["geo_rel_change"]) * 100 for o in OPS]
need("oracle", f"would gain {orc[0]:.2f}%, {orc[1]:.2f}%, and {orc[2]:.2f}%")
p4s = [p5["operating_points"][o]["controller_behaviour"]["p4_selected"]["share_quantum_changed"] * 100 for o in OPS]
need("P4 switch range", f"({min(p4s):.0f}–{max(p4s):.0f}% of workloads)")
r = [p5["operating_points"][o]["metrics"]["avg_response"]["p3_frozen"] for o in OPS]
need("resp small", f"+{r[0]['vs_RR1_mean_diff']:.2f} ticks (6 processes)")
need("resp medium", f"+{r[1]['vs_RR1_mean_diff']:.2f} ticks [{r[1]['vs_RR1_mean_diff_ci95'][0]:.2f}, {r[1]['vs_RR1_mean_diff_ci95'][1]:.2f}] (14)")
need("resp large", f"+{r[2]['vs_RR1_mean_diff']:.2f} ticks [{r[2]['vs_RR1_mean_diff_ci95'][0]:.2f}, {r[2]['vs_RR1_mean_diff_ci95'][1]:.2f}] (22)")
r4 = [p5["operating_points"][o]["metrics"]["avg_response"]["p4_selected"]["vs_RR1_mean_diff"] for o in OPS]
need("resp P4", f"(+{r4[0]:.1f}, +{r4[1]:.1f}, +{r4[2]:.1f} ticks)")
need("switches large", f"from {L5['context_switches']['RR1']['mean']:.1f} to {L5['context_switches']['p3_frozen']['mean']:.1f}")
for o in OPS:  # throughput and utilization identical across all methods
    M = p5["operating_points"][o]["metrics"]
    assert len({round(v["mean"], 12) for v in M["throughput"].values()}) == 1
    assert len({round(v["mean"], 12) for v in M["cpu_utilization"].values()}) == 1
need("thr/util large", f"{L5['throughput']['RR1']['mean']:.3f} jobs per tick and {L5['cpu_utilization']['RR1']['mean'] * 100:.1f}% utilization")
pool = p5["pooled_descriptive"]["p3_frozen vs RR1"]
need("pooled", f"it is {pct(pool['geo_rel_change'])}% (95% CI {pct(pool['geo_rel_change_ci95'][0])}% to {pct(pool['geo_rel_change_ci95'][1])}%)")
dg = f3["train_only_diagnostic"]
need("spearman", f"(Spearman {dg['spearman_true_job_burst_cv_vs_target']:.2f})")
need("spearman obs", f"related ({dg['spearman_observed_burst_cv_vs_target']:.2f})")
need("commit", "commit `e848cf2`")
assert not re.search(r"\b(kernel|hardware) (benchmark|measurement)s? (show|showed|demonstrate)", MD, re.I)

bad = [(l, t) for l, t, ok in checks if not ok]
print(f"{len(checks) - len(bad)}/{len(checks)} claims found verbatim in the report")
for l, t in bad:
    print(f"  MISSING [{l}]: {t}")
sys.exit(1 if bad else 0)
