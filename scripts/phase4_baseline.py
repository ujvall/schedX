"""Phase 4, step 1: reproduce the frozen Phase 3 controller and verify it
against the committed Phase 3 evidence. Read-only with respect to Phase 3.

Usage:  python scripts/phase4_baseline.py [--artifacts artifacts] [--out reports/phase4]

Writes  <out>/baseline_reproduction.json
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from schedx.adaptive_eval import METRICS, run_controller, summarize  # noqa: E402
from schedx.metrics import compute_metrics  # noqa: E402
from schedx.ml import library_versions  # noqa: E402
from schedx.phase3 import config_hash, load_split, load_traces  # noqa: E402
from schedx.policy_eval import burst_jobs  # noqa: E402
from schedx.scheduler import round_robin  # noqa: E402

P3 = ROOT / "reports" / "phase3"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default=str(ROOT / "artifacts"))
    ap.add_argument("--out", default=str(ROOT / "reports" / "phase4"))
    args = ap.parse_args()
    artifacts, out = Path(args.artifacts), Path(args.out)

    cfg2 = json.loads((ROOT / "configs" / "phase2.json").read_text(encoding="utf-8"))
    cfg3 = json.loads((ROOT / "configs" / "phase3.json").read_text(encoding="utf-8"))
    frozen = json.loads((P3 / "frozen_controller.json").read_text(encoding="utf-8"))
    bundle = joblib.load(artifacts / "phase3" / "controllers.joblib")
    _, split, meta = load_split(artifacts, cfg2)
    chash = config_hash(cfg3, meta["dataset_version"])
    if not (chash == frozen["config_hash"] == bundle["config_hash"]):
        sys.exit("frozen Phase 3 controller does not match config/dataset; cannot reproduce baseline")

    q0, ctrl = bundle["q0"], bundle["controllers"]["selected"]
    test_ids = list(split.index[split == "test"])
    traces = load_traces(artifacts, test_ids)
    committed = pd.read_csv(P3 / "per_workload_test.csv").set_index(["workload_id", "method"])

    rows, mismatches = [], []
    for w in test_ids:
        sched, q, source, t = run_controller(traces[w], ctrl.decide, ctrl.regime, q0, ctrl.after_completions)
        m = summarize(compute_metrics(sched))
        rr = summarize(compute_metrics(round_robin(burst_jobs(traces[w]), q0)))
        for method, metrics in (("adaptive_selected", m), ("best_fixed", rr)):
            c = committed.loc[(w, method)]
            for k in METRICS:
                if not np.isclose(c[k], metrics[k], rtol=0, atol=1e-12):
                    mismatches.append({"workload_id": w, "method": method, "metric": k,
                                       "committed": float(c[k]), "reproduced": metrics[k]})
        if committed.loc[(w, "adaptive_selected"), "decision_source"] != source:
            mismatches.append({"workload_id": w, "method": "adaptive_selected", "metric": "decision_source"})
        rows.append({"workload_id": w, "quantum": q, "source": source, "t_decision": t,
                     **{f"p3_{k}": v for k, v in m.items()}, **{f"rr{q0}_{k}": v for k, v in rr.items()}})
    df = pd.DataFrame(rows)

    report = {
        "purpose": "Reproduce the frozen Phase 3 controller on the established test split (read-only check).",
        "phase3_config_hash": chash,
        "dataset_version": meta["dataset_version"],
        "controller": frozen["controllers"]["selected"],
        "default_quantum_q0": q0,
        "n_test_workloads": len(test_ids),
        "reproduces_committed_phase3_evidence": not mismatches,
        "mismatches": mismatches[:20],
        "n_mismatches": len(mismatches),
        "decisions": {
            "chosen_quantum_counts": df["quantum"].value_counts().sort_index().to_dict(),
            "decision_source_counts": df["source"].value_counts().to_dict(),
            "share_quantum_changed": float((df["quantum"] != q0).mean()),
        },
        "test_means": {
            "phase3_controller": {k: float(df[f"p3_{k}"].mean()) for k in METRICS},
            f"RR{q0}": {k: float(df[f"rr{q0}_{k}"].mean()) for k in METRICS},
        },
        "library_versions": library_versions(),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "baseline_reproduction.json").write_text(json.dumps(report, indent=2, default=int), encoding="utf-8")
    print(f"reproduces committed Phase 3 evidence: {not mismatches} ({len(mismatches)} mismatches)")
    if mismatches:
        sys.exit(1)


if __name__ == "__main__":
    main()
