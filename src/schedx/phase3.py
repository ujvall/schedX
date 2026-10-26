"""Phase 3 experiment pipeline: fit/select on train+validation, then test once.

``fit_stage`` must only ever receive training and validation traces;
``test_stage`` only the held-out test traces plus the frozen controllers.
Protocol: docs/phase3_adaptive_scheduling.md §8.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.tree import export_text

from .adaptive_eval import (
    evaluate_fixed,
    evaluate_online_candidates,
    rr_name,
    run_controller,
    submission_features,
    summarize,
)
from .controller import QuantumController, best_candidate, fit_controller, score, tie_order
from .metrics import compute_metrics
from .ml import grouped_split
from .policy_eval import burst_jobs
from .scheduler import round_robin
from .stats import holm, mean_diff_ci, paired_comparison
from .workload import GeneratedProcess, dataset_version, processes_from_frame

Traces = dict[str, tuple[GeneratedProcess, ...]]


# --- Inputs ------------------------------------------------------------------------------


def config_hash(cfg3: dict, ds_version: str) -> str:
    blob = json.dumps({"phase3": cfg3, "dataset_version": ds_version}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def load_split(artifacts: Path, cfg2: dict) -> tuple[pd.DataFrame, pd.Series, dict]:
    """Phase 2 features, the Phase 2 grouped split, and dataset metadata."""
    meta = json.loads((artifacts / "dataset" / "dataset_meta.json").read_text(encoding="utf-8"))
    if meta["dataset_version"] != dataset_version(cfg2):
        raise ValueError("dataset does not match the Phase 2 config; rerun scripts/generate_dataset.py")
    features = pd.read_csv(artifacts / "dataset" / "features.csv")
    s = cfg2["split"]
    split = grouped_split(features, s["seed"], s["train"], s["validation"], s["test"])
    return features, split, meta


def load_traces(artifacts: Path, ids) -> Traces:
    """Rebuild traces for ``ids`` only (others are never materialized)."""
    ids = set(ids)
    df = pd.read_csv(artifacts / "dataset" / "processes.csv")
    return processes_from_frame(df[df.workload_id.isin(ids)])


def quanta_evaluated(cfg3: dict) -> list[int]:
    return sorted(set(cfg3["candidates"]) | {cfg3["phase2_baseline_quantum"]})


# --- Fit stage (train + validation only) -------------------------------------------------------


def choose_default(turnaround: pd.DataFrame) -> tuple[int, dict[int, float]]:
    """Best fixed quantum: lowest mean log regret; ties -> smaller quantum."""
    cands = sorted(turnaround.columns)
    log_regret = np.log(turnaround[cands].div(turnaround[cands].min(axis=1), axis=0)).mean()
    table = {int(q): float(log_regret[q]) for q in cands}
    return min(cands, key=lambda q: (table[q], q)), table


def counterfactual_data(traces: Traces, cfg3: dict) -> dict:
    """Run every candidate on every workload (training-time counterfactuals)."""
    cands = cfg3["candidates"]
    fixed = {w: evaluate_fixed(burst_jobs(p), cands, include_fcfs=False) for w, p in traces.items()}
    turn = pd.DataFrame({w: {q: m[rr_name(q)]["avg_turnaround"] for q in cands} for w, m in fixed.items()}).T
    q0, default_table = choose_default(turn)
    sub_X = pd.DataFrame({w: submission_features(p) for w, p in traces.items()}).T.astype(float)
    online = {}
    for m in cfg3["decision_after_completions"]:
        feats, outs, times = {}, {}, {}
        for w, procs in traces.items():
            jobs = burst_jobs(procs)
            base = round_robin(jobs, q0)
            t, f, out = evaluate_online_candidates(procs, jobs, base, q0, cands, m, fixed[w][rr_name(q0)])
            outs[w] = {q: v["avg_turnaround"] for q, v in out.items()}
            times[w] = t
            if f is not None:
                feats[w] = f
        online[m] = {
            "X": pd.DataFrame(feats).T.astype(float),
            "turnaround": pd.DataFrame(outs).T,
            "t_decision": pd.Series(times, dtype=object),
        }
    return {"q0": q0, "default_table": default_table, "fixed_turnaround": turn, "submission_X": sub_X, "online": online}


def _complexity(regime: str, family: str, m: int | None) -> tuple:
    return (0,) if family == "constant" else ((1,) if regime == "submission" else (2, m))


def search(data: dict, train_ids, val_ids, cfg3: dict) -> tuple[pd.DataFrame, dict[str, QuantumController]]:
    """Fit every configuration on train, score on validation, freeze winners.

    Validation score = mean over validation workloads of log(T_chosen/T_q0).
    Ties go to the simpler configuration (constant < submission < online,
    smaller m first), then to grid order.
    """
    cands, q0, seed = tuple(cfg3["candidates"]), data["q0"], cfg3["seed"]
    train_ids, val_ids = list(train_ids), list(val_ids)
    regimes = [("submission", None, data["submission_X"], data["fixed_turnaround"])]
    regimes += [("online", m, d["X"], d["turnaround"]) for m, d in data["online"].items()]

    rows, fitted = [], []
    configs = [("submission", None, "constant", {})]
    for regime, m, _, _ in regimes:
        for family, grid in cfg3["grids"].items():
            configs += [(regime, m, family, p) for p in grid]
    lookup = {(r, m): (X, T) for r, m, X, T in regimes}
    for regime, m, family, params in configs:
        X, T = lookup[(regime, m)]
        Xtr = X.loc[X.index.intersection(train_ids)]
        ctrl = fit_controller(regime, family, params, Xtr, T.loc[Xtr.index], cands, q0, seed, m)
        val = score(ctrl, X.loc[X.index.intersection(val_ids)], T.loc[val_ids])
        tr = score(ctrl, Xtr, T.loc[train_ids])
        decisions = ctrl.decide_batch(X.loc[X.index.intersection(val_ids)]) if len(X) else []
        rows.append({
            "regime": regime, "after_completions": m, "family": family, "params": json.dumps(params),
            "val_mean_log_ratio": float(val.mean()), "val_geo_rel_change": float(np.expm1(val.mean())),
            "train_mean_log_ratio": float(tr.mean()),
            "val_share_changed": float(np.mean([q != q0 for q, _ in decisions])) if decisions else 0.0,
            "complexity": _complexity(regime, family, m),
        })
        fitted.append(ctrl)
    table = pd.DataFrame(rows)

    def pick(mask) -> int:
        sub = table[mask]
        best = sub["val_mean_log_ratio"].min()
        tied = sub[sub["val_mean_log_ratio"] == best]
        return int(min(tied.index, key=lambda i: (tied.at[i, "complexity"], i)))

    sel = pick(table.index == table.index)
    best_sub = pick((table.regime == "submission") & (table.family != "constant"))
    best_on = pick(table.regime == "online")
    table["selected"] = table.index == sel
    table = table.drop(columns="complexity")
    return table, {"selected": fitted[sel], "best_submission": fitted[best_sub], "best_online": fitted[best_on]}


def variability_diagnostic(traces: Traces, data: dict, train_ids, m: int) -> dict:
    """Training-only diagnostic: how strongly does burst variability relate to
    the large-vs-small quantum choice, and how much of it is visible online?

    Spearman correlations with log(T_RR_max / T_RR_min) over training
    workloads. Uses retrospective burst CVs, so it is analysis, never a
    controller input.
    """
    from scipy.stats import spearmanr

    from .adaptive_eval import decision_time, online_observation
    from .observation import observation_features

    train_ids = [w for w in train_ids if w in traces]
    T = data["fixed_turnaround"].loc[train_ids]
    qmin, qmax = min(T.columns), max(T.columns)
    y = np.log(T[qmax] / T[qmin])
    true_cv, obs_cv = {}, {}
    for w in train_ids:
        jobs = burst_jobs(traces[w])
        b = np.array([j.burst for j in jobs], dtype=float)
        true_cv[w] = b.std() / b.mean()
        s = round_robin(jobs, data["q0"])
        t = decision_time(s, len(jobs), m)
        if t is not None:
            obs_cv[w] = observation_features(online_observation(traces[w], jobs, s, t))["obs_completed_burst_cv"]
    tc, oc = pd.Series(true_cv), pd.Series(obs_cv).dropna()
    rho = lambda a, b: float(spearmanr(a, b).statistic)
    return {
        "target": f"log(T_RR{qmax} / T_RR{qmin}) on training workloads",
        "after_completions": m,
        "spearman_true_job_burst_cv_vs_target": rho(tc, y[tc.index]),
        "spearman_observed_burst_cv_vs_target": rho(oc, y[oc.index]),
        "spearman_true_vs_observed_cv": rho(tc[oc.index], oc),
        "n_train": len(tc),
        "n_with_observation": len(oc),
    }


def tree_rules(ctrl: QuantumController) -> str | None:
    if ctrl.model is None:
        return None
    est = ctrl.model[-1]
    if not hasattr(est, "tree_"):
        return None
    names = [str(n) for n in ctrl.model[:-1].get_feature_names_out()]
    return export_text(est, feature_names=names, decimals=3)


def describe(ctrl: QuantumController) -> dict:
    return {"regime": ctrl.regime, "family": ctrl.family, "params": ctrl.params,
            "after_completions": ctrl.after_completions, "default": ctrl.default,
            "candidates": list(ctrl.candidates), "features": list(ctrl.features),
            "label": ctrl.describe(), "rules": tree_rules(ctrl)}


# --- Test stage (held-out workloads, frozen controllers) ---------------------------------------


DEPLOYABLE = ["fixed_RR2", "best_fixed", "phase2_map_predicted", "adaptive_selected",
              "adaptive_submission", "adaptive_online"]
REFERENCE = ["phase2_map_true", "oracle_fixed", "oracle_online"]


def test_stage(
    traces: Traces,
    controllers: dict[str, QuantumController],
    q0: int,
    cfg3: dict,
    labels: pd.Series,
    predicted: pd.Series,
    class_map: dict[str, str],
) -> pd.DataFrame:
    """Evaluate every method on every test workload. Returns a long table:
    one row per (workload, method) with the chosen policy and its metrics."""
    cands = cfg3["candidates"]
    order = tie_order(cands, q0)
    online_m = controllers["best_online"].after_completions
    keys = {"selected": "adaptive_selected", "best_submission": "adaptive_submission", "best_online": "adaptive_online"}
    rows = []
    for w, procs in traces.items():
        jobs = burst_jobs(procs)
        fixed = evaluate_fixed(jobs, quanta_evaluated(cfg3), include_fcfs=True)

        def add(method, policy, metrics, source="fixed", t=None):
            rows.append({"workload_id": w, "method": method, "policy": policy, "decision_source": source,
                         "t_decision": t, **metrics})

        add("fixed_RR2", rr_name(cfg3["phase2_baseline_quantum"]), fixed[rr_name(cfg3["phase2_baseline_quantum"])])
        add("best_fixed", rr_name(q0), fixed[rr_name(q0)])
        add("phase2_map_predicted", class_map[predicted[w]], fixed[class_map[predicted[w]]], "phase2_model")
        add("phase2_map_true", class_map[labels[w]], fixed[class_map[labels[w]]], "true_label")
        q_or = best_candidate({q: fixed[rr_name(q)]["avg_turnaround"] for q in cands}, order)
        add("oracle_fixed", rr_name(q_or), fixed[rr_name(q_or)], "oracle")

        base = round_robin(jobs, q0)
        t, _, outs = evaluate_online_candidates(procs, jobs, base, q0, cands, online_m, fixed[rr_name(q0)])
        q_on = best_candidate({q: v["avg_turnaround"] for q, v in outs.items()}, order)
        add("oracle_online", f"RR{q0}->RR{q_on}" if t is not None else rr_name(q0), outs[q_on], "oracle", t)

        for key, method in keys.items():
            ctrl = controllers[key]
            sched, q, source, td = run_controller(procs, ctrl.decide, ctrl.regime, q0, ctrl.after_completions)
            metrics = summarize(compute_metrics(sched))
            # The inference run must reproduce the matching counterfactual exactly.
            if ctrl.regime == "submission":
                assert metrics == fixed[rr_name(q)], (w, method)
            elif ctrl.after_completions == online_m:
                assert metrics == outs[q], (w, method)
            policy = rr_name(q) if ctrl.regime == "submission" or td is None else f"RR{q0}->RR{q}"
            add(method, policy, metrics, source, td)
    return pd.DataFrame(rows)


def sensitivity(method: pd.Series, baseline: pd.Series) -> dict:
    """Post-hoc robustness checks for the primary comparison. Added after the
    test evaluation to explain why the pre-specified Pratt Wilcoxon test and
    the bootstrap CI disagree; they do not replace the pre-specified test."""
    from scipy.stats import binomtest, wilcoxon

    lr = np.log(method / baseline)
    nz = lr[lr != 0]
    better, worse = int((nz < 0).sum()), int((nz > 0).sum())
    return {
        "note": "post hoc; does not replace the pre-specified Wilcoxon (pratt) test",
        "n_changed": int(len(nz)),
        "n_better": better,
        "n_worse": worse,
        "sum_log_gain_better": float(nz[nz < 0].sum()),
        "sum_log_loss_worse": float(nz[nz > 0].sum()),
        "wilcoxon_nonzero_only_p": float(wilcoxon(nz).pvalue) if len(nz) else 1.0,
        "sign_test_p": float(binomtest(better, len(nz)).pvalue) if len(nz) else 1.0,
    }


def _pivot(long: pd.DataFrame, metric: str) -> pd.DataFrame:
    return long.pivot(index="workload_id", columns="method", values=metric)


def analyze(long: pd.DataFrame, info: pd.DataFrame, cfg3: dict) -> dict:
    """All test-set statistics. ``info``: per-workload reporting variables
    (class, n_processes, burst_cv, n_jobs) indexed by workload id."""
    b = cfg3["bootstrap"]
    T = _pivot(long, "avg_turnaround")
    info = info.loc[T.index]
    cmp = lambda m, base: paired_comparison(T[m].to_numpy(), T[base].to_numpy(), b["seed"], b["n"])

    family = {f"{m} vs best_fixed": cmp(m, "best_fixed") for m in DEPLOYABLE if m != "best_fixed"}
    family.update({f"{m} vs fixed_RR2": cmp(m, "fixed_RR2") for m in DEPLOYABLE if m not in ("best_fixed", "fixed_RR2")})
    adj = holm({k: v["wilcoxon_p"] for k, v in family.items()})
    for k in family:
        family[k]["holm_p"] = adj[k]
    reference = {f"{m} vs best_fixed": cmp(m, "best_fixed") for m in REFERENCE}
    reference.update({f"{m} vs oracle_fixed": cmp(m, "oracle_fixed")
                      for m in ("best_fixed", "fixed_RR2", "adaptive_selected", "adaptive_submission", "adaptive_online")})

    # Share of the best_fixed -> oracle gap (in mean log turnaround) closed.
    gap = np.log(T["best_fixed"] / T["oracle_fixed"]).mean()
    closed = {m: (float((gap - np.log(T[m] / T["oracle_fixed"]).mean()) / gap) if gap > 0 else None)
              for m in ("adaptive_selected", "adaptive_submission", "adaptive_online", "phase2_map_predicted",
                        "phase2_map_true", "oracle_online")}

    methods = DEPLOYABLE + REFERENCE
    secondary = {}
    for metric in ("avg_turnaround", "avg_waiting", "avg_response", "throughput", "cpu_utilization", "context_switches"):
        P = _pivot(long, metric)
        secondary[metric] = {
            m: {"mean": float(P[m].mean()), "median": float(P[m].median()),
                **{f"vs_best_fixed_{k}": v for k, v in mean_diff_ci(P[m], P["best_fixed"], b["seed"], b["n"]).items()}}
            for m in methods
        }

    def groups(col, labels_):
        out = {}
        for g in labels_.cat.categories if hasattr(labels_, "cat") else sorted(labels_.unique()):
            ids = labels_.index[labels_ == g]
            out[str(g)] = {"n": int(len(ids)), **{
                m: paired_comparison(T.loc[ids, m].to_numpy(), T.loc[ids, "best_fixed"].to_numpy(), b["seed"], 2000)
                for m in ("adaptive_selected", "adaptive_online", "adaptive_submission", "oracle_fixed")}}
        return out

    breakdown = {
        "workload_class": groups("workload_class", info["workload_class"]),
        "n_processes_tercile": groups("n_processes", pd.qcut(info["n_processes"], 3)),
        "burst_cv_tercile": groups("burst_cv", pd.qcut(info["burst_cv"], 3)),
    }
    decisions = {}
    for m in ("adaptive_selected", "adaptive_submission", "adaptive_online", "oracle_fixed", "oracle_online",
              "phase2_map_predicted"):
        sub = long[long.method == m]
        decisions[m] = {"policy_counts": sub.policy.value_counts().to_dict(),
                        "source_counts": sub.decision_source.value_counts().to_dict()}
    return {
        "n_test_workloads": int(len(T)),
        "primary": {"comparison": "adaptive_selected vs best_fixed", **family["adaptive_selected vs best_fixed"]},
        "primary_sensitivity_post_hoc": sensitivity(T["adaptive_selected"], T["best_fixed"]),
        "deployable_comparisons": family,
        "reference_comparisons": reference,
        "oracle_gap_closed": closed,
        "secondary_metrics": secondary,
        "breakdowns": breakdown,
        "decisions": decisions,
    }
