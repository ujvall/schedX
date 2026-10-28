"""Phase 4 pipeline: improved online observation (docs/phase4_observation_model.md).

Builds on Phase 3 primitives without changing them. The only manipulated
variable is the controller's feature set: A + O (Phase 3) versus
A + O + E (Kaplan–Meier estimates of unfinished work). Candidates, decision
points, model families, grids, seeds, and splits are the same as Phase 3.

``fit_stage`` receives training and validation traces only; ``test_stage``
receives evaluation traces plus frozen controllers.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from .adaptive_eval import (
    decision_time,
    evaluate_fixed,
    evaluate_online_candidates,
    online_observation,
    rr_name,
    run_controller,
    summarize,
    switched,
)
from .controller import QuantumController, best_candidate, fit_controller, score, tie_order
from .estimator import censored_estimate, is_sufficient
from .metrics import compute_metrics
from .phase3 import Traces, choose_default
from .policy_eval import burst_jobs
from .scheduler import round_robin
from .stats import holm, mean_diff_ci, paired_comparison, signflip_test

REGIME = "online_censored"


def config_hash(cfg4: dict, dataset_version: str, phase3_hash: str) -> str:
    blob = json.dumps({"phase4": cfg4, "dataset_version": dataset_version, "phase3": phase3_hash}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def improved_features(processes, jobs, default_schedule, t) -> dict | None:
    """A + O + E features at decision tick ``t``; ``None`` if insufficient."""
    from .adaptive_eval import submission_features
    from .observation import observation_features

    obs = online_observation(processes, jobs, default_schedule, t)
    if not is_sufficient(obs):
        return None
    return {**submission_features(processes), **observation_features(obs), **censored_estimate(obs)}


# --- Fit stage (train + validation only) ------------------------------------------------


def counterfactual_data(traces: Traces, cfg4: dict) -> dict:
    """Every candidate on every workload (training-time counterfactuals),
    with A + O + E features at each decision point."""
    cands = cfg4["candidates"]
    fixed = {w: evaluate_fixed(burst_jobs(p), cands, include_fcfs=False) for w, p in traces.items()}
    turn = pd.DataFrame({w: {q: m[rr_name(q)]["avg_turnaround"] for q in cands} for w, m in fixed.items()}).T
    q0, default_table = choose_default(turn)
    online = {}
    for m in cfg4["decision_after_completions"]:
        feats, outs, times = {}, {}, {}
        for w, procs in traces.items():
            jobs = burst_jobs(procs)
            base = round_robin(jobs, q0)
            t, _, out = evaluate_online_candidates(procs, jobs, base, q0, cands, m, fixed[w][rr_name(q0)])
            outs[w] = {q: v["avg_turnaround"] for q, v in out.items()}
            times[w] = t
            f = improved_features(procs, jobs, base, t) if t is not None else None
            if f is not None:
                feats[w] = f
        online[m] = {"X": pd.DataFrame(feats).T.astype(float), "turnaround": pd.DataFrame(outs).T,
                     "t_decision": pd.Series(times, dtype=object)}
    return {"q0": q0, "default_table": default_table, "fixed_turnaround": turn, "online": online}


def search(data: dict, train_ids, val_ids, cfg4: dict, ablation: dict) -> tuple[pd.DataFrame, dict]:
    """Fit every online_censored configuration on train; score on validation.

    Returns the validation table and ``{"p4_selected", "p4_ablation"}``.
    ``ablation`` = the frozen Phase 3 controller's (family, params, m),
    retrained with A + O + E features: isolates the effect of the features.
    """
    cands, q0, seed = tuple(cfg4["candidates"]), data["q0"], cfg4["seed"]
    train_ids, val_ids = list(train_ids), list(val_ids)
    rows, fitted = [], []
    for m in cfg4["decision_after_completions"]:
        X, T = data["online"][m]["X"], data["online"][m]["turnaround"]
        Xtr = X.loc[X.index.intersection(train_ids)]
        Xva = X.loc[X.index.intersection(val_ids)]
        for family, grid in cfg4["grids"].items():
            for params in grid:
                ctrl = fit_controller(REGIME, family, params, Xtr, T.loc[Xtr.index], cands, q0, seed, m)
                val = score(ctrl, Xva, T.loc[val_ids])
                tr = score(ctrl, Xtr, T.loc[train_ids])
                decisions = ctrl.decide_batch(Xva)
                rows.append({"after_completions": m, "family": family, "params": json.dumps(params),
                             "val_mean_log_ratio": float(val.mean()), "val_geo_rel_change": float(np.expm1(val.mean())),
                             "train_mean_log_ratio": float(tr.mean()),
                             "val_share_changed": float(np.mean([q != q0 for q, _ in decisions])) if decisions else 0.0})
                fitted.append(ctrl)
    table = pd.DataFrame(rows)
    best = table["val_mean_log_ratio"].min()
    # Ties: smaller m first, then grid order (= table order within m).
    sel = int(table.index[table["val_mean_log_ratio"] == best].min())
    table["selected"] = table.index == sel

    m, family, params = ablation["after_completions"], ablation["family"], ablation["params"]
    X, T = data["online"][m]["X"], data["online"][m]["turnaround"]
    Xtr = X.loc[X.index.intersection(train_ids)]
    abl = fit_controller(REGIME, family, params, Xtr, T.loc[Xtr.index], cands, q0, seed, m)
    abl_val = score(abl, X.loc[X.index.intersection(val_ids)], T.loc[val_ids])
    return table, {"p4_selected": fitted[sel], "p4_ablation": abl,
                   "p4_ablation_val_mean_log_ratio": float(abl_val.mean())}


# --- Inference ---------------------------------------------------------------------------------


def run_p4(processes, ctrl: QuantumController, q0: int):
    """Inference path for a Phase 4 controller: run q0, observe at the m-th
    completion, decide once from A + O + E, continue. Never sees outcomes."""
    jobs = burst_jobs(processes)
    base = round_robin(jobs, q0)
    t = decision_time(base, len(jobs), ctrl.after_completions)
    if t is None:
        q, source = ctrl.decide(None)
        return base, q0, source, None
    feats = improved_features(processes, jobs, base, t)
    if feats is None:
        return base, q0, "fallback:insufficient_observation", t
    q, source = ctrl.decide(feats)
    return (base if q == q0 else switched(jobs, q0, q, t)), q, source, t


# --- Evaluation stage ------------------------------------------------------------------------------


def evaluate_set(traces: Traces, ctrls: dict, p3: dict, q0: int, cfg4: dict) -> pd.DataFrame:
    """Every method on every workload of one evaluation set (long table)."""
    cands = cfg4["candidates"]
    order = tie_order(cands, q0)
    m_sel = ctrls["p4_selected"].after_completions
    rows = []
    for w, procs in traces.items():
        jobs = burst_jobs(procs)
        fixed = evaluate_fixed(jobs, cands, include_fcfs=False)

        def add(method, policy, metrics, source="fixed", t=None):
            rows.append({"workload_id": w, "method": method, "policy": policy, "decision_source": source,
                         "t_decision": t, **metrics})

        add(f"RR{q0}", rr_name(q0), fixed[rr_name(q0)])
        q_or = best_candidate({q: fixed[rr_name(q)]["avg_turnaround"] for q in cands}, order)
        add("oracle_fixed", rr_name(q_or), fixed[rr_name(q_or)], "oracle")
        base = round_robin(jobs, q0)
        t, _, outs = evaluate_online_candidates(procs, jobs, base, q0, cands, m_sel, fixed[rr_name(q0)])
        q_on = best_candidate({q: v["avg_turnaround"] for q, v in outs.items()}, order)
        add("oracle_online", f"RR{q0}->RR{q_on}" if t is not None else rr_name(q0), outs[q_on], "oracle", t)

        for name, ctrl in (("p3_frozen", p3["selected"]), ("submission_ref", p3["best_submission"])):
            sched, q, source, td = run_controller(procs, ctrl.decide, ctrl.regime, q0, ctrl.after_completions)
            policy = rr_name(q) if ctrl.regime == "submission" or td is None else f"RR{q0}->RR{q}"
            add(name, policy, summarize(compute_metrics(sched)), source, td)
        for name in ("p4_selected", "p4_ablation"):
            sched, q, source, td = run_p4(procs, ctrls[name], q0)
            metrics = summarize(compute_metrics(sched))
            if name == "p4_selected":
                assert metrics == outs[q], (w, name)  # inference reproduces the counterfactual
            add(name, f"RR{q0}->RR{q}" if td is not None else rr_name(q0), metrics, source, td)
    return pd.DataFrame(rows)


def _comparison(T: pd.DataFrame, a: str, b: str, st: dict, n_boot: int | None = None) -> dict:
    lr = np.log(T[a] / T[b]).to_numpy()
    out = paired_comparison(T[a].to_numpy(), T[b].to_numpy(), st["bootstrap"]["seed"], n_boot or st["bootstrap"]["n"])
    out["pratt_wilcoxon_p"] = out.pop("wilcoxon_p")  # secondary in Phase 4
    sf = signflip_test(lr, st["signflip"]["seed"], st["signflip"]["monte_carlo_draws"], st["signflip"]["exact_max_nonzero"])
    out["signflip_p"], out["signflip_method"] = sf["p"], sf["method"]
    out["change_in_mean_turnaround"] = float(T[a].mean() / T[b].mean() - 1)
    return out


def analyze(long: pd.DataFrame, info: pd.DataFrame, cfg4: dict, q0: int) -> dict:
    """Pre-specified family (Holm, merged duplicates, decision rule) plus
    secondary and exploratory statistics for one evaluation set."""
    st = cfg4["statistics"]
    T = long.pivot(index="workload_id", columns="method", values="avg_turnaround")
    # Effective policy: "RRa->RRa" (decided to keep the default) runs exactly like "RRa".
    P = long.pivot(index="workload_id", columns="method", values="policy").apply(
        lambda col: col.map(lambda p: p.split("->")[0] if "->" in p and len(set(p.split("->"))) == 1 else p))
    alias = {"RR1": f"RR{q0}"}
    hyp = {h: [alias.get(x, x) for x in pair] for h, pair in cfg4["hypotheses"].items()}

    # Merge hypotheses whose two arms make identical decisions on every workload.
    groups: dict[tuple, list[str]] = {}
    for h, (a, b) in hyp.items():
        groups.setdefault((tuple(P[a]), tuple(P[b])), []).append(h)
    merged = {hs[0]: hs for hs in groups.values()}
    family = {h: _comparison(T, *hyp[h], st) for h in merged}
    adj = holm({h: v["signflip_p"] for h, v in family.items()})
    for h, v in family.items():
        v["hypothesis"] = f"{hyp[h][0]} vs {hyp[h][1]}"
        v["merged_with"] = merged[h][1:]
        v["holm_p"] = adj[h]
        v["supported_improvement"] = bool(adj[h] < st["alpha"] and v["geo_rel_change_ci95"][1] < 0)

    secondary_pairs = [("submission_ref", f"RR{q0}"), ("p3_frozen", f"RR{q0}"), ("oracle_fixed", f"RR{q0}"),
                       ("oracle_online", f"RR{q0}"), ("p4_ablation", f"RR{q0}"), ("p4_selected", "p4_ablation")]
    secondary = {f"{a} vs {b}": _comparison(T, a, b, st) for a, b in secondary_pairs}

    methods = ["RR" + str(q0), "p3_frozen", "p4_selected", "p4_ablation", "submission_ref", "oracle_fixed", "oracle_online"]
    metrics = {}
    for k in ("avg_turnaround", "avg_response", "avg_waiting", "throughput", "cpu_utilization", "context_switches"):
        M = long.pivot(index="workload_id", columns="method", values=k)
        metrics[k] = {
            m: {"mean": float(M[m].mean()), "median": float(M[m].median()),
                **{f"vs_p3_frozen_{x}": y for x, y in mean_diff_ci(M[m], M["p3_frozen"], st["bootstrap"]["seed"], st["bootstrap"]["n"]).items()},
                **{f"vs_RR{q0}_{x}": y for x, y in mean_diff_ci(M[m], M[f"RR{q0}"], st["bootstrap"]["seed"], st["bootstrap"]["n"]).items()}}
            for m in methods
        }
    changes = {}
    for m in ("p3_frozen", "p4_selected", "p4_ablation", "oracle_online"):
        sub = long[long.method == m]
        switched_ = sub.policy.str.contains("->") & ~sub.policy.str.endswith(f"->RR{q0}")
        changes[m] = {"share_quantum_changed": float(switched_.mean()), "policy_counts": sub.policy.value_counts().to_dict(),
                      "source_counts": sub.decision_source.value_counts().to_dict()}
    agree = float((P["p4_selected"] == P["p3_frozen"]).mean())

    info = info.loc[T.index]
    nb = st["breakdown_bootstrap_n"]

    def groups_(lab):
        out = {}
        for g in (lab.cat.categories if hasattr(lab, "cat") else sorted(lab.unique())):
            ids = lab.index[lab == g]
            sub = T.loc[ids]
            out[str(g)] = {"n": int(len(ids)), **{f"{a} vs {b}": _comparison(sub, a, b, st, nb)
                                                    for a, b in (("p4_selected", "p3_frozen"), ("p4_selected", f"RR{q0}"),
                                                                 ("p3_frozen", f"RR{q0}"), ("oracle_fixed", f"RR{q0}"))}}
        return out

    return {
        "n_workloads": int(len(T)),
        "primary_family": family,
        "primary_hypothesis": cfg4["primary_hypothesis"],
        "secondary_comparisons": secondary,
        "metrics": metrics,
        "quantum_changes": changes,
        "share_same_decision_p4_selected_vs_p3_frozen": agree,
        "breakdowns_exploratory": {
            "workload_class": groups_(info["workload_class"]),
            "burst_cv_tercile": groups_(pd.qcut(info["burst_cv"], 3)),
            "n_processes_tercile": groups_(pd.qcut(info["n_processes"], 3)),
        },
    }


def estimator_diagnostic(traces: Traces, data: dict, train_ids, ms) -> dict:
    """Training-only retrospective check of the estimator against the true
    bursts of the jobs released by the decision tick. Analysis only: true
    bursts are never controller inputs."""
    from scipy.stats import spearmanr

    from .observation import observation_features

    train_ids = [w for w in train_ids if w in traces]
    T = data["fixed_turnaround"].loc[train_ids]
    qmin, qmax = min(T.columns), max(T.columns)
    y = np.log(T[qmax] / T[qmin])
    out = {"target": f"log(T_RR{qmax} / T_RR{qmin})", "n_train": len(train_ids), "by_decision_point": {}}
    for m in ms:
        rows = {}
        for w in train_ids:
            jobs = burst_jobs(traces[w])
            base = round_robin(jobs, data["q0"])
            t = decision_time(base, len(jobs), m)
            if t is None:
                continue
            obs = online_observation(traces[w], jobs, base, t)
            o, e = observation_features(obs), censored_estimate(obs)
            rel = np.array([j.burst for j in jobs if j.pid in obs.released], dtype=float)
            allb = np.array([j.burst for j in jobs], dtype=float)
            rows[w] = {"naive_mean": o["obs_completed_burst_mean"], "km_rmst": e["km_rmst"],
                       "true_rmst": float(np.minimum(rel, e["km_tau"]).mean()),
                       "km_surv": e["km_surv_tau"], "true_surv": float((rel > e["km_tau"]).mean()),
                       "naive_cv": o["obs_completed_burst_cv"], "km_cv": e["km_restricted_cv"],
                       "true_cv": float(allb.std() / allb.mean()), "km_tau": e["km_tau"]}
        d = pd.DataFrame(rows).T.astype(float)
        rho = lambda a, b: float(spearmanr(d[a], b, nan_policy="omit").statistic)
        out["by_decision_point"][int(m)] = {
            "n": int(len(d)),
            "median_tau": float(d.km_tau.median()),
            "mean_abs_log_error_restricted_mean": {
                "completed_only": float(np.mean(np.abs(np.log(d.naive_mean / d.true_rmst)))),
                "kaplan_meier": float(np.mean(np.abs(np.log(d.km_rmst / d.true_rmst)))),
            },
            "mean_bias_tail_mass_km_minus_true": float((d.km_surv - d.true_surv).mean()),
            "spearman_with_target": {"completed_cv": rho("naive_cv", y[d.index]), "km_restricted_cv": rho("km_cv", y[d.index]),
                                     "km_surv_tau": rho("km_surv", y[d.index]), "km_rmst": rho("km_rmst", y[d.index])},
            "spearman_with_true_burst_cv": {"completed_cv": rho("naive_cv", d.true_cv), "km_restricted_cv": rho("km_cv", d.true_cv)},
        }
    return out
