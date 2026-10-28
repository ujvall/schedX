"""Phase 5 principal experiment: three independently generated operating points
evaluated with frozen controllers only (docs/phase5_experimental_protocol.md).

Nothing here fits or tunes a model. Each workload's job list is scheduled by
every method; the hash of each schedule's own input job list is recorded so
pairing can be verified from the raw CSV.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from .adaptive_eval import METRICS, run_controller, summarize
from .controller import best_candidate, tie_order
from .metrics import compute_metrics
from .phase4 import _comparison, run_p4
from .policy_eval import burst_jobs
from .scheduler import round_robin
from .stats import holm, mean_diff_ci, signflip_test
from .workload import GeneratorConfig, GeneratedProcess, Workload, generate_dataset

OP_ORDER = ("small", "medium", "large")


def generator_for(base: dict, n_processes: int) -> GeneratorConfig:
    """Phase 2 generator rules with the process count fixed to ``n_processes``."""
    return GeneratorConfig.from_dict({**base, "process_count": [n_processes, n_processes]})


def check_seeds(cfg5: dict) -> None:
    seeds = [op["master_seed"] for op in cfg5["operating_points"].values()]
    if len(set(seeds)) != len(seeds):
        raise ValueError("operating points must use distinct master seeds")
    reused = set(seeds) & set(cfg5["reserved_seeds_not_to_reuse"])
    if reused:
        raise ValueError(f"seeds {sorted(reused)} are already used by earlier datasets")


def generate_operating_points(cfg5: dict, generator_base: dict) -> dict[str, list[Workload]]:
    check_seeds(cfg5)
    out = {}
    for name in OP_ORDER:
        op = cfg5["operating_points"][name]
        out[name] = generate_dataset(op["n_workloads"], op["master_seed"], generator_for(generator_base, op["n_processes"]))
    return out


def _sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, separators=(",", ":")).encode()).hexdigest()


def trace_hash(processes: tuple[GeneratedProcess, ...]) -> str:
    """Identity of a workload's full input (arrivals, priorities, CPU and I/O traces)."""
    return _sha([[p.pid, p.arrival, p.priority, list(p.cpu_bursts), list(p.io_bursts)] for p in processes])


def jobs_hash(jobs) -> str:
    """Identity of the job list a scheduler actually received."""
    return _sha([[j.pid, j.arrival, j.burst] for j in jobs])


def manifest(ops: dict[str, list[Workload]]) -> pd.DataFrame:
    rows = []
    for name, workloads in ops.items():
        for w in workloads:
            jobs = burst_jobs(w.processes)
            rows.append({
                "operating_point": name, "workload_id": w.workload_id, "master_seed": w.master_seed,
                "workload_index": w.index, "n_processes": len(w.processes), "n_jobs": len(jobs),
                "total_cpu": sum(j.burst for j in jobs), "workload_class": w.label, "regime": w.regime,
                "burst_cv": w.burst_cv, "mean_interarrival": w.mean_interarrival,
                "trace_sha256": trace_hash(w.processes), "jobs_sha256": jobs_hash(jobs),
            })
    return pd.DataFrame(rows)


def evaluate_workload(w: Workload, p3, p4, q0: int, quanta) -> list[dict]:
    """Every method on one workload. Returns long rows with metrics and the
    hash of the job list each method actually scheduled."""
    jobs = burst_jobs(w.processes)
    rows = []

    def add(method, sched, policy, source, t):
        rows.append({"workload_id": w.workload_id, "method": method, "policy": policy, "decision_source": source,
                     "t_decision": t, "input_jobs_sha256": jobs_hash(sched.processes), **summarize(compute_metrics(sched))})

    fixed = {q: round_robin(jobs, q) for q in quanta}
    for q, s in fixed.items():
        add(f"RR{q}", s, f"RR{q}", "fixed", None)
    turn = {q: compute_metrics(s).avg_turnaround for q, s in fixed.items()}
    q_or = best_candidate(turn, tie_order(quanta, q0))
    add("oracle_fixed", fixed[q_or], f"RR{q_or}", "oracle", None)

    s, q, source, t = run_controller(w.processes, p3.decide, p3.regime, q0, p3.after_completions)
    add("p3_frozen", s, f"RR{q0}->RR{q}" if t is not None else f"RR{q0}", source, t)
    s, q, source, t = run_p4(w.processes, p4, q0)
    add("p4_selected", s, f"RR{q0}->RR{q}" if t is not None and source != "fallback:insufficient_observation" else f"RR{q0}",
        source, t)
    return rows


def _effective(policy: str) -> str:
    parts = policy.split("->")
    return parts[0] if len(parts) == 2 and parts[0] == parts[1] else policy


def _aggregate_change_ci(m: np.ndarray, b: np.ndarray, seed: int, n_boot: int) -> list[float]:
    idx = np.random.default_rng(seed).integers(0, len(m), size=(n_boot, len(m)))
    r = m[idx].mean(axis=1) / b[idx].mean(axis=1) - 1
    return [float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))]


def compare(T: pd.DataFrame, a: str, b: str, st: dict, n_boot: int | None = None) -> dict:
    out = _comparison(T, a, b, st, n_boot)
    out["change_in_mean_turnaround_ci95"] = _aggregate_change_ci(
        T[a].to_numpy(), T[b].to_numpy(), st["bootstrap"]["seed"], n_boot or st["bootstrap"]["n"])
    return out


def classify(c: dict, alpha: float, sesoi: float) -> str:
    lo, hi = c["geo_rel_change_ci95"]
    if c["holm_p"] < alpha and hi < 0:
        return "practically meaningful improvement" if hi <= -sesoi else "statistically supported improvement below the practical threshold"
    if -sesoi < lo and hi < sesoi:
        return "practically negligible difference"
    return "inconclusive"


def run_family(raw: pd.DataFrame, man: pd.DataFrame, hyps: dict, st: dict) -> dict:
    """Holm over one family; hypotheses with identical decision vectors are merged."""
    tables = {}
    for name in OP_ORDER:
        ids = man.loc[man.operating_point == name, "workload_id"]
        sub = raw[raw.workload_id.isin(ids)]
        tables[name] = (sub.pivot(index="workload_id", columns="method", values="avg_turnaround"),
                        sub.pivot(index="workload_id", columns="method", values="policy").apply(lambda c: c.map(_effective)))
    groups: dict[tuple, list[str]] = {}
    for h, (a, b) in hyps.items():
        op = h.rsplit("_", 1)[1]
        P = tables[op][1]
        groups.setdefault((op, tuple(P[a]), tuple(P[b])), []).append(h)
    fam = {}
    for hs in groups.values():
        h = hs[0]
        a, b = hyps[h]
        op = h.rsplit("_", 1)[1]
        c = compare(tables[op][0], a, b, st)
        c.update(hypothesis=f"{a} vs {b}", operating_point=op, merged_with=hs[1:])
        fam[h] = c
    adj = holm({h: v["signflip_p"] for h, v in fam.items()})
    for h, v in fam.items():
        v["holm_p"] = adj[h]
        v["classification"] = classify(v, st["alpha"], st["sesoi_relative"])
    return fam


def pooled(raw: pd.DataFrame, man: pd.DataFrame, a: str, b: str, st: dict) -> dict:
    """Equal-weight average of per-operating-point mean log ratios; bootstrap
    stratified by operating point (descriptive, outside both families)."""
    rng = np.random.default_rng(st["bootstrap"]["seed"])
    T = raw.pivot(index="workload_id", columns="method", values="avg_turnaround")
    per = {}
    for name in OP_ORDER:
        ids = man.loc[man.operating_point == name, "workload_id"]
        per[name] = np.log(T.loc[ids, a] / T.loc[ids, b]).to_numpy()
    est = np.mean([d.mean() for d in per.values()])
    boot = np.zeros(st["bootstrap"]["n"])
    for d in per.values():
        boot += d[rng.integers(0, len(d), size=(st["bootstrap"]["n"], len(d)))].mean(axis=1) / len(per)
    sizes = {len(d) for d in per.values()}
    sf = signflip_test(np.concatenate(list(per.values())), st["signflip"]["seed"],
                       st["signflip"]["monte_carlo_draws"], st["signflip"]["exact_max_nonzero"]) if len(sizes) == 1 else None
    return {"comparison": f"{a} vs {b}", "geo_rel_change": float(np.expm1(est)),
            "geo_rel_change_ci95": [float(np.expm1(np.percentile(boot, 2.5))), float(np.expm1(np.percentile(boot, 97.5)))],
            "signflip_p_descriptive": None if sf is None else sf["p"]}


def analyze(raw: pd.DataFrame, man: pd.DataFrame, cfg5: dict) -> dict:
    st = cfg5["statistics"]
    primary = run_family(raw, man, cfg5["primary_hypotheses"], st)
    secondary = run_family(raw, man, cfg5["secondary_hypotheses"], st)
    methods = [f"RR{q}" for q in cfg5["fixed_quanta"]] + ["p3_frozen", "p4_selected", "oracle_fixed"]
    per_op = {}
    for name in OP_ORDER:
        m_ = man[man.operating_point == name]
        sub = raw[raw.workload_id.isin(m_.workload_id)]
        metrics = {}
        for k in METRICS:
            M = sub.pivot(index="workload_id", columns="method", values=k)
            metrics[k] = {meth: {"mean": float(M[meth].mean()), "median": float(M[meth].median()),
                                 **{f"vs_RR1_{x}": y for x, y in mean_diff_ci(M[meth], M["RR1"], st["bootstrap"]["seed"],
                                                                                 st["bootstrap"]["n"]).items()}}
                          for meth in methods}
        behaviour = {}
        for meth in ("p3_frozen", "p4_selected"):
            s = sub[sub.method == meth]
            eff = s.policy.map(_effective)
            behaviour[meth] = {"share_quantum_changed": float((eff != f"RR{cfg5['default_quantum']}").mean()),
                               "share_fallback": float(s.decision_source.str.startswith("fallback").mean()),
                               "policy_counts": s.policy.value_counts().to_dict(),
                               "source_counts": s.decision_source.value_counts().to_dict()}
        T = sub.pivot(index="workload_id", columns="method", values="avg_turnaround")
        cls = m_.set_index("workload_id").workload_class
        by_class = {c: {"n": int((cls == c).sum()),
                        **{f"{a} vs RR1": compare(T.loc[cls.index[cls == c]], a, "RR1", st, 2000)["geo_rel_change"]
                           for a in ("p3_frozen", "p4_selected", "oracle_fixed")}}
                    for c in sorted(cls.unique())}
        per_op[name] = {
            "n_workloads": int(len(m_)), "n_processes_per_workload": int(m_.n_processes.iloc[0]),
            "total_processes": int(m_.n_processes.sum()), "total_jobs": int(m_.n_jobs.sum()),
            "mean_jobs_per_workload": float(m_.n_jobs.mean()),
            "class_counts": m_.workload_class.value_counts().to_dict(),
            "metrics": metrics, "controller_behaviour": behaviour,
            "oracle_fixed_vs_RR1": compare(T, "oracle_fixed", "RR1", st),
            "exploratory_by_class_geo_change": by_class,
        }
    return {
        "primary_family": primary,
        "secondary_family": secondary,
        "pooled_descriptive": {k: pooled(raw, man, a, b, st)
                               for k, (a, b) in {"p3_frozen vs RR1": ("p3_frozen", "RR1"),
                                                 "p4_selected vs RR1": ("p4_selected", "RR1"),
                                                 "oracle_fixed vs RR1": ("oracle_fixed", "RR1")}.items()},
        "operating_points": per_op,
    }


def processes_frame(workloads: list[Workload]) -> pd.DataFrame:
    """Raw input traces (one row per process)."""
    from .workload import to_frames

    return to_frames(workloads)[0]

