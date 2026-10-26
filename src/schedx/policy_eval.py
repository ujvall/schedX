"""Offline policy-value experiment (Phase 2). NOT adaptive scheduling.

Asks whether a workload's (predicted) class could plausibly inform the choice
of a non-clairvoyant policy, using a CPU-only *burst-trace projection*: each
CPU burst becomes an independent job released at its contention-free time
(arrival + earlier CPU bursts + earlier I/O waits). This open-loop projection
ignores that CPU contention would delay I/O completion; it is a CPU-only
experiment, not an I/O-aware simulation. See docs/phase2_ml.md §6.
"""

from __future__ import annotations

from functools import partial
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from .metrics import compute_metrics
from .models import Process, Schedule
from .scheduler import fcfs, round_robin
from .workload import CLASSES, GeneratedProcess


def candidate_policies(quanta: Iterable[int]) -> dict[str, Callable[[list[Process]], Schedule]]:
    """Non-clairvoyant candidates only: real schedulers do not know bursts."""
    return {"FCFS": fcfs, **{f"RR{q}": partial(round_robin, quantum=q) for q in quanta}}


def burst_jobs(processes: Iterable[GeneratedProcess]) -> list[Process]:
    jobs = []
    for p in processes:
        release = p.arrival
        for j, cpu in enumerate(p.cpu_bursts):
            jobs.append(Process(f"{p.pid}.{j}", release, cpu, p.priority))
            release += cpu + (p.io_bursts[j] if j < len(p.io_bursts) else 0)
    return jobs


def score_workloads(traces: dict[str, tuple[GeneratedProcess, ...]], quanta: Iterable[int]) -> pd.DataFrame:
    """Average job turnaround per (workload, candidate) under the projection."""
    policies = candidate_policies(quanta)
    rows = {}
    for wid, procs in traces.items():
        jobs = burst_jobs(procs)
        rows[wid] = {name: compute_metrics(run(jobs)).avg_turnaround for name, run in policies.items()}
    return pd.DataFrame.from_dict(rows, orient="index")[list(policies)]


def regret(scores: pd.DataFrame) -> pd.DataFrame:
    """score / best score on the same workload (1.0 = best candidate)."""
    return scores.div(scores.min(axis=1), axis=0)


def choose_policies(regret_fit: pd.DataFrame, labels: pd.Series) -> tuple[str, dict[str, str]]:
    """Fixed baseline and class->policy map, from fitting workloads only.

    Pass only train+validation workloads here; never test.
    """
    fixed = str(regret_fit.mean().idxmin())
    labels = labels.reindex(regret_fit.index)
    class_map = {c: str(regret_fit[labels == c].mean().idxmin()) for c in CLASSES if (labels == c).any()}
    return fixed, class_map


def _paired_bootstrap_ci(diff: np.ndarray, seed: int, n_boot: int = 2000) -> list[float]:
    """95% percentile CI of the mean of paired per-workload differences."""
    rng = np.random.default_rng(seed)
    means = diff[rng.integers(0, len(diff), size=(n_boot, len(diff)))].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def evaluate_choices(
    regret_test: pd.DataFrame,
    true_labels: pd.Series,
    predictions: pd.DataFrame,
    fixed: str,
    class_map: dict[str, str],
    seed: int = 0,
) -> dict:
    """Mean regret on test workloads for each decision rule.

    ``predictions`` has one row per (workload_id, horizon_k) with ``predicted``.
    Each rule is also compared with the fixed baseline on the same workloads:
    ``delta_vs_fixed`` < 0 means lower regret than the baseline, with a paired
    bootstrap 95% CI.
    """
    ids = regret_test.index
    base = regret_test[fixed].to_numpy()
    true_labels = true_labels.reindex(ids)

    def rule(policies: pd.Series) -> dict:
        r = np.array([regret_test.at[w, p] for w, p in policies.reindex(ids).items()])
        return {
            "mean_regret": float(r.mean()),
            "delta_vs_fixed": float((r - base).mean()),
            "delta_ci95": _paired_bootstrap_ci(r - base, seed),
            "differs_from_fixed": float((policies.reindex(ids) != fixed).mean()),
        }

    out = {
        "n_workloads": int(len(ids)),
        "fixed_baseline": {"mean_regret": float(base.mean())},
        "true_class_map": rule(true_labels.map(class_map)),
        "predicted_class_map_by_horizon": {},
        "oracle": {"mean_regret": 1.0},
        "candidate_mean_regret": {k: float(v) for k, v in regret_test.mean().items()},
    }
    for k, g in predictions.groupby("horizon_k"):
        out["predicted_class_map_by_horizon"][int(k)] = rule(g.set_index("workload_id")["predicted"].map(class_map))
    return out
