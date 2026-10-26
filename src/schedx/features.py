"""Workload features with explicit decision-time availability (Phase 2).

Every dataset column is assigned one category in ``CATALOG``:

* ``id``, ``target``, ``generation``, ``label_equivalent`` -- never model inputs.
* ``A`` -- known at batch submission (counts, arrivals, priorities).
* ``B`` -- measured online from each process's first ``k`` CPU bursts and the
  ``k - 1`` I/O waits between them. Bursts after the k-th are never read.
* ``C`` -- retrospective lifetime totals; only for the labelled
  retrospective experiment.

Model inputs are selected *by category* via ``feature_columns``, so a column
must be deliberately catalogued as A/B to ever reach a deployable model.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from .workload import GeneratedProcess, Workload

PRE, ONLINE, RETRO = "A", "B", "C"
DEPLOYABLE = (PRE, ONLINE)

CATALOG: dict[str, str] = {
    # identifiers / target
    "workload_id": "id",
    "pid": "id",
    "workload_class": "target",
    # generation parameters (workloads.csv / processes.csv)
    "master_seed": "generation",
    "workload_index": "generation",
    "regime": "generation",
    "p_io": "generation",
    "mean_interarrival": "generation",
    "burst_cv": "generation",
    "intended_kind": "generation",
    # the label is a threshold on these
    "share_io": "label_equivalent",
    "io_fraction": "label_equivalent",
    "process_kind": "label_equivalent",
    # A: pre-execution
    "n_processes": PRE,
    "arrival_span": PRE,
    "arrival_gap_mean": PRE,
    "arrival_gap_std": PRE,
    "arrival_rate": PRE,
    "priority_mean": PRE,
    "priority_std": PRE,
    # B: online, first k bursts
    "horizon_k": ONLINE,
    "obs_cpu_mean": ONLINE,
    "obs_cpu_std": ONLINE,
    "obs_cpu_min": ONLINE,
    "obs_cpu_max": ONLINE,
    "obs_cpu_median": ONLINE,
    "obs_proc_cpu_mean_std": ONLINE,
    "obs_io_mean": ONLINE,
    "obs_io_std": ONLINE,
    "obs_io_cpu_ratio": ONLINE,
    "obs_frac_io_dominant": ONLINE,
    "obs_io_per_proc": ONLINE,
    "obs_frac_terminated": ONLINE,
    # C: retrospective (whole lifetime)
    "total_cpu_mean": RETRO,
    "total_io_mean": RETRO,
    "lifetime_io_fraction": RETRO,
    "bursts_per_proc_mean": RETRO,
    "cpu_demand_per_tick": RETRO,
}

# Columns of features.csv that are candidate model features, in fixed order.
FEATURES_CSV_FEATURES = [c for c in CATALOG if CATALOG[c] in (PRE, ONLINE, RETRO)]


def feature_columns(categories: Iterable[str] = DEPLOYABLE) -> list[str]:
    """Feature names for the given availability categories, in catalog order."""
    cats = set(categories)
    bad = cats - {PRE, ONLINE, RETRO}
    if bad:
        raise ValueError(f"categories {sorted(bad)} are never model inputs")
    return [c for c in FEATURES_CSV_FEATURES if CATALOG[c] in cats]


def _std(xs) -> float:
    return float(np.std(xs)) if len(xs) else np.nan


def _mean(xs) -> float:
    return float(np.mean(xs)) if len(xs) else np.nan


def pre_execution_features(procs: tuple[GeneratedProcess, ...]) -> dict[str, float]:
    arrivals = sorted(p.arrival for p in procs)
    gaps = np.diff(arrivals)
    span = arrivals[-1] - arrivals[0]
    prios = [p.priority for p in procs]
    return {
        "n_processes": len(procs),
        "arrival_span": span,
        "arrival_gap_mean": _mean(gaps),
        "arrival_gap_std": _std(gaps),
        "arrival_rate": len(procs) / (span + 1),
        "priority_mean": _mean(prios),
        "priority_std": _std(prios),
    }


def online_features(procs: tuple[GeneratedProcess, ...], k: int) -> dict[str, float]:
    """Features from each process's first ``k`` CPU bursts and the I/O waits
    that completed before its k-th CPU burst. Nothing later is read."""
    if k < 1:
        raise ValueError("horizon k must be >= 1")
    cpu_all, io_all, proc_cpu_means = [], [], []
    io_dominant, terminated = [], 0
    for p in procs:
        cpu = p.cpu_bursts[:k]
        io = p.io_bursts[: k - 1]
        cpu_all.extend(cpu)
        io_all.extend(io)
        proc_cpu_means.append(np.mean(cpu))
        if io:
            io_dominant.append(sum(io) > sum(cpu))
        # Termination is observable: the process ended within the horizon.
        terminated += len(p.cpu_bursts) <= k
    return {
        "horizon_k": k,
        "obs_cpu_mean": _mean(cpu_all),
        "obs_cpu_std": _std(cpu_all),
        "obs_cpu_min": float(min(cpu_all)),
        "obs_cpu_max": float(max(cpu_all)),
        "obs_cpu_median": float(np.median(cpu_all)),
        "obs_proc_cpu_mean_std": _std(proc_cpu_means),
        "obs_io_mean": _mean(io_all),
        "obs_io_std": _std(io_all),
        "obs_io_cpu_ratio": sum(io_all) / sum(cpu_all) if io_all else np.nan,
        "obs_frac_io_dominant": _mean(io_dominant),
        "obs_io_per_proc": len(io_all) / len(procs),
        "obs_frac_terminated": terminated / len(procs),
    }


def retrospective_features(procs: tuple[GeneratedProcess, ...]) -> dict[str, float]:
    total_cpu = [p.total_cpu for p in procs]
    total_io = [p.total_io for p in procs]
    span = max(p.arrival for p in procs) - min(p.arrival for p in procs)
    return {
        "total_cpu_mean": _mean(total_cpu),
        "total_io_mean": _mean(total_io),
        "lifetime_io_fraction": sum(total_io) / (sum(total_io) + sum(total_cpu)),
        "bursts_per_proc_mean": _mean([len(p.cpu_bursts) for p in procs]),
        "cpu_demand_per_tick": sum(total_cpu) / (span + 1),
    }


def snapshot_features(procs: tuple[GeneratedProcess, ...], k: int) -> dict[str, float]:
    """All A, B and C features for one workload at horizon ``k``."""
    if not procs:
        raise ValueError("a workload needs at least one process")
    return {**pre_execution_features(procs), **online_features(procs, k), **retrospective_features(procs)}


def build_feature_table(workloads: Iterable[Workload], horizons: Iterable[int]) -> pd.DataFrame:
    """One row per (workload, horizon): id, features (catalog order), target."""
    horizons = list(horizons)
    rows = []
    for w in workloads:
        for k in horizons:
            rows.append({"workload_id": w.workload_id, **snapshot_features(w.processes, k), "workload_class": w.label})
    return pd.DataFrame(rows, columns=["workload_id", *FEATURES_CSV_FEATURES, "workload_class"])
