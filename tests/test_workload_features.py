"""Workload generation, labels, and leakage-sensitive feature extraction."""

import math
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from schedx.features import (
    CATALOG,
    FEATURES_CSV_FEATURES,
    build_feature_table,
    feature_columns,
    online_features,
    snapshot_features,
)
from schedx.workload import (
    CLASSES,
    GeneratorConfig,
    GeneratedProcess,
    config_dict,
    generate_dataset,
    generate_workload,
    process_kind,
    processes_from_frame,
    to_frames,
    workload_class,
)

CFG = GeneratorConfig()
WORKLOADS = generate_dataset(150, master_seed=11)


# --- Generation ------------------------------------------------------------------


def test_generation_is_reproducible_and_index_independent():
    again = generate_dataset(150, master_seed=11)
    assert again == WORKLOADS
    # Any workload can be regenerated alone from (seed, index).
    assert generate_workload(11, 37) == WORKLOADS[37]
    assert generate_workload(12, 37) != WORKLOADS[37]


def test_all_times_are_integer_ticks_within_config():
    lo, hi = CFG.process_count
    for w in WORKLOADS:
        assert lo <= len(w.processes) <= hi
        arrivals = [p.arrival for p in w.processes]
        assert arrivals[0] == 0 and arrivals == sorted(arrivals)
        for p in w.processes:
            assert isinstance(p.arrival, int) and p.arrival >= 0
            assert len(p.cpu_bursts) >= 1
            assert len(p.io_bursts) == len(p.cpu_bursts) - 1
            assert all(isinstance(b, int) and b >= 1 for b in p.cpu_bursts + p.io_bursts)
            assert 0 <= p.priority <= CFG.priority_max


def test_dataset_varies_its_generation_parameters():
    assert {w.regime for w in WORKLOADS} == {"cpu", "io", "mixed"}
    assert len({len(w.processes) for w in WORKLOADS}) > 10
    assert np.std([w.mean_interarrival for w in WORKLOADS]) > 1
    assert np.std([w.burst_cv for w in WORKLOADS]) > 0.1


def test_config_round_trips_through_json_form():
    assert GeneratorConfig.from_dict(config_dict(CFG)) == CFG


# --- Labels ------------------------------------------------------------------------


def test_label_thresholds():
    assert process_kind(0.5) == "cpu_bound" and process_kind(0.51) == "io_bound"
    assert workload_class(0.25) == "cpu_bound"
    assert workload_class(0.26) == "mixed"
    assert workload_class(0.74) == "mixed"
    assert workload_class(0.75) == "io_bound"


def test_labels_rederive_from_raw_traces_after_csv_round_trip(tmp_path):
    procs_df, wl_df = to_frames(WORKLOADS)
    procs_df.to_csv(tmp_path / "p.csv", index=False)
    wl_df.to_csv(tmp_path / "w.csv", index=False)
    traces = processes_from_frame(pd.read_csv(tmp_path / "p.csv"))
    labels = pd.read_csv(tmp_path / "w.csv").set_index("workload_id")["workload_class"]
    for w in WORKLOADS:
        assert traces[w.workload_id] == w.processes
        # Recompute the label from bursts alone, independent of the classes.
        kinds = []
        for p in traces[w.workload_id]:
            io, cpu = sum(p.io_bursts), sum(p.cpu_bursts)
            kinds.append(io / (io + cpu) > 0.5)
        share = sum(kinds) / len(kinds)
        expected = "cpu_bound" if share <= 0.25 else "io_bound" if share >= 0.75 else "mixed"
        assert labels[w.workload_id] == expected == w.label
    assert set(labels) == set(CLASSES)


def test_labels_are_not_just_the_generator_regime():
    # Labels come from realized traces, so some workloads differ from intent.
    mismatched = sum(w.label != {"cpu": "cpu_bound", "io": "io_bound", "mixed": "mixed"}[w.regime] for w in WORKLOADS)
    assert 0 < mismatched < len(WORKLOADS) / 2


# --- Features and leakage ---------------------------------------------------------


def test_main_feature_set_excludes_every_non_deployable_column():
    cols = feature_columns()
    assert cols, "deployable feature set must not be empty"
    assert {CATALOG[c] for c in cols} == {"A", "B"}
    for forbidden in ("workload_class", "share_io", "io_fraction", "process_kind", "regime", "p_io",
                      "mean_interarrival", "burst_cv", "workload_id", "lifetime_io_fraction"):
        assert forbidden not in cols
    with pytest.raises(ValueError):
        feature_columns(["target"])
    with pytest.raises(ValueError):
        feature_columns(["label_equivalent"])


def test_online_features_never_read_bursts_after_horizon():
    w = max(WORKLOADS, key=lambda w: min(len(p.cpu_bursts) for p in w.processes))
    k = 2
    tampered = tuple(
        replace(p, cpu_bursts=p.cpu_bursts[:k] + tuple(999 for _ in p.cpu_bursts[k:]),
                io_bursts=p.io_bursts[: k - 1] + tuple(999 for _ in p.io_bursts[k - 1:]))
        for p in w.processes
    )
    assert online_features(tampered, k) == pytest.approx(online_features(w.processes, k), nan_ok=True)
    # Sanity: the tampering is visible to retrospective features.
    assert snapshot_features(tampered, k)["total_cpu_mean"] != snapshot_features(w.processes, k)["total_cpu_mean"]


def test_online_features_use_bursts_within_horizon():
    p = GeneratedProcess("A", 0, 0, (4, 6, 8), (10, 20), "cpu_bound")
    f1 = online_features((p,), 1)
    f3 = online_features((p,), 3)
    assert f1["obs_cpu_mean"] == 4 and f3["obs_cpu_mean"] == 6
    assert math.isnan(f1["obs_io_mean"]) and f3["obs_io_mean"] == 15
    assert f1["obs_frac_terminated"] == 0 and f3["obs_frac_terminated"] == 1
    with pytest.raises(ValueError):
        online_features((p,), 0)


def test_feature_table_schema_and_missing_values():
    df = build_feature_table(WORKLOADS[:20], [1, 2, 3])
    assert list(df.columns) == ["workload_id", *FEATURES_CSV_FEATURES, "workload_class"]
    assert len(df) == 60 and set(df["horizon_k"]) == {1, 2, 3}
    io_cols = ["obs_io_mean", "obs_io_std", "obs_io_cpu_ratio", "obs_frac_io_dominant"]
    # At k=1 no I/O has completed: these are missing, never zero-filled.
    assert df.loc[df.horizon_k == 1, io_cols].isna().all().all()
    non_io = [c for c in FEATURES_CSV_FEATURES if c not in io_cols]
    assert not df[non_io].isna().any().any()
    assert (df.loc[df.horizon_k == 1, "obs_io_per_proc"] == 0).all()


def test_every_dataset_column_is_catalogued():
    procs_df, wl_df = to_frames(WORKLOADS[:3])
    feat = build_feature_table(WORKLOADS[:3], [1])
    raw_trace_cols = {"arrival", "priority", "cpu_bursts", "io_bursts", "n_cpu_bursts", "total_cpu", "total_io"}
    for col in set(wl_df.columns) | set(feat.columns) | (set(procs_df.columns) - raw_trace_cols):
        assert col in CATALOG, col
