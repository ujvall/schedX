"""Phase 3 observation boundary and counterfactual vs inference separation."""

import math
from dataclasses import replace

import pytest

from schedx import Process, Slice, adaptive_round_robin, compute_metrics, round_robin
from schedx.adaptive_eval import (
    decision_time,
    evaluate_fixed,
    evaluate_online_candidates,
    online_observation,
    run_controller,
    submission_features,
    summarize,
    switched,
)
from schedx.observation import O_FEATURES, observation_features, observe
from schedx.policy_eval import burst_jobs
from schedx.workload import generate_dataset

WORKLOADS = generate_dataset(40, master_seed=21)


def J(pid, arrival, burst):
    return Process(pid, arrival, burst)


# --- Observation boundary -----------------------------------------------------------


def test_observe_hand_example():
    jobs = [J("a", 0, 2), J("b", 0, 5), J("c", 3, 1), J("d", 9, 4)]
    tl = round_robin(jobs, 2).timeline  # a0-2 b2-4 c4-5 b5-7 b7-8 d9-...
    obs = observe(jobs, tl, 5, process_arrivals=[0, 0, 9])
    assert obs.released == {"a": 0, "b": 0, "c": 3}  # d not released yet
    assert obs.attained == {"a": 2, "b": 2, "c": 1}
    assert obs.completed == {"a": 2, "c": 1}  # b unfinished: its burst stays hidden
    assert (obs.busy, obs.context_switches, obs.processes_arrived) == (5, 2, 2)
    f = observation_features(obs)
    assert list(f) == O_FEATURES
    assert f["obs_completed_burst_mean"] == 1.5 and f["obs_pending_jobs"] == 1
    assert f["obs_pending_attained_mean"] == 2 and f["obs_busy_fraction"] == 1.0
    assert f["obs_procs_arrived_frac"] == pytest.approx(2 / 3)


def test_observe_clips_slices_at_decision_time():
    obs = observe([J("a", 0, 10)], [Slice(0, 10, "a")], 4, [0])
    assert obs.attained == {"a": 4} and obs.completed == {} and obs.busy == 4
    f = observation_features(obs)
    assert math.isnan(f["obs_completed_burst_mean"])  # nothing completed yet: missing, not 0
    with pytest.raises(ValueError):
        observe([J("a", 0, 1)], [], 0, [0])


def test_observation_never_depends_on_unfinished_or_future_bursts():
    """Lengthening any burst not completed by t must not change the observation
    (Round Robin is non-clairvoyant), and neither may future releases."""
    checked = 0
    for w in WORKLOADS:
        jobs = burst_jobs(w.processes)
        sched = round_robin(jobs, 2)
        t = decision_time(sched, len(jobs), 5)
        if t is None:
            continue
        obs = online_observation(w.processes, jobs, sched, t)
        tampered = [j if j.pid in obs.completed else replace(j, burst=j.burst + 50) for j in jobs]
        obs2 = online_observation(w.processes, tampered, round_robin(tampered, 2), t)
        assert observation_features(obs2) == pytest.approx(observation_features(obs), nan_ok=True)
        assert obs2.completed == obs.completed
        checked += 1
    assert checked > 20


# --- Decision time and online counterfactuals ------------------------------------------


def test_decision_time_is_mth_completion_or_none():
    jobs = [J("a", 0, 1), J("b", 0, 3), J("c", 0, 2)]
    s = round_robin(jobs, 1)  # completions: a@1, c@5, b@6
    assert decision_time(s, 3, 1) == 1
    assert decision_time(s, 3, 2) == 5
    assert decision_time(s, 3, 3) is None  # nothing left to decide
    with pytest.raises(ValueError):
        decision_time(s, 3, 0)


def test_switched_run_matches_default_before_decision_time():
    for w in WORKLOADS[:15]:
        jobs = burst_jobs(w.processes)
        base = round_robin(jobs, 2)
        t = decision_time(base, len(jobs), 5)
        if t is None:
            continue
        for q1 in (1, 4, 8):
            run = switched(jobs, 2, q1, t)
            before = [(s.start, min(s.end, t), s.pid) for s in run.timeline if s.start < t]
            expect = [(s.start, min(s.end, t), s.pid) for s in base.timeline if s.start < t]
            assert before == expect
            compute_metrics(run)  # valid schedule: every job runs exactly its burst


def test_online_counterfactuals_cover_all_candidates():
    w = max(WORKLOADS, key=lambda w: sum(len(p.cpu_bursts) for p in w.processes))
    jobs = burst_jobs(w.processes)
    base = round_robin(jobs, 2)
    base_m = summarize(compute_metrics(base))
    t, feats, out = evaluate_online_candidates(w.processes, jobs, base, 2, [1, 2, 4, 8], 5, base_m)
    assert t is not None and set(out) == {1, 2, 4, 8} and out[2] == base_m
    assert set(feats) == set(submission_features(w.processes)) | set(O_FEATURES)
    # Few jobs: no decision point, every candidate equals the default.
    tiny = [p for p in WORKLOADS if sum(len(x.cpu_bursts) for x in p.processes) <= 5]
    if tiny:
        j = burst_jobs(tiny[0].processes)
        s = round_robin(j, 2)
        m = summarize(compute_metrics(s))
        assert evaluate_online_candidates(tiny[0].processes, j, s, 2, [1, 2], 5, m) == (None, None, {1: m, 2: m})


def test_evaluate_fixed_reports_all_metrics():
    out = evaluate_fixed(burst_jobs(WORKLOADS[0].processes), [1, 2])
    assert list(out) == ["FCFS", "RR1", "RR2"]
    assert set(out["RR1"]) == {"avg_turnaround", "avg_waiting", "avg_response", "throughput",
                               "cpu_utilization", "context_switches"}
    # Work-conserving policies on identical jobs share throughput and utilization.
    assert out["RR1"]["throughput"] == out["FCFS"]["throughput"]
    assert out["RR1"]["cpu_utilization"] == out["RR2"]["cpu_utilization"]


# --- Inference path ------------------------------------------------------------------------------


def test_inference_only_sees_features_and_matches_counterfactual():
    """run_controller passes the decide() callback features only, and the
    resulting schedule equals the counterfactual run for the chosen quantum."""
    seen = []

    def decide(feats):
        seen.append(feats)
        return (8, "model") if feats is not None else (2, "fallback:no_observation")

    for w in WORKLOADS[:10]:
        jobs = burst_jobs(w.processes)
        base = round_robin(jobs, 2)
        base_m = summarize(compute_metrics(base))
        t, feats, out = evaluate_online_candidates(w.processes, jobs, base, 2, [2, 8], 5, base_m)
        sched, q, source, t2 = run_controller(w.processes, decide, "online", q0=2, after_completions=5)
        assert t2 == t
        assert summarize(compute_metrics(sched)) == (out[8] if t is not None else base_m)
        assert seen[-1] == feats  # identical feature dict in both paths
        if t is None:
            assert (q, source) == (2, "fallback:no_observation")
    for f in seen:
        assert f is None or set(f) <= set(submission_features(WORKLOADS[0].processes)) | set(O_FEATURES)


def test_submission_regime_decides_once_from_submission_features():
    calls = []
    w = WORKLOADS[3]
    sched, q, source, t = run_controller(w.processes, lambda f: calls.append(f) or (4, "model"), "submission", q0=2)
    assert calls == [submission_features(w.processes)] and (q, t) == (4, 0)
    assert [(s.start, s.end, s.pid) for s in sched.timeline] == \
        [(s.start, s.end, s.pid) for s in round_robin(burst_jobs(w.processes), 4).timeline]
    with pytest.raises(ValueError):
        run_controller(w.processes, lambda f: (2, "x"), "psychic", q0=2)
