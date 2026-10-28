"""Phase 4 censoring-aware estimator: correctness and information boundary."""

import math
from dataclasses import replace

import numpy as np
import pytest

from schedx import Process, round_robin
from schedx.adaptive_eval import decision_time, online_observation
from schedx.estimator import (
    E_FEATURES,
    MIN_COMPLETED,
    censored_estimate,
    is_sufficient,
    kaplan_meier,
    survival_at,
)
from schedx.observation import Observation, observe
from schedx.policy_eval import burst_jobs
from schedx.workload import generate_dataset

WORKLOADS = generate_dataset(40, master_seed=33)


def obs_from(completed: dict, pending: dict) -> Observation:
    released = {j: 0 for j in [*completed, *pending]}
    return Observation(t=10, released=released, attained={**completed, **pending}, completed=completed,
                       busy=10, context_switches=0, processes_arrived=1, processes_submitted=1)


# --- Kaplan-Meier --------------------------------------------------------------------


def test_kaplan_meier_hand_example():
    times, surv = kaplan_meier([1, 2, 2, 3, 4], [1, 1, 1, 0, 1])
    assert times.tolist() == [1, 2, 4]
    assert surv == pytest.approx([0.8, 0.4, 0.0])  # 4/5; *2/4; *0/1
    assert survival_at(times, surv, 0) == 1.0 and survival_at(times, surv, 3) == pytest.approx(0.4)


def test_job_censored_at_x_is_still_at_risk_at_x():
    # The unfinished job has received 2 ticks, so its burst is > 2.
    times, surv = kaplan_meier([2, 2], [0, 1])
    assert surv.tolist() == [0.5]


def test_without_censoring_km_is_the_empirical_distribution():
    rng = np.random.default_rng(0)
    b = rng.integers(1, 12, size=40)
    times, surv = kaplan_meier(b, np.ones_like(b))
    for x in range(0, 13):
        assert survival_at(times, surv, x) == pytest.approx((b > x).mean())
    e = censored_estimate(obs_from({f"j{i}": int(v) for i, v in enumerate(b)}, {}))
    assert e["km_censored_frac"] == 0 and e["km_surv_tau"] == 0
    assert e["km_rmst"] == pytest.approx(b.mean())  # tau = max, so min(b, tau) = b
    assert e["km_restricted_cv"] == pytest.approx(b.std() / b.mean())
    assert e["km_rmst_over_naive_mean"] == pytest.approx(1.0)


def test_censored_estimate_hand_example():
    e = censored_estimate(obs_from({"a": 1, "b": 2, "c": 2}, {"d": 3, "e": 1}))
    # durations 1,2,2,3c,1c: at 1: risk 5, d=1 -> .8 ; at 2: risk 3 (2,2,3), d=2 -> .8/3
    assert list(e) == E_FEATURES
    assert e["km_tau"] == 3 and e["km_censored_frac"] == pytest.approx(0.4)
    assert e["km_surv_tau"] == pytest.approx(0.8 / 3)
    assert e["km_rmst"] == pytest.approx(1 + 0.8 + 0.8 / 3)  # S(0)+S(1)+S(2)
    assert e["km_median"] == 2 and e["km_q25"] == 2  # S(1)=.8 > .75, S(2)=.27 <= .5
    assert e["km_rmst_over_naive_mean"] == pytest.approx((1 + 0.8 + 0.8 / 3) / (5 / 3))


def test_unidentified_quantile_is_missing_not_guessed():
    e = censored_estimate(obs_from({"a": 1, "b": 1, "c": 1}, {f"p{i}": 1 for i in range(12)}))
    assert e["km_surv_tau"] == pytest.approx(12 / 15)
    assert math.isnan(e["km_median"])  # median beyond the identified range


def test_km_reduces_bias_under_length_biased_completion():
    """Simulated RR1-style censoring: everyone has received the same service
    level c; jobs with burst <= c are complete. KM's restricted mean is unbiased;
    the completed-only mean is biased low."""
    rng = np.random.default_rng(1)
    naive_err, km_err = [], []
    for _ in range(200):
        b = rng.geometric(1 / 6, size=30)
        c = int(rng.integers(3, 9))
        done = {f"j{i}": int(x) for i, x in enumerate(b) if x <= c}
        pend = {f"j{i}": c for i, x in enumerate(b) if x > c}
        if len(done) < MIN_COMPLETED:
            continue
        e = censored_estimate(obs_from(done, pend))
        truth = np.minimum(b, e["km_tau"]).mean()
        km_err.append(e["km_rmst"] - truth)
        naive_err.append(np.mean(list(done.values())) - truth)
    assert abs(np.mean(km_err)) < 1e-9  # exact with a common censoring level
    assert np.mean(naive_err) < -0.5  # completed-only mean is clearly too low (about -0.9 ticks here)


# --- Insufficient observations ----------------------------------------------------------


def test_insufficient_observation_gives_missing_features():
    o = obs_from({"a": 1, "b": 2}, {"c": 5})
    assert not is_sufficient(o) and not is_sufficient(None)
    assert all(math.isnan(v) for v in censored_estimate(o).values())
    assert is_sufficient(obs_from({"a": 1, "b": 2, "c": 3}, {}))


# --- Information boundary on real workloads ---------------------------------------------------


def _estimate(w, jobs, m=10):
    s = round_robin(jobs, 1)
    t = decision_time(s, len(jobs), m)
    return (None, None) if t is None else (t, censored_estimate(online_observation(w.processes, jobs, s, t)))


def test_estimate_ignores_true_length_of_unfinished_bursts():
    checked = 0
    for w in WORKLOADS:
        jobs = burst_jobs(w.processes)
        t, e = _estimate(w, jobs)
        if t is None:
            continue
        s = round_robin(jobs, 1)
        done = online_observation(w.processes, jobs, s, t).completed
        longer = [j if j.pid in done else replace(j, burst=j.burst + 40) for j in jobs]
        t2, e2 = _estimate(w, longer)
        assert t2 == t and e2 == pytest.approx(e, nan_ok=True)
        checked += 1
    assert checked > 20


def test_late_arriving_jobs_contribute_nothing():
    for w in WORKLOADS[:15]:
        jobs = burst_jobs(w.processes)
        t, e = _estimate(w, jobs)
        if t is None:
            continue
        # Changing jobs released after t (length and exact release time) or
        # adding new late jobs must not change the estimate.
        late = [replace(j, burst=j.burst * 3 + 5, arrival=j.arrival + 7) if j.arrival > t else j for j in jobs]
        extra = late + [Process(f"late{i}", t + 1 + i, 50) for i in range(5)]
        for variant in (late, extra):
            s = round_robin(variant, 1)
            o = online_observation(w.processes, variant, s, t)
            assert censored_estimate(o) == pytest.approx(e, nan_ok=True)


def test_observation_released_set_excludes_future_jobs():
    jobs = [Process("a", 0, 3), Process("b", 0, 1), Process("c", 9, 2)]
    # RR1: a 0-1, b 1-2 (done), a 2-3, ... ; c is released only at t=9.
    o = observe(jobs, round_robin(jobs, 1).timeline, 3, [0, 0, 9])
    assert set(o.released) == {"a", "b"} and o.completed == {"b": 1} and o.attained["a"] == 2
