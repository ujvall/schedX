"""Training-time counterfactual evaluation and inference-time online runs (Phase 3).

Two clearly separated uses of the simulator:

* **Counterfactual evaluation** (training targets, oracles, reporting):
  ``evaluate_fixed`` and ``evaluate_online_candidates`` run *every* candidate
  on a workload. Only training/validation workloads may feed targets.
* **Inference** (``run_controller``): run the default quantum until the
  decision time, build the ``Observation`` from that prefix only, ask the
  controller once, then continue with its choice. The controller never sees
  any candidate's outcome.

Workloads use Phase 2's CPU-only burst-trace projection (``burst_jobs``):
each CPU burst is a job released at its contention-free time. I/O blocking is
not simulated.
"""

from __future__ import annotations

from typing import Callable, Iterable

from .features import pre_execution_features
from .metrics import ScheduleMetrics, compute_metrics
from .models import Process, Schedule
from .observation import Observation, observation_features, observe
from .policy_eval import burst_jobs, candidate_policies
from .scheduler import adaptive_round_robin
from .workload import GeneratedProcess

METRICS = ("avg_turnaround", "avg_waiting", "avg_response", "throughput", "cpu_utilization", "context_switches")


def summarize(m: ScheduleMetrics) -> dict[str, float]:
    return {k: float(getattr(m, k)) for k in METRICS}


def rr_name(q: int) -> str:
    return f"RR{q}"


def evaluate_fixed(jobs: list[Process], quanta: Iterable[int], include_fcfs: bool = True) -> dict[str, dict]:
    """Metrics of each fixed policy over the whole run."""
    policies = candidate_policies(quanta)
    if not include_fcfs:
        policies.pop("FCFS")
    return {name: summarize(compute_metrics(run(jobs))) for name, run in policies.items()}


def decision_time(default_schedule: Schedule, n_jobs: int, after_completions: int) -> int | None:
    """Tick at which the ``after_completions``-th job finishes under the
    default quantum, or ``None`` if the workload ends before a decision could
    change anything (fewer than ``after_completions + 1`` jobs)."""
    if after_completions < 1:
        raise ValueError("after_completions must be >= 1")
    if n_jobs <= after_completions:
        return None
    done: dict[str, int] = {}
    for s in default_schedule.timeline:
        if not s.idle:
            done[s.pid] = s.end  # last slice end = completion
    return sorted(done.values())[after_completions - 1]


def submission_features(processes: tuple[GeneratedProcess, ...]) -> dict[str, float]:
    """Category A features: known when the batch is submitted."""
    return pre_execution_features(processes)


def online_observation(
    processes: tuple[GeneratedProcess, ...], jobs: list[Process], default_schedule: Schedule, t: int
) -> Observation:
    return observe(jobs, default_schedule.timeline, t, [p.arrival for p in processes])


def switched(jobs: list[Process], q0: int, q1: int, t: int) -> Schedule:
    """Run ``q0`` for dispatches before ``t`` and ``q1`` from ``t`` on."""
    return adaptive_round_robin(jobs, lambda now: q0 if now < t else q1, name=f"RR{q0}->{q1}@{t}")


def evaluate_online_candidates(
    processes: tuple[GeneratedProcess, ...],
    jobs: list[Process],
    default_schedule: Schedule,
    q0: int,
    candidates: Iterable[int],
    after_completions: int,
    default_metrics: dict,
) -> tuple[int | None, dict[str, float] | None, dict[int, dict]]:
    """Counterfactual: outcome of switching to each candidate at the decision time.

    Returns ``(t_decision, features, {q1: metrics})``. Without a decision point
    every candidate equals the default run.
    """
    t = decision_time(default_schedule, len(jobs), after_completions)
    if t is None:
        return None, None, {q: default_metrics for q in candidates}
    feats = {**submission_features(processes), **observation_features(online_observation(processes, jobs, default_schedule, t))}
    out = {q: default_metrics if q == q0 else summarize(compute_metrics(switched(jobs, q0, q, t))) for q in candidates}
    return t, feats, out


def run_controller(
    processes: tuple[GeneratedProcess, ...],
    decide: Callable[[dict | None], tuple[int, str]],
    regime: str,
    q0: int,
    after_completions: int | None = None,
) -> tuple[Schedule, int, str, int | None]:
    """Inference path: the controller sees only features, never outcomes.

    ``regime`` is ``"submission"`` (decide once at t=0 from A features) or
    ``"online"`` (run q0, observe until the decision time, decide, continue).
    Returns ``(schedule, chosen_quantum, decision_source, t_decision)``.
    """
    jobs = burst_jobs(processes)
    if regime == "submission":
        q, source = decide(submission_features(processes))
        return adaptive_round_robin(jobs, lambda now: q, name=f"RR{q}"), q, source, 0
    if regime != "online":
        raise ValueError(f"unknown regime {regime!r}")
    default_schedule = adaptive_round_robin(jobs, lambda now: q0, name=f"RR{q0}")
    t = decision_time(default_schedule, len(jobs), after_completions)
    if t is None:
        q, source = decide(None)
        return default_schedule, q0, source, None
    obs = online_observation(processes, jobs, default_schedule, t)
    q, source = decide({**submission_features(processes), **observation_features(obs)})
    return (default_schedule if q == q0 else switched(jobs, q0, q, t)), q, source, t
