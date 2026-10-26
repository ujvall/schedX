"""What an online scheduler has observed by its decision time (Phase 3).

Phase 2's B features assumed the first k bursts of *every* process were
known, even for processes that had not arrived yet. Phase 3 replaces that
with an explicit boundary: an ``Observation`` contains only events that
happened strictly before the decision tick ``t`` in the schedule that
actually ran (plus jobs released at ``t``):

* which jobs have been released, and when;
* how much CPU each released job has received so far;
* the burst length of jobs that have *completed* (a burst's length is only
  known once it ends);
* CPU busy time and context switches so far;
* how many submitted processes have arrived.

Burst lengths of unfinished or unreleased jobs are never exposed. Under a
non-clairvoyant policy such as Round Robin, the schedule before ``t`` does
not depend on them either (tested).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .models import Process, Slice

O_FEATURES = [
    "obs_t_decision",
    "obs_jobs_released",
    "obs_jobs_completed",
    "obs_completed_burst_mean",
    "obs_completed_burst_median",
    "obs_completed_burst_std",
    "obs_completed_burst_cv",
    "obs_completed_burst_max",
    "obs_pending_jobs",
    "obs_pending_attained_mean",
    "obs_busy_fraction",
    "obs_release_rate",
    "obs_switch_rate",
    "obs_procs_arrived_frac",
]


@dataclass(frozen=True)
class Observation:
    t: int
    released: dict[str, int]  # job id -> release tick (<= t)
    attained: dict[str, int]  # job id -> CPU ticks received before t
    completed: dict[str, int]  # job id -> burst length, jobs finished by t
    busy: int
    context_switches: int
    processes_arrived: int
    processes_submitted: int


def observe(jobs: Iterable[Process], timeline: Iterable[Slice], t: int, process_arrivals: Iterable[int]) -> Observation:
    """Build the observation at tick ``t`` from the schedule that actually ran.

    ``timeline`` may extend past ``t``; it is clipped. ``process_arrivals``
    are the submitted processes' arrival ticks (known at submission).
    """
    if t <= 0:
        raise ValueError("decision time must be > 0")
    jobs = list(jobs)
    released = {j.pid: j.arrival for j in jobs if j.arrival <= t}
    attained = dict.fromkeys(released, 0)
    busy = switches = 0
    last = None
    for s in timeline:
        if s.start >= t:
            break
        if s.idle:
            continue
        ran = min(s.end, t) - s.start
        attained[s.pid] += ran
        busy += ran
        if last is not None and s.pid != last:
            switches += 1
        last = s.pid
    # A job is complete once it has received its whole burst before t; the
    # OS knows this because the burst ended (the process blocked or exited).
    burst = {j.pid: j.burst for j in jobs}
    completed = {pid: burst[pid] for pid, a in attained.items() if a == burst[pid]}
    arrivals = list(process_arrivals)
    return Observation(
        t=t,
        released=released,
        attained=attained,
        completed=completed,
        busy=busy,
        context_switches=switches,
        processes_arrived=sum(a <= t for a in arrivals),
        processes_submitted=len(arrivals),
    )


def observation_features(obs: Observation) -> dict[str, float]:
    done = np.array(list(obs.completed.values()), dtype=float)
    pending = [a for pid, a in obs.attained.items() if pid not in obs.completed]
    mean = float(done.mean()) if len(done) else np.nan
    std = float(done.std()) if len(done) else np.nan
    return {
        "obs_t_decision": obs.t,
        "obs_jobs_released": len(obs.released),
        "obs_jobs_completed": len(obs.completed),
        "obs_completed_burst_mean": mean,
        "obs_completed_burst_median": float(np.median(done)) if len(done) else np.nan,
        "obs_completed_burst_std": std,
        "obs_completed_burst_cv": std / mean if len(done) else np.nan,
        "obs_completed_burst_max": float(done.max()) if len(done) else np.nan,
        "obs_pending_jobs": len(pending),
        "obs_pending_attained_mean": float(np.mean(pending)) if pending else np.nan,
        "obs_busy_fraction": obs.busy / obs.t,
        "obs_release_rate": len(obs.released) / obs.t,
        "obs_switch_rate": obs.context_switches / obs.t,
        "obs_procs_arrived_frac": obs.processes_arrived / obs.processes_submitted,
    }
