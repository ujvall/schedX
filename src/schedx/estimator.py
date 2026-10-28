"""Censoring-aware online estimate of the burst-length distribution (Phase 4).

Under a small default quantum, short bursts finish first, so the completed
bursts seen at the decision time are a biased (short) sample. Unfinished
released jobs still carry information: a job that has received ``a`` ticks
without finishing has a burst **longer than** ``a``. This is right-censored
data, and the Kaplan–Meier (KM) estimator is the standard unbiased way to use
it.

Inputs come only from an ``Observation`` (``schedx.observation``): released
jobs, the CPU each has received before the decision tick, and the lengths
of completed bursts. Unreleased jobs and the true length of unfinished
bursts are never used.

Identification limit: KM identifies the distribution only up to
``tau`` = the largest observed time (completed burst or CPU received so
far). Beyond ``tau`` nothing is known. The features therefore report the
probability mass beyond ``tau`` (``km_surv_tau``) instead of guessing its
shape, and restricted statistics treat that mass as sitting at ``tau``
(a stated lower bound, not an estimate of the full distribution).
"""

from __future__ import annotations

import numpy as np

from .observation import Observation

E_FEATURES = [
    "km_tau",
    "km_censored_frac",
    "km_surv_tau",
    "km_rmst",
    "km_q25",
    "km_median",
    "km_restricted_cv",
    "km_rmst_over_naive_mean",
]

# Fewer completed bursts than this: the estimate is too unstable to use and
# every feature is reported missing, so the controller falls back.
MIN_COMPLETED = 3


def kaplan_meier(durations: np.ndarray, events: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Discrete-time KM survival ``S(x) = P(burst > x)`` at each distinct event time.

    ``durations``: completed burst length (event) or CPU received so far
    (censored; the burst is known to be longer). A job censored at ``x``
    is still at risk at ``x``, because its burst exceeds ``x``.
    """
    durations = np.asarray(durations, dtype=float)
    events = np.asarray(events, dtype=bool)
    times = np.unique(durations[events])
    surv, s = [], 1.0
    for x in times:
        at_risk = np.sum(durations >= x)
        d = np.sum((durations == x) & events)
        s *= 1.0 - d / at_risk
        surv.append(s)
    return times, np.array(surv)


def survival_at(times: np.ndarray, surv: np.ndarray, x: float) -> float:
    """S(x) from the KM step function (S = 1 before the first event)."""
    i = np.searchsorted(times, x, side="right") - 1
    return 1.0 if i < 0 else float(surv[i])


def _quantile(times: np.ndarray, surv: np.ndarray, p: float) -> float:
    """Smallest event time with P(burst <= x) >= p, or NaN if not identified."""
    hit = np.nonzero(surv <= 1.0 - p + 1e-12)[0]
    return float(times[hit[0]]) if len(hit) else np.nan


def censored_estimate(obs: Observation) -> dict[str, float]:
    """E features from the observation at the decision tick. All NaN (and
    ``insufficient``) when fewer than ``MIN_COMPLETED`` bursts have completed."""
    ids = list(obs.released)
    durations = np.array([obs.completed.get(j, obs.attained[j]) for j in ids], dtype=float)
    events = np.array([j in obs.completed for j in ids], dtype=bool)
    n_done = int(events.sum())
    nan = dict.fromkeys(E_FEATURES, np.nan)
    if n_done < MIN_COMPLETED:
        return nan
    tau = float(durations.max())
    times, surv = kaplan_meier(durations, events)
    s_tau = survival_at(times, surv, tau)
    # Restricted mean E[min(B, tau)] = sum_{k=0}^{tau-1} P(B > k) for integer bursts.
    rmst = float(sum(survival_at(times, surv, k) for k in range(int(tau))))
    # Restricted distribution: KM point masses at event times, tail mass at tau.
    prev = np.concatenate([[1.0], surv[:-1]])
    support = np.concatenate([times, [tau]])
    mass = np.concatenate([prev - surv, [s_tau]])
    mean_r = float((support * mass).sum())
    var_r = float((((support - mean_r) ** 2) * mass).sum())
    naive_mean = float(durations[events].mean())
    return {
        "km_tau": tau,
        "km_censored_frac": float(1 - n_done / len(ids)),
        "km_surv_tau": s_tau,
        "km_rmst": rmst,
        "km_q25": _quantile(times, surv, 0.25),
        "km_median": _quantile(times, surv, 0.5),
        "km_restricted_cv": float(np.sqrt(var_r) / mean_r) if mean_r > 0 else np.nan,
        "km_rmst_over_naive_mean": rmst / naive_mean,
    }


def is_sufficient(obs: Observation | None) -> bool:
    return obs is not None and len(obs.completed) >= MIN_COMPLETED
