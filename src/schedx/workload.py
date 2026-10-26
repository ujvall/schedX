"""Reproducible synthetic workload generator (Phase 2).

Each process is an alternating trace CPU, IO, CPU, ..., CPU of integer
bursts. I/O waits are *recorded* for classification; the Phase 1 engine does
not simulate I/O blocking. Generation rules are documented in
docs/phase2_ml.md; labels are computed from the realized traces, never from
the generator's intent.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

CLASSES = ("cpu_bound", "mixed", "io_bound")
REGIMES = ("cpu", "io", "mixed")

# Label thresholds (see docs/phase2_ml.md, "Target labels").
PROCESS_IO_THRESHOLD = 0.5
CPU_BOUND_MAX_SHARE = 0.25
IO_BOUND_MIN_SHARE = 0.75


@dataclass(frozen=True)
class GeneratorConfig:
    """Ranges are inclusive ``[low, high]`` pairs."""

    process_count: tuple[int, int] = (4, 24)
    mean_interarrival: tuple[float, float] = (1.0, 12.0)  # workload intensity
    burst_cv: tuple[float, float] = (0.3, 1.2)  # burst-length variability
    p_io_cpu_regime: float = 0.1
    p_io_io_regime: float = 0.9
    p_io_mixed_regime: tuple[float, float] = (0.3, 0.7)
    cpu_heavy_cpu_mean: tuple[float, float] = (6.0, 25.0)
    cpu_heavy_io_mean: tuple[float, float] = (2.0, 10.0)
    cpu_heavy_bursts: tuple[int, int] = (1, 6)
    io_heavy_cpu_mean: tuple[float, float] = (1.0, 5.0)
    io_heavy_io_mean: tuple[float, float] = (8.0, 40.0)
    io_heavy_bursts: tuple[int, int] = (3, 12)
    priority_max: int = 4

    @classmethod
    def from_dict(cls, d: dict) -> "GeneratorConfig":
        # JSON has no tuples; normalise lists so configs compare/hash equal.
        return cls(**{k: tuple(v) if isinstance(v, list) else v for k, v in d.items()})


@dataclass(frozen=True)
class GeneratedProcess:
    pid: str
    arrival: int
    priority: int
    cpu_bursts: tuple[int, ...]
    io_bursts: tuple[int, ...]  # len(cpu_bursts) - 1 waits between CPU bursts
    intended_kind: str  # generator intent; NOT the label

    @property
    def total_cpu(self) -> int:
        return sum(self.cpu_bursts)

    @property
    def total_io(self) -> int:
        return sum(self.io_bursts)

    @property
    def io_fraction(self) -> float:
        return self.total_io / (self.total_cpu + self.total_io)

    @property
    def kind(self) -> str:
        return process_kind(self.io_fraction)


@dataclass(frozen=True)
class Workload:
    workload_id: str
    master_seed: int
    index: int
    regime: str
    p_io: float
    mean_interarrival: float
    burst_cv: float
    processes: tuple[GeneratedProcess, ...] = field(repr=False)

    @property
    def share_io(self) -> float:
        return sum(p.kind == "io_bound" for p in self.processes) / len(self.processes)

    @property
    def label(self) -> str:
        return workload_class(self.share_io)


def process_kind(io_fraction: float) -> str:
    return "io_bound" if io_fraction > PROCESS_IO_THRESHOLD else "cpu_bound"


def workload_class(share_io: float) -> str:
    if share_io <= CPU_BOUND_MAX_SHARE:
        return "cpu_bound"
    if share_io >= IO_BOUND_MIN_SHARE:
        return "io_bound"
    return "mixed"


def _bursts(rng: np.random.Generator, mean: float, cv: float, n: int) -> tuple[int, ...]:
    # Gamma with the requested mean and coefficient of variation, rounded up
    # to whole ticks so every burst is an int >= 1.
    shape = 1.0 / cv**2
    raw = rng.gamma(shape, mean / shape, size=n)
    return tuple(int(max(1, np.ceil(x))) for x in raw)


def _uniform(rng: np.random.Generator, bounds) -> float:
    return float(rng.uniform(bounds[0], bounds[1]))


def _randint(rng: np.random.Generator, bounds) -> int:
    return int(rng.integers(bounds[0], bounds[1], endpoint=True))


def generate_workload(master_seed: int, index: int, cfg: GeneratorConfig = GeneratorConfig()) -> Workload:
    """Generate workload ``index`` of the dataset seeded by ``master_seed``.

    Depends only on ``(master_seed, index, cfg)``, so any single workload can
    be regenerated without the rest of the dataset.
    """
    rng = np.random.default_rng([master_seed, index])
    regime = REGIMES[int(rng.integers(len(REGIMES)))]
    if regime == "cpu":
        p_io = cfg.p_io_cpu_regime
    elif regime == "io":
        p_io = cfg.p_io_io_regime
    else:
        p_io = _uniform(rng, cfg.p_io_mixed_regime)
    n = _randint(rng, cfg.process_count)
    mean_gap = _uniform(rng, cfg.mean_interarrival)
    cv = _uniform(rng, cfg.burst_cv)

    gaps = np.rint(rng.exponential(mean_gap, size=n - 1)).astype(int)
    arrivals = [0, *np.cumsum(gaps).tolist()]

    procs = []
    for i, arrival in enumerate(arrivals):
        io_heavy = bool(rng.random() < p_io)
        if io_heavy:
            cpu_mean = _uniform(rng, cfg.io_heavy_cpu_mean)
            io_mean = _uniform(rng, cfg.io_heavy_io_mean)
            n_bursts = _randint(rng, cfg.io_heavy_bursts)
        else:
            cpu_mean = _uniform(rng, cfg.cpu_heavy_cpu_mean)
            io_mean = _uniform(rng, cfg.cpu_heavy_io_mean)
            n_bursts = _randint(rng, cfg.cpu_heavy_bursts)
        procs.append(
            GeneratedProcess(
                pid=f"P{i}",
                arrival=int(arrival),
                priority=int(rng.integers(0, cfg.priority_max, endpoint=True)),
                cpu_bursts=_bursts(rng, cpu_mean, cv, n_bursts),
                io_bursts=_bursts(rng, io_mean, cv, n_bursts - 1),
                intended_kind="io_bound" if io_heavy else "cpu_bound",
            )
        )
    return Workload(
        workload_id=f"w{master_seed}-{index:05d}",
        master_seed=master_seed,
        index=index,
        regime=regime,
        p_io=p_io,
        mean_interarrival=mean_gap,
        burst_cv=cv,
        processes=tuple(procs),
    )


def generate_dataset(n_workloads: int, master_seed: int, cfg: GeneratorConfig = GeneratorConfig()) -> list[Workload]:
    return [generate_workload(master_seed, i, cfg) for i in range(n_workloads)]


def _join(xs) -> str:
    return ";".join(map(str, xs))


def _split(s) -> tuple[int, ...]:
    # An empty trace round-trips through CSV as NaN.
    return tuple(int(x) for x in str(s).split(";")) if isinstance(s, str) and s else ()


def to_frames(workloads: list[Workload]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(processes_df, workloads_df)`` as documented in data_dictionary.md."""
    proc_rows, wl_rows = [], []
    for w in workloads:
        for p in w.processes:
            proc_rows.append(
                {
                    "workload_id": w.workload_id,
                    "pid": p.pid,
                    "arrival": p.arrival,
                    "priority": p.priority,
                    "cpu_bursts": _join(p.cpu_bursts),
                    "io_bursts": _join(p.io_bursts),
                    "n_cpu_bursts": len(p.cpu_bursts),
                    "total_cpu": p.total_cpu,
                    "total_io": p.total_io,
                    "io_fraction": p.io_fraction,
                    "process_kind": p.kind,
                    "intended_kind": p.intended_kind,
                }
            )
        wl_rows.append(
            {
                "workload_id": w.workload_id,
                "master_seed": w.master_seed,
                "workload_index": w.index,
                "regime": w.regime,
                "p_io": w.p_io,
                "mean_interarrival": w.mean_interarrival,
                "burst_cv": w.burst_cv,
                "n_processes": len(w.processes),
                "share_io": w.share_io,
                "workload_class": w.label,
            }
        )
    return pd.DataFrame(proc_rows), pd.DataFrame(wl_rows)


def processes_from_frame(df: pd.DataFrame) -> dict[str, tuple[GeneratedProcess, ...]]:
    """Rebuild per-workload process traces from ``processes.csv``."""
    out: dict[str, list[GeneratedProcess]] = {}
    for r in df.itertuples(index=False):
        out.setdefault(r.workload_id, []).append(
            GeneratedProcess(
                pid=r.pid,
                arrival=int(r.arrival),
                priority=int(r.priority),
                cpu_bursts=_split(r.cpu_bursts),
                io_bursts=_split(r.io_bursts),
                intended_kind=r.intended_kind,
            )
        )
    return {k: tuple(v) for k, v in out.items()}


def config_dict(cfg: GeneratorConfig) -> dict:
    return {k: list(v) if isinstance(v, tuple) else v for k, v in asdict(cfg).items()}
