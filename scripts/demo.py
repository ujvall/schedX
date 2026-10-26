"""Run every Phase 1 algorithm on one small workload and print the results.

Usage:  python scripts/demo.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from schedx import Process, compute_metrics, fcfs, priority, round_robin, sjf, srtf  # noqa: E402

WORKLOAD = [
    Process("P1", arrival=0, burst=7, priority=3),
    Process("P2", arrival=2, burst=4, priority=1),
    Process("P3", arrival=4, burst=1, priority=2),
    Process("P4", arrival=5, burst=4, priority=1),
]

RUNS = [
    fcfs(WORKLOAD),
    sjf(WORKLOAD),
    srtf(WORKLOAD),
    round_robin(WORKLOAD, quantum=2),
    priority(WORKLOAD),
    priority(WORKLOAD, preemptive=True),
]

for schedule in RUNS:
    m = compute_metrics(schedule)
    gantt = " ".join(f"[{s.start}-{s.end} {s.pid or 'idle'}]" for s in schedule.timeline)
    print(f"{schedule.algorithm}\n  {gantt}")
    print(
        f"  avg wait={m.avg_waiting:.2f}  avg TAT={m.avg_turnaround:.2f}  avg resp={m.avg_response:.2f}  "
        f"throughput={m.throughput:.3f}/tick  util={m.cpu_utilization:.0%}  switches={m.context_switches}\n"
    )
