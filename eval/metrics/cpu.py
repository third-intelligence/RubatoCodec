"""CPU count helper shared by the CPU-parallel metrics."""

import os


def usable_cpu_count() -> int:
    """Return the number of CPUs this process may use.

    Uses the scheduler affinity mask where available (Linux), which respects
    container and job-scheduler CPU limits, and falls back to os.cpu_count().
    """
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1
