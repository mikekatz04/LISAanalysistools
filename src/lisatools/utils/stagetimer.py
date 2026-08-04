"""Opt-in, device-synced stage timing for hot evaluation paths.

cProfile is a poor tool for these paths: GPU kernels are asynchronous, so the
cost lands on whichever python frame next synchronizes, and python-heavy frames
are inflated ~3x by the profiler itself. These helpers accumulate wall time with
an explicit device sync at each stage boundary instead.

Off by default (near-zero overhead); enable with ``MBHTDIONFLY_TIMING=1``.
Dump with :func:`report_timing`.
"""

import os
import time
from collections import defaultdict

import numpy as np

TIMING: bool = os.environ.get("MBHTDIONFLY_TIMING", "0") != "0"
TIMERS: dict = defaultdict(float)
TIMER_COUNTS: dict = defaultdict(int)


def _sync(xp) -> None:
    if xp is not np:
        try:
            xp.cuda.runtime.deviceSynchronize()
        except Exception:
            pass


class stage:
    """Context manager accumulating device-synced wall time into ``TIMERS``."""

    __slots__ = ("name", "xp", "t0")

    def __init__(self, name: str, xp):
        self.name = name
        self.xp = xp

    def __enter__(self):
        if TIMING:
            _sync(self.xp)
            self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if TIMING:
            _sync(self.xp)
            TIMERS[self.name] += time.perf_counter() - self.t0
            TIMER_COUNTS[self.name] += 1
        return False


def record(name: str, dt: float) -> None:
    """Add ``dt`` seconds to the ``name`` accumulator (already synced)."""
    TIMERS[name] += dt
    TIMER_COUNTS[name] += 1


def _atexit_report() -> None:  # pragma: no cover - diagnostic only
    print("\n" + report_timing(), flush=True)


def _signal_report(signum, frame) -> None:  # pragma: no cover - diagnostic only
    """Dump on SIGTERM/SIGINT.

    Slurm sends SIGTERM before SIGKILL at walltime; without this a job that
    runs out of time loses the whole accumulated report, since atexit does not
    run on an uncaught fatal signal.
    """
    print(f"\n[stagetimer] signal {signum} -- dumping accumulated timers",
          flush=True)
    print(report_timing(), flush=True)
    signal.signal(signum, signal.SIG_DFL)
    os.kill(os.getpid(), signum)


if TIMING:
    import atexit
    import signal

    atexit.register(_atexit_report)
    for _sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(_sig, _signal_report)
        except (ValueError, OSError):
            pass  # not on the main thread / not supported here


def report_timing(reset: bool = False) -> str:
    """Human-readable dump of the accumulated stage timers."""
    if not TIMERS:
        return "stage timing: no data (set MBHTDIONFLY_TIMING=1)"
    total = TIMERS.get("total") or sum(TIMERS.values())
    lines = ["stage timing (device-synced wall seconds):"]
    for name, val in sorted(TIMERS.items(), key=lambda kv: -kv[1]):
        n = TIMER_COUNTS[name]
        lines.append(
            f"  {name:<30s} {val:9.2f} s  {100 * val / total:5.1f}%  "
            f"{n:6d} calls  {val / max(n, 1):8.4f} s/call"
        )
    if reset:
        TIMERS.clear()
        TIMER_COUNTS.clear()
    return "\n".join(lines)
