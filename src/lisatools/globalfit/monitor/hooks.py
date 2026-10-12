"""The after-save hook: page + snapshot on the saver rank, watched.

USER SPEC 2026-09-26: "[production of tar file and html] should occur
right after the file is saved, on the savers rank ... but should throw
major warnings if it is holding anything up."

WHAT "HOLDING ANYTHING UP" MEANS HERE, PRECISELY
------------------------------------------------
The saver rank is the run's ONLY writer, and the sampler's ``save_step``
is a BLOCKING pickled send to it. So while this hook is building a page,
the saver is not in ``recv``, and a sampler that reaches its next save
can block on the send until the build finishes. That is the failure this
watchdog exists to make impossible to miss -- it is invisible from the
sampler's own logs, which show only a slow iteration.

Three independent signals, all reported, because each one catches a
case the others do not:

1. **A save was already queued when we finished.** ``comm.iprobe`` after
   the build. This is the direct evidence that work waited on us.
2. **The build ate a large fraction of the save interval.** Measured
   against the observed gap between the last two saves, not a guess:
   the run's cadence is what it is. Above ``GF_MONITOR_WARN_FRAC``
   (default 0.25) it is reported even if nothing happened to be queued,
   because it means the next one probably will be.
3. **States were dropped.** The saver loop coalesces when its queue
   exceeds ``coalesce_threshold`` and warns; if the drop counter moved
   across our build, we caused it, and that is a DATA LOSS event
   (iterations that never reached the store) rather than a slowdown.

Signal 3 escalates to a banner and AUTO-DISABLES the hook for the rest
of the run. Losing stored iterations to a diagnostic is never a trade
worth making silently, and an operator who wants it back can restart
with a longer ``GF_MONITOR_ITER``.

``GF_MONITOR_PAGE_SHORT=1`` (2026-10-03) builds the SHORT page
(``_short.py``: topline, lnL and leaf count over time, the other sources,
phase-maximised overlap, current noise; under a minute -- the same block
opens the full page) INSTEAD of the full one, at
``RUN_DIR_monitor_short.html``
beside the run folder. The full page stays the default. ``GF_MONITOR_OUT``
names the FULL page only; the short page always takes its default name, so
the two can never overwrite each other. ``GF_MONITOR_PAGE=0`` still turns
the page off whichever kind is selected.

Nothing here can raise into the saver loop.
"""

from __future__ import annotations

import os
import time
from logging import getLogger
from typing import Optional

logger = getLogger(__name__)

__all__ = ["after_save", "build_products", "MonitorWatchdog"]


def _flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default) == "1"


def _num(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except ValueError:
        logger.warning("%s is not a number; using %s", name, default)
        return default


class MonitorWatchdog:
    """Per-run state for the hook: cadence, timing history, escalation."""

    def __init__(self):
        self.disabled_reason: Optional[str] = None
        self.last_save_wall: Optional[float] = None
        self.save_interval: Optional[float] = None
        self.builds = 0
        self.total_build_s = 0.0
        self.delayed = 0

    # -- cadence bookkeeping -------------------------------------------
    def note_save(self) -> None:
        """Called on every save, whether or not a page is built.

        The interval between consecutive saves is the only honest
        denominator for "is this hook expensive": it is the run's real
        cadence, including the F-stat epochs that make some iterations
        many times longer than others.
        """
        now = time.monotonic()
        if self.last_save_wall is not None:
            gap = now - self.last_save_wall
            # Exponential blend rather than last-gap: one F-stat epoch
            # should not convince the watchdog it has all day.
            self.save_interval = (gap if self.save_interval is None
                                  else 0.5 * self.save_interval + 0.5 * gap)
        self.last_save_wall = now

    def should_build(self, i: int) -> bool:
        if self.disabled_reason is not None:
            return False
        if not _flag("GF_MONITOR_AFTER_SAVE"):
            return False
        every = max(int(_num("GF_MONITOR_ITER", 1)), 1)
        return i > 0 and (i % every) == 0

    # -- the report ----------------------------------------------------
    def report(self, elapsed: float, *, queued_after: bool,
               dropped_delta: int) -> None:
        self.builds += 1
        self.total_build_s += elapsed
        frac = (elapsed / self.save_interval) if self.save_interval else None
        warn_frac = _num("GF_MONITOR_WARN_FRAC", 0.25)

        if dropped_delta > 0:
            self.disabled_reason = (
                f"dropped {dropped_delta} save state(s) during a page build")
            logger.warning(
                "\n"
                "########################################################\n"
                "##  [GF_MONITOR] STOPPING: THE PAGE COST STORED        \n"
                "##  ITERATIONS.                                        \n"
                "##                                                     \n"
                "##  %d save state(s) were dropped while a page/snapshot\n"
                "##  build held the saver rank (%.1f s). Those          \n"
                "##  iterations are NOT in the store and cannot be      \n"
                "##  recovered -- the saver coalesces to the newest      \n"
                "##  payload when its queue backs up.                   \n"
                "##                                                     \n"
                "##  The hook is now OFF for the rest of this run. The  \n"
                "##  run itself is unaffected and continues.            \n"
                "##                                                     \n"
                "##  To re-enable with room to breathe, restart with a  \n"
                "##  larger GF_MONITOR_ITER (build every Nth save) or   \n"
                "##  leave GF_MONITOR_AFTER_SAVE unset and build the    \n"
                "##  page offline from a snapshot.                      \n"
                "########################################################",
                dropped_delta, elapsed)
            return

        if queued_after or (frac is not None and frac > warn_frac):
            self.delayed += 1
            logger.warning(
                "\n"
                "########################################################\n"
                "##  [GF_MONITOR] THE PAGE BUILD IS HOLDING THE SAVER   \n"
                "##  RANK UP.                                           \n"
                "##                                                     \n"
                "##  build %.1f s%s                                     \n"
                "##  a save was %swaiting when it finished              \n"
                "##  %d of %d builds have delayed a save                \n"
                "##                                                     \n"
                "##  The sampler's save_step is a BLOCKING send to this \n"
                "##  rank, so this time can appear as a slow iteration  \n"
                "##  with no explanation in the sampler's own log.      \n"
                "##                                                     \n"
                "##  Raise GF_MONITOR_ITER to build less often.         \n"
                "########################################################",
                elapsed,
                (" = %.0f%% of the %.0f s save interval"
                 % (100 * frac, self.save_interval)) if frac else "",
                "" if queued_after else "NOT ",
                self.delayed, self.builds)
        else:
            logger.info(
                "[GF_MONITOR] page+snapshot in %.1f s%s (no save waited; "
                "%d built, %d delayed so far).",
                elapsed,
                (" = %.0f%% of the %.0f s save interval"
                 % (100 * frac, self.save_interval)) if frac else "",
                self.builds, self.delayed)


def after_save(gb_reader, comm, main_rank, i, watchdog, *,
               dropped_so_far: int = 0) -> None:
    """Build the page (and snapshot) if asked, and report the cost.

    Called from the saver loop's quiet gap. Never raises: a diagnostic
    that can kill the run's only writer is worse than no diagnostic.
    """
    try:
        if not watchdog.should_build(i):
            return
        # A queued save always wins. Checked here as well as by the
        # caller because a plot build may have run in between and taken
        # minutes.
        if comm.iprobe(source=main_rank):
            return

        run_dir = os.path.dirname(os.path.abspath(gb_reader.filename))
        elapsed = build_products(run_dir, watchdog)
        if elapsed is None:
            return

        watchdog.report(
            elapsed,
            queued_after=bool(comm.iprobe(source=main_rank)),
            dropped_delta=max(0, dropped_so_far - getattr(
                watchdog, "_dropped_seen", 0)),
        )
        watchdog._dropped_seen = dropped_so_far
    except Exception as e:                        # noqa: BLE001
        logger.warning(
            "[GF_MONITOR] after-save hook failed (%s: %s). Saves are "
            "unaffected.", type(e).__name__, e)


def build_products(run_dir, watchdog=None):
    """Build the page and/or the snapshot tar for ``run_dir``; seconds taken.

    ``None`` when both products are switched off (nothing was built). The
    one implementation behind both callers: the saver-rank hook above and
    the DEDICATED MONITOR RANK (:mod:`.rank`, 2026-10-04). May raise; both
    callers catch.
    """
    out = os.environ.get("GF_MONITOR_OUT")
    timeout = _num("GF_MONITOR_TIMEOUT", 1800.0)

    # TAR-ONLY MODE (GF_MONITOR_PAGE=0). The two products cost very
    # different amounts: the tar is an extract plus a gzip, the page is a
    # whole interpreter's worth of rendering every panel. Building only
    # the tar and rendering the page offline with
    # `python -m lisatools.globalfit.monitor.from_tar SNAP.tar.gz` is the
    # cheap way to keep per-iteration snapshots; GF_MONITOR_SNAPSHOT can
    # likewise turn the tar off while keeping the page.
    _want_page = _flag("GF_MONITOR_PAGE", "1")
    _want_snap = _flag("GF_MONITOR_SNAPSHOT", "1")
    if not _want_page and not _want_snap:
        # Not an error, but it is certainly not what anyone meant: the
        # hook is armed and would produce nothing at all.
        if not getattr(watchdog, "_warned_no_products", False):
            if watchdog is not None:
                watchdog._warned_no_products = True
            logger.warning(
                "[GF_MONITOR] GF_MONITOR_AFTER_SAVE is on but both "
                "GF_MONITOR_PAGE and GF_MONITOR_SNAPSHOT are 0, so "
                "the hook produces nothing. Unset "
                "GF_MONITOR_AFTER_SAVE to disable it properly.")
        return None

    st = time.perf_counter()
    if _want_page and _flag("GF_MONITOR_PAGE_SHORT"):
        # The short page, in place of the full one (seconds, not minutes).
        # Its own default name, never GF_MONITOR_OUT, which names the full
        # page.
        from . import build_short_monitor

        build_short_monitor(run_dir, check=False)
    elif _want_page:
        from . import build_monitor

        build_monitor(run_dir, out, timeout=timeout, check=False)
    if _want_snap:
        from .snapshot import build_snapshot

        # GF_MONITOR_SNAPSHOT_SHORT=1 -> <run>_short.tar.gz: log
        # information and the most recent state, nothing large. It is a
        # SEPARATE file from the full snapshot, so turning it on does not
        # overwrite or invalidate whatever full tar the run has already
        # produced.
        # GF_MONITOR_SNAPSHOT_FSTAT=1 -> the F-stat fit's epoch caches
        # (gb_fstat_fit/**, GBs at 1 yr) ride along; default OUT, only
        # their DONE.json markers ship (user ruling 2026-10-03: "default to
        # leaving them out"). The CLI spelling is --add-fstat.
        build_snapshot(run_dir, short=_flag("GF_MONITOR_SNAPSHOT_SHORT"),
                       include_fstat=_flag("GF_MONITOR_SNAPSHOT_FSTAT"))
    return time.perf_counter() - st
