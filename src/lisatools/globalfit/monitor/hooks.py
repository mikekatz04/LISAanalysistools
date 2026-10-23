"""The after-save hook: page + snapshot on the saver rank, watched.

Two entry points:

- :func:`build_products` -- THE build: the monitor page
  (:func:`~lisatools.globalfit.monitor.build_monitor`, or the short page)
  and the snapshot tar
  (:func:`~lisatools.globalfit.monitor.snapshot.build_snapshot`) for one run
  directory. In the 9mo/12mo jobs it runs on the dedicated monitor rank
  (``GF_MONITOR_RANK=1``) inside the memory-guarded child of
  :mod:`~lisatools.globalfit.monitor.buildguard` (its default target).
- :func:`after_save` with :class:`MonitorWatchdog` -- the saver-rank hook,
  called by the saver loop
  (:func:`~lisatools.globalfit.hdfbackend.save_to_backend_asynchronously_and_plot`)
  after every save ONLY when no monitor rank is active (the launcher
  disarms ``GF_MONITOR_RANK`` when the task count does not fit). The saver
  always builds a watchdog and calls :meth:`MonitorWatchdog.note_save`.

Knobs (``"1"`` = on unless stated): ``GF_MONITOR_AFTER_SAVE`` (default
``0``; launchers ``1``; gates both the hook and the monitor rank's builds),
``GF_MONITOR_ITER`` (build every Nth save; default 1 here, launchers 10),
``GF_MONITOR_WARN_FRAC`` (default 0.25), ``GF_MONITOR_PAGE`` (default 1),
``GF_MONITOR_PAGE_SHORT`` (default 0), ``GF_MONITOR_OUT`` (full-page path),
``GF_MONITOR_TIMEOUT`` (child-render timeout, default 1800 s),
``GF_MONITOR_SNAPSHOT`` (default 1), ``GF_MONITOR_SNAPSHOT_SHORT`` and
``GF_MONITOR_SNAPSHOT_FSTAT`` (default 0).

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
    """``True`` when env ``name`` (or ``default`` when unset) is exactly ``"1"``."""
    return os.environ.get(name, default) == "1"


def _num(name: str, default: float) -> float:
    """Env ``name`` as a float; ``default`` when unset or empty (warned when unparsable)."""
    try:
        return float(os.environ.get(name, "") or default)
    except ValueError:
        logger.warning("%s is not a number; using %s", name, default)
        return default


class MonitorWatchdog:
    """Per-run state for the hook: cadence, timing history, escalation.

    One instance per saver loop. :meth:`note_save` runs on every save;
    :meth:`should_build` and :meth:`report` only on the saver-hook path
    (:func:`after_save`).

    Attributes:
        disabled_reason (str or None): Set when a build cost stored
            iterations; the hook then stays off for the rest of the run.
        last_save_wall (float or None): ``time.monotonic()`` of the last save.
        save_interval (float or None): Blended save interval [s]
            (``0.5 * old + 0.5 * newest gap``).
        builds (int): Builds reported.
        total_build_s (float): Total build time [s].
        delayed (int): Builds that held a save up (queued save or over the
            ``GF_MONITOR_WARN_FRAC`` share of the interval).
    """

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
        """Return whether save ``i`` should build.

        True when the hook is not disabled, ``GF_MONITOR_AFTER_SAVE=1``, and
        ``i > 0`` is a multiple of ``GF_MONITOR_ITER`` (default 1).

        Args:
            i: The saver's save counter.
        """
        if self.disabled_reason is not None:
            return False
        if not _flag("GF_MONITOR_AFTER_SAVE"):
            return False
        every = max(int(_num("GF_MONITOR_ITER", 1)), 1)
        return i > 0 and (i % every) == 0

    # -- the report ----------------------------------------------------
    def report(self, elapsed: float, *, queued_after: bool,
               dropped_delta: int) -> None:
        """Log the cost of one build and escalate (module docstring, signals 1-3).

        A dropped save state disables the hook (banner); a queued save or a
        build above ``GF_MONITOR_WARN_FRAC`` of the save interval is a
        "holding the saver up" banner; otherwise one INFO line.

        Args:
            elapsed: Build time [s].
            queued_after: Whether a save was already waiting when the build
                finished.
            dropped_delta: Save states dropped by the saver during the build.
        """
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

    Skips the build when :meth:`MonitorWatchdog.should_build` says no or a
    save is already queued (``comm.iprobe`` on ``main_rank``); otherwise
    calls :func:`build_products` on the store's directory and reports the
    cost.

    Args:
        gb_reader: The saver's backend (its ``filename`` locates the run
            directory).
        comm: The run communicator.
        main_rank (int): The head rank (the sender of saves).
        i (int): The saver's save counter.
        watchdog (MonitorWatchdog): The saver's watchdog.
        dropped_so_far: Total save states the saver has dropped.
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

    Products, by env knob: the full page (``GF_MONITOR_PAGE``, default on;
    path ``GF_MONITOR_OUT`` or ``<run_dir>_monitor.html``; build errors are
    logged, not raised) or, with ``GF_MONITOR_PAGE_SHORT=1``, the short page
    in its place; then the snapshot tar (``GF_MONITOR_SNAPSHOT``, default on;
    ``GF_MONITOR_SNAPSHOT_SHORT=1`` for the short tar,
    ``GF_MONITOR_SNAPSHOT_FSTAT=1`` to include the F-stat caches). The page
    reads the run's ``GF_MONITOR_ADD_INFO`` itself.

    Args:
        run_dir (str): The run directory.
        watchdog (MonitorWatchdog, optional): Used only to warn once when
            both products are off.

    Returns:
        float or None: Seconds taken, or ``None`` when nothing was built.

    Example:
        .. code-block:: python

            from lisatools.globalfit.monitor.hooks import build_products

            seconds = build_products("/data/gf_prod_9mo_v9")
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

        # No add_info here, on purpose: the page then reads the run's own
        # GF_MONITOR_ADD_INFO export (the 9mo / 1yr launchers overlay the 6mo
        # run on the comparison figures; _add_info.py).
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
