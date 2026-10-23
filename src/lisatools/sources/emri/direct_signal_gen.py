"""Global-fit adapter for the direct-to-WDM EMRI template (``EMRI_LIKELIHOOD=direct``).

:class:`EMRIDirectWDMSignalGen` turns a batch of waveform-basis rows into WDM
coefficients on the run's ACTIVE box, the same array a production template
(:class:`~lisatools.sources.emri.response.EMRIWaveWrap`: FEW + ResponseWrapper +
dense TD->WDM transform) lands on:

* :class:`~lisatools.sources.emri.wdm_direct.EMRIDirectWDM` assembles every row on the
  wavelet grid of the run (same ``Nf``, ``Nt``, ``dt``; its own ``pixel_edge`` layers at
  each grid end stay zero, so they must lie outside the active time box), restricted to the
  run's active frequency band (``f_band``, set here);
* the adapter crops that to the run domain's ``active_slice_t``;
* a row FEW refuses (out of its domain of validity) is reported in the returned ``ok``
  mask instead of failing the batch (the move scores it at the ``-1e300`` floor, as the
  container path does).

Optional trajectory fan-out (``traj_workers > 0``, ``EMRI_TRAJ_WORKERS``): the rows'
inspirals are integrated on a small process pool (``few.trajectory.pool``) before the
batch, and FEW's generator is served from a cache that holds only the current batch.
The pool's workers are started EAGERLY, all at once, with the ``__main__`` script hidden
from ``multiprocessing``'s spawn preparation: a spawned child otherwise re-imports the
parent's main script (``run_combined_staged.py``, which imports the global-fit stack and,
through it, MPI). A pool that fails to start or breaks later is shut down and the batch
continues serially (logged once).
"""

from __future__ import annotations

import contextlib
import logging
import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from ...utils.exceptions import WaveformDomainError

logger = logging.getLogger(__name__)

__all__ = ["EMRIDirectWDMSignalGen", "spawn_executor_without_main"]

#: seconds the eager worker start may take before the pool is abandoned
POOL_START_TIMEOUT = 300.0
#: the pool's [EMRI_DIRECT] counters are logged on the 1st, (1+N)th, ... pooled batch
POOL_LOG_EVERY = 50


@contextlib.contextmanager
def _main_hidden_from_spawn():
    """Hide ``__main__.__file__`` while child processes are spawned.

    ``multiprocessing.spawn.get_preparation_data`` makes every child import the
    parent's main SCRIPT (as ``__mp_main__``) when ``__main__`` has a ``__file__`` and
    no ``__spec__`` -- i.e. under ``python script.py``. Without it the child imports
    only what the submitted callable needs."""
    main = sys.modules.get("__main__")
    had = main is not None and "__file__" in vars(main)
    saved = vars(main).pop("__file__") if had else None
    try:
        yield
    finally:
        if had:
            main.__file__ = saved


def spawn_executor_without_main(n_workers, warm_fn, warm_args=(), timeout=POOL_START_TIMEOUT):
    """A spawn-context ``ProcessPoolExecutor`` with ALL ``n_workers`` processes started.

    ``n_workers`` warm-up tasks ``warm_fn(*warm_args)`` are submitted at once with the
    main script hidden (:func:`_main_hidden_from_spawn`), so every worker exists before
    the context is left -- ``ProcessPoolExecutor`` never spawns again unless one dies
    (which breaks the pool).

    Args:
        n_workers (int): Number of worker processes.
        warm_fn (callable): Picklable warm-up task run once per worker.
        warm_args (tuple, optional): Arguments of ``warm_fn``. Default ``()``.
        timeout (float, optional): Seconds each warm-up result may take. Default
            :data:`POOL_START_TIMEOUT` (300).

    Returns:
        concurrent.futures.ProcessPoolExecutor: The started executor.

    Raises:
        Exception: Whatever the warm-up raised (including ``TimeoutError``), or
            ``RuntimeError`` if fewer than ``n_workers`` processes exist after the warm-up;
            the executor is shut down and its processes killed first.
    """
    n_workers = int(n_workers)
    ex = ProcessPoolExecutor(max_workers=n_workers, mp_context=multiprocessing.get_context("spawn"))
    try:
        with _main_hidden_from_spawn():
            futs = [ex.submit(warm_fn, *warm_args) for _ in range(n_workers)]
            for f in futs:
                f.result(timeout=timeout)
        # every worker must exist now: one spawned later would import the main script
        started = len(getattr(ex, "_processes", None) or {})
        if started != n_workers:
            raise RuntimeError(
                f"pool warm-up started {started} of {n_workers} workers; a worker spawned "
                "after the warm-up would import the parent's main script"
            )
    except BaseException:
        _kill_executor(ex)
        raise
    return ex


def _kill_executor(ex):
    """Shut ``ex`` down without waiting (pending futures cancelled) and kill its processes."""
    procs = list(getattr(ex, "_processes", {}).values())
    ex.shutdown(wait=False, cancel_futures=True)
    for p in procs:
        with contextlib.suppress(Exception):
            p.kill()


class EMRIDirectWDMSignalGen:
    """Rows -> direct-to-WDM EMRI templates on the run's active box.

    The ``EMRI_LIKELIHOOD=direct`` template generator of the 9mo / 12mo runs: built per device
    by ``globalfit.stock.erebor.source_runtime.get_emri_direct_gen``; the EMRI move
    (:class:`~lisatools.globalfit.moves.emridirectmove.EMRIDirectLikeMove`) scores batches
    with :meth:`templates`, and the engine's EMRI ``signal_gen`` builds residual templates
    through :meth:`__call__`. On construction it sets ``direct.f_band`` to the run's active
    frequency layers (when unset), so the direct template builds only those layers.

    Args:
        direct (EMRIDirectWDM): :class:`~lisatools.sources.emri.wdm_direct.EMRIDirectWDM`
            built on the run's FULL wavelet grid (``WDMSettings(Nf, Nt, dt)``, no active box).
        domain_settings (WDMSettings): the run's :class:`~lisatools.domains.WDMSettings`
            (active box).
        nchannels (int, optional): channels kept (the production wrap's ``nchannels``);
            ``None`` keeps all.
        runtime_kwargs (dict, optional): per-call FEW kwargs every template gets (the
            production wrap's ``runtime_kwargs``: the mode-selection threshold); call kwargs
            win.
        traj_workers (int, optional): trajectory pool size (``EMRI_TRAJ_WORKERS``); 0
            (default) integrates serially.

    Attributes:
        last_stats (dict): ``direct.last_stats`` of the last :meth:`templates` call.
        pool_totals (dict): Pool counters (``batches``, ``rows``, ``computed``, ``errors``).

    Raises:
        ValueError: If ``direct``'s grid is not the run's ``(Nf, Nt, dt)``, or the run's
            active time box reaches the ``pixel_edge`` layers the direct template leaves
            zero at each grid end.

    Example:
        .. code-block:: python

            gen = EMRIDirectWDMSignalGen(direct, run_wdm_settings, nchannels=3,
                                         runtime_kwargs={"mode_selection_threshold": 1e-5})
            arr, ok = gen.templates(rows)    # (n, nch, Nf_active, Nt_active), (n,) bool
            sig = gen(*rows[0])              # WDMSignal on the run domain
    """

    def __init__(self, direct, domain_settings, *, nchannels=None, runtime_kwargs=None,
                 traj_workers=0):
        full = direct.wdm
        dom = domain_settings
        if (int(full.Nf), int(full.Nt)) != (int(dom.Nf), int(dom.Nt)) or abs(
            float(full.data_dt) - float(dom.data_dt)
        ) > 1e-12:
            raise ValueError(
                f"EMRI direct template grid (Nf={full.Nf}, Nt={full.Nt}, dt={full.data_dt}) is not "
                f"the run's wavelet grid (Nf={dom.Nf}, Nt={dom.Nt}, dt={dom.data_dt})."
            )
        edge = int(direct.pixel_edge)
        if int(dom.ind_min_t) < edge or int(dom.ind_max_t) >= int(dom.Nt) - edge:
            raise ValueError(
                f"the run's active time box [{dom.ind_min_t}, {dom.ind_max_t}] reaches the "
                f"{edge} edge layers the direct template leaves zero at each end of the "
                f"{dom.Nt}-layer grid."
            )
        self.direct = direct
        self.domain_settings = dom
        # the direct generator builds the run's active frequency band only (its batch arrays
        # hold layers [m_lo, m_hi)); the time crop stays here
        sl_f = dom.active_slice_f
        if hasattr(direct, "f_band") and direct.f_band is None and int(sl_f.stop) <= int(full.Nf):
            direct.f_band = (int(sl_f.start), int(sl_f.stop))
        self.nchannels = None if nchannels is None else int(nchannels)
        self.runtime_kwargs = dict(runtime_kwargs or {})
        self.traj_workers = max(0, int(traj_workers))
        self._executor = None
        self._traj_pool = None
        self._traj_cache = None
        self.pool_totals = dict(batches=0, rows=0, computed=0, errors=0)
        self.last_stats = {}

    # ------------------------------------------------------------------
    # templates
    # ------------------------------------------------------------------

    def templates(self, rows, **kwargs):
        """Templates of many waveform-basis rows on the run's active box.

        Runs the optional trajectory pre-computation, then ONE
        :meth:`~lisatools.sources.emri.wdm_direct.EMRIDirectWDM.batch` call over all rows
        (``skip_domain_errors=True``), and crops to the active frequency layers and time box.

        Args:
            rows (np.ndarray): ``(n, 14)`` FEW waveform-basis rows (one row is accepted).
            **kwargs: Per-call FEW kwargs, merged over ``runtime_kwargs``.

        Returns:
            tuple: ``(arr, ok)``. ``arr`` is ``(n, nch, Nf_active, Nt_active)`` float64 on the
            backend's array module; a row with ``ok[i] = False`` (FEW domain refusal) is
            all-zero there. ``ok`` is a host ``(n,)`` bool array.

        Raises:
            ValueError: If the direct template's band does not hold the active layers.
        """
        kw = dict(self.runtime_kwargs)
        kw.update(kwargs)
        rows = [np.asarray(r, dtype=np.float64) for r in np.atleast_2d(np.asarray(rows, dtype=np.float64))]
        n = len(rows)
        self._precompute_trajectories(rows, kw)
        try:
            full = self.direct.batch(rows, chunk_rows=max(1, n), skip_domain_errors=True, **kw)
        finally:
            if self._traj_cache is not None:
                self._traj_cache.clear()
        self.last_stats = dict(self.direct.last_stats)
        ok = np.ones(n, dtype=bool)
        ok[list(self.direct.last_failed_rows)] = False
        sl_f = self.domain_settings.active_slice_f
        sl_t = self.domain_settings.active_slice_t
        band = getattr(self.direct, "f_band", None)
        if band is not None:                                  # full holds layers [m_lo, m_hi) only
            if int(full.shape[2]) != band[1] - band[0] or not band[0] <= sl_f.start <= sl_f.stop <= band[1]:
                raise ValueError(f"direct template band {band} does not hold the active layers {sl_f}")
            sl_f = slice(sl_f.start - band[0], sl_f.stop - band[0])
        nch = full.shape[1] if self.nchannels is None else self.nchannels
        arr = full[:, :nch, sl_f, sl_t].copy()
        del full
        return arr, ok

    def __call__(self, *params, **kwargs):
        """ONE template as a :class:`~lisatools.domains.WDMSignal` on the run domain.

        Args:
            *params: The 14 FEW waveform-basis parameters.
            **kwargs: Per-call FEW kwargs (see :meth:`templates`).

        Returns:
            WDMSignal: ``(nch, Nf_active, Nt_active)`` on ``domain_settings``.

        Raises:
            WaveformDomainError: When FEW refuses the parameters.
        """
        from ...domains import WDMSignal

        arr, ok = self.templates(np.asarray(params, dtype=np.float64)[None, :], **kwargs)
        if not ok[0]:
            raise WaveformDomainError("EMRI direct template: FEW refused the parameters")
        return WDMSignal(arr[0], self.domain_settings)

    # ------------------------------------------------------------------
    # trajectory fan-out
    # ------------------------------------------------------------------

    def _precompute_trajectories(self, rows, kw):
        """Integrate the rows' inspirals on the pool into the trajectory cache (no-op without a
        pool or for fewer rows than workers); any pool failure disables the pool (logged)."""
        # a chunk smaller than the pool (an in-model step: ntemps x walkers-per-rank rows)
        # is integrated serially: the capture + IPC round trip would cost more than the
        # one or two trajectories it parallelises; the eigen-table sweeps fill the pool
        if self.traj_workers <= 0 or len(rows) < max(2, self.traj_workers):
            return
        try:
            if self._traj_pool is None:
                self._start_pool()
            direct = self.direct
            self.last_pool_stats = st = self._traj_cache.precompute(
                lambda *p: direct._mode_list(p, kw), rows, self._traj_pool
            )
            tot = self.pool_totals
            tot["batches"] += 1
            for k in ("rows", "computed", "errors"):
                tot[k] += int(st.get(k, 0))
            if tot["batches"] % POOL_LOG_EVERY == 1:
                # a pool whose captured calls stopped matching the real ones adds work
                # without saving any: hits must track the rows computed
                logger.info(
                    "[EMRI_DIRECT] trajectory pool: %d batches, %d rows, %d trajectories "
                    "computed, %d worker errors; cache hits %s, misses %s.", tot["batches"],
                    tot["rows"], tot["computed"], tot["errors"],
                    getattr(self._traj_cache, "hits", "?"), getattr(self._traj_cache, "misses", "?"),
                )
        except Exception as exc:  # noqa: BLE001 - the pool is an accelerator only
            logger.warning(
                "[EMRI_DIRECT] trajectory pool disabled (%s: %s); integrating serially.",
                type(exc).__name__, exc,
            )
            self.close_pool()
            self.traj_workers = 0

    def _start_pool(self):
        """Start the spawn pool and install FEW's ``TrajectoryCache`` on the generator."""
        from few.trajectory.pool import (
            TrajectoryCache,
            TrajectoryPool,
            _worker_run,
            inspiral_init_kwargs_from,
        )

        gen = self.direct.few_gen
        init = inspiral_init_kwargs_from(gen)
        ex = spawn_executor_without_main(self.traj_workers, _worker_run, (init, []))
        self._executor = ex
        self._traj_pool = TrajectoryPool(n_workers=self.traj_workers, inspiral_init_kwargs=init,
                                         executor=ex)
        self._traj_cache = TrajectoryCache.install(gen)
        cores = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
        logger.info("[EMRI_DIRECT] trajectory pool: %d spawn workers started (this process may "
                    "run on %s cores).", self.traj_workers, cores)
        if cores is not None and cores < self.traj_workers + 1:
            logger.warning(
                "[EMRI_DIRECT] %d trajectory workers share %d core(s) with this rank (MPI "
                "pinning or --cpus-per-task): the fan-out cannot run in parallel.",
                self.traj_workers, cores,
            )

    def close_pool(self):
        """Shut the pool down and restore the FEW generator's own inspiral module."""
        if self._traj_cache is not None:
            from few.trajectory.pool import TrajectoryCache

            TrajectoryCache.uninstall(self.direct.few_gen)
        if self._executor is not None:
            _kill_executor(self._executor)
        self._executor = self._traj_pool = self._traj_cache = None
