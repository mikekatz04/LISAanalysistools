"""The run's own residual, written beside the store for the monitor page.

``GF_RESIDUAL_SNAPSHOT_EVERY=N`` (default 3; 0 turns it off): after every
N-th iteration the head writes
``<store stem>_residual_snapshot.npz`` holding, in the run's own WDM
analysis domain (active band and time span only):

* ``data`` -- the data exactly as the run loaded it
  (``general_info.input_data_residual_array``; for a mojito COMBINED run the
  pre-summed stream with every source class and the noise in it);
* ``residual`` -- the max-lnL cold walker's residual, i.e. that data minus
  EVERY fitted signal (GB, VGB, MBH, EMRI, SOBBH): the array its likelihood
  is computed from;
* ``residuals`` -- EVERY cold walker's residual, shape
  ``(nwalkers, 3, Nf_active, Nt_active)``, in the order of ``walkers``
  (``walkers[k]`` is the cold walker index of ``residuals[k]``; today
  ``arange(nwalkers)``). ``residual`` is ``residuals[walkers.index(walker)]``,
  kept as its own key so a reader written before 2026-10-11 still works;
* ``f_edges`` / ``t_edges`` (Hz / s, active band), ``store_row`` (the
  store row this state was saved as: the resume point plus this job's
  iteration), ``iteration`` (this JOB's iteration count, which restarts at 0
  on every resume), ``walker``, ``lnl`` (every cold walker), ``nf`` / ``nt``
  / ``dt``.

A sidecar written before 2026-10-11 has no ``residuals`` / ``walkers``: it
holds the max-lnL walker alone, and a reader must not treat ``residual`` as
any other walker's.

Why the run writes it rather than the monitor rebuilding it (Mike
2026-10-07): the monitor's data / template / residual panel used to subtract
only GB + VGB from per-type bricks it picked itself, so every MBH merger
in the window showed up as a "residual" (an MBH at 111 d on the 6mo page).
Regenerating every source class outside the run would need the run's exact
generators and settings; the run already holds the exact answer. Why every
walker (2026-10-11): the page's "whitened residual per GB band" panels
compare the max- and the min-lnL walker, and with only the max-lnL one on
disk the min-lnL curve was rebuilt as data minus its GB + VGB alone -- every
MBH / EMRI / SOBBH left in -- which read as a 5-7x excess below 1.5 mHz on
the 9mo page for a walker only ~800 nats behind.

Arrays are stored as float32 (complex64 for a complex WDM basis): this is a
plotting product, not an analysis input. Each walker costs
``3 * Nf_active * Nt_active * 4`` bytes (13.7 MB on the 9mo grid, so four
walkers add ~55 MB to a ~27 MB file; ``residual`` duplicates one of them).
The stack is filled walker by walker, so at most one walker's host copy in
the run's own dtype is alive at a time. Multi-rank: every walker block's
LEAD rank returns its block's residuals through the ``residual_snapshot``
fan-out builtin (see :data:`RESIDUAL_SNAPSHOT_OP`) and the head places each
block at its ``[w0, w1)`` rows; replicas of a block (``R > 1``) are not
asked. The data comes from the head's own copy.

Nothing here may break the run: :func:`maybe_snapshot` logs and carries on.

Where it runs: the head's :class:`~lisatools.globalfit.recipe.Recipe` calls
:func:`maybe_snapshot` from its per-iteration hook (``Recipe.__call__``, eryn's
``stopping_fn``) with the ``snapshot_context`` that
:meth:`~lisatools.globalfit.run.GlobalFit._build_acs_and_recipe` attaches
(the head's ACA, the data holder, the store path, the resume row). The
compute ranks answer the gather through the :data:`RESIDUAL_SNAPSHOT_OP`
builtin registered in :meth:`~lisatools.globalfit.run.GlobalFit.prepare_compute`.
The 9mo/12mo launchers leave ``GF_RESIDUAL_SNAPSHOT_EVERY`` unset, so the
default cadence (every 3rd iteration) applies; the monitor rank reads the file.
"""
from __future__ import annotations

import logging
import os
import time

import numpy as np

logger = logging.getLogger(__name__)

#: Sidecar file name: ``<store without .h5>`` + this suffix.
SNAPSHOT_SUFFIX = "_residual_snapshot.npz"
#: Cadence knob, in iterations; 0 = off.
SNAPSHOT_EVERY_ENV = "GF_RESIDUAL_SNAPSHOT_EVERY"
#: Default cadence: every 3rd iteration (~15 min on the 6mo run; one ~1 s
#: fan-out + a file of (2 + nwalkers) float32 grids, ~80 MB on the 9mo run
#: with 4 walkers, overwritten in place).
SNAPSHOT_EVERY_DEFAULT = 3
#: Fan-out builtin: a walker block's lead returns its block's residuals,
#: ``(rows, 3, Nf_active, Nt_active)`` in the plotting dtype.
RESIDUAL_SNAPSHOT_OP = "residual_snapshot"


def snapshot_every() -> int:
    """The snapshot cadence in iterations, read from ``GF_RESIDUAL_SNAPSHOT_EVERY``.

    Read on every call (no caching), so the knob takes effect at the next
    iteration.

    Returns:
        int: The cadence; :data:`SNAPSHOT_EVERY_DEFAULT` (3) when the variable
        is unset, 0 (off) when it is empty, negative or not an integer (the
        last case logs a warning).
    """
    try:
        return max(0, int(os.environ.get(SNAPSHOT_EVERY_ENV,
                                         str(SNAPSHOT_EVERY_DEFAULT)) or 0))
    except ValueError:
        logger.warning("%s=%r is not an integer; residual snapshots are off.",
                       SNAPSHOT_EVERY_ENV, os.environ.get(SNAPSHOT_EVERY_ENV))
        return 0


def snapshot_path(store_path: str) -> str:
    """The sidecar path for the store at ``store_path``.

    Args:
        store_path: The run's main HDF store path.

    Returns:
        str: ``store_path`` with a trailing ``.h5`` replaced by
        :data:`SNAPSHOT_SUFFIX` (appended when there is no ``.h5``).

    Example:
        >>> from lisatools.globalfit.residual_snapshot import snapshot_path
        >>> snapshot_path("/runs/gf_9mo_parameter_estimation_main.h5")
        '/runs/gf_9mo_parameter_estimation_main_residual_snapshot.npz'
    """
    stem = store_path[:-3] if store_path.endswith(".h5") else store_path
    return stem + SNAPSHOT_SUFFIX


def _plot_array(arr) -> np.ndarray:
    """Host copy in a plotting dtype (float32, or complex64 if complex)."""
    a = arr.get() if hasattr(arr, "get") else np.asarray(arr)
    return a.astype(np.complex64 if np.iscomplexobj(a) else np.float32)


def _stack_residuals(acs, rows):
    """``(len(rows), *shape)`` plotting-dtype stack of ``acs[i].data.arr``.

    Filled walker by walker: one walker's host copy in the run's own dtype
    (float64 on the cluster) is alive at a time and is cast straight into
    its float32 row, so no second full-size copy of the block exists.
    """
    rows = [int(i) for i in rows]
    out = None
    for k, i in enumerate(rows):
        a = acs[i].data.arr
        a = a.get() if hasattr(a, "get") else np.asarray(a)
        if out is None:
            out = np.empty((len(rows),) + a.shape,
                           np.complex64 if np.iscomplexobj(a) else np.float32)
        out[k] = a
        del a
    return out


def walker_residual(model, payload):
    """Fan-out builtin body for :data:`RESIDUAL_SNAPSHOT_OP` on a compute rank.

    Registered by :meth:`~lisatools.globalfit.run.GlobalFit.prepare_compute`.

    Args:
        model (GlobalFitInfo): The rank's model; its
            ``analysis_container_arr`` holds this rank's walker block.
        payload (dict | None): ``{"rows": [local ACA row indices]}``, or
            ``None`` for a rank that was not asked (a replica that is not its
            block's lead).

    Returns:
        np.ndarray | None: ``(len(rows), *residual_shape)`` residual stack in
        the plotting dtype (float32 / complex64), or ``None`` without a
        payload.
    """
    if payload is None:
        return None
    return _stack_residuals(model.analysis_container_arr, payload["rows"])


def _block_leads(layout):
    """One rank per walker block (its lead), as ``concat_blocks`` picks them."""
    leads = getattr(layout, "block_leads", None)
    if leads is None:  # stub layouts without the block API
        leads = ((layout.head_rank,) if getattr(layout, "replica_mode", False)
                 else layout.compute_ranks)
    return tuple(int(r) for r in leads)


def _gather_residuals(acs, fanout, nwalkers):
    """Every cold walker's residual, ``(nwalkers, ...)`` in global walker order."""
    if fanout is None:
        return _stack_residuals(acs, range(int(nwalkers)))
    layout = fanout.layout
    leads = _block_leads(layout)

    def _payload(rank, w0, w1):
        return ({"rows": list(range(int(w1) - int(w0)))}
                if int(rank) in leads else None)

    def _local(payload, model):
        return None if payload is None else _stack_residuals(acs, payload["rows"])

    def _merge(results):
        # Placed by each block's own [w0, w1), not by arrival or dict order:
        # the rows are then right even if the lead order ever stops being
        # the walker order.
        out = None
        for r in leads:
            part = results.get(r)
            w0, w1 = (int(v) for v in layout.block_of(r))
            if part is None or part.shape[0] != w1 - w0:
                raise RuntimeError(
                    f"residual snapshot: block lead rank {r} (walkers "
                    f"{w0}:{w1}) returned "
                    f"{None if part is None else part.shape}")
            if out is None:
                out = np.empty((int(nwalkers),) + part.shape[1:], part.dtype)
            out[w0:w1] = part
        return out

    return fanout.run(RESIDUAL_SNAPSHOT_OP, move=None,
                      per_rank_payload=_payload, local_body=_local,
                      merge=_merge)


def take_snapshot(*, acs, data_holder, store_path, iteration, fanout=None,
                  store_iteration0=0):
    """Write the sidecar (every cold walker's residual); returns its path.

    Gathers every walker's lnL (``acs.likelihood`` single-process, else
    ``fanout.gather_likelihood``), picks the max-lnL walker, gathers every
    walker's residual, and writes the npz atomically (tmp file +
    ``os.replace``). The file keys are listed in the module docstring.

    Args:
        acs (AnalysisContainerArray): The head's container array (its own
            walker block under the walker-block layout).
        data_holder: The run's ``general_info.input_data_residual_array``
            (its ``data_res_arr.arr`` is written as ``data``).
        store_path (str): The main HDF store path (see :func:`snapshot_path`).
        iteration (int): This job's index of the iteration that just ran.
        fanout (WalkerFanout, optional): The head's fan-out; ``None`` in a
            single-process run.
        store_iteration0 (int, optional): The store's iteration when this job
            resumed (0 for a fresh store); ``store_row`` is this plus
            ``iteration``.

    Returns:
        str: The sidecar path written.

    Raises:
        RuntimeError: The gathered residuals do not match the walker count, or
            the data and residual shapes differ.
    """
    if fanout is None:
        _ll = acs.likelihood(complex=False)
        lls = np.asarray(_ll.get() if hasattr(_ll, "get") else _ll, float)
    else:
        lls = np.asarray(fanout.gather_likelihood(acs), float)
    finite = np.where(np.isfinite(lls), lls, -np.inf)
    w = int(np.argmax(finite))
    nw = int(lls.size)

    residuals = _gather_residuals(acs, fanout, nw)
    if residuals is None or residuals.shape[0] != nw:
        raise RuntimeError(
            f"residual snapshot: {nw} walker likelihoods but residuals "
            f"{None if residuals is None else residuals.shape}")
    residual = residuals[w]

    data = _plot_array(data_holder.data_res_arr.arr)
    if data.shape != residual.shape:
        raise RuntimeError(f"residual snapshot: data {data.shape} and residual "
                           f"{residual.shape} differ")
    s = acs[0].data.settings
    path = snapshot_path(store_path)
    tmp = path + ".tmp.npz"
    np.savez(
        tmp, data=data, residual=residual,
        residuals=residuals, walkers=np.arange(nw),
        f_edges=np.asarray(_plot_array(s.f_arr_edges), float),
        t_edges=np.asarray(_plot_array(s.t_arr_edges), float),
        iteration=int(iteration), store_row=int(store_iteration0) + int(iteration),
        walker=w, lnl=lls,
        nf=int(s.Nf), nt=int(s.Nt), dt=float(s.data_dt),
        written_at=time.time())
    os.replace(tmp, path)   # a reader never sees a partial file
    return path


def maybe_snapshot(*, iteration, acs, data_holder, store_path, fanout=None,
                   store_iteration0=0):
    """Cadence gate + never-raise wrapper around :func:`take_snapshot`.

    ``iteration`` is the index of the iteration that just ran; with
    ``N = snapshot_every()`` a snapshot is written after iterations
    ``N-1, 2N-1, ...``. Any exception from :func:`take_snapshot` is logged as
    a ``[RESIDUAL_SNAPSHOT]`` warning and swallowed. Arguments are those of
    :func:`take_snapshot` (all keyword-only), normally passed as the recipe's
    ``snapshot_context``.

    Returns:
        str | None: The path written, or ``None`` when the cadence skips this
        iteration, the cadence is 0, a required input is missing, or the
        write failed.
    """
    every = snapshot_every()
    if every <= 0 or (int(iteration) + 1) % every:
        return None
    if acs is None or data_holder is None or not store_path:
        return None
    try:
        t0 = time.perf_counter()
        path = take_snapshot(acs=acs, data_holder=data_holder,
                             store_path=store_path, iteration=iteration,
                             fanout=fanout, store_iteration0=store_iteration0)
        logger.info("[RESIDUAL_SNAPSHOT] store row %d (job iteration %d) -> %s "
                    "(%.1f s)", int(store_iteration0) + int(iteration),
                    int(iteration), os.path.basename(path),
                    time.perf_counter() - t0)
        return path
    except Exception as exc:  # noqa: BLE001 - a monitor product never breaks the run
        logger.warning("[RESIDUAL_SNAPSHOT] iteration %d failed (%r); the run "
                       "continues.", int(iteration), exc, exc_info=True)
        return None
