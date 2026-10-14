"""Specialized ``eryn`` HDF5 backends for the global fit (GB / MBH / EMRI / GF)."""

import os
import shutil
import time
import warnings
from logging import getLogger
from typing import Optional

import h5py
import numpy as np
from eryn.backends import HDFBackend as eryn_HDFBackend

from ..domains import (
    DomainSettingsBase,
    FDSettings,
    STFTSettings,
    WDMSettings,
)
from . import midit_checkpoint
from .plot import RunResultsProduction
from .state import (
    EMRIState,
    GBState,
    GFState,
    MBHState,
    ModuleSubState,
    SOBBHState,
)

import logging

logger = logging.getLogger(__name__)

logger = getLogger(__name__)

#: on-disk storage-format version written by GFHDFBackend.reset. 2 = the
#: cold-chain storage rework (main backend stores only the engine ladder;
#: each branch's tempered ensemble lives in its sub-backend).
GF_FORMAT_VERSION = 2


# ----------------------------------------------------------------------
# Domain-settings (de)serialization helpers.
#
# Each concrete DomainSettings class exposes ``args`` / ``kwargs``
# properties that round-trip its constructor. We persist those, plus a
# class-name discriminator, so the backend can rebuild the exact
# DomainSettings instance from disk later. No string-level "fd"/"wdm"
# flag — the discriminator is the class name itself.
# ----------------------------------------------------------------------


_DOMAIN_SETTINGS_CLASSES = {
    "FDSettings": FDSettings,
    "STFTSettings": STFTSettings,
    "WDMSettings": WDMSettings,
}


def _domain_attr_value(v):
    """Coerce a kwarg value into something HDF5-friendly (or skip None)."""
    if v is None:
        return None
    if isinstance(v, (str, bytes, bool, int, float, np.floating, np.integer)):
        return v
    if isinstance(v, np.ndarray):
        return v
    # Fall back to string repr for backends / orbits / etc. The user can
    # reconstruct those separately; we just want the round-trip to not
    # crash.
    return str(v)


def _write_domain_settings(group: h5py.Group, settings: DomainSettingsBase) -> None:
    """Persist a :class:`DomainSettingsBase` into ``group`` under ``"domain_settings"``.

    Stores the class name plus the ``args`` / ``kwargs`` from the
    instance. For WDM the array-shaped kwargs (``window`` / ``omega``)
    are written as datasets; everything else lives in attributes.
    """
    if "domain_settings" in group:
        del group["domain_settings"]
    ds_group = group.create_group("domain_settings")
    ds_group.attrs["class_name"] = type(settings).__name__

    args_group = ds_group.create_group("args")
    for i, val in enumerate(settings.args):
        coerced = _domain_attr_value(val)
        if coerced is None:
            continue
        if isinstance(coerced, np.ndarray):
            args_group.create_dataset(str(i), data=coerced)
        else:
            args_group.attrs[str(i)] = coerced

    kwargs_group = ds_group.create_group("kwargs")
    for key, val in settings.kwargs.items():
        coerced = _domain_attr_value(val)
        if coerced is None:
            continue
        if isinstance(coerced, np.ndarray):
            kwargs_group.create_dataset(key, data=coerced)
        else:
            kwargs_group.attrs[key] = coerced


def _read_domain_settings(
    group: h5py.Group, force_backend: Optional[str] = None
) -> Optional[DomainSettingsBase]:
    """Reconstruct a :class:`DomainSettingsBase` from a ``"domain_settings"`` subgroup.

    Returns ``None`` if the group does not contain a serialised settings
    block (e.g. older files).
    """
    if "domain_settings" not in group:
        return None
    ds_group = group["domain_settings"]
    class_name = ds_group.attrs.get("class_name")
    if isinstance(class_name, bytes):
        class_name = class_name.decode()
    cls = _DOMAIN_SETTINGS_CLASSES.get(class_name)
    if cls is None:
        raise ValueError(f"Unknown domain settings class on disk: {class_name!r}")

    args_group = ds_group["args"]
    indices = sorted(
        list(args_group.attrs.keys()) + list(args_group.keys()),
        key=lambda s: int(s),
    )
    args = []
    for idx in indices:
        if idx in args_group.attrs:
            args.append(args_group.attrs[idx])
        else:
            args.append(args_group[idx][...])

    kwargs_group = ds_group["kwargs"]
    kwargs = {}
    for key in kwargs_group.attrs.keys():
        val = kwargs_group.attrs[key]
        if isinstance(val, bytes):
            val = val.decode()
        kwargs[key] = val
    for key in kwargs_group.keys():
        kwargs[key] = kwargs_group[key][...]
    if force_backend is not None:
        kwargs["force_backend"] = force_backend
    return cls(*args, **kwargs)



def _validate_resume_readable(path: str) -> None:
    """Read what a resume reads; raise if the file cannot serve one.

    Deliberately narrow: ``get_last_sample`` pulls the LAST stored row of
    every sub-backend dataset, and that is the read that raises
    ``OSError: filter returned failure during read`` on a torn gzip chunk.
    Checking one row per dataset costs milliseconds even on the 23-month
    store, whereas decompressing everything would cost more than the copy.
    """
    with h5py.File(path, "r") as f:
        root = f["global_fit"] if "global_fit" in f else f
        it = int(root.attrs.get("iteration", 0))
        row = max(it - 1, 0)

        # ---- THE RECIPE GROUP (2026-09-29) --------------------------------
        # A resume reads this too -- prepare_main -> add_recipe opens every
        # recipe step group and reads its attrs -- and until now this
        # function did not, which is how the 3-month run lost BOTH copies at
        # once.
        #
        # Job 666 died at resume on
        #   KeyError: 'Unable to synchronously open object
        #              (message not aligned)'
        # a malformed OBJECT HEADER on one recipe step group, while every
        # chain and sub-backend in the same file read perfectly. A recipe
        # step group is tiny but its attrs (status, order num,
        # completed_iteration, start_iteration) are rewritten at every stage
        # transition, and an attr write rewrites the header -- so a kill or
        # MPI abort in that window tears exactly this and nothing else.
        #
        # ⚠ WHY IT COST THE BACKUP TOO. `_atomic_backup_copy` is a BYTE copy
        # (shutil.copyfile) gated on THIS function. Because the sweep below
        # stops at sub_backend, a torn recipe header passed validation, was
        # duplicated faithfully into the running backup copy, and
        # `promote_backup_if_store_unreadable` never fired either -- it
        # validates the primary, sees no fault in the region it checks and
        # reports the primary fine. Both generations were lost to a gap in
        # the check, not to two independent failures.
        #
        # ⚠ AHEAD of the sub_backend early-return on purpose: a noise-only
        # store has no sub_backend and can still carry a torn recipe.
        # Reading it is microseconds (five groups, two attrs each).
        if bool(root.attrs.get("has_recipe", False)) and "recipe" in root:
            recipe_group = root["recipe"]
            for key in recipe_group:
                try:
                    step = recipe_group[key]      # opens the object header
                    step.attrs["status"]
                    step.attrs["order num"]
                except Exception as e:            # noqa: BLE001
                    raise ValueError(
                        f"recipe step {key!r} is unreadable "
                        f"({type(e).__name__}: {e}). The link resolves but the "
                        "object does not, which is a torn object header -- the "
                        "signature of a kill during an attr write at a stage "
                        "transition. A resume reads this in add_recipe and "
                        "would die there."
                    ) from e

        if "sub_backend" not in root:
            return
        for branch in root["sub_backend"]:
            grp = root["sub_backend"][branch]
            for key in grp:
                dset = grp[key]
                if getattr(dset, "ndim", 0) and dset.shape[0] > row:
                    dset[row]          # decompresses that row's chunks
                else:
                    dset[()]

        # ---- SEMANTIC completeness of the last row ------------------------
        # Readability is not enough. The main backend and each sub-backend are
        # NOT flushed atomically, so a kill BETWEEN them (a spot preemption --
        # ~1.7/day on the 3-month partition) leaves a row whose main half is
        # written and whose sub-backend half is still zeros. That row
        # decompresses perfectly, so every check above passes; the run then
        # resumes into it and MPI-ABORTS at the first propose with
        #   [vgb] cold-chain inds mismatch between the main state and its
        #   sub-state (550 differing leaf slots)
        # from GFState.check_cold_row -- after the ranks are up, minutes in,
        # and with no self-heal because the promotion path never fired.
        # (Observed: 3mo v8 10-walker, job 464, stored iteration 1.)
        #
        # So run that same cold-row comparison HERE, on the h5, where raising
        # hands the store to promote_backup_if_store_unreadable() -- the
        # backup lags by at most one save step, so recovery costs one
        # iteration instead of a dead job.
        #
        # Comparing inds (rather than "is the row all zeros") is what keeps
        # this free of false positives: a branch that legitimately holds zero
        # leaves early in a run has all-False inds on BOTH sides, which agree.
        main_inds = root.get("inds")
        if main_inds is not None:
            for branch in root["sub_backend"]:
                sub = root["sub_backend"][branch]
                if branch not in main_inds or "inds" not in sub:
                    continue
                m, s = main_inds[branch], sub["inds"]
                if m.shape[0] <= row or s.shape[0] <= row:
                    continue
                a = np.asarray(m[row])
                b = np.asarray(s[row])
                # main: (ntemps_main, nsamplers, nwalkers, nleaves) -> cold row
                # sub : (ntemps_sub, nwalkers, nleaves)             -> cold row
                while a.ndim > b.ndim - 1:
                    a = a[0]
                b = b[0]
                if a.shape != b.shape:
                    continue
                n_diff = int(np.count_nonzero(a != b))
                if n_diff:
                    raise ValueError(
                        f"[{branch}] stored row {row} is INCOMPLETE: the main "
                        f"backend and the sub-backend disagree on "
                        f"{n_diff} cold leaf slots. This is a torn save -- the "
                        "two halves are not flushed atomically, so a job killed "
                        "between them (spot preemption) leaves exactly this. "
                        "Resuming it would MPI-abort at the first propose."
                    )


def promote_backup_if_store_unreadable(path: str) -> bool:
    """Self-heal a torn store by promoting the running backup copy.

    WHY THIS EXISTS. The runs live on a SPOT partition, so a job can be
    preempted at any instant -- including mid-write. Appending into a
    gzip-compressed HDF5 file that is killed part-way leaves chunks that
    open fine and only fail when READ, so the NEXT job dies at resume with
    ``filter returned failure during read``. That happened twice; both
    times the fix was mechanical (swap in ``*_running_backup_copy.h5``),
    which is exactly the kind of thing that should not need a human at
    03:00. With the MPI abort in place the failure is at least fast and
    loud -- but fast and loud still means the run is stopped.

    So: validate the primary the way a RESUME reads it; if that raises,
    validate the backup, and only if the backup is sound swap them --
    preserving the damaged file as ``*_CORRUPT_<n>.h5`` for forensics
    rather than deleting evidence. The backup lags by at most one save
    step (``backup_iter=1`` on the dedicated saver rank), so the cost of
    recovery is bounded and small.

    Returns True if a promotion happened. Raises nothing on the happy
    path; if BOTH files are unreadable it returns False and lets the
    caller fail normally -- there is nothing safe left to do, and
    inventing a state would be worse than stopping.
    """
    if not os.path.exists(path):
        return False
    try:
        _validate_resume_readable(path)
        return False                      # primary is fine, nothing to do
    except Exception as primary_err:
        backup = path[:-3] + "_running_backup_copy.h5"
        logger.error(
            "STORE UNREADABLE at resume (%s: %s). This is the signature of a "
            "job killed mid-write -- e.g. a spot preemption.",
            type(primary_err).__name__, primary_err)
        if not os.path.exists(backup):
            logger.error("No running backup copy beside it; cannot self-heal.")
            return False
        try:
            _validate_resume_readable(backup)
        except Exception as backup_err:
            logger.error(
                "The running backup is ALSO unreadable (%s: %s); not "
                "promoting it. Both copies are damaged.",
                type(backup_err).__name__, backup_err)
            return False
        # Keep the damaged file: it is the only record of the lost
        # iterations and of how the tear looks.
        n = 0
        while os.path.exists(f"{path[:-3]}_CORRUPT_{n}.h5"):
            n += 1
        corrupt_name = f"{path[:-3]}_CORRUPT_{n}.h5"
        os.replace(path, corrupt_name)
        shutil.copyfile(backup, path)
        try:
            with open(path, "rb") as fh:
                os.fsync(fh.fileno())
        except Exception:
            pass
        logger.error(
            "SELF-HEALED: promoted %s over the damaged store (kept as %s). "
            "The run resumes from the backup's last complete iteration, so "
            "at most the iterations written since the last backup are lost.",
            os.path.basename(backup), os.path.basename(corrupt_name))
        return True


#: Per-row dataset in the main group: the leg-ender name each row was saved
#: after ("" for rows saved by a stage without legs). See lisatools.globalfit.legs.
SAVED_AFTER_DS = "saved_after"


def write_saved_after(g, iteration, name):
    """Record that row ``iteration - 1`` of main group ``g`` was saved after
    leg-ender ``name``. Creates the resizable string dataset on first use
    (earlier rows read back as unknown) and grows it as rows are added."""
    n = int(iteration)
    if n <= 0:
        return
    if SAVED_AFTER_DS not in g:
        ds = g.create_dataset(
            SAVED_AFTER_DS, shape=(n,), maxshape=(None,),
            dtype=h5py.string_dtype(encoding="utf-8"), chunks=(64,))
        ds[:] = [""] * n
    else:
        ds = g[SAVED_AFTER_DS]
        if int(ds.shape[0]) < n:
            old = int(ds.shape[0])
            ds.resize((n,))
            ds[old:n] = [""] * (n - old)
    ds[n - 1] = str(name)


def stamp_move_order(g, step_name, order):
    """Stamp the stage's ordered move-name list on its recipe group, ONCE.

    An existing stamp is kept: it records the composition the stage's rows
    were written under, which is what a resume compares the live list to.
    Returns True when written.
    """
    import json

    grp = g.get("recipe")
    if grp is None or step_name not in grp:
        return False
    s = grp[step_name]
    if s.attrs.get("move_order") is not None:
        return False
    s.attrs["move_order"] = json.dumps([str(x) for x in order])
    return True


def _atomic_backup_copy(src: str) -> None:
    """Refresh ``<src>_running_backup_copy.h5`` so it can never be torn.

    WHY (2026-08-15 forensics): a job killed while writing the live store
    leaves permanently unreadable gzip chunks -- that is what stranded the
    3-month run, whose primary raised ``filter returned failure during read``
    on ``sub_backend/gb/*`` while this backup stayed intact and became the
    only way back. The backup is therefore the last line of defence, and the
    previous implementation copied STRAIGHT ONTO it: a kill during that copy
    (a many-second window on a multi-GB store) would have corrupted the
    fallback too, leaving nothing.

    Write to a sibling temp file, flush it to disk, then ``os.replace`` it
    into position. ``replace`` is atomic on POSIX, so a reader -- or a kill
    at ANY instant -- sees either the previous complete backup or the new
    complete one, never a half-written file. The temp lives in the same
    directory so the rename stays within one filesystem (across mounts it
    would silently degrade to a copy, reintroducing the very window this
    closes).

    Never raises: a failed backup must not kill a healthy run. The stale
    backup simply survives, which is strictly better than dying here.
    """
    dst = src[:-3] + "_running_backup_copy.h5"
    tmp = dst + ".tmp"
    try:
        shutil.copyfile(src, tmp)
        # VALIDATE BEFORE REPLACING. copyfile is a BYTE copy -- it neither
        # knows nor cares whether the source is a sound HDF5 file -- so a
        # primary that has already torn would be faithfully copied over the
        # one good fallback, and both generations would be lost. Read exactly
        # what a RESUME reads (the last stored row of every sub-backend):
        # that is the code path that actually failed, it is one row rather
        # than a multi-GB sweep, and a failure here is an EARLY WARNING that
        # the live store is damaged while the previous backup is still good.
        _validate_resume_readable(tmp)
        # copyfile returns once the data is in the page cache; force it to
        # the device before the rename, so a machine-level crash cannot
        # leave the new name pointing at unwritten blocks.
        with open(tmp, "rb") as fh:
            os.fsync(fh.fileno())
        os.replace(tmp, dst)          # atomic swap
    except Exception as e:            # noqa: BLE001 -- see docstring
        logger.warning(
            "running backup copy NOT refreshed (%s: %s). The previous backup "
            "is untouched and still usable. NOTE: if this is an OSError about "
            "reading data, the LIVE STORE is already damaged -- that backup "
            "is now your recovery point.", type(e).__name__, e)
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass


# ---- store writes routed to the saver rank ---------------------------------
# USER RULING 2026-10-07: "In general, let's make sure all file saves go
# through the saver rank." The rows always did; the recipe-group stamps
# (stage completion + the next stage's start, GF_PERSIST_STAGE_START's start,
# the galfor ratchet's stop flag) were written by the HEAD opening the store
# in append mode itself, and when one coincided with the saver writing a row
# two processes wrote one HDF5 file and the recipe group tore (3mo 2026-09-28;
# 6mo replica_pe -> full_pe 2026-10-04). With a saver rank the head now sends
# them as a ``{STORE_WRITE_KEY: method, "args", "kwargs"}`` request on the SAME
# channel as the rows (default tag, so MPI's non-overtaking order puts it after
# the row it closes and before the next), the saver runs the SAME backend
# method in arrival order, and acknowledges on STORE_WRITE_ACK_TAG.

#: Payload key of a routed store write (head -> saver, on the row channel).
STORE_WRITE_KEY = "store_write"

#: Tag of the saver's acknowledgement (saver -> head). Only acks use it, so an
#: ack can never be mistaken for anything else the head receives on the run
#: communicator. (The monitor rank's notices are on the WORLD communicator.)
STORE_WRITE_ACK_TAG = 7802

#: The backend methods the saver runs for the head. Anything else is refused.
ROUTED_STORE_WRITES = ("completed_recipe_step", "stamp_stage_start", "stamp_stage_flag")


def _is_store_write(payload) -> bool:
    return isinstance(payload, dict) and STORE_WRITE_KEY in payload


def _run_routed_store_write(gb_reader, comm, main_rank, payload) -> None:
    """Saver side: run one routed store write, then acknowledge it to the head.

    Never raises into the loop. A failure is logged HERE with the method name
    and its arguments, and travels back in the acknowledgement; the head
    re-raises it, i.e. exactly where the direct write would have raised.
    """
    method = payload.get(STORE_WRITE_KEY)
    args = tuple(payload.get("args", ()) or ())
    kwargs = dict(payload.get("kwargs", {}) or {})
    t0 = time.perf_counter()
    try:
        if method not in ROUTED_STORE_WRITES:
            raise ValueError(
                f"{method!r} is not a routed store write (allowed: "
                f"{ROUTED_STORE_WRITES})")
        ack = {"ok": True, "result": getattr(gb_reader, method)(*args, **kwargs)}
    except Exception as exc:  # noqa: BLE001 -- reported to the head, never fatal here
        logger.exception(
            "[STORE-WRITE] saver rank FAILED %s(args=%r, kwargs=%r): %s: %s",
            method, args, kwargs, type(exc).__name__, exc)
        ack = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    else:
        try:
            _rows = int(gb_reader.iteration)
        except Exception:  # noqa: BLE001 -- a log detail
            _rows = None
        logger.info(
            "[STORE-WRITE] saver rank ran %s(args=%r, kwargs=%r) with %s row(s) "
            "stored, in %.3f s", method, args, kwargs, _rows,
            time.perf_counter() - t0)
    comm.send(ack, dest=main_rank, tag=STORE_WRITE_ACK_TAG)


# THE MONITOR HOOK MOVED INTO THE PACKAGE (2026-09-26). This file used to
# carry the script lookup and the subprocess call inline; both now live in
# lisatools.globalfit.monitor, which ships the generator itself, so there
# is no filesystem hunt for a scripts/ directory and no second copy of the
# invocation. Imported lazily inside the loop -- hdfbackend is imported by
# every rank, and only the saver ever needs this.

def save_to_backend_asynchronously_and_plot(
    gb_reader,
    comm,
    main_rank,
    plot_container=None,
    plot_iter=100,
    backup_iter=1,
    coalesce_threshold=3,
    warm_self_hook=None,
):
    """Async writer loop: receive states from ``main_rank`` and persist to HDF5.

    Runs on the dedicated results rank (np >= 3). Receives
    ``{save_args, save_kwargs}`` payloads, calls
    :meth:`GFHDFBackend.save_step_main`, periodically copies the file to a
    running backup, regenerates the diagnostic plots, and exits when sent
    ``{"finish_run": True}``. Also runs the head's routed store writes
    (``{STORE_WRITE_KEY: method, ...}``, one of :data:`ROUTED_STORE_WRITES`)
    in arrival order with the rows and acknowledges each on
    :data:`STORE_WRITE_ACK_TAG`, so this rank is the only process that
    writes the store while the run samples (user ruling 2026-10-07).

    Saves always take priority over plots (parallel-resources plan P2):

    - The loop blocks directly on ``recv`` — the incoming message IS the
      clock (no fixed sleep).
    - Backpressure: if the loop falls behind and more than
      ``coalesce_threshold`` save payloads are queued, only the newest is
      written and the number of dropped intermediate states is logged
      loudly (never silently).
    - Plots run only when the incoming queue is empty; a queued save defers
      the plot to the next quiet gap, and a plot failure never kills the
      saver.

    Args:
        gb_reader: The :class:`GFHDFBackend` to write into.
        comm: MPI communicator.
        main_rank: Rank that produces the save payloads.
        plot_container: Optional eryn ``PlotContainer``; its ``backend`` is
            pointed at ``gb_reader`` so plots render from the file this
            loop writes. ``None`` disables plotting.
        plot_iter: Save-step cadence for plot regeneration.
        backup_iter: Save-step cadence at which the HDF file is copied to a
            ``*_running_backup_copy.h5``. DEFAULT 1 -- EVERY save step.
            This loop only ever runs on the DEDICATED saver rank (see
            ``run.py``'s ``elif self.rank == self.results_rank``), so the
            copy is already off the sampler's critical path: the main rank
            hands off and returns to proposing while this happens. The old
            cadence of 5 was sized for a saver sharing the sampler's
            process, where every copy stole wall time from the science; here
            it only widened the recovery gap. Since a torn primary is
            recovered from this file, "every iteration" means losing at most
            ONE iteration instead of up to five. ``None`` is treated as 1.
            MEASURED LIMIT (2026-09-14, 3mo v8 job 479): the copy is a
            full-file copy + fsync, and the sampler's ``save_step`` is a
            BLOCKING pickled send to this rank -- so once the store is
            large the sampler waits for the previous copy: 60-80 s per
            iteration on a 14 GB store (24% of a 5.4 min PE iteration),
            0.5 s on a 1.5 GB one. The production scripts now export
            ``BACKUP_ITER=10`` (``general_info.backup_iter``); the
            mid-iteration checkpoint (600 s) still bounds the loss of a
            torn store to that interval.
        coalesce_threshold: Queue depth above which intermediate states are
            dropped in favour of the newest.
        warm_self_hook: Optional
            :class:`~lisatools.globalfit.warmstart.self_refit.WarmSelfHook`
            (``GB_WARM_SELF=1``): after every batch that saved a row -- once
            the plots and the monitor notice are done -- it builds a GB
            warm-start self-refit epoch when the stage's cadence is due,
            reading the store through THIS backend's ``open("r")`` and
            closing it before it returns. Watched like the monitor hook
            (a build that holds a save up is reported); never raises.
            ``None`` (default) leaves this loop exactly as it was.
    """
    logger.info("results rank: starting async save/plot loop")
    if plot_container is not None:
        plot_container.backend = gb_reader
    run = True
    i = 0
    total_dropped = 0
    # Per-run state for the page/snapshot hook: cadence history and the
    # escalation that turns the hook off if it ever costs stored
    # iterations. Constructed unconditionally and cheaply -- it also
    # records the save interval when the hook is off, so turning it on
    # mid-campaign has a real denominator to judge against.
    from .monitor.hooks import MonitorWatchdog

    _monitor_watchdog = MonitorWatchdog()
    while run:
        payloads = [comm.recv(source=main_rank)]
        # Drain the backlog without blocking: saves must never queue behind
        # this loop's own work.
        while comm.iprobe(source=main_rank):
            payloads.append(comm.recv(source=main_rank))

        finish = any(
            isinstance(p, dict) and p.get("finish_run", False) for p in payloads
        )
        _rows = [k for k, p in enumerate(payloads)
                 if isinstance(p, dict) and "save_args" in p]
        keep = set(_rows)
        if len(_rows) > coalesce_threshold:
            # the newest row, plus the row right before each routed store
            # write: a stage stamp must land AFTER the row it closes
            keep = {_rows[-1]}
            _last = None
            for k, p in enumerate(payloads):
                if k in _rows:
                    _last = k
                elif _is_store_write(p) and _last is not None:
                    keep.add(_last)
            dropped = len(_rows) - len(keep)
            total_dropped += dropped
            logger.warning(
                "results rank fell behind: dropping %d intermediate save "
                "state(s), keeping the newest (%d dropped so far this run).",
                dropped,
                total_dropped,
            )
        states = [payloads[k] for k in sorted(keep)]

        # IN ARRIVAL ORDER: rows and routed store writes interleave exactly as
        # the head sent them (see STORE_WRITE_KEY).
        for k, save_dict in enumerate(payloads):
            if _is_store_write(save_dict):
                _run_routed_store_write(gb_reader, comm, main_rank, save_dict)
                continue
            if k not in keep:
                continue
            st = time.perf_counter()
            gb_reader.save_step_main(
                *save_dict["save_args"], **save_dict["save_kwargs"]
            )
            logger.debug("save step took %.3f s", time.perf_counter() - st)
            i += 1
            # Every save, built page or not: the gap between consecutive
            # saves is the only honest denominator for "is the page hook
            # expensive", and it has to include the saves it skipped.
            _monitor_watchdog.note_save()
            if warm_self_hook is not None:
                warm_self_hook.note_save()
            # None -> every step: general_info.backup_iter defaults to None
            # in the engine settings, and `i % None` is a TypeError, so a
            # run that never set it would have crashed the saver here.
            _every = int(backup_iter) if backup_iter else 1
            if (i % _every) == 0:
                _atomic_backup_copy(gb_reader.filename)

        if finish:
            run = False
            continue

        # A batch that saved no row (a routed store write alone) has nothing
        # new for the plots or the page; re-running them for the same save
        # count would only build them twice.
        if not states:
            continue

        # Plots only in a quiet gap: a queued save always wins.
        if (
            plot_container is not None
            and i > 0
            and (i % plot_iter) == 0
            and not comm.iprobe(source=main_rank)
        ):
            st = time.perf_counter()
            try:
                plot_container.produce_plots()
            except Exception:
                logger.exception(
                    "diagnostic plotting failed on the results rank "
                    "(saves continue unaffected)"
                )
            else:
                logger.info(
                    "diagnostic plots regenerated in %.1f s",
                    time.perf_counter() - st,
                )

        # THE HTML PAGE + SNAPSHOT TAR (2026-09-26). Same quiet-gap
        # contract as the plots directly above -- a queued save always
        # wins -- but its OWN cadence: plot_iter is 100 by default and the
        # point of the page is that it tracks the run.
        #
        # DEFAULT OFF. This rank is the run's only writer; opting a
        # ~140-second, 2.5 GB subprocess into that position is a decision
        # for the launcher to make explicitly (GF_MONITOR_AFTER_SAVE=1),
        # not a default that arrives with an upgrade.
        #
        # ``total_dropped`` is handed over so the watchdog can tell a
        # slowdown (recoverable, warn) from lost stored iterations
        # (unrecoverable, stop building). Re-checking iprobe is inside
        # the hook: the plot build above may have taken minutes.
        #
        # DEDICATED MONITOR RANK (GF_MONITOR_RANK=1, 2026-10-04): when the
        # driver split one off, this loop only POSTS a non-blocking notice
        # and goes straight back to recv -- the build happens on that rank,
        # so it can never hold a save (job 717: the save after every
        # in-place build stalled ~400 s with every GPU idle). See
        # monitor/rank.py.
        from .monitor import rank as _monitor_rank

        if _monitor_rank.active():
            if states:
                _monitor_rank.notify_saved(gb_reader.filename, i, total_dropped)
        else:
            from .monitor.hooks import after_save as _monitor_after_save

            _monitor_after_save(gb_reader, comm, main_rank, i,
                                _monitor_watchdog, dropped_so_far=total_dropped)

        # GB WARM-START SELF REFIT (GB_WARM_SELF=1, user rulings 2026-10-13):
        # LAST in the quiet gap, after the monitor notice went out. The hook
        # re-checks iprobe itself (a queued save always wins) and reports
        # through its own watchdog when a build held a save up.
        if warm_self_hook is not None:
            warm_self_hook(gb_reader, comm, main_rank, i,
                           dropped_so_far=total_dropped)

    logger.info(
        "results rank: finished (%d states saved, %d dropped)", i, total_dropped
    )
    return


class GFHDFBackend(eryn_HDFBackend):
    """``eryn`` HDF backend extended with per-branch sub-backends and recipe tracking.

    Each branch (``gb``, ``mbh``, ...) may register its own backend class via
    ``sub_backend``; on every save step the global :class:`GFHDFBackend` calls
    each sub-backend's :meth:`save_step` so per-branch counters / band info /
    temperature ladders persist alongside the eryn state.

    Args:
        *args: Forwarded to :class:`eryn.backends.HDFBackend`.
        name: Top-level HDF5 group the run is stored under. Defaults to
            ``"global_fit"`` (eryn's own default is ``"mcmc"``). Files written
            before this default changed store their run under ``"mcmc"``; such
            a file is detected on open and honored, with a warning, so existing
            runs stay readable and resumable.
        comm: Optional MPI communicator (required together with
            ``save_plot_rank`` for asynchronous saves).
        sub_backend: Mapping ``{branch_name: backend_class}`` for branch-
            specific datasets.
        sub_state_bases: Mapping ``{branch_name: state_class}`` to wrap
            states pulled from the file with the right :class:`GFState`
            subclass.
        save_plot_rank: Rank to forward save jobs to, and the recipe-group
            stamps (:data:`ROUTED_STORE_WRITES`) with them.
        **kwargs: Forwarded to :class:`eryn.backends.HDFBackend`.

    Raises:
        ValueError: If only one of ``comm`` / ``save_plot_rank`` is provided.
    """

    #: Group name used by files written before the ``"mcmc"`` -> ``"global_fit"``
    #: rename (eryn's ``HDFBackend`` default).
    LEGACY_NAME = "mcmc"

    def __init__(
        self,
        *args,
        name: str = "global_fit",
        comm=None,
        sub_backend=None,
        sub_state_bases=None,
        save_plot_rank=None,
        **kwargs,
    ):

        super().__init__(*args, name=name, **kwargs)

        # Back-compat: an existing file written under the old default keeps its
        # "mcmc" group. Adopt it rather than silently looking for a group that
        # is not there (which would read as "uninitialized" and start the run
        # over the top of real samples).
        if name != self.LEGACY_NAME and self.filename is not None:
            try:
                if os.path.exists(self.filename):
                    with h5py.File(self.filename, "r") as f:
                        if name not in f and self.LEGACY_NAME in f:
                            warnings.warn(
                                f"{self.filename!r} stores its run under the legacy "
                                f"{self.LEGACY_NAME!r} group (the default is now "
                                f"{name!r}); reading it as {self.LEGACY_NAME!r}. Pass "
                                f"name={self.LEGACY_NAME!r} explicitly to silence this.",
                                DeprecationWarning,
                                stacklevel=2,
                            )
                            self.name = self.LEGACY_NAME
            except OSError:
                # unreadable/locked file: leave the requested name in place and
                # let the real open() below report the problem properly
                pass

        if comm is not None or save_plot_rank is not None:
            if comm is None or save_plot_rank is None:
                raise ValueError("If providing comm/save_plot_rank, must provide both.")

        self.comm = comm
        self.save_plot_rank = save_plot_rank

        self.sub_backend = sub_backend
        if self.sub_backend is not None:
            # Sub-backends write into OUR file, under f[name]["sub_backend"], so
            # they must use the same group name we resolved above -- including
            # the legacy fallback. They subclass eryn's HDFBackend directly, so
            # left to their own default they would look for "mcmc" while the
            # parent writes "global_fit".
            self.sub_backend = {
                key: self.sub_backend[key](*args, name=self.name, **kwargs)
                for key in self.sub_backend
                if self.sub_backend[key] is not None
            }
            # Stamp each sub-backend with its branch so per-branch group
            # names resolve correctly (e.g. GBHDFBackend serves both the
            # "gb" and "vgb" branches; default class attr is "gb").
            for key, sub in self.sub_backend.items():
                sub.sub_name = key

        self.sub_state_bases = sub_state_bases
        self.recipe_added = False

    @property
    def format_version(self):
        """The on-disk ``gf_format_version`` (``None`` for pre-rework files)."""
        with self.open() as f:
            return f[self.name].attrs.get("gf_format_version", None)

    def check_format_version(self, context: str):
        """Raise with an actionable message when the file predates format v2."""
        version = self.format_version
        if version is None or int(version) < GF_FORMAT_VERSION:
            raise ValueError(
                f"{context}: {self.filename!r} was written with the "
                f"pre-cold-chain storage layout (gf_format_version="
                f"{version}). This code stores only the cold chain in the "
                "main backend, with each branch's tempered ensemble in its "
                "sub-backend, and cannot resume old-layout files (clean "
                "break, no migration). Start a fresh backend, or check out "
                "the pre-rework code to keep using this file."
            )

    # ------------------------------------------------------------------
    # Domain-settings persistence
    # ------------------------------------------------------------------

    def write_domain_settings(self, settings: DomainSettingsBase) -> None:
        """Record the run's :class:`DomainSettingsBase` into the file.

        Idempotent: a second call replaces the previously stored block.
        """
        with self.open("a") as f:
            _write_domain_settings(f[self.name], settings)

    def read_domain_settings(
        self, force_backend: Optional[str] = None
    ) -> Optional[DomainSettingsBase]:
        """Read back the run's :class:`DomainSettingsBase` (or ``None``)."""
        with self.open() as f:
            if self.name not in f:
                return None
            return _read_domain_settings(f[self.name], force_backend=force_backend)

    def write_noise_model_identity(self, identity: dict) -> None:
        """Record the semantic noise-model identity beside the domain settings.

        Plain scalar attrs (instrument component class, WDM PSD method, delay
        table / modulation digests, data epoch). A resume compares them so a
        store sampled under one likelihood is never silently continued under
        another whose arrays merely have the same shapes. Idempotent.
        """
        with self.open("a") as f:
            grp = f[self.name]
            if "noise_model_identity" in grp:
                del grp["noise_model_identity"]
            sub = grp.create_group("noise_model_identity")
            for key, value in identity.items():
                sub.attrs[key] = value

    def read_noise_model_identity(self) -> Optional[dict]:
        """Read back the stored noise-model identity (``None`` if absent)."""
        with self.open() as f:
            if self.name not in f or "noise_model_identity" not in f[self.name]:
                return None
            sub = f[self.name]["noise_model_identity"]
            out = {}
            for key, value in sub.attrs.items():
                if isinstance(value, bytes):
                    value = value.decode()
                elif isinstance(value, np.bool_):
                    value = bool(value)
                elif isinstance(value, np.floating):
                    value = float(value)
                elif isinstance(value, np.integer):
                    value = int(value)
                elif isinstance(value, np.str_):
                    value = str(value)
                out[key] = value
            return out

    @property
    def reset_kwargs(self):
        """Get reset_kwargs including sub-backend kwargs from h5 file."""
        base_kwargs = super().reset_kwargs
        if self.sub_backend is not None:
            # First, determine which sub-backend groups exist in the file.
            existing_keys = set()
            try:
                with self.open() as f:
                    if self.name in f and "sub_backend" in f[self.name]:
                        existing_keys = set(f[self.name]["sub_backend"].keys())
            except OSError:
                # No file yet (fresh run) -- nothing on disk to read back.
                pass
            # Now read reset_kwargs from each existing sub-backend
            # (outside the parent file handle to avoid nested opens).
            for key, sub_backend_tmp in self.sub_backend.items():
                if key in existing_keys:
                    try:
                        base_kwargs.update(sub_backend_tmp.reset_kwargs)
                    except (KeyError, OSError) as e:
                        logger.warning(
                            "could not read reset_kwargs from sub-backend %r: %s",
                            key,
                            e,
                        )
        return base_kwargs

    def reset(self, *args, **kwargs):
        """Wipe the file and re-initialize all sub-backends.

        Sub-backend kwargs are saved before ``super().reset()`` is called,
        because the parent ``reset`` deletes the entire HDF5 group; the
        saved kwargs (merged with any new caller kwargs) are then used to
        reset each sub-backend.
        """
        # Per-branch sub-backend reset kwargs (``{branch: {...}}``). Routes
        # branch-specific sizing (e.g. GB and VGB each carry their OWN
        # ``num_bands`` / ``band_edges``) to the matching sub-backend, since a
        # flat kwargs merge would let two GB-style branches clobber each
        # other's band structure. Popped here so eryn's ``super().reset()``
        # never sees it.
        sub_reset_kwargs = kwargs.pop("sub_reset_kwargs", None) or {}

        # Store sub-backend kwargs before super().reset() deletes the HDF5 group.
        # super().reset() calls `del f[self.name]` which wipes everything,
        # including the sub_backend group. We need to preserve the kwargs
        # so sub-backends can be re-created.
        sub_backend_saved_kwargs = {}
        if self.sub_backend is not None:
            # First, determine which sub-backend groups exist in the file.
            existing_keys = set()
            try:
                with self.open() as f:
                    if self.name in f and "sub_backend" in f[self.name]:
                        existing_keys = set(f[self.name]["sub_backend"].keys())
            except OSError:
                # No file yet (fresh run) -- nothing on disk to preserve.
                pass
            # Now read reset_kwargs from each existing sub-backend
            # (outside the parent file handle to avoid nested opens).
            for key, sub_backend_tmp in self.sub_backend.items():
                if key in existing_keys:
                    try:
                        sub_backend_saved_kwargs[key] = sub_backend_tmp.reset_kwargs
                    except (KeyError, OSError) as e:
                        logger.warning(
                            "could not preserve reset_kwargs from sub-backend %r "
                            "across reset: %s",
                            key,
                            e,
                        )

        # regular reset — this deletes and recreates f[self.name]
        super().reset(*args, **kwargs)

        if self.sub_backend is not None:
            with self.open("a") as f:
                g = f[self.name]
                if "sub_backend" not in g:
                    g.create_group("sub_backend")

            for key, sub_backend_tmp in self.sub_backend.items():
                # Use saved kwargs if available, otherwise fall back to
                # kwargs passed directly to this reset call.
                sub_backend_kwargs = sub_backend_saved_kwargs.get(key, {})
                # Per-branch overrides win (branch-specific band structure);
                # they take precedence over both the saved and the shared
                # kwargs so two GB-style branches don't share one band grid.
                for kw_key, kw_val in sub_reset_kwargs.get(key, {}).items():
                    sub_backend_kwargs[kw_key] = kw_val
                # Merge in any kwargs passed by the caller (e.g. num_mbhs,
                # num_bands, band_edges). Don't use hasattr() to check for
                # reset_kwargs — it's a @property that reads from the HDF5
                # file which was just wiped by super().reset().
                for kw_key, kw_val in kwargs.items():
                    if kw_key not in sub_backend_kwargs:
                        sub_backend_kwargs[kw_key] = kw_val
                # nwalkers is the sub-backends' first positional argument; a
                # per-branch value in the routed kwargs replaces it there.
                sub_args = list(args)
                if "nwalkers" in sub_backend_kwargs and len(sub_args) >= 1:
                    sub_args[0] = sub_backend_kwargs.pop("nwalkers")
                sub_backend_tmp.reset(*sub_args, **sub_backend_kwargs)

        with self.open("a") as f:
            f[self.name].attrs["has_recipe"] = False
            # storage-format discriminator: 2 = cold-chain main backend with
            # per-branch tempered sub-backends (the cold-chain storage
            # rework). Files without the attr predate the rework and cannot
            # be resumed by this code (clean break).
            f[self.name].attrs["gf_format_version"] = GF_FORMAT_VERSION

    def grow(self, ngrow, *args):
        """Grow the main backend and every sub-backend by ``ngrow`` rows."""
        super().grow(ngrow, *args)

        # open the file in append mode
        if self.sub_backend is not None:
            with self.open("a") as f:
                # resize all the arrays accordingly
                g = f[self.name]
                ntot = g.attrs["iteration"] + ngrow

                for sub_backend_tmp in self.sub_backend.values():
                    if sub_backend_tmp is None:
                        continue
                    sub_backend_tmp.grow(ngrow, *args)

    def save_step_main(self, state, *args, **kwargs):
        """Persist a single sample to the main backend and every sub-backend."""
        super().save_step(state, *args, **kwargs)

        # open for appending in with statement
        with self.open("a") as f:
            g = f[self.name]
            # get the iteration left off on
            # minus one because it was updated in the super function
            iteration = g.attrs["iteration"] - 1

            # Search LEGS (user design 2026-09-30): the row's leg-ender name,
            # the resume position, plus the stage's ordered move list, once.
            # Both written HERE, on the saver -- the single writer -- never
            # from the head (the recipe-group tears of 09-28/29).
            _after = getattr(state, "gf_saved_after", None)
            if _after:
                write_saved_after(g, iteration + 1, _after)
                _order = getattr(state, "gf_move_order", None)
                _stage = getattr(state, "gf_stage_name", None)
                if _order and _stage:
                    stamp_move_order(g, _stage, _order)

            if self.sub_backend is not None:
                # resize all the arrays accordingly

                sub_group = g["sub_backend"]
                for sub_state in self.sub_backend.values():
                    if sub_state is None:
                        continue
                    sub_state.save_step(state, *args, **kwargs)

    def save_step(self, *args, **kwargs):
        """Save a step to the backend

        Args:
            state (GFState): The :class:`GFState` of the ensemble.
            accepted (ndarray): An array of boolean flags indicating whether
                or not the proposal for each walker was accepted.
            rj_accepted (ndarray, optional): An array of the number of accepted steps
                for the reversible jump proposal for each walker.
                If :code:`self.rj` is True, then rj_accepted must be an array with
                :code:`rj_accepted.shape == accepted.shape`. If :code:`self.rj`
                is False, then rj_accepted must be None, which is the default.
            swaps_accepted (ndarray, optional): 1D array with number of swaps accepted
                for the in-model step. (default: ``None``)
            moves_accepted_fraction (dict, optional): Dict of acceptance fraction arrays for all of the
                moves in the sampler. This dict must have the same keys as ``self.move_keys``.
                (default: ``None``)

        """

        # Timed either way: the sync branch is the full gzip+h5 write on the
        # sampler process (the cost a dedicated saver rank would remove);
        # the handoff branch is the blocking pickled send (the cost that
        # REMAINS on the sampler under mpiexec >= 3) -- both numbers are
        # needed to judge the trade.
        t0 = time.perf_counter()
        if not self._has_saver_rank():
            self.save_step_main(*args, **kwargs)
            mode = "sync write"
        else:
            state = args[0]
            self.comm.send({"save_args": args, "save_kwargs": kwargs}, dest=self.save_plot_rank)
            mode = f"handoff to rank {self.save_plot_rank}"
        try:
            mb = os.path.getsize(self.filename) / 1e6
        except OSError:
            mb = float("nan")
        logger.info("[SAVE] save_step %.2f s (%s; h5 now %.1f MB)",
                    time.perf_counter() - t0, mode, mb)
        # Mid-iteration checkpoint bookkeeping: a real save landed (sync
        # write, or handed to the saver rank), so checkpoints written from
        # here on belong to the NEXT iteration; earlier ones go stale.
        midit_checkpoint.note_saved()

    def _has_saver_rank(self) -> bool:
        """Whether this process hands its store writes to a dedicated saver rank.

        THE ONE TEST for both the rows (:meth:`save_step`) and the routed
        store writes (:meth:`_write_on_saver`), so the two always share one
        writer. A communicator of 3+ ranks: below that the saver is aliased
        to the head (``communication.ranks.resolve_roles``), and the size-2
        dedicated-saver fallback still saves its rows synchronously here.
        The saver's own backend is built without ``comm``, so a write it
        runs for the head lands directly.
        """
        return self.comm is not None and self.comm.Get_size() >= 3

    def _write_on_saver(self, method, *args, **kwargs):
        """Run the store write ``method(*args, **kwargs)`` on the saver rank.

        Returns ``(True, result)`` once the saver has ACKNOWLEDGED it, or
        ``(False, None)`` when there is no saver rank and the caller must
        write directly (single process, ``fit.sample()``, size < 3, the
        saver's own backend, offline tools).

        The request travels on the row channel, so it lands after every row
        already handed off and before the next one. Waiting for the
        acknowledgement is what keeps a read straight after the write
        correct (the next stage's ``setup_run`` reads
        ``stage_start_iteration`` and ``iteration`` right after the boundary
        stamp); it costs one wait for the saver's queue at each stage
        boundary. A write that failed on the saver raises here.
        """
        if not self._has_saver_rank():
            return False, None
        t0 = time.perf_counter()
        self.comm.send({STORE_WRITE_KEY: method, "args": args, "kwargs": kwargs},
                       dest=self.save_plot_rank)
        ack = self.comm.recv(source=self.save_plot_rank, tag=STORE_WRITE_ACK_TAG)
        if not (isinstance(ack, dict) and ack.get("ok")):
            err = ack.get("error") if isinstance(ack, dict) else repr(ack)
            raise RuntimeError(
                f"store write {method}(args={args!r}, kwargs={kwargs!r}) FAILED "
                f"on the saver rank {self.save_plot_rank}: {err} (the saver "
                "logged the traceback)")
        logger.info(
            "[STORE-WRITE] %s(args=%r, kwargs=%r) routed to saver rank %d, "
            "acknowledged in %.2f s", method, args, kwargs, self.save_plot_rank,
            time.perf_counter() - t0)
        return True, ack.get("result")

    def get_a_sample(self, it):
        """Access a sample in the chain

        Args:
            it (int): iteration of State to return.

        Returns:
            State: :class:`eryn.state.State` object containing the sample from the chain.

        Raises:
            AttributeError: Backend is not initialized.

        """
        if (not self.initialized) or self.iteration <= 0:
            raise AttributeError(
                "you must run the sampler with " "'store == True' before accessing the " "results"
            )

        tmp_state = super().get_a_sample(it)
        state = GFState(tmp_state, sub_state_bases=self.sub_state_bases, is_eryn_state_input=True)

        # open for appending in with statement
        if self.sub_backend is not None:
            # resize all the arrays accordingly
            sub_states = {}
            sub_state_bases = {}
            for key in self.branch_names:
                sub_backend_tmp = self.sub_backend.get(key, None)
                if sub_backend_tmp is None:
                    sub_states[key] = None
                    sub_state_bases[key] = None
                    continue

                sub_states[key] = sub_backend_tmp.get_a_sample(it)
                sub_state_bases[key] = type(sub_states[key])

        else:
            sub_states = None
            sub_state_bases = None

        state.sub_states = sub_states
        state.sub_state_bases = sub_state_bases
        return state

    @property
    def has_recipe(self):
        """Whether a :class:`Recipe` has been recorded in this file."""
        with self.open() as f:
            return f[self.name].attrs["has_recipe"]

    @property
    def recipe(self):
        """Reconstructed dict of the on-disk recipe steps in their original order."""
        assert self.has_recipe
        with self.open() as f:
            _recipe = {}
            order = []
            keys = []
            for key in f[self.name]["recipe"]:
                _recipe[key] = {
                    key: val for key, val in f[self.name]["recipe"][key].attrs.items()
                }
                order.append(_recipe[key]["order num"])
                keys.append(key)

            new_order = np.argsort(np.asarray(order))
            recipe = {}
            for i in new_order:
                recipe[keys[i]] = _recipe[keys[i]]

        return recipe

    def add_recipe(self, recipe):
        """Persist a :class:`Recipe` into this backend's HDF5 metadata.

        On a store that already carries a recipe the stored statuses are
        read back into ``recipe``. The stored steps must all be present in
        the new recipe, in their stored relative order; a recipe that drops
        or reorders a stored step cannot resume the store and is refused.
        NEW steps are accepted when every one of them comes AFTER the stage
        the store is in (its first incomplete stored step) -- user design
        2026-10-03: "gb search 4 will be called replica pe", inserted between
        gb_search_3 and full_pe of a store already in gb_search_3. Each new
        step gets its group (status False), the order numbers are rewritten
        to the new positions, and any incomplete step after the active one
        loses a stale ``start_iteration`` stamp (a rewind had left full_pe
        reading 55 on the 6mo store) so the handover stamps it afresh.
        """
        if self.has_recipe:
            with self.open("a") as f:
                recipe_group = f[self.name]["recipe"]
                new_names = [step["name"] for step in recipe.recipe]
                # the stored step names: the group's keys when it has them (an
                # h5py group does; the damaged-store test doubles do not, and
                # for those the recipe's own names are the only list)
                _keys = getattr(recipe_group, "keys", None)
                stored_names = (list(_keys()) if callable(_keys)
                                else [n for n in new_names if n in recipe_group])
                stored = {}
                for key in stored_names:
                    # ⚠ THE LINK EXISTING DOES NOT MEAN THE OBJECT READS.
                    # Opening the group reads its object header, and a
                    # header torn by a kill mid-attr-write fails here with
                    # HDF5's own wording wrapped in a KeyError:
                    #
                    #   KeyError: 'Unable to synchronously open object
                    #              (message not aligned)'
                    #
                    # which reads as "no such recipe step" and sends the
                    # next person looking for a recipe bug that does not
                    # exist. Measured 2026-09-29, 3mo job 666: every chain,
                    # log_like and sub_backend in that store read fine and
                    # the run reached "initial log likelihood" -- only this
                    # group was damaged. Attr writes (status,
                    # completed_iteration, start_iteration) rewrite the
                    # object header, so this group is rewritten far more
                    # often than its size suggests and an MPI abort landing
                    # there is exactly how it tears. Say so, and name the
                    # recovery point, rather than raising the raw KeyError.
                    try:
                        grp = recipe_group[key]
                        stored[key] = (bool(grp.attrs["status"]), int(grp.attrs["order num"]))
                    except (KeyError, OSError) as e:
                        _bak = self.filename[:-3] + "_running_backup_copy.h5"
                        raise RuntimeError(
                            f"the recipe step {key!r} is present as a link in "
                            f"{self.filename} but its object CANNOT BE READ "
                            f"({type(e).__name__}: {e}). This is HDF5 file "
                            f"DAMAGE, not a missing or renamed recipe step -- "
                            f"a job killed or MPI-aborted while an attr was "
                            f"being written to this group leaves its object "
                            f"header malformed. The rest of the store may be "
                            f"perfectly readable, as the chains were here. "
                            f"RECOVERY, in order of preference: (1) transplant "
                            f"just this group from {_bak} (keeps every stored "
                            f"iteration); (2) restore that backup wholesale "
                            f"(loses at most `backup_iter` saves). COPY THE "
                            f"PRIMARY ASIDE BEFORE EITHER -- never repair the "
                            f"only copy."
                        ) from e
                missing = [k for k in stored if k not in new_names]
                if missing:
                    raise ValueError(
                        f"the store {self.filename} carries recipe step(s) {missing} that this "
                        f"run's recipe {new_names} does not: a recipe that DROPS or renames a "
                        "stored step cannot resume the store. Restore the step name or start "
                        "a fresh backend.")
                stored_order = [k for k, _ in sorted(stored.items(), key=lambda kv: kv[1][1])]
                in_new = [n for n in new_names if n in stored]
                if stored_order != in_new:
                    raise ValueError(
                        f"the stored recipe steps are ordered {stored_order} but this run's "
                        f"recipe orders them {in_new}: reordering stored steps cannot resume "
                        "the store.")
                added = [n for n in new_names if n not in stored]
                if added:
                    # the stage the store is IN: its first incomplete stored step
                    # (every stored step complete -> the recipe had finished; new
                    # steps then extend it from the end)
                    _active = next((i for i, n in enumerate(new_names)
                                    if n in stored and not stored[n][0]), None)
                    bound = (_active if _active is not None
                             else max(new_names.index(n) for n in stored))
                    bad = [n for n in added if new_names.index(n) <= bound]
                    if bad:
                        raise ValueError(
                            f"recipe step(s) {bad} would be inserted at or before the stage "
                            f"the store is in ({new_names[bound]!r}); stages can only be ADDED "
                            "after the active stage. Reorder the recipe or start a fresh "
                            "backend.")
                    for n in added:
                        grp = recipe_group.create_group(n)
                        grp.attrs["status"] = False
                    cleared = []
                    for n in new_names[bound + 1:]:
                        grp = recipe_group[n]
                        if (not bool(grp.attrs.get("status", False))
                                and "start_iteration" in grp.attrs):
                            del grp.attrs["start_iteration"]
                            cleared.append(n)
                    logger.info(
                        "[RECIPE] the store's recipe is EXTENDED: step(s) %s added after the "
                        "active stage %r (status False; the run resumes in %r as before); "
                        "order renumbered to %s%s.", added, new_names[bound], new_names[bound],
                        new_names, (f"; stale start_iteration cleared on {cleared}"
                                    if cleared else ""))
                for i, recipe_step in enumerate(recipe.recipe):
                    grp = recipe_group[recipe_step["name"]]
                    recipe_step["status"] = bool(grp.attrs["status"])
                    _order = grp.attrs.get("order num")      # a just-added group has none yet
                    if _order is None or int(_order) != i + 1:
                        if not added:
                            raise ValueError(
                                f"recipe step {recipe_step['name']!r} is stored at order "
                                f"{_order} but sits at position {i + 1} in this run's recipe.")
                        grp.attrs["order num"] = i + 1

        else:
            _tmp = recipe.to_file()
            with self.open("a") as f:
                recipe_group = f[self.name].create_group("recipe")
                for i, (key, val) in enumerate(_tmp.items()):
                    recipe_step_group = recipe_group.create_group(key)
                    recipe_step_group.attrs["status"] = val
                    recipe_step_group.attrs["order num"] = i + 1

                f[self.name].attrs["has_recipe"] = True

    def completed_recipe_step(self, step_name, next_step_name=None):
        """Mark ``step_name`` as completed in the on-disk recipe metadata.

        Also stamps ``completed_iteration``: the stored iteration the step
        ended at. The status flag alone says WHETHER a stage finished, never
        WHERE -- and with the v9 three-stage GB search (``gb_search_1/2/3``)
        every per-iteration trace in the monitor now spans several stages
        whose caps, SNR floors, phase-max setting and F-stat peak floor all
        differ. Without the boundary those traces are unreadable: a leaf
        count that jumps is either a discovery or a stage lowering its
        floors, and nothing on the page could tell them apart.

        Written best-effort. A stamp is a diagnostic, and failing a run
        because one could not be written would be absurd; a missing stamp
        just means the monitor draws no boundary (every store written before
        this existed is in that state).

        With a saver rank this runs THERE (:meth:`_write_on_saver`), after the
        row it closes, and returns once it has landed: the head writing this
        group while the saver wrote a row is how the recipe group tore at
        the 2026-09-28 and 2026-10-04 stage transitions.
        """
        routed, _ = self._write_on_saver(
            "completed_recipe_step", step_name, next_step_name=next_step_name)
        if routed:
            return
        # ⚠ READ THE ITERATION FIRST. ``self.iteration`` OPENS THE FILE in
        # "r", and doing that from inside the ``with self.open("a")`` below
        # is a same-file reopen with incompatible flags. On this stack it
        # succeeds; on an older HDF5 it behaves like a lock, and eryn's
        # backend retry loop spins 100 x 1 s before raising -- which the
        # ``except`` would then swallow, costing ~100 s and 100 stderr lines
        # per stage boundary and producing no stamp at all.
        try:
            _it = int(self.iteration)
        except Exception as exc:  # noqa: BLE001 -- the stamp is diagnostic
            logger.debug("recipe step %s: iteration unreadable (%r)",
                         step_name, exc)
            _it = None
        with self.open("a") as f:
            recipe_group = f[self.name]["recipe"]
            recipe_step_group = recipe_group[step_name]
            recipe_step_group.attrs["status"] = True
            if _it is not None:
                recipe_step_group.attrs["completed_iteration"] = _it
                # The NEXT step starts where this one completed. Stamped in
                # the SAME open as the completion (one write event at the
                # boundary, not two) so a stage's start survives a resume:
                # the galfor ratchet's schedule is keyed on the stage-local
                # iteration and would otherwise nudge again on every
                # relaunch. Companion to ``stage_start_iteration`` /
                # ``stamp_stage_start``.
                #
                # THE STAMP IS RESET HERE (user ruling 2026-10-07: "keep
                # GF_PERSIST_STAGE_START, but it should reset at the end of a
                # stage"). Until then an existing stamp was kept, so a stale
                # start on the next step -- a rewound, migrated or hand-reset
                # store -- would have been restored on the next launch and the
                # stage's convergence clock would have read as already
                # elapsed. The boundary is the one place the start is known
                # exactly, and a plain resume never reaches this line, so the
                # origin a resume restores is still the one the boundary wrote.
                if next_step_name is not None and next_step_name in recipe_group:
                    _nxt = recipe_group[next_step_name]
                    _old = _nxt.attrs.get("start_iteration")
                    if _old is not None and int(_old) != _it:
                        logger.info(
                            "recipe step %s: start_iteration %d -> %d, reset at "
                            "the end of %s.", next_step_name, int(_old), _it,
                            step_name)
                    _nxt.attrs["start_iteration"] = _it

    # ---- search legs: the row's leg-ender name + the stage's move order ----
    def saved_after(self, it):
        """The leg-ender name row ``it`` was saved after, or ``None``.

        ``None`` for a store written before legs existed, a row saved by a
        stage without legs, or a row outside the dataset.
        """
        return self.saved_after_history(int(it), int(it) + 1)[0]

    def saved_after_history(self, start, stop):
        """``saved_after`` for rows ``start..stop-1`` in one file open."""
        start, stop = int(start), int(stop)
        out = [None] * max(stop - start, 0)
        if stop <= start:
            return out
        try:
            with self.open("r") as f:
                ds = f[self.name].get(SAVED_AFTER_DS)
                if ds is None:
                    return out
                n = int(ds.shape[0])
                for k, it in enumerate(range(start, stop)):
                    if 0 <= it < n:
                        v = ds[it]
                        if isinstance(v, bytes):
                            v = v.decode("utf-8", "replace")
                        out[k] = str(v) or None
        except Exception as exc:  # noqa: BLE001 -- a resume hint, never fatal
            logger.debug("saved_after rows %d..%d unreadable (%r)", start, stop, exc)
        return out

    def stage_move_order(self, step_name):
        """The ordered move-name list stamped for ``step_name``, or ``None``."""
        import json

        try:
            with self.open("r") as f:
                grp = f[self.name].get("recipe")
                if grp is None or step_name not in grp:
                    return None
                raw = grp[step_name].attrs.get("move_order")
                if raw is None:
                    return None
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8", "replace")
                return [str(x) for x in json.loads(str(raw))]
        except Exception as exc:  # noqa: BLE001
            logger.debug("recipe step %s: move_order unreadable (%r)", step_name, exc)
            return None

    def stage_start_iteration(self, step_name):
        """The stored iteration a step STARTED at, or ``None``.

        The companion to ``completed_iteration``'s "where did it end".
        Written by :meth:`stamp_stage_start` and read on resume so a stage's
        own convergence clock survives a restart -- see
        ``RJRecipeStep.setup_run`` and ``GF_PERSIST_STAGE_START``.

        Returns ``None`` for a store written before this existed, for a step
        that has not started, or on any read error: the caller then falls
        back to the live iteration, which is the pre-2026-09-26 behaviour.
        """
        try:
            with self.open("r") as f:
                grp = f[self.name].get("recipe")
                if grp is None or step_name not in grp:
                    return None
                val = grp[step_name].attrs.get("start_iteration")
                return None if val is None else int(val)
        except Exception as exc:  # noqa: BLE001 -- diagnostic, never fatal
            logger.debug("recipe step %s: start_iteration unreadable (%r)",
                         step_name, exc)
            return None

    def stage_flag(self, step_name, key):
        """A per-stage attribute stamped by :meth:`stamp_stage_flag`, or ``None``.

        The galfor ratchet's data-driven stop lives here
        (``galfor_ratchet_done``): the decision that the last release gained
        too little is made once, in process, and a relaunch must not ratchet
        again because it forgot (user ruling 2026-10-02: no gate issues).
        ``None`` for an old store, an unknown step or any read error.
        """
        try:
            with self.open("r") as f:
                grp = f[self.name].get("recipe")
                if grp is None or step_name not in grp:
                    return None
                val = grp[step_name].attrs.get(str(key))
                return None if val is None else val
        except Exception as exc:  # noqa: BLE001 -- diagnostic, never fatal
            logger.debug("recipe step %s: flag %s unreadable (%r)", step_name, key, exc)
            return None

    def stamp_stage_flag(self, step_name, key, value) -> bool:
        """Write a per-stage attribute (best effort; True when it landed).

        Runs on the saver rank when there is one (:meth:`_write_on_saver`).
        """
        routed, landed = self._write_on_saver("stamp_stage_flag", step_name, key, value)
        if routed:
            return bool(landed)
        try:
            with self.open("a") as f:
                grp = f[self.name].get("recipe")
                if grp is None or step_name not in grp:
                    return False
                grp[step_name].attrs[str(key)] = value
                return True
        except Exception as exc:  # noqa: BLE001 -- diagnostic, never fatal
            logger.debug("recipe step %s: could not stamp %s (%r)", step_name, key, exc)
            return False

    def stamp_stage_start(self, step_name, iteration):
        """Record the stored iteration ``step_name`` started at.

        Best-effort and idempotent-by-caller: :meth:`stage_start_iteration`
        is consulted first, so an existing stamp is never overwritten and a
        resume keeps the ORIGINAL start rather than the restart's.

        Runs on the saver rank when there is one (:meth:`_write_on_saver`).
        """
        routed, _ = self._write_on_saver("stamp_stage_start", step_name, iteration)
        if routed:
            return
        try:
            with self.open("a") as f:
                grp = f[self.name].get("recipe")
                if grp is None or step_name not in grp:
                    return
                grp[step_name].attrs["start_iteration"] = int(iteration)
        except Exception as exc:  # noqa: BLE001 -- diagnostic, never fatal
            logger.debug("recipe step %s: could not stamp start_iteration "
                         "(%r)", step_name, exc)


def _uniform_value(arr, dtype):
    """The one value every element of ``arr`` holds, as a ``dtype`` scalar.

    ``None`` when the elements differ (or ``arr`` is empty): an HDF5 fill
    value is a single scalar. NaN counts as one value.
    """
    a = np.asarray(arr)
    if a.size == 0:
        return None
    first = a.reshape(-1)[0]
    if a.dtype.kind in "fc" and np.isnan(first):
        same = bool(np.all(np.isnan(a)))
    else:
        same = bool(np.all(a == first))
    if not same:
        return None
    return np.asarray(first).astype(dtype)[()]


class ModuleSubBackend(eryn_HDFBackend):
    """Generic per-branch sub-backend writing under ``sub_backend/<branch>``.

    "The sub-state IS the schema": this backend has no dataset table of its
    own. ``reset`` derives every dataset from a zeroed template built by
    ``state_class.make_template`` (statics written once, everything else a
    growable per-iteration dataset shaped ``(0, *arr.shape)``); ``save_step``
    writes ``state.sub_states[self.sub_name].storage_arrays()`` by name
    (creating the dataset of any array the store lacks, earlier rows at the
    array's fresh-state value, or at the sub-state's
    ``predating_fill_values`` entry where it names one) and then zeroes the
    sub-state's delta
    counters; ``get_a_sample`` rebuilds a
    sub-state via ``state_class.from_stored``. Subclasses set
    :attr:`state_class`; :class:`GFHDFBackend` stamps :attr:`sub_name` with
    the branch name at construction.
    """

    state_class = ModuleSubState
    sub_name: str = None

    def _group(self, f):
        """This branch's HDF5 group inside an open file handle."""
        return f[self.name]["sub_backend"][self.sub_name]

    def reset(self, nwalkers, *args, ntemps=1, **kwargs):
        """Create this branch's datasets from a zeroed template sub-state."""
        template = self.state_class.make_template(nwalkers, ntemps, **kwargs)
        # Arrays named in legacy_dtype_names keep the backend float dtype
        # (continuity with pre-rework files); everything else stores its
        # in-memory dtype (bool inds, int counters, ...).
        legacy_dtype = set(getattr(template, "legacy_dtype_names", ()))

        with self.open("a") as f:
            grp = f[self.name]["sub_backend"].create_group(self.sub_name)

            for name, value in template.storage_attrs().items():
                grp.attrs[name] = value

            for name, arr in template.static_arrays().items():
                dtype = self.dtype if name in legacy_dtype else np.asarray(arr).dtype
                grp.create_dataset(
                    name,
                    data=arr,
                    dtype=dtype,
                    compression=self.compression,
                    compression_opts=self.compression_opts,
                )

            for name, arr in template.storage_arrays().items():
                arr = np.asarray(arr)
                dtype = self.dtype if name in legacy_dtype else arr.dtype
                grp.create_dataset(
                    name,
                    (0,) + arr.shape,
                    maxshape=(None,) + arr.shape,
                    dtype=dtype,
                    compression=self.compression,
                    compression_opts=self.compression_opts,
                )

    @property
    def reset_kwargs(self):
        """Dimension kwargs read back from this branch's group on disk."""
        out = {}
        with self.open() as f:
            grp = self._group(f)
            # Attrs/statics added after a file was written (e.g. the
            # 2026-08-15 ``cap_edges`` / ``num_cap_cells`` leaf-cap grid) are
            # simply absent on older stores: skip them so the sub-state's own
            # defaults apply, rather than failing the resume.
            for name in self.state_class.dim_attr_names:
                if name in grp.attrs:
                    out[name] = grp.attrs[name]
            for name in self.state_class.static_names:
                if name in grp:
                    out[name] = grp[name][:]
        return out

    def grow(self, ngrow, *args):
        """Grow every non-static dataset by ``ngrow`` rows."""
        with self.open("a") as f:
            g = f[self.name]
            grp = g["sub_backend"][self.sub_name]
            ntot = g.attrs["iteration"] + ngrow
            static = set(self.state_class.static_names)
            for key in grp:
                if key in static:
                    continue
                grp[key].resize(ntot, axis=0)

    def save_step(self, state, *args, **kwargs):
        """Persist this iteration's sub-state arrays, then zero delta counters."""
        with self.open("a") as f:
            g = f[self.name]
            # minus one because the parent save_step already advanced it
            iteration = g.attrs["iteration"] - 1
            grp = g["sub_backend"][self.sub_name]

            arrays = state.sub_states[self.sub_name].storage_arrays()
            # dead-leaf masking on write only, using eryn's own
            # store_missing_leaves sentinel (the live sub-state keeps its
            # stale coords, exactly like the main chain)
            if "chain" in arrays and "inds" in arrays:
                chain = np.array(arrays["chain"], copy=True)
                chain[~np.asarray(arrays["inds"], dtype=bool)] = self.store_missing_leaves
                arrays = {**arrays, "chain": chain}

            # An array with no dataset (a store created before the array
            # existed, or created from a template that did not allocate it)
            # gets its dataset NOW. Until 2026-10-10 it was skipped, silently,
            # at every save: the 9mo store (jobs 751-753) never held the
            # per-(walker, band) RJ shutoff valve, so every resume restarted
            # it from scratch.
            missing = [name for name in arrays if name not in grp]
            if missing:
                self._create_missing_datasets(
                    grp, state.sub_states[self.sub_name], arrays, missing, iteration
                )

            for name, arr in arrays.items():
                grp[name][iteration] = arr

        state.sub_states[self.sub_name].reset_delta_counters()

    def _create_missing_datasets(self, grp, sub_state, arrays, names, iteration):
        """Create per-iteration datasets for ``names`` inside ``grp``.

        Same layout as :meth:`reset` (step axis growable, same dtype rule and
        compression) and the same length as the group's existing
        per-iteration datasets, so :meth:`grow` and every row read stay
        aligned. The earlier rows read back as the FRESH-state value of the
        array (``-inf`` for a running max, ``-1`` for an unset step stamp,
        ...), taken from a template built like ``reset``'s, i.e. what the
        dataset would have started with -- never a fabricated 0, which as the
        valve's all-time cold-lnL max could never be beaten (the band lnL is
        ``-1/2 <r|r>``, never positive) and would shut every pair of a walker
        that loaded such a row. Zeros only where there is no single fresh
        value or no template can be built.

        A sub-state can name a different value for rows that PREDATE the
        array (``predating_fill_values``; GB: the valve's window-rule stamp
        reads 1, the rule those rows were learned under, where a fresh
        valve is stamped with the current rule). That value wins.
        """
        static = set(self.state_class.static_names)
        lengths = [
            int(grp[key].shape[0])
            for key in grp
            if key not in static and len(getattr(grp[key], "shape", ())) > 0
        ]
        nrows = max(lengths + [int(iteration) + 1])
        legacy_dtype = set(getattr(sub_state, "legacy_dtype_names", ()))
        predating = dict(getattr(sub_state, "predating_fill_values", None) or {})
        _need_fresh = [n for n in names if n not in predating]
        fresh = (self._fresh_storage_arrays(sub_state, _need_fresh)
                 if _need_fresh else {})

        for name in names:
            arr = np.asarray(arrays[name])
            dtype = self.dtype if name in legacy_dtype else arr.dtype
            fill = None
            if name in predating:
                fill = np.asarray(predating[name]).astype(dtype)[()]
            elif name in fresh and np.shape(fresh[name]) == arr.shape:
                fill = _uniform_value(fresh[name], dtype)
            if fill is not None and fill == 0:
                fill = None  # h5py's own default fill is already zero (NaN != 0)
            grp.create_dataset(
                name,
                (nrows,) + arr.shape,
                maxshape=(None,) + arr.shape,
                dtype=dtype,
                compression=self.compression,
                compression_opts=self.compression_opts,
                fillvalue=fill,
            )
            logger.info(
                "[STORE] created missing per-iteration dataset sub_backend/%s/%s "
                "(shape %s, %s; %d earlier rows %s)",
                self.sub_name,
                name,
                (nrows,) + arr.shape,
                np.dtype(dtype).name,
                int(iteration),
                "zero-filled"
                if fill is None
                else f"filled with the fresh-state value {fill.item()!r}",
            )

    def _fresh_storage_arrays(self, sub_state, names) -> dict:
        """``{name: fresh-state array}`` for ``names``, from a template.

        Built as :meth:`reset` builds its template: ``make_template`` on the
        live sub-state's ``reset_kwargs``. Tried first without the tempered
        dimensions, which skips allocating a whole zeroed ensemble when only
        band arrays are missing. Names no template allocates are left out
        (the caller zero-fills those) and named in a warning.
        """
        try:
            kw = dict(sub_state.reset_kwargs)
            nwalkers = kw.pop("nwalkers")
            ntemps = kw.pop("ntemps")
        except Exception as exc:  # noqa: BLE001 -- the fill is not worth a failed save
            logger.warning(
                "[STORE] no fresh-state template for sub_backend/%s (%s: %s); "
                "the new datasets' earlier rows are zero-filled.",
                self.sub_name, type(exc).__name__, exc)
            return {}
        best = {}
        for drop in (("nleaves_max", "ndim"), ()):
            try:
                template = type(sub_state).make_template(
                    nwalkers, ntemps, **{k: v for k, v in kw.items() if k not in drop}
                )
                out = template.storage_arrays()
            except Exception:  # noqa: BLE001 -- try the next form
                continue
            got = {name: out[name] for name in names if name in out}
            if len(got) > len(best):
                best = got
            if len(best) == len(names):
                return best
        logger.warning(
            "[STORE] the fresh-state template for sub_backend/%s does not "
            "allocate %s; their earlier rows are zero-filled.",
            self.sub_name, [name for name in names if name not in best])
        return best

    def get_value(self, name, thin=1, discard=0, slice_vals=None):
        """Read one of this branch's datasets (statics returned whole).

        Args:
            name (str): Name of value requested.
            thin (int, optional): Take only every ``thin`` steps from the
                chain. (default: ``1``)
            discard (int, optional): Discard the first ``discard`` steps in
                the chain as burn-in. (default: ``0``)
            slice_vals (indexing np.ndarray or slice, optional): If provided,
                slice the array directly from the HDF5 file with
                slice = ``slice_vals``; ``thin`` and ``discard`` are then
                ignored. (default: ``None``)
        """
        if not self.initialized:
            raise AttributeError(
                "You must run the sampler with " "'store == True' before accessing the " "results"
            )

        if slice_vals is None:
            slice_vals = slice(discard + thin - 1, self.iteration, thin)

        with self.open() as f:
            g = f[self.name]
            if g.attrs["iteration"] <= 0:
                raise AttributeError(
                    "You must run the sampler with "
                    "'store == True' before accessing the "
                    "results"
                )
            grp = g["sub_backend"][self.sub_name]
            if name not in grp:
                raise ValueError(f"No {name} in this backend.")
            if name in self.state_class.static_names:
                return grp[name][:]
            return grp[name][slice_vals]

    def get_a_sample(self, it):
        """Rebuild this branch's sub-state from iteration ``it``."""
        thin = self.iteration - it if it != self.iteration else 1
        discard = it + 1 - thin
        slice_vals = slice(discard + thin - 1, self.iteration, thin)

        with self.open() as f:
            grp = self._group(f)
            static = set(self.state_class.static_names)
            arrays = {key: grp[key][slice_vals] for key in grp if key not in static}
            # ``static`` may name datasets a pre-existing file never wrote
            # (cap_edges, 2026-08-15) -- from_stored fills those in.
            statics = {key: grp[key][:] for key in static if key in grp}
            attrs = dict(grp.attrs)

        return self.state_class.from_stored(arrays, statics=statics, attrs=attrs)


class PerLeafLadderHDFBackend(ModuleSubBackend):
    """Sub-backend persisting a per-leaf temperature ladder (``betas_all``)."""

    def get_betas_all(self, **kwargs):
        """The stored ``(nsteps, nleaves, ntemps)`` ladder history.

        Accepts the standard ``thin`` / ``discard`` / ``slice_vals`` kwargs
        of :meth:`get_value`.
        """
        return self.get_value("betas_all", **kwargs)


class GBHDFBackend(ModuleSubBackend):
    """Sub-backend that persists per-band GB sampler counters.

    Serves any GB-style banded branch; :class:`GFHDFBackend` stamps
    :attr:`sub_name` with the branch name at construction (``"gb"``,
    ``"vgb"``, ...), which keys this sub-backend's HDF group.
    """

    state_class = GBState
    sub_name: str = "gb"

    @property
    def num_bands(self):
        """Get num_bands from h5 file."""
        with self.open() as f:
            return f[self.name]["sub_backend"][self.sub_name].attrs["num_bands"]

    @property
    def band_edges(self):
        """Get band_edges from h5 file."""
        with self.open() as f:
            return f[self.name]["sub_backend"][self.sub_name]["band_edges"][:]

    def get_value(self, name, thin=1, discard=0, slice_vals=None):
        """Returns a requested value to user.

        This function helps to streamline the backend for both
        basic and hdf backend.

        Args:
            name (str): Name of value requested.
            thin (int, optional): Take only every ``thin`` steps from the
                chain. (default: ``1``)
            discard (int, optional): Discard the first ``discard`` steps in
                the chain as burn-in. (default: ``0``)
            slice_vals (indexing np.ndarray or slice, optional): If provided, slice the array directly
                from the HDF5 file with slice = ``slice_vals``. ``thin`` and ``discard`` will be
                ignored if slice_vals is not ``None``. This is particularly useful if files are
                very large and the user only wants a small subset of the overall array.
                (default: ``None``)

        Returns:
            dict or np.ndarray: Values requested.

        """
        # check if initialized
        if not self.initialized:
            raise AttributeError(
                "You must run the sampler with " "'store == True' before accessing the " "results"
            )

        if name != "band_info":
            # anything else (chain, inds, ...) is a plain per-dataset read
            return super().get_value(
                name, thin=thin, discard=discard, slice_vals=slice_vals
            )

        if slice_vals is None:
            slice_vals = slice(discard + thin - 1, self.iteration, thin)

        successful = False
        num_try = 0

        while not successful and num_try < 100:
            try:
                # open the file wrapped in a "with" statement
                with self.open() as f:
                    # get the group that everything is stored in
                    g = f[self.name]
                    iteration = g.attrs["iteration"]
                    if iteration <= 0:
                        raise AttributeError(
                            "You must run the sampler with "
                            "'store == True' before accessing the "
                            "results"
                        )

                    gb_group = g["sub_backend"][self.sub_name]
                    # band_* datasets only: the group also holds the tempered
                    # chain/inds, which are not part of the band_info dict.
                    v_all = {
                        key: gb_group[key][slice_vals]
                        for key in gb_group
                        if key.startswith("band_") and key != "band_edges"
                    }
                    v_all["band_edges"] = gb_group["band_edges"][:]
                    successful = True
            except OSError:
                num_try += 1
                logger.warning(f"Tried to open h5 file {num_try} times.")
                time.sleep(20.0)
        if not successful:
            raise OSError("Unable to open h5 file after many tries.")

        return v_all

    def get_band_info(self, **kwargs):
        """Get the stored chain of MCMC samples

        Args:
            thin (int, optional): Take only every ``thin`` steps from the
                chain. (default: ``1``)
            discard (int, optional): Discard the first ``discard`` steps in
                the chain as burn-in. (default: ``0``)
            slice_vals (indexing np.ndarray or slice, optional): This is only available in :class:`eryn.backends.hdfbackend`.
                If provided, slice the array directly from the HDF5 file with slice = ``slice_vals``.
                ``thin`` and ``discard`` will be ignored if slice_vals is not ``None``.
                This is particularly useful if files are very large and the user only wants a
                small subset of the overall array. (default: ``None``)

        Returns:
            dict: MCMC samples
                The dictionary contains np.ndarrays of samples
                across the branches.

        """
        tmp = self.get_value("band_info", **kwargs)
        tmp["initialized"] = True
        return tmp

class MBHHDFBackend(PerLeafLadderHDFBackend):
    """Sub-backend that persists the per-leaf MBH temperature ladder."""

    state_class = MBHState
    sub_name: str = "mbh"

    @property
    def num_mbhs(self):
        """Get num_mbhs from the mbh sub-backend group in the h5 file."""
        with self.open() as f:
            return self._group(f).attrs["num_mbhs"]


class EMRIHDFBackend(PerLeafLadderHDFBackend):
    """Sub-backend that persists the per-leaf EMRI temperature ladder."""

    state_class = EMRIState
    sub_name: str = "emri"

    @property
    def num_emris(self):
        """Get num_emris from the emri sub-backend group in the h5 file."""
        with self.open() as f:
            return self._group(f).attrs["num_emris"]


class SOBBHHDFBackend(PerLeafLadderHDFBackend):
    """Sub-backend that persists the per-leaf SOBBH temperature ladder.

    Mirrors :class:`EMRIHDFBackend` — one ``betas_all`` row per leaf, sliced
    out at save time and restored from disk for warm-starts.
    """

    state_class = SOBBHState
    sub_name: str = "sobbh"

    @property
    def num_sobbhs(self):
        """Get num_sobbhs from the sobbh sub-backend group in the h5 file."""
        with self.open() as f:
            return self._group(f).attrs["num_sobbhs"]


