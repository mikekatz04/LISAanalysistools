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


def save_to_backend_asynchronously_and_plot(
    gb_reader,
    comm,
    main_rank,
    plot_container=None,
    plot_iter=100,
    backup_iter=5,
    coalesce_threshold=3,
):
    """Async writer loop: receive states from ``main_rank`` and persist to HDF5.

    Runs on the dedicated results rank (np >= 3). Receives
    ``{save_args, save_kwargs}`` payloads, calls
    :meth:`GFHDFBackend.save_step_main`, periodically copies the file to a
    running backup, regenerates the diagnostic plots, and exits when sent
    ``{"finish_run": True}``.

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
            ``*_running_backup_copy.h5``.
        coalesce_threshold: Queue depth above which intermediate states are
            dropped in favour of the newest.
    """
    logger.info("results rank: starting async save/plot loop")
    if plot_container is not None:
        plot_container.backend = gb_reader
    run = True
    i = 0
    total_dropped = 0
    while run:
        payloads = [comm.recv(source=main_rank)]
        # Drain the backlog without blocking: saves must never queue behind
        # this loop's own work.
        while comm.iprobe(source=main_rank):
            payloads.append(comm.recv(source=main_rank))

        finish = any(
            isinstance(p, dict) and p.get("finish_run", False) for p in payloads
        )
        states = [p for p in payloads if isinstance(p, dict) and "save_args" in p]
        if len(states) > coalesce_threshold:
            dropped = len(states) - 1
            total_dropped += dropped
            logger.warning(
                "results rank fell behind: dropping %d intermediate save "
                "state(s), keeping the newest (%d dropped so far this run).",
                dropped,
                total_dropped,
            )
            states = states[-1:]

        for save_dict in states:
            st = time.perf_counter()
            gb_reader.save_step_main(
                *save_dict["save_args"], **save_dict["save_kwargs"]
            )
            logger.debug("save step took %.3f s", time.perf_counter() - st)
            i += 1
            if (i % backup_iter) == 0:
                _atomic_backup_copy(gb_reader.filename)

        if finish:
            run = False
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
        save_plot_rank: Rank to forward save jobs to.
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
        if self.comm is None or self.comm.Get_size() < 3:
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
        """Persist a :class:`Recipe` into this backend's HDF5 metadata."""
        if self.has_recipe:
            with self.open() as f:
                recipe_group = f[self.name]["recipe"]
                for i, recipe_step in enumerate(recipe.recipe):
                    key = recipe_step["name"]
                    assert key in recipe_group
                    recipe_step_group = recipe_group[key]
                    recipe.recipe[i]["status"] = recipe_step_group.attrs["status"]
                    order_i_in_file = recipe_step_group.attrs["order num"]
                    assert order_i_in_file == i + 1

        else:
            _tmp = recipe.to_file()
            with self.open("a") as f:
                recipe_group = f[self.name].create_group("recipe")
                for i, (key, val) in enumerate(_tmp.items()):
                    recipe_step_group = recipe_group.create_group(key)
                    recipe_step_group.attrs["status"] = val
                    recipe_step_group.attrs["order num"] = i + 1

                f[self.name].attrs["has_recipe"] = True

    def completed_recipe_step(self, step_name):
        """Mark ``step_name`` as completed in the on-disk recipe metadata."""
        with self.open("a") as f:
            recipe_group = f[self.name]["recipe"]
            recipe_step_group = recipe_group[step_name]
            recipe_step_group.attrs["status"] = True


class ModuleSubBackend(eryn_HDFBackend):
    """Generic per-branch sub-backend writing under ``sub_backend/<branch>``.

    "The sub-state IS the schema": this backend has no dataset table of its
    own. ``reset`` derives every dataset from a zeroed template built by
    ``state_class.make_template`` (statics written once, everything else a
    growable per-iteration dataset shaped ``(0, *arr.shape)``); ``save_step``
    writes ``state.sub_states[self.sub_name].storage_arrays()`` by name and
    then zeroes the sub-state's delta counters; ``get_a_sample`` rebuilds a
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

            for name, arr in arrays.items():
                if name not in grp:
                    # e.g. arrays added on an HDF file created before they
                    # existed: skip rather than corrupt the resume.
                    continue
                grp[name][iteration] = arr

        state.sub_states[self.sub_name].reset_delta_counters()

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


