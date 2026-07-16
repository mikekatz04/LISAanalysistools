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
from .state import EMRIState, GBState, GFState, MBHState, SOBBHState

logger = getLogger(__name__)


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
                shutil.copy(
                    gb_reader.filename,
                    gb_reader.filename[:-3] + "_running_backup_copy.h5",
                )

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

        self.sub_state_bases = sub_state_bases
        self.recipe_added = False

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
            except Exception:
                pass
            # Now read reset_kwargs from each existing sub-backend
            # (outside the parent file handle to avoid nested opens).
            for key, sub_backend_tmp in self.sub_backend.items():
                if key in existing_keys:
                    try:
                        base_kwargs.update(sub_backend_tmp.reset_kwargs)
                    except Exception:
                        pass
        return base_kwargs

    def reset(self, *args, **kwargs):
        """Wipe the file and re-initialize all sub-backends.

        Sub-backend kwargs are saved before ``super().reset()`` is called,
        because the parent ``reset`` deletes the entire HDF5 group; the
        saved kwargs (merged with any new caller kwargs) are then used to
        reset each sub-backend.
        """
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
            except Exception:
                pass
            # Now read reset_kwargs from each existing sub-backend
            # (outside the parent file handle to avoid nested opens).
            for key, sub_backend_tmp in self.sub_backend.items():
                if key in existing_keys:
                    try:
                        sub_backend_saved_kwargs[key] = sub_backend_tmp.reset_kwargs
                    except Exception:
                        pass

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
                # Merge in any kwargs passed by the caller (e.g. num_mbhs,
                # num_bands, band_edges). Don't use hasattr() to check for
                # reset_kwargs — it's a @property that reads from the HDF5
                # file which was just wiped by super().reset().
                for kw_key, kw_val in kwargs.items():
                    if kw_key not in sub_backend_kwargs:
                        sub_backend_kwargs[kw_key] = kw_val
                sub_backend_tmp.reset(*args, **sub_backend_kwargs)

        with self.open("a") as f:
            f[self.name].attrs["has_recipe"] = False

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

        if self.comm is None or self.comm.Get_size() < 3:
            self.save_step_main(*args, **kwargs)

        else:
            state = args[0]
            self.comm.send({"save_args": args, "save_kwargs": kwargs}, dest=self.save_plot_rank)

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


class GBHDFBackend(eryn_HDFBackend):
    """Sub-backend that persists per-band GB sampler counters."""

    def reset(self, nwalkers, *args, ntemps=1, num_bands=None, band_edges=None, **kwargs):
        """Create the per-band datasets used to back :class:`GBState`."""
        if num_bands is None or band_edges is None:
            raise ValueError("Must provide num_bands and band_edges kwargs.")

        # open file in append mode
        with self.open("a") as f:
            g = f[self.name]["sub_backend"]

            band_info = g.create_group("gb")

            band_info.create_dataset(
                "band_edges",
                data=band_edges,
                dtype=self.dtype,
                compression=self.compression,
                compression_opts=self.compression_opts,
            )

            band_info.attrs["num_bands"] = len(band_edges) - 1

            band_info.create_dataset(
                "band_temps",
                (0, num_bands, ntemps),
                maxshape=(None, num_bands, ntemps),
                dtype=self.dtype,
                compression=self.compression,
                compression_opts=self.compression_opts,
            )

            band_info.create_dataset(
                "band_swaps_proposed",
                (0, num_bands, ntemps - 1),
                maxshape=(None, num_bands, ntemps - 1),
                dtype=self.dtype,
                compression=self.compression,
                compression_opts=self.compression_opts,
            )

            band_info.create_dataset(
                "band_swaps_accepted",
                (0, num_bands, ntemps - 1),
                maxshape=(None, num_bands, ntemps - 1),
                dtype=self.dtype,
                compression=self.compression,
                compression_opts=self.compression_opts,
            )

            band_info.create_dataset(
                "band_num_proposed",
                (0, num_bands, ntemps),
                maxshape=(None, num_bands, ntemps),
                dtype=self.dtype,
                compression=self.compression,
                compression_opts=self.compression_opts,
            )

            band_info.create_dataset(
                "band_num_accepted",
                (0, num_bands, ntemps),
                maxshape=(None, num_bands, ntemps),
                dtype=self.dtype,
                compression=self.compression,
                compression_opts=self.compression_opts,
            )

            band_info.create_dataset(
                "band_num_proposed_rj",
                (0, num_bands, ntemps),
                maxshape=(None, num_bands, ntemps),
                dtype=self.dtype,
                compression=self.compression,
                compression_opts=self.compression_opts,
            )

            band_info.create_dataset(
                "band_num_accepted_rj",
                (0, num_bands, ntemps),
                maxshape=(None, num_bands, ntemps),
                dtype=self.dtype,
                compression=self.compression,
                compression_opts=self.compression_opts,
            )

            band_info.create_dataset(
                "band_num_binaries",
                (0, ntemps, nwalkers, num_bands),
                maxshape=(None, ntemps, nwalkers, num_bands),
                dtype=self.dtype,
                compression=self.compression,
                compression_opts=self.compression_opts,
            )

            # Per-band progressive leaf-cap state (search mode; see
            # lisatools.globalfit.state.ensure_leaf_cap_fields).
            for cap_key in ("band_leaf_cap", "band_cap_iters", "band_best_ll"):
                band_info.create_dataset(
                    cap_key,
                    (0, num_bands),
                    maxshape=(None, num_bands),
                    dtype=self.dtype,
                    compression=self.compression,
                    compression_opts=self.compression_opts,
                )

    @property
    def num_bands(self):
        """Get num_bands from h5 file."""
        with self.open() as f:
            return f[self.name]["sub_backend"]["gb"].attrs["num_bands"]

    @property
    def band_edges(self):
        """Get band_edges from h5 file."""
        with self.open() as f:
            return f[self.name]["sub_backend"]["gb"]["band_edges"][:]

    @property
    def reset_kwargs(self):
        """Get reset_kwargs from h5 file."""
        return dict(num_bands=self.num_bands, band_edges=self.band_edges)

    def grow(self, ngrow, *args):
        """Grow every per-band dataset by ``ngrow`` rows."""
        # open the file in append mode
        with self.open("a") as f:
            g = f[self.name]
            band_info = g["sub_backend"]["gb"]
            # resize all the arrays accordingly
            ntot = g.attrs["iteration"] + ngrow
            for key in band_info:
                if key == "band_edges":
                    continue
                band_info[key].resize(ntot, axis=0)

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
            raise ValueError(f"No {name} in this backend.")

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

                    gb_group = g["sub_backend"]["gb"]
                    v_all = {
                        key: gb_group[key][slice_vals] for key in gb_group if key != "band_edges"
                    }
                    v_all["band_edges"] = gb_group["band_edges"][:]
                    successful = True
            except OSError:
                num_try += 1
                print(f"Tried to open h5 file {num_try} times.")
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

    def save_step(self, state, *args, **kwargs):
        """Persist the current ``GBState.band_info`` arrays for one iteration."""
        # open for appending in with statement
        with self.open("a") as f:
            g = f[self.name]
            # get the iteration left off on
            # minus one because it was updated in the super function
            iteration = g.attrs["iteration"] - 1

            gb_group = g["sub_backend"]["gb"]

            # make sure the backend has all the information needed to store everything
            for key in [
                "num_bands",
            ]:
                if not hasattr(self, key):
                    setattr(self, key, g.attrs[key])

            # branch-specific
            for name, dat in state.sub_states["gb"].band_info.items():
                if not isinstance(dat, np.ndarray) or name == "band_edges":
                    continue
                if name not in gb_group:
                    # e.g. the leaf-cap arrays on an HDF file created before
                    # they existed: skip rather than corrupt the resume.
                    continue
                gb_group[name][iteration] = dat

        # reset the counter for band info
        state.sub_states["gb"].reset_band_counters()

    def get_a_sample(self, it):
        """Access a sample in the chain

        Args:
            it (int): iteration of GFState to return.

        Returns:
            GFState: :class:`eryn.state.GFState` object containing the sample from the chain.

        Raises:
            AttributeError: Backend is not initialized.

        """

        thin = self.iteration - it if it != self.iteration else 1
        discard = it + 1 - thin

        band_info = self.get_band_info(discard=discard, thin=thin)
        sample = GBState(None, band_info=band_info)
        sample.band_info["initialized"] = True
        return sample


class MBHHDFBackend(eryn_HDFBackend):
    """Sub-backend that persists the per-leaf MBH temperature ladder."""

    def reset(self, nwalkers, *args, ntemps=1, num_mbhs: int = None, **kwargs):
        """Create a ``betas_all`` dataset of shape ``(*, num_mbhs, ntemps)``."""
        if num_mbhs is None:
            raise ValueError("Must provide num_mbhs kwarg.")

        # open file in append mode
        with self.open("a") as f:
            g = f[self.name]["sub_backend"]

            mbh_group = g.create_group("mbh")

            mbh_group.attrs["num_mbhs"] = num_mbhs

            mbh_group.create_dataset(
                "betas_all",
                (0, num_mbhs, ntemps),
                maxshape=(None, num_mbhs, ntemps),
                dtype=self.dtype,
                compression=self.compression,
                compression_opts=self.compression_opts,
            )

    @property
    def num_mbhs(self):
        """Get num_bands from h5 file."""
        with self.open() as f:
            mbh_group = f[self.name]["sub_backend"]["mbh"]
            return mbh_group.attrs["num_mbhs"]

    @property
    def reset_kwargs(self):
        """Get reset_kwargs from h5 file."""
        return dict(num_mbhs=self.num_mbhs)

    def grow(self, ngrow, *args):
        """Grow ``betas_all`` by ``ngrow`` rows."""
        # open the file in append mode
        with self.open("a") as f:
            g = f[self.name]
            mbh_group = f[self.name]["sub_backend"]["mbh"]
            # resize all the arrays accordingly
            ntot = g.attrs["iteration"] + ngrow
            mbh_group["betas_all"].resize(ntot, axis=0)

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

        if name != "betas_all":
            raise ValueError(f"No {name} in this backend.")

        if slice_vals is None:
            slice_vals = slice(discard + thin - 1, self.iteration, thin)

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

            mbh_group = g["sub_backend"]["mbh"]
            v_all = mbh_group["betas_all"][slice_vals]
        return v_all

    def get_betas_all(self, **kwargs):
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
        return self.get_value("betas_all", **kwargs)

    def save_step(self, state, *args, **kwargs):
        """Persist this iteration's ``betas_all`` from the current MBH sub-state."""
        # open for appending in with statement
        with self.open("a") as f:
            g = f[self.name]
            # get the iteration left off on
            # minus one because it was updated in the super function
            iteration = g.attrs["iteration"] - 1
            mbh_group = g["sub_backend"]["mbh"]
            mbh_group["betas_all"][iteration] = state.sub_states["mbh"].betas_all

    def get_a_sample(self, it):
        """Access a sample in the chain

        Args:
            it (int): iteration of GFState to return.

        Returns:
            GFState: :class:`eryn.state.GFState` object containing the sample from the chain.

        Raises:
            AttributeError: Backend is not initialized.

        """
        thin = self.iteration - it if it != self.iteration else 1
        discard = it + 1 - thin

        betas_all = self.get_betas_all(discard=discard, thin=thin)

        sample = MBHState(None, betas_all=betas_all)
        return sample


# TODO: @ alessandro, we can use the same for EMRIs and MBHs
# for now, but I assume we will want it separate in the end


class EMRIHDFBackend(eryn_HDFBackend):
    """Sub-backend that persists the per-leaf EMRI temperature ladder."""

    def reset(self, nwalkers, *args, ntemps=1, num_emris: int = None, **kwargs):
        """Create a ``betas_all`` dataset of shape ``(*, num_emris, ntemps)``."""
        if num_emris is None:
            raise ValueError("Must provide num_emris kwarg.")

        # open file in append mode
        with self.open("a") as f:
            g = f[self.name]["sub_backend"]

            emri_group = g.create_group("emri")

            emri_group.attrs["num_emris"] = num_emris

            emri_group.create_dataset(
                "betas_all",
                (0, num_emris, ntemps),
                maxshape=(None, num_emris, ntemps),
                dtype=self.dtype,
                compression=self.compression,
                compression_opts=self.compression_opts,
            )

    @property
    def num_emris(self):
        """Get num_bands from h5 file."""
        with self.open() as f:
            return f[self.name].attrs["num_emris"]

    @property
    def reset_kwargs(self):
        """Get reset_kwargs from h5 file."""
        return dict(num_emris=self.num_emris)

    def grow(self, ngrow, *args):
        """Grow ``betas_all`` by ``ngrow`` rows."""
        # open the file in append mode
        with self.open("a") as f:
            g = f[self.name]
            emri_group = f[self.name]["sub_backend"]["emri"]
            # resize all the arrays accordingly
            ntot = g.attrs["iteration"] + ngrow
            emri_group["betas_all"].resize(ntot, axis=0)

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

        if name != "betas_all":
            raise ValueError(f"No {name} in this backend.")

        if slice_vals is None:
            slice_vals = slice(discard + thin - 1, self.iteration, thin)

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

            emri_group = g["sub_backend"]["emri"]
            v_all = emri_group["betas_all"][slice_vals]
        return v_all

    def get_betas_all(self, **kwargs):
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
        return self.get_value("betas_all", **kwargs)

    def save_step(self, state, *args, **kwargs):
        """Persist this iteration's ``betas_all`` from the current EMRI sub-state."""
        # open for appending in with statement
        with self.open("a") as f:
            g = f[self.name]
            # get the iteration left off on
            # minus one because it was updated in the super function
            iteration = g.attrs["iteration"] - 1
            emri_group = g["sub_backend"]["emri"]
            emri_group["betas_all"][iteration] = state.sub_states["emri"].betas_all

    def get_a_sample(self, it):
        """Access a sample in the chain

        Args:
            it (int): iteration of GFState to return.

        Returns:
            GFState: :class:`eryn.state.GFState` object containing the sample from the chain.

        Raises:
            AttributeError: Backend is not initialized.

        """
        thin = self.iteration - it if it != self.iteration else 1
        discard = it + 1 - thin

        betas_all = self.get_betas_all(discard=discard, thin=thin)

        sample = EMRIState(None, betas_all=betas_all)
        return sample


class SOBBHHDFBackend(eryn_HDFBackend):
    """Sub-backend that persists the per-leaf SOBBH temperature ladder.

    Mirrors :class:`EMRIHDFBackend` — one ``betas_all`` row per leaf, sliced
    out at save time and restored from disk for warm-starts.
    """

    def reset(self, nwalkers, *args, ntemps=1, num_sobbhs: int = None, **kwargs):
        """Create a ``betas_all`` dataset of shape ``(*, num_sobbhs, ntemps)``."""
        if num_sobbhs is None:
            raise ValueError("Must provide num_sobbhs kwarg.")

        with self.open("a") as f:
            g = f[self.name]["sub_backend"]

            sobbh_group = g.create_group("sobbh")

            sobbh_group.attrs["num_sobbhs"] = num_sobbhs

            sobbh_group.create_dataset(
                "betas_all",
                (0, num_sobbhs, ntemps),
                maxshape=(None, num_sobbhs, ntemps),
                dtype=self.dtype,
                compression=self.compression,
                compression_opts=self.compression_opts,
            )

    @property
    def num_sobbhs(self):
        """Get num_sobbhs from h5 file."""
        with self.open() as f:
            return f[self.name].attrs["num_sobbhs"]

    @property
    def reset_kwargs(self):
        """Get reset_kwargs from h5 file."""
        return dict(num_sobbhs=self.num_sobbhs)

    def grow(self, ngrow, *args):
        """Grow ``betas_all`` by ``ngrow`` rows."""
        with self.open("a") as f:
            g = f[self.name]
            sobbh_group = f[self.name]["sub_backend"]["sobbh"]
            ntot = g.attrs["iteration"] + ngrow
            sobbh_group["betas_all"].resize(ntot, axis=0)

    def get_value(self, name, thin=1, discard=0, slice_vals=None):
        """Return ``betas_all`` slices from the file. See :meth:`EMRIHDFBackend.get_value`."""
        if not self.initialized:
            raise AttributeError(
                "You must run the sampler with 'store == True' before accessing the results"
            )

        if name != "betas_all":
            raise ValueError(f"No {name} in this backend.")

        if slice_vals is None:
            slice_vals = slice(discard + thin - 1, self.iteration, thin)

        with self.open() as f:
            g = f[self.name]
            iteration = g.attrs["iteration"]
            if iteration <= 0:
                raise AttributeError(
                    "You must run the sampler with 'store == True' before accessing the results"
                )

            sobbh_group = g["sub_backend"]["sobbh"]
            v_all = sobbh_group["betas_all"][slice_vals]
        return v_all

    def get_betas_all(self, **kwargs):
        """Get the stored SOBBH ``betas_all`` history."""
        return self.get_value("betas_all", **kwargs)

    def save_step(self, state, *args, **kwargs):
        """Persist this iteration's ``betas_all`` from the current SOBBH sub-state."""
        with self.open("a") as f:
            g = f[self.name]
            iteration = g.attrs["iteration"] - 1
            sobbh_group = g["sub_backend"]["sobbh"]
            sobbh_group["betas_all"][iteration] = state.sub_states["sobbh"].betas_all

    def get_a_sample(self, it):
        """Access a sample in the chain. See :meth:`EMRIHDFBackend.get_a_sample`."""
        thin = self.iteration - it if it != self.iteration else 1
        discard = it + 1 - thin

        betas_all = self.get_betas_all(discard=discard, thin=thin)

        sample = SOBBHState(None, betas_all=betas_all)
        return sample
