from __future__ import annotations

import os
import time
import shutil
from typing import Any, Dict, List, Optional, Union
from collections.abc import Iterable

import h5py
import numpy as np
from numpy.typing import NDArray

from eryn.backends import HDFBackend as eryn_HDFBackend
from eryn.backends import Backend
from .state import EMRIState, GBState, GFState, MBHState

VERSION = "lisa_global_fit_backend"

def save_to_backend_asynchronously_and_plot(
    gb_reader, comm, main_rank, head_rank, plot_iter, backup_iter
):

    print("starting run SAVE")
    run_results_production = (
        None  ## RunResultsProduction(None, None, add_gbs=False, add_mbhs=False)
    )
    run = True
    i = 0
    while run:
        print("WAITING FOR DATA")
        save_dict = comm.recv(source=main_rank)
        print("RECEIVED FOR DATA")
        if "finish_run" in save_dict and save_dict["finish_run"]:
            run = False
            continue

        time.sleep(15.0)  # to allow for ending the code
        save_args = save_dict["save_args"]
        save_kwargs = save_dict["save_kwargs"]
        print("attempting to save step")
        st = time.perf_counter()
        gb_reader.save_step_main(*save_args, **save_kwargs)
        et = time.perf_counter()
        print("SAVE STEP, time:", et - st)
        # if ((i + 1) % plot_iter) == 0:
        #     print("ASK FOR DATA FOR PLOT")
        #     comm.send({"send": True}, dest=head_rank, tag=91)
        #     current_info = comm.recv(source=head_rank, tag=92)

        #     # remove GPU component for GB waveform build
        #     current_info.current_info["gb"]["get_templates"].initialization_kwargs["use_gpu"] = False
        #     current_info.current_info["mbh"]["get_templates"].initialization_kwargs["use_gpu"] = False
        #     current_info.current_info["gb"]["get_templates"].runtime_kwargs["use_c_implementation"] = False

        #     print("STARTING PLOT")
        #     run_results_production.build_plots(current_info)
        #     print("FINISHED PLOT")

        if ((i + 1) % backup_iter) == 0:
            print("copy to backup file")
            # copy to backup file
            shutil.copy(gb_reader.filename, gb_reader.filename[:-3] + "_running_backup_copy.h5")

        i += 1
    return

class NewHDFBackend(Backend):
    """A backend that stores the chain in an HDF5 file using h5py

    .. note:: You must install `h5py <http://www.h5py.org/>`_ to use this
        backend.

    Args:
        filename (str): The name of the HDF5 file where the chain will be
            saved.
        name (str, optional): The name of the group where the chain will
            be saved. (default: ``"mcmc"``)
        read_only (bool, optional): If ``True``, the backend will throw a
            ``RuntimeError`` if the file is opened with write access.
            (default: ``False``)
        dtype (dtype, optional): Dtype to use for data storage. If None,
            program uses np.float64. (default: ``None``)
        compression (str, optional): Compression type for h5 file. See more information
            in the
            `h5py documentation <https://docs.h5py.org/en/stable/high/dataset.html#filter-pipeline>`_.
            (default: ``None``)
        compression_opts (int, optional): Compression level for h5 file. See more information
            in the
            `h5py documentation <https://docs.h5py.org/en/stable/high/dataset.html#filter-pipeline>`_.
            (default: ``None``)
        store_missing_leaves (double, optional): Number to store for leaves that are not
            used in a specific step. (default: ``np.nan``)


    """

    def __init__(
        self,
        filename,
        name="mcmc",
        read_only=False,
        dtype=None,
        compression=None,
        compression_opts=None,
        store_missing_leaves=np.nan,
        buffer_size: int = 10
    ):
        if h5py is None:
            raise ImportError("you must install 'h5py' to use the NewHDFBackend")

        # store all necessary quantities
        self.filename = filename
        self.name = name
        self.read_only = read_only
        self.compression = compression
        self.compression_opts = compression_opts
        if dtype is None:
            self.dtype_set = False
            self.dtype = np.float64
        else:
            self.dtype_set = True
            self.dtype = dtype

        self.store_missing_leaves = store_missing_leaves

        # setup buffering
        if buffer_size < 1:
            raise ValueError("buffer_size must be >= 1.")
        
        self.buffer_size = buffer_size
        self._buffer_count = 0
        self._buffer: Dict[str, Any] = {}        
        
    @property
    def initialized(self):
        """Check if backend file has been initialized properly."""
        if not os.path.exists(self.filename):
            return False
        try:
            with self.open() as f:
                return self.name in f
        except (OSError, IOError):
            return False

    def open(self, mode="r"):
        """Opens the h5 file in the proper mode.

        Args:
            mode (str, optional): Mode to open h5 file.

        Returns:
            H5 file object: Opened file.

        Raises:
            RuntimeError: If backend is opened for writing when it is read-only.

        """

        if self.read_only and mode != "r":
            raise RuntimeError(
                "The backend has been loaded in read-only "
                "mode. Set `read_only = False` to make "
                "changes."
            )

        # open the file
        file_opened = False

        try_num = 0
        max_tries = 50
        while not file_opened:
            try:
                f = h5py.File(self.filename, mode)
                file_opened = True
                
            except (BlockingIOError, OSError) as e:
                try_num += 1
                if try_num >= max_tries:
                    raise BlockingIOError("Max tries exceeded trying to open h5 file.")
                print("Failed to open h5 file. Trying again.")
                time.sleep(2.0)

        # get the data type and store it if it is not previously set
        if not self.dtype_set and self.name in f:
            # get the group from the file
            g = f[self.name]
            if "chain" in g:
                # get the model names in chain
                keys = list(g["chain"])

                # they all have the same dtype so use the first one
                try:
                    self.dtype = g["chain"][keys[0]].dtype

                    # we now have it
                    self.dtype_set = True
                # catch error if the chain has not been initialized yet
                except IndexError:
                    pass

        return f

    def reset(
        self,
        nwalkers,
        ndims,
        nleaves_max=1,
        ntemps=1,
        branch_names=None,
        nbranches=1,
        rj=False,
        moves=None,
        key_order=None,
        **info,
    ):
        """Clear the state of the chain and empty the backend

        Args:
            nwalkers (int): The size of the ensemble
            ndims (int, list of ints, or dict): The number of dimensions for each branch. If
                ``dict``, keys should be the branch names and values the associated dimensionality.
            nleaves_max (int, list of ints, or dict, optional): Maximum allowable leaf count for each branch.
                It should have the same length as the number of branches.
                If ``dict``, keys should be the branch names and values the associated maximal leaf value.
                (default: ``1``)
            ntemps (int, optional): Number of rungs in the temperature ladder.
                (default: ``1``)
            branch_names (str or list of str, optional): Names of the branches used. If not given,
                branches will be names ``model_0``, ..., ``model_n`` for ``n`` branches.
                (default: ``None``)
            nbranches (int, optional): Number of branches. This is only used if ``branch_names is None``.
                (default: ``1``)
            rj (bool, optional): If True, reversible-jump techniques are used.
                (default: ``False``)
            moves (list, optional): List of all of the move classes input into the sampler.
                (default: ``None``)
            key_order (dict, optional): Keys are ``branch_names`` and values are lists of key ordering for each
                branch. For example, ``{"model_0": ["x1", "x2", "x3"]}``. 
                (default: ``None``)
            **info (dict, optional): Any other key-value pairs to be added
                as attributes to the backend. These are also added to the HDF5 file.

        """

        # open file in append mode
        with self.open("a") as f:
            # we are resetting so if self.name in the file we need to delete it
            if self.name in f:
                del f[self.name]

            # turn things into lists/dicts if needed
            if branch_names is not None:
                if isinstance(branch_names, str):
                    branch_names = [branch_names]

                elif not isinstance(branch_names, list):
                    raise ValueError("branch_names must be string or list of strings.")

            else:
                branch_names = ["model_{}".format(i) for i in range(nbranches)]

            nbranches = len(branch_names)

            if isinstance(ndims, int):
                assert len(branch_names) == 1
                ndims = {branch_names[0]: ndims}

            elif isinstance(ndims, list) or isinstance(ndims, np.ndarray):
                assert len(branch_names) == len(ndims)
                ndims = {bn: nd for bn, nd in zip(branch_names, ndims)}

            elif isinstance(ndims, dict):
                assert len(list(ndims.keys())) == len(branch_names)
                for key in ndims:
                    if key not in branch_names:
                        raise ValueError(
                            f"{key} is in ndims but does not appear in branch_names: {branch_names}."
                        )
            else:
                raise ValueError("ndims is to be a scalar int, list or dict.")

            if isinstance(nleaves_max, int):
                assert len(branch_names) == 1
                nleaves_max = {branch_names[0]: nleaves_max}

            elif isinstance(nleaves_max, list) or isinstance(nleaves_max, np.ndarray):
                assert len(branch_names) == len(nleaves_max)
                nleaves_max = {bn: nl for bn, nl in zip(branch_names, nleaves_max)}

            elif isinstance(nleaves_max, dict):
                assert len(list(nleaves_max.keys())) == len(branch_names)
                for key in nleaves_max:
                    if key not in branch_names:
                        raise ValueError(
                            f"{key} is in nleaves_max but does not appear in branch_names: {branch_names}."
                        )
            else:
                raise ValueError("nleaves_max is to be a scalar int, list, or dict.")

            # store all the info needed in memory and in the file

            g = f.create_group(self.name)

            g.attrs["version"] = VERSION
            g.attrs["nbranches"] = len(branch_names)
            g.attrs["branch_names"] = branch_names
            g.attrs["ntemps"] = ntemps
            g.attrs["nwalkers"] = nwalkers
            g.attrs["has_blobs"] = False
            g.attrs["rj"] = rj
            g.attrs["iteration"] = 0

            # create info group
            g.create_group("info")
            # load info into class and into file
            for key, value in info.items():
                setattr(self, key, value)
                if isinstance(value, np.ndarray) and value.size > 1000:
                    g["info"].create_dataset(key, data=value)
                else:
                    g["info"].attrs[key] = value

            # store nleaves max and ndims dicts
            g.create_group("ndims")
            for key, value in ndims.items():
                g["ndims"].attrs[key] = value

            g.create_group("nleaves_max")
            for key, value in nleaves_max.items():
                g["nleaves_max"].attrs[key] = value

            # prepare all the data sets

            g.create_dataset(
                "accepted",
                data=np.zeros((ntemps, nwalkers)),
                compression=self.compression,
                compression_opts=self.compression_opts,
            )

            g.create_dataset(
                "swaps_accepted",
                data=np.zeros((ntemps - 1,)),
                compression=self.compression,
                compression_opts=self.compression_opts,
            )

            if self.rj:
                g.create_dataset(
                    "rj_accepted",
                    data=np.zeros((ntemps, nwalkers)),
                    compression=self.compression,
                    compression_opts=self.compression_opts,
                )

            g.create_dataset(
                "log_like",
                (0, ntemps, nwalkers),
                maxshape=(None, ntemps, nwalkers),
                dtype=self.dtype,
                compression=self.compression,
                compression_opts=self.compression_opts,
            )

            g.create_dataset(
                "log_prior",
                (0, ntemps, nwalkers),
                maxshape=(None, ntemps, nwalkers),
                dtype=self.dtype,
                compression=self.compression,
                compression_opts=self.compression_opts,
            )

            g.create_dataset(
                "betas",
                (0, ntemps),
                maxshape=(None, ntemps),
                dtype=self.dtype,
                compression=self.compression,
                compression_opts=self.compression_opts,
            )

            # setup data sets for branch-specific items

            chain = g.create_group("chain")
            inds = g.create_group("inds")
            k_o_g = g.create_group("key_order")

            for name in branch_names:
                nleaves = self.nleaves_max[name]
                ndim = self.ndims[name]
                chain.create_dataset(
                    name,
                    (0, ntemps, nwalkers, nleaves, ndim),
                    maxshape=(None, ntemps, nwalkers, nleaves, ndim),
                    dtype=self.dtype,
                    compression=self.compression,
                    compression_opts=self.compression_opts,
                )

                inds.create_dataset(
                    name,
                    (0, ntemps, nwalkers, nleaves),
                    maxshape=(None, ntemps, nwalkers, nleaves),
                    dtype=bool,
                    compression=self.compression,
                    compression_opts=self.compression_opts,
                )

                if key_order is not None:
                    k_o_g.attrs[name] = key_order[name] 

            # store move specific information
            if moves is not None:
                move_group = g.create_group("moves")
                # setup info and keys
                for full_move_name in moves:

                    single_move = move_group.create_group(full_move_name)

                    # prepare information dictionary
                    single_move.create_dataset(
                        "acceptance_fraction",
                        (ntemps, nwalkers),
                        maxshape=(ntemps, nwalkers),
                        dtype=self.dtype,
                        compression=self.compression,
                        compression_opts=self.compression_opts,
                    )

            else:
                self.move_info = None

            self.blobs = None

    @property
    def key_order(self):
        """Key order of parameters for each model."""
        with self.open() as f:
            return {key: value for key, value in f[self.name]["key_order"].attrs.items()}

    @property
    def nwalkers(self):
        """Get nwalkers from h5 file."""
        with self.open() as f:
            return f[self.name].attrs["nwalkers"]

    @property
    def ntemps(self):
        """Get ntemps from h5 file."""
        with self.open() as f:
            return f[self.name].attrs["ntemps"]

    @property
    def rj(self):
        """Get rj from h5 file."""
        with self.open() as f:
            return f[self.name].attrs["rj"]

    @property
    def nleaves_max(self):
        """Get nleaves_max from h5 file."""
        with self.open() as f:
            return {
                key: f[self.name]["nleaves_max"].attrs[key]
                for key in f[self.name]["nleaves_max"].attrs
            }

    @property
    def ndims(self):
        """Get ndims from h5 file."""
        with self.open() as f:
            return {
                key: f[self.name]["ndims"].attrs[key]
                for key in f[self.name]["ndims"].attrs
            }

    @property
    def move_keys(self):
        """Get move_keys from h5 file."""
        with self.open() as f:
            return list(f[self.name]["moves"])

    @property
    def branch_names(self):
        """Get branch names from h5 file."""
        with self.open() as f:
            return list(f[self.name].attrs["branch_names"])

    @property
    def nbranches(self):
        """Get number of branches from h5 file."""
        with self.open() as f:
            return f[self.name].attrs["nbranches"]

    @property
    def reset_args(self):
        """Get reset_args from h5 file."""
        return [self.nwalkers, self.ndims]

    @property
    def reset_kwargs(self):
        """Get reset_kwargs from h5 file."""
        return dict(
            nleaves_max=self.nleaves_max,
            ntemps=self.ntemps,
            branch_names=self.branch_names,
            rj=self.rj,
            moves=self.moves if hasattr(self, 'moves') else None,
            key_order=self.key_order,
        )

    def has_blobs(self):
        """Returns ``True`` if the model includes blobs"""
        with self.open() as f:
            return f[self.name].attrs["has_blobs"]

    def get_value(self, name, thin=1, discard=0, slice_vals=None, temp_index=None, branch_names=None):
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
            temp_index (int, optional): Integer for the desired temperature index.
                If ``None``, will return all temperatures. (default: ``None``)
            branch_names (str or list, optional): Specific branch names requested. (default: ``None``)
            
        Returns:
            dict or np.ndarray: Values requested.

        """
        # check if initialized
        if not self.initialized:
            raise AttributeError(
                "You must run the sampler with "
                "'store == True' before accessing the "
                "results."
                "When using the HDF backend, make sure you have the file"
                "path correctly set. This is the error that"
                "is given if the backend cannot find the file."
            )

        if slice_vals is None:
            slice_vals = slice(discard + thin - 1, self.iteration, thin)

        # make sure branch_names input is a list
        if branch_names is not None:
            if isinstance(branch_names, str):
                branches_names = [branch_names]

        branch_names_in = self.branch_names if branch_names is None else branch_names

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

                    if temp_index is None:
                        temp_index = np.arange(self.ntemps)
                    else:
                        assert isinstance(temp_index, int)

                    if name == "chain":
                        v_all = {key: g["chain"][key][slice_vals, temp_index] for key in branch_names_in}

                    elif name == "inds":
                        v_all = {key: g["inds"][key][slice_vals, temp_index] for key in branch_names_in}
                    
                    elif name == "blobs" and not g.attrs["has_blobs"]:
                        v_all = None
                        
                    else:
                        v_all = g[name][slice_vals, temp_index]

                    successful = True
            
            except OSError:
                num_try += 1
                print(f"Unable to read h5 file {num_try} times.")
                time.sleep(20.0)
            

        if not successful:
            raise OSError("Attempted to open file max try number of times. Likely cannot read data.")
                
        return v_all

    def get_move_info(self):
        """Get move information.

        Returns:
            dict: Keys are move names and values are dictionaries with information on the moves.

        """
        # setup output dictionary
        move_info_out = {}
        with self.open() as f:
            g = f[self.name]

            # iterate through everything and produce a dictionary
            for move_name in g["moves"]:
                move_info_out[move_name] = {}
                for info_name in g["moves"][move_name]:
                    move_info_out[move_name][info_name] = g["moves"][move_name][
                        info_name
                    ][:]

        return move_info_out

    @property
    def shape(self):
        """The dimensions of the ensemble

        Returns:
            dict: Shape of samples
                Keys are ``branch_names`` and values are tuples with
                shapes of individual branches: (ntemps, nwalkers, nleaves_max, ndim).

        """
        # open file wrapped in with
        with self.open() as f:
            g = f[self.name]
            return {
                key: (
                    g.attrs["ntemps"],
                    g.attrs["nwalkers"],
                    self.nleaves_max[key],
                    self.ndims[key],
                )
                for key in g.attrs["branch_names"]
            }

    @property
    def iteration(self):
        """Number of iterations stored in the hdf backend so far."""
        with self.open() as f:
            return f[self.name].attrs["iteration"]

    @property
    def accepted(self):
        """Number of accepted moves per walker."""
        with self.open() as f:
            return f[self.name]["accepted"][...]

    @property
    def rj_accepted(self):
        """Number of accepted rj moves per walker."""
        with self.open() as f:
            return f[self.name]["rj_accepted"][...]

    @property
    def swaps_accepted(self):
        """Number of accepted swaps."""
        with self.open() as f:
            return f[self.name]["swaps_accepted"][...]

    @property
    def random_state(self):
        """Get the random state"""
        with self.open() as f:
            elements = [
                v
                for k, v in sorted(f[self.name].attrs.items())
                if k.startswith("random_state_")
            ]
        return elements if len(elements) else None

    def grow(self, ngrow, blobs):
        """Expand the storage space by some number of samples

        Args:
            ngrow (int): The number of steps to grow the chain.
            blobs (None or np.ndarray): The current array of blobs. This is used to compute the
                dtype for the blobs array.

        """
        self._check_blobs(blobs)

        # open the file in append mode
        with self.open("a") as f:
            g = f[self.name]

            # resize all the arrays accordingly

            ntot = g.attrs["iteration"] + ngrow
            for key in g["chain"]:
                g["chain"][key].resize(ntot, axis=0)
                g["inds"][key].resize(ntot, axis=0)

            g["log_like"].resize(ntot, axis=0)
            g["log_prior"].resize(ntot, axis=0)
            g["betas"].resize(ntot, axis=0)

            # deal with blobs
            if blobs is not None:
                has_blobs = g.attrs["has_blobs"]
                # if blobs have not been added yet
                if not has_blobs:
                    nwalkers = g.attrs["nwalkers"]
                    ntemps = g.attrs["ntemps"]
                    g.create_dataset(
                        "blobs",
                        (ntot, ntemps, nwalkers, blobs.shape[-1]),
                        maxshape=(None, ntemps, nwalkers, blobs.shape[-1]),
                        dtype=self.dtype,
                        compression=self.compression,
                        compression_opts=self.compression_opts,
                    )
                else:
                    # resize the blobs if they have been there
                    g["blobs"].resize(ntot, axis=0)
                    if g["blobs"].shape[1:] != blobs.shape:
                        raise ValueError(
                            "Existing blobs have shape {} but new blobs "
                            "requested with shape {}".format(
                                g["blobs"].shape[1:], blobs.shape
                            )
                        )
                g.attrs["has_blobs"] = True

    def save_step(
        self,
        state,
        accepted,
        rj_accepted=None,
        swaps_accepted=None,
        moves_accepted_fraction=None,
    ):
        """Save a step to the backend

        Args:
            state (State): The :class:`State` of the ensemble.
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
        file_opened = False
        max_tries = 100
        try_num = 0
        while not file_opened:
            try:
                # Open the file exactly ONCE per step
                with self.open("a") as f:
                    self._save_step_to_group(
                        f[self.name],
                        state,
                        accepted,
                        rj_accepted,
                        swaps_accepted,
                        moves_accepted_fraction,
                    )
                file_opened = True
            except (BlockingIOError, OSError):
                try_num += 1
                if try_num >= max_tries:
                    raise BlockingIOError("Max tries exceeded trying to open h5 file.")
                time.sleep(10.0)
    
    def _save_step_to_group(
        self,
        g: h5py.Group,
        state: Any,
        accepted: NDArray[np.float64],
        rj_accepted: Optional[NDArray[np.float64]] = None,
        swaps_accepted: Optional[NDArray[np.float64]] = None,
        moves_accepted_fraction: Optional[Dict[str, NDArray[np.float64]]] = None,
    ) -> None:
        """Internal write method that accepts an open HDF5 Group."""
        iteration: int = g.attrs["iteration"]

        # Ensure attributes are loaded
        for key in ["rj", "ntemps", "nwalkers", "nbranches", "branch_names", "ndims"]:
            if not hasattr(self, key):
                setattr(self, key, g.attrs[key])

        self._check(
            state,
            accepted,
            rj_accepted=rj_accepted,
            swaps_accepted=swaps_accepted,
        )

        for name, model in state.branches.items():
            g["inds"][name][iteration] = model.inds
            
            # --- PERFORMANCE FIX: Avoid np.repeat ---
            # Copy coordinates, then use boolean mask broadcasting 
            # to zero out missing leaves. This avoids allocating a massive array.
            coords_in = model.coords.copy()
            coords_in[~model.inds] = self.store_missing_leaves
            
            g["chain"][name][iteration] = coords_in

        g["log_like"][iteration, :] = state.log_like
        g["log_prior"][iteration, :] = state.log_prior
        
        if state.blobs is not None:
            g["blobs"][iteration, :] = state.blobs
        if state.betas is not None:
            g["betas"][iteration, :] = state.betas
            
        g["accepted"][:] += accepted
        if swaps_accepted is not None:
            g["swaps_accepted"][:] += swaps_accepted
        if self.rj and rj_accepted is not None:
            g["rj_accepted"][:] += rj_accepted

        for i, v in enumerate(state.random_state):
            g.attrs[f"random_state_{i}"] = v

        g.attrs["iteration"] = iteration + 1

        if moves_accepted_fraction is not None:
            if "moves" not in g:
                raise ValueError("moves_info was not initialized.")
            for move_key in self.move_keys:
                g["moves"][move_key]["acceptance_fraction"][:] = moves_accepted_fraction[move_key]

    def flush(self) -> None:
        """Manually trigger a flush of the in-memory buffer to the HDF5 file."""
        if self._buffer_count == 0:
            return

        file_opened = False
        max_tries = 100
        try_num = 0
        while not file_opened:
            try:
                # Open the file ONCE to flush everything
                with self.open("a") as f:
                    self._flush_buffer_to_group(f[self.name])
                file_opened = True
            except (BlockingIOError, OSError):
                try_num += 1
                if try_num >= max_tries:
                    raise BlockingIOError("Max tries exceeded trying to open h5 file.")
                time.sleep(2.0)
                
    def _append_to_buffer(
        self,
        state: Any,
        accepted: NDArray[np.float64],
        rj_accepted: Optional[NDArray[np.float64]] = None,
        swaps_accepted: Optional[NDArray[np.float64]] = None,
        moves_accepted_fraction: Optional[Dict[str, NDArray[np.float64]]] = None,
    ) -> None:
        """Stores the current step in memory instead of immediately writing to HDF5."""
        
        # Initialize buffer dictionary if empty
        if not self._buffer:
            self._buffer = {
                "chain": {name: [] for name in state.branches},
                "inds": {name: [] for name in state.branches},
                "log_like": [],
                "log_prior": [],
                "blobs": [],
                "betas": [],
                "accepted": np.zeros_like(accepted),
                "rj_accepted": np.zeros_like(rj_accepted) if rj_accepted is not None else None,
                "swaps_accepted": np.zeros_like(swaps_accepted) if swaps_accepted is not None else None,
            }

        # 1. Branch-specific data
        for name, model in state.branches.items():
            self._buffer["inds"][name].append(model.inds.copy())
            
            coords_in = model.coords.copy()
            coords_in[~model.inds] = self.store_missing_leaves
            self._buffer["chain"][name].append(coords_in)

        # 2. Likelihood, priors, and general sampler data
        self._buffer["log_like"].append(state.log_like.copy())
        self._buffer["log_prior"].append(state.log_prior.copy())
        
        if state.blobs is not None:
            self._buffer["blobs"].append(state.blobs.copy())
        if state.betas is not None:
            self._buffer["betas"].append(state.betas.copy())

        # 3. Accumulate acceptance statistics
        self._buffer["accepted"] += accepted
        if swaps_accepted is not None:
            self._buffer["swaps_accepted"] += swaps_accepted
        if getattr(self, "rj", False) and rj_accepted is not None:
            self._buffer["rj_accepted"] += rj_accepted

        # 4. Overwrite states that just need the most recent value
        self._buffer["random_state"] = state.random_state
        if moves_accepted_fraction is not None:
            self._buffer["moves_fraction"] = moves_accepted_fraction

        self._buffer_count += 1

    def _flush_buffer_to_group(self, g: h5py.Group) -> None:
        """Writes the buffered list of numpy arrays to the HDF5 group via slices."""
        if self._buffer_count == 0:
            return

        iteration: int = g.attrs["iteration"]
        end_iter: int = iteration + self._buffer_count

        # Bulk write datasets
        for name in self._buffer["chain"]:
            g["inds"][name][iteration:end_iter] = np.stack(self._buffer["inds"][name], axis=0)
            g["chain"][name][iteration:end_iter] = np.stack(self._buffer["chain"][name], axis=0)

        g["log_like"][iteration:end_iter] = np.stack(self._buffer["log_like"], axis=0)
        g["log_prior"][iteration:end_iter] = np.stack(self._buffer["log_prior"], axis=0)

        if self._buffer["blobs"]:
            g["blobs"][iteration:end_iter] = np.stack(self._buffer["blobs"], axis=0)
        if self._buffer["betas"]:
            g["betas"][iteration:end_iter] = np.stack(self._buffer["betas"], axis=0)

        # Bulk update sums and state
        g["accepted"][:] += self._buffer["accepted"]
        if self._buffer["swaps_accepted"] is not None:
            g["swaps_accepted"][:] += self._buffer["swaps_accepted"]
        if getattr(self, "rj", False) and self._buffer["rj_accepted"] is not None:
            g["rj_accepted"][:] += self._buffer["rj_accepted"]

        for i, v in enumerate(self._buffer["random_state"]):
            g.attrs[f"random_state_{i}"] = v

        if "moves_fraction" in self._buffer:
            for move_key in self.move_keys:
                g["moves"][move_key]["acceptance_fraction"][:] = self._buffer["moves_fraction"][move_key]

        # Update the main iteration attribute
        g.attrs["iteration"] = end_iter

        # Clear buffer memory
        self._buffer.clear()
        self._buffer_count = 0



class GFHDFBackend(NewHDFBackend):
    def __init__(
        self,
        *args,
        comm=None,
        sub_backend=None,
        sub_state_bases=None,
        save_plot_rank=None,
        **kwargs,
    ):

        super().__init__(*args, **kwargs)

        if comm is not None or save_plot_rank is not None:
            if comm is None or save_plot_rank is None:
                raise ValueError("If providing comm/save_plot_rank, must provide both.")

        self.comm = comm
        self.save_plot_rank = save_plot_rank

        self.sub_backend = sub_backend
        if self.sub_backend is not None:
            self.sub_backend = {
                key: self.sub_backend[key](*args, **kwargs)
                for key in self.sub_backend
                if self.sub_backend[key] is not None
            }

        self.sub_state_bases = sub_state_bases
        self.recipe_added = False

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

    def save_step_main(self, state: Any, *args: Any, **kwargs: Any) -> None:
        """Appends to buffer. Flushes all backends if buffer is full."""
        self._append_to_buffer(state, *args, **kwargs)
        
        if self.sub_backend is not None:
            for key, sub_state in self.sub_backend.items():
                if sub_state is not None:
                    sub_state._append_to_buffer(state, *args, **kwargs)

        # If buffer threshold is met, trigger the flush which opens the file ONCE
        if self._buffer_count >= self.buffer_size:
            self.flush()

    def save_step(self, *args: Any, **kwargs: Any) -> None:
        """Routes the step depending on MPI size."""
        if self.comm is None or self.comm.Get_size() < 3:
            self.save_step_main(*args, **kwargs)
        else:
            self.comm.send({"save_args": args, "save_kwargs": kwargs}, dest=self.save_plot_rank)

    def _flush_buffer_to_group(self, g: h5py.Group) -> None:
        """Propagates the flush command down to sub-backends."""
        # Parent flush writes its data and updates the global `iteration` attribute
        super()._flush_buffer_to_group(g)
        
        if self.sub_backend is not None:
            sub_group = g["sub_backend"]
            for key, sub_state in self.sub_backend.items():
                if sub_state is not None:
                    sub_state._flush_buffer_to_group(sub_group[key])

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
        with self.open() as f:
            return f[self.name].attrs["has_recipe"]

    @property
    def recipe(self):
        assert self.has_recipe
        with self.open() as f:
            _recipe = {}
            order = []
            keys = []
            for key in f[self.name]["recipe"]:
                _recipe[key] = {key: val for key, val in f["mcmc"]["recipe"][key].attrs.items()}
                order.append(_recipe[key]["order num"])
                keys.append(key)

            new_order = np.argsort(np.asarray(order))
            recipe = {}
            for i in new_order:
                recipe[keys[i]] = _recipe[keys[i]]

        return recipe

    def add_recipe(self, recipe):
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
        with self.open("a") as f:
            recipe_group = f[self.name]["recipe"]
            recipe_step_group = recipe_group[step_name]
            recipe_step_group.attrs["status"] = True


class GBHDFBackend(NewHDFBackend):

    def reset(self, nwalkers, *args, ntemps=1, num_bands=None, band_edges=None, **kwargs):
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
    
    def _save_step_to_group(self, g: h5py.Group, state: Any, *args: Any, **kwargs: Any) -> None:
        # Get the iteration left off on (minus one because parent updated it)
        iteration: int = g.parent.parent.attrs["iteration"] - 1

        for key in ["num_bands"]:
            if not hasattr(self, key):
                setattr(self, key, g.attrs[key])

        for name, dat in state.sub_states["gb"].band_info.items():
            if not isinstance(dat, np.ndarray) or name == "band_edges":
                continue
            g[name][iteration] = dat

        state.sub_states["gb"].reset_band_counters()
        
    def _append_to_buffer(self, state: Any, *args: Any, **kwargs: Any) -> None:
        if not self._buffer:
            # ONLY create lists for items that are actually numpy arrays
            self._buffer = {
                name: [] 
                for name, dat in state.sub_states["gb"].band_info.items() 
                if name != "band_edges" and isinstance(dat, np.ndarray)
            }

        for name, dat in state.sub_states["gb"].band_info.items():
            if not isinstance(dat, np.ndarray) or name == "band_edges":
                continue
            self._buffer[name].append(dat.copy())
            
        state.sub_states["gb"].reset_band_counters()
        self._buffer_count += 1

    def _flush_buffer_to_group(self, g: h5py.Group) -> None:
        if self._buffer_count == 0:
            return
            
        # GFHDFBackend incremented the main iteration, so we subtract our count to find the start point
        end_iter: int = g.parent.parent.attrs["iteration"]
        start_iter: int = end_iter - self._buffer_count

        for name in self._buffer:
            g[name][start_iter:end_iter] = np.stack(self._buffer[name], axis=0)

        self._buffer.clear()
        self._buffer_count = 0


class MBHHDFBackend(NewHDFBackend):

    def reset(self, nwalkers, *args, ntemps=1, num_mbhs: int = None, **kwargs):
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

    def _save_step_to_group(self, g: h5py.Group, state: Any, *args: Any, **kwargs: Any) -> None:
        iteration: int = g.parent.parent.attrs["iteration"] - 1
        g["betas_all"][iteration] = state.sub_states["mbh"].betas_all
        
    def _append_to_buffer(self, state: Any, *args: Any, **kwargs: Any) -> None:
        if not self._buffer:
            self._buffer = {"betas_all": []}
        self._buffer["betas_all"].append(state.sub_states["mbh"].betas_all.copy())
        self._buffer_count += 1

    def _flush_buffer_to_group(self, g: h5py.Group) -> None:
        if self._buffer_count == 0:
            return
        end_iter: int = g.parent.parent.attrs["iteration"]
        start_iter: int = end_iter - self._buffer_count
        g["betas_all"][start_iter:end_iter] = np.stack(self._buffer["betas_all"], axis=0)
        self._buffer.clear()
        self._buffer_count = 0

# TODO: @ alessandro, we can use the same for EMRIs and MBHs
# for now, but I assume we will want it separate in the end


class EMRIHDFBackend(NewHDFBackend):

    def reset(self, nwalkers, *args, ntemps=1, num_emris: int = None, **kwargs):
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
    
    def _save_step_to_group(self, g: h5py.Group, state: Any, *args: Any, **kwargs: Any) -> None:
        iteration: int = g.parent.parent.attrs["iteration"] - 1
        g["betas_all"][iteration] = state.sub_states["emri"].betas_all
        
    def _append_to_buffer(self, state: Any, *args: Any, **kwargs: Any) -> None:
        if not self._buffer:
            self._buffer = {"betas_all": []}
        self._buffer["betas_all"].append(state.sub_states["emri"].betas_all.copy())
        self._buffer_count += 1

    def _flush_buffer_to_group(self, g: h5py.Group) -> None:
        if self._buffer_count == 0:
            return
        end_iter: int = g.parent.parent.attrs["iteration"]
        start_iter: int = end_iter - self._buffer_count
        g["betas_all"][start_iter:end_iter] = np.stack(self._buffer["betas_all"], axis=0)
        self._buffer.clear()
        self._buffer_count = 0
