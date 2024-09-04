import numpy as np
from eryn.backends import HDFBackend as eryn_HDFBackend
from .state import State
from .plot import RunResultsProduction
import time
import shutil


def save_to_backend_asynchronously_and_plot(gb_reader, comm, main_rank, head_rank, plot_iter, backup_iter):

    print("starting run SAVE")
    run_results_production = None ## RunResultsProduction(None, None, add_gbs=False, add_mbhs=False)
    run = True
    i = 0
    while run:
        print("WAITING FOR DATA")
        save_dict = comm.recv(source=main_rank)
        print("RECEIVED FOR DATA")
        if "finish_run" in save_dict and save_dict["finish_run"]:
            run = False
            continue

        time.sleep(15.)  # to allow for ending the code
        save_args = save_dict["save_args"]
        save_kwargs = save_dict["save_kwargs"]
        print("attempting to save step")
        st = time.perf_counter()
        gb_reader.save_step_main(*save_args, **save_kwargs)
        et = time.perf_counter()
        print("SAVE STEP, time:", et - st)
        if ((i + 1) % plot_iter) == 0:
            print("ASK FOR DATA FOR PLOT")
            comm.send({"send": True}, dest=head_rank, tag=91)
            current_info = comm.recv(source=head_rank, tag=92)

            # remove GPU component for GB waveform build
            current_info.current_info["gb"]["get_templates"].initialization_kwargs["use_gpu"] = False
            current_info.current_info["mbh"]["get_templates"].initialization_kwargs["use_gpu"] = False
            current_info.current_info["gb"]["get_templates"].runtime_kwargs["use_c_implementation"] = False

            print("STARTING PLOT")
            run_results_production.build_plots(current_info)
            print("FINISHED PLOT")

        if ((i + 1) % backup_iter) == 0:
            print("copy to backup file")
            # copy to backup file
            shutil.copy(gb_reader.filename, gb_reader.filename[:-3] + "_running_backup_copy.h5")

        i += 1
    return 


class HDFBackend(eryn_HDFBackend):

    def __init__(self, *args, comm=None, save_plot_rank=None, **kwargs):

        super().__init__(*args, **kwargs)

        if comm is not None or save_plot_rank is not None:
            if comm is None or save_plot_rank is None:
                raise ValueError("If providing comm/save_plot_rank, must provide both.")

        self.comm = comm
        self.save_plot_rank = save_plot_rank
    
    def reset(self, nwalkers, *args, ntemps=1, num_bands=None, band_edges=None, **kwargs):
        if num_bands is None or band_edges is None:
            raise ValueError("Must provide num_bands and band_edges kwargs.")

        # regular reset
        super().reset(nwalkers, *args, ntemps=ntemps, **kwargs)

        # open file in append mode
        with self.open("a") as f:
            g = f[self.name]

            band_info = g.create_group("band_info")

            band_info.create_dataset(
                "band_edges",
                data=band_edges,
                dtype=self.dtype,
                compression=self.compression,
                compression_opts=self.compression_opts,
            )

            band_info.attrs["num_bands"] = len(band_edges)

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
            return f[self.name]["band_info"].attrs["num_bands"]

    @property
    def band_edges(self):
        """Get band_edges from h5 file."""
        with self.open() as f:
            return f[self.name]["band_info"]["band_edges"][:]

    @property
    def reset_kwargs(self):
        """Get reset_kwargs from h5 file."""
        return dict(
            nleaves_max=self.nleaves_max,
            ntemps=self.ntemps,
            branch_names=self.branch_names,
            rj=self.rj,
            moves=self.moves,
            num_bands=self.num_bands
        )

    def grow(self, ngrow, *args):
        super().grow(ngrow, *args)
        
        # open the file in append mode
        with self.open("a") as f:
            g = f[self.name]

            # resize all the arrays accordingly
            ntot = g.attrs["iteration"] + ngrow
            for key in g["band_info"]:
                if key == "band_edges":
                    continue
                g["band_info"][key].resize(ntot, axis=0)

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
                "You must run the sampler with "
                "'store == True' before accessing the "
                "results"
            )

        if name != "band_info":
            return super().get_value(name, thin=thin, discard=discard, slice_vals=slice_vals) 

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

                    v_all = {key: g["band_info"][key][slice_vals] for key in g["band_info"] if key != "band_edges"}
                    v_all["band_edges"] = g["band_info"]["band_edges"][:]
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

    def save_step_main(
        self,
        state,
        *args, 
        **kwargs
    ):

        super().save_step(state, *args, **kwargs)
        
        # open for appending in with statement
        with self.open("a") as f:
            g = f[self.name]
            # get the iteration left off on
            # minus one because it was updated in the super function
            iteration = g.attrs["iteration"] - 1

            # make sure the backend has all the information needed to store everything
            for key in [
                "num_bands",
            ]:
                if not hasattr(self, key):
                    setattr(self, key, g.attrs[key])

            # branch-specific
            for name, dat in state.band_info.items():
                if not isinstance(dat, np.ndarray) or name == "band_edges":
                    continue
                g["band_info"][name][iteration] = dat

        # reset the counter for band info
        state.reset_band_counters()
        
    def save_step(
        self,
        *args, 
        **kwargs
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

        if self.comm is None or self.comm.Get_size() < 3:
            self.save_step_main(*args, **kwargs)
        
        else:
            state = args[0]
            mgh = state.mgh
            state.mgh = None
            self.comm.send({"save_args": args, "save_kwargs": kwargs}, dest=self.save_plot_rank)
            state.mgh = mgh

    def get_a_sample(self, it):
        """Access a sample in the chain

        Args:
            it (int): iteration of State to return.

        Returns:
            State: :class:`eryn.state.State` object containing the sample from the chain.

        Raises:
            AttributeError: Backend is not initialized.

        """
        sample = State(super().get_a_sample(it))

        thin = self.iteration - it if it != self.iteration else 1
        discard = it + 1 - thin

        sample.band_info = self.get_band_info(discard=discard, thin=thin)
        sample.band_info["initialized"] = True

        return sample

    


class MBHHDFBackend(eryn_HDFBackend):

    def reset(self, nwalkers, *args, ntemps=1, num_mbhs: int=None, **kwargs):
        if num_mbhs is None:
            raise ValueError("Must provide num_mbhs kwarg.")

        # regular reset
        super().reset(nwalkers, *args, ntemps=ntemps, **kwargs)

        # open file in append mode
        with self.open("a") as f:
            g = f[self.name]

            g.attrs["num_mbhs"] = num_mbhs

            g.create_dataset(
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
            return f[self.name].attrs["num_mbhs"]

    @property
    def reset_kwargs(self):
        """Get reset_kwargs from h5 file."""
        return dict(
            nleaves_max=self.nleaves_max,
            ntemps=self.ntemps,
            branch_names=self.branch_names,
            rj=self.rj,
            moves=self.moves,
            num_bands=self.num_mbhs
        )

    def grow(self, ngrow, *args):

        super().grow(ngrow, *args)
        
        # open the file in append mode
        with self.open("a") as f:
            g = f[self.name]

            # resize all the arrays accordingly
            ntot = g.attrs["iteration"] + ngrow
            g["betas_all"].resize(ntot, axis=0)

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
                "You must run the sampler with "
                "'store == True' before accessing the "
                "results"
            )

        if name != "betas_all":
            return super().get_value(name, thin=thin, discard=discard, slice_vals=slice_vals) 

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

            v_all = g["betas_all"][slice_vals]
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

    def save_step(
        self,
        state,
        *args, 
        **kwargs
    ):

        super().save_step(state, *args, **kwargs)
        
        # open for appending in with statement
        with self.open("a") as f:
            g = f[self.name]
            # get the iteration left off on
            # minus one because it was updated in the super function
            iteration = g.attrs["iteration"] - 1

            g["betas_all"][iteration] = state.betas_all

    def get_a_sample(self, it):
        """Access a sample in the chain

        Args:
            it (int): iteration of State to return.

        Returns:
            State: :class:`eryn.state.State` object containing the sample from the chain.

        Raises:
            AttributeError: Backend is not initialized.

        """
        sample = State(super().get_a_sample(it))

        thin = self.iteration - it if it != self.iteration else 1
        discard = it + 1 - thin

        sample.betas_all = self.get_betas_all(discard=discard, thin=thin)

        return sample

    