import numpy as np
from eryn.backends import HDFBackend as eryn_HDFBackend
from .state import GFState, MBHState, EMRIState, GBState
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


class GFHDFBackend(eryn_HDFBackend):
    def __init__(self, *args, comm=None, sub_backend=None, sub_state_bases=None, save_plot_rank=None, **kwargs):

        super().__init__(*args, **kwargs)

        if comm is not None or save_plot_rank is not None:
            if comm is None or save_plot_rank is None:
                raise ValueError("If providing comm/save_plot_rank, must provide both.")

        self.comm = comm
        self.save_plot_rank = save_plot_rank
        
        
        self.sub_backend = sub_backend
        if self.sub_backend is not None:
            self.sub_backend = {key: self.sub_backend[key](*args, **kwargs) for key in self.sub_backend if self.sub_backend[key] is not None}
    
        self.sub_state_bases = sub_state_bases
        self.recipe_added = False

    def reset(self, *args, **kwargs):
        # regular reset
        super().reset(*args, **kwargs)
        
        if self.sub_backend is not None:
            with self.open("a") as f:
                g = f[self.name]
                if "sub_backend" not in g:
                    g.create_group("sub_backend")
            
            for sub_backend_tmp in self.sub_backend.values():
                sub_backend_tmp.reset(*args, **kwargs)

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

            if self.sub_backend is not None:
                # resize all the arrays accordingly

                sub_group = g["sub_backend"]
                for sub_state in self.sub_backend.values():
                    if sub_state is None:
                        continue
                    sub_state.save_step(state, *args, **kwargs)
        
    def save_step(
        self,
        *args, 
        **kwargs
    ):
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
                "you must run the sampler with "
                "'store == True' before accessing the "
                "results"
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
            return f[self.name].attrs["recipe"]

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

class GBHDFBackend(eryn_HDFBackend):

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
            return f[self.name]["sub_backend"]["gb"].attrs["num_bands"]

    @property
    def band_edges(self):
        """Get band_edges from h5 file."""
        with self.open() as f:
            return f[self.name]["sub_backend"]["gb"]["band_edges"][:]

    @property
    def reset_kwargs(self):
        """Get reset_kwargs from h5 file."""
        return dict(
            num_bands=self.num_bands
        )

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
                "You must run the sampler with "
                "'store == True' before accessing the "
                "results"
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
                    v_all = {key: gb_group[key][slice_vals] for key in gb_group if key != "band_edges"}
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

    def save_step(
        self,
        state,
        *args, 
        **kwargs
    ):
 
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

    def reset(self, nwalkers, *args, ntemps=1, num_mbhs: int=None, **kwargs):
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
            return f[self.name].attrs["num_mbhs"]

    @property
    def reset_kwargs(self):
        """Get reset_kwargs from h5 file."""
        return dict(
            num_mbhs=self.num_mbhs
        )

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
                "You must run the sampler with "
                "'store == True' before accessing the "
                "results"
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

    def save_step(
        self,
        state,
        *args, 
        **kwargs
    ):

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

    def reset(self, nwalkers, *args, ntemps=1, num_emris: int=None, **kwargs):
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
        return dict(
            num_emris=self.num_emris
        )

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
                "You must run the sampler with "
                "'store == True' before accessing the "
                "results"
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

    def save_step(
        self,
        state,
        *args, 
        **kwargs
    ):

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

    