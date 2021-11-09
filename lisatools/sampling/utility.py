import os

import numpy as np

from emcee.backends.hdf import HDFBackend
from emcee import __version__
from emcee import autocorr


class ModifiedHDFBackend(HDFBackend):
    # TODO: adjust for compression_opts

    def reset(
        self, nwalkers, ndim, ntemps=1, ntemps_target=1, injection=None, test_inds=None
    ):
        """Clear the state of the chain and empty the backend
        Args:
            nwakers (int): The size of the ensemble
            ndim (int): The number of dimensions
        """
        with self.open("a") as f:
            if self.name in f:
                del f[self.name]

            g = f.create_group(self.name)
            g.attrs["version"] = __version__
            g.attrs["nwalkers"] = nwalkers
            g.attrs["ndim"] = ndim
            g.attrs["nwalkers_per_temp"] = int(nwalkers / ntemps)
            g.attrs["ntemps"] = ntemps
            g.attrs["ntemps_target"] = ntemps_target
            g.attrs["has_blobs"] = False
            g.attrs["iteration"] = 0
            g.attrs["test_inds"] = test_inds
            if injection is not None:
                g.attrs["injection"] = injection

                if test_inds is not None:
                    g.attrs["test_injection_values"] = injection

                else:
                    g.attrs["test_injection_values"] = None

            g.create_dataset(
                "accepted",
                data=np.zeros(nwalkers),
                # compression=self.compression,
                # compression_opts=self.compression_opts,
            )
            g.create_dataset(
                "betas",
                (0, ntemps),
                maxshape=(None, ntemps),
                dtype=self.dtype,
                # compression=self.compression,
                # compression_opts=self.compression_opts,
            )
            g.create_dataset(
                "chain",
                (0, nwalkers, ndim),
                maxshape=(None, nwalkers, ndim),
                dtype=self.dtype,
                # compression=self.compression,
                # compression_opts=self.compression_opts,
            )
            g.create_dataset(
                "log_prob",
                (0, nwalkers),
                maxshape=(None, nwalkers),
                dtype=self.dtype,
                # compression=self.compression,
                # compression_opts=self.compression_opts,
            )

    def get_dims(self):
        if not self.initialized:
            raise AttributeError(
                "You must run the sampler with "
                "'store == True' before accessing the "
                "results"
            )
        with self.open() as f:
            g = f[self.name]
            nwalkers = g.attrs["nwalkers_per_temp"]
            ndim = g.attrs["ndim"]
            ntemps = g.attrs["ntemps"]

        return (ntemps, nwalkers, ndim)

    def get_chain(self, squeeze=False, **kwargs):
        """Get the stored chain of MCMC samples
        Args:
            flat (Optional[bool]): Flatten the chain across the ensemble.
                (default: ``False``)
            thin (Optional[int]): Take only every ``thin`` steps from the
                chain. (default: ``1``)
            discard (Optional[int]): Discard the first ``discard`` steps in
                the chain as burn-in. (default: ``0``)
        Returns:
            array[..., nwalkers, ndim]: The MCMC samples.
        """
        ntemps, nwalkers, ndim = self.get_dims()

        out = self.get_value("chain", **kwargs)
        if squeeze:
            out = out.squeeze()
        return out

    def get_temps(self, **kwargs):
        ntemps, nwalkers, ndim = self.get_dims()
        out = self.get_value("betas", **kwargs)
        return out

    def get_value(self, name, flat=False, thin=1, discard=0):
        if not self.initialized:
            raise AttributeError(
                "You must run the sampler with "
                "'store == True' before accessing the "
                "results"
            )
        # TODO: add compression stuff
        with self.open() as f:
            g = f[self.name]
            iteration = g.attrs["iteration"]
            if iteration <= 0:
                raise AttributeError(
                    "You must run the sampler with "
                    "'store == True' before accessing the "
                    "results"
                )

            if name == "blobs" and not g.attrs["has_blobs"]:
                return None

            v = g[name][discard + thin - 1 : self.iteration : thin]

            ntemps, nwalkers, ndim = self.get_dims()

            if name == "accepted" or name == "betas":
                return v

            elif name == "log_prob":
                v = v.reshape(-1, ntemps, nwalkers)
                if flat:
                    v = v.flatten()
                return v

            elif name == "blobs":
                if v.ndim == 3:
                    _, _, nblobs = v.shape
                    v = v.reshape(-1, ntemps, nwalkers, nblobs)
                else:
                    v = v.reshape(-1, ntemps, nwalkers)
                return v

            v = v.reshape(-1, ntemps, nwalkers, ndim)
            if flat:
                v = v.reshape(-1, ndim)
            return v

    def get_autocorr_time(self, discard=0, thin=1, ntemps_target=1, **kwargs):
        """Compute an estimate of the autocorrelation time for each parameter
        Args:
            thin (Optional[int]): Use only every ``thin`` steps from the
                chain. The returned estimate is multiplied by ``thin`` so the
                estimated time is in units of steps, not thinned steps.
                (default: ``1``)
            discard (Optional[int]): Discard the first ``discard`` steps in
                the chain as burn-in. (default: ``0``)
        Other arguments are passed directly to
        :func:`emcee.autocorr.integrated_time`.
        Returns:
            array[ndim]: The integrated autocorrelation time estimate for the
                chain for each parameter.
        """
        x = self.get_chain(discard=discard, thin=thin, squeeze=False)
        ntemps, nwalkers, ndim = self.get_dims()
        x_keep = x[:, :ntemps_target].reshape(-1, ntemps_target * nwalkers, ndim)
        return thin * autocorr.integrated_time(x_keep, **kwargs)

    def get_attr(self, attr):
        with self.open() as f:
            g = f[self.name]
            iteration = g.attrs[attr]

        return iteration

    def save_step(self, state, accepted):
        """Save a step to the backend
        Args:
            state (State): The :class:`State` of the ensemble.
            accepted (ndarray): An array of boolean flags indicating whether
                or not the proposal for each walker was accepted.
        """
        self._check(state, accepted)

        with self.open("a") as f:
            g = f[self.name]
            iteration = g.attrs["iteration"]

            g["chain"][iteration, :, :] = state.coords
            g["log_prob"][iteration, :] = state.log_prob
            if state.blobs is not None:
                g["blobs"][iteration, :] = state.blobs
            g["accepted"][:] += accepted

            for i, v in enumerate(state.random_state):
                g.attrs["random_state_{0}".format(i)] = v

            g.attrs["iteration"] = iteration + 1

    def grow_temps(self, ngrow):
        """Expand the storage space by some number of samples
            Args:
                ngrow (int): The number of steps to grow the chain.
                blobs: The current array of blobs. This is used to compute the
                    dtype for the blobs array.
            """
        with self.open("a") as f:
            g = f[self.name]
            ntot = g.attrs["iteration"] + ngrow
            g["betas"].resize(ntot, axis=0)

    def save_temps(self, temps):
        with self.open("a") as f:
            g = f[self.name]
            iteration = g.attrs["iteration"]

            g["betas"][iteration, :] = temps


class RelBinUpdate:
    def __init__(self, update_kwargs, set_d_d_zero=False):
        self.update_kwargs = update_kwargs
        self.set_d_d_zero = set_d_d_zero

    def __call__(self, it, sample_state, sampler, **kwargs):

        samples = sample_state.branches_coords["mbh"].reshape(-1, sampler.ndims[0])
        lp_max = sample_state.log_prob.argmax()
        best = samples[lp_max]

        lp = sample_state.log_prob.flatten()
        sorted = np.argsort(lp)
        inds_best = sorted[-1000:]
        inds_worst = sorted[:1000]

        best_full = sampler.log_prob_fn.f.parameter_transforms["mbh"].both_transforms(
            best, copy=True
        )

        sampler.log_prob_fn.f.template_model._init_rel_bin_info(
            best_full, **self.update_kwargs
        )

        if self.set_d_d_zero:
            sampler.log_prob_fn.f.template_model.base_d_d = 0.0

        # TODO: make this a general update function in Eryn (?)
        # samples[inds_worst] = samples[inds_best].copy()
        samples = samples.reshape(sampler.ntemps, sampler.nwalkers, 1, sampler.ndims[0])
        logp = sampler.compute_log_prior({"mbh": samples})
        logL, blobs = sampler.compute_log_prob({"mbh": samples}, logp=logp)

        sample_state.branches["mbh"].coords = samples
        sample_state.log_prob = logL
        sample_state.blobs = blobs

        # sampler.backend.save_step(sample_state, np.full_like(lp, True))
