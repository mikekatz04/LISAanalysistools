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

    def get_value(self, name, flat=False, thin=1, discard=0):
        if not self.initialized:
            raise AttributeError(
                "You must run the sampler with "
                "'store == True' before accessing the "
                "results"
            )
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

            if name == "accepted":
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
