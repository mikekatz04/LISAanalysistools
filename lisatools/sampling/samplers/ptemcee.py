import time

import numpy as np

import emcee

from lisatools.sampling.moves.ptredblue import PTStretchMove


class LogUniformPrior:
    def __init__(self, ranges):
        self.minimums = np.asarray([range_i[0] for range_i in ranges])
        self.maximums = np.asarray([range_i[1] for range_i in ranges])

    def __call__(self, x):

        temp = np.sum(
            (x < self.minimums[np.newaxis, :]) * 1.0
            + (x > self.maximums[np.newaxis, :]) * 1.0,
            axis=-1,
        )

        temp[temp > 0.0] = -np.inf

        return temp


class LogProb:
    def __init__(
        self,
        betas,
        ndim_full,
        lnlike,
        lnprior,
        lnlike_kwargs={},
        test_inds=None,
        fill_values=None,
        subset=None,
    ):

        self.lnlike_kwargs = lnlike_kwargs
        self.lnlike = lnlike
        self.lnprior = lnprior

        self.betas = betas
        self.ntemps = len(self.betas)

        self.ndim_full = ndim_full

        if test_inds is not None:
            if fill_values is None:
                raise ValueError("If providing test_inds, need to provide fill_values.")

            self.need_to_fill = True
            self.test_inds = test_inds

            self.fill_inds = np.delete(np.arange(ndim_full), self.test_inds)

            self.fill_values = fill_values

        else:
            self.need_to_fill = False

        self.subset = subset

    def __call__(self, x):
        prior_vals = self.lnprior(x)
        inds_eval = np.squeeze(np.where(np.isinf(prior_vals) != True))

        loglike_vals = np.full(x.shape[0], -np.inf)

        if self.need_to_fill:
            x_in = np.zeros((x.shape[0], self.ndim_full))
            x_in[:, self.test_inds] = x
            x_in[:, self.fill_inds] = self.fill_values[np.newaxis, :]

        else:
            x_in = x

        if self.subset is None:
            temp = self.lnlike.get_ll(x_in[inds_eval], **self.lnlike_kwargs)

        else:
            num_inds = len(inds_eval)
            ind_skip = np.arange(self.subset, num_inds, self.subset)
            inds_eval_temp = [inds for inds in np.split(inds_eval, ind_skip)]

            temp = np.concatenate(
                [
                    self.lnlike.get_ll(x_in[inds], **self.lnlike_kwargs)
                    for inds in inds_eval_temp
                ]
            )

        loglike_vals[inds_eval] = temp

        # tempered like
        logP = -loglike_vals.reshape(self.ntemps, -1) * self.betas[
            :, None
        ] + prior_vals.reshape(self.ntemps, -1)
        return np.array([logP.flatten(), -loglike_vals, prior_vals]).T


class PTEmceeSampler:
    def __init__(
        self,
        nwalkers,
        ndim,
        ndim_full,
        lnprob,
        prior_ranges,
        subset=None,
        lnlike_kwargs={},
        test_inds=None,
        fill_values=None,
        fp=None,
        autocorr_iter_count=100,
        autocorr_multiplier=100,
        sampler_kwargs={},
    ):

        self.nwalkers, self.ndim, self.ndim_full = nwalkers, ndim, ndim_full

        betas = np.array([1.0, 0.1, 0.01, 0.0])

        self.lnprior = LogUniformPrior(prior_ranges)
        self.lnprob = LogProb(
            betas,
            ndim_full,
            lnprob,
            self.lnprior,
            lnlike_kwargs,
            subset=subset,
            test_inds=test_inds,
            fill_values=fill_values,
        )

        self.autocorr_iter_count = autocorr_iter_count
        self.autocorr_multiplier = autocorr_multiplier

        self.subset = subset

        if fp is None:
            backend = emcee.backends.Backend()

        else:
            backend = emcee.backends.HDFBackend(fp)
            backend.reset(nwalkers, ndim)

        # TODO: add block if nwalkers / betas is not okay
        pt_move = PTStretchMove(betas, int(nwalkers / len(betas)), ndim)
        self.sampler = emcee.EnsembleSampler(
            nwalkers,
            ndim,
            self.lnprob,
            vectorize=True,
            moves=pt_move,
            backend=backend,
            **sampler_kwargs
        )

    def sample(self, x0, max_iter, show_progress=False):

        # We'll track how the average autocorrelation time estimate changes
        index = 0
        autocorr = np.empty(max_iter)

        # This will be useful to testing convergence
        old_tau = np.inf

        st = time.perf_counter()
        # Now we'll sample for up to max_n steps
        for sample in self.sampler.sample(
            x0, iterations=max_iter, progress=show_progress
        ):
            # Only check convergence every 100 steps
            """
            if self.sampler.iteration % self.autocorr_iter_count:
                continue

            # Compute the autocorrelation time so far
            # Using tol=0 means that we'll always get an estimate even
            # if it isn't trustworthy

            # TODO: fix this for parallel tempering
            tau = self.sampler.get_autocorr_time(tol=0)
            autocorr[index] = np.mean(tau)
            index += 1

            # Check convergence
            converged = np.all(tau * self.autocorr_multiplier < self.sampler.iteration)
            converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)

            if converged:
                break
            old_tau = tau
            """
            pass

        et = time.perf_counter()

        duration = et - st

        print("timing:", duration)
