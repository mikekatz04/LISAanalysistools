import numpy as np

import emcee


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
        ndim_full,
        lnlike,
        lnprior,
        lnlike_kwargs={},
        test_inds=None,
        fill_values=None,
    ):

        self.lnlike_kwargs = lnlike_kwargs
        self.lnlike = lnlike
        self.lnprior = lnprior

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

    def __call__(self, x):
        prior_vals = self.lnprior(x)
        inds_eval = np.where(np.isinf(prior_vals) != True)

        loglike_vals = np.full(x.shape[0], -np.inf)

        if self.need_to_fill:
            x_in = np.zeros((x.shape[0], self.ndim_full))
            x_in[:, self.test_inds] = x
            x_in[:, self.fill_inds] = self.fill_values[np.newaxis, :]

        else:
            x_in = x

        loglike_vals[inds_eval] = self.lnlike.get_ll(
            x_in[inds_eval], **self.lnlike_kwargs
        )

        return -loglike_vals


class EmceeSampler:
    def __init__(
        self,
        nwalkers,
        ndim,
        ndim_full,
        lnprob,
        prior_ranges,
        lnlike_kwargs={},
        test_inds=None,
        fill_values=None,
        fp=None,
    ):

        self.nwalkers, self.ndim, self.ndim_full = nwalkers, ndim, ndim_full

        self.lnprior = LogUniformPrior(prior_ranges)
        self.lnprob = LogProb(
            ndim_full,
            lnprob,
            self.lnprior,
            lnlike_kwargs,
            test_inds=test_inds,
            fill_values=fill_values,
        )

        if fp is None:
            backend = emcee.backends.Backend()

        else:
            backend = emcee.backends.HDFBackend(fp)
            backend.reset(nwalkers, ndim)

        self.sampler = emcee.EnsembleSampler(
            nwalkers, ndim, self.lnprob, vectorize=True, backend=backend
        )

    def sample(self, x0, max_iter, show_progress=False):

        # We'll track how the average autocorrelation time estimate changes
        # index = 0
        # autocorr = np.empty(max_n)

        # This will be useful to testing convergence
        # old_tau = np.inf

        # Now we'll sample for up to max_n steps
        for sample in self.sampler.sample(
            x0, iterations=max_iter, progress=show_progress
        ):
            # Only check convergence every 100 steps
            """
            if sampler.iteration % 100:
                continue

            # Compute the autocorrelation time so far
            # Using tol=0 means that we'll always get an estimate even
            # if it isn't trustworthy
            tau = sampler.get_autocorr_time(tol=0)
            autocorr[index] = np.mean(tau)
            index += 1

            # Check convergence
            converged = np.all(tau * 100 < sampler.iteration)
            converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
            if converged:
                break
            old_tau = tau
            """
            pass
