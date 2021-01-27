import time

import numpy as np

import emcee
from lisatools.sampling.utility import ModifiedHDFBackend
from lisatools.sampling.plot import PlotContainer

from ptemcee.sampler import default_beta_ladder

from lisatools.sampling.moves.ptredblue import PTStretchMove


def get_pt_autocorr_time(x, discard=0, thin=1, **kwargs):
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
    return thin * emcee.autocorr.integrated_time(x, **kwargs)


class LogPrior:
    def __init__(self, priors):
        self.priors = priors

    def __call__(self, x):

        prior_vals = np.zeros((x.shape[0]))
        for i, (prior_i, x_i) in enumerate(zip(self.priors, x.T)):
            temp = prior_i.logpdf(x_i)

            prior_vals[np.isinf(temp)] += -np.inf

        return prior_vals


class LogProb:
    def __init__(
        self,
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
        inds_eval = np.atleast_1d(np.squeeze(np.where(np.isinf(prior_vals) != True)))

        loglike_vals = np.full(x.shape[0], np.inf)

        if len(inds_eval) == 0:
            return np.array([-loglike_vals, prior_vals]).T

        # if self.lnlike.parameter_transforms is not None:
        #    self.lnlike.parameter_transforms.transform_inplace_parameters(
        #        x.T, self.test_inds
        #    )

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

        return np.array([-loglike_vals, prior_vals]).T


class PTEmceeSampler:
    def __init__(
        self,
        nwalkers,
        ndim,
        ndim_full,
        lnprob,
        priors,
        subset=None,
        lnlike_kwargs={},
        injection=None,
        test_inds=None,
        fill_values=None,
        fp=None,
        autocorr_iter_count=100,
        autocorr_multiplier=1000,  # higher in ptemcee
        plot_iterations=-1,
        plot_source=None,
        plot_kwargs={},
        betas=None,
        ntemps=None,
        ntemps_target_extra=0,
        Tmax=None,
        burn=None,
        periodic=None,
        sampler_kwargs={},
    ):

        self.nwalkers, self.ndim, self.ndim_full = nwalkers, ndim, ndim_full

        if betas is None:
            if ntemps == 1:
                betas = np.array([1.0])
            else:
                ntemps_ladder = ntemps - ntemps_target_extra
                betas = default_beta_ladder(ndim, ntemps=ntemps_ladder, Tmax=Tmax)
                if ntemps_target_extra > 0:
                    betas = np.concatenate([np.full(ntemps_target_extra, 1.0), betas])

        self.betas = betas
        self.ntemps = len(betas)
        self.ntemps_target = 1 + ntemps_target_extra
        self.all_walkers = self.nwalkers * len(betas)
        self.burn = burn

        self.lnprior = LogPrior(priors)
        self.lnprob = LogProb(
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

        self.plot_iterations = plot_iterations
        if fp is None:
            backend = emcee.backends.Backend()
            if plot_iterations > 0:
                raise NotImplementedError

        else:
            backend = ModifiedHDFBackend(fp)
            backend.reset(
                self.all_walkers,
                ndim,
                ntemps=self.ntemps,
                injection=injection,
                test_inds=test_inds,
            )

            if self.plot_iterations > 0:
                if plot_source is None:
                    raise ValueError("plot_in_run is True. plot_source must be given.")
                self.plot_gen = PlotContainer(fp, plot_source, **plot_kwargs)

        self.injection = injection
        self.test_inds = test_inds
        self.ndim = ndim

        # TODO: add block if nwalkers / betas is not okay
        pt_move = PTStretchMove(betas, nwalkers, ndim, periodic=periodic)
        self.sampler = emcee.EnsembleSampler(
            self.all_walkers,
            ndim,
            self.lnprob,
            vectorize=True,
            moves=pt_move,
            backend=backend,
            **sampler_kwargs
        )

    def sample(self, x0, iterations=10000, **sampler_kwargs):
        # We'll track how the average autocorrelation time estimate changes
        index = 0
        autocorr = np.empty(iterations)

        # This will be useful to testing convergence
        old_tau = np.inf

        st = time.perf_counter()

        if self.burn is not None:
            for sample in self.sampler.sample(
                x0, iterations=self.burn, store=False, **sampler_kwargs
            ):
                x0 = sample.coords
            print("Burn Finished")

        # Now we'll sample for up to max_n steps
        for sample in self.sampler.sample(x0, iterations=iterations, **sampler_kwargs):
            # Only check convergence every 100 steps

            if self.sampler.iteration % self.autocorr_iter_count and (
                self.sampler.iteration % self.plot_iterations
                or self.plot_iterations <= 0
            ):
                continue

            # Compute the autocorrelation time so far
            # Using tol=0 means that we'll always get an estimate even
            # if it isn't trustworthy
            tau = self.sampler.get_autocorr_time(tol=0)
            autocorr[index] = np.mean(tau)
            index += 1

            print(index, tau, tau * self.autocorr_multiplier, self.sampler.iteration)

            # Check convergence
            converged = np.all(tau * self.autocorr_multiplier < self.sampler.iteration)
            converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)

            if (
                self.sampler.iteration % self.plot_iterations == 0
                and self.plot_iterations > 0
            ):
                self.plot_gen.generate_corner()

            if converged:
                break
            old_tau = tau
            pass

        et = time.perf_counter()

        duration = et - st

        print("timing:", duration)
