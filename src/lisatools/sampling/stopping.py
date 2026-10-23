"""Stopping criteria for ``eryn``-based LISA samplers.

Each class is an ``eryn`` stopping hook: pass an instance as
``stopping_fn=`` to :class:`~eryn.ensemble.EnsembleSampler` and
:meth:`~eryn.ensemble.EnsembleSampler.run_mcmc` calls it as
``stopping_fn(i, state, sampler)`` every ``stopping_iterations`` stored steps;
a ``True`` return ends the run.

Live classes: :class:`SNRStopping` (stop once the SNR blob of the best sample
clears a limit; LATW tutorial 07) and :class:`SearchConvergeStopping` (stop
when the best log-likelihood plateaus; used by the dev MBH search
:mod:`lisatools.globalfit.mbhsearch` and the legacy
:mod:`lisatools.globalfit.moves.mbhspecialmove`). The 9mo / 12mo global-fit
runs use none of them (their stages stop through the recipe's own stopping
functions, and production MBH runs ``MBH_LIKELIHOOD=batched``, not the legacy
MBH move). The other classes have no callers.
"""

import time

import numpy as np
from eryn.utils.stopping import Stopping
from eryn.utils.utility import thermodynamic_integration_log_evidence

import logging

logger = logging.getLogger(__name__)


class SNRStopping(Stopping):
    """Stop sampling once the SNR of the best sample exceeds a threshold.

    At each call the stored sample with the highest log-likelihood (over all
    stored steps, temperatures and walkers) is found, and its SNR is read
    from the first column of the sampler blobs, so the likelihood must
    return the SNR as blob 0.

    Args:
        snr_limit (float, optional): SNR threshold above which sampling should
            be stopped. Default ``100.0``.
        verbose (bool, optional): If ``True``, log the best sample's SNR and
            log-likelihood on each call. Default ``False``.

    Example:
        >>> from lisatools.sampling.stopping import SNRStopping
        >>> stop = SNRStopping(snr_limit=12.0)
        >>> stop.snr_limit
        12.0

        Then pass it to the sampler:

        .. code-block:: python

            sampler = EnsembleSampler(..., stopping_fn=stop, stopping_iterations=10)
    """

    def __init__(self, snr_limit=100.0, verbose=False):
        self.snr_limit = snr_limit
        self.verbose = verbose

    def __call__(self, iter, sample, sampler):
        """Return ``True`` once the best sample's SNR exceeds ``self.snr_limit``.

        Args:
            iter (int): Current sampler iteration index (unused; kept for the
                ``eryn`` stopping API).
            sample (eryn.state.State): Latest sample (unused; kept for the
                ``eryn`` stopping API).
            sampler (eryn.ensemble.EnsembleSampler): Active sampler from which
                to read log-likelihoods and blobs (blob 0 = SNR).

        Returns:
            bool: ``True`` if the SNR at the maximum log-likelihood sample
            exceeds the threshold, ``False`` otherwise.
        """

        ind = sampler.get_log_like().argmax()

        log_best = sampler.get_log_like().max()
        snr_best = sampler.get_blobs()[:, :, :, 0].flatten()[ind]
        # d_h_best = sampler.get_blobs()[:, :, :, 1].flatten()[ind]
        # h_h_best = sampler.get_blobs()[:, :, :, 2].flatten()[ind]

        if self.verbose:
            logger.info(
                "snr_best",
                snr_best,
                "limit:",
                self.snr_limit,
                "loglike:",
                log_best,
                # d_h_best,
                # h_h_best,
            )

        if snr_best > self.snr_limit:
            return True

        else:
            return False


# TODO check: keep/remove/document -- no callers (src, tests, scripts, LATW); LATW 07: unused
class NLeavesSearchStopping:
    """Stop a search once the maximum leaf count stops growing.

    Compares the maximum number of active ``"gb"`` leaves across the most
    recent ``convergence_iter`` iterations to the maximum across all earlier
    iterations and stops if the recent window did not grow.

    Args:
        convergence_iter: Number of trailing iterations used to define the
            "recent" window.
        verbose: If ``True``, print diagnostics on each call.
    """

    def __init__(self, convergence_iter=5, verbose=False):
        self.convergence_iter = convergence_iter
        self.verbose = verbose

    def __call__(self, i, sample, sampler):
        """Return ``True`` if the leaf count has plateaued.

        Args:
            i: Current iteration index (unused).
            sample: Latest sample (unused).
            sampler: Active ``eryn`` sampler whose backend exposes
                ``get_nleaves`` and ``iteration``.

        Returns:
            ``True`` once recent maxima no longer exceed earlier maxima.
        """

        if not hasattr(self, "st"):
            self.st = time.perf_counter()

        current_iter = sampler.backend.iteration

        stop = False
        if current_iter > self.convergence_iter:

            nleaves_cc = sampler.backend.get_nleaves()["gb"][:, 0]

            # do not include most recent
            nleaves_cc_max_old = nleaves_cc[: -self.convergence_iter].max()
            nleaves_cc_max_new = nleaves_cc[-self.convergence_iter :].max()

            if nleaves_cc_max_old >= nleaves_cc_max_new:
                stop = True

            else:
                stop = False

            if self.verbose:
                dur = (time.perf_counter() - self.st) / 3600.0  # hours
                logger.info(
                    "\nnleaves max old:\n",
                    nleaves_cc_max_old,
                    "\nnleaves max new:\n",
                    nleaves_cc_max_new,
                    f"\nTIME TO NOW: {dur} hours",
                )

        return stop


class SearchConvergeStopping(Stopping):
    """Stop when the best log-likelihood has not improved for ``n_iters`` calls.

    Tracks the best log-likelihood seen so far and increments a consecutive
    counter every time the new best differs from the previous best by less
    than ``diff``. The counter is reset whenever a meaningful improvement is
    seen. When the counter reaches ``n_iters`` the call returns ``True`` and
    the counter is reset to 0 (so the instance can be reused for the next
    search). ``start_iteration`` is only the number of stored steps
    DISCARDED before the maximum is taken; unlike
    :class:`eryn.utils.stopping.SearchConvergeStopping` (which also returns
    ``False`` for calls with ``iter < start_iteration``), every call is
    evaluated.

    Args:
        n_iters (int, optional): Number of consecutive non-improving calls
            required to stop. Default ``30``.
        diff (float, optional): Absolute log-likelihood threshold below which
            a change is considered insignificant. Default ``1.0``.
        verbose (bool, optional): If ``True``, log convergence diagnostics on
            each call. Default ``False``.
        start_iteration (int, optional): Number of leading stored steps to
            discard before evaluating the best log-likelihood. Default ``0``.

    Example:
        >>> from lisatools.sampling.stopping import SearchConvergeStopping
        >>> stop = SearchConvergeStopping(n_iters=30, diff=1.0, start_iteration=0)
        >>> stop.iters_consecutive
        0
    """

    def __init__(self, n_iters=30, diff=1.0, verbose=False, start_iteration=0):
        self.n_iters = n_iters
        self.iters_consecutive = 0
        self.past_like_best = -np.inf
        self.diff = diff
        self.verbose = verbose
        self.start_iteration = start_iteration

    def __call__(self, iter, sample, sampler):
        """Return ``True`` when the best log-likelihood has converged.

        Args:
            iter (int): Sampler iteration index (unused).
            sample (eryn.state.State): Latest sample (unused).
            sampler (eryn.ensemble.EnsembleSampler): Sampler whose stored
                log-likelihoods are read.

        Returns:
            bool: ``True`` after ``n_iters`` consecutive calls without a change
            of at least ``diff`` in the best log-likelihood.
        """

        like_best = sampler.get_log_like(discard=self.start_iteration).max()

        if np.abs(like_best - self.past_like_best) < self.diff:
            self.iters_consecutive += 1

        else:
            self.iters_consecutive = 0
            self.past_like_best = like_best

        if self.verbose:
            logger.info(
                "\nITERS CONSECUTIVE:\n",
                self.iters_consecutive,
                self.past_like_best,
                like_best,
            )

        if self.iters_consecutive >= self.n_iters:
            self.iters_consecutive = 0
            return True

        else:
            return False


# GBBandLogLConvergeStopping was removed with the legacy
# MultiGPUDataHolder it consumed (parallel-resources plan P4).


# TODO check: keep/remove/document -- no callers (src, tests, scripts, LATW); LATW 07: unused
class SearchConvergeStopping2(Stopping):
    """Variant of :class:`SearchConvergeStopping` with a circular look-back buffer.

    In addition to the running best-log-likelihood check used by
    :class:`SearchConvergeStopping`, this variant keeps the most recent
    ``iter_back_check`` best log-likelihoods in a circular buffer and uses
    their spread as a secondary convergence test.

    Args:
        n_iters: Number of consecutive non-improving calls required to stop.
        diff: Log-likelihood improvement threshold.
        verbose: If ``True``, print diagnostics on each call.
        start_iteration: Number of leading iterations to discard.
        iter_back_check: Length of the circular look-back buffer used for the
            spread-based convergence check.
    """

    def __init__(self, n_iters=30, diff=0.1, verbose=False, start_iteration=0, iter_back_check=-1):
        self.n_iters = n_iters
        self.iters_consecutive = 0
        self.past_like_best = -np.inf
        self.diff = diff
        self.verbose = verbose
        self.start_iteration = start_iteration
        self.iter_back_check = iter_back_check
        self.time = 0
        self.back_check = [None for _ in range(self.iter_back_check)]
        self.last_sampler_iteration = self.start_iteration
        self.back_check_ind = 0
        self.stop_here = True

    def __call__(self, iter, sample, sampler):
        """Return ``True`` once the chain has plateaued under both checks."""

        self.time += 1

        if sampler.iteration <= self.start_iteration:
            return False

        lps = sampler.get_log_like(discard=self.start_iteration)[
            self.last_sampler_iteration - self.start_iteration :
        ]
        try:
            like_best = lps.max()
        except:
            breakpoint()
        self.last_sampler_iteration = sampler.iteration

        if np.any(np.asarray(self.back_check) == None):
            for i in range(len(self.back_check)):
                if self.back_check[i] is None:
                    self.back_check[i] = like_best
                    return False

        first_check = like_best - self.past_like_best > self.diff
        second_check = np.all(like_best >= np.asarray(self.back_check))

        # spread in stored values is below difference
        third_check = (
            np.asarray(self.back_check).max() - np.asarray(self.back_check).min() < self.diff
        )

        update = (
            (first_check and second_check and self.past_like_best == -np.inf)
            or (self.past_like_best == -np.inf and third_check)
            or (self.past_like_best > -np.inf and first_check)
        )

        self.back_check[self.back_check_ind] = like_best
        self.back_check_ind = (self.back_check_ind + 1) % len(self.back_check)

        if update:
            self.past_like_best = like_best
            self.iters_consecutive = 0

        elif self.past_like_best > -np.inf:
            self.iters_consecutive += 1

        if self.verbose:
            logger.info(
                "\nITERS CONSECUTIVE:\n",
                self.iters_consecutive,
                f"previous best: {self.past_like_best}, overall best: {like_best},",
                "first check:",
                first_check,
                "second check:",
                second_check,
            )

        if self.iters_consecutive >= self.n_iters:
            self.iters_consecutive = 0
            return True

        else:
            return False


# TODO check: keep/remove/document -- no callers; work in progress (always returns False)
class EvidenceStopping(Stopping):
    """Stopping criterion based on the parallel-tempering log-evidence estimate.

    Uses :func:`eryn.utils.utility.thermodynamic_integration_log_evidence` to
    estimate the evidence each call. Currently a work in progress: the call
    method computes the evidence and returns ``False``.

    Args:
        diff: Log-evidence change threshold (intended for a future
            implementation of the actual stopping check).
        verbose: If ``True``, print diagnostics on each call.
    """

    # TODO/DOCS: EvidenceStopping.__call__ currently always returns False
    # (the threshold-based stop logic is unreachable below the early return).
    # Documenting current behavior; intent appears to be evidence-change-based
    # stopping using ``self.diff``.
    def __init__(self, diff=0.5, verbose=False):
        self.diff = diff
        self.verbose = verbose

    def __call__(self, iter, sample, sampler):
        """Compute and log the current log-evidence; always returns ``False``."""

        betas = sampler.get_betas()[-1]
        logls = sampler.get_log_like().mean(axis=(0, 2))

        logZ, dlogZ = thermodynamic_integration_log_evidence(betas, logls)
        logger.info(logZ, dlogZ)
        return False

        if self.verbose:
            logger.info(
                "snr_best",
                snr_best,
                "limit:",
                self.snr_limit,
                "loglike:",
                log_best,
                # d_h_best,
                # h_h_best,
            )

        if snr_best > self.snr_limit:
            return True

        else:
            return False


# TODO check: keep/remove/document -- no callers (src, tests, scripts, LATW); LATW 07: unused
class MPICommunicateStopping(Stopping):
    """Stopping wrapper that broadcasts a stop decision over MPI.

    On the rank designated by ``stopper_rank``, ``stop_fn`` is evaluated and,
    if it returns ``True``, a stop signal is sent to every rank in
    ``other_ranks``. Other ranks poll for that signal each call.

    Args:
        stopper_rank: MPI rank that decides whether to stop.
        other_ranks: List of ranks to notify when stopping.
        stop_fn: Callable evaluated on the stopper rank; must accept the
            same ``*args, **kwargs`` as :meth:`__call__`.
    """

    def __init__(self, stopper_rank, other_ranks, stop_fn=None):

        self.stopper_rank = stopper_rank
        self.other_ranks = other_ranks
        self.stop_fn = stop_fn

    def add_comm(self, comm):
        """Attach the MPI communicator used for send / receive operations."""
        self.comm = comm

    def __call__(self, *args, **kwargs):
        """Evaluate or receive the stop decision depending on the local MPI rank."""

        if not hasattr(self, "comm"):
            raise ValueError("Must add comm via add_comm method before __call__ is used.")

        if not hasattr(self, "rank"):
            self.rank = self.comm.Get_rank()
            if not self.rank == self.stopper_rank and not self.rank in self.other_ranks:
                raise ValueError(
                    f"Rank ({self.rank}) is not available in other ranks list ({self.other_ranks}). Must be either stopper rank ({self.stopper_rank}) or in other ranks list."
                )

            if self.stopper_rank == self.rank and self.stop_fn is None:
                raise ValueError(
                    "Rank is equivalent to stopper rank but stop_fn is not provided. It must be provided."
                )

        if self.rank == self.stopper_rank:
            stop = self.stop_fn(*args, **kwargs)

            if stop:
                for rank in self.other_ranks:
                    tag = int(str(rank) + "1000")
                    self.comm.isend(True, dest=rank, tag=tag)

        else:
            tag = int(str(self.rank) + "1000")
            check_stop = self.comm.irecv(source=self.stopper_rank, tag=tag)

            if check_stop.get_status():
                stop = check_stop.wait()

            else:
                check_stop.cancel()
                stop = False

        return stop
