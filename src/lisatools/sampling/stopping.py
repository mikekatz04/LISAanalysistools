"""Stopping criteria for ``eryn``-based LISA samplers."""

import time

import numpy as np
from eryn.utils.stopping import Stopping
from eryn.utils.utility import thermodynamic_integration_log_evidence


class SNRStopping(Stopping):
    """Stop sampling once the best in-chain SNR exceeds a threshold.

    The SNR is read from the first column of the sampler blobs.

    Args:
        snr_limit: SNR threshold above which sampling should be stopped.
        verbose: If ``True``, print the current best SNR and log-likelihood
            on each call.
    """

    def __init__(self, snr_limit=100.0, verbose=False):
        self.snr_limit = snr_limit
        self.verbose = verbose

    def __call__(self, iter, sample, sampler):
        """Return ``True`` once the maximum SNR seen exceeds ``self.snr_limit``.

        Args:
            iter: Current sampler iteration index (unused; kept for the
                ``eryn`` stopping API).
            sample: Latest sample (unused; kept for the ``eryn`` stopping API).
            sampler: Active ``eryn`` sampler from which to read log-likelihoods
                and blobs.

        Returns:
            ``True`` if the best SNR exceeds the threshold, ``False`` otherwise.
        """

        ind = sampler.get_log_like().argmax()

        log_best = sampler.get_log_like().max()
        snr_best = sampler.get_blobs()[:, :, :, 0].flatten()[ind]
        # d_h_best = sampler.get_blobs()[:, :, :, 1].flatten()[ind]
        # h_h_best = sampler.get_blobs()[:, :, :, 2].flatten()[ind]

        if self.verbose:
            print(
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
                print(
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
    seen.

    Args:
        n_iters: Number of consecutive non-improving calls required to stop.
        diff: Absolute log-likelihood threshold below which a change is
            considered insignificant.
        verbose: If ``True``, print convergence diagnostics on each call.
        start_iteration: Number of leading iterations to discard before
            evaluating the best log-likelihood.
    """

    def __init__(self, n_iters=30, diff=1.0, verbose=False, start_iteration=0):
        self.n_iters = n_iters
        self.iters_consecutive = 0
        self.past_like_best = -np.inf
        self.diff = diff
        self.verbose = verbose
        self.start_iteration = start_iteration

    def __call__(self, iter, sample, sampler):
        """Return ``True`` when the best log-likelihood has converged."""

        like_best = sampler.get_log_like(discard=self.start_iteration).max()

        if np.abs(like_best - self.past_like_best) < self.diff:
            self.iters_consecutive += 1

        else:
            self.iters_consecutive = 0
            self.past_like_best = like_best

        if self.verbose:
            print(
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


class GBBandLogLConvergeStopping(Stopping):
    """Per-frequency-band log-likelihood convergence criterion for GB searches.

    Splits the frequency axis at ``band_edges`` and tracks the best
    log-likelihood within each band independently. A band is marked converged
    when its best log-likelihood has not improved by more than ``diff`` for
    ``n_iters`` consecutive calls. Sampling stops once all bands are
    converged.

    Args:
        fd: 1D array of frequencies indexing the data; ``band_edges`` are
            located in this array via ``searchsorted``.
        band_edges: Array of frequency band edges (length ``num_bands + 1``).
        n_iters: Number of consecutive non-improving calls required to mark
            a band converged.
        diff: Log-likelihood improvement threshold.
        verbose: If ``True``, print per-call diagnostics.
        start_iteration: Number of leading iterations to discard.
    """

    def __init__(self, fd, band_edges, n_iters=30, diff=1.0, verbose=False, start_iteration=0):
        self.band_edge_inds = np.searchsorted(fd, band_edges, side="right") - 1
        self.num_bands = self.band_edge_inds.shape[0] - 1
        self.converged = np.zeros(self.num_bands, dtype=bool)
        self.iters_consecutive = np.zeros(self.num_bands, dtype=int)
        self.past_like_best = np.full(self.num_bands, -np.inf)
        self.n_iters = n_iters
        self.diff = diff
        self.verbose = verbose
        self.start_iteration = start_iteration

    def add_mgh(self, mgh):
        """Attach a multi-GPU data holder used to compute per-band likelihoods."""
        self.mgh = mgh

    def __call__(self, i, sample, sampler):
        """Update per-band convergence state and return ``True`` once all bands have converged."""

        ll_per_band = self.mgh.get_ll(band_edge_inds=self.band_edge_inds).max(axis=0)

        ll_movement = (ll_per_band - self.past_like_best) > self.diff

        self.iters_consecutive[~ll_movement] += 1
        self.iters_consecutive[ll_movement] = 0

        self.converged = self.iters_consecutive >= self.n_iters

        self.past_like_best[ll_movement] = ll_per_band[ll_movement]

        # for move in sampler.all_moves:
        #     move.converged_sub_bands = self.converged.copy()

        if self.verbose:
            print(
                "Num still going:",
                (~self.converged).sum(),
                "\nChanged here:",
                (ll_movement).sum(),
            )

        if np.all(self.converged):
            return True
        else:
            return False


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
            print(
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
        """Compute and print the current log-evidence; always returns ``False``."""

        betas = sampler.get_betas()[-1]
        logls = sampler.get_log_like().mean(axis=(0, 2))

        logZ, dlogZ = thermodynamic_integration_log_evidence(betas, logls)
        print(logZ, dlogZ)
        return False

        if self.verbose:
            print(
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
