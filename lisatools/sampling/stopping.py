import numpy as np

from eryn.utils.stopping import Stopping


class SNRStopping(Stopping):
    def __init__(self, snr_limit=100.0, verbose=False):
        self.snr_limit = snr_limit
        self.verbose = verbose

    def __call__(self, iter, sample, sampler):

        ind = sampler.get_log_prob().argmax()

        log_best = sampler.get_log_prob().max()
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


class SearchConvergeStopping(Stopping):
    def __init__(self, n_iters=30, diff=1.0, verbose=False):
        self.n_iters = n_iters
        self.iters_consecutive = 0
        self.past_like_best = -np.inf
        self.diff = diff
        self.verbose = verbose

    def __call__(self, iter, sample, sampler):

        like_best = sampler.get_log_prob().max()

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
            return True

        else:
            return False
