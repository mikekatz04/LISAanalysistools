import numpy as np

from eryn.utils.stopping import Stopping
from eryn.utils.utility import thermodynamic_integration_log_evidence


class SNRStopping(Stopping):
    def __init__(self, snr_limit=100.0, verbose=False):
        self.snr_limit = snr_limit
        self.verbose = verbose

    def __call__(self, iter, sample, sampler):

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


class SearchConvergeStopping(Stopping):
    def __init__(self, n_iters=30, diff=1.0, verbose=False, start_iteration=0):
        self.n_iters = n_iters
        self.iters_consecutive = 0
        self.past_like_best = -np.inf
        self.diff = diff
        self.verbose = verbose
        self.start_iteration = start_iteration

    def __call__(self, iter, sample, sampler):

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



class SearchConvergeStopping2(Stopping):
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

        self.time += 1

        if sampler.iteration <= self.start_iteration:
            return False

        lps = sampler.get_log_like(discard=self.start_iteration)[self.last_sampler_iteration - self.start_iteration:]
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
        third_check = np.asarray(self.back_check).max() - np.asarray(self.back_check).min() < self.diff

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
                "first check:", first_check,
                "second check:", second_check
            )

        if self.iters_consecutive >= self.n_iters:
            self.iters_consecutive = 0
            return True

        else:
            return False



class EvidenceStopping(Stopping):
    def __init__(self, diff=0.5, verbose=False):
        self.diff = diff
        self.verbose = verbose

    def __call__(self, iter, sample, sampler):

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



        
