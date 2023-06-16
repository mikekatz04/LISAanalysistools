import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

from sklearn.mixture import GaussianMixture
import numpy as np
import matplotlib.pyplot as plt
# from eryn.moves.multipletry import logsumexp
from scipy.special import logsumexp
import h5py
import corner
import multiprocessing as mp

def compute_gaussian_likelihood(weights, means, covs, normalized_samples):
    
    _, num_components = weights.shape
    num_leaves, num_samples, ndim = normalized_samples.shape
    n = float(ndim)


    cov_inv = np.zeros_like(covs)
    dets = np.zeros(covs.shape[:2])
    for i in range(num_leaves):
        for j in range(num_components):
            cov_inv[i, j] = np.linalg.inv(covs[i, j])
            dets[i, j] = np.linalg.det(covs[i, j])
    prefactor = 1 / ((2 * np.pi) ** (n / 2) * dets ** (1/2))
    log_kernel_vals = (-1. / 2. * np.sum(np.matmul((normalized_samples[:, None, :, :] - means[:, :, None, :]), cov_inv) * (normalized_samples[:, None, :, :] - means[:, :, None, :]), axis=-1))

    log_normal_vals = np.log(weights[:, :, None]) + np.log(prefactor[:, :, None]) + log_kernel_vals
    
    log_r_ic = log_normal_vals - logsumexp(log_normal_vals, axis=1)[:, None, :]

    # update
    weights[:] = np.exp(logsumexp(log_r_ic, axis=-1)) / num_samples
    means[:] = np.exp(logsumexp(log_r_ic[:, :, :, None] + np.log(normalized_samples)[:, None, :, :], axis=2) - logsumexp(log_r_ic, axis=-1)[:, :, None])
    tmp = normalized_samples[:, None, :, :] - means[:, :, None, :]
    covs[:] = np.exp(logsumexp(log_r_ic[:, :, :, None, None] + np.log(tmp[:, :, :, None, :] * tmp[:, :, :, :, None]), axis=2) - logsumexp(log_r_ic, axis=-1)[:, :, None, None])
    
    breakpoint()

def fit_gaussian_mixture_model(n_components, samples):

    min_vals = samples.min(axis=1)
    max_vals = samples.max(axis=1)

    normalized_samples = (samples - min_vals[:, None, :]) / (max_vals - min_vals)[:, None, :]

    num_leaves, num_samples, ndim = normalized_samples.shape

    weights = np.tile(np.ones(n_components) / n_components, (num_leaves, 1))


    inds_leaves = np.repeat(np.arange(num_leaves), n_components)

    fix = np.ones(num_leaves, dtype=bool)
    inds_samples = np.zeros((num_leaves, n_components), dtype=int)
    while np.any(fix):
        inds_samples[fix] = np.random.randint(num_samples, size=(num_leaves * n_components)).reshape(num_leaves, n_components)[fix]
        if n_components > 1:
            fix = np.any(np.diff(np.sort(inds_samples, axis=-1), axis=-1) == 0, axis=-1)
        else:
            fix[:] = False

    inds_samples = inds_samples.flatten()
    means = normalized_samples[inds_leaves, inds_samples].reshape(num_leaves, n_components, ndim)
    covs = np.repeat(np.array([np.cov(normalized_samples[i].T) for i in range(num_leaves)])[:, None], n_components, axis=1)

    compute_gaussian_likelihood(weights, means, covs, normalized_samples)
    breakpoint()


def fit_each_leaf(samples):
    run = True
    min_bic = np.inf
    sample_mins = samples.min(axis=0)
    sample_maxs = samples.max(axis=0)
    samples[:] = ((samples - sample_mins) / (sample_maxs - sample_mins)) * 2 - 1
    for n_components in range(1, 20):
        if not run:
            continue
        #fit_gaussian_mixture_model(n_components, samples)
        #breakpoint()
        mixture = GaussianMixture(n_components=n_components, verbose=False, verbose_interval=2)

        mixture.fit(samples)
        test_bic = mixture.bic(samples)
        # print(n_components, test_bic)
        if test_bic < min_bic:
            min_bic = test_bic
            keep_mix = mixture
            keep_components = n_components
            
        else:
            run = False

            # print(leaf, n_components - 1, et - st)
        
    """if keep_components >= 9:
        new_samples = keep_mix.sample(n_samples=100000)[0]
        old_samples = samples
        fig = corner.corner(old_samples, hist_kwargs=dict(density=True, color="r"), color="r", plot_datapoints=False, plot_density=False)
        corner.corner(new_samples, hist_kwargs=dict(density=True, color="b"), color="b", plot_datapoints=False, plot_contours=True, plot_density=False, fig=fig)
        fig.savefig("mix_check.png")
        plt.close()
        breakpoint()"""
    # print(keep_components)
    return [keep_mix.weights_, keep_mix.means_, keep_mix.covariances_, np.array([np.linalg.inv(keep_mix.covariances_[i]) for i in range(len(keep_mix.weights_))]), np.array([np.linalg.det(keep_mix.covariances_[i]) for i in range(len(keep_mix.weights_))]), sample_mins, sample_maxs]


if __name__ == "__main__":
    
    start = 50
    end = 400
    index = 1000
    keep = np.array([0, 1, 2, 4, 6, 7])
    
    with h5py.File("search_find_individual_posteriors_3.h5", "r") as fp:
        num_iters, num_leaves, num_temps, num_walkers, _ = fp["samples"].shape
    
    ndim = len(keep)
    output = []
    import time
    st1 = time.perf_counter()
    change = 200
    start_ind = 4800
    for end_ind in range(change + start_ind, num_leaves, change):
        with h5py.File("search_find_individual_posteriors_3.h5", "r") as fp:
            samples_all = fp["samples"][start:, start_ind:end_ind, 0, :, keep].transpose(1, 0, 2, 3)
            
        args = []
        for leaf in range(change): 
            args.append((samples_all[leaf].reshape(-1, ndim),))

        with mp.Pool(10) as pool:
            gmm_info = pool.starmap(fit_each_leaf, args)
        
        """gmm_info = [None for tmp in args]
        
        for leaf, tmp in enumerate(args):
            gmm_info[leaf] = fit_each_leaf(*tmp)
            print(leaf)"""

        weights = [tmp[0] for tmp in gmm_info]
        means = [tmp[1] for tmp in gmm_info]
        covs = [tmp[2] for tmp in gmm_info]
        invcovs = [tmp[3] for tmp in gmm_info]
        dets = [tmp[4] for tmp in gmm_info]
        mins = [tmp[5] for tmp in gmm_info]
        maxs = [tmp[6] for tmp in gmm_info]

        output = [weights, means, covs, invcovs, dets, mins, maxs]
        
        import pickle
        with open(f"gmm_info_numbers_{start_ind}_to_{end_ind - 1}.pickle", "wb") as fp:
            pickle.dump(output, fp, pickle.HIGHEST_PROTOCOL)

        print(start_ind, end_ind -1)
        start_ind = end_ind