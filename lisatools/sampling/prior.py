import numpy as np
from scipy import stats

from ..utils.constants import *
from ..sensitivity import get_sensitivity
from eryn.moves.multipletry import logsumexp

try:
    import cupy as cp

except (ModuleNotFoundError, ImportError) as e:
    pass


class SNRPrior:
    def __init__(self, rho_star, use_cupy=False):
        self.rho_star = rho_star
        self.use_cupy = use_cupy

    def pdf(self, rho):
        
        xp = np if not self.use_cupy else cp

        p = xp.zeros_like(rho)
        good = rho > 0.0
        p[good] = 3 * rho[good] / (4 * self.rho_star ** 2 * (1 + rho[good] / (4 * self.rho_star)) ** 5)
        return p

    def logpdf(self, rho):
        xp = np if not self.use_cupy else cp
        return xp.log(self.pdf(rho))

    def cdf(self, rho):
        xp = np if not self.use_cupy else cp
        c = xp.zeros_like(rho)
        good = rho > 0.0
        c[good] = 768 * self.rho_star ** 3 * (1 / (768. * self.rho_star ** 3) - (rho[good] + self.rho_star)/(3. * (rho[good] + 4 * self.rho_star) ** 4))
        return c

    def rvs(self, size=1):
        if isinstance(size, int):
            size = (size,)

        xp = np if not self.use_cupy else cp

        u = xp.random.rand(*size)

        rho = (-4*self.rho_star + xp.sqrt(-32*self.rho_star**2 - (32*(-self.rho_star**2 + u*self.rho_star**2))/(1 - u) + 
      (3072*2**0.3333333333333333*xp.cbrt(-1 + 3*u - 3*u**2 + u**3)*
         (self.rho_star**4 - u*self.rho_star**4))/
       ((-1 + u)**2*xp.cbrt(-1769472*self.rho_star**6 + 1769472*u*self.rho_star**6 - 
            xp.sqrt(3131031158784*u*self.rho_star**12 - 6262062317568*u**2*self.rho_star**12 + 
              3131031158784*u**3*self.rho_star**12))) + 
      xp.cbrt(-1769472*self.rho_star**6 + 1769472*u*self.rho_star**6 - 
          xp.sqrt(3131031158784*u*self.rho_star**12 - 6262062317568*u**2*self.rho_star**12 + 
            3131031158784*u**3*self.rho_star**12))/
       (3.*2**0.3333333333333333*xp.cbrt(-1 + 3*u - 3*u**2 + u**3)))/2.
     + xp.sqrt(32*self.rho_star**2 + (32*(-self.rho_star**2 + u*self.rho_star**2))/(1 - u) - 
      (3072*2**0.3333333333333333*xp.cbrt(-1 + 3*u - 3*u**2 + u**3)*
         (self.rho_star**4 - u*self.rho_star**4))/
       ((-1 + u)**2*xp.cbrt(-1769472*self.rho_star**6 + 1769472*u*self.rho_star**6 - 
            xp.sqrt(3131031158784*u*self.rho_star**12 - 6262062317568*u**2*self.rho_star**12 + 
              3131031158784*u**3*self.rho_star**12))) - 
      xp.cbrt(-1769472*self.rho_star**6 + 1769472*u*self.rho_star**6 - 
          xp.sqrt(3131031158784*u*self.rho_star**12 - 6262062317568*u**2*self.rho_star**12 + 
            3131031158784*u**3*self.rho_star**12))/
       (3.*2**0.3333333333333333*xp.cbrt(-1 + 3*u - 3*u**2 + u**3)) + 
      (2048*self.rho_star**3 - (2048*u*self.rho_star**3)/(-1 + u))/
       (4.*xp.sqrt(-32*self.rho_star**2 - (32*(-self.rho_star**2 + u*self.rho_star**2))/(1 - u) + 
           (3072*2**0.3333333333333333*
              xp.cbrt(-1 + 3*u - 3*u**2 + u**3)*(self.rho_star**4 - u*self.rho_star**4)
              )/
            ((-1 + u)**2*xp.cbrt(-1769472*self.rho_star**6 + 1769472*u*self.rho_star**6 - 
                 xp.sqrt(3131031158784*u*self.rho_star**12 - 6262062317568*u**2*self.rho_star**12 + 
                   3131031158784*u**3*self.rho_star**12))) + 
           xp.cbrt(-1769472*self.rho_star**6 + 1769472*u*self.rho_star**6 - 
               xp.sqrt(3131031158784*u*self.rho_star**12 - 6262062317568*u**2*self.rho_star**12 + 
                 3131031158784*u**3*self.rho_star**12))/
            (3.*2**0.3333333333333333*
              xp.cbrt(-1 + 3*u - 3*u**2 + u**3)))))/2.)

        return rho


class AmplitudeFromSNR:
    def __init__(self, L, Tobs, **noise_kwargs):
        self.f_star = 1 / (2. * np.pi * L) * C_SI
        self.Tobs = Tobs
        self.noise_kwargs = noise_kwargs

    def __call__(self, rho, f0):
        factor = 1./2. * np.sqrt((self.Tobs * np.sin(f0 / self.f_star) ** 2) / get_sensitivity(f0, sens_fn="noisepsd_AE", **self.noise_kwargs))
        amp = rho / factor
        return (amp, f0)

    def forward(self, amp, f0):
        factor = 1./2. * np.sqrt((self.Tobs * np.sin(f0 / self.f_star) ** 2) / get_sensitivity(f0, sens_fn="noisepsd_AE", **self.noise_kwargs))
        rho = amp * factor
        return (rho, f0)

class FullGaussianMixtureModel:
    def __init__(self, gb, weights, means, covs, invcovs, dets, mins, maxs, limit=10.0, use_cupy=False):
        
        self.use_cupy = use_cupy
        if use_cupy:
            xp = cp
        else:
            xp = np

        self.gb = gb

        indexing = []
        for i, weight in enumerate(weights):
            index_base = np.full_like(weight, i, dtype=int)
            indexing.append(index_base)
        
        self.indexing = xp.asarray(np.concatenate(indexing))
        # invidivual weights / total number of components to uniformly choose from them
        self.weights = xp.asarray(np.concatenate(weights, axis=0) * 1 / len(weights))

        assert xp.allclose(self.weights.sum(), 1.0)
        self.means = xp.asarray(np.concatenate(means, axis=0))
        self.covs = xp.asarray(np.concatenate(covs, axis=0))
        self.invcovs = xp.asarray(np.concatenate(invcovs, axis=0))
        self.dets = xp.asarray(np.concatenate(dets, axis=0))
        self.ndim = self.means.shape[1]

        self.mins = xp.asarray(np.vstack(mins))
        self.maxs = xp.asarray(np.vstack(maxs))

        self.mins_in_pdf = self.mins[self.indexing].T.flatten().copy()
        self.maxs_in_pdf = self.maxs[self.indexing].T.flatten().copy()
        self.means_in_pdf = self.means.T.flatten().copy()
        self.invcovs_in_pdf = self.invcovs.transpose(1, 2, 0).flatten().copy()

        self.cumulative_weights = xp.concatenate([xp.array([0.0]), xp.cumsum(self.weights)])

        self.min_limit_f = self.map_back_frequency(-1. * limit, self.mins[self.indexing, 1], self.maxs[self.indexing, 1]) 
        self.max_limit_f = self.map_back_frequency(+1. * limit, self.mins[self.indexing, 1], self.maxs[self.indexing, 1]) 

        # compute the jacobian
        self.log_det_J = (self.ndim * np.log(2) - xp.sum(xp.log(self.maxs - self.mins), axis=-1))[self.indexing].copy()

        """self.inds_sort_min_limit_f = xp.argsort(self.min_limit_f)
        self.inds_sort_max_limit_f = xp.argsort(self.max_limit_f)
        self.sorted_min_limit_f = self.min_limit_f[self.inds_sort_min_limit_f]
        self.sorted_max_limit_f = self.max_limit_f[self.inds_sort_max_limit_f]
        """
    def logpdf(self, x):

        if self.use_cupy:
            xp = cp
        else:
            xp = np

        assert len(x.shape) == 2
        assert x.shape[1] == self.ndim

        k = self.ndim

        inds_sort = xp.argsort(x[:, 1])
        f_sort = x[:, 1][inds_sort]
        points_sorted = x[inds_sort]

        ind_min_limit = xp.searchsorted(f_sort, self.min_limit_f, side="left")
        ind_max_limit = xp.searchsorted(f_sort, self.max_limit_f, side="right")

        diff = (ind_max_limit - ind_min_limit)
        cs = xp.concatenate([xp.array([0]), xp.cumsum(diff)])
        tmp = xp.arange(cs[-1])
        keep_component_map = xp.searchsorted(cs, tmp, side="right") - 1
        keep_point_map = tmp - cs[keep_component_map] + ind_min_limit[keep_component_map]
        max_components = diff.max().item()
        
        int_check = int(1e6)
        assert int_check > self.min_limit_f.shape[0]
        special_point_component_map = int_check * keep_point_map + keep_component_map

        sorted_special = xp.sort(special_point_component_map)

        points_keep_in = (sorted_special / float(int_check)).astype(int)
        components_keep_in = sorted_special - points_keep_in * int_check

        unique_points, unique_starts = xp.unique(points_keep_in, return_index=True)
        start_index_in_pdf = xp.concatenate([unique_starts, xp.array([len(points_keep_in)])]).astype(xp.int32)
        assert xp.all(xp.diff(unique_starts) > 0)
        
        points_sorted_in = points_sorted[unique_points]

        logpdf_out_tmp = xp.zeros(points_sorted_in.shape[0])

        self.gb.compute_logpdf(logpdf_out_tmp, components_keep_in.astype(xp.int32), points_sorted_in,
                    self.weights, self.mins_in_pdf, self.maxs_in_pdf, self.means_in_pdf, self.invcovs_in_pdf, self.dets, self.log_det_J, 
                    points_sorted_in.shape[0], start_index_in_pdf, self.weights.shape[0], x.shape[1])

        # need to reverse the sort
        logpdf_out = xp.full(x.shape[0], -xp.inf)
        logpdf_out[xp.sort(inds_sort[unique_points])] = logpdf_out_tmp[xp.argsort(inds_sort[unique_points])]
        return logpdf_out
        """# breakpoint()

        # map to reduced domain
        x_mapped = (self.map_input(points_sorted[:, None, :], self.mins[None, :, :], self.maxs[None, :, :]))[:, self.indexing]
        
        diff = x_mapped - self.means[None, :, :]
        log_main_part = -1./2. * xp.einsum("...k,...k", diff, xp.einsum("...jk,...k->...j", self.invcovs, diff))
        log_norm_factor = (k / 2) * xp.log(2 * np.pi) + (1 / 2) * xp.log(self.dets)
        log_weighted_pdf = (xp.log(self.weights) + log_norm_factor)[None, :] + log_main_part
        
        logpdf_full_dist_tmp = logsumexp(log_weighted_pdf, axis=-1, xp=xp)
        logpdf_full_dist = logpdf_full_dist_tmp[xp.argsort(inds_sort)]

        breakpoint()
        assert xp.allclose(logpdf_full_dist, logpdf_out)

        return logpdf_full_dist"""

    def map_input(self, x, mins, maxs):
        return ((x - mins) / (maxs - mins)) * 2. - 1.

    def map_back_frequency(self, x, mins, maxs):
        return (x + 1.) * 1. / 2. * (maxs - mins) + mins
        
    def rvs(self, size=(1,)):

        if isinstance(size, int):
            size = (size,)

        if self.use_cupy:
            xp = cp
        else:
            xp = np

        # choose which component
        draw = xp.random.rand(*size)
        component = (xp.searchsorted(self.cumulative_weights, draw.flatten(), side="right") - 1).reshape(draw.shape)

        mean_here = self.means[component]
        cov_here = self.covs[component]

        new_points = mean_here + xp.einsum("...kj,...j->...k", cov_here, np.random.randn(*(component.shape + (self.ndim,))))

        index_here = self.indexing[component]
        mins_here = self.mins[index_here]
        maxs_here = self.maxs[index_here]
        new_points_mapped = self.map_back_frequency(new_points, mins_here, maxs_here)
        
        return new_points_mapped
        

        
        