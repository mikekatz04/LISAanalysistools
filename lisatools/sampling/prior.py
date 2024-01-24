import numpy as np
from scipy import stats

from ..utils.constants import *
from ..sensitivity import get_sensitivity
from eryn.moves.multipletry import logsumexp

from typing import Union, Optional, Tuple, List

import sys
sys.path.append("/data/mkatz/LISAanalysistools/lisaflow/flow/experiments/rvs/gf_search/")
# from galaxy_ffdot import GalaxyFFdot
# from galaxy import Galaxy

try:
    import cupy as cp

except (ModuleNotFoundError, ImportError) as e:
    pass

class AmplitudeFrequencySNRPrior:
    def __init__(self, rho_star, frequency_prior, L, Tobs, use_cupy=False, **noise_kwargs):
        self.rho_star = rho_star
        self.frequency_prior = frequency_prior

        self.transform = AmplitudeFromSNR(L, Tobs, use_cupy=use_cupy, **noise_kwargs)
        self.snr_prior = SNRPrior(rho_star, use_cupy=use_cupy)
        
        # must be after transform and snr_prior due to setter
        self.use_cupy = use_cupy
        
    @property
    def use_cupy(self):
        return self._use_cupy

    @use_cupy.setter
    def use_cupy(self, use_cupy):
        self._use_cupy = use_cupy
        self.transform.use_cupy = use_cupy
        self.snr_prior.use_cupy = use_cupy
        self.frequency_prior.use_cupy = use_cupy

    def pdf(self, *args, **noise_kwargs):
        return np.exp(self.logpdf(*args, **noise_kwargs))

    def logpdf(self, amp, f0_ms, **noise_kwargs):

        xp = np if not self.use_cupy else cp

        f0 = f0_ms / 1e3
        rho, f0 = self.transform.forward(amp, f0, **noise_kwargs)

        rho_pdf = self.snr_prior.pdf(rho)

        Jac = xp.abs(rho / amp)

        logpdf_amp = np.log(np.abs(Jac * rho_pdf))
        logpdf_f = self.frequency_prior.logpdf(f0_ms)

        return logpdf_amp + logpdf_f

    def rvs(self, size=1, f0_input=None, **noise_kwargs):
        if isinstance(size, int):
            size = (size,)

        xp = np if not self.use_cupy else cp

        if f0_input is None:
            f0_ms = self.frequency_prior.rvs(size=size)
        else:
            f0_ms = f0_input
            assert f0_input.shape[:-1] == size
        
        f0 = f0_ms / 1e3

        rho = self.snr_prior.rvs(size=size)

        amp, _ = self.transform(rho, f0, **noise_kwargs)

        return (amp, f0_ms)






class SNRPrior:
    def __init__(self, rho_star, use_cupy=False):
        self.rho_star = rho_star
        self.use_cupy = use_cupy

    @property
    def use_cupy(self):
        return self._use_cupy

    @use_cupy.setter
    def use_cupy(self, use_cupy):
        self._use_cupy = use_cupy

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
    def __init__(self, L, Tobs, fd=None, use_cupy=False, **noise_kwargs):
        self.f_star = 1 / (2. * np.pi * L) * C_SI
        self.Tobs = Tobs
        self.noise_kwargs = noise_kwargs
        
        xp = np if not use_cupy else cp
        if fd is not None:
            self.fd = xp.asarray(fd)
        else:
            self.fd = fd

        # got to be after fd
        self.use_cupy = use_cupy

    @property
    def use_cupy(self):
        return self._use_cupy

    @use_cupy.setter
    def use_cupy(self, use_cupy):
        self._use_cupy = use_cupy
        if use_cupy and not isinstance(self.fd, cp.ndarray):
            self.fd = cp.asarray(self.fd)
        elif not use_cupy and isinstance(self.fd, cp.ndarray):
            self.fd = self.fd.get()

    def interp_psd(self, f0, psds, walker_inds=None):
        assert self.fd is not None
        xp = np if not self.use_cupy else cp
        psds = xp.atleast_2d(psds)
        
        if xp == cp and not isinstance(self.fd, cp.ndarray):
            self.fd = xp.asarray(self.fd)
        try:
            inds_fd = xp.searchsorted(self.fd, f0, side="right") - 1
        except:
            breakpoint()
        if walker_inds is None:
            walker_inds = xp.zeros_like(f0, dtype=int)

        new_psds = (psds[(walker_inds, inds_fd + 1)] - psds[(walker_inds, inds_fd)]) / (self.fd[inds_fd + 1] - self.fd[inds_fd]) * (f0 - self.fd[inds_fd]) + psds[(walker_inds, inds_fd)]
        return new_psds

    def __call__(self, rho, f0, **noise_kwargs):

        xp = np if not self.use_cupy else cp

        if noise_kwargs == {}:
            noise_kwargs = self.noise_kwargs

        Sn_f = self.get_Sn_f(f0, **noise_kwargs)

        factor = 1./2. * np.sqrt((self.Tobs * np.sin(f0 / self.f_star) ** 2) / Sn_f)
        amp = rho / factor
        return (amp, f0)

    def get_Sn_f(self, f0, psds=None, walker_inds=None, Sn_f=None, **noise_kwargs):
        if Sn_f is not None:
            assert len(f0) == len(Sn_f)
            assert isinstance(f0, type(Sn_f))

        elif psds is not None:
            Sn_f = self.interp_psd(f0, psds, walker_inds=walker_inds)
        else:
            Sn_f = get_sensitivity(f0, **noise_kwargs)

        return Sn_f

    def forward(self, amp, f0, **noise_kwargs):

        if noise_kwargs == {}:
            noise_kwargs = self.noise_kwargs

        Sn_f = self.get_Sn_f(f0, **noise_kwargs)

        factor = 1./2. * np.sqrt((self.Tobs * np.sin(f0 / self.f_star) ** 2) / Sn_f)
        rho = amp * factor
        return (rho, f0)


class GBPriorWrap:
    def __init__(self, ndim, full_prior_container, gen_frequency_alone=False):
        self.base_prior = full_prior_container
        self.use_cupy = full_prior_container.use_cupy
        self.ndim = ndim
        self.gen_frequency_alone = gen_frequency_alone

        if gen_frequency_alone:
            self.keys_sep = [1, 2, 3, 4, 5, 6, 7]
        else:
            self.keys_sep = [2, 3, 4, 5, 6, 7]

    @property
    def priors_in(self):
        return self.base_prior.priors_in

    def logpdf(self, x, **noise_kwargs):
        xp = np if not self.use_cupy else cp
        assert x.shape[1] == self.ndim and x.ndim == 2

        logpdf_everything_else = self.base_prior.logpdf(x, keys=self.keys_sep)

        f0 = xp.asarray(x[:, 1])
        amp = xp.asarray(x[:, 0])
        logpdf_A_f = self.base_prior.priors_in[(0, 1)].logpdf(amp, f0, **noise_kwargs)

        return logpdf_A_f + logpdf_everything_else

    def rvs(self, size=1, ignore_amp=False, **kwargs):
        xp = np if not self.use_cupy else cp
        if isinstance(size, int):
            size = (size,)
        
        arr = xp.zeros(size + (self.ndim,)).reshape(-1, self.ndim)

        diff = self.ndim - len(self.keys_sep)
        assert diff >= 0

        arr[:, :] = self.base_prior.rvs(size, keys=self.keys_sep).reshape(-1, self.ndim)

        if not ignore_amp:
            f0_input = arr[:, 1] if self.gen_frequency_alone else None
            arr[:, :diff] = xp.asarray(self.base_prior.priors_in[(0, 1)].rvs(size, f0_input=f0_input, **kwargs)).reshape(diff, -1).T

        arr = arr.reshape(size + (self.ndim,))
        return arr


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
        

# class FlowDist:
#     def __init__(self, config: dict, model: Union[Galaxy, GalaxyFFdot], fit: str, ndim: int):

#         self.dist = model(config)
#         self.dist.load_fit()

#         param_min, param_max = np.loadtxt(fit)
#         self.dist.set_min(param_min)
#         self.dist.set_max(param_max)
        
#         self.config = config
#         self.fit = fit
#         self.ndim = ndim

#     def rvs(self, size: Optional[Union[int, tuple]]=(1,)) -> cp.ndarray:
#         if isinstance(size, int):
#             size = (size,)

#         total_samp = int(np.prod(size))
#         samples = self.dist.sample(total_samp).reshape(size + (self.ndim,))
#         return samples 

#     def logpdf(self, x: cp.ndarray) -> cp.ndarray:
#         assert x.shape[-1] == self.ndim
#         log_prob = self.dist.log_prob(x.reshape(-1, self.ndim)).reshape(x.shape[:-1])
#         return log_prob

# class GalaxyFlowDist(FlowDist):
#     def __init__(self):
#         config = '/data/mkatz/LISAanalysistools/lisaflow/flow/experiments/configs/gbs/density_galaxy.yaml'
#         model = Galaxy
#         fit = '/data/mkatz/LISAanalysistools/lisaflow/flow/experiments/rvs/minmax_galaxy_sangria.txt'
#         ndim = 3
#         super().__init__(config, model, fit, ndim)

#     def logpdf(self, x: cp.ndarray) -> cp.ndarray:
#         # adjust amplitudes to exp
#         x[:, 0] = np.log(x[:, 0])
#         return super().logpdf(x)

#     def rvs(self, size: Optional[Union[int, tuple]]=(1,)) -> cp.ndarray:
#         if isinstance(size, int):
#             size = (size,)

#         samples = super().rvs(size=size)
#         samples = samples.reshape(-1, samples.shape[-1])
#         samples[:, 0] = np.exp(samples[:, 0])
#         samples = samples.reshape(size + (samples.shape[-1],))
#         return samples

# class FFdotFlowDist(FlowDist):
#     def __init__(self):
#         config = '/data/mkatz/LISAanalysistools/lisaflow/flow/experiments/configs/gbs/density_f.yaml'
#         model = GalaxyFFdot
#         fit = '/data/mkatz/LISAanalysistools/lisaflow/flow/experiments/rvs/minmax_ffdot_sangria.txt'
#         ndim = 2
#         super().__init__(config, model, fit, ndim)
        
    

    
        
        
            
            