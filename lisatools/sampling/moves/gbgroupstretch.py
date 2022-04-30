# -*- coding: utf-8 -*-

import numpy as np
from scipy import stats
import warnings

try:
    import cupy as xp

    gpu_available = True
except ModuleNotFoundError:
    import numpy as xp

    gpu_available = False

from eryn.moves import GroupStretchMove
from eryn.prior import PriorContainer
from eryn.utils.utility import groups_from_inds
from .gbmultipletryrj import GBMutlipleTryRJ

from ...diagnostic import inner_product


__all__ = ["GBGroupStretchMove"]


# MHMove needs to be to the left here to overwrite GBBruteRejectionRJ RJ proposal method
class GBGroupStretchMove(GroupStretchMove, GBMutlipleTryRJ):
    """Generate Revesible-Jump proposals for GBs with try-force rejection

    Will use gpu if template generator uses GPU.

    Args:
        priors (object): :class:`PriorContainer` object that has ``logpdf``
            and ``rvs`` methods.

    """

    def __init__(
        self,
        gb_args,
        gb_kwargs,
        start_ind_limit=10,
        *args,
        **kwargs
    ):

        self.name = "gbgroupstretch"
        self.start_ind_limit = start_ind_limit
        GBMutlipleTryRJ.__init__(self, *gb_args, **gb_kwargs)
        GroupStretchMove.__init__(self, *args, **kwargs)

    def setup_gbs(self, branch):
        coords = branch.coords
        inds = branch.inds
        supps = branch.branch_supplimental
        ntemps, nwalkers, nleaves_max, ndim = branch.shape
        all_remaining_coords = coords[inds]
        remaining_wave_info = supps[inds]

        num_remaining = len(all_remaining_coords)
        # TODO: make faster?
        points_out = np.zeros((num_remaining, self.nfriends, ndim))
        

        freqs = all_remaining_coords[:, 1]

        distances = np.abs(freqs[None, :] - freqs[:, None])
        distances[distances == 0.0] = 1e300
        
        """
        distances = self.xp.full((num_remaining,num_remaining), 1e300)
        for i, coords_here in enumerate(all_remaining_coords):
            A_here = remaining_wave_info["A"][i]
            E_here = remaining_wave_info["E"][i]
            sig_len = len(A_here)
            start_ind_here = remaining_wave_info["start_inds"][i].item()
            freqs_here = (self.xp.arange(sig_len) + start_ind_here) * self.df
            psd_here = self.psd[0][start_ind_here - self.start_freq_ind: start_ind_here - self.start_freq_ind + sig_len]
    
            h_h = inner_product([A_here, E_here], [A_here, E_here], f_arr=freqs_here, PSD=psd_here, use_gpu=self.use_gpu)
            
            for j in range(i, num_remaining):
                if j == i:
                    continue
                A_check = remaining_wave_info["A"][j]
                E_check = remaining_wave_info["E"][j]
                start_ind_check = remaining_wave_info["start_inds"][j].item()
                if abs(start_ind_here - start_ind_check) > self.start_ind_limit:
                    continue
                start_ind = self.xp.max(self.xp.asarray([start_ind_here, start_ind_check])).item()
                end_ind = self.xp.min(self.xp.asarray([start_ind_here + sig_len, start_ind_check + sig_len])).item()
                sig_len_new = end_ind - start_ind

                start_ind_now_here = start_ind - start_ind_here
                slice_here = slice(start_ind_now_here, start_ind_now_here + sig_len_new)

                start_ind_now_check = start_ind - start_ind_check
                slice_check = slice(start_ind_now_check, start_ind_now_check + sig_len_new)
                d_h = inner_product([A_here[slice_here], E_here[slice_here]], [A_check[slice_check], E_check[slice_check]], f_arr=freqs_here[slice_here], PSD=psd_here[slice_here], use_gpu=self.use_gpu)
                
                distances[i, j] = abs(1.0 - d_h.real / h_h.real)
                distances[j, i] = distances[i, j]
            print(i)

        keep = self.xp.argsort(distances)[:self.nfriends]
        try:
            keep = keep.get()
        except AttributeError:
            pass
        

        breakpoint()
        """
        keep = np.argsort(distances, axis=1)[:, :self.nfriends]
        supps[inds] = {"group_move_points": all_remaining_coords[keep]}

    def setup_noise_params(self, branch):
        
        coords = branch.coords
        supps = branch.branch_supplimental
        ntemps, nwalkers, nleaves_max, ndim = branch.shape

        par0 = coords[:, :, :, 0].flatten()

        distances = np.abs(par0[None, :] - par0[:, None])
        distances[distances == 0.0] = 1e300
        
        keep = np.argsort(distances, axis=1)[:, :self.nfriends]
        supps[:] = {"group_move_points": coords.reshape(-1, 1)[keep].reshape(ntemps, nwalkers, nleaves_max, self.nfriends, ndim)}

    def find_friends(self, branches):
        for i, (name, branch) in enumerate(branches.items()):
            if name == "gb":
                self.setup_gbs(branch)
            elif name == "noise_params":
                self.setup_noise_params(branch)
            else:
                raise NotImplementedError
