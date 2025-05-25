import numpy as np
import cupy as xp
from copy import deepcopy

from eryn.moves import RedBlueMove, StretchMove
# from eryn.state import State
from lisatools.globalfit.state import GFState
from lisatools.sampling.moves.skymodehop import SkyMove
from bbhx.likelihood import NewHeterodynedLikelihood
from tqdm import tqdm
from .globalfitmove import GlobalFitMove
from .addremovemove import ResidualAddOneRemoveOneMove

class MBHSpecialMove(ResidualAddOneRemoveOneMove, GlobalFitMove, RedBlueMove):
    def __init__(self, *args, **kwargs):
        
        RedBlueMove.__init__(self, **kwargs)
        
        ResidualAddOneRemoveOneMove.__init__(self, *args, **kwargs)

    def setup_likelihood_here(self, coords):
        # TODO: should we try to pick specifically based on max ll for MBHs rather than data as a whole
        start_likelihood = self.acs.likelihood()
        keep_het = start_likelihood.argmax()

        data_index = xp.arange(self.nwalkers, dtype=np.int32)
        noise_index = xp.arange(self.nwalkers, dtype=np.int32)
        het_coords = np.tile(coords[keep_het], (self.nwalkers, 1))

        # self.waveform_like_kwargs = dict(
        #     **self.waveform_like_kwargs,
        #     constants_index=data_index
        # )

        self.like_fn = NewHeterodynedLikelihood(
            self.waveform_gen,
            self.fd,
            model.analysis_container_arr.data_shaped[0],
            model.analysis_container_arr.psd_shaped[0],
            het_coords,
            256,
            data_index=data_index,
            noise_index=noise_index,
            gpu=xp.cuda.runtime.getDevice(),  # self.use_gpu,
        )
        data_index = walker_inds_base.astype(np.int32)
        noise_index = walker_inds_base.astype(np.int32)

        # set d_d term in the likelihood
        self.like_fn.d_d = self.like_fn(
            removal_coords_in, 
            **self.waveform_like_kwargs
        ) + self.acs.likelihood(noise_only=True)[data_index]

        xp.get_default_memory_pool().free_all_blocks()
        
    def compute_like(self, new_points_in, data_index):
        assert data_index is not None
        logl = like_het.get_ll(
            new_points_in, 
            constants_index=data_index,
        )
                    
        return logl

    def get_waveform_here(self, coords):
        breakpoint()
        xp.get_default_memory_pool().free_all_blocks()
        waveforms = xp.zeros((coords.shape[0], self.acs.nchannels, self.acs.data_length), dtype=complex)
        
        for i in range(coords.shape[0]):
            waveforms[i] = self.waveform_gen(*coords[i], **self.waveform_gen_kwargs)
        
        return waveforms

    def replace_residuals(self, old_state, new_state):
        fd = xp.asarray(self.acs.fd)
        old_contrib = [None, None]
        new_contrib = [None, None]
        for leaf in range(old_state.branches["mbh"].shape[-2]):
            removal_coords = old_state.branches["mbh"].coords[0, :, leaf]
            removal_coords_in = self.transform_fn.both_transforms(removal_coords)
            removal_waveforms = self.waveform_gen(*removal_coords_in.T, fill=True, freqs=fd, **self.mbh_kwargs).transpose(1, 0, 2)
            
            add_coords = new_state.branches["mbh"].coords[0, :, leaf]
            add_coords_in = self.transform_fn.both_transforms(add_coords)
            add_waveforms = self.waveform_gen(*add_coords_in.T, fill=True, freqs=fd, **self.mbh_kwargs).transpose(1, 0, 2)

            if leaf == 0:
                old_contrib[0] = removal_waveforms[0]
                old_contrib[1] = removal_waveforms[1]
                new_contrib[0] = add_waveforms[0]
                new_contrib[1] = add_waveforms[1]
            else:
                old_contrib[0] += removal_waveforms[0]
                old_contrib[1] += removal_waveforms[1]
                new_contrib[0] += add_waveforms[0]
                new_contrib[1] += add_waveforms[1]
            
        self.acs.swap_out_in_base_data(old_contrib, new_contrib)
        xp.get_default_memory_pool().free_all_blocks()

