import h5py
import numpy as np
import shutil

try:
    import cupy as cp
    gpu_available = True
except (ModuleNotFoundError, ImportError) as e:
    import numpy as cp
    gpu_available = True

from eryn.moves.tempering import TemperatureControl, make_ladder

from lisatools.detector import EqualArmlengthOrbits
from eryn.moves import TemperatureControl
from lisatools.utils.constants import *
from gbgpu.utils.utility import get_fdot
from eryn.state import BranchSupplemental
from lisatools.globalfit.hdfbackend import GFHDFBackend, GBHDFBackend, MBHHDFBackend, EMRIHDFBackend
from lisatools.globalfit.utils import SetupInfoTransfer, AllSetupInfoTransfer
from lisatools.globalfit.run import CurrentInfoGlobalFit, GlobalFit
# from global_fit_input.global_fit_settings import get_global_fit_settings

from lisatools.globalfit.state import GFBranchInfo, AllGFBranchInfo
from lisatools.globalfit.state import MBHState, EMRIState, GBState

from bbhx.utils.transform import *

from lisatools.globalfit.generatefuncs import *
from lisatools.utils.utility import AET
from lisatools.sampling.prior import SNRPrior, AmplitudeFromSNR, AmplitudeFrequencySNRPrior, GBPriorWrap

from eryn.prior import uniform_dist
from eryn.utils import TransformContainer
from eryn.prior import ProbDistContainer

from eryn.moves import StretchMove
from lisatools.sampling.moves.skymodehop import SkyMove

from eryn.moves import CombineMove
from lisatools.globalfit.moves import GBSpecialStretchMove, GBSpecialRJRefitMove, GBSpecialRJSearchMove, GBSpecialRJPriorMove, PSDMove, MBHSpecialMove, ResidualAddOneRemoveOneMove, GBSpecialRJSerialSearchMCMC, GFCombineMove
from lisatools.globalfit.galaxyglobal import make_gmm
from lisatools.globalfit.moves import GlobalFitMove
from lisatools.utils.utility import tukey

from lisatools.globalfit.stock.erebor import get_gb_erebor_settings, get_mbh_erebor_settings


# import few

def dtrend(t, y):
    # @Nikos data setup
    m, b = np.polyfit(t, y, 1)
    ytmp = y - (m * t + b)
    ydetrend = ytmp - np.mean(ytmp)
    return ydetrend

# basic transform functions for pickling
def f_ms_to_s(x):
    return x * 1e-3


def mbh_dist_trans(x):
    return x * PC_SI * 1e9  # Gpc

from eryn.utils.updates import Update

from lisatools.globalfit.recipe import Recipe, RecipeStep
import time


class PSDSearchRecipeStep(RecipeStep):
    def setup_run(self, iteration, last_sample, sampler):
        # making sure
        sampler.moves = self.moves
        sampler.weights = self.weights

    def stopping_function(self, iteration, last_sample, sampler):
        # this will already be converged to max logl
        return True


from lisatools.sampling.stopping import SearchConvergeStopping


class MBHSearchStep(RecipeStep):

    def __init__(self, *args, moves=None, weights=None, **kwargs):
        self.stopper = SearchConvergeStopping(*args, **kwargs)
        super().__init__(moves=moves, weights=weights)

    def setup_run(self, iteration, last_sample, sampler):
        # making sure
        sampler.moves = self.moves
        sampler.weights = self.weights
        
    def stopping_function(self, *args, **kwargs):
        # this will already be converged to max logl
        # I think it should be just True becuase 
        # the inner proposal is taking care of this. 
        return True  # self.stopper(*args, **kwargs)


class GBRunStep(RecipeStep):

    def __init__(self, *args, convergence_iter=5, thin_by=1, verbose=False, **kwargs):
        self.convergence_iter = convergence_iter
        self.verbose = verbose
        RecipeStep.__init__(self, *args, **kwargs)
        self.thin_by = thin_by

    def stopping_function(self, i, sample, sampler):

        if not hasattr(self, "st"):
            self.st = time.perf_counter()

        current_iter = sampler.backend.iteration

        stop = False
        if current_iter > self.convergence_iter:

            nleaves_cc = sampler.backend.get_nleaves(branch_names=["gb"], temp_index=0)["gb"]

            # do not include most recent
            nleaves_cc_max_old = nleaves_cc[:-self.convergence_iter].max()
            nleaves_cc_max_new = nleaves_cc[-self.convergence_iter:].max()

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
                    f"\nTIME TO NOW: {dur} hours"
                )

        return stop
        
    def setup_run(self, iteration, last_sample, sampler):
        # TODO: maybe make this the defaul setup
        sampler.moves = self.moves
        sampler.weights = self.weights
        sampler.override_thin_by = self.thin_by
        sampler.yield_step = self.thin_by
        sampler.checkpoint_step = self.thin_by
        for move in sampler.moves:
            move.periodic = sampler.periodic
            move.ntemps = sampler.ntemps 
            move.temperature_control = sampler.temperature_control


def setup_recipe(recipe, gf_branch_info, curr, acs, priors, state):
    # _ = setup_mbh_functionality(gf_branch_info, curr, acs, priors, state):
    # _ = setup_emri_functionality(gf_branch_info, curr, acs, priors, state):
    gb_info = curr.source_info["gb"]
    mbh_info = curr.source_info["mbh"]
    # TODO: adjust this indide current info
    general_info = curr.general_info
    nwalkers = curr.general_info["nwalkers"]
    ntemps = curr.general_info["ntemps"]

    gpus = curr.general_info["gpus"]
    cp.cuda.runtime.setDevice(gpus[0])
    from gbgpu.gbgpu import GBGPU
    import gbgpu 

    _gb_backend = gbgpu.get_backend("cuda12x")
    _gb_backend.set_cuda_device(gpus[0])

    gb = GBGPU(force_backend="cuda12x")
   
    gb.gpus = gpus
    nwalkers = curr.general_info["nwalkers"]
    ntemps = curr.general_info["ntemps"]
    
    # setup psd search move
    effective_ndim = gf_branch_info.ndims["psd"] + gf_branch_info.ndims["galfor"]
    temperature_control = TemperatureControl(nwalkers, effective_ndim, ntemps=ntemps, Tmax=np.inf, permute=False)
    
    psd_move_args = (gb, acs, priors)

    psd_move_kwargs = dict(
        num_repeats=500,
        live_dangerously=True,
        gibbs_sampling_setup=[{
            "psd": np.ones((1, gf_branch_info.ndims["psd"]), dtype=bool),
            "galfor": np.ones((1, gf_branch_info.ndims["galfor"]), dtype=bool)
        }],
        temperature_control=temperature_control
    )
    
    psd_search_move = PSDMove(
        *psd_move_args, 
        max_logl_mode=True,
        name="psd search move",
        **psd_move_kwargs,
    )

    psd_pe_move = PSDMove(
        *psd_move_args, 
        max_logl_mode=False,
        name="psd pe move",
        **psd_move_kwargs,
    )
    # TODO: put this under the hood
    psd_search_move.accepted = np.zeros((ntemps, nwalkers))
    psd_pe_move.accepted = np.zeros((ntemps, nwalkers))

    recipe.add_recipe_component(PSDSearchRecipeStep(moves=[psd_search_move]), name="psd search")
    
    # return SetupInfoTransfer(
    #     name="psd",
    #     in_model_moves=[],  # (psd_move, 1.0)],
    # )
    band_edges = gb_info.band_edges
    band_N_vals = gb_info.band_N_vals
    gb_betas = gb_info.betas
    
    # TODO: make sure all temperatures are read in from file including psd
    band_temps = np.tile(np.asarray(gb_betas), (len(band_edges) - 1, 1))

    gpu_priors_in = deepcopy(priors["gb"].priors_in)
    for key, item in gpu_priors_in.items():
        item.use_cupy = True

    from eryn.prior import ProbDistContainer
    gpu_priors = {"gb": ProbDistContainer(gpu_priors_in, use_cupy=True)}

    gpus = curr.general_info["gpus"]
    cp.cuda.runtime.setDevice(gpus[0])
    gb.gpus = gpus
    nleaves_max_gb = state.branches["gb"].shape[-2]
    waveform_kwargs = gb_info.waveform_kwargs
    if "N" in waveform_kwargs:
        waveform_kwargs.pop("N")

    # print("REMOVE THIS AFTER TESTING")
    # inj = np.load("ldc2a_inj_395_405_bands.npy")
    # injection_params = np.array([
    #     inj["Amplitude"], 
    #     inj["Frequency"],
    #     inj["FrequencyDerivative"],
    #     np.zeros_like(inj["FrequencyDerivative"]),
    #     inj["InitialPhase"],
    #     inj["Inclination"],
    #     inj["Polarization"],
    #     inj["EclipticLongitude"],
    #     inj["EclipticLatitude"]
    # ]).T
    
    # coords_in_in = np.tile(injection_params, (nwalkers, 1))
    # data_index = np.repeat(cp.arange(nwalkers, dtype=np.int32), injection_params.shape[0])
    # factors = cp.ones_like(data_index, dtype=np.float64)
    # N_vals = cp.full_like(data_index, 256, dtype=np.int32)
    
    # # ll_source_0 = acs.likelihood(source_only=True)
    # # ll_all_0 = acs.likelihood()
    # gb.generate_global_template(
    #     coords_in_in,
    #     data_index,
    #     acs.linear_data_arr,
    #     data_length=acs.data_length,
    #     factors=factors,
    #     data_splits=acs.gpu_map,
    #     # start_freq_ind=start_freq_ind,  # included in waveform_kwargs
    #     N=N_vals,
    #     **waveform_kwargs,
    # )
    # ll_source_1 = acs.likelihood(source_only=True)
    # ll_all_1 = acs.likelihood()
    # gb.d_d = 0.0
    # check = gb.get_ll(
    #     coords_in_in,
    #     acs.linear_data_arr,
    #     acs.linear_psd_arr,
    #     data_index=data_index,
    #     noise_index=data_index,
    #     data_length=acs.data_length,
    #     data_splits=acs.gpu_map,
    #     # start_freq_ind=start_freq_ind,  # included in waveform_kwargs
    #     N=N_vals,
    #     **waveform_kwargs,
    # )
    
    # gb.generate_global_template(
    #     coords_in_in,
    #     data_index,
    #     acs.linear_data_arr,
    #     data_length=acs.data_length,
    #     factors=-1 * factors,
    #     data_splits=acs.gpu_map,
    #     # start_freq_ind=start_freq_ind,  # included in waveform_kwargs
    #     N=N_vals,
    #     **waveform_kwargs,
    # )
    # ll_source_2 = acs.likelihood(source_only=True)
    # ll_all_2 = acs.likelihood()

    if state.branches["gb"].inds[0].sum() > 0:
        
        coords_out_gb = state.branches["gb"].coords[0,
            state.branches["gb"].inds[0]
        ]

        coords_out_gb[:, 3] = coords_out_gb[:, 3] % (2 * np.pi)
        coords_out_gb[:, 5] = coords_out_gb[:, 5] % (1 * np.pi)
        coords_out_gb[:, 6] = coords_out_gb[:, 6] % (2 * np.pi)
        
        check = priors["gb"].logpdf(coords_out_gb)
        if np.any(np.isinf(check)):
            raise ValueError("Starting priors are inf.")

        coords_in_in = gb_info.transform.both_transforms(coords_out_gb)

        band_inds = np.searchsorted(band_edges, coords_in_in[:, 1], side="right") - 1

        walker_vals = np.tile(
            np.arange(nwalkers), (nleaves_max_gb, 1)
        ).transpose((1, 0))[state.branches["gb"].inds[0]]

        data_index_1 = walker_vals  # ((band_inds % 2) + 0) * nwalkers + walker_vals

        data_index = cp.asarray(data_index_1).astype(
            cp.int32
        )

        # goes in as -h
        factors = -cp.ones_like(data_index, dtype=cp.float64)

        from gbgpu.utils.utility import get_N

        N_vals = band_N_vals[band_inds]

        print("before global template")
        # TODO: add test to make sure that the genertor send in the general information matches this one
        gb.gpus = gpus

        gb.generate_global_template(
            coords_in_in,
            data_index,
            acs.linear_data_arr,
            data_length=acs.data_length,
            factors=factors,
            data_splits=acs.gpu_map,
            N=N_vals,
            **waveform_kwargs,
        )

        print("after global template")
        # del data_index
        # del factors
        # cp.get_default_memory_pool().free_all_blocks()
    # import matplotlib.pyplot as plt
    # # plt.loglog(acs.f_arr, np.abs(acs.linear_data_arr[0][0:acs.data_length].get()))
    # plt.loglog(general_info["fd"], np.abs(general_info["A_inj"]))
    # plt.loglog(acs.f_arr, np.abs(acs[0].data_res_arr[0].get()), '--')
    # plt.xlim(band_edges.min(), band_edges.max())
    # plt.savefig("check0.png")
    # plt.close()
    # # plt.loglog(acs.f_arr, np.abs(acs.linear_data_arr[0][acs.data_length:2*acs.data_length].get()))
    # plt.loglog(general_info["fd"], np.abs(general_info["E_inj"]))
    # plt.loglog(acs.f_arr, np.abs(acs[0].data_res_arr[1].get()), '--')
    # plt.xlim(band_edges.min(), band_edges.max())
    # plt.savefig("check1.png")
    # plt.close()

    band_edges = gb_info.band_edges
    num_sub_bands = len(band_edges)
    betas_gb = gb_info.betas

    adjust_temps = False

    if hasattr(state, "band_info"):
        band_info_check = deepcopy(state.band_info)
        adjust_temps = True
    #    del state.band_info

    band_temps = np.tile(np.asarray(betas_gb), (len(band_edges) - 1, 1))
    state.sub_states["gb"].initialize_band_information(nwalkers, ntemps, band_edges, band_temps)
    if adjust_temps:
        state.sub_states["gb"].band_info["band_temps"][:] = band_info_check["band_temps"][0, :]

    band_inds_in = np.zeros((ntemps, nwalkers, nleaves_max_gb), dtype=int)
    N_vals_in = np.zeros((ntemps, nwalkers, nleaves_max_gb), dtype=int)

    if state.branches["gb"].inds.sum() > 0:
        f_in = state.branches["gb"].coords[state.branches["gb"].inds][:, 1] / 1e3
        band_inds_in[state.branches["gb"].inds] = np.searchsorted(band_edges, f_in, side="right") - 1
        N_vals_in[state.branches["gb"].inds] = band_N_vals[band_inds_in[state.branches["gb"].inds]]

    # branch_supp_base_shape = (ntemps, nwalkers, nleaves_max_gb)
    # state.branches["gb"].branch_supplemental = BranchSupplemental(
    #     {"N_vals": N_vals_in, "band_inds": band_inds_in}, base_shape=branch_supp_base_shape, copy=True
    # )

    mbh_info = curr.source_info["mbh"]
    from bbhx.waveformbuild import BBHWaveformFD

    wave_gen = BBHWaveformFD(
        **mbh_info.initialize_kwargs
    )

    if np.any(mbh_inds := state.branches_inds["mbh"][0]):
        for leaf in range(mbh_inds.shape[-1]):
            if mbh_inds[0, leaf]:
                assert np.all(mbh_inds[:, leaf])
                inj_coords = state.branches_coords["mbh"][0, :, leaf]
                inj_coords_in = mbh_info.transform.both_transforms(inj_coords)
                # TODO: fix freqs input with backend
                AET = wave_gen(*inj_coords_in.T, fill=True, freqs=cp.asarray(acs.f_arr),**mbh_info.waveform_kwargs)
                acs.add_signal_to_residual(AET[:, :2])

    if False:  # hasattr(state, "betas_all") and state.betas_all is not None:
            betas_all = state.sub_states["mbh"].betas_all
    else:
        print("remove the False above")
        betas_all = np.tile(make_ladder(mbh_info.ndim, ntemps=ntemps), (mbh_info.nleaves_max, 1))

    # to make the states work 
    betas = betas_all[0]
    state.sub_states["mbh"].betas_all = betas_all

    inner_moves = mbh_info.inner_moves
    tempering_kwargs = dict(ntemps=ntemps, Tmax=np.inf, permute=False)
    
    coords_shape = (ntemps, nwalkers, mbh_info.nleaves_max, mbh_info.ndim)

    mbh_move_args = (
        "mbh",  # branch_name,
        coords_shape,
        wave_gen,
        tempering_kwargs,
        mbh_info.waveform_kwargs.copy(),  # waveform_gen_kwargs,
        mbh_info.waveform_kwargs.copy(),  # waveform_like_kwargs,
        acs,
        mbh_info.num_prop_repeats,
        mbh_info.transform,
        priors,
        inner_moves,
        acs.df
    )
    
    mbh_search_move = MBHSpecialMove(*mbh_move_args, name="mbh_search", run_search=True, force_backend="cuda12x", file_backend=curr.backend, search_fp=general_info["file_information"]["mbh_search_file"])
    mbh_pe_move = MBHSpecialMove(*mbh_move_args, name="mbh_pe", run_search=False, force_backend="cuda12x")

    mbh_search_moves = GFCombineMove([mbh_search_move, psd_search_move])  # GFCombineMove([psd_search_move, mbh_search_move, psd_search_move])
    mbh_search_moves.accepted = np.zeros((ntemps, nwalkers))
    
    recipe.add_recipe_component(MBHSearchStep(moves=[mbh_search_moves], n_iters=5, verbose=True), name="mbh search")
    

    ########### GB
    fd = general_info["fd"].copy()

    gb_move_args = (
        gb,
        priors,
        general_info["start_freq_ind"],
        acs.data_length,
        acs,
        np.asarray(fd),
        band_edges,
        band_N_vals,
        gpu_priors,
    )

    gb_move_kwargs = dict(
        waveform_kwargs=waveform_kwargs,
        parameter_transforms=gb_info.transform,
        provide_betas=True,
        skip_supp_names_update=["group_move_points"],
        random_seed=general_info["random_seed"],
        force_backend="cuda12x",
        nfriends=nwalkers,
        **gb_info.group_proposal_kwargs
    )

    gb_search_prune_move = GBSpecialRJPriorMove(
        *gb_move_args, 
        rj_proposal_distribution=gpu_priors,
        name="rj_prior_search",
        use_prior_removal=True,  
        phase_maximize=False,  
        ranks_needed=0,
        run_swaps=True, 
        gpus=[],
        **gb_move_kwargs
    )

    gb_search_fstat_mcmc_move = GBSpecialRJSerialSearchMCMC(
        *gb_move_args, 
        rj_proposal_distribution=None,
        is_rj_prop=True,
        run_swaps=False, 
        name="rj_fstat_mcmc_search",
        phase_maximize=True, 
        ranks_needed=0,
        gpus=[],
        **gb_move_kwargs
    )

    backend_name = general_info["file_information"]["fp_main"]
    gb_search_refit_move = GBSpecialRJRefitMove(
        *gb_move_args, 
        rj_proposal_distribution=None,
        is_rj_prop=True,
        run_swaps=False, 
        name="rj_refit_search",
        fp=backend_name,
        phase_maximize=True,  # gb_info["pe_info"]["rj_phase_maximize"],
        ranks_needed=0,
        gpus=[],
        **gb_move_kwargs
    )

    gb_search_moves = GFCombineMove([psd_search_move, mbh_pe_move, gb_search_fstat_mcmc_move, gb_search_refit_move, gb_search_prune_move, mbh_pe_move, psd_search_move])
    gb_search_moves.accepted = np.zeros((ntemps, nwalkers))
    
    recipe.add_recipe_component(GBRunStep(moves=[gb_search_moves], convergence_iter=5, verbose=True), name="gb search")
    
    gb_pe_prior_move = GBSpecialRJPriorMove(
        *gb_move_args, 
        rj_proposal_distribution=gpu_priors,
        name="rj_prior",
        use_prior_removal=False,  # gb_info["pe_info"]["use_prior_removal"],
        phase_maximize=False,  # should probably be false if pruning  # gb_info["pe_info"]["rj_phase_maximize"],
        ranks_needed=0,
        run_swaps=True, 
        gpus=[],
        **gb_move_kwargs
    )

    gb_pe_fstat_mcmc_move = GBSpecialRJSerialSearchMCMC(
        *gb_move_args, 
        rj_proposal_distribution=None,
        run_swaps=True, 
        name="rj_fstat_mcmc",
        phase_maximize=False, 
        ranks_needed=0,
        gpus=[],
        **gb_move_kwargs
    )

    gb_pe_refit_move = GBSpecialRJRefitMove(
        *gb_move_args, 
        rj_proposal_distribution=None,
        run_swaps=True, 
        name="rj_refit",
        fp=backend_name,
        phase_maximize=False,  # gb_info["pe_info"]["rj_phase_maximize"],
        ranks_needed=0,
        gpus=[],
        **gb_move_kwargs
    )
    full_pe_moves = [psd_pe_move, mbh_pe_move, gb_pe_prior_move, gb_pe_refit_move, gb_pe_fstat_mcmc_move]
    full_pe_weights = [0.3, 0.6, 0.08, 0.02]
    recipe.add_recipe_component(GBRunStep(moves=full_pe_moves, weights=full_pe_weights, thin_by=5, convergence_iter=100, verbose=True), name="gb pe")
            

def setup_mbh_functionality(recipe, gf_branch_info, curr, acs, priors, state):

    nwalkers = curr.general_info["nwalkers"]
    ntemps = curr.general_info["ntemps"]
    mbh_info = curr.source_info["mbh"]
    from bbhx.waveformbuild import BBHWaveformFD

    wave_gen = BBHWaveformFD(
        **mbh_info.initialize_kwargs
    )

    if False:  # hasattr(state, "betas_all") and state.betas_all is not None:
            betas_all = state.sub_states["mbh"].betas_all
    else:
        print("remove the False above")
        betas_all = np.tile(make_ladder(mbh_info.ndim, ntemps=ntemps), (mbh_info.nleaves_max, 1))

    # to make the states work 
    betas = betas_all[0]
    state.sub_states["mbh"].betas_all = betas_all

    inner_moves = mbh_info.inner_moves
    tempering_kwargs = dict(ntemps=ntemps, Tmax=np.inf, permute=False)
    
    coords_shape = (ntemps, nwalkers, mbh_info.nleaves_max, mbh_info.ndim)

    mbh_move_args = (
        "mbh",  # branch_name,
        coords_shape,
        wave_gen,
        tempering_kwargs,
        mbh_info.waveform_kwargs.copy(),  # waveform_gen_kwargs,
        mbh_info.waveform_kwargs.copy(),  # waveform_like_kwargs,
        acs,
        mbh_info.num_prop_repeats,
        mbh_info.transform,
        priors,
        inner_moves,
        acs.df
    )
    
    mbh_move = MBHSpecialMove(*mbh_move_args, force_backend="cuda12x",)
    breakpoint()
    return SetupInfoTransfer(
        name="mbh",
        in_model_moves=[], #mbh_move],
    )


class WrapEMRI:
    def __init__(self, waveform_gen_td, nchannels, tukey_alpha, start_freq_ind, end_freq_ind, dt):
        self.waveform_gen_td = waveform_gen_td
        self.tukey_alpha = tukey_alpha
        self.nchannels = nchannels
        self.start_freq_ind, self.end_freq_ind = start_freq_ind, end_freq_ind
        self.dt = dt

    def __call__(self, *args, **kwargs):
        AET_t = cp.asarray(self.waveform_gen_td(*args, **kwargs))
        fft_input = AET_t * tukey(AET_t.shape[-1], self.tukey_alpha, xp=cp)[None, :]
        # TODO: adjust this if it needs 3rd axis?
        AET_f = self.dt * cp.fft.rfft(fft_input, axis=-1)[:self.nchannels, self.start_freq_ind: self.end_freq_ind]
        return AET_f


def setup_emri_functionality(recipe, gf_branch_info, curr, acs, priors, state):

    from lisatools.sources.emri import EMRITDIWaveform  

    nwalkers = curr.general_info["nwalkers"]
    ntemps = curr.general_info["ntemps"]
    emri_info = curr.source_info["emri"]

    # TODO: mix this with the actual generatefuncs.py 
    emri_gen = WrapEMRI(EMRITDIWaveform(**emri_info["initialize_kwargs"]), acs.nchannels, curr.general_info["tukey_alpha"], curr.general_info["start_freq_ind"], curr.general_info["end_freq_ind"], curr.general_info["dt"])

    num_emris = gf_branch_info.nleaves_max["emri"]
    ndim = gf_branch_info.ndims["emri"]
    # need to inject emris since they are not in the data set yet. 
    emri_inj_params = priors["emri"].rvs(size=(num_emris,))
    emri_inj_params_in = emri_info["transform"].both_transforms(emri_inj_params)   
    start = curr.general_info["start_freq_ind"]
    end = curr.general_info["end_freq_ind"]
    for inj in emri_inj_params_in:
        AET_f = emri_gen(*inj)
        # this will go into every residual because it is the data
        for i in range(nwalkers):
            # TODO: let's chat about how to work all the "remove from" / "add from" etc.
            acs.remove_signal_from_residual(AET_f, data_index=np.array([i]))
    
    emri_start_params = emri_inj_params[None, None] * (1 + 1e-6 * np.random.randn(ntemps, nwalkers, num_emris, ndim))
    emri_start_params_in = emri_info["transform"].both_transforms(emri_start_params.reshape(-1, ndim)).reshape(emri_start_params.shape[:-1] + (-1,))   
    
    # now we need to actually subtract cold-chain from their specific residuals
    for i in range(nwalkers):
        for leaf in range(num_emris):
            temp = emri_start_params_in[0, i, leaf]
            AET_f = emri_gen(*temp)
            acs.add_signal_to_residual(AET_f, data_index=np.array([i]))
  
    cp.get_default_memory_pool().free_all_blocks()
    from eryn.state import Branch
    state.branches["emri"] = Branch(emri_start_params)
    
    if False:  # hasattr(state, "betas_all") and state.betas_all is not None:
            betas_all = state.sub_states["emri"].betas_all
    else:
        print("remove the False above")
        betas_all = np.tile(make_ladder(emri_info["pe_info"]["ndim"], ntemps=ntemps), (emri_info["pe_info"]["nleaves_max"], 1))

    # to make the states work 
    betas = betas_all[0]
    state.sub_states["emri"].betas_all = betas_all

    inner_moves = emri_info["pe_info"]["inner_moves"]
    # should do skip_swap_branches in the add one / remove one move
    tempering_kwargs = dict(ntemps=ntemps, Tmax=np.inf, permute=False)
    
    coords_shape = (ntemps, nwalkers, emri_info["pe_info"]["nleaves_max"], emri_info["pe_info"]["ndim"])

    emri_move_args = (
        "emri",  # branch_name,
        coords_shape,
        emri_gen,
        tempering_kwargs,
        emri_info["waveform_kwargs"].copy(),  # waveform_gen_kwargs,
        emri_info["waveform_kwargs"].copy(),  # waveform_like_kwargs,
        acs,
        emri_info["pe_info"]["num_prop_repeats"],
        emri_info["transform"],
        priors,
        inner_moves,
        acs.df
    )
    
    emri_move = ResidualAddOneRemoveOneMove(*emri_move_args, force_backend="cuda12x")
    
    return SetupInfoTransfer(
        name="emri",
        in_model_moves=[],  # emri_move],
    )


def get_global_fit_settings(copy_settings_file=False):
    ###############################
    ###############################
    ###  Global Fit File Setup  ###
    ###############################
    ###############################

    file_information = {}
    file_store_dir = "/scratch/335-lisa/mlkatz/gf_output/"
    file_information["file_store_dir"] = file_store_dir
    base_file_name = "rework_8th_run_through"
    file_information["base_file_name"] = base_file_name
    file_information["plot_base"] = file_store_dir + base_file_name + '/output_plots.png'

    file_information["fp_main"] = file_store_dir + base_file_name + "_parameter_estimation_main.h5"
    
    file_information["fp_gb_gmm_info"] = file_store_dir + base_file_name + "_gmm_info.pickle"

    file_information["gb_main_chain_file"] = file_store_dir + base_file_name + "_gb_main_chain_file.h5"
    file_information["gb_all_chain_file"] = file_store_dir + base_file_name + "_gb_all_chain_file.h5"

    file_information["mbh_main_chain_file"] = file_store_dir + base_file_name + "_mbh_main_chain_file.h5"
    file_information["mbh_search_file"] = file_store_dir + base_file_name + "_mbh_search_tmp_file.h5"
    
    # file_information["past_file_for_start"] = file_store_dir + "rework_6th_run_through" + "_parameter_estimation_main.h5"
    if copy_settings_file:
        shutil.copy(__file__, file_store_dir + base_file_name + "_" + __file__.split("/")[-1])
    
    ###############################
    ###############################
    ###  Global Fit data Setup  ###
    ###############################
    ###############################

    ldc_source_file = "/scratch/335-lisa/mlkatz/LDC2_sangria_training_v2.h5"
    with h5py.File(ldc_source_file, "r") as f:
        tXYZ = f["obs"]["tdi"][:]

        # remove sources
        # for source in ["mbhb"]:  # , "dgb", "igb"]:  # "vgb" ,
        #     change_arr = f["sky"][source]["tdi"][:]
        #     for change in ["X", "Y", "Z"]:
        #         tXYZ[change] -= change_arr[change]

        # tXYZ = f["sky"]["dgb"]["tdi"][:]
        # tXYZ["X"] += f["sky"]["dgb"]["tdi"][:]["X"]
        # tXYZ["Y"] += f["sky"]["dgb"]["tdi"][:]["Y"]
        # tXYZ["Y"] += f["sky"]["dgb"]["tdi"][:]["Z"]

    t, X, Y, Z = (
        tXYZ["t"].squeeze(),
        tXYZ["X"].squeeze(),
        tXYZ["Y"].squeeze(),
        tXYZ["Z"].squeeze(),
    )

    dt = t[1] - t[0]
    _Tobs = 2.0 * YRSID_SI / 12.0
    Nobs = int(_Tobs / dt)  # len(t)
    t = t[:Nobs]
    X = X[:Nobs]
    Y = Y[:Nobs]
    Z = Z[:Nobs]

    Tobs = Nobs * dt
    df = 1 / Tobs

    # TODO: @nikos what do you think about the window needed here. For this case at 1 year, I do not think it matters. But for other stuff.
    # the time domain waveforms like emris right now will apply this as well
    tukey_alpha = 0.05
    tukey_here = tukey(X.shape[0], tukey_alpha)
    X = dtrend(t, tukey_here * X.copy())
    Y = dtrend(t, tukey_here * Y.copy())
    Z = dtrend(t, tukey_here * Z.copy())

    # f***ing dt
    Xf, Yf, Zf = (np.fft.rfft(X) * dt, np.fft.rfft(Y) * dt, np.fft.rfft(Z) * dt)
    Af, Ef, Tf = AET(Xf, Yf, Zf)
    # Af[:] = 0.0
    # Ef[:] = 0.0
    # Tf[:] = 0.0

    # TODO: MAKE DEFAULT SETUP CLASSES FOR EACH PIECE

    start_freq_ind = 0
    # start_freq_ind = int(0.004 / df)
    # TODO: check this. 
    # This is here because of data storage size 
    # and an issue I think with a zero in the response psd
    # end_freq_ind = int(0.030 / df)  # len(A_inj)  # ADD THIS BACK TO ALLOW FOR FULL FFT
    # end_freq_ind = int(0.007 / df)  # len(A_inj) - 1
    end_freq_ind = len(Af)  # ALLOW FOR FULL FFT
    
    A_inj, E_inj = (
        Af[start_freq_ind:end_freq_ind],
        Ef[start_freq_ind:end_freq_ind],
    )

    data_length = len(A_inj)
    fd = (np.arange(data_length) + start_freq_ind) * df
    
    # TODO: connect LISA to SSB for MBHs to numerical orbits

    generate_current_state = GenerateCurrentState(A_inj, E_inj)

    gpus = [0]
    cp.cuda.runtime.setDevice(gpus[0])
    # few.get_backend('cuda12x')
    nwalkers = 36
    ntemps = 24

    orbits = EqualArmlengthOrbits()
    gpu_orbits = EqualArmlengthOrbits(force_backend="cuda12x")
    all_general_info = dict(
        file_information=file_information,
        fd=fd,
        A_inj=A_inj,
        E_inj=E_inj,
        t=t, 
        X=X,
        Y=Y,
        Z=Z,
        orbits=orbits,
        gpu_orbits=gpu_orbits, 
        data_length=data_length,
        start_freq_ind=start_freq_ind,
        end_freq_ind=end_freq_ind,
        df=df,
        Tobs=Tobs,
        dt=dt,
        source_file=ldc_source_file,
        generate_current_state=generate_current_state,
        random_seed=1024,
        begin_new_likelihood=False,
        plot_iter=4,
        backup_iter=5,
        nwalkers=nwalkers,
        ntemps=ntemps,
        tukey_alpha=tukey_alpha,
        gpus=gpus

    )


    ###############################
    ###############################
    ######    Rank/GPU setup  #####
    ###############################
    ###############################

    head_rank = 1

    main_rank = 0
    main_gpu = gpus[0]
    other_gpus = gpus[1:]

    # run results rank will be next available rank if used
    # gmm_ranks will be all other ranks

    rank_info = dict(
        head_rank=head_rank,
        main_rank=main_rank
    )

    gpu_assignments = dict(
        main_gpu=main_gpu,
        other_gpus=other_gpus
    )

    ##################################
    ##################################
    ###  Galactic Binary Settings  ###
    ##################################
    ##################################

    # limits on parameters
    gb_setup = get_gb_erebor_settings()

    ##################################
    ##################################
    ###  PSD Settings  ###############
    ##################################
    ##################################


    priors_psd = {
        0: uniform_dist(6.0e-12, 20.0e-12),  # Soms_d
        1: uniform_dist(1.0e-15, 20.0e-15),  # Sa_a
        2: uniform_dist(6.0e-12, 20.0e-12),  # Soms_d
        3: uniform_dist(1.0e-15, 20.0e-15),  # Sa_a
    }

    psd_kwargs = dict(sens_fn="A1TDISens")
    psd_initialize_kwargs = {}

    ### Galactic Foreground Settings #
 
    priors_galfor = {
        0: uniform_dist(1e-45, 2e-43),  # amp
        1: uniform_dist(1e-4, 5e-2),  # knee
        2: uniform_dist(0.01, 3.0),  # alpha
        3: uniform_dist(1e0, 1e7),  # Slope1
        4: uniform_dist(5e1, 8e3),  # Slope2
    }

    search_stopping_kwargs = dict(
        n_iters=5,
        diff=0.01,
        verbose=False
    )

    # mcmc info for main run
    psd_main_run_mcmc_info = dict(
        branch_names=["psd", "galfor"],
        ndims={"psd": 4, "galfor": 5},
        nleaves_max={"psd": 1, "galfor": 1},
        ntemps=10,
        nwalkers=50,
        progress=False,
        thin_by=100,
        update_iterations=20,
        stop_kwargs=search_stopping_kwargs,
        stopping_iterations=1
    )

    all_psd_info = dict(
        periodic=None,
        priors={"psd": ProbDistContainer(priors_psd), "galfor": ProbDistContainer(priors_galfor)},
        psd_kwargs=psd_kwargs,
        initalize_kwargs=psd_initialize_kwargs,
        pe_info=psd_main_run_mcmc_info,
        stopping_iterations=1,
    )


    ##################################
    ##################################
    ### MBHB Settings ################
    ##################################
    ##################################


    mbh_setup = get_mbh_erebor_settings()

    ##################################
    ##################################
    ### EMRI Settings ################
    ##################################
    ##################################


    # for transforms
    # this is an example of how you would fill parameters 
    # if you want to keep them fixed
    # (you need to remove them from the other parts of initialization)
    fill_dict_emri = {
       "ndim_full": 14,
       "fill_values": np.array([1.0, 0.0]), # inclination and Phi_theta
       "fill_inds": np.array([5, 12]),
    }

    # priors
    priors_emri = {
        0: uniform_dist(np.log(1e5), np.log(1e6)),  # M total mass
        1: uniform_dist(1.0, 100.0),  # mu
        2: uniform_dist(0.01, 0.98),  # a
        3: uniform_dist(12.0, 16.0),  # p0
        4: uniform_dist(0.001, 0.4),  # e0
        5: uniform_dist(0.01, 100.0),  # dist in Gpc
        6: uniform_dist(-0.99999, 0.99999),  # qS
        7: uniform_dist(0.0, 2 * np.pi),  # phiS
        8: uniform_dist(-0.99999, 0.99999),  # qK
        9: uniform_dist(0.0, 2 * np.pi),  # phiK
        10: uniform_dist(0.0, 2 * np.pi),  # Phi_phi0
        11: uniform_dist(0.0, 2 * np.pi),  # Phi_r0
    }

    # transforms from pe to waveform generation
    # after the fill happens (this is a little confusing)
    # on my list of things to improve
    parameter_transforms_emri = {
        0: np.exp,  # M 
        7: np.arccos, # qS
        9: np.arccos,  # qK
    }

    transform_fn_emri = TransformContainer(
        parameter_transforms=parameter_transforms_emri,
        fill_dict=fill_dict_emri,

    )

    # sampler treats periodic variables by wrapping them properly
    # TODO: really need to create map for transform_fn with keywork names
    periodic_emri = {
        "emri": {7: 2 * np.pi, 9: np.pi, 10: 2 * np.pi, 11: 2 * np.pi}
    }

    response_kwargs = dict(
        t0=30000.0,
        order=25,
        tdi="1st generation",
        tdi_chan="AE",
        orbits=gpu_orbits,
        force_backend="cuda12x",
    )

    # TODO: I prepared this for Kerr but have not used it with the Kerr waveform yet
    # so spin results are currently meaningless and will lead to slower code
    # waveform kwargs
    initialize_kwargs_emri = dict(
        T=Tobs / YRSID_SI, # TODO: check these conversions all align
        dt=dt,
        emri_waveform_args=("FastKerrEccentricEquatorialFlux",),
        emri_waveform_kwargs=dict(force_backend="cuda12x"),
        response_kwargs=response_kwargs,
    )
    
    
    # for EMRI waveform class initialization
    waveform_kwargs_emri = {}  #  deepcopy(initialize_kwargs_emri)

    get_emri = GetEMRITemplates(
        initialize_kwargs_emri,
        waveform_kwargs_emri,
        start_freq_ind,
        end_freq_ind
    )

    inner_moves_emri = [
        (StretchMove(), 1.0)
    ]

    # mcmc info for main run
    emri_main_run_mcmc_info = dict(
        branch_names=["emri"],
        nleaves_max=1,
        ndim=12,
        ntemps=ntemps,
        nwalkers=nwalkers,
        num_prop_repeats=1,
        inner_moves=inner_moves_emri,
        progress=False,
        thin_by=1,
       # stop_kwargs=mix_stopping_kwargs,
        # stopping_iterations=1
    )

    all_emri_info = dict(
        setup_func=setup_emri_functionality,
        periodic=periodic_emri,
        priors={"emri": ProbDistContainer(priors_emri)},
        transform=transform_fn_emri,
        waveform_kwargs=waveform_kwargs_emri,
        initialize_kwargs=initialize_kwargs_emri,
        pe_info=emri_main_run_mcmc_info,
        get_templates=get_emri,
    )

    ##############
    ## READ OUT ##
    ##############

    # TODO: needs to be okay if there is only one branch
    gf_branch_information = (
        GFBranchInfo("mbh", mbh_setup.ndim, mbh_setup.nleaves_max, 0, branch_state=MBHState, branch_backend=MBHHDFBackend) 
        + GFBranchInfo("gb", gb_setup.ndim, gb_setup.nleaves_max, 0, branch_state=GBState, branch_backend=GBHDFBackend) 
        # + GFBranchInfo("emri", 12, 1, 1, branch_state=EMRIState, branch_backend=EMRIHDFBackend)  # TODO: generalize this class?
        + GFBranchInfo("galfor", 5, 1, 1) 
        + GFBranchInfo("psd", 4, 1, 1)
    )

    curr_info = CurrentInfoGlobalFit({
        "gf_branch_information": gf_branch_information,
        "source_info":{
            "gb": gb_setup,
            "mbh": mbh_setup,
            "psd": all_psd_info,
            # "emri": all_emri_info,
        },
        "general": all_general_info,
        "rank_info": rank_info,
        "setup_function": setup_recipe,
        "gpu_assignments": gpu_assignments,
    })

    return curr_info



if __name__ == "__main__":
    settings = get_global_fit_settings()
    breakpoint()
