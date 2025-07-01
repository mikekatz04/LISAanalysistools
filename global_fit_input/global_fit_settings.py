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
from gbgpu.utils.constants import *
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

import few

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

class PSDSearchRecipeStep(RecipeStep):

    def setup_run(self, iteration, last_sample, sampler):
        # making sure
        self.moves[0].max_logl_mode = True
        sampler.moves = self.moves
        sampler.weights = self.weights

    def stopping_function(self, iteration, last_sample, sampler):
        # this will already be converged to max logl
        return True


class GBSearchStep(RecipeStep):
    def setup_run(self, iteration, last_sample, sampler):
        sampler.moves = self.moves
        sampler.weights = self.weights

    def stopping_function(self, iteration, last_sample, sampler):
        # this will already be converged to max logl
        return True
    

def setup_gb_functionality(gf_branch_info, curr, acs, priors, state):
    gb_info = curr.source_info["gb"]
    # TODO: adjust this indide current info
    recipe = curr.current_info["recipe"]
    general_info = curr.general_info
    band_edges = gb_info["band_edges"]
    band_N_vals = gb_info["band_N_vals"]
    betas = gb_info["pe_info"]["betas"]
    nwalkers = curr.general_info["nwalkers"]
    ntemps = curr.general_info["ntemps"]
    
    band_temps = np.tile(np.asarray(betas), (len(band_edges) - 1, 1))

    gpu_priors_in = deepcopy(priors["gb"].priors_in)
    for key, item in gpu_priors_in.items():
        item.use_cupy = True

    from eryn.prior import ProbDistContainer
    gpu_priors = {"gb": ProbDistContainer(gpu_priors_in, use_cupy=True)}


    from gbgpu.gbgpu import GBGPU
    gb = GBGPU(use_gpu=True)
    gpus = curr.general_info["gpus"]
    cp.cuda.runtime.setDevice(gpus[0])
    gb.gpus = gpus
    nleaves_max_gb = state.branches["gb"].shape[-2]
    waveform_kwargs = gb_info["pe_info"]["pe_waveform_kwargs"].copy()
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

        check = priors["gb"].logpdf(coords_out_gb)

        if np.any(np.isinf(check)):
            raise ValueError("Starting priors are inf.")

        coords_out_gb[:, 3] = coords_out_gb[:, 3] % (2 * np.pi)
        coords_out_gb[:, 5] = coords_out_gb[:, 5] % (1 * np.pi)
        coords_out_gb[:, 6] = coords_out_gb[:, 6] % (2 * np.pi)
        
        coords_in_in = gb_info["transform"].both_transforms(coords_out_gb)

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


        band_mean_f = (band_edges[1:] + band_edges[:-1]) / 2
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
    # breakpoint()

    band_edges = gb_info["band_edges"]
    num_sub_bands = len(band_edges)
    betas_gb = gb_info["pe_info"]["betas"]

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
    band_mean_f = (band_edges[1:] + band_edges[:-1]) / 2
    from gbgpu.utils.utility import get_N

    if state.branches["gb"].inds.sum() > 0:
        f_in = state.branches["gb"].coords[state.branches["gb"].inds][:, 1] / 1e3
        band_inds_in[state.branches["gb"].inds] = np.searchsorted(band_edges, f_in, side="right") - 1
        N_vals_in[state.branches["gb"].inds] = band_N_vals.get()[band_inds_in[state.branches["gb"].inds]]

    branch_supp_base_shape = (ntemps, nwalkers, nleaves_max_gb)
    state.branches["gb"].branch_supplemental = BranchSupplemental(
        {"N_vals": N_vals_in, "band_inds": band_inds_in}, base_shape=branch_supp_base_shape, copy=True
    )

    ########### GB

    gb_kwargs = dict(
        waveform_kwargs=waveform_kwargs,
        parameter_transforms=gb_info["transform"],
        provide_betas=True,
        skip_supp_names_update=["group_move_points"],
        random_seed=general_info["random_seed"],
        use_gpu=True,
        nfriends=nwalkers,
        phase_maximize=gb_info["pe_info"]["in_model_phase_maximize"],
        ranks_needed=0,
        gpus=[],
        name="group_prop",
        **gb_info["pe_info"]["group_proposal_kwargs"]
    )

    fd = general_info["fd"].copy()

    gb_args = (
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

    # gb_move = GBSpecialStretchMove(
    #     *gb_args,
    #     **gb_kwargs,
    # )

    # add the other
    # gb_move.gb.gpus = gpus

    rj_moves_in = []
    rj_moves_in_frac = []

    gb_kwargs_rj = dict(
        waveform_kwargs=waveform_kwargs,
        parameter_transforms=gb_info["transform"],
        search=False,
        provide_betas=True,
        skip_supp_names_update=["group_move_points"],
        random_seed=general_info["random_seed"],
        use_gpu=True,
        rj_proposal_distribution=gpu_priors,
        name="rj_prior",
        use_prior_removal=True,  # gb_info["pe_info"]["use_prior_removal"],
        nfriends=nwalkers,
        phase_maximize=False,  # should probably be false if pruning  # gb_info["pe_info"]["rj_phase_maximize"],
        ranks_needed=0,
        run_swaps=True, 
        gpus=[],
        **gb_info["pe_info"]["group_proposal_kwargs"]  # needed for it to work
    )
    print("using prior removal and combine move")

    gb_args_rj = (
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

    rj_move_prior = GBSpecialRJPriorMove(
        *gb_args_rj,
        **gb_kwargs_rj,
    )

    # rj_moves_in.append(rj_move_prior)
    # rj_moves_in_frac.append(gb_info["pe_info"]["rj_prior_fraction"])

    ranks_needed_here = 0
    gb_kwargs_rj2 = dict(
        waveform_kwargs=waveform_kwargs,
        parameter_transforms=gb_info["transform"],
        search=False,
        provide_betas=True,
        skip_supp_names_update=["group_move_points"],
        random_seed=general_info["random_seed"],
        use_gpu=True,
        rj_proposal_distribution=gpu_priors,
        run_swaps=False, 
        name="rj_fstat_mcmc_search",
        use_prior_removal=False,  # gb_info["pe_info"]["use_prior_removal"],
        nfriends=nwalkers,
        phase_maximize=True,  # gb_info["pe_info"]["rj_phase_maximize"],
        ranks_needed=ranks_needed_here,
        gpus=[],
        **gb_info["pe_info"]["group_proposal_kwargs"]  # needed for it to work
    )

    rj_serial_search_move = GBSpecialRJSerialSearchMCMC(
        *gb_args_rj,
        **gb_kwargs_rj2,
    )
    rj_serial_search_move.rj_proposal_distribution = None

    rj_moves_in.append(rj_serial_search_move)
    rj_moves_in_frac.append(gb_info["pe_info"]["rj_prior_fraction"])

    # gb_kwargs_rj_search = dict(
    #     waveform_kwargs=waveform_kwargs,
    #     parameter_transforms=gb_info["transform"],
    #     search=False,
    #     provide_betas=True,
    #     skip_supp_names_update=["group_move_points"],
    #     random_seed=general_info["random_seed"],
    #     use_gpu=True,
    #     rj_proposal_distribution={"gb": make_gmm(gb, curr.gb_info["search_gmm_info"])},
    #     name="rj_search",
    #     use_prior_removal=gb_info["pe_info"]["use_prior_removal"],
    #     nfriends=nwalkers,
    #     phase_maximize=gb_info["pe_info"]["rj_phase_maximize"],
    #     ranks_needed=5,
    #     gpus=[6],
    #     **gb_info["pe_info"]["group_proposal_kwargs"]  # needed for it to work
    # )

    # gb_args_rj_search = (
    #     gb,
    #     priors,
    #     general_info["start_freq_ind"],
    #     acs.data_length,
    #     acs,
    #     np.asarray(fd),
    #     band_edges,
        # band_N_vals,
    #     gpu_priors,
    # )

    # rj_move_search = GBSpecialRJSearchMove(
    #     *gb_args_rj_search,
    #     psd_like=psd_move.compute_log_like,  # cleanup
    #     **gb_kwargs_rj_search,
    # )

    # rj_moves_in.append(rj_move_search)
    # rj_moves_in_frac.append(gb_info["pe_info"]["rj_search_fraction"])
    
    total_frac = sum(rj_moves_in_frac)

    effective_ndim = gf_branch_info.ndims["psd"] + gf_branch_info.ndims["galfor"]
    temperature_control = TemperatureControl(nwalkers, effective_ndim, ntemps=ntemps, Tmax=np.inf, permute=False)
    # need permute to be False
    psd_move = PSDMove(
        gb, acs, priors, num_repeats=500, max_logl_mode=True,
        live_dangerously=True, 
        name="psd move",
        gibbs_sampling_setup=[{
            "psd": np.ones((1, gf_branch_info.ndims["psd"]), dtype=bool),
            "galfor": np.ones((1, gf_branch_info.ndims["galfor"]), dtype=bool)
        }],
        temperature_control=temperature_control
    )
    rj_moves = [(rj_move_i, rj_move_frac_i / total_frac) for rj_move_i, rj_move_frac_i in zip(rj_moves_in, rj_moves_in_frac)]

    # moves = GFCombineMove([rj_serial_search_move, gb_move, rj_move_prior])
    moves = GFCombineMove([psd_move, rj_serial_search_move, rj_move_prior, psd_move])
    moves.accepted = np.zeros((ntemps, nwalkers))
    
    tmp = recipe.recipe[1]
    assert tmp["name"] == "gb search"
    tmp["adjust"].moves = [moves]
    
    return SetupInfoTransfer(
        name="gb",
        in_model_moves=[psd_move],  # moves],  # ], #[moves],  # [gb_move] + rj_moves,  # [gb_move] + rj_moves, # probably better to run them all together
        # rj_moves=rj_moves,
    )

            
def setup_mbh_functionality(gf_branch_info, curr, acs, priors, state):

    nwalkers = curr.general_info["nwalkers"]
    ntemps = curr.general_info["ntemps"]
    mbh_info = curr.source_info["mbh"]
    from bbhx.waveformbuild import BBHWaveformFD

    wave_gen = BBHWaveformFD(
        **mbh_info["initialize_kwargs"]
    )

    if False:  # hasattr(state, "betas_all") and state.betas_all is not None:
            betas_all = state.sub_states["mbh"].betas_all
    else:
        print("remove the False above")
        betas_all = np.tile(make_ladder(mbh_info["pe_info"]["ndim"], ntemps=ntemps), (mbh_info["pe_info"]["nleaves_max"], 1))

    # to make the states work 
    betas = betas_all[0]
    state.sub_states["mbh"].betas_all = betas_all

    inner_moves = mbh_info["pe_info"]["inner_moves"]
    tempering_kwargs = dict(ntemps=ntemps, Tmax=np.inf, permute=False)
    
    coords_shape = (ntemps, nwalkers, mbh_info["pe_info"]["nleaves_max"], mbh_info["pe_info"]["ndim"])

    mbh_move_args = (
        "mbh",  # branch_name,
        coords_shape,
        wave_gen,
        tempering_kwargs,
        mbh_info["waveform_kwargs"].copy(),  # waveform_gen_kwargs,
        mbh_info["waveform_kwargs"].copy(),  # waveform_like_kwargs,
        acs,
        mbh_info["pe_info"]["num_prop_repeats"],
        mbh_info["transform"],
        priors,
        inner_moves,
        acs.df
    )
    
    mbh_move = MBHSpecialMove(*mbh_move_args, use_gpu=True,)

    return SetupInfoTransfer(
        name="mbh",
        in_model_moves=[], #mbh_move],
    )


def setup_psd_functionality(gf_branch_info, curr, acs, priors, state):
    gpus = curr.general_info["gpus"]
    cp.cuda.runtime.setDevice(gpus[0])
    gb = GBGPU(use_gpu=True)
    gb.gpus = gpus
    nwalkers = curr.general_info["nwalkers"]
    ntemps = curr.general_info["ntemps"]
    recipe = curr.current_info["recipe"]
    
    effective_ndim = gf_branch_info.ndims["psd"] + gf_branch_info.ndims["galfor"]
    temperature_control = TemperatureControl(nwalkers, effective_ndim, ntemps=ntemps, Tmax=np.inf, permute=False)
    psd_move = PSDMove(
        gb, acs, priors, num_repeats=500, max_logl_mode=False,
        live_dangerously=True,
        name="psd move 2", 
        gibbs_sampling_setup=[{
            "psd": np.ones((1, gf_branch_info.ndims["psd"]), dtype=bool),
            "galfor": np.ones((1, gf_branch_info.ndims["galfor"]), dtype=bool)
        }],
        temperature_control=temperature_control
    )
    psd_move.accepted = np.zeros((ntemps, nwalkers))
    tmp = recipe.recipe[0]
    assert tmp["name"] == "psd search"
    tmp["adjust"].moves = [psd_move]

    return SetupInfoTransfer(
        name="psd",
        in_model_moves=[],  # (psd_move, 1.0)],
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


def setup_emri_functionality(gf_branch_info, curr, acs, priors, state):

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
    
    emri_move = ResidualAddOneRemoveOneMove(*emri_move_args, use_gpu=True,)
    
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
    file_store_dir = "global_fit_output/"
    file_information["file_store_dir"] = file_store_dir
    base_file_name = "rework_2nd_run_through"
    file_information["base_file_name"] = base_file_name
    file_information["plot_base"] = file_store_dir + base_file_name + '/output_plots.png'

    file_information["fp_psd_search_initial"] = file_store_dir + base_file_name + "_initial_search_psd.h5"
    file_information["fp_psd_search"] = file_store_dir + base_file_name + "_search_psd.h5"
    file_information["fp_mbh_search_base"] = file_store_dir + base_file_name + "_search_mbh"

    file_information["fp_main"] = file_store_dir + base_file_name + "_parameter_estimation_main.h5"
    file_information["fp_gb_pe"] = file_store_dir + base_file_name + "_parameter_estimation_gb.h5"
    file_information["fp_psd_pe"] = file_store_dir + base_file_name + "_parameter_estimation_psd.h5"
    file_information["fp_mbh_pe"] = file_store_dir + base_file_name + "_parameter_estimation_mbh.h5"

    file_information["fp_gb_gmm_info"] = file_store_dir + base_file_name + "_gmm_info.pickle"

    file_information["gb_main_chain_file"] = file_store_dir + base_file_name + "_gb_main_chain_file.h5"
    file_information["gb_all_chain_file"] = file_store_dir + base_file_name + "_gb_all_chain_file.h5"

    file_information["mbh_main_chain_file"] = file_store_dir + base_file_name + "_mbh_main_chain_file.h5"

    file_information["status_file"] = file_store_dir + base_file_name + "_status_file.txt"

    if copy_settings_file:
        shutil.copy(__file__, file_store_dir + base_file_name + "_" + __file__.split("/")[-1])
    
    ###############################
    ###############################
    ###  Global Fit data Setup  ###
    ###############################
    ###############################

    ldc_source_file = "LDC2_sangria_training_v2.h5"
    with h5py.File(ldc_source_file, "r") as f:
        tXYZ = f["obs"]["tdi"][:]

        # remove sources
        for source in ["mbhb"]:  # , "dgb", "igb"]:  # "vgb" ,
            change_arr = f["sky"][source]["tdi"][:]
            for change in ["X", "Y", "Z"]:
                tXYZ[change] -= change_arr[change]

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
    _Tobs = YEAR / 12.0
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

    start_freq_ind = 0
    # start_freq_ind = int(0.004 / df)
    # TODO: check this. 
    # This is here because of data storage size 
    # and an issue I think with a zero in the response psd
    end_freq_ind = int(0.030 / df)  # len(A_inj) - 1
    # end_freq_ind = int(0.007 / df)  # len(A_inj) - 1
    
    A_inj, E_inj = (
        Af[start_freq_ind:end_freq_ind],
        Ef[start_freq_ind:end_freq_ind],
    )

    data_length = len(A_inj)
    fd = (np.arange(data_length) + start_freq_ind) * df
    
    # TODO: connect LISA to SSB for MBHs to numerical orbits

    generate_current_state = GenerateCurrentState(A_inj, E_inj)

    gpus = [7]
    cp.cuda.runtime.setDevice(gpus[0])
    few.get_backend('cuda12x')
    nwalkers = 36
    ntemps = 24

    orbits = EqualArmlengthOrbits()
    gpu_orbits = EqualArmlengthOrbits(use_gpu=True)
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
        backup_iter=10,
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
    delta_safe = 1e-5
    A_lims = [7e-26, 1e-19]
    f0_lims = [0.05e-3, 2.5e-2]  # TODO: this upper limit leads to an issue at 23 mHz where there is no source?
    m_chirp_lims = [0.001, 1.0]
    # now with negative fdots
    fdot_max_val = get_fdot(f0_lims[-1], Mc=m_chirp_lims[-1])
    fdot_lims = [-fdot_max_val, fdot_max_val]
    phi0_lims = [0.0, 2 * np.pi]
    iota_lims = [0.0 + delta_safe, np.pi - delta_safe]
    psi_lims = [0.0, np.pi]
    lam_lims = [0.0, 2 * np.pi]
    beta_sky_lims = [-np.pi / 2.0 + delta_safe, np.pi / 2.0 - delta_safe]

    # band separation setup
    max_start_ind_gb = (f0_lims[0]  / df).astype(int)
    min_end_ind_gb = (f0_lims[1]  / df).astype(int)

    from gbgpu.utils.utility import get_N
    test_freq_low = 0.0008
    test_freq_mid = 0.002
    test_freq_high = 0.012

    if Tobs < YEAR / 2.0:
        oversample = 2
    else:
        oversample = 4

    low_N = get_N(1e-30, test_freq_low, Tobs=Tobs, oversample=oversample)[0]
    mid_N = get_N(1e-30, test_freq_mid, Tobs=Tobs, oversample=oversample)[0]
    high_N = get_N(1e-30, test_freq_high, Tobs=Tobs, oversample=oversample)[0]
    
    first_barrier = (0.001 / df).astype(int) - mid_N
    second_barrier = (0.01 / df).astype(int) - high_N

    buffer = 10
    min_width_low = 2 * low_N + buffer
    min_width_mid = 2 * mid_N + buffer
    min_width_high = 2 * high_N + buffer

    # setup low band
    num_bands_low = ((first_barrier - max_start_ind_gb) // min_width_low) + 1
    assert first_barrier - num_bands_low * min_width_low > 0
    _start_ind = first_barrier - num_bands_low * min_width_low
    low_fs_inds_propose = np.arange(_start_ind, first_barrier, min_width_low)
    low_Ns = np.full_like(low_fs_inds_propose, low_N, dtype=int)

    # setup mid band
    num_bands_mid = ((second_barrier - first_barrier) // min_width_mid)
    mid_fs_inds_propose = np.arange(first_barrier, second_barrier, min_width_mid)
    second_barrier_final = mid_fs_inds_propose[-1]
    # remove this last one, it will be added by the upper bands
    mid_fs_inds_propose = mid_fs_inds_propose[:-1]
    mid_Ns = np.full_like(mid_fs_inds_propose, mid_N, dtype=int)
    
    # setup high band
    num_bands_high = ((min_end_ind_gb - second_barrier_final) // min_width_high) + 1
    assert second_barrier_final + num_bands_high * min_width_high < end_freq_ind
    _end_ind = second_barrier_final + num_bands_high * min_width_high
    high_fs_inds_propose = np.arange(second_barrier_final, _end_ind + min_width_high, min_width_high)
    high_Ns = np.full_like(high_fs_inds_propose, high_N, dtype=int)
    band_edge_inds = np.concatenate([low_fs_inds_propose, mid_fs_inds_propose, high_fs_inds_propose])   
    band_edges = band_edge_inds * df 
    band_N_vals = cp.asarray(np.concatenate([low_Ns, mid_Ns, high_Ns]))
    # band_edges = band_edges[395:]

    print("NEED TO THINK ABOUT mCHIRP prior")
    f0_lims = [band_edges[1].min(), band_edges[-2].max()]

    fdot_max_val = get_fdot(f0_lims[-1], Mc=m_chirp_lims[-1])

    fdot_lims = [-fdot_max_val, fdot_max_val]
    
    num_sub_bands = len(band_edges)

    # waveform settings
    # TODO: check beta versus theta
    gb_waveform_kwargs = dict(
        dt=dt, T=Tobs, use_c_implementation=True, oversample=oversample
    )

    pe_gb_waveform_kwargs = dict(
        dt=dt, T=Tobs, use_c_implementation=True, oversample=oversample, start_freq_ind=start_freq_ind
    )

    gb_initialize_kwargs = dict(use_gpu=True, gpus=gpus)

    gb_transform_fn_in = {
        0: np.exp,
        1: f_ms_to_s,
        5: np.arccos,
        8: np.arcsin,
    }

    gb_fill_dict = {"fill_inds": np.array([3]), "ndim_full": 9, "fill_values": np.array([0.0])}

    gb_transform_fn = TransformContainer(
        parameter_transforms=gb_transform_fn_in, fill_dict=gb_fill_dict
    )

    gb_periodic = {"gb": {3: 2 * np.pi, 5: np.pi, 6: 2 * np.pi}}

    # prior setup
    rho_star = 5.0
    # snr_prior = SNRPrior(rho_star)

    # frequency_prior = uniform_dist(*(np.asarray(f0_lims) * 1e3))
    print("Decide how to treat fdot prior")
    priors_gb = {
        0: uniform_dist(*(np.log(np.asarray(A_lims)))),
        1: uniform_dist(*(np.asarray(f0_lims) * 1e3)), # AmplitudeFrequencySNRPrior(rho_star, frequency_prior, L, Tobs, fd=fd),  # use sangria as a default
        2: uniform_dist(*fdot_lims),
        3: uniform_dist(*phi0_lims),
        4: uniform_dist(*np.cos(iota_lims)),
        5: uniform_dist(*psi_lims),
        6: uniform_dist(*lam_lims),
        7: uniform_dist(*np.sin(beta_sky_lims)),
    }

    # TODO: orbits check against sangria/sangria_hm

    # priors_gb_fin = GBPriorWrap(8, ProbDistContainer(priors_gb))
    priors_gb_fin = {"gb": ProbDistContainer(priors_gb)}

    snrs_ladder = np.array([1., 1.5, 2.0, 3.0, 4.0, 5.0, 7.5, 10.0, 15.0, 20.0, 35.0, 50.0, 75.0, 125.0, 250.0, 5e2])
    ntemps_pe = 24  # len(snrs_ladder)
    # betas =  1 / snrs_ladder ** 2  # make_ladder(ndim * 10, Tmax=5e6, ntemps=ntemps_pe)
    betas = 1 / 1.2 ** np.arange(ntemps_pe)
    betas[-1] = 0.0001

    stopping_kwargs = dict(
        n_iters=1000,
        diff=1.0,
        verbose=True
    )

    stop_search_kwargs = dict(
        convergence_iter=5,  # really * thin_by
        verbose=True
    )

    # mcmc info for main run
    gb_main_run_mcmc_info = dict(
        branch_names=["gb"],
        nleaves_max=15000,
        ndim=8,
        ntemps=len(betas),
        betas=betas,
        nwalkers=nwalkers,
        start_resample_iter=-1,  # -1 so that it starts right at the start of PE
        iter_count_per_resample=10,
        pe_waveform_kwargs=pe_gb_waveform_kwargs,
        group_proposal_kwargs=dict(
            n_iter_update=1,
            live_dangerously=True,
            a=1.75,
            num_repeat_proposals=200
        ),
        other_tempering_kwargs=dict(
            adaptation_time=2,
            permute=True
        ),
        use_prior_removal=False,
        rj_refit_fraction=0.2,
        rj_search_fraction=0.2,
        rj_prior_fraction=0.6,
        nsteps=10000,
        update_iterations=1,
        thin_by=3,
        progress=True,
        rho_star=rho_star,
        stop_kwargs=stopping_kwargs,
        stop_search_kwargs=dict(convergence_iter=5, verbose=True),  # really 5 * thin_by
        stopping_iterations=1,
        in_model_phase_maximize=False,
        rj_phase_maximize=False,
    )

    # mcmc info for search runs
    gb_search_run_mcmc_info = dict(
        ndim=8,
        ntemps=10,
        nwalkers=100,
        pe_waveform_kwargs=pe_gb_waveform_kwargs,
        m_chirp_lims=[0.001, 1.2],
        snr_lim=5.0,
        # stop_kwargs=dict(newly_added_limit=1, verbose=True),
        stopping_iterations=1,
    )

    # template generator
    get_gb_templates = GetGBTemplates(
        gb_initialize_kwargs,
        gb_waveform_kwargs
    )

    all_gb_info = dict(
        setup_func=setup_gb_functionality,
        band_edges=band_edges,
        band_N_vals=band_N_vals,
        periodic=gb_periodic,
        priors=priors_gb_fin,
        transform=gb_transform_fn,
        waveform_kwargs=gb_waveform_kwargs,
        initialize_kwargs=gb_initialize_kwargs,
        pe_info=gb_main_run_mcmc_info,
        search_info=gb_search_run_mcmc_info,
        get_templates=get_gb_templates,
    )



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

    psd_kwargs = dict(sens_fn="A1TDISens")  # , use_gpu=False)
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
        setup_func=setup_psd_functionality,
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


    # for transforms
    fill_dict_mbh = {
        "ndim_full": 12,
        "fill_values": np.array([0.0]),
        "fill_inds": np.array([6]),
    }

    # priors
    priors_mbh = {
        0: uniform_dist(np.log(1e4), np.log(1e8)),
        1: uniform_dist(0.01, 0.999999999),
        2: uniform_dist(-0.99999999, +0.99999999),
        3: uniform_dist(-0.99999999, +0.99999999),
        4: uniform_dist(0.01, 1000.0),
        5: uniform_dist(0.0, 2 * np.pi),
        6: uniform_dist(-1.0 + 1e-6, 1.0 - 1e-6),
        7: uniform_dist(0.0, 2 * np.pi),
        8: uniform_dist(-1.0 + 1e-6, 1.0 - 1e-6),
        9: uniform_dist(0.0, np.pi),
        10: uniform_dist(0.0, Tobs + 3600.0),
    }

    # transforms from pe to waveform generation
    parameter_transforms_mbh = {
        0: np.exp,
        4: mbh_dist_trans,
        7: np.arccos,
        9: np.arcsin,
        (0, 1): mT_q,
        (11, 8, 9, 10): LISA_to_SSB,
    }

    transform_fn_mbh = TransformContainer(
        parameter_transforms=parameter_transforms_mbh,
        fill_dict=fill_dict_mbh,
    )

    # sampler treats periodic variables by wrapping them properly
    periodic_mbh = {
        "mbh": {5: 2 * np.pi, 7: 2 * np.pi, 9: np.pi}
    }

    # waveform kwargs
    initialize_kwargs_mbh = dict(
        amp_phase_kwargs=dict(run_phenomd=True),
        response_kwargs=dict(TDItag="AET", orbits=gpu_orbits),
        use_gpu=True
    )

    # for MBH waveform class initialization
    waveform_kwargs_mbh = dict(
        modes=[(2,2)],
        length=1024,
    )

    get_mbh = GetMBHTemplates(
        initialize_kwargs_mbh,
        waveform_kwargs_mbh
    )

    inner_moves = [
        (SkyMove(which="both"), 0.02),
        (SkyMove(which="long"), 0.05),
        (SkyMove(which="lat"), 0.05),
        (StretchMove(), 0.88)
    ]

    mix_stopping_kwargs = dict(
        n_iters=5,
        diff=0.01,
        verbose=False
    )

    # mcmc info for main run
    mbh_main_run_mcmc_info = dict(
        branch_names=["mbh"],
        nleaves_max=15,
        ndim=11,
        ntemps=10,
        nwalkers=50,
        num_prop_repeats=200,
        inner_moves=inner_moves,
        progress=False,
        thin_by=1,
        stop_kwargs=mix_stopping_kwargs,
        stopping_iterations=1
    )

    mbh_search_kwargs = {
        "modes": [(2, 2)],
        "length": 1024,
        "shift_t_limits": True,
        "phase_marginalize": True
    }

    search_stopping_kwargs = dict(
        n_iters=50,
        diff=0.01,
        verbose=False
    )

    mbh_search_run_info = dict(
        ntemps=10,
        nwalkers=100,
        mbh_kwargs=mbh_search_kwargs,
        time_splits=8,
        max_num_per_gpu=2, 
        verbose=False,
        snr_lim=20.0, 
        stop_kwargs=search_stopping_kwargs,
        stopping_iterations=4
    )

    all_mbh_info = dict(
        setup_func=setup_mbh_functionality,
        periodic=periodic_mbh,
        priors={"mbh": ProbDistContainer(priors_mbh)},
        transform=transform_fn_mbh,
        waveform_kwargs=waveform_kwargs_mbh,
        initialize_kwargs=initialize_kwargs_mbh,
        pe_info=mbh_main_run_mcmc_info,
        search_info=mbh_search_run_info,
        get_templates=get_mbh,
        stop_kwargs=search_stopping_kwargs,
    )


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
        use_gpu=True,
    )

    # TODO: I prepared this for Kerr but have not used it with the Kerr waveform yet
    # so spin results are currently meaningless and will lead to slower code
    # waveform kwargs
    initialize_kwargs_emri = dict(
        T=Tobs / YRSID_SI, # TODO: check these conversions all align
        dt=dt,
        emri_waveform_args=("FastKerrEccentricEquatorialFlux",),
        emri_waveform_kwargs=dict(use_gpu=True),
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
        # GFBranchInfo("mbh", 11, 15, 15, branch_state=MBHState, branch_backend=MBHHDFBackend) 
        GFBranchInfo("gb", 8, 8000, 0, branch_state=GBState, branch_backend=GBHDFBackend) 
        # + GFBranchInfo("emri", 12, 1, 1, branch_state=EMRIState, branch_backend=EMRIHDFBackend)  # TODO: generalize this class?
        + GFBranchInfo("galfor", 5, 1, 1) 
        + GFBranchInfo("psd", 4, 1, 1)
    )

    recipe = Recipe()
    recipe.add_recipe_component(PSDSearchRecipeStep(), name="psd search")
    recipe.add_recipe_component(GBSearchStep(), name="gb search")
    
    return {
        "gf_branch_information": gf_branch_information,
        "source_info":{
            "gb": all_gb_info,
            # "mbh": all_mbh_info,
            "psd": all_psd_info,
            # "emri": all_emri_info,
        },
        "general": all_general_info,
        "rank_info": rank_info,
        "gpu_assignments": gpu_assignments,
        "recipe": recipe,
    }



if __name__ == "__main__":
    settings = get_global_fit_settings()
    breakpoint()