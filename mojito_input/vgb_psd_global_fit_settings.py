import h5py
import numpy as np
import shutil
import logging

logger = logging.getLogger(__name__)
level = logging.INFO
logger.setLevel(level)

if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(level)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    # * Prevent duplicate messages*
    logger.propagate = False


GPU_BACKEND = "cuda12x"
try:
    import lisatools
    lisatools.get_backend("lisatools_" + GPU_BACKEND)
    import cupy as cp
    gpu_available = True
    logger.info("STATUS: GPUs are available and running on " + GPU_BACKEND)
except:
    import numpy as cp
    gpu_available = False
    logger.info("STATUS: Either cupy could not be imported or gpu backends were not found...")
    logger.info("WARNING: switching to cpus and numpy.")
    

from eryn.moves.tempering import TemperatureControl, make_ladder

from lisatools.detector import EqualArmlengthOrbits, L1Orbits
from eryn.moves import TemperatureControl
from lisatools.utils.constants import YRSID_SI
from gbgpu.utils.utility import get_fdot
from eryn.state import BranchSupplemental
from lisatools.globalfit.hdfbackend import GFHDFBackend, GBHDFBackend, MBHHDFBackend, EMRIHDFBackend
from lisatools.globalfit.utils import SetupInfoTransfer, AllSetupInfoTransfer
from lisatools.globalfit.run import CurrentInfoGlobalFit, GlobalFit
# from global_fit_input.global_fit_settings import get_global_fit_settings

from lisatools.globalfit.state import GFBranchInfo, AllGFBranchInfo
from lisatools.globalfit.state import MBHState, EMRIState, GBState

# from bbhx.utils.transform import *

from lisatools.globalfit.generatefuncs import *
from lisatools.utils.utility import AET
from lisatools.sampling.prior import SNRPrior, AmplitudeFromSNR, AmplitudeFrequencySNRPrior, GBPriorWrap

from lisatools.globalfit.stock.erebor import (
    GalForSetup, GalForSettings, PSDSetup, PSDSettings,
    MBHSetup, MBHSettings, GBSetup, GBSettings
)

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
from lisatools.analysiscontainer import AnalysisContainerArray
from lisatools.datacontainer import DataResidualArray

# import few


# basic transform functions for pickling
def f_ms_to_s(x):
    return x * 1e-3

from eryn.utils.updates import Update

from lisatools.globalfit.preprocessing import L1ProcessingStep
from lisatools.globalfit.recipe import Recipe, RecipeStep
import time

from lisatools.globalfit.engine import GlobalFitSettings, GeneralSetup, GeneralSettings, RankInfo
from lisatools.globalfit.recipe_steps import SearchRecipeStep, PERecipeStep, RJRecipeStep, build_psd_moves, build_gb_moves, setup_state_for_injection, scatter_around_injection
from lisatools.globalfit.priors.gbpriors import get_fdot_mojito

################

### DEFINE RECIPE

#############

MOJITO_REFERENCE_TIME = 97729089.327664


def setup_recipe(
    recipe: Recipe, 
    engine_info, 
    curr: CurrentInfoGlobalFit, 
    acs: AnalysisContainerArray, 
    priors: dict[str, ProbDistContainer], 
    state
):
    general_info = curr.general_info
    nwalkers: int = general_info.nwalkers
    ntemps: int = general_info.ntemps
    gpus: list[int] = curr.general_info.gpus
    cp.cuda.runtime.setDevice(gpus[0])

    #* =============================== INJECT SOURCES =================================
    # Sampling basis: ``[logA, f0 [mHz], fdot, phi0, cos_iota, psi, lam, sin_beta]``
    spread_gb = np.array([1e-4, 1e-5, 1e-14, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4])
    setup_state_for_injection(curr, state, "VGB", "gb", spread=spread_gb)

    # Sampling basis: ``[S_acc, S_oms]``
    # spread_psd = np.array([1.5e-13, 3.0e-17]) # 1 percent
    # psd_info: PSDSettings = curr.source_info["psd"]
    # assert psd_info.injection is not None and psd_info.betas is not None
    # scatter_around_injection(state, "psd", psd_info.injection, spread=spread_psd, betas=psd_info.betas)
    
    #* ================================= BUILD MOVES ==================================
    num_repeats_psd = 60
    permute_every_psd = 50
    psd_search_move, psd_pe_move = build_psd_moves(
        engine_info, curr, acs, priors, 
        num_repeats=num_repeats_psd,
        permute_every=permute_every_psd
    )
    gb_search_moves, gb_pe_moves = build_gb_moves(
        engine_info, curr, acs, priors, state
    )

    #* ================================= SETUP SEARCH ================================= 
    # all_search_moves = [psd_search_move] + gb_search_moves
    # all_search_weights = [0.5, 0.4, 0.1]
    # gf_search_move = GFCombineMove(moves=all_search_moves, weights=all_search_weights, verbose=True, share_temperature_control=False)
    # gf_search_move.accepted = np.zeros((ntemps, nwalkers))
    recipe.add_recipe_component(SearchRecipeStep(moves=[psd_search_move]), name="psd search")

    # recipe.add_recipe_component(RJRecipeStep(moves=[gf_search_move], convergence_iter=10), name="gb + psd search")
    # search_weights = [0.8, 0.2]
    # recipe.add_recipe_component(RJRecipeStep(moves=gb_search_moves, weights=search_weights, convergence_iter=10), name="gb search")
    
    #* ========================== SETUP PARAMETER ESTIMATION ========================== 
    all_pe_moves = gb_pe_moves + [psd_pe_move]
    pe_weights = [0.4, 0.08, 0.02, 0.5]
    recipe.add_recipe_component(RJRecipeStep(moves=all_pe_moves, weights=pe_weights, thin_by=1, convergence_iter=500), name="gb_psd_pe")


#######################
##### SETTINGS ###########
###############


def get_gb_erebor_settings(general_set: GeneralSetup) -> GBSetup:
    delta_safe = 1e-9

    A_lims = [10**(-23.2), 1e-20]
    f0_lims = [1e-4, 0.023] # reset by band limits
    
    m_chirp_lims = [0.03, 1.34]
    # fdot_max_val = get_fdot(f0_lims[-1], Mc=m_chirp_lims[-1])
    
    fdot_lims = [get_fdot_mojito(f0_lims[-1], sign="-"), get_fdot_mojito(f0_lims[-1], sign="+")] # also reset in band limits
    phi0_lims = [0.0, 2 * np.pi]
    iota_lims = [0.0 + delta_safe, np.pi - delta_safe]
    psi_lims = [0.0, np.pi]
    alpha_lims = [0.0, 2 * np.pi]
    delta_lims = [-np.pi / 2.0 + delta_safe, np.pi / 2.0 - delta_safe]
    
    input_data_arr: DataResidualArray = general_set.input_data_residual_array
    start_freq = float(input_data_arr.settings.f_arr[0])
    end_freq = float(input_data_arr.settings.f_arr[-1])

    Tobs = 1/getattr(input_data_arr.settings, "df")

    oversample = 4
    extra_buffer = 5
    
    assert start_freq and end_freq and general_set.Tobs and general_set.preprocess_kwargs
    start_freq_ind = int(start_freq * general_set.Tobs)

    initialize_kwargs = dict(
        orbits=general_set.gpu_orbits if gpu_available else general_set.orbits, 
        t0=general_set.data_t0,
        force_backend=general_set.gpu_backend
        )

    # geometric spacing 
    betas = 1 / 1.2 ** np.arange(general_set.ntemps)
    betas[-1] = 0.0001
    
    data_start_freq_ind = int(input_data_arr.settings.f_arr[0] / input_data_arr.settings.df)
    
    search_kwargs = dict(
        nwalkers = 32,
        ntemps = 24,
        shutoff_band_iteration = 5,
        shutoff_frequency_threshold = None, # 4e-3 
        burn_1 = 200,
        nsteps_1 = 200,
        snr_threshold = 8.0,
        burn_2 = 500,
        nsteps_2 = 500,
        refit_start_iteration = 5
    )
    
    waveform_kwargs = dict(
        dt=general_set.dt,
        T=Tobs,
        use_c_implementation=True,
        start_freq_ind=data_start_freq_ind,
        tdi_channel_setup="XYZ",
        tdi2=True,
        oversample=oversample,
        window=general_set.window_type,
        window_alpha=general_set.window_alpha
    )
    
    gb_settings = GBSettings(
        A_lims=A_lims,
        f0_lims=f0_lims,
        m_chirp_lims=m_chirp_lims,
        fdot_lims=fdot_lims,
        phi0_lims=phi0_lims,
        iota_lims=iota_lims,
        psi_lims=psi_lims,
        alpha_lims=alpha_lims,
        delta_lims=delta_lims,
        start_freq=start_freq,
        end_freq=end_freq,
        oversample=oversample,
        extra_buffer=extra_buffer,
        # Start_resample_iter, Iter_count_per_resample, !group_proposal_kwargs (handled later?)
        start_freq_ind=start_freq_ind,
        # t0=t0_gbs,
        # tdi_setup="XYZ",
        # use_tdi2=True,
        Tobs=Tobs,
        dt=general_set.dt,
        initialize_kwargs=initialize_kwargs,
        waveform_kwargs=waveform_kwargs,
        # Transform, Priors, Periodic (handled later!)
        nleaves_max=100,
        nleaves_min=0,
        ndim=8,
        betas=betas,
        log_dir=general_set.file_store_dir,
        num_repeat_proposals=50, 
        search_kwargs=search_kwargs        
    )

    gb_setup = GBSetup(gb_settings)
    return gb_setup


def get_psd_erebor_settings(general_set: GeneralSetup) -> PSDSetup:
    initialize_kwargs_psd = dict()

    priors_psd = {
                r'$S_{\rm oms}$': uniform_dist(6.0e-12, 20.0e-11),  # Soms_d
                r'$S_{\rm tm}$': uniform_dist(1.0e-15, 20.0e-14),  # Sa_a
            }
    priors = {"psd": ProbDistContainer(priors_psd)}
    injection = np.array([15e-12, 3e-15]) # for diagnostic plots

    psd_settings = PSDSettings(
        # Psd_kwargs, Nleaves_max, Nleaves_min, Transform_fn (handled later or fixed)
        ndim=2,
        injection=injection,
        Tobs=general_set.Tobs,
        dt=general_set.dt,
        initialize_kwargs=initialize_kwargs_psd,
        #* ?transform, ?periodic, !betas, !other_tempering_kwargs, ?branch_state, ?branch_backend
        priors=priors,
        log_dir=general_set.file_store_dir
    )

    return PSDSetup(psd_settings)


def get_galfor_erebor_settings(general_set: GeneralSetup) -> GalForSetup:    

    galfor_settings = GalForSettings(
        # galfor_kwargs, nleaves_max, nleaves_min, ndim (handled later or fixed)
        Tobs=general_set.Tobs,
        dt=general_set.dt,
        initialize_kwargs={},
        log_dir=general_set.file_store_dir
    )

    return GalForSetup(galfor_settings)



def get_general_erebor_settings() -> GeneralSetup:    
    
    Tobs = 0.75 * YRSID_SI
    dt = 5.0
    start_freq, end_freq = [5e-5, 0.025] # [0.0138032364, 0.0220867393] # 

    # head_dir = "/sps/lisaf/crondeel/Erebor_dev/_data_sets/mojito/"
    # head_dir = "/workspace/rrondeel/erebor/outputs/testing/"
    # data_input_path = "/workspace/ggfitlisa/ldc/mojito_light/"
    # base_file_name = "vgb_psd"
    # file_store_dir = head_dir 

    head_dir = "/data/asantini/packages/LISAanalysistools/"
    data_input_path = "/data/asantini/globalfit/MOJITO_DATA/mojito_light_2p5s/"
    base_file_name = 'vgb_0' #"test_mbh_18_with_covariance"
    file_store_dir = head_dir + "mojito_output/"
    
    gpus = [0]
    cp.cuda.runtime.setDevice(gpus[0])
    import jax
    jax.config.update("jax_cuda_visible_devices", ",".join(str(gpu) for gpu in gpus))
    nwalkers = 8
    ntemps = 10

    window_type = "tukey"
    window_taper_duration = 1 / start_freq
    normalize_window = True
    
    basis_domain = "fd"
    # stft_dt = 24 * 3600.0
    
    source_ids = [0]
    base_file_name += f"_{basis_domain}"
    
    processor_init_kwargs = dict(
        L1_folder=data_input_path,
        source_types=['noise', 'vgb'], # 'mbhb', 'gb',
        source_ids=dict(vgb=source_ids),
        verbose=True,
        do_plots=True,
        orbits_class=L1Orbits,
        orbits_kwargs=dict(force_backend=GPU_BACKEND, frame="ecliptic", armlength=2493162305.42235) #icrs
    )

    downsample_kwargs = {
        "target_fs": 1/dt,  # Hz — target sampling rate (None = no downsampling).
        "window": (
            "kaiser",
            31.0,
        ),  # Kaiser window beta parameter (higher = more aggressive anti-aliasing)
    }

    highpass_kwargs = {
        "cutoff": 1e-5,  # Hz — highpass cutoff frequency
        "order": 2,  # Butterworth filter order
        "zero_phase": True,
    }

    lowpass_kwargs = {
        "cutoff": 1e-1,  # Hz — lowpass cutoff frequency
        "order": 2,  # Butterworth filter order
        "zero_phase": True,
    }
    
    trim_kwargs = {
        "duration": 200 * 3600,  # seconds — duration to trim from each end
        "is_percent": False,  # If True, 'duration' is interpreted as a percentage of the total signal length
        "trimming_type": "from_each_end",  # "from_each_end" or "from_start"
    }

    preprocess_kwargs = dict(
        highpass_kwargs=highpass_kwargs,
        lowpass_kwargs=lowpass_kwargs,
        trim_kwargs=trim_kwargs,
        downsample_kwargs=downsample_kwargs,
        Tobs=Tobs,
    )
    
    sensitivity_init_kwargs = dict(tdi_generation=2, mask_percentage=0.02) # use_splines=True

    general_settings = GeneralSettings(
        Tobs=Tobs,
        dt=dt,
        file_store_dir=file_store_dir,
        base_file_name=base_file_name,
        start_freq=start_freq,
        end_freq=end_freq,
        basis_domain=basis_domain,
        random_seed=103209,
        backup_iter=5,
        nwalkers=nwalkers,
        ntemps=ntemps,
        window_type=window_type,
        window_taper_duration=window_taper_duration,
        gpu_backend=GPU_BACKEND,
        gpus=gpus,
        data_processor=L1ProcessingStep,
        processor_init_kwargs=processor_init_kwargs,
        preprocess_kwargs=preprocess_kwargs,
        normalize_window=normalize_window,
        sensitivity_init_kwargs=sensitivity_init_kwargs,
    )

    general_setup = GeneralSetup(general_settings)
    return general_setup


def get_global_fit_settings(copy_settings_file=False):

    general_setup = get_general_erebor_settings()

    # file_information["past_file_for_start"] = file_store_dir + "rework_6th_run_through" + "_parameter_estimation_main.h5"
    if copy_settings_file:
        shutil.copy(
            __file__, 
            general_setup.file_store_dir 
            + general_setup.base_file_name 
            + "_" 
            + __file__.split("/")[-1]
        )

    ###############################
    ###############################
    ######    Rank/GPU setup  #####
    ###############################
    ###############################

    head_rank = 1

    main_rank = 0
    
    # run results rank will be next available rank if used
    # gmm_ranks will be all other ranks

    rank_info = RankInfo(
        head_rank=head_rank,
        main_rank=main_rank
    )

    ##################################
    ##################################
    ###  Galactic Binary Settings  ###
    ##################################
    ##################################

    # limits on parameters
    gb_setup = get_gb_erebor_settings(general_setup)

    ##################################
    ##################################
    ###  PSD Settings  ###############
    ##################################
    ##################################


    psd_setup = get_psd_erebor_settings(general_setup)

    ##################################
    ##################################
    ###  Galfor Settings  ############
    ##################################
    ##################################


    # galfor_setup = get_galfor_erebor_settings(general_setup)

    ##################################
    ##################################
    ### EMRI Settings ################
    ##################################
    ##################################


    # emri_setup = get_emri_erebor_settings(general_setup)

    ##############
    ## READ OUT ##
    ##############

    global_settings = GlobalFitSettings(
        source_info={
            "gb": gb_setup,
            "psd": psd_setup,
            # "galfor": galfor_setup,
        },
        general_info=general_setup,
        rank_info=rank_info,
        setup_function=setup_recipe,
    )

    curr_info = CurrentInfoGlobalFit(global_settings)

    return curr_info



if __name__ == "__main__":
    settings = get_global_fit_settings()
    breakpoint()
