from __future__ import annotations

import h5py
import numpy as np
import shutil
import logging

try:
    import cupy as cp

    gpu_available = True
except (ModuleNotFoundError, ImportError) as e:
    import numpy as cp

    gpu_available = False

from typing import TYPE_CHECKING

from lisatools.detector import L1Orbits
from lisatools.domaincomputation import DomainComputationGroupArray
from lisatools.utils.constants import *
from eryn.state import BranchSupplemental
from lisatools.globalfit.run import CurrentInfoGlobalFit
from lisatools.globalfit.stock.erebor import PSDSetup, PSDSettings, MBHSetup, MBHSettings, GBSetup, GBSettings, get_fdot_mojito


from eryn.prior import uniform_dist, log_uniform
from eryn.utils import TransformContainer
from eryn.prior import ProbDistContainer

from eryn.moves import StretchMove, TemperatureControl
from eryn.moves.tempering import make_ladder
from lisatools.globalfit.moves import GFCombineMove, MultiGPUPSDMove, TDMBHSpecialMove
from lisatools.globalfit.engine import GlobalFitSettings, GeneralSetup, GeneralSettings, RankInfo
from lisatools.globalfit.recipe_steps import subtract_initial_signal
from lisatools.utils.constants import YRSID_SI

from eryn.utils.updates import Update

from lisatools.globalfit.preprocessing import L1ProcessingStep
from lisatools.globalfit.recipe_steps import (
    SearchRecipeStep,
    PERecipeStep,
    build_psd_moves,
    build_gb_moves,
    build_mbh_moves_phenom,
    scatter_around_injection,
    mbh_catalogue_to_sampling_basis,
    setup_state_for_injection,
)

from lisatools.globalfit.postprocessing import (
    StochasticMetadata,
    SourceMetadata
)

if TYPE_CHECKING:
    from lisatools.globalfit.recipe import Recipe
    from lisatools.analysiscontainer import AnalysisContainerArray
    from lisatools.datacontainer import DataResidualArray
    from lisatools.globalfit.engine import EngineInfo


MOJITO_REFERENCE_TIME = 97729089.327664

logger = logging.getLogger(__name__)


################

### DEFINE RECIPE

#############


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
    cp.cuda.runtime.setDevice(curr.general_info.gpus[0])

    #* =============================== INJECT SOURCES =================================
    # Sampling basis: ``[logA, f0 [mHz], fdot, phi0, cos_iota, psi, lam, sin_beta]``
    spread_gb = np.array([1e-9, 1e-11, 1e-17, 1e-9, 1e-9, 1e-9, 1e-9, 1e-9])
    # iteratively_resolved_population = np.load("iteratively_resolved_gbs_075yrs_snr7.npy")
    # subset_inds = np.array([int(name.split('_')[1]) for name in iteratively_resolved_population["Name"]])
    subset_inds = None
    setup_state_for_injection(curr, state, "VGB", "gb", spread=spread_gb, subset_inds=subset_inds)

    
    #* ================================= BUILD MOVES ==================================
    gb_search_moves, gb_pe_moves = build_gb_moves(
        engine_info, curr, acs, priors, state
    )

    #* ================================= SETUP SEARCH ================================= 
    # search_weights = [0.8, 0.15, 0.05]
    # recipe.add_recipe_component(RJRecipeStep(moves=gb_search_moves, weights=search_weights, convergence_iter=10), name="gb search")
    
    #* ========================== SETUP PARAMETER ESTIMATION ========================== 
    all_pe_moves = gb_pe_moves 
    pe_weights = [0.8, 0.16, 0.04] # [0.05, 0.45, 0.5] # 
    recipe.add_recipe_component(PERecipeStep(moves=all_pe_moves, weights=pe_weights, thin_by=1, convergence_iter=500), name="gb_pe")
    
    moves_info = "".join([f"Move {all_pe_moves[i].name} has weight {w}, " for i, w in enumerate(pe_weights)])
    logger.info(f"For PE: {moves_info}")


#######################
##### SETTINGS ###########
###############

def get_gb_erebor_settings(general_set: GeneralSetup) -> tuple[GBSetup, SourceMetadata]:
    
    waveform_model = "GBGPU"
    waveform_model_code_link = "https://github.com/Erebor-L2D/GBGPU/tree/cd1l-run0"
    prior_model_code_link = "https://priors-database-f0027f.gitlab.io/mojito_light_1a.html#massive-black-hole-binaries-mbhb"

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

    band_edges = gb_setup.band_edges.copy()
    band_starts = band_edges[:-1]
    band_ends = band_edges[1:]
    frequency_ranges = [(start, end) for start, end in zip(band_starts, band_ends)]

    metadata_initialize_kwargs = initialize_kwargs.copy()
    _ = metadata_initialize_kwargs.pop("orbits", None)  # remove orbits from
    metadata_initialize_kwargs['force_backend'] = 'cpu'  # override to cpu for metadata to avoid issues with serialization

    gb_metadata = SourceMetadata(
        source_type="gb",
        frequency_ranges=frequency_ranges,
        waveform_model=waveform_model,
        waveform_model_code_link=waveform_model_code_link,
        prior_model_code_link=prior_model_code_link,
        waveform_model_config=dict(init_kwargs=metadata_initialize_kwargs, runtime_kwargs=waveform_kwargs),
    )

    return gb_setup, gb_metadata


def get_general_erebor_settings() -> GeneralSetup:

    global_fit_codename = "erebor"
    global_fit_version = "CDL1run1_v2"
    global_fit_contact = "ereborl2d@googlegroups.com"
    global_fit_code_link = "https://github.com/Erebor-L2D"
    global_fit_input_data_link = ""
    global_fit_input_reference = "mojito light"
    global_fit_noise_model = "parametric"
    global_fit_noise_model_code_link = "https://github.com/Erebor-L2D" #todo populate repositories
    comment = "making a shorter run to have something for tomorrow"

    submission_folder = "/workspace/rrondeel/erebor/vgb_run_2/"

    # source_ids = [18, 5, 16]

    Tobs = 9.0 * YRSID_SI / 12.0
    dt = 5.0
    start_freq = 1e-4
    end_freq = 2.5e-2

    head_dir = "/workspace/rrondeel/erebor/"
    data_input_path = "/workspace/ggfitlisa/ldc/mojito_light/"
    base_file_name = global_fit_version #"test_mbh_18_with_covariance"
    file_store_dir = head_dir + "vgb_run_2/"

    gpus = [0]
    cp.cuda.runtime.setDevice(gpus[0])
    # Restrict JAX to only see the target GPU — must be set before JAX backend init
    import jax

    jax.config.update("jax_cuda_visible_devices", ",".join(str(gpu) for gpu in gpus))

    backend = "cuda12x" if gpus is not None else "cpu"
    nwalkers = 30
    ntemps = 24

    window_type = "tukey"
    window_taper_duration = 1 / start_freq
    normalize_window = True

    basis_domain = "fd"
    stft_dt = 1 * 24 * 3600.0 if basis_domain == "stft" else None  # hours

    base_file_name += f"_{basis_domain}"

    processor_init_kwargs = dict(
        L1_folder=data_input_path,
        source_types=["vgb", "noise"],  #'vgb', 'gb', "mbhb",
        source_ids=dict(), # mbhb=source_ids
        verbose=True,
        do_plots=True,
        orbits_class=L1Orbits,
        orbits_kwargs=dict(force_backend=backend, frame="icrs", armlength=2493162305.42235),  # icrs
    )

    downsample_kwargs = {
        "target_fs": 1 / dt,  # Hz — target sampling rate (None = no downsampling).
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
        "duration": 0.02,  # seconds — duration to trim from each end
        "is_percent": True,  # If True, 'duration' is interpreted as a percentage of the total signal length
        "trimming_type": "from_each_end",  # "from_each_end" or "from_start"
    }

    preprocess_kwargs = dict(
        highpass_kwargs=highpass_kwargs,
        lowpass_kwargs=lowpass_kwargs,
        trim_kwargs=trim_kwargs,
        downsample_kwargs=downsample_kwargs,
        Tobs=Tobs,
    )

    sensitivity_init_kwargs = dict(tdi_generation=2, mask_percentage=0.02)

    general_settings = GeneralSettings(
        Tobs=Tobs,
        dt=dt,
        file_store_dir=file_store_dir,
        base_file_name=base_file_name,
        start_freq=start_freq,
        end_freq=end_freq,
        basis_domain=basis_domain,
        stft_dt=stft_dt,
        random_seed=103209,
        backup_iter=5,
        nwalkers=nwalkers,
        ntemps=ntemps,
        window_type=window_type,
        window_taper_duration=window_taper_duration,
        gpus=gpus,
        data_processor=L1ProcessingStep,
        processor_init_kwargs=processor_init_kwargs,
        preprocess_kwargs=preprocess_kwargs,
        normalize_window=normalize_window,
        sensitivity_init_kwargs=sensitivity_init_kwargs,
        global_fit_codename=global_fit_codename,
        global_fit_version=global_fit_version,
        global_fit_contact=global_fit_contact,
        global_fit_code_link=global_fit_code_link,
        input_data_link=global_fit_input_data_link,
        input_reference=global_fit_input_reference,
        noise_model=global_fit_noise_model,
        noise_model_code_link=global_fit_noise_model_code_link,
        submission_parent_folder=submission_folder,
        comment=comment
    )

    general_setup = GeneralSetup(general_settings)
    return general_setup


def get_global_fit_settings(copy_settings_file=False):

    general_setup = get_general_erebor_settings()

    if copy_settings_file:
        shutil.copy(
            __file__,
            general_setup.file_store_dir
            + general_setup.base_file_name
            + "_"
            + __file__.split("/")[-1],
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

    rank_info = RankInfo(head_rank=head_rank, main_rank=main_rank)

    ##################################
    ##################################
    ###  PSD Settings  ###############
    ##################################
    ##################################

    # psd_setup, psd_metadata = get_psd_erebor_settings(general_setup)

    ##################################
    ##################################
    ###  MBH Settings  ###############
    ##################################
    ##################################

    # mbh_setup, mbh_metadata = get_mbh_erebor_settings(general_setup)

    ##################################
    ##################################
    ###  GB Settings  ###############
    ##################################
    ##################################

    gb_setup, gb_metadata = get_gb_erebor_settings(general_setup)

    ##############
    ## READ OUT ##
    ##############

    global_settings = GlobalFitSettings(
        source_info={
            # "mbh": mbh_setup,
            "gb": gb_setup,
            # "psd": psd_setup,
        },
        general_info=general_setup,
        rank_info=rank_info,
        setup_function=setup_recipe,
        source_metadata={
            # "mbh": mbh_metadata,
            "gb": gb_metadata,
            # "psd": psd_metadata,
        }
    )

    curr_info = CurrentInfoGlobalFit(global_settings)

    return curr_info


if __name__ == "__main__":
    settings = get_global_fit_settings()
    breakpoint()