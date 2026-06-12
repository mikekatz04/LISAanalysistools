"""Run with: uv run python /workspace/rrondeel/pop_inf/LISAanalysistools/scripts/run_global.py -sfp /workspace/rrondeel/pop_inf/LISAanalysistools/population_inference_input/galaxy_psd_global_fit_settings.py """

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
from lisatools.globalfit.run import CurrentInfoGlobalFit
from lisatools.globalfit.stock.erebor import (
    GalForSetup, GalForSettings, PSDSetup, PSDSettings,
    MBHSetup, MBHSettings, GBSetup, GBSettings
)
from eryn.prior import ProbDistContainer

from lisatools.globalfit.engine import GlobalFitSettings, GeneralSetup, GeneralSettings, RankInfo
from lisatools.globalfit.recipe_steps import subtract_initial_signal
from lisatools.utils.constants import YRSID_SI
from lisatools.globalfit.generatefuncs import *



from eryn.prior import uniform_dist, log_uniform
from eryn.utils import TransformContainer
from eryn.prior import ProbDistContainer

from lisatools.globalfit.preprocessing import L1ProcessingStep
from lisatools.globalfit.recipe_steps import (
    SearchRecipeStep,
    PERecipeStep,
    RJRecipeStep,
    build_psd_moves,
    build_gb_moves,
    build_mbh_moves_phenom,
    scatter_around_injection,
    mbh_catalogue_to_sampling_basis,
    setup_state_for_injection,
)
from lisatools.globalfit.moves import GFCombineMove
from lisatools.globalfit.postprocessing import (
    StochasticMetadata,
    SourceMetadata
)
from lisatools.globalfit.moves import GlobalFitMove

if TYPE_CHECKING:
    from lisatools.globalfit.recipe import Recipe
    from lisatools.analysiscontainer import AnalysisContainerArray
    from lisatools.datacontainer import DataResidualArray
    from lisatools.globalfit.engine import EngineInfo



logger = logging.getLogger(__name__)

def f_ms_to_s(x):
    return x * 1e-3

def ten_to_the_x(x):
    return 10.0 ** x

MOJITO_REFERENCE_TIME = 97729089.327664

#####################

### DEFINE RECIPE ###

#####################


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
    psd_info = curr.source_info["psd"]

    #* =============================== INJECT SOURCES =================================
    # Sampling basis: ``[logA, f0 [mHz], fdot, phi0, cos_iota, psi, lam, sin_beta]``
    spread_gb = np.array([1e-30, 1e-30, 1e-30, 1e-30, 1e-30, 1e-30, 1e-30, 1e-30])
    iteratively_resolved_population_path = "/workspace/ggfitlisa/ldc/mojito_light/catalogues/iteratively_resolved_gbs_075yrs_snr7.npy"
    iteratively_resolved_population = np.load(iteratively_resolved_population_path, allow_pickle=True)

    frequencies = iteratively_resolved_population["Frequency"]

    # iteratively_resolved_population["Polarization"] = iteratively_resolved_population["Polarization"] % np.pi  # ensure polarization is within [0, pi]

    in_band = (frequencies > curr.source_info["gb"].new_f0_lims[0]) & (frequencies < curr.source_info["gb"].new_f0_lims[1])
    logger.info(
        f"Keeping {np.sum(in_band)} out of {len(iteratively_resolved_population)} iteratively resolved GB \
        sources within the band limits {curr.source_info['gb'].new_f0_lims[0]} \
        - {curr.source_info['gb'].new_f0_lims[1]}"
    )
    iteratively_resolved_population = iteratively_resolved_population[in_band]
    subset_inds = np.array([int(name.split('_')[1]) for name in iteratively_resolved_population["Name"]])
    logger.info(f"Injecting {len(subset_inds)} GB sources from iteratively resolved population.")
    # subset_inds = None
    setup_state_for_injection(curr, state, "GB", "gb", spread=spread_gb, subset_inds=subset_inds, priors=priors)

    
    #* ================================= BUILD MOVES ==================================
    psd_search_move, psd_pe_move = build_psd_moves(
        engine_info, curr, acs, priors, num_repeats=psd_info.num_prop_repeats
    )
    
    gb_search_moves, gb_pe_moves = build_gb_moves(
        engine_info, curr, acs, priors, state
    )

    #* ================================= SETUP SEARCH ================================= 
    recipe.add_recipe_component(SearchRecipeStep(moves=[psd_search_move]), name="init psd search")

    # search_weights = [0.8, 0.15, 0.05]
    # recipe.add_recipe_component(RJRecipeStep(moves=gb_search_moves, weights=search_weights, convergence_iter=10), name="gb search")
    
    #* ========================== SETUP PARAMETER ESTIMATION ========================== 

    prior_combined = GFCombineMove(moves=[gb_pe_moves[0], psd_pe_move], share_temperature_control=False)
    fstat_combined = GFCombineMove(moves=[gb_pe_moves[1], psd_pe_move], share_temperature_control=False)
    refit_combined = GFCombineMove(moves=[gb_pe_moves[2], psd_pe_move], share_temperature_control=False)

    pe_weights = [0.8, 0.16, 0.04] # [0.05, 0.45, 0.5] # 
    recipe.add_recipe_component(PERecipeStep(
        moves=[prior_combined, fstat_combined, refit_combined], 
        weights=pe_weights, 
        thin_by=1, 
        convergence_iter=500
    ), name="gb_psd_pe")
        


#######################
##### SETTINGS ########
#######################

# LOG10_TM_ASD_RANGE = (-16.0, -13.0)
# LOG10_OMS_ASD_RANGE = (-12.0, -10.0)

# # Galactic foreground prior ranges
# LOG10_AMP_RANGE = (-46.0, -43.0)
# ALPHA_RANGE = (1.0, 8.0)
# LOG10_FREQ1_RANGE = (np.log10(1e-3), np.log10(1e-2))
# LOG10_FREQ2_RANGE = (np.log10(1e-4), np.log10(1e-2))
# LOG10_FKNEE_RANGE = (np.log10(1e-3), np.log10(1e-1))


def get_psd_erebor_settings(general_set: GeneralSetup) -> tuple[PSDSetup, StochasticMetadata]:

    frequency_ranges = [(general_set.start_freq, general_set.end_freq)]
    prior_model = "uniform"
    model_config = dict(use_splines=False, num_params=2)  # for now just two parameters, but can be extended to include splines or other features in the future
    
    # waveform kwargs
    initialize_kwargs_psd = dict()
    
    # injection for diagnostic plots
    injection = np.array([np.log10(15e-12), np.log10(3e-15)])

    #? Will be changed to relative path in the future
    prior_file_psd = "/workspace/rrondeel/pop_inf/LISAanalysistools/src/lisatools/globalfit/prior_files/mojito_priors/instrumental_noise_analytical_mojito.prior"

    psd_settings = PSDSettings(
        Tobs=general_set.Tobs,
        dt=general_set.dt,
        prior_file=prior_file_psd,
        initialize_kwargs=initialize_kwargs_psd,
        ndim=2,
        injection=injection,
        log_dir=general_set.file_store_dir,
        num_prop_repeats=100,
    )
    prior_model_config = {
        r"$S_{\rm oms}$": (-12.0, -10.0),
        r"$S_{\rm tm}$": (-16.0, -13.0),
    }
    psd_metadata = StochasticMetadata(
        model_config=model_config,
        frequency_ranges=frequency_ranges,
        prior_model=prior_model,
        prior_model_code_link="",  # todo populate repositories
        prior_model_config=prior_model_config,
    )

    return PSDSetup(psd_settings), psd_metadata

def get_galfor_erebor_settings(general_set: GeneralSetup) -> tuple[GalForSetup, StochasticMetadata]:

    frequency_ranges = [(general_set.start_freq, general_set.end_freq)]
    prior_model = "uniform"
    model_config = dict(num_params=5, galactic_grid_kwargs=general_set.sensitivity_init_kwargs["galactic_grid_kwargs"])  
    # for now just two parameters, but can be extended to include splines or other features in the future

    #? Will be changed to relative path in the future
    prior_file_galfor = "/workspace/rrondeel/pop_inf/LISAanalysistools/src/lisatools/globalfit/prior_files/mojito_priors/galactic_foreground_stationary_mojito.prior"
    
    galfor_settings = GalForSettings(
        Tobs=general_set.Tobs,
        dt=general_set.dt,
        prior_file=prior_file_galfor,
        initialize_kwargs={},
        ndim=5,
    )
    prior_model_config = {
        r'$\log_{10} A_{\rm gal}$': (-46.0, -43.0),
        r'$\alpha_{\rm gal}$': (1.0, 3.0),
        r'$\log_{10} f_1$': (np.log10(1e-3), np.log10(1e-1)),
        r'$\log_{10} f_{\rm knee}$': (np.log10(1e-3), np.log10(1e-1)),
        r'$\log_{10} f_2$': (np.log10(1e-4), np.log10(1e-2)),
    }
    galfor_metadata = StochasticMetadata(
        model_config=model_config,
        frequency_ranges=frequency_ranges,
        prior_model=prior_model,
        prior_model_code_link="",  # todo populate repositories
        prior_model_config=prior_model_config,
    )

    return GalForSetup(galfor_settings), galfor_metadata

  

def get_gb_erebor_settings(general_set: GeneralSetup) -> tuple[GBSetup, SourceMetadata]:
    
    waveform_model = "GBGPU"
    waveform_model_code_link = "https://github.com/Erebor-L2D/GBGPU/tree/cdl1-run0"
    prior_model_code_link = "https://priors-database-f0027f.gitlab.io/mojito_light_1a.html#massive-black-hole-binaries-mbhb"
    
    #? Will be changed to relative path in the future
    prior_file_gb = "/workspace/rrondeel/pop_inf/LISAanalysistools/src/lisatools/globalfit/prior_files/mojito_priors/galactic_binary_mojito.prior"
    
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
        force_backend=general_set.gpu_backend,
        flip_ref_phase=True
    )

    # geometric spacing 
    betas = 1 / 1.2 ** np.arange(general_set.ntemps)
    betas[-1] = 0.0001
    
    data_start_freq_ind = int(input_data_arr.settings.f_arr[0] / input_data_arr.settings.df)
    
    search_kwargs = dict(
        nwalkers = 16,
        ntemps = 24,
        shutoff_band_iteration = 10,
        shutoff_frequency_threshold = 4e-3,
        burn_1 = 500,
        nsteps_1 = 400,
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
        prior_file=prior_file_gb,
        # A_lims=A_lims,
        # f0_lims=f0_lims,
        # m_chirp_lims=m_chirp_lims,
        # fdot_lims=fdot_lims,
        # phi0_lims=phi0_lims,
        # iota_lims=iota_lims,
        # psi_lims=psi_lims,
        # alpha_lims=alpha_lims,
        # delta_lims=delta_lims,
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
        nleaves_max=6000,
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
    global_fit_version = "GFPRIORS_v1"
    global_fit_contact = "ereborl2d@googlegroups.com"
    global_fit_code_link = "https://github.com/Erebor-L2D/LISAanalysistools/releases/tag/cdl1-run_0"
    global_fit_input_data_link = ""
    global_fit_input_reference = "mojito light"
    global_fit_noise_model = "parametric"
    global_fit_noise_model_code_link = "https://github.com/Erebor-L2D/LISAanalysistools/blob/9d63bb1e63e7b8f640d3780551d9421df5245992/src/lisatools/sensitivity.py#L1797" #todo populate repositories
    comment = "Testing new prior setup that is compatible with population inference"

    submission_folder = None # "/work/asantini/globalfit/l3c_exchange/mojito_light_results/"

    num_iterations = 300

    # source_ids = [18, 5, 16]

    Tobs = 9.0 * YRSID_SI / 12.0
    dt = 5.0
    start_freq = 1e-4
    end_freq = 2.9e-2

    head_dir = "/workspace/rrondeel/pop_inf/_runs/"
    data_input_path = "/workspace/ggfitlisa/ldc/mojito_light/"
    base_file_name = global_fit_version
    file_store_dir = head_dir + "prior_testing/"
    # head_dir = "/data/asantini/packages/LISAanalysistools/"
    # data_input_path = "/data/asantini/globalfit/MOJITO_DATA/mojito_light_2p5s/"
    # base_file_name = global_fit_version #"test_mbh_18_with_covariance"
    # file_store_dir = head_dir + "mojito_output/"

    gpus = [0]
    cp.cuda.runtime.setDevice(gpus[0])
    # Restrict JAX to only see the target GPU — must be set before JAX backend init
    import jax

    jax.config.update("jax_cuda_visible_devices", ",".join(str(gpu) for gpu in gpus))

    backend = "cuda12x" if gpus is not None else "cpu"
    nwalkers = 24
    ntemps = 16

    window_type = "tukey"
    window_taper_duration = 1 / start_freq
    normalize_window = True

    basis_domain = "fd"
    stft_dt = 1 * 24 * 3600.0 if basis_domain == "stft" else None  # hours

    base_file_name += f"_{basis_domain}"

    processor_init_kwargs = dict(
        L1_folder=data_input_path,
        source_types=["gb", "noise"],  #'vgb', 'gb', 'mbhb'
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
    
    galactic_grid_kwargs = dict(
        R_d=2.18,     # disk radial scale length [kpc]
        z_d=0.48,     # disk vertical scale height [kpc]
        # alpha0=1.006863,  # Initial orbital phase α0 [rad]
        # beta0=2.384498,   # Initial constellation rotation β0 [rad]
        N_lambda=90, # sky grid longitude points
        N_beta=60,   # sky grid latitude points
    )

    sensitivity_init_kwargs = dict(tdi_generation=2, mask_percentage=0.02, galactic_grid_kwargs=galactic_grid_kwargs)

    general_settings = GeneralSettings(
        num_iterations=num_iterations,
        Tobs=Tobs,
        dt=dt,
        file_store_dir=file_store_dir,
        base_file_name=base_file_name,
        start_freq=start_freq,
        end_freq=end_freq,
        basis_domain=basis_domain,
        stft_dt=stft_dt,
        random_seed=1434768955,
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
    ######    Rank/GPU setup  #####
    ###############################

    head_rank = 1
    main_rank = 0

    rank_info = RankInfo(
        head_rank=head_rank,
        main_rank=main_rank,
    )

    ##################################
    ###  PSD + GalFor Settings  ######
    ##################################

    psd_setup, psd_metadata = get_psd_erebor_settings(general_setup)

    galfor_setup, galfor_metadata = get_galfor_erebor_settings(general_setup)
    
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
            "gb": gb_setup,
            "psd": psd_setup,
            "galfor": galfor_setup,
        },
        general_info=general_setup,
        rank_info=rank_info,
        setup_function=setup_recipe,
        source_metadata={
            "gb": gb_metadata,
            "psd": psd_metadata,
            "galfor": galfor_metadata,
        }
    )

    curr_info = CurrentInfoGlobalFit(global_settings)
    breakpoint()
    return curr_info


if __name__ == "__main__":
    settings = get_global_fit_settings()
