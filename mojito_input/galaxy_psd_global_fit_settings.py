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



from eryn.prior import uniform_dist
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
from lisatools.globalfit.priors.gbpriors import get_fdot_mojito
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

    #* =============================== INJECT SOURCES =================================
    # Sampling basis: ``[logA, f0 [mHz], fdot, phi0, cos_iota, psi, lam, sin_beta]``
    spread_gb = np.array([1e-9, 1e-11, 1e-17, 1e-9, 1e-9, 1e-9, 1e-9, 1e-9])
    iteratively_resolved_population = np.load("/workspace/ggfitlisa/ldc/mojito_light/catalogues/iteratively_resolved_gbs_075yrs_snr7.npy")
    subset_inds = np.array([int(name.split('_')[1]) for name in iteratively_resolved_population["Name"]])
    subset_inds = None
    setup_state_for_injection(curr, state, "GB", "gb", spread=spread_gb, subset_inds=subset_inds)

    
    #* ================================= BUILD MOVES ==================================
    psd_search_move, psd_pe_move = build_psd_moves(
        engine_info, curr, acs, priors
    )
    
    gb_search_moves, gb_pe_moves = build_gb_moves(
        engine_info, curr, acs, priors, state
    )

    #* ================================= SETUP SEARCH ================================= 
    recipe.add_recipe_component(SearchRecipeStep(moves=[psd_search_move]), name="init psd search")

    # search_weights = [0.8, 0.15, 0.05]
    # recipe.add_recipe_component(RJRecipeStep(moves=gb_search_moves, weights=search_weights, convergence_iter=10), name="gb search")
    
    #* ========================== SETUP PARAMETER ESTIMATION ========================== 
    all_pe_moves = GFCombineMove(moves=(gb_pe_moves + [psd_pe_move]), share_temperature_control=False)
    pe_weights = [0.4, 0.08, 0.02, 0.5] # [0.05, 0.45, 0.5] # 
    recipe.add_recipe_component(PERecipeStep(moves=all_pe_moves, weights=pe_weights, thin_by=1, convergence_iter=500), name="gb_pe")
    
    # moves_info = "".join([f"Move {all_pe_moves[i].name} has weight {w}, " for i, w in enumerate(pe_weights)])
    # logger.info(f"For PE: {moves_info}")
    


#######################
##### SETTINGS ########
#######################

LOG10_TM_ASD_RANGE = (-16.0, -13.0)
LOG10_OMS_ASD_RANGE = (-12.0, -10.0)

# Galactic foreground prior ranges
LOG10_AMP_RANGE = (-46.0, -43.0)
ALPHA_RANGE = (1.0, 8.0)
LOG10_FREQ1_RANGE = (np.log10(1e-3), np.log10(1e-2))
LOG10_FREQ2_RANGE = (np.log10(1e-4), np.log10(1e-2))
LOG10_FKNEE_RANGE = (np.log10(1e-3), np.log10(1e-1))


def get_psd_erebor_settings(general_set: GeneralSetup):
    """
    Build PSD and galactic foreground branch setups.

    PSD branch (ndim=2): [Soms_d, Sa_a]

    Galactic foreground branch (ndim=5): [Amp, f_knee, alpha, f_1, f_2]
    Parameter ordering must match what PSDMove.psd_log_like expects:
        galfor_pars[:, 0] = Amp
        galfor_pars[:, 1] = f_knee   (kn_all)
        galfor_pars[:, 2] = alpha
        galfor_pars[:, 3] = f_1  (f_1_all)
        galfor_pars[:, 4] = f_2  (f_2_all)

    R_d, z_d, alpha0, beta0 are NOT inferred — they are fixed in GeneralSettings
    via galactic_grid_kwargs and initialized once in GeneralSetup.init_data_information.
    """
    frequency_ranges = [(general_set.start_freq, general_set.end_freq)]
    prior_model = "uniform"
    model_config = dict(use_splines=False, num_params=2)  # for now just two parameters, but can be extended to include splines or other features in the future
    
    initialize_kwargs_psd = dict()

    psd_input_basis = [r'$\log_{10} S_{\rm oms}$', r'$\log_{10} S_{\rm tm}$']
    psd_transform = TransformContainer(
        input_basis=psd_input_basis,
        output_basis=psd_input_basis,
        parameter_transforms={
            psd_input_basis[0]: ten_to_the_x,
            psd_input_basis[1]: ten_to_the_x,
        },
    )

    # ---- PSD priors ----
    priors_psd = {
        psd_input_basis[0]: uniform_dist(*LOG10_OMS_ASD_RANGE),
        psd_input_basis[1]: uniform_dist(*LOG10_TM_ASD_RANGE),
    }

    galfor_input_basis = [
        r'$\log_{10} A_{\rm gal}$',
        r'$\log_{10} f_{\rm knee}$',
        r'$\alpha_{\rm gal}$',
        r'$\log_{10} f_1$',
        r'$\log_{10} f_2$',
    ]
    galfor_transform = TransformContainer(
        input_basis=galfor_input_basis,
        output_basis=galfor_input_basis,
        parameter_transforms={
            galfor_input_basis[0]: ten_to_the_x,
            galfor_input_basis[1]: ten_to_the_x,
            galfor_input_basis[3]: ten_to_the_x,
            galfor_input_basis[4]: ten_to_the_x,
        },
    )

    # ---- Galactic foreground spectral priors ----
    # Only the spectral envelope is inferred; the sky geometry (R_avg) is fixed.
    priors_galfor = {
        galfor_input_basis[0]: uniform_dist(*LOG10_AMP_RANGE),
        galfor_input_basis[1]: uniform_dist(*LOG10_FKNEE_RANGE),
        galfor_input_basis[2]: uniform_dist(*ALPHA_RANGE),
        galfor_input_basis[3]: uniform_dist(*LOG10_FREQ1_RANGE),
        galfor_input_basis[4]: uniform_dist(*LOG10_FREQ2_RANGE),
    }

    # ---- PSD setup ----
    psd_settings = PSDSettings(
        Tobs=general_set.Tobs,
        dt=general_set.dt,
        initialize_kwargs=initialize_kwargs_psd,
        psd_kwargs={"transform_fn": psd_transform},
        priors={"psd": ProbDistContainer(priors_psd)},
        ndim=2,
    )
    psd_setup = PSDSetup(psd_settings)

    # ---- Galactic foreground setup ----
    galfor_settings = GalForSettings(
        Tobs=general_set.Tobs,
        dt=general_set.dt,
        initialize_kwargs={},
        transform=galfor_transform,
        priors={"galfor": ProbDistContainer(priors_galfor)},
        ndim=5,
    )
    galfor_setup = GalForSetup(galfor_settings)

    prior_model_config = {
        "S_oms": (6.0e-12, 20.0e-11),
        "S_tm": (1.0e-15, 20.0e-14),
    }
    stoch_metadata = StochasticMetadata(
        model_config=model_config,
        frequency_ranges=frequency_ranges,
        prior_model=prior_model,
        prior_model_code_link="",  # todo populate repositories
        prior_model_config=prior_model_config,
    )

    return psd_setup, galfor_setup, stoch_metadata

  

def get_gb_erebor_settings(general_set: GeneralSetup) -> tuple[GBSetup, SourceMetadata]:
    
    waveform_model = "GBGPU"
    waveform_model_code_link = "https://github.com/Erebor-L2D/GBGPU/tree/cdl1-run0"
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
        nwalkers = 16,
        ntemps = 24,
        shutoff_band_iteration = 10,
        shutoff_frequency_threshold = 4e-3,
        burn_1 = 1000,
        nsteps_1 = 400,
        snr_threshold = 8.0,
        burn_2 = 1000,
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
    global_fit_version = "CDL1run1_v2"
    global_fit_contact = "ereborl2d@googlegroups.com"
    global_fit_code_link = "https://github.com/Erebor-L2D/LISAanalysistools/releases/tag/cdl1-run_0"
    global_fit_input_data_link = ""
    global_fit_input_reference = "mojito light"
    global_fit_noise_model = "parametric"
    global_fit_noise_model_code_link = "https://github.com/Erebor-L2D/LISAanalysistools/blob/9d63bb1e63e7b8f640d3780551d9421df5245992/src/lisatools/sensitivity.py#L1797" #todo populate repositories
    comment = "new test run for galaxy+noise."

    submission_folder = "/work/asantini/globalfit/l3c_exchange/mojito_light_results/"

    num_iterations = 300

    # source_ids = [18, 5, 16]

    Tobs = 9.0 * YRSID_SI / 12.0
    dt = 5.0
    start_freq = 1e-4
    end_freq = 2.5e-2

    head_dir = "/data/asantini/packages/LISAanalysistools/"
    data_input_path = "/data/asantini/globalfit/MOJITO_DATA/mojito_light_2p5s/"
    base_file_name = global_fit_version #"test_mbh_18_with_covariance"
    file_store_dir = head_dir + "mojito_output/"

    gpus = [0]
    cp.cuda.runtime.setDevice(gpus[0])
    # Restrict JAX to only see the target GPU — must be set before JAX backend init
    import jax

    jax.config.update("jax_cuda_visible_devices", ",".join(str(gpu) for gpu in gpus))

    backend = "cuda12x" if gpus is not None else "cpu"
    nwalkers = 30
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
        orbits_kwargs=dict(force_backend=backend, frame="icrs"),  # icrs
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

    psd_setup, galfor_setup, stoch_metadata = get_psd_erebor_settings(general_setup)
    
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
            "psd": stoch_metadata,
        }
    )

    curr_info = CurrentInfoGlobalFit(global_settings)

    return curr_info


if __name__ == "__main__":
    settings = get_global_fit_settings()
