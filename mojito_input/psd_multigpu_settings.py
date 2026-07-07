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


from lisatools.detector import L1Orbits
from lisatools.domains import FDSettings, STFTSettings
from lisatools.domaincomputation import DomainComputationGroupArray
from lisatools.utils.constants import *
from eryn.state import BranchSupplemental
from lisatools.globalfit.run import CurrentInfoGlobalFit
from lisatools.globalfit.stock.erebor import PSDSetup, PSDSettings, MBHSetup, MBHSettings


from eryn.prior import uniform_dist, log_uniform, ProbDistContainer
from eryn.utils import TransformContainer

from eryn.moves import TemperatureControl
from eryn.moves.tempering import make_ladder
from lisatools.globalfit.moves import GFCombineMove, MultiGPUPSDMove, TDMBHSpecialMove
from lisatools.globalfit.engine import GlobalFitSettings, GeneralSetup, GeneralSettings, RankInfo
from lisatools.globalfit.recipe import subtract_initial_signal
from lisatools.utils.constants import YRSID_SI

from eryn.utils.updates import Update

from lisatools.globalfit.preprocessing import L1ProcessingStep
from lisatools.globalfit.recipe import (
    SearchRecipeStep,
    PERecipeStep,
    build_psd_moves,
    build_mbh_moves_phenom,
    scatter_around_injection,
    mbh_catalogue_to_sampling_basis,
)

from lisatools.globalfit.postprocessing import StochasticMetadata

logger = logging.getLogger(__name__)

def setup_recipe(recipe, engine_info, curr, acs, priors, state):

    cp.cuda.runtime.setDevice(curr.general_info.gpus[0])

    dcga = DomainComputationGroupArray(acs=acs)

    general_info = curr.general_info
    nwalkers: int = general_info.nwalkers
    ntemps: int = general_info.ntemps
    Tmax: float = 1e6
    num_repeats: int = 100
    permute_every: int = 50

    psd_info = curr.source_info["psd"]
    # mbh_info = curr.source_info["mbh"]

    effective_ndim = engine_info.ndims["psd"]
    temperature_control = TemperatureControl(
        effective_ndim, nwalkers, ntemps=ntemps, Tmax=Tmax, permute=False
    )

    psd_move_kwargs = dict(
        num_repeats=num_repeats,
        permute_every=permute_every,
        live_dangerously=True,
        psd_transform_fn=psd_info.transform,
        temperature_control=temperature_control,
        use_gpu=True,
        run_async=False,
        run_threaded=False
    )

    psd_search_move = MultiGPUPSDMove(
        dcga, priors, max_logl_mode=True, name="psd search move", **psd_move_kwargs
    )
    psd_pe_move = MultiGPUPSDMove(dcga, priors, max_logl_mode=False, name="psd pe move", **psd_move_kwargs)

    psd_search_move.accepted = np.zeros((ntemps, nwalkers))
    psd_pe_move.accepted = np.zeros((ntemps, nwalkers))

    #psd_search_move, psd_pe_move = build_psd_moves(engine_info, curr, acs, priors, num_repeats=num_repeats, permute_every=permute_every, Tmax=Tmax)

    recipe.add_recipe_component(SearchRecipeStep(moves=[psd_search_move]), name="psd search")

    # Initialize MBH walkers from catalogue injection parameters
    
    recipe.add_recipe_component(PERecipeStep(moves=[psd_pe_move]), name="psd pe")


#######################
##### SETTINGS ########
#######################

def get_psd_erebor_settings(general_set: GeneralSetup) -> PSDSetup:

    frequency_ranges = [(general_set.start_freq, general_set.end_freq)]
    prior_model = "uniform"
    model_config = dict(use_splines=False, num_params=2)  # for now just two parameters, but can be extended to include splines or other features in the future

    if prior_model == "uniform":
        logger.info("Using uniform prior for PSD parameters.")
        prior_fn = uniform_dist

    elif prior_model == "log_uniform":
        logger.info("Using log-uniform prior for PSD parameters.")
        prior_fn = log_uniform
    else:
        raise ValueError(f"Unsupported prior model: {prior_model}")
    
    prior_model_config = {
        "S_oms": (6.0e-12, 20.0e-11),
        "S_tm": (1.0e-15, 20.0e-14),
    }

    # waveform kwargs
    initialize_kwargs_psd = dict()

    priors_psd = {
        r"$S_{\rm oms}$": prior_fn(*prior_model_config["S_oms"]),  # Soms_d
        r"$S_{\rm tm}$": prior_fn(*prior_model_config["S_tm"]),  # Sa_a
    }

    priors = {"psd": ProbDistContainer(priors_psd)}

    injection = np.array([15e-12, 3e-15])  # for diagnostic plots

    psd_settings = PSDSettings(
        Tobs=general_set.Tobs,
        dt=general_set.dt,
        initialize_kwargs=initialize_kwargs_psd,
        priors=priors,
        ndim=2,
        injection=injection,
        log_dir=general_set.file_store_dir,
        num_prop_repeats=50,
    )

    psd_metadata = StochasticMetadata(
        model_config=model_config,
        frequency_ranges=frequency_ranges,
        prior_model=prior_model,
        prior_model_code_link="",  # todo populate repositories
        prior_model_config=prior_model_config,
    )

    return PSDSetup(psd_settings), psd_metadata


def get_general_erebor_settings() -> GeneralSetup:
    # limits on parameters
    # now with negative fdots

    global_fit_codename = "erebor"
    global_fit_version = "run0_v8"
    global_fit_contact = "ereborl2d@googlegroups.com"
    global_fit_code_link = "https://github.com/Erebor-L2D"
    global_fit_input_data_link = "https://nextcloud-dcc-fi-csc-okd-globalstorage1.2.rahtiapp.fi/apps/files/files/4641?dir=/brickmarket/mojito_light_v1_0_0"
    global_fit_input_reference = "mojito light"
    global_fit_noise_model = "parametric"
    global_fit_noise_model_code_link = "https://github.com/Erebor-L2D" #todo populate repositories

    submission_folder = None #"/work/asantini/globalfit/l3c_exchange/mojito_light_results/"

    source_ids = [18]

    Tobs = 3.0 * YRSID_SI / 12.0
    dt = 5.0
    start_freq = 1e-4
    end_freq = 2.9e-2

    head_dir = "/data/asantini/packages/LISAanalysistools/"
    data_input_path = "/data/asantini/globalfit/MOJITO_DATA/mojito_light_2p5s/"
    base_file_name = "3_months_psd_noswaps_debug_newmove" #"test_mbh_18_with_covariance"
    file_store_dir = head_dir + "mojito_output/"

    gpus = [0]
    cp.cuda.runtime.setDevice(gpus[0])
    # Restrict JAX to only see the target GPU — must be set before JAX backend init
    import jax
    jax.config.update("jax_cuda_visible_devices", ",".join(str(gpu) for gpu in gpus))

    backend = "cuda12x" if gpus is not None else "cpu"
    nwalkers = 24
    ntemps = 10

    window_type = "tukey"
    window_taper_duration = 1 / start_freq
    normalize_window = True

    basis_domain = "fd"
    stft_dt = 7 * 24 * 3600.0 if basis_domain == "stft" else None  # hours

    base_file_name += f"_{basis_domain}"

    processor_init_kwargs = dict(
        L1_folder=data_input_path,
        source_types=["noise"],  #'vgb', 'gb'
        #source_ids=dict(mbhb=source_ids),
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

    # Domain communicated by settings factory, not a string flag (sprint
    # rule): the engine calls ``factory(times, dt, force_backend)`` after
    # loading the data so the grid is sized against the real time array.
    if basis_domain == "stft":
        domain_settings = STFTSettings.make_factory(
            big_dt=stft_dt, min_freq=start_freq, max_freq=end_freq
        )
    else:
        domain_settings = FDSettings.make_factory(
            min_freq=start_freq, max_freq=end_freq
        )

    general_settings = GeneralSettings(
        Tobs=Tobs,
        dt=dt,
        file_store_dir=file_store_dir,
        base_file_name=base_file_name,
        domain_settings=domain_settings,
        random_seed=103209,
        backup_iter=5,
        nwalkers=nwalkers,
        ntemps=ntemps,
        window_type=window_type,
        window_taper_duration=window_taper_duration,
        gpus=gpus,
        data_processor_class=L1ProcessingStep,
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
    )

    general_setup = GeneralSetup(general_settings)
    # Band/STFT metadata consumed by the per-source setup functions
    # (no longer GeneralSettings fields post-merge; the analysis band
    # itself lives on domain_settings).
    general_setup.start_freq = start_freq
    general_setup.end_freq = end_freq
    general_setup.stft_dt = stft_dt
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

    psd_setup, psd_metadata = get_psd_erebor_settings(general_setup)

    ##############
    ## READ OUT ##
    ##############

    global_settings = GlobalFitSettings(
        source_info={
            "psd": psd_setup,
        },
        general_info=general_setup,
        rank_info=rank_info,
        setup_function=setup_recipe,
        source_metadata={
            "psd": psd_metadata,
        },
    )

    curr_info = CurrentInfoGlobalFit(global_settings)

    return curr_info


if __name__ == "__main__":
    settings = get_global_fit_settings()
    breakpoint()
