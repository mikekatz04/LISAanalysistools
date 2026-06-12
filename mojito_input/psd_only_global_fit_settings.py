import logging
import shutil

import h5py
import numpy as np

try:
    import cupy as cp

    gpu_available = True
except (ModuleNotFoundError, ImportError) as e:
    import numpy as cp

    gpu_available = True

from eryn.moves import CombineMove, StretchMove
from eryn.prior import ProbDistContainer, uniform_dist

from eryn.state import BranchSupplemental
from eryn.utils import TransformContainer
from eryn.utils.updates import Update

from lisatools.detector import EqualArmlengthOrbits, L1Orbits

from lisatools.globalfit.engine import GeneralSettings, GeneralSetup, GlobalFitSettings, RankInfo
from lisatools.globalfit.generatefuncs import *
from lisatools.globalfit.preprocessing import L1ProcessingStep
from lisatools.globalfit.recipe_steps import PERecipeStep, SearchRecipeStep, build_psd_moves
from lisatools.globalfit.run import CurrentInfoGlobalFit, GlobalFit
from lisatools.globalfit.state import AllGFBranchInfo, EMRIState, GBState, GFBranchInfo, MBHState
from lisatools.globalfit.stock.erebor import (
    GalForSettings,
    GalForSetup,
    GBSettings,
    GBSetup,
    MBHSettings,
    MBHSetup,
    PSDSettings,
    PSDSetup,
)
from lisatools.globalfit.utils import AllSetupInfoTransfer, SetupInfoTransfer
from lisatools.sampling.moves.skymodehop import SkyMove
from lisatools.sampling.prior import (
    AmplitudeFrequencySNRPrior,
    AmplitudeFromSNR,
    GBPriorWrap,
    SNRPrior,
)
from lisatools.utils.constants import *
from lisatools.utils.constants import YRSID_SI
from lisatools.utils.utility import AET, tukey

# from global_fit_input.global_fit_settings import get_global_fit_settings


# from bbhx.utils.transform import *


def ten_to_the_x(x):
    return 10.0 ** x


def setup_recipe(recipe, engine_info, curr, acs, priors, state):
    cp.cuda.runtime.setDevice(curr.general_info.gpus[0])

    psd_search_move, psd_pe_move = build_psd_moves(engine_info, curr, acs, priors)

    recipe.add_recipe_component(SearchRecipeStep(moves=[psd_search_move]), name="psd search")
    recipe.add_recipe_component(PERecipeStep(moves=[psd_pe_move]), name="psd pe")


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

    return psd_setup, galfor_setup


def get_general_erebor_settings() -> GeneralSetup:

    from lisatools.utils.constants import YRSID_SI
    Tobs =  YRSID_SI 
    dt = 2.5
    start_freq = 1e-4
    end_freq = 2.9e-2

    data_input_path = "/data/asantini/globalfit/MOJITO_DATA/mojito_light_2p5s/"
    base_file_name = "psd_noise_gal"
    file_store_dir = "/work/fpozzoli/test_lisatools/LISAanalysistools/" + "mojito_output/"

    gpus = [0]
    cp.cuda.runtime.setDevice(gpus[0])

    nwalkers = 30
    ntemps = 4

    # Restrict JAX to only see the target GPU — must be set before JAX backend init
    import jax

    jax.config.update("jax_cuda_visible_devices", ",".join(str(gpu) for gpu in gpus))
    # few.get_backend('cuda12x')
    backend = "cuda12x" if gpus is not None else "cpu"
    nwalkers = 30
    ntemps = 4

    window_type = "tukey"
    window_taper_duration = 0.1 * 7 * 24 * 3600.0
    normalize_window = True

    basis_domain = "stft"  # fd
    stft_dt = 7 * 24 * 3600.0 if basis_domain == "stft" else None  # how many hours

    processor_init_kwargs = dict(
        L1_folder=data_input_path,
        source_types=["noise"],
        verbose=True,
        do_plots=True,
        orbits_class=L1Orbits,
        orbits_kwargs=dict(force_backend=backend, frame="icrs"),
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

    sensitivity_init_kwargs = dict(
        tdi_generation=2,
        mask_percentage=0.02,
        use_splines=False,
    )

    # ---- Fixed galactic grid parameters (NOT inferred) ----
    # alpha0, beta0: LISA orbit orientation angles [rad].
    # These should match the orbit file used; 0.0 is the default for
    # equal-armlength/Keplerian orbits.  For numerical orbits, read them
    # from the orbit file or set to the appropriate value.
    galactic_grid_kwargs = dict(
        R_d=2.18,     # disk radial scale length [kpc]
        z_d=0.48,     # disk vertical scale height [kpc]
        alpha0=1.006863,  # Initial orbital phase α0 [rad]
        beta0=2.384498,   # Initial constellation rotation β0 [rad]
        N_lambda=90, # sky grid longitude points
        N_beta=60,   # sky grid latitude points
    )


    general_settings = GeneralSettings(
        Tobs=Tobs,
        dt=dt,
        file_store_dir=file_store_dir,
        base_file_name=base_file_name,
        start_freq=start_freq,
        end_freq=end_freq,
        basis_domain=basis_domain,
        stft_dt=stft_dt,
        random_seed=12345,
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
        galactic_grid_kwargs=galactic_grid_kwargs,
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

    # run results rank will be next available rank if used
    # gmm_ranks will be all other ranks

    rank_info = RankInfo(head_rank=head_rank, main_rank=main_rank)

    ##################################
    ###  PSD + GalFor Settings  ######
    ##################################

    psd_setup, galfor_setup = get_psd_erebor_settings(general_setup)

    ##############
    ## READ OUT ##
    ##############

    global_settings = GlobalFitSettings(
        source_info={
            "psd":    psd_setup,
            "galfor": galfor_setup,
        },
        general_info=general_setup,
        rank_info=rank_info,
        setup_function=setup_recipe,
    )

    curr_info = CurrentInfoGlobalFit(global_settings)

    return curr_info

if __name__ == "__main__":
    settings = get_global_fit_settings()