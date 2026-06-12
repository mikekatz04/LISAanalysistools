import h5py
import numpy as np
import shutil
import logging

try:
    import cupy as cp
    gpu_available = True
except (ModuleNotFoundError, ImportError) as e:
    import numpy as cp
    gpu_available = True

from lisatools.detector import EqualArmlengthOrbits
from lisatools.domains import FDSettings, STFTSettings
from lisatools.utils.constants import *
from eryn.state import BranchSupplemental
from lisatools.globalfit.hdfbackend import GFHDFBackend, GBHDFBackend, MBHHDFBackend, EMRIHDFBackend
from lisatools.globalfit.utils import SetupInfoTransfer, AllSetupInfoTransfer
from lisatools.globalfit.run import CurrentInfoGlobalFit, GlobalFit

from lisatools.globalfit.state import GFBranchInfo, AllGFBranchInfo
from lisatools.globalfit.state import MBHState, EMRIState, GBState

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
from lisatools.globalfit.moves import (
    GBSpecialStretchMove, GBSpecialRJRefitMove, GBSpecialRJSearchMove,
    GBSpecialRJPriorMove, MBHSpecialMove, GBSpecialRJSerialSearchMCMC, GFCombineMove
)
from lisatools.globalfit.galaxyglobal import make_gmm
from lisatools.globalfit.moves import GlobalFitMove
from lisatools.utils.utility import tukey

from lisatools.globalfit.engine import GlobalFitSettings, GeneralSetup, GeneralSettings, RankInfo

from eryn.utils.updates import Update

from lisatools.globalfit.preprocessing import L1ProcessingStep
from lisatools.globalfit.recipe_steps import SearchRecipeStep, PERecipeStep, build_psd_moves


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
    Tobs = YRSID_SI
    dt = 2.5

    data_input_path = "/data/asantini/globalfit/MOJITO_DATA/mojito_light_2p5s/"
    base_file_name = "psd_noise_gal_fd"
    file_store_dir = "/work/fpozzoli/test_lisatools/LISAanalysistools/" + "mojito_output/"

    gpus = [cp.cuda.runtime.getDevice()]
    cp.cuda.runtime.setDevice(gpus[0])

    nwalkers = 30
    ntemps = 4

    winalpha = 0.1
    wintype = "tukey"

    # FD domain -> stationary (single-time-bin) galaxy response.
    basis_domain = "fd"

    processor_init_kwargs = dict(
        L1_folder=data_input_path,
        source_types=['noise', 'gb'],
        source_ids={'gb': [0]},
        verbose=True,
        do_plots=True,
    )

    preprocess_kwargs = dict(normalize=True)

    sensitivity_init_kwargs = dict(
        tdi_generation=2,
        mask_percentage=0.02,
        use_splines=False,
    )

    galactic_grid_kwargs = None

    # Domain communicated by settings factory, not a string flag (sprint
    # rule): the engine calls ``factory(times, dt, force_backend)`` after
    # loading the data so the grid is sized against the real time array.
    assert basis_domain != "stft", "stft basis needs an stft_dt"
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
        winalpha=winalpha,
        wintype=wintype,
        gpus=gpus,
        data_processor_class=L1ProcessingStep,
        processor_init_kwargs=processor_init_kwargs,
        preprocess_kwargs=preprocess_kwargs,
        sensitivity_init_kwargs=sensitivity_init_kwargs,
        galactic_grid_kwargs=galactic_grid_kwargs,
    )

    general_setup = GeneralSetup(general_settings)
    # Band/STFT metadata consumed by the per-source setup functions
    # (no longer GeneralSettings fields post-merge; the analysis band
    # itself lives on domain_settings).
    general_setup.start_freq = start_freq
    general_setup.end_freq = end_freq
    return general_setup


def get_global_fit_settings(copy_settings_file=False):

    general_setup = get_general_erebor_settings()

    if copy_settings_file:
        shutil.copy(
            __file__,
            general_setup.file_store_dir + general_setup.base_file_name + "_" + __file__.split("/")[-1]
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

    psd_setup, galfor_setup = get_psd_erebor_settings(general_setup)

    ##############
    ## READ OUT ##
    ##############

    global_settings = GlobalFitSettings(
        source_info={
            "psd": psd_setup,
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
