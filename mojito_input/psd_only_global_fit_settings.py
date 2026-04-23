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

from lisatools.detector import EqualArmlengthOrbits

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


#from bbhx.utils.transform import *












def setup_recipe(recipe, engine_info, curr, acs, priors, state):
    cp.cuda.runtime.setDevice(curr.general_info.gpus[0])

    psd_search_move, psd_pe_move = build_psd_moves(engine_info, curr, acs, priors)

    recipe.add_recipe_component(SearchRecipeStep(moves=[psd_search_move]), name="psd search")
    recipe.add_recipe_component(PERecipeStep(moves=[psd_pe_move]), name="psd pe")
    
    
#######################
##### SETTINGS ###########
###############


def get_psd_erebor_settings(general_set: GeneralSetup) -> PSDSetup:
    
    # waveform kwargs
    initialize_kwargs_psd = dict()

    priors_psd = {
                r'$S_{\rm oms}$': uniform_dist(6.0e-12, 20.0e-11),  # Soms_d
                r'$S_{\rm tm}$': uniform_dist(1.0e-15, 20.0e-14),  # Sa_a
            }
    priors = {"psd": ProbDistContainer(priors_psd)}

    psd_settings = PSDSettings(
        Tobs=general_set.Tobs,
        dt=general_set.dt,
        initialize_kwargs=initialize_kwargs_psd,
        priors=priors,
        ndim=2
    )

    return PSDSetup(psd_settings)


def get_general_erebor_settings() -> GeneralSetup:
    # now with negative fdot
    Tobs = 1. * YRSID_SI / 12.0
    dt = 2.5
    start_freq = 5e-5
    end_freq = 1e-1

    head_dir = "/data/asantini/packages/LISAanalysistools/"
    data_input_path = "/data/asantini/globalfit/MOJITO_DATA/mojito_light_2p5s/"
    base_file_name = "matrix_tryout"
    file_store_dir = head_dir + "mojito_output/"

    # TODO: connect LISA to SSB for MBHs to numerical orbits

    gpus = [3]
    cp.cuda.runtime.setDevice(gpus[0])
    # Restrict JAX to only see the target GPU — must be set before JAX backend init
    import jax
    jax.config.update("jax_cuda_visible_devices", ",".join(str(gpu) for gpu in gpus))
    # few.get_backend('cuda12x')
    nwalkers = 30
    ntemps = 4

    window_type = "tukey"
    window_taper_duration = 1 / start_freq
    normalize_window = True

    basis_domain = "stft" # fd
    stft_dt = 24 * 3600.0  if basis_domain == "stft" else None # how many hours

    processor_init_kwargs = dict(L1_folder=data_input_path,
                                 source_types=['noise'],
                                 verbose=True,
                                 do_plots=True,
                                )

    downsample_kwargs = {
        "target_fs": 0.2,  # Hz — target sampling rate (None = no downsampling).
        "window": ("kaiser", 31.0)  # Kaiser window beta parameter (higher = more aggressive anti-aliasing)
    }

    highpass_kwargs = {
        'cutoff': 1e-5,  # Hz — highpass cutoff frequency
        'order': 2,  # Butterworth filter order
        'zero_phase': True
    }

    lowpass_kwargs = {
        'cutoff': 1e-1,  # Hz — lowpass cutoff frequency
        'order': 2,  # Butterworth filter order
        'zero_phase': True
    }

    trim_kwargs = {
        'duration': 200 * 3600,  # seconds — duration to trim from each end
        'is_percent': False,  # If True, 'duration' is interpreted as a percentage of the total signal length
        'trimming_type': "from_each_end"  # "from_each_end" or "from_start"
    }
    
    preprocess_kwargs = dict(
        highpass_kwargs=highpass_kwargs,
        lowpass_kwargs=lowpass_kwargs,
        trim_kwargs=trim_kwargs,
        downsample_kwargs=downsample_kwargs,
        Tobs=Tobs
    )

    sensitivity_init_kwargs = dict(tdi_generation=2, 
                                   mask_percentage=0.02,
                                   use_splines=False)

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
    )

    general_setup = GeneralSetup(general_settings)
    return general_setup


def get_global_fit_settings(copy_settings_file=False):

    general_setup = get_general_erebor_settings()

    # file_information["past_file_for_start"] = file_store_dir + "rework_6th_run_through" + "_parameter_estimation_main.h5"
    if copy_settings_file:
        shutil.copy(__file__, general_setup.file_store_dir + general_setup.base_file_name + "_" + __file__.split("/")[-1])

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
    ###  PSD Settings  ###############
    ##################################
    ##################################


    psd_setup = get_psd_erebor_settings(general_setup)

    ##############
    ## READ OUT ##
    ##############


    global_settings = GlobalFitSettings(
        source_info={
            "psd": psd_setup,
            # "emri": all_emri_info,
        },
        general_info=general_setup,
        rank_info=rank_info,
        setup_function=setup_recipe,
    )

    curr_info = CurrentInfoGlobalFit(global_settings)

    return curr_info



if __name__ == "__main__":
    settings = get_global_fit_settings()
    # breakpoint()
