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
from lisatools.utils.constants import *
#from gbgpu.utils.utility import get_fdot
from eryn.state import BranchSupplemental
from lisatools.globalfit.hdfbackend import GFHDFBackend, GBHDFBackend, MBHHDFBackend, EMRIHDFBackend
from lisatools.globalfit.utils import SetupInfoTransfer, AllSetupInfoTransfer
from lisatools.globalfit.run import CurrentInfoGlobalFit, GlobalFit
# from global_fit_input.global_fit_settings import get_global_fit_settings

from lisatools.globalfit.state import GFBranchInfo, AllGFBranchInfo
from lisatools.globalfit.state import MBHState, EMRIState, GBState

#from bbhx.utils.transform import *

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
from scipy.stats import multivariate_normal

from eryn.moves import StretchMove
from lisatools.sampling.moves.skymodehop import SkyMove

from eryn.moves import CombineMove
from lisatools.globalfit.moves import GBSpecialStretchMove, GBSpecialRJRefitMove, GBSpecialRJSearchMove, GBSpecialRJPriorMove, MBHSpecialMove, GBSpecialRJSerialSearchMCMC, GFCombineMove
from lisatools.globalfit.galaxyglobal import make_gmm
from lisatools.globalfit.moves import GlobalFitMove
from lisatools.utils.utility import tukey


# import few
from lisatools.globalfit.engine import GlobalFitSettings, GeneralSetup, GeneralSettings, RankInfo


from eryn.utils.updates import Update

from lisatools.globalfit.preprocessing import L1ProcessingStep
from lisatools.globalfit.recipe_steps import SearchRecipeStep, PERecipeStep, build_psd_moves


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
    if general_set.use_splines: # [Somsi,Stmi, ...S, f oms internal S, f tm ... Somsn,Stmn ]
        priors_psd = {}
        for mdl in ["oms", "tm"]:
            priors_psd[rf"$\log_{{10}}S_{{\rm {mdl}, i}}$"] = multivariate_normal( 0.0, .1) # uniform_dist(general_set.lowerbound, general_set.upperbound)
            for nk in range(1, general_set.nknots-1):
                priors_psd[rf"$\log_{{10}}S_{{\rm {mdl}, {nk}}}$"] = multivariate_normal( 0.0, .1) # uniform_dist(general_set.lowerbound, general_set.upperbound) 
                priors_psd[rf"$\log_{{10}}f_{{\rm {mdl}, {nk}}}$"] = uniform_dist(np.log10(general_set.start_freq), np.log10(general_set.end_freq))
            priors_psd[rf"$\log_{{10}}S_{{\rm {mdl}, e}}$"] = multivariate_normal( 0.0, .1) # uniform_dist(general_set.lowerbound, general_set.upperbound)

        noise_fill_dict = {
                rf"$\log_{{10}}f_{{\rm oms, i}}$": np.log10(general_set.start_freq), 
                rf"$\log_{{10}}f_{{\rm oms, e}}$": np.log10(general_set.end_freq),  
                rf"$\log_{{10}}f_{{\rm tm, i}}$": np.log10(general_set.start_freq), 
                rf"$\log_{{10}}f_{{\rm tm, e}}$": np.log10(general_set.end_freq), 
                r"$S_{\rm oms}$": 15e-12,  
                r"$S_{\rm tm}$": 3e-15,  
            }
        output_basis = [r'$S_{\rm oms}$', r'$S_{\rm tm}$'] + list(priors_psd.keys())
        output_basis.insert(3, rf"$\log_{{10}}f_{{\rm oms, i}}$")
        output_basis.insert(2+2*general_set.nknots-1, rf"$\log_{{10}}f_{{\rm oms, e}}$")
        output_basis.insert(2+2*general_set.nknots+1, rf"$\log_{{10}}f_{{\rm tm, i}}$")
        output_basis.append(rf"$\log_{{10}}f_{{\rm tm, e}}$")
        transfcont = TransformContainer(
                input_basis=list(priors_psd.keys()),
                output_basis=output_basis,
                parameter_transforms=None,
                fill_dict=noise_fill_dict,
            )
    else:
        priors_psd = {
                    r'$S_{\rm oms}$': uniform_dist(6.0e-12, 20.0e-11), # Soms_d
                    r'$S_{\rm tm}$': uniform_dist(1.0e-15, 20.0e-14),  # Sa_a
                }
        transfcont = None

    priors = {"psd": ProbDistContainer(priors_psd)}

    psd_settings = PSDSettings(
        Tobs=general_set.Tobs,
        dt=general_set.dt,
        initialize_kwargs=initialize_kwargs_psd,
        priors=priors,
        ndim=len(priors_psd),
        transform_fn=transfcont,
    )

    return PSDSetup(psd_settings)


def get_general_erebor_settings(use_splines=True) -> GeneralSetup:
       # limits on parameters
    delta_safe = 1e-5
    # now with negative fdots
    
    from lisatools.utils.constants import YRSID_SI
    Tobs = .5 * YRSID_SI # / 12.0 # .1 * YRSID_SI / 12.0
    dt = 2.5

    head_dir = "/home/karnesis/work/Git/LISAanalysistools/"
    #ldc_source_file = head_dir + "emri_sangria_injection.h5"
    data_input_path = "/mnt/wd_hdd_6TB/nikos/DATA/global_fit/mojito_lite/"
    base_file_name = "noise-only_spline-1yr"
    file_store_dir = "/mnt/wd_hdd_6TB/nikos/DATA/global_fit/gf_output/splines_1_yr/"

    # TODO: connect LISA to SSB for MBHs to numerical orbits

    gpus = [cp.cuda.runtime.getDevice()]
    cp.cuda.runtime.setDevice(gpus[0])
    # Restrict JAX to only see the target GPU — must be set before JAX backend init
    import jax
    jax.config.update("jax_cuda_visible_devices", str(gpus[0]))
    # few.get_backend('cuda12x')
    nwalkers = 40
    ntemps = 4

    winalpha = 0.1 # 0.1 # bh tryout
    wintype = "tukey"

    basis_domain = "stft" # fd
    stft_dt = 7 * 24 * 3600.0  # how many hours

    processor_init_kwargs = dict(L1_folder=data_input_path,
                                 source_types=['noise'],
                                 verbose=True,
                                 do_plots=True,
                                )
    
    preprocess_kwargs = dict(normalize=True)

    sensitivity_init_kwargs = dict(tdi_generation=2, 
                                   mask_percentage=0.02,
                                   use_splines=use_splines)

    general_settings = GeneralSettings(
        Tobs=Tobs,
        dt=dt,
        file_store_dir=file_store_dir,
        base_file_name=base_file_name,
        start_freq=1e-4,
        end_freq=2e-2,
        basis_domain=basis_domain,
        stft_dt=stft_dt,
        random_seed=12345,
        backup_iter=5,
        nwalkers=nwalkers,
        ntemps=ntemps,
        winalpha=winalpha,
        wintype=wintype,
        use_splines=use_splines, 
        nknots=6,
        lowerbound=-1, 
        upperbound=1,
        gpus=gpus,
        data_processor=L1ProcessingStep,
        processor_init_kwargs=processor_init_kwargs,
        preprocess_kwargs=preprocess_kwargs,
        sensitivity_init_kwargs=sensitivity_init_kwargs,
        #remove_from_data=["mbhb", "dgb", "igb", "vgb"],
        #channels=["X", "Y", "Z"],  # , "T"
        #noise_model=sangria
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
