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

from eryn.moves.tempering import TemperatureControl, make_ladder

from lisatools.detector import L1Orbits
from eryn.moves import TemperatureControl
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

from eryn.moves import StretchMove
from lisatools.sampling.moves.skymodehop import SkyMove

from eryn.moves import CombineMove
from lisatools.globalfit.moves import GBSpecialStretchMove, GBSpecialRJRefitMove, GBSpecialRJSearchMove, GBSpecialRJPriorMove, PSDMove, MBHSpecialMove, ResidualAddOneRemoveOneMove, GBSpecialRJSerialSearchMCMC, GFCombineMove
from lisatools.globalfit.galaxyglobal import make_gmm
from lisatools.globalfit.moves import GlobalFitMove
from lisatools.utils.utility import tukey


# import few
from lisatools.globalfit.engine import GlobalFitSettings, GeneralSetup, GeneralSettings, RankInfo

from eryn.utils.updates import Update

from lisatools.globalfit.recipe import Recipe, RecipeStep
import time

from lisatools.globalfit.preprocessing import L1ProcessingStep

from lisaconstants import ASTRONOMICAL_YEAR as YRSID_SI


################

### DEFINE RECIPE

#############


class PSDSearchRecipeStep(RecipeStep):
    def setup_run(self, iteration, last_sample, sampler):
        # making sure
        sampler.moves = self.moves
        sampler.weights = self.weights

    def stopping_function(self, iteration, last_sample, sampler):
        # this will already be converged to max logl
        return True


class PSDPERecipeStep(RecipeStep):
    def setup_run(self, iteration, last_sample, sampler):
        # making sure
        sampler.moves = self.moves
        sampler.weights = self.weights

    def stopping_function(self, iteration, last_sample, sampler):
        # this will already be converged to max logl
        return False


from lisatools.sampling.stopping import SearchConvergeStopping


################

### DEFINE RECIPE

#############


def setup_recipe(recipe, engine_info, curr, acs, priors, state):
   
    # TODO: adjust this indide current info
    general_info = curr.general_info
    nwalkers = curr.general_info.nwalkers
    ntemps = curr.general_info.ntemps

    psd_info = curr.source_info["psd"]
    mbh_info = curr.source_info["mbh"]

    gpus = curr.general_info.gpus
    cp.cuda.runtime.setDevice(gpus[0])
    
    # setup psd search move
    effective_ndim = engine_info.ndims["psd"]  #  + engine_info.ndims["galfor"]
    Tmax = 1e6
    temperature_control = TemperatureControl(effective_ndim, nwalkers, ntemps=ntemps, Tmax=Tmax, permute=False)
    
    psd_move_args = (acs, priors)

    psd_move_kwargs = dict(
        num_repeats=60,
        live_dangerously=True,
        psd_transform_fn = psd_info.transform_fn,
        sensitivity_backend = general_info.sensitivity_backend,
        # gibbs_sampling_setup=[{
        #     "psd": np.ones((1, engine_info.ndims["psd"]), dtype=bool),
        #     "galfor": np.ones((1, engine_info.ndims["galfor"]), dtype=bool)
        # }],
        temperature_control=temperature_control
    )
    
    psd_search_move = PSDMove(
        *psd_move_args, 
        max_logl_mode=True,
        name="psd search move",
        **psd_move_kwargs,
    )

    psd_pe_move = PSDMove(
        *psd_move_args, 
        max_logl_mode=False,
        name="psd pe move",
        **psd_move_kwargs,
    )
    # TODO: put this under the hood
    psd_search_move.accepted = np.zeros((ntemps, nwalkers))
    psd_pe_move.accepted = np.zeros((ntemps, nwalkers))

    recipe.add_recipe_component(PSDSearchRecipeStep(moves=[psd_search_move]), name="psd search")
    recipe.add_recipe_component(PSDPERecipeStep(moves=[psd_pe_move]), name="psd pe")

    # now do the black holes

    from lisatools.sources.bbh.waveform import PhenomTHMTDIWaveform

    wave_gen = PhenomTHMTDIWaveform(**mbh_info.initialize_kwargs)

    if np.any(mbh_inds := state.branches_inds["mbh"][0]):
        for leaf in range(mbh_inds.shape[-1]):
            if mbh_inds[0, leaf]:
                assert np.all(mbh_inds[:, leaf])
                inj_coords = state.branches_coords["mbh"][0, :, leaf]
                inj_coords_in = mbh_info.transform.both_transforms(inj_coords)
                
                signals_in = wave_gen(*inj_coords_in.T, **mbh_info.waveform_kwargs)
                breakpoint()
                acs.add_signal_to_residual(signals_in)
    
    betas_all = np.tile(make_ladder(mbh_info.ndim, ntemps=ntemps), (mbh_info.nleaves_max, 1))
    state.sub_states["mbh"].betas_all = betas_all

    inner_moves = mbh_info.inner_moves
    tempering_kwargs = dict(ntemps=ntemps, Tmax=np.inf, permute=False)
    coords_shape = (ntemps, nwalkers, mbh_info.nleaves_max, mbh_info.ndim)

    mbh_move_args = (
        "mbh",  # branch_name,
        coords_shape,
        wave_gen,
        tempering_kwargs,
        mbh_info.waveform_kwargs.copy(),  # waveform_gen_kwargs,
        mbh_info.waveform_kwargs.copy(),  # waveform_like_kwargs,
        acs,
        mbh_info.num_prop_repeats,
        mbh_info.transform,
        priors,
        inner_moves,
    )

    mbh_pe_move = ResidualAddOneRemoveOneMove(*mbh_move_args)
    mbh_pe_move.accepted = np.zeros((ntemps, nwalkers))

    
#######################
##### SETTINGS ###########
###############

def get_mbh_global_fit_settings(general_set: GeneralSetup) -> MBHSetup:
    
    hms = [21, 33, 44]

    tlowfit = True # use a fit to set the starting time of the root finder used in t(f)
    tol = 1e-12 # root finding tolerance

    wave_kwargs = dict(
        higher_modes=hms,
            include_negative_modes=True, # negative m modes will be produced by simmetry
            t_low_fit=tlowfit,
            coarse_grain=False, # if false it will generate the waveform on a dense time grid with the specified timestep
            atol=tol,
            rtol=tol,
    )

    response_kwargs = dict(
        sampling_frequency=1.0/general_set.dt,
        tdi='2nd generation',
        orbits=general_set.gpu_orbits if gpu_available else general_set.orbits,
        order=30
    )

    waveform_init_kwargs = dict(
        waveform_kwargs=wave_kwargs,
        response_kwargs=response_kwargs,
        waveform_t0 = 97729089.327664,
        data_t0 = general_set.data_t0,
        dt = general_set.dt,
        Tobs = 3. / 12., # this is only for the waveform generation, not the data, which is still general_set.Tobs
        start_freq=5e-5,
        ref_freq=2.0886886878886526e-05, # source 18
        buffer_time=3000,
        tukey_alpha=general_set.tukey_alpha,
        is_tref=False,
        force_backend=general_set.force_backend,
    )


    waveform_runtime_kwargs = dict(
        # this is for the waveform generation during the run, not the initialization
        output_domain=general_set.basis_domain.capitalize(),
        domain_kwargs=general_set.basis_kwargs,
    )

    mbh_settings = MBHSettings(
        Tobs=general_set.Tobs,
        dt=general_set.dt,
        initialize_kwargs=waveform_init_kwargs,
        waveform_kwargs=waveform_runtime_kwargs,
        nleaves_max=1,
        nleaves_min=1,
        ndim=11
    )

    return MBHSetup(mbh_settings)



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
       # limits on parameters
    # now with negative fdots

    source_ids = [18]
    
    Tobs = 9. * YRSID_SI / 12.0
    dt = 2.5

    head_dir = "/data/asantini/packages/LISAanalysistools/"
    #ldc_source_file = head_dir + "emri_sangria_injection.h5"
    data_input_path = "/data/asantini/globalfit/MOJITO_DATA/mojito_light_2p5s/"
    base_file_name = "mbh_psd_separate_1st_try"
    file_store_dir = head_dir + "mojito_output/"

    gpus = [2]
    cp.cuda.runtime.setDevice(gpus[0])
    backend="cuda12x" if gpus is not None else "cpu"
    nwalkers = 20
    ntemps = 10

    tukey_alpha = 0.05

    basis_domain = "stft"
    stft_dt = 6 * 3600.0  # hours

    processor_init_kwargs = dict(L1_folder=data_input_path,
                                 source_types=['noise', 'mbhb'],
                                 source_ids=dict(mbhb=source_ids),
                                 verbose=True,
                                 do_plots=True,
                                 orbits_class=L1Orbits,
                                 orbirs_kwargs=dict(force_backend=backend, frame="icrs")
                                )
    
    preprocess_kwargs = dict(plot_folder=file_store_dir)

    sensitivity_init_kwargs = dict(tdi_generation=2, mask_percentage=0.02)

    general_settings = GeneralSettings(
        Tobs=Tobs,
        dt=dt,
        file_store_dir=file_store_dir,
        base_file_name=base_file_name,
        start_freq=1e-5,
        end_freq=1e-1,
        basis_domain=basis_domain,
        stft_dt=stft_dt,
        random_seed=103209,
        backup_iter=5,
        nwalkers=nwalkers,
        ntemps=ntemps,
        tukey_alpha=tukey_alpha,
        gpus=gpus,
        data_processor=L1ProcessingStep,
        processor_init_kwargs=processor_init_kwargs,
        preprocess_kwargs=preprocess_kwargs,
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
    ###  MBH Settings  ###############
    ##################################
    ##################################


    mbh_setup = get_mbh_erebor_settings(general_setup)
    

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
            "mbh": mbh_setup,
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
    breakpoint()
