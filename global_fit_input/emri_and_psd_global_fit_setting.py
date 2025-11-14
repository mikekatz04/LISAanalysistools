import h5py
import numpy as np
import shutil

try:
    import cupy as cp
    gpu_available = True
except (ModuleNotFoundError, ImportError) as e:
    import numpy as cp
    gpu_available = True

from eryn.moves.tempering import TemperatureControl, make_ladder

from lisatools.detector import EqualArmlengthOrbits
from eryn.moves import TemperatureControl
from lisatools.utils.constants import *
from gbgpu.utils.utility import get_fdot
from eryn.state import BranchSupplemental
from lisatools.globalfit.hdfbackend import GFHDFBackend, GBHDFBackend, MBHHDFBackend, EMRIHDFBackend
from lisatools.globalfit.utils import SetupInfoTransfer, AllSetupInfoTransfer
from lisatools.globalfit.run import CurrentInfoGlobalFit, GlobalFit
# from global_fit_input.global_fit_settings import get_global_fit_settings

from lisatools.globalfit.state import GFBranchInfo, AllGFBranchInfo
from lisatools.globalfit.state import MBHState, EMRIState, GBState

from bbhx.utils.transform import *

from lisatools.globalfit.generatefuncs import *
from lisatools.utils.utility import AET
from lisatools.sampling.prior import SNRPrior, AmplitudeFromSNR, AmplitudeFrequencySNRPrior, GBPriorWrap

from lisatools.globalfit.stock.erebor import (
    GalForSetup, GalForSettings, PSDSetup, PSDSettings,
    MBHSetup, MBHSettings, GBSetup, GBSettings, EMRISetup, EMRISettings,
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
from lisatools.globalfit.engine import GlobalFitSettings, GeneralSetup, GeneralSettings


from eryn.utils.updates import Update

from lisatools.globalfit.recipe import Recipe, RecipeStep
import time


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
    
class EMRIPERecipeStep(RecipeStep):
    def __init__(self, *args, moves=None, weights=None, **kwargs):
        super().__init__(moves=moves, weights=weights)
    
    def setup_run(self, iteration, last_sample, sampler):
        # making sure
        sampler.moves = self.moves
        sampler.weights = self.weights

    def stopping_function(self, iteration, last_sample, sampler):
        return False


from lisatools.sampling.stopping import SearchConvergeStopping


################

### DEFINE RECIPE

#############


def setup_recipe(recipe, engine_info, curr, acs, priors, state):

    from lisatools.sources.emri import EMRITDIWaveform  
   
    emri_info = curr.source_info["emri"]
    # TODO: adjust this indide current info
    general_info = curr.general_info
    nwalkers = curr.general_info.nwalkers
    ntemps = curr.general_info.ntemps

    gpus = curr.general_info.gpus
    cp.cuda.runtime.setDevice(gpus[0])
    
    # setup psd search move
    effective_ndim = engine_info.ndims["psd"]  #  + engine_info.ndims["galfor"]
    temperature_control = TemperatureControl(effective_ndim, nwalkers, ntemps=ntemps, Tmax=np.inf, permute=False)
    
    psd_move_args = (acs, priors)

    psd_move_kwargs = dict(
        num_repeats=500,
        live_dangerously=True,
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
    #recipe.add_recipe_component(PSDPERecipeStep(moves=[psd_pe_move]), name="psd pe")

    wave_gen = WrapEMRI(EMRITDIWaveform(**emri_info.initialize_kwargs), acs.nchannels, curr.general_info.tukey_alpha, curr.general_info.start_freq_ind, curr.general_info.end_freq_ind, curr.general_info.dt)
    
    betas_all = np.tile(make_ladder(emri_info.ndim, ntemps=ntemps), (emri_info.nleaves_max, 1))

    # to make the states work 
    betas = betas_all[0]
    state.sub_states["emri"].betas_all = betas_all

    inner_moves = emri_info.inner_moves
    tempering_kwargs = dict(ntemps=ntemps, Tmax=np.inf, permute=False)
    
    coords_shape = (ntemps, nwalkers, emri_info.nleaves_max, emri_info.ndim)

    emri_move_args = (
        "emri",
        coords_shape,
        wave_gen,
        tempering_kwargs,
        emri_info.waveform_kwargs.copy(),
        emri_info.waveform_kwargs.copy(),
        acs,
        emri_info.num_prop_repeats,
        emri_info.transform,
        priors,
        inner_moves,
        acs.df
    )
    emri_pe_move = ResidualAddOneRemoveOneMove(*emri_move_args)

    emri_pe_moves = GFCombineMove(moves=[emri_pe_move, psd_pe_move], share_temperature_control=False)
    emri_pe_moves.accepted = np.zeros((ntemps, nwalkers), dtype=int)

    recipe.add_recipe_component(EMRIPERecipeStep(moves=[emri_pe_moves]), name="emri pe")


##########################
##### SETTINGS ###########
##########################

class WrapEMRI:
    def __init__(self, waveform_gen_td, nchannels, tukey_alpha, start_freq_ind, end_freq_ind, dt):
        self.waveform_gen_td = waveform_gen_td
        self.tukey_alpha = tukey_alpha
        self.nchannels = nchannels
        self.start_freq_ind, self.end_freq_ind = start_freq_ind, end_freq_ind
        self.dt = dt

    def __call__(self, *args, **kwargs):
        AET_t = cp.asarray(self.waveform_gen_td(*args, **kwargs))
        fft_input = AET_t * tukey(AET_t.shape[-1], self.tukey_alpha, xp=cp)[None, :]
        # TODO: adjust this if it needs 3rd axis?
        AET_f = self.dt * cp.fft.rfft(fft_input, axis=-1)[:self.nchannels, self.start_freq_ind: self.end_freq_ind]
        return AET_f

def get_emri_erebor_settings(general_set: GeneralSetup) -> EMRISetup:

    injection_parameters_file = '/data/asantini/packages/LISAanalysistools/injection_params.npz'
    delta_prior = 1e-2

    gpu_orbits = EqualArmlengthOrbits(force_backend="cuda12x")
    # waveform kwargs
    response_kwargs = dict(
        t0=30000.0,
        order=25,
        tdi="1st generation",
        tdi_chan="AE",
        orbits=gpu_orbits,
        force_backend="cuda12x",
        remove_garbage="zero",  # removes the beginning of the signal that has bad information
    )

    # TODO: I prepared this for Kerr but have not used it with the Kerr waveform yet
    # so spin results are currently meaningless and will lead to slower code
    # waveform kwargs
    initialize_kwargs_emri = dict(
        T=general_set.Tobs / YRSID_SI, # TODO: check these conversions all align
        dt=general_set.dt,
        emri_waveform_args=("FastKerrEccentricEquatorialFlux",),
        emri_waveform_kwargs=dict(force_backend="cuda12x"),
        response_kwargs=response_kwargs,
    )

    injection_params = np.load(injection_parameters_file)['injection_params']
    fill_values = np.array([injection_params[5], injection_params[12]])

    injection_sampling = deepcopy(injection_params)
    injection_sampling[0] = np.log(injection_sampling[0])  # log mass
    injection_sampling[7] = np.cos(injection_sampling[7])  # cos qK
    injection_sampling[9] = np.cos(injection_sampling[9])  # cos qS

    injection_sampling = np.delete(injection_sampling, [5, 12])
    

    emri_settings = EMRISettings(
        Tobs=general_set.Tobs,
        dt=general_set.dt,
        fill_values=fill_values,
        injection=injection_sampling,
        delta_prior=delta_prior,
        initialize_kwargs=initialize_kwargs_emri,
        nleaves_max=1,
        nleaves_min=1,
        ndim=12
    )

    return EMRISetup(emri_settings)

def get_psd_erebor_settings(general_set: GeneralSetup) -> PSDSetup:
    
    # waveform kwargs
    initialize_kwargs_psd = dict()

    from lisatools.utils.constants import YRSID_SI
    Tobs = YRSID_SI
    dt = 10.0

    psd_settings = PSDSettings(
        Tobs=general_set.Tobs,
        dt=general_set.dt,
        initialize_kwargs=initialize_kwargs_psd,
    )

    return PSDSetup(psd_settings)



# def get_galfor_erebor_settings(general_set: GeneralSetup) -> GalForSetup:
    
#     from lisatools.detector import EqualArmlengthOrbits

#     from lisatools.utils.constants import YRSID_SI
#     Tobs = YRSID_SI
#     dt = 10.0

#     galfor_settings = GalForSettings(
#         Tobs=general_set.Tobs,
#         dt=general_set.dt,
#         initialize_kwargs={},
#     )

#     return GalForSetup(galfor_settings)



def get_general_erebor_settings() -> GeneralSetup:
       # limits on parameters
    delta_safe = 1e-5
    # now with negative fdots
    
    from lisatools.utils.constants import YRSID_SI
    Tobs = YRSID_SI * 2.0 / 12.0
    dt = 10.0

    emri_source_file = "/data/asantini/packages/LISAanalysistools/emri_sangria_injection.h5"
    base_file_name = "emri_psd_3rd_try"
    file_store_dir = "/data/asantini/packages/LISAanalysistools/global_fit_output/"

    # TODO: connect LISA to SSB for MBHs to numerical orbits

    gpus = [5]
    cp.cuda.runtime.setDevice(gpus[0])
    # few.get_backend('cuda12x')
    nwalkers = 36
    ntemps = 2

    tukey_alpha = 0.05

    orbits = EqualArmlengthOrbits()
    gpu_orbits = EqualArmlengthOrbits(force_backend="cuda12x")

    general_settings = GeneralSettings(
        Tobs=Tobs,
        dt=dt,
        file_store_dir=file_store_dir,
        base_file_name=base_file_name,
        data_input_path=emri_source_file,
        orbits=orbits,
        gpu_orbits=gpu_orbits, 
        start_freq_ind=0,
        end_freq_ind=None,
        random_seed=103209,
        backup_iter=5,
        nwalkers=nwalkers,
        ntemps=ntemps,
        tukey_alpha=tukey_alpha,
        gpus=gpus,
        remove_from_data=[],
    )

    general_setup = GeneralSetup(general_settings)
    return general_setup


from lisatools.globalfit.engine import RankInfo


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
    ###  EMRI Settings  ##############
    ##################################
    ##################################

    emri_setup = get_emri_erebor_settings(general_setup)

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
            "emri": emri_setup,
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
