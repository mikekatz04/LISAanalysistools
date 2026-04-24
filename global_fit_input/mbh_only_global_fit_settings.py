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



# basic transform functions for pickling
def mbh_dist_trans(x):
    return x * PC_SI * 1e9  # Gpc

from eryn.utils.updates import Update

from lisatools.globalfit.recipe import Recipe, RecipeStep
import time

from lisatools.globalfit.engine import GlobalFitSettings, GeneralSetup, GeneralSettings


################

### DEFINE RECIPE

#############


class MBHSearchStep(RecipeStep):

    def __init__(self, *args, moves=None, weights=None, **kwargs):
        super().__init__(moves=moves, weights=weights)

    def setup_run(self, iteration, last_sample, sampler):
        # making sure
        sampler.moves = self.moves
        sampler.weights = self.weights
        
    def stopping_function(self, *args, **kwargs):
        # this will already be converged to max logl
        # I think it should be just True becuase 
        # the inner proposal is taking care of this. 
        return True  # self.stopper(*args, **kwargs)


class MBHPEStep(RecipeStep):

    def __init__(self, *args, moves=None, weights=None, **kwargs):
        super().__init__(moves=moves, weights=weights)

    def setup_run(self, iteration, last_sample, sampler):
        # making sure
        sampler.moves = self.moves
        sampler.weights = self.weights
        
    def stopping_function(self, *args, **kwargs):
        # this will already be converged to max logl
        # I think it should be just True becuase 
        # the inner proposal is taking care of this. 
        return False  # self.stopper(*args, **kwargs)


################

### DEFINE RECIPE

#############


def setup_recipe(recipe, engine_info, curr, acs, priors, state):
    # _ = setup_mbh_functionality(engine_info, curr, acs, priors, state):
    # _ = setup_emri_functionality(engine_info, curr, acs, priors, state):
    mbh_info = curr.source_info["mbh"]
    # TODO: adjust this indide current info
    general_info = curr.general_info
    nwalkers = curr.general_info.nwalkers
    ntemps = curr.general_info.ntemps

    gpus = curr.general_info.gpus
    cp.cuda.runtime.setDevice(gpus[0])
    
    mbh_info = curr.source_info["mbh"]
    from bbhx.waveformbuild import BBHWaveformFD

    wave_gen = BBHWaveformFD(
        **mbh_info.initialize_kwargs
    )

    if np.any(mbh_inds := state.branches_inds["mbh"][0]):
        for leaf in range(mbh_inds.shape[-1]):
            if mbh_inds[0, leaf]:
                assert np.all(mbh_inds[:, leaf])
                inj_coords = state.branches_coords["mbh"][0, :, leaf]
                inj_coords_in = mbh_info.transform.both_transforms(inj_coords)
                # TODO: fix freqs input with backend
                AET = wave_gen(*inj_coords_in.T, fill=True, freqs=cp.asarray(acs.f_arr),**mbh_info.waveform_kwargs)
                acs.add_signal_to_residual(AET[:, :2])

    if False:  # hasattr(state, "betas_all") and state.betas_all is not None:
            betas_all = state.sub_states["mbh"].betas_all
    else:
        print("remove the False above")
        betas_all = np.tile(make_ladder(mbh_info.ndim, ntemps=ntemps), (mbh_info.nleaves_max, 1))

    # to make the states work 
    betas = betas_all[0]
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
        acs.df
    )

    search_fp = general_info.file_store_dir + general_info.base_file_name + mbh_info.mbh_search_file_key + ".h5"
    mbh_search_move = MBHSpecialMove(*mbh_move_args, name="mbh_search", run_search=True, force_backend="cuda12x", file_backend=curr.backend, search_fp=search_fp)
    mbh_pe_move = MBHSpecialMove(*mbh_move_args, name="mbh_pe", run_search=False, force_backend="cuda12x")
    mbh_search_move.accepted = np.zeros((ntemps, nwalkers), dtype=int)
    mbh_pe_move.accepted = np.zeros((ntemps, nwalkers), dtype=int)
    recipe.add_recipe_component(MBHSearchStep(moves=[mbh_search_move], n_iters=5, verbose=True), name="mbh search")
    recipe.add_recipe_component(MBHPEStep(moves=[mbh_pe_move], n_iters=5, verbose=True), name="mbh pe")
    
#######################
##### SETTINGS ###########
###############


def get_mbh_erebor_settings(general_set: GeneralSetup) -> MBHSetup:
    
    gpu_orbits = EqualArmlengthOrbits(force_backend="cuda12x")
    # waveform kwargs
    initialize_kwargs_mbh = dict(
        amp_phase_kwargs=dict(run_phenomd=True),
        response_kwargs=dict(TDItag="AET", orbits=gpu_orbits),
        force_backend="cuda12x",
    )

    from lisatools.utils.constants import YRSID_SI
    Tobs = YRSID_SI
    dt = 10.0

    mbh_settings = MBHSettings(
        Tobs=general_set.Tobs,
        dt=general_set.dt,
        initialize_kwargs=initialize_kwargs_mbh,
        nleaves_max=15,
        nleaves_min=0,
        ndim=11
    )

    return MBHSetup(mbh_settings)


from lisatools.detector import sangria


def get_general_erebor_settings() -> GeneralSetup:
       # limits on parameters
    delta_safe = 1e-5
    # now with negative fdots
    
    from lisatools.utils.constants import YRSID_SI
    Tobs = 2. * YRSID_SI / 12.0
    dt = 10.0

    ldc_source_file = "/scratch/335-lisa/mlkatz/LDC2_sangria_training_v2.h5"
    base_file_name = "mbh_separate_1st_try_parameter_estimation_main.h5"
    file_store_dir = "/scratch/335-lisa/mlkatz/gf_output/"

    # TODO: connect LISA to SSB for MBHs to numerical orbits

    gpus = [0]
    cp.cuda.runtime.setDevice(gpus[0])
    # few.get_backend('cuda12x')
    nwalkers = 36
    ntemps = 24

    tukey_alpha = 0.05

    orbits = EqualArmlengthOrbits()
    gpu_orbits = EqualArmlengthOrbits(force_backend="cuda12x")

    general_settings = GeneralSettings(
        Tobs=Tobs,
        dt=dt,
        file_store_dir=file_store_dir,
        base_file_name=base_file_name,
        data_input_path=ldc_source_file,
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
        remove_from_data=["dgb", "igb"],
        fixed_psd_kwargs=dict(model=sangria)
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
    ###  Galactic Binary Settings  ###
    ##################################
    ##################################

   
    ##################################
    ##################################
    ### MBHB Settings ################
    ##################################
    ##################################


    mbh_setup = get_mbh_erebor_settings(general_setup)

    ##################################
    ##################################
    ### EMRI Settings ################
    ##################################
    ##################################


    # # for transforms
    # # this is an example of how you would fill parameters 
    # # if you want to keep them fixed
    # # (you need to remove them from the other parts of initialization)
    # fill_dict_emri = {
    #    "ndim_full": 14,
    #    "fill_values": np.array([1.0, 0.0]), # inclination and Phi_theta
    #    "fill_inds": np.array([5, 12]),
    # }

    # # priors
    # priors_emri = {
    #     0: uniform_dist(np.log(1e5), np.log(1e6)),  # M total mass
    #     1: uniform_dist(1.0, 100.0),  # mu
    #     2: uniform_dist(0.01, 0.98),  # a
    #     3: uniform_dist(12.0, 16.0),  # p0
    #     4: uniform_dist(0.001, 0.4),  # e0
    #     5: uniform_dist(0.01, 100.0),  # dist in Gpc
    #     6: uniform_dist(-0.99999, 0.99999),  # qS
    #     7: uniform_dist(0.0, 2 * np.pi),  # phiS
    #     8: uniform_dist(-0.99999, 0.99999),  # qK
    #     9: uniform_dist(0.0, 2 * np.pi),  # phiK
    #     10: uniform_dist(0.0, 2 * np.pi),  # Phi_phi0
    #     11: uniform_dist(0.0, 2 * np.pi),  # Phi_r0
    # }

    # # transforms from pe to waveform generation
    # # after the fill happens (this is a little confusing)
    # # on my list of things to improve
    # parameter_transforms_emri = {
    #     0: np.exp,  # M 
    #     7: np.arccos, # qS
    #     9: np.arccos,  # qK
    # }

    # transform_fn_emri = TransformContainer(
    #     parameter_transforms=parameter_transforms_emri,
    #     fill_dict=fill_dict_emri,

    # )

    # # sampler treats periodic variables by wrapping them properly
    # # TODO: really need to create map for transform_fn with keywork names
    # periodic_emri = {
    #     "emri": {7: 2 * np.pi, 9: np.pi, 10: 2 * np.pi, 11: 2 * np.pi}
    # }

    # response_kwargs = dict(
    #     t0=30000.0,
    #     order=25,
    #     tdi="1st generation",
    #     tdi_chan="AE",
    #     orbits=general_setup.gpu_orbits,
    #     force_backend="cuda12x",
    # )

    # # TODO: I prepared this for Kerr but have not used it with the Kerr waveform yet
    # # so spin results are currently meaningless and will lead to slower code
    # # waveform kwargs
    # initialize_kwargs_emri = dict(
    #     T=general_setupTobs / YRSID_SI, # TODO: check these conversions all align
    #     dt=dt,
    #     emri_waveform_args=("FastKerrEccentricEquatorialFlux",),
    #     emri_waveform_kwargs=dict(force_backend="cuda12x"),
    #     response_kwargs=response_kwargs,
    # )
    
    
    # # for EMRI waveform class initialization
    # waveform_kwargs_emri = {}  #  deepcopy(initialize_kwargs_emri)

    # get_emri = GetEMRITemplates(
    #     initialize_kwargs_emri,
    #     waveform_kwargs_emri,
    #     start_freq_ind,
    #     end_freq_ind
    # )

    # inner_moves_emri = [
    #     (StretchMove(), 1.0)
    # ]

    # # mcmc info for main run
    # emri_main_run_mcmc_info = dict(
    #     branch_names=["emri"],
    #     nleaves_max=1,
    #     ndim=12,
    #     ntemps=ntemps,
    #     nwalkers=nwalkers,
    #     num_prop_repeats=1,
    #     inner_moves=inner_moves_emri,
    #     progress=False,
    #     thin_by=1,
    #    # stop_kwargs=mix_stopping_kwargs,
    #     # stopping_iterations=1
    # )

    # all_emri_info = dict(
    #     setup_func=setup_emri_functionality,
    #     periodic=periodic_emri,
    #     priors={"emri": ProbDistContainer(priors_emri)},
    #     transform=transform_fn_emri,
    #     waveform_kwargs=waveform_kwargs_emri,
    #     initialize_kwargs=initialize_kwargs_emri,
    #     pe_info=emri_main_run_mcmc_info,
    #     get_templates=get_emri,
    # )

    ##############
    ## READ OUT ##
    ##############


    global_settings = GlobalFitSettings(
        source_info={
            "mbh": mbh_setup,
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
