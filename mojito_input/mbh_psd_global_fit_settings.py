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
from lisatools.utils.constants import *
from eryn.state import BranchSupplemental
from lisatools.globalfit.run import CurrentInfoGlobalFit
from lisatools.globalfit.stock.erebor import PSDSetup, PSDSettings, MBHSetup, MBHSettings


from eryn.prior import uniform_dist
from eryn.utils import TransformContainer
from eryn.prior import ProbDistContainer

from eryn.moves import CombineMove
from lisatools.globalfit.moves import GFCombineMove


from lisatools.globalfit.engine import GlobalFitSettings, GeneralSetup, GeneralSettings, RankInfo

from eryn.utils.updates import Update

from lisatools.globalfit.preprocessing import L1ProcessingStep
from lisatools.globalfit.recipe_steps import SearchRecipeStep, PERecipeStep, build_psd_moves, build_mbh_moves_phenom, scatter_around_injection, mbh_catalogue_to_sampling_basis

from lisaconstants import ASTRONOMICAL_YEAR as YRSID_SI


def setup_recipe(recipe, engine_info, curr, acs, priors, state):
    cp.cuda.runtime.setDevice(curr.general_info.gpus[0])

    psd_search_move, psd_pe_move = build_psd_moves(engine_info, curr, acs, priors)

    recipe.add_recipe_component(SearchRecipeStep(moves=[psd_search_move]), name="psd search")

    # Initialize MBH walkers from catalogue injection parameters
    catalogue = getattr(curr.general_info, 'catalogue', {})
    mbh_catalogue = catalogue.get('MBHB', {})
    if mbh_catalogue:
        mbh_info = curr.source_info["mbh"]
        injection_params_list = []
        for source_id in sorted(mbh_catalogue.keys()):
            entry = mbh_catalogue[source_id]
            sampling_params = mbh_catalogue_to_sampling_basis(entry)
            injection_params_list.append(sampling_params)

        injection_params = np.array(injection_params_list)

        # Per-parameter spread for the Gaussian scatter
        spread = 1e-2

        scatter_around_injection(
            state, "mbh", injection_params, spread,
        )

    # todo test this 
    
    _, mbh_pe_move = build_mbh_moves_phenom(curr, acs, priors, state)
    mbh_pe_moves = GFCombineMove(moves=[mbh_pe_move, psd_pe_move], share_temperature_control=False)
    recipe.add_recipe_component(PERecipeStep(moves=[mbh_pe_moves]), name="mbh+psd pe")

    
#######################
##### SETTINGS ########
#######################

def get_mbh_erebor_settings(general_set: GeneralSetup) -> MBHSetup:
    
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
        order=40
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
        #is_tref=False,
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
        ndim=11,
        num_prop_repeats=5,
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
    
    Tobs = 5. * YRSID_SI / 12.0
    dt = 2.5

    head_dir = "/data/asantini/packages/LISAanalysistools/"
    #ldc_source_file = head_dir + "emri_sangria_injection.h5"
    data_input_path = "/data/asantini/globalfit/MOJITO_DATA/mojito_light_2p5s/"
    base_file_name = "mbh_psd_separate_2nd_try"
    file_store_dir = head_dir + "mojito_output/"

    gpus = [2]
    cp.cuda.runtime.setDevice(gpus[0])
    # Restrict JAX to only see the target GPU — must be set before JAX backend init
    import jax
    jax.config.update("jax_cuda_visible_devices", str(gpus[0]))

    backend="cuda12x" if gpus is not None else "cpu"
    nwalkers = 20
    ntemps = 2

    tukey_alpha = 0.05

    basis_domain = "stft"
    stft_dt = 8 * 3600.0  # hours

    processor_init_kwargs = dict(L1_folder=data_input_path,
                                 source_types=['noise', 'mbhb'],
                                 source_ids=dict(mbhb=source_ids),
                                 verbose=True,
                                 do_plots=True,
                                 orbits_class=L1Orbits,
                                 orbits_kwargs=dict(force_backend=backend, frame="ecliptic")
                                )
    
    preprocess_kwargs = dict(plot_folder=file_store_dir)

    sensitivity_init_kwargs = dict(tdi_generation=2, mask_percentage=0.02)

    general_settings = GeneralSettings(
        Tobs=Tobs,
        dt=dt,
        file_store_dir=file_store_dir,
        base_file_name=base_file_name,
        start_freq=5e-5,
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
