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
from lisatools.globalfit.recipe import (
    SearchRecipeStep,
    PERecipeStep,
    build_psd_moves,
    build_mbh_moves_phenom,
    scatter_around_injection,
    mbh_catalogue_to_sampling_basis,
)

from lisaconstants import ASTRONOMICAL_YEAR as YRSID_SI


def setup_recipe(recipe, engine_info, curr, acs, priors, state):
    cp.cuda.runtime.setDevice(curr.general_info.gpus[0])

    psd_search_move, psd_pe_move = build_psd_moves(engine_info, curr, acs, priors, permute_every=50)

    recipe.add_recipe_component(SearchRecipeStep(moves=[psd_search_move]), name="psd search")

    # Initialize MBH walkers from catalogue injection parameters
    catalogue = getattr(curr.general_info, "catalogue", {})
    mbh_catalogue = catalogue.get("MBHB", {})
    if mbh_catalogue:
        mbh_info = curr.source_info["mbh"]
        injection_params_list = []
        for source_id in sorted(mbh_catalogue.keys()):
            entry = mbh_catalogue[source_id]
            sampling_params = mbh_catalogue_to_sampling_basis(entry)
            injection_params_list.append(sampling_params)

        injection_params = np.array(injection_params_list)

        # Store injection truths for diagnostic plots
        curr.source_info["mbh"].injection = injection_params

        # Per-parameter spread for the Gaussian scatter
        spread = np.array([1e-4, 1e-4, 1e-4, 1e-4, 1e-3, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2])

        scatter_around_injection(
            state,
            "mbh",
            injection_params,
            spread,
        )

    # todo test this

    _, mbh_pe_move = build_mbh_moves_phenom(curr, acs, priors, state, permute_every=40)
    mbh_pe_moves = GFCombineMove(moves=[mbh_pe_move, psd_pe_move], share_temperature_control=False)
    recipe.add_recipe_component(PERecipeStep(moves=[mbh_pe_moves]), name="mbh+psd pe")


#######################
##### SETTINGS ########
#######################


def get_mbh_erebor_settings(general_set: GeneralSetup) -> MBHSetup:

    hms = [21, 33, 44]

    tlowfit = True  # use a fit to set the starting time of the root finder used in t(f)
    tol = 1e-12  # root finding tolerance

    wave_kwargs = dict(
        higher_modes=hms,
        include_negative_modes=True,  # negative m modes will be produced by simmetry
        t_low_fit=tlowfit,
        coarse_grain=False,  # if false it will generate the waveform on a dense time grid with the specified timestep
        atol=tol,
        rtol=tol,
    )

    response_kwargs = dict(
        sampling_frequency=1.0 / general_set.dt,
        tdi_generation="2nd generation",
        tdi_channels="XYZ",
        orbits=general_set.gpu_orbits if gpu_available else general_set.orbits,
        order=40,
    )

    waveform_init_kwargs = dict(
        waveform_kwargs=wave_kwargs,
        waveform_t0=97729089.327664,
        data_td_settings=general_set.data_td_settings,
        Tobs=1.0
        / 12.0
        / 2.0
        * YRSID_SI,  # this is only for the waveform generation, not the data, which is still general_set.Tobs
        start_freq=1e-4,
        ref_freq=2.0886886878886526e-05,  # source 5
        buffer_time=6000,
        stft_dt=general_set.stft_dt,
        freq_min=general_set.start_freq,
        freq_max=general_set.end_freq,
        tukey_alpha=general_set.window_alpha,
        force_backend=general_set.force_backend,
        **response_kwargs,
    )

    waveform_runtime_kwargs = dict(
        # this is for the waveform generation during the run, not the initialization
        # output_domain=general_set.basis_domain.capitalize(),
        # domain_kwargs=general_set.basis_kwargs,
    )

    mbh_settings = MBHSettings(
        Tobs=general_set.Tobs,
        dt=general_set.dt,
        initialize_kwargs=waveform_init_kwargs,
        waveform_kwargs=waveform_runtime_kwargs,
        nleaves_max=1,
        nleaves_min=1,
        ndim=11,
        num_prop_repeats=40,
        log_dir=general_set.file_store_dir,
    )

    return MBHSetup(mbh_settings)


def get_psd_erebor_settings(general_set: GeneralSetup) -> PSDSetup:

    # waveform kwargs
    initialize_kwargs_psd = dict()

    priors_psd = {
        r"$S_{\rm oms}$": uniform_dist(6.0e-12, 20.0e-11),  # Soms_d
        r"$S_{\rm tm}$": uniform_dist(1.0e-15, 20.0e-14),  # Sa_a
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
    )

    return PSDSetup(psd_settings)


def get_general_erebor_settings() -> GeneralSetup:
    # limits on parameters
    # now with negative fdots

    source_ids = [18]

    Tobs = 1.0 * YRSID_SI / 12.0
    dt = 5.0
    start_freq = 1e-4
    end_freq = 2.9e-2

    head_dir = "/data/asantini/packages/LISAanalysistools/"
    data_input_path = "/data/asantini/globalfit/MOJITO_DATA/mojito_light_2p5s/"
    base_file_name = "test_mbh_18_with_covariance"
    file_store_dir = head_dir + "mojito_output/"

    gpus = [0]
    cp.cuda.runtime.setDevice(gpus[0])
    # Restrict JAX to only see the target GPU — must be set before JAX backend init
    import jax

    jax.config.update("jax_cuda_visible_devices", ",".join(str(gpu) for gpu in gpus))

    backend = "cuda12x" if gpus is not None else "cpu"
    nwalkers = 30
    ntemps = 3

    window_type = "tukey"
    window_taper_duration = 0.1 * 7 * 24 * 3600.0
    normalize_window = True

    basis_domain = "stft"
    stft_dt = 7 * 24 * 3600.0 if basis_domain == "stft" else None  # hours

    base_file_name += f"_{basis_domain}"

    processor_init_kwargs = dict(
        L1_folder=data_input_path,
        source_types=["noise", "mbhb"],  #'vgb', 'gb'
        source_ids=dict(mbhb=source_ids),
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
        "duration": 70 * 24 * 3600,  # seconds — duration to trim from each end
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
