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
from lisatools.domaincomputation import DomainComputationGroupArray
from lisatools.utils.constants import *
from eryn.state import BranchSupplemental
from lisatools.globalfit.run import CurrentInfoGlobalFit
from lisatools.globalfit.stock.erebor import PSDSetup, PSDSettings, EMRISetup, EMRISettings


from eryn.prior import uniform_dist, log_uniform
from eryn.utils import TransformContainer
from eryn.prior import ProbDistContainer
from eryn.moves import StretchMove, TemperatureControl 
from eryn.moves.tempering import make_ladder

from lisatools.domains import STFTSettings, FDSettings
from lisatools.sensitivity import XYZSensitivityBackend
from lisatools.globalfit.moves import GFCombineMove, MultiGPUPSDMove, TDMBHSpecialMove, EMRISpecialMove
from lisatools.globalfit.engine import GlobalFitSettings, GeneralSetup, GeneralSettings, RankInfo
from lisatools.globalfit.recipe_steps import emri_catalogue_to_sampling_basis, subtract_initial_signal
from lisatools.utils.constants import YRSID_SI
from lisatools.globalfit.preprocessing import L1ProcessingStep
from lisatools.globalfit.recipe_steps import (
    SearchRecipeStep,
    PERecipeStep,
    build_psd_moves,
    build_mbh_moves_phenom,
    scatter_around_injection,
    mbh_catalogue_to_sampling_basis,
)

from lisatools.globalfit.postprocessing import (
    StochasticMetadata,
    SourceMetadata
)

logger = logging.getLogger(__name__)

MOJITO_REFERENCE_TIME = 97729089.327664

def setup_recipe(recipe, engine_info, curr, acs, priors, state):

    cp.cuda.runtime.setDevice(curr.general_info.gpus[0])

    general_info = curr.general_info
    nwalkers: int = general_info.nwalkers
    ntemps: int = general_info.ntemps
    Tmax: float = 1.0e6
    permute_every: int = 20

    emri_info = curr.source_info["emri"]
    psd_info = curr.source_info["psd"]

    effective_ndim = engine_info.ndims["psd"]
    temperature_control = TemperatureControl(
        effective_ndim, nwalkers, ntemps=ntemps, Tmax=Tmax, permute=False
    )

    psd_move_kwargs = dict(
        num_repeats=psd_info.num_prop_repeats,
        permute_every=permute_every,
        live_dangerously=True,
        psd_transform_fn=psd_info.transform,
        temperature_control=temperature_control,
        use_gpu=True,
        run_async=True,
        run_threaded=False
    )

    psd_search_move = MultiGPUPSDMove(
        acs, priors, max_logl_mode=True, name="psd search move", **psd_move_kwargs
    )
    psd_pe_move = MultiGPUPSDMove(acs, priors, max_logl_mode=False, name="psd pe move", **psd_move_kwargs)

    psd_search_move.accepted = np.zeros((ntemps, nwalkers))
    psd_pe_move.accepted = np.zeros((ntemps, nwalkers))

    #psd_search_move, psd_pe_move = build_psd_moves(engine_info, curr, acs, priors, permute_every=50)

    # recipe.add_recipe_component(SearchRecipeStep(moves=[psd_search_move]), name="psd search")

    #* ========================= *#
    
    # Initialize EMRI walkers from catalogue injection parameters
    catalogue = getattr(curr.general_info, "catalogue", {})
    emri_catalogue = catalogue.get("EMRI", {})
    if emri_catalogue:
        emri_info = curr.source_info["emri"]
        injection_params_list = []
        for source_id in sorted(emri_catalogue.keys()):
            entry = emri_catalogue[source_id]
            sampling_params = emri_catalogue_to_sampling_basis(entry)
            injection_params_list.append(sampling_params)

        injection_params = np.array(injection_params_list)

        # Store injection truths for diagnostic plots
        curr.source_info["emri"].injection = injection_params

        # Per-parameter spread for the Gaussian scatter
        spread = 1e-5

        scatter_around_injection(
            state,
            "emri",
            injection_params,
            spread,
            priors=priors,
        )
        
    from lisatools.sources.emri.waveform import EMRITDIWaveform

    # todo test this
    wave_gen = EMRITDIWaveform(**emri_info.initialize_kwargs)
    # breakpoint()
    subtract_initial_signal(acs, state, wave_gen.get_signals_for_residuals, "emri", emri_info)

    if emri_info.betas is None:
        emri_info.betas = make_ladder(emri_info.ndim, ntemps=ntemps)
    betas_all = np.tile(emri_info.betas, (emri_info.nleaves_max, 1))
    state.sub_states["emri"].betas_all = betas_all
    logger.debug(f"EMRI betas: {emri_info.betas}")

    coords_shape = (ntemps, nwalkers, emri_info.nleaves_max, emri_info.ndim)

    emri_move_kwargs = dict(
        dcga=acs,
        waveform_gen=wave_gen,
        branch_name="emri",
        coords_shape=coords_shape,
        waveform_gen_method="get_signals_for_residuals",
        waveform_gen_kwargs=emri_info.waveform_kwargs.copy(),
        waveform_like_method="__call__",
        waveform_like_kwargs=emri_info.waveform_kwargs.copy(),
        num_repeats=emri_info.num_prop_repeats,
        transform_fn=emri_info.transform,
        priors=priors,
        inner_moves=emri_info.inner_moves,
        betas_all=betas_all,
        permute_every=permute_every,
        pad_out_of_prior=True,
        run_async=True,
        run_threaded=True,
        randomize_split=True,
        # Cap concurrent EMRI waveform+likelihood evaluations per GPU to bound
        # peak device memory; None runs all of a split's walkers at once. With
        # >1 GPU this still runs num_gpus * batch_size_per_gpu walkers in parallel.
        batch_size_per_gpu=4,
    )

    emri_pe_move = EMRISpecialMove(**emri_move_kwargs)
    emri_pe_move.accepted = np.zeros((ntemps, nwalkers))

    #_, mbh_pe_move = build_mbh_moves_phenom(curr, acs, priors, state, permute_every=40)
    # emri_pe_moves = GFCombineMove(moves=[emri_pe_move, psd_pe_move], share_temperature_control=False)
    recipe.add_recipe_component(PERecipeStep(moves=[emri_pe_move]), name="emri pe")


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
        log_dir=general_set.artifacts_file_dir,
        priors=priors,
        ndim=2,
        injection=injection,
        num_prop_repeats=500,
    )

    psd_metadata = StochasticMetadata(
        model_config=model_config,
        frequency_ranges=frequency_ranges,
        prior_model=prior_model,
        prior_model_code_link="",  # todo populate repositories
        prior_model_config=prior_model_config,
    )

    return PSDSetup(psd_settings), psd_metadata


def get_emri_erebor_settings(general_set: GeneralSetup) -> EMRISetup:

    waveform_model = "EMRITDIWaveform"
    waveform_model_code_link = "https://github.com/Erebor-L2D/LISAanalysistools/blob/9d63bb1e63e7b8f640d3780551d9421df5245992/src/lisatools/sources/emri/waveform.py#L130"
    prior_model_code_link = ""
    frequency_ranges = [(general_set.start_freq, general_set.end_freq)]

    wave_kwargs = dict(
            sum_kwargs = {"pad_output": True,},
            inspiral_kwargs = {"DENSE_STEPPING": 0,  # sparsely sampled trajectory
                #"max_init_len": int(1e8),  # length of trajectories well under 1000
            },
            amplitude_kwargs = {}
        )

    response_kwargs = dict(
        sampling_frequency=1.0 / general_set.dt,
        tdi_generation="2nd generation",
        tdi_channels="XYZ",
        orbits=general_set.gpu_orbits if gpu_available else general_set.orbits,
        order=30,
    )

    waveform_init_kwargs = dict(
        waveform_kwargs=wave_kwargs,
        waveform_t0=MOJITO_REFERENCE_TIME,
        data_td_settings=general_set.data_td_settings,
        buffer_time=15_000,
        stft_dt=general_set.stft_dt,
        freq_min=general_set.start_freq,
        freq_max=general_set.end_freq,
        tukey_alpha=general_set.window_alpha,
        force_backend=general_set.force_backend,
        fft_batch_size=1,
        **response_kwargs,
    )

    waveform_runtime_kwargs = dict(mode_selection_threshold = 1e-4)

    betas = 1 / 1.2 ** np.arange(general_set.ntemps)  # Geometric ladder with ratio 1.2

    input_basis = [
        "logm1",
        "m2",
        "a",
        "p0",
        "e0",
        "dist",
        "cosqK",
        "phiK",
        "Phi_phi0",
        "Phi_r0",
        "alpha",
        "sin_delta",
    ]

    output_basis = [
        "m1",
        "m2",
        "a",
        "p0",
        "e0",
        "xI0",
        "dist",
        "qK",
        "phiK",
        "Phi_phi0",
        "Phi_theta0",
        "Phi_r0",
        "alpha",
        "delta",
    ]

    # for transforms
    emri_fill_dict = {
        "xI0": 1.0,  # inclination
        "Phi_theta0": 0.0,  # Phi_theta
    }

    def decouple_spin_sign(a, xI0):
        # Decouple the spin sign from a

        xI0 = np.sign(a) * 1.0  # inclination is either 1 or -1 depending on the sign of a
        a = np.abs(a)  # take the absolute value of a
        return a, xI0

    def couple_spin_sign(a, xI0):
        # Couple the spin sign back into a
        a = np.sign(xI0) * a  # restore the sign of a based on xI0
        xI0 = 1.0
        return a, xI0

    emri_transform_fn_in = {
        "m1": np.exp,  # M
        ("a", "xI0"): decouple_spin_sign,
        "qK": np.arccos,  # qK
        "delta": np.arcsin,  # delta
    }

    emri_inverse_transform_fn_in = {
        "m1": np.log,  # M
        "qK": np.cos,  # qS
        ("a", "xI0"): couple_spin_sign,
        "delta": np.sin,  # delta
    }
    key_map = {
        "logm1": "m1",
        "cosqK": "qK",
        "sin_delta": "delta",
    }

    transform = TransformContainer(
        input_basis=input_basis,
        output_basis=output_basis,
        parameter_transforms=emri_transform_fn_in,
        fill_dict=emri_fill_dict,
        inverse_parameter_transforms=emri_inverse_transform_fn_in,
        key_map=key_map
    )

    periodic = {
        "emri": {
                "phiK": 2 * np.pi,
                "Phi_phi0": 2 * np.pi,
                "Phi_r0": 2 * np.pi,
                "alpha": 2 * np.pi,
                }
    }

    priors_emri = {
            "logm1": uniform_dist(np.log(1e5), np.log(5e6)),  # log m1
            "m2": uniform_dist(1, 100),  # m2
            "a": uniform_dist(-0.999, 0.999),  # a
            "p0": uniform_dist(5.0, 100.0),  # p0
            "e0": uniform_dist(0.001, 0.8),  # e0
            "dist": uniform_dist(0.01, 100.0),  # dist in Gpc
            "qK": uniform_dist(-0.99999, 0.99999),  # qK
            "phiK": uniform_dist(0.0, 2 * np.pi),  # phiK
            "Phi_phi0": uniform_dist(0.0, 2 * np.pi),  # Phi_phi0
            "Phi_r0": uniform_dist(0.0, 2 * np.pi),  # Phi_r0
            "alpha": uniform_dist(0.0, 2 * np.pi),  # alpha
            "sin_delta": uniform_dist(-0.99999, 0.99999),  # delta
        }
    priors = {"emri": ProbDistContainer(priors_emri)}

    emri_settings = EMRISettings(
        log_dir=general_set.artifacts_file_dir,
        Tobs=general_set.Tobs,
        dt=general_set.dt,
        initialize_kwargs=waveform_init_kwargs,
        transform=transform,
        periodic=periodic,
        priors=priors,
        waveform_kwargs=waveform_runtime_kwargs,
        nleaves_max=len(general_set.processor_init_kwargs["source_ids"]["emri"]),
        nleaves_min=len(general_set.processor_init_kwargs["source_ids"]["emri"]),
        ndim=12,
        num_prop_repeats=5,
        betas=betas,
        inner_moves=[StretchMove(),],
        logm1_lims = None,
        m2_lims = None,
        a_lims = None,
        p0_lims = None,
        e0_lims = None,
    )

    wf_metadata = waveform_init_kwargs.copy()
    _ = wf_metadata.pop("data_td_settings", None)  # remove data_td_settings
    _ = wf_metadata.pop("orbits", None)  # remove orbits from metadata to avoid issues with serialization
    wf_metadata['force_backend'] = 'cpu'


    emri_metadata = SourceMetadata(
        source_type="EMRI",
        frequency_ranges=frequency_ranges,
        waveform_model=waveform_model,
        waveform_model_code_link=waveform_model_code_link,
        waveform_model_config=wf_metadata,
        prior_model_code_link=prior_model_code_link,
    )

    return EMRISetup(emri_settings), emri_metadata

def get_general_erebor_settings() -> GeneralSetup:
    # limits on parameters
    # now with negative fdots

    global_fit_codename = "erebor"
    global_fit_version = "CDL1run0_v0"
    global_fit_contact = "ereborl2d@googlegroups.com"
    global_fit_code_link = "https://github.com/Erebor-L2D/LISAanalysistools/releases/tag/cdl1-run_0"
    global_fit_input_data_link = ""
    global_fit_input_reference = "mojito light"
    global_fit_noise_model = "parametric"
    global_fit_noise_model_code_link = "https://github.com/Erebor-L2D/LISAanalysistools/blob/9d63bb1e63e7b8f640d3780551d9421df5245992/src/lisatools/sensitivity.py#L1797" #todo populate repositories
    comment = ""

    submission_folder = None #"/work/asantini/globalfit/erebor_org_setup/mojito_runs/"

    num_iterations = 500

    source_ids = [0]

    Tobs = 1 * YRSID_SI
    dt = 5.0
    start_freq = 1e-4
    end_freq = 2.9e-2

    head_dir = "/data/asantini/globalfit/erebor_org_setup/mojito_runs/"
    data_input_path = "/data/asantini/globalfit/MOJITO_DATA/mojito_light_2p5s/"
    base_file_name = "test_emri"
    file_store_dir = head_dir

    gpus = [1]
    cp.cuda.runtime.setDevice(gpus[0])
    # Restrict JAX to only see the target GPU — must be set before JAX backend init
    import jax

    jax.config.update("jax_cuda_visible_devices", ",".join(str(gpu) for gpu in gpus))

    backend = "cuda12x" if gpus is not None else "cpu"
    nwalkers = 20
    ntemps = 1

    window_type = "tukey"
    window_taper_duration = 1 / start_freq
    normalize_window = True

    basis_domain = "stft"
    stft_dt = 1 * 24 * 3600.0 if basis_domain == "stft" else None  # hours

    if basis_domain == "stft":
        domain_settings = STFTSettings.make_factory(
            big_dt=stft_dt, min_freq=start_freq, max_freq=end_freq
        )
    else:
        domain_settings = FDSettings.make_factory(min_freq=start_freq, max_freq=end_freq)

    base_file_name += f"_{basis_domain}"

    processor_init_kwargs = dict(
        L1_folder=data_input_path,
        source_types=["emri"],  #'vgb', 'gb'
        source_ids=dict(emri=source_ids),
        verbose=True,
        do_plots=True,
        orbits_class=L1Orbits,
        store_individual_timeseries=True,
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

    sensitivity_init_kwargs = dict(tdi_generation=2, mask_percentage=0.02, average_transfer_functions=True)

    general_settings = GeneralSettings(
        num_iterations=num_iterations,
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
        sensitivity_backend_class=XYZSensitivityBackend,
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
        comment=comment
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

    ##################################
    ##################################
    ###  EMRI Settings  ###############
    ##################################
    ##################################

    emri_setup, emri_metadata = get_emri_erebor_settings(general_setup)

    ##############
    ## READ OUT ##
    ##############

    global_settings = GlobalFitSettings(
        source_info={
            "emri": emri_setup,
            "psd": psd_setup,
        },
        general_info=general_setup,
        rank_info=rank_info,
        setup_function=setup_recipe,
        source_metadata={
            "emri": emri_metadata,
            "psd": psd_metadata,
        }
    )

    curr_info = CurrentInfoGlobalFit(global_settings)

    return curr_info


if __name__ == "__main__":
    settings = get_global_fit_settings()
    breakpoint()
