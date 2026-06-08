from __future__ import annotations

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

from typing import TYPE_CHECKING

from lisatools.detector import L1Orbits
from lisatools.domaincomputation import DomainComputationGroupArray
from lisatools.utils.constants import *
from eryn.state import BranchSupplemental
from lisatools.globalfit.run import CurrentInfoGlobalFit
from lisatools.globalfit.stock.erebor import PSDSetup, PSDSettings, MBHSetup, MBHSettings, GBSetup, GBSettings, get_fdot_mojito


from eryn.prior import uniform_dist, log_uniform
from eryn.utils import TransformContainer
from eryn.prior import ProbDistContainer

from eryn.moves import StretchMove, TemperatureControl
from eryn.moves.tempering import make_ladder
from lisatools.globalfit.moves import GFCombineMove, MultiGPUPSDMove, TDMBHSpecialMove
from lisatools.globalfit.engine import GlobalFitSettings, GeneralSetup, GeneralSettings, RankInfo
from lisatools.utils.constants import YRSID_SI

from eryn.utils.updates import Update

from lisatools.globalfit.preprocessing import L1ProcessingStep
from lisatools.globalfit.recipe_steps import (
    SearchRecipeStep,
    PERecipeStep,
    RJRecipeStep,
    build_psd_moves,
    build_gb_moves,
    build_mbh_moves_phenom,
    scatter_around_injection,
    mbh_catalogue_to_sampling_basis,
    setup_state_for_injection,
    subtract_initial_signal
)

from lisatools.globalfit.postprocessing import (
    StochasticMetadata,
    SourceMetadata
)

if TYPE_CHECKING:
    from lisatools.globalfit.recipe import Recipe
    from lisatools.analysiscontainer import AnalysisContainerArray
    from lisatools.datacontainer import DataResidualArray
    from lisatools.globalfit.engine import EngineInfo


MOJITO_REFERENCE_TIME = 97729089.327664
MOJITO_AVERAGE_ARMLENGTH = 2493162305.42235

logger = logging.getLogger(__name__)

def setup_recipe(
    recipe: Recipe, 
    engine_info: EngineInfo, 
    curr: CurrentInfoGlobalFit, 
    acs: AnalysisContainerArray, 
    priors: dict[str, ProbDistContainer], 
    state):

    cp.cuda.runtime.setDevice(curr.general_info.gpus[0])

    dcga = DomainComputationGroupArray(acs=acs)

    general_info = curr.general_info
    nwalkers: int = general_info.nwalkers
    ntemps: int = general_info.ntemps
    Tmax: float = 1.0e6
    permute_every: int = 40

    mbh_info = curr.source_info["mbh"]
    psd_info = curr.source_info["psd"]
    gb_info = curr.source_info["gb"]

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
        run_async=False,
        run_threaded=False
    )

    psd_search_move = MultiGPUPSDMove(
        dcga, priors, max_logl_mode=True, name="psd search move", **psd_move_kwargs
    )
    psd_pe_move = MultiGPUPSDMove(dcga, priors, max_logl_mode=False, name="psd pe move", **psd_move_kwargs)

    psd_search_move.accepted = np.zeros((ntemps, nwalkers))
    psd_pe_move.accepted = np.zeros((ntemps, nwalkers))

    #psd_search_move, psd_pe_move = build_psd_moves(engine_info, curr, acs, priors, permute_every=50)

    recipe.add_recipe_component(SearchRecipeStep(moves=[psd_search_move]), name="psd search")

    #* ========================= *#
    
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
        spread = np.array([1e-4, 1e-3, 1e-3, 1e-3, 1e-3, 1e-1, 1e-1, 1e-1, 1e-2, 1e-2, 1])

        scatter_around_injection(
            state,
            "mbh",
            injection_params,
            spread,
            priors=priors,
        )
        
    from lisatools.sources.bbh.waveform import PhenomTHMTDIWaveform

    # todo test this
    wave_gen = PhenomTHMTDIWaveform(**mbh_info.initialize_kwargs)
    # breakpoint()
    subtract_initial_signal(acs, state, wave_gen.get_signals_for_residuals, "mbh", mbh_info)

    if mbh_info.betas is None:
        mbh_info.betas = make_ladder(mbh_info.ndim, ntemps=ntemps)
    betas_all = np.tile(mbh_info.betas, (mbh_info.nleaves_max, 1))
    state.sub_states["mbh"].betas_all = betas_all
    logger.debug(f"MBH betas: {mbh_info.betas}")

    coords_shape = (ntemps, nwalkers, mbh_info.nleaves_max, mbh_info.ndim)

    mbh_move_kwargs = dict(
        dcga=dcga,
        waveform_gen=wave_gen,
        branch_name="mbh",
        coords_shape=coords_shape,
        waveform_gen_kwargs=mbh_info.waveform_kwargs.copy(),
        waveform_like_kwargs={},
        num_repeats=mbh_info.num_prop_repeats,
        transform_fn=mbh_info.transform,
        priors=priors,
        inner_moves=mbh_info.inner_moves,
        betas_all=betas_all,
        permute_every=permute_every,
        pad_out_of_prior=True,
        run_async=True,
        run_threaded=True,
        randomize_split=True
    )

    mbh_pe_move = TDMBHSpecialMove(**mbh_move_kwargs)
    mbh_pe_move.accepted = np.zeros((ntemps, nwalkers))


    #* =========== GB ============== *#
    # Sampling basis: ``[logA, f0 [mHz], fdot, phi0, cos_iota, psi, lam, sin_beta]``
    spread = np.array([1e-10, 1e-11, 1e-17, 1e-9, 1e-9, 1e-9, 1e-9, 1e-9])
    setup_state_for_injection(curr, state, "VGB", "gb", spread=spread)

    gb_search_moves, gb_pe_moves = build_gb_moves(
        engine_info, curr, acs, priors, state
    )

    search_weights = [0.8, 0.2]
    recipe.add_recipe_component(RJRecipeStep(moves=gb_search_moves, weights=search_weights, convergence_iter=10), name="gb search")

    #_, mbh_pe_move = build_mbh_moves_phenom(curr, acs, priors, state, permute_every=40)
    pe_moves = GFCombineMove(moves=[gb_pe_moves[0], mbh_pe_move, psd_pe_move], share_temperature_control=False)
    recipe.add_recipe_component(PERecipeStep(moves=[pe_moves]), name="mbh + psd pe")


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
        num_prop_repeats=100,
    )

    psd_metadata = StochasticMetadata(
        model_config=model_config,
        frequency_ranges=frequency_ranges,
        prior_model=prior_model,
        prior_model_code_link="",  # todo populate repositories
        prior_model_config=prior_model_config,
    )

    return PSDSetup(psd_settings), psd_metadata

def get_gb_erebor_settings(general_set: GeneralSetup) -> tuple[GBSetup:, SourceMetadata]:

    waveform_model = "GBGPU"
    waveform_model_code_link = "https://github.com/Erebor-L2D/GBGPU/tree/cdl1-run0"
    prior_model_code_link = "https://priors-database-f0027f.gitlab.io/mojito_light_1a.html#massive-black-hole-binaries-mbhb"

    input_data_arr: DataResidualArray = general_set.input_data_residual_array

    eps = 0.0001  
    start_freq = float(input_data_arr.settings.f_arr[0]) * (1 + eps)
    end_freq = float(input_data_arr.settings.f_arr[-1]) * (1 - eps)  # avoid edge effects by staying slightly within the band limits defined by the input data array

    Tobs = 1/getattr(input_data_arr.settings, "df")
    
    delta_safe = 1e-9

    A_lims = [10**(-23.2), 1e-20]
    f0_lims = [start_freq, end_freq]#[general_set.start_freq, general_set.end_freq]  # reset by band limits

    m_chirp_lims = [0.03, 1.34]
    # fdot_max_val = get_fdot(f0_lims[-1], Mc=m_chirp_lims[-1])
    
    fdot_lims = [get_fdot_mojito(f0_lims[-1] * 2, sign="-"), get_fdot_mojito(f0_lims[-1] * 2, sign="+")] # also reset in band limits
    phi0_lims = [0.0, 2 * np.pi]
    iota_lims = [0.0 + delta_safe, np.pi - delta_safe]
    psi_lims = [0.0, np.pi]
    alpha_lims = [0.0, 2 * np.pi]
    delta_lims = [-np.pi / 2.0 + delta_safe, np.pi / 2.0 - delta_safe]

    oversample = 4
    extra_buffer = 5
    
    assert start_freq and end_freq and general_set.Tobs and general_set.preprocess_kwargs
    start_freq_ind = input_data_arr.start_freq_ind

    initialize_kwargs = dict(
        orbits=general_set.gpu_orbits if gpu_available else general_set.orbits, 
        t0=general_set.data_t0,
        force_backend=general_set.gpu_backend
    )

    # geometric spacing 
    betas = 1 / 1.2 ** np.arange(general_set.ntemps)
    betas[-1] = 0.0001
        
    search_kwargs = dict(
        nwalkers = 16,
        ntemps = 24,
        shutoff_band_iteration = 10,
        shutoff_frequency_threshold = 4e-3,
        burn_1 = 1000,
        nsteps_1 = 400,
        snr_threshold = 8.0,
        burn_2 = 1000,
        nsteps_2 = 500,
        refit_start_iteration = 5
    )
    
    waveform_kwargs = dict(
        dt=general_set.dt,
        T=Tobs,
        use_c_implementation=True,
        start_freq_ind=start_freq_ind,
        tdi_channel_setup="XYZ",
        tdi2=True,
        oversample=oversample,
        window=general_set.window_type,
        window_alpha=general_set.window_alpha
    )
    
    gb_settings = GBSettings(
        A_lims=A_lims,
        f0_lims=f0_lims,
        m_chirp_lims=m_chirp_lims,
        fdot_lims=fdot_lims,
        phi0_lims=phi0_lims,
        iota_lims=iota_lims,
        psi_lims=psi_lims,
        alpha_lims=alpha_lims,
        delta_lims=delta_lims,
        start_freq=start_freq,
        end_freq=end_freq,
        oversample=oversample,
        extra_buffer=extra_buffer,
        # Start_resample_iter, Iter_count_per_resample, !group_proposal_kwargs (handled later?)
        start_freq_ind=start_freq_ind,
        # t0=t0_gbs,
        # tdi_setup="XYZ",
        # use_tdi2=True,
        Tobs=Tobs,
        dt=general_set.dt,
        initialize_kwargs=initialize_kwargs,
        waveform_kwargs=waveform_kwargs,
        # Transform, Priors, Periodic (handled later!)
        nleaves_max=100,
        nleaves_min=0,
        ndim=8,
        betas=betas,
        log_dir=general_set.artifacts_file_dir,
        num_repeat_proposals=100, 
        search_kwargs=search_kwargs        
    )

    gb_setup = GBSetup(gb_settings)
    
    band_edges = gb_setup.band_edges.copy()
    band_starts = band_edges[:-1]
    band_ends = band_edges[1:]
    frequency_ranges = [(start, end) for start, end in zip(band_starts, band_ends)]

    metadata_initialize_kwargs = initialize_kwargs.copy()
    _ = metadata_initialize_kwargs.pop("orbits", None)  # remove orbits from
    metadata_initialize_kwargs['force_backend'] = 'cpu'  # override to cpu for metadata to avoid issues with serialization

    gb_metadata = SourceMetadata(
        source_type="gb",
        frequency_ranges=frequency_ranges,
        waveform_model=waveform_model,
        waveform_model_code_link=waveform_model_code_link,
        prior_model_code_link=prior_model_code_link,
        waveform_model_config=dict(init_kwargs=metadata_initialize_kwargs, runtime_kwargs=waveform_kwargs),
    )

    return gb_setup, gb_metadata


def get_mbh_erebor_settings(general_set: GeneralSetup) -> MBHSetup:

    waveform_model = "PhenomTHMTDIWaveform"
    waveform_model_code_link = "https://github.com/Erebor-L2D/LISAanalysistools/blob/9d63bb1e63e7b8f640d3780551d9421df5245992/src/lisatools/sources/bbh/waveform.py#L130"
    prior_model_code_link = "https://priors-database-f0027f.gitlab.io/mojito_light_1a.html#massive-black-hole-binaries-mbhb"
    frequency_ranges = [(general_set.start_freq, general_set.end_freq)]

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
        order=30,
    )

    waveform_init_kwargs = dict(
        waveform_kwargs=wave_kwargs,
        waveform_t0=MOJITO_REFERENCE_TIME,
        data_td_settings=general_set.data_td_settings,
        Tobs=1.0
        / 12.0
        * YRSID_SI,  # this is only for the waveform generation, not the data, which is still general_set.Tobs
        start_freq=7e-5,
        use_reference_time=True,
        buffer_time=15_000,
        stft_dt=general_set.stft_dt,
        freq_min=general_set.start_freq,
        freq_max=general_set.end_freq,
        tukey_alpha=general_set.window_alpha,
        force_backend=general_set.force_backend,
        fft_batch_size=2,
        **response_kwargs,
    )

    waveform_runtime_kwargs = dict()

    betas = 1 / 1.2 ** np.arange(general_set.ntemps)  # Geometric ladder with ratio 1.2

    basis = [
        r"$\log M$",
        r"$Q$",
        r"$s_{1z}$",
        r"$s_{2z}$",
        r"$d_L$",
        r"$\phi_{\rm ref}$",
        r"$\cos \iota$",
        r"$\psi$",
        r"$\alpha$",
        r"$\sin \delta$",
        r"$t_{\rm plunge}$",
    ]

    def gpc_to_mpc(x):
        """
        Transform from Gpc to Mpc, for distance prior.
        """
        return x * 1e3

    def mT_Q(M, Q):
        """
        Transform from total mass and mass ratio m1/m2 to m1 and m2.
        """
        m2 = M / (1 + Q)
        m1 = Q * m2
        assert np.all(m1 >= m2), "m1 should be the larger mass"
        return m1, m2

    mbh_transform_fn_in = {
        r"$\log M$": np.exp,
        r"$d_L$": gpc_to_mpc,
        r"$\cos \iota$": np.arccos,
        r"$\sin \delta$": np.arcsin,
        (r"$\log M$", r"$Q$"): mT_Q,
        # (r"$t_{\rm plunge}$", r"$\lambda$", r"$\sin \delta$", r"$\psi$"): LISA_to_SSB,
        # ("lam", "sin_beta", "psi"): ecliptic_to_icrs,
    }

    transform = TransformContainer(
        input_basis=basis,
        output_basis=basis,
        parameter_transforms=mbh_transform_fn_in,
        fill_dict={},
    )

    periodic = {"mbh": {r"$\phi_{\rm ref}$": 2 * np.pi, r"$\alpha$": 2 * np.pi, r"$\psi$": np.pi}}

    priors_mbh = {
                r"$\log M$": uniform_dist(np.log(1e5), np.log(1e8)),
                r"$Q$": log_uniform(1., 10.),
                r"$s_{1z}$": uniform_dist(-0.99999999, +0.99999999),
                r"$s_{2z}$": uniform_dist(-0.99999999, +0.99999999),
                r"$d_L$": uniform_dist(1, 150.0), # uniform_dist(0.01, 1000.0),
                r"$\phi_{\rm ref}$": uniform_dist(0.0, 2 * np.pi),
                r"$\cos \iota$": uniform_dist(-1.0 + 1e-6, 1.0 - 1e-6),
                r"$\psi$": uniform_dist(0.0, np.pi), #is this right?
                r"$\alpha$": uniform_dist(0.0, 2 * np.pi),
                r"$\sin \delta$": uniform_dist(-1.0 + 1e-6, 1.0 - 1e-6),
                r"$t_{\rm plunge}$": uniform_dist(0.0, general_set.Tobs + 3600.0),
            }
    priors = {"mbh": ProbDistContainer(priors_mbh)}

    mbh_settings = MBHSettings(
        log_dir=general_set.artifacts_file_dir,
        Tobs=general_set.Tobs,
        dt=general_set.dt,
        initialize_kwargs=waveform_init_kwargs,
        transform=transform,
        periodic=periodic,
        priors=priors,
        waveform_kwargs=waveform_runtime_kwargs,
        nleaves_max=len(general_set.processor_init_kwargs["source_ids"]["mbhb"]),
        nleaves_min=len(general_set.processor_init_kwargs["source_ids"]["mbhb"]),
        ndim=11,
        num_prop_repeats=50,
        betas=betas,
        inner_moves=[StretchMove(),]
    )

    wf_metadata = waveform_init_kwargs.copy()
    _ = wf_metadata.pop("data_td_settings", None)  # remove data_td_settings
    _ = wf_metadata.pop("orbits", None)  # remove orbits from metadata to avoid issues with serialization
    wf_metadata['force_backend'] = 'cpu'


    mbh_metadata = SourceMetadata(
        source_type="MBHB",
        frequency_ranges=frequency_ranges,
        waveform_model=waveform_model,
        waveform_model_code_link=waveform_model_code_link,
        waveform_model_config=wf_metadata,
        prior_model_code_link=prior_model_code_link,
    )

    return MBHSetup(mbh_settings), mbh_metadata

def get_general_erebor_settings() -> GeneralSetup:
    # limits on parameters
    # now with negative fdots

    global_fit_codename = "erebor"
    global_fit_version = "CDL1run1_v3"
    global_fit_contact = "ereborl2d@googlegroups.com"
    global_fit_code_link = "https://github.com/Erebor-L2D/LISAanalysistools/releases/tag/cdl1-run_0"
    global_fit_input_data_link = ""
    global_fit_input_reference = "mojito light"
    global_fit_noise_model = "parametric"
    global_fit_noise_model_code_link = "https://github.com/Erebor-L2D/LISAanalysistools/blob/9d63bb1e63e7b8f640d3780551d9421df5245992/src/lisatools/sensitivity.py#L1797" #todo populate repositories
    comment = "new test run for mbhb+vgb+noise after implementing clustering and sorting."

    submission_folder = "/work/asantini/globalfit/l3c_exchange/mojito_light_results/"

    num_iterations = 150

    source_ids = [18, 5, 16, 7, 2, 12]

    Tobs = 9.0 * YRSID_SI / 12.0
    dt = 5.0
    start_freq = 1e-4
    end_freq = 2.9e-2

    head_dir = "/data/asantini/globalfit/erebor/mojito_runs/"
    data_input_path = "/data/asantini/globalfit/MOJITO_DATA/mojito_light_2p5s/"
    base_file_name = "joint_run_vgb_v2"
    file_store_dir = head_dir

    gpus = [0]
    cp.cuda.runtime.setDevice(gpus[0])
    # Restrict JAX to only see the target GPU — must be set before JAX backend init
    import jax

    jax.config.update("jax_cuda_visible_devices", ",".join(str(gpu) for gpu in gpus))

    backend = "cuda12x" if gpus is not None else "cpu"
    nwalkers = 30
    ntemps = 4

    window_type = "tukey"
    window_taper_duration = 1 / start_freq
    normalize_window = True

    basis_domain = "fd"
    stft_dt = 1 * 24 * 3600.0 if basis_domain == "stft" else None  # hours

    base_file_name += f"_{basis_domain}"

    processor_init_kwargs = dict(
        L1_folder=data_input_path,
        source_types=["mbhb", "vgb", "noise"],  #'vgb', 'gb'
        source_ids=dict(mbhb=source_ids),
        verbose=True,
        do_plots=True,
        orbits_class=L1Orbits,
        orbits_kwargs=dict(force_backend=backend, frame="icrs", armlength=MOJITO_AVERAGE_ARMLENGTH),  # icrs
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

    sensitivity_init_kwargs = dict(tdi_generation=2, mask_percentage=0.02)

    general_settings = GeneralSettings(
        num_iterations=num_iterations,
        Tobs=Tobs,
        dt=dt,
        file_store_dir=file_store_dir,
        base_file_name=base_file_name,
        start_freq=start_freq,
        end_freq=end_freq,
        basis_domain=basis_domain,
        stft_dt=stft_dt,
        random_seed=103209,
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
    ###  MBH Settings  ###############
    ##################################
    ##################################

    mbh_setup, mbh_metadata = get_mbh_erebor_settings(general_setup)

    ##################################
    ##################################
    ###  GB Settings  ###############
    ##################################
    ##################################

    gb_setup, gb_metadata = get_gb_erebor_settings(general_setup)

    ##############
    ## READ OUT ##
    ##############

    global_settings = GlobalFitSettings(
        source_info={
            "mbh": mbh_setup,
            "gb": gb_setup,
            "psd": psd_setup,
        },
        general_info=general_setup,
        rank_info=rank_info,
        setup_function=setup_recipe,
        source_metadata={
            "mbh": mbh_metadata,
            "gb": gb_metadata,
            "psd": psd_metadata,
        }
    )

    curr_info = CurrentInfoGlobalFit(global_settings)

    return curr_info


if __name__ == "__main__":
    settings = get_global_fit_settings()
    breakpoint()
