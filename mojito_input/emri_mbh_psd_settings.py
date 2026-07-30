import h5py
import numpy as np
import os
import shutil
import logging

from functools import partial

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
from lisatools.globalfit.stock.erebor import PSDSetup, PSDSettings, EMRISetup, EMRISettings, MBHSetup, MBHSettings
from lisatools.sources.utils import from_lisa_frame, to_lisa_frame
from lisatools.sampling.moves.skymodehop import SkyMove

from eryn.prior import uniform_dist, log_uniform
from eryn.utils import TransformContainer
from eryn.prior import ProbDistContainer
from eryn.moves import StretchMove, TemperatureControl, DEMove, ConditionalFlowMove
from eryn.moves.tempering import make_ladder
from eryn.flows import ZukoFlow, WhiteningTransform, OneHotLeafConditioning, ProcessExecutor

from lisatools.domains import STFTSettings, FDSettings
from lisatools.sensitivity import XYZSensitivityBackend
from lisatools.globalfit.moves import GFCombineMove, MultiGPUPSDMove, TDMBHSpecialMove, EMRISpecialMove
from lisatools.globalfit.engine import GlobalFitSettings, GeneralSetup, GeneralSettings, RankInfo
from lisatools.utils.constants import YRSID_SI
from lisatools.globalfit.preprocessing import L1ProcessingStep
from lisatools.globalfit.recipe_steps import (
    SearchRecipeStep,
    PERecipeStep,
    IterationCountRecipeStep,
    build_psd_moves,
    build_mbh_moves_phenom,
    scatter_around_injection,
    mbh_catalogue_to_sampling_basis,
    emri_catalogue_to_sampling_basis,
    subtract_initial_signal
)

from lisatools.globalfit.postprocessing import (
    StochasticMetadata,
    SourceMetadata
)

logger = logging.getLogger(__name__)

MOJITO_REFERENCE_TIME = 97729089.327664

def attach_flow_trainers(info, label):
    """Start a ProcessExecutor for every ConditionalFlowMove among ``info.inner_moves``.

    The executor is attached here rather than in the settings function:
    CurrentInfoGlobalFit deep-copies the settings, and a live ProcessExecutor
    (mp queues + spawned worker) does not survive a deepcopy. Constructing it
    here also ensures only the sampler process spawns the trainer.

    Returns a copy of the inner-move list with the flow moves at zero weight
    (train-but-don't-propose), usable as a burn-in move set.
    """
    burnin_inner_moves = []
    for inner_move in info.inner_moves:
        move_obj = inner_move[0] if isinstance(inner_move, tuple) else inner_move
        move_weight = inner_move[1] if isinstance(inner_move, tuple) else 1.0
        if isinstance(move_obj, ConditionalFlowMove):
            move_obj.executor = ProcessExecutor(move_obj.flow, **move_obj.executor_init_kwargs)
            logger.info(
                f"{label} flow trainer started on "
                f"{move_obj.executor.worker_device} (sampling flow on cpu)"
            )
            burnin_inner_moves.append((move_obj, 0.0))
        else:
            burnin_inner_moves.append((move_obj, move_weight))
    return burnin_inner_moves


def setup_recipe(recipe, engine_info, curr, acs, priors, state):

    cp.cuda.runtime.setDevice(curr.general_info.gpus[0])

    general_info = curr.general_info
    nwalkers: int = general_info.nwalkers
    ntemps: int = general_info.ntemps
    Tmax: float = 1.0e6
    permute_every: int = 100

    emri_info = curr.source_info["emri"]
    mbh_info = curr.source_info["mbh"]
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
        run_threaded=True
    )

    psd_search_move = MultiGPUPSDMove(
        acs, priors, max_logl_mode=True, name="psd search move", **psd_move_kwargs
    )
    psd_pe_move = MultiGPUPSDMove(acs, priors, max_logl_mode=False, name="psd pe move", **psd_move_kwargs)

    psd_search_move.accepted = np.zeros((ntemps, nwalkers))
    psd_pe_move.accepted = np.zeros((ntemps, nwalkers))

    recipe.add_recipe_component(SearchRecipeStep(moves=[psd_search_move]), name="psd search")

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
        spread = 1e-6

        scatter_around_injection(
            state,
            "emri",
            injection_params,
            spread,
            priors=priors,
        )
        
    from lisatools.sources.emri.waveform import EMRITDIWaveform

    # todo test this
    emri_wave_gen = EMRITDIWaveform(**emri_info.initialize_kwargs)
    # breakpoint()
    subtract_initial_signal(acs, state, emri_wave_gen.get_signals_for_residuals, "emri", emri_info)

    if emri_info.betas is None:
        emri_info.betas = make_ladder(emri_info.ndim, ntemps=ntemps)
    betas_all = np.tile(emri_info.betas, (emri_info.nleaves_max, 1))
    state.sub_states["emri"].betas_all = betas_all
    logger.debug(f"EMRI betas: {emri_info.betas}")

    coords_shape_emri = (betas_all.shape[1], nwalkers, emri_info.nleaves_max, emri_info.ndim)

    burnin_inner_moves_emri = attach_flow_trainers(emri_info, "EMRI")

    emri_move_kwargs = dict(
        dcga=acs,
        waveform_gen=emri_wave_gen,
        branch_name="emri",
        coords_shape=coords_shape_emri,
        waveform_gen_method="get_signals_for_residuals",
        waveform_gen_kwargs=emri_info.waveform_kwargs.copy(),
        waveform_like_method="__call__",
        waveform_like_kwargs=emri_info.waveform_kwargs.copy(),
        num_repeats=emri_info.num_prop_repeats,
        transform_fn=emri_info.transform,
        priors=priors,
        inner_moves=emri_info.inner_moves,
        betas_all=betas_all,
        permute_every=25,
        pad_out_of_prior=True,
        flow_buffer_thin=5,
        run_async=True,
        run_threaded=True,
        randomize_split=True,
        batch_size_per_gpu=5,
    )

    emri_pe_move = EMRISpecialMove(**emri_move_kwargs)
    emri_pe_move.accepted = np.zeros((ntemps, nwalkers))


    #* ========================= *#
    
    # Initialize MBH walkers from catalogue injection parameters
    catalogue = getattr(curr.general_info, "catalogue", {})
    mbh_catalogue = catalogue.get("MBHB", {})
    if mbh_catalogue:
        mbh_info = curr.source_info["mbh"]
        injection_params_list = []
        for source_id in sorted(mbh_catalogue.keys()):
            entry = mbh_catalogue[source_id]
            sampling_params = mbh_catalogue_to_sampling_basis(entry, to_lisa_frame=partial(to_lisa_frame, orbits=general_info.orbits, t_ref=MOJITO_REFERENCE_TIME))
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
    mbh_wave_gen = PhenomTHMTDIWaveform(**mbh_info.initialize_kwargs)
    # breakpoint()
    subtract_initial_signal(acs, state, mbh_wave_gen.get_signals_for_residuals, "mbh", mbh_info)

    if mbh_info.betas is None:
        mbh_info.betas = make_ladder(mbh_info.ndim, ntemps=ntemps)
    betas_all = np.tile(mbh_info.betas, (mbh_info.nleaves_max, 1))
    state.sub_states["mbh"].betas_all = betas_all
    logger.debug(f"MBH betas: {mbh_info.betas}")

    coords_shape = (ntemps, nwalkers, mbh_info.nleaves_max, mbh_info.ndim)

    burnin_inner_moves_mbh = attach_flow_trainers(mbh_info, "MBH")

    mbh_move_kwargs = dict(
        dcga=acs,
        waveform_gen=mbh_wave_gen,
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
        flow_buffer_thin=5,
        run_async=True,
        run_threaded=True,
        randomize_split=True
    )

    mbh_pe_move = TDMBHSpecialMove(**mbh_move_kwargs)
    mbh_pe_move.accepted = np.zeros((ntemps, nwalkers))

    pe_moves = GFCombineMove(moves=[mbh_pe_move, emri_pe_move, psd_pe_move], share_temperature_control=False)
    recipe.add_recipe_component(PERecipeStep(moves=[pe_moves]), name="emri pe")

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

    input_basis = [
        r"$\log M$",
        r"$Q$",
        r"$s_{1z}$",
        r"$s_{2z}$",
        r"$d_L$",
        r"$\phi_{\rm ref}$",
        r"$\cos \iota$",
        r"$\psi_L$",
        r"$\lambda_L$",
        r"$\sin \beta_L$",
        r"$t_L$",
    ]

    output_basis = [
        r"$m_1$",
        r"$m_2$",
        r"$s_{1z}$",
        r"$s_{2z}$",
        r"$d_L$",
        r"$\phi_{\rm ref}$",
        r"$\iota$",
        r"$\psi$",
        r"$\alpha$",
        r"$\delta$",
        r"$t_{\rm plunge}$",
    ]

    key_map = {
        r"$\log M$": r"$m_1$",
        r"$Q$": r"$m_2$",
        r"$\cos \iota$": r"$\iota$",
        r"$\psi_L$": r"$\psi$",
        r"$\lambda_L$": r"$\alpha$",
        r"$\sin \beta_L$": r"$\delta$",
        r"$t_L$": r"$t_{\rm plunge}$",
    }

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
    
    _to_lisa_frame = partial(to_lisa_frame, orbits=general_set.orbits, t_ref=MOJITO_REFERENCE_TIME)
    _from_lisa_frame = partial(from_lisa_frame, orbits=general_set.orbits, t_ref=MOJITO_REFERENCE_TIME)

    mbh_transform_fn_in = {
        r"$m_1$": np.exp,
        r"$d_L$": gpc_to_mpc,
        r"$\iota$": np.arccos,
        r"$\delta$": np.arcsin,
        (r"$m_1$", r"$m_2$"): mT_Q,
        (r"$t_{\rm plunge}$", r"$\alpha$", r"$\delta$", r"$\psi$"): _from_lisa_frame,
    }

    inverse_mbh_transform_fn_in = {
        r"$m_1$": np.log,
        r"$d_L$": gpc_to_mpc,
        r"$\iota$": np.cos,
        r"$\delta$": np.sin,
        (r"$m_1$", r"$m_2$"): mT_Q,
        (r"$t_{\rm plunge}$", r"$\alpha$", r"$\delta$", r"$\psi$"): _to_lisa_frame,
    }

    transform = TransformContainer(
        input_basis=input_basis,
        output_basis=output_basis,
        parameter_transforms=mbh_transform_fn_in,
        inverse_parameter_transforms=inverse_mbh_transform_fn_in,
        key_map=key_map,
        fill_dict={},
    )

    periodic = {"mbh": {r"$\phi_{\rm ref}$": 2 * np.pi, r"$\lambda_L$": 2 * np.pi, r"$\psi_L$": np.pi}}

    priors_mbh = {
                r"$\log M$": uniform_dist(np.log(1e5), np.log(1e8)),
                r"$Q$": log_uniform(1., 10.),
                r"$s_{1z}$": uniform_dist(-0.99999999, +0.99999999),
                r"$s_{2z}$": uniform_dist(-0.99999999, +0.99999999),
                r"$d_L$": uniform_dist(1, 150.0), # uniform_dist(0.01, 1000.0),
                r"$\phi_{\rm ref}$": uniform_dist(0.0, 2 * np.pi),
                r"$\cos \iota$": uniform_dist(-1.0 + 1e-6, 1.0 - 1e-6),
                r"$\psi_L$": uniform_dist(0.0, np.pi), #is this right?
                r"$\lambda_L$": uniform_dist(0.0, 2 * np.pi),
                r"$\sin \beta_L$": uniform_dist(-1.0 + 1e-6, 1.0 - 1e-6),
                r"$t_L$": uniform_dist(0.0, general_set.Tobs + 3600.0),
            }
    priors = {"mbh": ProbDistContainer(priors_mbh)}

    #* ===== Flow proposal (online-trained normalizing flow) ===== *#

    nleaves_max_mbh = len(general_set.processor_init_kwargs["source_ids"]["mbhb"])

    # Trainer GPU: must stay off the ACS/sampling devices (general_set.gpus).
    # Shared with the EMRI trainer (separate worker processes on the same
    # device; each net is small, so memory is not a concern).
    flow_train_gpu = 4
    assert flow_train_gpu not in (general_set.gpus or []), (
        "flow trainer must run on a GPU not used by the ACS"
    )

    # periodic angles in the sampling basis (note psi_L has period pi)
    flow_periodic = {
        input_basis.index(name): (0.0, periodic["mbh"][name])
        for name in [r"$\phi_{\rm ref}$", r"$\psi_L$", r"$\lambda_L$"]
    }

    # Per-leaf whitening (shared=False, one-hot leaf conditioning): the 6 MBHBs
    # live in different regions of parameter space; per-condition maps keep each
    # leaf ~N(0,1) in latent space and the shared NSF learns per-leaf shape
    # modulations selected through the leaf one-hot. periodic_in_cholesky folds
    # the periodic sky/phase angles into the whitening Cholesky (exact, constant
    # Jacobian). See tasks/todo_flow_mbh_proposal.md for the offline acceptance
    # sweep that fixed train_noise=0 and this buffer/whitening config; the
    # per-mode mixture variant it also explored was pruned (measured +0.04
    # overall, failing the >=0.08 gate on the multimodal leaves 3/4/5).
    #
    # periodic_aliases (ZukoFlow default 1) sums the coords density over the
    # periodic alias lattice, which is what makes it normalized over the period.
    # It matters most HERE: periodic_in_cholesky compresses the sky/phase periods
    # to only ~4-19 latent sigma on leaves 3/4/5, so the flow spills across the
    # window and the legacy single-image density was off by up to 590 nats on
    # 1.4-3.2% of draws. Cost: those leaves keep 21-27 aliases (~20x on the
    # density call, ~1.3 s / 1500 pts); leaves 0/2 collapse to 1 alias and are
    # free. WATCH STEP TIME on relaunch -- set periodic_aliases=0 to trade the
    # normalization back for the old speed.
    flow = ZukoFlow(
        dims=len(input_basis),
        flow_class="NSF",
        device="cpu",  # proposal-side net; training runs on the executor's GPU
        conditioning=OneHotLeafConditioning(nleaves_max=nleaves_max_mbh),
        data_transform=WhiteningTransform(
            ndim=len(input_basis), periodic=flow_periodic, shared=False,
            periodic_in_cholesky=True,
        ),
        seed=general_set.random_seed,
        transforms=8,
        hidden_features=(128, 128, 128),
        bins=8,
    )

    # harvest_every=None: training data comes exclusively from the add/remove
    # move's per-leaf ring buffer (submit_by_leaf once per MCMC step); setup()
    # still polls the executor and hot-reloads new weights every step. Until
    # the first trained snapshot arrives the move is an identity pass-through.
    flow_move = ConditionalFlowMove(flow, "mbh", executor=None, harvest_every=None)

    # The ProcessExecutor itself is built in setup_recipe (a live executor does
    # not survive the deepcopy inside CurrentInfoGlobalFit); its parameters are
    # defined here. With flow_buffer_thin=5 the move submits
    # (num_repeats/5) * nwalkers = 240 semi-independent rows per leaf per step:
    # min_train_samples=1500 triggers the first fit ~7 steps after submissions
    # start and max_buffer_samples=6000 keeps a ~25-step history window per
    # leaf (staleness is free: splice-tested lag cost ~0 out to >=12 steps).
    # Overfit guards: val_split="temporal" holds out the NEWEST rows so early
    # stopping measures fresh-point NLL (the MH factor). train_noise MUST stay
    # 0 here: the jitter is applied per-dim in the whitened latent space, where
    # per-condition whitening makes each sky-mode island (and the razor
    # phase/timing correlations) far thinner than 0.1 -- 0.1 jitter puffed the
    # training data by 10-100x in those directions and cost the proposals
    # 10^3-10^5 nats (splice test 2026-07-14: acceptance ceiling ~0.60;
    # train_noise 0.1 -> 0 + buffer 2k -> 4k rows moved implied acceptance
    # from ~0.000 to ~0.2-0.3; see tasks/todo_flow_mbh_proposal.md).
    flow_artifacts_dir = os.path.join(general_set.artifacts_file_dir, "mbh_flow")
    flow_move.executor_init_kwargs = dict(
        worker_device=f"cuda:{flow_train_gpu}",
        epochs_per_round=150,
        min_train_samples=1500,
        max_buffer_samples=6000,
        fit_kwargs=dict(
            batch_size=1024,
            lr=1e-3,
            lr_annealing=True,
            optimizer="adamw",
            patience=30,
            validation_fraction=0.15,
            val_split="temporal",
            train_noise=0.0,
        ),
        refit_transform_every=1,
        torch_num_threads=2,
        seed=general_set.random_seed,
        save_path=os.path.join(flow_artifacts_dir, "mbh_flow_latest.h5"),
        diagnostics_dir=flow_artifacts_dir,
        plot_corner=True,
    )

    sky_inds_map = dict(cosinc=6, lam=8, sinbeta=9, psi=7, t_ref=10)
    inner_moves = [
        (StretchMove(), 0.35),
        (DEMove(), 0.3),
        (SkyMove(ind_map=sky_inds_map, coord_frame="lisa", which='lat'), 0.05),
        (SkyMove(ind_map=sky_inds_map, coord_frame="lisa", which='long'), 0.05),
        (SkyMove(ind_map=sky_inds_map, coord_frame="lisa", which='both'), 0.05),
        (flow_move, 0.2),
    ]

    mbh_settings = MBHSettings(
        log_dir=general_set.artifacts_file_dir,
        Tobs=general_set.Tobs,
        dt=general_set.dt,
        initialize_kwargs=waveform_init_kwargs,
        transform=transform,
        periodic=periodic,
        priors=priors,
        waveform_kwargs=waveform_runtime_kwargs,
        nleaves_max=nleaves_max_mbh,
        nleaves_min=nleaves_max_mbh,
        ndim=11,
        num_prop_repeats=50,
        betas=betas,
        inner_moves=inner_moves
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


def get_emri_erebor_settings(general_set: GeneralSetup) -> EMRISetup:

    ntemps_emri = 1

    waveform_model = "EMRITDIWaveform"
    waveform_model_code_link = "https://github.com/Erebor-L2D/LISAanalysistools/blob/9d63bb1e63e7b8f640d3780551d9421df5245992/src/lisatools/sources/emri/waveform.py#L130"
    prior_model_code_link = ""
    frequency_ranges = [(general_set.start_freq, general_set.end_freq)]

    wave_kwargs = dict(
            sum_kwargs = {"pad_output": True,},
            inspiral_kwargs = {"DENSE_STEPPING": 0,  # sparsely sampled trajectory
                #"max_init_len": int(1e8),  # length of trajectories well under 1000
                "buffer_length": int(1e2),  # length of trajectories well under 1000
                "err": 1e-8
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
        # Launch the response/TDI kernels without their internal
        # cudaDeviceSynchronize (the move synchronizes before consuming
        # results); together with the GIL-released response bindings this lets
        # the per-GPU threads overlap instead of alternating.
        run_async=True,  # TODO restore True once memory headroom exists (cudaMallocAsync vs cupy pool contention + hard-abort on OOM)
        **response_kwargs,
    )

    waveform_runtime_kwargs = dict(mode_selection_threshold = 1e-4)

    betas = 1 / 1.2 ** np.arange(ntemps_emri)  # Geometric ladder with ratio 1.2

    input_basis = [
        "log_m1",
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
        "log_m1": "m1",
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
            "log_m1": uniform_dist(np.log(1e5), np.log(5e6)),  # log m1
            "m2": uniform_dist(1, 200),  # m2
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

    #* ===== Flow proposal (online-trained normalizing flow) ===== *#

    nleaves_max_emri = len(general_set.processor_init_kwargs["source_ids"]["emri"])

    # Trainer GPU: must stay off the ACS/sampling devices (general_set.gpus).
    flow_train_gpu = 4
    assert flow_train_gpu not in (general_set.gpus or []), (
        "flow trainer must run on a GPU not used by the ACS"
    )

    # periodic angles in the sampling basis
    flow_periodic = {
        input_basis.index(name): (0.0, periodic["emri"][name])
        for name in ["phiK", "Phi_phi0", "Phi_r0", "alpha"]
    }

    flow = ZukoFlow(
        dims=len(input_basis),
        flow_class="NSF",
        device="cpu",  # proposal-side net; training runs on the executor's GPU
        conditioning=OneHotLeafConditioning(nleaves_max=nleaves_max_emri),
        # periodic_aliases (ZukoFlow default 1) is FREE here: the 4 EMRI phase
        # periods are 22-1748 latent sigma wide, so every alias is pruned and
        # the density collapses to a single evaluation (measured single-image
        # error exactly 0 on both leaves). Kept on so the density stays
        # normalized if a future fit compresses those periods.
        # periodic_in_cholesky: the 4 EMRI phase angles carry razor
        # correlations with the intrinsic parameters; whitening them jointly
        # lifted exact-MH offline acceptance 0.43 -> 0.50 (2026-07-14).
        data_transform=WhiteningTransform(
            ndim=len(input_basis), periodic=flow_periodic, shared=False,
            periodic_in_cholesky=True,
        ),
        seed=general_set.random_seed,
        transforms=8,
        hidden_features=(128, 128, 128),
        bins=8,
    )

    # harvest_every=None: training data comes exclusively from the add/remove
    # move's per-leaf ring buffer (submit_by_leaf once per MCMC step); setup()
    # still polls the executor and hot-reloads new weights every step. Until
    # the first trained snapshot arrives the move is an identity pass-through.
    flow_move = ConditionalFlowMove(flow, "emri", executor=None, harvest_every=None)

    # The ProcessExecutor itself is built in setup_recipe (a live executor does
    # not survive the deepcopy inside CurrentInfoGlobalFit); its parameters are
    # defined here. With flow_buffer_thin=5 the move submits
    # (num_repeats/5) * nwalkers = 240 semi-independent rows per leaf per step:
    # min_train_samples=1500 triggers the first fit ~7 steps after submissions
    # start and max_buffer_samples=4000 keeps a ~17-step history window per
    # leaf (staleness splice-tested free; exact-MH offline scoring 2026-07-14:
    # 2x buffer lifted EMRI acceptance 0.31->0.44, 3x added nothing).
    # Anti-memorization guards (the trainer previously collapsed onto the
    # correlated walker tracks, killing acceptance): val_split="temporal" holds
    # out the NEWEST rows so early stopping measures fresh-point NLL (the MH
    # factor). train_noise stays 0: the jitter acts in whitened latent space
    # and smears the sharpest posterior directions (exact-MH acceptance 0.25
    # at noise 0.1 vs 0.31-0.44 at noise 0). The temporal split + patience are
    # the anti-memorization guards that remain.
    # Monitoring: the latest fit is checkpointed atomically to HDF5 (reload
    # later via ZukoFlow.load(save_path)) and each training round writes loss/
    # val-NLL plots plus a corner overlay of training samples vs flow draws.
    flow_artifacts_dir = os.path.join(general_set.artifacts_file_dir, "emri_flow")
    flow_move.executor_init_kwargs = dict(
        worker_device=f"cuda:{flow_train_gpu}",
        epochs_per_round=150,
        min_train_samples=1500,
        max_buffer_samples=4000,
        fit_kwargs=dict(
            batch_size=1024,
            lr=1e-3,
            lr_annealing=True,
            optimizer="adamw",
            patience=30,
            validation_fraction=0.15,
            val_split="temporal",
            train_noise=0.0,
        ),
        refit_transform_every=1,
        torch_num_threads=2,
        seed=general_set.random_seed,
        save_path=os.path.join(flow_artifacts_dir, "emri_flow_latest.h5"),
        diagnostics_dir=flow_artifacts_dir,
        plot_corner=True,
    )

    emri_settings = EMRISettings(
        log_dir=general_set.artifacts_file_dir,
        Tobs=general_set.Tobs,
        dt=general_set.dt,
        initialize_kwargs=waveform_init_kwargs,
        transform=transform,
        periodic=periodic,
        priors=priors,
        waveform_kwargs=waveform_runtime_kwargs,
        nleaves_max=nleaves_max_emri,
        nleaves_min=nleaves_max_emri,
        ndim=12,
        num_prop_repeats=50,
        betas=betas,
        inner_moves=[(StretchMove(), 0.4), (DEMove(), 0.4), (flow_move, 0.2)],
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

    source_ids = dict(
        emri=[0, 3],
        mbhb=[18, 5, 16, 7, 2, 12]
        )

    Tobs = 1 * YRSID_SI
    dt = 5.
    start_freq = 1e-4
    end_freq = 2.9e-2

    head_dir = "/data/asantini/globalfit/erebor_org_setup/mojito_runs/"
    data_input_path = "/data/asantini/globalfit/MOJITO_DATA/mojito_light_2p5s/"
    base_file_name = "test_flow_joint_sources"
    file_store_dir = head_dir

    gpus = [5, 6]
    cp.cuda.runtime.setDevice(gpus[0])
    # Restrict JAX to only see the target GPU — must be set before JAX backend init
    import jax

    jax.config.update("jax_cuda_visible_devices", ",".join(str(gpu) for gpu in gpus))

    backend = "cuda12x" if gpus is not None else "cpu"
    nwalkers = 24
    ntemps = 4

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
        source_types=["noise", "mbhb", "emri"],  #'vgb', 'gb'
        source_ids=source_ids,
        verbose=True,
        do_plots=True,
        orbits_class=L1Orbits,
        store_individual_timeseries=True,
        orbits_kwargs=dict(force_backend="cpu", frame="icrs"),  # icrs
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
        "duration": 200 * 3600,  # seconds — duration to trim from each end
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
    ###  MBH Settings  ###############
    ##################################
    ##################################

    mbh_setup, mbh_metadata = get_mbh_erebor_settings(general_setup)

    ##################################
    ##################################
    ###  EMRI Settings  ##############
    ##################################
    ##################################

    emri_setup, emri_metadata = get_emri_erebor_settings(general_setup)

    ##############
    ## READ OUT ##
    ##############

    global_settings = GlobalFitSettings(
        source_info={
            "emri": emri_setup,
            "mbh": mbh_setup,
            "psd": psd_setup,
        },
        general_info=general_setup,
        rank_info=rank_info,
        setup_function=setup_recipe,
        source_metadata={
            "emri": emri_metadata,
            "mbh": mbh_metadata,
            "psd": psd_metadata,
        }
    )

    curr_info = CurrentInfoGlobalFit(global_settings)

    return curr_info


if __name__ == "__main__":
    settings = get_global_fit_settings()
    breakpoint()
