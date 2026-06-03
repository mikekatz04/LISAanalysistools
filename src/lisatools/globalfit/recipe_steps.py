import time
import logging
import typing
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import cupy as cp

from lisatools.analysiscontainer import AnalysisContainerArray
from lisatools.datacontainer import DataResidualArray

from bbhx.utils.transform import SSB_to_LISA
from gbgpu.gbgpu import GBGPU
from eryn.moves.tempering import TemperatureControl, make_ladder
from eryn.prior import ProbDistContainer

from ..sources.utils import icrs_to_ecliptic, evolve_galactic_binary
from .engine import Setup, GlobalFitEngine
from .moves import PSDMove, ResidualAddOneRemoveOneMove, GBSpecialRJPriorMove, GBSpecialRJSerialSearchMCMC, GBSpecialRJRefitMove
from .moves.gbspecialstretch import GBSpecialBase
from .recipe import RecipeStep, BaseRecipeStep
from .run import CurrentInfoGlobalFit
from .state import GFState
from .stock.erebor import GBSetup, GeneralSetup

logger = logging.getLogger(__name__)

MOJITO_REFERENCE_TIME = 97729089.327664

class SearchRecipeStep(BaseRecipeStep):
    """
    Recipe step that completes immediately (one-shot search/initialisation). 
    Used when the stopping criterion is embedded in the move.
    """

    def stopping_function(self, *args, **kwargs):
        return True


class PERecipeStep(BaseRecipeStep):
    """Recipe step that runs indefinitely (ongoing parameter estimation)."""

    def stopping_function(self, *args, **kwargs):
        return False


class RJRecipeStep(BaseRecipeStep):

    def __init__(
        self, 
        *args, 
        convergence_iter: int = 5, 
        thin_by: int = 1, 
        **kwargs
    ):
        BaseRecipeStep.__init__(self, *args, **kwargs)
        self.convergence_iter = convergence_iter
        self.thin_by = thin_by

    def stopping_function(
        self, 
        i, 
        sample, 
        sampler: GlobalFitEngine
    ) -> bool:

        if not hasattr(self, "st"):
            self.st = time.perf_counter()

        current_iter = sampler.backend.iteration

        assert isinstance(current_iter, (int, np.integer))
        
        stop = False
        if current_iter > self.convergence_iter:
            #? Actual convergence should be related to the same number of sources above SNR XX for Y itterations
            nleaves_cc = sampler.backend.get_nleaves(branch_names=["gb"], temp_index=0)["gb"]

            # do not include most recent
            nleaves_cc_max_old = nleaves_cc[:-self.convergence_iter].max()
            nleaves_cc_max_new = nleaves_cc[-self.convergence_iter:].max()

            if nleaves_cc_max_old >= nleaves_cc_max_new:
                stop = True

            else:
                stop = False

            
            dur = (time.perf_counter() - self.st) / 3600.0  # hours
            print(f"Previous nleaves: {nleaves_cc_max_old} --> new nleaves: {nleaves_cc_max_new}")
            print(f"TIME SINCE START: {dur} hours")

        return stop
        
    def setup_run(
        self, 
        iteration, 
        last_sample, 
        sampler: GlobalFitEngine
    ):
        # TODO: maybe make this the default setup
        sampler.moves = self.moves
        sampler.weights = self.weights
        sampler.yield_step = self.thin_by
        sampler.checkpoint_step = self.thin_by
        # sampler.override_thin_by = self.thin_by --> # TODO check this one
        
        for move in self.moves: 
            if sampler.periodic is not None and move.periodic is None:
                print(f"Setting periodicity of move {move} to {sampler.periodic}")
                move.periodic = sampler.periodic
            if sampler.temperature_control is not None and move.temperature_control is None:
                print(f"Setting temperature control of move {move} to {sampler.temperature_control}")
                move.temperature_control = sampler.temperature_control
            
            # TODO: do we also need to set these? I think the current settings setup has ntemps covered, not sure about temp_cntrl            
            # move.ntemps = sampler.ntemps 
            

def scatter_around_injection(
    state: GFState,
    branch_name: str,
    injection_params: np.ndarray,
    spread: float | np.ndarray,
    reverse_transform: typing.Callable | None = None,
    betas: np.ndarray | None = None,
    priors: ProbDistContainer | None = None,
    max_resample_tries: int = 50,
):
    """
    Initialize branch coordinates by scattering walkers around injection parameters.

    For each leaf, draws coordinates from a multivariate Gaussian centered on
    the (transformed) injection parameters.  Higher-temperature chains receive
    proportionally wider scatter when ``betas`` is provided.  Initialized
    leaves are marked as active (``inds = True``).

    When ``priors`` is supplied, any draw that lies outside the prior support
    (``logpdf == -inf``) is rejected and redrawn.  This is essential for
    sampling bases that contain ``arcsin``/``arccos`` transforms (e.g. MBH
    ``sin_beta``, ``cos_iota``) where an out-of-support initial coordinate
    silently produces NaN once the transform pipeline runs, eventually
    surfacing as a CUDA illegal-memory-access in downstream kernels.

    The function modifies ``state`` in-place, so it can be called from
    ``setup_recipe`` (before MCMC) or from a ``RecipeStep.setup_run``
    (between recipe phases).

    Parameters
    ----------
    state : GFState
        Sampler state to modify in-place.
    branch_name : str
        Name of the branch to initialize (e.g. ``"mbh"``, ``"emri"``).
    injection_params : ndarray
        True source parameters in the **physical** (waveform) basis.
        Shape ``(ndim_phys,)`` for a single leaf, or
        ``(nleaves, ndim_phys)`` for multiple leaves.
    spread : float, ndarray
        Controls the width of the Gaussian scatter (in sampling basis).

        * *scalar* -- isotropic standard deviation for every parameter.
        * *1-D array* ``(ndim,)`` -- per-parameter standard deviations.
        * *2-D array* ``(ndim, ndim)`` -- full covariance matrix
          (shared across leaves).
        * *3-D array* ``(nleaves, ndim, ndim)`` -- per-leaf covariance
          matrices.
    reverse_transform : callable, optional
        Converts a single parameter vector from physical basis to
        sampling basis: ``(ndim_phys,) -> (ndim_sampling,)``.
        If *None*, ``injection_params`` are assumed to already be in
        the sampling basis.
    betas : ndarray of shape ``(ntemps,)``, optional
        Inverse-temperature ladder.  When provided the covariance for
        temperature index *t* is scaled by ``1 / betas[t]`` so that
        hotter chains start with a wider scatter.
    priors : ProbDistContainer, optional
        Prior container for ``branch_name``.  When given, walker draws
        outside the prior support are rejected and redrawn (up to
        ``max_resample_tries`` per walker).  Without it, this routine can
        seed walkers that lie outside arcsin/arccos domains, producing
        NaNs once the transform pipeline runs.
    max_resample_tries : int, optional
        Hard cap on resampling attempts per walker before raising.  Only
        used when ``priors`` is supplied.  Default 50 — for any reasonable
        scatter and prior, this is wildly more than needed; the cap exists
        only to surface pathological configs (e.g. the entire scatter
        landing outside the prior) instead of looping forever.
    """
    coords = state.branches_coords[branch_name]
    ntemps, nwalkers, nleaves_max, ndim = coords.shape

    injection_params = np.atleast_2d(np.asarray(injection_params, dtype=float))
    # Physical → sampling basis
    if reverse_transform is not None:
        injection_sampling = np.array([reverse_transform(p) for p in injection_params])
    else:
        injection_sampling = injection_params

    nleaves_init = injection_sampling.shape[0]
    assert (
        nleaves_init <= nleaves_max
    ), f"More injection leaves ({nleaves_init}) than nleaves_max ({nleaves_max})"
    assert (
        injection_sampling.shape[-1] == ndim
    ), f"Injection ndim ({injection_sampling.shape[-1]}) != branch ndim ({ndim})"

    # Build covariance matrix/matrices
    spread = np.asarray(spread, dtype=float)
    if spread.ndim == 0:
        cov = spread.item() ** 2 * np.eye(ndim)
        covs = np.tile(cov, (nleaves_init, 1, 1))
    elif spread.ndim == 1:
        cov = np.diag(spread**2)
        covs = np.tile(cov, (nleaves_init, 1, 1))
    elif spread.ndim == 2:
        covs = np.tile(spread, (nleaves_init, 1, 1))
    elif spread.ndim == 3:
        assert spread.shape == (nleaves_init, ndim, ndim)
        covs = spread
    else:
        raise ValueError(f"spread must be scalar, 1-D, 2-D, or 3-D; got shape {spread.shape}")

    if betas is not None:
        logger.info(f"Scaling initial covariance by betas: {betas}")

    leaf_prior = priors[branch_name] if priors is not None else None

    for leaf in range(nleaves_init):
        center = injection_sampling[leaf]
        leaf_cov = covs[leaf]
        for t in range(ntemps):
            if betas is not None:
                scaled_cov = leaf_cov / max(betas[t], 1e-10)
            else:
                scaled_cov = leaf_cov

            draws = np.random.multivariate_normal(center, scaled_cov, size=nwalkers)

            if leaf_prior is not None:
                bad = ~np.isfinite(leaf_prior.logpdf(draws))
                tries = 0
                while bad.any():
                    if tries >= max_resample_tries:
                        n_bad = int(bad.sum())
                        raise RuntimeError(
                            f"scatter_around_injection: leaf={leaf} temp={t}: "
                            f"{n_bad}/{nwalkers} walkers still outside prior support "
                            f"after {max_resample_tries} resample passes. "
                            f"Injection sampling-basis params = {center.tolist()}. "
                            f"Likely the injection sits on / outside a prior edge, or "
                            f"the scatter is too wide for the prior range."
                            f"Last resampled points (showing up to 10): {draws[bad][:10].tolist()}"
                        )
                    redraws = np.random.multivariate_normal(
                        center, scaled_cov, size=int(bad.sum())
                    )
                    draws[bad] = redraws
                    bad = ~np.isfinite(leaf_prior.logpdf(draws))
                    tries += 1

            coords[t, :, leaf] = draws

        state.branches_inds[branch_name][:, :, leaf] = True


def mbh_catalogue_to_sampling_basis(catalogue_entry: dict, trim_duration: float = 0.0) -> np.ndarray:
    """Convert a single Mojito MBHB catalogue entry to MBH sampling basis.

    The sampling basis is:
    ``[logM, q, s1z, s2z, dist, phi_ref, cos_iota, psi, lam, sin_beta, t_plunge]``

    Parameters
    ----------
    catalogue_entry : dict
        Dictionary of catalogue parameters for one MBHB source, as
        stored by ``L1DataLoader.catalogue['MBHB'][source_id]``.

    Returns
    -------
    np.ndarray
        Parameter vector of shape ``(11,)`` in the MBH sampling basis
        (LISA frame for sky/time parameters).
    """
    m1 = float(catalogue_entry["PrimaryMassSSBFrame"])
    m2 = float(catalogue_entry["SecondaryMassSSBFrame"])

    # Ensure m1 >= m2
    if m2 > m1:
        m1, m2 = m2, m1

    logM = np.log(m1 + m2)
    q = m2 / m1
    Q = m1 / m2
    logq = np.log(q)

    s1z = float(catalogue_entry["PrimarySpinCompZ"])
    s2z = float(catalogue_entry["SecondarySpinCompZ"])
    dist = float(catalogue_entry["LuminosityDistance"]) / 1e3  # Mpc -> Gpc
    phi_ref = float(catalogue_entry["PhaseReferenceSourceFrame"]) % (2 * np.pi)
    cos_iota = np.cos(float(catalogue_entry["InclinationAngle"]))

    # Sky coordinates: ICRS -> ecliptic -> SSB -> LISA
    ra = float(catalogue_entry["RightAscension"]) % (2 * np.pi)
    dec = float(catalogue_entry["Declination"])
    sin_dec = np.sin(dec)
    psi_icrs = float(catalogue_entry["PolarisationAngle"]) % np.pi  # ensure polarization is within [0, pi]
    lam_ecl, beta_ecl, psi_ssb = icrs_to_ecliptic(ra, dec, psi_icrs)
    t_ssb = float(catalogue_entry["TimeCoalescencePhenomTPHMSSBFrame"])

    # logger.debug(f"Catalogue entry: RA={ra}, Dec={dec}, psi_icrs={psi_icrs}, t_ssb={t_ssb}")
    
    # t_L, lam_L, beta_L, psi_L = SSB_to_LISA(t_ssb, lam_ecl, beta_ecl, psi_ssb)
    
    # lam_L = lam_L % (2 * np.pi)
    # psi_L = psi_L % np.pi
    # logger.debug(f"Converted to LISA frame: t_L={t_L}, lambda_L={lam_L}, beta_L={beta_L}, psi_L={psi_L}")
    # sin_beta_L = np.sin(beta_L)

    #return np.array([logM, Q, s1z, s2z, dist, phi_ref, cos_iota, psi_L, lam_L, sin_beta_L, t_L])
    return np.array([logM, Q, s1z, s2z, dist, phi_ref, cos_iota, psi_icrs, ra, sin_dec, t_ssb])


def gb_catalogue_to_sampling_basis(catalogue_entry: dict, trim_duration: float = 0.0) -> np.ndarray:
    """Converts the (V)GB catalogue entries to the sampling basis. 
    The index 0 in f0 and phi0 refer to the frequency and phase at the start of the data.

    The sampling basis is:
    ``[logA, f0 [mHz], fdot, phi0, cos_iota, psi, lam, sin_beta]``

    Parameters
    ----------
    catalogue_entry : dict
        Dictionary of catalogue parameters for all (V)GBs, as
        stored by ``L1DataLoader.catalogue['(V)GB'][source_id]``.

    Returns
    -------
    np.ndarray
        Parameter vector of shape ``(8,)`` in the (V)GB sampling basis
        (ICRS or LISA frame for sky/time parameters).
    """
    amp = np.array(catalogue_entry["Amplitude"])
    logA = np.log(amp)
    
    f_ref = np.array(catalogue_entry["GW22FrequencySourceFrame"])
    fdot = np.array(catalogue_entry["GW22FrequencyDerivativeSourceFrame"])
    phi_ref = np.array(catalogue_entry["TrueAnomaly"])
    t_ref = np.unique(np.array(catalogue_entry["TimeReferenceSSBFrame"]))
    
    assert len(t_ref) == 1
    t_ref = t_ref.item()
    t_init = t_ref + trim_duration
    
    f_init, phi_init, _ = evolve_galactic_binary(t_ref, t_init, f_ref, phi_ref, fdot, phase_sign=-1)
    
    f0_mHz = f_init * 1e3
    cos_iota = np.cos(np.array(catalogue_entry["InclinationAngle"]))# % (np.pi)

    ra = np.array(catalogue_entry["RightAscension"]) # alpha
    dec = np.array(catalogue_entry["Declination"]) # delta
    psi_icrs = np.array(catalogue_entry["PolarisationAngle"]) % np.pi  # ensure polarization is within [0, pi]
    # lam_ecl, beta_ecl, psi_ecl= icrs_to_ecliptic(ra, dec, psi_icrs)

    alpha = ra % (2 * np.pi)
    sin_delta = np.sin(dec)

    return np.array([logA, f0_mHz, fdot, phi_init, cos_iota, psi_icrs, alpha, sin_delta]).T


def setup_state_for_injection(curr: CurrentInfoGlobalFit, state: GFState, source_type: str, branch_name: str, spread: float | np.ndarray  = 1e-5, subset_inds = None, priors: ProbDistContainer | None = None):
    """Initialize 'branch_name' walkers from catalogue injection parameters"""

    catalogue = getattr(curr.general_info, "catalogue", {})
    catalogue = catalogue.get(source_type, {})
    if catalogue:
        injection_params_list = []
        for source_id in sorted(catalogue.keys()):
            entry = catalogue[source_id]
            
            func_name = f"{branch_name}_catalogue_to_sampling_basis"
            conversion_func = globals().get(func_name)

            assert conversion_func and callable(conversion_func), f"catalogue_to_sampling_basis function for {branch_name} was not found."
            assert curr.general_info.preprocess_kwargs

            trim_duration = curr.general_info.data_t0 - MOJITO_REFERENCE_TIME # curr.general_info.data_processor.original_t0
            sampling_params = conversion_func(entry, trim_duration=trim_duration)

            injection_params_list.append(sampling_params)

        injection_params = np.array(injection_params_list)
        
        ndim = state.branches_coords[branch_name].shape[-1]
        if injection_params.ndim == 3:
            injection_params = injection_params.reshape(-1, ndim)

        if subset_inds is not None:
            injection_params = injection_params[subset_inds, :]
        
        # Store injection truths for diagnostic plots
        try:
            setattr(curr.source_info[branch_name], "injection", injection_params)
        except AttributeError:
            logger.warning(f"No injection data is saved for {branch_name}.")
        
        scatter_around_injection(
            state, branch_name, injection_params, spread, betas=getattr(curr.source_info[branch_name], "betas"), priors=priors
        )


def subtract_initial_signal(
    acs: AnalysisContainerArray,
    state: GFState,
    wave_gen: typing.Callable,
    source_name: str,
    source_info: Setup,
):
    xp = acs.xp
    if np.any(inds := state.branches_inds[source_name][0]):
        logger.info(f"Subtracting initial signals for {source_name}")
        counter = 0
        for leaf in range(inds.shape[-1]):
            if inds[0, leaf]:
                assert np.all(inds[:, leaf])
                inj_coords = state.branches_coords[source_name][0, :, leaf]
                inj_coords_in = xp.asarray(source_info.transform.both_transforms(inj_coords))

                # logger.debug(f"CUDA device here: {cp.cuda.runtime.getDevice()}")  # Debugging line to check current CUDA device

                # C-order columns are non-contiguous (stride = ndim*8), so ascontiguousarray
                # is forced to allocate a fresh, pool-aligned buffer for each parameter —
                # avoiding the misalignment that arises with F-order when nwalkers is odd.
                signals_in = wave_gen(*[xp.ascontiguousarray(col) for col in inj_coords_in.T], **source_info.waveform_kwargs)
                for w in range(len(signals_in)):
                    ll_here = acs.acs[w].template_likelihood(template=DataResidualArray(signals_in[w]), include_psd_info=False)
                #     if acs.gpus is not None:
                #         device = acs.gpu_map[w]
                #         with cp.cuda.Device(device):
                #             ll_here = acs.acs[w].template_likelihood(template=DataResidualArray(signals_in[w]), include_psd_info=False)
                #     else:
                #         ll_here = acs.acs[w].template_likelihood(template=DataResidualArray(signals_in[w]), include_psd_info=False)

                    logger.debug(f"Initial log-likelihood contribution from walker {w}, leaf {leaf}: {ll_here}.")
                acs.add_signal_to_residual(signals_in)
                counter += 1
                
                # if acs.gpus is not None:
                #     acs.synchronize()  # Ensure GPU computations are complete before logging
                #     # acs.xp.get_default_memory_pool().free_all_blocks()
                #     cp.cuda.runtime.setDevice(main_device)  # Switch back to main device after subtraction
                #     logger.debug(f"Switched back to main CUDA device {main_device} after subtraction.")
                    
        logger.debug(f"Subtracted {counter} initial signals for {source_name}")
    else:
        logger.info(f"No initial signals for {source_name}")

    #breakpoint()

def build_psd_moves(
    engine_info: Setup,
    curr: CurrentInfoGlobalFit,
    acs: AnalysisContainerArray,
    priors: dict,
    *,
    num_repeats: int = 60,
    permute_every: int = 50,
    Tmax: float = 1e6,
) -> tuple[PSDMove, PSDMove]:
    """Build PSD search and PE moves.

    Both moves share the same ``acs``, ``priors``, and
    ``TemperatureControl`` instance, so updates to ``acs`` (e.g. signal
    subtraction by another branch) are visible to both moves at runtime.

    Parameters
    ----------
    engine_info :
        Engine info object exposing ``ndims``.
    curr : CurrentInfoGlobalFit
        Current run info; reads ``source_info["psd"]`` and ``general_info``.
    acs :
        Shared analysis container (passed by reference).
    priors : dict
        Shared priors dict (passed by reference).
    num_repeats : int, optional
        Number of internal PSD move repeats. Default 60.
    Tmax : float, optional
        Maximum temperature for ``TemperatureControl``. Default 1e6.

    Returns
    -------
    psd_search_move : PSDMove
    psd_pe_move : PSDMove
    """
    general_info = curr.general_info
    nwalkers: int = general_info.nwalkers
    ntemps: int = general_info.ntemps
    psd_info = curr.source_info["psd"]
    galfor_info = curr.source_info.get("galfor", None)

    effective_ndim = engine_info.ndims["psd"] if galfor_info is None else engine_info.ndims["galfor"] + engine_info.ndims["psd"] 
    temperature_control = TemperatureControl(
        effective_ndim, nwalkers, ntemps=ntemps, Tmax=Tmax, permute=False
    )

    psd_move_kwargs = dict(
        num_repeats=num_repeats,
        permute_every=permute_every,
        live_dangerously=True,
        psd_transform_fn=psd_info.transform,
        galfor_transform_fn=galfor_info.transform if galfor_info is not None else None,
        sensitivity_backend=general_info.sensitivity_backend,
        temperature_control=temperature_control,
        use_gpu=True,
    )

    psd_search_move = PSDMove(
        acs, priors, max_logl_mode=True, name="psd search move", **psd_move_kwargs
    )
    psd_pe_move = PSDMove(acs, priors, max_logl_mode=False, name="psd pe move", **psd_move_kwargs)

    psd_search_move.accepted = np.zeros((ntemps, nwalkers))
    psd_pe_move.accepted = np.zeros((ntemps, nwalkers))

    return psd_search_move, psd_pe_move


def build_mbh_moves_phenom(
    curr: CurrentInfoGlobalFit, 
    acs: AnalysisContainerArray, 
    priors: dict, 
    state: GFState,
    permute_every: int = 20,
    ) -> tuple[typing.Callable, ResidualAddOneRemoveOneMove]:
    """Build MBH PE move using ``PhenomTHMTDIWaveform`` + ``ResidualAddOneRemoveOneMove``.

    Sets ``state.sub_states['mbh'].betas_all`` as a side effect.

    Parameters
    ----------
    curr : CurrentInfoGlobalFit
        Current run info; reads ``source_info["mbh"]`` and ``general_info``.
    acs :
        Shared analysis container (passed by reference).
    priors : dict
        Shared priors dict (passed by reference).
    state :
        Current sampler state; ``sub_states["mbh"].betas_all`` is set here.

    Returns
    -------
    wave_gen : PhenomTHMTDIWaveform
    mbh_pe_move : ResidualAddOneRemoveOneMove
    """
    from lisatools.sources.bbh.waveform import PhenomTHMTDIWaveform

    mbh_info = curr.source_info["mbh"]
    nwalkers = curr.general_info.nwalkers
    ntemps = curr.general_info.ntemps

    wave_gen = PhenomTHMTDIWaveform(**mbh_info.initialize_kwargs)
    # breakpoint()
    subtract_initial_signal(acs, state, wave_gen.get_signals_for_residuals, "mbh", mbh_info)

    if mbh_info.betas is None:
        mbh_info.betas = make_ladder(mbh_info.ndim, ntemps=ntemps)
    betas_all = np.tile(mbh_info.betas, (mbh_info.nleaves_max, 1))
    state.sub_states["mbh"].betas_all = betas_all
    logger.debug(f"MBH betas: {mbh_info.betas}")

    coords_shape = (ntemps, nwalkers, mbh_info.nleaves_max, mbh_info.ndim)

    mbh_move_args = (
        "mbh",  # branch_name
        coords_shape,
        wave_gen.get_signals_for_residuals,
        # tempering_kwargs,
        mbh_info.waveform_kwargs.copy(),  # waveform_gen_kwargs
        dict(propagate_data_res_kwargs=False),  # waveform_like_kwargs
        acs,
        mbh_info.num_prop_repeats,
        mbh_info.transform,
        priors,
        mbh_info.inner_moves,
    )

    mbh_pe_move = ResidualAddOneRemoveOneMove(*mbh_move_args, betas_all=betas_all, permute_every=permute_every)
    mbh_pe_move.accepted = np.zeros((ntemps, nwalkers))

    return wave_gen, mbh_pe_move


@dataclass
class GBWaveformDict(typing.TypedDict):
    dt: float
    T: float
    use_c_implementation: bool
    start_freq_ind: int
    tdi_channel_setup: str
    tdi2: bool
    window: None | str
    window_alpha: float


def build_gb_moves(
    engine_info: Setup,
    curr: CurrentInfoGlobalFit,
    acs: AnalysisContainerArray,
    priors: dict,
    state: GFState,
    *,
    Tmax: float = 1e6,
) -> typing.Tuple[typing.List[GBSpecialBase], typing.List[GBSpecialBase]]:
    """Build GB search and PE moves.

    Both moves share the same ``acs``, ``priors``, and
    ``TemperatureControl`` instance, so updates to ``acs`` (e.g. signal
    subtraction by another branch) are visible to both moves at runtime.

    Parameters
    ----------
    engine_info :
        Engine info object exposing ``ndims``.
    curr : CurrentInfoGlobalFit
        Current run info; reads ``source_info["gb"]`` and ``general_info``.
    acs :
        Shared analysis container (passed by reference).
    priors : dict
        Shared priors dict (passed by reference).
    num_repeats : int, optional
        Number of internal GB move repeats. Default 60.
    Tmax : float, optional
        Maximum temperature for ``TemperatureControl``. Default 1e6.

    Returns
    -------
    gb_search_moves : List[GBSpecialBase]
    gb_pe_moves : List[GBSpecialBase]
    """    
    gb_info: GBSetup = curr.source_info["gb"]
    general_info: GeneralSetup = curr.general_info
    nwalkers: int = general_info.nwalkers
    ntemps: int = general_info.ntemps
    data_start_freq_ind = int(acs.start_freq_ind[0])
    
    gb_betas = gb_info.betas
    gpus: list[int] = general_info.gpus
    
    #* Setting up gbgpu on correct backend and gpu(s) for correct orbits and timeshift
    from gbgpu.gbgpu import GBGPU
    
    cp.cuda.runtime.setDevice(gpus[0])
    gb = GBGPU(**gb_info.initialize_kwargs)
    # cp.cuda.runtime.setDevice(gpus[0])
    gb.gpus = gpus

    logger.debug(f"GBGPU initialized at t0 = {gb_info.initialize_kwargs['t0']}")
    logger.debug(f"GBGPU initialized with gpus: {gb.gpus} and backend: {gb.backend}")
    
    #* Make sure that priors are evaluated on gpus
    gpu_priors_in = deepcopy(priors["gb"].priors_in)
    for _, item in gpu_priors_in.items():
        item.use_cupy = True
    gpu_priors = {"gb": ProbDistContainer(gpu_priors_in, use_cupy=True)}
    
    nleaves_max_gb = state.branches["gb"].shape[-2]
    
    #* Get band information
    band_edges = gb_info.band_edges
    band_N_vals = gb_info.band_N_vals
    assert band_edges is not None
    assert band_N_vals is not None

    #* This checks if the initialization has any gbs in it (when injecting gbs) and adjusts acs accordingly
    if state.branches["gb"].inds[0].sum() > 0:

        coords_out_gb = state.branches["gb"].coords[0,
            state.branches["gb"].inds[0]
        ]
        coords_out_gb[:, 3] = coords_out_gb[:, 3] % (2 * np.pi)
        coords_out_gb[:, 5] = coords_out_gb[:, 5] % (1 * np.pi)
        coords_out_gb[:, 6] = coords_out_gb[:, 6] % (2 * np.pi)
        
        check = priors["gb"].logpdf(coords_out_gb)
        if np.any(np.isinf(check)):

            # check which prior is inf
            inf_indices = np.where(np.isinf(check))[0]
            inf_coords = coords_out_gb[inf_indices]
            logger.error(f"Found {len(inf_indices)} coordinates with inf logpdf under GB priors. Example inf coordinates: {inf_coords[:5]}") 

            logger.info("Prior bounds for GB parameters:")
            for param_name, prior in priors["gb"].priors_in.items():
                logger.info(f"  {param_name}: [{prior.min_val},{prior.max_val}]")
            breakpoint()
            raise ValueError("Starting priors are inf. If injecting, try reducing spread.")

        coords_in_in = gb_info.transform.both_transforms(coords_out_gb)

        band_inds = np.searchsorted(band_edges, coords_in_in[:, 1], side="right") - 1

        walker_vals = np.tile(
            np.arange(nwalkers), (nleaves_max_gb, 1)
        ).transpose((1, 0))[state.branches["gb"].inds[0]]

        data_index_1 = walker_vals  # ((band_inds % 2) + 0) * nwalkers + walker_vals

        data_index = cp.asarray(data_index_1).astype(
            cp.int32
        )
        # goes in as -h
        factors = -cp.ones_like(data_index, dtype=cp.float64)

        N_vals = band_N_vals[band_inds]

        logger.info("Removing GBs from residuals")
        template_in = deepcopy(acs.linear_data_arr)
        gb.generate_global_template(
            coords_in_in,
            data_index,
            acs.linear_data_arr,
            data_length=acs.data_length,
            factors=factors,
            data_splits=acs.gpu_map,
            N=N_vals,
            **gb_info.waveform_kwargs,
        )
        max_diff_templates = cp.abs(template_in[0]-acs.linear_data_arr[0]).max()
        del template_in
        logger.debug(f"The difference in residuals in/out = {max_diff_templates:5e}")

    acs[0].data_res_arr.data_res_arr.plot(channel=0, filename=curr.general_info.artifacts_file_dir + "data_post_subtraction.png")

    #* Check if we need to adjust the band temps, and adjust if required
    adjust_temps = False
    state_band_info = getattr(state, "band_info", None)
    if state_band_info is not None:
        band_info_check = deepcopy(state_band_info)
        adjust_temps = True
        #    del state.band_info

    band_temps = np.tile(np.asarray(gb_betas), (len(band_edges) - 1, 1))
    #print(f"")
    state.sub_states["gb"].initialize_band_information(nwalkers, ntemps, band_edges, band_temps)
    if adjust_temps:
        state.sub_states["gb"].band_info["band_temps"][:] = band_info_check["band_temps"][0, :]

    # TODO Check if the block below is needed... I.e., do we need band_inds in brach supplemental?
    # band_inds_in = np.zeros((ntemps, nwalkers, nleaves_max_gb), dtype=int)
    # N_vals_in = np.zeros((ntemps, nwalkers, nleaves_max_gb), dtype=int)

    # if state.branches["gb"].inds.sum() > 0:
    #     f_in = state.branches["gb"].coords[state.branches["gb"].inds][:, 1] / 1e3
    #     band_inds_in[state.branches["gb"].inds] = np.searchsorted(band_edges, f_in, side="right") - 1
    #     N_vals_in[state.branches["gb"].inds] = band_N_vals[band_inds_in[state.branches["gb"].inds]]

    # branch_supp_base_shape = (ntemps, nwalkers, nleaves_max_gb)
    # state.branches["gb"].branch_supplemental = BranchSupplemental(
    #     {"N_vals": N_vals_in, "band_inds": band_inds_in}, base_shape=branch_supp_base_shape, copy=True
    # )

    #* Assembling args and kwargs
    gb_move_args = (
        gb,
        priors,
        data_start_freq_ind,
        acs.end_shape[0],
        acs,
        acs.settings.f_arr,
        band_edges,
        band_N_vals,
        gpu_priors,
    )

    effective_ndim = engine_info.ndims["gb"]
    temperature_control = TemperatureControl(
        effective_ndim, nwalkers, ntemps=ntemps, Tmax=Tmax, permute=False
    )
    gb_move_kwargs = dict(
        waveform_kwargs=gb_info.waveform_kwargs,
        parameter_transforms=gb_info.transform,
        provide_betas=True,
        skip_supp_names_update=["group_move_points"],
        random_seed=general_info.random_seed,
        force_backend=general_info.gpu_backend,
        nfriends=nwalkers,
        temperature_control=temperature_control,
        use_gpu=True, 
        num_repeat_proposals=gb_info.num_repeat_proposals,
        search_kwargs=gb_info.search_kwargs
    )

    #* ============================================= SEARCH MOVES =============================================
    gb_search_prune_move = GBSpecialRJPriorMove(
        *gb_move_args, 
        rj_proposal_distribution=gpu_priors,
        name="rj_prior_search",
        use_prior_removal=True,  
        phase_maximize=False,  
        ranks_needed=0,
        run_swaps=True, 
        gpus=[],
        **gb_move_kwargs 
    )
    gb_search_prune_move.accepted = np.zeros((ntemps, nwalkers))
    
    gb_search_fstat_mcmc_move = GBSpecialRJSerialSearchMCMC(
        *gb_move_args, 
        rj_proposal_distribution=None,
        is_rj_prop=True,
        run_swaps=False, 
        name="rj_fstat_mcmc_search",
        phase_maximize=True, 
        ranks_needed=0,
        gpus=[],
        **gb_move_kwargs
    )
    gb_search_fstat_mcmc_move.accepted = np.zeros((ntemps, nwalkers))

    gb_search_refit_move = GBSpecialRJRefitMove(
        *gb_move_args, 
        rj_proposal_distribution=None,
        is_rj_prop=True,
        run_swaps=False, 
        name="rj_refit_search",
        fp=general_info.main_file_path,
        phase_maximize=True,  # gb_info["pe_info"]["rj_phase_maximize"],
        ranks_needed=0,
        gpus=[],
        **gb_move_kwargs
    )
    gb_search_refit_move.accepted = np.zeros((ntemps, nwalkers))
    
    gb_search_moves = [gb_search_fstat_mcmc_move, gb_search_prune_move] # gb_search_refit_move, Refit currently not used for search
    
    #* ============================================= PARAMETER ESTIMATION MOVES =============================================
    gb_pe_prior_move = GBSpecialRJPriorMove(
        *gb_move_args, 
        rj_proposal_distribution=gpu_priors,
        name="rj_prior",
        use_prior_removal=False,  # gb_info["pe_info"]["use_prior_removal"],
        phase_maximize=False,  # should probably be false if pruning  # gb_info["pe_info"]["rj_phase_maximize"],
        ranks_needed=0,
        run_swaps=True, 
        gpus=[],
        **gb_move_kwargs
    )
    gb_pe_prior_move.accepted = np.zeros((ntemps, nwalkers))

    gb_pe_fstat_mcmc_move = GBSpecialRJSerialSearchMCMC(
        *gb_move_args, 
        rj_proposal_distribution=None,
        run_swaps=True, 
        name="rj_fstat_mcmc",
        phase_maximize=False, 
        ranks_needed=0,
        gpus=[],
        **gb_move_kwargs
    )
    gb_pe_fstat_mcmc_move.accepted = np.zeros((ntemps, nwalkers))

    gb_pe_refit_move = GBSpecialRJRefitMove(
        *gb_move_args, 
        rj_proposal_distribution=None,
        run_swaps=True, 
        name="rj_refit",
        fp=general_info.main_file_path,
        phase_maximize=False,  # gb_info["pe_info"]["rj_phase_maximize"],
        ranks_needed=0,
        gpus=[],
        **gb_move_kwargs
    )
    gb_pe_refit_move.accepted = np.zeros((ntemps, nwalkers))
    
    gb_pe_moves = [gb_pe_prior_move, gb_pe_refit_move, gb_pe_fstat_mcmc_move]

    return gb_search_moves, gb_pe_moves