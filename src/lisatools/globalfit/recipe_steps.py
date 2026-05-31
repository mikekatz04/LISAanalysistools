"""Built-in recipe steps and helpers for assembling Erebor-style global-fit runs."""

import time
import logging
import typing
from copy import deepcopy

import numpy as np
try:
    import cupy as cp
except ModuleNotFoundError:
    import numpy as cp

from lisatools.analysiscontainer import AnalysisContainerArray
# DataResidualArray is now a deprecation shim; we pass DomainBase children
# (or raw arrays via the AnalysisContainer/template APIs) directly.
from lisatools.domains import FDSettings, WDMSettings

# from bbhx.utils.transform import SSB_to_LISA
from eryn.moves.tempering import TemperatureControl, make_ladder
from eryn.prior import ProbDistContainer

from ..sources.utils import icrs_to_ecliptic
from .engine import Setup, GlobalFitEngine
from .moves import PSDMove, ResidualAddOneRemoveOneMove, GBSpecialRJPriorMove, GBSpecialRJSerialSearchMCMC, GBSpecialRJRefitMove
from .moves.gbspecialstretch import GBSpecialBase
from .recipe import RecipeStep, BaseRecipeStep
from .run import CurrentInfoGlobalFit
from .state import GFState
from .stock.erebor import GBSetup, GeneralSetup


logger = logging.getLogger(__name__)


class SearchRecipeStep(BaseRecipeStep):
    """Recipe step that completes immediately (one-shot search/initialization).

    Used when the stopping criterion is embedded inside the move itself
    rather than at the recipe level.
    """

    def stopping_function(self, *args, **kwargs):
        """Always stop after one call."""
        return True


class PERecipeStep(BaseRecipeStep):
    """Recipe step that runs indefinitely (ongoing parameter estimation)."""

    def stopping_function(self, *args, **kwargs):
        """Never stop on its own — relies on outer stopping logic."""
        return False


class RJRecipeStep(BaseRecipeStep):
    """Reversible-jump recipe step that stops once GB leaf count plateaus.

    Args:
        convergence_iter: Window length used to compare older vs newer
            cold-chain max leaf counts.
        thin_by: Forwarded thinning factor applied to the sampler.
    """

    def __init__(
        self,
        *args,
        convergence_iter: int = 5,
        thin_by: int = 1,
        **kwargs
    ):
        RecipeStep.__init__(self, *args, **kwargs)
        self.convergence_iter = convergence_iter
        self.thin_by = thin_by

    def stopping_function(
        self,
        i,
        sample,
        sampler: GlobalFitEngine
    ) -> bool:
        """Stop when the cold chain stops growing in number of GB leaves."""

        if not hasattr(self, "st"):
            self.st = time.perf_counter()

        current_iter = sampler.backend.iteration
        assert isinstance(current_iter, int)
        
        stop = False
        if current_iter > self.convergence_iter:

            nleaves_cc = sampler.backend.get_nleaves(branch_names=["gb"], temp_index=0)["gb"]

            # do not include most recent
            nleaves_cc_max_old = nleaves_cc[:-self.convergence_iter].max()
            nleaves_cc_max_new = nleaves_cc[-self.convergence_iter:].max()

            if nleaves_cc_max_old >= nleaves_cc_max_new:
                stop = True

            else:
                stop = False

            
            dur = (time.perf_counter() - self.st) / 3600.0  # hours
            logger.info(f"Previous nleaves: {nleaves_cc_max_old} --> new nleaves: {nleaves_cc_max_new}")
            logger.info(f"TIME SINCE START: {dur} hours")

        return stop
        
    def setup_run(
        self,
        iteration,
        last_sample,
        sampler: GlobalFitEngine
    ):
        """Configure the sampler for this RJ recipe step (moves, weights, thinning)."""
        # TODO: maybe make this the defaul setup
        sampler.moves = self.moves
        sampler.weights = self.weights
        sampler.yield_step = self.thin_by
        sampler.checkpoint_step = self.thin_by
        # sampler.override_thin_by = self.thin_by --> # TODO check this one
        
        for move in self.moves: 
            if sampler.periodic is not None and move.periodic is None:
                logger.debug(f"Setting periodicity of move {move} to {sampler.periodic}")
                move.periodic = sampler.periodic
            
            # TODO: do we also need to set these? I think the current settings setup has ntemps covered, not sure about temp_cntrl            
            # move.ntemps = sampler.ntemps 
            # move.temperature_control = sampler.temperature_control
            

def scatter_around_injection(
    state: GFState,
    branch_name: str,
    injection_params: np.ndarray,
    spread: float | np.ndarray,
    reverse_transform: typing.Callable | None = None,
    betas: np.ndarray | None = None,
):
    """
    Initialize branch coordinates by scattering walkers around injection parameters.

    For each leaf, draws coordinates from a multivariate Gaussian centered on
    the (transformed) injection parameters.  Higher-temperature chains receive
    proportionally wider scatter when ``betas`` is provided.  Initialized
    leaves are marked as active (``inds = True``).

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

    for leaf in range(nleaves_init):
        center = injection_sampling[leaf]
        leaf_cov = covs[leaf]
        for t in range(ntemps):
            if betas is not None:
                scaled_cov = leaf_cov / max(betas[t], 1e-10)
            else:
                scaled_cov = leaf_cov

            draws = np.random.multivariate_normal(center, scaled_cov, size=nwalkers)
            coords[t, :, leaf] = draws

        state.branches_inds[branch_name][:, :, leaf] = True


def mbh_catalogue_to_sampling_basis(catalogue_entry: dict) -> np.ndarray:
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

    s1z = float(catalogue_entry["PrimarySpinCompZ"])
    s2z = float(catalogue_entry["SecondarySpinCompZ"])
    dist = float(catalogue_entry["LuminosityDistance"]) / 1e3  # Mpc -> Gpc
    phi_ref = float(catalogue_entry["PhaseReferenceSourceFrame"]) % (2 * np.pi)
    cos_iota = np.cos(float(catalogue_entry["InclinationAngle"]))

    # Sky coordinates: ICRS -> ecliptic -> SSB -> LISA
    ra = float(catalogue_entry["RightAscension"])
    dec = float(catalogue_entry["Declination"])
    psi_icrs = float(catalogue_entry["PolarisationAngle"])
    psi_ssb, lam_ecl, beta_ecl = icrs_to_ecliptic(psi_icrs, ra, dec)
    t_ssb = float(catalogue_entry["TimeCoalescencePhenomTPHMSSBFrame"])

    logger.debug(f"Catalogue entry: RA={ra}, Dec={dec}, psi_icrs={psi_icrs}, t_ssb={t_ssb}")
    
    t_L, lam_L, beta_L, psi_L = SSB_to_LISA(t_ssb, lam_ecl, beta_ecl, psi_ssb)
    
    lam_L = lam_L % (2 * np.pi)
    psi_L = psi_L % np.pi
    logger.debug(f"Converted to LISA frame: t_L={t_L}, lambda_L={lam_L}, beta_L={beta_L}, psi_L={psi_L}")
    sin_beta_L = np.sin(beta_L)

    return np.array([logM, q, s1z, s2z, dist, phi_ref, cos_iota, psi_L, lam_L, sin_beta_L, t_L])


def subtract_initial_signal(
    acs: AnalysisContainerArray,
    state: GFState,
    wave_gen: typing.Callable,
    source_name: str,
    source_info: Setup,
):
    """Subtract pre-injected source templates from the residual buffers in ``acs``.

    Used at run start when a recipe seeds branches with known signal
    parameters (e.g. catalog injections); the corresponding template is
    removed from the residual so the sampler sees only the noise + other
    sources.

    Args:
        acs: Shared :class:`AnalysisContainerArray` whose residuals are
            modified in place.
        state: Current sampler state.
        wave_gen: Waveform generator for ``source_name``.
        source_name: Branch name (e.g. ``"mbh"``, ``"emri"``).
        source_info: Per-source :class:`Setup` providing transforms /
            waveform kwargs.
    """
    if np.any(inds := state.branches_inds[source_name][0]):
        logger.info(f"Subtracting initial signals for {source_name}")
        counter = 0
        for leaf in range(inds.shape[-1]):
            if inds[0, leaf]:
                assert np.all(inds[:, leaf])
                inj_coords = state.branches_coords[source_name][0, :, leaf]
                inj_coords_in = source_info.transform.both_transforms(inj_coords)
                signals_in = wave_gen(*inj_coords_in.T, **source_info.waveform_kwargs)
                for w in range(len(signals_in)):
                    ll_here = acs.acs[w].template_likelihood(template=signals_in[w], include_psd_info=False)
                    logger.debug(f"Initial log-likelihood contribution from walker {w}, leaf {leaf}: {ll_here}")
                acs.add_signal_to_residual(signals_in)
                counter += 1
        logger.debug(f"Subtracted {counter} initial signals for {source_name}")
    else:
        logger.info(f"No initial signals for {source_name}")


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

    effective_ndim = engine_info.ndims["psd"]
    temperature_control = TemperatureControl(
        effective_ndim, nwalkers, ntemps=ntemps, Tmax=Tmax, permute=False
    )

    psd_move_kwargs = dict(
        num_repeats=num_repeats,
        permute_every=permute_every,
        live_dangerously=True,
        psd_transform_fn=psd_info.transform_fn,
        sensitivity_backend=general_info.sensitivity_backend,
        temperature_control=temperature_control,
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

    subtract_initial_signal(acs, state, wave_gen, "mbh", mbh_info)

    betas_all = np.tile(make_ladder(mbh_info.ndim, ntemps=ntemps), (mbh_info.nleaves_max, 1))
    state.sub_states["mbh"].betas_all = betas_all

    coords_shape = (ntemps, nwalkers, mbh_info.nleaves_max, mbh_info.ndim)

    mbh_move_args = (
        "mbh",  # branch_name
        coords_shape,
        wave_gen,
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


# TODO would it make more sense to split this up into search and pe?
def build_gb_moves(
    engine_info: Setup,
    curr: CurrentInfoGlobalFit,
    acs: AnalysisContainerArray,
    priors: dict,
    state: GFState,
    *,
    num_repeats: int = 60,
    permute_every: int = 50,
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
    band_edges = gb_info.band_edges
    band_N_vals = gb_info.band_N_vals
    gb_betas = gb_info.betas
    gpus: list[int] = general_info.gpus

    domain_settings = general_info.domain_settings

    #* Setting up gbgpu on the correct backend and (if any) gpu(s).
    #* CPU path keeps numpy and avoids the cupy-only setDevice call.
    from gbgpu.gbgpu import GBGPU
    import gbgpu

    gb_force_backend = general_info.force_backend
    _gb_backend = gbgpu.get_backend(gb_force_backend)
    if gpus is not None:
        _gb_backend.set_cuda_device(gpus[0])
    # NOTE: ``GBGPU.__init__`` no longer accepts ``t0``; it gets the
    # reference time from the orbits object's t0. We keep ``gb_info.t0``
    # around because the WDM-domain ``GBWDMComputations`` consumes it
    # directly.
    gb = GBGPU(force_backend=gb_force_backend, orbits=general_info.gpu_orbits)
    if gpus is not None:
        cp.cuda.runtime.setDevice(gpus[0])
        gb.gpus = gpus
    else:
        gb.gpus = None

    #* Make sure that priors are evaluated on gpus (when available).
    # On CPU runs we keep ``use_cupy=False`` because eryn's prior.xp reads
    # ``cp`` unconditionally when ``use_cupy=True`` and raises NameError
    # if cupy isn't installed.
    use_gpu_priors = gpus is not None
    gpu_priors_in = deepcopy(priors["gb"].priors_in)
    for _, item in gpu_priors_in.items():
        item.use_cupy = use_gpu_priors
    gpu_priors = {"gb": ProbDistContainer(gpu_priors_in, use_cupy=use_gpu_priors)}
    
    nleaves_max_gb = state.branches["gb"].shape[-2]
    waveform_kwargs = gb_info.waveform_kwargs
    if "N" in waveform_kwargs:
        waveform_kwargs.pop("N")

    #* This checks if the initialization has any gbs in it and adjusts acs accordingly
    if state.branches["gb"].inds[0].sum() > 0:

        coords_out_gb = state.branches["gb"].coords[0,
            state.branches["gb"].inds[0]
        ]

        coords_out_gb[:, 3] = coords_out_gb[:, 3] % (2 * np.pi)
        coords_out_gb[:, 5] = coords_out_gb[:, 5] % (1 * np.pi)
        coords_out_gb[:, 6] = coords_out_gb[:, 6] % (2 * np.pi)

        check = priors["gb"].logpdf(coords_out_gb)
        if np.any(np.isinf(check)):
            raise ValueError("Starting priors are inf.")

        coords_in_in = gb_info.transform.both_transforms(coords_out_gb)

        band_inds = np.searchsorted(band_edges, coords_in_in[:, 1], side="right") - 1

        walker_vals = np.tile(
            np.arange(nwalkers), (nleaves_max_gb, 1)
        ).transpose((1, 0))[state.branches["gb"].inds[0]]

        data_index_1 = walker_vals  # ((band_inds % 2) + 0) * nwalkers + walker_vals

        data_index = cp.asarray(data_index_1).astype(cp.int32)
        # goes in as -h (subtract initial template from data residual)
        factors = -cp.ones_like(data_index, dtype=cp.float64)

        N_vals = band_N_vals[band_inds]

        logger.debug("Generating global GB template")
        if gpus is not None:
            gb.gpus = gpus

        if isinstance(domain_settings, FDSettings):
            #* TODO: add test to make sure the generator matches the general information.
            template_in = acs.linear_data_arr
            gb.generate_global_template(
                coords_in_in,
                data_index,
                acs.linear_data_arr,
                data_length=acs.data_length,
                factors=factors,
                data_splits=acs.gpu_map,
                N=N_vals,
                **waveform_kwargs,
            )
            max_diff_templates = cp.abs(template_in - acs.linear_data_arr).max()
            logger.debug(
                f"Global GB template generated with max template in/out diff = "
                f"{max_diff_templates:5e}"
            )
        elif isinstance(domain_settings, WDMSettings):
            if gb_info.gb_wdm_comp is None:
                raise ValueError(
                    "WDM-domain GB initialization requires "
                    "gb_info.gb_wdm_comp; build a GBWDMComputations in the "
                    "settings file and pass it via GBSettings.gb_wdm_comp."
                )
            num_bin = coords_in_in.shape[0]
            xp = gb_info.gb_wdm_comp.xp
            factors_arr = xp.asarray(factors).astype(xp.float64)
            gb_info.gb_wdm_comp.fill_global_wdm(
                coords_in_in,
                acs.linear_data_arr[0],
                acs,
                convert_to_ra_dec=False,
                data_index=xp.asarray(data_index),
                factors=factors_arr,
            )
        else:
            raise NotImplementedError(
                f"Domain settings {type(domain_settings).__name__} are not "
                f"supported for GB initialization."
            )


    #* Check if we need to adjust the band temps, and adjust if required
    adjust_temps = False
    state_band_info = getattr(state, "band_info", None)
    if state_band_info is not None:
        band_info_check = deepcopy(state_band_info)
        adjust_temps = True
        #    del state.band_info

    band_temps = np.tile(np.asarray(gb_betas), (len(band_edges) - 1, 1))
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
    #* ``fd`` is no longer a positional — the move derives it from
    #* ``acs.settings`` so the same call works for FDSettings and
    #* WDMSettings (and any future domain).
    gb_move_args = (
        gb,
        priors,
        gb_info.start_freq_ind,
        acs.data_length,
        acs,
        band_edges,
        band_N_vals,
        gpu_priors,
    )

    gb_move_kwargs = dict(
        waveform_kwargs=waveform_kwargs,
        parameter_transforms=gb_info.transform,
        provide_betas=True,
        skip_supp_names_update=["group_move_points"],
        random_seed=general_info.random_seed,
        force_backend=general_info.force_backend,
        nfriends=nwalkers,
        # gb_wdm_comp is None for the FD path (default) and a
        # GBWDMComputations instance for the WDM path. The move's Buffer
        # then dispatches on the AC's DomainSettings to pick the right
        # likelihood engine -- no string-level mode flag.
        gb_wdm_comp=gb_info.gb_wdm_comp,
        **gb_info.group_proposal_kwargs
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
    
    gb_search_moves = [gb_search_fstat_mcmc_move, gb_search_refit_move, gb_search_prune_move]
    
    # OLD stuff, but still here for **inspiration**
    # # gb_search_moves = GFCombineMove([psd_search_move, mbh_pe_move, gb_search_fstat_mcmc_move, gb_search_refit_move, gb_search_prune_move, mbh_pe_move, psd_search_move])
    # gb_search_moves = GFCombineMove([gb_search_fstat_mcmc_move, gb_search_refit_move, gb_search_prune_move, psd_search_move])
    # gb_search_moves.accepted = np.zeros((ntemps, nwalkers))
    # recipe.add_recipe_component(GBRunStep(moves=[gb_search_moves], convergence_iter=5, verbose=True), name="gb search")
    
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