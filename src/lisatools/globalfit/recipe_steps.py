import logging
from typing import Callable

import numpy as np

from lisatools.analysiscontainer import AnalysisContainerArray

try:
    import cupy as cp
except (ModuleNotFoundError, ImportError):
    import numpy as cp

from bbhx.utils.transform import SSB_to_LISA
from eryn.moves.tempering import TemperatureControl, make_ladder

from ..sources.bbh.waveform import PhenomTHMTDIWaveform
from ..sources.utils import icrs_to_ecliptic
from .engine import Setup
from .moves import PSDMove, ResidualAddOneRemoveOneMove
from .recipe import RecipeStep
from .run import CurrentInfoGlobalFit
from .state import GFState

logger = logging.getLogger(__name__)


class SearchRecipeStep(RecipeStep):
    """Recipe step that completes immediately (one-shot search/initialisation)."""

    def __init__(self, *args, moves=None, weights=None, **kwargs):
        super().__init__(moves=moves, weights=weights)

    def setup_run(self, iteration, last_sample, sampler):
        sampler.moves = self.moves
        sampler.weights = self.weights

    def stopping_function(self, *args, **kwargs):
        return True


class PERecipeStep(RecipeStep):
    """Recipe step that runs indefinitely (ongoing parameter estimation)."""

    def __init__(self, *args, moves=None, weights=None, **kwargs):
        super().__init__(moves=moves, weights=weights)

    def setup_run(self, iteration, last_sample, sampler):
        sampler.moves = self.moves
        sampler.weights = self.weights

    def stopping_function(self, *args, **kwargs):
        return False


def scatter_around_injection(
    state: GFState,
    branch_name: str,
    injection_params: np.ndarray,
    spread: float | np.ndarray,
    reverse_transform: Callable | None = None,
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
    coords = state.branches[branch_name].coords
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

        state.branches[branch_name].inds[:, :, leaf] = True  # todo check this. unsure this is the right way to access the indices


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
    lam_ecl, beta_ecl = icrs_to_ecliptic(ra, dec)

    psi_ssb = float(catalogue_entry["PolarisationAngle"])
    t_ssb = float(catalogue_entry["TimeCoalescencePhenomTPHMSSBFrame"])

    t_L, lam_L, beta_L, psi_L = SSB_to_LISA(t_ssb, lam_ecl, beta_ecl, psi_ssb)

    lam_L = lam_L % (2 * np.pi)
    psi_L = psi_L % np.pi
    sin_beta_L = np.sin(beta_L)

    return np.array([logM, q, s1z, s2z, dist, phi_ref, cos_iota, psi_L, lam_L, sin_beta_L, t_L])


def subtract_initial_signal(
    acs: AnalysisContainerArray,
    state: GFState,
    wave_gen: Callable,
    source_name: str,
    source_info: Setup,
):

    if np.any(inds := state.branches_inds[source_name][0]):
        logger.info(f"Subtracting initial signals for {source_name}")
        counter = 0
        for leaf in range(inds.shape[-1]):
            if inds[0, leaf]:
                assert np.all(inds[:, leaf])
                inj_coords = state.branches_coords[source_name][0, :, leaf]
                inj_coords_in = source_info.transform.both_transforms(inj_coords)
                signals_in = wave_gen(*inj_coords_in.T, **source_info.waveform_kwargs)
                acs.add_signal_to_residual(signals_in)
                counter += 1
        logger.info(f"Subtracted {counter} initial signals for {source_name}")
    else:
        logger.info(f"No initial signals for {source_name}")


def build_psd_moves(
    engine_info: Setup,
    curr: CurrentInfoGlobalFit,
    acs: AnalysisContainerArray,
    priors: dict,
    *,
    num_repeats: int = 60,
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
    nwalkers = general_info.nwalkers
    ntemps = general_info.ntemps
    psd_info = curr.source_info["psd"]

    effective_ndim = engine_info.ndims["psd"]
    temperature_control = TemperatureControl(
        effective_ndim, nwalkers, ntemps=ntemps, Tmax=Tmax, permute=False
    )

    psd_move_kwargs = dict(
        num_repeats=num_repeats,
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
    curr: CurrentInfoGlobalFit, acs: AnalysisContainerArray, priors: dict, state: GFState
) -> tuple[PhenomTHMTDIWaveform, ResidualAddOneRemoveOneMove]:
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
        mbh_info.waveform_kwargs.copy(),  # waveform_like_kwargs
        acs,
        mbh_info.num_prop_repeats,
        mbh_info.transform,
        priors,
        mbh_info.inner_moves,
    )

    mbh_pe_move = ResidualAddOneRemoveOneMove(*mbh_move_args, betas_all=betas_all)
    mbh_pe_move.accepted = np.zeros((ntemps, nwalkers))

    return wave_gen, mbh_pe_move
