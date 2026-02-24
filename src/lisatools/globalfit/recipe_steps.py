import numpy as np

try:
    import cupy as cp
except (ModuleNotFoundError, ImportError):
    import numpy as cp

from eryn.moves.tempering import TemperatureControl, make_ladder

from lisatools.globalfit.moves import PSDMove, ResidualAddOneRemoveOneMove
from lisatools.globalfit.recipe import RecipeStep


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


def scatter_around_injection(state, branch_name, injection_params, spread,
                             reverse_transform=None, betas=None):
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
        injection_sampling = np.array(
            [reverse_transform(p) for p in injection_params]
        )
    else:
        injection_sampling = injection_params

    nleaves_init = injection_sampling.shape[0]
    assert nleaves_init <= nleaves_max, (
        f"More injection leaves ({nleaves_init}) than nleaves_max ({nleaves_max})"
    )
    assert injection_sampling.shape[-1] == ndim, (
        f"Injection ndim ({injection_sampling.shape[-1]}) != branch ndim ({ndim})"
    )

    # Build covariance matrix/matrices
    spread = np.asarray(spread, dtype=float)
    if spread.ndim == 0:
        cov = spread.item() ** 2 * np.eye(ndim)
        covs = np.tile(cov, (nleaves_init, 1, 1))
    elif spread.ndim == 1:
        cov = np.diag(spread ** 2)
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

        state.branches[branch_name].inds[:, :, leaf] = True


def build_psd_moves(engine_info, curr, acs, priors, *, num_repeats=60, Tmax=1e6):
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

    psd_search_move = PSDMove(acs, priors, max_logl_mode=True, name="psd search move", **psd_move_kwargs)
    psd_pe_move = PSDMove(acs, priors, max_logl_mode=False, name="psd pe move", **psd_move_kwargs)

    psd_search_move.accepted = np.zeros((ntemps, nwalkers))
    psd_pe_move.accepted = np.zeros((ntemps, nwalkers))

    return psd_search_move, psd_pe_move


def build_mbh_moves_phenom(curr, acs, priors, state):
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

    if np.any(mbh_inds := state.branches_inds["mbh"][0]):
        for leaf in range(mbh_inds.shape[-1]):
            if mbh_inds[0, leaf]:
                assert np.all(mbh_inds[:, leaf])
                inj_coords = state.branches_coords["mbh"][0, :, leaf]
                inj_coords_in = mbh_info.transform.both_transforms(inj_coords)
                signals_in = wave_gen(*inj_coords_in.T, **mbh_info.waveform_kwargs)
                breakpoint()
                acs.add_signal_to_residual(signals_in)

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
