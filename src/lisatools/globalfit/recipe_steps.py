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
