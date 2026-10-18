"""Move that swaps individual sources in/out of the residual data array."""

from __future__ import annotations

import collections
import inspect
import logging
import os
import time
import warnings
from contextlib import nullcontext
from copy import deepcopy
from typing import Any, Callable, Optional, TYPE_CHECKING

try:
    import cupy as xp
    _xp_is_cupy = True
except (ImportError, ModuleNotFoundError):
    # CPU-only fallback so the moves package imports cleanly when cupy is
    # unavailable (e.g. on a Mac development host).
    import numpy as xp
    _xp_is_cupy = False
import numpy as np


def _free_pool() -> None:
    """No-op stand-in for ``cupy.get_default_memory_pool().free_all_blocks()`` on CPU."""
    if _xp_is_cupy:
        xp.get_default_memory_pool().free_all_blocks()


# ---- search-stage cold -> hot rotation (user ruling 2026-10-14) -------------
#: default proposal cadence of the rotation: 125 proposals = 5 proposes of 25
#: in-model repeats (first "every 200 proposals for a source (which includes
#: the 25 repeats so 8 iterations)", then "every 5 iterations during the
#: search stages rather than 8", both 2026-10-14)
COLD_TO_HOT_EVERY_DEFAULT = 125
#: stage kinds the rotation runs in: every search kind, never PE
COLD_TO_HOT_STAGE_KINDS = ("search", "rj")

# ---- safety cap of the eigen step-scale adaptation (ruling 2026-10-15) -------
#: default {BRANCH}_EIGEN_ADAPT_MAX_FACTOR: the adapted sigma never exceeds 100x
#: the information-matrix sigma. Inert unless {BRANCH}_EIGEN_AXIS_ADAPT is on.
EIGEN_ADAPT_MAX_FACTOR_DEFAULT = 100.0

# ---- per-axis acceptance accounting + controller (ruling 2026-10-18) ---------
#: {BRANCH}_EIGEN_AXIS_ACC_MIN / _MAX / _FACTOR defaults: an axis whose cold
#: acceptance over a refresh window falls below MIN has its step divided by
#: FACTOR, one above MAX multiplied by it (docs/emri-sampling-audit.md, B2)
EIGEN_AXIS_ACC_MIN_DEFAULT = 0.1
EIGEN_AXIS_ACC_MAX_DEFAULT = 0.5
EIGEN_AXIS_ACC_FACTOR_DEFAULT = 2.0
#: cold proposals an axis needs before the controller judges it; fewer carry
#: over to the next window (one window at 10 visits x 25 repeats x 1 cold
#: rung x 1/12 of the axis draws is ~20 per axis)
EIGEN_AXIS_ACC_MIN_PROPOSALS = 8


def factor_capped_sigmas(sigmas_fisher, sigmas_adapted, max_factor):
    """``max(sigma_fisher, min(sigma_adapted, max_factor x sigma_fisher))`` per axis.

    Applied to :func:`eigen_refresh.adapt_axis_sigmas`' output (``max(fisher,
    min(width, box cap))``) this is exactly ``max(fisher, min(width, box cap,
    max_factor x fisher))`` for every ``max_factor``: the Fisher floor and the
    prior-box cap are unchanged, and the factor cap sits under the floor, so it
    can never push a sigma below the information-matrix one (``max_factor < 1``
    just leaves the floor). ``max_factor`` ``None`` / ``<= 0`` / non-finite:
    no cap, ``sigmas_adapted`` returned as is (same object).

    Why (12mo, first tar with the adaptation live): the narrowest Fisher axis
    (the best-measured phase / time combination) came out x600-4900 wider on
    MBH leaves 4/7/8/9/10/11 and EMRI 0/2/7, at the prior-box cap, because the
    rung chains straddle DISCRETE degenerate solutions (sky-mode hops, phase
    degeneracies) inside the 50-visit window; ``draw_axis_step`` picks one axis
    per repeat, so such an axis wastes ~1/ndim of the proposals.
    """
    if max_factor is None:
        return sigmas_adapted
    mf = float(max_factor)
    if not np.isfinite(mf) or mf <= 0.0:
        return sigmas_adapted
    sf = np.asarray(sigmas_fisher, dtype=float)
    return np.maximum(sf, np.minimum(np.asarray(sigmas_adapted, dtype=float), sf * mf))


def cold_to_hot_offset(event: int, nwalkers: int) -> int:
    """Walker offset of rotation ``event``: ``1 + event mod (nwalkers - 1)``.

    Never 0 (a walker never receives its own cold chain) and, over
    ``nwalkers - 1`` consecutive events, every other walker once. 0 when
    there is no partner (fewer than two walkers).
    """
    nw = int(nwalkers)
    if nw < 2:
        return 0
    return 1 + int(event) % (nw - 1)


def cold_to_hot_due(clock: int, repeats: int, cadence: int):
    """``(fires, new_clock)`` after a propose of ``repeats`` proposals.

    The clock counts proposals (in-model repeats) since the last event and
    is advanced BEFORE the test, so the first propose after a (re)start
    never fires and, at 25 repeats and cadence 125, events land on proposes
    5, 10, 15, ... A cadence <= 0 disables the rotation (clock held at 0).
    A cadence shorter than one propose fires every propose and keeps the
    remainder.
    """
    cad = int(cadence)
    if cad <= 0:
        return False, 0
    new = int(clock) + int(repeats)
    if new >= cad:
        return True, new % cad
    return False, new


def rotate_cold_into_hot(coords, inds, log_prior, *, hot: int, offset: int) -> int:
    """Copy every walker's COLD rung into the ``hot`` rung of walker ``w + offset``.

    In place on a per-leaf ladder sub-state's arrays: ``coords``
    ``(ntemps, nwalkers, nleaves, ndim)``, ``inds`` ``(ntemps, nwalkers,
    nleaves)`` and, when given, ``log_prior`` ``(nleaves, ntemps,
    nwalkers)`` (the copied point's prior is the source walker's cold
    prior: same prior, same point). Only leaves alive on the source
    walker's cold rung are copied; the destination's ``inds`` follow. The
    destination's ``log_like`` is NOT touched: every leaf visit rescores
    all rungs against the walker's own exposed residual (``prev_logl`` in
    ``propose_local``), which is where the copied point gets its
    likelihood. Returns the number of (walker, leaf) copies; 0 when ``hot``
    is the cold rung, the offset is 0, or there is nothing alive.
    """
    hot = int(hot)
    offset = int(offset)
    nt, nw = int(coords.shape[0]), int(coords.shape[1])
    if hot <= 0 or hot >= nt or nw < 2 or offset % nw == 0:
        return 0
    cold_coords = np.array(coords[0], copy=True)          # (nwalkers, nleaves, ndim)
    cold_inds = np.array(inds[0], copy=True)              # (nwalkers, nleaves)
    nl = int(coords.shape[2])
    if log_prior is not None and (
        np.ndim(log_prior) != 3 or tuple(np.shape(log_prior)) != (nl, nt, nw)
    ):
        log_prior = None                                  # not the per-leaf ladder layout
    cold_lp = None if log_prior is None else np.array(log_prior[:, 0, :], copy=True)  # (nleaves, nwalkers)
    n = 0
    for w in range(nw):
        dst = (w + offset) % nw
        alive = np.nonzero(cold_inds[w])[0]
        if alive.size == 0:
            continue
        coords[hot, dst, alive] = cold_coords[w, alive]
        inds[hot, dst, alive] = True
        if cold_lp is not None:
            log_prior[alive, hot, dst] = cold_lp[alive, w]
        n += int(alive.size)
    return n


from eryn.moves import EigenAxisMove, Move, StretchMove, TemperatureControl
from eryn.prior import ProbDistContainer
from eryn.utils.transform import TransformContainer

from tqdm import tqdm

from ...analysiscontainer import AnalysisContainerArray
from ...domaincomputation import DomainComputationGroupArray
from ...domains import DomainBase, DomainBaseArray, WDMSignal
from ...sampling.moves.walkerdiffjump import KIND_MODE, KIND_SCALED, WalkerDifferenceJumpMove
from ...sampling.snrladder import (
    LADDER_TMAX_MIN_DEFAULT,
    ceiling_tmax,
    check_clip,
    describe_leaf,
    leaf_alive,
    snr_changed,
    snr_keyed_ladder,
    snr_tmax,
    substate_leaf_snrs,
    top_snr_value,
)
from ...utils.device import sensitivity_to_current_device
from ...utils.utility import asnumpy, get_array_module
from .. import midit_checkpoint
from . import eigen_refresh, eigen_table_persist
from .globalfitmove import GlobalFitMove
from .walkerfanout import WalkerFanoutMixin, pooled_ladder_step

logger = logging.getLogger(__name__)


def describe_inner_moves(moves, weights) -> str:
    """One-line ``ClassName[(which, frame)] weight [(jump mix f x k)] | ...``
    summary of an inner-move stack (``SkyMove`` carries its ``which`` /
    ``coord_frame``; the eigen jump-mixture move its ``jump_mix`` /
    ``jump_mix_factor``, e.g. ``EigenAxisJumpMixMove 1.000 (jump mix 0.10 x
    3.0)`` -- the log proof that ``{BRANCH}_EIGEN_JUMP_MIX`` took effect)."""
    parts = []
    for move, weight in zip(moves, weights):
        label = type(move).__name__
        which = getattr(move, "which", None)
        frame = getattr(move, "coord_frame", None)
        if which is not None and frame is not None:
            label = f"{label}({which}, {frame})"
        part = f"{label} {float(weight):.3f}"
        mix = getattr(move, "jump_mix", None)
        if mix is not None and float(mix) > 0.0:
            part += (
                f" (jump mix {float(mix):.2f} x "
                f"{float(getattr(move, 'jump_mix_factor', 1.0)):.1f})"
            )
        if isinstance(move, WalkerDifferenceJumpMove):
            part += f" (cross-walker, mode frac {float(move.mode_frac):.2f})"
        parts.append(part)
    return " | ".join(parts)
DEBUG_MODE = False

if TYPE_CHECKING:
    from ...sources.waveformbase import TDWaveformBase
    from ...domaincomputation import DomainComputationGroupArray


class MoveSignalGen:
    """Engine-convention params-based generator adapter for one move.

    Follows the same call convention as the engine-installed per-branch
    generators (e.g. the stock ``SourceSignalGen``):
    ``fn(*sampling_params, apply_transform=True, **waveform_kwargs) ->
    DomainBase``, with the branch transform applied inside by default and
    ``apply_transform=False`` meaning the caller already applied it (the
    add/remove move always calls this way — its choreography transforms once
    up front, so the transform is applied exactly once either way).

    Registered into a container's dict-based ``signal_gen`` under the move's
    branch name when the container does not already carry a usable generator
    there. Reads ``transform_fn`` / ``waveform_gen`` off the move at call
    time so later reassignment (e.g. the multi-GPU subclass swapping
    ``waveform_gen`` after the base ``__init__``) stays honored; a specific
    ``waveform_gen`` override (heterodyne hook) can be pinned instead.
    """

    def __init__(self, move: "ResidualAddOneRemoveOneMove", waveform_gen: Callable = None):
        self.move = move
        self._waveform_gen = waveform_gen

    @property
    def waveform_gen(self) -> Callable:
        return self._waveform_gen if self._waveform_gen is not None else self.move.waveform_gen

    def __call__(self, *params, apply_transform: bool = True, leaf_inds=None, **kwargs):
        if apply_transform:
            params = self.move.transform_fn.both_transforms(
                np.asarray(params, dtype=float), leaf_inds=leaf_inds
            )
        return self.waveform_gen(*params, **kwargs)


class ResidualAddOneRemoveOneMove(WalkerFanoutMixin, GlobalFitMove, StretchMove, Move):
    """
    Move that handles adding and removing sources to and from the residuals stored in the analysis container array.
    This is done by first removing the contribution of the current sources in the cold chain from the residual,
    then proposing new sources for this leaf, and then adding back in the contribution of the new sources to the residual.
    This way we can make sure that the likelihoods are computed correctly for each proposed source and that the likelihoods are consistent with the current state of the residuals in the analysis container array.

    Args:
        branch_name: name of the branch that this move will operate on.
        coords_shape: shape of the coordinates of the sources in the branch that this move will operate on.
        waveform_gen: function that generates the waveforms for the sources given their (waveform-basis) coordinates. All operations on the analysis container array run through the containers' ``signal_gen`` machinery (:meth:`AnalysisContainer.build_template` / :meth:`AnalysisContainer.calculate_signal_likelihood`) and default to the generator the engine already installed on each container under ``branch_name``; ``waveform_gen`` is the fallback, wrapped as a :class:`MoveSignalGen` and added to containers that do not carry the branch (see :meth:`_resolve_signal_gen_override`). Because the move transforms coordinates once up front, the containers' generators are always called with ``apply_transform=False`` — the transform is applied exactly once on every path.
        waveform_gen_kwargs: keyword arguments for the waveform generator function.
        waveform_like_kwargs: keyword arguments for the likelihood computation function.
        acs: analysis container array that contains the residuals and other information needed for the likelihood computation.
        num_repeats: number of times to repeat the proposal step for each leaf.
        transform_fn: transform container that contains the transforms to be applied to the coordinates before generating waveforms and computing likelihoods.
        priors: prior distribution container that contains the prior distributions for the sources in the branch.
        inner_moves: list of moves and their corresponding weights to be used for proposing new sources for the leaf.
        Tmax: maximum temperature for the temperature control.
        betas_all: array of betas for all leaves and temperatures. Shape is (nleaves_max, ntemps). If None, betas will be initialized as in TemperatureControl.
        permute_every: cadence (in proposes / engine iterations) for the walker-permuting (fancy) temperature swap: it fires at most once per leaf visit, on the final in-model repeat, on proposes ``permute_every, 2*permute_every, ...`` (1 = every propose; the first propose after a restart never fires for cadence > 1); <= 0 disables it entirely (``{BRANCH}_PERMUTE_EVERY`` env override).
        pad_out_of_prior: whether to pad proposed sources that are out of the prior bounds to avoid JIT compilation issues. If True, proposed sources that are out of the prior bounds will be replaced with the first in-prior point.
        eigen_refresh_every: leaf-visit cadence of the eigen inner-move table
            refresh (:meth:`refresh_inner_move_tables`). ``None`` reads
            ``{BRANCH}_EIGEN_REFRESH``, then 10.
        eigen_eps_rel: finite-difference step of the likelihood ("ll")
            information-matrix route, as a fraction of each prior-box width.
            ``None`` reads ``{BRANCH}_EIGEN_EPS_REL``, then 1e-4.
        eigen_table_scope: ``"walker_max"`` (one table at the max-lnL cold
            walker) or ``"per_walker"`` (one per (temperature, walker) point);
            ll route only. ``None`` reads ``{BRANCH}_EIGEN_SCOPE``, then
            ``"walker_max"``. See :meth:`_eigen_scope`.
        eigen_info: information-matrix route of the eigen tables: ``"ll"``
            (likelihood second differences, the default) or ``"gram"`` (the
            Gram matrix ``<dh_a|dh_b>`` of the move's own batched templates;
            only on moves that implement ``_gram_templates``). ``None`` reads
            ``{BRANCH}_EIGEN_INFO``, then ``"ll"``. See :meth:`_eigen_info` and
            ``docs/eigen-info-routes.md``.
        eigen_gram_target: Gram route only: the template change each
            column's step is tuned to, as a fraction of ``||h||``; ``0`` keeps
            the fixed starting steps. ``None`` reads
            ``{BRANCH}_EIGEN_GRAM_TARGET``, then 1e-3. See :meth:`_gram_target`.
        eigen_gram_eps_rel: Gram route only: the starting step as a fraction
            of each prior-box width. ``None`` reads
            ``{BRANCH}_EIGEN_GRAM_EPS_REL``, then the ll route's
            :meth:`_eigen_eps_rel`.
        eigen_axis_adapt: widen each eigen table axis's step to the chain's
            own robust width along it (directions kept; floor = the
            information-matrix sigma, cap = the table's own cap). ``None``
            reads ``{BRANCH}_EIGEN_AXIS_ADAPT``, then ``False`` (off: every
            table bit-identical to the information-matrix one). See
            :meth:`_eigen_adapt_table`.
        eigen_adapt_visits: ring-buffer length K of the adaptation, in leaf
            visits (the end-of-visit rung ensemble of each). ``None`` reads
            ``{BRANCH}_EIGEN_ADAPT_VISITS``, then 50.
        eigen_adapt_min_visits: the buffer must hold this many visits before
            a refresh adapts (until then the table is the information-matrix
            one). ``None`` reads ``{BRANCH}_EIGEN_ADAPT_MIN_VISITS``, then 10.
        eigen_adapt_pool: under the walker-block fan-out, adapt every rank's
            step scale from the UNION of all walkers' rung ensembles (the
            head ships the full ensemble's rung coordinates in every propose
            payload) instead of the rank's own walkers. Acts only with
            ``eigen_axis_adapt`` on. ``None`` reads
            ``{BRANCH}_EIGEN_ADAPT_POOL``, then ``False``. See
            :meth:`_note_eigen_pool_sample`.
        eigen_temper_sigmas: rung ``k`` of the eigen inner move steps with
            ``sigma / sqrt(beta_k)`` (capped at the table's own cap) instead
            of the cold sigma at every rung; the stored / persisted table
            stays the cold one. ``None`` reads
            ``{BRANCH}_EIGEN_TEMPER_SIGMAS``, then ``False``. See
            :meth:`_eigen_visit_rung_scale`.
        eigen_adapt_max_factor: safety cap of the step-scale adaptation: an
            adapted sigma never exceeds this multiple of the
            information-matrix sigma (:func:`factor_capped_sigmas`). ``None``
            reads ``{BRANCH}_EIGEN_ADAPT_MAX_FACTOR``, then 100; ``<= 0`` or
            ``inf`` = no cap. Inert unless ``eigen_axis_adapt`` is on.
        eigen_axis_acc_log: tally the eigen inner move's proposals and
            acceptances per table axis (cold rung vs hot rungs) and log them
            once per table refresh (telemetry only). ``None`` reads
            ``{BRANCH}_EIGEN_AXIS_ACC_LOG``, then ``False``. See
            :meth:`_note_inner_outcome`.
        eigen_axis_acc_tune: the acceptance-driven per-axis step controller
            (implies the accounting): at each table refresh an axis whose
            cold acceptance over the window fell below ``eigen_axis_acc_min``
            has its step divided by ``eigen_axis_acc_factor`` (never below
            the information-matrix sigma), one above ``eigen_axis_acc_max``
            multiplied by it (never above the adapted width, or the box cap
            with the adaptation off). ``None`` reads
            ``{BRANCH}_EIGEN_AXIS_ACC_TUNE``, then ``False``. See
            :meth:`_eigen_axis_tune_table`.
        eigen_axis_acc_min / eigen_axis_acc_max / eigen_axis_acc_factor:
            the controller's band and step (``{BRANCH}_EIGEN_AXIS_ACC_MIN`` /
            ``_MAX`` / ``_FACTOR``; 0.1 / 0.5 / 2).
        ladder_reanchor_nats: the lost-rung guard's threshold in lnL nats
            (:meth:`_reanchor_lost_rungs`); ``0`` disables the guard.
            ``None`` reads ``{BRANCH}_LADDER_REANCHOR_NATS``, then 200.
        ladder_top_snr: SNR-keyed per-leaf ladders (user ruling 2026-10-18,
            default off): ``> 0`` makes each leaf's ladder
            ``make_ladder(ndim, ntemps, Tmax=clip((SNR_leaf / top)^2,
            ladder_tmax_min, ladder_tmax))``. ``None`` reads
            ``{BRANCH}_LADDER_TOP_SNR``, then 0 (off). See
            :meth:`_maybe_update_snr_ladders` and
            :mod:`lisatools.sampling.snrladder`.
        ladder_tmax / ladder_tmax_min: the clip's ceiling / floor (``None``
            reads ``{BRANCH}_LADDER_TMAX`` / ``{BRANCH}_LADDER_TMAX_MIN``, then
            the uncapped make_ladder top / 4). Inert with the knob off.
        ladder_top_snr_refresh: every N proposes, rebuild the ladder of a leaf
            whose SNR moved more than 20 % since its ladder was built (0 =
            never; ``None`` reads ``{BRANCH}_LADDER_TOP_SNR_REFRESH``).
        ladder_top_snr_refresh_pe: allow that refresh in PE stages too
            (``None`` reads ``{BRANCH}_LADDER_TOP_SNR_REFRESH_PE``, then off:
            search stages only).
        **kwargs: additional keyword arguments for the Move class.
    """

    def __init__(
        self,
        branch_name: str,
        coords_shape: tuple,
        waveform_gen: Callable,
        waveform_gen_kwargs: dict,
        waveform_like_kwargs: dict,
        acs: AnalysisContainerArray,
        num_repeats: int,
        transform_fn: TransformContainer,
        priors: ProbDistContainer,
        inner_moves: list,
        Tmax: float = np.inf,
        betas_all: np.ndarray = None,
        permute_every: int = 20,
        pad_out_of_prior: bool = False,
        dcga: DomainComputationGroupArray = None,
        waveform_gen_method: str = None,
        waveform_like_method: str = None,
        run_async: bool = False,
        run_threaded: bool = False,
        eigen_refresh_every: int = None,
        eigen_eps_rel: float = None,
        eigen_table_scope: str = None,
        eigen_info: str = None,
        eigen_gram_target: float = None,
        eigen_gram_eps_rel: float = None,
        eigen_axis_adapt: bool = None,
        eigen_adapt_visits: int = None,
        eigen_adapt_min_visits: int = None,
        eigen_adapt_pool: bool = None,
        eigen_temper_sigmas: bool = None,
        eigen_adapt_max_factor: float = None,
        eigen_axis_acc_log: bool = None,
        eigen_axis_acc_tune: bool = None,
        eigen_axis_acc_min: float = None,
        eigen_axis_acc_max: float = None,
        eigen_axis_acc_factor: float = None,
        ladder_reanchor_nats: float = None,
        cold_to_hot_every: int = None,
        ladder_top_snr: float = None,
        ladder_tmax: float = None,
        ladder_tmax_min: float = None,
        ladder_top_snr_refresh: int = None,
        ladder_top_snr_refresh_pe: bool = None,
        **kwargs,
    ):

        # Move.__init__(self, **kwargs)
        StretchMove.__init__(self, **kwargs)

        self.ntemps, self.nwalkers, self.nleaves_max, self.ndim = coords_shape
        # search-stage cold -> hot rotation (see _maybe_rotate_cold_into_hot);
        # kwarg > {BRANCH}_COLD_TO_HOT_EVERY env > 125 proposals
        self.cold_to_hot_every = cold_to_hot_every
        self._cold_to_hot_clock = 0
        self.n_cold_to_hot_events = 0
        self._cold_to_hot_offsets_seen = []

        # ONE move structure (2026-07 multi-GPU pass): pass ``dcga=`` to run
        # waveform generation + likelihood scoring on per-device replicas —
        # how many GPUs the move uses is set by the DCGA it is built with.
        # Without a DCGA the move runs the per-container ACA path (which is
        # itself shard-aware; see compute_acs_like). Same object either way.
        # With ``dcga``, ``waveform_gen`` is the generator OBJECT (its
        # ``.kwargs`` seed the replicas) and ``waveform_gen_method`` names
        # the method; without, ``waveform_gen`` is the bound callable.
        self._dcga = dcga
        self._run_async = run_async
        self._run_threaded = run_threaded
        self._waveform_gen_obj = None
        self.waveform_gen_method = waveform_gen_method
        self.waveform_like_method = waveform_like_method or waveform_gen_method
        if dcga is not None:
            if acs is None:
                acs = dcga.acs
            if waveform_gen_method is not None:
                self._waveform_gen_obj = waveform_gen
                waveform_gen = getattr(waveform_gen, waveform_gen_method)

        self.branch_name = branch_name
        self.acs = acs
        self.waveform_gen = waveform_gen
        self.num_repeats = num_repeats
        self.transform_fn = transform_fn
        self.priors = priors
        self.waveform_gen_kwargs = waveform_gen_kwargs
        self.waveform_like_kwargs = waveform_like_kwargs
        moves_tmp = [move[0] if isinstance(move, tuple) else move for move in inner_moves]
        move_weights = [move[1] if isinstance(move, tuple) else 1.0 for move in inner_moves]
        self.moves = moves_tmp
        self.move_weights = move_weights / np.sum(move_weights)
        # The resolved inner-move stack, at move construction: this runs on
        # every rank after the run's logger is up, so it is the ground truth
        # for what the branch actually proposes with (the build-time report
        # in stock.erebor.source_runtime used to be silenced by the branch
        # Setups' logger init -- 2026-10-13, 9mo job 758).
        logger.info(
            "[%s] inner-move stack: %s",
            self.branch_name.upper(),
            describe_inner_moves(self.moves, self.move_weights),
        )

        self.temperature_controls = [None for _ in range(self.nleaves_max)]
        for i in range(self.nleaves_max):
            if betas_all is not None:
                assert betas_all.shape == (self.nleaves_max, self.ntemps)
                betas_in = betas_all[i]
            else:
                betas_in = None

            self.temperature_controls[i] = TemperatureControl(
                self.ndim,
                self.nwalkers,
                betas=betas_in,
                permute=False,
                ntemps=self.ntemps,
                Tmax=Tmax,
                skip_swap_branches=None,  # will fill in after first run through move
            )
        
        self.permute_every = permute_every
        self.pad_out_of_prior = pad_out_of_prior

        # eigen inner-move table refresh (see refresh_inner_move_tables). Each
        # knob resolves kwarg > {BRANCH}_<NAME> env > default at USE time
        # (None defers to the env), so an attribute set after construction
        # wins too: _eigen_refresh_cadence / _eigen_eps_rel / _eigen_scope /
        # _eigen_info / _gram_target / _gram_eps_rel.
        self.eigen_refresh_every = eigen_refresh_every
        self.eigen_eps_rel = eigen_eps_rel
        self.eigen_table_scope = eigen_table_scope
        self.eigen_info = eigen_info
        self.eigen_gram_target = eigen_gram_target
        self.eigen_gram_eps_rel = eigen_gram_eps_rel
        # step-scale adaptation from the chain (_eigen_adapt_table): the same
        # kwarg > env > default at USE time
        self.eigen_axis_adapt = eigen_axis_adapt
        self.eigen_adapt_visits = eigen_adapt_visits
        self.eigen_adapt_min_visits = eigen_adapt_min_visits
        # cross-rank pooling of the adaptation buffer and the per-rung
        # tempered step (both kwarg > env > off at USE time)
        self.eigen_adapt_pool = eigen_adapt_pool
        self.eigen_temper_sigmas = eigen_temper_sigmas
        # the adaptation's safety cap (kwarg > env > 100 at USE time)
        self.eigen_adapt_max_factor = eigen_adapt_max_factor
        # per-axis acceptance log + controller (ruling 2026-10-18; kwarg > env
        # > off / 0.1 / 0.5 / 2 at USE time; _note_inner_outcome,
        # _eigen_axis_tune_table)
        self.eigen_axis_acc_log = eigen_axis_acc_log
        self.eigen_axis_acc_tune = eigen_axis_acc_tune
        self.eigen_axis_acc_min = eigen_axis_acc_min
        self.eigen_axis_acc_max = eigen_axis_acc_max
        self.eigen_axis_acc_factor = eigen_axis_acc_factor
        # lost-rung guard threshold (kwarg > env > 200 at USE time, like the
        # eigen knobs above; see _ladder_reanchor_nats)
        self.ladder_reanchor_nats = ladder_reanchor_nats
        # SNR-keyed per-leaf ladders (ruling 2026-10-18; kwarg > env > off at
        # USE time; _maybe_update_snr_ladders). ``snr_ladder_plan`` is what
        # the builder resolved at construction (recipe.resolve_snr_keyed_ladders);
        # None = nothing resolved (a move built outside the builder).
        self.ladder_top_snr = ladder_top_snr
        self.ladder_tmax = ladder_tmax
        self.ladder_tmax_min = ladder_tmax_min
        self.ladder_top_snr_refresh = ladder_top_snr_refresh
        self.ladder_top_snr_refresh_pe = ladder_top_snr_refresh_pe

        self._setup_debug()

        # make sure to propagate the periodic information to the inner moves if it is included in kwargs
        if 'periodic' in kwargs:
            self.periodic = kwargs['periodic']

        if self._dcga is not None and self._waveform_gen_obj is not None:
            self.create_waveform_gen_replicas()

    # ------------------------------------------------------------------
    # eigen inner-move table refresh
    # ------------------------------------------------------------------

    def _eigen_refresh_cadence(self):
        """kwarg > ``{BRANCH}_EIGEN_REFRESH`` env > 10 leaf-visits."""
        val = getattr(self, "eigen_refresh_every", None)
        if val is None:
            val = os.environ.get(
                f"{str(self.branch_name).upper()}_EIGEN_REFRESH", None
            )
        if val is None:
            val = 10
        return max(1, int(val))

    def _eigen_eps_rel(self):
        """kwarg > ``{BRANCH}_EIGEN_EPS_REL`` env > 1e-4 of the prior box."""
        val = getattr(self, "eigen_eps_rel", None)
        if val is None:
            val = os.environ.get(
                f"{str(self.branch_name).upper()}_EIGEN_EPS_REL", None
            )
        if val is None:
            val = 1e-4
        return float(val)

    def _eigen_info(self):
        """kwarg > ``{BRANCH}_EIGEN_INFO`` env > ``"ll"`` (lower-cased).

        ``"ll"``: the information matrix from likelihood second differences
        (the scope routes of :meth:`_eigen_scope`). ``"gram"``: the Gram
        (expected Fisher) matrix ``<dh_a|dh_b>`` of the move's OWN batched
        templates (:meth:`_build_eigen_table_gram`) -- no residual term, so
        positive semi-definite, one template batch per refresh. Any other
        value means ``"ll"``. Moves without a ``_gram_templates`` hook stay
        on ``"ll"``; a failed Gram build warns and falls back to it.

        Why a second route (SOBBH chunked toy, 2026-10-06): at its production
        step (1e-4 of the prior box) the ll route gave an f_low curvature
        ~250x too small (that step is far into f_low's nonlinear range; the
        resulting f_low step ~16x too wide), spin entries dominated by lnL
        noise, and mass entries ~1.5x off the Gram.
        """
        val = getattr(self, "eigen_info", None)
        if val is None:
            val = os.environ.get(
                f"{str(self.branch_name).upper()}_EIGEN_INFO", "ll")
        return str(val).strip().lower()

    def _eigen_scope(self):
        """kwarg > ``{BRANCH}_EIGEN_SCOPE`` env > ``"walker_max"``.

        ``"per_walker"``: one table per (temperature, walker) point of the
        leaf — stashed full-shape; the red/blue seam installs the
        split-sliced view (:meth:`_install_eigen_split_table`).
        ``"walker_max"``: ONE shared table, built at the max-lnL
        cold-chain walker.
        """
        val = getattr(self, "eigen_table_scope", None)
        if val is None:
            val = os.environ.get(
                f"{str(self.branch_name).upper()}_EIGEN_SCOPE", None
            )
        if val is None:
            return "walker_max"
        val = str(val).strip().lower()
        if val not in ("per_walker", "walker_max"):
            eigen_refresh.logger.warning(
                "[eigen_refresh] %s: unknown eigen_table_scope %r (use "
                "'per_walker' or 'walker_max'); using 'walker_max'",
                self.branch_name, val,
            )
            return "walker_max"
        return val

    # ------------------------------------------------------------------
    # step-scale adaptation from the chain ({BRANCH}_EIGEN_AXIS_ADAPT)
    # ------------------------------------------------------------------

    def _eigen_switch(self, attr, knob):
        """Attribute ``attr`` > ``{BRANCH}_<knob>`` env > ``False``.

        Strings take 1/0, true/false, yes/no, on/off (case-insensitive);
        anything else is off.
        """
        val = getattr(self, attr, None)
        if val is None:
            val = os.environ.get(
                f"{str(self.branch_name).upper()}_{knob}", None)
        if val is None:
            return False
        if isinstance(val, str):
            return val.strip().lower() in ("1", "true", "yes", "on")
        return bool(val)

    def _eigen_axis_adapt(self):
        """kwarg > ``{BRANCH}_EIGEN_AXIS_ADAPT`` env > ``False``.

        Strings take 1/0, true/false, yes/no, on/off (case-insensitive);
        anything else is off.
        """
        return self._eigen_switch("eigen_axis_adapt", "EIGEN_AXIS_ADAPT")

    def _eigen_adapt_pool(self):
        """kwarg > ``{BRANCH}_EIGEN_ADAPT_POOL`` env > ``False``.

        Pool the step-scale adaptation over EVERY walker's rung ensemble
        under the walker-block fan-out (:meth:`_note_eigen_pool_sample`).
        Acts only together with :meth:`_eigen_axis_adapt`.
        """
        return self._eigen_switch("eigen_adapt_pool", "EIGEN_ADAPT_POOL")

    def _eigen_temper_sigmas(self):
        """kwarg > ``{BRANCH}_EIGEN_TEMPER_SIGMAS`` env > ``False``.

        Step rung ``k`` with ``sigma / sqrt(beta_k)``
        (:meth:`_eigen_visit_rung_scale`).
        """
        return self._eigen_switch("eigen_temper_sigmas", "EIGEN_TEMPER_SIGMAS")

    def _eigen_adapt_visits(self):
        """kwarg > ``{BRANCH}_EIGEN_ADAPT_VISITS`` env > 50 (ring length K)."""
        val = getattr(self, "eigen_adapt_visits", None)
        if val is None:
            val = os.environ.get(
                f"{str(self.branch_name).upper()}_EIGEN_ADAPT_VISITS", None)
        return max(1, int(val)) if val is not None else 50

    def _eigen_adapt_max_factor(self):
        """kwarg > ``{BRANCH}_EIGEN_ADAPT_MAX_FACTOR`` env > 100; ``None`` = no cap.

        ``<= 0``, ``inf`` or ``nan`` mean no cap (``None``). An unreadable
        value warns once and keeps the default. Inert unless
        :meth:`_eigen_axis_adapt` is on (it only acts inside the adaptation).
        """
        val = getattr(self, "eigen_adapt_max_factor", None)
        if val is None:
            val = os.environ.get(
                f"{str(self.branch_name).upper()}_EIGEN_ADAPT_MAX_FACTOR", None)
            if isinstance(val, str) and not val.strip():
                val = None
        if val is None:
            val = EIGEN_ADAPT_MAX_FACTOR_DEFAULT
        try:
            f = float(val)
        except (TypeError, ValueError):
            if not getattr(self, "_eigen_max_factor_warned", False):
                self._eigen_max_factor_warned = True
                logger.warning(
                    "[%s_EIGEN] unreadable eigen_adapt_max_factor %r; using %g",
                    str(self.branch_name).upper(), val, EIGEN_ADAPT_MAX_FACTOR_DEFAULT)
            f = EIGEN_ADAPT_MAX_FACTOR_DEFAULT
        if not np.isfinite(f) or f <= 0.0:
            return None
        return f

    def _eigen_adapt_min_visits(self):
        """kwarg > ``{BRANCH}_EIGEN_ADAPT_MIN_VISITS`` env > 10."""
        val = getattr(self, "eigen_adapt_min_visits", None)
        if val is None:
            val = os.environ.get(
                f"{str(self.branch_name).upper()}_EIGEN_ADAPT_MIN_VISITS", None)
        return max(1, int(val)) if val is not None else 10

    def _eigen_fisher_dict(self):
        if getattr(self, "_eigen_fisher_sigmas", None) is None:
            self._eigen_fisher_sigmas = {}
        return self._eigen_fisher_sigmas

    def _eigen_carried_dict(self):
        if getattr(self, "_eigen_carried", None) is None:
            self._eigen_carried = {}
        return self._eigen_carried

    def _eigen_periodic_columns(self):
        """``(column indices, periods)`` of this branch's periodic parameters.

        Read off the move's periodic spec (an
        :class:`eryn.utils.PeriodicContainer`, installed from the sampler's
        at stage setup), falling back to an inner move's, then to a raw
        ``{branch: {int column: period}}`` dict. String-keyed raw dicts carry
        no column order here and resolve to nothing.
        """
        cands = [getattr(self, "_periodic", None)]
        cands += [getattr(m, "periodic", None) for m in getattr(self, "moves", [])]
        for per in cands:
            if per is None:
                continue
            inds = getattr(per, "inds_periodic", None)
            periods = getattr(per, "periods", None)
            if isinstance(inds, dict) and isinstance(periods, dict):
                if self.branch_name in inds:
                    return (np.asarray(inds[self.branch_name], dtype=int),
                            np.asarray(periods[self.branch_name], dtype=float))
                continue
            if isinstance(per, dict):
                spec = per.get(self.branch_name) or {}
                cols = [(int(k), float(v)) for k, v in spec.items()
                        if isinstance(k, (int, np.integer))]
                if cols:
                    return (np.array([c for c, _ in cols], dtype=int),
                            np.array([p for _, p in cols], dtype=float))
        return np.zeros(0, dtype=int), np.zeros(0, dtype=float)

    def _note_eigen_adapt_sample(self, leaf, work, betas):
        """Push the leaf's END-OF-VISIT rung ensemble into its ring buffer.

        ``work.coords[:ntemps, :, leaf]`` -- every rung of every walker THIS
        rank's body sampled (one walker per rank in production) -- with the
        visit's inverse temperatures. Keeps the last
        :meth:`_eigen_adapt_visits` visits per leaf. No-op unless the
        adaptation is on and an eigen inner move is in the stack. Never
        persisted: after a restart it refills (the sidecar carries the
        adapted scale itself).
        """
        if not self._eigen_axis_adapt():
            return
        if not any(isinstance(m, EigenAxisMove) for m in self.moves):
            return
        try:
            nt = int(self.ntemps)
            x = np.array(asnumpy(work.coords[:nt, :, int(leaf)]),
                         dtype=float, copy=True)
            b = np.array(asnumpy(betas), dtype=float, copy=True).ravel()[:x.shape[0]]
            if b.size < x.shape[0]:
                return
            self._eigen_ring("_eigen_adapt_buffer", leaf).append((x, b))
        except Exception as exc:  # noqa: BLE001 -- telemetry must not crash a propose
            logger.warning(
                "[%s_EIGEN] leaf %d: adaptation sample not recorded (%r)",
                str(self.branch_name).upper(), int(leaf), exc,
            )

    def _eigen_ring(self, attr, leaf):
        """``leaf``'s ring buffer in the ``{leaf: deque}`` dict ``attr``.

        Created on first use; re-sized (keeping the newest entries) when
        :meth:`_eigen_adapt_visits` changed since.
        """
        K = self._eigen_adapt_visits()
        buf = getattr(self, attr, None)
        if buf is None:
            buf = {}
            setattr(self, attr, buf)
        ring = buf.get(int(leaf))
        if ring is None or ring.maxlen != K:
            ring = buf[int(leaf)] = collections.deque(
                [] if ring is None else list(ring)[-K:], maxlen=K)
        return ring

    # ---- cross-rank pooling ({BRANCH}_EIGEN_ADAPT_POOL, ruling 2026-10-15) ----

    def _eigen_pool_block_head(self):
        """True on the HEAD of a walker-block fan-out.

        That is the only layout whose bodies see a PART of the ensemble:
        single process and the flat (``GF_GPU_ROUTING`` replica) body run
        the full ensemble on one process, so their own buffer already IS
        the union and nothing needs shipping.
        """
        if not self.fanout_active:
            return False
        layout = self.fanout.layout
        return not (getattr(layout, "replica_mode", False)
                    and self._flat_body_ok(layout))

    def _note_eigen_pool_sample(self, state):
        """Head, before the state is sliced: the full rung ensemble per leaf.

        User ruling 2026-10-15 (9mo, EMRI leaf 3): walker 0's hot rungs
        spanned the whole degenerate ridge while walkers 1 and 3 sat with
        EVERY rung within 1e-4; a rank that adapts from its own walker can
        only learn the 1e-4. With ``{BRANCH}_EIGEN_ADAPT_POOL`` (and
        ``{BRANCH}_EIGEN_AXIS_ADAPT``) on, the head takes ``coords[:ntemps,
        :, leaf]`` of EVERY walker for every alive leaf, with the leaf's
        ladder ``betas_all[leaf]``, and :meth:`fanout_payload_extra` ships
        it (``ntemps x nwalkers x ndim`` floats per leaf) to every compute
        rank, the head's own body included; :meth:`fanout_apply_extra`
        appends it to the rank's POOLED ring buffer
        (``_eigen_pool_buffer``, the same K visits), and
        :meth:`_eigen_adapt_table` adapts from that union on the rank's OWN
        table axes. The state at the top of a propose is the previous
        propose's end-of-visit ensemble, so this is the same sample the
        single-process body buffers at the end of each visit (its betas
        one pooled ladder step later). Called before the cold -> hot
        rotation edits a hot rung. A no-op off the block head, with either
        switch off, or without an eigen inner move.
        """
        self._eigen_pool_outgoing = None
        if not self._eigen_pool_block_head():
            return
        if not (self._eigen_axis_adapt() and self._eigen_adapt_pool()):
            return
        if not any(isinstance(m, EigenAxisMove) for m in getattr(self, "moves", ())):
            return
        try:
            sub = (getattr(state, "sub_states", None) or {}).get(self.branch_name)
            if sub is None or not getattr(sub, "tempered_initialized", False):
                return
            coords = np.asarray(asnumpy(sub.coords), dtype=float)
            inds = np.asarray(asnumpy(sub.inds), dtype=bool)
            betas_all = np.asarray(asnumpy(sub.betas_all), dtype=float)
            nt = min(int(self.ntemps), int(coords.shape[0]), int(betas_all.shape[-1]))
            out = {}
            for leaf in range(int(coords.shape[2])):
                if not inds[0, 0, leaf]:
                    continue
                out[int(leaf)] = (np.array(coords[:nt, :, leaf], copy=True),
                                  np.array(betas_all[leaf, :nt], copy=True))
            self._eigen_pool_outgoing = out or None
        except Exception as exc:  # noqa: BLE001 -- telemetry must not crash a propose
            logger.warning(
                "[%s_EIGEN] pooled adaptation sample not shipped (%r); the ranks "
                "adapt from their own walkers this propose",
                str(self.branch_name).upper(), exc,
            )

    def _push_eigen_pool(self, pool):
        """Rank: append the head's shipped union sample to the pooled ring."""
        if not self._eigen_axis_adapt():
            return
        try:
            for leaf, (x, b) in dict(pool).items():
                self._eigen_ring("_eigen_pool_buffer", leaf).append(
                    (np.array(x, dtype=float, copy=True),
                     np.array(b, dtype=float, copy=True).ravel()))
        except Exception as exc:  # noqa: BLE001 -- telemetry must not crash a propose
            logger.warning(
                "[%s_EIGEN] pooled adaptation sample not recorded (%r)",
                str(self.branch_name).upper(), exc,
            )

    def _eigen_ring_walkers(self, attr, leaf):
        """Walker count of the widest entry in ``leaf``'s ring ``attr`` (0: empty)."""
        ring = (getattr(self, attr, None) or {}).get(int(leaf))
        return max((int(x.shape[1]) for x, _ in ring), default=0) if ring else 0

    def _eigen_adapt_rung_samples(self, leaf, pool=False):
        """``([(X_k, beta_k), ...], n_visits)`` pooled per rung from the buffer.

        Rung ``k`` pools every buffered visit's rung-``k`` rows over the
        rank's walkers, each with the beta it was drawn at. ``pool=True``
        reads the cross-rank union buffer (:meth:`_note_eigen_pool_sample`)
        instead of the rank's own.
        """
        attr = "_eigen_pool_buffer" if pool else "_eigen_adapt_buffer"
        ring = (getattr(self, attr, None) or {}).get(int(leaf))
        if not ring:
            return [], 0
        nt = max(x.shape[0] for x, _ in ring)
        out = []
        for k in range(nt):
            xs = [x[k] for x, _ in ring if x.shape[0] > k]
            bs = [np.full(x.shape[1], b[k]) for x, b in ring if x.shape[0] > k]
            out.append((np.concatenate(xs, axis=0), np.concatenate(bs)))
        return out, len(ring)

    def _eigen_adapt_table(self, leaf, table):
        """Widen ``table``'s step scale per axis to the chain's own width.

        User ruling 2026-10-14 (9mo job 763): every EMRI samples a long
        degenerate intrinsic ridge whose direction the information-matrix
        table HAS (its widest intrinsic axis), but the chain's own cold
        spread along it was 1.4-56x the table sigma, so the eigen move crept
        along it in steps ``ratio^2`` iterations from crossing.

        * Samples: the leaf's ring buffer (:meth:`_note_eigen_adapt_sample`),
          the last K visits x every rung x the rank's walkers. With
          ``{BRANCH}_EIGEN_ADAPT_POOL`` on and the head's shipped union
          buffer holding at least the minimum (:meth:`_note_eigen_pool_sample`),
          the last K visits x every rung x EVERY walker instead, on this
          rank's own axes; the log then says ``pooled over W walkers``
          (single process / the flat replica body: the own buffer, which
          already holds every walker).
        * Cold units: each sample's deviation from its rung's own robust
          centre, periodic columns wrapped to ``[-P/2, P/2)`` first, times
          ``sqrt(beta)`` (:func:`eigen_refresh.cold_unit_deviations`).
        * Width: ``1.4826 x MAD`` of the coordinates in the table's axis
          basis (:func:`eigen_refresh.empirical_axis_widths`).
        * Adapted sigma: ``max(sigma_fisher, min(width, cap))``, the cap the
          table is built with (:func:`eigen_refresh.adapt_axis_sigmas`),
          then the safety cap ``max(sigma_fisher, min(that, max_factor x
          sigma_fisher))`` (:func:`factor_capped_sigmas`,
          ``{BRANCH}_EIGEN_ADAPT_MAX_FACTOR``, default 100): together
          ``max(fisher, min(width, box cap, max_factor x fisher))``. The
          axes are returned unchanged.

        Fewer than :meth:`_eigen_adapt_min_visits` buffered visits: the
        table is the information-matrix one, unless the leaf carries an
        adapted scale adopted from the sidecar after a restart -- that scale
        is then mapped onto the new axes
        (:func:`eigen_refresh.carried_axis_widths`) until the buffer is long
        enough. The information-matrix sigmas are kept per leaf
        (``_eigen_fisher_sigmas``) for the sidecar and the telemetry. Any
        failure warns and returns the information-matrix table.
        """
        leaf = int(leaf)
        axes, sig_f = table
        floor = np.array(asnumpy(sig_f), dtype=float, copy=True)
        self._eigen_fisher_dict()[leaf] = floor
        try:
            axes_np = np.asarray(asnumpy(axes), dtype=float)
            rungs, n_vis = self._eigen_adapt_rung_samples(leaf)
            need = self._eigen_adapt_min_visits()
            pooled = 0
            if self._eigen_adapt_pool():
                p_rungs, p_vis = self._eigen_adapt_rung_samples(leaf, pool=True)
                if p_vis >= need:
                    # the head's union of every walker's rung ensemble
                    rungs, n_vis = p_rungs, p_vis
                    pooled = self._eigen_ring_walkers("_eigen_pool_buffer", leaf)
                elif not getattr(self, "_fanout_body", False):
                    # single process / the flat replica body: the own
                    # buffer already holds every walker
                    pooled = self._eigen_ring_walkers("_eigen_adapt_buffer", leaf)
            emp, source = None, None
            if n_vis >= need:
                inds, periods = self._eigen_periodic_columns()
                dev = eigen_refresh.cold_unit_deviations(rungs, inds, periods)
                if dev is not None:
                    emp = eigen_refresh.empirical_axis_widths(axes_np, dev)
                    source = (f"{n_vis} visits, "
                              f"{sum(1 for _, b in rungs if np.any(b > 0))} rungs")
                    if pooled > 1:
                        source += f", pooled over {pooled} walkers"
                self._eigen_carried_dict().pop(leaf, None)
            elif leaf in self._eigen_carried_dict():
                a_old, s_old = self._eigen_carried_dict()[leaf]
                emp = eigen_refresh.carried_axis_widths(axes_np, a_old, s_old)
                source = (f"carried from the persisted table, buffer "
                          f"{n_vis} of {need} visits")
            if emp is None or np.shape(emp) != floor.shape:
                return table
            widths = eigen_refresh.prior_box_widths(
                self.priors[self.branch_name], self.ndim)
            sig_box = eigen_refresh.adapt_axis_sigmas(axes_np, floor, emp, widths)
            mf = self._eigen_adapt_max_factor()
            sig_a = factor_capped_sigmas(floor, sig_box, mf)
            if not np.all(np.isfinite(sig_a)):
                raise ValueError("non-finite adapted sigmas")
            n_capped = int(np.sum(np.any(
                (sig_a < sig_box).reshape(-1, floor.shape[-1]), axis=0)))
            self._log_eigen_adapt(leaf, axes_np, floor, sig_a, widths, source,
                                  factor_capped=(mf, n_capped))
            return (axes, sig_a)
        except Exception as exc:  # noqa: BLE001 -- never crash the sampler
            logger.warning(
                "[%s_EIGEN] leaf %d: step-scale adaptation failed (%r); "
                "using the information-matrix table",
                str(self.branch_name).upper(), leaf, exc,
            )
            return table

    def _log_eigen_adapt(self, leaf, axes, floor, adapted, widths, source,
                         factor_capped=None):
        """One INFO line when any axis moved by more than 20 %.

        ``[EMRI_EIGEN] leaf k: step scale adapted on N of 12 axes, max x12.3
        (axis 3, ridge), median x1.8 (50 visits, 8 rungs)``. The ratio is
        adapted / information-matrix sigma (per-walker tables: the largest
        over the tables, per axis); the median is over the N adapted axes;
        ``ridge`` marks the table's widest unrailed axis (largest sigma as a
        fraction of its cap -- the flattest direction the matrix measured).
        With the pooled buffer the parenthesis adds ``pooled over W
        walkers``; with ``{BRANCH}_EIGEN_TEMPER_SIGMAS`` on, this refresh's
        tempering note (:meth:`_eigen_temper_note`) rides in it too, so the
        refresh logs the tempering once. ``factor_capped = (max_factor, N)``
        with ``N > 0`` appends ``, factor-capped at x100 on N axes``; the
        ratios are taken AFTER the cap, so a capped line reads ``max
        x100.0``.
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(floor > 0, adapted / floor, 1.0)
        d = ratio.shape[-1]
        r = np.nanmax(ratio.reshape(-1, d), axis=0)
        moved = r > 1.2
        if not np.any(moved):
            return
        note = getattr(self, "_eigen_refresh_note", None)
        if note:
            source = f"{source}, {note}"
            self._eigen_refresh_note = None
        cap = eigen_refresh.axis_caps(axes, widths)
        with np.errstate(divide="ignore", invalid="ignore"):
            frac = np.nanmax((floor / cap).reshape(-1, d), axis=0)
        unrailed = frac < 1.0 - 1e-9
        ridge = (int(np.argmax(np.where(unrailed, frac, -np.inf)))
                 if np.any(unrailed) else None)
        j = int(np.argmax(r))
        capped_note = ""
        if factor_capped is not None and factor_capped[0] is not None and factor_capped[1] > 0:
            capped_note = (f", factor-capped at x{float(factor_capped[0]):g} on "
                           f"{int(factor_capped[1])} axes")
        # {BRANCH}_EIGEN_AXIS_ACC_LOG / _TUNE (2026-10-18): how the widened
        # axes fared over the window that just ended ("" with both off)
        capped_note += self._widened_axis_acceptance_note(moved)
        logger.info(
            "[%s_EIGEN] leaf %d: step scale adapted on %d of %d axes, max x%.1f "
            "(axis %d%s), median x%.1f (%s)%s",
            str(self.branch_name).upper(), int(leaf), int(moved.sum()), d,
            float(r[j]), j, ", ridge" if j == ridge else "",
            float(np.median(r[moved])), source, capped_note,
        )

    def refresh_inner_move_tables(self, leaf, work, betas=None):
        """Feed per-leaf ``(axes, sigmas)`` tables to eigen inner moves.

        Called once per leaf visit in ``propose`` — after
        ``setup_likelihood_here`` (the leaf is exposed, so ``compute_like``
        scores against the right residual) and after ``self._current_leaf``
        is set (per-leaf transform fills resolve). No-op unless one of
        ``self.moves`` is an :class:`eryn.moves.EigenAxisMove`.

        One table product per leaf (:meth:`_build_eigen_table`: a shared
        table at the max-lnL cold walker, or per-(temperature, walker) tables
        under the per_walker scope), recomputed every
        :meth:`_eigen_refresh_cadence` visits. Between refreshes the table
        is frozen, which is what keeps the inner move's ``factors == 0``
        honest (the same adaptive-kernel status the GB in-model tables
        carry: refresh only at block boundaries, never inside the repeat
        sweep).

        The first visit to a leaf looks in the run's eigen-table sidecar
        first (:mod:`~lisatools.globalfit.moves.eigen_table_persist`), so
        a restarted process does not repay minutes of information-matrix
        build per leaf; every build is written back through. That changes
        NOTHING about detailed balance — the table was already frozen
        between refreshes, and a frozen symmetric proposal is correct MH
        whatever point its curvature came from. (The sidecar does not record
        the ``{BRANCH}_EIGEN_INFO`` route, for the same reason: after a
        restart that switches routes, the adopted table serves until the
        leaf's next refresh.)

        With ``{BRANCH}_EIGEN_AXIS_ADAPT`` on, every refresh then widens the
        built table's step SCALE per axis to the chain's own width
        (:meth:`_eigen_adapt_table`); the axes are kept. Like the table
        itself the adapted scale changes only here, at a leaf-visit
        boundary, so the proposal stays frozen and symmetric inside each
        repeat sweep.

        With ``{BRANCH}_EIGEN_TEMPER_SIGMAS`` on, every visit also fixes the
        per-rung step multiplier ``1 / sqrt(beta_k)`` from ``betas`` (the
        visit's ladder; default: the leaf's temperature control) for the
        whole visit (:meth:`_eigen_visit_rung_scale`); the stored and
        persisted table stays the COLD one, and the tempered rows are
        installed per split (:meth:`_install_eigen_split_table`).

        2026-10-18 (the EMRI sampling audit): at a refresh the per-axis
        tallies of the window that just ended (:meth:`_note_inner_outcome`,
        ``{BRANCH}_EIGEN_AXIS_ACC_LOG`` / ``_TUNE``) feed the adaptation
        line, the acceptance controller (:meth:`_eigen_axis_tune_table`,
        after the adaptation, before the install) and the window's log
        lines; every visit then fixes the cross-walker jump's partner spec
        (:meth:`_de_leaf_spec`) when one is in the stack.
        """
        eigen_moves = self._eigen_inner_moves()
        if not eigen_moves:
            return
        if not hasattr(self, "_eigen_tables"):
            self._eigen_tables = {}
            self._eigen_visit_count = {}
        leaf = int(leaf)
        if leaf not in self._eigen_tables:
            self._adopt_persisted_eigen_table(leaf)
        visits = self._eigen_visit_count.get(leaf, 0)
        self._eigen_visit_count[leaf] = visits + 1
        temper = self._eigen_temper_sigmas()
        if temper and betas is None:
            tcs = getattr(self, "temperature_controls", None)
            if tcs is not None and leaf < len(tcs) and tcs[leaf] is not None:
                betas = tcs[leaf].betas
        if (leaf not in self._eigen_tables
                or visits % self._eigen_refresh_cadence() == 0):
            # the tempering note rides in the adaptation line when that
            # line fires (_log_eigen_adapt), else it gets its own line
            self._eigen_refresh_note = (
                self._eigen_temper_note(betas) if temper else None)
            # the per-axis acceptance of the window that ends here (None
            # with the accounting off): read by the adaptation line and the
            # controller, logged below (ruling 2026-10-18)
            report = self._eigen_axis_window_report(leaf)
            self._eigen_axis_report = report
            table = self._build_eigen_table(leaf, work)
            adapted = self._eigen_axis_adapt()
            if adapted:
                table = self._eigen_adapt_table(leaf, table)
            else:
                # off: no floor bookkeeping, the table is the built one
                getattr(self, "_eigen_fisher_sigmas", {}).pop(leaf, None)
            if self._eigen_axis_acc_tune():
                table = self._eigen_axis_tune_table(leaf, table, adapted, report)
            self._eigen_tables[leaf] = table
            self._persist_eigen_table(leaf)
            note = getattr(self, "_eigen_refresh_note", None)
            if note:
                logger.info("[%s_EIGEN] leaf %d: step %s",
                            str(self.branch_name).upper(), leaf, note)
            self._eigen_refresh_note = None
            self._log_eigen_axis_window(leaf, report)
            self._eigen_axis_report = None
        # the cross-walker jump's partner spec for this visit (no-op without
        # a WalkerDifferenceJumpMove in the stack)
        if self._de_moves():
            self._de_leaf_spec(leaf)
        axes, sigmas = self._eigen_tables[leaf]
        tempered = self._eigen_visit_rung_scale(leaf, betas) if temper else None
        if not temper:
            getattr(self, "_eigen_rung_scales", {}).pop(leaf, None)
        if getattr(axes, "ndim", 2) == 4:
            # per-(temp, walker) tables: the red/blue seam installs the
            # split-sliced view right before each get_proposal call
            return
        if tempered is not None:
            # per-rung rows: installed per split by the same seam; the
            # full-ensemble view goes in now for callers without the seam
            full = np.ones((len(tempered), int(np.shape(work.coords)[1])), dtype=bool)
            for move in eigen_moves:
                self._install_eigen_split_table(move, leaf, full)
            return
        for move in eigen_moves:
            move.set_axes(self.branch_name, axes, sigmas)

    # ---- per-rung tempered step ({BRANCH}_EIGEN_TEMPER_SIGMAS, 2026-10-15) ----

    def _eigen_rung_scale(self, betas):
        """``1 / sqrt(beta_k)`` per sampled rung (``beta <= 0``: 1), host numpy.

        :func:`eigen_refresh.temper_sigmas` applied to unit sigmas: the
        target at inverse temperature ``beta`` is ``exp(beta lnL)``, whose
        width is ``1 / sqrt(beta)`` times the cold one, while
        ``EigenAxisMove`` applies no temperature scaling of its own.
        """
        b = np.asarray(asnumpy(betas), dtype=float).ravel()[: int(self.ntemps)]
        return eigen_refresh.temper_sigmas(np.ones_like(b), b)

    def _eigen_temper_note(self, betas):
        """``"tempered 1/sqrt(beta) over 8 rungs, x1.00..x2.45"`` (or ``None``)."""
        if betas is None:
            return None
        s = self._eigen_rung_scale(betas)
        if not s.size:
            return None
        return (f"tempered 1/sqrt(beta) over {s.size} rungs, "
                f"x{float(np.min(s)):.2f}..x{float(np.max(s)):.2f}")

    def _eigen_temper_widths(self):
        """The prior-box widths the tempered steps are capped with (cached)."""
        w = getattr(self, "_eigen_temper_box", None)
        if w is None:
            w = self._eigen_temper_box = eigen_refresh.prior_box_widths(
                self.priors[self.branch_name], self.ndim)
        return w

    def _eigen_tempered_rows(self, axes, sigmas, scale):
        """``max(sigma, min(sigma x scale, cap))`` row by row.

        ``axes`` ``(..., ndim, ndim)``, ``sigmas`` ``(..., ndim)``, ``scale``
        broadcastable to ``sigmas[..., 0]``. The cap is the table's own
        (:func:`eigen_refresh.axis_caps`), so an axis railed at the prior
        box stays there; the cold sigma is the floor (a custom table above
        its cap keeps its own sigma, as in the adaptation).
        """
        sig = np.asarray(asnumpy(sigmas), dtype=float)
        cap = eigen_refresh.axis_caps(np.asarray(asnumpy(axes), dtype=float),
                                      self._eigen_temper_widths())
        wide = sig * np.asarray(scale, dtype=float)[..., None]
        return np.maximum(sig, np.minimum(wide, cap))

    def _eigen_visit_rung_scale(self, leaf, betas):
        """Fix this visit's per-rung step for ``leaf``; ``None`` when unusable.

        User ruling 2026-10-15 (9mo, EMRI leaf 3): walkers 1 and 3 had
        ladders whose EVERY rung stayed within 1e-4 because every rung
        proposed with the same cold-sized sigma (the shared source table is
        untempered; only the noise branches' ``temper_sigmas`` widened it).
        With ``{BRANCH}_EIGEN_TEMPER_SIGMAS`` on, rung ``k`` steps with
        ``sigma / sqrt(beta_k)``, capped at the table's cap and never below
        the cold sigma (:meth:`_eigen_tempered_rows`); the cold rung
        (``beta = 1``) is unchanged bit for bit. ``betas`` are the visit's
        ladder as it stood at the start of the visit and the scale is fixed
        for the whole visit, so it depends only on the rung, never on the
        point: each rung's proposal stays a frozen symmetric kernel. Stored
        as ``_eigen_rung_scales[leaf] = (scale, tempered shared sigmas or
        None)``; a per-walker stash is tempered row by row at install time
        (the lost-rung guard edits its rows during the visit).
        """
        store = getattr(self, "_eigen_rung_scales", None)
        if store is None:
            store = self._eigen_rung_scales = {}
        store.pop(int(leaf), None)
        if betas is None:
            if not getattr(self, "_eigen_temper_warned", False):
                self._eigen_temper_warned = True
                logger.warning(
                    "[%s_EIGEN] leaf %d: no ladder to temper the eigen step with; "
                    "every rung steps with the cold sigma",
                    str(self.branch_name).upper(), int(leaf))
            return None
        try:
            scale = self._eigen_rung_scale(betas)
            if scale.size != int(self.ntemps):
                raise ValueError(f"{scale.size} betas for {int(self.ntemps)} rungs")
            axes, sigmas = self._eigen_tables[int(leaf)]
            shared = None
            if getattr(axes, "ndim", 2) == 2:
                shared = self._eigen_tempered_rows(
                    np.broadcast_to(np.asarray(asnumpy(axes)), (scale.size,) + np.shape(axes)),
                    np.broadcast_to(np.asarray(asnumpy(sigmas)), (scale.size,) + np.shape(sigmas)),
                    scale)
            store[int(leaf)] = (scale, shared)
            return scale
        except Exception as exc:  # noqa: BLE001 -- never crash the sampler
            logger.warning(
                "[%s_EIGEN] leaf %d: tempered eigen step failed (%r); every rung "
                "steps with the cold sigma", str(self.branch_name).upper(), int(leaf), exc)
            return None

    # ------------------------------------------------------------------
    # per-axis acceptance accounting + the acceptance-driven controller
    # ({BRANCH}_EIGEN_AXIS_ACC_LOG / _TUNE; the EMRI sampling audit,
    # user ruling 2026-10-18, docs/emri-sampling-audit.md items B1 / B2)
    # ------------------------------------------------------------------

    def _eigen_inner_moves(self):
        """Every eigen move the tables must reach: the stack's own and the
        fallbacks of its cross-walker jumps (usually the same object)."""
        moves = [m for m in getattr(self, "moves", ()) if isinstance(m, EigenAxisMove)]
        for de in self._de_moves():
            fb = getattr(de, "fallback", None)
            if isinstance(fb, EigenAxisMove) and not any(fb is m for m in moves):
                moves.append(fb)
        return moves

    def _eigen_float_knob(self, attr, knob, default):
        """Attribute ``attr`` > ``{BRANCH}_<knob>`` env > ``default`` (a float).

        An empty env value counts as unset; an unreadable one warns once per
        knob and keeps the default.
        """
        val = getattr(self, attr, None)
        if val is None:
            val = os.environ.get(f"{str(self.branch_name).upper()}_{knob}", None)
            if isinstance(val, str) and not val.strip():
                val = None
        if val is None:
            return float(default)
        try:
            return float(val)
        except (TypeError, ValueError):
            warned = getattr(self, "_eigen_knob_warned", None)
            if warned is None:
                warned = self._eigen_knob_warned = set()
            if knob not in warned:
                warned.add(knob)
                logger.warning("[%s_EIGEN] unreadable %s %r; using %g",
                               str(self.branch_name).upper(), attr, val, float(default))
            return float(default)

    def _eigen_axis_acc_log(self):
        """kwarg > ``{BRANCH}_EIGEN_AXIS_ACC_LOG`` env > ``False``."""
        return self._eigen_switch("eigen_axis_acc_log", "EIGEN_AXIS_ACC_LOG")

    def _eigen_axis_acc_tune(self):
        """kwarg > ``{BRANCH}_EIGEN_AXIS_ACC_TUNE`` env > ``False``."""
        return self._eigen_switch("eigen_axis_acc_tune", "EIGEN_AXIS_ACC_TUNE")

    def _eigen_axis_acc_bounds(self):
        """``(acc_min, acc_max, factor)``; an inconsistent set warns once and
        falls back to the defaults (0.1, 0.5, 2)."""
        lo = self._eigen_float_knob("eigen_axis_acc_min", "EIGEN_AXIS_ACC_MIN",
                                    EIGEN_AXIS_ACC_MIN_DEFAULT)
        hi = self._eigen_float_knob("eigen_axis_acc_max", "EIGEN_AXIS_ACC_MAX",
                                    EIGEN_AXIS_ACC_MAX_DEFAULT)
        f = self._eigen_float_knob("eigen_axis_acc_factor", "EIGEN_AXIS_ACC_FACTOR",
                                   EIGEN_AXIS_ACC_FACTOR_DEFAULT)
        if not (0.0 <= lo < hi <= 1.0 and f > 1.0 and np.isfinite(f)):
            if not getattr(self, "_eigen_acc_bounds_warned", False):
                self._eigen_acc_bounds_warned = True
                logger.warning(
                    "[%s_EIGEN] inconsistent axis-acceptance controller knobs (min %r, "
                    "max %r, factor %r); using %g / %g / %g", str(self.branch_name).upper(),
                    lo, hi, f, EIGEN_AXIS_ACC_MIN_DEFAULT, EIGEN_AXIS_ACC_MAX_DEFAULT,
                    EIGEN_AXIS_ACC_FACTOR_DEFAULT)
            return (EIGEN_AXIS_ACC_MIN_DEFAULT, EIGEN_AXIS_ACC_MAX_DEFAULT,
                    EIGEN_AXIS_ACC_FACTOR_DEFAULT)
        return lo, hi, f

    def _eigen_axis_accounting(self):
        """Tally per-axis outcomes? The log or the controller is on and the
        stack carries an eigen move."""
        return ((self._eigen_axis_acc_log() or self._eigen_axis_acc_tune())
                and bool(self._eigen_inner_moves()))

    def _leaf_tally(self, attr, leaf, shape):
        """``leaf``'s zero-initialised float tally in the ``{leaf: array}`` dict ``attr``."""
        d = getattr(self, attr, None)
        if d is None:
            d = {}
            setattr(self, attr, d)
        arr = d.get(int(leaf))
        if arr is None or arr.shape != tuple(shape):
            arr = d[int(leaf)] = np.zeros(shape, dtype=float)
        return arr

    def _eigen_axis_tune_dict(self):
        if getattr(self, "_eigen_axis_tune", None) is None:
            self._eigen_axis_tune = {}
        return self._eigen_axis_tune

    def _walker_label(self):
        """``"walker 1"`` / ``"walkers 0-3"``: the global walkers this body runs."""
        w0 = self._walker_offset()
        nw = int(getattr(self, "nwalkers", 1) or 1)
        return f"walker {w0}" if nw == 1 else f"walkers {w0}-{w0 + nw - 1}"

    def _derive_axis_picks(self, leaf, old_points, new_points, split_mask):
        """The eigen axis each row stepped along, from its displacement.

        For an eigen move that does not report its picks (the plain eryn
        :class:`~eryn.moves.EigenAxisMove`, or the jump mixture at mix 0):
        ``c = A^-1 dy`` with the periodic columns of ``dy`` wrapped to
        ``[-P/2, P/2)`` has ONE nonzero entry for a one-axis draw. A periodic
        step longer than half a period lands on the wrong axis; telemetry
        only. Rows that did not move get -2.
        """
        tabs = getattr(self, "_eigen_tables", None) or {}
        if int(leaf) not in tabs:
            return None
        axes = np.asarray(asnumpy(tabs[int(leaf)][0]), dtype=float)
        d = axes.shape[-1]
        x0 = np.asarray(asnumpy(old_points), dtype=float).reshape(-1, d)
        x1 = np.asarray(asnumpy(new_points), dtype=float).reshape(-1, d)
        inds, periods = self._eigen_periodic_columns()
        dy = eigen_refresh.wrap_periodic(x1 - x0, inds, periods)
        if axes.ndim == 2:
            c = dy @ np.linalg.inv(axes).T
        else:
            a_rows = axes[np.asarray(split_mask, dtype=bool)].reshape(-1, d, d)
            if a_rows.shape[0] != dy.shape[0]:
                return None
            c = np.linalg.solve(a_rows, dy[..., None])[..., 0]
        pick = np.argmax(np.abs(c), axis=-1).astype(int)
        pick[~np.any(dy != 0.0, axis=-1)] = -2
        return pick

    def _note_inner_outcome(self, move_here, leaf, rung_of_row, keep,
                            old_points, new_points, split_mask):
        """Tally one split's proposals and acceptances (B1, and the B3 jumps).

        Called by ``propose_local`` after each split's Metropolis test.
        Rows are the split's (temperature, walker) rows in C order -- the
        order the eigen move draws them in. Per leaf: ``_eigen_axis_tally``
        ``(band, [proposed, accepted], axis)`` for the eigen draws at the
        table width (band 0 the cold rung, 1 the hot rungs), ``_eigen_jump_tally``
        ``(band, [proposed, accepted])`` for the jump mixture's wide draws
        (kept out of the per-axis counts: their step is ``jump_factor`` x the
        table's), ``_de_tally`` ``(kind, band, [proposed, accepted])`` for
        the cross-walker jumps (kind 0 the mode jump, 1 the scaled one).
        Out-of-prior draws count as proposals, as in the in-model counters.
        No random numbers, no state: the proposal stream is untouched. A
        failure warns once and stops nothing.
        """
        is_de = isinstance(move_here, WalkerDifferenceJumpMove)
        accounting = self._eigen_axis_accounting()
        if not (is_de or accounting):
            return
        try:
            name = self.branch_name
            keep = np.asarray(asnumpy(keep), dtype=bool).ravel()
            band = (np.asarray(rung_of_row).ravel() > 0).astype(int)
            n = keep.size
            kind = np.zeros(n, dtype=int)
            eig = move_here
            if is_de:
                kind = np.asarray(move_here.last_kind.get(name, kind), dtype=int).ravel()
                if kind.size != n:
                    kind = np.zeros(n, dtype=int)
                de = self._leaf_tally("_de_tally", leaf, (2, 2, 2))
                for j, kd in enumerate((KIND_MODE, KIND_SCALED)):
                    m = kind == kd
                    np.add.at(de[j, :, 0], band[m], 1.0)
                    np.add.at(de[j, :, 1], band[m & keep], 1.0)
                eig = move_here.fallback
            if not accounting or not isinstance(eig, EigenAxisMove):
                return
            src = move_here if is_de else eig
            pick = (getattr(src, "last_pick", None) or {}).get(name)
            pick = None if pick is None else np.asarray(pick, dtype=int).ravel()
            if pick is None or pick.size != n:
                pick = np.full(n, -2, dtype=int)
            todo = (kind == 0) & (pick == -2)
            if np.any(todo):
                derived = self._derive_axis_picks(leaf, old_points, new_points, split_mask)
                if derived is not None and derived.size == n:
                    pick = np.where(todo, derived, pick)
            base = (kind == 0) & (pick >= 0)
            scale = (getattr(eig, "last_jump_scale", None) or {}).get(name)
            if scale is not None and np.size(scale) == n:
                wide = base & (np.asarray(scale, dtype=float).ravel() > 1.0)
                base &= ~wide
                jt = self._leaf_tally("_eigen_jump_tally", leaf, (2, 2))
                np.add.at(jt[:, 0], band[wide], 1.0)
                np.add.at(jt[:, 1], band[wide & keep], 1.0)
            t = self._leaf_tally("_eigen_axis_tally", leaf, (2, 2, int(self.ndim)))
            ok = base & (pick < int(self.ndim))
            np.add.at(t[:, 0, :], (band[ok], pick[ok]), 1.0)
            np.add.at(t[:, 1, :], (band[ok & keep], pick[ok & keep]), 1.0)
        except Exception as exc:  # noqa: BLE001 -- telemetry must not crash a propose
            if not getattr(self, "_eigen_tally_warned", False):
                self._eigen_tally_warned = True
                logger.warning("[%s_EIGEN] leaf %d: per-axis acceptance not recorded (%r)",
                               str(self.branch_name).upper(), int(leaf), exc)

    def _eigen_axis_window_report(self, leaf):
        """The tallies since ``leaf``'s last refresh, then reset (``None`` when empty).

        ``{"axis": (2, 2, ndim), "jump": (2, 2), "de": (2, 2, 2), "de_visits":
        (2,)}`` (missing entries ``None``), consumed at the refresh: the
        adaptation line, the controller and the window log read it.
        """
        out = {}
        for key, attr in (("axis", "_eigen_axis_tally"), ("jump", "_eigen_jump_tally"),
                          ("de", "_de_tally"), ("de_visits", "_de_visit_tally")):
            d = getattr(self, attr, None) or {}
            arr = d.pop(int(leaf), None)
            out[key] = None if arr is None else np.array(arr, copy=True)
        if all(v is None or not np.any(v) for v in out.values()):
            return None
        return out

    @staticmethod
    def _rates(prop, acc):
        prop = np.asarray(prop, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(prop > 0, np.asarray(acc, dtype=float) / np.where(prop > 0, prop, 1.0),
                            np.nan)

    @staticmethod
    def _fmt_rates(r):
        return " ".join("-" if not np.isfinite(v) else f"{v:.2f}" for v in np.ravel(r))

    def _widened_axis_acceptance_note(self, moved):
        """``", cold acceptance of the widened axes 8:0.00 9:0.04"`` from the
        current window report (``""`` without one)."""
        rep = getattr(self, "_eigen_axis_report", None)
        if not rep or rep.get("axis") is None:
            return ""
        t = rep["axis"]
        moved = np.asarray(moved, dtype=bool).ravel()
        if t.shape[-1] != moved.size or not np.any(moved):
            return ""
        acc = self._rates(t[0, 0], t[0, 1])
        parts = [f"{k}:{'-' if not np.isfinite(acc[k]) else f'{acc[k]:.2f}'}"
                 for k in np.flatnonzero(moved)]
        return ", cold acceptance of the widened axes " + " ".join(parts)

    def _log_eigen_axis_window(self, leaf, report):
        """The refresh window's acceptance lines (B1 and the B3 jumps).

        ``[EMRI_EIGEN] leaf k (walker w): axis acceptance over the last 10
        visits, cold 0.42 ... (n 14-22), hot 0.47 ...; jump-mix draws cold
        a/b hot c/d`` with ``{BRANCH}_EIGEN_AXIS_ACC_LOG`` or ``_TUNE`` on,
        and ``[EMRI_DE] leaf k (walker w): cross-walker jumps over the last
        10 visits (active on 5): mode cold a/b hot c/d, scaled cold e/f hot
        g/h`` with a jump in the stack.
        """
        if not report:
            return
        br = str(self.branch_name).upper()
        who = self._walker_label()
        nvis = self._eigen_refresh_cadence()
        t = report.get("axis")
        if t is not None and np.any(t[:, 0]) and (
                self._eigen_axis_acc_log() or self._eigen_axis_acc_tune()):
            n_cold = t[0, 0]
            jt = report.get("jump")
            jump = ""
            if jt is not None and np.any(jt[:, 0]):
                jump = (f"; jump-mix draws cold {int(jt[0, 1])}/{int(jt[0, 0])} "
                        f"hot {int(jt[1, 1])}/{int(jt[1, 0])}")
            logger.info(
                "[%s_EIGEN] leaf %d (%s): axis acceptance over the last %d visits, cold %s "
                "(n %d-%d per axis), hot %s%s", br, int(leaf), who, nvis,
                self._fmt_rates(self._rates(t[0, 0], t[0, 1])),
                int(np.min(n_cold)), int(np.max(n_cold)),
                self._fmt_rates(self._rates(t[1, 0], t[1, 1])), jump)
        de = report.get("de")
        if self._de_moves() and (de is not None or report.get("de_visits") is not None):
            dv = report.get("de_visits")
            de = np.zeros((2, 2, 2)) if de is None else de
            act = "" if dv is None else f" (active on {int(dv[0])} of {int(dv[1])})"
            logger.info(
                "[%s_DE] leaf %d (%s): cross-walker jumps over the last %d visits%s: mode "
                "cold %d/%d hot %d/%d, scaled cold %d/%d hot %d/%d", br, int(leaf), who, nvis,
                act, int(de[0, 0, 1]), int(de[0, 0, 0]), int(de[0, 1, 1]), int(de[0, 1, 0]),
                int(de[1, 0, 1]), int(de[1, 0, 0]), int(de[1, 1, 1]), int(de[1, 1, 0]))

    def _eigen_axis_tune_table(self, leaf, table, adapted, report):
        """The acceptance-driven per-axis step controller (B2), at a refresh.

        Why (9mo job 773, rows 840-1007): the adaptation widened the EMRI
        tables' narrow intrinsic axes (logm1 / p0 / m2 / a / e0 combinations,
        axes 8-11) by x14-100 -- the pooled union measures the OFFSETS
        between walkers sitting on different islands of the leaf's comb, not
        a traversable width -- and the cold walkers' own spread along those
        axes is 0.04-0.2 of the widened step, so each such draw is a 5-25
        sigma jump that is never accepted ("widened but rejected": 4 of 12
        axes dead on EMRI 3, 3 on EMRI 0).

        Per axis ``k`` a multiplier ``m_k`` on the table's sigma (the adapted
        one with ``{BRANCH}_EIGEN_AXIS_ADAPT``, else the information-matrix
        one) is kept per leaf. The window's cold-rung proposals and
        acceptances along ``k`` accumulate until ``k`` has at least
        :data:`EIGEN_AXIS_ACC_MIN_PROPOSALS`; then cold acceptance <
        ``acc_min``: ``m_k /= factor``; > ``acc_max``: ``m_k *= factor``; the
        accumulator restarts. The installed sigma is ``max(floor, min(m_k x
        sigma, upper))`` with ``floor`` = the information-matrix sigma and
        ``upper`` = the adapted sigma (adaptation on) or the table's
        prior-box cap (off), and ``m_k`` is clipped so the step stays in
        that band. The multiplier changes only here, at a leaf-visit
        boundary, so every repeat sweep runs a frozen symmetric kernel (the
        status of the table refresh itself). The tempered rungs scale the
        tuned cold sigma. ``m`` rides in the sidecar (``axis_tune``).
        """
        leaf = int(leaf)
        axes, sig = table
        try:
            sig = np.asarray(asnumpy(sig), dtype=float)
            d = sig.shape[-1]
            acc_min, acc_max, f = self._eigen_axis_acc_bounds()
            ctrl = self._leaf_tally("_eigen_axis_ctrl", leaf, (2, d))
            t = None if not report else report.get("axis")
            if t is not None and t.shape[-1] == d:
                ctrl[0] += t[0, 0]
                ctrl[1] += t[0, 1]
            m = self._eigen_axis_tune_dict().get(leaf)
            m = np.ones(d) if m is None or np.size(m) != d else np.array(m, dtype=float)
            floor = sig
            if adapted:
                fl = self._eigen_fisher_dict().get(leaf)
                if fl is not None and np.shape(fl) == sig.shape:
                    floor = np.asarray(fl, dtype=float)
            if adapted:
                upper = sig
            else:
                upper = eigen_refresh.axis_caps(
                    np.asarray(asnumpy(axes), dtype=float),
                    eigen_refresh.prior_box_widths(self.priors[self.branch_name], self.ndim))
                upper = np.maximum(upper, sig)
            ready = ctrl[0] >= EIGEN_AXIS_ACC_MIN_PROPOSALS
            acc = self._rates(ctrl[0], ctrl[1])
            m_old = m.copy()
            m[ready & (acc < acc_min)] /= f
            m[ready & (acc > acc_max)] *= f
            with np.errstate(divide="ignore", invalid="ignore"):
                lo_m = np.nanmin(np.where(sig > 0, floor / sig, 1.0).reshape(-1, d), axis=0)
                hi_m = np.nanmax(np.where(sig > 0, upper / sig, 1.0).reshape(-1, d), axis=0)
            m = np.clip(m, lo_m, np.maximum(hi_m, lo_m))
            # what actually changed (a step already at its floor / ceiling
            # does not count)
            down = m < m_old * (1.0 - 1e-12)
            up = m > m_old * (1.0 + 1e-12)
            ctrl[:, ready] = 0.0
            self._eigen_axis_tune_dict()[leaf] = m
            new = np.maximum(floor, np.minimum(sig * m, upper))
            if not np.all(np.isfinite(new)):
                raise ValueError("non-finite tuned sigmas")
            if np.any(down) or np.any(up):
                def _list(mask):
                    ks = np.flatnonzero(mask)
                    if not ks.size:
                        return "none"
                    return (",".join(str(k) for k in ks)
                            + f" (cold acc {np.nanmin(acc[mask]):.2f}-{np.nanmax(acc[mask]):.2f})")
                logger.info(
                    "[%s_EIGEN] leaf %d (%s): acceptance-tuned step: shrunk axes %s, widened "
                    "%s (band %.2f-%.2f, factor %g); step now x%.2f..x%.2f of the %s sigma",
                    str(self.branch_name).upper(), leaf, self._walker_label(), _list(down),
                    _list(up), acc_min, acc_max, f, float(np.min(m)), float(np.max(m)),
                    "adapted" if adapted else "information-matrix")
            return (axes, new)
        except Exception as exc:  # noqa: BLE001 -- never crash the sampler
            logger.warning(
                "[%s_EIGEN] leaf %d: acceptance-tuned step failed (%r); using the "
                "untuned table", str(self.branch_name).upper(), leaf, exc)
            return table

    # ------------------------------------------------------------------
    # the cross-walker difference jump (B3; lisatools.sampling.moves.
    # walkerdiffjump.WalkerDifferenceJumpMove in the inner-move stack)
    # ------------------------------------------------------------------

    def _de_moves(self):
        """The stack's :class:`WalkerDifferenceJumpMove` instances."""
        return [m for m in getattr(self, "moves", ()) if isinstance(m, WalkerDifferenceJumpMove)]

    def _de_next_parity(self):
        """This propose's walker-group parity (deterministic, state-free).

        The ``k``-th distinct global iteration (``gf_iteration``, the stage
        combine's stamp) in which this move proposes has parity ``k mod 2``;
        without a stamp (a move run outside a recipe) every propose counts as
        a new one. When the move proposes every iteration (full_pe,
        source_search) this is ``gf_iteration mod 2`` up to the job's
        starting offset; under an EVEN source cadence (a gb_search stage
        with ``GB_SEARCH_{N}_SOURCE_EVERY`` = 2, 4, ...) ``gf_iteration mod
        2`` would pin one group, this still alternates. Like
        ``gf_iteration`` it restarts with each process; it reads nothing
        from the chain, so the schedule adds no state dependence.
        """
        it = getattr(self, "gf_iteration", None)
        ticks = int(getattr(self, "_de_propose_ticks", 0))
        self._de_propose_ticks = ticks + 1
        key = ("iteration", int(it)) if it is not None else ("propose", ticks)
        if key != getattr(self, "_de_last_key", None):
            self._de_iter_count = int(getattr(self, "_de_iter_count", 0)) + 1
            self._de_last_key = key
        return int(self._de_iter_count) % 2

    def _set_de_pool(self, pool, parity):
        self._de_pool = pool
        self._de_parity = None if parity is None else int(parity)
        self._de_pool_token = int(getattr(self, "_de_pool_token", 0)) + 1

    def _note_de_partner_sample(self, state, reuse=None):
        """Head / single process, top of propose: every walker's rung ensemble.

        The start-of-propose ``coords[:ntemps, :, leaf]`` of EVERY walker per
        alive leaf (taken before the cold -> hot rotation edits a hot rung)
        and this propose's group parity (:meth:`_de_next_parity`). The body
        reads them from ``_de_pool`` (single process, the flat replica
        body); the walker-block head also ships them (``de_pool`` /
        ``de_parity`` in the payload). ``reuse`` is the pooled-adaptation
        sample of the same state (:meth:`_note_eigen_pool_sample`): its
        arrays are reused, so the pickled payload carries them once. No-op
        without a :class:`WalkerDifferenceJumpMove` in the stack.
        """
        self._de_pool_outgoing = None
        if not self._de_moves():
            self._set_de_pool(None, None)
            return
        try:
            sub = (getattr(state, "sub_states", None) or {}).get(self.branch_name)
            if sub is None or not getattr(sub, "tempered_initialized", False):
                self._set_de_pool(None, None)
                return
            if reuse:
                out = {int(leaf): x for leaf, (x, _b) in reuse.items()}
            else:
                coords = np.asarray(asnumpy(sub.coords), dtype=float)
                inds = np.asarray(asnumpy(sub.inds), dtype=bool)
                nt = min(int(self.ntemps), int(coords.shape[0]))
                out = {int(leaf): np.array(coords[:nt, :, leaf], copy=True)
                       for leaf in range(int(coords.shape[2])) if inds[0, 0, leaf]}
            self._set_de_pool(out or None, self._de_next_parity())
            if self._eigen_pool_block_head() and out:
                self._de_pool_outgoing = (out, self._de_parity)
        except Exception as exc:  # noqa: BLE001 -- never crash the sampler
            self._set_de_pool(None, None)
            logger.warning(
                "[%s_DE] cross-walker partner pool not taken (%r); every walker runs "
                "its ordinary stack this propose", str(self.branch_name).upper(), exc)

    def _de_leaf_spec(self, leaf):
        """``(active walkers (nwalkers_local,), partners (P, ndim))`` or ``None``.

        Active: the body's walkers whose GLOBAL index ``g`` has ``(g +
        parity) % 2 == 1``; partners: every rung of every walker of the
        OTHER parity, from the start-of-propose pool (unique rows, finite
        only). ``None`` (every walker runs its ordinary stack) without a
        pool, without an active walker here, or with fewer than two distinct
        partner states. Cached per (leaf, pool); each new pool counts one
        visit in the window tally.
        """
        key = (int(leaf), int(getattr(self, "_de_pool_token", 0)))
        cache = getattr(self, "_de_spec_cache", None)
        if cache is not None and cache[0] == key:
            return cache[1]
        spec = None
        pool = (getattr(self, "_de_pool", None) or {}).get(int(leaf))
        parity = getattr(self, "_de_parity", None)
        if pool is not None and parity is not None:
            pool = np.asarray(pool, dtype=float)
            nw_glob = int(pool.shape[1])
            w0 = self._walker_offset()
            g = w0 + np.arange(int(self.nwalkers))
            active = ((g + int(parity)) % 2) == 1
            partner_w = [w for w in range(nw_glob) if (w + int(parity)) % 2 == 0]
            if partner_w and np.any(active):
                P = pool[:, partner_w, :].reshape(-1, pool.shape[-1])
                P = P[np.all(np.isfinite(P), axis=1)]
                P = np.unique(P, axis=0) if P.size else P
                if P.shape[0] >= 2:
                    spec = (active, P)
            vt = self._leaf_tally("_de_visit_tally", leaf, (2,))
            vt[0] += 1.0 if spec is not None else 0.0
            vt[1] += 1.0
        self._de_spec_cache = (key, spec)
        return spec

    def _install_de_split(self, move_here, leaf, split_mask):
        """Per split: the jump's active rows and partners; the fallback's table.

        No-op unless ``move_here`` is a :class:`WalkerDifferenceJumpMove`.
        The fallback eigen move gets this split's table rows first (the
        same seam the eigen move itself uses), then the jump its spec
        (:meth:`_de_leaf_spec`) mapped onto the split's ``(temperature,
        walker)`` rows; no spec clears it, so every row takes the fallback.
        """
        if not isinstance(move_here, WalkerDifferenceJumpMove):
            return
        self._install_eigen_split_table(move_here.fallback, leaf, split_mask)
        name = self.branch_name
        try:
            spec = self._de_leaf_spec(leaf)
            if spec is None:
                move_here.clear_partners(name)
                return
            active_w, partners = spec
            S1 = np.asarray(split_mask, dtype=bool)
            nt = int(S1.shape[0])
            n_here = int(np.count_nonzero(S1)) // max(nt, 1)
            walker_of_row = np.nonzero(S1)[1].reshape(nt, n_here)
            active = np.asarray(active_w, dtype=bool)[walker_of_row]
            if not np.any(active):
                move_here.clear_partners(name)
                return
            move_here.set_partners(name, active, partners)
        except Exception as exc:  # noqa: BLE001 -- never crash the sampler
            move_here.clear_partners(name)
            if not getattr(self, "_de_install_warned", False):
                self._de_install_warned = True
                logger.warning("[%s_DE] leaf %d: cross-walker jump not installed (%r); "
                               "the fallback proposes", str(name).upper(), int(leaf), exc)

    # ------------------------------------------------------------------
    # eigen table persistence (the sidecar next to the run's store)
    # ------------------------------------------------------------------

    def _eigen_sidecar_path(self):
        """Sidecar path for this run, or ``None`` when persistence is off.

        The store path reaches the move layer exactly the way the GB
        refit move's ``fp`` does: stamped by the builder from
        ``curr.general_info.main_file_path`` (``eigen_store_path``), with
        the armed mid-iteration-checkpoint singleton
        (:func:`~lisatools.globalfit.midit_checkpoint.main_store_path`) as
        the fallback for moves built outside the recipe. Both are set on
        the sampling rank only, which is what keeps the writer single.
        """
        if not eigen_table_persist.persist_enabled():
            return None
        store = getattr(self, "eigen_store_path", None)
        if not store:
            store = midit_checkpoint.main_store_path()
        return eigen_table_persist.sidecar_path(store)

    def _eigen_data_identity(self):
        """The data's ``df`` (= 1/Tobs), or ``None`` if unreachable.

        A physics guard for the sidecar: a table built against a
        different observation length is meaningless. Best effort only —
        domains that do not expose ``df`` leave the guard abstaining
        (the sidecar path is already run-scoped, so it is belt and
        braces, not the primary protection).
        """
        try:
            return float(self.acs[0].data_res_arr.df)
        except Exception:  # noqa: BLE001 — a guard must not raise
            return None

    def _eigen_persist_guards(self):
        """The STRICT guard set shared by the load and the write."""
        return dict(
            scope=self._eigen_scope(),
            ndim=int(self.ndim),
            ntemps=int(self.ntemps),
            nwalkers=int(self.nwalkers),
            data_identity=self._eigen_data_identity(),
        )

    def _note_eigen_expansion_point(self, leaf, x0):
        """Record where a table was built (sidecar diagnostics only)."""
        if not hasattr(self, "_eigen_x0"):
            self._eigen_x0 = {}
        self._eigen_x0[int(leaf)] = np.asarray(x0, dtype=float).copy()

    def _adopt_persisted_eigen_table(self, leaf):
        """Reload ``leaf``'s table + visit counter; ``True`` if adopted.

        Adopting the counter as well is what keeps the refresh cadence
        continuous across a restart: the stored value is the counter as
        of the last build, so the next refresh lands a cadence interval
        after that build rather than a cadence interval after the
        relaunch.
        """
        path = self._eigen_sidecar_path()
        if path is None:
            return False
        try:
            got = eigen_table_persist.load_entry(
                path, self.branch_name, leaf, return_fisher=True, return_tune=True,
                **self._eigen_persist_guards()
            )
        except Exception as exc:  # never fail a run over its own cache
            eigen_refresh.logger.warning(
                "[eigen_refresh] %s leaf %d: eigen-table sidecar read "
                "failed (%r); rebuilding", self.branch_name, leaf, exc,
            )
            return False
        if got is None:
            return False
        axes, sigmas, visits, fisher, tune = got
        # the acceptance controller's multiplier ({BRANCH}_EIGEN_AXIS_ACC_TUNE,
        # 2026-10-18): restored, and divided out of the scale carried into
        # the next refresh, which re-applies it (no double application)
        tune = tune if self._eigen_axis_acc_tune() else None
        if tune is not None:
            self._eigen_axis_tune_dict()[leaf] = np.array(tune, dtype=float)
        if (getattr(self, "eigen_store_readonly", False)
                and np.asarray(axes).ndim >= 4):
            # a compute rank reading the head's sidecar: a per-(temp, walker)
            # stash is the HEAD'S block, not this rank's walkers -- rebuild
            eigen_refresh.logger.info(
                "[eigen_refresh] %s leaf %d: persisted table is a per-walker "
                "stash of another rank's block; rebuilding locally",
                self.branch_name, leaf,
            )
            return False
        # step-scale adaptation: the entry's ``sigmas`` are the ADAPTED
        # scale, ``sigmas_fisher`` (if present) the information-matrix floor
        if self._eigen_axis_adapt():
            floor = np.array(sigmas if fisher is None else fisher, copy=True)
            self._eigen_fisher_dict()[leaf] = floor
            if fisher is not None:
                # the safety cap holds for an adopted scale too (a sidecar
                # written before {BRANCH}_EIGEN_ADAPT_MAX_FACTOR existed, or
                # with a larger factor, would otherwise serve uncapped until
                # the leaf's next refresh)
                mf = self._eigen_adapt_max_factor()
                capped = factor_capped_sigmas(floor, sigmas, mf)
                n_cap = int(np.count_nonzero(np.asarray(capped) < np.asarray(sigmas)))
                if n_cap:
                    logger.info(
                        "[%s_EIGEN] leaf %d: persisted step scale factor-capped at "
                        "x%g on %d entries", str(self.branch_name).upper(), int(leaf),
                        float(mf), n_cap)
                    sigmas = capped
            if fisher is not None and not np.array_equal(fisher, sigmas):
                # carried over the restart: maps onto the next refresh's
                # axes while the (unpersisted) ring buffer refills
                carried = np.array(sigmas, dtype=float, copy=True)
                if tune is not None:
                    carried = carried / np.asarray(tune, dtype=float)
                self._eigen_carried_dict()[leaf] = (
                    np.array(axes, copy=True), carried)
        elif fisher is not None:
            # the adaptation is OFF now: run the information-matrix table,
            # never a scale adapted by an earlier process
            sigmas = fisher
        self._eigen_tables[leaf] = (axes, sigmas)
        self._eigen_visit_count[leaf] = int(visits)
        eigen_refresh.logger.info(
            "[eigen_refresh] %s leaf %d: adopted the persisted eigen table "
            "(shape %s, visits %d%s) from %s — skipping the information-"
            "matrix build", self.branch_name, leaf,
            getattr(axes, "shape", None), int(visits),
            "" if fisher is None else (
                ", adapted step scale" if self._eigen_axis_adapt()
                else ", information-matrix sigmas (adaptation off)"),
            path,
        )
        return True

    def _persist_eigen_table(self, leaf):
        """Write-through the table just built for ``leaf``."""
        if getattr(self, "eigen_store_readonly", False):
            return  # compute rank: the head is the sidecar's single writer
        path = self._eigen_sidecar_path()
        if path is None:
            return
        try:
            axes, sigmas = self._eigen_tables[int(leaf)]
            # the information-matrix floor rides along only while the
            # step-scale adaptation is on (off: the pre-adaptation entry)
            fisher = (getattr(self, "_eigen_fisher_sigmas", {}).get(int(leaf))
                      if self._eigen_axis_adapt() else None)
            # the controller's multiplier rides along only while it is on
            tune = (self._eigen_axis_tune_dict().get(int(leaf))
                    if self._eigen_axis_acc_tune() else None)
            entry = eigen_table_persist.make_entry(
                axes, sigmas,
                self._eigen_visit_count.get(int(leaf), 0),
                x0=getattr(self, "_eigen_x0", {}).get(int(leaf)),
                store_iteration=getattr(self, "_dbg_step", None),
                sigmas_fisher=fisher, axis_tune=tune,
                **self._eigen_persist_guards(),
            )
            eigen_table_persist.save_entry(
                path, self.branch_name, leaf, entry
            )
        except Exception as exc:  # never fail a run over its own cache
            eigen_refresh.logger.warning(
                "[eigen_refresh] %s leaf %d: eigen-table sidecar write "
                "failed (%r); continuing", self.branch_name, leaf, exc,
            )

    def _install_eigen_split_table(self, move_here, leaf, split_mask):
        """Install the (temp, walker)-sliced eigen table for one split.

        ``split_mask`` is the same ``(ntemps, nwalkers)`` boolean the
        proposal ``sets`` are built with, so the sliced table rows line up
        with the coords rows exactly. No-op for shared tables (installed
        at refresh time) and for non-eigen inner moves -- unless
        ``{BRANCH}_EIGEN_TEMPER_SIGMAS`` fixed a per-rung step for this
        visit (:meth:`_eigen_visit_rung_scale`): then every row of rung
        ``k`` gets its tempered sigma, for a shared table and a per-walker
        stash alike (same rows, same random draws, only the step widths
        differ).
        """
        if not isinstance(move_here, EigenAxisMove):
            return
        tabs = getattr(self, "_eigen_tables", None)
        if not tabs or int(leaf) not in tabs:
            return
        axes, sigmas = tabs[int(leaf)]
        tempered = (getattr(self, "_eigen_rung_scales", None) or {}).get(int(leaf))
        if getattr(axes, "ndim", 2) != 4:
            if tempered is None:
                return
            # shared table, one tempered sigma vector per rung
            sig_rung = tempered[1]
            nt, ndim = sig_rung.shape
            n_here = int(np.count_nonzero(split_mask)) // nt
            ax = np.broadcast_to(np.asarray(asnumpy(axes), dtype=float),
                                 (nt, n_here, 1, ndim, ndim))
            sg = np.broadcast_to(sig_rung[:, None, None, :], (nt, n_here, 1, ndim))
            move_here.set_axes(self.branch_name, ax, sg)
            return
        ndim = axes.shape[-1]
        ax = axes[split_mask]
        sg = sigmas[split_mask]
        if tempered is not None:
            # rows come out rung-major (C order), so the rung of each is
            # its first nonzero index
            rung = np.nonzero(np.asarray(split_mask))[0]
            sg = self._eigen_tempered_rows(ax, sg, tempered[0][rung])
        ax = ax.reshape(self.ntemps, -1, 1, ndim, ndim)
        sg = sg.reshape(self.ntemps, -1, 1, ndim)
        move_here.set_axes(self.branch_name, ax, sg)

    def _build_eigen_table(self, leaf, work):
        """One refresh product for ``leaf``, never raising.

        An ``eigen_table_builder`` attribute (installed by a branch setup,
        e.g. from a Settings ``info_matrix_gen``) overrides the default
        likelihood-difference routes — signature
        ``builder(move, leaf, widths) -> (axes, sigmas)`` (any shape
        :meth:`eryn.moves.EigenAxisMove.set_axes` accepts, or the stashed
        ``(ntemps, nwalkers, ndim[, ndim])`` per-walker form).
        Otherwise ``{BRANCH}_EIGEN_INFO=gram`` (:meth:`_eigen_info`, on a move
        with a ``_gram_templates`` hook) builds from the Gram matrix, falling
        back with a warning to the likelihood route on any error; the
        likelihood route's shape is picked by :meth:`_eigen_scope`.
        """
        widths = eigen_refresh.prior_box_widths(
            self.priors[self.branch_name], self.ndim
        )
        builder = getattr(self, "eigen_table_builder", None)
        if builder is not None:
            try:
                return builder(self, leaf, widths)
            except Exception as exc:  # degrade, never crash the sampler
                eigen_refresh.logger.warning(
                    "[eigen_refresh] %s leaf %d custom builder failed (%r); "
                    "using the identity fallback table",
                    self.branch_name, leaf, exc,
                )
                return eigen_refresh._fallback_table(widths)

        if (self._eigen_info() == "gram"
                and callable(getattr(self, "_gram_templates", None))):
            try:
                return self._build_eigen_table_gram(leaf, work, widths)
            except Exception as exc:  # degrade to the likelihood route
                eigen_refresh.logger.warning(
                    "[eigen_refresh] %s leaf %d Gram build failed (%r); "
                    "using the likelihood second-difference route",
                    self.branch_name, leaf, exc,
                )
        if self._eigen_scope() == "per_walker":
            return self._build_eigen_tables_per_walker(leaf, work, widths)
        return self._build_eigen_table_walker_max(leaf, work, widths)

    def _eigen_best_walker(self, leaf, work):
        """``(walker, cold_rows)`` for ``leaf``: the max-lnL COLD walker and
        the cold-chain sampling rows ``(nwalkers, ndim)`` it was picked from.

        Shared by the walker_max ll route and the Gram route. Scoring failure
        degrades to walker 0 with a warning -- never to a crash.
        """
        cold = np.asarray(asnumpy(work.coords[0, :, leaf]), dtype=np.float64)
        best = 0
        try:
            idx = np.arange(self.nwalkers, dtype=np.int32)
            ll = np.asarray(
                asnumpy(self.compute_like(self._to_phys(cold), data_index=idx)),
                dtype=float,
            )
            best = int(np.nanargmax(ll))
        except Exception as exc:
            eigen_refresh.logger.warning(
                "[eigen_refresh] %s leaf %d max-lnL walker selection failed "
                "(%r); using walker 0", self.branch_name, leaf, exc,
            )
        return best, cold

    # ------------------------------------------------------------------
    # Gram information matrix ({BRANCH}_EIGEN_INFO=gram). A subclass opts
    # in by implementing
    #     _gram_templates(phys_rows, walker) -> (arr, box)
    # -- one WDM template per waveform-basis row, stacked on axis 0 of
    # ``arr``, all on the WDM settings ``box`` -- built with the SAME
    # generator its scorer differences against (SOBBH: the chunked / lookup
    # fill; EMRI: the direct adapter; MBH: the batched windowed generator).
    # Anything it raises sends the refresh back to the ll route. Overriding
    # _gram_context puts the build on the walker's shard device.
    # ------------------------------------------------------------------

    def _gram_context(self, walker):
        """Device context for walker ``walker``'s Gram templates.

        The base move has no device routing, so this is a no-op context;
        sharded subclasses (EMRI direct, MBH batched) enter the device that
        owns ``walker``.
        """
        return nullcontext()

    def _build_eigen_table_gram(self, leaf, work, widths):
        """ONE shared ``(axes, sigmas)`` table for ``leaf`` from the Gram
        matrix at the max-lnL cold walker (:meth:`_eigen_best_walker`).

        The matrix is :meth:`_gram_info` at that walker's cold row; the
        table is the same eigen decomposition the ll route's walker_max
        scope produces (:func:`eigen_refresh._table_from_info`), so the
        inner move sees an identically shaped product. ``{BRANCH}_EIGEN_SCOPE``
        does not apply: the Gram route always builds one shared table.
        """
        best, cold = self._eigen_best_walker(leaf, work)
        x0 = cold[best]
        t0 = time.perf_counter()
        info, steps = self._gram_info(x0, best, widths, return_steps=True)
        # record the centre the final stencil used: x0 nudged inside the
        # prior box by the final steps (_gram_info)
        lo, hi = eigen_refresh.prior_box_bounds(
            self.priors[self.branch_name], x0.size)
        self._note_eigen_expansion_point(
            leaf, eigen_refresh.nudge_inside(x0, steps, lo, hi))
        eigen_refresh.logger.info(
            "[eigen_refresh] %s leaf %d Gram info matrix at walker %d in %.2f s",
            self.branch_name, leaf, best, time.perf_counter() - t0,
        )
        return eigen_refresh._table_from_info(info, widths)

    def _gram_eps_rel(self):
        """kwarg > ``{BRANCH}_EIGEN_GRAM_EPS_REL`` env > the ll route's
        :meth:`_eigen_eps_rel`: the Gram route's STARTING step, as a
        fraction of each prior-box width.

        Template differences have no second-difference cancellation, so they
        can (and for chirping sources must) step far smaller than the ll
        route; with the default target the start only seeds the one rescale
        pass of :meth:`_gram_info`.
        """
        val = getattr(self, "eigen_gram_eps_rel", None)
        if val is None:
            val = os.environ.get(
                f"{str(self.branch_name).upper()}_EIGEN_GRAM_EPS_REL", None)
        return float(val) if val is not None else float(self._eigen_eps_rel())

    def _gram_target(self):
        """kwarg > ``{BRANCH}_EIGEN_GRAM_TARGET`` env > 1e-3.

        The template change ``||h(x + s e_i) - h(x - s e_i)|| / 2`` each
        column's step ``s_i`` is tuned to, as a fraction of ``||h||`` (~1e-3
        rad of phase); see :meth:`_gram_info`. ``0`` keeps the fixed
        :meth:`_gram_eps_rel` steps. Needed because no single relative step
        works for every column of a chirping source: on the SOBBH chunked toy
        (2026-10-06) f_low goes nonlinear above ~1e-6 of its box while the
        mass columns hit the template's float noise floor below ~1e-7. With
        the tuning the Gram converges (target 3e-3 vs 1e-3: 5e-3
        correlation-normalized; 1e-4 vs 1e-3: 4e-4).
        """
        val = getattr(self, "eigen_gram_target", None)
        if val is None:
            val = os.environ.get(
                f"{str(self.branch_name).upper()}_EIGEN_GRAM_TARGET", "1e-3")
        return float(val)

    def _gram_info(self, x0, walker, widths, return_steps=False):
        """Gram matrix ``<dh_a|dh_b>`` at sampling point ``x0``, scored with
        walker ``walker``'s noise.

        ``dh_i = (h(x0 + s_i e_i) - h(x0 - s_i e_i)) / (2 s_i)``: central
        differences in the SAMPLING basis of the move's own batched templates
        (:meth:`_gram_templates`, one batch of ``2 * ndim + 1`` rows per
        pass). The inner products go through walker ``walker``'s container
        -- the same ``_slice_to_template`` PSD slice and
        :func:`~lisatools.diagnostic.inner_product` normalization the batched
        scorers use -- so the matrix is the likelihood's expected curvature.
        WDM templates only.

        Steps: ``s_i`` starts at :meth:`_gram_eps_rel` of the prior box; with
        :meth:`_gram_target` ``> 0`` ONE rescale pass sets each ``s_i`` so the
        half-difference ``||h(x0 + s_i e_i) - h(x0 - s_i e_i)|| / 2`` equals
        target x ``||h(x0)||`` (each factor clipped to [1e-3, 1e3], each step
        to [1e-12, 1e-2] of the box; a column whose template does not move
        at all takes the 1e3 factor). The templates are then rebuilt at the
        tuned steps.

        Each pass centres its stencil on ``x0`` nudged inside the prior box
        by that pass's steps (:func:`eigen_refresh.nudge_inside`), so no row
        leaves the prior (9mo job 751: an EMRI walker 1.3e-5 inside its
        ``cos(qK)`` bound got a NaN template from the ``-`` row on every
        refresh). A non-finite matrix raises a ``ValueError`` naming the
        columns whose half-differences were non-finite.

        Args:
            x0: ``(ndim,)`` sampling-basis expansion point.
            walker: walker whose container (data, PSD, device) scores it.
            widths: ``(ndim,)`` prior-box widths (the step unit).
            return_steps: also return the final steps (tests).

        Returns:
            ``(ndim, ndim)`` matrix, or ``(matrix, steps)``.
        """
        from ...diagnostic import inner_product

        best = int(walker)
        x0 = np.asarray(x0, dtype=float)
        widths = np.asarray(widths, dtype=float)
        nd = int(x0.size)
        lo, hi = eigen_refresh.prior_box_bounds(self.priors[self.branch_name], nd)
        steps = self._gram_eps_rel() * widths
        target = self._gram_target()
        # the scorers' inner-product kwargs, minus the PSD (passed explicitly
        # below) and the flags that change inner_product's return form
        like_kw = {
            k: v for k, v in (getattr(self, "waveform_like_kwargs", None) or {}).items()
            if k not in ("psd", "complex", "include_psd_info")
        }
        ac = self.acs.acs.flatten()[best]

        def ip(x, y, psd):
            v = inner_product(x, y, psd=psd, **like_kw)
            return float(np.real(v.get() if hasattr(v, "get") else v))

        def half_differences(s):
            """``(h0, [half_i], psd, xc)`` at steps ``s`` around ``xc`` =
            ``x0`` nudged inside the prior box by ``s``;
            ``half_i = (h(xc + s_i e_i) - h(xc - s_i e_i)) / 2``."""
            xc = eigen_refresh.nudge_inside(x0, s, lo, hi)
            # rows 2i / 2i+1 = xc +/- s[i] e_i; the last row is xc itself
            X = np.repeat(xc[None, :], 2 * nd + 1, axis=0)
            for i in range(nd):
                X[2 * i, i] += s[i]
                X[2 * i + 1, i] -= s[i]
            arr, box = self._gram_templates(self._to_phys(X), best)
            h0 = WDMSignal(arr[2 * nd], box)
            _, _, psd = ac._slice_to_template(h0)
            # the walker's PSD may live on another GPU than the templates
            # (SOBBH: comp device vs walker shard) -- keep the products local.
            # The module is the TEMPLATES' one, not ``self.xp``: that is
            # eryn's use_gpu-derived module, and the source-move builders
            # never pass use_gpu, so it is numpy on GPU runs and made this
            # guard a no-op (9mo job 751: diagnostic.py:248/294 on 2 ranks)
            psd = sensitivity_to_current_device(get_array_module(arr), psd)
            half = [WDMSignal(0.5 * (arr[2 * i] - arr[2 * i + 1]), box)
                    for i in range(nd)]
            return h0, half, psd, xc

        with self._gram_context(best):
            h0, half, psd, xc = half_differences(steps)
            if target > 0:
                hn = np.sqrt(max(ip(h0, h0, psd), 0.0))
                dn = np.array([np.sqrt(max(ip(d, d, psd), 0.0)) for d in half])
                scale = np.where(dn > 0, target * hn / np.where(dn > 0, dn, 1.0), 1e3)
                steps = np.clip(steps * np.clip(scale, 1e-3, 1e3),
                                1e-12 * widths, 1e-2 * widths)
                h0 = half = None  # release the first pass before the second
                h0, half, psd, xc = half_differences(steps)
            info = np.zeros((nd, nd))
            for a in range(nd):
                for b in range(a, nd):
                    info[a, b] = info[b, a] = (
                        ip(half[a], half[b], psd) / (steps[a] * steps[b]))
        if not np.all(np.isfinite(info)):
            # name the culprit for the fallback's log line: the columns whose
            # half-difference norm dn_i = ||half_i|| = s_i sqrt(G_ii) is
            # non-finite (rows with a non-finite entry if none is)
            with np.errstate(invalid="ignore"):
                dn = np.sqrt(np.abs(np.diag(info))) * steps
            bad = np.flatnonzero(~np.isfinite(dn))
            if bad.size == 0:
                bad = np.flatnonzero(~np.all(np.isfinite(info), axis=1))
            raise ValueError(
                f"non-finite information matrix: non-finite template "
                f"half-differences in columns {bad.tolist()} (centre "
                f"{xc[bad].tolist()}, steps {steps[bad].tolist()}, prior box "
                f"{lo[bad].tolist()} .. {hi[bad].tolist()})"
            )
        return (info, steps) if return_steps else info

    def _build_eigen_table_walker_max(self, leaf, work, widths):
        """ONE shared table, built at the max-lnL COLD-chain walker.

        The corner sweep scores against that walker's own data/noise
        (``data_index``). Selection failure degrades to walker 0 with a
        warning — never to a crash.
        """
        best, cold = self._eigen_best_walker(leaf, work)
        x0 = cold[best]
        eps_rel = self._eigen_eps_rel()
        bounds = eigen_refresh.prior_box_bounds(
            self.priors[self.branch_name], self.ndim)
        # the corners sit around x0 nudged inside the prior box (the
        # builder's bounds=); record that point, not the raw cold row
        self._note_eigen_expansion_point(leaf, eigen_refresh.nudge_inside(
            x0, eps_rel * np.asarray(widths, dtype=float), *bounds))

        def call_ll(x):
            x = np.atleast_2d(x)
            data_index = np.full(x.shape[0], best, dtype=np.int32)
            return self.compute_like(
                self._to_phys(x), data_index=data_index
            )

        return eigen_refresh.eigen_table_from_ll(
            call_ll, x0, widths, eps_rel=eps_rel, bounds=bounds
        )

    def _build_eigen_tables_per_walker(self, leaf, work, widths):
        """Per-(temperature, walker) tables for the leaf, ONE batched sweep.

        Every point gets its OWN information matrix at its own coords
        against its own walker's data/noise (no borrowing). Rows reach
        ``compute_like`` as whole (ntemps*nwalkers)-point blocks, so the
        per-walker ``data_index`` tiles by ``rows // n`` (the
        information_matrix_from_ll batching invariant). Returns the
        stashed ``(ntemps, nwalkers, ndim, ndim)`` / ``(..., ndim)`` form
        the red/blue seam slices per split.
        """
        nt, nw = self.ntemps, self.nwalkers
        pts = np.asarray(
            asnumpy(work.coords[:nt, :, leaf]), dtype=np.float64
        ).reshape(nt * nw, -1)
        point_walker = np.tile(np.arange(nw, dtype=np.int32), nt)
        eps_rel = self._eigen_eps_rel()
        bounds = eigen_refresh.prior_box_bounds(
            self.priors[self.branch_name], self.ndim)
        # each point's corners sit around it nudged inside the prior box
        self._note_eigen_expansion_point(leaf, eigen_refresh.nudge_inside(
            pts, eps_rel * np.asarray(widths, dtype=float), *bounds
        ).reshape(nt, nw, -1))

        def call_ll(x):
            x = np.atleast_2d(x)
            reps = x.shape[0] // point_walker.size
            data_index = np.tile(point_walker, reps)
            return self.compute_like(
                self._to_phys(x), data_index=data_index
            )

        axes, sigmas = eigen_refresh.eigen_tables_from_ll_batch(
            call_ll, pts, widths, eps_rel=eps_rel, bounds=bounds
        )
        ndim = axes.shape[-1]
        return (axes.reshape(nt, nw, ndim, ndim),
                sigmas.reshape(nt, nw, ndim))

    # ------------------------------------------------------------------
    # DCGA (multi-device) spread — absorbed from the legacy
    # MultiGPUResidualAddRemoveMove in the 2026-07 multi-GPU pass (one
    # move structure; active when the move is constructed with ``dcga=``).
    # ------------------------------------------------------------------

    @property
    def dcga(self) -> DomainComputationGroupArray:
        return self._dcga

    @property
    def run_async(self) -> bool:
        return self._run_async

    @property
    def run_threaded(self) -> bool:
        return self._run_threaded

    def create_waveform_gen_replicas(self):
        """One waveform generator per device, seeded from the generator
        object's ``.kwargs`` with the split's device-local orbits."""
        gen_obj = self._waveform_gen_obj
        if not hasattr(gen_obj, "kwargs"):
            raise ValueError(
                "Waveform generator must have a 'kwargs' attribute that "
                "contains the keyword arguments to initialize the waveform "
                "generator (needed to build per-device replicas)."
            )
        self._waveform_generators = []
        for i, device in enumerate(
            self.dcga.gpus
            if self.dcga.gpus is not None
            else [None] * self.dcga.num_splits
        ):
            with self.dcga.device_context(device):
                init_kwargs = gen_obj.kwargs.copy()
                if "orbits" in init_kwargs:
                    init_kwargs["orbits"] = (
                        self.dcga.computation_groups[i].orbits
                    )
                self._waveform_generators.append(
                    gen_obj.__class__(**init_kwargs)
                )

    @property
    def waveform_generators(self) -> list:
        return self._waveform_generators

    def make_args_tuple(self, coords) -> tuple:
        """Unpack coordinates into per-parameter arrays for the generator
        (JAX needs the separate memory addresses)."""
        return tuple(xp.array(coords[:, i]) for i in range(coords.shape[1]))

    def prepare_inputs(self, coords, data_index):
        """Partition coordinates/indices by split and place them on device."""
        positions_per_split, data_intra_index_per_split, _ = (
            self.dcga.unpack_indices(data_index)
        )
        coords_per_split = self.dcga.unpack_coords(
            positions_per_split, coords, keep_tuple=True
        )

        data_intra_index_per_split, coords_per_split = (
            self.dcga.place_on_device(
                items=(data_intra_index_per_split, coords_per_split)
            )
        )

        waveform_args_per_split = self.dcga._loop_operation(
            operation=[
                self.make_args_tuple for _ in self.dcga.computation_groups
            ],
            operation_args_per_split=coords_per_split,
        )

        return (
            positions_per_split,
            data_intra_index_per_split,
            waveform_args_per_split,
        )

    def aggregate_waveforms(self, waveforms_per_split) -> list[DomainBase]:
        """Flatten per-split waveform lists into one list."""
        waveforms = []
        for waveforms_split in waveforms_per_split:
            waveforms.extend(waveforms_split)
        return waveforms

    def _get_waveform_here_dcga(self, coords: np.ndarray) -> list[DomainBase]:
        self.free_gpu_memory()

        data_index = np.arange(coords.shape[0], dtype=np.int32)

        _, _, waveform_args_per_split = self.prepare_inputs(coords, data_index)

        operations = [
            getattr(waveform_gen, self.waveform_gen_method)
            for waveform_gen in self.waveform_generators
        ]

        waveforms_out = self.dcga._loop_operation(
            operation=operations,
            operation_args_per_split=waveform_args_per_split,
            operation_kwargs=self.waveform_gen_kwargs,
            aggregate_fn=self.aggregate_waveforms,
            run_threaded=self.run_threaded,
        )

        self.dcga.synchronize()

        return waveforms_out

    def _compute_like_dcga(
        self, coords_in: np.ndarray, data_index: np.ndarray
    ) -> np.ndarray:
        (
            positions_per_split,
            data_intra_index_per_split,
            waveform_args_per_split,
        ) = self.prepare_inputs(coords_in, data_index)

        waveform_like_operations = [
            getattr(waveform_gen, self.waveform_like_method)
            for waveform_gen in self.waveform_generators
        ]

        likelihood_args_per_split = self.dcga._loop_operation(
            operation=waveform_like_operations,
            operation_args_per_split=waveform_args_per_split,
            operation_kwargs=self.waveform_like_kwargs,
            positions_per_split=positions_per_split,
            run_threaded=self.run_threaded,
        )

        if not self.run_async:
            self.dcga.synchronize()

        likelihoods = self.dcga.compute_signal_likelihood(
            positions_per_split=positions_per_split,
            data_intra_per_split=data_intra_index_per_split,
            noise_intra_per_split=data_intra_index_per_split,
            likelihood_args_per_split=likelihood_args_per_split,
            likelihood_kwargs={'run_async': self.run_async},
            run_threaded=self.run_threaded,
        )

        # Release GPU arrays before freeing the pool: the large template
        # arrays must be dereferenced before free_all_blocks() so those
        # blocks are actually returned rather than staying pool-owned.
        del (
            likelihood_args_per_split,
            waveform_args_per_split,
            data_intra_index_per_split,
        )

        self.free_gpu_memory()

        if np.any(~np.isfinite(likelihoods)):
            logger.warning(
                f"Non-finite likelihoods encountered: {likelihoods}. "
                "This could be a sign of numerical issues."
            )
            if DEBUG_MODE:
                breakpoint()

        likelihoods = np.where(
            np.isfinite(likelihoods), likelihoods, -1e300
        )

        return likelihoods

    # ------------------------------------------------------------------
    # Debug / instrumentation plotting (mirrors the GB special-stretch move,
    # adapted to the "remove source from residual -> sample -> put it back"
    # choreography of this move). Activated per branch by the capitalised
    # ``{BRANCH_NAME}_DEBUG`` env var (e.g. EMRI_DEBUG / SOBBH_DEBUG /
    # MBH_DEBUG); companion knobs mirror GB's:
    #   {BRANCH}_DEBUG            "1" arms all debug hooks (default off)
    #   {BRANCH}_DEBUG_DIR        output folder (default ./gf_output/{branch}_debug/)
    #   {BRANCH}_DEBUG_PLOT_WALKER the single walker whose leaf is plotted (0)
    #   {BRANCH}_DEBUG_PLOT_LEAF   the single leaf plotted (unset -> the first
    #                              alive leaf each step)
    #   {BRANCH}_DEBUG_EVERY       plot only every Nth propose step (1)
    # ------------------------------------------------------------------
    def _setup_debug(self) -> None:
        """Read the per-branch ``{BRANCH}_DEBUG`` env options (called from __init__)."""
        p = str(self.branch_name).upper()
        self._dbg_prefix = p
        self.debug = bool(int(os.environ.get(f"{p}_DEBUG", "0")))
        self.debug_plot_dir = os.environ.get(
            f"{p}_DEBUG_DIR", f"./gf_output/{self.branch_name}_debug/"
        )
        self.debug_plot_walker = int(os.environ.get(f"{p}_DEBUG_PLOT_WALKER", "0"))
        _leaf = os.environ.get(f"{p}_DEBUG_PLOT_LEAF")
        self.debug_plot_leaf = int(_leaf) if _leaf not in (None, "") else None
        self.debug_every = max(1, int(os.environ.get(f"{p}_DEBUG_EVERY", "1")))
        # prev_logl consistency check (DEFAULT ON): cross-check the move's
        # scoring path (compute_like — the DCGA/GPU path when active) against
        # the per-container ACS path at the current-state points each leaf
        # visit. "0" disables, "1" warns (default), "strict" raises. Costs one
        # extra ntemps*nwalkers likelihood batch per leaf per propose.
        self.check_ll_mode = (
            os.environ.get(f"{p}_CHECK_LL")
            or os.environ.get("ADDREMOVE_CHECK_LL", "1")
        ).strip().lower()
        # ``{BRANCH}_CHECK_LL_EVERY`` wins when set; otherwise the class
        # default (every visit here; subclasses whose cross-check is a costly
        # slow-path rebuild override ``_check_ll_every_default``).
        self.check_ll_every = max(
            1, int(os.environ.get(f"{p}_CHECK_LL_EVERY") or self._check_ll_every_default)
        )
        # Per-leaf cold-chain <d|h>/<h|h> sub-state record. On the slow
        # container path it costs one EXTRA nwalkers waveform batch per
        # leaf per iteration, so it defaults OFF here; fast-kernel
        # subclasses flip ``_record_dh_default`` (their record is one
        # cheap vectorized call). ``{BRANCH}_RECORD_DH`` overrides either
        # way; when off the sub-state d_h/h_h stay NaN.
        self.record_inner_products = bool(int(
            os.environ.get(f"{p}_RECORD_DH", self._record_dh_default)
        ))
        # Tempering diagnostics: {BRANCH}_SWAP_DEBUG=1 logs one [SWAP_DBG]
        # line per repeat (cold-chain lnL before/after the temperature swap,
        # inner acceptance, per-rung swap counts, fancy flag).
        self.swap_debug = bool(int(os.environ.get(f"{p}_SWAP_DEBUG", "0")))
        # {BRANCH}_PERMUTE_EVERY overrides the ctor permute_every (a
        # CADENCE in proposes since 2026-08-27; 1 = every propose); <= 0
        # DISABLES the walker-permuting (fancy) swap entirely — the A/B
        # lever for isolating fancy-swap pathologies.
        _pe_env = os.environ.get(f"{p}_PERMUTE_EVERY") or os.environ.get(
            "ADDREMOVE_PERMUTE_EVERY"
        )
        if _pe_env is not None:
            self.permute_every = int(_pe_env)
            logger.info(
                "[%s] permute_every overridden via env -> %d (<=0 disables "
                "the fancy swap)", p, self.permute_every,
            )
        # Colour scale keyed to the RESIDUAL (where the other sources are
        # already subtracted) and log10 by default, so this source stays
        # visible even in a crowded band. {P}_DEBUG_LOG=0 -> linear.
        self.debug_log = bool(int(os.environ.get(f"{p}_DEBUG_LOG", "1")))
        self._dbg_plot_counter = 0
        self._dbg_step = 0
        if self.debug:
            logger.info(
                "[%s_DEBUG] armed: dir=%s walker=%d leaf=%s every=%d",
                p, self.debug_plot_dir, self.debug_plot_walker,
                self.debug_plot_leaf, self.debug_every,
            )

    def _dbg_trace_this_step(self) -> bool:
        """True when the current propose step should produce debug plots."""
        return self.debug and (self._dbg_step % self.debug_every == 0)

    def _dbg_residual(self, walker: int):
        """Host copy of ``walker``'s current residual array (domain-native shape)."""
        return asnumpy(self.acs[int(walker)].data.arr).copy()

    def _dbg_template(self, coords_row_in, walker: int):
        """Host copy of one source's template array, in the residual's domain shape.

        ``coords_row_in`` is a single already-transformed (waveform-basis)
        coordinate row. Built through the walker's container ``signal_gen``
        machinery — the same path :meth:`_apply_cold_chain_sources` folds into
        the residual — so the plotted template matches the fit exactly.
        Returns ``None`` on any failure (debug must never break the sampler).
        """
        try:
            ac = self.acs[int(walker)]
            sig = ac.build_template(
                {self.branch_name: np.asarray(coords_row_in)},
                waveform_kwargs=self._branch_waveform_kwargs(),
                signal_gen=self._resolve_signal_gen_override(ac),
                apply_transform=False,
            )
            arr = sig.arr if hasattr(sig, "arr") else sig
            return asnumpy(arr).copy()
        except Exception as exc:  # noqa: BLE001
            logger.debug("[%s_DEBUG] template build skipped: %r", self._dbg_prefix, exc)
            return None

    def _dbg_weighted_ll(self, residual_arr, walker) -> float:
        """``-1/2 <r|r>`` for a residual snapshot, PSD-weighted.

        Never hand-roll the inner product: build the snapshot as the domain
        object the walker's container expects and let the container's OWN
        installed :meth:`AnalysisContainer.inner_product` apply the sensitivity
        weighting (the same primitive the sampler scores with). A raw
        ``sum|r|^2`` drops the PSD entirely and is not a likelihood.
        """
        ac = self.acs[int(walker)]
        settings = ac.data.settings
        snapshot = settings.associated_class(asnumpy(residual_arr), settings)
        saved = ac.data
        try:
            ac.data = snapshot
            return -0.5 * float(np.real(ac.inner_product()))
        finally:
            ac.data = saved

    def _debug_plot_source_sequence(self, leaf, walker, snaps, template, meta) -> None:
        """Save a GB-style flip-book of ``[template | data | residual]`` frames.

        Mirrors the GB special-stretch move's per-moment 3x3 sequence figures:
        one figure PER MOMENT of the remove -> sample -> re-add choreography,
        each with rows = TDI channels (X/Y/Z) and columns

          1. TOTAL TEMPLATE — this leaf's recovered source ``h`` (fixed),
          2. TOTAL DATA     — the data this source is fit against, i.e. the
                              source-isolated residual with every other source
                              already removed (fixed reference),
          3. RESIDUAL       — the ACS residual AT THAT MOMENT (varies).

        The frames are the entry (source folded into the fit), removed (source
        isolated for sampling), and refit (updated source folded back) states.
        Template and data columns are held fixed across frames while the
        residual column changes, so flipping through the numbered files
        ``..._f0_...`` / ``_f1_`` / ``_f2_`` animates the source leaving and
        re-entering the residual directly. A single colour scale (per channel)
        is shared across ALL frames and columns so the flip is calibrated.
        Domain-adaptive (WDM image / FD|TD line); any failure is logged and
        swallowed.

        NOTE vs GB: this move does not mutate the residual during the repeat
        loop (proposals are scored on the fly), so the GB "before/after
        add-back" pair collapses to a single ``removed`` frame here.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            if template is None:
                return
            tmpl = np.asarray(template)
            # Fixed "data" reference = the source-isolated residual (others
            # removed). Fall back to entry + template if the exposed snapshot
            # is missing.
            data = snaps.get("exposed")
            if data is None and snaps.get("before_removal") is not None:
                data = np.asarray(snaps["before_removal"]) + tmpl
            if data is None:
                return
            data = np.asarray(data)
            if tmpl.shape != data.shape:
                return  # template build failed / shape mismatch -> skip cleanly

            frames = [
                ("f0", "entry: source in fit", snaps.get("before_removal")),
                ("f1", "removed: source isolated", snaps.get("exposed")),
                ("f2", "refit: source re-added", snaps.get("after_readd")),
            ]
            frames = [(k, t, np.asarray(r)) for k, t, r in frames if r is not None]
            if not frames:
                return

            nch = data.shape[0] if data.ndim >= 2 else 1
            chan_labels = (
                ["X", "Y", "Z"] if nch == 3 else [f"chan {c}" for c in range(nch)]
            )
            kind = meta.get("domain", "wdm")

            def _ch(a, c):
                # Signed real coefficients (WDM w_mn are real), so a diverging
                # colour map centres an emptied template / clean residual at
                # white and shows signal as colour.
                a = np.real(np.asarray(a))
                a = a[None] if a.ndim == 1 else a
                return a[c] if a.shape[0] > c else a[0]

            # Diverging colour scale (the WDM class default, RdBu), symmetric
            # about zero and keyed to the TOTAL DATA column — the
            # source-isolated residual, i.e. THIS source's own signal (other
            # sources are already subtracted there, so a crowded band cannot
            # blow out the scale). Keying to the residual instead (the old
            # behavior) backfires on clean/noiseless fits: the residual is
            # numerical dust, its percentile becomes the vmax, and the whole
            # plane lights up with ~1e-31-level leakage texture. Keyed to the
            # signal, a clean residual renders WHITE and colour appears only
            # where the source sits. SymLogNorm (the ``debug_log`` default)
            # keeps the weak wings visible; one norm per channel shared
            # across all frames -> the flip-book is calibrated.
            from matplotlib.colors import Normalize, SymLogNorm

            use_log = bool(getattr(self, "debug_log", True)) and kind == "wdm"
            norm_row = []
            for ch in range(nch):
                a = np.abs(_ch(data, ch))
                vmax = float(np.max(a)) if np.any(a > 0) else 1.0
                vmax = vmax or 1.0
                norm_row.append(
                    SymLogNorm(linthresh=vmax * 1e-3, vmin=-vmax, vmax=vmax, base=10)
                    if use_log else Normalize(vmin=-vmax, vmax=vmax)
                )

            # Crop the frequency axis to the layers the source actually
            # occupies (a narrow-band source spans a handful of the full
            # band's layers — GB debug plots read well because they are
            # per-band slabs; mirror that). Margin keeps leakage wings
            # visible; the full band returns if the source is broadband.
            ylim = None
            if kind == "wdm":
                occ = np.zeros(0, dtype=int)
                for ch in range(nch):
                    slab = np.abs(_ch(data, ch))
                    if slab.ndim == 2 and np.any(slab > 0):
                        thr = float(np.max(slab)) * 1e-3
                        occ = np.union1d(occ, np.where(np.any(slab > thr, axis=1))[0])
                if occ.size:
                    pad = 25
                    lo = max(int(occ.min()) - pad, 0)
                    hi = int(occ.max()) + pad
                    ylim = (lo, hi)

            rr_entry = (
                self._dbg_weighted_ll(snaps["before_removal"], walker)
                if snaps.get("before_removal") is not None else float("nan")
            )
            os.makedirs(self.debug_plot_dir, exist_ok=True)
            base = f"{self.branch_name}_debug_leaf{leaf}_w{walker}_s{self._dbg_step:04d}"

            for fkey, ftitle, residual in frames:
                # The "total template" is the CURRENT total template = data -
                # residual. When the source is removed (f1) it leaves the model
                # and reappears in the residual, so with a single source its
                # power switches from the template column to the residual
                # column across the flip-book.
                total_template = np.asarray(data) - np.asarray(residual)
                cols = [("total template", total_template), ("total data", data),
                        (f"residual — {ftitle}", residual)]
                fig, axes = plt.subplots(
                    nch, 3, figsize=(3.6 * 3, 2.7 * nch), squeeze=False, sharex=True,
                )
                for ci, (ctitle, a) in enumerate(cols):
                    for ch in range(nch):
                        ax = axes[ch][ci]
                        slab = _ch(a, ch)
                        if kind == "wdm" and slab.ndim == 2:
                            im = ax.imshow(
                                slab, aspect="auto", origin="lower",
                                cmap="RdBu", norm=norm_row[ch],
                            )
                            if ylim is not None:
                                ax.set_ylim(*ylim)
                            if ci == 2:
                                cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
                                cb.set_label("coeff", fontsize=7)
                        else:
                            ax.plot(_ch(a, ch).ravel(), lw=0.6)
                        if ch == 0:
                            ax.set_title(ctitle, fontsize=9)
                        if ci == 0:
                            ax.set_ylabel(chan_labels[ch], fontsize=9)
                rr_here = self._dbg_weighted_ll(residual, walker)
                fig.suptitle(
                    f"{self.branch_name} leaf {leaf} walker {walker} step "
                    f"{self._dbg_step} — {ftitle}  | -0.5<r|r>_psd entry={rr_entry:.3e} "
                    f"here={rr_here:.3e}",
                    fontsize=10,
                )
                fname = os.path.join(self.debug_plot_dir, f"{base}_{fkey}.png")
                fig.tight_layout(rect=[0, 0, 1, 0.96])
                fig.savefig(fname, dpi=120, bbox_inches="tight")
                plt.close(fig)
                logger.info("[%s_DEBUG] saved -> %s", self._dbg_prefix, fname)
            self._dbg_plot_counter += 1
        except Exception as exc:  # noqa: BLE001
            logger.info("[%s_DEBUG] plot skipped: %r", self._dbg_prefix, exc)

    @property
    def periodic(self):
        """Periodicity dictionary propagated to all inner moves."""
        return self._periodic

    @periodic.setter
    def periodic(self, periodic):
        self._periodic = periodic
        if periodic is not None:
            for tmp_move in self.moves:
                if tmp_move.periodic is None:
                    tmp_move.periodic = periodic

    def free_gpu_memory(self):
        if self._dcga is not None:
            self._dcga.free_gpu_memory()
            return
        if self.xp is not np:
            self.xp.get_default_memory_pool().free_all_blocks()

    def check_add_skip_swap_info(self, state):
        """Populate ``skip_swap_branches`` on each per-leaf temperature control.

        Branches other than :attr:`branch_name` are skipped during temperature
        swapping so this move does not corrupt unrelated branches.
        """

        if self.temperature_controls[0].skip_swap_branches is not None:
            return

        if len(state.branches) > 1:
            skip_swap_branches = [key for key in state.branches.keys()]
            skip_swap_branches.remove(self.branch_name)

        else:
            skip_swap_branches = []

        for i in range(self.nleaves_max):
            self.temperature_controls[i].skip_swap_branches = skip_swap_branches

    @property
    def own_signal_gen(self) -> MoveSignalGen:
        """This move's engine-convention generator adapter (lazy singleton)."""
        if getattr(self, "_own_signal_gen", None) is None:
            self._own_signal_gen = MoveSignalGen(self)
        return self._own_signal_gen

    def _supports_apply_transform(self, gen) -> bool:
        """``True`` when ``gen`` explicitly accepts the ``apply_transform`` kwarg.

        The move feeds waveform-basis parameters, so a generator is only
        usable as the branch default if it can be told to skip its internal
        sampling→waveform transform (a bare ``**kwargs`` doesn't count — it
        would silently forward the flag into the wave wrap). Results are
        cached per generator object.
        """
        cache = getattr(self, "_apply_transform_support_cache", None)
        if cache is None:
            cache = self._apply_transform_support_cache = {}
        key = id(gen)
        if key not in cache:
            try:
                parameters = inspect.signature(gen).parameters
                cache[key] = any(
                    p.name == "apply_transform"
                    and p.kind in (p.KEYWORD_ONLY, p.POSITIONAL_OR_KEYWORD)
                    for p in parameters.values()
                )
            except (TypeError, ValueError):
                cache[key] = False
        return cache[key]

    def _resolve_signal_gen_override(self, ac):
        """Resolve the branch generator for one container: installed first.

        Default (returns ``None``): the generator already installed on the
        container's dict-based ``signal_gen`` under :attr:`branch_name` —
        the same generator the engine uses for residual rebuilds — provided
        it supports the ``apply_transform`` kwarg the move relies on.

        When the container does not carry that model name, this move's
        :attr:`own_signal_gen` is added: durably registered when that cannot
        clobber anything (no generator installed, or a dict without the
        key), otherwise (bare single-callable ``signal_gen``, or an
        installed entry without ``apply_transform`` support) injected for
        the call only via the returned per-call ``{branch: gen}`` mapping.
        """
        gen_map = getattr(ac, "_signal_gen", None)
        if isinstance(gen_map, dict):
            installed = gen_map.get(self.branch_name)
            if installed is not None:
                if self._supports_apply_transform(installed):
                    return None
                # installed under our name but cannot skip its transform:
                # score/apply with our own generator, without clobbering it.
                return {self.branch_name: self.own_signal_gen}
            new_map = dict(gen_map)
            new_map[self.branch_name] = self.own_signal_gen
            ac.signal_gen = new_map
            return None
        if gen_map is None:
            ac.signal_gen = {self.branch_name: self.own_signal_gen}
            return None
        # bare single-callable installed by someone else: never clobber it.
        return {self.branch_name: self.own_signal_gen}

    def _branch_waveform_kwargs(self) -> dict:
        """Per-model ``waveform_kwargs`` mapping for this branch's generator."""
        return {self.branch_name: self.waveform_gen_kwargs}

    def _apply_cold_chain_sources(self, coords, sign):
        """One-at-a-time template build + apply to keep peak RAM flat.

        ``sign=-1`` subtracts (add_back), ``sign=+1`` adds (remove). ``coords``
        are already-transformed (waveform-basis) rows, one per walker. Each
        walker's template is assembled by its container's ``signal_gen``
        machinery (:meth:`AnalysisContainer.build_template`), defaulting to
        the generator the engine installed for this branch — with
        ``apply_transform=False`` since the move already applied the branch
        transform (applied exactly once either way) — and falling back to
        this move's own :class:`MoveSignalGen` when the container doesn't
        carry the branch (see :meth:`_resolve_signal_gen_override`). The
        template is then applied with the device-aware
        :meth:`AnalysisContainerArray.signal_operation`. Never holds more
        than a single source's waveform in memory.
        """
        # Delegated to the ACA (2026-07-28): apply_signal_from_params is the
        # shard-aware fused build+apply primitive (one live template per
        # split, per-split device contexts, gc/pool hygiene) — the move no
        # longer reimplements the ACA's residual machinery. A current-state
        # source outside the waveform's domain of validity is skipped
        # ("skip"): deterministic, so the add and remove passes skip
        # identically — its template is zero, matching its -1e300 sentinel.
        self.acs.apply_signal_from_params(
            sign,
            {self.branch_name: coords},
            index=np.arange(int(np.shape(coords)[0])),
            waveform_kwargs=self._branch_waveform_kwargs(),
            signal_gen_resolver=self._resolve_signal_gen_override,
            apply_transform=False,
            domain_error="skip",
        )

    def add_back_in_cold_chain_sources(self, coords):
        """Subtract current cold-chain sources from the residual (one walker at a time)."""
        self._apply_cold_chain_sources(coords, sign=-1)

    def remove_cold_chain_sources(self, coords):
        """Add current cold-chain sources back into the residual (one walker at a time)."""
        self._apply_cold_chain_sources(coords, sign=+1)

        # ll_tmp3 = self.acs.likelihood(
        #     source_only=True
        # )  #  - xp.sum(xp.log(xp.asarray(psd[:2])), axis=(0, 2))).get()

    def get_waveform_here(self, coords: np.ndarray) -> DomainBaseArray | list[DomainBase]:
        """Get the waveforms for the given source coordinates.

        Each call to ``waveform_gen`` returns a :class:`~lisatools.domains.DomainBase`
        (e.g. :class:`~lisatools.domains.STFTSignal` or
        :class:`~lisatools.domains.FDSignal`).  The results are collected into a
        :class:`~lisatools.domains.DomainBaseArray`, which batches them for
        vectorized downstream operations when all signals share the same
        domain settings.

        Args:
            coords: Source coordinates, shape ``(n_sources, ndim)``.

        Returns:
            :class:`~lisatools.domains.DomainBaseArray` of length ``n_sources``.

        """
        if self._dcga is not None:
            return self._get_waveform_here_dcga(coords)

        _free_pool()

        waveforms = []
        for i in range(coords.shape[0]):
            waveforms.append(self.waveform_gen(*coords[i], **self.waveform_gen_kwargs))

        return DomainBaseArray(waveforms)

    def setup_likelihood_here(self, coords):
        """Set up state before computing the likelihood (DCGA path computes
        the per-split <d|d> terms; the plain path needs nothing)."""
        if self._dcga is not None:
            self._dcga.compute_d_d_terms()

    def compute_acs_like(self, coords_in, data_index, signal_gen=None, **kwargs):
        """
        Compute the likelihood for the given coordinates and data index using the analysis container array.

        Args:
            coords_in: already-transformed (waveform-basis) coordinates of the sources for which we want to compute the likelihood. Shape is (n_sources, ndim). The containers' generators are called with ``apply_transform=False`` so the branch transform — applied once by this move's choreography — is never applied twice.
            data_index: index of the data for which we want to compute the likelihood. Shape is (n_sources,).
            signal_gen: optional waveform generator override. By default (``None``) each targeted container's own installed generator for this branch is used (falling back to this move's generator when the container doesn't carry it — see :meth:`_resolve_signal_gen_override`), so scoring and residual bookkeeping share one template-assembly path. Pass a waveform-basis callable to score with a different generator than the one used for proposing, for example when using heterodyned likelihoods; it is wrapped in :class:`MoveSignalGen` and swapped in per call.
            kwargs: additional keyword arguments for the likelihood computation function.

        Returns:
            ll: likelihood for the given coordinates and data index. Shape is (n_sources,).
        """
        # TODO: we should probably move the prior in here even though
        # in general with current setup it should only be points in the prior
        # that make it here
        # Normalize to a 2D ``(n_sources, ndim)`` batch: ``atleast_2d`` on a
        # 1D vector yields ``(1, ndim)`` (row), which is exactly one source --
        # NOT ndim sources.
        # Host-normalize: sampling-basis params are host-convention scalars
        # for the engine generators (they np.asarray the row).
        coords_in = np.atleast_2d(asnumpy(coords_in))
        data_index_np = np.atleast_1d(asnumpy(data_index)).astype(int)
        source_only = kwargs.pop("source_only", False)

        # Params-based likelihood through each targeted container's
        # signal_gen machinery (template assembly + scoring live in
        # :meth:`AnalysisContainer.calculate_signal_likelihood`; the move no
        # longer generates waveforms itself). Rows are evaluated one at a
        # time, which is required anyway: every generator currently wired
        # into this move is SINGLE-SOURCE (EMRI FEW GenerateEMRIWaveform;
        # the SOBBH / MBH tdionfly + legacy ResponseWrapper wraps all build
        # ONE combined template per call), so a batched
        # ``signal_gen(*coords_in.T)`` would crash or silently sum candidates.
        custom_override = (
            None if signal_gen is None
            else {self.branch_name: MoveSignalGen(self, signal_gen)}
        )

        # Delegated to the ACA (2026-07-28): scoring goes through the ACA's
        # vectorized dispatcher (calculate_signal_likelihood), which owns the
        # shard grouping, per-split device contexts, and the -1e300
        # WaveformDomainError sentinel — the move no longer reimplements the
        # sharded scoring loop. The per-container generator override is
        # resolved up front (this also durably registers the move's generator
        # on containers that don't carry the branch); rows are grouped by
        # resolved override so each group is ONE vectorized call.
        n_rows = len(data_index_np)
        if custom_override is not None:
            groups = [(custom_override, np.arange(n_rows))]
        else:
            def _ov_key(ov):
                return (
                    None if ov is None
                    else tuple(sorted((k, id(v)) for k, v in ov.items()))
                )

            by_key: dict = {}
            for pos, i in enumerate(data_index_np):
                ov = self._resolve_signal_gen_override(self.acs[int(i)])
                key = _ov_key(ov)
                if key not in by_key:
                    by_key[key] = (ov, [])
                by_key[key][1].append(pos)
            groups = [
                (ov, np.asarray(rows, dtype=int))
                for ov, rows in by_key.values()
            ]

        ll = np.full(n_rows, -1e300, dtype=float)
        for ov, rows in groups:
            vals = self.acs.calculate_signal_likelihood(
                {self.branch_name: coords_in[rows]},
                index=data_index_np[rows],
                domain_error_value=-1e300,
                signal_gen=ov,
                waveform_kwargs=self._branch_waveform_kwargs(),
                apply_transform=False,
                source_only=source_only,
                **kwargs,
            )
            ll[rows] = np.real(np.asarray(vals))
        return ll

    def compute_like(self, coords_in, data_index):
        """Score rows; in one-walker replica mode the rows scatter over the replicas.

        Every likelihood evaluation of this move funnels through here: the
        entry ``prev_logl``, the in-model scoring, the fancy-swap re-scoring
        (``log_like_for_fancy_swaping``) and the eigen-table builds. The
        rank-local scorer is :meth:`compute_like_local` (DCGA or container
        path) -- subclasses override THAT, never this. The gathered
        ``d_h``/``h_h`` side outputs land on ``self._last_d_h/_last_h_h``
        (the chunked SOBBH scorer's convention).
        """
        if not self.rows_active():
            return self.compute_like_local(coords_in, data_index)
        coords_np = np.atleast_2d(np.asarray(asnumpy(coords_in), dtype=np.float64))
        idx = np.atleast_1d(np.asarray(asnumpy(data_index))).astype(np.int32)
        # ``walkers=idx``: under the flat control plane the body runs at the
        # FULL ensemble width, so a row's ``data_index`` IS its global walker
        # -- which is exactly the key the router needs to send it to a rank
        # that holds that walker, and to rewrite it to that rank's local ACA
        # row. At one walker block the router ignores it and the split is the
        # original contiguous one.
        out = self.row_fanout.run(
            "ll_rows", {"coords": coords_np, "data_index": idx},
            local_body=self._score_rows_local, walkers=idx,
        )
        self._last_d_h = out["d_h"]
        self._last_h_h = out["h_h"]
        return out["ll"]

    def compute_like_local(self, coords_in, data_index):
        """Rank-local scorer (the pre-replica ``compute_like`` body)."""
        if self._dcga is not None:
            return self._compute_like_dcga(coords_in, data_index)
        return self.compute_acs_like(
            coords_in, data_index, **self.waveform_like_kwargs
        )

    def _score_rows_local(self, rows):
        """Serve one row chunk: ``{"ll", "d_h", "h_h"}`` in row order (empty-safe)."""
        coords = np.asarray(rows["coords"], dtype=np.float64)
        idx = np.asarray(rows["data_index"]).astype(np.int32)
        n = int(coords.shape[0])
        if n == 0:
            return {"ll": np.zeros(0), "d_h": np.zeros(0), "h_h": np.zeros(0)}
        self._last_d_h = None
        self._last_h_h = None
        ll = np.asarray(self.compute_like_local(coords, idx), dtype=float).reshape(n)

        def _side(arr):
            if arr is None:
                return np.full(n, np.nan)
            return np.asarray(arr, dtype=float).reshape(n)

        return {"ll": ll, "d_h": _side(self._last_d_h), "h_h": _side(self._last_h_h)}

    def serve_ll_rows(self, payload, clock, model):
        return self._score_rows_local(payload["rows"])

    def _check_like_local(self, coords_in, data_index):
        """The check_ll cross-check: the container path with the MOVE's own generator."""
        return self.compute_acs_like(
            coords_in,
            data_index=data_index,
            signal_gen=self.waveform_gen,
            **self.waveform_like_kwargs,
        )

    def compute_check_like(self, coords_in, data_index):
        if not self.rows_active():
            return np.asarray(self._check_like_local(coords_in, data_index))
        coords_np = np.atleast_2d(np.asarray(asnumpy(coords_in), dtype=np.float64))
        idx = np.atleast_1d(np.asarray(asnumpy(data_index))).astype(np.int32)
        out = self.row_fanout.run(
            "ll_rows_check", {"coords": coords_np, "data_index": idx},
            local_body=self._check_rows_local, walkers=idx,
        )
        return out["ll"]

    def _check_rows_local(self, rows):
        n = int(np.shape(rows["coords"])[0])
        if n == 0:
            return {"ll": np.zeros(0)}
        ll = self._check_like_local(np.asarray(rows["coords"]), np.asarray(rows["data_index"]))
        return {"ll": np.asarray(ll, dtype=float).reshape(n)}

    def serve_ll_rows_check(self, payload, clock, model):
        return self._check_rows_local(payload["rows"])

    def _replay_cold_chain(self, kind, coords_in, leaf):
        """Expose / setup / fold the leaf's cold-chain source on EVERY replica.

        Gated on replica mode itself (``self.row_fanout is None``), NOT on
        ``rows_active()`` / the ``{PREFIX}_LIKELIHOOD_FANOUT`` knob: that knob
        only changes who SCORES rows (head-only vs fanned-out), it must not
        stop the replays that keep every replica's residual state aligned —
        otherwise, with the knob off, replicas would score against stale
        residuals. Single process (no row_fanout): a direct local call,
        identical to the pre-replica choreography.
        """
        payload = {
            "kind": str(kind),
            "coords": np.asarray(asnumpy(coords_in), dtype=np.float64),
            "leaf": int(leaf),
        }
        if self.row_fanout is None:
            self._apply_cold_chain_replay(payload)
            return
        self.row_fanout.replay("ar_replay", payload, local_body=self._apply_cold_chain_replay)

    def _replay_block_rows(self, coords):
        """The rows of a replayed cold-chain batch that THIS rank's ACA holds.

        A replay ships the flat body's cold-chain rows: one per walker of the
        WHOLE ensemble, in global walker order. Under one walker block
        (one-walker replica mode) every rank holds every walker, so that is
        the whole batch. With several blocks AND replicas (the
        ``GF_GPU_ROUTING`` both-axes layout) a rank's ACA holds only its
        block ``[w0, w1)``: applying all N rows indexed ``0..N-1`` to a B-row
        ACA fails the ``signal_operation`` bound (and with B > 1 would expose
        another block's walkers into this one's rows). Expose, setup and fold
        all go through here, so a rank's window/offset and the template it
        later folds come from the same rows.
        """
        fanout = getattr(self, "fanout", None)
        if self.row_fanout is None or fanout is None:
            return coords
        layout = fanout.layout
        if int(getattr(layout, "n_blocks", 1)) <= 1:
            return coords
        w0, w1 = layout.block_of(fanout.rank)
        return coords[int(w0):int(w1)]

    def _apply_cold_chain_replay(self, payload):
        kind = payload["kind"]
        self._current_leaf = int(payload["leaf"])
        coords = self._replay_block_rows(np.asarray(payload["coords"], dtype=np.float64))
        if kind == "expose":
            self.remove_cold_chain_sources(coords)
        elif kind == "setup":
            self.setup_likelihood_here(coords)
        elif kind == "fold":
            self.add_back_in_cold_chain_sources(coords)
        else:
            raise ValueError(f"unknown cold-chain replay kind {kind!r}")

    def serve_ar_replay(self, payload, clock, model):
        self._apply_cold_chain_replay(payload)
        return None

    # ---- search-stage cold -> hot rotation (user ruling 2026-10-14) ---------
    def _cold_to_hot_every(self) -> int:
        """kwarg > ``{BRANCH}_COLD_TO_HOT_EVERY`` env > 125 proposals (<= 0 = off)."""
        val = getattr(self, "cold_to_hot_every", None)
        if val is None:
            val = os.environ.get(
                f"{str(self.branch_name).upper()}_COLD_TO_HOT_EVERY", None
            )
        if val is None or (isinstance(val, str) and not val.strip()):
            val = COLD_TO_HOT_EVERY_DEFAULT
        return int(val)

    def fanout_prepare_state(self, model, state):
        """The mixin seam: runs on the full ensemble before any slicing.

        The pooled adaptation sample is taken FIRST (block head only,
        :meth:`_note_eigen_pool_sample`): the rotation below edits a hot rung.
        The cross-walker jump's partner pool (:meth:`_note_de_partner_sample`,
        every layout, the same arrays when both are on) likewise. The
        SNR-keyed ladder update (:meth:`_maybe_update_snr_ladders`) runs LAST,
        so a rebuilt ladder applies from this propose on and never inside
        one.
        """
        self._note_eigen_pool_sample(state)
        self._note_de_partner_sample(
            state, reuse=getattr(self, "_eigen_pool_outgoing", None))
        self._maybe_rotate_cold_into_hot(state)
        # LAST: the samples above were drawn at the ladders they carry; a
        # rebuilt ladder applies from this propose on ({BRANCH}_LADDER_TOP_SNR;
        # a no-op with the knob off)
        self._maybe_update_snr_ladders(state)

    def _maybe_rotate_cold_into_hot(self, state) -> bool:
        """Every ``cold_to_hot_every`` proposals in a SEARCH stage, copy each
        walker's cold chain into the hottest sampled rung of another walker.

        User ruling 2026-10-14 (the 1yr flagship's EMRI 0, held by walker 1
        at SNR 28.7 and lost by walkers 0, 2 and 3 at SNR 4-8): "during all
        search parts of the sampler, when source search is run (EMRI, MBHB,
        SOBHB), we should be copying the cold-chain into the hot chain
        rotating by walker so it does not pass to itself ... every 200
        proposals for a source (which includes the 25 repeats so 8
        iterations)", then "every 5 iterations during the search stages
        rather than 8" (default 125 proposals = 5 proposes of 25 repeats;
        in the gb_search stages the source moves propose once per CYCLE, so
        that is every 5 cycles there and every 5 rounds in source_search).
        A cross-walker exchange through the ladder: walker
        ``w``'s cold point lands in the hottest rung the move samples
        (``self.ntemps - 1``) of walker ``w + k``, ``k = 1 + event mod
        (nwalkers - 1)`` (never itself; every other walker in turn), for
        every alive leaf of the branch at once. At the next leaf visit the
        copied rung is scored against ITS walker's exposed residual like
        every other rung: a point that beats the walker's own is swapped
        down the ladder by the per-repeat temperature swaps; one that sits
        far below is re-anchored by the lost-rung guard. Nothing else moves
        (cold rung, residual, ladder), so no replay is needed; the edit is
        made here, on the full ensemble, before the state is sliced per
        rank.

        Fires only when ``gf_stage_kind`` is a search kind (``"search"``,
        ``"rj"``: source_search, gb_search_seed/1/2), never in PE and never
        when the stage is unknown; needs two or more walkers and two or more
        sampled rungs. Not reversible (no detailed balance; search only).
        The clock is in-memory: a resume starts it from 0. Logged once per
        event (``[<BRANCH>] cold->hot rotate ...``); counted in
        ``n_cold_to_hot_events``.
        """
        # counters live here too (a subclass may bypass __init__)
        self.n_cold_to_hot_events = int(getattr(self, "n_cold_to_hot_events", 0))
        cadence = self._cold_to_hot_every()
        kind = getattr(self, "gf_stage_kind", None)
        if cadence <= 0 or kind not in COLD_TO_HOT_STAGE_KINDS:
            return False
        fires, self._cold_to_hot_clock = cold_to_hot_due(
            getattr(self, "_cold_to_hot_clock", 0), int(self.num_repeats), cadence
        )
        if not fires:
            return False
        sub = (getattr(state, "sub_states", None) or {}).get(self.branch_name)
        if sub is None or not getattr(sub, "tempered_initialized", False):
            return False
        coords = sub.coords
        nw = int(coords.shape[1])
        hot = min(int(self.ntemps), int(coords.shape[0])) - 1
        event = int(getattr(self, "n_cold_to_hot_events", 0))
        offset = cold_to_hot_offset(event, nw)
        if hot < 1 or offset == 0:
            return False
        n = rotate_cold_into_hot(
            coords, sub.inds, getattr(sub, "log_prior", None), hot=hot, offset=offset
        )
        self.n_cold_to_hot_events = event + 1
        seen = getattr(self, "_cold_to_hot_offsets_seen", None)
        if seen is None:
            seen = self._cold_to_hot_offsets_seen = []
        seen.append(int(offset))
        logger.info(
            "[%s] cold->hot rotate event %d (stage kind %s, every %d proposals): "
            "offset %d over %d walkers -- %d (walker, leaf) cold points copied into "
            "rung %d (the hottest sampled); scored at the next leaf visit, swapped "
            "down if better, re-anchored if lost.",
            str(self.branch_name).upper(), event + 1, kind, cadence, offset, nw, n, hot,
        )
        return True

    # ---- SNR-keyed per-leaf ladders (user ruling 2026-10-18) ------------------
    #: what the builder resolved (``recipe.resolve_snr_keyed_ladders``):
    #: ``{"ref": {leaf: SNR the leaf's ladder was built at (or its first
    #: measured SNR after a plain resume)}, "pending": {leaves whose SNR was
    #: unknown at construction}}``. ``None`` = nothing resolved (a move built
    #: outside the builder): every alive leaf is then pending.
    snr_ladder_plan = None

    def _ladder_top_snr(self) -> float:
        """kwarg > ``{BRANCH}_LADDER_TOP_SNR`` env > 0 (off); on/true/yes = 2.0."""
        val = getattr(self, "ladder_top_snr", None)
        if val is None:
            val = os.environ.get(f"{str(self.branch_name).upper()}_LADDER_TOP_SNR", None)
        return top_snr_value(val)

    def _ladder_tmax_min(self) -> float:
        """kwarg > ``{BRANCH}_LADDER_TMAX_MIN`` env > 4 (the floor of a leaf's T_max)."""
        return self._eigen_float_knob("ladder_tmax_min", "LADDER_TMAX_MIN",
                                      LADDER_TMAX_MIN_DEFAULT)

    def _ladder_tmax_ceiling(self, ntemps) -> float:
        """kwarg > ``{BRANCH}_LADDER_TMAX`` env > the uncapped make_ladder top."""
        val = getattr(self, "ladder_tmax", None)
        if val is None:
            env = os.environ.get(f"{str(self.branch_name).upper()}_LADDER_TMAX", None)
            val = None if (env is None or not env.strip()) else float(env)
        return ceiling_tmax(self.ndim, ntemps, val)

    def _ladder_top_snr_refresh(self) -> int:
        """kwarg > ``{BRANCH}_LADDER_TOP_SNR_REFRESH`` env > 0 (never), in proposes."""
        val = getattr(self, "ladder_top_snr_refresh", None)
        if val is None:
            val = os.environ.get(
                f"{str(self.branch_name).upper()}_LADDER_TOP_SNR_REFRESH", None)
        if val is None or (isinstance(val, str) and not val.strip()):
            return 0
        return max(0, int(val))

    def _ladder_top_snr_refresh_pe(self) -> bool:
        """kwarg > ``{BRANCH}_LADDER_TOP_SNR_REFRESH_PE`` env > off."""
        return self._eigen_switch("ladder_top_snr_refresh_pe", "LADDER_TOP_SNR_REFRESH_PE")

    def _maybe_update_snr_ladders(self, state):
        """Build / refresh SNR-keyed per-leaf ladders BETWEEN proposes (head side).

        User ruling 2026-10-18: with ``{BRANCH}_LADDER_TOP_SNR`` > 0 each
        leaf's ladder is ``make_ladder(ndim, ntemps, Tmax=T_leaf)``, ``T_leaf
        = clip((SNR_leaf / top)^2, {BRANCH}_LADDER_TMAX_MIN,
        {BRANCH}_LADDER_TMAX)`` (:mod:`lisatools.sampling.snrladder`).
        ``SNR_leaf`` is ``sqrt(<h|h>)`` at the max-lnL cold walker, read off
        the sub-state's per-leaf inner products (``d_h`` / ``h_h``, recorded
        at the end of every leaf visit; under the walker-block fan-out the
        head's merged state carries every walker's).

        Runs from :meth:`fanout_prepare_state`, on the process that owns the
        whole ensemble (single process, the flat head, the block head before
        it slices), so a ladder only ever changes BETWEEN proposes: inside a
        propose the swaps, the lost-rung guard and the tempered eigen steps
        use the visit's ladder, exactly as before. The new row is written to
        ``sub.betas_all[leaf]`` of the state being proposed from: the visit
        reads it, the block slices carry it by value, the head's pooled
        adaptation starts from it, and the saver persists it with the row
        (``PerLeafLadderState.storage_arrays``). The rung STATES are left
        where they are; a rung the new temperatures orphan is re-anchored by
        the lost-rung guard at its next visit (the reset-on-resume path's
        behaviour).

        Per alive leaf:

        * PENDING (its SNR was unknown when the ladder was constructed: a
          fresh store, or a reset store that never recorded ``<h|h>``): the
          ladder is built at the first propose that finds a recorded SNR.
          This is the construction itself, deferred, so it happens once per
          leaf in any stage;
        * no baseline (a plain resume kept the stored ladder): the first
          measured SNR becomes the refresh baseline, nothing is rebuilt;
        * REFRESH (``{BRANCH}_LADDER_TOP_SNR_REFRESH`` = N > 0, every N
          proposes of the move, i.e. of every alive leaf): a leaf whose SNR
          moved more than 20 % from the SNR its ladder was built at gets
          its ladder rebuilt (a copy's SNR jump, a lost source). An ADAPTIVE
          change -- the kernel is not detailed-balance exact while it
          happens -- so it runs in the search stages only (``gf_stage_kind``
          search / rj) unless ``{BRANCH}_LADDER_TOP_SNR_REFRESH_PE=1``
          (Mike's rule: no adaptive tricks in PE unless ruled); an unknown
          stage never refreshes. A rebuild whose ``T_leaf`` equals the
          current top (both at the clip) only moves the baseline.

        Off (the default): returns at once, no state touched, no RNG drawn.
        Returns ``[(leaf, snr, T_leaf), ...]`` of the rebuilt ladders.
        """
        top = self._ladder_top_snr()
        if not top > 0.0:
            return []
        sub = (getattr(state, "sub_states", None) or {}).get(self.branch_name)
        if sub is None or not getattr(sub, "tempered_initialized", False):
            return []
        betas_all = getattr(sub, "betas_all", None)
        if betas_all is None or np.ndim(betas_all) != 2:
            return []
        nt = int(np.shape(betas_all)[-1])
        nl = min(int(self.nleaves_max), int(np.shape(betas_all)[0]))
        if nt < 2:
            return []
        lo, hi = check_clip(self._ladder_tmax_min(), self._ladder_tmax_ceiling(nt))
        plan = getattr(self, "snr_ladder_plan", None)
        if plan is None:
            plan = self.snr_ladder_plan = {"ref": {}, "pending": set(range(nl))}
        ref = plan.setdefault("ref", {})
        pending = plan.setdefault("pending", set())
        self._snr_ladder_clock = int(getattr(self, "_snr_ladder_clock", 0)) + 1
        every = self._ladder_top_snr_refresh()
        kind = getattr(self, "gf_stage_kind", None)
        stage_ok = kind in COLD_TO_HOT_STAGE_KINDS or (
            kind == "pe" and self._ladder_top_snr_refresh_pe())
        due = every > 0 and self._snr_ladder_clock % every == 0 and stage_ok
        measured = substate_leaf_snrs(sub, nl)
        tag = f"[{str(self.branch_name).upper()}_LADDER]"
        inds = getattr(sub, "inds", None)
        changes, unmeasured = [], []
        for leaf in range(nl):
            if not leaf_alive(inds, leaf):
                continue
            snr, walker, skind = measured[leaf]
            finite = snr is not None and bool(np.isfinite(snr))
            built = ref.get(leaf)
            if leaf in pending:
                if not finite:
                    unmeasured.append(leaf)
                    continue
                why = "construction, deferred to the leaf's first recorded SNR"
            elif built is None:
                if finite:
                    ref[leaf] = float(snr)
                    logger.info(
                        "%s leaf %d: refresh baseline SNR %.1f (the stored ladder is kept, "
                        "T_max %.1f)", tag, leaf, float(snr),
                        float(1.0 / np.asarray(betas_all[leaf], dtype=float)[-1]))
                continue
            elif due and finite and snr_changed(snr, built):
                why = (f"refresh: SNR {float(built):.1f} at the last build, "
                       f"{100.0 * (float(snr) / float(built) - 1.0):+.0f} %, stage kind {kind}")
            else:
                continue
            t_leaf, raw, clip = snr_tmax(snr, top, lo, hi)
            row = np.asarray(betas_all[leaf], dtype=float)
            current = float(1.0 / row[-1]) if row[-1] > 0.0 else float("inf")
            if leaf not in pending and np.isclose(t_leaf, current, rtol=1e-9, atol=0.0):
                ref[leaf] = float(snr)
                logger.info(
                    "%s leaf %d: SNR %.1f -> %.1f (%+.0f %%) but T_max stays %.1f (%s); "
                    "ladder kept, baseline moved", tag, leaf, float(built), float(snr),
                    100.0 * (float(snr) / float(built) - 1.0), t_leaf,
                    "clipped" if clip else "unchanged")
                continue
            new = snr_keyed_ladder(self.ndim, nt, t_leaf)
            sub.betas_all[leaf][:nt] = new
            tcs = getattr(self, "temperature_controls", None) or []
            if leaf < len(tcs) and tcs[leaf] is not None:
                try:
                    if np.size(tcs[leaf].betas) == nt:
                        tcs[leaf].betas[:] = new
                except Exception:  # noqa: BLE001 -- the visit reloads it from betas_all
                    pass
            ref[leaf] = float(snr)
            pending.discard(leaf)
            changes.append((leaf, float(snr), float(t_leaf)))
            logger.info(describe_leaf(tag, leaf, snr, t_leaf, raw, clip, top, lo, hi, new, why,
                                      kind=skind, walker=walker))
        if unmeasured and self._snr_ladder_clock >= 2 and not getattr(
                self, "_snr_ladder_warned", False):
            self._snr_ladder_warned = True
            logger.warning(
                "%s %s_LADDER_TOP_SNR=%g: leaves %s have no recorded <h|h> yet (is "
                "%s_RECORD_DH on?); they keep their configured ladder until one is recorded",
                tag, str(self.branch_name).upper(), top, self._int_runs(unmeasured),
                str(self.branch_name).upper())
        return changes

    def _fancy_swap_fires(self, repeat):
        """Walker-permuting swap cadence; never with one walker (identity permutation)."""
        return (
            self.permute_every > 0
            and repeat == self.num_repeats - 1
            and self._fancy_swap_clock % self.permute_every == 0
            and int(self.nwalkers) > 1
        )

    #: Default for the per-leaf <d|h>/<h|h> record ({BRANCH}_RECORD_DH
    #: overrides): OFF on the slow container path (one extra nwalkers
    #: waveform batch per leaf); fast-kernel subclasses set "1".
    _record_dh_default = "0"
    #: ``{BRANCH}_CHECK_LL_EVERY`` when the env var is unset (see _setup_debug)
    _check_ll_every_default = "1"

    def _record_leaf_inner_products(self, new_state, add_coords_in, leaf):
        """Record this leaf's cold-chain ``<d|h>``, ``<h|h>`` on the sub-state.

        Called once per leaf at the end of the in-model block, just before
        the accepted cold-chain sources are folded back into the residual
        (the likelihood setup for this leaf is still bound). The base
        implementation pays one extra ``nwalkers``-row evaluation through
        :meth:`compute_acs_like` so each container stashes
        ``last_d_h``/``last_h_h`` (the DCGA kernel path records nothing) —
        which is why it is gated OFF by default (``{BRANCH}_RECORD_DH=1``
        arms it). Fast-likelihood subclasses override this to record from
        their own kernel's inner-product outputs instead (cheap; default
        ON there).

        Args:
            new_state: The move's working :class:`GFState` (its sub-state
                receives the values).
            add_coords_in: Waveform-basis cold-chain coords for this leaf,
                shape ``(nwalkers, ndim)``.
            leaf: Leaf index being recorded.
        """
        if not getattr(self, "record_inner_products", False):
            return
        _sub = (getattr(new_state, "sub_states", None) or {}).get(self.branch_name)
        if _sub is None or getattr(_sub, "d_h", None) is None:
            return
        self.compute_acs_like(
            add_coords_in,
            np.arange(self.nwalkers, dtype=np.int32),
            **self.waveform_like_kwargs,
        )
        for _w in range(self.nwalkers):
            _d_h_w = getattr(self.acs[_w], "last_d_h", None)
            _h_h_w = getattr(self.acs[_w], "last_h_h", None)
            if _d_h_w is not None and _h_h_w is not None:
                _sub.d_h[_w, leaf] = float(np.real(_d_h_w))
                _sub.h_h[_w, leaf] = float(np.real(_h_h_w))

    # ---- lost-rung guard (user ruling 2026-10-13) -----------------------------
    #: per-propose record of the guard, ``{leaf: [(global walker, rung,
    #: deficit nats), ...]}``; reset at the top of every propose_local
    _ladder_reanchor_log = None

    def _ladder_reanchor_nats(self) -> float:
        """kwarg > ``{BRANCH}_LADDER_REANCHOR_NATS`` env > 200 nats (<= 0 = off)."""
        val = getattr(self, "ladder_reanchor_nats", None)
        if val is None:
            val = os.environ.get(
                f"{str(self.branch_name).upper()}_LADDER_REANCHOR_NATS", None
            )
        if val is None or (isinstance(val, str) and not val.strip()):
            val = 200.0
        return float(val)

    def _walker_offset(self) -> int:
        """Global index of this body's walker 0.

        Under the walker-block fan-out every rank runs ``propose_local`` on
        its own slice ``[w0, w1)``, so local walker ``j`` is global walker
        ``w0 + j``. Single process and the flat (replica) body run the whole
        ensemble: offset 0.
        """
        if not getattr(self, "_fanout_body", False):
            return 0
        fanout = getattr(self, "fanout", None)
        if fanout is None or getattr(self, "row_fanout", None) is not None:
            return 0
        try:
            return int(fanout.layout.block_of(fanout.rank)[0])
        except Exception:  # noqa: BLE001 -- a log label must never raise
            return 0

    @staticmethod
    def _int_runs(values) -> str:
        """``[2, 3, 4, 7]`` -> ``"2-4,7"``."""
        vals = sorted(int(v) for v in values)
        out, i = [], 0
        while i < len(vals):
            j = i
            while j + 1 < len(vals) and vals[j + 1] == vals[j] + 1:
                j += 1
            out.append(f"{vals[i]}" if i == j else f"{vals[i]}-{vals[j]}")
            i = j + 1
        return ",".join(out)

    def _reanchor_lost_rungs(self, work, leaf, prev_logl, prev_logp, betas):
        """Re-anchor this leaf's LOST hot rungs to the cold state.

        Why (9mo v9, 2026-10-13): the loudest EMRI (leaf 3, SNR 49) had rungs
        2-7 ~1,000 nats below its cold rung on every walker since the first
        source_search block -- the hot rungs lost the source and never found
        it again (in-model acceptance 0.8-0.85 = wandering on a flat floor,
        swap acceptance between rungs 1 and 2 = 0.00), so the cold walkers
        sat on separate islands with no bridge. User ruling 2026-10-13: a
        rung more than ``THRESH`` nats below its OWN cold rung for that leaf
        is re-anchored to the cold state at the start of the leaf's block.

        Criterion, per walker ``w`` and hot rung ``k >= 1``::

            lnL[0, w] - lnL[k, w] > THRESH + (ndim / 2) * (1 / beta_k - 1)

        ``lnL`` is ``prev_logl``: the walker's total lnL with this leaf at
        that rung's parameters, scored against the walker's exposed residual
        at this visit -- the quantity the move's MH steps and swaps use (the
        sub-state ``log_like[leaf]`` is the end-of-visit copy of it). The
        second term is the rung's EXPECTED drop below the cold rung for an
        ``ndim``-dimensional Gaussian at temperature ``1 / beta_k`` -- the
        "expected tempering drop" the ruling measured the 200 nats against
        (<= 15.5 nats on the 8-rung 1/1.2^k ladder, where this reduces to
        the ruling as stated; ~700-800 at the top rung of the 8-rung
        make_ladder ladder, where a hot rung that far down is legitimate).
        ``beta_k == 0`` (an infinite-temperature rung) never fires. A
        cold rung at the out-of-domain sentinel (or non-finite) never
        anchors anything; a non-finite hot rung under a valid cold rung is
        re-anchored.

        What is copied, cold -> rung ``k`` of walker ``w`` (the set the
        temperature swap exchanges between rungs: coords, inds, lnL, log
        prior -- branch supplementals are ``None`` for these branches --
        plus this leaf's per-(temperature, walker) eigen table row when the
        ``per_walker`` scope is on): ``work.coords[k, w, leaf]``,
        ``work.inds[k, w, leaf]``, ``prev_logl[k, w]``, ``prev_logp[k, w]``.
        The tempered posterior is recomputed by the caller from the edited
        arrays, and the end-of-visit telemetry writes ``log_like`` /
        ``log_prior`` from them. The cold rung, the residual and the ladder
        are untouched (a within-walker, within-leaf edit), so nothing needs
        replaying across ranks: under the walker-block fan-out each rank
        edits its own slice and ``merge_walkers`` returns the columns.

        Not reversible (no detailed balance); fires only in the pathological
        state. Runs in search and PE stages alike. Threshold:
        ``{BRANCH}_LADDER_REANCHOR_NATS`` (default 200; 0 disables).

        Returns ``(prev_logl, prev_logp, records)``; ``records`` is a list of
        ``(global walker, rung, deficit)`` (empty when nothing fired, in
        which case the inputs are returned untouched).
        """
        thresh = self._ladder_reanchor_nats()
        nt = int(self.ntemps)
        if not thresh > 0.0 or nt < 2:
            return prev_logl, prev_logp, []
        ll = np.asarray(asnumpy(prev_logl), dtype=float)
        nt = min(nt, ll.shape[0])
        b = np.asarray(asnumpy(betas), dtype=float).ravel()[:nt]
        if nt < 2 or b.size < nt:
            return prev_logl, prev_logp, []
        cold = ll[0]
        valid_cold = np.isfinite(cold) & (cold > -1e299)
        hot = ll[1:nt]
        deficit = cold[None, :] - hot
        with np.errstate(divide="ignore", invalid="ignore"):
            margin = 0.5 * float(self.ndim) * (1.0 / b[1:nt] - 1.0)
        margin = np.where(np.isnan(margin), np.inf, np.maximum(margin, 0.0))
        lost = valid_cold[None, :] & (
            (deficit > thresh + margin[:, None]) | ~np.isfinite(hot)
        )
        if not np.any(lost):
            return prev_logl, prev_logp, []

        k_idx, w_idx = np.nonzero(lost)
        k_idx = k_idx + 1
        leaf = int(leaf)

        work.coords[k_idx, w_idx, leaf] = work.coords[0, w_idx, leaf]
        work.inds[k_idx, w_idx, leaf] = work.inds[0, w_idx, leaf]
        if not isinstance(prev_logl, np.ndarray):
            prev_logl = np.array(asnumpy(prev_logl), dtype=float)
        if not isinstance(prev_logp, np.ndarray):
            prev_logp = np.array(asnumpy(prev_logp), dtype=float)
        prev_logl[k_idx, w_idx] = prev_logl[0, w_idx]
        prev_logp[k_idx, w_idx] = prev_logp[0, w_idx]

        # per-(temperature, walker) eigen tables (per_walker scope): the
        # re-anchored row takes the cold row's table (built at the cold point)
        tabs = getattr(self, "_eigen_tables", None)
        if tabs and leaf in tabs:
            axes, sigmas = tabs[leaf]
            if getattr(axes, "ndim", 2) == 4:
                try:
                    xp_t = get_array_module(axes)
                    ki, wi = xp_t.asarray(k_idx), xp_t.asarray(w_idx)
                    axes[ki, wi] = axes[0, wi]
                    sigmas[ki, wi] = sigmas[0, wi]
                    # the adaptation's information-matrix floor follows
                    floor = getattr(self, "_eigen_fisher_sigmas", {}).get(leaf)
                    if floor is not None and np.ndim(floor) == 3:
                        floor[k_idx, w_idx] = floor[0, w_idx]
                except Exception as exc:  # noqa: BLE001 -- the next refresh rebuilds it
                    logger.warning(
                        "[%s_LADDER] leaf %d: per-walker eigen table rows not "
                        "re-anchored (%r); the next table refresh rebuilds them",
                        str(self.branch_name).upper(), leaf, exc,
                    )

        w0 = self._walker_offset()
        defs = deficit[k_idx - 1, w_idx]
        records = [
            (int(w) + w0, int(k), float(d)) for k, w, d in zip(k_idx, w_idx, defs)
        ]
        log = getattr(self, "_ladder_reanchor_log", None)
        if log is None:
            log = self._ladder_reanchor_log = {}
        log.setdefault(leaf, []).extend(records)

        by_walker = {}
        for w, k, _ in records:
            by_walker.setdefault(w, []).append(k)
        where = "; ".join(
            f"walker {w} rung{'s' if len(ks) > 1 else ''} {self._int_runs(ks)}"
            for w, ks in sorted(by_walker.items())
        )
        finite = defs[np.isfinite(defs)]
        span = (
            f"{finite.min():.0f}-{finite.max():.0f} nats below cold"
            if finite.size else "non-finite lnL"
        )
        m_used = margin[np.unique(k_idx - 1)]
        logger.info(
            "[%s_LADDER] leaf %d: re-anchored %d rung(s) to the cold state "
            "(%s, %s; threshold %.0f + expected tempering drop %.1f-%.1f nats)",
            str(self.branch_name).upper(), leaf, len(records), where, span,
            thresh, float(np.min(m_used)), float(np.max(m_used)),
        )
        return prev_logl, prev_logp, records

    def setup(self, model, state):
        """Per-iteration setup hook (no-op by default)."""
        return

    def _verify_entry_vs_acs(self, prev_logl, cold_ref, leaf):
        """The expose invariant (2026-07-28): the cold-chain ``prev_logl``
        scored against the freshly EXPOSED residual must reproduce the full
        ACS likelihoods from just before the expose — algebraically
        ``r_exposed - h_current == r_full``, so the two are the same number.
        A large per-walker SPREAD in the difference means the expose/fold
        choreography (or the scoring path) is inconsistent with the residual
        state — this is the check that catches the d-2h expose-sign class of
        bug that corrupted all add/remove sampling from 6d33265 (2026-05-22)
        until today. A constant offset with ~zero spread is a benign
        normalization convention. Shares {BRANCH}_CHECK_LL gating/severity.
        """
        if cold_ref is None:
            return
        cold = np.asarray(prev_logl[0], dtype=float)
        ref = np.asarray(cold_ref, dtype=float).reshape(cold.shape)
        both = (
            np.isfinite(cold) & np.isfinite(ref)
            & (cold > -1e299) & (ref > -1e299)
        )
        if not np.any(both):
            return
        diff = cold[both] - ref[both]
        spread = float(diff.max() - diff.min())
        med = float(np.median(diff))
        if spread <= 1e-1 and abs(med) <= 5.0:
            return
        msg = (
            f"{self.branch_name} leaf {leaf}: EXPOSE INVARIANT VIOLATED — "
            f"cold prev_logl vs pre-expose ACS lnL: median offset "
            f"{med:.6e}, spread {spread:.6e} over {int(both.sum())} walkers. "
            "Large spread/offset means the exposed residual or scoring path "
            "is wrong (e.g. template subtracted instead of added at expose)."
        )
        if self.check_ll_mode == "strict":
            raise ValueError(msg)
        logger.warning(msg)
        if DEBUG_MODE:
            breakpoint()

    def _verify_prev_logl(self, prev_logl, old_coords_in, data_index_in, leaf):
        """Cross-check the move's scoring path against the ACS container path.

        ``prev_logl`` came from :meth:`compute_like` — the DCGA (multi-device)
        path when active, else the container path with the engine-installed
        generator and ``waveform_like_kwargs``. This recomputes the SAME
        current-state points through :meth:`compute_acs_like` with the move's
        OWN generator and the move's own ``waveform_like_kwargs`` VERBATIM —
        the same convention ``compute_like`` applies. It must NEVER force
        ``source_only=True``: on runs with a sampled psd branch the full lnL
        carries a walker-indexed noise term (~1e8 here), so a forced
        source-only recompute differs by exactly that per-walker constant and
        the old pooled-spread gate read the WALKER-TO-WALKER noise variation
        (~2e6) as "drift" — 8 false warnings per iteration on the healthy 6mo
        job 487, verified benign 2026-09-14 (the convention-matched
        ``_verify_entry_vs_acs`` stayed silent at a 0.1 gate; SOBBH's
        override predates this fix for the same reason).

        The gate is the WITHIN-walker spread — the difference varying across
        temperatures of ONE walker. Real residual/scoring bookkeeping drift
        (the failure mode that silently walks a chain away from truth) shows
        up there; ANY per-walker-constant convention offset (e.g. the DCGA
        replica path's source-only lnL) cancels out of it identically.
        Default mode warns loudly; ``{BRANCH}_CHECK_LL=strict`` raises; ``0``
        disables; ``{BRANCH}_CHECK_LL_EVERY=N`` thins the cost.
        """
        acs_like = (
            self.compute_check_like(old_coords_in, data_index_in)
            .reshape(prev_logl.shape)
            .real
        )
        both = (
            np.isfinite(prev_logl)
            & np.isfinite(acs_like)
            & (prev_logl > -1e299)
            & (acs_like > -1e299)
        )
        if not np.any(both):
            return
        diff_full = np.where(both, prev_logl - acs_like, np.nan)
        with warnings.catch_warnings():
            # all-NaN walker columns are fine (fully invalid walkers)
            warnings.simplefilter("ignore", RuntimeWarning)
            within = np.nanmax(diff_full, axis=0) - np.nanmin(diff_full, axis=0)
            within_max = float(np.nanmax(np.nan_to_num(within, nan=0.0)))
        max_abs = float(np.abs(diff_full[both]).max())
        if within_max <= 1e-1:
            # No within-walker drift. A residual constant offset per walker
            # is a convention difference (source-only replica paths), not a
            # bookkeeping bug — record it quietly for forensics only.
            if max_abs > 5.0:
                logger.debug(
                    "%s leaf %s: prev_logl vs ACS differ by a per-walker "
                    "CONSTANT (max|diff|=%.3e, within-walker spread %.3e) — "
                    "convention offset, not drift.",
                    self.branch_name, leaf, max_abs, within_max,
                )
            return
        msg = (
            f"{self.branch_name} leaf {leaf}: prev_logl (move scoring path) vs "
            f"ACS container path show WITHIN-WALKER drift: max per-walker "
            f"spread={within_max:.6e} across temperatures "
            f"(max|diff|={max_abs:.6e} over {int(both.sum())} points). This is "
            "residual/scoring bookkeeping drift (real bug) — per-walker "
            "constant conventions cannot produce it."
        )
        if self.check_ll_mode == "strict":
            raise ValueError(msg)
        logger.warning(msg)
        if DEBUG_MODE:
            breakpoint()

    #: Leaf currently being processed by the per-leaf proposal loop. The
    #: transform helper below reads it because some transform call sites
    #: (the tempering likelihood callback) have an eryn-fixed signature and
    #: cannot receive leaf indices as arguments.
    _current_leaf: Optional[int] = None

    def _to_phys(self, coords):
        """Sampling -> physical rows through the branch transform.

        For containers with PER-LEAF fills (Eryn per-leaf ``fill_dict``,
        e.g. the EMRI xI0/Phi_theta0 prograde-retrograde flags), every row
        handled by this move belongs to the leaf currently being processed
        (``self._current_leaf``); scalar-fill containers (SOBBH/MBH) take
        the plain path.
        """
        if getattr(self.transform_fn, "n_leaf_fills", None) is not None:
            assert self._current_leaf is not None, (
                "per-leaf transform used outside the per-leaf loop"
            )
            leaf_inds = np.full(
                np.shape(coords)[:-1], int(self._current_leaf), dtype=int
            )
            return self.transform_fn.both_transforms(coords, leaf_inds=leaf_inds)
        return self.transform_fn.both_transforms(coords)

    def log_like_for_fancy_swaping(self, x, supps=None, branch_supps=None, **kwargs):
        """
        Compute the log likelihood for the given coordinates and data index for use in fancy swapping.
        This is needed because when permuting the coordinates during tempering, we need to recompute the likelihood against the new set of residuals and covariance matrix.

        Args:
            x: Dictionary of coordinates of the sources for which we want to compute the likelihood.
                The coordinates are expected to be in the shape (ntemps, nwalkers, nleaves_max, ndim).
            supps: supplimental information for the likelihood computation. #todo add
            branch_supps: Branch supplimental. #todo add

        Returns:
            ll: likelihood for the given coordinates and data index. Shape is (ntemps, nwalkers).
            blobs: blobs for the given coordinates and data index. Default is None.
        """
        assert x[self.branch_name].ndim == 4 and x[self.branch_name].shape[1] == self.nwalkers
        # Single-leaf slices only: rows are scored per walker, and with
        # per-leaf transform fills the fixed ``_current_leaf`` identity must
        # apply to every row (a multi-leaf slice would silently apply one
        # leaf's fills to all leaves).
        assert x[self.branch_name].shape[2] == 1, (
            "log_like_for_fancy_swaping expects a single-leaf coordinate slice"
        )
        # shape is (nwalkers, 1 (nleaves_max), ndim)
        ntemps = x[self.branch_name].shape[0]

        coords = x[self.branch_name].reshape(-1, x[self.branch_name].shape[-1])
        data_index_in = np.tile(np.arange(self.nwalkers), (ntemps, 1)).flatten().astype(np.int32)

        coords_in = self._to_phys(coords)

        # TODO: need to be careful here when heterodyning about if it is "close"
        output = (
            self.compute_like(
                coords_in,
                data_index=data_index_in,
            )
            .reshape((ntemps, self.nwalkers))
            .real
        )
        return output, None  # AS: match psd? I'm not sure

    def get_split_inds(self):
        all_inds = np.tile(np.arange(self.nwalkers), (self.ntemps, 1))
        inds = all_inds % self.nsplits
        if self.randomize_split:
            [np.random.shuffle(x) for x in inds]

        return inds

    def _repeat_split_masks(self, move_here):
        """Ensemble partition for ONE in-model repeat: a list of row masks.

        ``len(masks)`` IS the number of :meth:`compute_like` calls that
        repeat will make — each mask is scored in one batch.

        :class:`~eryn.moves.StretchMove` KEEPS the red/blue split: its
        proposal stretches each walker toward a walker drawn from the
        COMPLEMENTARY sub-ensemble, so the halves are load-bearing for its
        detailed balance (updating a walker that another walker is
        simultaneously being stretched against breaks it). That path is
        untouched — ``self.nsplits`` masks from :meth:`get_split_inds`,
        including any subclass override of the shuffle (MBH pins equal
        per-GPU block sizes there).

        Every other inner move in this stack is MH-style — the production
        default is :class:`~eryn.moves.EigenAxisMove`, which proposes from
        each row's own coords and a table frozen for the whole repeat sweep.
        Such a proposal reads NOTHING from the other rows, and the move does
        not mutate the residual during the repeat loop, so every
        ``(temp, walker)`` row is an independent MH step against a target
        that the other rows cannot change. Splitting it only cuts the
        scoring batch in half and pays the per-call kernel overhead twice
        (job 508: 2 calls/repeat at ~54 rows each, cost dominated by the
        row-independent part of the call). One full-ensemble mask = one
        call per repeat, the identical Markov kernel.

        ESCAPE (2026-09-16): ``SOBBH_SINGLE_CALL=0`` forces the red/blue
        halves back on for EVERY inner move, restoring the pre-483953fa
        two-calls-per-repeat structure without a revert. It exists so the
        cluster can bisect batch-size effects (a full ensemble is ~120-129
        rows against the old ~60) against a suspected kernel-side fault.
        Default ``1`` = the single full-ensemble call.
        """
        if (isinstance(move_here, StretchMove)
                or os.environ.get("SOBBH_SINGLE_CALL", "1").strip() != "1"):
            inds = self.get_split_inds()
            return [inds == split for split in range(self.nsplits)]
        return [np.ones((self.ntemps, self.nwalkers), dtype=bool)]

    # ---- multi-rank fan-out (WalkerFanoutMixin hooks) ------------------------
    #: per-propose {leaf: (sum swaps_accepted, sum swaps_proposed)} over repeats;
    #: reset at the top of every propose_local, read by fanout_reply_extra
    _fanout_swap_tally = None

    def _fanout_note_swaps(self, leaf, tc):
        """Accumulate this repeat's swap counts for ``leaf`` (Eryn re-zeroes them per call)."""
        tally = getattr(self, "_fanout_swap_tally", None)
        if tally is None:
            tally = self._fanout_swap_tally = {}
        acc = np.asarray(tc.swaps_accepted, dtype=float).ravel()
        prop = np.asarray(tc.swaps_proposed, dtype=float).ravel()
        a, p = tally.get(int(leaf), (0.0, 0.0))
        tally[int(leaf)] = (a + acc, p + prop)

    def _record_leaf_telemetry(self, new_state, leaf, proposed, accepted,
                               prev_logl, prev_logp):
        """Fill this leaf visit's acceptance counters and lnL on the sub-state.

        ``PerLeafLadderState`` has declared ``in_model_proposed`` /
        ``in_model_accepted``, the per-rung-pair swap counters and the
        per-leaf ``log_like`` / ``log_prior`` from the start, but NOTHING
        ever wrote them for the addremove branches, so every MBH / EMRI /
        SOBBH store on disk records zeros: the three branches have no
        acceptance telemetry at all (found 2026-09-17 in the 4-GPU store --
        53 iterations, every counter exactly zero, while the same fields are
        populated for gb / vgb / psd / galfor). PSD fills its equivalents
        from ``_tally_*``; this is the addremove side of the same contract.

        Semantics, chosen to match what the storage layer already assumes:

        * ``in_model_proposed[leaf, rung]`` counts ONE proposal per walker
          per repeat. Every walker is proposed for exactly once per repeat
          (the red/blue split masks partition the ensemble), and an
          out-of-prior draw still counts as a proposal, so the denominator
          is ``num_repeats * nwalkers`` and the ratio is the true MH
          acceptance rate of the inner move.
        * ``swaps_*[leaf, pair]`` is the per-repeat sum over the visit --
          Eryn re-zeroes its control's counters on every
          ``temperature_swaps`` call, which is why ``_fanout_note_swaps``
          accumulates them.
        * ``log_like`` / ``log_prior`` are the POST-swap per-leaf values for
          this leaf, one row per (rung, walker).

        Counters are DELTAS. A rank's slice starts at zero and
        :meth:`ModuleSubState.merge_walkers` SUM-adds them across walker
        blocks, so the head ends the propose with ensemble-wide totals, and
        ``reset_delta_counters`` zeroes them after each save. ``log_like``
        and ``log_prior`` carry a walker axis and are merged by column, so a
        rank writes only its own block's rows.
        """
        sub = (getattr(new_state, "sub_states", None) or {}).get(self.branch_name)
        if sub is None or not getattr(sub, "tempered_initialized", False):
            return
        leaf = int(leaf)
        nt = min(int(self.ntemps), int(sub.ntemps))

        for name, src in (("in_model_proposed", proposed),
                          ("in_model_accepted", accepted)):
            dst = getattr(sub, name, None)
            if dst is not None:
                dst[leaf, :nt] += np.asarray(src, dtype=dst.dtype)[:nt]

        tally = (getattr(self, "_fanout_swap_tally", None) or {}).get(leaf)
        if tally is not None:
            for name, src in (("swaps_accepted", tally[0]),
                              ("swaps_proposed", tally[1])):
                dst = getattr(sub, name, None)
                src = np.asarray(src, dtype=float).ravel()
                if dst is None or not src.size:
                    continue
                n = min(src.size, dst.shape[-1])
                dst[leaf, :n] += src[:n].astype(dst.dtype)

        for name, src in (("log_like", prev_logl), ("log_prior", prev_logp)):
            dst = getattr(sub, name, None)
            if dst is None:
                continue
            src = np.asarray(src, dtype=float)
            dst[leaf, :nt, :] = src[:nt, : dst.shape[-1]]

    def fanout_temperature_controls(self):
        return list(self.temperature_controls)

    def fanout_payload_extra(self):
        extra = {
            "fancy_swap_clock": int(getattr(self, "_fancy_swap_clock", 0)),
            "dbg_step": int(getattr(self, "_dbg_step", 0)),
        }
        # {BRANCH}_EIGEN_ADAPT_POOL: the full ensemble's rung coordinates,
        # taken by fanout_prepare_state this propose (absent otherwise, so
        # the payload is unchanged with the switch off)
        pool = getattr(self, "_eigen_pool_outgoing", None)
        if pool:
            extra["eigen_pool"] = pool
            self._eigen_pool_outgoing = None
            x0 = next(iter(pool.values()))[0]
            logger.debug(
                "[%s_EIGEN] propose payload: pooled adaptation sample, %d leaves x "
                "%d rungs x %d walkers (%d floats)", str(self.branch_name).upper(),
                len(pool), int(x0.shape[0]), int(x0.shape[1]),
                int(sum(np.size(x) + np.size(b) for x, b in pool.values())))
        # the cross-walker jump (a WalkerDifferenceJumpMove in the stack): the
        # start-of-propose rung ensembles of every walker + this propose's
        # group parity (absent otherwise: the payload is unchanged)
        de = getattr(self, "_de_pool_outgoing", None)
        if de:
            extra["de_pool"], extra["de_parity"] = de[0], int(de[1])
            self._de_pool_outgoing = None
        return extra

    def fanout_apply_extra(self, extra):
        if "fancy_swap_clock" in extra:
            self._fancy_swap_clock = int(extra["fancy_swap_clock"])
        if "dbg_step" in extra:
            self._dbg_step = int(extra["dbg_step"])
        if extra.get("eigen_pool"):
            # every rank, the head's own body included, buffers the union
            self._push_eigen_pool(extra["eigen_pool"])
        if "de_pool" in extra:
            # every rank, the head's own body included: the jump's partners
            self._set_de_pool(extra["de_pool"], extra.get("de_parity"))

    def fanout_reply_extra(self, part):
        return {
            "swap_tally": {
                int(leaf): (np.asarray(a, dtype=float), np.asarray(p, dtype=float))
                for leaf, (a, p) in (getattr(self, "_fanout_swap_tally", None) or {}).items()
            }
        }

    def fanout_merge_extra(self, replies, new_state):
        """Pool the per-leaf swap tallies over ranks; adapt each visited ladder ONCE.

        The authoritative per-leaf ladder lives on the sub-state
        (``betas_all[leaf]``, read at the top of the leaf loop, written back at
        its end); ``merge_walkers`` never touches ladders, so the head writes
        the adapted ladder there itself and mirrors it on its own control.
        """
        pooled = {}
        for extra in replies.values():
            for leaf, (acc, prop) in extra.get("swap_tally", {}).items():
                a, p = pooled.get(int(leaf), (0.0, 0.0))
                pooled[int(leaf)] = (
                    a + np.asarray(acc, dtype=float), p + np.asarray(prop, dtype=float)
                )
        sub = new_state.sub_states[self.branch_name]
        for leaf in sorted(pooled):
            acc, prop = pooled[leaf]
            tc = self.temperature_controls[leaf]
            ntemps = int(tc.ntemps)
            betas0 = np.array(sub.betas_all[leaf][:ntemps], copy=True)
            new = pooled_ladder_step(tc, betas0, acc, prop)
            sub.betas_all[leaf][:ntemps] = new
            tc.betas[:] = new
        # head-side per-propose checkpoint (the per-leaf hook is off in rank bodies);
        # the guard is defensive: merge only ever runs on the head after gf_serve
        if not self._fanout_body:
            midit_checkpoint.maybe_write(
                new_state, tag=f"{self.branch_name} propose", prepare=self._sync_cold_row
            )

    def propose_local(self, model, state):
        """Generate proposals by removing, updating, and re-adding each leaf.

        Runs on the FULL state single-process and on this rank's walker slice
        under several compute ranks (``WalkerFanoutMixin.propose`` dispatches).

        # TODO/DOCS: detailed acceptance/swap accounting; see code for the per-leaf
        loop and the ``compute_like`` calls used to score proposals.

        Args:
            model: ``eryn`` model object containing ``analysis_container_arr``,
                ``random``, etc.
            state: Current sampler state.

        Returns:
            Tuple ``(new_state, accepted)``.
        """
        self.setup(model, state)
        tic = time.time()
        # Fancy-swap cadence clock: one tick per propose (engine
        # iteration). In-memory; firings land on proposes N, 2N, ...
        # (user ruling 2026-08-27: iteration 1 -- the build/JIT-transient
        # iteration -- must NOT pay the swap).
        self._fancy_swap_clock = getattr(self, "_fancy_swap_clock", 0) + 1
        self._fanout_swap_tally = {}
        self._ladder_reanchor_log = {}

        # cold-row agreement between the main state and this branch's
        # sub-state (GF_SUBSTATE_CHECK=0 disables)
        self._check_substate_consistency(state, [self.branch_name])

        # The working ensemble is the SUB-STATE's tempered branch (module
        # ladder); the main state carries only the engine's cold chain, so
        # eryn-facing ``accepted`` arrays use the engine shape.
        work_in = self._work_branch(state)
        engine_ntemps = state.log_like.shape[0]

        if not np.any(work_in.inds):
            _accepted = np.zeros((engine_ntemps, self.nwalkers), dtype=bool)
            return state, _accepted

        new_state = deepcopy(state)
        # the copy's working branch: a view over new_state's sub-state arrays
        work = self._work_branch(new_state)
        # fresh per-iteration inner-product record (NaN = dead / not visited)
        _sub0 = (getattr(new_state, "sub_states", None) or {}).get(self.branch_name)
        if _sub0 is not None and getattr(_sub0, "d_h", None) is not None:
            _sub0.d_h[:] = np.nan
            _sub0.h_h[:] = np.nan

        # per-step debug reset (trace at most one leaf per propose call)
        self._dbg_step += 1
        self._dbg_leaf_done = False
        self._dbg_snaps = {}

        # self.acs = model.analysis_container_arr
        self.check_add_skip_swap_info(state)

        # mapping information
        temp_inds_base = np.repeat(np.arange(self.ntemps)[:, None], self.nwalkers, axis=-1)
        walker_inds_base = np.tile(np.arange(self.nwalkers), (self.ntemps, 1))

        # randomize order
        leaves_random_order = np.random.permutation(np.arange(self.nleaves_max))
        for leaf in leaves_random_order:
            # guard against leaves with False
            assert np.all(
                work_in.inds[0, 0, leaf] == work_in.inds[:, :, leaf]
            )
            if not work_in.inds[0, 0, leaf]:
                continue
            # leaf identity for the per-leaf transform fills (see _to_phys)
            self._current_leaf = int(leaf)
            # second step of randomizing order (making sure it does not run over)

            # fill this temperature control with temperatures from current state
            temperature_control_here = self.temperature_controls[leaf]

            temperature_control_here.betas[:] = new_state.sub_states[self.branch_name].betas_all[
                leaf
            ][
                : self.ntemps
            ]  # as: make sure only local ntemps are used
            ntemps_full = new_state.sub_states[self.branch_name].betas_all[leaf].shape[0]

            ndim = work.coords.shape[-1]

            # Debug: trace ONE leaf per step (the configured leaf, else the
            # first alive one). Snapshot the residual the sampler will score
            # against, before/after isolating this leaf, and plot it.
            _dbg_leaf = (
                self.debug
                and not self._dbg_leaf_done
                and self._dbg_step % self.debug_every == 0
                and (self.debug_plot_leaf is None or leaf == self.debug_plot_leaf)
            )
            _dbg_w = min(int(self.debug_plot_walker), self.nwalkers - 1)
            # Debug must never crash the sampler: every snapshot is guarded
            # and disarms this leaf's tracing on any failure.
            if _dbg_leaf:
                try:
                    self._dbg_snaps = {"before_removal": self._dbg_residual(_dbg_w)}
                    self._dbg_domain = type(self.acs[_dbg_w].data).__name__.replace(
                        "Signal", ""
                    ).lower()
                except Exception as exc:  # noqa: BLE001
                    logger.info("[%s_DEBUG] snapshot skipped: %r", self._dbg_prefix, exc)
                    _dbg_leaf = False

            # Cold-chain ACS reference BEFORE the expose: consumed by the
            # [STAGE] print and by the default-on expose invariant
            # (_verify_entry_vs_acs) — scoring the current coords against the
            # exposed residual must reproduce these values.
            _cold_ref = None
            if self.swap_debug or (
                self.check_ll_mode != "0"
                and self._dbg_step % self.check_ll_every == 0
            ):
                # ensemble-wide: this is compared against prev_logl, which
                # is at the body's state width (see ensemble_likelihood)
                _cold_ref = asnumpy(self.ensemble_likelihood(self.acs))
            if self.swap_debug:
                logger.info(
                    "[STAGE] %s leaf %d PRE-EXPOSE  ACS cold lnL min/med/max "
                    "%.3f / %.3f / %.3f",
                    self.branch_name, leaf,
                    float(np.min(_cold_ref)),
                    float(np.median(_cold_ref)),
                    float(np.max(_cold_ref)),
                )

            # remove cold chain sources FROM THE FIT: add their templates back
            # into the residual (r = d - h  ->  r = d), exposing this leaf.
            # NOTE 2026-07-28: propose() had these two calls SWAPPED since the
            # 6d33265 sign refactor (2026-05-22) — the expose step SUBTRACTED
            # the template (r -> d - 2h), so every proposal was scored against
            # a corrupted target (prev_logl ~ -2<d|d> at truth) and accepted
            # updates folded back against it degraded the true residual.
            removal_coords = work.coords[0, :, leaf]
            removal_coords_in = self._to_phys(removal_coords)
            self._replay_cold_chain("expose", removal_coords_in, leaf)

            if _dbg_leaf:
                # source-isolated residual = the "data" this source is fit
                # against (all other sources removed); the fixed reference for
                # the flip-book decomposition.
                try:
                    self._dbg_snaps["exposed"] = self._dbg_residual(_dbg_w)
                except Exception as exc:  # noqa: BLE001
                    logger.info("[%s_DEBUG] snapshot skipped: %r", self._dbg_prefix, exc)
                    _dbg_leaf = False

            self._replay_cold_chain("setup", removal_coords_in, leaf)

            # per-leaf eigen tables for EigenAxisMove inner moves (no-op for
            # stretch-only stacks); must run with the leaf exposed and
            # _current_leaf set. The visit-start ladder fixes the per-rung
            # tempered step ({BRANCH}_EIGEN_TEMPER_SIGMAS) for the visit.
            self.refresh_inner_move_tables(
                leaf, work, betas=temperature_control_here.betas)

            old_coords = (
                work.coords[: self.ntemps, :, leaf].reshape(-1, ndim)
            )
            old_coords_in = self._to_phys(old_coords)

            data_index_in = (
                np.tile(np.arange(self.nwalkers), (self.ntemps, 1)).flatten().astype(np.int32)
            )
            # TODO: fix this
            # prev_logl = self.waveform_gen.get_direct_ll(fd, data_residuals.flatten(), psd.flatten(), self.df, *old_coords_in.T, noise_index=noise_index, data_index=data_index, **self.waveform_kwargs).reshape((ntemps, nwalkers)).real.get()
            # TODO: check if psd term is included properly here at each step
            # TODO: and check data index here
            prev_logl = (
                self.compute_like(
                    old_coords_in,
                    data_index=data_index_in,
                )
                .reshape((self.ntemps, self.nwalkers))
                .real
            )

            if self.check_ll_mode != "0" and (
                self._dbg_step % self.check_ll_every == 0
            ):
                self._verify_prev_logl(prev_logl, old_coords_in, data_index_in, leaf)
                self._verify_entry_vs_acs(prev_logl, _cold_ref, leaf)

            if self.swap_debug:
                logger.info(
                    "[SWAP_DBG] %s leaf %d ENTRY cold prev_logl min/med/max "
                    "%.3f / %.3f / %.3f",
                    self.branch_name, leaf,
                    float(prev_logl[0].min()),
                    float(np.median(prev_logl[0])),
                    float(prev_logl[0].max()),
                )

            # -1e300 is the out-of-domain sentinel, not a numerical issue.
            if np.any((prev_logl < -1e10) & (prev_logl > -1e299)) or np.any(prev_logl > 1e30):
                logger.warning(f"Very low log likelihood encountered in propose: min = {prev_logl.min()}, max = {prev_logl.max()}. This could be a sign of numerical issues.")
                if DEBUG_MODE:
                    breakpoint()

            prev_logp = (
                self.priors[self.branch_name]
                .logpdf(old_coords)
                .reshape((self.ntemps, self.nwalkers))
            )

            # LOST-RUNG GUARD (user ruling 2026-10-13): a hot rung of this
            # leaf sitting far below the leaf's own cold rung is re-anchored
            # to the cold state BEFORE the in-model repeats (coords + the
            # per-rung bookkeeping the swaps carry). Search AND PE: this body
            # is the propose of every stage, and under the walker-block
            # fan-out each rank runs it on its own block (merge_walkers
            # returns the edited columns to the head).
            prev_logl, prev_logp, _ = self._reanchor_lost_rungs(
                work, leaf, prev_logl, prev_logp, temperature_control_here.betas
            )

            prev_logP = temperature_control_here.compute_log_posterior_tempered(
                prev_logl, prev_logp
            )

            # fix this need to compute prev_logl for all walkers
            _free_pool()
            # per-rung acceptance tally for THIS leaf visit, written to the
            # sub-state at the end of the visit (_record_leaf_telemetry).
            _leaf_proposed = np.zeros(self.ntemps, dtype=np.int64)
            _leaf_accepted = np.zeros(self.ntemps, dtype=np.int64)
            for repeat in tqdm(
                range(self.num_repeats), desc=f"{self.branch_name} update, leaf {leaf}",
                disable=not getattr(self, "progress", False),
            ):

                # pick move
                move_here = self.moves[
                    model.random.choice(np.arange(len(self.moves)), p=self.move_weights)
                ]

                # logger.debug(f"move here: {move_here.__class__.__name__}")

                # Partition the ensemble for this repeat: red/blue halves
                # for a stretch inner move (complementary ensembles are
                # load-bearing there), ONE full-ensemble block for an
                # MH-style one (see _repeat_split_masks). len(split_masks)
                # is the number of scoring calls this repeat makes.
                accepted = np.zeros((ntemps_full, self.nwalkers), dtype=bool)
                split_masks = self._repeat_split_masks(move_here)

                # prepare accepted fraction
                # accepted_here = np.zeros((self.ntemps, self.nwalkers), dtype=bool)
                for split, S1 in enumerate(split_masks):
                    # get split information
                    num_total_here = np.sum(S1)
                    nwalkers_here = np.sum(S1[0])

                    temp_inds_here = temp_inds_base[S1]
                    walker_inds_here = walker_inds_base[S1]

                    # prepare the sets for each model
                    # goes into the proposal as (ntemps * (nwalkers / subset size), nleaves_max, ndim)
                    sets = [
                        work.coords[: self.ntemps][mask_j][:, leaf]
                        .reshape(self.ntemps, -1, 1, ndim)
                        for mask_j in split_masks
                    ]

                    old_points = sets[split].reshape((self.ntemps, nwalkers_here, ndim))

                    # setup s and c based on splits
                    s = {self.branch_name: sets[split]}
                    c = {self.branch_name: sets[:split] + sets[split + 1 :]}

                    # per-(temp, walker) eigen tables are sliced to THIS
                    # split's rows (no-op for shared tables / other moves)
                    self._install_eigen_split_table(move_here, leaf, S1)
                    # the cross-walker jump: its fallback's rows and this
                    # visit's active rows + partners (no-op for other moves)
                    self._install_de_split(move_here, leaf, S1)

                    # Get the move-specific proposal.
                    if isinstance(move_here, StretchMove):
                        q, factors = move_here.get_proposal(s, c, model.random)

                    else:
                        q, factors = move_here.get_proposal(s, model.random)

                    new_points = q[self.branch_name].reshape((self.ntemps, nwalkers_here, ndim))

                    # Compute prior of the proposed position
                    # new_inds_prior is adjusted if product-space is used
                    logp = self.priors[self.branch_name].logpdf(new_points.reshape(-1, ndim))
                    in_prior = ~np.isinf(logp)
                    logl = np.full_like(logp, -1e300)

                    if np.any(in_prior):
                        if self.pad_out_of_prior and np.any(~in_prior):
                            padded = new_points.reshape(-1, ndim).copy()
                            padded[~in_prior] = new_points.reshape(-1, ndim)[in_prior][0]
                            new_points_in = self._to_phys(padded)

                            data_index = np.asarray(walker_inds_here.astype(np.int32))

                            all_logl = self.compute_like(new_points_in, data_index=data_index)
                            logl = np.where(in_prior, all_logl, -1e300)

                        else:
                            new_points_in = self._to_phys(
                                new_points.reshape(-1, ndim)[in_prior]
                            )

                            # Compute the lnprobs of the proposed position.
                            data_index = np.asarray(walker_inds_here[in_prior].astype(np.int32))
                
                            logl[in_prior] = self.compute_like(
                                    new_points_in,
                                    data_index=data_index,
                                )
                    
                    if DEBUG_MODE:
                        logger.debug(f"average proposed logl: {logl[in_prior].mean()}.")
    
                    # -1e300 is the out-of-domain sentinel, not a numerical issue.
                    _logl_in = logl[in_prior]
                    if np.any((_logl_in < -1e10) & (_logl_in > -1e299)) or np.any(_logl_in > 1e30):
                        logger.warning(f"Suspicious likelihood encountered in propose: min = {logl[~np.isinf(logp)].min()}, max = {logl[~np.isinf(logp)].max()}. This could be a sign of numerical issues.")
                        if DEBUG_MODE:
                            breakpoint()
                    # print(f"new logl: {logl}. elapsed: {time.time() - tic}")

                    logl = logl.reshape(self.ntemps, nwalkers_here)

                    logp = logp.reshape(self.ntemps, nwalkers_here)
                    prev_logp_here = prev_logp[S1].reshape(self.ntemps, nwalkers_here)

                    prev_logl_here = prev_logl[S1].reshape(self.ntemps, nwalkers_here)

                    prev_logP_here = temperature_control_here.compute_log_posterior_tempered(
                        prev_logl_here, prev_logp_here
                    )
                    logP = temperature_control_here.compute_log_posterior_tempered(logl, logp)

                    lnpdiff = factors + logP - prev_logP_here

                    keep = lnpdiff > np.log(model.random.rand(self.ntemps, nwalkers_here))

                    # per-axis / cross-walker acceptance tallies (no-op with
                    # {BRANCH}_EIGEN_AXIS_ACC_LOG / _TUNE off and no jump in
                    # the stack; draws no random numbers)
                    self._note_inner_outcome(move_here, leaf, temp_inds_here, keep,
                                             old_points, new_points, S1)

                    temp_inds_update = temp_inds_here[keep.flatten()]
                    walker_inds_update = walker_inds_here[keep.flatten()]

                    accepted[: self.ntemps][(temp_inds_update, walker_inds_update)] = True

                    # update state information
                    work.coords[
                        (
                            temp_inds_update,
                            walker_inds_update,
                            np.full_like(walker_inds_update, leaf),
                        )
                    ] = new_points[keep].reshape(len(temp_inds_update), ndim)

                    prev_logl[(temp_inds_update, walker_inds_update)] = logl[keep].flatten()
                    prev_logp[(temp_inds_update, walker_inds_update)] = logp[keep].flatten()
                    prev_logP[(temp_inds_update, walker_inds_update)] = logP[keep].flatten()

                # acceptance tracking (module-ladder resolution; the engine
                # presets an engine-shaped array, so re-shape on first use)
                if np.shape(getattr(self, "accepted", None)) != accepted.shape:
                    self.accepted = np.zeros(accepted.shape)
                self.accepted += accepted
                # print(self.accepted[0])
                self.num_proposals += 1
                # Every walker is proposed for exactly once per repeat (the
                # split masks partition the ensemble), so the denominator is
                # nwalkers per rung; out-of-prior draws count as proposals.
                _leaf_accepted += np.asarray(
                    accepted[: self.ntemps], dtype=bool).sum(axis=1)
                _leaf_proposed += int(self.nwalkers)

                # TODO: include PSD likelihood in swaps?
                # temperature swaps
                # make swaps
                coords_for_swap = {
                    self.branch_name: work.coords[:, :, leaf].copy()[:, :, None]
                }

                # Walker-permuting (fancy) swap: at most ONCE per leaf
                # visit, on the FINAL in-model repeat — and only on every
                # ``permute_every``-th propose (user ruling 2026-08-27:
                # the fancy swap measured ~17 min of an ~18.5-min MBH
                # leaf visit at 6 rungs, ~5x the in-model work; a
                # per-iteration permute is unaffordable). permute_every
                # is a CADENCE in proposes again: 1 = every propose (the
                # previous behavior), N = every Nth. The FIRST propose
                # after a (re)start never fires (user ruling 2026-08-27):
                # firings land on proposes N, 2N, 3N, ... <= 0 disables
                # entirely ({BRANCH}_PERMUTE_EVERY env override).
                fancy_swap = self._fancy_swap_fires(repeat)
                if fancy_swap:
                    logger.debug(
                        "%s leaf %d repeat %d: fancy (walker-permuting) "
                        "temperature swap firing",
                        self.branch_name, leaf, repeat,
                    )
                #if fancy_swap:
                    # logger.debug(f"Permuting walkers before swap.")
                compute_log_like = self.log_like_for_fancy_swaping

                if self.swap_debug:
                    _cold_before = np.array(prev_logl[0], copy=True)

                # TODO: check permute make sure it is okay
                (
                    coords_for_swap,
                    prev_logP,
                    prev_logl,
                    prev_logp,
                    inds,
                    blobs,
                    supps,
                    branch_supps,
                ) = temperature_control_here.temperature_swaps(
                    coords_for_swap,
                    prev_logP.copy(),
                    prev_logl.copy(),
                    prev_logp.copy(),
                    branch_supps={self.branch_name: None},  # TODO: adjust this to be flexible
                    fancy_swap=fancy_swap,
                    compute_log_like=compute_log_like,
                    permute_here=fancy_swap,
                )

                self._fanout_note_swaps(leaf, temperature_control_here)

                if self.swap_debug:
                    _cold_after = np.asarray(prev_logl[0])
                    _drop = _cold_before - _cold_after
                    logger.info(
                        "[SWAP_DBG] %s leaf %d rep %d fancy=%d | inner cold "
                        "acc %d/%d | cold lnL max/med %.3f / %.3f -> %.3f / "
                        "%.3f | worst walker drop %.3f | swaps_acc %s",
                        self.branch_name, leaf, repeat, int(fancy_swap),
                        int(np.asarray(accepted[0]).sum()), self.nwalkers,
                        float(_cold_before.max()),
                        float(np.median(_cold_before)),
                        float(_cold_after.max()),
                        float(np.median(_cold_after)),
                        float(_drop.max()),
                        np.array2string(
                            np.asarray(
                                temperature_control_here.swaps_accepted
                            ),
                            precision=0,
                        ),
                    )

                temperature_control_here.adapt_temps()

                work.coords[:, :, leaf] = coords_for_swap[
                    self.branch_name
                ][:, :, 0]

            # the end-of-visit rung ensemble -> the leaf's ring buffer for the
            # eigen step-scale adaptation ({BRANCH}_EIGEN_AXIS_ADAPT; no-op off)
            self._note_eigen_adapt_sample(leaf, work, temperature_control_here.betas)

            # ll_tmp1 = -1/2 * 4 * self.df * xp.sum(data_residuals[:2].conj() * data_residuals[:2] / psd[:2], axis=(0, 2)).get()

            # add back cold chain sources
            _free_pool()

            add_coords = work.coords[0, :, leaf]
            add_coords_in = self._to_phys(add_coords)

            _dbg_tmpl = None
            if _dbg_leaf:
                # recovered template = the accepted cold-chain source for the
                # traced walker (built before it is folded back in).
                _dbg_tmpl = self._dbg_template(add_coords_in[_dbg_w], _dbg_w)

            self._record_leaf_inner_products(new_state, add_coords_in, leaf)

            # fold the (updated) cold-chain sources back INTO the fit:
            # subtract their templates from the residual (r = d -> d - h_new).
            self._replay_cold_chain("fold", add_coords_in, leaf)

            if self.swap_debug:
                _acs_cold_post = asnumpy(self.ensemble_likelihood(self.acs))
                _believed = np.asarray(prev_logl[0], dtype=float)
                _delta = _believed - _acs_cold_post
                logger.info(
                    "[STAGE] %s leaf %d POST-READD ACS cold lnL min/med/max "
                    "%.3f / %.3f / %.3f | move-believed cold prev_logl "
                    "min/med/max %.3f / %.3f / %.3f | believed-ACS median "
                    "offset %.3f spread %.3f  (spread ~0 = consistent "
                    "bookkeeping; large spread = state/residual corruption)",
                    self.branch_name, leaf,
                    float(np.min(_acs_cold_post)),
                    float(np.median(_acs_cold_post)),
                    float(np.max(_acs_cold_post)),
                    float(np.min(_believed)),
                    float(np.median(_believed)),
                    float(np.max(_believed)),
                    float(np.median(_delta)),
                    float(np.max(_delta) - np.min(_delta)),
                )

            if os.environ.get("ADDREMOVE_ROUNDTRIP"):
                _rc = np.asarray(removal_coords)[_dbg_w]
                _ac = np.asarray(add_coords)[_dbg_w]
                _same = bool(np.allclose(_rc, _ac))
                _b = self._dbg_snaps.get("before_removal")
                _a = self._dbg_snaps.get("after_readd")
                _dd = lambda x: float(np.sum(np.abs(np.asarray(x)) ** 2)) if x is not None else float("nan")
                # capture the current residual explicitly (after_readd snapshot
                # is only taken under _dbg_leaf below)
                _now = self._dbg_residual(_dbg_w)
                logger.warning(
                    "[ROUNDTRIP %s leaf %d w%d] coords_same=%s  |dcoords|=%.3e  "
                    "sum|resid|^2: before=%.6e  after_readd=%.6e",
                    self.branch_name, leaf, _dbg_w, _same,
                    float(np.max(np.abs(_rc - _ac))), _dd(_b), _dd(_now),
                )

            if _dbg_leaf:
                try:
                    self._dbg_snaps["after_readd"] = self._dbg_residual(_dbg_w)
                    self._debug_plot_source_sequence(
                        leaf, _dbg_w, self._dbg_snaps, _dbg_tmpl,
                        {"domain": getattr(self, "_dbg_domain", "wdm")},
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.info("[%s_DEBUG] plot skipped: %r", self._dbg_prefix, exc)
                self._dbg_leaf_done = True

            # read out all betas from temperature controls
            new_state.sub_states[self.branch_name].betas_all[leaf][
                : self.ntemps
            ] = temperature_control_here.betas

            # acceptance / swap counters + per-leaf lnL for this visit
            self._record_leaf_telemetry(
                new_state, leaf, _leaf_proposed, _leaf_accepted,
                prev_logl, prev_logp,
            )
            # print(leaf)

            # ll_tmp2 = -1/2 * 4 * self.df * xp.sum(data_residuals[:2].conj() * data_residuals[:2] / psd[:2], axis=(0, 2)).get()

            logger.info(f"✓ {self.branch_name} leaf {leaf} complete ({time.time() - tic:.1f}s)")

            # Mid-iteration checkpoint at the leaf boundary (throttled;
            # no-op unless run.py armed the module). Everything accepted so
            # far this propose is already folded into ``new_state`` (the
            # sub-state work views + the residual); ``prepare`` mirrors the
            # branch's cold row into the main state ONLY when a write is
            # actually due, so a resume rebuilds exactly the residual this
            # loop is carrying. This is the hook that matters for the
            # minutes-per-leaf PE branches (MBH/EMRI), where a whole-branch
            # propose outlives a spot-preemption window.
            if not self._fanout_body:
                midit_checkpoint.maybe_write(
                    new_state,
                    tag=f"{self.branch_name} leaf {leaf}",
                    prepare=self._sync_cold_row,
                )

        # udpate at the end
        logger.info(f"✓ {self.branch_name} proposal complete — all leaves processed ({time.time() - tic:.1f}s total)")
        # new_state.log_like[(temp_inds_update, walker_inds_update)] = logl.flatten()
        # new_state.log_prior[(temp_inds_update, walker_inds_update)] = logp.flatten()
        # print("before computing current likelihood. elapsed: ", time.time() - tic)
        # ENSEMBLE-WIDE, not block-local (2026-09-23). ``new_state`` below is
        # written at the width of the state this body was handed, and under
        # the flat control plane that is the WHOLE ensemble while this rank's
        # ACA holds only its own block -- so a bare ``self.acs.likelihood()``
        # would be B rows for an N-row assignment. At one block per rank this
        # returns exactly the local read it replaces.
        current_ll = (
            self.ensemble_likelihood(self.acs)
        )  #  - xp.sum(xp.log(xp.asarray(psd[:2])), axis=(0, 2))).get()
        # print("after computing current likelihood. elapsed: ", time.time() - tic)
        if np.any(current_ll < 0.0):
            logger.warning(f"The ACS likelihood should always be positive given the psd contribution, but got {current_ll.min()}")
            logger.warning(f"The minimum proposed likelihood was {prev_logl.min()}.")
            if DEBUG_MODE:
                breakpoint()

        _free_pool()
        # TODO: add check with last used logl

        current_lp = (
            self.priors[self.branch_name]
            .logpdf(work.coords[0, :, :].reshape(-1, ndim))
            .reshape(work.shape[1:-1])
            .sum(axis=-1)
        )

        new_state.log_like[0] = current_ll
        # new_state.log_prior[0] = current_lp
        _free_pool()
        if not hasattr(self, "best_last_ll"):
            self.best_last_ll = current_ll.max()
            self.low_last_ll = current_ll.min()
        # print(self.branch_name, self.best_last_ll, current_ll.max(), current_ll.max() - self.best_last_ll)
        # print(current_ll.max(), self.best_last_ll, current_ll.min(), self.low_last_ll)
        self.best_last_ll = current_ll.max()
        self.low_last_ll = current_ll.min()

        if self.temperature_control is None:
            # this really does not matter
            self.temperature_control = self.temperature_controls[0]

        self.temperature_control.swaps_accepted = self.temperature_controls[0].swaps_accepted

        # new_state.log_prior[:] = model.compute_log_prior_fn(new_state.branches_coords, inds=new_state.branches_inds, supps=new_state.supplimental)
        # breakpoint()
        new_state.log_like[:] = (
            current_ll #self.acs.likelihood()
        )  #  - xp.sum(xp.log(xp.asarray(psd[:2])), axis=(0, 2))).get()

        self.free_gpu_memory()

        # assert np.abs(new_state.log_like[0] - self.acs.get_ll(include_psd_info=True)).max() < 1e-4
        # breakpoint()
        logger.debug(f"mean accepted fraction: {np.mean(self.accepted[0] / self.num_proposals)}. elapsed: {time.time() - tic}")
        # mirror the updated cold row into the main (engine) state and hand
        # eryn an engine-shaped acceptance array
        self._sync_cold_row(new_state)
        return new_state, np.asarray(accepted)[:engine_ntemps]

    def replace_residuals(self, old_state, new_state):
        """Swap the cold-chain contribution from ``old_state`` to ``new_state``.

        Domain-agnostic: each template is assembled through the containers'
        ``signal_gen`` machinery (see :meth:`_apply_cold_chain_sources`) as a
        :class:`~lisatools.domains.DomainBase` (FD, STFT, WDM, ...) and the
        :class:`AnalysisContainerArray` dispatches the actual residual
        update by basis. The old FD-only implementation hand-assembled
        per-channel arrays from ``self.acs.fd``; that path is now
        unnecessary because :meth:`add_signal_to_residual` /
        :meth:`remove_signal_from_residual` work on any domain.
        """
        for leaf in range(old_state.branches[self.branch_name].shape[-2]):
            removal_coords = old_state.branches[self.branch_name].coords[0, :, leaf]
            self._current_leaf = int(leaf)
            removal_coords_in = self._to_phys(removal_coords)
            self._apply_cold_chain_sources(removal_coords_in, sign=+1)

            add_coords = new_state.branches[self.branch_name].coords[0, :, leaf]
            add_coords_in = self._to_phys(add_coords)
            self._apply_cold_chain_sources(add_coords_in, sign=-1)

        _free_pool()


# stft_tof addition: multi-GPU variant of the general add/remove move with
# per-device waveform-generator replicas. Kept through the merge; its
# interaction with the dev-side ACA sharding is reviewed in the dedicated
# multi-GPU pass.
class MultiGPUResidualAddRemoveMove(ResidualAddOneRemoveOneMove):
    """Deprecated alias — :class:`ResidualAddOneRemoveOneMove` is ONE
    structure that takes ``dcga=`` directly (2026-07 multi-GPU pass). This
    shim maps the legacy ``MultiGPUResidualAddRemoveMove(dcga, waveform_gen,
    ...)`` signature onto it.
    """

    def __init__(
        self,
        dcga: DomainComputationGroupArray,
        waveform_gen: Any,
        branch_name: str,
        coords_shape: tuple,
        waveform_gen_method: str,
        waveform_gen_kwargs: dict,
        waveform_like_kwargs: dict,
        num_repeats: int,
        transform_fn: TransformContainer,
        priors: ProbDistContainer,
        inner_moves: list,
        Tmax: float = np.inf,
        betas_all: np.ndarray = None,
        permute_every: int = 20,
        pad_out_of_prior: bool = False,
        run_async: bool = False,
        run_threaded: bool = False,
        waveform_like_method: str = None,
        **kwargs,
    ):
        import warnings

        warnings.warn(
            "MultiGPUResidualAddRemoveMove is deprecated: construct "
            "ResidualAddOneRemoveOneMove(..., dcga=...) directly (one move "
            "structure).",
            DeprecationWarning,
            stacklevel=2,
        )
        ResidualAddOneRemoveOneMove.__init__(
            self,
            branch_name=branch_name,
            coords_shape=coords_shape,
            waveform_gen=waveform_gen,
            waveform_gen_kwargs=waveform_gen_kwargs,
            waveform_like_kwargs=waveform_like_kwargs,
            acs=dcga.acs,
            num_repeats=num_repeats,
            transform_fn=transform_fn,
            priors=priors,
            inner_moves=inner_moves,
            Tmax=Tmax,
            betas_all=betas_all,
            permute_every=permute_every,
            pad_out_of_prior=pad_out_of_prior,
            dcga=dcga,
            waveform_gen_method=waveform_gen_method,
            waveform_like_method=waveform_like_method,
            run_async=run_async,
            run_threaded=run_threaded,
            **kwargs,
        )
