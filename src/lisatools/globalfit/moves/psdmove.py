"""MCMC move that updates the LISA noise model (psd / galfor / sgwb branches).

Noise-move split (2026-07): ONE parameterized move class. Each instance
samples ``sampled_branches`` under its OWN temperature ladder while the
likelihood always evaluates the full noise model, with non-sampled noise
branches frozen at their cold-chain rows. ``sampled_branches=None``
reproduces the historical joint psd+galfor+sgwb move exactly.

stft_tof merge (2026-06): hybrid likelihood routing. The move keeps the
dev-side domain-agnostic structure (per-walker sensitivity matrices installed
on the AnalysisContainers, scored via ``acs.likelihood()``), and adds the
stft_tof C++ XYZ-sensitivity kernel (``psd_likelihood_wrap`` shared-memory
sums) as a fast path when the data basis is FD or STFT and the ACA holds a
single shard. The WDM basis falls back to the ACA route until the WDM
counterpart kernels land in domains.cu; on the ACA route the walker-batched
``PSD_BATCH`` scorer (default on) builds every walker's covariance in one
batched pass (:meth:`PSDMove._compute_log_like_batched`).

Pipeline position. :func:`lisatools.globalfit.recipe.build_noise_moves` builds
one ``(search, pe)`` pair of :class:`PSDMove` per noise branch, each with its
own ladder; the stock names are ``psd_pe`` / ``galfor_pe`` / ``sgwb_pe`` and
``psd_search`` / ``galfor_search`` / ``sgwb_search`` (``docs/conventions.md``,
"Noise moves are split"). On the 9mo / 12mo runs (``erebor.all_sources``, no
sgwb branch) only ``psd_pe`` and ``galfor_pe`` run: in ``full_pe`` and inside
``gb_search_2``'s cycle-end noise search (a
:class:`~lisatools.globalfit.moves.globalfitmove.MaxLogLCombineMove` over the
two); the noise is pinned at launch, so no noise-search stage runs and the
``*_search`` twins are built but never staged.

Knobs: ladder size ``PSD_NTEMPS`` / ``GALFOR_NTEMPS`` / ``SGWB_NTEMPS`` (settings
fields, default 12; branches sampled by ONE move must share a size, else the build
raises), repeats per
propose ``{PSD,GALFOR,SGWB}_NUM_PROP_REPEATS`` (settings default 50; launchers 10;
``full_pe`` raises them to ``PE_INMODEL_REPEATS=25`` at entry), the inner
proposal ``{P}_INNER_MOVE_KIND`` (unset -> ``eigen`` for a block of fewer than
three walkers), the eigen tables ``{P}_EIGEN_INFO`` (``ll`` default; launchers
``fisher``), ``{P}_EIGEN_SCOPE``, ``{P}_EIGEN_TEMPER_SIGMAS``, ``{P}_EIGEN_REFRESH``,
``{P}_EIGEN_EPS_REL``, the batched scorer ``PSD_BATCH``, and the debug plot hook
``{P}_DEBUG*``; see :class:`PSDMove`.
"""

import os
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import logging

import numpy as np
try:
    import cupy as cp
except (ModuleNotFoundError, ImportError):
    import numpy as cp
from eryn.model import Model

# Propose-level stage timers, same machinery and same reporting format the GB
# moves use (_ProposeTimer / _tspan in gbspecialstretch). psd + galfor were
# the only uninstrumented branches: with GB at 47.7% of a 1,018 s iteration
# and VGB at 14.4%, the remaining ~38% could only be inferred as a residual.
# gbspecialstretch does not import this module, so there is no cycle.
from .gbspecialstretch import _ProposeTimer, _tspan
from eryn.moves import RedBlueMove, StretchMove
from eryn.state import BranchSupplemental
from eryn.state import State as eryn_State
from eryn.utils.transform import TransformContainer

from tqdm import tqdm

from ...analysiscontainer import AnalysisContainerArray
from ...diagnostic import (
    batched_residual_full_source_and_noise_likelihoods,
    psd_debug_checks_enabled,
)
from ...domaincomputation import DomainComputationGroupArray
from ...domains import (
    CoarseWDMSettings,
    FDSettings,
    FDSignal,
    STFTSettings,
    WDMSettings,
)
from ...sensitivity import (
    _XYZ_ELEMENT_SENS,
    _mat3x3_det_inv,
    CompositeSensitivityBackend,
    GalacticForeground,
    InstrumentNoise,
    SGWB,
    XYZSensitivityBackend,
)
from ...stochastic import check_stochastic
from ...utils.device import device_context
from ...utils.utility import asnumpy
from ..state import GFState
from .globalfitmove import GlobalFitMove
from .walkerfanout import WalkerFanoutMixin, pooled_ladder_step

logger = logging.getLogger(__name__)


# TODO check: keep/remove/document -- galfor ratchet only (PSDMove.forced_noise_step); GALFOR_RATCHET default 0, launchers 0, ratchet dropped from production 2026-10-03
def clip_shift_to_prior(coords, delta, prior_fn, max_iter=48):
    """Shift every row of ``coords`` by ``delta``, pulling any row that would
    leave the prior back to the box edge DIMENSION BY DIMENSION.

    The galfor ratchet's forced step applies one fixed vector to every rung
    and walker. After a release the ladder spreads, and a hot rung can sit
    within one nudge of a prior bound (6mo job 675, 2026-10-01: rungs 3 and
    6 were 0.07 / 0.002 dex above the ``fk`` floor and the second nudge was
    refused outright, which killed the run). A row at the edge now takes
    as much of each component as the box allows -- found by bisection on
    the fraction of that component's shift, with the other components
    unshifted -- and every other row takes the full vector.

    Exact for a product (box) prior, which is every noise prior in the
    stock recipes. For any other prior the combined shift is re-checked
    and a row still outside raises rather than ships.

    Args:
        coords: ``(ntemps, nwalkers, 1, ndim)`` sampled-basis coordinates.
        delta: ``(ndim,)`` shift.
        prior_fn: ``prior_fn(arr) -> (ntemps, nwalkers)`` log-prior of a
            trial coordinate array of the same shape, ``-inf`` outside.
        max_iter: bisection depth (48 -> the edge to 1 part in 3e14 of the
            shift).

    Returns:
        ``(shifted, clipped)``: the new coordinates and the ``(ntemps,
        nwalkers)`` mask of rows that did not take the full shift.

    Raises:
        ValueError: a row is outside the prior BEFORE the shift (a store
            resumed under a narrower prior, say) -- clipping recovers a
            shift that would leave the box, it must not paper over a row
            that was never inside it; or the prior is not separable and a
            row is still outside after the per-component clip.
    """
    coords = np.asarray(coords, dtype=float)
    delta = np.asarray(delta, dtype=float)
    if delta.shape != (coords.shape[-1],):
        raise ValueError(
            f"clip_shift_to_prior: shift has shape {delta.shape}; expected "
            f"({coords.shape[-1]},).")
    shifted = coords + delta
    bad = ~np.isfinite(np.asarray(prior_fn(shifted), dtype=float))
    if not bad.any():
        return shifted, bad
    base_ok = np.isfinite(np.asarray(prior_fn(coords), dtype=float))
    if not base_ok[bad].all():
        rows = [tuple(int(i) for i in r) for r in np.argwhere(bad & ~base_ok)]
        raise ValueError(
            f"clip_shift_to_prior: {len(rows)} row(s) (rung, walker) "
            f"{rows} are outside the prior BEFORE the shift; nothing was "
            "changed.")
    frac = np.ones(bad.shape + (delta.size,))
    for d in np.flatnonzero(delta != 0.0):
        unit = np.zeros_like(delta)
        unit[d] = delta[d]
        ok_full = np.isfinite(np.asarray(prior_fn(coords + unit), dtype=float))
        need = bad & ~ok_full
        if not need.any():
            continue
        lo = np.zeros(bad.shape)
        hi = np.ones(bad.shape)
        for _ in range(int(max_iter)):
            mid = 0.5 * (lo + hi)
            trial = coords + mid[..., None, None] * unit
            ok = np.isfinite(np.asarray(prior_fn(trial), dtype=float))
            lo = np.where(ok, mid, lo)
            hi = np.where(ok, hi, mid)
        frac[need, d] = lo[need]
    shifted = coords + frac[:, :, None, :] * delta
    still = ~np.isfinite(np.asarray(prior_fn(shifted), dtype=float))
    if still.any():
        rows = [tuple(int(i) for i in r) for r in np.argwhere(still)]
        raise ValueError(
            f"clip_shift_to_prior: rows (rung, walker) {rows} are still "
            "outside the prior after clipping each component to its own "
            "edge -- the prior is not a product box; nothing was changed.")
    return shifted, bad

DEBUG_MODE = False


class PSDMove(WalkerFanoutMixin, GlobalFitMove, StretchMove):
    """Noise-model move: samples ``sampled_branches`` of the noise model.

    Samples ``sampled_branches`` of the noise model (``psd``, ``galfor``,
    ``sgwb``) while the likelihood always evaluates the FULL noise model —
    non-sampled noise branches contribute their cold-chain coordinates (per
    walker), the same "temper your own ensemble against the cold-chain rest"
    structure the add/remove source branches use. With the default
    ``sampled_branches=None`` every noise branch present in the state is
    sampled jointly (the historical single-joint-move behavior). The stock
    recipes build one instance pair per branch
    (:func:`~lisatools.globalfit.recipe.build_noise_moves`: ``psd_pe`` /
    ``psd_search``, ``galfor_pe`` / ``galfor_search``, ``sgwb_pe`` / ``sgwb_search``).

    The ``propose(model, state)`` contract. ``propose`` is the
    :class:`~lisatools.globalfit.moves.walkerfanout.WalkerFanoutMixin`'s: with one
    compute rank it calls :meth:`propose_local` directly; with several it slices
    the state per walker block, runs :meth:`propose_local` on every rank and
    merges. :meth:`propose_local` then:

    1. :meth:`_begin_propose`: checks the noise branches' cold rows against their
       sub-states, freezes the NON-sampled noise branches at their cold rows
       (``_fixed_noise_coords``), builds the working state from the sampled
       branches' SUB-STATES (this move's own ladder, shape
       ``(ntemps_module, nwalkers)``), resolves the inner proposal and, for
       ``eigen``, refreshes the eigen tables every ``eigen_refresh_every``
       visits;
    2. runs the inner loop: ``num_repeats`` x :meth:`run_move`
       (:meth:`run_move_for_loop`, the PE path), the plateau loop
       (``max_logl_mode``, :meth:`run_move_max_likelihood`) or the tiled ensemble
       search (search stages with ``ensemble_search``);
    3. :meth:`_finish_propose`: writes each SAMPLED branch's full ladder into its
       sub-state (coords, ``log_like``, ``log_prior``, ``betas`` and this
       iteration's in-model / swap counters), mirrors row 0 into the main state
       (``_sync_cold_row``), sets the cold ``log_prior`` to the FULL noise-model
       prior sum, publishes every walker's cold-row noise model on its container
       (``sens_mat``, then ``reset_linear_psd_arr``) and sets the cold
       ``log_like`` to the refreshed container likelihoods (``acs.likelihood``).

    It returns ``(new_state, accepted)`` with ``accepted`` cut to the engine
    ladder. Non-sampled branches' sub-states are never written: each noise
    sub-state is owned by the move that samples it.

    Ladders and swaps. Each instance carries its own ``temperature_control``
    (built by ``build_noise_moves`` with ``ntemps`` from the lead branch's
    ``{P}_NTEMPS``, default 12, or its ``betas``; ``Tmax`` 1e6;
    ``walker_inds`` excluded from swaps). Every repeat of :meth:`run_move` runs
    an identity-paired swap within each walker; the walker-permuting ("fancy")
    swap fires every ``permute_every`` repeats and never with one walker per
    block (the production layout). Under several compute ranks the ranks'
    ladders are frozen and the head adapts the shared ladder once per propose
    from the pooled swap tallies (:meth:`fanout_merge_extra`).

    Args:
        acs: AnalysisContainerArray containing the data and sensitivity information
            (``dcga.acs`` when ``None`` and a ``dcga`` is given).
        priors (dict): Branch name -> prior container for the noise branches.
        *args: Forwarded to :class:`GlobalFitMove` and eryn's
            :class:`~eryn.moves.StretchMove`.
        sampled_branches: which noise branches THIS move samples (subset of
            ``["psd", "galfor", "sgwb"]``, reordered canonically; an unknown name
            raises ``ValueError``). ``None`` samples all present branches
            jointly. Each move instance carries its own ``temperature_control``
            ladder for its sampled block.
        num_repeats: In-model repeats per propose (and per plateau-loop pass).
            Default 1; ``build_noise_moves`` passes the branch's
            ``num_prop_repeats``.
        max_logl_mode: True makes :meth:`propose_local` run
            :meth:`run_move_max_likelihood` (repeat until the cold-chain maximum
            stops improving for 5 passes): the ``*_search`` instances. Default
            ``False``.
        psd_kwargs: Extra keyword arguments forwarded to :meth:`psd_log_like`
            (kernel fast path only), which currently ignores them. Default ``{}``.
        sensitivity_backend: The run's sensitivity backend
            (``general_info.sensitivity_backend``: a
            :class:`~lisatools.sensitivity.CompositeSensitivityBackend` on the
            stock WDM runs, an :class:`~lisatools.sensitivity.XYZSensitivityBackend`
            on the kernel fast path).
        psd_transform_fn: TransformContainer for transforming the PSD parameters.
        galfor_transform_fn: TransformContainer for transforming the galactic
            foreground parameters. NOTE the long-standing order convention is
            preserved as-is: the galfor prior/sampling basis is labelled
            ``[amp, knee, alpha, Slope1, Slope2]`` while the kernel unpack
            reads ``[Amp, alpha, f_1, kn, f_2]`` (see
            :meth:`prepare_likelihood_inputs`; history in the commented
            reorder block in ``run.py``). Do not reorder here.
        sgwb_transform_fn: TransformContainer for the SGWB parameters. Applied
            on the ACA routes (:meth:`_build_sensitivity_for_walker`, the
            batched scorer); the kernel fast path has no SGWB term and is
            refused for an sgwb-bearing model.
        permute_every: Cadence, in repeats, of the walker-permuting ("fancy")
            temperature swap (fires on repeat indices divisible by it, and only
            with more than one walker in the block). ``<= 0`` (e.g. -1) turns
            it off. Default 10; ``build_noise_moves`` passes 50. Identity swaps
            run every repeat.
        tolerance: minimum allowed distance between spline knot positions in the
            sensitivity model (kernel fast path with splines only). Default 0.
        dcga: Run the kernel fast path on per-device replicas (multi-GPU
            single-process runs). Default ``None``.
        run_async: Forwarded to the DCGA likelihood. Default ``False``.
        run_threaded: Threaded per-split dispatch. Default ``False``.
        build_threads: CPU threads for per-walker sensitivity builds (see
            :attr:`build_threads`). Default 1.
        coarse_runtime (CoarseWDMRuntime, optional): The all-source coarse sidecar
            (``COARSE_GPU_MODE``); ``None`` (default, and the 9mo / 12mo runs)
            leaves every route on the fine likelihood.
        inner_move_kind: the in-model proposal, ``"eigen"`` (eryn
            ``EigenAxisMove`` on per-rung information-matrix tables) or
            ``"stretch"``; ``None`` resolves from the walker count (see
            :meth:`_resolve_inner_kind`). Settings field ``inner_move_kind``,
            env ``{PSD,GALFOR,SGWB}_INNER_MOVE_KIND``.
        eigen_refresh_every: proposes between eigen table refreshes (Settings
            field, env ``{P}_EIGEN_REFRESH``). Default 10.
        eigen_eps_rel: finite-difference step of both information-matrix
            routes (likelihood second differences, and the covariance
            differences of the ``"fisher"`` route), as a fraction of each
            prior-box width (Settings field, env ``{P}_EIGEN_EPS_REL``). Default
            1e-4.
        eigen_info: information-matrix route of the eigen tables, ``"ll"``
            (likelihood second differences, the default) or ``"fisher"``
            (the expected noise Fisher, :meth:`_noise_fisher`; PSD_BATCH
            route only, falls back to ``"ll"`` on any error). Applies to
            every branch this move samples; ``None`` reads
            ``{BRANCH}_EIGEN_INFO`` per branch (``PSD_EIGEN_INFO``,
            ``GALFOR_EIGEN_INFO``, ``SGWB_EIGEN_INFO``), then ``"ll"``. See
            ``docs/eigen-info-routes.md``.
        ensemble_search: Arm the tiled per-walker ensemble search in
            search-kind stages (:meth:`run_move_ensemble_search`; settings field,
            env ``{P}_ENSEMBLE_SEARCH``). Default ``False``.
        ensemble_repeats: Tile copies ``R`` per walker (floored at 2).
            Default 10.
        ensemble_spread_lo: Lower target of each walker's tiled lnL
            spread. Default 1.0.
        ensemble_spread_hi: Upper target of the spread. Default 10.0.
        ensemble_scale0: Starting perturbation scale, in prior-box
            fractions. Default 1e-3.
        ensemble_scale_tries: Rescaling rounds of the spread loop (floored
            at 1). Default 4.
        **kwargs: Forwarded to :class:`GlobalFitMove` (``name``, required) and
            eryn's :class:`~eryn.moves.StretchMove` (``temperature_control``,
            ``live_dangerously``, ``use_gpu``, ``periodic``, ...).

    Attributes:
        NOISE_BRANCHES (tuple): ``("psd", "galfor", "sgwb")``, the canonical order.
        sampled_branches (list[str] or None): See ``sampled_branches``.
        num_repeats (int): See ``num_repeats`` (the fan-out ships a live value to
            the ranks, e.g. ``PE_INMODEL_REPEATS`` in ``full_pe``).
        max_logl_mode (bool): See ``max_logl_mode``.
        psd_batch (bool): ``PSD_BATCH`` at construction (see the note).
        prev_logl (np.ndarray): The last :meth:`compute_log_like` output.

    Note:
        ``PSD_BATCH`` (seeds :attr:`psd_batch`, default ``1``): on the ACA
        (composite / WDM) route, score each walker batch through ONE batched
        covariance build + ONE batched likelihood per shard instead of the
        per-walker Python loop — same numbers (parity asserted at
        rtol <= 1e-12 in ``tests/test_psd_move_batched.py``, for the
        ``fold`` and ``layer_calibrated`` WDM PSD methods), O(1) kernel
        launches per scoring batch per device instead of O(nwalkers), and no
        per-batch sens-mat install/restore or ``linear_psd_arr`` repack.
        The galfor / SGWB magnitudes follow the backend's
        ``wdm_psd_method`` (``WDM_PSD_METHOD``) exactly as the per-walker
        components do (see :meth:`_batched_stochastic_mag`).
        ``PSD_BATCH=0`` restores the historical per-walker path (the
        ``_force_parent_path``-style escape hatch). The batched route
        engages only when supported (CompositeSensitivityBackend with the
        instrument basis cache, a linear-in-noise-levels instrument model,
        3x3 channel structure, no frequency-layer mask, full-likelihood
        containers); anything else keeps the per-walker path, with one
        ``[PSD_BATCH] ... OFF (reason)`` log line.

        Other env knobs read by this class: ``{BRANCH}_EIGEN_INFO`` (see
        ``eigen_info``); ``{P}_EIGEN_SCOPE`` (:meth:`_eigen_scope`) and
        ``{P}_EIGEN_TEMPER_SIGMAS`` (:meth:`_eigen_temper_sigmas`), with ``P`` =
        ``PSD`` / ``GALFOR`` / ``SGWB`` after the sampled branch;
        ``{P}_DEBUG``, ``{P}_DEBUG_DIR``, ``{P}_DEBUG_PLOT_WALKER``,
        ``{P}_DEBUG_EVERY`` (the residual-vs-model plot hook; ``FG_*`` aliases the
        galfor prefix); ``GB_PROP_TIMING_SYNC`` (device sync in the
        ``[PSD_TIMING]`` span timer, default ``0``).

    Example:
        The stock layer builds the instances; a recipe names them:

        .. code-block:: python

            from lisatools.globalfit.recipe import build_noise_moves

            psd_search, psd_pe = build_noise_moves(
                engine_info, curr, acs, priors, sampled_branches=["psd"])
            galfor_search, galfor_pe = build_noise_moves(
                engine_info, curr, acs, priors, sampled_branches=["galfor"])

            # on a stock fit the recipe refers to them by stock name:
            fit.recipe.get_move("psd_pe", stage="full_pe")
    """
    fanout_flat_body = False  # see the mixin: PSD reads walker count/devices off the LOCAL ACA


    # canonical noise-model branch order (psd is mandatory in the model;
    # galfor/sgwb optional)
    NOISE_BRANCHES = ("psd", "galfor", "sgwb")
    # Do not trade runtime for an unbounded second copy of the PSD plane on
    # uncoarsened two-year WDM runs.  The try5 coarse cache is only ~6.4 MiB;
    # the equivalent Q=1 cache would approach 1 GiB for 14 walkers.
    FIXED_COMPONENT_CACHE_MAX_BYTES = 256 * 1024**2
    # Omitted foreground layers perturb every diagonal covariance by less
    # than this fraction of the frozen instrument covariance.  At Q=150 the
    # resulting worst-case accumulated log-likelihood error is O(1e-4), far
    # below both floating Monte Carlo scatter and proposal-scale delta-log-L,
    # while the try5 posterior keeps only ~38 of 59 active layers.
    GALFOR_SUBBAND_REL_TOL = 1e-10
    # A WDM layer averages a finite frequency window.  The cheap layer-center
    # support estimate can cross the relative threshold one layer earlier than
    # that exact average near a sharp foreground cutoff.  Retain two neighbors
    # on either side before performing the exact selected-layer fold.
    GALFOR_SUBBAND_GUARD_LAYERS = 2

    # per-iteration acceptance deltas; allocated by propose() at the module
    # ladder shape, so run_move() skips the tally when called on its own.
    _tally_in_model_proposed = None
    _tally_in_model_accepted = None
    _tally_swaps_proposed = None
    _tally_swaps_accepted = None
    fanout_assigns_counters = True  # body ASSIGNS sub.in_model_*/swaps_* -> zero, then sum

    def __init__(
        self,
        acs: AnalysisContainerArray,
        priors,
        *args,
        sampled_branches: list = None,
        num_repeats: int = 1,
        max_logl_mode: bool = False,
        psd_kwargs: dict = {},
        sensitivity_backend: XYZSensitivityBackend = None,
        psd_transform_fn: TransformContainer = None,
        galfor_transform_fn: TransformContainer = None,
        sgwb_transform_fn: TransformContainer = None,
        # 10 (user ruling 2026-08-12): fancy (walker-permuting, re-scored)
        # swap cadence; identity naive swaps fire every repeat regardless.
        permute_every: int = 10,
        tolerance: float = 0.0,
        dcga: DomainComputationGroupArray = None,
        run_async: bool = False,
        run_threaded: bool = False,
        build_threads: int = 1,
        coarse_runtime=None,
        inner_move_kind: str = None,
        eigen_refresh_every: int = 10,
        eigen_eps_rel: float = 1e-4,
        eigen_info: str = None,
        # tiled per-walker ensemble search -- SEARCH stages only, see
        # _ensemble_search_active / run_move_ensemble_search
        ensemble_search: bool = False,
        ensemble_repeats: int = 10,
        ensemble_spread_lo: float = 1.0,
        ensemble_spread_hi: float = 10.0,
        ensemble_scale0: float = 1e-3,
        ensemble_scale_tries: int = 4,
        **kwargs,
    ):

        GlobalFitMove.__init__(self, *args, **kwargs)
        StretchMove.__init__(self, *args, **kwargs)
        # ONE move structure (2026-07 multi-GPU pass): pass ``dcga=`` to run
        # the kernel fast path through per-device replicas — how many GPUs
        # the move uses is set by the DCGA it is built with. Without a DCGA
        # the move runs its single-device path. Same object, same propose.
        self._dcga = dcga
        self._run_async = run_async
        self._run_threaded = run_threaded
        # ACA-route per-walker sensitivity builds, spread over threads. See
        # :attr:`build_threads`. Pool is lazy and never pickled.
        self._build_threads = max(1, int(build_threads))
        self._build_pool = None
        # One-time serial warm of the backend's walker-independent caches,
        # tracked PER DEVICE on multi-GPU ACAs (see the note in
        # :meth:`_build_batch`). Holds the devices already warmed (``None``
        # is the CPU / single-device entry).
        self._build_warmed = set()
        # All-source coarse sidecar (plan-2): a CoarseWDMRuntime owning the
        # per-walker coarse statistics and the coarse backend. None (default)
        # leaves every existing route untouched. The runtime is consulted
        # only by compute_coarse_log_like and the propose-entry refresh; the
        # fine callback, the containers' sens_mat, and the packed ACA buffer
        # never see coarse state.
        self.coarse_runtime = coarse_runtime
        # inner proposal: eryn stretch (needs >= 2 walkers) or the eigen-axis MH
        # move fed by per-rung likelihood-difference info matrices (one-walker
        # default). Resolved per propose from the module ladder's walker count.
        self.inner_move_kind = inner_move_kind
        self.eigen_refresh_every = max(1, int(eigen_refresh_every or 10))
        self.eigen_eps_rel = float(eigen_eps_rel or 1e-4)
        # None defers to {BRANCH}_EIGEN_INFO at refresh time (_eigen_info)
        self.eigen_info = eigen_info
        self._inner_kind = None
        self._inner_kind_logged = False
        self._eigen_inner = None
        self._eigen_visits = 0
        self.ensemble_search = bool(ensemble_search)
        self.ensemble_repeats = max(2, int(ensemble_repeats or 10))
        self.ensemble_spread_lo = float(ensemble_spread_lo)
        self.ensemble_spread_hi = float(ensemble_spread_hi)
        self.ensemble_scale0 = float(ensemble_scale0)
        self.ensemble_scale_tries = max(1, int(ensemble_scale_tries or 1))
        self._ensemble_inner = None
        self._eigen_inner_accepted_shape = None
        self._fixed_component_covariances_coarse = {}
        if dcga is not None:
            if acs is None:
                acs = dcga.acs
            if sensitivity_backend is None:
                sensitivity_backend = (
                    dcga.computation_groups[0].sensitivity_backend
                )
        self.acs = acs
        self.psd_kwargs = psd_kwargs
        self.priors = priors
        self.num_repeats = num_repeats
        self.max_logl_mode = max_logl_mode
        self.starting_now = True

        if sampled_branches is not None:
            unknown = [b for b in sampled_branches if b not in self.NOISE_BRANCHES]
            if unknown:
                raise ValueError(
                    f"Unknown noise branch(es) {unknown}; sampled_branches must be a "
                    f"subset of {list(self.NOISE_BRANCHES)}."
                )
            # canonical order regardless of input order
            sampled_branches = [b for b in self.NOISE_BRANCHES if b in sampled_branches]
        self.sampled_branches = sampled_branches
        # cold-row snapshot of the non-sampled noise branches, refreshed at
        # every propose(); empty when this move samples the whole model
        self._fixed_noise_coords = {}
        # Per-propose, per-walker additive covariances for noise branches that
        # this Metropolis-within-Gibbs move freezes at their cold rows.  These
        # arrays are exact and immutable for the whole repeat block.
        self._fixed_component_covariances = {}
        self._fixed_total_covariances = {}
        self._fixed_total_loglikes = None
        self._fixed_total_loglike_frequency_terms = None

        self.sensitivity_backend = sensitivity_backend
        self.psd_transform_fn = psd_transform_fn
        self.galfor_transform_fn = galfor_transform_fn
        self.sgwb_transform_fn = sgwb_transform_fn

        self.permute_every = permute_every
        self.tolerance = tolerance

        # Debug escape hatch (see the note in the legacy MultiGPUPSDMove):
        # when True, psd_log_like bypasses the DCGA routing entirely.
        self._force_parent_path = False

        # Walker-batched ACA-route scoring (see the class docstring's
        # PSD_BATCH entry). ``_batch_runtime`` caches the per-device
        # walker-independent pieces (instrument bases handles, modulation
        # matrices, extra-component covariances) — built once per device and
        # persisted for the run (allocate-once rule); dropped from pickles.
        self.psd_batch = bool(int(os.environ.get("PSD_BATCH", "1")))
        self._batch_runtime = {}
        self._batch_route_state = None  # lazily resolved: True/False
        self._batch_route_reason = None
        self._batch_mem_logged = set()

        self._setup_debug()

    # Noise-model twin of the per-branch ``{BRANCH}_DEBUG`` plot hooks in
    # addremovemove/gbspecialstretch. The prefix follows the move's sampled
    # branch: ``PSD_DEBUG`` on the psd move, ``GALFOR_DEBUG`` (alias
    # ``FG_DEBUG``) on the galfor move, ``SGWB_DEBUG`` on sgwb. Knobs:
    #   {P}_DEBUG             "1" arms the plot hook (default off)
    #   {P}_DEBUG_DIR         output folder (default ./gf_output/{branch}_debug/)
    #   {P}_DEBUG_PLOT_WALKER walker / analysis container plotted (0)
    #   {P}_DEBUG_EVERY       plot only every Nth propose (1)
    def _setup_debug(self) -> None:
        """Read the ``{PSD,GALFOR/FG,SGWB}_DEBUG`` env options (from __init__)."""
        sampled = list(self.sampled_branches or self.NOISE_BRANCHES)
        if sampled == ["galfor"]:
            prefixes = ("GALFOR", "FG")
        elif sampled == ["sgwb"]:
            prefixes = ("SGWB",)
        else:
            prefixes = ("PSD",)
        self._debug_prefix = prefixes[0]

        def _opt(name, default):
            for p in prefixes:
                v = os.environ.get(f"{p}_{name}")
                if v not in (None, ""):
                    return v
            return default

        self.debug = bool(int(_opt("DEBUG", "0")))
        self.debug_plot_dir = _opt("DEBUG_DIR", f"./gf_output/{sampled[0]}_debug/")
        self.debug_plot_walker = int(_opt("DEBUG_PLOT_WALKER", "0"))
        self.debug_every = max(1, int(_opt("DEBUG_EVERY", "1")))
        self._debug_step = 0
        self._debug_walker_params = None
        if self.debug:
            logger.info(
                "[%s_DEBUG] armed: dir=%s walker=%d every=%d",
                self._debug_prefix, self.debug_plot_dir,
                self.debug_plot_walker, self.debug_every,
            )

    def _debug_plot_psd_vs_residual(self) -> None:
        """Overlay the walker's current noise model on the residual's
        per-layer power. For the WDM basis ``E[w_mn^2] == S_wdm[m]`` (the
        fold validation contract), so on a converged noise fit the two
        curves sit on top of each other; a mismatch localizes which layers
        the fit is missing."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            def _h(a):
                return a.get() if hasattr(a, "get") else np.asarray(a)

            w = self.debug_plot_walker
            ac = self.acs[w]
            res = _h(ac.data.arr)
            S = _h(ac.sens_mat[:])

            nch = res.shape[0]
            res_pow = np.abs(res) ** 2
            if res_pow.ndim == 3:
                res_pow = res_pow.mean(axis=-1)
            if S.ndim == res.ndim + 1 and S.shape[0] == S.shape[1]:
                Sd = np.stack([S[i, i] for i in range(nch)])
            else:
                Sd = S[:nch]
            if Sd.ndim == 3:
                Sd = Sd.mean(axis=-1)

            settings = getattr(ac.sens_mat, "basis_settings", None)
            f = _h(settings.f_arr) if hasattr(settings, "f_arr") else None
            n_layer = res_pow.shape[-1]
            if f is None or len(f) != n_layer:
                f = np.arange(n_layer, dtype=float)
            good = f > 0

            # Instrument-only reference (same psd params, foreground/sgwb
            # zeroed): the gap between this and the total model IS the
            # fitted foreground contribution per layer — without it a
            # galfor-absorbed galaxy and no galaxy at all look identical.
            S_instr = None
            dbg = self._debug_walker_params
            if dbg is not None and (dbg[2] is not None or dbg[3] is not None):
                try:
                    Si = _h(self._build_sensitivity_for_walker(
                        dbg[0], dbg[1], None, None)[:])
                    if Si.ndim == res.ndim + 1 and Si.shape[0] == Si.shape[1]:
                        S_instr = np.stack([Si[i, i] for i in range(nch)])
                    else:
                        S_instr = Si[:nch]
                    if S_instr.ndim == 3:
                        S_instr = S_instr.mean(axis=-1)
                except Exception as e:
                    logger.warning(
                        "[%s_DEBUG] instrument-only reference skipped: %r",
                        self._debug_prefix, e,
                    )

            os.makedirs(self.debug_plot_dir, exist_ok=True)
            move = getattr(self, "gf_move_name", self._debug_prefix.lower())
            labels = ("X", "Y", "Z")
            fig, axes = plt.subplots(nch, 1, figsize=(8, 2.6 * nch), sharex=True)
            axes = np.atleast_1d(axes)
            for i in range(nch):
                ch = labels[i] if i < len(labels) else str(i)
                axes[i].loglog(f[good], res_pow[i][good], lw=0.8,
                               label=f"residual E[w^2] ({ch})")
                axes[i].loglog(f[good], np.real(Sd[i])[good], "k--", lw=1.0,
                               label="noise model S (total)")
                if S_instr is not None:
                    axes[i].loglog(f[good], np.real(S_instr[i])[good], "r:",
                                   lw=1.0, label="instrument-only S")
                axes[i].legend(fontsize=7, loc="upper right")
                axes[i].set_ylabel("power")
            axes[-1].set_xlabel("f [Hz]")
            axes[0].set_title(f"{move}: walker {w}, propose {self._debug_step}")
            fname = os.path.join(
                self.debug_plot_dir,
                f"{move}_psd_vs_res_{self._debug_step:05d}.png",
            )
            fig.savefig(fname, dpi=110, bbox_inches="tight")
            plt.close(fig)
            logger.info("[%s_DEBUG] wrote %s", self._debug_prefix, fname)
        except Exception as e:  # debug hooks must never kill a run
            logger.warning(
                "[%s_DEBUG] psd-vs-residual plot skipped: %r",
                self._debug_prefix, e,
            )

    @property
    def dcga(self) -> DomainComputationGroupArray:
        """The multi-device DCGA the move was built with (``None`` on the stock runs)."""
        return self._dcga

    @property
    def run_async(self) -> bool:
        """Forwarded to the DCGA likelihood (``run_async`` constructor argument)."""
        return self._run_async

    @property
    def run_threaded(self) -> bool:
        """Force threaded per-split dispatch (``run_threaded`` constructor argument)."""
        return self._run_threaded

    @property
    def build_threads(self) -> int:
        """Threads used for the ACA route's per-walker sensitivity builds.

        1 (default) keeps the historical serial loop. Higher values spread
        one batch's builds over :attr:`build_pool`; the batching in
        :meth:`compute_log_like` already guarantees one row per walker per
        batch, so the workers touch disjoint ``acs[w]``.

        This pays off only because the work releases the GIL: the LAT
        nanobind kernels are declared with
        ``nb::call_guard<nb::gil_scoped_release>()`` and the composite
        assembly is numpy. Measured on the stock noise grid (0.3-8 mHz =>
        59x1024 active cells): 3.11 ms/walker serial -> 1.73 at 4 threads
        (1.8x) -> 1.78 at 6, i.e. it **saturates at ~4**. The arrays are
        small enough that Python-level orchestration, not the kernels, sets
        the floor, so do not expect linear scaling and do not raise this
        past ~4 hoping for more.

        Output is bitwise-identical to the serial path: the rows are
        independent and nothing is reduced across them.
        """
        return self._build_threads

    @property
    def build_pool(self) -> ThreadPoolExecutor:
        """Lazy pool for :attr:`build_threads` (mirrors ``AnalysisContainerArray``)."""
        if self._build_pool is None:
            self._build_pool = ThreadPoolExecutor(max_workers=self._build_threads)
        return self._build_pool

    @property
    def _skip_linear_psd_repack(self) -> bool:
        """True when the ACA route can leave ``acs.linear_psd_arr`` untouched.

        ``reset_linear_psd_arr`` repacks EVERY container's ``invC`` (a
        ``(3,3,*basis)`` array — 4.5 MB each on the stock noise grid) into the
        ACA's contiguous buffer and rebinds ``sens_mat.invC`` to a view of it.
        The buffer exists for ONE reason: so a single C++/CUDA kernel can take
        one flat pointer spanning all walkers and index it by ``data_index``.
        Its only consumers are ``DomainComputationGroupArray`` (multi-GPU
        replicas) and the GB chunked-heterodyne kernels in ``chunked_het``.

        The ACA likelihood route reads ``ac.sens_mat.invC`` per container and
        never touches the buffer, so on a run with no such kernel the repack is
        pure memcpy — measured at 16.1 ms/call, ~4.7 s of an ~11 s sampling
        iteration on the stock noise fit (~40% of it).

        .. warning::
           **TODO (WDM C++ likelihood kernel).** This skip is only valid while
           the WDM basis has NO batched C++ likelihood kernel — that absence is
           exactly why this move falls back to the ACA route
           (:meth:`_kernel_fast_path_available` accepts FD/STFT only). When the
           WDM kernel lands in ``domains.cu`` it WILL read
           ``linear_psd_arr[0]``, and this must go back to repacking — or the
           kernel will silently score a stale inverse covariance. That failure
           mode is quiet and has bitten before (the walker-0 frozen-readout bug,
           2026-07-10, documented in ``chunked_het.py``). Delete this property
           and its call sites in the same commit that adds the kernel.

           Gating on the backend name is DELIBERATELY conservative but is NOT a
           proof of "no consumer": ``chunked_het`` is backend-dispatched and
           runs on CPU too, so a CPU run that puts a GB chunked-het move and
           this move on ONE ACA would read a stale buffer. That configuration
           does not exist today (the stock CPU noise fits carry no GB branch),
           and the ``_dcga`` guard below covers the multi-GPU consumer.
        """
        if self._dcga is not None:
            return False  # DCGA replicas read the buffer directly
        backend = getattr(self.sensitivity_backend, "backend", None)
        return str(getattr(backend, "name", "")).endswith("_cpu")

    def __getstate__(self):
        """Drop the thread pool from pickles / deepcopies (sprint pickle rule)."""
        state = dict(self.__dict__)
        state["_build_pool"] = None
        state["_fixed_component_covariances"] = {}
        state["_fixed_total_covariances"] = {}
        state["_fixed_total_loglikes"] = None
        state["_fixed_total_loglike_frequency_terms"] = None
        # runtime device arrays (per-device bases/modulations/extras);
        # purely derived state, rebuilt on first use in the copy
        state["_batch_runtime"] = {}
        return state

    # ------------------------------------------------------------------
    # sampled-vs-fixed noise-branch resolution
    # ------------------------------------------------------------------

    def _noise_model_branches(self, state) -> list:
        """Noise branches present in ``state`` (the full noise MODEL)."""
        return [key for key in self.NOISE_BRANCHES if key in state.branches_coords]

    def _resolve_sampled(self, state) -> list:
        """Branches THIS move samples; validated against the state's model."""
        model = self._noise_model_branches(state)
        if self.sampled_branches is None:
            return model
        missing = [b for b in self.sampled_branches if b not in model]
        if missing:
            raise ValueError(
                f"PSDMove configured to sample {self.sampled_branches} but branch(es) "
                f"{missing} are not in the state (noise model: {model})."
            )
        return list(self.sampled_branches)

    def _merged_noise_rows(self, coords, logp_keep, walker_inds_keep) -> dict:
        """Per-row full-noise-model coordinates (sampling basis).

        Sampled branches come from the proposal ``coords`` dict (rows
        surviving the prior cut); fixed branches come from the cold-row
        snapshot taken at ``propose`` entry, indexed by each row's physical
        walker. This is the single seam both likelihood routes consume.
        """
        merged = {}
        for key in self.NOISE_BRANCHES:
            if key in coords:
                merged[key] = coords[key][logp_keep][:, 0]
            elif key in self._fixed_noise_coords:
                merged[key] = self._fixed_noise_coords[key][walker_inds_keep]
        return merged

    def _cold_noise_log_prior(self, state) -> np.ndarray:
        """Sum of ALL noise-branch priors at the cold-row coordinates.

        Written into the engine ``log_prior[0]`` so the cold prior does not
        flip-flop between per-branch partial priors as split noise moves run
        in sequence within a stage.
        """
        lp = None
        for key in self._noise_model_branches(state):
            coords_b = np.asarray(state.branches_coords[key][0])
            ndim = coords_b.shape[-1]
            vals = (
                self.priors[key]
                .logpdf(coords_b.reshape(-1, ndim))
                .reshape(coords_b.shape[:2])
                .sum(axis=-1)
            )
            lp = vals if lp is None else lp + vals
        return lp

    # ------------------------------------------------------------------
    # stft_tof kernel fast path
    # ------------------------------------------------------------------

    def transform_coords(self, coords: list, return_cupy: bool = False, xp=None):
        """Map sampling-basis noise rows to the kernel fast path's physical basis.

        Kernel fast path only (FD / STFT basis; :meth:`psd_log_like`). Applies
        ``psd_transform_fn`` / ``galfor_transform_fn`` when set.

        Args:
            coords: ``[psd_rows]`` or ``[psd_rows, galfor_rows]``, each
                ``(n, ndim)``. Without galfor rows a negligible foreground
                ``[1e-200, 1e-3, 1.0, 1.0, 1.0]`` is tiled in.
            return_cupy: Return arrays converted with ``xp.asarray``. Default
                ``False``.
            xp (module, optional): Array module; default ``self.acs.xp``.

        Returns:
            tuple: ``(psd_pars, galfor_pars)``, each ``(n, ndim)``.
        """
        if xp is None:
            xp = self.acs.xp  # Use the array library from the analysis container

        if self.psd_transform_fn is not None:
            psd_pars = self.psd_transform_fn.both_transforms(coords[0])
        else:
            psd_pars = coords[0]

        if len(coords) == 1:
            galfor_pars = np.tile(np.array([1e-200, 1e-3, 1.0, 1.0, 1.0]), (psd_pars.shape[0], 1))
        else:
            if self.galfor_transform_fn is not None:
                galfor_pars = self.galfor_transform_fn.both_transforms(coords[1])
            else:
                galfor_pars = coords[1]

        if return_cupy:
            psd_pars = xp.asarray(psd_pars)
            galfor_pars = xp.asarray(galfor_pars)

        return psd_pars, galfor_pars

    def prepare_likelihood_inputs(self, psd_pars, galfor_pars) -> tuple:
        """Unpack physical-basis rows into the C++ XYZ kernel's argument tuple.

        Kernel fast path only. Every component is copied into its own
        allocation (no aliasing of the input buffers while an async kernel
        runs). With ``sensitivity_backend.use_splines`` the spline knot
        positions / amplitudes (columns ``3::2`` / ``2:-1:2``) are split into
        the two noise levels and sorted by position.

        Args:
            psd_pars (xp.ndarray): ``(n, >=2)`` physical PSD rows (column 0
                ``Soms_d``, 1 ``Sa_a``, then spline knots when used).
            galfor_pars (xp.ndarray): ``(n, 5)`` rows unpacked as
                ``[Amp, alpha, f_1, kn, f_2]`` (see the ``galfor_transform_fn`` note
                in the class docstring).

        Returns:
            tuple: nine arrays, ``Soms_d``, ``Sa_a``, ``Amp``, ``alpha``,
            ``f_1``, ``kn``, ``f_2``, knot positions, knot amplitudes; the last
            two are ``None`` without splines.
        """

        xp = self.acs.xp  # Use the appropriate array library (numpy or cupy)

        # Use .copy() (not xp.ascontiguousarray) to guarantee each kernel input
        # owns an independent allocation. ascontiguousarray is a no-op when the
        # slice is already contiguous — which is the case for a column of a
        # Fortran-order 2D array. In that case Soms_d_in_all and Sa_a_in_all
        # would alias psd_pars's buffer, and the CUDA kernel would read through
        # those aliased pointers — exposing it to cupy memory-pool reuse
        # hazards while the kernel runs asynchronously. .copy() forces a
        # fresh, contiguous, owning allocation.
        Soms_d_in_all = psd_pars[:, 0].copy()
        Sa_a_in_all = psd_pars[:, 1].copy()

        if self.sensitivity_backend.use_splines:
            knots_positions = xp.asarray(psd_pars[:, 3::2])
            knots_amplitudes = xp.asarray(psd_pars[:, 2:-1:2])
            half = knots_positions.shape[1] // 2  # Get the mid of the array
            # put the 2 noise levels on the batch axis
            spline_knots_amplitude = xp.stack(
                (knots_amplitudes[:, :half], knots_amplitudes[:, half:])
            )
            spline_knots_position = xp.stack(
                (knots_positions[:, :half], knots_positions[:, half:])
            )

            # Sort
            sort_indices = xp.argsort(spline_knots_position, axis=2)
            # Apply the same indices to both arrays
            spline_knots_position = xp.take_along_axis(
                spline_knots_position, sort_indices, axis=2
            )
            spline_knots_amplitude = xp.take_along_axis(
                spline_knots_amplitude, sort_indices, axis=2
            )
            # now check if any knot position is not too close together
        else:
            spline_knots_position = None
            spline_knots_amplitude = None

        # Same defensive copy for the galactic-foreground components. The
        # previous code unpacked `*xp.ascontiguousarray(galfor_pars.T)` which,
        # for an already-contiguous input, would unpack 5 views into the same
        # base array — same aliasing hazard as Soms/Sa. Copying each row
        # independently gives 5 owning 1D arrays.
        galfor_pars_T = galfor_pars.T  # view; transpose is always free
        Amp_all = galfor_pars_T[0].copy()
        alpha_all = galfor_pars_T[1].copy()
        f_1_all = galfor_pars_T[2].copy()
        kn_all = galfor_pars_T[3].copy()
        f_2_all = galfor_pars_T[4].copy()

        likelihood_args = (
            Soms_d_in_all,
            Sa_a_in_all,
            Amp_all,
            alpha_all,
            f_1_all,
            kn_all,
            f_2_all,
            spline_knots_position,
            spline_knots_amplitude,
        )

        return likelihood_args

    def psd_log_like(self, x: list, supps=None, **sens_kwargs) -> np.ndarray:
        """Score noise rows through the C++ XYZ sensitivity kernel (stft_tof fast path).

        Used by :meth:`_score_rows` when :meth:`_kernel_fast_path_available`
        (FD / STFT basis, single shard or a DCGA, no sgwb branch); never on a
        WDM run. With a DCGA it routes to :meth:`_psd_log_like_dcga`. Rows whose
        spline knots sit closer than ``tolerance`` score ``-1e300``.

        Args:
            x: ``[psd_rows]`` or ``[psd_rows, galfor_rows]`` (sampling basis).
            supps (dict): ``{"walker_inds": (n,) ints}``, the walker (data row)
                of each row. Required.
            **sens_kwargs: ``psd_kwargs``; accepted and not used by either
                path.

        Returns:
            np.ndarray: ``(n,)`` log-likelihoods (host).

        Raises:
            ValueError: ``supps`` is ``None``.
        """
        if supps is None:
            raise ValueError("Must provide supps to identify the data streams.")

        if self._dcga is not None and not getattr(self, "_force_parent_path", False):
            # GPU-spread path: per-device replicas via the DCGA this move was
            # constructed with (one move structure; see __init__).
            return self._psd_log_like_dcga(x, supps=supps, **sens_kwargs)

        xp = self.acs.xp  # Use the appropriate array library (numpy or cupy)

        wi = supps["walker_inds"]
        data_index_all = xp.asarray(wi).astype(np.int32)

        psd_pars, galfor_pars = self.transform_coords(x, return_cupy=True, xp=xp)

        likelihood_args = self.prepare_likelihood_inputs(psd_pars, galfor_pars)

        ll = self.sensitivity_backend.compute_log_like(
            self.acs.linear_data_arr[0], data_index_all, *likelihood_args
        )

        if likelihood_args[-2] is not None and likelihood_args[-1] is not None:
            invalid_knots = xp.any(
                xp.diff(10 ** likelihood_args[-2], axis=2) < self.tolerance, axis=(0, 2)
            )
        else:
            invalid_knots = xp.zeros(psd_pars.shape[0], dtype=bool)

        ll[invalid_knots] = -1e300

        return ll.get() if hasattr(ll, "get") else ll

    def _kernel_fast_path_available(self, has_sgwb: bool = False) -> bool:
        """True when the C++ XYZ sensitivity kernel can score this proposal.

        Conditions: a sensitivity backend is configured, the data basis is FD
        or STFT (the kernel's two domains — the WDM counterpart is planned in
        domains.cu), the ACA holds a single shard (the kernel reads
        ``linear_data_arr[0]`` directly), and the noise MODEL carries no sgwb
        branch — the kernel has no SGWB component, so scoring an sgwb-bearing
        model through it (sampled or fixed) would silently drop that term.
        """
        if has_sgwb:
            return False
        if self.sensitivity_backend is None:
            return False
        basis = self.sensitivity_backend.basis_settings
        if not isinstance(basis, (FDSettings, STFTSettings)):
            return False
        # A DCGA-configured move scores each shard on its own replica, so
        # the kernel path stays available on multi-shard ACAs.
        return self._dcga is not None or len(self.acs.linear_data_arr) == 1

    # ------------------------------------------------------------------
    # dev ACA path + hybrid dispatch
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # multi-GPU walker-shard plumbing (ACA route)
    # ------------------------------------------------------------------
    # The ACA shards walkers across devices (``acs.gpu_map`` /
    # ``acs.gpu_splits``); scoring already runs per shard inside the owning
    # device context (``AnalysisContainerArray._vectorized_dispatch``). The
    # helpers below give the per-walker sensitivity BUILDS the same
    # discipline: each build enters the owning walker's ``device_context``
    # and evaluates against a per-device domain-settings replica (the
    # ``source_runtime`` precedent), so nothing reads the primary device's
    # caches cross-device. CPU / single-GPU paths are byte-identical to the
    # historical behavior (device ``None`` -> nullcontext + shared objects).

    def _walker_device(self, walker_index: int):
        """The CUDA device owning walker ``walker_index``'s shard (None on CPU)."""
        # getattr chain: some unit tests build the move via ``__new__`` with
        # no ACA at all — treat that like the CPU path.
        gpus = getattr(getattr(self, "acs", None), "gpus", None)
        if gpus is None:
            return None
        return int(self.acs.gpu_map[int(walker_index)])

    def _backend_settings_for_device(self, device):
        """Per-device replica of the sensitivity backend's domain settings.

        CPU / single-GPU / the run's primary device return the backend's own
        (shared) ``basis_settings`` unchanged — zero extra memory, identical
        objects. A non-primary device returns the cached device-local replica
        from ``source_runtime._device_local_domain_settings_on`` (built once
        per (settings, device) and kept for the run), whose WDM window /
        fold caches live on that device.
        """
        backend = self.sensitivity_backend
        settings = getattr(backend, "basis_settings", None)
        gpus = getattr(getattr(self, "acs", None), "gpus", None)
        if settings is None or device is None or gpus is None:
            return settings
        primary = int(gpus[0])
        if int(device) == primary:
            return settings
        from ..stock.erebor.source_runtime import (
            _device_local_domain_settings_on,
        )

        return _device_local_domain_settings_on(
            settings, self.acs.xp, int(device), primary
        )

    @staticmethod
    def _to_physical(transform_fn, params):
        """One branch's sampling-basis row -> the physical basis the model wants.

        ``both_transforms`` can hand back a leading axis for a single row, so
        squeeze it the way :meth:`SensitivityBackendBase.__call__` does.
        """
        if transform_fn is None or params is None:
            return params
        out = transform_fn.both_transforms(np.asarray(params, dtype=float))
        return np.atleast_1d(np.asarray(out).squeeze())

    def _build_sensitivity_for_walker(
        self, walker_index: int, psd_params, galfor_params, sgwb_params=None
    ):
        """Build the per-walker sensitivity matrix for the given parameters.

        Takes every branch's row in ITS SAMPLING BASIS and applies that
        branch's transform here, so a transformed branch behaves the same on
        this route as on the kernel fast path (:meth:`transform_coords`) --
        the backend only ever knows about the psd transform, so a log-sampled
        galfor/sgwb would otherwise reach the model unexponentiated.

        Routes through the configured :class:`XYZSensitivityBackend`. The
        returned :class:`SensitivityMatrix` is what will be installed on the
        AnalysisContainer for the matching walker when we accept proposals
        (see :meth:`propose`).

        Multi-GPU: the build runs inside the walker's owning
        ``device_context`` and (composite backend) against a per-device
        domain-settings replica, so the produced matrix — covariance,
        ``invC``, ``detC`` — is resident on the walker's shard device.
        """
        # ``sgwb_params`` is only forwarded when present so the legacy
        # XYZSensitivityBackend (whose __call__ has no sgwb kwarg) keeps
        # working for runs without an sgwb branch.
        psd_params = self._to_physical(self.psd_transform_fn, psd_params)
        galfor_params = self._to_physical(self.galfor_transform_fn, galfor_params)
        sgwb_params = self._to_physical(self.sgwb_transform_fn, sgwb_params)
        fixed = getattr(self, "_fixed_component_covariances", {}).get(
            int(walker_index), {}
        )
        if fixed and hasattr(self.sensitivity_backend, "matrix_from_params"):
            return self.sensitivity_backend.matrix_from_params(
                f"walker_{walker_index}",
                None if "psd" in fixed else psd_params,
                galfor_params=None if "galfor" in fixed else galfor_params,
                sgwb_params=None if "sgwb" in fixed else sgwb_params,
                fixed_covariances=fixed,
            )

        extra = {} if sgwb_params is None else dict(sgwb_params=sgwb_params)
        dev = self._walker_device(walker_index)
        settings_here = self._backend_settings_for_device(dev)
        if settings_here is not None and settings_here is not getattr(
            self.sensitivity_backend, "basis_settings", None
        ):
            # only forwarded when it differs so duck-typed / legacy backends
            # without the kwarg keep working on the shared-settings path
            extra["basis_settings"] = settings_here
        xp = getattr(getattr(self, "acs", None), "xp", np)
        with device_context(xp, dev):
            # NB: psd_params / galfor_params were already mapped to physical
            # units at the top of this method (needed by the frozen-component
            # route above), so they must NOT be transformed a second time here.
            return self.sensitivity_backend(
                f"walker_{walker_index}",
                psd_params,
                galfor_params=galfor_params,
                **extra,
            )

    def _fixed_component_cache_available(self) -> bool:
        """Whether exact additive-component caching is safe on this route."""
        if self._dcga is not None:
            return False
        backend = getattr(self.sensitivity_backend, "backend", None)
        return (
            str(getattr(backend, "name", "")).endswith("_cpu")
            and hasattr(self.sensitivity_backend, "component_covariance")
        )

    def _prepare_fixed_component_covariances(self) -> None:
        """Build each frozen branch once per walker for this proposal block."""
        self._fixed_component_covariances = {}
        self._fixed_total_covariances = {}
        self._fixed_total_loglikes = None
        self._fixed_total_loglike_frequency_terms = None
        if not self._fixed_noise_coords or not self._fixed_component_cache_available():
            return

        transforms = {
            "psd": self.psd_transform_fn,
            "galfor": self.galfor_transform_fn,
            "sgwb": self.sgwb_transform_fn,
        }
        tasks = [
            (branch, w, rows[w])
            for branch, rows in self._fixed_noise_coords.items()
            for w in range(len(rows))
        ]

        def _one(task):
            branch, w, params = task
            physical = self._to_physical(transforms[branch], params)
            covariance = self.sensitivity_backend.component_covariance(
                branch, physical, name=f"fixed_{branch}_{w}"
            )
            return branch, int(w), covariance

        # Warm shared basis/spectral caches without a race, then spread the
        # remaining independent rows exactly like the ordinary build path.
        results = []
        task_count = len(tasks)
        if tasks:
            first = _one(tasks.pop(0))
            estimated_bytes = int(getattr(first[2], "nbytes", 0)) * task_count
            if estimated_bytes > self.FIXED_COMPONENT_CACHE_MAX_BYTES:
                logger.info(
                    "skipping frozen noise-component cache: estimated %.1f MiB "
                    "exceeds %.1f MiB limit",
                    estimated_bytes / 1024**2,
                    self.FIXED_COMPONENT_CACHE_MAX_BYTES / 1024**2,
                )
                return
            results.append(first)
            # per-device warm bookkeeping (see :meth:`_build_batch`); this
            # route is CPU-only, so the one task warms the whole set.
            self._build_warmed.add(self._walker_device(first[1]))
        if self._build_threads > 1 and len(tasks) > 1:
            results.extend(self.build_pool.map(_one, tasks))
        else:
            results.extend(_one(task) for task in tasks)
        for branch, w, covariance in results:
            self._fixed_component_covariances.setdefault(w, {})[branch] = covariance

        # A galfor-only split move changes no other model component.  Keep the
        # exact full-band frozen sum so its likelihood can serve as a baseline;
        # each proposal then recomputes only the layers where foreground power
        # measurably changes that sum.  Fixed extras are not represented in the
        # branch cache, so retain the ordinary full-band path when present.
        if (
            self.sampled_branches == ["galfor"]
            and not getattr(self.sensitivity_backend, "extra_components", ())
        ):
            required = set(self._fixed_noise_coords)
            for w, components in self._fixed_component_covariances.items():
                if set(components) != required:
                    self._fixed_total_covariances = {}
                    break
                ordered = [
                    components[key]
                    for key in self.NOISE_BRANCHES
                    if key in components
                ]
                total = ordered[0]
                for covariance in ordered[1:]:
                    total = total + covariance
                self._fixed_total_covariances[w] = total

    def _coarse_batch_fast_path_available(self) -> bool:
        """True for the exact CPU coarse-WDM batched reduction."""
        if not self._fixed_component_cache_available():
            return False
        if not hasattr(self.sensitivity_backend, "covariance_from_params"):
            return False
        acs = self.acs.flatten()
        if len(acs) == 0 or getattr(acs[0], "coarse_stats", None) is None:
            return False
        stat = acs[0].coarse_stats
        return all(getattr(ac, "coarse_stats", None) is stat for ac in acs)

    def _galfor_subband_fast_path_available(self) -> bool:
        """Whether a galfor-only move can score against a frozen baseline."""
        return (
            self._coarse_batch_fast_path_available()
            and self.sampled_branches == ["galfor"]
            and len(self._fixed_total_covariances) == len(self.acs.flatten())
            and hasattr(self.sensitivity_backend, "galfor_coarse_profile")
            and hasattr(
                self.sensitivity_backend,
                "galfor_coarse_profile_estimate",
            )
            and hasattr(
                self.sensitivity_backend,
                "galfor_coarse_covariance_from_profile",
            )
        )

    def _galfor_profile_frequency_mask(self, profile, fixed_covariance):
        """Layers whose foreground diagonal is non-negligible vs ``fixed``."""
        column, modulation = profile
        column = np.asarray(column)
        modulation = np.asarray(modulation)
        if modulation.ndim == 2:
            modulation_diagonal = np.diag(modulation)[:, None]
        else:
            modulation_diagonal = np.stack(
                [modulation[channel, channel] for channel in range(3)], axis=0
            )
        fixed_diagonal = np.real(
            np.stack(
                [fixed_covariance[channel, channel] for channel in range(3)],
                axis=0,
            )
        )
        numerator = (
            np.abs(modulation_diagonal)[:, None, :]
            * np.abs(column)[None, :, None]
        )
        denominator = np.abs(fixed_diagonal)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(
                denominator > 0.0,
                numerator / denominator,
                np.where(numerator > 0.0, np.inf, 0.0),
            )
        return np.any(
            ratio > self.GALFOR_SUBBAND_REL_TOL, axis=(0, 2)
        )

    def _compute_galfor_subband_loglike(
        self, stat, walker_inds_keep, galfor_coords
    ):
        """Full baseline plus band-limited foreground likelihood corrections."""
        from ...coarsewdm import (
            coarse_wdm_log_likelihood_batch,
            coarse_wdm_log_likelihood_batch_frequency_terms,
        )

        nwalkers = len(self.acs.flatten())
        if self._fixed_total_loglikes is None:
            fixed_batch = np.stack(
                [self._fixed_total_covariances[w] for w in range(nwalkers)],
                axis=0,
            )
            self._fixed_total_loglike_frequency_terms = np.asarray(
                coarse_wdm_log_likelihood_batch_frequency_terms(
                    stat, fixed_batch
                )
            )
            self._fixed_total_loglikes = np.sum(
                self._fixed_total_loglike_frequency_terms, axis=1
            )

        out = np.empty(walker_inds_keep.shape[0], dtype=float)
        remaining = np.arange(walker_inds_keep.shape[0])
        while remaining.size:
            _, first = np.unique(walker_inds_keep[remaining], return_index=True)
            batch = remaining[np.sort(first)]
            physical_params = []
            masks = []
            for row in batch:
                row = int(row)
                w = int(walker_inds_keep[row])
                physical = self._to_physical(
                    self.galfor_transform_fn, galfor_coords[row]
                )
                estimate = (
                    self.sensitivity_backend.galfor_coarse_profile_estimate(
                        physical
                    )
                )
                if estimate is None:
                    return None
                physical_params.append(physical)
                masks.append(
                    self._galfor_profile_frequency_mask(
                        estimate, self._fixed_total_covariances[w]
                    )
                )

            support = np.any(masks, axis=0)
            guard = self.GALFOR_SUBBAND_GUARD_LAYERS
            if guard and np.any(support):
                expanded = support.copy()
                for shift in range(1, guard + 1):
                    expanded[shift:] |= support[:-shift]
                    expanded[:-shift] |= support[shift:]
                support = expanded
            frequency_indices = np.flatnonzero(support)
            walkers = np.asarray(walker_inds_keep[batch], dtype=int)
            if frequency_indices.size == 0:
                out[batch] = self._fixed_total_loglikes[walkers]
            else:
                profiles = [
                    self.sensitivity_backend.galfor_coarse_profile(
                        physical,
                        frequency_indices=frequency_indices,
                    )
                    for physical in physical_params
                ]
                if any(profile is None for profile in profiles):
                    return None
                fixed_subband = np.stack(
                    [
                        self._fixed_total_covariances[w][
                            :, :, frequency_indices, :
                        ]
                        for w in walkers
                    ],
                    axis=0,
                )
                foreground = np.stack(
                    [
                        self.sensitivity_backend.galfor_coarse_covariance_from_profile(
                            profile, frequency_indices
                        )
                        for profile in profiles
                    ],
                    axis=0,
                )
                fixed_subband_logl = np.sum(
                    self._fixed_total_loglike_frequency_terms[
                        walkers[:, None], frequency_indices[None, :]
                    ],
                    axis=1,
                )
                total_subband_logl = coarse_wdm_log_likelihood_batch(
                    stat,
                    fixed_subband + foreground,
                    frequency_indices=frequency_indices,
                )
                out[batch] = (
                    self._fixed_total_loglikes[walkers]
                    + np.asarray(total_subband_logl)
                    - np.asarray(fixed_subband_logl)
                )
            remaining = np.setdiff1d(remaining, batch)
        return out

    def _build_covariance_batch(
        self, batch, walker_inds_keep, psd_coords, galfor_coords, sgwb_coords
    ):
        """Build raw covariances for one-walker-per-row coarse batch."""
        def _one(row):
            row = int(row)
            w = int(walker_inds_keep[row])
            psd_here = self._to_physical(self.psd_transform_fn, psd_coords[row])
            galfor_here = (
                None
                if galfor_coords is None
                else self._to_physical(self.galfor_transform_fn, galfor_coords[row])
            )
            sgwb_here = (
                None
                if sgwb_coords is None
                else self._to_physical(self.sgwb_transform_fn, sgwb_coords[row])
            )
            fixed = self._fixed_component_covariances.get(w, {})
            return self.sensitivity_backend.covariance_from_params(
                f"walker_{w}",
                None if "psd" in fixed else psd_here,
                galfor_params=None if "galfor" in fixed else galfor_here,
                sgwb_params=None if "sgwb" in fixed else sgwb_here,
                fixed_covariances=fixed,
            )

        rows = list(batch)
        covariances = []
        if rows:
            # serial per-device cache warm (see :meth:`_build_batch`); this
            # route is CPU-only, so the first row warms every later one.
            dev = self._walker_device(int(walker_inds_keep[int(rows[0])]))
            if dev not in self._build_warmed:
                self._build_warmed.add(dev)
                covariances.append(_one(rows.pop(0)))
        if self._build_threads > 1 and len(rows) > 1:
            covariances.extend(self.build_pool.map(_one, rows))
        else:
            covariances.extend(_one(row) for row in rows)
        return np.stack(covariances, axis=0)

    # ------------------------------------------------------------------
    # All-source coarse sidecar (plan-2): surrogate scoring seam
    # ------------------------------------------------------------------
    @property
    def coarse_sidecar_active(self) -> bool:
        """True when this move carries a live sidecar runtime (mode != off)."""
        rt = self.coarse_runtime
        return rt is not None and getattr(rt, "mode", "off") != "off"

    def _resolve_mode_like_fns(self):
        """``(mode, init_fn, model_fn, swap_fn)`` for the current coarse mode.

        ``init_fn`` seeds the working state's log-likes (the state invariant),
        ``model_fn`` scores eryn's stage-1 proposals, ``swap_fn`` feeds the
        temperature swaps. off: fine everywhere. search_approx: coarse
        everywhere (an optimizer of the surrogate, not a posterior sampler).
        delayed_acceptance: FINE state invariant + coarse stage-1 + fine
        swaps — stage 2 in :meth:`_propose_delayed_acceptance` closes the
        kernel on the fine target.
        """
        mode = self.coarse_runtime.mode if self.coarse_sidecar_active else "off"
        if mode == "auto":
            # Stage-dependent (user ruling 2026-08-28): optimise the surrogate
            # while searching, sample the exact target while doing PE. The kind
            # is stamped by the running stage's GFCombineMove immediately
            # before this propose (stock moves are shared by name, so it cannot
            # be static). "rj" counts as a search: gb_search is a search by
            # name and intent, and it is where the wall time is. Unstamped
            # (a bare move outside any stage) falls back to the EXACT mode.
            kind = getattr(self, "gf_stage_kind", None)
            mode = "search_approx" if kind in ("search", "rj") else "delayed_acceptance"
        if mode == "search_approx":
            return (
                mode,
                self.compute_coarse_log_like,
                self.compute_coarse_log_like,
                self.compute_coarse_log_like,
            )
        if mode == "delayed_acceptance":
            return (
                mode,
                self.compute_log_like,
                self.compute_coarse_log_like,
                self.compute_log_like,
            )
        return mode, self.compute_log_like, self.compute_log_like, self.compute_log_like

    def _prepare_fixed_component_covariances_coarse(self) -> None:
        """Coarse twin of :meth:`_prepare_fixed_component_covariances`.

        Frozen-branch covariances on the COARSE grid for the sidecar's
        candidate assembly. Rebuilt every proposal block (the cold rows a
        preceding noise move accepted may have moved); coarse arrays are ~Q
        times smaller than fine, so this is cheap and stays serial.
        """
        self._fixed_component_covariances_coarse = {}
        if not self.coarse_sidecar_active or not self._fixed_noise_coords:
            return
        backend = self.coarse_runtime.coarse_backend
        transforms = {
            "psd": self.psd_transform_fn,
            "galfor": self.galfor_transform_fn,
            "sgwb": self.sgwb_transform_fn,
        }
        for branch, rows in self._fixed_noise_coords.items():
            for w in range(len(rows)):
                physical = self._to_physical(transforms[branch], rows[w])
                covariance = backend.component_covariance(
                    branch, physical, name=f"coarse_fixed_{branch}_{w}"
                )
                self._fixed_component_covariances_coarse.setdefault(int(w), {})[
                    branch
                ] = covariance

    def _build_coarse_covariance_batch(
        self, batch, walker_inds_keep, psd_coords, galfor_coords, sgwb_coords
    ):
        """Coarse candidate covariances for one one-row-per-walker batch."""
        runtime = self.coarse_runtime
        backend = runtime.coarse_backend

        def _one(row):
            row = int(row)
            w = int(walker_inds_keep[row])
            psd_here = self._to_physical(self.psd_transform_fn, psd_coords[row])
            galfor_here = (
                None
                if galfor_coords is None
                else self._to_physical(self.galfor_transform_fn, galfor_coords[row])
            )
            sgwb_here = (
                None
                if sgwb_coords is None
                else self._to_physical(self.sgwb_transform_fn, sgwb_coords[row])
            )
            fixed = self._fixed_component_covariances_coarse.get(w, {})
            return backend.covariance_from_params(
                f"coarse_walker_{w}",
                None if "psd" in fixed else psd_here,
                galfor_params=None if "galfor" in fixed else galfor_here,
                sgwb_params=None if "sgwb" in fixed else sgwb_here,
                fixed_covariances=fixed,
            )

        rows = [_one(row) for row in batch]
        return runtime.xp.stack([runtime.xp.asarray(r) for r in rows], axis=0)

    def compute_coarse_log_like(
        self, coords, inds=None, logp=None, supps=None, branch_supps=None
    ):
        """Surrogate PSD/GALFOR log-likelihood against the coarse sidecar.

        Same eryn row semantics as :meth:`compute_log_like` (prior mask,
        ``walker_inds`` row mapping, frozen-branch merge), scored against the
        sidecar runtime's per-walker coarse statistics. Never touches
        ``state.log_like``, any container's ``sens_mat``, or the packed ACA
        buffer — the values live only in move-local arrays.
        """
        if not self.coarse_sidecar_active:
            raise RuntimeError(
                "compute_coarse_log_like needs a live coarse sidecar runtime "
                "(coarse_runtime with mode != 'off')."
            )
        if logp is None:
            logp = self.compute_log_prior(
                coords, inds=inds, supps=supps, branch_supps=branch_supps
            )
        logl = np.full_like(logp, -1e300)
        logp_keep = ~np.isinf(logp)
        if not np.any(logp_keep):
            return logl, None
        if supps is None:
            raise ValueError("Must provide supps to identify the data streams.")

        walker_inds_all = np.asarray(supps.holder["walker_inds"]).reshape(logp.shape)
        walker_inds_keep = walker_inds_all[logp_keep].astype(int)
        merged = self._merged_noise_rows(coords, logp_keep, walker_inds_keep)
        psd_coords = merged["psd"]
        galfor_coords = merged.get("galfor")
        sgwb_coords = merged.get("sgwb")

        # Rows grouped by owning device (None = CPU / single device), then by
        # unique walker within the group — each batch builds and scores under
        # its device's context against that device's statistic rows.
        tmp_logl = np.empty(walker_inds_keep.shape[0], dtype=float)
        row_device = [self._walker_device(int(w)) for w in walker_inds_keep]
        for device in sorted(set(row_device), key=repr):
            rows_dev = np.asarray(
                [i for i, d in enumerate(row_device) if d == device], dtype=int
            )
            remaining = rows_dev
            while remaining.size:
                _, first = np.unique(walker_inds_keep[remaining], return_index=True)
                batch = remaining[np.sort(first)]
                with device_context(self.acs.xp, device):
                    covariances = self._build_coarse_covariance_batch(
                        batch,
                        walker_inds_keep,
                        psd_coords,
                        galfor_coords,
                        sgwb_coords,
                    )
                tmp_logl[batch] = np.asarray(
                    asnumpy(
                        self.coarse_runtime.coarse_log_like_batch(
                            covariances,
                            walker_inds_keep[batch],
                            device=device,
                        )
                    ),
                    dtype=float,
                )
                remaining = np.setdiff1d(remaining, batch)
        logl[logp_keep] = tmp_logl
        return logl, None

    def _build_batch(self, batch, walker_inds_keep, psd_coords, galfor_coords, sgwb_coords):
        """Install the proposed sensitivity matrix for every row in ``batch``.

        ``batch`` holds at most one row per walker (see the batching in
        :meth:`compute_log_like`), so each row owns its ``acs[w]`` outright
        and the rows are independent. With :attr:`build_threads` > 1 they
        run concurrently in :attr:`build_pool`.

        The first row seen for EACH device is always built serially: the
        sensitivity backend fills walker-independent caches on its first
        call (``CompositeSensitivityBackend._instrument_basis_cache`` -- the
        per-WDM-time-slice fold under ``--unequal-arm`` is minutes of work),
        those caches are per device (each keyed on the device / its settings
        replica), and letting N threads race into a cold cache would just do
        that work N times. Once a device is warm, every later call on it is
        a lookup.
        """
        def _one(row):
            row = int(row)
            w = int(walker_inds_keep[row])
            galfor_here = None if galfor_coords is None else galfor_coords[row]
            sgwb_here = None if sgwb_coords is None else sgwb_coords[row]
            self.acs[w].sens_mat = self._build_sensitivity_for_walker(
                w, psd_coords[row], galfor_here, sgwb_here
            )

        self._run_rows_per_split(_one, list(batch), walker_inds_keep)

    def _run_rows_per_split(self, one_fn, rows, walker_inds_keep):
        """Run ``one_fn(row)`` for every row, grouped by owning ACA split.

        The first row seen for each not-yet-warm device runs serially first
        (the per-device cache warm described in :meth:`_build_batch`); the
        rest dispatch through ``acs._run_per_split`` (the
        ``addremovemove.compute_acs_like`` / ``_vectorized_dispatch``
        primitive): serial on CPU / single split, one thread per populated
        split under ``run_threaded`` (multi-GPU). ``one_fn`` enters its own
        walker's ``device_context`` (see
        :meth:`_build_sensitivity_for_walker`), so the workers need no extra
        device pinning. Within a split, :attr:`build_threads` > 1 keeps the
        historical intra-split thread spread.

        The move's ``run_threaded`` flag forces threaded dispatch; otherwise
        the ACA's own ``run_threaded`` default decides (the
        ``compute_acs_like`` precedent).
        """
        pending = []
        for row in rows:
            dev = self._walker_device(int(walker_inds_keep[int(row)]))
            if dev not in self._build_warmed:
                # serial per-device cache warm
                self._build_warmed.add(dev)
                one_fn(row)
            else:
                pending.append(row)
        rows = pending
        if not len(rows):
            return

        def _run_group(group_rows):
            if self._build_threads > 1 and len(group_rows) > 1:
                # list() forces consumption so worker exceptions re-raise.
                list(self.build_pool.map(one_fn, group_rows))
            else:
                for row in group_rows:
                    one_fn(row)

        acs = getattr(self, "acs", None)
        if acs is None or not (
            hasattr(acs, "_run_per_split") and hasattr(acs, "_split_rows")
        ):
            _run_group(list(rows))
            return

        rows_arr = np.asarray(list(rows), dtype=int)
        split_to_pos = acs._split_rows(
            np.asarray(walker_inds_keep, dtype=int)[rows_arr]
        )

        def _worker(split, positions):
            _run_group(rows_arr[np.asarray(positions, dtype=int)])

        acs._run_per_split(
            _worker, split_to_pos,
            run_threaded=True if self._run_threaded else None,
        )

    # ------------------------------------------------------------------
    # walker-batched ACA-route scoring (PSD_BATCH; tier 3)
    # ------------------------------------------------------------------
    # The per-walker ACA route builds one CompositeSensitivityMatrix per
    # proposal row (2 basis multiplies + a WDM stochastic fold + a 3x3
    # det/inv per walker) and reads one likelihood per container -- O(nb)
    # Python-orchestrated kernel chains per scoring batch. Everything in
    # that chain is either LINEAR in per-walker scalars (the instrument
    # bases), a per-walker 1-D curve feeding a linear fold (the stochastic
    # components), or elementwise over the basis grid (_mat3x3_det_inv, the
    # likelihood sums) -- so a walker axis slots straight in and the whole
    # batch is O(1) kernel launches per device. Parity with the per-walker
    # route is exact by construction (same operations, same order); tested
    # at rtol <= 1e-12 in tests/test_psd_move_batched.py.

    def _batched_route_ready(self) -> bool:
        """Whether the PSD_BATCH walker-batched route scores this move.

        Resolved once (the run's configuration cannot change mid-run) and
        logged; ``_batch_route_reason`` records why the per-walker path was
        kept."""
        if self._batch_route_state is None:
            self._batch_route_state = ok = self._resolve_batched_route()
            if ok:
                logger.info("[PSD_BATCH] walker-batched ACA scoring: ON")
            else:
                logger.info(
                    "[PSD_BATCH] walker-batched ACA scoring: OFF (%s)",
                    self._batch_route_reason,
                )
        return self._batch_route_state

    def _resolve_batched_route(self) -> bool:
        """Check every PSD_BATCH precondition; on failure record the reason.

        False (with ``_batch_route_reason`` set) for ``PSD_BATCH=0``, a backend
        that is not a :class:`~lisatools.sensitivity.CompositeSensitivityBackend`
        with an instrument basis cache and an ``InstrumentNoise`` component, a
        coarse-WDM basis, a domain other than FD / WDM, an instrument model not
        linear in the noise levels, an ACA without the batched plumbing or with a
        channel structure other than 3 x 3, a compressing frequency-layer mask,
        or a ``likelihood_source_only`` container.
        """
        def _no(reason):
            self._batch_route_reason = reason
            return False

        if not self.psd_batch:
            return _no("PSD_BATCH=0")
        backend = self.sensitivity_backend
        if not isinstance(backend, CompositeSensitivityBackend):
            return _no(
                f"sensitivity backend is {type(backend).__name__}, "
                "not CompositeSensitivityBackend"
            )
        if backend._instrument_basis_cache is None:
            return _no("instrument basis cache disabled")
        if not issubclass(backend.instrument_component_cls, InstrumentNoise):
            return _no("instrument component is not an InstrumentNoise")
        if isinstance(backend.basis_settings, CoarseWDMSettings):
            # The batched scorer gathers the FINE residual buffer and scores
            # it against ``C`` directly. On a coarse grid the model lives on
            # (Nf_active, Ncoarse) while the residual is still fine, and the
            # coarse likelihood is a different quantity anyway (it reads the
            # precomputed sufficient statistic + frozen WS dof). The coarse
            # routes in ``compute_log_like`` handle this case and run first;
            # this is the second guard so the ordering there is not the only
            # thing standing between a coarse run and a wrong likelihood.
            return _no("coarse WDM likelihood uses its own batched route")
        if not isinstance(backend.basis_settings, (FDSettings, WDMSettings)):
            return _no(
                f"unsupported domain {type(backend.basis_settings).__name__}"
            )
        try:
            if not self._batch_instrument_probe()._linear_in_noise_levels():
                return _no("instrument model is not linear in the noise levels")
        except Exception as exc:  # duck-typed test backends etc.
            return _no(f"instrument probe failed: {exc!r}")
        acs = getattr(self, "acs", None)
        for attr in (
            "linear_data_arr", "gpu_splits", "split_map", "nchannels",
            "end_shape", "shape_sens", "_run_per_split", "_split_rows",
        ):
            if not hasattr(acs, attr):
                return _no(f"ACA lacks {attr}")
        if tuple(acs.shape_sens) != (3, 3):
            return _no(f"channel structure {tuple(acs.shape_sens)} != (3, 3)")
        try:
            settings = acs.settings
        except Exception:
            settings = backend.basis_settings
        if getattr(settings, "frequency_layer_mask", None) is not None:
            # ``apply_frequency_layer_mask`` is a no-op on active-grid
            # arrays (``arr.shape[-2] == Nf_active`` — exactly what the
            # ACA's data / invC slices are); it only compresses FULL-grid
            # arrays, and the batched route does not reproduce that.
            if tuple(acs.end_shape) != tuple(settings.basis_shape_active):
                return _no(
                    "frequency_layer_mask would compress the data "
                    f"(end_shape {tuple(acs.end_shape)} != active basis "
                    f"{tuple(settings.basis_shape_active)})"
                )
        try:
            for w in range(len(acs)):
                if getattr(acs[w], "likelihood_source_only", False):
                    return _no("a container is likelihood_source_only")
        except Exception as exc:
            return _no(f"container check failed: {exc!r}")
        return True

    def _batch_instrument_probe(self):
        """A representative instrument component sharing the backend's cache.

        Built exactly the way :meth:`CompositeSensitivityBackend._build_matrix`
        builds the per-walker components (same class, same kwargs, same
        basis-cache dict, a model of the same class), so ``probe._bases``
        warms/reads the SAME cache entry the per-walker route uses and
        ``probe._linear_in_noise_levels()`` answers for the real per-walker
        models."""
        backend = self.sensitivity_backend
        model = backend.instrument_model_cls(
            1.0, 1.0, backend._orbits, f"{backend.model_name}:psd_batch_probe"
        )
        component_kwargs = dict(
            tdi_generation=backend.tdi_generation,
            model=model,
            fill_nans=backend.instrument_fill_nans,
            **backend.instrument_component_kwargs,
        )
        component_kwargs["basis_cache"] = backend._instrument_basis_cache
        return backend.instrument_component_cls(**component_kwargs)

    def _batch_device_runtime(self, device) -> dict:
        """Per-device walker-independent pieces of the batched build.

        Instrument bases (via the shared basis cache), stochastic model
        classes + modulation matrices, and the extra components' (fixed)
        covariances -- built ONCE per device inside its context and kept for
        the run (allocate-once rule); never pickled."""
        rt = self._batch_runtime.get(device)
        if rt is not None:
            return rt
        backend = self.sensitivity_backend
        settings = self._backend_settings_for_device(device)
        probe = self._batch_instrument_probe()
        B_oms, B_acc = probe._bases(settings)
        galfor_probe = GalacticForeground(
            foreground_params=(0.0,) * 5,
            modulation=backend.galfor_modulation,
            tdi_generation=backend.tdi_generation,
            stochastic_fn=backend.galfor_stochastic_fn,
        )
        sgwb_probe = SGWB(
            sgwb_params=(0.0,),
            stochastic_fn=backend.sgwb_stochastic_fn,
            tdi_generation=backend.tdi_generation,
        )
        rt = dict(
            settings=settings,
            B_oms=B_oms,
            B_acc=B_acc,
            Xsens=_XYZ_ELEMENT_SENS[backend.tdi_generation][0],
            galfor_fn=check_stochastic(backend.galfor_stochastic_fn),
            galfor_mod=galfor_probe.time_modulation(settings),
            sgwb_fn=check_stochastic(backend.sgwb_stochastic_fn),
            sgwb_mod=sgwb_probe.time_modulation(settings),
            extra_covs=[
                comp.covariance(settings) for comp in backend.extra_components
            ],
        )
        self._batch_runtime[device] = rt
        return rt

    def _batched_stochastic_mag(self, settings, param_rows, stoch_fn, Xsens, xp):
        """Per-walker stochastic-only magnitudes, batched.

        WDM: evaluates each walker's (cheap, 1-D) spectrum on the fold's
        rFFT bins, then runs ONE batched fold through the existing
        ``FDSignal.wdmtransform(is_psd=True)`` with the walkers stacked on
        the channel axis -- the exact per-walker ``_wdm_layer_psd`` math
        (see :func:`~lisatools.sensitivity.get_sensitivity`), one gather /
        window-square / sum for the whole batch. Returns ``(nb, Nf_active)``
        columns (broadcast along time downstream, mirroring the stationary
        ``repeat``). FD: per-row :func:`get_sensitivity` (1-D, cheap),
        stacked to ``(nb, Nf)``.

        Any other ``wdm_psd_method`` (production: ``layer_calibrated``, the
        64-node window quadrature) goes per row through the SAME column
        helper the per-walker components use. Until 2026-10-06 this always
        folded, so under ``layer_calibrated`` the noise moves scored galfor
        exactly while the installed per-walker sensitivity used the
        quadrature (on the 6-month grid: 4e-4 lnL at the production galfor
        estimate, up to 0.3 lnL across wide prior draws). Parity:
        ``test_compute_log_like_parity_layer_calibrated``."""
        method = getattr(self.sensitivity_backend, "wdm_psd_method", "fold")
        if isinstance(settings, WDMSettings) and method != "fold":
            from ...sensitivity import _wdm_stationary_psd_column

            rows = []
            for p in param_rows:
                col = xp.asarray(_wdm_stationary_psd_column(
                    settings, Xsens, (),
                    dict(stochastic_params=tuple(np.asarray(p, dtype=float)),
                         stochastic_function=stoch_fn, include_instrument=False),
                    method,
                ))
                col[xp.isnan(col)] = 0.0
                rows.append(col)
            return xp.stack(rows)
        if isinstance(settings, WDMSettings):
            f_full = xp.fft.rfftfreq(settings.N, settings.data_dt)
            df = float(f_full[1] - f_full[0])
            idx = settings.fold_frequency_indices
            f_idx = f_full[idx]
            psd_full = None
            for r in range(len(param_rows)):
                psd_active = xp.asarray(
                    Xsens.get_Sn(
                        f_idx,
                        stochastic_params=tuple(
                            np.asarray(param_rows[r], dtype=float)
                        ),
                        stochastic_function=stoch_fn,
                        include_instrument=False,
                    )
                )
                if psd_full is None:
                    psd_full = xp.zeros(
                        (len(param_rows), f_full.shape[0]),
                        dtype=psd_active.dtype,
                    )
                psd_full[r, idx] = psd_active
            psd_fd = FDSignal(
                psd_full,
                FDSettings(
                    f_full.shape[0], df, force_backend=settings.backend
                ),
            )
            folded = xp.real(
                psd_fd.wdmtransform(settings=settings, is_psd=True)
            )
            col = folded[:, :, 0]
            # get_sensitivity's fill_nans=0.0 on the folded PSD
            col[xp.isnan(col)] = 0.0
            return col

        from ...sensitivity import get_sensitivity

        rows = [
            xp.asarray(
                get_sensitivity(
                    settings,
                    sens_fn=Xsens,
                    stochastic_params=tuple(np.asarray(p, dtype=float)),
                    stochastic_function=stoch_fn,
                    include_instrument=False,
                    fill_nans=0.0,
                )
            )
            for p in param_rows
        ]
        return xp.stack(rows)

    @staticmethod
    def _add_modulated_mag(C, mag, mod, settings, xp):
        """Add a stochastic component's ``mag x modulation`` into ``C`` in place.

        The batched twin of :meth:`SeparableComponent.covariance` for the
        magnitude-on-every-element components (galfor / sgwb): ``C`` is
        ``(nch, nch, nb, *basis)``; ``mag`` is ``(nb, Nf)`` for the WDM
        basis (broadcast along the trailing time axis) or ``(nb, *basis)``
        generally."""
        from ...sensitivity import _basis_time_axis

        nch = C.shape[0]
        nbasis = C.ndim - 3
        if mag.ndim - 1 < nbasis:
            mag = mag[..., None]  # WDM column -> broadcast over time
        if mod is None:
            for i in range(nch):
                for j in range(nch):
                    C[i, j] += mag
            return
        mod = xp.asarray(mod)
        if mod.ndim == 2:
            for i in range(nch):
                for j in range(nch):
                    C[i, j] += mag * mod[i, j]
            return
        if mod.ndim == 3:
            time_axis = _basis_time_axis(settings)
            if time_axis is None:
                raise ValueError(
                    "time-varying modulation on a domain with no time axis"
                )
            ntime = mod.shape[2]
            shape = [1] * (1 + nbasis)  # broadcast against (nb, *basis)
            shape[1 + time_axis] = ntime
            for i in range(nch):
                for j in range(nch):
                    C[i, j] += mag * mod[i, j].reshape(shape)
            return
        raise ValueError("modulation must be 2D (nch,nch) or 3D (nch,nch,Ntime).")

    def _batched_covariance(self, rt, psd_phys, galfor_phys, sgwb_phys, xp):
        """The PSD_BATCH noise covariance ``C[3, 3, nb, *basis]``, one slice
        per physical-basis row, on the device of ``rt``.

        ``rt`` is :meth:`_batch_device_runtime` (the cached instrument bases,
        stochastic models and modulations of one device). ``psd_phys`` is
        ``(nb, >=2)``, columns 0 / 1 the ``Soms_d`` / ``Sa_a`` levels (squared
        here); ``galfor_phys`` / ``sgwb_phys`` are ``(nb, ndim)`` rows or
        ``None`` for an absent branch. The ONE model both
        :meth:`_score_split_batch` (the scorer) and
        :meth:`_noise_fisher` (the ``"fisher"`` eigen route) evaluate, so the
        Fisher is the curvature of exactly the likelihood being sampled.
        """
        settings = rt["settings"]
        nb = int(np.shape(psd_phys)[0])
        basis_nd = rt["B_oms"].ndim - 2
        # instrument: C[..] = Soms_d^2 * B_oms + Sa_a^2 * B_acc, walker axis
        # broadcast (the exact per-walker InstrumentNoise.base_covariance
        # fast path, LISAModel stores the squared levels)
        Soms2 = xp.asarray(np.asarray(psd_phys[:, 0], dtype=float) ** 2)
        Sa2 = xp.asarray(np.asarray(psd_phys[:, 1], dtype=float) ** 2)
        bshape = (1, 1, nb) + (1,) * basis_nd
        C = (
            rt["B_oms"][:, :, None] * Soms2.reshape(bshape)
            + rt["B_acc"][:, :, None] * Sa2.reshape(bshape)
        )
        if galfor_phys is not None:
            mag = self._batched_stochastic_mag(
                settings, galfor_phys, rt["galfor_fn"], rt["Xsens"], xp
            )
            self._add_modulated_mag(C, mag, rt["galfor_mod"], settings, xp)
        if sgwb_phys is not None:
            mag = self._batched_stochastic_mag(
                settings, sgwb_phys, rt["sgwb_fn"], rt["Xsens"], xp
            )
            self._add_modulated_mag(C, mag, rt["sgwb_mod"], settings, xp)
        for cov in rt["extra_covs"]:
            C += cov[:, :, None]
        return C

    def _noise_fisher(self, branch, points, pt_walker, other, widths):
        """Expected Fisher of the noise likelihood in ``branch``'s SAMPLING
        basis, one ``(d, d)`` matrix per expansion point:

            F_ij = kappa * sum_pix Tr(C^-1 dC/dx_i C^-1 dC/dx_j)

        with ``kappa = settings.logdet_factor`` (the likelihood's own
        log-det weight; this is the expected Hessian of
        ``-1/2 * 4 diff * d^T C^-1 d - kappa * log det C`` when that
        likelihood is self-consistent). Needs only the MODEL covariance --
        no residual, so it is positive semi-definite everywhere, unlike the
        observed information (34 of 55 galfor builds non-positive on the 3mo
        run). ``dC/dx`` by central differences of :meth:`_batched_covariance`
        (smooth, noise-free) at ``eigen_eps_rel`` of the prior box; pixels
        the likelihood drops (non-finite / zero det at the expansion point)
        are dropped here too.

        Args:
            branch: the noise branch the matrix is taken over.
            points: ``(n_pts, d)`` expansion points in ``branch``'s sampling
                basis.
            pt_walker: ``(n_pts,)`` walker of each point; selects the device
                (shard) the covariance is built on.
            other: ``{key: (n_pts, d_key) rows or None}`` for the OTHER noise
                branches (sampling basis), held fixed at each point; ``None``
                (or a missing key) = branch absent from the model.
            widths: ``(d,)`` prior-box widths of ``branch`` (the step unit).

        Returns:
            ``(n_pts, d, d)`` host array.

        Raises when the PSD_BATCH route is not engaged (the caller then
        falls back to the likelihood route).
        """
        from .eigen_refresh import nudge_inside, prior_box_bounds

        if not self._batched_route_ready():
            raise RuntimeError("noise Fisher needs the PSD_BATCH route")
        acs = self.acs
        xp = acs.xp
        tfs = {"psd": self.psd_transform_fn, "galfor": self.galfor_transform_fn,
               "sgwb": self.sgwb_transform_fn}
        points = np.atleast_2d(np.asarray(points, dtype=np.float64))
        n_pts, nd = points.shape
        steps = float(self.eigen_eps_rel) * np.asarray(widths, dtype=float)
        # keep every x +/- step row inside the prior box: a point within one
        # step of a bound (galfor alpha at GALFOR_ALPHA_MAX, fk at the
        # 0.8 mHz floor) is expanded one step in from that bound instead of
        # differencing the model outside its prior (same rule as the source
        # Gram route after the 9mo job-751 EMRI 3 defect, 2026-10-10). Only
        # points INSIDE the box are nudged: a live walker always is, and an
        # expansion point handed in from outside it (synthetic tests) is
        # expanded where it stands, as before.
        lo, hi = prior_box_bounds(self.priors[branch], nd)
        in_box = np.all((points >= lo) & (points <= hi), axis=1)
        if np.any(in_box):
            points = points.copy()
            points[in_box] = nudge_inside(points[in_box], steps, lo, hi)
        out = np.zeros((n_pts, nd, nd))
        for p in range(n_pts):
            # rows 2i / 2i+1 = x +/- steps[i] e_i; the last row is x itself
            X = np.repeat(points[p][None, :], 2 * nd + 1, axis=0)
            for i in range(nd):
                X[2 * i, i] += steps[i]
                X[2 * i + 1, i] -= steps[i]
            # physical-basis rows of every noise branch: ``branch`` varies,
            # the others repeat their fixed row at this point
            phys = {}
            for key in self.NOISE_BRANCHES:
                if key == branch:
                    rows = X
                elif other.get(key) is None:
                    rows = None
                else:
                    rows = np.tile(np.asarray(other[key][p], dtype=float),
                                   (X.shape[0], 1))
                phys[key] = None if rows is None else np.stack(
                    [np.asarray(self._to_physical(tfs[key], r), dtype=float) for r in rows])
            w = int(pt_walker[p])
            split = int(next(iter(acs._split_rows(np.array([w])))))
            device = None if acs.gpus is None else int(acs.gpus[split])
            with device_context(xp, device):
                rt = self._batch_device_runtime(device)
                kappa = float(getattr(rt["settings"], "logdet_factor", 1.0))
                C = self._batched_covariance(
                    rt, phys["psd"], phys.get("galfor"), phys.get("sgwb"), xp)
                # C^-1 and the kept-pixel mask at the expansion point (last row)
                detC, invC = _mat3x3_det_inv(C[:, :, -1:], xp)
                keep = (xp.isfinite(detC) & (detC != 0.0))[0]
                invC = xp.where(xp.isfinite(invC), invC, 0.0)[:, :, 0]
                # A_i = C^-1 dC/dx_i per pixel (channel axes first)
                A = []
                for i in range(nd):
                    dC = (C[:, :, 2 * i] - C[:, :, 2 * i + 1]) / (2.0 * steps[i])
                    A.append(xp.einsum("ab...,bc...->ac...", invC, dC))
                del C
                for i in range(nd):
                    for j in range(i, nd):
                        tr = xp.einsum("ab...,ba...->...", A[i], A[j])  # Tr(A_i A_j)
                        v = float(asnumpy(xp.sum(xp.where(keep, tr, 0.0))))
                        out[p, i, j] = out[p, j, i] = kappa * v
        return out

    def _score_split_batch(
        self, split, device, walkers, psd_phys, galfor_phys, sgwb_phys
    ) -> np.ndarray:
        """Score one shard's walker batch: batched build + batched likelihood.

        Must be called inside ``device``'s context. ``walkers`` are the
        (unique) physical walker indices owned by ``split``; params are
        physical-basis host rows."""
        acs = self.acs
        xp = acs.xp
        rt = self._batch_device_runtime(device)
        settings = rt["settings"]
        nb = len(walkers)
        C = self._batched_covariance(rt, psd_phys, galfor_phys, sgwb_phys, xp)

        # batched det/inv + the same sanitization as _setup_det_and_inv
        detC, invC = _mat3x3_det_inv(C, xp)
        C_nbytes = C.nbytes
        # C is only consumed by the PSD_DEBUG_CHECKS validation; with the
        # checks off, drop our reference here so the allocator can reuse the
        # covariance stack for the (temporary-heavy) sanitization + likelihood.
        sens_for_check = C if psd_debug_checks_enabled() else None
        del C

        # Unconditional in-place masked writes: the `if bool(xp.any(...))`
        # probes these replaced were a full-array reduction AND a
        # device->host sync per scoring call, to save a single extra pass we
        # pay anyway. Same values either way (non-finite invC -> 0, non-finite
        # detC -> 1); ``copyto`` keeps it allocation-free.
        xp.copyto(invC, 0.0, where=~xp.isfinite(invC))
        xp.copyto(detC, 1.0, where=~xp.isfinite(detC))

        # residual gather straight off the shard's contiguous buffer (the
        # per-container views are rebinds INTO it, so it is authoritative)
        split_rows = np.asarray(acs.gpu_splits[split], dtype=int)
        order = np.argsort(split_rows)
        intra = order[
            np.searchsorted(split_rows[order], np.asarray(walkers, dtype=int))
        ]
        shaped = acs.linear_data_arr[split].reshape(
            (len(split_rows), int(acs.nchannels)) + tuple(acs.end_shape)
        )
        # `intra` is the identity whenever the batch is the whole shard (the
        # common case): take a view instead of copying the full residual.
        if intra.shape[0] == shaped.shape[0] and np.array_equal(
            intra, np.arange(shaped.shape[0])
        ):
            res_batch = shaped
        else:
            res_batch = shaped[intra]

        if (device, nb) not in self._batch_mem_logged:
            self._batch_mem_logged.add((device, nb))
            logger.info(
                "[PSD_BATCH] device %s: batched build for %d walkers -- "
                "C %.1f MB + invC %.1f MB + detC %.1f MB + residual gather "
                "%.1f MB (total ~%.1f MB)",
                device, nb, C_nbytes / 1e6, invC.nbytes / 1e6,
                detC.nbytes / 1e6, res_batch.nbytes / 1e6,
                (C_nbytes + invC.nbytes + detC.nbytes + res_batch.nbytes) / 1e6,
            )

        ll = batched_residual_full_source_and_noise_likelihoods(
            res_batch, sens_for_check, invC, detC, settings
        )
        return asnumpy(ll)

    def _score_walker_batch(
        self, batch, walker_inds_keep, psd_coords, galfor_coords, sgwb_coords
    ) -> np.ndarray:
        """Score one one-row-per-walker batch through the batched route.

        Applies the branch transforms per row on the host (identical to the
        per-walker :meth:`_build_sensitivity_for_walker` seam), then runs one
        :meth:`_score_split_batch` per populated shard through the ACA's
        ``_run_per_split`` (threaded per split on multi-GPU)."""
        acs = self.acs
        xp = acs.xp
        batch = np.asarray(batch, dtype=int)
        walkers = np.asarray(walker_inds_keep, dtype=int)[batch]

        def _phys(transform_fn, coords):
            if coords is None:
                return None
            return np.stack([
                np.asarray(
                    self._to_physical(transform_fn, coords[int(r)]),
                    dtype=float,
                )
                for r in batch
            ])

        psd_phys = _phys(self.psd_transform_fn, psd_coords)
        galfor_phys = _phys(self.galfor_transform_fn, galfor_coords)
        sgwb_phys = _phys(self.sgwb_transform_fn, sgwb_coords)

        out = np.empty(len(batch), dtype=float)
        split_to_pos = acs._split_rows(walkers)

        def _worker(split, positions):
            positions = np.asarray(positions, dtype=int)
            device = None if acs.gpus is None else int(acs.gpus[split])
            with device_context(xp, device):
                ll = self._score_split_batch(
                    split, device, walkers[positions], psd_phys[positions],
                    None if galfor_phys is None else galfor_phys[positions],
                    None if sgwb_phys is None else sgwb_phys[positions],
                )
            out[positions] = np.asarray(ll, dtype=float)

        acs._run_per_split(
            _worker, split_to_pos,
            run_threaded=True if self._run_threaded else None,
        )
        return out

    def _compute_log_like_batched(
        self, walker_inds_keep, psd_coords, galfor_coords, sgwb_coords
    ) -> np.ndarray:
        """Batched scoring of all surviving rows, grouped one-row-per-walker.

        The grouping mirrors the per-walker route's batching loop -- here it
        bounds the batched build's memory at ``nwalkers`` covariance stacks
        per group (not ``ntemps * nwalkers``), NOT because anything is
        installed on the containers (nothing is)."""
        n = int(np.asarray(walker_inds_keep).shape[0])
        tmp_logl = np.empty(n, dtype=float)
        remaining = np.arange(n)
        while remaining.size:
            _, first = np.unique(walker_inds_keep[remaining], return_index=True)
            batch = remaining[np.sort(first)]
            tmp_logl[batch] = self._score_walker_batch(
                batch, walker_inds_keep, psd_coords, galfor_coords, sgwb_coords
            )
            remaining = np.setdiff1d(remaining, batch)
        return tmp_logl

    def compute_log_like(self, coords, inds=None, logp=None, supps=None, branch_supps=None):
        """Compute the full-noise-model log-likelihood of the proposed rows (eryn callback).

        Rows with a ``-inf`` prior keep ``-1e300``. The surviving rows are
        merged with the frozen non-sampled branches (:meth:`_merged_noise_rows`,
        each row's walker from ``walker_inds``) and scored through
        :meth:`compute_psd_rows` -> :meth:`_score_rows`, whose tiers are, in
        order: the C++ XYZ kernel fast path (FD / STFT basis, single shard; the
        stft_tof hybrid routing), the CPU coarse-WDM galfor sub-band and coarse
        batch routes, the walker-batched ``PSD_BATCH`` route (the stock WDM
        runs), and the per-walker container route (install each proposed
        noise model as the walker container's ``sens_mat`` in one-row-per-walker
        batches, read the containers' likelihoods, then restore the originals).

        Args:
            coords (dict): Branch-keyed coordinates, each
                ``(ntemps, nwalkers, 1, ndim)`` in the sampling basis.
            inds (dict, optional): Branch-keyed leaf occupancy flags (only
                forwarded to :meth:`compute_log_prior`, which ignores them).
            logp (np.ndarray, optional): Pre-computed ``(ntemps, nwalkers)`` log
                prior; computed if ``None``.
            supps (eryn.state.BranchSupplemental): Eryn supplemental information;
                ``walker_inds`` selects which physical walker each row maps to.
                Required unless every row has a ``-inf`` prior.
            branch_supps (dict, optional): Branch supplementals (unused here).

        Returns:
            tuple[np.ndarray, None]: ``(logl, None)``, ``logl`` shaped like ``logp``
            (also kept on :attr:`prev_logl` when any row survives the prior cut).

        Raises:
            ValueError: ``supps`` is ``None`` while some row has a finite prior.
        """
        if logp is None:
            logp = self.compute_log_prior(coords, inds=inds, supps=supps, branch_supps=branch_supps)

        assert logp is not None
        logl = np.full_like(logp, -1e300)

        logp_keep = ~np.isinf(logp)
        if not np.any(logp_keep):
            warnings.warn("All points entering likelihood have a log prior of minus inf.")
            return logl, None

        if supps is None:
            raise ValueError("Must provide supps to identify the data streams.")

        # ``walker_inds`` is broadcast (ntemps, nwalkers) — flatten and mask
        # to the rows that survived the prior cut. ``BranchSupplemental``'s
        # ``__getitem__`` expects an integer/slice index, so reach into
        # ``.holder`` to fetch the named entry.
        walker_inds_all = np.asarray(supps.holder["walker_inds"]).reshape(logp.shape)
        walker_inds_keep = walker_inds_all[logp_keep].astype(int)

        # full noise model per row: sampled branches from the proposal,
        # fixed branches from the cold-row snapshot (per walker)
        merged = self._merged_noise_rows(coords, logp_keep, walker_inds_keep)
        psd_coords = merged["psd"]
        has_galfor = "galfor" in merged
        galfor_coords = merged.get("galfor")
        has_sgwb = "sgwb" in merged
        sgwb_coords = merged.get("sgwb")

        logl[logp_keep] = self.compute_psd_rows(
            walker_inds_keep,
            psd_coords,
            galfor_coords if has_galfor else None,
            sgwb_coords if has_sgwb else None,
        )
        self.prev_logl = logl.copy()
        return logl, None

    def _score_rows(self, walker_inds_keep, psd_coords, galfor_coords, sgwb_coords):
        """Rank-local tier dispatch: ``(n,)`` log-likelihoods for merged noise rows.

        Row ``i`` scores the full noise model ``(psd, galfor, sgwb)[i]`` against
        walker ``walker_inds_keep[i]``'s residual on THIS rank's ACA. Tier order
        is unchanged from the pre-replica ``compute_log_like``: C++ kernel,
        galfor sub-band coarse, coarse batch, PSD_BATCH, container fallback.
        """
        walker_inds_keep = np.asarray(walker_inds_keep).astype(int).reshape(-1)
        n = int(walker_inds_keep.shape[0])
        if n == 0:
            return np.zeros(0, dtype=float)
        has_galfor = galfor_coords is not None
        has_sgwb = sgwb_coords is not None

        if self._kernel_fast_path_available(has_sgwb=has_sgwb):
            input_args = [psd_coords, galfor_coords] if has_galfor else [psd_coords]
            # A plain dict, not a BranchSupplemental: the old call site sliced
            # the real supps (`supps[logp_keep]`), whose __getitem__ (a
            # base-shape slicer, not a key lookup) returns exactly a
            # ``{"walker_inds": <kept rows>}`` dict -- both consumers
            # (psd_log_like's parent path and the DCGA path) only ever index
            # it by that one string key.
            supps_keep = {"walker_inds": walker_inds_keep}
            return np.asarray(
                self.psd_log_like(input_args, supps=supps_keep, **self.psd_kwargs), dtype=float
            ).reshape(n)

        # The coarse-WDM routes come FIRST: they score against the coarse
        # sufficient statistic, which the PSD_BATCH route below knows nothing
        # about (it gathers the FINE residual buffer). Both are guarded, but
        # order is the first line of defence.
        if self._galfor_subband_fast_path_available():
            stat = self.acs.flatten()[0].coarse_stats
            tmp_logl = self._compute_galfor_subband_loglike(stat, walker_inds_keep, galfor_coords)
            if tmp_logl is not None:
                return np.asarray(tmp_logl, dtype=float).reshape(n)

        if self._coarse_batch_fast_path_available():
            from ...coarsewdm import coarse_wdm_log_likelihood_batch

            stat = self.acs.flatten()[0].coarse_stats
            tmp_logl = np.empty(walker_inds_keep.shape[0], dtype=float)
            remaining = np.arange(walker_inds_keep.shape[0])
            while remaining.size:
                _, first = np.unique(walker_inds_keep[remaining], return_index=True)
                batch = remaining[np.sort(first)]
                covariances = self._build_covariance_batch(
                    batch,
                    walker_inds_keep,
                    psd_coords,
                    galfor_coords if has_galfor else None,
                    sgwb_coords if has_sgwb else None,
                )
                tmp_logl[batch] = np.asarray(
                    coarse_wdm_log_likelihood_batch(stat, covariances)
                )
                remaining = np.setdiff1d(remaining, batch)
            return tmp_logl

        if self._batched_route_ready():
            # PSD_BATCH walker-batched route (tier 3): one batched covariance
            # build + one batched likelihood per shard per scoring group.
            # Nothing is installed on the containers, so there is no
            # sens-mat snapshot/restore and no linear_psd_arr repack here
            # (the publish repack in propose() still runs and is what the
            # rest of the run reads).
            return np.asarray(
                self._compute_log_like_batched(
                    walker_inds_keep, psd_coords, galfor_coords, sgwb_coords
                ), dtype=float,
            ).reshape(n)

        # Cache and restore the per-walker sensitivity matrix so we don't
        # corrupt the state seen by other moves. After all proposals are
        # scored we put each AC's original sens_mat back; the caller (the
        # ``propose`` loop) reinstalls the accepted PSD onto the ACs.
        # There are ntemps*nwalkers proposal rows but only nwalkers ACs, and
        # ``walker_inds`` maps every rung of walker w onto acs[w]. Scoring all
        # rows in one pass would let each rung overwrite the previous one's
        # sens_mat, and the single batched ``likelihood()`` would then hand
        # every rung of a walker the SAME value (the last one written) --
        # the cold rung scored with a hot rung's parameters, so the Metropolis
        # ratio compares unrelated likelihoods and accepts nearly everything.
        # Score in batches instead, each holding a walker at most once.
        original_sens = {}
        tmp_logl = np.empty(walker_inds_keep.shape[0], dtype=float)
        try:
            remaining = np.arange(walker_inds_keep.shape[0])
            while remaining.size:
                # first remaining row for each distinct walker -> one batch
                _, first = np.unique(walker_inds_keep[remaining], return_index=True)
                batch = remaining[np.sort(first)]
                # Snapshot the originals serially before any building: the
                # threaded path overwrites acs[w].sens_mat, so this must not
                # race with it (and it is a bare attribute read anyway).
                for row in batch:
                    w = int(walker_inds_keep[row])
                    if w not in original_sens:
                        original_sens[w] = self.acs[w].sens_mat
                self._build_batch(
                    batch,
                    walker_inds_keep,
                    psd_coords,
                    galfor_coords if has_galfor else None,
                    sgwb_coords if has_sgwb else None,
                )
                # See _skip_linear_psd_repack -- WARNING there before adding a
                # WDM C++ likelihood kernel. acs.likelihood() reads
                # ac.sens_mat.invC, not the buffer, so on a no-kernel run this
                # repack is ~16 ms/call of memcpy nobody reads.
                if not self._skip_linear_psd_repack:
                    self.acs.reset_linear_psd_arr()
                walker_ll = asnumpy(np.asarray(self.acs.likelihood()))
                tmp_logl[batch] = walker_ll[walker_inds_keep[batch].astype(int)]
                remaining = np.setdiff1d(remaining, batch)
        finally:
            for w, sens in original_sens.items():
                self.acs[w].sens_mat = sens
            if not self._skip_linear_psd_repack:
                self.acs.reset_linear_psd_arr()

        return tmp_logl

    def compute_psd_rows(self, walker_inds_keep, psd_coords, galfor_coords, sgwb_coords):
        """The scoring seam: scatters the rows over the replicas in one-walker replica mode.

        Every likelihood this move needs goes through here: the proposal
        scoring (:meth:`compute_log_like`) and the ``ll`` eigen-table route
        (:meth:`_refresh_eigen_tables`). Without an active row fan-out (one-walker
        replica mode, opt-in through ``GF_GPU_ROUTING=1``, with the
        ``{P}_LIKELIHOOD_FANOUT`` knob on) this is :meth:`_score_rows` on this
        rank's ACA.

        Args:
            walker_inds_keep (np.ndarray): ``(n,)`` walker (container) index per row.
            psd_coords (np.ndarray): ``(n, ndim_psd)`` sampling-basis rows.
            galfor_coords (np.ndarray or None): ``(n, ndim_galfor)`` rows, or ``None``
                (no galfor).
            sgwb_coords (np.ndarray or None): ``(n, ndim_sgwb)`` rows, or ``None``
                (no sgwb).

        Returns:
            np.ndarray: ``(n,)`` log-likelihoods of the full noise model.
        """
        if not self.rows_active():
            return self._score_rows(walker_inds_keep, psd_coords, galfor_coords, sgwb_coords)
        # NOT WALKER-ROUTED, AND IT DOES NOT NEED TO BE (2026-09-23).
        # ``fanout_flat_body = False`` keeps this family on the BLOCK path,
        # so ``propose_local`` only ever runs against the rank's own block
        # and every row here already belongs to a walker this rank holds.
        # The unrouted contiguous split is therefore correct by
        # construction, at any number of blocks.
        #
        # The reason PSD is not flat: it derives its walker count and device
        # map from the LOCAL ACA (``len(self.acs.flatten())``,
        # ``self.acs.gpu_map[...]``), which are WRONG -- not merely slow --
        # under a full-ensemble body. Making it flat would buy a scatter
        # over R ranks of a move that is ~1% of the iteration, so the
        # de-ACA pass that would enable it is not worth its risk today.
        #
        # This assert is the guard that keeps the above true: if someone
        # flips ``fanout_flat_body``, the rows stop being block-local and
        # this fires instead of scoring against the wrong residual.
        assert not self.fanout_flat_body, (
            f"{self.fanout_knob_prefix()}: the PSD scoring seam assumes a "
            "block-local body (fanout_flat_body=False); a flat body needs "
            "the rows walker-routed first"
        )
        rows = {"walker_inds": np.asarray(walker_inds_keep).astype(np.int32).reshape(-1)}
        for key, arr in (("psd", psd_coords), ("galfor", galfor_coords), ("sgwb", sgwb_coords)):
            if arr is not None:
                rows[key] = np.asarray(arr, dtype=np.float64)
        out = self.row_fanout.run("psd_rows", rows, local_body=self._serve_psd_rows_local)
        return out["ll"]

    def _serve_psd_rows_local(self, rows):
        """Score one row chunk of a replica scatter: ``{"ll": (n,)}`` (empty-safe)."""
        n = int(np.shape(rows["walker_inds"])[0])
        if n == 0:
            return {"ll": np.zeros(0)}
        ll = self._score_rows(rows["walker_inds"], rows.get("psd"), rows.get("galfor"), rows.get("sgwb"))
        return {"ll": np.asarray(ll, dtype=float).reshape(n)}

    def serve_psd_rows(self, payload, clock, model):
        """Compute-rank handler of the ``psd_rows`` scatter (one-walker replica mode)."""
        return self._serve_psd_rows_local(payload["rows"])

    # ---- replays: every replica mirrors the head's propose-begin prep and
    # publish, gated on replica mode itself (``self.row_fanout is None``),
    # NOT on ``rows_active()`` / the ``{PREFIX}_LIKELIHOOD_FANOUT`` knob —
    # that knob only changes who SCORES rows; it must not stop the replays
    # that keep every replica's fixed-noise state / sens_mat aligned. ----
    def _replay_noise_begin(self):
        """Run the propose-begin prep (:meth:`_apply_noise_begin`) here and on every replica."""
        payload = {
            "kind": "begin",
            "fixed_noise_coords": {
                k: np.asarray(v, dtype=np.float64) for k, v in self._fixed_noise_coords.items()
            },
        }
        if self.row_fanout is None:
            self._apply_psd_replay(payload)
            return
        self.row_fanout.replay("psd_replay", payload, local_body=self._apply_psd_replay)

    def _apply_noise_begin(self, payload):
        """Install the frozen noise rows and rebuild the fixed-component caches.

        Sets ``_fixed_noise_coords`` from the payload, then
        :meth:`_prepare_fixed_component_covariances` (builds only on a CPU
        sensitivity backend with ``component_covariance`` and at least one
        frozen branch, without a DCGA; a no-op on GPU runs) and, with a coarse
        sidecar, its refresh and coarse fixed covariances.
        """
        self._fixed_noise_coords = {
            k: np.asarray(v, dtype=np.float64) for k, v in payload["fixed_noise_coords"].items()
        }
        self._prepare_fixed_component_covariances()
        if self.coarse_sidecar_active:
            self.coarse_runtime.refresh_P(self.acs)
            self._prepare_fixed_component_covariances_coarse()

    def _replay_noise_publish(self, new_state):
        """Publish the cold-row noise model of walker 0 onto every replica's container."""
        bc = new_state.branches_coords

        def _row(key):
            return np.asarray(bc[key][0, 0, 0], dtype=np.float64) if key in bc else None

        payload = {"kind": "publish", "psd": _row("psd"), "galfor": _row("galfor"), "sgwb": _row("sgwb")}
        if self.row_fanout is None:
            self._apply_psd_replay(payload)
            return
        self.row_fanout.replay("psd_replay", payload, local_body=self._apply_psd_replay)

    def _apply_psd_replay(self, payload):
        """Apply one replay payload: ``"begin"`` (frozen rows) or ``"publish"`` (walker 0's model).

        Raises ``ValueError`` on any other ``kind``.
        """
        kind = payload["kind"]
        if kind == "begin":
            self._apply_noise_begin(payload)
        elif kind == "publish":
            # Same "sens_refresh" bucket the non-replica publish path times
            # (see propose_local's _publish_one) so the profiler bucket is
            # not silently zero in replica mode; _tspan no-ops when the
            # propose-level timer is absent (e.g. a bare-replay unit test).
            with _tspan(getattr(self, "_prop_timer", None), "sens_refresh"):
                new_sens = self._build_sensitivity_for_walker(
                    0, payload["psd"], payload["galfor"], payload["sgwb"]
                )
            self.acs[0].sens_mat = new_sens
            self.acs.reset_linear_psd_arr()
        else:
            raise ValueError(f"unknown noise replay kind {kind!r}")

    def serve_psd_replay(self, payload, clock, model):
        """Compute-rank handler of the ``psd_replay`` command (replica mode); returns ``None``."""
        self._apply_psd_replay(payload)
        return None

    def _fancy_swap_fires(self, move_i, nwalkers):
        """Walker-permuting swap cadence; never with one walker (identity permutation).

        ``permute_every <= 0`` (e.g. -1) disables it; the check runs before the modulo.
        """
        return (
            int(self.permute_every) > 0
            and int(move_i) % int(self.permute_every) == 0
            and int(nwalkers) > 1
        )

    def fanout_knob_prefix(self):
        """Upper-cased lead sampled branch (``PSD`` / ``GALFOR`` / ``SGWB``) for knob names."""
        return str((self.sampled_branches or ["psd"])[0]).upper()

    def compute_log_prior(self, branches_coords, *args, **kwargs):
        """Sum the per-branch log priors over THIS move's sampled branches.

        Fixed (non-sampled) noise branches are constants during this move's
        Metropolis-Hastings updates, so their priors cancel in the acceptance
        ratio and must not be summed here (a split move using the joint prior
        would double-count them against another noise move).

        Args:
            branches_coords: Branch-keyed dict of coordinates, each
                ``(ntemps, nwalkers, 1, ndim)``.
            *args: Ignored (eryn callback signature).
            **kwargs: Ignored (eryn callback signature).

        Returns:
            np.ndarray: ``(ntemps, nwalkers)`` prior log probabilities (``None``
            when no sampled branch is in ``branches_coords``).
        """
        # wait to get ntemps, nwalkers
        logp = None
        for key in self.NOISE_BRANCHES:
            if key not in branches_coords:
                continue
            if self.sampled_branches is not None and key not in self.sampled_branches:
                continue
            ntemps, nwalkers, _, ndim = branches_coords[key].shape
            if logp is None:
                logp = np.zeros((ntemps, nwalkers))

            logp[:] += (
                self.priors[key]
                .logpdf(branches_coords[key].reshape(-1, ndim))
                .reshape(ntemps, nwalkers)
            )
        return logp

    def _propose_delayed_acceptance(self, model, state):
        """One inner-proposal repeat under two-stage (delayed) acceptance.

        Stage 1 reuses the resolved inner proposal's (eigen or stretch, see
        :meth:`_inner_propose`) own accept VERBATIM as the surrogate screen:
        the working state's log-likes are swapped to the coarse values (so
        eryn's ``logP(x)`` is the tempered coarse posterior) and ``model``
        carries :meth:`compute_coarse_log_like` (so ``logP(y)`` is too); the
        prior and the proposal's own factor therefore enter exactly once,
        in stage 1. Stage 2 corrects the stage-1 survivors with

            log alpha_2 = beta * [(Lf(y) - Lc(y)) - (Lf(x) - Lc(x))]

        (priors and factors cancel), reverting rejects to their pre-proposal
        coordinates. The returned state's ``log_like`` is FINE on every row
        — the invariant the temperature swaps and the sub-state bookkeeping
        rely on. Fine evaluations are spent only on stage-1 survivors (the
        prior-mask trick: non-survivor rows enter the fine callback with a
        ``-inf`` prior and are skipped).

        The independent stage-2 uniforms come from ``model.random`` AFTER
        eryn's stage-1 draws, so a fixed seed reproduces the full kernel.
        """
        # Pre-proposal snapshot: coords + FINE lls + priors. eryn's propose
        # may mutate ``state`` in place, so this copy is the revert source
        # and the alpha_2 reference. is_eryn_state_input: the working state
        # is the module-ladder tmp_state (dict-constructed, no sub-states);
        # the plain-GFState copy path would trip over its absent
        # sub_state_bases, and the snapshot needs none of that machinery.
        state_before = GFState(state, copy=True, is_eryn_state_input=True)
        fine_x = np.asarray(state_before.log_like)

        coarse_x = self.compute_coarse_log_like(
            state.branches_coords,
            logp=state.log_prior,
            supps=state.supplemental,
        )[0]

        # Stage 1: eryn stretch accept on the tempered coarse posterior.
        # The temperature control is DETACHED for this call: eryn's propose
        # ends with temper_comps (the per-repeat identity swap), which would
        # permute rungs between the stage-1 accept and the stage-2
        # correction. The swap is re-applied below, after the fine invariant
        # is restored — so it swaps on FINE values, as every other swap does.
        state.log_like[:] = coarse_x
        _tc = self.temperature_control
        self.temperature_control = None
        try:
            # Stage-1 draw: the eigen-axis MH move or the vanilla eryn
            # stretch, whichever this propose block resolved (see
            # :meth:`_inner_propose` for the dispatch and MRO rationale).
            new_state, accepted1 = self._inner_propose(model, state)
        finally:
            self.temperature_control = _tc
        acc = np.asarray(accepted1, dtype=bool)

        if not acc.any():
            new_state.log_like[:] = fine_x
            self._da_debug_last = None
            new_state = self.temperature_control.temper_comps(new_state)
            return new_state, acc

        # Stage 2: fine correction on the survivors only.
        coarse_y = np.asarray(new_state.log_like).copy()
        survivor_logp = np.where(acc, np.asarray(new_state.log_prior), -np.inf)
        fine_y = self.compute_log_like(
            new_state.branches_coords,
            logp=survivor_logp,
            supps=new_state.supplemental,
        )[0]
        betas = np.asarray(self.temperature_control.betas)[:, None]
        log_alpha2 = betas * ((fine_y - coarse_y) - (fine_x - coarse_x))
        u2 = model.random.uniform(size=acc.shape)
        with np.errstate(divide="ignore"):
            keep = acc & (np.log(u2) < log_alpha2)
        revert = acc & ~keep

        for key in new_state.branches:
            new_state.branches[key].coords[revert] = state_before.branches[
                key
            ].coords[revert]
        new_state.log_prior[revert] = np.asarray(state_before.log_prior)[revert]
        # FINE invariant restored: survivors carry their fresh fine values,
        # everything else keeps the pre-proposal fine values.
        new_state.log_like[:] = np.where(keep, fine_y, fine_x)

        # Surrogate-accuracy audit, free of charge: the stage-2 exponent IS
        # the coarse-vs-fine DELTA-logL error that scripts/noise/coarse_q_scan.py
        # measures offline -- a perfect surrogate gives 0 and stage 2 always
        # accepts. Record it per repeat; propose() logs one [COARSE_AUDIT] line.
        _dd = ((fine_y - coarse_y) - (fine_x - coarse_x))[acc]
        if getattr(self, "_da_audit", None) is not None and _dd.size:
            self._da_audit.append(
                (int(acc.sum()), int(keep.sum()), int(acc.size), np.abs(_dd))
            )
        self._da_debug_last = {
            "accepted_stage1": acc,
            "keep": keep,
            "fine_x": fine_x,
            "fine_y": fine_y,
            "coarse_x": coarse_x,
            "coarse_y": coarse_y,
            "log_alpha2": log_alpha2,
            "coords_final_prewap": np.array(
                new_state.branches["psd"].coords, copy=True
            )
            if "psd" in new_state.branches
            else None,
        }
        # The per-repeat identity swap eryn would have run inside propose,
        # now on the restored FINE values.
        new_state = self.temperature_control.temper_comps(new_state)
        return new_state, keep

    def run_move(self, move_i, model, state):
        """Run one in-model repeat: the inner proposal, then the temperature swaps.

        The inner proposal is :meth:`_inner_propose` (eigen-axis MH or eryn
        stretch), or :meth:`_propose_delayed_acceptance` with a coarse sidecar in
        ``delayed_acceptance`` mode. Its acceptances are added to the per-propose
        in-model tallies. Then ``temperature_control.temperature_swaps`` runs:
        identity-paired swaps every repeat, the walker-permuting swap when
        ``move_i`` is a multiple of ``permute_every`` and the block holds more
        than one walker;
        the swap acceptances are added to the swap tallies (written into the
        sub-states by :meth:`_finish_propose`).

        Args:
            move_i (int): Repeat index used to schedule the walker-permuting swap.
            model (eryn.model.Model): The working model of :meth:`_begin_propose`.
            state (GFState): Current working (module-ladder) state.

        Returns:
            tuple: ``(new_state, accepted)``, ``accepted`` shaped
            ``(ntemps_module, nwalkers)``.
        """
        if (
            self.coarse_sidecar_active
            and self.coarse_runtime.mode == "delayed_acceptance"
        ):
            new_state, accepted = self._propose_delayed_acceptance(model, state)
        else:
            # The eigen-axis MH move or the vanilla eryn stretch, whichever
            # this propose block resolved (see :meth:`_inner_propose`).
            new_state, accepted = self._inner_propose(model, state)

        # in-model bookkeeping: eryn returns (ntemps, nwalkers) acceptances and
        # every walker is proposed once per call, so the per-temperature deltas
        # are the row sums / the walker count. Tallies are reset per propose()
        # and written into the sub-states there (see delta_counter_names).
        acc = np.asarray(accepted)
        acc = acc.reshape(acc.shape[0], -1)
        if self._tally_in_model_accepted is not None:
            self._tally_in_model_accepted += acc.sum(axis=-1).astype(int)
            self._tally_in_model_proposed += acc.shape[-1]

        # Tempering, addremove-style two-tier cadence (2026-08-12): a cheap
        # IDENTITY-paired naive swap fires EVERY repeat -- valid within a
        # walker because every rung of walker j scores against walker j's own
        # residual, so the dbeta*dlogl factor is exact and costs no
        # likelihood evaluations -- and the walker-PERMUTING fancy swap
        # (re-scores the swapped params against the DESTINATION walkers'
        # residuals through compute_log_like) fires only every
        # ``permute_every`` repeats. Previously NO swap ran between fancy
        # events, so the ladder sat unexchanged for the other repeats.
        do_fancy = self._fancy_swap_fires(move_i, int(np.shape(new_state.log_like)[1]))
        x = new_state.branches_coords
        logl = new_state.log_like
        logp = new_state.log_prior
        branch_supps = new_state.branches_supplemental
        supps = new_state.supplemental

        logP = self.compute_log_posterior(logl, logp)
        x, logP, logl, logp, inds, blobs, supps, branch_supps = (
            self.temperature_control.temperature_swaps(
                x,
                logP,
                logl,
                logp,
                supps=supps,
                branch_supps=branch_supps,
                # search_approx swaps on the surrogate (it optimizes that
                # target); off/delayed_acceptance swap on the FINE values —
                # identity swaps read the fine state.log_like and fancy swaps
                # re-score through the fine callback (plan-2 §3.2).
                compute_log_like=getattr(self, "_swap_like_fn", None)
                or self.compute_log_like,
                compute_log_prior=self.compute_log_prior,
                fancy_swap=do_fancy,
                permute_here=do_fancy,
            )
        )

        for name in x:
            new_state.branches[name].coords[:] = x[name][:]
            new_state.branches[name].branch_supplemental = branch_supps[name]

        new_state.log_like[:] = logl[:]
        new_state.log_prior[:] = logp[:]
        new_state.supplemental = supps

        # swap bookkeeping: TemperatureControl refreshes swaps_accepted on
        # every call and holds swaps_proposed fixed at nwalkers per rung.
        tc = self.temperature_control
        if self._tally_swaps_accepted is not None:
            sa = getattr(tc, "swaps_accepted", None)
            sp = getattr(tc, "swaps_proposed", None)
            if sa is not None:
                self._tally_swaps_accepted += np.asarray(sa).ravel().astype(int)
            if sp is not None:
                self._tally_swaps_proposed += np.asarray(sp).ravel().astype(int)

        return new_state, accepted

    def run_move_for_loop(self, model, state, num_repeats):
        """Run :meth:`run_move` ``num_repeats`` times sequentially (the PE path).

        Args:
            model (eryn.model.Model): The working model of :meth:`_begin_propose`.
            state (GFState): Working (module-ladder) state.
            num_repeats (int): Repeat count (``self.num_repeats`` from
                :meth:`propose_local`).

        Returns:
            tuple: ``(state, accepted)`` of the last repeat.
        """
        for i in tqdm(
            range(num_repeats), desc="psd update",
            disable=not getattr(self, "progress", False),
        ):
            state, accepted = self.run_move(i, model, state)
        return state, accepted

    def run_move_max_likelihood(self, model, state):
        """Repeat :meth:`run_move_for_loop` until the max log-like plateaus.

        Used in search-style runs (``max_logl_mode=True``, the ``*_search``
        instances). The loop counts consecutive iterations during which the
        cold-chain max log-likelihood no longer increases (counting only after
        it has changed once) and exits after ``num_checks`` (hard-coded 5)
        such iterations. There is no iteration cap.

        Under several compute ranks this plateau is the RANK's walker block's
        maximum, so search termination is per block -- see the module docstring
        of :mod:`~lisatools.globalfit.moves.walkerfanout` ("Semantics that
        change").
        """
        num_checks = 5
        num_so_far = 0
        max_logl = -np.inf
        changed_once = False
        while num_so_far < num_checks:
            state, accepted = self.run_move_for_loop(model, state, self.num_repeats)

            if state.log_like[0].max() != max_logl and not np.isinf(max_logl):
                changed_once = True

            if state.log_like[0].max() > max_logl:
                max_logl = state.log_like[0].max()
                num_so_far = 0
            else:
                if changed_once:
                    num_so_far += 1

        return state, accepted

    # ---- per-walker tiled ensemble search (SEARCH stages only) ---------------
    #
    # TODO(noise-ensemble-search): this currently rides inside the existing
    # noise search proposal, gated on gf_stage_kind. If it earns its keep,
    # split it into its own stock move name so it can be selected per stage
    # and A/B'd without a knob.
    def _ensemble_search_active(self) -> bool:
        """Is the tiled per-walker ensemble search armed for THIS propose?

        Two conditions, both required: the branch knob
        ``{PREFIX}_ENSEMBLE_SEARCH`` is on, AND this propose is running
        inside a SEARCH stage.

        The stage test is ``gf_stage_kind`` -- the per-propose stamp
        ``GFCombineMove`` sets from the running ``Stage.kind``, the same
        discriminator :meth:`_resolve_mode_like_fns` already uses to pick
        the coarse sidecar mode. It must NOT be ``max_logl_mode`` and must
        NOT be the GB ``"search" in self.name`` idiom: ``JointMaxLogLSearch``
        wraps the ``*_pe`` moves, not the ``*_search`` ones, so the live
        object is literally named ``"psd pe move"`` and carries
        ``max_logl_mode=False`` -- both of those tests would silently never
        fire. Keying off the stage is also what puts the move back in
        regular mode for free: the SAME move object is reused by the
        ``noise_vgb_pe`` / ``full_pe`` stages, where ``gf_stage_kind`` is
        ``"pe"``, and takes the ordinary path there.
        """
        if not bool(getattr(self, "ensemble_search", False)):
            return False
        return getattr(self, "gf_stage_kind", None) in ("search", "rj")

    def _ensemble_fold(self, arr4, R):
        """``(nt, B, 1, ndim)`` cold state -> ``(B*nt, R, 1, ndim)`` inner ensemble.

        The row order IS Eryn's folded-sampler order: row ``b * nt + t``
        holds walker ``b``'s rung ``t``, which is what
        ``Move.sampler_id_rows`` (``np.repeat(np.arange(nsamplers),
        ntemps_per_sampler)``) and ``TemperatureControl``'s
        ``act * ntemps + i`` row arithmetic both assume. Every reshape in
        this block relies on it, so it is asserted in the tests rather than
        re-derived at each site.
        """
        nt, B = arr4.shape[:2]
        tail = arr4.shape[2:]
        rows = np.transpose(arr4, (1, 0) + tuple(range(2, arr4.ndim)))
        rows = rows.reshape((B * nt, 1) + tail)
        return np.repeat(rows, int(R), axis=1)

    def _ensemble_spread_scale(self, model, state, inner_supps, R, widths, names):
        """Closed loop on the MEASURED logL spread (user ruling 2026-09-18).

        Perturb, score, and rescale until each walker's own inner ensemble
        spans ``[ensemble_spread_lo, ensemble_spread_hi]`` log-likelihood.
        The loop is per-walker and vectorised: one batched likelihood call
        covers every walker's whole ``nt x R`` block, and the rescale
        ``scale *= sqrt(target / spread)`` is applied per walker.

        Copy 0 of every ``(walker, rung)`` is left UNPERTURBED so the
        incumbent always survives the block -- the search can then only
        improve on what it was handed, and there is always at least one
        finite point to measure a spread against when a wide draw puts the
        rest outside the prior.
        """
        nt, B = state.branches_coords[names[0]].shape[:2]
        base = {k: self._ensemble_fold(np.asarray(state.branches_coords[k]), R)
                for k in names}
        rng = model.random
        step = {k: rng.normal(size=base[k].shape) for k in names}
        for k in names:
            step[k][:, 0] = 0.0                      # keep the incumbent

        lo = float(self.ensemble_spread_lo)
        hi = float(self.ensemble_spread_hi)
        target = 0.5 * (lo + hi)
        scale = np.full(B, float(self.ensemble_scale0), dtype=float)
        coords = logp = logl = None

        for _ in range(max(1, int(self.ensemble_scale_tries))):
            rows = np.repeat(scale, nt)[:, None, None, None]
            coords = {k: base[k] + rows * widths[k] * step[k] for k in names}
            logp = self.compute_log_prior(coords)
            logl = np.asarray(
                model.compute_log_like_fn(coords, logp=logp, supps=inner_supps)[0]
            )
            blk = np.where(np.isfinite(logl), logl, np.nan).reshape(B, nt * R)
            with np.errstate(invalid="ignore"):
                spread = np.nanmax(blk, axis=1) - np.nanmin(blk, axis=1)
            # a walker with <2 finite points has no measurable spread: treat
            # it as far too wide and shrink hard rather than divide by nan.
            n_ok = np.isfinite(blk).sum(axis=1)
            spread = np.where(n_ok >= 2, spread, np.inf)
            if np.all((spread >= lo) & (spread <= hi)):
                break
            with np.errstate(divide="ignore", invalid="ignore"):
                adj = np.sqrt(target / spread)
            adj = np.where(np.isfinite(adj) & (adj > 0.0), adj, 0.25)
            scale = np.clip(scale * np.clip(adj, 0.1, 10.0), 1e-8, 1e2)

        return coords, logp, logl, scale

    def run_move_ensemble_search(self, model, state):
        """One block of the tiled per-walker ensemble search.

        For each walker of this rank's block, independently and with no
        cross-walker interaction: tile its parameters into ``R`` repeats
        (default 10), spread them by a measured 1-10 logL, run the eryn
        stretch on that inner ensemble with full ensemble-permutation
        tempering and per-walker ladder adaptation, then hand back the best
        point that walker's ``(ntemps x R)`` set reached.

        **Why this exists.** ``StretchMove`` needs a complement, which a
        one-walker block does not have (:meth:`_resolve_inner_kind` refuses
        it outright), so a 4-GPU/4-walker run falls back to the eigen-axis
        proposal -- and that path was returning a non-positive information
        matrix on 34 of 55 galfor builds. Tiling manufactures a complement
        INSIDE each walker and takes the information matrix off the search
        critical path entirely.

        **Almost nothing here is new.** The geometry is Eryn's folded
        sampler axis: outer walker -> ``nsamplers``, rung -> ``ntemps``, the
        ``R`` repeats -> ``nwalkers``. Red/blue splits and stretch
        complements are drawn per row of axis 0 and never mix rows, so an
        unmodified ``StretchMove`` on a folded ``(B*nt, R, 1, ndim)`` array
        IS "R copies per rung, each walker its own independent ensemble".
        ``TemperatureControl(permute=True, nsamplers=B)`` gives the
        full-permutation swaps within each walker's own ladder, and its
        ``temper_comps`` adapts each walker's betas from that walker's own
        swap acceptance. :meth:`compute_log_like` and
        :meth:`compute_log_prior` are already shape-agnostic, and
        :meth:`compute_psd_rows` already takes repeating walker indices, so
        the scoring seam is untouched.

        SEARCH ONLY -- the argmax at the end is a deterministic hill-climb,
        admissible under the standing search-waiver policy and gated to
        search stages by :meth:`_ensemble_search_active`.
        """
        from eryn.moves.tempering import TemperatureControl

        from .eigen_refresh import prior_box_widths

        names = [k for k in self.NOISE_BRANCHES if k in state.branches_coords]
        nt, B = state.branches_coords[names[0]].shape[:2]
        R = max(2, int(self.ensemble_repeats))
        ndim_total = sum(int(state.branches_coords[k].shape[-1]) for k in names)

        if R < 2 * ndim_total and not getattr(self, "_ens_width_logged", False):
            logger.warning(
                "[%s] ensemble search: %d repeats is below eryn's red/blue floor "
                "of 2*ndim_total=%d for branches %s -- the stretch complement is "
                "thin. Raise {PREFIX}_ENSEMBLE_REPEATS.",
                self.fanout_knob_prefix(), R, 2 * ndim_total, names,
            )
            self._ens_width_logged = True

        # data_index: every one of a walker's R*nt rows must score against
        # THAT walker's own ACA residual. Row b*nt + t -> walker b. Getting
        # this wrong scores walkers against each other's residuals (the PSD
        # fancy-swap walker_inds defect, in a new place).
        w_rows = np.repeat(np.arange(B, dtype=np.int64), nt)
        inner_supps = BranchSupplemental(
            {"walker_inds": np.tile(w_rows[:, None], (1, R))},
            base_shape=(B * nt, R),
            copy=True,
        )

        widths = {
            k: np.asarray(
                prior_box_widths(self.priors[k], int(state.branches_coords[k].shape[-1])),
                dtype=float,
            )
            for k in names
        }

        coords, logp, logl, scale = self._ensemble_spread_scale(
            model, state, inner_supps, R, widths, names
        )

        # The inner ladder is PRIVATE: a fresh control, never published and
        # never handed to the fan-out, so it adapts during the run even
        # under several compute ranks (install_walker_fanout pins only the
        # move's own control non-adaptive). Seeded from the outer ladder.
        betas0 = np.asarray(self.temperature_control.betas, dtype=float)[:nt]
        inner_tc = TemperatureControl(
            ndim_total, R,
            betas=betas0, ntemps=nt, nsamplers=B,
            permute=True,                       # full ensemble permutation
            skip_swap_supp_names=["walker_inds"],
        )
        inner = self._ensemble_inner_move()
        inner.temperature_control = inner_tc
        inner.accepted = np.zeros((B * nt, R))

        inner_state = GFState(coords, copy=True, supplemental=inner_supps)
        inner_state.log_prior = logp
        inner_state.log_like = logl

        inner_model = Model(
            model.log_like_fn,
            model.compute_log_like_fn,
            self.compute_log_prior,
            inner_tc,
            model.map_fn,
            model.random,
        )

        acc_sum = np.zeros((B * nt, R))
        for _ in range(max(1, int(self.num_repeats))):
            inner_state, acc = inner.propose(inner_model, inner_state)
            acc_sum += np.asarray(acc, dtype=float)
        n_rounds = max(1, int(self.num_repeats))

        return self._ensemble_take_best(state, inner_state, acc_sum, n_rounds, names,
                                        nt, B, R, scale)

    def _ensemble_inner_move(self):
        """The cached eryn ``StretchMove`` of the ensemble search (``periodic`` re-synced)."""
        if getattr(self, "_ensemble_inner", None) is None:
            # live_dangerously: the red/blue floor is about the OUTER
            # ensemble's dimensionality; here the complement is manufactured
            # and its width is the knob (warned about above).
            self._ensemble_inner = StretchMove(
                live_dangerously=True, periodic=self.periodic
            )
        self._ensemble_inner.periodic = self.periodic
        return self._ensemble_inner

    def _ensemble_take_best(self, state, inner_state, acc_sum, n_rounds, names,
                            nt, B, R, scale):
        """Fold the inner ensemble back onto the module ladder.

        The COLD row of each walker takes that walker's global argmax over
        its whole ``(nt x R)`` set -- the user's "take the max logL of each
        walker and temperature set of 10 and make that the parameters and
        logL to pass on", and the cold row is what the residual and the
        enclosing plateau criterion actually read. Each HOT rung takes the
        argmax over its OWN R copies instead of the global best, so the
        ladder keeps its spread rather than collapsing onto one point.
        """
        ll = np.asarray(inner_state.log_like, dtype=float).reshape(B, nt, R)
        lp = np.asarray(inner_state.log_prior, dtype=float).reshape(B, nt, R)

        bw = np.arange(B)[:, None]
        bt = np.arange(nt)[None, :]
        best_r = np.argmax(np.where(np.isfinite(ll), ll, -np.inf), axis=2)   # (B, nt)
        flat = np.where(np.isfinite(ll), ll, -np.inf).reshape(B, nt * R)
        best_t, best_r0 = np.divmod(np.argmax(flat, axis=1), R)              # (B,)
        ba = np.arange(B)

        for k in names:
            ndim_k = int(state.branches_coords[k].shape[-1])
            c = np.asarray(inner_state.branches_coords[k]).reshape(B, nt, R, 1, ndim_k)
            out = c[bw, bt, best_r]                       # (B, nt, 1, ndim)
            out[:, 0] = c[ba, best_t, best_r0]            # cold row: global best
            state.branches_coords[k][:] = np.transpose(out, (1, 0, 2, 3))

        ll_out = ll[bw, bt, best_r]
        lp_out = lp[bw, bt, best_r]
        ll_out[:, 0] = ll[ba, best_t, best_r0]
        lp_out[:, 0] = lp[ba, best_t, best_r0]
        state.log_like = np.ascontiguousarray(ll_out.T)
        state.log_prior = np.ascontiguousarray(lp_out.T)

        # bookkeeping on the OUTER move, so acceptance_fraction stays finite
        # and the sub-state counters are not silently zero (the same reason
        # _inner_propose mirrors RedBlueMove's own bookkeeping by hand).
        acc = (acc_sum.reshape(B, nt, R).sum(axis=2) / (n_rounds * R)).T
        self.accepted += acc
        self.num_proposals += 1
        self._tally_in_model_proposed += n_rounds * R * B
        self._tally_in_model_accepted += (
            acc_sum.reshape(B, nt, R).sum(axis=(0, 2)).astype(int)
        )
        sa = getattr(self._ensemble_inner.temperature_control, "swaps_accepted", None)
        sp = getattr(self._ensemble_inner.temperature_control, "swaps_proposed", None)
        if sa is not None and sp is not None:
            self._tally_swaps_accepted += (
                np.asarray(sa).reshape(B, -1).sum(axis=0).astype(int)
            )
            self._tally_swaps_proposed += (
                np.asarray(sp).reshape(B, -1).sum(axis=0).astype(int)
            )
        if not getattr(self, "_ens_logged", False):
            logger.info(
                "[%s] ensemble search: %d repeats x %d rungs x %d walkers, "
                "perturbation scale %s prior-box fractions",
                self.fanout_knob_prefix(), R, nt, B,
                np.array2string(scale, precision=3),
            )
            self._ens_logged = True
        return state, acc

    # ---- inner proposal: stretch or eigen-axis MH ----------------------------
    def _resolve_inner_kind(self, nwalkers_block, nwalkers_run=None) -> str:
        """Resolve the inner proposal kind for one propose.

        ``nwalkers_block`` is the per-rank BLOCK width this propose is
        actually working with (what the stretch-complement check must key
        off — a 1-walker block cannot stretch even on a many-walker run).
        ``nwalkers_run`` is the RUN's total walker count (defaults to
        ``nwalkers_block`` for single-process / non-fanout callers); it is
        logged for context only. The ``inner_move_kind is None`` default
        resolves off the BLOCK width (2026-09-17 ruling): ``"eigen"`` when
        the block holds fewer than three walkers, else ``"stretch"``. The
        stretch's complement is the LOCAL block under the walker-block layout
        (cross-rank complement pooling is WP8, not built), so a 1-walker block
        has no complement at all and a 2-walker block's complement is a single
        point (the proposal degenerates to a line through it) -- both are what
        a 4-GPU, 4-walker campaign run produces (one walker per rank), and
        the one-walker replica mode (run nwalkers == 1) is the same case.
        An explicit ``{P}_INNER_MOVE_KIND`` always wins (``stretch`` at a
        1-walker block is still refused below). Logs the resolved kind ONCE
        per move.
        """
        if nwalkers_run is None:
            nwalkers_run = nwalkers_block
        kind = self.inner_move_kind
        if kind is None:
            kind = "eigen" if int(nwalkers_block) < 3 else "stretch"
        kind = str(kind).strip().lower()
        if kind not in ("eigen", "stretch"):
            prefix = self.fanout_knob_prefix()
            raise ValueError(
                f"PSDMove {prefix}_INNER_MOVE_KIND {self.inner_move_kind!r}: "
                "use 'eigen' or 'stretch'"
            )
        if kind == "stretch" and int(nwalkers_block) == 1:
            prefix = self.fanout_knob_prefix()
            raise ValueError(
                "PSDMove: the stretch inner proposal needs at least two walkers in the "
                f"block (there is no complement with one); block nwalkers={nwalkers_block}, "
                f"run nwalkers={nwalkers_run}; set {prefix}_INNER_MOVE_KIND=eigen"
            )
        if not getattr(self, "_inner_kind_logged", False):
            logger.info(
                "[%s] resolved PSD inner proposal kind -> %r (block nwalkers=%d, "
                "run nwalkers=%d)", self.fanout_knob_prefix(), kind,
                int(nwalkers_block), int(nwalkers_run),
            )
            self._inner_kind_logged = True
        return kind

    def _eigen_inner_move(self):
        """The cached eryn ``EigenAxisMove`` (axis mode), synced to this move's ladder and wrap."""
        if self._eigen_inner is None:
            from eryn.moves import EigenAxisMove

            self._eigen_inner = EigenAxisMove(mode="axis", periodic=self.periodic)
        # follow the move's control (None during the delayed-acceptance stage 1)
        # and periodic wrap on every call, not just at first construction.
        self._eigen_inner.temperature_control = self.temperature_control
        self._eigen_inner.periodic = self.periodic
        return self._eigen_inner

    def _eigen_scope(self):
        """``{PREFIX}_EIGEN_SCOPE``: ``cold_per_walker`` (default) or ``per_temp``.

        ``cold_per_walker`` (2026-09-18 user ruling) builds the information
        matrix ONLY on the COLD row, one table per walker, and shares each
        walker's table up its own ladder. Two reasons:

        * **Correctness.** The hot rungs sit far from the mode, where the
          log-likelihood genuinely has negative-curvature directions -- the
          3-month run logged ``8/12 matrices had a non-positive eigenvalue
          (worst lambda/lambda_max = -inf) -> projected onto the PSD cone``,
          12 being galfor's ntemps. Expanding only at the cold row removes
          that class of table entirely.
        * **Cost.** Builds drop from ``ntemps`` to ``nwalkers`` points --
          12 -> 1 for galfor/psd at one walker per rank.

        It also fixes a smaller latent issue: the legacy path expanded at
        every rung but scored all of them against walker 0's residual.

        ``per_temp`` restores the pre-2026-09-18 behaviour exactly (every
        rung at walker 0, shared across walkers, no sigma tempering) for
        A/B work. Mirrors ``AddRemove._eigen_scope``'s knob spelling.
        """
        val = getattr(self, "eigen_table_scope", None)
        if val is None:
            val = os.environ.get(f"{self._debug_prefix}_EIGEN_SCOPE", None)
        if val is None:
            return "cold_per_walker"
        val = str(val).strip().lower()
        if val not in ("cold_per_walker", "per_temp"):
            logger.warning(
                "[%s] unknown EIGEN_SCOPE %r (use 'cold_per_walker' or "
                "'per_temp'); using 'cold_per_walker'",
                self._debug_prefix, val,
            )
            return "cold_per_walker"
        return val

    def _eigen_temper_sigmas(self) -> bool:
        """``{PREFIX}_EIGEN_TEMPER_SIGMAS`` (default ON under cold_per_walker).

        Widens the shared cold sigma by ``1/sqrt(beta)`` per rung. Only
        meaningful when a cold table is reused up the ladder; see
        :func:`~lisatools.globalfit.moves.eigen_refresh.temper_sigmas` for
        why omitting it freezes the hot chains. ``0`` hands every rung the
        bare cold sigma, which is what ``AddRemove``'s ``walker_max`` scope
        still does (tracked separately).
        """
        raw = os.environ.get(f"{self._debug_prefix}_EIGEN_TEMPER_SIGMAS")
        if raw is None:
            return True
        return str(raw).strip().lower() not in ("0", "false", "no", "")

    def _eigen_info(self, branch):
        """Information-matrix route for ``branch``'s eigen tables
        (lower-cased): ``eigen_info`` attribute > ``{BRANCH}_EIGEN_INFO`` env
        > ``"ll"``.

        ``"fisher"``: the expected noise Fisher (:meth:`_noise_fisher`);
        anything else: likelihood second differences. The attribute applies
        to every branch this move samples; the env is read per branch
        (``PSD_EIGEN_INFO``, ``GALFOR_EIGEN_INFO``, ``SGWB_EIGEN_INFO``).
        """
        val = getattr(self, "eigen_info", None)
        if val is None:
            val = os.environ.get(f"{str(branch).upper()}_EIGEN_INFO", "ll")
        return str(val).strip().lower()

    def _refresh_eigen_tables(self, tmp_branches_coords):
        """Per-branch eigen tables, installed on the eigen inner move.

        Expansion points follow :meth:`_eigen_scope`: ``cold_per_walker``
        (each walker's cold row, shared up its ladder with tempered sigmas)
        or the legacy ``per_temp`` (every rung at walker 0).

        Information matrix per point, by :meth:`_eigen_info` per branch:
        ``"ll"`` (default) -- likelihood second differences; ``call_ll`` for
        branch ``b`` varies only ``b``'s parameters, the other sampled
        branches sit at their own point values (rows arrive as whole
        ``n_pts``-point blocks, the batching invariant of
        ``information_matrix_from_ll``) and the fixed branches at the cold
        row; scoring goes through :meth:`compute_psd_rows`, so it scatters
        too. ``"fisher"`` -- the expected Fisher (:meth:`_noise_fisher`) at
        the same points with the same held branches; on any error it warns
        and takes the ll route.
        """
        from .eigen_refresh import (
            _tables_from_info_batch,
            eigen_tables_from_ll_batch,
            prior_box_bounds,
            prior_box_widths,
            temper_sigmas,
        )

        names = list(tmp_branches_coords)
        first = np.asarray(tmp_branches_coords[names[0]])
        ntemps, nwalkers = int(first.shape[0]), int(first.shape[1])
        inner = self._eigen_inner_move()
        cold_only = self._eigen_scope() == "cold_per_walker"
        # EXPANSION POINTS. cold_per_walker: the COLD row, one point per
        # walker -> nwalkers builds. per_temp (legacy): every rung at walker
        # 0 -> ntemps builds, shared across walkers.
        sl = (0, slice(None)) if cold_only else (slice(None), 0)
        point = {
            b: np.asarray(tmp_branches_coords[b], dtype=np.float64)[sl][:, 0, :]
            for b in names
        }
        n_pts = int(point[names[0]].shape[0])
        # Which walker's data each expansion point belongs to. Under
        # cold_per_walker each point is that walker's own cold state, so it
        # must be scored against THAT walker's residual/PSD; the legacy path
        # scored every rung against walker 0, which is what it expanded at.
        pt_walker = (np.arange(n_pts, dtype=np.int32) if cold_only
                     else np.zeros(n_pts, dtype=np.int32))
        fixed = {k: np.asarray(v, dtype=np.float64)[0] for k, v in self._fixed_noise_coords.items()}
        for b in names:
            ndim_b = int(point[b].shape[1])
            widths = prior_box_widths(self.priors[b], ndim_b)

            def call_ll(x, _b=b):
                x = np.atleast_2d(np.asarray(x, dtype=np.float64))
                # information_matrix_from_ll batches whole n_pts-point
                # blocks in row order, so per-point metadata tiles by
                # rows // n_pts (the same invariant the addremove builder
                # relies on).
                idx = np.arange(x.shape[0]) % n_pts
                rows = {}
                for key in self.NOISE_BRANCHES:
                    if key == _b:
                        rows[key] = x
                    elif key in point:
                        rows[key] = point[key][idx]
                    elif key in fixed:
                        rows[key] = np.tile(fixed[key], (x.shape[0], 1))
                return self.compute_psd_rows(
                    pt_walker[idx],
                    rows.get("psd"), rows.get("galfor"), rows.get("sgwb"),
                )

            axes = None
            if self._eigen_info(b) == "fisher":
                # {BRANCH}_EIGEN_INFO=fisher: the expected Fisher at the SAME
                # expansion points, the other noise branches held where
                # call_ll holds them (sampled: their own point rows; fixed:
                # the cold row; absent: None). Any failure -> the ll route.
                try:
                    _t0 = time.perf_counter()
                    other = {}
                    for k in self.NOISE_BRANCHES:
                        if k == b:
                            continue
                        if k in point:
                            other[k] = point[k]
                        elif k in fixed:
                            other[k] = np.tile(fixed[k], (n_pts, 1))
                        else:
                            other[k] = None
                    info = self._noise_fisher(b, point[b], pt_walker, other, widths)
                    axes, sigmas = _tables_from_info_batch(info, widths)
                    logger.info("[eigen_refresh] %s: noise Fisher tables at %d point(s) "
                                "in %.2f s", b, n_pts, time.perf_counter() - _t0)
                except Exception as exc:  # degrade to the likelihood route
                    logger.warning("[eigen_refresh] %s noise Fisher failed (%r); using "
                                   "likelihood second differences", b, exc)
                    axes = None
            if axes is None:
                # bounds: the second-difference corners stay inside the
                # prior box (a point within eps of a bound is expanded one
                # step in), as on the source branches' routes
                axes, sigmas = eigen_tables_from_ll_batch(
                    call_ll, point[b], widths, eps_rel=self.eigen_eps_rel,
                    bounds=prior_box_bounds(self.priors[b], ndim_b),
                )
            if cold_only:
                # one table per walker, shared up the ladder; the sigmas are
                # widened per rung because EigenAxisMove applies no beta
                # scaling of its own (see eigen_refresh.temper_sigmas).
                axes5 = np.broadcast_to(
                    axes[None, :, None], (ntemps, nwalkers, 1, ndim_b, ndim_b)
                ).copy()
                sig4 = np.broadcast_to(
                    sigmas[None, :, None], (ntemps, nwalkers, 1, ndim_b)
                )
                if self._eigen_temper_sigmas():
                    sig4 = temper_sigmas(
                        sig4, np.asarray(self.temperature_control.betas)
                    )
                sig4 = np.ascontiguousarray(sig4)
            else:
                axes5 = np.broadcast_to(
                    axes[:, None, None], (ntemps, nwalkers, 1, ndim_b, ndim_b)
                ).copy()
                sig4 = np.broadcast_to(
                    sigmas[:, None, None], (ntemps, nwalkers, 1, ndim_b)
                ).copy()
            inner.set_axes(b, axes5, sig4)

    def _inner_propose(self, model, state):
        """One in-model step: the eigen-axis MH move or the vanilla eryn stretch.

        ``StretchMove.propose`` is named outright (it IS ``RedBlueMove.propose``)
        rather than reached through the MRO past the fan-out mixin -- this is
        the sole call site for both the stage-1 delayed-acceptance draw and
        the ordinary ``run_move`` step, so both routes get the same dispatch.
        """
        if self._inner_kind == "eigen":
            inner = self._eigen_inner_move()
            ntemps, nwalkers = next(iter(state.branches_coords.values())).shape[:2]
            # EigenAxisMove IS an MHMove: its propose() ends with
            # ``self.accepted += accepted``, so the inner move's OWN
            # (otherwise unused) accepted counter must be initialized before
            # its first use -- eryn's ensemble normally does this once per
            # move before running it; here nothing else does, since the
            # inner move is never handed to an ensemble directly.
            if getattr(self, "_eigen_inner_accepted_shape", None) != (ntemps, nwalkers):
                inner.accepted = np.zeros((ntemps, nwalkers))
                self._eigen_inner_accepted_shape = (ntemps, nwalkers)
            out_state, out_acc = inner.propose(model, state)
            # Mirror RedBlueMove.propose (Eryn red_blue.py:326-327): advance
            # the OUTER move's own accepted/num_proposals too, not just the
            # inner move's -- otherwise Move.acceptance_fraction sees
            # num_proposals == 0 forever (zeros/0 -> RuntimeWarning + all-nan,
            # poisoning the enclosing CombineMove's average at every
            # checkpoint). `self.accepted` is sized to the module ladder
            # shape by propose_local before any run_move call.
            self.accepted += np.asarray(out_acc)
            self.num_proposals += 1
            return out_state, out_acc
        return StretchMove.propose(self, model, state)

    # ---- multi-rank fan-out (WalkerFanoutMixin hooks) ------------------------
    # Multi-rank semantics this family changes (walkerfanout module docstring,
    # "Semantics that change"): the search-mode plateau is per walker block,
    # the pooled ladder adapts from run_move's explicit swap tallies (a
    # different swap population from single mode's), and the control's own
    # swaps_accepted stored by eryn/HDF is the head block's count.
    def fanout_temperature_controls(self):
        """The one ladder this move adapts (frozen on the ranks by the fan-out install)."""
        return [self.temperature_control]

    def fanout_payload_extra(self):
        """Head -> ranks: the current ``betas`` (and a pending ratchet shift, if any)."""
        extra = {"betas": np.array(self.temperature_control.betas, dtype=float, copy=True)}
        if self._forced_step_deltas is not None:
            # the galfor ratchet's forced shift: every rank applies the same
            # vector to its own walker block (host numpy, like the ladder)
            extra["forced_step_deltas"] = {
                k: np.asarray(v, dtype=float).copy()
                for k, v in self._forced_step_deltas.items()}
        return extra

    def fanout_apply_extra(self, extra):
        """Rank side: rebind the shipped ``betas``; set (or clear) the forced shift."""
        if "betas" in extra:
            # rebind, never write in place: the control may alias the recipe's betas array
            self.temperature_control.betas = np.asarray(extra["betas"], dtype=float).copy()
        # ALWAYS assigned: a rank must never carry a forced shift into the
        # next ordinary propose because the head did not ship a fresh one
        fd = extra.get("forced_step_deltas")
        self._forced_step_deltas = (
            None if fd is None
            else {str(k): np.asarray(v, dtype=float) for k, v in dict(fd).items()})

    @staticmethod
    def _tally_or_empty(arr):
        """``arr`` as a flat float array, or an empty one for ``None``."""
        return np.zeros(0) if arr is None else np.asarray(arr, dtype=float).ravel()

    def fanout_reply_extra(self, part):
        """Rank -> head: this block's per-propose swap tallies (accepted, proposed)."""
        return {
            "swaps_accepted": self._tally_or_empty(self._tally_swaps_accepted),
            "swaps_proposed": self._tally_or_empty(self._tally_swaps_proposed),
        }

    def fanout_merge_extra(self, replies, new_state):
        """Pool swap tallies over ranks, adapt the shared ladder ONCE, publish it.

        The body copies ``tc.betas`` into every sampled sub-state
        (``sub.betas[:] = ...``) but ``merge_walkers`` never writes ladders back,
        so the head republishes the adapted ladder here.

        The pooled ratio is built from ``run_move``'s explicit
        ``temperature_swaps`` tallies (fancy swaps included), a different swap
        population from the single-process ladder's -- see the module docstring
        of :mod:`~lisatools.globalfit.moves.walkerfanout` ("Semantics that
        change").
        """
        if not replies:
            return
        tc = self.temperature_control
        acc = sum(np.asarray(e["swaps_accepted"], dtype=float) for e in replies.values())
        prop = sum(np.asarray(e["swaps_proposed"], dtype=float) for e in replies.values())
        # rebind, never write in place: the control may alias the recipe's betas array
        tc.betas = pooled_ladder_step(tc, np.array(tc.betas, copy=True), acc, prop)
        for key in (self.fanout_branches or self.sampled_branches or []):
            sub = (getattr(new_state, "sub_states", None) or {}).get(key)
            if sub is not None and getattr(sub, "betas", None) is not None:
                sub.betas[:] = tc.betas

    def propose_local(self, model, state):
        """Propose a noise-model update and refresh per-walker sensitivity matrices.

        Builds a temporary :class:`GFState` containing only THIS move's
        sampled noise branches (non-sampled noise branches are frozen at
        their cold rows and merged into every likelihood evaluation), runs
        the inner loop -- :meth:`run_move_max_likelihood` with
        ``max_logl_mode``, else :meth:`run_move_ensemble_search` when
        :meth:`_ensemble_search_active`, else :meth:`run_move_for_loop` (the
        production path) -- then writes the accepted coordinates back into a
        copy of ``state`` (full ladder into the sampled branches' sub-states,
        cold row mirrored into the main state) and refreshes each walker's
        sensitivity matrix in :attr:`acs`. See the class docstring for the
        full contract.

        Runs on the FULL state single-process and on this rank's walker slice
        under several compute ranks (``WalkerFanoutMixin.propose`` dispatches).

        The begin (working state) and finish (write-back + publish) halves
        are :meth:`_begin_propose` / :meth:`_finish_propose`, shared with
        :meth:`forced_noise_step` so a forced step publishes through exactly
        this path.

        Args:
            model (eryn.model.Model): The eryn model (``analysis_container_arr``,
                ``map_fn``, ``random``).
            state (GFState): The current state (this rank's block under the
                fan-out).

        Returns:
            tuple: ``(new_state, accepted)``, ``accepted`` cut to the engine
            ladder (the cold row).

        Raises:
            ValueError: A configured sampled branch is missing from the state,
                the sampled branches' ladders differ in shape, or a cold-row
                check fails.
        """
        deltas = self._forced_step_deltas
        ctx = self._begin_propose(model, state, resolve_inner=deltas is None)
        tmp_state, tmp_model, _tm = ctx.tmp_state, ctx.tmp_model, ctx.tm

        if deltas is not None:
            # THE FORCED STEP (galfor ratchet). No Metropolis loop: shift the
            # working coordinates on every rung and walker of THIS body's
            # slice, recompute prior and likelihood, accept unconditionally.
            # Runs wherever an ordinary body runs -- on the head for a
            # single-process / flat layout, on each compute rank for the
            # block layout (the deltas travel in the propose payload's
            # ``extra``) -- so the scoring seams and the publish below see
            # exactly the width they always see.
            with _tspan(_tm, "sample"):
                tmp_state, accepted = self._apply_forced_step(
                    state, ctx, tmp_state, deltas)
            return self._finish_propose(state, ctx, tmp_state, accepted)

        with _tspan(_tm, "sample"):
            if self.max_logl_mode:
                tmp_state, accepted = self.run_move_max_likelihood(tmp_model, tmp_state)

            elif self._ensemble_search_active():
                # Tiled per-walker ensemble, SEARCH stages only. One block
                # per propose: the enclosing MaxLogLCombineMove plateau is
                # what "run until the maximum likelihood converges" means
                # here, so this stays chunked exactly like run_move_for_loop
                # rather than owning a second convergence loop.
                tmp_state, accepted = self.run_move_ensemble_search(tmp_model, tmp_state)

            else:
                tmp_state, accepted = self.run_move_for_loop(tmp_model, tmp_state, self.num_repeats)

        return self._finish_propose(state, ctx, tmp_state, accepted)

    # ---- the forced noise step (galfor ratchet) -----------------------------
    #: Pending forced shift ``{branch: (ndim,) array}`` in the sampled basis,
    #: or None. Set by :meth:`forced_noise_step` for the duration of ONE
    #: propose; shipped to the compute ranks in the propose payload's
    #: ``extra`` (:meth:`fanout_payload_extra` / :meth:`fanout_apply_extra`).
    _forced_step_deltas = None

    # TODO check: keep/remove/document -- only caller is noise_ratchet.py (galfor ratchet); GALFOR_RATCHET default 0, launchers 0, dropped from production 2026-10-03
    def forced_noise_step(self, model, state, deltas):
        """Shift sampled noise coordinates by a fixed vector and PUBLISH.

        The galfor ratchet's "forced lowering" (user design 2026-09-30):
        ``deltas`` maps a branch this move samples to a shift in the SAMPLED
        basis (the run's ``log10`` galfor basis: ``[amp, fk, alpha, f_1,
        f_2]``), applied to EVERY rung and walker. No Metropolis step: the
        new coordinates are accepted unconditionally, their log-prior and
        log-likelihood recomputed, and the result written back and published
        to every walker's container through the same finish half an ordinary
        accepted proposal uses -- so the residual, the sensitivity matrices,
        the replicas and the stored cold row all agree.

        ⚠ ROUTED THROUGH :meth:`propose`, never straight into the body: under
        the block layout each compute rank's ACA holds only its own walkers,
        and the mixin's ``propose`` is what slices the ensemble per rank and
        merges the replies (6mo job 671 died on a 4-walker state against a
        1-walker ACA when this was called directly). The shift itself rides
        in the payload ``extra``, so every rank applies the same vector.

        Refuses (``ValueError``, ``state`` untouched, nothing shipped) a
        branch this move does not sample, a malformed shift, or a row that
        is outside the prior BEFORE the shift. A row that the shift would
        carry OUT of the prior box is CLIPPED to the edge, component by
        component (:func:`clip_shift_to_prior`), and named in a WARNING
        together with how many COLD rows were clipped -- the cold rows are
        the measurement, a hot rung at the edge is not. (Until 2026-10-01
        such a row refused the whole nudge; 6mo job 675 died on two hot
        rungs within one nudge of the ``fk`` floor after the first release
        had spread the ladder.)

        Returns:
            Tuple ``(new_state, accepted)`` with ``accepted`` all True at the
            engine ladder shape.
        """
        noise_branches = self._resolve_sampled(state)
        deltas = {str(k): np.asarray(v, dtype=float) for k, v in dict(deltas).items()}
        for key, d in deltas.items():
            if key not in noise_branches:
                raise ValueError(
                    f"forced_noise_step: {key!r} is not sampled by this move "
                    f"(sampled: {noise_branches}).")
            ndim = int(np.shape(state.branches_coords[key])[-1])
            if d.shape != (ndim,):
                raise ValueError(
                    f"forced_noise_step: shift for {key!r} has shape {d.shape}; "
                    f"expected ({ndim},).")
        # prior check on the FULL ensemble before anything is shipped: the
        # prior needs no ACA, and a refusal must leave every rank untouched.
        # Rows the shift would carry out of the box are clipped to its edge
        # (the ranks repeat the same deterministic clip on their own slice
        # in _apply_forced_step); the head's pass is what names them.
        base = {key: np.asarray(self._work_branch(state, key).coords, dtype=float)
                for key in noise_branches}
        for key, d in deltas.items():
            shifted, clipped = clip_shift_to_prior(
                base[key], d,
                lambda arr, _k=key: self.compute_log_prior({**base, _k: arr}))
            if clipped.any():
                rows = ", ".join(f"rung {int(t)} walker {int(w)}"
                                 for t, w in np.argwhere(clipped))
                took = shifted - base[key]
                short = np.abs(took - d)
                dims = [int(i) for i in np.flatnonzero(short.max(axis=(0, 1, 2)) > 1e-12)]
                logger.warning(
                    "[GALFOR_RATCHET %s] forced noise step CLIPPED to the prior box "
                    "on %d of %d rows of %r (%s; dims %s; largest shortfall %.4g in "
                    "the sampled basis; cold rows clipped: %d) -- every other row "
                    "takes the full shift %s",
                    getattr(self, "name", "psd"), int(clipped.sum()), clipped.size,
                    key, rows, dims, float(short.max()), int(clipped[0].sum()),
                    d.tolist())
        logger.info(
            "[GALFOR_RATCHET %s] forced noise step: %s on every rung and walker; "
            "cold lnL before (walker mean) %.1f",
            getattr(self, "name", "psd"),
            ", ".join(f"{k} += {v.tolist()}" for k, v in deltas.items()),
            float(np.mean(state.log_like[0])))
        self._forced_step_deltas = deltas
        try:
            new_state, accepted = self.propose(model, state)
        finally:
            self._forced_step_deltas = None
        logger.info(
            "[GALFOR_RATCHET %s] forced noise step published; cold lnL after "
            "(walker mean) %.1f", getattr(self, "name", "psd"),
            float(np.mean(new_state.log_like[0])))
        return new_state, accepted

    def _apply_forced_step(self, state, ctx, tmp_state, deltas):
        """Body half of the forced step on THIS body's working state.

        The same deterministic per-row clip the head ran on the full
        ensemble (:func:`clip_shift_to_prior`), on this body's own rows: a
        rank holds only its walkers, and the clip of a row depends on that
        row alone, so head and ranks agree without shipping per-row shifts.
        """
        base = {key: np.asarray(v, dtype=float)
                for key, v in tmp_state.branches_coords.items()}
        for key, d in deltas.items():
            if key not in base:
                continue
            shifted, _ = clip_shift_to_prior(
                base[key], np.asarray(d, dtype=float),
                lambda arr, _k=key: self.compute_log_prior({**base, _k: arr}))
            tmp_state.branches[key].coords[:] = shifted
        coords = tmp_state.branches_coords
        logp = np.asarray(self.compute_log_prior(coords))
        if np.any(np.isinf(logp)):
            # the head checked the full ensemble already; a rank can only get
            # here if the two disagree, which is a bug worth a loud stop
            raise ValueError(
                "forced_noise_step: the shifted coordinates leave the prior on "
                f"{int(np.sum(np.isinf(logp)))} of {logp.size} rows of this body.")
        logl = ctx.init_like_fn(coords, logp=logp, supps=tmp_state.supplemental)[0]
        tmp_state.log_prior = logp
        tmp_state.log_like = np.asarray(logl)
        return tmp_state, np.ones((ctx.nt_mod, ctx.nwalkers_mod), dtype=bool)

    def _begin_propose(self, model, state, resolve_inner: bool = True):
        """The begin half of :meth:`propose_local`: freeze the fixed noise
        branches, replay the begin prep, build the working (module-ladder)
        state and model. ``resolve_inner=False`` skips the inner-proposal
        resolution (eigen tables etc.) for a caller that runs no inner loop.

        Returns a namespace with ``tmp_state``, ``tmp_model``, ``tm``,
        ``noise_branches``, ``engine_ntemps``, ``nt_mod``, ``nwalkers_mod``,
        ``init_like_fn``, ``before_vals``, ``t0``.
        """
        from types import SimpleNamespace

        _t_prop0 = time.perf_counter()
        # GB_PROP_TIMING_SYNC=1 attributes device time to the span that
        # launched it rather than the one that forces the sync (same
        # semantics as the GB moves). Resolved defensively: PSDMove has no
        # guaranteed ``xp``/``backend``, and a timer must never be able to
        # take down a propose.
        _sync = None
        if os.environ.get("GB_PROP_TIMING_SYNC", "0") == "1":
            _xp = getattr(self, "xp", None)
            if getattr(getattr(self, "backend", None), "uses_cupy", False) and _xp is not None:
                _sync = _xp.cuda.runtime.deviceSynchronize
        self._prop_timer = _tm = _ProposeTimer(sync_fn=_sync)

        model_branches = self._noise_model_branches(state)
        noise_branches = self._resolve_sampled(state)

        # cold-row agreement between the main state and the noise-model
        # branches' sub-states — fixed inputs included, they feed the
        # likelihood too (GF_SUBSTATE_CHECK=0 disables)
        self._check_substate_consistency(state, model_branches)
        engine_ntemps = state.log_like.shape[0]

        # freeze the non-sampled noise branches at their cold rows for the
        # duration of this proposal block (consumed by _merged_noise_rows)
        self._fixed_noise_coords = {
            key: np.asarray(state.branches_coords[key][0, :, 0]).copy()
            for key in model_branches
            if key not in noise_branches
        }
        # Correctness-first lifecycle (plan-2 §4.3): every noise proposal
        # block re-reads every walker's CURRENT residual — a source move may
        # have run since the last block. Residual epochs are a recorded later
        # optimization, deliberately not built yet. One-walker replica mode:
        # this begin-of-propose prep must run on every replica, not just the
        # head, so it is a replay (see _replay_noise_begin / _apply_noise_begin).
        self._replay_noise_begin()

        # The working ensembles are the SUB-STATES' tempered branches (this
        # move's module ladder); the main state carries only the engine's
        # cold chain.
        tmp_branches_coords = {
            key: self._work_branch(state, key).coords for key in noise_branches
        }

        shapes = {key: tmp_branches_coords[key].shape[:2] for key in noise_branches}
        if len(set(shapes.values())) != 1:
            raise ValueError(
                "Noise branches sampled by ONE move must share (ntemps, nwalkers); "
                f"got {shapes}. Match the branches' ntemps knobs "
                "(PSD_NTEMPS / GALFOR_NTEMPS / SGWB_NTEMPS) or sample them "
                "with separate moves."
            )

        # move-local supplemental at the MODULE ladder shape (the main
        # state's supplemental is engine-shaped)
        nt_mod, nwalkers_mod = tmp_branches_coords[noise_branches[0]].shape[:2]

        # inner proposal resolution: eigen-axis MH (one-walker default) or
        # the vanilla eryn stretch, from THIS block's module-ladder walker
        # count -- must run after the begin replay above (the tables score
        # merged rows that need the fixed-component covariances) and after
        # nt_mod/nwalkers_mod are known. The default kind keys off THIS
        # BLOCK's walker count (a block of fewer than three walkers has no
        # usable stretch complement); nwalkers_run is logged for context --
        # see _resolve_inner_kind.
        nwalkers_run = (
            self.fanout.layout.nwalkers
            if getattr(self, "fanout", None) is not None
            else nwalkers_mod
        )
        if not resolve_inner:
            pass  # a forced step runs no inner loop: no kind, no tables
        elif self._ensemble_search_active():
            # The tiled ensemble supplies its own complement, so neither the
            # block-width resolution nor the information matrix applies:
            # _resolve_inner_kind would either pick `eigen` (and pay for the
            # table this move exists to avoid) or REFUSE an explicit
            # `stretch` at a one-walker block -- and that refusal is about
            # the OUTER block, which is exactly the condition being worked
            # around here.
            self._inner_kind = "ensemble"
        else:
            self._inner_kind = self._resolve_inner_kind(
                nwalkers_mod, nwalkers_run=nwalkers_run
            )
            if self._inner_kind == "eigen":
                if self._eigen_visits % self.eigen_refresh_every == 0:
                    self._refresh_eigen_tables(tmp_branches_coords)
                self._eigen_visits += 1

        # per-iteration acceptance deltas, accumulated across the repeat block
        # by run_move and written into each sampled branch's sub-state below.
        self._da_audit = []
        self._tally_in_model_proposed = np.zeros(nt_mod, dtype=int)
        self._tally_in_model_accepted = np.zeros(nt_mod, dtype=int)
        self._tally_swaps_proposed = np.zeros(max(nt_mod - 1, 0), dtype=int)
        self._tally_swaps_accepted = np.zeros(max(nt_mod - 1, 0), dtype=int)
        tmp_supps = BranchSupplemental(
            {"walker_inds": np.tile(np.arange(nwalkers_mod), (nt_mod, 1))},
            base_shape=(nt_mod, nwalkers_mod),
            copy=True,
        )

        # module-ladder acceptance tracking (the stage combine-move setter
        # presets an engine-shaped array, so re-shape on first use)
        _acc = getattr(self, "_accepted", None)
        if _acc is None or np.shape(_acc) != (nt_mod, nwalkers_mod):
            self.accepted = np.zeros((nt_mod, nwalkers_mod))

        tmp_state = GFState(tmp_branches_coords, copy=True, supplemental=tmp_supps)

        # ensuring it is up to date. Should not change anything.
        before_vals = model.analysis_container_arr.likelihood().copy()

        # Coarse-sidecar mode dispatch (plan-2 §3.2/§6):
        #   off               — fine everywhere (the historical path).
        #   search_approx     — the whole inner loop optimizes the coarse
        #                       surrogate (proposals, working state, swaps);
        #                       NOT a posterior sampler. The end-of-propose
        #                       publication below still recomputes the fine
        #                       cold state, so the next move sees fine only.
        #   delayed_acceptance— the working state's log_like invariant stays
        #                       FINE; eryn scores stage-1 proposals with the
        #                       coarse callback and _propose_delayed_acceptance
        #                       applies the stage-2 fine correction. Swaps are
        #                       fine-valued by construction.
        _mode, _init_like_fn, _model_like_fn, self._swap_like_fn = (
            self._resolve_mode_like_fns()
        )

        tmp_state.log_prior = self.compute_log_prior(tmp_branches_coords)
        tmp_state.log_like = _init_like_fn(
            tmp_branches_coords, logp=tmp_state.log_prior, supps=tmp_state.supplemental
        )[0]
        self.starting_now = False

        tmp_model = Model(
            state,
            _model_like_fn,
            self.compute_log_prior,
            self.temperature_control,
            model.map_fn,
            model.random,
        )

        return SimpleNamespace(
            t0=_t_prop0, tm=_tm, noise_branches=noise_branches,
            engine_ntemps=engine_ntemps, nt_mod=nt_mod, nwalkers_mod=nwalkers_mod,
            tmp_state=tmp_state, tmp_model=tmp_model, init_like_fn=_init_like_fn,
            before_vals=before_vals,
        )

    def _publish_noise_rows(self, branches_coords, tm=None):
        """Install every LOCAL walker's cold-row noise model on its container.

        ``branches_coords`` is engine-shaped (``[0, w, 0]`` is walker ``w``'s
        cold row). Same per-shard dispatch as the scoring builds: each
        walker's build runs on its owning device, split groups run
        concurrently under run_threaded (multi-GPU), serially otherwise.
        Shared by the end of every propose (:meth:`_finish_propose`) and by
        a passive replica's :meth:`fanout_apply_replica_sync`, so a replica
        reproduces its block lead's publish through the very same code.
        """
        nwalkers = len(self.acs)

        def _publish_one(w):
            w = int(w)
            psd_params = branches_coords["psd"][0, w, 0]
            if "galfor" in branches_coords:
                galfor_params = branches_coords["galfor"][0, w, 0]
            else:
                galfor_params = None
            if "sgwb" in branches_coords:
                sgwb_params = branches_coords["sgwb"][0, w, 0]
            else:
                sgwb_params = None

            with _tspan(tm, "sens_refresh"):
                new_sens = self._build_sensitivity_for_walker(
                    w, psd_params, galfor_params, sgwb_params
                )
            self.acs[w].sens_mat = new_sens
            if self.debug and w == self.debug_plot_walker:
                # accepted params of the plotted walker, for the
                # instrument-only reference curve in the debug overlay.
                # Non-replica path only: in replica mode the publish runs
                # through _replay_noise_publish/_apply_psd_replay instead of
                # _publish_one, so this capture never fires there.
                self._debug_walker_params = (
                    w, psd_params, galfor_params, sgwb_params
                )

        self._run_rows_per_split(
            _publish_one, np.arange(nwalkers), np.arange(nwalkers)
        )

        # NOT gated by _skip_linear_psd_repack, deliberately. This is the one
        # repack that PUBLISHES the accepted noise model to the rest of the
        # run, and it fires once per propose (~16 ms) rather than once per
        # scoring batch (~295x per iteration) -- so skipping it would buy
        # ~0.15% while opening the buffer to staleness across moves. Keeping
        # it means the only window where linear_psd_arr is stale is INSIDE
        # compute_log_like's scoring loop, where no other move can observe it.
        self.acs.reset_linear_psd_arr()

    # ---- the block path under replicas (GF_GPU_ROUTING, several blocks) --
    def fanout_replica_sync_payload(self, new_state, w0, w1):
        """Head: the merged cold noise rows of block ``[w0, w1)`` for its
        PASSIVE replicas, plus which of them this move held FIXED.

        PSD/galfor run the BLOCK path under replicas over several blocks
        (``fanout_flat_body = False``): only the lead proposes, so only the
        lead's containers carry the new noise model. The passive replicas
        rebuild exactly what the lead built -- its propose-begin prep (the
        non-sampled branches frozen at their cold rows, which this move
        never changes) and its publish -- from these rows.
        """
        bc = new_state.branches_coords
        model = self._noise_model_branches(new_state)
        sampled = set(self._resolve_sampled(new_state))
        rows = {
            key: np.array(np.asarray(bc[key])[0:1, int(w0):int(w1), 0:1],
                          dtype=np.float64, copy=True)
            for key in ("psd", "galfor", "sgwb") if key in bc
        }
        fixed = {
            key: np.array(np.asarray(bc[key])[0, int(w0):int(w1), 0],
                          dtype=np.float64, copy=True)
            for key in model if key not in sampled
        }
        return {"rows": rows, "fixed_noise_coords": fixed}

    def fanout_apply_replica_sync(self, payload):
        """Passive replica: the lead's propose-begin prep, then its publish."""
        self._apply_noise_begin({"fixed_noise_coords": payload["fixed_noise_coords"]})
        self._publish_noise_rows(payload["rows"], getattr(self, "_prop_timer", None))

    def _finish_propose(self, state, ctx, tmp_state, accepted):
        """The finish half of :meth:`propose_local`: write the working state's
        sampled branches back into a copy of ``state`` (full ladder into the
        sub-states, cold row into the engine state), publish each walker's
        accepted noise model onto its container (replicas included), and
        return ``(new_state, accepted)`` at the engine ladder shape."""
        _tm = ctx.tm
        noise_branches = ctx.noise_branches
        engine_ntemps = ctx.engine_ntemps
        _t_prop0 = ctx.t0

        # CHECK THIS STATE SETUP
        new_state = GFState(state, copy=True)

        for key in noise_branches:
            if key not in tmp_state.branches:
                continue
            # full-ladder write into the sub-state (shared-memory view),
            # cold row mirrored into the main (engine) state. Only the
            # SAMPLED branches are written: each noise branch's sub-state is
            # owned by the move that samples it (full-noise-model tempered
            # lnL of ITS ensemble + ITS move's adapted ladder).
            self._work_branch(new_state, key).coords[:] = tmp_state.branches[key].coords[:]
            self._sync_cold_row(new_state, key)
            sub = (new_state.sub_states or {}).get(key)
            if sub is not None and getattr(sub, "tempered_initialized", False):
                sub.log_like[:] = tmp_state.log_like[:]
                sub.log_prior[:] = tmp_state.log_prior[:]
                if hasattr(sub, "betas") and sub.betas is not None:
                    sub.betas[:] = self.temperature_control.betas
                # acceptance deltas for this iteration (the backend zeroes
                # them after each save, so these are per-iteration counts).
                # rj_* stay zero: this move never changes leaf count.
                sub.in_model_proposed[:] = self._tally_in_model_proposed
                sub.in_model_accepted[:] = self._tally_in_model_accepted
                if sub.swaps_proposed.size:
                    sub.swaps_proposed[:] = self._tally_swaps_proposed
                    sub.swaps_accepted[:] = self._tally_swaps_accepted

        # the engine state keeps only the cold row (row 0 rewritten from the
        # refreshed ACS below). The cold prior is the FULL noise-model sum so
        # it doesn't flip-flop between partial priors as split moves run.
        new_state.log_like[0] = tmp_state.log_like[0]
        new_state.log_prior[0] = self._cold_noise_log_prior(new_state)

        if self.row_fanout is not None:
            # Gated on replica mode itself, not rows_active()/the knob: the
            # publish must reach every replica's container regardless of
            # {PREFIX}_LIKELIHOOD_FANOUT (see _replay_noise_publish).
            self._replay_noise_publish(new_state)  # walker 0 on every replica, head included
        else:
            self._publish_noise_rows(new_state.branches_coords, _tm)
        after_vals = self.acs.likelihood()

        if self.debug:
            if self._debug_step % self.debug_every == 0:
                self._debug_plot_psd_vs_residual()
            self._debug_step += 1

        new_state.log_like[0] = after_vals
        # eryn-facing acceptance uses the engine (cold-chain) shape
        _tm.count("walkers", len(self.acs))
        _tm.count("branches", len(noise_branches))
        audit = getattr(self, "_da_audit", None)
        if audit:
            _s1 = sum(a[0] for a in audit)
            _s2 = sum(a[1] for a in audit)
            _n = sum(a[2] for a in audit)
            _dd = np.concatenate([a[3] for a in audit])
            logger.info(
                "[COARSE_AUDIT %s] stage1 %d/%d (%.3f) stage2 %d/%d (%.3f) "
                "|dlogl| med %.3e p95 %.3e max %.3e  (0 = exact surrogate; "
                "stage2 acceptance IS the accuracy metric)",
                getattr(self, "name", "psd"),
                _s1, _n, _s1 / max(_n, 1),
                _s2, _s1, _s2 / max(_s1, 1),
                float(np.median(_dd)),
                float(np.percentile(_dd, 95)),
                float(np.max(_dd)),
            )
        logger.info("[PSD_TIMING %s] %s", getattr(self, "name", "psd"),
                    _tm.report(time.perf_counter() - _t_prop0,
                               top=("sample", "sens_refresh", "rows_per_split")))
        self._prop_timer = None
        return new_state, np.asarray(accepted)[:engine_ntemps]


    # ------------------------------------------------------------------
    # DCGA (multi-device) kernel path — absorbed from the legacy
    # MultiGPUPSDMove in the 2026-07 multi-GPU pass (one move structure).
    # ------------------------------------------------------------------

    def _psd_log_like_dcga(self, x: list, supps=None, **sens_kwargs):
        """Kernel fast path through the DCGA's per-device replicas."""
        if supps is None:
            raise ValueError("Must provide supps to identify the data streams.")

        wi = supps["walker_inds"]

        psd_pars, galfor_pars = self.transform_coords(x, return_cupy=False)

        data_index_all = np.asarray(wi).astype(np.int32)

        positions_per_split, data_intra_index_per_split, _ = self.dcga.unpack_indices(data_index_all)
        coords_per_split = self.dcga.unpack_coords(positions_per_split, (psd_pars, galfor_pars))

        data_intra_index_per_split, coords_per_split = self.dcga.place_on_device(
            items=(data_intra_index_per_split, coords_per_split)
        )

        likelihood_args_per_split = self.dcga._loop_operation(
            operation=self.prepare_likelihood_inputs,
            operation_args_per_split=coords_per_split,
            positions_per_split=positions_per_split,
        )

        ll = self.dcga.compute_psd_likelihood(
            positions_per_split,
            data_intra_index_per_split,
            data_intra_index_per_split,
            likelihood_args_per_split,
            likelihood_kwargs={'run_async': self.run_async},
            run_threaded=self.run_threaded,
        )

        # now check if any knot position is not too close together
        if self.sensitivity_backend.use_splines:
            invalid_knots_mask = np.zeros(psd_pars.shape[0], dtype=bool)
            for i, likelihood_args in enumerate(likelihood_args_per_split):
                if likelihood_args is None:
                    continue
                spline_knots_position = (
                    likelihood_args[-2].get()
                    if hasattr(likelihood_args[-2], "get")
                    else likelihood_args[-2]
                )

                invalid_knots_mask[positions_per_split[i]] = np.any(
                    np.diff(10**spline_knots_position, axis=2) < self.tolerance, axis=(0, 2)
                )
            ll[invalid_knots_mask] = -1e300

        self.dcga.free_gpu_memory()
        return ll


# TODO check: keep/remove/document -- deprecated alias; no callers in src/scripts/LATW (exported from moves/__init__ + rst; used by tests/test_psd_move_multi_shard.py, tests/test_walkerfanout_mixin.py)
class MultiGPUPSDMove(PSDMove):
    """Deprecated alias — :class:`PSDMove` is ONE structure that takes
    ``dcga=`` directly (2026-07 multi-GPU pass). This shim maps the legacy
    ``MultiGPUPSDMove(dcga, priors, ...)`` signature onto it.
    """

    def __init__(self, dcga: DomainComputationGroupArray, priors, *args,
                 run_async: bool = False, run_threaded: bool = False,
                 **kwargs):
        warnings.warn(
            "MultiGPUPSDMove is deprecated: construct "
            "PSDMove(acs, priors, dcga=...) directly (one move structure).",
            DeprecationWarning, stacklevel=2,
        )
        PSDMove.__init__(
            self, dcga.acs, priors, *args, dcga=dcga,
            run_async=run_async, run_threaded=run_threaded, **kwargs,
        )
