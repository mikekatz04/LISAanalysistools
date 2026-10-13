"""Move that swaps individual sources in/out of the residual data array."""

from __future__ import annotations

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


from eryn.moves import EigenAxisMove, Move, StretchMove, TemperatureControl
from eryn.prior import ProbDistContainer
from eryn.utils.transform import TransformContainer

from tqdm import tqdm

from ...analysiscontainer import AnalysisContainerArray
from ...domaincomputation import DomainComputationGroupArray
from ...domains import DomainBase, DomainBaseArray, WDMSignal
from ...utils.device import sensitivity_to_current_device
from ...utils.utility import asnumpy, get_array_module
from .. import midit_checkpoint
from . import eigen_refresh, eigen_table_persist
from .globalfitmove import GlobalFitMove
from .walkerfanout import WalkerFanoutMixin, pooled_ladder_step

logger = logging.getLogger(__name__)


def describe_inner_moves(moves, weights) -> str:
    """One-line ``ClassName[(which, frame)] weight | ...`` summary of an
    inner-move stack (``SkyMove`` carries its ``which`` / ``coord_frame``)."""
    parts = []
    for move, weight in zip(moves, weights):
        label = type(move).__name__
        which = getattr(move, "which", None)
        frame = getattr(move, "coord_frame", None)
        if which is not None and frame is not None:
            label = f"{label}({which}, {frame})"
        parts.append(f"{label} {float(weight):.3f}")
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
        **kwargs,
    ):

        # Move.__init__(self, **kwargs)
        StretchMove.__init__(self, **kwargs)

        self.ntemps, self.nwalkers, self.nleaves_max, self.ndim = coords_shape

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

    def refresh_inner_move_tables(self, leaf, work):
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
        """
        eigen_moves = [m for m in self.moves if isinstance(m, EigenAxisMove)]
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
        if (leaf not in self._eigen_tables
                or visits % self._eigen_refresh_cadence() == 0):
            self._eigen_tables[leaf] = self._build_eigen_table(leaf, work)
            self._persist_eigen_table(leaf)
        axes, sigmas = self._eigen_tables[leaf]
        if getattr(axes, "ndim", 2) == 4:
            # per-(temp, walker) tables: the red/blue seam installs the
            # split-sliced view right before each get_proposal call
            return
        for move in eigen_moves:
            move.set_axes(self.branch_name, axes, sigmas)

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
                path, self.branch_name, leaf, **self._eigen_persist_guards()
            )
        except Exception as exc:  # never fail a run over its own cache
            eigen_refresh.logger.warning(
                "[eigen_refresh] %s leaf %d: eigen-table sidecar read "
                "failed (%r); rebuilding", self.branch_name, leaf, exc,
            )
            return False
        if got is None:
            return False
        axes, sigmas, visits = got
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
        self._eigen_tables[leaf] = (axes, sigmas)
        self._eigen_visit_count[leaf] = int(visits)
        eigen_refresh.logger.info(
            "[eigen_refresh] %s leaf %d: adopted the persisted eigen table "
            "(shape %s, visits %d) from %s — skipping the information-"
            "matrix build", self.branch_name, leaf,
            getattr(axes, "shape", None), int(visits), path,
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
            entry = eigen_table_persist.make_entry(
                axes, sigmas,
                self._eigen_visit_count.get(int(leaf), 0),
                x0=getattr(self, "_eigen_x0", {}).get(int(leaf)),
                store_iteration=getattr(self, "_dbg_step", None),
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
        at refresh time) and for non-eigen inner moves.
        """
        if not isinstance(move_here, EigenAxisMove):
            return
        tabs = getattr(self, "_eigen_tables", None)
        if not tabs or int(leaf) not in tabs:
            return
        axes, sigmas = tabs[int(leaf)]
        if getattr(axes, "ndim", 2) != 4:
            return
        ndim = axes.shape[-1]
        ax = axes[split_mask].reshape(self.ntemps, -1, 1, ndim, ndim)
        sg = sigmas[split_mask].reshape(self.ntemps, -1, 1, ndim)
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
        return {
            "fancy_swap_clock": int(getattr(self, "_fancy_swap_clock", 0)),
            "dbg_step": int(getattr(self, "_dbg_step", 0)),
        }

    def fanout_apply_extra(self, extra):
        if "fancy_swap_clock" in extra:
            self._fancy_swap_clock = int(extra["fancy_swap_clock"])
        if "dbg_step" in extra:
            self._dbg_step = int(extra["dbg_step"])

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
            # _current_leaf set
            self.refresh_inner_move_tables(leaf, work)

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
