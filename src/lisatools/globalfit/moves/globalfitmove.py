"""Base mix-in tying ``eryn`` moves to global-fit MPI/GPU bookkeeping.

Also home of the declarative :class:`Move` base — the single user-facing
"move" concept of the global-fit recipe layer (a move is a proposal; the
words are synonymous). A ``Move`` is cheap to construct and picklable; the
heavy work happens in its required ``setup(ctx)`` hook, run once at recipe
materialization with a :class:`MoveBuildContext`.
"""

import dataclasses
import os
import resource
import sys
import time
import typing

import numpy as np
from eryn.moves import CombineMove

from .. import midit_checkpoint


def ensure_fine_noise_covariance_current(acs, general_info) -> None:
    """Assert the canonical fine noise state is what source moves consume.

    Cheap, assertion-only source-move precondition for coarse-sidecar runs
    (plan-2 §6.4): every container's sensitivity matrix must live on the
    run's FINE domain settings with the fine active shape. A violation is a
    programming error in a noise move's publication sequence — fail loudly;
    source moves never rebuild an unknown state or continue on coarse
    covariance. No-op unless a coarse sidecar runtime is active.
    """
    runtime = getattr(general_info, "coarse_wdm_runtime", None)
    if runtime is None or getattr(runtime, "mode", "off") == "off":
        return
    fine_settings = general_info.domain_settings
    coarse_settings = getattr(general_info, "coarse_wdm_settings", None)
    for w in range(len(acs)):
        sens_mat = acs[w].sens_mat
        basis = getattr(sens_mat, "basis_settings", None)
        if coarse_settings is not None and basis == coarse_settings:
            raise AssertionError(
                f"walker {w}: COARSE sensitivity matrix visible at a "
                "source-move boundary — the noise move failed to publish the "
                "fine state before returning."
            )
        if basis != fine_settings:
            raise AssertionError(
                f"walker {w}: sens_mat basis settings are not the run's fine "
                "domain settings (source moves must see only full fine state)."
            )
        got = tuple(sens_mat.data_shape)
        want = tuple(fine_settings.basis_shape_active)
        if got[-len(want):] != want:
            raise AssertionError(
                f"walker {w}: sens_mat data_shape {got} does not end in the "
                f"fine active shape {want}."
            )


@dataclasses.dataclass
class MoveBuildContext:
    """Runtime context handed to every ``Move.setup`` at materialization.

    Carries everything a move needs to build itself: the runtime
    :class:`~lisatools.globalfit.recipe.Recipe`, the engine/branch metadata,
    the built :class:`~lisatools.globalfit.run.GlobalFitSetup` (``curr``), the
    shared :class:`~lisatools.analysiscontainer.AnalysisContainerArray`
    (``acs``), the priors dict, the initial state, the variant-provided
    ``stock_moves`` name lookup, and the sampler shape (``ntemps`` /
    ``nwalkers``).
    """

    recipe: typing.Any
    engine_info: typing.Any
    curr: typing.Any
    acs: typing.Any
    priors: dict
    state: typing.Any
    stock_moves: dict = dataclasses.field(default_factory=dict)
    ntemps: typing.Optional[int] = None
    nwalkers: typing.Optional[int] = None


class Move:
    """Declarative, picklable recipe move (cheap-construct / heavy-setup).

    A move IS a proposal — one object per adjustment the fit makes each
    iteration. The base class resolves ``name`` against the variant's stock
    moves at materialization; there are exactly two ways to customize:

    * subclass ``Move`` and override :meth:`setup` — build and return the
      runtime eryn move there (or return ``None`` to mean "``self`` is the
      runtime move", in which case the subclass also implements ``propose``);
    * pass a plain ``fn(model, state) -> (new_state, accepted)`` to
      ``add_move`` — it is wrapped in a
      :class:`~lisatools.globalfit.moves.functionmove.FunctionMove`.

    Args:
        name: Unique (per recipe) move name. For the base class this is the
            stock-move name to look up (``"rj_prior"``, ``"psd_pe"``, ...).
        branch: Branch this move samples (validated against enabled branches
            at materialization).
        debug: Move-level debug override applied at materialization via
            ``set_debug`` — ``None`` keeps the move's env-resolved default,
            ``True``/``False`` force it, a dict passes options
            (``plot_dir``/``plot_walker``/``plot_leaf``/``plot_band``/
            ``every``). Wins over the stage-level ``Stage.debug``.
    """

    def __init__(
        self,
        name: str,
        *,
        branch: typing.Optional[str] = None,
        debug: typing.Optional[typing.Union[bool, dict]] = None,
    ):
        if not isinstance(name, str) or not name:
            raise ValueError(f"Move needs a non-empty string name, got {name!r}.")
        self.name = name
        self.branch = branch
        self.debug = debug
        self._runtime = None

    @property
    def is_stock(self) -> bool:
        """True when this move resolves through the stock-name lookup (base ``setup``)."""
        return type(self).setup is Move.setup

    def stock_dependencies(self) -> typing.List[str]:
        """Stock-move names this move's :meth:`setup` pulls from ``ctx``.

        Only meaningful on subclasses that override :meth:`setup`: a custom
        move resolves by its own code rather than by name, so the variant
        setup functions cannot tell from ``Recipe.stock_names()`` that it
        COMPOSES stock moves. Declare them here and they get built.

        Empty by default -- a custom move that builds everything itself needs
        nothing. Returning a name that the variant does not provide fails at
        materialization with the available list, which is the intended
        behaviour (better than a silent no-op).
        """
        return []

    def setup(self, ctx: MoveBuildContext):
        """Build/return the runtime eryn move (``None`` -> ``self`` IS the runtime move).

        Default implementation: stock lookup — resolve
        ``ctx.stock_moves[self.name]``, the move the variant's setup function
        built under this name.
        """
        try:
            return ctx.stock_moves[self.name]
        except KeyError:
            raise ValueError(
                f"Move {self.name!r}: no stock move under this name (available: "
                f"{sorted(ctx.stock_moves)}). Subclass Move and override setup(ctx) "
                "to build a custom move, or add a plain fn(model, state) via add_move."
            ) from None

    def materialize(self, ctx: MoveBuildContext):
        """Framework entry point: run :meth:`setup` and record the runtime move."""
        runtime = self.setup(ctx)
        if runtime is None:
            runtime = self
        self._runtime = runtime
        return runtime

    @property
    def runtime(self):
        """The materialized runtime move (``None`` before materialization)."""
        return self._runtime

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_runtime"] = None
        return state

    def __repr__(self):
        extra = f", branch={self.branch!r}" if self.branch else ""
        return f"{type(self).__name__}({self.name!r}{extra})"


class _RuntimeMove(Move):
    """Private wrapper for a fully constructed runtime move added to a recipe.

    Created by the recipe's move coercion when a user passes an
    already-built eryn move to ``add_move``/``Stage(moves=[...])``. NOTE:
    constructed moves often hold arrays or device state — a fit configured
    with one may not be picklable (deliberately not dropped on pickle,
    matching the historical ``instance`` caveat).
    """

    def __init__(self, move, *, name=None, branch=None, debug=None):
        name = name or getattr(move, "name", None)
        if not name:
            raise ValueError(
                "A constructed move needs a name: pass add_move(..., name=...) "
                "or give the move object a .name attribute."
            )
        super().__init__(name, branch=branch, debug=debug)
        self.move = move

    def setup(self, ctx: MoveBuildContext):
        return self.move


class GlobalFitMove:
    """Mix-in providing MPI rank / GPU bookkeeping for global-fit moves.

    Subclasses combine this mix-in with an ``eryn`` move to expose the extra
    state the global-fit driver needs (assigned MPI ranks, GPU device list,
    likelihood-comparison cadence, and buffer-reset cadence).

    Args:
        name: Required identifier used in logging and bookkeeping.
        iters_compare_likelihood: Iteration cadence at which the move should
            cross-check likelihoods between worker ranks. ``-1`` disables.
        iters_reset_buffers: Iteration cadence at which auxiliary buffers
            should be reset. ``-1`` disables.
    """

    def _work_branch(self, state, branch_name=None):
        """The ensemble a move samples: the sub-state's tempered Branch view.

        The returned eryn ``Branch`` shares memory with the sub-state's
        coords/inds (the module owns its full ladder; the main state carries
        only the engine's cold chain). Falls back to the main state's branch
        when no tempered sub-state exists (e.g. simple-API branches).
        """
        name = branch_name if branch_name is not None else self.branch_name
        sub_states = getattr(state, "sub_states", None) or {}
        sub = sub_states.get(name)
        if sub is not None and getattr(sub, "tempered_initialized", False):
            return sub.branch
        return state.branches[name]

    def _sync_cold_row(self, state, branch_name=None):
        """Mirror the sub-state's cold row into the main (engine) state."""
        name = branch_name if branch_name is not None else self.branch_name
        sub_states = getattr(state, "sub_states", None) or {}
        sub = sub_states.get(name)
        if sub is not None and getattr(sub, "tempered_initialized", False):
            sub.sync_cold_row(state, name)

    def _check_substate_consistency(self, state, branch_names=None):
        """Verify main-state cold rows match each sub-state's row 0.

        Delegates to :meth:`ModuleSubState.check_cold_row` for every
        tempered-initialized sub-state (or just ``branch_names``). No-op when
        ``GF_SUBSTATE_CHECK=0`` or the state carries no sub-states.
        """
        if not substate_check_enabled():
            return
        sub_states = getattr(state, "sub_states", None)
        if not sub_states:
            return
        names = branch_names if branch_names is not None else sub_states.keys()
        for name in names:
            sub = sub_states.get(name)
            if sub is None or not getattr(sub, "tempered_initialized", False):
                continue
            sub.check_cold_row(state, name)

    def __init__(
        self,
        *args,
        name=None,
        iters_compare_likelihood=-1,
        iters_reset_buffers=-1,
        **kwargs,
    ):
        assert name is not None
        self.name = name

        # should be inside moves so you know where it fails
        # rather than in the engine
        self.iters_compare_likelihood = iters_compare_likelihood
        self.iters_reset_buffers = iters_reset_buffers

    @property
    def iters_reset_buffers(self) -> int:
        """Iteration cadence at which auxiliary buffers are reset."""
        return self._iters_reset_buffers

    @iters_reset_buffers.setter
    def iters_reset_buffers(self, iters_reset_buffers: int):
        assert isinstance(iters_reset_buffers, int)
        self._iters_reset_buffers = iters_reset_buffers

    @property
    def iters_compare_likelihood(self) -> int:
        """Iteration cadence at which inter-rank likelihoods are cross-checked."""
        return self._iters_compare_likelihood

    @iters_compare_likelihood.setter
    def iters_compare_likelihood(self, iters_compare_likelihood: int):
        assert isinstance(iters_compare_likelihood, int)
        self._iters_compare_likelihood = iters_compare_likelihood

    @property
    def comm(self):
        """The MPI communicator object owned by this move."""
        return self._comm

    @comm.setter
    def comm(self, comm):
        # if hasattr(self, "update_comm_special") and self.update_comm_special and hasattr(self, "moves"):
        #     ranks_needed = []
        #     for move in self.moves:
        #         if isinstance(move, tuple) or isinstance(move, list):
        #             assert len(move) == 2
        #             move = move[0]

        #         move.comm = comm

        self._comm = comm

    # NOTE: the ranks/assign_ranks/ranks_needed move->rank machinery was
    # removed with the dead dispatch (parallel-resources plan P3).

    def set_debug(
        self,
        enabled: bool = True,
        *,
        plot_dir=None,
        plot_walker=None,
        plot_leaf=None,
        plot_band=None,
        every=None,
        log=None,
    ) -> None:
        """Enable/disable this move's debug instrumentation (move/stage level).

        Uniform across the debug-capable moves — the GB special-stretch move
        (band-indexed: ``plot_band``) and ``ResidualAddOneRemoveOneMove``
        (leaf-indexed: ``plot_leaf``). Options left ``None`` keep whatever the
        move already resolved from its env vars (``GB_DEBUG`` /
        ``{BRANCH}_DEBUG``); precedence is move-spec > stage-spec > env.

        Applied by ``Stage.setup`` from the ``debug`` field on a
        :class:`Move`/``Stage``, or callable directly on a built move.
        Moves without debug hooks (e.g. ``PSDMove``) simply carry the flag
        inertly.
        """
        self.debug = bool(enabled)
        if plot_dir is not None:
            self.debug_plot_dir = str(plot_dir)
        if plot_walker is not None:
            self.debug_plot_walker = int(plot_walker)
        if plot_leaf is not None and hasattr(self, "debug_plot_leaf"):
            self.debug_plot_leaf = int(plot_leaf)
        if plot_band is not None and hasattr(self, "debug_plot_band"):
            self.debug_plot_band = int(plot_band)
        if every is not None and hasattr(self, "debug_every"):
            self.debug_every = max(1, int(every))
        if log is not None and hasattr(self, "debug_log"):
            self.debug_log = bool(log)
        if self.debug and not getattr(self, "debug_plot_dir", None):
            self.debug_plot_dir = "./gf_output/debug/"

    @property
    def gpus(self):
        """List of GPU device indices assigned to this move (default empty)."""
        if not hasattr(self, "_gpus"):
            return []
        return self._gpus

    @gpus.setter
    def gpus(self, gpus):
        assert isinstance(gpus, list)
        for tmp in gpus:
            assert isinstance(tmp, int)

        self._gpus = gpus


def _gf_rss_mb() -> float:
    """Peak resident set size of this process in MB (monotone; darwin=bytes)."""
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return ru / 1e6 if sys.platform == "darwin" else ru / 1e3


def _gf_gpu_pool_mb():
    """(used, total) MB of the default cupy memory pool, or (None, None).

    Only queries cupy when the run has already imported it — a CPU run must
    never trigger a cupy/CUDA initialization from instrumentation.
    """
    if "cupy" not in sys.modules:
        return None, None
    try:
        pool = sys.modules["cupy"].get_default_memory_pool()
        return pool.used_bytes() / 1e6, pool.total_bytes() / 1e6
    except Exception:
        return None, None


def substate_check_enabled() -> bool:
    """Whether the cold-chain sub-state consistency check is armed.

    Always on; disable with ``GF_SUBSTATE_CHECK=0`` (read at call time).
    """
    return os.environ.get("GF_SUBSTATE_CHECK", "1") != "0"


class GFCombineMove(CombineMove, GlobalFitMove):
    """An ``eryn`` :class:`CombineMove` that participates in global-fit bookkeeping.

    Inheriting both :class:`~eryn.moves.CombineMove` and :class:`GlobalFitMove`
    lets a sequence of moves share rank/GPU state and likelihood-comparison
    cadence with the rest of the global fit.

    ``GF_MOVE_TIMING=1`` arms per-move instrumentation in :meth:`propose`:
    one ``[GF_TIMING]`` line per wrapped move per iteration (wall time, peak
    RSS, cupy pool) plus a ``move=__total__`` per-iteration line.  The env var
    is read at call time — construction, pickling, and deepcopy behavior are
    untouched, and with the variable unset the eryn code path runs unchanged.
    ``GF_MOVE_TIMING_SYNC=1`` additionally synchronizes the current cupy
    stream at each move boundary so asynchronous kernel time is attributed to
    the move that launched it.

    ``random_choice=True`` draws ONE wrapped move per step (uniformly, with
    replacement, from ``model.random``) instead of running all of them in a
    fixed order -- the stock eryn move-selection semantics. Default ``False``
    keeps eryn's run-them-all behaviour, so existing stages are untouched.
    Useful for a PE stage where the moves are alternative proposals for the
    same state rather than a pipeline that has to run start to finish.

    ``weighted_cycle=True`` (user ruling 2026-08-26, the GB PE storage
    fix): keep the SEARCH-style single-propose cycle -- every drawn
    sub-move executes inside THIS one propose call, so the backend stores
    ONE row per iteration -- but draw the cycle's composition each
    iteration: ``len(moves)`` draws WITH replacement from the wrapped
    moves, weighted by ``move_weights`` (default equal weights). The draw
    comes from ``model.random`` so a fixed seed reproduces the run. When
    both flags are set, ``weighted_cycle`` governs. Search stages pass
    neither flag and keep the fixed sequential order.
    """

    update_comm_special = True

    def __init__(
        self,
        *args,
        random_choice: bool = False,
        weighted_cycle: bool = False,
        move_weights=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.random_choice = bool(random_choice)
        self.weighted_cycle = bool(weighted_cycle)
        self.move_weights = self._validate_move_weights(move_weights)

    def _validate_move_weights(self, move_weights):
        """Normalize ``move_weights`` to a probability vector (or None).

        None means equal weights (resolved at draw time so the move count
        is always current). Anything else must match ``len(self.moves)``,
        be non-negative, and carry positive total mass.
        """
        if move_weights is None:
            return None
        w = np.asarray(move_weights, dtype=float)
        if w.shape != (len(self.moves),):
            raise ValueError(
                f"move_weights has shape {w.shape}; expected "
                f"({len(self.moves)},) to match the wrapped moves."
            )
        if np.any(w < 0.0) or not np.isfinite(w).all():
            raise ValueError("move_weights must be finite and non-negative.")
        total = w.sum()
        if total <= 0.0:
            raise ValueError("move_weights must have positive total mass.")
        return w / total

    def propose(self, model, state):
        # Simple-API branches (sub_states all None, e.g. erebor.blank) temper
        # on the engine ladder through plain eryn moves, which legitimately
        # rebuild the State -- only arm the guards when a real module
        # sub-state is present.
        _subs = getattr(state, "sub_states", None) or {}
        had_sub_states = any(sub is not None for sub in _subs.values())
        if had_sub_states:
            # stage-entry guard: every branch's cold row must agree between
            # the main state and its sub-state (GF_SUBSTATE_CHECK=0 disables)
            self._check_substate_consistency(state)

        state, accepted = self._propose_moves(model, state)

        if had_sub_states:
            if getattr(state, "sub_states", None) is None:
                raise ValueError(
                    f"stage {getattr(self, 'gf_stage_name', '?')!r}: a wrapped "
                    "move returned a plain eryn State, destroying the "
                    "GFState sub-states. Every global-fit move must return "
                    "the input state or a new GFState (e.g. "
                    "GFState(state, copy=True))."
                )
        return state, accepted

    def _propose_moves_once(self, model, state):
        """One pass over the wrapped moves (the plain GFCombineMove body)."""
        return GFCombineMove._propose_moves(self, model, state)

    def _propose_moves(self, model, state):
        if getattr(self, "weighted_cycle", False) and len(self.moves) > 1:
            # GB PE cycle style (user ruling 2026-08-26): one propose runs
            # a full drawn cycle -- len(moves) sub-moves drawn WITH
            # replacement by weight -- so the backend stores one row per
            # iteration instead of one per move (the random_choice mode's
            # 1-row-per-move storage artifact).
            # TODO: think about replacement -- a weighted draw WITHOUT
            # replacement (a weighted permutation) would guarantee every
            # move runs each iteration while keeping the random order.
            n = len(self.moves)
            w = self.move_weights
            if w is None:
                w = np.full(n, 1.0 / n)
            idx = model.random.choice(n, size=n, replace=True, p=w)
            timing = os.environ.get("GF_MOVE_TIMING", "0") == "1"
            accepted_out = None
            for k in idx:
                move = self.moves[int(k)]
                if isinstance(move, tuple):  # (move, weight) entry
                    move = move[0]
                if timing:
                    name = getattr(move, "gf_move_name", type(move).__name__)
                    print(
                        f"[GF_TIMING] stage="
                        f"{getattr(self, 'gf_stage_name', '?')} "
                        f"weighted_cycle -> {name}",
                        flush=True,
                    )
                state, accepted = move.propose(model, state)
                accepted_out = (
                    accepted.copy() if accepted_out is None
                    else accepted_out + accepted
                )
                # sub-move boundary: coherent resume point (throttled; no-op
                # unless run.py armed the module on this rank)
                midit_checkpoint.maybe_write(
                    state,
                    tag=(
                        f"stage={getattr(self, 'gf_stage_name', '?')} after "
                        f"{getattr(move, 'gf_move_name', type(move).__name__)}"
                    ),
                )
            return state, accepted_out

        if getattr(self, "random_choice", False) and len(self.moves) > 1:
            # One move per step, drawn from the sampler's RNG so the choice
            # is reproducible under a fixed seed. Sub-state guards in
            # propose() still apply to whichever move ran.
            k = int(model.random.randint(len(self.moves)))
            move = self.moves[k]
            if isinstance(move, tuple):  # (move, weight); weight unused here
                move = move[0]
            name = getattr(move, "gf_move_name", type(move).__name__)
            if os.environ.get("GF_MOVE_TIMING", "0") == "1":
                print(f"[GF_TIMING] stage={getattr(self, 'gf_stage_name', '?')} "
                      f"random_choice -> {name}", flush=True)
            return move.propose(model, state)

        timing = os.environ.get("GF_MOVE_TIMING", "0") == "1"
        if not timing and not midit_checkpoint.armed():
            return super().propose(model, state)

        # Unified sequential loop: eryn CombineMove.propose semantics plus
        # optional [GF_TIMING] instrumentation and mid-iteration checkpoints
        # at the sub-move boundaries (each wrapped move returns a coherent
        # state -- cold rows synced -- so every boundary is a resume point).
        self._gf_timing_iter = getattr(self, "_gf_timing_iter", 0) + 1
        it = self._gf_timing_iter
        stage = getattr(self, "gf_stage_name", "?")
        sync = timing and os.environ.get("GF_MOVE_TIMING_SYNC", "0") == "1"

        accepted_out = None
        t_all = time.perf_counter()
        for move in self.moves:
            if isinstance(move, tuple):
                move = move[0]
            rss0 = _gf_rss_mb()
            t0 = time.perf_counter()
            state, accepted = move.propose(model, state)
            if sync and "cupy" in sys.modules:
                try:
                    sys.modules["cupy"].cuda.get_current_stream().synchronize()
                except Exception:
                    pass
            name = getattr(move, "gf_move_name", type(move).__name__)
            if timing:
                wall = time.perf_counter() - t0
                rss1 = _gf_rss_mb()
                used, total = _gf_gpu_pool_mb()
                print(
                    f"[GF_TIMING] stage={stage} move={name} it={it} "
                    f"wall_s={wall:.4f} rss_mb={rss1:.0f} d_rss_mb={rss1 - rss0:.0f} "
                    f"gpu_used_mb={used if used is not None else -1:.0f} "
                    f"gpu_pool_mb={total if total is not None else -1:.0f}",
                    flush=True,
                )
            if accepted_out is None:
                accepted_out = accepted.copy()
            else:
                accepted_out += accepted
            midit_checkpoint.maybe_write(
                state, tag=f"stage={stage} after {name}"
            )
        if timing:
            print(
                f"[GF_TIMING] stage={stage} move=__total__ it={it} "
                f"wall_s={time.perf_counter() - t_all:.4f} rss_mb={_gf_rss_mb():.0f}",
                flush=True,
            )
        return state, accepted_out


class MaxLogLCombineMove(GFCombineMove):
    """Combine move whose max-logL convergence spans ALL wrapped moves.

    ``PSDMove(max_logl_mode=True)`` converges each noise branch SEPARATELY:
    every move runs its own ``run_move_max_likelihood`` loop to its own
    plateau. For a joint search that is the wrong criterion -- psd and galfor
    are two parameterizations of ONE noise model, so maximizing them
    independently can stall each at a plateau the pair could climb past
    together (galfor moves change the residual the psd move is fitting, and
    vice versa).

    This promotes the criterion one level: the wrapped moves each run a
    single pass, and THIS object owns the loop, checking the cold-chain max
    log-likelihood over the whole set. Effectively one move, maximizing
    ``logl`` over repeated iterations of psd AND galfor together.

    Wrap the ``*_pe`` moves, not the ``*_search`` ones. ``max_logl_mode`` is
    consulted in exactly one place (``psdmove.py:918``) -- it only chooses
    between the plateau loop and a single ``run_move_for_loop`` -- so the pe
    move IS the search move minus its private loop, which is precisely the
    inner behaviour wanted here. Wrapping the search moves would nest two
    plateau loops.

    The plateau rule mirrors ``PSDMove.run_move_max_likelihood`` -- exit
    after ``num_checks`` consecutive non-improving iterations, counting only
    once the likelihood has moved at least once (so a move that has not yet
    taken effect cannot trip the exit immediately) -- with one refinement:
    an improvement only counts (resets the counter) when it beats the
    baseline by more than ``tol`` log-likelihood units. A stretch ensemble
    near the mode produces small strict improvements almost every iteration,
    so a search on a ~1e7-scale lnL would otherwise polish the noise floor
    indefinitely. Sub-``tol`` gains do NOT advance the baseline either, so a
    slow accumulated climb (more than ``tol`` per ``num_checks`` window)
    still registers as progress and keeps the stage alive.
    """

    def __init__(
        self, *args, num_checks: int = 5, max_iter: int = 0, tol: float = 5.0, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.num_checks = int(num_checks)
        # Absolute lnL units. The default is deliberately soft (a search
        # stage only needs the noise model roughly converged before the next
        # stage samples it in PE mode); MAXLOGL_TOL overrides, 0 restores
        # the strict any-improvement rule.
        self.tol = float(os.environ.get("MAXLOGL_TOL", tol))
        # Hard ceiling on the plateau loop. 0 = unbounded, which is what
        # PSDMove.run_move_max_likelihood does -- fine when one move owns a
        # cheap likelihood, dangerous here: this can wrap several moves over
        # a basis with no compiled kernel (the WDM noise likelihood falls
        # back to the ACA route), so a likelihood that keeps twitching by
        # 1e-9 never trips the plateau and the stage never ends.
        # MAXLOGL_MAX_ITER overrides.
        self.max_iter = int(os.environ.get("MAXLOGL_MAX_ITER", max_iter))

    def _propose_moves(self, model, state):
        # CHUNKED plateau loop (2026-08-12): at most MAXLOGL_ITERS_PER_STEP
        # inner iterations per propose() call, with the plateau bookkeeping
        # persisted on the instance across calls. The engine saves the
        # backend between propose() calls, so a killed run resumes from the
        # last chunk's ensemble instead of restarting the whole search --
        # previously the ENTIRE search ran inside one engine iteration and
        # a stop during it lost everything. SearchRecipeStep consults
        # ``maxlogl_plateau_done`` to advance the stage. Inside a non-search
        # stage (the joint noise move rides along in gb_search), a plateaued
        # instance keeps taking one inner iteration per call, so the noise
        # model keeps sampling underneath.
        max_inner = int(os.environ.get("MAXLOGL_ITERS_PER_STEP", "10"))
        if not hasattr(self, "_ml_state"):
            self._ml_state = dict(
                num_so_far=0,
                max_logl=-np.inf,   # true running best (logging only)
                ref_logl=-np.inf,   # baseline at last counter reset
                changed_once=False,
                n_iter=0,
            )
        ms = self._ml_state
        accepted = None
        t0 = time.perf_counter()
        stage = getattr(self, "gf_stage_name", "?")
        # Progress EVERY iteration by default. This loop has no natural
        # output and no fixed length, so without it a long stage is
        # indistinguishable from a hung one -- which is exactly what the
        # first full-band run looked like. MAXLOGL_LOG_EVERY=0 silences it.
        log_every = int(os.environ.get("MAXLOGL_LOG_EVERY", "1"))
        inner = 0
        while inner < max_inner and (
            ms["num_so_far"] < self.num_checks or accepted is None
        ):
            state, accepted = self._propose_moves_once(model, state)
            ms["n_iter"] += 1
            inner += 1
            cur = float(state.log_like[0].max())
            if cur != ms["ref_logl"] and not np.isinf(ms["ref_logl"]):
                ms["changed_once"] = True
            if cur > ms["max_logl"]:
                ms["max_logl"] = cur
            improved = cur > ms["ref_logl"] + self.tol or np.isinf(ms["ref_logl"])
            if improved:
                ms["ref_logl"] = cur
                ms["num_so_far"] = 0
            elif ms["changed_once"]:
                ms["num_so_far"] += 1
            if log_every and (ms["n_iter"] % log_every == 0):
                print(
                    f"[MAXLOGL] stage={stage} iter={ms['n_iter']} "
                    f"logl={cur:.6f} best={ms['max_logl']:.6f} "
                    + ("IMPROVED" if improved
                       else f"flat {ms['num_so_far']}/{self.num_checks}") +
                    f" changed_once={ms['changed_once']} "
                    f"elapsed_s={time.perf_counter() - t0:.1f}",
                    flush=True,
                )
            if self.max_iter and ms["n_iter"] >= self.max_iter:
                print(
                    f"[MAXLOGL] stage={stage} hit max_iter={self.max_iter} "
                    f"before plateau (best={ms['max_logl']:.6f}); advancing.",
                    flush=True,
                )
                break
        done = ms["num_so_far"] >= self.num_checks or (
            self.max_iter and ms["n_iter"] >= self.max_iter
        )
        first_done = done and not getattr(self, "maxlogl_plateau_done", False)
        self.maxlogl_plateau_done = bool(done)
        if first_done:
            print(
                f"[MAXLOGL] stage={stage} done after {ms['n_iter']} iterations "
                f"(best={ms['max_logl']:.6f}, {time.perf_counter() - t0:.1f}s)",
                flush=True,
            )
        elif not done:
            print(
                f"[MAXLOGL] stage={stage} chunk break at iter={ms['n_iter']} "
                f"({inner}/{max_inner} this step; checkpointing).",
                flush=True,
            )
        return state, accepted
