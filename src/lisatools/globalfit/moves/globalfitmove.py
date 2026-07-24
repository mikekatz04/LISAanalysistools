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
    """

    update_comm_special = True

    def propose(self, model, state):
        if os.environ.get("GF_MOVE_TIMING", "0") != "1":
            return super().propose(model, state)

        self._gf_timing_iter = getattr(self, "_gf_timing_iter", 0) + 1
        it = self._gf_timing_iter
        stage = getattr(self, "gf_stage_name", "?")
        sync = os.environ.get("GF_MOVE_TIMING_SYNC", "0") == "1"

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
            wall = time.perf_counter() - t0
            rss1 = _gf_rss_mb()
            used, total = _gf_gpu_pool_mb()
            name = getattr(move, "gf_move_name", type(move).__name__)
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
        print(
            f"[GF_TIMING] stage={stage} move=__total__ it={it} "
            f"wall_s={time.perf_counter() - t_all:.4f} rss_mb={_gf_rss_mb():.0f}",
            flush=True,
        )
        return state, accepted_out
