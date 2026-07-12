"""Base mix-in tying ``eryn`` moves to global-fit MPI/GPU bookkeeping."""

import numpy as np
from eryn.moves import CombineMove


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

        Applied by :func:`materialize_recipe` from the ``debug`` field on a
        ``MoveSpec``/``StageSpec``, or callable directly on a built move.
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


class GFCombineMove(CombineMove, GlobalFitMove):
    """An ``eryn`` :class:`CombineMove` that participates in global-fit bookkeeping.

    Inheriting both :class:`~eryn.moves.CombineMove` and :class:`GlobalFitMove`
    lets a sequence of moves share rank/GPU state and likelihood-comparison
    cadence with the rest of the global fit.
    """

    update_comm_special = True
