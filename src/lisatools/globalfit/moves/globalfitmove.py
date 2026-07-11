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

    ranks_initialized = False

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

    # NOTE: ranks_needed was defined twice on this class; the asserting
    # version below (which always won at class creation) is the survivor.

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

    @property
    def ranks(self):
        """List of MPI ranks assigned to this move via :meth:`assign_ranks`."""
        return self._ranks

    def assign_ranks(self, ranks):
        """Record the MPI ranks dedicated to this move.

        Args:
            ranks: List of MPI rank indices.
        """
        assert isinstance(ranks, list)
        self.ranks_initialized = True
        self._ranks = ranks

    @property
    def ranks_needed(self):
        """Number of MPI ranks this move requires (default 0)."""
        if not hasattr(self, "_ranks_needed"):
            return 0
        return self._ranks_needed

    @ranks_needed.setter
    def ranks_needed(self, ranks_needed):
        assert isinstance(ranks_needed, int)
        self._ranks_needed = ranks_needed


class GFCombineMove(CombineMove, GlobalFitMove):
    """An ``eryn`` :class:`CombineMove` that participates in global-fit bookkeeping.

    Inheriting both :class:`~eryn.moves.CombineMove` and :class:`GlobalFitMove`
    lets a sequence of moves share rank/GPU state and likelihood-comparison
    cadence with the rest of the global fit.
    """

    update_comm_special = True
