import numpy as np
from eryn.moves import CombineMove

class GlobalFitMove:
    ranks_initialized = False

    def __init__(self, *args, name=None, iters_compare_likelihood=-1, iters_reset_buffers=-1, **kwargs):
        assert name is not None
        self.name = name

        # should be inside moves so you know where it fails
        # rather than in the engine
        self.iters_compare_likelihood = iters_compare_likelihood
        self.iters_reset_buffers = iters_reset_buffers

    @property
    def iters_reset_buffers(self) -> int:
        return self._iters_reset_buffers

    @iters_reset_buffers.setter
    def iters_reset_buffers(self, iters_reset_buffers: int):
        assert isinstance(iters_reset_buffers, int)
        self._iters_reset_buffers = iters_reset_buffers

    @property
    def iters_compare_likelihood(self) -> int:
        return self._iters_compare_likelihood

    @iters_compare_likelihood.setter
    def iters_compare_likelihood(self, iters_compare_likelihood: int):
        assert isinstance(iters_compare_likelihood, int)
        self._iters_compare_likelihood = iters_compare_likelihood
        
    @property
    def comm(self):
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
    
    @property
    def ranks_needed(self):
        if not hasattr(self, "_ranks_needed"):
            return 0
        return self._ranks_needed

    @ranks_needed.setter
    def ranks_needed(self, ranks_needed):
        self._ranks_needed = ranks_needed

    @property
    def gpus(self):
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
        return self._ranks

    def assign_ranks(self, ranks):
        assert isinstance(ranks, list)
        self.ranks_initialized = True
        self._ranks = ranks
    
    @property
    def ranks_needed(self): 
        if not hasattr(self, "_ranks_needed"):
            return 0
        return self._ranks_needed

    @ranks_needed.setter
    def ranks_needed(self, ranks_needed):
        assert isinstance(ranks_needed, int)
        self._ranks_needed = ranks_needed
       

class GFCombineMove(CombineMove, GlobalFitMove):
    update_comm_special = True