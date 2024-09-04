import numpy as np


class GlobalFitMove:
    ranks_initialized = False

    @property
    def comm(self):
        return self._comm

    @comm.setter
    def comm(self, comm):
        self._comm = comm
    
    @property
    def ranks_needed(self):
        return 0

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
       