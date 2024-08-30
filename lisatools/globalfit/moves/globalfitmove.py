import numpy as np


class GlobalFitMove:

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
    def gpus_needed(self):
        return 

    @property
    def ranks(self):
        return self._ranks

    def assign_ranks(self, ranks):
        assert isinstance(ranks, list)
        self._ranks = ranks