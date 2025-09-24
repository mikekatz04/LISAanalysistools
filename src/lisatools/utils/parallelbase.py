from typing import Optional, Sequence, TypeVar, Union
import types


from gpubackendtools import ParallelModuleBase


class LISAToolsParallelModule(ParallelModuleBase):
    def __init__(self, force_backend=None):
        force_backend_in = ('lisatools', force_backend) if isinstance(force_backend, str) else force_backend
        super().__init__(force_backend_in)
