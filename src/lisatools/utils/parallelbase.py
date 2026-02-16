import types
from typing import Optional, Sequence, TypeVar, Union

from gpubackendtools import ParallelModuleBase


class LISAToolsParallelModule(ParallelModuleBase):
    def __init__(self, *args, force_backend=None, **kwargs):
        force_backend_in = (
            ("lisatools", force_backend)
            if isinstance(force_backend, str)
            else force_backend
        )
        super().__init__(force_backend_in)
