from __future__ import annotations
import dataclasses
import enum
import types
import typing
import abc
from typing import Optional, Sequence, TypeVar, Union
from ..utils.exceptions import *

from gpubackendtools.gpubackendtools import BackendMethods, CpuBackend, Cuda11xBackend, Cuda12xBackend
from gpubackendtools.utils.exceptions import *

@dataclasses.dataclass
class LISAToolsBackendMethods(BackendMethods):
    get_ll: typing.Callable[(...), None]
    fill_global: typing.Callable[(...), None]


class LISAToolsBackend:
    get_ll: typing.Callable[(...), None]
    fill_global: typing.Callable[(...), None]

    def __init__(self, lisatools_backend_methods):

        # set direct lisatools methods
        # pass rest to general backend
        assert isinstance(lisatools_backend_methods, LISAToolsBackendMethods)

        self.get_ll = lisatools_backend_methods.get_ll
        self.fill_global = lisatools_backend_methods.fill_global


class LISAToolsCpuBackend(CpuBackend, LISAToolsBackend):
    """Implementation of the CPU backend"""
    
    _backend_name = "lisatools_backend_cpu"
    _name = "lisatools_cpu"
    def __init__(self, *args, **kwargs):
        CpuBackend.__init__(self, *args, **kwargs)
        LISAToolsBackend.__init__(self, self.cpu_methods_loader())

    @staticmethod
    def cpu_methods_loader() -> LISAToolsBackendMethods:
        try:
            import lisatools_backend_cpu.utils
            
        except (ModuleNotFoundError, ImportError) as e:
            raise BackendUnavailableException(
                "'cpu' backend could not be imported."
            ) from e

        numpy = LISAToolsCpuBackend.check_numpy()

        return LISAToolsBackendMethods(
            get_ll=lisatools_backend_cpu.utils.get_ll,
            fill_global=lisatools_backend_cpu.utils.fill_global,
            xp=numpy,
        )


class LISAToolsCuda11xBackend(Cuda11xBackend, LISAToolsBackend):

    """Implementation of CUDA 11.x backend"""
    _backend_name : str = "lisatools_backend_cuda11x"
    _name = "lisatools_cuda11x"

    def __init__(self, *args, **kwargs):
        Cuda11xBackend.__init__(self, *args, **kwargs)
        LISAToolsBackend.__init__(self, self.cuda11x_module_loader())
        
    @staticmethod
    def cuda11x_module_loader():
        try:
            import lisatools_backend_cuda11x.utils

        except (ModuleNotFoundError, ImportError) as e:
            raise BackendUnavailableException(
                "'cuda11x' backend could not be imported."
            ) from e

        try:
            import cupy
        except (ModuleNotFoundError, ImportError) as e:
            raise MissingDependencies(
                "'cuda11x' backend requires cupy", pip_deps=["cupy-cuda11x"]
            ) from e

        return LISAToolsBackendMethods(
            get_ll=lisatools_backend_cuda11x.utils.get_ll,
            fill_global=lisatools_backend_cuda11x.utils.fill_global,
            xp=cupy,
        )

class LISAToolsCuda12xBackend(Cuda12xBackend, LISAToolsBackend):
    """Implementation of CUDA 12.x backend"""
    _backend_name : str = "lisatools_backend_cuda12x"
    _name = "lisatools_cuda12x"
    
    def __init__(self, *args, **kwargs):
        Cuda12xBackend.__init__(self, *args, **kwargs)
        LISAToolsBackend.__init__(self, self.cuda12x_module_loader())
        
    @staticmethod
    def cuda12x_module_loader():
        try:
            import lisatools_backend_cuda12x.utils

        except (ModuleNotFoundError, ImportError) as e:
            raise BackendUnavailableException(
                "'cuda12x' backend could not be imported."
            ) from e

        try:
            import cupy
        except (ModuleNotFoundError, ImportError) as e:
            raise MissingDependencies(
                "'cuda12x' backend requires cupy", pip_deps=["cupy-cuda12x"]
            ) from e

        return LISAToolsBackendMethods(
            get_ll=lisatools_backend_cuda12x.utils.get_ll,
            fill_global=lisatools_backend_cuda12x.utils.fill_global,
            xp=cupy,
        )


KNOWN_BACKENDS = {
    "cuda12x": LISAToolsCuda12xBackend,
    "cuda11x": LISAToolsCuda11xBackend,
    "cpu": LISAToolsCpuBackend,
}

"""List of existing backends, per default order of preference."""
