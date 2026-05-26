"""Backend definitions and method tables for the LISA Analysis Tools native modules."""

from __future__ import annotations

import abc
import dataclasses
import enum
import types
import typing
from typing import Optional, Sequence, TypeVar, Union

from gpubackendtools.exceptions import *
from gpubackendtools.gpubackendtools import (
    BackendMethods,
    CpuBackend,
    Cuda11xBackend,
    Cuda12xBackend,
    Cuda13xBackend
)

from ..utils.exceptions import *


@dataclasses.dataclass
class LISAToolsBackendMethods(BackendMethods):
    """Container of native (C++/CUDA) symbols exposed by a LISA Analysis Tools backend.

    Extends :class:`gpubackendtools.gpubackendtools.BackendMethods` with the
    LISA-specific entries that each backend module must provide. When adding a
    new C++/CUDA function to the package, add a field here and populate it from
    every backend's ``*_module_loader``.

    Attributes:
        OrbitsWrap: Native ``OrbitsWrap`` class (CPU or GPU variant) used as the
            low-level wrapper around an ``Orbits`` instance.
        Orbits: Native ``Orbits`` class providing detector geometry on the
            backend device.
        check_orbits: Native function used to validate orbit data.
    """

    OrbitsWrap: object
    Orbits: object
    # SensitivityMatrixWrap: object  # XYZBackend disabled (symbol issues on Linux)
    check_orbits: typing.Callable[(...), None]
    # psd_likelihood: typing.Callable[(...), None]
    # compute_logpdf: typing.Callable[(...), None]


class LISAToolsBackend:
    """Mixin attaching LISA-specific native symbols to a backend class.

    Concrete CPU/CUDA backends combine this mixin with the matching
    :class:`gpubackendtools` backend base (e.g. ``CpuBackend``,
    ``Cuda12xBackend``) to expose both the generic ``xp`` array module and the
    LISA-specific native classes (``OrbitsWrap``, ``Orbits``, ``check_orbits``).

    Args:
        lisatools_backend_methods: The :class:`LISAToolsBackendMethods` instance
            produced by the backend's module loader. Its fields are copied onto
            ``self`` so they are available as plain attributes.
    """

    # TODO: not ClassVar?
    OrbitsWrap: object
    Orbits: object
    check_orbits: typing.Callable[(...), None]
    # SensitivityMatrixWrap: object  # XYZBackend disabled (symbol issues on Linux)
    # psd_likelihood: typing.Callable[(...), None]
    # compute_logpdf: typing.Callable[(...), None]

    def __init__(self, lisatools_backend_methods):

        # set direct lisatools methods
        # pass rest to general backend
        assert isinstance(lisatools_backend_methods, LISAToolsBackendMethods)
        self.OrbitsWrap = lisatools_backend_methods.OrbitsWrap
        self.Orbits = lisatools_backend_methods.Orbits
        self.check_orbits = lisatools_backend_methods.check_orbits
        # self.SensitivityMatrixWrap = lisatools_backend_methods.SensitivityMatrixWrap  # XYZBackend disabled
        # self.psd_likelihood = lisatools_backend_methods.psd_likelihood
        # self.compute_logpdf = lisatools_backend_methods.compute_logpdf


class LISAToolsCpuBackend(CpuBackend, LISAToolsBackend):
    """CPU backend, backed by the ``lisatools_backend_cpu`` native module."""

    _backend_name = "lisatools_backend_cpu"
    _name = "lisatools_cpu"

    def __init__(self, *args, **kwargs):
        CpuBackend.__init__(self, *args, **kwargs)
        LISAToolsBackend.__init__(self, self.cpu_methods_loader())

    @staticmethod
    def cpu_methods_loader() -> LISAToolsBackendMethods:
        """Load CPU native symbols and return a populated :class:`LISAToolsBackendMethods`.

        Raises:
            BackendUnavailableException: If ``lisatools_backend_cpu`` cannot be
                imported (e.g. the CPU extension was not built).
        """
        try:
            import lisatools_backend_cpu.pycppdetector

        except (ModuleNotFoundError, ImportError) as e:
            raise BackendUnavailableException("'cpu' backend could not be imported.") from e

        numpy = LISAToolsCpuBackend.check_numpy()

        return LISAToolsBackendMethods(
            OrbitsWrap=lisatools_backend_cpu.pycppdetector.OrbitsWrapCPU,
            Orbits=lisatools_backend_cpu.pycppdetector.OrbitsCPU,
            check_orbits=lisatools_backend_cpu.pycppdetector.check_orbits,
            # SensitivityMatrixWrap=lisatools_backend_cpu.pycppdetector.XYZSensitivityMatrixWrapCPU,  # XYZBackend disabled
            # psd_likelihood=lisatools_backend_cpu.pycppdetector.psd_likelihood,
            # compute_logpdf=lisatools_backend_cpu.pycppdetector.compute_logpdf,
            xp=numpy,
        )


class LISAToolsCuda11xBackend(Cuda11xBackend, LISAToolsBackend):
    """CUDA 11.x backend, backed by ``lisatools_backend_cuda11x``."""

    _backend_name: str = "lisatools_backend_cuda11x"
    _name = "lisatools_cuda11x"

    def __init__(self, *args, **kwargs):
        Cuda11xBackend.__init__(self, *args, **kwargs)
        LISAToolsBackend.__init__(self, self.cuda11x_module_loader())

    @staticmethod
    def cuda11x_module_loader():
        """Load CUDA 11.x native symbols and return a :class:`LISAToolsBackendMethods`.

        Raises:
            BackendUnavailableException: If ``lisatools_backend_cuda11x`` is not
                available.
            MissingDependencies: If ``cupy`` (specifically ``cupy-cuda11x``) is
                not installed.
        """
        try:
            import lisatools_backend_cuda11x.pycppdetector
            # import lisatools_backend_cuda11x.psd

        except (ModuleNotFoundError, ImportError) as e:
            raise BackendUnavailableException("'cuda11x' backend could not be imported.") from e

        try:
            import cupy
        except (ModuleNotFoundError, ImportError) as e:
            raise MissingDependencies(
                "'cuda11x' backend requires cupy", pip_deps=["cupy-cuda11x"]
            ) from e

        return LISAToolsBackendMethods(
            OrbitsWrap=lisatools_backend_cuda11x.pycppdetector.OrbitsWrapGPU,
            Orbits=lisatools_backend_cuda11x.pycppdetector.OrbitsGPU,
            check_orbits=lisatools_backend_cuda11x.pycppdetector.check_orbits,
            # SensitivityMatrixWrap=lisatools_backend_cuda11x.pycppdetector.XYZSensitivityMatrixWrapGPU,  # XYZBackend disabled
            # psd_likelihood=lisatools_backend_cuda11x.pycppdetector.psd_likelihood,
            # compute_logpdf=lisatools_backend_cuda11x.pycppdetector.compute_logpdf,
            xp=cupy,
        )


class LISAToolsCuda12xBackend(Cuda12xBackend, LISAToolsBackend):
    """CUDA 12.x backend, backed by ``lisatools_backend_cuda12x``."""

    _backend_name: str = "lisatools_backend_cuda12x"
    _name = "lisatools_cuda12x"

    def __init__(self, *args, **kwargs):
        Cuda12xBackend.__init__(self, *args, **kwargs)
        LISAToolsBackend.__init__(self, self.cuda12x_module_loader())

    @staticmethod
    def cuda12x_module_loader():
        """Load CUDA 12.x native symbols and return a :class:`LISAToolsBackendMethods`.

        Raises:
            BackendUnavailableException: If ``lisatools_backend_cuda12x`` is not
                available.
            MissingDependencies: If ``cupy`` (specifically ``cupy-cuda12x``) is
                not installed.
        """
        try:
            import lisatools_backend_cuda12x.pycppdetector

        except (ModuleNotFoundError, ImportError) as e:
            raise BackendUnavailableException("'cuda12x' backend could not be imported.") from e

        try:
            import cupy
        except (ModuleNotFoundError, ImportError) as e:
            raise MissingDependencies(
                "'cuda12x' backend requires cupy", pip_deps=["cupy-cuda12x"]
            ) from e

        return LISAToolsBackendMethods(
            OrbitsWrap=lisatools_backend_cuda12x.pycppdetector.OrbitsWrapGPU,
            Orbits=lisatools_backend_cuda12x.pycppdetector.OrbitsGPU,
            check_orbits=lisatools_backend_cuda12x.pycppdetector.check_orbits,
            # SensitivityMatrixWrap=lisatools_backend_cuda12x.pycppdetector.XYZSensitivityMatrixWrapGPU,  # XYZBackend disabled
            # psd_likelihood=lisatools_backend_cuda12x.pycppdetector.psd_likelihood,
            # compute_logpdf=lisatools_backend_cuda12x.pycppdetector.compute_logpdf,
            xp=cupy,
        )

class LISAToolsCuda13xBackend(Cuda13xBackend, LISAToolsBackend):
    """CUDA 13.x backend, backed by ``lisatools_backend_cuda13x``."""

    _backend_name: str = "lisatools_backend_cuda13x"
    _name = "lisatools_cuda13x"

    def __init__(self, *args, **kwargs):
        Cuda13xBackend.__init__(self, *args, **kwargs)
        LISAToolsBackend.__init__(self, self.cuda13x_module_loader())

    @staticmethod
    def cuda13x_module_loader():
        """Load CUDA 13.x native symbols and return a :class:`LISAToolsBackendMethods`.

        Raises:
            BackendUnavailableException: If ``lisatools_backend_cuda13x`` is not
                available.
            MissingDependencies: If ``cupy`` (specifically ``cupy-cuda13x``) is
                not installed.
        """
        try:
            import lisatools_backend_cuda13x.pycppdetector

            # import lisatools_backend_cuda13x.psd

        except (ModuleNotFoundError, ImportError) as e:
            raise BackendUnavailableException("'cuda13x' backend could not be imported.") from e

        try:
            import cupy
        except (ModuleNotFoundError, ImportError) as e:
            raise MissingDependencies(
                "'cuda13x' backend requires cupy", pip_deps=["cupy-cuda13x"]
            ) from e

        return LISAToolsBackendMethods(
            OrbitsWrap=lisatools_backend_cuda13x.pycppdetector.OrbitsWrapGPU,
            Orbits=lisatools_backend_cuda13x.pycppdetector.OrbitsGPU,
            check_orbits=lisatools_backend_cuda13x.pycppdetector.check_orbits,
            # SensitivityMatrixWrap=lisatools_backend_cuda13x.pycppdetector.XYZSensitivityMatrixWrapGPU,  # XYZBackend disabled
            # psd_likelihood=lisatools_backend_cuda13x.pycppdetector.psd_likelihood,
            # compute_logpdf=lisatools_backend_cuda13x.pycppdetector.compute_logpdf,
            xp=cupy,
        )
"""List of existing backends, per default order of preference."""
