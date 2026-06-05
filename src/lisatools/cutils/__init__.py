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

    # Phase 3L.7k (2026-06-04): LISA-response Wraps absorbed from the
    # now-retiring fastlisaresponse backend. Owners of the underlying
    # C++ classes are LAT (pycppdetector); the fields below let consumer
    # code that used to reach `self.backend.X` on the FLR backend
    # reach the same names on the LAT backend.
    TDSplineTDIWaveformWrap: object
    FDSplineTDIWaveformWrap: object
    LISAResponseWrap: object
    LISAResponse: object
    # `OrbitsWrap_responselisa` is the lisa-on-gpu-era response-flavored
    # OrbitsWrap. Kept as a separate field for the transition; consumers
    # that don't need it should use the plain `OrbitsWrap` above.
    OrbitsWrap_responselisa: object
    TDIConfigWrap: object
    TDIConfig: object
    CubicSplineWrap_responselisa: object
    WDMSettingsWrap: object
    WDMDomainWrap: object
    FDDomainWrap: object
    # `TDITypeDict`: {"XYZ": TDI_XYZ, "AET": TDI_AET, "AE": TDI_AE} --
    # consumer-side helper for the TDI flavor int enum used by the
    # chunked-het + signal-het kernels.
    TDITypeDict: object


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
    # Phase 3L.7k LISA-response Wraps (see LISAToolsBackendMethods).
    TDSplineTDIWaveformWrap: object
    FDSplineTDIWaveformWrap: object
    LISAResponseWrap: object
    LISAResponse: object
    OrbitsWrap_responselisa: object
    TDIConfigWrap: object
    TDIConfig: object
    CubicSplineWrap_responselisa: object
    WDMSettingsWrap: object
    WDMDomainWrap: object
    FDDomainWrap: object
    TDITypeDict: object

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
        # Phase 3L.7k -- LISA-response wraps absorbed from
        # fastlisaresponse cutils backend (which is being retired).
        self.TDSplineTDIWaveformWrap = lisatools_backend_methods.TDSplineTDIWaveformWrap
        self.FDSplineTDIWaveformWrap = lisatools_backend_methods.FDSplineTDIWaveformWrap
        self.LISAResponseWrap = lisatools_backend_methods.LISAResponseWrap
        self.LISAResponse = lisatools_backend_methods.LISAResponse
        self.OrbitsWrap_responselisa = lisatools_backend_methods.OrbitsWrap_responselisa
        self.TDIConfigWrap = lisatools_backend_methods.TDIConfigWrap
        self.TDIConfig = lisatools_backend_methods.TDIConfig
        self.CubicSplineWrap_responselisa = lisatools_backend_methods.CubicSplineWrap_responselisa
        self.WDMSettingsWrap = lisatools_backend_methods.WDMSettingsWrap
        self.WDMDomainWrap = lisatools_backend_methods.WDMDomainWrap
        self.FDDomainWrap = lisatools_backend_methods.FDDomainWrap
        self.TDITypeDict = lisatools_backend_methods.TDITypeDict


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

        _lat_pd = lisatools_backend_cpu.pycppdetector
        return LISAToolsBackendMethods(
            OrbitsWrap=_lat_pd.OrbitsWrapCPU,
            Orbits=_lat_pd.OrbitsCPU,
            check_orbits=_lat_pd.check_orbits,
            # SensitivityMatrixWrap=_lat_pd.XYZSensitivityMatrixWrapCPU,  # XYZBackend disabled
            # psd_likelihood=_lat_pd.psd_likelihood,
            # compute_logpdf=_lat_pd.compute_logpdf,
            # Phase 3L.7k LISA-response wraps absorbed from fastlisaresponse.
            TDSplineTDIWaveformWrap=_lat_pd.TDSplineTDIWaveformWrapCPU,
            FDSplineTDIWaveformWrap=_lat_pd.FDSplineTDIWaveformWrapCPU,
            LISAResponseWrap=_lat_pd.LISAResponseWrapCPU,
            LISAResponse=_lat_pd.LISAResponseCPU,
            OrbitsWrap_responselisa=_lat_pd.OrbitsWrapCPU_responselisa,
            TDIConfigWrap=_lat_pd.TDIConfigWrapCPU,
            TDIConfig=_lat_pd.TDIConfigCPU,
            CubicSplineWrap_responselisa=_lat_pd.CubicSplineWrapCPU_responselisa,
            WDMSettingsWrap=_lat_pd.WDMSettingsWrapCPU,
            WDMDomainWrap=_lat_pd.WDMDomainWrapCPU,
            FDDomainWrap=_lat_pd.FDDomainWrapCPU,
            TDITypeDict={"XYZ": _lat_pd.TDI_XYZ, "AET": _lat_pd.TDI_AET, "AE": _lat_pd.TDI_AE},
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

        _lat_pd = lisatools_backend_cuda11x.pycppdetector
        return LISAToolsBackendMethods(
            OrbitsWrap=_lat_pd.OrbitsWrapGPU,
            Orbits=_lat_pd.OrbitsGPU,
            check_orbits=_lat_pd.check_orbits,
            # SensitivityMatrixWrap=_lat_pd.XYZSensitivityMatrixWrapGPU,  # XYZBackend disabled
            # psd_likelihood=_lat_pd.psd_likelihood,
            # compute_logpdf=_lat_pd.compute_logpdf,
            # Phase 3L.7k LISA-response wraps absorbed from fastlisaresponse.
            TDSplineTDIWaveformWrap=_lat_pd.TDSplineTDIWaveformWrapGPU,
            FDSplineTDIWaveformWrap=_lat_pd.FDSplineTDIWaveformWrapGPU,
            LISAResponseWrap=_lat_pd.LISAResponseWrapGPU,
            LISAResponse=_lat_pd.LISAResponseGPU,
            OrbitsWrap_responselisa=_lat_pd.OrbitsWrapGPU_responselisa,
            TDIConfigWrap=_lat_pd.TDIConfigWrapGPU,
            TDIConfig=_lat_pd.TDIConfigGPU,
            CubicSplineWrap_responselisa=_lat_pd.CubicSplineWrapGPU_responselisa,
            WDMSettingsWrap=_lat_pd.WDMSettingsWrapGPU,
            WDMDomainWrap=_lat_pd.WDMDomainWrapGPU,
            FDDomainWrap=_lat_pd.FDDomainWrapGPU,
            TDITypeDict={"XYZ": _lat_pd.TDI_XYZ, "AET": _lat_pd.TDI_AET, "AE": _lat_pd.TDI_AE},
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

        _lat_pd = lisatools_backend_cuda12x.pycppdetector
        return LISAToolsBackendMethods(
            OrbitsWrap=_lat_pd.OrbitsWrapGPU,
            Orbits=_lat_pd.OrbitsGPU,
            check_orbits=_lat_pd.check_orbits,
            # SensitivityMatrixWrap=_lat_pd.XYZSensitivityMatrixWrapGPU,  # XYZBackend disabled
            # psd_likelihood=_lat_pd.psd_likelihood,
            # compute_logpdf=_lat_pd.compute_logpdf,
            # Phase 3L.7k LISA-response wraps absorbed from fastlisaresponse.
            TDSplineTDIWaveformWrap=_lat_pd.TDSplineTDIWaveformWrapGPU,
            FDSplineTDIWaveformWrap=_lat_pd.FDSplineTDIWaveformWrapGPU,
            LISAResponseWrap=_lat_pd.LISAResponseWrapGPU,
            LISAResponse=_lat_pd.LISAResponseGPU,
            OrbitsWrap_responselisa=_lat_pd.OrbitsWrapGPU_responselisa,
            TDIConfigWrap=_lat_pd.TDIConfigWrapGPU,
            TDIConfig=_lat_pd.TDIConfigGPU,
            CubicSplineWrap_responselisa=_lat_pd.CubicSplineWrapGPU_responselisa,
            WDMSettingsWrap=_lat_pd.WDMSettingsWrapGPU,
            WDMDomainWrap=_lat_pd.WDMDomainWrapGPU,
            FDDomainWrap=_lat_pd.FDDomainWrapGPU,
            TDITypeDict={"XYZ": _lat_pd.TDI_XYZ, "AET": _lat_pd.TDI_AET, "AE": _lat_pd.TDI_AE},
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

        _lat_pd = lisatools_backend_cuda13x.pycppdetector
        return LISAToolsBackendMethods(
            OrbitsWrap=_lat_pd.OrbitsWrapGPU,
            Orbits=_lat_pd.OrbitsGPU,
            check_orbits=_lat_pd.check_orbits,
            # SensitivityMatrixWrap=_lat_pd.XYZSensitivityMatrixWrapGPU,  # XYZBackend disabled
            # psd_likelihood=_lat_pd.psd_likelihood,
            # compute_logpdf=_lat_pd.compute_logpdf,
            # Phase 3L.7k LISA-response wraps absorbed from fastlisaresponse.
            TDSplineTDIWaveformWrap=_lat_pd.TDSplineTDIWaveformWrapGPU,
            FDSplineTDIWaveformWrap=_lat_pd.FDSplineTDIWaveformWrapGPU,
            LISAResponseWrap=_lat_pd.LISAResponseWrapGPU,
            LISAResponse=_lat_pd.LISAResponseGPU,
            OrbitsWrap_responselisa=_lat_pd.OrbitsWrapGPU_responselisa,
            TDIConfigWrap=_lat_pd.TDIConfigWrapGPU,
            TDIConfig=_lat_pd.TDIConfigGPU,
            CubicSplineWrap_responselisa=_lat_pd.CubicSplineWrapGPU_responselisa,
            WDMSettingsWrap=_lat_pd.WDMSettingsWrapGPU,
            WDMDomainWrap=_lat_pd.WDMDomainWrapGPU,
            FDDomainWrap=_lat_pd.FDDomainWrapGPU,
            TDITypeDict={"XYZ": _lat_pd.TDI_XYZ, "AET": _lat_pd.TDI_AET, "AE": _lat_pd.TDI_AE},
            xp=cupy,
        )
"""List of existing backends, per default order of preference."""
