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

    .. warning::

       **A new field MUST be keyword-only with a ``None`` default.** Declare
       it at the END of the field list as::

           new_symbol: typing.Optional[typing.Callable[(...), None]] = (
               dataclasses.field(default=None, kw_only=True)
           )

       This dataclass is SUBCLASSED by downstream packages -- GBGPU's
       ``gbgpu.cutils.GBGPUBackendMethods``, and the same pattern in BBHx /
       FEW -- which add their own REQUIRED fields and construct the subclass
       from their OWN module loaders. Those loaders cannot know about a
       field LAT added yesterday, so a REQUIRED field here breaks every
       downstream backend load at import time. That is exactly what commit
       ``0f0fc73a`` did with ``gb_inmodel_gate_compact`` /
       ``gb_inmodel_accept_apply``: cluster job 354 (2026-08-28) died with
       ``TypeError: GBGPUBackendMethods.__init__() missing 2 required
       positional arguments``.

       ``kw_only=True`` is load-bearing, not decoration: a plain ``= None``
       default would place a defaulted field ahead of the subclass's
       required ones and fail at CLASS-DEFINITION time with "non-default
       argument follows default argument". Keyword-only fields are exempt
       from that ordering rule.

       ``None`` is also the honest value: it means "this backend module does
       not carry the symbol", which is the same state a stale ``.so``
       produces via the loaders' ``getattr(_lat_pd, name, None)``. Every
       consumer must already handle it. Same lesson as ``ddbe414`` -- the
       compile/import-time contract is the one that bites in production,
       and it bites in the OTHER repo.

    Attributes:
        OrbitsWrap: Native ``OrbitsWrap`` class (CPU or GPU variant) used as the
            low-level wrapper around an ``Orbits`` instance.
        Orbits: Native ``Orbits`` class providing detector geometry on the
            backend device.
        check_orbits: Native function used to validate orbit data.
    """

    OrbitsWrap: object
    Orbits: object
    # stft_tof merge (2026-06): XYZ sensitivity backend reactivated (the
    # former "symbol issues on Linux" were the missing CPU/GPU aliases).
    SensitivityMatrixWrap: object
    GalacticGridSetup: object
    GalacticGridWrap: object
    # 2026-06 domains consolidation: STFT/FD domain wraps. The incoming stft
    # FDDomainWrap is exposed as FDDomainForStftWrap (the Phase-3L.1
    # chunked-het FDDomainWrap below owns the FDDomainWrap name).
    STFTDomainWrap: object
    FDDomainForStftWrap: object
    STFTFresnelWrap: object
    check_orbits: typing.Callable[(...), None]
    psd_likelihood: typing.Callable[(...), None]
    compute_logpdf: typing.Callable[(...), None]

    # Phase 3L.7k (2026-06-04): LISA-response Wraps absorbed from the
    # now-retiring fastlisaresponse backend. Owners of the underlying
    # C++ classes are LAT (pycppdetector); the fields below let consumer
    # code that used to reach `self.backend.X` on the FLR backend
    # reach the same names on the LAT backend.
    TDSplineTDIWaveformWrap: object
    FDSplineTDIWaveformWrap: object
    LISAResponseWrap: object
    LISAResponse: object
    # Phase 3L.7p (2026-06-04): OrbitsWrap removed.
    # Consumers reach for OrbitsWrap above.
    TDIConfigWrap: object
    TDIConfig: object
    # Canonical name `CubicSplineWrap` (the `_responselisa` tail was
    # dropped everywhere, mirroring the Phase 3L.7p OrbitsWrap collapse).
    CubicSplineWrap: object
    WDMSettingsWrap: object
    WDMDomainWrap: object
    FDDomainWrap: object
    # `TDITypeDict`: {"XYZ": TDI_XYZ, "AET": TDI_AET, "AE": TDI_AE} --
    # consumer-side helper for the TDI flavor int enum used by the
    # chunked-het + signal-het kernels.
    TDITypeDict: object

    # --- KEYWORD-ONLY, DEFAULTED: fields added after downstream packages
    # --- started subclassing this dataclass. See the class docstring.
    #
    # Global-fit routing kernels (gf_routing_kernels.cu, 2026-08-27): the
    # fused GB in-model pre-score gate/compaction and post-score
    # accept/bookkeeping chains. ``None`` when the backend module does not
    # carry them -- either a stale ``.so`` built before they landed (LAT's
    # own loaders below use ``getattr(..., None)``) or a downstream loader
    # that never heard of them. The call site (GB_INMODEL_ACCEPT_KERNEL)
    # checks for None and falls back to the python chain with a warning.
    gb_inmodel_gate_compact: typing.Optional[typing.Callable[(...), None]] = (
        dataclasses.field(default=None, kw_only=True)
    )
    gb_inmodel_accept_apply: typing.Optional[typing.Callable[(...), None]] = (
        dataclasses.field(default=None, kw_only=True)
    )


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
    # stft_tof merge (2026-06): XYZ sensitivity backend reactivated.
    SensitivityMatrixWrap: object
    GalacticGridSetup: object
    GalacticGridWrap: object
    # 2026-06 domains consolidation: STFT/FD domain wraps.
    STFTDomainWrap: object
    FDDomainForStftWrap: object
    STFTFresnelWrap: object
    psd_likelihood: typing.Callable[(...), None]
    compute_logpdf: typing.Callable[(...), None]
    # Global-fit routing kernels (see LISAToolsBackendMethods). ``None`` when
    # the loaded backend module does not carry them.
    gb_inmodel_gate_compact: typing.Optional[typing.Callable[(...), None]]
    gb_inmodel_accept_apply: typing.Optional[typing.Callable[(...), None]]
    # Phase 3L.7k LISA-response Wraps (see LISAToolsBackendMethods).
    TDSplineTDIWaveformWrap: object
    FDSplineTDIWaveformWrap: object
    LISAResponseWrap: object
    LISAResponse: object
    TDIConfigWrap: object
    TDIConfig: object
    CubicSplineWrap: object
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
        # stft_tof merge (2026-06): XYZ sensitivity backend reactivated.
        self.SensitivityMatrixWrap = lisatools_backend_methods.SensitivityMatrixWrap
        self.GalacticGridSetup = lisatools_backend_methods.GalacticGridSetup
        self.GalacticGridWrap = lisatools_backend_methods.GalacticGridWrap
        # 2026-06 domains consolidation: STFT/FD domain wraps.
        self.STFTDomainWrap = lisatools_backend_methods.STFTDomainWrap
        self.FDDomainForStftWrap = lisatools_backend_methods.FDDomainForStftWrap
        self.STFTFresnelWrap = lisatools_backend_methods.STFTFresnelWrap
        self.psd_likelihood = lisatools_backend_methods.psd_likelihood
        self.compute_logpdf = lisatools_backend_methods.compute_logpdf
        # Global-fit routing kernels (gf_routing_kernels.cu).
        self.gb_inmodel_gate_compact = lisatools_backend_methods.gb_inmodel_gate_compact
        self.gb_inmodel_accept_apply = lisatools_backend_methods.gb_inmodel_accept_apply
        # Phase 3L.7k -- LISA-response wraps absorbed from
        # fastlisaresponse cutils backend (which is being retired).
        self.TDSplineTDIWaveformWrap = lisatools_backend_methods.TDSplineTDIWaveformWrap
        self.FDSplineTDIWaveformWrap = lisatools_backend_methods.FDSplineTDIWaveformWrap
        self.LISAResponseWrap = lisatools_backend_methods.LISAResponseWrap
        self.LISAResponse = lisatools_backend_methods.LISAResponse
        self.TDIConfigWrap = lisatools_backend_methods.TDIConfigWrap
        self.TDIConfig = lisatools_backend_methods.TDIConfig
        self.CubicSplineWrap = lisatools_backend_methods.CubicSplineWrap
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
            import gbt_backend_cpu.interp
            import lisatools_backend_cpu.pycppdetector

        except (ModuleNotFoundError, ImportError) as e:
            raise BackendUnavailableException("'cpu' backend could not be imported.") from e

        numpy = LISAToolsCpuBackend.check_numpy()
    
        _lat_pd = lisatools_backend_cpu.pycppdetector
        return LISAToolsBackendMethods(
            OrbitsWrap=_lat_pd.OrbitsWrapCPU,
            Orbits=_lat_pd.OrbitsCPU,
            check_orbits=_lat_pd.check_orbits,
            # stft_tof merge (2026-06): XYZ sensitivity backend reactivated.
            SensitivityMatrixWrap=_lat_pd.XYZSensitivityMatrixWrapCPU,
            GalacticGridSetup=_lat_pd.GalacticGridSetup,
            GalacticGridWrap=_lat_pd.GalacticGridWrapCPU,
            # 2026-06 domains consolidation: STFT/FD domain wraps.
            STFTDomainWrap=_lat_pd.STFTDomainWrapCPU,
            FDDomainForStftWrap=_lat_pd.FDDomainForStftWrapCPU,
            STFTFresnelWrap=_lat_pd.STFTFresnelWrapCPU,
            psd_likelihood=_lat_pd.psd_likelihood,
            compute_logpdf=_lat_pd.compute_logpdf,
            # Global-fit routing kernels (2026-08-27). ``getattr`` with a None
            # fallback so a backend module built before they landed still
            # imports; the GB_INMODEL_ACCEPT_KERNEL call site checks for None.
            gb_inmodel_gate_compact=getattr(_lat_pd, "gb_inmodel_gate_compact", None),
            gb_inmodel_accept_apply=getattr(_lat_pd, "gb_inmodel_accept_apply", None),
            # Phase 3L.7k LISA-response wraps absorbed from fastlisaresponse.
            TDSplineTDIWaveformWrap=_lat_pd.TDSplineTDIWaveformWrapCPU,
            FDSplineTDIWaveformWrap=_lat_pd.FDSplineTDIWaveformWrapCPU,
            LISAResponseWrap=_lat_pd.LISAResponseWrapCPU,
            LISAResponse=_lat_pd.LISAResponseCPU,
            TDIConfigWrap=_lat_pd.TDIConfigWrapCPU,
            TDIConfig=_lat_pd.TDIConfigCPU,
            # GBT is the single registrant for CubicSplineWrap (same
            # pattern as downstream packages consuming LAT's OrbitsWrap).
            CubicSplineWrap=gbt_backend_cpu.interp.CubicSplineWrapCPU,
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
            import gbt_backend_cuda11x.interp
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
            # stft_tof merge (2026-06): XYZ sensitivity backend reactivated.
            SensitivityMatrixWrap=_lat_pd.XYZSensitivityMatrixWrapGPU,
            GalacticGridSetup=_lat_pd.GalacticGridSetup,
            GalacticGridWrap=_lat_pd.GalacticGridWrapGPU,
            # 2026-06 domains consolidation: STFT/FD domain wraps.
            STFTDomainWrap=_lat_pd.STFTDomainWrapGPU,
            FDDomainForStftWrap=_lat_pd.FDDomainForStftWrapGPU,
            STFTFresnelWrap=_lat_pd.STFTFresnelWrapGPU,
            psd_likelihood=_lat_pd.psd_likelihood,
            compute_logpdf=_lat_pd.compute_logpdf,
            # Global-fit routing kernels (2026-08-27). ``getattr`` with a None
            # fallback so a backend module built before they landed still
            # imports; the GB_INMODEL_ACCEPT_KERNEL call site checks for None.
            gb_inmodel_gate_compact=getattr(_lat_pd, "gb_inmodel_gate_compact", None),
            gb_inmodel_accept_apply=getattr(_lat_pd, "gb_inmodel_accept_apply", None),
            # Phase 3L.7k LISA-response wraps absorbed from fastlisaresponse.
            TDSplineTDIWaveformWrap=_lat_pd.TDSplineTDIWaveformWrapGPU,
            FDSplineTDIWaveformWrap=_lat_pd.FDSplineTDIWaveformWrapGPU,
            LISAResponseWrap=_lat_pd.LISAResponseWrapGPU,
            LISAResponse=_lat_pd.LISAResponseGPU,
            TDIConfigWrap=_lat_pd.TDIConfigWrapGPU,
            TDIConfig=_lat_pd.TDIConfigGPU,
            # GBT is the single registrant for CubicSplineWrap.
            CubicSplineWrap=gbt_backend_cuda11x.interp.CubicSplineWrapGPU,
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
            import gbt_backend_cuda12x.interp
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
            # stft_tof merge (2026-06): XYZ sensitivity backend reactivated.
            SensitivityMatrixWrap=_lat_pd.XYZSensitivityMatrixWrapGPU,
            GalacticGridSetup=_lat_pd.GalacticGridSetup,
            GalacticGridWrap=_lat_pd.GalacticGridWrapGPU,
            # 2026-06 domains consolidation: STFT/FD domain wraps.
            STFTDomainWrap=_lat_pd.STFTDomainWrapGPU,
            FDDomainForStftWrap=_lat_pd.FDDomainForStftWrapGPU,
            STFTFresnelWrap=_lat_pd.STFTFresnelWrapGPU,
            psd_likelihood=_lat_pd.psd_likelihood,
            compute_logpdf=_lat_pd.compute_logpdf,
            # Global-fit routing kernels (2026-08-27). ``getattr`` with a None
            # fallback so a backend module built before they landed still
            # imports; the GB_INMODEL_ACCEPT_KERNEL call site checks for None.
            gb_inmodel_gate_compact=getattr(_lat_pd, "gb_inmodel_gate_compact", None),
            gb_inmodel_accept_apply=getattr(_lat_pd, "gb_inmodel_accept_apply", None),
            # Phase 3L.7k LISA-response wraps absorbed from fastlisaresponse.
            TDSplineTDIWaveformWrap=_lat_pd.TDSplineTDIWaveformWrapGPU,
            FDSplineTDIWaveformWrap=_lat_pd.FDSplineTDIWaveformWrapGPU,
            LISAResponseWrap=_lat_pd.LISAResponseWrapGPU,
            LISAResponse=_lat_pd.LISAResponseGPU,
            TDIConfigWrap=_lat_pd.TDIConfigWrapGPU,
            TDIConfig=_lat_pd.TDIConfigGPU,
            # GBT is the single registrant for CubicSplineWrap.
            CubicSplineWrap=gbt_backend_cuda12x.interp.CubicSplineWrapGPU,
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
            import gbt_backend_cuda13x.interp
            import lisatools_backend_cuda13x.pycppdetector

            # import lisatools_backend_cuda12x.psd

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
            # stft_tof merge (2026-06): XYZ sensitivity backend reactivated.
            SensitivityMatrixWrap=_lat_pd.XYZSensitivityMatrixWrapGPU,
            GalacticGridSetup=_lat_pd.GalacticGridSetup,
            GalacticGridWrap=_lat_pd.GalacticGridWrapGPU,
            # 2026-06 domains consolidation: STFT/FD domain wraps.
            STFTDomainWrap=_lat_pd.STFTDomainWrapGPU,
            FDDomainForStftWrap=_lat_pd.FDDomainForStftWrapGPU,
            STFTFresnelWrap=_lat_pd.STFTFresnelWrapGPU,
            psd_likelihood=_lat_pd.psd_likelihood,
            compute_logpdf=_lat_pd.compute_logpdf,
            # Global-fit routing kernels (2026-08-27). ``getattr`` with a None
            # fallback so a backend module built before they landed still
            # imports; the GB_INMODEL_ACCEPT_KERNEL call site checks for None.
            gb_inmodel_gate_compact=getattr(_lat_pd, "gb_inmodel_gate_compact", None),
            gb_inmodel_accept_apply=getattr(_lat_pd, "gb_inmodel_accept_apply", None),
            # Phase 3L.7k LISA-response wraps absorbed from fastlisaresponse.
            TDSplineTDIWaveformWrap=_lat_pd.TDSplineTDIWaveformWrapGPU,
            FDSplineTDIWaveformWrap=_lat_pd.FDSplineTDIWaveformWrapGPU,
            LISAResponseWrap=_lat_pd.LISAResponseWrapGPU,
            LISAResponse=_lat_pd.LISAResponseGPU,
            TDIConfigWrap=_lat_pd.TDIConfigWrapGPU,
            TDIConfig=_lat_pd.TDIConfigGPU,
            # GBT is the single registrant for CubicSplineWrap.
            CubicSplineWrap=gbt_backend_cuda13x.interp.CubicSplineWrapGPU,
            WDMSettingsWrap=_lat_pd.WDMSettingsWrapGPU,
            WDMDomainWrap=_lat_pd.WDMDomainWrapGPU,
            FDDomainWrap=_lat_pd.FDDomainWrapGPU,
            TDITypeDict={"XYZ": _lat_pd.TDI_XYZ, "AET": _lat_pd.TDI_AET, "AE": _lat_pd.TDI_AE},
            xp=cupy,
        )

"""List of existing backends, per default order of preference."""
