"""LISA Analysis Tools."""

# ruff: noqa: E402
try:
    from ._version import __version__  # pylint: disable=E0401,E0611
    from ._version import __version_tuple__

except ModuleNotFoundError:
    from importlib.metadata import PackageNotFoundError  # pragma: no cover
    from importlib.metadata import version

    try:
        __version__ = version(__name__)
        __version_tuple__ = tuple(__version__.split("."))
    except PackageNotFoundError:  # pragma: no cover
        __version__ = "unknown"
        __version_tuple__ = (0, 0, 0, "unknown")
    finally:
        del version, PackageNotFoundError

_is_editable: bool
try:
    from . import _editable

    _is_editable = True
    del _editable
except (ModuleNotFoundError, ImportError):
    _is_editable = False


def get_include() -> str:
    """Absolute path to LISAanalysistools' public C++/CUDA header directory.

    Downstream sprint packages (GBGPU, BBHx, FastEMRIWaveforms) add this
    to their compiler include path so that

        #include "Detector.hpp"           // class Orbits -- LAT-owned
        #include "orbits_view.hpp"        // OrbitsView POD (preferred)
        #include "PSD.hpp"
        #include "lisatools_header_abi.hpp"   // ABI version + wrapper-owner toggle

    resolve against the installed wheel.

    Pair with ``gpubackendtools.get_include()`` -- a downstream typically
    needs both: GBT for gbt_global.h + cuda_complex.hpp + InterpolateDevice.hh,
    LAT for the LISA-specific public headers.
    """
    import os.path
    return os.path.join(os.path.dirname(__file__), "cutils")


def get_cmake_module_path() -> str:
    """Absolute path to the directory containing ``LISAanalysisToolsConfig.cmake``.

    For downstreams that prefer ``find_package(LISAanalysisTools CONFIG)``
    over the Python shell-out form of :func:`get_include`.

    Example::

        execute_process(
          COMMAND ${Python_EXECUTABLE} -c
          "import lisatools; print(lisatools.get_cmake_module_path())"
          OUTPUT_VARIABLE LAT_CMAKE_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
        find_package(LISAanalysisTools CONFIG REQUIRED PATHS ${LAT_CMAKE_DIR})
        target_link_libraries(my_kernel PRIVATE LISAanalysisTools::headers)

    Returns the same directory as :func:`get_include`.
    """
    import os.path
    return os.path.join(os.path.dirname(__file__), "cutils")


from gpubackendtools import Globals, get_backend, get_first_backend, has_backend

from . import cutils, utils
from .cutils import LISAToolsCpuBackend, LISAToolsCuda11xBackend, LISAToolsCuda12xBackend, LISAToolsCuda13xBackend

add_backends = {
    "lisatools_cpu": LISAToolsCpuBackend,
    "lisatools_cuda11x": LISAToolsCuda11xBackend,
    "lisatools_cuda12x": LISAToolsCuda12xBackend,
    "lisatools_cuda13x": LISAToolsCuda13xBackend,
}

# Pure-JAX backend (subpackage gated on `import jax`). Optional.
try:
    from .jax import LISAToolsJaxBackend as _LISAToolsJaxBackend
    if _LISAToolsJaxBackend is not None:
        add_backends["lisatools_jax"] = _LISAToolsJaxBackend
except (ImportError, ModuleNotFoundError):
    pass

Globals().backends_manager.add_backends(add_backends)

from gpubackendtools import get_backend as _get_backend
from gpubackendtools import get_first_backend as _get_first_backend
from gpubackendtools import has_backend as _has_backend
from gpubackendtools.gpubackendtools import Backend


def get_backend(backend: str) -> Backend:
    __doc__ = _get_backend.__doc__
    if "lisatools_" not in backend:
        return _get_backend("lisatools_" + backend)
    else:
        return _get_backend(backend)


def has_backend(backend: str) -> Backend:
    __doc__ = _has_backend.__doc__
    if "lisatools_" not in backend:
        return _has_backend("lisatools_" + backend)
    else:
        return _has_backend(backend)


def get_first_backend(backend: str) -> Backend:
    __doc__ = _get_first_backend.__doc__
    if "lisatools_" not in backend:
        return _get_first_backend("lisatools_" + backend)
    else:
        return _get_first_backend(backend)


from .analysiscontainer import AnalysisContainer
from .datacontainer import DataResidualArray
from .sensitivity import SensitivityMatrix, get_sensitivity

__all__ = [
    "__version__",
    "__version_tuple__",
    "_is_editable",
    "get_include",
    "get_cmake_module_path",
    "get_logger",
    "get_config",
    "get_config_setter",
    "get_backend",
    "get_file_manager",
    "has_backend",
    "get_backend",
    "get_first_backend",
]
