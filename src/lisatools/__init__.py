"""LISA Analysis Tools (``lisatools``): LISA data-analysis and global-fit library.

Importing the package registers the LISA Analysis Tools compute backends with the
``gpubackendtools`` registry (``lisatools_cpu``, ``lisatools_cuda11x``, ``lisatools_cuda12x``,
``lisatools_cuda13x``, plus ``lisatools_jax`` when ``import jax`` succeeds) and re-exports the
central analysis objects:

- :class:`~lisatools.analysiscontainer.AnalysisContainer`: data + sensitivity + signal model, the
  likelihood object.
- :class:`~lisatools.datacontainer.DataResidualArray`: the data / residual arrays.
- :class:`~lisatools.sensitivity.SensitivityMatrix` and
  :func:`~lisatools.sensitivity.get_sensitivity`: noise models.
- :func:`get_backend`, :func:`has_backend`, :func:`get_first_backend`: backend lookup by short
  name (``"cpu"``, ``"cuda12x"``, ...).
- :func:`get_include`, :func:`get_cmake_module_path`: header / CMake paths for downstream
  native builds (GBGPU, BBHx, FastEMRIWaveforms).

Example:
    >>> import lisatools
    >>> be = lisatools.get_backend("cpu")
    >>> be.xp.__name__
    'numpy'
"""

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
    """Return the absolute path to LISAanalysistools' public C++/CUDA header directory.

    Downstream packages (GBGPU, BBHx, FastEMRIWaveforms) add this to their compiler include
    path so that

    .. code-block:: c++

        #include "Detector.hpp"           // class Orbits -- LAT-owned
        #include "orbits_view.hpp"        // OrbitsView POD (preferred)
        #include "PSD.hpp"
        #include "lisatools_header_abi.hpp"   // ABI version + wrapper-owner toggle

    resolve against the installed wheel.

    Pair with ``gpubackendtools.get_include()``: a downstream typically needs both, GBT for
    ``gbt_global.h`` + ``cuda_complex.hpp`` + ``InterpolateDevice.hh``, LAT for the
    LISA-specific public headers.

    Returns:
        str: The installed ``lisatools/cutils`` directory.

    Example:
        >>> import lisatools
        >>> lisatools.get_include().endswith("cutils")
        True
    """
    import os.path
    return os.path.join(os.path.dirname(__file__), "cutils")


def get_cmake_module_path() -> str:
    """Return the absolute path to the directory containing ``LISAanalysisToolsConfig.cmake``.

    For downstreams that prefer ``find_package(LISAanalysisTools CONFIG)`` over the Python
    shell-out form of :func:`get_include`.

    Returns:
        str: The installed ``lisatools/cutils`` directory (the same directory as
        :func:`get_include`).

    Example:
        .. code-block:: cmake

            execute_process(
              COMMAND ${Python_EXECUTABLE} -c
              "import lisatools; print(lisatools.get_cmake_module_path())"
              OUTPUT_VARIABLE LAT_CMAKE_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
            find_package(LISAanalysisTools CONFIG REQUIRED PATHS ${LAT_CMAKE_DIR})
            target_link_libraries(my_kernel PRIVATE LISAanalysisTools::headers)
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
    """Return a LISA Analysis Tools compute backend by name.

    A name without the ``"lisatools_"`` prefix gets it prepended, then
    :func:`gpubackendtools.get_backend` resolves it against the backends registered by this
    package at import. Accepted names:

    - ``"cpu"``, ``"cuda11x"``, ``"cuda12x"``, ``"cuda13x"``: that backend.
    - ``"cuda"``: the first loadable of ``cuda13x``, ``cuda12x``, ``cuda11x`` (in that order).
    - ``"gpu"``: alias of ``"cuda"``.
    - ``"jax"``: the pure-JAX backend, registered only when ``import jax`` succeeds.
    - a full registry name containing ``"lisatools_"`` (e.g. ``"lisatools_cpu"``), passed
      through unchanged.

    Args:
        backend: Backend name, see above.

    Returns:
        Backend: The loaded backend. ``backend.xp`` is ``numpy`` (CPU), ``cupy`` (CUDA) or
        ``jax.numpy`` (JAX); the native classes (``OrbitsWrap``, ``WDMSettingsWrap``, ...) are
        attributes, see :class:`~lisatools.cutils.LISAToolsBackendMethods`.

    Raises:
        gpubackendtools.gpubackendtools.BackendAccessException: When the backend (or, for
            ``"cuda"`` / ``"gpu"``, every CUDA backend) is unavailable or disabled.
        ValueError: When the name is not a registered backend.

    Example:
        >>> import lisatools
        >>> lisatools.get_backend("cpu").name
        'lisatools_cpu'
    """
    # Assigns a local name only (it does not set the function's __doc__); the
    # docstring above is what help() shows.
    __doc__ = _get_backend.__doc__
    if "lisatools_" not in backend:
        return _get_backend("lisatools_" + backend)
    else:
        return _get_backend(backend)


def has_backend(backend: str) -> Backend:
    """Report whether a LISA Analysis Tools backend can be loaded (``False`` when unavailable).

    The name is prefixed with ``"lisatools_"`` (unless it already contains it) and passed to
    :func:`gpubackendtools.has_backend`, which tries to load the backend once and caches the
    result.

    Accepted names: ``"cpu"``, ``"cuda11x"``, ``"cuda12x"``, ``"cuda13x"``, ``"jax"`` (when
    JAX is importable), or a full ``"lisatools_<name>"`` registry name. The ``"cuda"`` and
    ``"gpu"`` aliases are honoured by :func:`get_backend` but not here: ``gpubackendtools``
    matches them only as the bare strings ``"cuda"`` / ``"gpu"``, and the prefixed
    ``"lisatools_cuda"`` is not a registered name.

    Args:
        backend: Backend name, see above.

    Returns:
        bool: ``True`` when the backend's native module (and ``cupy`` for CUDA) loads.

    Raises:
        ValueError: When the name is not a registered backend (including ``"cuda"`` and
            ``"gpu"``, see above).

    Example:
        >>> import lisatools
        >>> lisatools.has_backend("cpu")
        True
    """
    # Assigns a local name only (it does not set the function's __doc__); the
    # docstring above is what help() shows.
    __doc__ = _has_backend.__doc__
    if "lisatools_" not in backend:
        return _has_backend("lisatools_" + backend)
    else:
        return _has_backend(backend)


def get_first_backend(backend: str) -> Backend:
    """Return the first loadable backend of a preference list (thin wrapper, see Warning).

    Prefixes ``"lisatools_"`` onto ``backend`` (unless it already contains it) and forwards
    to :func:`gpubackendtools.get_first_backend`, which expects a SEQUENCE of names.

    Args:
        backend: Backend name(s), see Warning.

    Returns:
        Backend: The first loadable backend.

    Raises:
        gpubackendtools.gpubackendtools.BackendAccessException: When none of the names can be
            loaded.
        ValueError: For a ``str`` argument, see Warning.
        TypeError: For a ``list`` argument, see Warning.

    Warning:
        As written this wrapper does not work: a ``str`` argument becomes one prefixed string
        that ``gpubackendtools`` iterates character by character, so the first character
        fails as an unknown backend name (``ValueError``); a ``list`` argument fails the
        string concatenation (``TypeError``). Package classes resolve their default backend
        through ``ParallelModuleBase.supported_backends()`` instead. Use
        :func:`get_backend` with ``"cuda"`` for "first available CUDA backend".
    """
    # Assigns a local name only (it does not set the function's __doc__); the
    # docstring above is what help() shows.
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
