"""Base class for LISA Analysis Tools modules that dispatch to a backend.

:class:`LISAToolsParallelModule` is the parent of every backend-dispatching class in the
package (:class:`~lisatools.detector.Orbits`, the domain settings, the sensitivity backends,
the WDM / chunked-het computation classes, the response classes). Backend choice is made once,
at construction, through ``force_backend=``; methods then dispatch through ``self.backend`` /
``self.xp``.
"""

import types
from typing import Optional, Sequence, TypeVar, Union

from gpubackendtools import ParallelModuleBase


class LISAToolsParallelModule(ParallelModuleBase):
    """Project-specific :class:`ParallelModuleBase` that scopes backend lookup to ``lisatools``.

    Subclasses inherit the standard ``gpubackendtools`` backend-dispatch
    machinery (``xp``, ``backend``, ``backend_name``, ``build_with_same_backend``) but
    resolve string ``force_backend`` values against the LISA Analysis Tools backend
    registry (``"cpu"``, ``"cuda11x"``, ``"cuda12x"``, ``"cuda13x"``, ``"cuda"``,
    ``"gpu"``, and ``"jax"`` when JAX is importable).

    Args:
        *args: Accepted and ignored (not forwarded to
            :class:`gpubackendtools.ParallelModuleBase`).
        force_backend (str, tuple, list, Backend, optional): A backend name, a
            ``(package, name)`` pair, or a ``Backend`` instance. A bare string not already
            starting with ``"lisatools_"`` is rewritten to ``("lisatools", force_backend)``
            so the lookup is restricted to LISA Analysis Tools' registered backends.
            ``None`` (default) picks the first loadable entry of the subclass's
            ``supported_backends()``.
        **kwargs: Accepted and ignored (not forwarded).

    Example:
        >>> from lisatools.detector import EqualArmlengthOrbits
        >>> orbits = EqualArmlengthOrbits(force_backend="cpu")
        >>> orbits.backend.name
        'lisatools_cpu'
    """

    _BACKEND_PREFIX = "lisatools"

    def __init__(self, *args, force_backend=None, **kwargs):
        if isinstance(force_backend, str) and not force_backend.startswith(
            self._BACKEND_PREFIX + "_"
        ):
            force_backend = (self._BACKEND_PREFIX, force_backend)
        super().__init__(force_backend)

    @staticmethod
    def GPU_RECOMMENDED_WITH_JAX() -> list[str]:
        """``GPU_RECOMMENDED()`` platform tags with the JAX backend appended.

        The JAX backend is a host-side pure-Python path (no compiled wheel
        needed); it is listed after the GPU options so the default
        "first available" pick stays GPU when both are present. Returns bare
        platform tags (``cuda13x`` / ``cpu`` / ...); the consuming
        :meth:`supported_backends` prepends the ``_BACKEND_PREFIX`` namespace
        to match the names registered in ``Globals().backends_manager``.
        """
        return ["cuda13x", "cuda12x", "cuda11x", "cpu", "jax"]
