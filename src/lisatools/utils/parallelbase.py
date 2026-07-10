"""Base class for LISA Analysis Tools modules that dispatch to a backend."""

import types
from typing import Optional, Sequence, TypeVar, Union

from gpubackendtools import ParallelModuleBase


class LISAToolsParallelModule(ParallelModuleBase):
    """Project-specific :class:`ParallelModuleBase` that scopes backend lookup to ``lisatools``.

    Subclasses inherit the standard ``gpubackendtools`` backend-dispatch
    machinery (``xp``, ``backend``, ``has_backend``) but resolve string
    ``force_backend`` values against the LISA Analysis Tools backend registry
    (``"cpu"``, ``"cuda11x"``, ``"cuda12x"``, ``"cuda13x"``, ``"cuda"``,
    ``"gpu"``).

    Args:
        *args: Forwarded to :class:`gpubackendtools.ParallelModuleBase`.
        force_backend: Either a backend name (``str``) or a ``(package, name)``
            tuple. A bare string is rewritten to ``("lisatools", force_backend)``
            so the lookup is restricted to LISA Analysis Tools' registered
            backends. ``None`` (default) lets the framework pick the preferred
            available backend.
        **kwargs: Forwarded to :class:`gpubackendtools.ParallelModuleBase`.
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
