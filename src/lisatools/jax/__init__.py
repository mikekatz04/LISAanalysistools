"""JAX backend for lisatools.

Mirrors :mod:`lisatools.cutils` for the CPU / CUDA backends but uses
``jax.numpy`` and pure-Python (autograd-friendly) implementations of
the orbits / sensitivity-matrix wrappers. Downstream packages
(``fastlisaresponse.jax``, etc.) consume this backend the same way
they consume the cuda/cpu ones -- via the ``backend.OrbitsWrap`` slot
on the resolved :class:`LISAToolsBackend` instance.

The subpackage import is gated on ``import jax``.
"""
from __future__ import annotations

try:
    import jax  # noqa: F401
    import jax.numpy as jnp  # noqa: F401
    _HAS_JAX = True
except (ImportError, ModuleNotFoundError):
    _HAS_JAX = False

if _HAS_JAX:
    from .backend import LISAToolsJaxBackend
    from .orbits import OrbitsWrapJAX

    __all__ = [
        "LISAToolsJaxBackend",
        "OrbitsWrapJAX",
    ]
else:
    __all__ = []
