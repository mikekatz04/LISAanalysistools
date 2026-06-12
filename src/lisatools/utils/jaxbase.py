"""Deprecation shim: ``lisatools.utils.jaxbase`` moved to ``lisatools.jax.jaxbase``.

The JaxBase mixin is JAX infrastructure, so it lives with the rest of the
JAX backend code under ``lisatools.jax`` (post stft_tof merge relocation).
"""
import warnings

warnings.warn(
    "lisatools.utils.jaxbase has moved to lisatools.jax.jaxbase; "
    "update your imports (this shim will be removed).",
    DeprecationWarning,
    stacklevel=2,
)

from ..jax.jaxbase import *  # noqa: F401,F403
from ..jax.jaxbase import JaxBase  # noqa: F401
