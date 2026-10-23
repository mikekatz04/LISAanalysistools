"""JAX backend for lisatools.

Mirrors :mod:`lisatools.cutils` for the CPU / CUDA backends but uses
``jax.numpy`` and pure-Python (jit / grad friendly) implementations. The
backend itself (:class:`~lisatools.jax.backend.LISAToolsJaxBackend`,
registered as ``lisatools_jax`` by ``lisatools/__init__`` when ``import jax``
succeeds) provides the orbits wrapper :class:`~lisatools.jax.orbits.OrbitsWrapJAX`
in its ``OrbitsWrap`` slot; the sensitivity-matrix wrapper has no JAX version.
Downstream JAX code (``gbgpu.jax``, ``bbhx.jax``) consumes this backend the
same way it consumes the CPU / CUDA ones, and builds on the two
subpackages:

- :mod:`lisatools.jax.response`: the LISA response (sky projection,
  ``get_tdi_Xf_single``), :class:`~lisatools.jax.response.tdi_config.TDIConfigWrapJAX`,
  amplitude / phase extraction and the :class:`~lisatools.jax.response.base.JaxAmpPhaseSource`
  source contract.
- :mod:`lisatools.jax.wdm`: the WDM settings / domain / wavelet-lookup
  wrappers and :func:`~lisatools.jax.wdm.fast_inner.fast_wdm_inner_jax`.

Per the backend implementation hierarchy (``docs/conventions.md``) the JAX
path may differ internally from the C++ one but must match it at the
inner-product level. The 9mo / 12mo runs compute on ``cuda13x``; the only
piece of this package they use is :class:`~lisatools.jax.jaxbase.JaxBase`
(the phentax MBH generator's mixin). Select the backend per object with
``force_backend="jax"`` or fetch it with ``lisatools.get_backend("jax")``.

The subpackage import is gated on ``import jax``: without jax ``__all__`` is
empty and nothing is imported.

Example:
    .. code-block:: python

        import lisatools
        backend = lisatools.get_backend("jax")   # requires jax
        backend.xp                                # jax.numpy
        # OrbitsWrapJAX; see its Warning on the argument order
        orbits_wrap = backend.OrbitsWrap(*orbits.pycppdetector_args)
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
    # Generic LISA-response + WDM JAX subpackages, absorbed from
    # fastlisaresponse.jax at Phase 3.
    from . import response, wdm

    __all__ = [
        "LISAToolsJaxBackend",
        "OrbitsWrapJAX",
        "response",
        "wdm",
    ]
else:
    __all__ = []
