"""JAX implementations of the LISA TDI / response infrastructure.

Absorbed from ``fastlisaresponse.jax`` at Phase 3 of the sprint reorg.

Public entry points:

* :class:`~lisatools.jax.response.base.JaxAmpPhaseSource` — base for source
  classes that provide ``amplitude(t, params)`` and ``phase(t, params)``.
* :class:`~lisatools.jax.response.tdi_config.TDIConfigWrapJAX` — frozen
  container mirroring the C++ ``TDIConfigWrap``.
* :func:`~lisatools.jax.response.projection.get_sky_vectors`,
  :func:`~lisatools.jax.response.projection.get_phase_ref`,
  :func:`~lisatools.jax.response.projection.get_tdi_Xf_single`
  — sky projection + TDI helpers (was ``fastlisaresponse.jax.projection``).
* :func:`~lisatools.jax.response.amp_phase_extract.extract_amplitude_and_phase`,
  :func:`~lisatools.jax.response.amp_phase_extract.unwrap_phase`,
  :func:`~lisatools.jax.response.amp_phase_extract.extract_and_unwrap`
  — post-processing, the JAX ports of the C++ ``new_extract_amplitude_and_phase``
  / ``new_unwrap_phase`` (was ``fastlisaresponse.jax.amp_phase_extract``).

The consumers are the GBGPU / BBHx JAX paths (``gbgpu.jax``,
``bbhx.jax.sources``); the global-fit runs do not use this subpackage.
Subpackage import is gated on ``import jax`` at :mod:`lisatools.jax`.
"""
from __future__ import annotations

from .base import JaxAmpPhaseSource
from .tdi_config import TDIConfigWrapJAX
from .projection import get_phase_ref, get_sky_vectors, get_tdi_Xf_single
from .amp_phase_extract import (
    extract_amplitude_and_phase,
    extract_and_unwrap,
    unwrap_phase,
)

__all__ = [
    "JaxAmpPhaseSource",
    "TDIConfigWrapJAX",
    "get_phase_ref",
    "get_sky_vectors",
    "get_tdi_Xf_single",
    "extract_amplitude_and_phase",
    "extract_and_unwrap",
    "unwrap_phase",
]
