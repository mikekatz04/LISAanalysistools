"""JAX implementations of the LISA TDI / response infrastructure.

Absorbed from ``fastlisaresponse.jax`` at Phase 3 of the sprint reorg.

Public entry points:

* :class:`JaxAmpPhaseSource` — base for source classes that provide
  ``amplitude(t, params)`` and ``phase(t, params)``.
* :class:`TDIConfigWrapJAX` — frozen JAX container mirroring the C++
  ``TDIConfigWrap``.
* :func:`get_sky_vectors`, :func:`get_phase_ref`, :func:`get_tdi_Xf_single`
  — sky projection + TDI helpers (was ``fastlisaresponse.jax.projection``).
* :func:`new_extract_amplitude_and_phase`, :func:`new_unwrap_phase`
  — post-processing (was ``fastlisaresponse.jax.amp_phase_extract``).

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
