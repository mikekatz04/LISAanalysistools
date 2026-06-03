"""JAX implementations of the generic WDM-domain infrastructure.

Absorbed from ``fastlisaresponse.jax.wdm`` at Phase 3 of the sprint
reorg. Only the *generic* (waveform-family-agnostic) bits live here:

* :class:`WaveletLookupTableWrapJAX` — JAX port of the C++
  ``WaveletLookupTable``.
* :class:`WDMSettingsWrapJAX` — POD config (Nf, Nt, dt, ...).
* :class:`WDMDomainWrapJAX` — observed-data WDM coefficient container.
* :func:`fast_wdm_inner_jax` — per-pixel projection + frequency
  estimator (was ``fastlisaresponse.jax.wdm.fast_inner.fast_wdm_inner_jax``).

GB-specific kernels (``gb_wdm_get_ll`` / ``gb_wdm_fill_global`` /
``gb_wdm_swap_ll`` and their heterodyne variants) live in
:mod:`gbgpu.jax` after Phase 3F; SOBBH-specific equivalents are
in :mod:`bbhx.jax` after Phase 3G.

Subpackage import is gated on ``import jax`` at :mod:`lisatools.jax`.
"""
from __future__ import annotations

from .wavelet_lookup import WaveletLookupTableWrapJAX
from .wdm_settings import WDMSettingsWrapJAX
from .wdm_domain import WDMDomainWrapJAX
from .fast_inner import fast_wdm_inner_jax

__all__ = [
    "WaveletLookupTableWrapJAX",
    "WDMSettingsWrapJAX",
    "WDMDomainWrapJAX",
    "fast_wdm_inner_jax",
]
