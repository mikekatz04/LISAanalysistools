"""JAX implementations of the generic WDM-domain infrastructure.

Absorbed from ``fastlisaresponse.jax.wdm`` at Phase 3 of the sprint
reorg. Only the *generic* (waveform-family-agnostic) bits live here:

* :class:`~lisatools.jax.wdm.wavelet_lookup.WaveletLookupTableWrapJAX` — JAX
  port of the C++ ``WaveletLookupTable``.
* :class:`~lisatools.jax.wdm.wdm_settings.WDMSettingsWrapJAX` — POD config
  (layer ``df`` / ``dt``, ``Nf``, ``Nt``, channels, active band).
* :class:`~lisatools.jax.wdm.wdm_domain.WDMDomainWrapJAX` — observed-data WDM
  coefficient container.
* :func:`~lisatools.jax.wdm.fast_inner.fast_wdm_inner_jax` — per-pixel
  projection + frequency estimator (was
  ``fastlisaresponse.jax.wdm.fast_inner.fast_wdm_inner_jax``).

GB-specific kernels (``gb_wdm_get_ll`` / ``gb_wdm_fill_global`` /
``gb_wdm_swap_ll`` and their heterodyne variants) live in
``gbgpu.jax`` (``gbgpu.jax.wdm``) after Phase 3F; SOBBH-specific
equivalents are in ``bbhx.jax`` after Phase 3G. The global-fit runs (CUDA
WDM path) do not use this subpackage.

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
